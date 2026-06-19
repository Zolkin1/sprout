# Copyright 2026 Zachary Olkin. All rights reserved.

"""Open-loop motion-sequence tracking command.

Tracks a pre-generated fixed motion clip (see ``clips/``) by a pure step
index -- no contact gates, no online re-anchoring, no skill cross-fades.
At reset each env is assigned a clip and placed exactly on it; thereafter
the reference advances by ``step = episode_length_buf`` and the robot's
measured outputs are computed relative to the clip's *pre-computed* world
anchor schedule (the reference stance-foot landing poses baked in offline).

Because the anchor schedule is fixed at generation time, there is no
re-anchoring to the measured foot during training, so neither foot gets a
"free zero" -- the left/right tracking asymmetry of the online command is
gone.  The CLF tracking (``y_des``/``y_act`` in the stance-anchored,
yaw-aligned frame) and reward are otherwise unchanged.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.utils.math import quat_from_angle_axis

from .base_trajectory_cmd import BaseTrajectoryCommand
from .clips.clip_io import load_clip_library
from .clips.clip_fingerprint import fingerprint_clip_config, verify_clip_fingerprint
from .clips.clip_manager import MotionClipManager
from .world_lift import (
    lift_base_world_state,
    lift_orientation,
    lift_orientation_batched,
    lift_position,
    lift_position_batched,
)
from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg

# Per-frame debug-marker colors (RGB 0-1); matches the viser/plot tooling.
_DEBUG_FRAME_COLOR_MAP: dict[str, tuple[float, float, float]] = {
    "pelvis_link": (0.95, 0.55, 0.10),
    "left_ankle_roll_link": (0.20, 0.45, 1.00),
    "right_ankle_roll_link": (0.90, 0.20, 0.25),
    "left_wrist_yaw_link": (0.20, 0.80, 0.85),
    "right_wrist_yaw_link": (0.80, 0.30, 0.90),
}
_DEBUG_FRAME_FALLBACK_COLOR: tuple[float, float, float] = (0.6, 0.6, 0.6)
# Frame-name substrings treated as "hand" end-effectors (filtered out of the
# debug ribbon unless cfg.debug_vis_show_hands is True).
_DEBUG_HAND_FRAME_KEYS: tuple[str, ...] = ("wrist", "hand")
_DEBUG_VIZ_PRIM_PATH: str = "/Visuals/motion_seq_ref"
_DEBUG_VIZ_EPS: float = 1e-6  # guard against zero-length segments / degenerate axes


class MotionSequenceCommand(BaseTrajectoryCommand):
    """Track fixed pre-generated motion clips open-loop."""

    cfg: MotionSequenceCommandCfg

    # Lazily built once the simulator is up (CommandTerm may call
    # ``_set_debug_vis_impl`` before ``_post_init``).  See the debug-vis block.
    _debug_markers: object | None = None
    _debug_vis_pending: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _create_manager(self, cfg: MotionSequenceCommandCfg, env) -> MotionClipManager:
        """Load the dense clip library and wrap it in a gather manager."""
        device = cfg.clip_device or env.device
        self._clip_lib = load_clip_library(cfg.path, device)
        return MotionClipManager(self._clip_lib)

    def _verify_contact_frames(self) -> None:
        """The clip's stance reference frame must be a tracked contact body."""
        ref = self.manager.ref_frame
        if ref not in self.contact_bodies:
            raise ValueError(
                f"Clip reference frame '{ref}' is not in contact_bodies {self.contact_bodies}."
            )

    def _post_init(self) -> None:
        """Allocate clip state, validate dt, map contacts, record fingerprint."""
        n, dev = self.num_envs, self.device
        T = self.manager.num_steps

        # The clip must be sampled at the control dt so step == step counter.
        step_dt = float(self.env.step_dt)
        if abs(self.manager.dt - step_dt) > 1e-6:
            raise ValueError(
                f"Clip dt {self.manager.dt} != env.step_dt {step_dt}. Regenerate clips "
                f"with --dt {step_dt}."
            )

        # Per-env tracking state.
        self.clip_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.start_step = torch.zeros(n, dtype=torch.long, device=dev)
        self._step = torch.zeros(n, dtype=torch.long, device=dev)
        # Per-env world placement of the clip frame (yaw-only ground frame).
        self.clip_world_pos = torch.zeros(n, 3, device=dev)
        self.clip_world_quat = torch.zeros(n, 4, device=dev)
        self.clip_world_quat[:, 3] = 1.0  # identity (xyzw, w last -- Isaac Lab convention)
        # Phase carrier so the base ``_update_command`` can read manager.phase.
        self.manager.phase = torch.zeros(n, device=dev)

        # Base-frame output indices for world reconstruction (placement / obs).
        # Force the orientation channels to xyzw order (the YAML stores ori
        # scalar-first as ori_w, but Isaac Lab's quat math is scalar-last), so
        # the lifted base quaternion matches root_quat_w / write_root_pose.
        base = self.cfg.base_frame_name
        self._base_idx = dict(self._reference_world_indices(base))
        self._base_idx["ori"] = torch.tensor(
            [self.ordered_pos_output_names.index(f"{base}:ori_{a}") for a in ("x", "y", "z", "w")],
            dtype=torch.long, device=dev,
        )

        # Map stored contact columns (clip contact_frames order) to the
        # command's contact_bodies order; -1 where the clip has no schedule.
        clip_cf = self.manager.contact_frames
        self._contact_col = torch.tensor(
            [clip_cf.index(b) if b in clip_cf else -1 for b in self.contact_bodies],
            dtype=torch.long, device=dev,
        )

        self._record_clip_fingerprint(self.cfg)

        # --- Debug-viz setup (per reference frame: pos indices + color) ---
        self._debug_frame_names = [
            n[: -len(":pos_x")] for n in self.ordered_pos_output_names if n.endswith(":pos_x")
        ]
        # Optionally drop hand/wrist end-effector frames from the ribbon.
        if not self.cfg.debug_vis_show_hands:
            self._debug_frame_names = [
                f for f in self._debug_frame_names
                if not any(k in f for k in _DEBUG_HAND_FRAME_KEYS)
            ]
        self._debug_frame_pos_idx = [
            torch.tensor(
                [self.ordered_pos_output_names.index(f"{f}:pos_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            )
            for f in self._debug_frame_names
        ]
        self._debug_frame_colors = [
            _DEBUG_FRAME_COLOR_MAP.get(f, _DEBUG_FRAME_FALLBACK_COLOR) for f in self._debug_frame_names
        ]
        # Forward-window step offsets (current step .. +horizon), subsampled.
        horizon_steps = max(1, int(round(self.cfg.debug_vis_horizon_s / self.manager.dt)))
        stride = max(1, int(self.cfg.debug_vis_stride))
        self._debug_offsets = torch.arange(0, horizon_steps, stride, dtype=torch.long, device=dev)

    def _record_clip_fingerprint(self, cfg: MotionSequenceCommandCfg) -> None:
        """Persist (and verify) the loaded clip library's fingerprint.

        Warns loudly if ``cfg.clip_fingerprint`` (injected at play/resume from
        the trained-against ``env.yaml``) differs from the loaded library's,
        then writes the loaded value back so a fresh run serialises it.
        """
        loaded = self._clip_lib.attrs.get("clip_fingerprint")
        if loaded is None:
            return
        trained = getattr(cfg, "clip_fingerprint", None)
        if trained is not None:
            ok, msg = verify_clip_fingerprint(trained, loaded)
            if not ok:
                print("\n" + "=" * 78 + f"\n[WARN] {msg}\n" + "=" * 78 + "\n")
        cfg.clip_fingerprint = loaded

    # ------------------------------------------------------------------
    # Lifecycle (overrides the online phase/gate machinery)
    # ------------------------------------------------------------------

    def _update_command(self) -> None:
        """Advance the step index, gather reference + measured, update CLF."""
        self._advance_step()
        self.get_measured_outputs()
        self.get_desired_outputs()
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot = vdot
        self.v = vcur

    def _advance_step(self) -> None:
        """Set the per-env clip step from the episode step counter.

        ``episode_length_buf`` is stable within a step, so this is idempotent
        across Isaac Lab's twice-per-step command compute -- no double-advance.
        """
        T = self.manager.num_steps
        self._step = torch.clamp(self.start_step + self.env.episode_length_buf, max=T - 1)
        self.manager.phase = self._step.float() / max(T - 1, 1)

    def _world_anchor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env world anchor pose = clip placement ∘ clip anchor[step]."""
        cap, caq = self.manager.gather_anchor(self.clip_id, self._step)
        wpos = lift_position(self.clip_world_pos, self.clip_world_quat, cap)
        wquat = lift_orientation(self.clip_world_quat, caq)
        return wpos, wquat

    def get_measured_outputs(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> None:
        """Compute measured outputs relative to the clip's world anchor.

        Bypasses all online re-anchoring: the anchor is read from the clip's
        pre-computed schedule (lifted into the world by the per-env clip
        placement), stored in ``ref_poses``, then handed to the inherited
        :meth:`compute_measured_output`.
        """
        wpos, wquat = self._world_anchor()
        self.ref_poses[:, :3] = wpos
        self.ref_poses[:, 3:7] = wquat
        self.compute_measured_output(wpos, wquat)

    def get_desired_outputs(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> None:
        """Gather the (stance-anchored) reference outputs for the current step."""
        ids = env_ids if env_ids is not None else slice(None)
        y, dy = self.manager.gather_desired(self.clip_id[ids], self._step[ids])
        self.y_des[ids] = y
        self.dy_des[ids] = dy

    def get_contact_state(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the clip's expected contact schedule in contact_bodies order."""
        stored = self.manager.gather_contact(self.clip_id, self._step)  # [N, C_clip]
        out = torch.zeros(self.num_envs, len(self.contact_bodies), device=self.device)
        have = self._contact_col >= 0
        cols = self._contact_col[have]
        out[:, have] = stored[:, cols].float()
        if env_ids is not None:
            return out[env_ids]
        return out

    # ------------------------------------------------------------------
    # World-frame reference (for reset placement, obs, termination)
    # ------------------------------------------------------------------

    def reference_world_state(self) -> torch.Tensor:
        """World-frame reference state ``[N, 3+4+3+3+2*nj]`` at the current step.

        ``[base_pos_w, base_quat_w(wxyz), base_lin_w, base_ang_w, joint_pos,
        joint_vel]`` -- lifts the stance-anchored ``y_des`` through the current
        world anchor (``ref_poses``).  Used by the reset event and observations.
        """
        return lift_base_world_state(self.y_des, self.dy_des, self.ref_poses, self._base_idx)

    def reference_world_state_at(
        self, clip_id: torch.Tensor, step: torch.Tensor, world_pos: torch.Tensor, world_quat: torch.Tensor,
    ) -> torch.Tensor:
        """World reference state for an arbitrary (clip, step, placement).

        Used by the reset event to compute spawn poses before the per-step
        buffers are populated.

        Args:
            clip_id: ``[K]`` clip indices.
            step: ``[K]`` step indices.
            world_pos: ``[K, 3]`` clip-placement world positions.
            world_quat: ``[K, 4]`` clip-placement world quaternions (wxyz).

        Returns:
            ``[K, 3+4+3+3+2*nj]`` world reference state.
        """
        y, dy = self.manager.gather_desired(clip_id, step)
        cap, caq = self.manager.gather_anchor(clip_id, step)
        wpos = lift_position(world_pos, world_quat, cap)
        wquat = lift_orientation(world_quat, caq)
        anchor = torch.cat([wpos, wquat], dim=-1)
        return lift_base_world_state(y, dy, anchor, self._base_idx)

    # ------------------------------------------------------------------
    # Debug visualization (standard CommandTerm debug_vis hook)
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Toggle the reference-trajectory marker visualizer.

        Draws, per env, a forward window of each reference frame's world
        position as colored cylinder segments -- a "ribbon" of where the clip
        says each body should go (matching the online command's marker style).
        Per-step update is in :meth:`_debug_vis_callback`.
        """
        if debug_vis:
            if self._debug_markers is None and hasattr(self, "_debug_frame_pos_idx"):
                self._debug_markers = self._create_debug_markers()
            if self._debug_markers is not None:
                self._debug_markers.set_visibility(True)
            else:
                self._debug_vis_pending = True
        else:
            self._debug_vis_pending = False
            if self._debug_markers is not None:
                self._debug_markers.set_visibility(False)

    def _create_debug_markers(self):
        """Build a :class:`VisualizationMarkers` with one cylinder prototype per frame.

        Mirrors the online command's marker style: the reference ribbon is
        drawn as thin cylinder *segments* between consecutive sampled points
        (oriented + z-scaled per segment in :meth:`_debug_vis_callback`),
        rather than a point cloud of spheres.  The native cylinder height is
        ``1.0`` so the per-segment z-scale equals the segment length.
        """
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN MotionSequenceCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_marker_radius)
        prototypes = {
            name: sim_utils.CylinderCfg(
                radius=radius,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
            for name, color in zip(self._debug_frame_names, self._debug_frame_colors)
        }
        cfg = VisualizationMarkersCfg(prim_path=_DEBUG_VIZ_PRIM_PATH, markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN MotionSequenceCommand] could not create VisualizationMarkers: {exc}")
            return None

    def _debug_vis_callback(self, event) -> None:
        """Draw the upcoming reference ribbon as cylinder segments per env.

        For each reference frame, the forward window's world positions are
        connected by one thin cylinder per consecutive pair (the online
        command's marker style): each cylinder is centred on the segment
        midpoint, oriented so its local +z lies along the segment, and
        z-scaled to the segment length.  The prototype index selects the
        frame's colour.
        """
        if self._debug_markers is None:
            if not self._debug_vis_pending:
                return
            self._debug_markers = self._create_debug_markers()
            if self._debug_markers is None:
                return
            self._debug_markers.set_visibility(True)
            self._debug_vis_pending = False

        n = self.num_envs
        T = self.manager.num_steps
        F = len(self._debug_frame_pos_idx)
        steps = (self._step[:, None] + self._debug_offsets[None, :]).clamp(max=T - 1)  # [n, Ns]
        ns = steps.shape[1]
        if F == 0 or n == 0 or ns < 2:
            return
        cid = self.clip_id[:, None].expand(n, ns)                                       # [n, Ns]

        # World anchor schedule for the window: clip placement ∘ clip anchor.
        ap = self.manager.lib.anchor_pos[cid, steps]                                     # [n, Ns, 3]
        aq = self.manager.lib.anchor_quat[cid, steps]                                    # [n, Ns, 4]
        wap = lift_position_batched(self.clip_world_pos, self.clip_world_quat, ap)
        waq = lift_orientation_batched(self.clip_world_quat, aq)

        # Per-frame world positions over the window: [n, Ns, F, 3].
        frame_idx = torch.stack(self._debug_frame_pos_idx, dim=0)                        # [F, 3]
        rel = self.manager.lib.y_des[cid, steps][:, :, frame_idx]                        # [n, Ns, F, 3]
        world_pos = lift_position_batched(
            wap.reshape(n * ns, 3), waq.reshape(n * ns, 4), rel.reshape(n * ns, F, 3),
        ).reshape(n, ns, F, 3)

        # Per-segment midpoint, direction, length (one cylinder per segment per frame).
        starts = world_pos[:, :-1]                                                       # [n, Ns-1, F, 3]
        ends = world_pos[:, 1:]
        centers = 0.5 * (starts + ends)
        vecs = ends - starts
        lens = torch.linalg.norm(vecs, dim=-1).clamp(min=_DEBUG_VIZ_EPS)                  # [n, Ns-1, F]
        dirs = vecs / lens.unsqueeze(-1)

        # Quaternion rotating each cylinder's local +z onto the segment direction.
        # axis = +z x dir = (-dir.y, dir.x, 0); degenerate (dir || +/-z) -> +x fallback.
        angle = torch.acos(dirs[..., 2].clamp(-1.0, 1.0))
        axis = torch.stack([-dirs[..., 1], dirs[..., 0], torch.zeros_like(dirs[..., 0])], dim=-1)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(axis)
        axis = torch.where(axis_norm > _DEBUG_VIZ_EPS, axis, fallback)
        quats = quat_from_angle_axis(angle.reshape(-1), axis.reshape(-1, 3))             # [M, 4] xyzw

        # Native cylinder height is 1.0; z-scale by the segment length.
        scales = torch.ones_like(dirs)
        scales[..., 2] = lens

        # Marker prototype index == frame index (insertion order in _create_debug_markers).
        marker_indices = (
            torch.arange(F, device=self.device, dtype=torch.long)
            .view(1, 1, F).expand(n, ns - 1, F).reshape(-1)
        )

        self._debug_markers.visualize(
            translations=centers.reshape(-1, 3),
            orientations=quats,
            scales=scales.reshape(-1, 3),
            marker_indices=marker_indices,
        )
