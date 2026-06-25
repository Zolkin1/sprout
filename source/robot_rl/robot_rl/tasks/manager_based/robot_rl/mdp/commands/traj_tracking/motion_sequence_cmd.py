# Copyright 2026 Zachary Olkin. All rights reserved.

"""Open-loop motion-sequence tracking command (world-frame CLF).

Tracks a pre-generated fixed motion clip (see ``clips/``) by a pure step
index -- no contact gates, no online re-anchoring, no skill cross-fades.
Clips are stored in a single clip-world frame (the per-step stance anchor is
baked in at generation time), so the only runtime transform is a per-env
*placement* (the clip's yaw + xy in the sim world, fixed at reset).  At reset
each env is assigned a clip and placed exactly on it; thereafter the reference
advances by ``step = episode_length_buf``.

The CLF runs in the **absolute world frame**: the desired outputs
``y_des``/``dy_des`` are the clip's outputs lifted through the placement, and
the measured outputs ``y_act``/``dy_act`` are read with an identity reference
(no yaw re-alignment).  This is the same world frame
:class:`GeneratorTrackingCommand` uses, so the two tasks share one frame.
Because the placement is fixed per episode, there is no re-anchoring to the
measured foot during training -- the left/right tracking asymmetry of the
online command is gone.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.utils.math import quat_from_angle_axis

from .base_trajectory_cmd import BaseTrajectoryCommand
from .sagittal_reflector import NamedReflector, SagittalReflectionConfig
from .clips.clip_io import load_clip_library
from .clips.clip_fingerprint import fingerprint_clip_config, verify_clip_fingerprint
from .clips.clip_manager import MotionClipManager
from .world_lift import (
    lift_base_world_state,
    lift_orientation,
    lift_orientation_batched,
    lift_position,
    lift_position_batched,
    lift_vector,
    lift_vector_batched,
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
        # Optional deterministic-reset overrides (default None => random sampling
        # in reset_on_motion_sequence is unchanged).  When set to [N] long tensors
        # they force the per-env clip id / start step at reset -- used by
        # scripts/rsl_rl/record_teacher_clips.py to roll out every clip exactly
        # once.  Training never sets these.
        self.forced_clip_id: torch.Tensor | None = None
        self.forced_start_step: torch.Tensor | None = None
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

        # --- World-frame assembly (shared with GeneratorTrackingCommand) ---
        # Output frames whose Cartesian channels are lifted clip-world -> world:
        # the base plus every end-effector reference frame, in resolved order.
        # (GeneratorTrackingCommand rebuilds these in its FM frame order.)
        self._build_world_frame_maps([self.cfg.base_frame_name] + self.resolve_ee_frames())

        # Identity reference for the world-frame measured outputs (no alignment).
        self._zeros3 = torch.zeros(n, 3, device=dev)
        self._ident_quat = torch.zeros(n, 4, device=dev)
        self._ident_quat[:, 3] = 1.0  # xyzw identity

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
                if getattr(cfg, "strict_fingerprint", False):
                    raise RuntimeError(msg)
                print("\n" + "=" * 78 + f"\n[WARN] {msg}\n" + "=" * 78 + "\n")
        cfg.clip_fingerprint = loaded

    # ------------------------------------------------------------------
    # Lifecycle (overrides the online phase/gate machinery)
    # ------------------------------------------------------------------

    def _build_world_frame_maps(self, frames: list[str]) -> None:
        """Build the world-frame channel index maps + per-step cache for ``frames``.

        ``frames`` is ``[base] + ee`` (the base frame followed by the ee
        reference frames whose Cartesian channels get lifted clip-world ->
        world).  Populates the flat scatter indices into ``y_des``/``dy_des``,
        a name->row lookup, and the ``[N, nf, 13]`` world-frame cache served by
        :meth:`reference_world_frames` / :meth:`reference_world_state`.
        """
        self._wf_frames = list(frames)
        self._wf_frame_index = {f: i for i, f in enumerate(self._wf_frames)}
        pos_list, ori_list, lin_list, ang_list = [], [], [], []
        for f in self._wf_frames:
            fidx = self._frame_output_indices(f)
            pos_list.append(fidx["pos"]); ori_list.append(fidx["ori"])
            lin_list.append(fidx["lin_vel"]); ang_list.append(fidx["ang_vel"])
        self._wf_pos_flat = torch.cat(pos_list)   # [nf*3] -> y_des
        self._wf_ori_flat = torch.cat(ori_list)   # [nf*4] -> y_des
        self._wf_lin_flat = torch.cat(lin_list)   # [nf*3] -> dy_des
        self._wf_ang_flat = torch.cat(ang_list)   # [nf*3] -> dy_des
        self._wf_world_frames = torch.zeros(self.num_envs, len(self._wf_frames), 13, device=self.device)

    def _update_command(self) -> None:
        """Advance the step, build world-frame desired + measured, update CLF."""
        self._advance_step()
        self.get_desired_outputs()                 # clip-world y_des/dy_des
        frames = self._clip_world_frames()         # lift through the placement
        self._write_world_des(frames)              # base+ee Cartesian -> world
        # Measured outputs in the world frame (identity reference -> no alignment).
        self.compute_measured_output(self._zeros3, self._ident_quat)
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

    def _placement(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env world placement of the clip (yaw-only ground frame, fixed at reset)."""
        return self.clip_world_pos, self.clip_world_quat

    def _lift_ee_world(
        self, frame_names: list[str], anchor_pos: torch.Tensor, anchor_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Lift the clip-world ``y_des``/``dy_des`` channels of ``frame_names`` through an anchor.

        Reads the (clip-world) stored outputs, so this must run before
        :meth:`_write_world_des` overwrites them.  Returns ``[N, K, 13]`` per
        frame ``[pos_w(3), quat_w(4, xyzw), lin_w(3), ang_w(3)]``.
        """
        out = []
        for f in frame_names:
            idx = self._frame_output_indices(f)
            pos_w = lift_position(anchor_pos, anchor_quat, self.y_des[:, idx["pos"]])
            quat_w = lift_orientation(anchor_quat, self.y_des[:, idx["ori"]])
            lin_w = lift_vector(anchor_quat, self.dy_des[:, idx["lin_vel"]])
            ang_w = lift_vector(anchor_quat, self.dy_des[:, idx["ang_vel"]])
            out.append(torch.cat([pos_w, quat_w, lin_w, ang_w], dim=-1))
        return torch.stack(out, dim=1)

    def _clip_world_frames(self) -> torch.Tensor:
        """Lift the current clip-world outputs through the per-env placement.

        Sets ``ref_poses`` to the placement, then lifts the base + every ee
        reference frame from the stored clip-world frame into the absolute world
        frame.  Must run before :meth:`_write_world_des` (reads clip-world
        ``y_des``).

        Returns:
            ``frames [N, nf, 13]`` -- ``[base] + ee`` in ``self._wf_frames``
            order, each ``[pos_w(3), quat_w(4, xyzw), lin_w(3), ang_w(3)]``.
        """
        wpos, wquat = self._placement()
        self.ref_poses[:, :3] = wpos
        self.ref_poses[:, 3:7] = wquat
        base13 = lift_base_world_state(self.y_des, self.dy_des, self.ref_poses, self._base_idx)[:, :13]
        if len(self._wf_frames) > 1:
            ee = self._lift_ee_world(self._wf_frames[1:], wpos, wquat)  # [N, K, 13]
            return torch.cat([base13[:, None, :], ee], dim=1)          # [N, nf, 13]
        return base13[:, None, :]

    def _write_world_des(self, frames: torch.Tensor) -> None:
        """Scatter world-frame ``[N, nf, 13]`` frames into y_des/dy_des, and cache them.

        Caches the (final, source-selected) world frames so the post-step
        recorder API (:meth:`reference_world_state` /
        :meth:`reference_world_frames`) reads them back without re-lifting the
        now-overwritten ``y_des``.
        """
        n = self.num_envs
        self.y_des[:, self._wf_pos_flat] = frames[:, :, 0:3].reshape(n, -1)
        self.y_des[:, self._wf_ori_flat] = frames[:, :, 3:7].reshape(n, -1)
        self.dy_des[:, self._wf_lin_flat] = frames[:, :, 7:10].reshape(n, -1)
        self.dy_des[:, self._wf_ang_flat] = frames[:, :, 10:13].reshape(n, -1)
        self._wf_world_frames = frames

    def get_desired_outputs(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> None:
        """Gather the (clip-world) reference outputs for the current step."""
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
    # Commanded velocity (upstream command that generated the clip)
    # ------------------------------------------------------------------

    @property
    def command_velocity_available(self) -> bool:
        """True if the clip library carries a per-step commanded velocity (``target_vel``)."""
        return self.manager.has_target_vel()

    def get_command_velocity(self) -> torch.Tensor:
        """Per-env commanded base velocity ``[N, 3]`` = (vx, vy, wz) at the current step.

        This is the *command* that generated the clip (library ``target_vel``), not
        the achieved reference velocity (``dy_des``).  It is the signal to condition
        a future-horizon flow-matching model on -- upstream of the reference and
        available at deployment.  Recorded by ``scripts/rsl_rl/record_data.py``.
        """
        if not self.command_velocity_available:
            raise RuntimeError(
                "Clip library has no target_vel; regenerate clips with a generator that writes it."
            )
        return self.manager.gather_target_vel(self.clip_id, self._step)

    def get_command_velocity_horizon(self, num_future: int) -> torch.Tensor:
        """Commanded velocity over the next ``num_future`` steps, ``[N, num_future, 3]``.

        Covers steps ``s+1 .. s+num_future`` (clamped to the clip end), matching the
        flow-matching future window whose history ends at the current step ``s``.
        Used at inference to condition the decoder on the command horizon.
        """
        if not self.command_velocity_available:
            raise RuntimeError(
                "Clip library has no target_vel; regenerate clips with a generator that writes it."
            )
        T = self.manager.num_steps
        offs = torch.arange(num_future, device=self.device)
        steps = (self._step[:, None] + 1 + offs[None, :]).clamp(max=T - 1)  # [N, num_future]
        cid = self.clip_id[:, None].expand_as(steps)
        return self.manager.gather(self.manager.lib.target_vel, cid, steps)

    # ------------------------------------------------------------------
    # Future window (look-ahead into the clip for the teacher policy)
    # ------------------------------------------------------------------

    def _future_step_indices(self, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(clip_id, step)`` index grids ``[N, K]`` for the given future ``offsets``.

        Steps are ``self._step + offset`` clamped to the clip end, so the window
        saturates on the final frame near a clip's end (same convention as
        :meth:`get_command_velocity_horizon` and the debug-viz ribbon).
        """
        T = self.manager.num_steps
        steps = (self._step[:, None] + offsets[None, :]).clamp(max=T - 1)  # [N, K]
        cid = self.clip_id[:, None].expand_as(steps)
        return cid, steps

    def get_future_command_velocity(self, offsets: torch.Tensor) -> torch.Tensor:
        """Commanded base velocity ``[N, K, 3]`` = (vx, vy, wz) at the future ``offsets``.

        The upstream ``target_vel`` (heading frame, deployable) at steps
        ``self._step + offsets`` -- the look-ahead counterpart of
        :meth:`get_command_velocity`.
        """
        if not self.command_velocity_available:
            raise RuntimeError(
                "Clip library has no target_vel; regenerate clips with a generator that writes it."
            )
        cid, steps = self._future_step_indices(offsets)
        return self.manager.gather(self.manager.lib.target_vel, cid, steps)

    def get_future_world_base(
        self, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame reference base pose+velocity over the future ``offsets``.

        Returns ``(pos_w, quat_w, lin_w, ang_w)`` of shapes ``[N, K, 3]``,
        ``[N, K, 4]`` (xyzw), ``[N, K, 3]``, ``[N, K, 3]``.  Lifts the clip-world
        base channels through the per-env placement (a single yaw+xy transform),
        so ``quat_w`` matches the convention consumed by the ``base_pose_error``
        observation and the debug-viz ribbon.
        """
        cid, steps = self._future_step_indices(offsets)
        bi = self._base_idx
        y = self.manager.gather(self.manager.lib.y_des, cid, steps)    # [N, K, P] clip-world
        dy = self.manager.gather(self.manager.lib.dy_des, cid, steps)  # [N, K, V]
        pos_w = lift_position_batched(self.clip_world_pos, self.clip_world_quat, y[:, :, bi["pos"]])
        quat_w = lift_orientation_batched(self.clip_world_quat, y[:, :, bi["ori"]])
        lin_w = lift_vector_batched(self.clip_world_quat, dy[:, :, bi["lin_vel"]])
        ang_w = lift_vector_batched(self.clip_world_quat, dy[:, :, bi["ang_vel"]])
        return pos_w, quat_w, lin_w, ang_w

    # ------------------------------------------------------------------
    # World-frame reference (for reset placement, obs, termination)
    # ------------------------------------------------------------------

    def reference_world_state(self) -> torch.Tensor:
        """World-frame reference state ``[N, 3+4+3+3+2*nj]`` at the current step.

        ``[base_pos_w, base_quat_w(wxyz), base_lin_w, base_ang_w, joint_pos,
        joint_vel]`` -- served from the per-step world cache populated by
        :meth:`_write_world_des` (base row) plus the frame-invariant joint
        channels.  Used by the reset event, observations, and terminations.
        """
        joint_pos = self.y_des[:, self._base_idx["joint_pos"]]
        joint_vel = self.dy_des[:, self._base_idx["joint_vel"]]
        return torch.cat([self._wf_world_frames[:, 0, :], joint_pos, joint_vel], dim=-1)

    def _frame_output_indices(self, frame_name: str) -> dict[str, torch.Tensor]:
        """Resolve (and cache) the y_des/dy_des indices for one output frame.

        Returns long-tensor index sets keyed by ``pos`` (3), ``ori`` (4, xyzw),
        ``lin_vel`` (3), ``ang_vel`` (3) for ``frame_name``.
        """
        cache = getattr(self, "_frame_idx_cache", None)
        if cache is None:
            cache = {}
            self._frame_idx_cache = cache
        if frame_name in cache:
            return cache[frame_name]
        pos_names = self.ordered_pos_output_names
        vel_names = self.ordered_vel_output_names
        dev = self.device
        entry = {
            "pos": torch.tensor(
                [pos_names.index(f"{frame_name}:pos_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            ),
            "ori": torch.tensor(
                [pos_names.index(f"{frame_name}:ori_{a}") for a in ("x", "y", "z", "w")],
                dtype=torch.long, device=dev,
            ),
            "lin_vel": torch.tensor(
                [vel_names.index(f"{frame_name}:pos_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            ),
            "ang_vel": torch.tensor(
                [vel_names.index(f"{frame_name}:ori_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            ),
        }
        cache[frame_name] = entry
        return entry

    def reference_world_frames(self, frame_names: list[str]) -> torch.Tensor:
        """World-frame reference state for extra output frames (e.g. feet/hands).

        Served from the per-step world cache populated by
        :meth:`_write_world_des` -- so it reflects the selected source (clip or
        generator) and never re-lifts the world-overwritten ``y_des``.  Used by
        the data recorder to record ee reference states.

        Args:
            frame_names: Output frame names; each must be one of the command's
                world-frame ee frames (``self._wf_frames`` minus the base).

        Returns:
            ``[N, K, 13]`` per frame ``[pos_w(3), quat_w(4, xyzw), lin_vel_w(3),
            ang_vel_w(3)]`` in the given frame order.
        """
        idxs = [self._wf_frame_index[f] for f in frame_names]
        return self._wf_world_frames[:, idxs, :]

    def get_reference_world_state(self, base_frame_name: str) -> torch.Tensor:
        """World-frame reference state, matching the base-class recording API.

        ``scripts/rsl_rl/record_data.py`` auto-detects the trajectory command via
        this method name and calls it with ``--ref_base_frame``.  The inherited
        :meth:`BaseTrajectoryCommand.get_reference_world_state` re-resolves the
        orientation indices in name-appearance order, which is wrong for the
        clip library (it stores ``ori`` scalar-first); this override delegates to
        the xyzw-correct :meth:`reference_world_state` instead.

        Args:
            base_frame_name: Trajectory output frame used as the robot root; must
                match ``cfg.base_frame_name`` (the only base frame this clip-based
                command is wired for).

        Returns:
            World-frame reference state, shape ``[N, 3 + 4 + 3 + 3 + 2*nj]``.
        """
        if base_frame_name != self.cfg.base_frame_name:
            raise ValueError(
                f"MotionSequenceCommand is wired for base frame "
                f"'{self.cfg.base_frame_name}'; got '{base_frame_name}'."
            )
        return self.reference_world_state()

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
        placement = torch.cat([world_pos, world_quat], dim=-1)
        return lift_base_world_state(y, dy, placement, self._base_idx)

    # ------------------------------------------------------------------
    # Symmetry helpers (end-effector reference)
    # ------------------------------------------------------------------

    def resolve_ee_frames(self, frame_names: list[str] | None = None) -> list[str]:
        """Resolve the end-effector frame list/order for the ``ee_ref`` obs.

        Single source of truth shared by :func:`mdp.ee_ref_relative_to_base`
        and its symmetry reflector (:meth:`get_symmetric_ee`) so the two can
        never drift.  When ``frame_names`` is ``None``, returns every reference
        output frame that exposes a position, except the base frame, sorted
        alphabetically (matching the obs default).

        Args:
            frame_names: Explicit frame list, or ``None`` for the default.

        Returns:
            Ordered end-effector frame names.
        """
        if frame_names is not None:
            return list(frame_names)
        base = self.cfg.base_frame_name
        return sorted(
            {n.split(":", 1)[0] for n in self.ordered_pos_output_names if ":pos_" in n} - {base}
        )

    def get_symmetric_ee(
        self, ee: torch.Tensor, frame_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Return the left-right symmetric reflection of base-relative ee positions.

        :func:`mdp.ee_ref_relative_to_base` stacks each frame's xyz position
        (expressed in the reference base frame).  Under sagittal reflection each
        left/right frame swaps and each frame's y component negates -- a
        :class:`NamedReflector` over ``{frame}:pos_{x,y,z}`` names does both.
        Reflectors are cached per resolved frame tuple.

        Args:
            ee: ``[N, 3*F]`` or ``[N, history_length * 3*F]`` ee-position tensor.
            frame_names: Same frame list the obs used (``None`` -> default).

        Returns:
            Symmetric ee positions of the same shape.
        """
        frames = self.resolve_ee_frames(frame_names)
        key = tuple(frames)
        cache = getattr(self, "_ee_reflector_cache", None)
        if cache is None:
            cache = {}
            self._ee_reflector_cache = cache
        reflector = cache.get(key)
        if reflector is None:
            names = [f"{f}:pos_{a}" for f in frames for a in ("x", "y", "z")]
            reflector = NamedReflector(SagittalReflectionConfig(), names, self.device)
            cache[key] = reflector
        return reflector.reflect_with_history(ee)

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

    def _clip_ribbon_env_mask(self) -> torch.Tensor | None:
        """Per-env bool mask of which envs draw the clip ribbon (``None`` => all).

        Subclasses override this to suppress the clip ribbon for envs tracking a
        different reference (e.g. the flow-matching generator). Returning
        ``None`` keeps the default behavior (draw every env).
        """
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

        # Optionally restrict the ribbon to a subset of envs (e.g. drop envs that
        # are tracking a different reference). Subselect every per-env quantity so
        # the batched lift stays aligned; hide the markers if no env qualifies.
        mask = self._clip_ribbon_env_mask()
        if mask is not None:
            env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                self._debug_markers.set_visibility(False)
                return
            self._debug_markers.set_visibility(True)
            step, clip_id = self._step[env_ids], self.clip_id[env_ids]
            clip_world_pos, clip_world_quat = self.clip_world_pos[env_ids], self.clip_world_quat[env_ids]
        else:
            step, clip_id = self._step, self.clip_id
            clip_world_pos, clip_world_quat = self.clip_world_pos, self.clip_world_quat

        n = step.shape[0]
        T = self.manager.num_steps
        F = len(self._debug_frame_pos_idx)
        steps = (step[:, None] + self._debug_offsets[None, :]).clamp(max=T - 1)          # [n, Ns]
        ns = steps.shape[1]
        if F == 0 or n == 0 or ns < 2:
            return
        cid = clip_id[:, None].expand(n, ns)                                            # [n, Ns]

        # Per-frame clip-world positions over the window, lifted to sim-world by
        # the per-env placement (a single yaw+xy transform; no per-step anchor).
        frame_idx = torch.stack(self._debug_frame_pos_idx, dim=0)                        # [F, 3]
        rel = self.manager.gather(self.manager.lib.y_des, cid, steps)[:, :, frame_idx]    # [n, Ns, F, 3] clip-world
        world_pos = lift_position_batched(
            clip_world_pos, clip_world_quat, rel.reshape(n, ns * F, 3),
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
