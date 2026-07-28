# Copyright 2026 Zachary Olkin. All rights reserved.

"""Closed-loop, robot-state-conditioned flow-matching tracking command (world-frame CLF).

Extends :class:`MotionSequenceCommand`: every env tracks the **flow-matching generator**,
which conditions on the **live robot state** history (the N most-recent measured steps, in
the FM's xyz+yaw ground frame), a root-height token, and the future velocity command, and
predicts the F-step ego-anchored clip future. Because the FM is anchored on the robot's own
xyz+yaw pose each re-plan and predicts node 0 = "now" (offset ≈ 0), no re-centering hack is
needed -- the prediction is already in the robot's frame. This is the deploy counterpart of
the recorded ``ego_anchored_clip_future`` training data (each re-inference == a snap).

The clip library supplies only the reset spawn pose + the ``target_vel`` command; the clip
reference is the warmup fallback for the first N steps until the history window fills. The
desired/measured outputs are in the world frame, so the inherited CLF reward, observations,
and terminations work uniformly. All base/ee velocities are the **link** (actor-frame) twist
(the FM is conditioned on ``root_link_{lin,ang}_vel_w``; ``y_act`` comes from ``body_link_vel_w``).
"""

from __future__ import annotations

import os

import torch
import warp as wp

from isaaclab.envs.mdp import height_scan as mdp_height_scan
from isaaclab.managers import SceneEntityCfg

from robot_rl.flow_matching.online import (
    BatchedHistory,
    build_cmd_tokens,
    build_height_token,
    build_history_tokens,
    load_fm_context,
    sample_world,
)

from .motion_sequence_cmd import MotionSequenceCommand
from .fm_tracking_cmd_cfg import FmTrackingCommandCfg


class FmTrackingCommand(MotionSequenceCommand):
    """Track a robot-state-conditioned flow-matching generator in the world frame."""

    cfg: FmTrackingCommandCfg

    # Debug-viz marker handles (lazily built once the sim is up).
    _fm_markers: object | None = None
    _source_markers: object | None = None

    def _post_init(self) -> None:
        """Load the FM context, the rolling robot-state history, and prediction buffers."""
        super()._post_init()
        n, dev = self.num_envs, self.device

        if not self.command_velocity_available:
            raise RuntimeError(
                "FmTrackingCommand requires a clip library with target_vel (the FM velocity "
                "command). Regenerate clips with a generator that writes it."
            )
        if not self.cfg.fm_checkpoint:
            raise ValueError(
                "FmTrackingCommandCfg.fm_checkpoint must be set to a trained robot-state-"
                "conditioned (ego_anchored_clip_future) flow-matching .pt checkpoint."
            )
        if not os.path.isfile(self.cfg.fm_checkpoint):
            raise FileNotFoundError(f"FmTrackingCommandCfg.fm_checkpoint '{self.cfg.fm_checkpoint}' does not exist.")
        if int(self.cfg.fm_update_every) < 1:
            raise ValueError(f"fm_update_every must be >= 1, got {self.cfg.fm_update_every}.")

        self.ctx = load_fm_context(
            self.cfg.fm_checkpoint, device=self.cfg.fm_device or dev,
            num_sampling_steps=self.cfg.fm_num_sampling_steps,
            zero_noise=self.cfg.fm_zero_noise,
        )
        if self.cfg.fm_zero_noise:
            print("[INFO FmTrackingCommand] fm_zero_noise=True: the flow integrates from x0=0, "
                  "so predictions are deterministic given the conditioning (diagnostic mode, "
                  "off the training distribution).")
        if self.ctx.cmd_dim <= 0:
            raise ValueError("FmTrackingCommand expects a command-conditioned FM checkpoint (cmd_dim>0).")

        # The FM's extra frames must be exactly the CLF's end-effector output frames.
        ee_frames = self.resolve_ee_frames()
        if set(self.ctx.frame_names) != set(ee_frames):
            raise ValueError(
                f"FM checkpoint extra frames {self.ctx.frame_names} != command ee output "
                f"frames {ee_frames}; re-record/-train the FM with matching --state_frames."
            )

        # Rolling per-env history of the LIVE ROBOT state (N steps), in FM joint/frame order.
        self._history = BatchedHistory(n, self.ctx.N, dev)

        # World-frame index maps in FM frame order ([base] + FM ee), so the per-step select
        # aligns the clip and FM frames without a permutation.
        self._build_world_frame_maps([self.cfg.base_frame_name] + list(self.ctx.frame_names))

        # Joint slots (clip-output order) + FM->clip and robot->FM joint permutations.
        joint_out_names = [nm.split(":", 1)[1] for nm in self.ordered_pos_output_names if nm.startswith("joint:")]
        self._joint_slots_pos = torch.tensor(
            [self.ordered_pos_output_names.index(f"joint:{nm}") for nm in joint_out_names],
            dtype=torch.long, device=dev,
        )
        self._joint_slots_vel = torch.tensor(
            [self.ordered_vel_output_names.index(f"joint:{nm}") for nm in joint_out_names],
            dtype=torch.long, device=dev,
        )
        self._fm_jperm = torch.tensor(
            [self.ctx.joint_names.index(nm) for nm in joint_out_names], dtype=torch.long, device=dev
        )
        robot_joints = list(self.robot.data.joint_names)
        self._robot_to_fm_jperm = torch.tensor(
            [robot_joints.index(nm) for nm in self.ctx.joint_names], dtype=torch.long, device=dev
        )
        self._ee_body_idx = (
            torch.tensor([self.robot.body_names.index(f) for f in self.ctx.frame_names], dtype=torch.long, device=dev)
            if self.ctx.num_extra_frames > 0 else None
        )

        # Prediction buffers + cadence state (every env runs the FM).
        self._pending_reset = torch.zeros(n, dtype=torch.bool, device=dev)
        f, nj, k = self.ctx.F, self.ctx.nj, self.ctx.num_extra_frames
        self._pred_root_pos = torch.zeros(n, f, 3, device=dev)
        self._pred_root_quat = torch.zeros(n, f, 4, device=dev)
        self._pred_root_lin = torch.zeros(n, f, 3, device=dev)
        self._pred_root_ang = torch.zeros(n, f, 3, device=dev)
        self._pred_jp = torch.zeros(n, f, nj, device=dev)
        self._pred_jv = torch.zeros(n, f, nj, device=dev)
        self._pred_frame_pos = torch.zeros(n, f, k, 3, device=dev)
        self._pred_frame_quat = torch.zeros(n, f, k, 4, device=dev)
        self._pred_frame_lin = torch.zeros(n, f, k, 3, device=dev)
        self._pred_frame_ang = torch.zeros(n, f, k, 3, device=dev)
        self._pred_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self._pred_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._fm_reinfer_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._last_step_seen = -1
        self._arange_n = torch.arange(n, device=dev)

        # --- FM-prediction debug ribbon: map each clip-ribbon frame (base built by the
        # base _post_init, re-ordered above) to its slot in the FM prediction buffers.
        self._fm_pred_specs: list[tuple[str, tuple[float, float, float], str, int]] = []
        for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
            if name == self.cfg.base_frame_name:
                self._fm_pred_specs.append((name, color, "base", -1))
            elif name in self.ctx.frame_names:
                self._fm_pred_specs.append((name, color, "frame", self.ctx.frame_names.index(name)))
        # Show EVERY predicted node in the FM ribbon (dense), independent of the
        # clip ribbon's debug_vis_stride subsampling.
        self._fm_debug_offsets = torch.arange(0, self.ctx.F, dtype=torch.long, device=dev)

    def reset(self, env_ids=None):
        """Latch the per-env episode reset (consumed in :meth:`_snapshot_and_infer`)."""
        if env_ids is None:
            self._pending_reset[:] = True
        else:
            self._pending_reset[env_ids] = True
        return super().reset(env_ids)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------
    def _update_command(self) -> None:
        """World-frame desired+measured outputs (FM once warmed up, clip during warmup) + CLF."""
        self._advance_step()
        self._snapshot_and_infer()

        # Re-anchor the CLIP (warmup fallback) onto the robot on the FM re-plan steps, so
        # the fallback is robot-anchored like the FM. Inert when ego_anchor_fraction == 0.
        if self.cfg.ego_anchor_fraction > 0.0:
            self._reanchor_clip_ids(torch.nonzero(self._fm_reinfer_mask, as_tuple=False).squeeze(-1))
        # Resolve the placement for THIS step's anchor-mask segment (the base class does this in its
        # own _update_command, which this override replaces). Inert for a non-ego env: its snapshot
        # still carries zero correction, so every segment resolves to the terrain-cell anchor.
        self._refresh_current_placement()

        self.get_desired_outputs()                                   # clip joints (frame-invariant)
        clip_frames = self._clip_world_frames()                      # [N, nf, 13] warmup fallback

        fm_base, fm_ee, fm_jp, fm_jv = self._fm_current_outputs()
        fm_frames = torch.cat([fm_base[:, None, :], fm_ee], dim=1)   # [N, nf, 13]
        use_fm = self._pred_valid

        sel = torch.where(use_fm[:, None, None], fm_frames, clip_frames)
        self._write_world_des(sel)

        fm_jp_slots, fm_jv_slots = fm_jp[:, self._fm_jperm], fm_jv[:, self._fm_jperm]
        self.y_des[:, self._joint_slots_pos] = torch.where(
            use_fm[:, None], fm_jp_slots, self.y_des[:, self._joint_slots_pos]
        )
        self.dy_des[:, self._joint_slots_vel] = torch.where(
            use_fm[:, None], fm_jv_slots, self.dy_des[:, self._joint_slots_vel]
        )

        # Measured outputs in the world frame (identity reference -> no alignment).
        self.compute_measured_output(self._zeros3, self._ident_quat)
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot, self.v = vdot, vcur

    # ------------------------------------------------------------------
    # Robot-state history snapshot + inference (once per control step)
    # ------------------------------------------------------------------
    def _robot_snapshot(self, ids: torch.Tensor | slice = slice(None)) -> dict[str, torch.Tensor]:
        """Snapshot the live robot state (FM joint/frame order, world frame) for ``ids``."""
        d = self.robot.data
        snap = {
            "root_pos": wp.to_torch(d.root_pos_w)[ids].clone(),
            "root_quat": wp.to_torch(d.root_quat_w)[ids].clone(),           # xyzw
            "root_lin": wp.to_torch(d.root_link_lin_vel_w)[ids].clone(),
            "root_ang": wp.to_torch(d.root_link_ang_vel_w)[ids].clone(),
            "jp": wp.to_torch(d.joint_pos)[ids][:, self._robot_to_fm_jperm].clone(),
            "jv": wp.to_torch(d.joint_vel)[ids][:, self._robot_to_fm_jperm].clone(),
        }
        if self._ee_body_idx is not None:
            fp = wp.to_torch(d.body_pos_w)[ids][:, self._ee_body_idx]
            fq = wp.to_torch(d.body_quat_w)[ids][:, self._ee_body_idx]
            fl = wp.to_torch(d.body_link_vel_w)[ids][:, self._ee_body_idx]  # [.,K,6]
            snap["frames"] = torch.cat([fp, fq, fl[..., :3], fl[..., 3:]], dim=-1).clone()  # [.,K,13]
        return snap

    def _snapshot_and_infer(self) -> None:
        """Append the live robot state to history and (re-)run FM inference once per step."""
        step = int(getattr(self.env, "common_step_counter", self._last_step_seen + 1))
        if step == self._last_step_seen:
            return
        self._last_step_seen = step

        just_reset = self._pending_reset.clone()
        self._pending_reset[just_reset] = False
        if bool(just_reset.any()):
            self._pred_valid = self._pred_valid & ~just_reset
            self._pred_idx = torch.where(just_reset, torch.zeros_like(self._pred_idx), self._pred_idx)

        self._history.invalidate(just_reset)
        self._history.append(self._robot_snapshot())

        valid = self._history.valid()
        # Envs whose window has not filled yet (post-reset). With fm_null_history_warmup the FM
        # is still sampled for them, conditioned on the learned null-history embedding it was
        # trained with, so the generator drives from step 0 and no clip fallback is needed.
        null_hist = self._history.null_mask()
        if self.cfg.fm_null_history_warmup:
            gate = torch.ones_like(valid)
        else:
            gate = valid
            self._pred_valid = self._pred_valid & valid
        reinfer = (self.env.episode_length_buf % max(int(self.cfg.fm_update_every), 1)) == 0
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        infer_mask = gate & (reinfer | ~self._pred_valid | exhausted)
        self._fm_reinfer_mask = infer_mask
        # Advance the applied index on steps that did not re-plan (only meaningful once a
        # prediction exists; ``_pred_valid`` implies a filled window in the non-null mode).
        advance = self._pred_valid & ~infer_mask
        self._pred_idx = torch.where(
            advance, torch.clamp(self._pred_idx + 1, max=self.ctx.F - 1), self._pred_idx
        )
        if bool(infer_mask.any()):
            self._run_fm(infer_mask, null_hist)

    @torch.no_grad()
    def _run_fm(self, mask: torch.Tensor, null_hist: torch.Tensor | None = None) -> None:
        """Run batched FM inference for the masked envs (robot-state history) and store predictions.

        ``null_hist`` is the per-env "history not yet complete" mask; those envs are sampled with
        the model's learned ``null_history`` embedding instead of their (incomplete, possibly
        pre-reset) window. ``stack_padded`` is used so the window is always N long even on the
        first steps of a run -- its content is discarded wherever ``null_hist`` is set.
        """
        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        rp = self._history.stack_padded("root_pos")[ids]    # [V,N,3] live robot history
        rq = self._history.stack_padded("root_quat")[ids]
        rl = self._history.stack_padded("root_lin")[ids]
        ra = self._history.stack_padded("root_ang")[ids]
        jp = self._history.stack_padded("jp")[ids]          # [V,N,nj] FM order
        jv = self._history.stack_padded("jv")[ids]
        frames = self._history.stack_padded("frames")[ids] if self.ctx.num_extra_frames > 0 else None

        history = build_history_tokens(self.ctx, rp, rq, rl, ra, jp, jv, frames=frames)
        # Height token = the live height scan at "now" (per-ray ground clearance from the
        # pelvis scanner), matching training; the same mdp.height_scan the teacher obs uses.
        scan = mdp_height_scan(self.env, SceneEntityCfg("height_scanner"))[ids]  # [V,K]
        height = build_height_token(self.ctx, scan)
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        cmd_tokens = build_cmd_tokens(self.ctx, fut_cmd)
        # Anchor = the robot NOW (the most-recent history step IS the live robot); the FM
        # predicts in this xyz+yaw frame, so no re-centering is needed.
        p_ref, q_ref = rp[:, -1].clone(), rq[:, -1].clone()
        world = sample_world(self.ctx, history, height, cmd_tokens, p_ref, q_ref,
                             history_null_mask=(None if null_hist is None else null_hist[ids]))

        self._pred_root_pos[ids] = world["root_pos_w"]
        self._pred_root_quat[ids] = world["root_quat_w"]
        self._pred_root_lin[ids] = world["root_link_lin_vel_w"]
        self._pred_root_ang[ids] = world["root_link_ang_vel_w"]
        self._pred_jp[ids] = world["joint_pos"]
        self._pred_jv[ids] = world["joint_vel"]
        if self.ctx.num_extra_frames > 0:
            self._pred_frame_pos[ids] = world["frame_pos_w"]
            self._pred_frame_quat[ids] = world["frame_quat_w"]
            self._pred_frame_lin[ids] = world["frame_lin_vel_w"]
            self._pred_frame_ang[ids] = world["frame_ang_vel_w"]
        # Node 0 = "now" (the anchor step, offset ≈ 0). Track from node 0 and advance one
        # per step, resetting to 0 at each re-plan -- exactly the recorded snap regime.
        self._pred_idx[ids] = 0
        self._pred_valid[ids] = True

    def _fm_current_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather the FM prediction at each env's current advance index (world frame)."""
        ar, idx = self._arange_n, self._pred_idx
        base = torch.cat([
            self._pred_root_pos[ar, idx], self._pred_root_quat[ar, idx],
            self._pred_root_lin[ar, idx], self._pred_root_ang[ar, idx],
        ], dim=-1)  # [N,13]
        ee = torch.cat([
            self._pred_frame_pos[ar, idx], self._pred_frame_quat[ar, idx],
            self._pred_frame_lin[ar, idx], self._pred_frame_ang[ar, idx],
        ], dim=-1)  # [N,K,13]
        return base, ee, self._pred_jp[ar, idx], self._pred_jv[ar, idx]

    @property
    def fm_active(self) -> torch.Tensor:
        """Per-env ``[N]`` bool: True where the FM (not the clip warmup) drives the reference."""
        return self._pred_valid

    # ------------------------------------------------------------------
    # Future window (FM prediction for valid envs, clip during warmup)
    # ------------------------------------------------------------------
    def get_future_world_base(
        self, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame future reference base pose+velocity, source-selected per env.

        Serves the FM prediction's future (``_pred_idx + offsets``, clamped to the F-step
        horizon) for warmed-up envs and the clip look-ahead for the rest -- mirroring the
        current-step source selection so the policy's ``future_base_*`` obs are consistent.
        """
        pos_c, quat_c, lin_c, ang_c = super().get_future_world_base(offsets)
        idx = torch.clamp(self._pred_idx[:, None] + offsets[None, :], max=self.ctx.F - 1)  # [N,K]
        ar = self._arange_n[:, None].expand_as(idx)
        use_fm = self._pred_valid[:, None, None]
        return (
            torch.where(use_fm, self._pred_root_pos[ar, idx], pos_c),
            torch.where(use_fm, self._pred_root_quat[ar, idx], quat_c),
            torch.where(use_fm, self._pred_root_lin[ar, idx], lin_c),
            torch.where(use_fm, self._pred_root_ang[ar, idx], ang_c),
        )

    # ------------------------------------------------------------------
    # Debug visualization (FM-prediction ribbon + source marker on the clip ribbon)
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Toggle the inherited clip ribbon plus the FM-prediction / source markers."""
        super()._set_debug_vis_impl(debug_vis)
        if not debug_vis:
            for m in (self._fm_markers, self._source_markers):
                if m is not None:
                    m.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Draw the clip ribbon (cylinders, warmup only) + FM prediction (spheres) + source marker."""
        super()._debug_vis_callback(event)
        if self.cfg.debug_vis_show_fm_prediction:
            self._draw_fm_prediction_ribbon()
        if self.cfg.debug_vis_show_source:
            self._draw_source_markers()

    def _clip_ribbon_env_mask(self) -> torch.Tensor:
        """Draw the inherited clip ribbon only for envs NOT yet tracking the FM (warmup).

        FM-active envs (``_pred_valid``) show the FM-prediction spheres instead."""
        return ~self._pred_valid

    def _draw_fm_prediction_ribbon(self) -> None:
        """Draw the FM's predicted world-frame trajectory as per-frame spheres (FM-active envs)."""
        if self._fm_markers is None:
            self._fm_markers = self._create_fm_markers()
            if self._fm_markers is None:
                return
        if not self._fm_pred_specs:
            self._fm_markers.set_visibility(False)
            return
        ids = torch.nonzero(self._pred_valid, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            self._fm_markers.set_visibility(False)
            return
        offs = self._fm_debug_offsets
        translations, marker_indices = [], []
        for proto_idx, (_name, _color, kind, k) in enumerate(self._fm_pred_specs):
            pts = (self._pred_root_pos[ids][:, offs] if kind == "base"
                   else self._pred_frame_pos[ids][:, offs, k]).reshape(-1, 3)
            translations.append(pts)
            marker_indices.append(torch.full((pts.shape[0],), proto_idx, dtype=torch.long, device=self.device))
        self._fm_markers.set_visibility(True)
        self._fm_markers.visualize(
            translations=torch.cat(translations, dim=0), marker_indices=torch.cat(marker_indices, dim=0),
        )

    def _draw_source_markers(self) -> None:
        """Float a marker above each robot: green = FM live, yellow = warming up (clip)."""
        if self._source_markers is None:
            self._source_markers = self._create_source_markers()
            if self._source_markers is None:
                return
        idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # 1 => warmup
        idx[self._pred_valid] = 0                                              # 0 => FM live
        pos = wp.to_torch(self.robot.data.root_pos_w).clone()
        pos[:, 2] += float(self.cfg.debug_vis_source_marker_height)
        self._source_markers.set_visibility(True)
        self._source_markers.visualize(translations=pos, marker_indices=idx)

    def _create_fm_markers(self):
        """Build a sphere-per-frame marker set for the FM-prediction ribbon."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN FmTrackingCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_fm_marker_radius)
        prototypes = {
            f"fm_{name}": sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
            for name, color, _kind, _k in self._fm_pred_specs
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/fm_pred", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN FmTrackingCommand] could not create FM markers: {exc}")
            return None

    def _create_source_markers(self):
        """Build the two-color (fm=green / warmup=yellow) tracking-source indicator set."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN FmTrackingCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_source_marker_radius)
        colors = {"fm": (0.10, 0.90, 0.20), "warmup": (0.95, 0.85, 0.10)}  # order fixes idx: 0=fm, 1=warmup
        prototypes = {
            name: sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            for name, color in colors.items()
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/fm_track_source", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN FmTrackingCommand] could not create source markers: {exc}")
            return None
