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
        )
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
        self._pred_valid = self._pred_valid & valid
        reinfer = (self.env.episode_length_buf % max(int(self.cfg.fm_update_every), 1)) == 0
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        infer_mask = valid & (reinfer | ~self._pred_valid | exhausted)
        self._fm_reinfer_mask = infer_mask
        advance = valid & ~infer_mask
        self._pred_idx = torch.where(
            advance, torch.clamp(self._pred_idx + 1, max=self.ctx.F - 1), self._pred_idx
        )
        if bool(infer_mask.any()):
            self._run_fm(infer_mask)

    @torch.no_grad()
    def _run_fm(self, mask: torch.Tensor) -> None:
        """Run batched FM inference for the masked envs (robot-state history) and store predictions."""
        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        rp = self._history.stack("root_pos")[ids]    # [V,N,3] live robot history
        rq = self._history.stack("root_quat")[ids]
        rl = self._history.stack("root_lin")[ids]
        ra = self._history.stack("root_ang")[ids]
        jp = self._history.stack("jp")[ids]          # [V,N,nj] FM order
        jv = self._history.stack("jv")[ids]
        frames = self._history.stack("frames")[ids] if self.ctx.num_extra_frames > 0 else None

        history = build_history_tokens(self.ctx, rp, rq, rl, ra, jp, jv, frames=frames)
        height = build_height_token(self.ctx, rp[:, -1, 2])          # robot root z above ground (now)
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        cmd_tokens = build_cmd_tokens(self.ctx, fut_cmd)
        # Anchor = the robot NOW (the most-recent history step IS the live robot); the FM
        # predicts in this xyz+yaw frame, so no re-centering is needed.
        p_ref, q_ref = rp[:, -1].clone(), rq[:, -1].clone()
        world = sample_world(self.ctx, history, height, cmd_tokens, p_ref, q_ref)

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
