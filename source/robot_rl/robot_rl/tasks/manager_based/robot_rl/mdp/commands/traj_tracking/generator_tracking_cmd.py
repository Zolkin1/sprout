# Copyright 2026 Zachary Olkin. All rights reserved.

"""Flow-matching generator-tracking command (world-frame CLF).

Extends :class:`MotionSequenceCommand`.  Every env is reset onto a pre-made clip
(supplying the FM velocity command ``target_vel`` and the spawn pose), but a
configurable fraction of envs track the **flow-matching generator's** predicted
future state instead of the clip reference:

  * **clip-tracking envs** (fraction ``clip_fraction``) follow the clip -- the
    stable teacher signal that lets the tracker bootstrap while its own rollout
    history (the FM input) is still garbage;
  * **FM-tracking envs** follow the generator's predicted next state, after the
    rolling history fills (they fall back to the clip during the warmup window).

Both sources drive a single **world-frame** CLF: the desired outputs
``y_des``/``dy_des`` and the measured outputs ``y_act``/``dy_act`` are expressed
in the world frame (no stance-anchor re-anchoring), so the inherited CLF reward,
observations, and terminations work uniformly for either source.  The world
frame is a single stationary inertial frame, valid for the CLF's finite-
difference ``vdot``.

Base-velocity convention note: the FM was trained on ``root_lin_vel_w`` (the
root *CoM* velocity, as recorded by ``record_data.py``), while the measured
``y_act`` base velocity comes from ``body_link_vel_w`` (the pelvis *link* frame).
For the pelvis these differ by ``omega x r_com`` -- a small, second-order
mismatch on the 6 base-velocity channels only (positions/orientations/joints are
exact).  Documented here rather than silently reconciled.
"""

from __future__ import annotations

import torch
import warp as wp

from robot_rl.flow_matching.online import (
    BatchedHistory,
    build_history_token,
    load_fm_context,
    sample_world,
)

from .motion_sequence_cmd import MotionSequenceCommand
from .generator_tracking_cmd_cfg import GeneratorTrackingCommandCfg


class GeneratorTrackingCommand(MotionSequenceCommand):
    """Track a flow-matching generator (or the clip) in the world frame."""

    cfg: GeneratorTrackingCommandCfg

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _post_init(self) -> None:
        """Set up the FM context, history buffer, source mask, and world index maps."""
        super()._post_init()
        n, dev = self.num_envs, self.device

        if not self.command_velocity_available:
            raise RuntimeError(
                "GeneratorTrackingCommand requires a clip library with target_vel "
                "(the FM velocity command). Regenerate clips with a generator that writes it."
            )
        if not self.cfg.fm_checkpoint:
            raise ValueError("GeneratorTrackingCommandCfg.fm_checkpoint must be set.")

        # The motion-sequence task is flat ground -> the FM history token has no
        # height-scan columns. (Update scan_dim here if a perceptive variant is added.)
        self.ctx = load_fm_context(
            self.cfg.fm_checkpoint,
            device=self.cfg.fm_device or dev,
            num_sampling_steps=self.cfg.fm_num_sampling_steps,
            scan_dim=0,
        )
        if self.ctx.future_cmd_dim <= 0:
            raise ValueError(
                "GeneratorTrackingCommand expects a future-command (cmd_array='target_vel') "
                "flow-matching checkpoint; the loaded checkpoint has no decoder command "
                "conditioning (cmd_dim=0)."
            )

        # The FM's extra frames must be exactly the CLF's end-effector output frames.
        ee_frames = self.resolve_ee_frames()
        if set(self.ctx.frame_names) != set(ee_frames):
            raise ValueError(
                f"FM checkpoint extra frames {self.ctx.frame_names} != command ee output "
                f"frames {ee_frames}; re-record/-train the FM with --state_frames matching them."
            )

        # -- history buffer + joint/frame permutations (robot order -> FM order) --
        robot_joints = list(self.robot.data.joint_names)
        self._joint_perm = torch.tensor(
            [robot_joints.index(j) for j in self.ctx.joint_names], dtype=torch.long, device=dev
        )
        self._ee_body_idx = (
            torch.tensor([self.robot.body_names.index(f) for f in self.ctx.frame_names],
                         dtype=torch.long, device=dev)
            if self.ctx.num_extra_frames > 0 else None
        )
        self._history = BatchedHistory(n, self.ctx.H, dev)

        # -- world-output index maps: frames = [base] + FM ee frames, in FM order --
        self._wf_frames = [self.cfg.base_frame_name] + list(self.ctx.frame_names)
        pos_list, ori_list, lin_list, ang_list = [], [], [], []
        for f in self._wf_frames:
            idx = self._frame_output_indices(f)
            pos_list.append(idx["pos"]); ori_list.append(idx["ori"])
            lin_list.append(idx["lin_vel"]); ang_list.append(idx["ang_vel"])
        self._wf_pos_flat = torch.cat(pos_list)   # [nf*3] -> y_des
        self._wf_ori_flat = torch.cat(ori_list)   # [nf*4] -> y_des
        self._wf_lin_flat = torch.cat(lin_list)   # [nf*3] -> dy_des
        self._wf_ang_flat = torch.cat(ang_list)   # [nf*3] -> dy_des

        # -- joint slots (clip-output order) + FM->clip joint permutation --
        joint_out_names = [n.split(":", 1)[1] for n in self.ordered_pos_output_names if n.startswith("joint:")]
        self._joint_slots_pos = torch.tensor(
            [self.ordered_pos_output_names.index(f"joint:{n}") for n in joint_out_names],
            dtype=torch.long, device=dev,
        )
        self._joint_slots_vel = torch.tensor(
            [self.ordered_vel_output_names.index(f"joint:{n}") for n in joint_out_names],
            dtype=torch.long, device=dev,
        )
        self._fm_jperm = torch.tensor(
            [self.ctx.joint_names.index(n) for n in joint_out_names], dtype=torch.long, device=dev
        )

        # -- per-env source mask (True => FM), prediction buffers, cadence state --
        self._fm_mask = torch.zeros(n, dtype=torch.bool, device=dev)  # re-rolled each reset
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

        # World-frame base reference buffer (read by reference_world_state()).
        self._ref_world_base = torch.zeros(n, 13, device=dev)
        # Constant frame for the world-frame measured outputs.
        self._zeros3 = torch.zeros(n, 3, device=dev)
        self._ident_quat = torch.zeros(n, 4, device=dev)
        self._ident_quat[:, 3] = 1.0  # xyzw identity

        self._last_step_seen = -1
        self._arange_n = torch.arange(n, device=dev)

    # ------------------------------------------------------------------
    # Per-step update (overrides the clip-only path)
    # ------------------------------------------------------------------

    def _update_command(self) -> None:
        """World-frame desired+measured outputs (clip or FM per env) and CLF update."""
        self._advance_step()
        self._snapshot_and_infer()

        # Clip desired outputs (stance-anchored). Joints are frame-invariant; the
        # base/ee Cartesian channels are overwritten with world values below.
        self.get_desired_outputs()

        # Clip world reference (base + ee), lifted through the clip's world anchor.
        wpos, wquat = self._world_anchor()
        self.ref_poses[:, :3] = wpos
        self.ref_poses[:, 3:7] = wquat
        clip_base = super().reference_world_state()                 # [N, 13 + 2nj]
        clip_ee = self.reference_world_frames(self.ctx.frame_names)  # [N, K, 13]

        # FM world reference at the current advance index.
        fm_base, fm_ee, fm_jp, fm_jv = self._fm_current_outputs()
        use_fm = self._fm_mask & self._pred_valid

        self._assemble_world_des(clip_base[:, :13], clip_ee, fm_base, fm_ee, fm_jp, fm_jv, use_fm)

        # Measured outputs in the world frame (identity reference -> no alignment).
        self.compute_measured_output(self._zeros3, self._ident_quat)

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot = vdot
        self.v = vcur

    # ------------------------------------------------------------------
    # History snapshot + flow-matching inference (once per control step)
    # ------------------------------------------------------------------

    def _snapshot_and_infer(self) -> None:
        """Append the current world state to history and (re-)run FM inference.

        Guarded to run once per control step (Isaac Lab computes commands twice
        per step). Freshly-reset envs (``episode_length_buf == 0``) get a fresh
        source assignment and an invalidated history window.
        """
        step = int(getattr(self.env, "common_step_counter", self._last_step_seen + 1))
        if step == self._last_step_seen:
            return
        self._last_step_seen = step

        just_reset = self.env.episode_length_buf == 0
        if bool(just_reset.any()):
            # Re-roll the per-env source (True => FM) for envs that just reset.
            roll_fm = torch.rand(self.num_envs, device=self.device) >= self.cfg.clip_fraction
            self._fm_mask = torch.where(just_reset, roll_fm, self._fm_mask)
            self._pred_valid = self._pred_valid & ~just_reset
            self._pred_idx = torch.where(just_reset, torch.zeros_like(self._pred_idx), self._pred_idx)

        # Snapshot the current world state (FM joint/frame order) into the history.
        perm = self._joint_perm
        snap = {
            "root_pos": wp.to_torch(self.robot.data.root_pos_w).clone(),
            "root_quat": wp.to_torch(self.robot.data.root_quat_w).clone(),     # xyzw
            "root_lin": wp.to_torch(self.robot.data.root_lin_vel_w).clone(),   # com vel
            "root_ang": wp.to_torch(self.robot.data.root_ang_vel_w).clone(),   # com ang vel
            "jp": wp.to_torch(self.robot.data.joint_pos)[:, perm].clone(),
            "jv": wp.to_torch(self.robot.data.joint_vel)[:, perm].clone(),
            "cmd": self.get_command_velocity().clone(),                        # past target_vel [N,3]
        }
        if self._ee_body_idx is not None:
            fpos = wp.to_torch(self.robot.data.body_pos_w)[:, self._ee_body_idx]
            fquat = wp.to_torch(self.robot.data.body_quat_w)[:, self._ee_body_idx]
            flink = wp.to_torch(self.robot.data.body_link_vel_w)[:, self._ee_body_idx]
            snap["frames"] = torch.cat([fpos, fquat, flink[..., :3], flink[..., 3:]], dim=-1).clone()  # [N,K,13]

        # Invalidate reset envs' windows, then append (this snapshot = step 1 of the new episode).
        self._history.invalidate(just_reset)
        self._history.append(snap)

        # Decide which FM envs to (re-)infer vs advance the prediction index.
        valid = self._history.valid()
        self._pred_valid = self._pred_valid & valid          # lose validity if the window broke
        fm_valid = valid & self._fm_mask
        reinfer = (step % max(int(self.cfg.fm_update_every), 1)) == 0
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        infer_mask = fm_valid & (reinfer | ~self._pred_valid | exhausted)
        advance = fm_valid & ~infer_mask
        self._pred_idx = torch.where(
            advance, torch.clamp(self._pred_idx + 1, max=self.ctx.F - 1), self._pred_idx
        )
        if bool(infer_mask.any()):
            self._run_fm(infer_mask)

    @torch.no_grad()
    def _run_fm(self, mask: torch.Tensor) -> None:
        """Run batched flow-matching inference for the masked envs and store predictions."""
        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        rp = self._history.stack("root_pos")[ids]    # [V,H,3]
        rq = self._history.stack("root_quat")[ids]
        rl = self._history.stack("root_lin")[ids]
        ra = self._history.stack("root_ang")[ids]
        jp = self._history.stack("jp")[ids]          # [V,H,nj] FM order
        jv = self._history.stack("jv")[ids]
        cmd = self._history.stack("cmd")[ids]         # [V,H,3] past target_vel
        frames = self._history.stack("frames")[ids] if self._ee_body_idx is not None else None

        token = build_history_token(self.ctx, rp, rq, rl, ra, jp, jv, cmd, frames=frames, scan=None)
        # Future command horizon (steps s+1..s+F) for the decoder, current-step subset.
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        # Anchor each prediction at the current base pose (the deployable relative frame).
        p_ref = rp[:, -1]
        q_ref = rq[:, -1]
        world = sample_world(self.ctx, token, p_ref, q_ref, future_cmd=fut_cmd)

        self._pred_root_pos[ids] = world["root_pos_w"]
        self._pred_root_quat[ids] = world["root_quat_w"]
        self._pred_root_lin[ids] = world["root_lin_vel_w"]
        self._pred_root_ang[ids] = world["root_ang_vel_w"]
        self._pred_jp[ids] = world["joint_pos"]
        self._pred_jv[ids] = world["joint_vel"]
        if self._ee_body_idx is not None:
            self._pred_frame_pos[ids] = world["frame_pos_w"]
            self._pred_frame_quat[ids] = world["frame_quat_w"]
            self._pred_frame_lin[ids] = world["frame_lin_vel_w"]
            self._pred_frame_ang[ids] = world["frame_ang_vel_w"]
        self._pred_idx[ids] = 0
        self._pred_valid[ids] = True

    def _fm_current_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather the FM prediction at each env's current advance index.

        Returns:
            ``(base[N,13], ee[N,K,13], jp[N,nj], jv[N,nj])`` in the world frame;
            ``jp``/``jv`` are in the FM joint order (permuted to clip slots later).
            Values are only meaningful where ``self._pred_valid`` is True.
        """
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

    # ------------------------------------------------------------------
    # World-frame desired-output assembly
    # ------------------------------------------------------------------

    def _assemble_world_des(
        self,
        clip_base: torch.Tensor,
        clip_ee: torch.Tensor,
        fm_base: torch.Tensor,
        fm_ee: torch.Tensor,
        fm_jp: torch.Tensor,
        fm_jv: torch.Tensor,
        use_fm: torch.Tensor,
    ) -> None:
        """Write world-frame ``y_des``/``dy_des`` (clip or FM per env) in the ordered layout.

        Args:
            clip_base: Clip world base ``[N,13]`` = ``[pos3, quat4, lin3, ang3]``.
            clip_ee: Clip world ee frames ``[N,K,13]`` (same per-frame layout).
            fm_base: FM world base ``[N,13]``.
            fm_ee: FM world ee frames ``[N,K,13]``.
            fm_jp: FM joint positions ``[N,nj]`` (FM order).
            fm_jv: FM joint velocities ``[N,nj]`` (FM order).
            use_fm: ``[N]`` bool; True => take the FM reference for that env.
        """
        n = self.num_envs
        clip_frames = torch.cat([clip_base[:, None, :], clip_ee], dim=1)  # [N, 1+K, 13]
        fm_frames = torch.cat([fm_base[:, None, :], fm_ee], dim=1)        # [N, 1+K, 13]
        sel = torch.where(use_fm[:, None, None], fm_frames, clip_frames)  # [N, nf, 13]

        self.y_des[:, self._wf_pos_flat] = sel[:, :, 0:3].reshape(n, -1)
        self.y_des[:, self._wf_ori_flat] = sel[:, :, 3:7].reshape(n, -1)
        self.dy_des[:, self._wf_lin_flat] = sel[:, :, 7:10].reshape(n, -1)
        self.dy_des[:, self._wf_ang_flat] = sel[:, :, 10:13].reshape(n, -1)

        # FM envs use the FM-predicted joints (reordered to clip slots); clip envs
        # keep the gathered clip joints (joints are frame-invariant).
        fm_jp_slots = fm_jp[:, self._fm_jperm]
        fm_jv_slots = fm_jv[:, self._fm_jperm]
        self.y_des[:, self._joint_slots_pos] = torch.where(
            use_fm[:, None], fm_jp_slots, self.y_des[:, self._joint_slots_pos]
        )
        self.dy_des[:, self._joint_slots_vel] = torch.where(
            use_fm[:, None], fm_jv_slots, self.dy_des[:, self._joint_slots_vel]
        )

        # Cache the selected world base for reference_world_state().
        self._ref_world_base = sel[:, 0, :]

    # ------------------------------------------------------------------
    # World-frame reference state (base) for obs / termination
    # ------------------------------------------------------------------

    def reference_world_state(self) -> torch.Tensor:
        """World-frame reference state ``[N, 3+4+3+3+2*nj]`` (selected source).

        Built directly from the world-frame base buffer + the world-frame
        ``y_des``/``dy_des`` joint channels (joints reordered to robot order),
        so it reflects the FM prediction for FM-tracking envs and the clip for
        clip-tracking envs -- replacing the clip-only stance-anchor lift.
        """
        joint_pos = self.y_des[:, self._base_idx["joint_pos"]]
        joint_vel = self.dy_des[:, self._base_idx["joint_vel"]]
        return torch.cat([self._ref_world_base, joint_pos, joint_vel], dim=-1)
