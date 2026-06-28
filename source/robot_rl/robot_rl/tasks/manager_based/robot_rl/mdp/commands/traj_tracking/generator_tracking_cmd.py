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

Base-velocity convention: every base/ee velocity is the **link** (actor-frame)
twist.  The FM is trained and conditioned on ``root_link_{lin,ang}_vel_w`` (see
``record_data.py``) and the measured ``y_act`` base velocity comes from
``body_link_vel_w`` -- the same pelvis link frame.  No CoM velocities are used
anywhere, so the FM reference and the CLF's measured outputs share one frame
(no ``omega x r_com`` mismatch).
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_euler_xyz, quat_mul, yaw_quat

from robot_rl.flow_matching.online import (
    BatchedHistory,
    build_cmd_tokens,
    build_cond_states,
    load_fm_context,
    sample_world,
)

from .motion_sequence_cmd import MotionSequenceCommand
from .generator_tracking_cmd_cfg import GeneratorTrackingCommandCfg
from .closed_loop_command import compute_closed_loop_command, roll_buckets
from .world_lift import lift_base_world_state


class GeneratorTrackingCommand(MotionSequenceCommand):
    """Track a flow-matching generator (or the clip) in the world frame."""

    cfg: GeneratorTrackingCommandCfg

    # Generator-tracking debug-viz marker handles (lazily built once the sim is
    # up, alongside the inherited clip-ribbon ``_debug_markers``).
    _fm_markers: object | None = None
    _source_markers: object | None = None
    _ref_vel_markers: object | None = None
    _act_vel_markers: object | None = None

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

        self.ctx = load_fm_context(
            self.cfg.fm_checkpoint,
            device=self.cfg.fm_device or dev,
            num_sampling_steps=self.cfg.fm_num_sampling_steps,
        )
        if self.ctx.cmd_dim <= 0:
            raise ValueError(
                "GeneratorTrackingCommand expects a command-conditioned (cmd_array='target_vel') "
                "flow-matching checkpoint; the loaded checkpoint has no command "
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
        # Body indices for the FM extra frames. The PARC-style design conditions on
        # AND predicts the same end-effector frames, so every history snapshot carries
        # them in the predicted-frame order ``ctx.frame_names`` (consumed both as clean
        # conditioning and, on the predicted side, by the tracker's ee_ref obs).
        self._hist_ee_body_idx = (
            torch.tensor([self.robot.body_names.index(f) for f in self.ctx.frame_names],
                         dtype=torch.long, device=dev)
            if self.ctx.num_extra_frames > 0 else None
        )
        self._history = BatchedHistory(n, self.ctx.N, dev)

        # Rebuild the world-frame index maps (base built them in resolve_ee_frames
        # order) so the frame order matches the FM prediction layout exactly --
        # the per-step select then aligns the clip and FM frames without a perm.
        self._build_world_frame_maps([self.cfg.base_frame_name] + list(self.ctx.frame_names))

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
        # Closed-loop FM envs (subset of _fm_mask): feed the FM a position-tracking
        # PD command instead of the clip's open-loop target_vel, and grade the
        # pose-drift termination against the clip (re-rolled each reset).
        self._closed_loop_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        # Per-step closed-loop command cache (computed for all envs; applied only
        # to closed-loop FM envs via the get_command_velocity* overrides).
        self._cmd_vel_cache = torch.zeros(n, 3, device=dev)
        # Clip world reference frames at the current step (cached in _update_command
        # for termination_reference_world_state); placeholder until first update.
        self._clip_world_frames_buf = self._wf_world_frames
        # Latched per-env episode-reset flag (set in reset(), consumed in
        # _snapshot_and_infer). episode_length_buf == 0 is unobservable here --
        # see reset() for why -- so we record the reset directly.
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

        # (_zeros3 / _ident_quat / _wf_world_frames are set by the base _post_init.)
        self._last_step_seen = -1
        self._arange_n = torch.arange(n, device=dev)

        # --- FM-prediction debug ribbon: map each (filtered) clip-ribbon frame
        # to its slot in the FM prediction buffers. The base frame reads the
        # predicted root; each ee frame reads its index in the predicted-frame
        # buffer; frames the FM does not predict are dropped from the ribbon.
        # ``self._debug_frame_names`` / ``_colors`` are built by the base _post_init.
        self._fm_pred_specs: list[tuple[str, tuple[float, float, float], str, int]] = []
        for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
            if name == self.cfg.base_frame_name:
                self._fm_pred_specs.append((name, color, "base", -1))
            elif name in self.ctx.frame_names:
                self._fm_pred_specs.append((name, color, "frame", self.ctx.frame_names.index(name)))
        # Subsample the F-step prediction horizon for the ribbon (same stride knob).
        self._fm_debug_offsets = torch.arange(
            0, self.ctx.F, max(1, int(self.cfg.debug_vis_stride)), dtype=torch.long, device=dev
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_ids=None):
        """Latch the per-env episode reset for the FM source machinery.

        ``_snapshot_and_infer`` cannot detect resets via ``episode_length_buf ==
        0``: Isaac Lab's ``_reset_idx`` calls ``command_manager.reset`` (which
        feeds ``_resample_command -> _update_command -> _snapshot_and_infer``)
        *before* it zeroes ``episode_length_buf``, and the once-per-step guard
        then skips the post-reset ``command_manager.compute``.  So the reset env
        is never observed at ``episode_length_buf == 0`` mid-rollout.  Record the
        reset here instead -- this hook runs with the exact reset ``env_ids`` for
        both the initial and every subsequent episode boundary.
        """
        if env_ids is None:
            self._pending_reset[:] = True
        else:
            self._pending_reset[env_ids] = True
        return super().reset(env_ids)

    # ------------------------------------------------------------------
    # Per-step update (overrides the clip-only path)
    # ------------------------------------------------------------------

    def _update_command(self) -> None:
        """World-frame desired+measured outputs (clip or FM per env) and CLF update."""
        self._advance_step()
        self._snapshot_and_infer()

        # Clip desired outputs (clip-world). Joints are frame-invariant; the
        # base/ee Cartesian channels are overwritten with world values below.
        self.get_desired_outputs()

        # Clip world reference (base + ee), lifted through the per-env placement
        # (frames in self._wf_frames == [base] + FM ee order).
        clip_frames = self._clip_world_frames()                      # [N, nf, 13]
        # Cache the clip world frames so the pose-drift termination can grade
        # closed-loop FM envs against the clip even when they track the FM.
        self._clip_world_frames_buf = clip_frames

        # FM world reference at the current advance index, in the same order.
        fm_base, fm_ee, fm_jp, fm_jv = self._fm_current_outputs()
        fm_frames = torch.cat([fm_base[:, None, :], fm_ee], dim=1)    # [N, nf, 13]
        use_fm = self._fm_mask & self._pred_valid

        # Per-env source select, then write the world-frame Cartesian channels.
        sel = torch.where(use_fm[:, None, None], fm_frames, clip_frames)
        self._write_world_des(sel)

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
        per step: once via ``command_manager.reset`` inside ``_reset_idx`` and
        once via ``command_manager.compute``). Freshly-reset envs (latched in
        :meth:`reset`) get a fresh source assignment and an invalidated history
        window.
        """
        step = int(getattr(self.env, "common_step_counter", self._last_step_seen + 1))
        if step == self._last_step_seen:
            return
        self._last_step_seen = step

        # Consume the latched reset flag (set in reset()); see that method for
        # why episode_length_buf == 0 cannot be used here.
        just_reset = self._pending_reset.clone()
        self._pending_reset[just_reset] = False
        if bool(just_reset.any()):
            # Re-roll the per-env source (True => FM) for envs that just reset, and
            # the open/closed-loop split within the FM envs (closed-loop is a subset
            # of FM).
            new_fm, new_cl = roll_buckets(
                torch.rand(self.num_envs, device=self.device),
                torch.rand(self.num_envs, device=self.device),
                self.cfg.clip_fraction,
                self.cfg.closed_loop_fraction,
            )
            self._fm_mask = torch.where(just_reset, new_fm, self._fm_mask)
            self._closed_loop_mask = torch.where(just_reset, new_cl, self._closed_loop_mask)
            self._pred_valid = self._pred_valid & ~just_reset
            self._pred_idx = torch.where(just_reset, torch.zeros_like(self._pred_idx), self._pred_idx)

        # Time-based clip->FM handoff (mirrors the MuJoCo runner's --clip_handoff_time):
        # when set, every env tracks the clip until it passes the handoff step, then the
        # FM -- a single mid-episode switch overriding the random clip_fraction roll, so
        # the IsaacLab side can reproduce the sim2sim handoff for comparison. The FM
        # history is filled for all envs every step above, so the prediction is valid
        # the moment an env crosses the handoff.
        if self.cfg.fm_handoff_time_s >= 0.0:
            handoff_step = round(self.cfg.fm_handoff_time_s / self.manager.dt)
            self._fm_mask = self._step >= handoff_step
            # Handoff is an eval/sim2sim-comparison path; keep every env open-loop
            # so the FM is fed the clip command (matches the MuJoCo runner).
            self._closed_loop_mask = torch.zeros_like(self._closed_loop_mask)

        # Closed-loop velocity command for this step (computed for all envs; the
        # get_command_velocity* overrides apply it only to closed-loop FM envs).
        # Must run before _run_fm so the FM command horizon reads the fresh value.
        self._refresh_command_velocity_cache()

        # Snapshot the current world state (FM joint/frame order) into the history.
        perm = self._joint_perm
        snap = {
            "root_pos": wp.to_torch(self.robot.data.root_pos_w).clone(),
            "root_quat": wp.to_torch(self.robot.data.root_quat_w).clone(),     # xyzw
            "root_lin": wp.to_torch(self.robot.data.root_link_lin_vel_w).clone(),  # link vel
            "root_ang": wp.to_torch(self.robot.data.root_link_ang_vel_w).clone(),  # link ang vel
            "jp": wp.to_torch(self.robot.data.joint_pos)[:, perm].clone(),
            "jv": wp.to_torch(self.robot.data.joint_vel)[:, perm].clone(),
        }
        if self._hist_ee_body_idx is not None:
            fpos = wp.to_torch(self.robot.data.body_pos_w)[:, self._hist_ee_body_idx]
            fquat = wp.to_torch(self.robot.data.body_quat_w)[:, self._hist_ee_body_idx]
            flink = wp.to_torch(self.robot.data.body_link_vel_w)[:, self._hist_ee_body_idx]
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
        rp = self._history.stack("root_pos")[ids]    # [V,N,3]
        rq = self._history.stack("root_quat")[ids]
        rl = self._history.stack("root_lin")[ids]
        ra = self._history.stack("root_ang")[ids]
        jp = self._history.stack("jp")[ids]          # [V,N,nj] FM order
        jv = self._history.stack("jv")[ids]
        frames = self._history.stack("frames")[ids] if self._hist_ee_body_idx is not None else None

        cond, known = build_cond_states(self.ctx, rp, rq, rl, ra, jp, jv, frames=frames)
        # Commanded-velocity profile over the window -> strided command tokens.
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        cmd_tokens = build_cmd_tokens(self.ctx, fut_cmd)
        # Default: anchor each prediction at the current base pose (step N-1).
        p_ref = rp[:, -1].clone()
        q_ref = rq[:, -1].clone()
        if self.cfg.fm_anchor_on_prev_traj:
            # Re-anchor on the *previous* prediction's ground pose so the reference
            # drifts globally with poor tracking (clip-like). Envs with a stored
            # prior prediction re-anchor; the rest (first inference after a reset /
            # window break -- _pred_valid still False here, before the set below)
            # stay robot-anchored. _pred_root_* / _pred_idx still hold the prior
            # trajectory and consumed index at this point (overwritten further down).
            had_prev = self._pred_valid[ids]              # [V] bool
            off = self._pred_idx[ids]                     # [V] currently-applied ref step
            prev_p = self._pred_root_pos[ids, off]        # [V, 3] world (prior traj)
            prev_q = self._pred_root_quat[ids, off]       # [V, 4] xyzw
            # Ground-corrected anchor: prior reference's xy + yaw, robot's z + tilt.
            p_ref_prev = p_ref.clone()
            p_ref_prev[:, :2] = prev_p[:, :2]
            dz_yaw = quat_mul(yaw_quat(prev_q), quat_conjugate(yaw_quat(q_ref)))  # world z-rot
            q_ref_prev = quat_mul(dz_yaw, q_ref)          # robot tilt, re-yawed to prior heading
            p_ref = torch.where(had_prev[:, None], p_ref_prev, p_ref)
            q_ref = torch.where(had_prev[:, None], q_ref_prev, q_ref)
        world = sample_world(self.ctx, cond, known, p_ref, q_ref, cmd_tokens=cmd_tokens)

        self._pred_root_pos[ids] = world["root_pos_w"]
        self._pred_root_quat[ids] = world["root_quat_w"]
        self._pred_root_lin[ids] = world["root_link_lin_vel_w"]
        self._pred_root_ang[ids] = world["root_link_ang_vel_w"]
        self._pred_jp[ids] = world["joint_pos"]
        self._pred_jv[ids] = world["joint_vel"]
        if self.ctx.num_extra_frames > 0:  # target (predicted) extra frames
            self._pred_frame_pos[ids] = world["frame_pos_w"]
            self._pred_frame_quat[ids] = world["frame_quat_w"]
            self._pred_frame_lin[ids] = world["frame_lin_vel_w"]
            self._pred_frame_ang[ids] = world["frame_ang_vel_w"]
        # Frames 0..N-1 reproduce the conditioning; novel future starts at N, so the
        # tracker's "current reference" index begins one step ahead of "now" (= N-1).
        self._pred_idx[ids] = self.ctx.N
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
    # Closed-loop velocity command (position-tracking PD over the clip)
    # ------------------------------------------------------------------

    def _use_pd_mask(self) -> torch.Tensor:
        """Per-env ``[N]`` bool: True where the closed-loop PD command is active.

        Closed-loop FM envs use the PD command once their history window is valid
        (the FM is live); during the brief warmup they track the clip, so they
        fall back to the clip ``target_vel`` -- keeping the command consistent
        with the reference each env is actually following.
        """
        return self._closed_loop_mask & self._history.valid()

    @torch.no_grad()
    def _refresh_command_velocity_cache(self) -> None:
        """Recompute the per-env closed-loop velocity command for this step.

        Tracks the clip's *moving* base position reference with feedforward (the
        clip ``target_vel``) plus a proportional position controller, expressed
        in the robot heading frame (the ``(vx, vy, wz)`` ``target_vel``
        convention).  Computed for every env (cheap); only closed-loop FM envs
        consume it (see :meth:`_use_pd_mask`).
        """
        # Clip base world pose at the current step.  Gather + lift directly (this
        # runs before _update_command repopulates the clip y_des/dy_des).
        y = self.manager.gather(self.manager.lib.y_des, self.clip_id, self._step)
        dy = self.manager.gather(self.manager.lib.dy_des, self.clip_id, self._step)
        pose = torch.cat([self.clip_world_pos, self.clip_world_quat], dim=-1)  # [N, 7]
        clip_base = lift_base_world_state(y, dy, pose, self._base_idx)         # [N, ...]
        clip_pos_w = clip_base[:, 0:3]
        clip_quat_w = clip_base[:, 3:7]                                        # xyzw

        v_ff = self.manager.gather_target_vel(self.clip_id, self._step)        # [N, 3] feedforward
        self._cmd_vel_cache = compute_closed_loop_command(
            v_ff,
            clip_pos_w,
            clip_quat_w,
            wp.to_torch(self.robot.data.root_pos_w),
            wp.to_torch(self.robot.data.root_quat_w),                          # xyzw
            wp.to_torch(self.robot.data.root_link_lin_vel_b),
            wp.to_torch(self.robot.data.root_link_ang_vel_b),
            kp_pos=self.cfg.pd_pos_kp,
            kd_pos=self.cfg.pd_pos_kd,
            kp_yaw=self.cfg.pd_yaw_kp,
            kd_yaw=self.cfg.pd_yaw_kd,
            max_lin=self.cfg.pd_max_lin_vel,
            max_ang=self.cfg.pd_max_ang_vel,
        )

    def get_command_velocity(self) -> torch.Tensor:
        """Per-env commanded base velocity ``[N, 3]``: PD command for closed-loop
        FM envs, the clip ``target_vel`` for the rest (clip + open-loop FM)."""
        base = super().get_command_velocity()
        use_pd = self._use_pd_mask()
        return torch.where(use_pd[:, None], self._cmd_vel_cache, base)

    def get_command_velocity_horizon(self, num_future: int) -> torch.Tensor:
        """Commanded-velocity horizon ``[N, num_future, 3]`` (the FM conditioning).

        Closed-loop FM envs hold the current PD command constant across the
        horizon; the rest use the clip's open-loop ``target_vel`` horizon.
        """
        base = super().get_command_velocity_horizon(num_future)
        use_pd = self._use_pd_mask()
        pd = self._cmd_vel_cache[:, None, :].expand_as(base)
        return torch.where(use_pd[:, None, None], pd, base)

    def get_future_command_velocity(self, offsets: torch.Tensor) -> torch.Tensor:
        """Look-ahead commanded velocity ``[N, K, 3]`` (the ``future_command_velocity`` obs).

        Closed-loop FM envs broadcast the current PD command over the look-ahead
        window; the rest use the clip's open-loop ``target_vel``.
        """
        base = super().get_future_command_velocity(offsets)
        use_pd = self._use_pd_mask()
        pd = self._cmd_vel_cache[:, None, :].expand_as(base)
        return torch.where(use_pd[:, None, None], pd, base)

    def termination_reference_world_state(self) -> torch.Tensor:
        """Reference state for the pose-drift termination, clip-relative for
        closed-loop FM envs.

        Open-loop FM envs keep the FM reference (the original behavior); clip
        envs already use the clip.  For closed-loop FM envs the base row is
        overwritten with the clip world base pose so a robot that stalls while
        the clip marches on accrues drift and terminates.
        """
        state = self.reference_world_state()
        cl = self._closed_loop_mask
        if bool(cl.any()):
            state = state.clone()
            state[cl, 0:13] = self._clip_world_frames_buf[cl, 0, :]
        return state

    # ------------------------------------------------------------------
    # Introspection for play-time logging / plots
    # ------------------------------------------------------------------

    @property
    def fm_active(self) -> torch.Tensor:
        """Per-env ``[N]`` bool: True where the FM prediction (not the clip) drives the reference.

        This is the exact authoritative-source mask used in :meth:`_update_command`
        (``_fm_mask & _pred_valid``): True only for envs assigned to the generator
        *and* past their history warmup.  Flips False->True at the clip->FM handoff
        (or end of warmup) and True->False on a window break -- the boundary the
        play plots mark with a transition line.
        """
        return self._fm_mask & self._pred_valid

    @property
    def fm_pred_idx(self) -> torch.Tensor:
        """Per-env ``[N]`` long: the currently-applied step within the F-step prediction window.

        Reset to :attr:`fm_novel_start` (``ctx.N``) on each fresh inference and
        advanced by one per control step in between, so a non-increase between
        consecutive steps marks a re-inference -- used to pick out distinct
        predictions for the peel-off plot.
        """
        return self._pred_idx

    @property
    def fm_window_len(self) -> int:
        """Length ``F`` of the FM prediction window (steps stored per inference)."""
        return int(self.ctx.F)

    @property
    def fm_novel_start(self) -> int:
        """Window index ``N`` where the novel future begins (steps ``0..N-1`` reproduce history)."""
        return int(self.ctx.N)

    @torch.no_grad()
    def predicted_outputs_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Full F-step FM prediction expressed in the ``y_des`` / ``dy_des`` output layout.

        Mirrors the single-index conversion in :meth:`_update_command` but for
        every window step at once, so the play-time logger can draw the predicted
        trajectories peeling off the tracked reference.

        Returns:
            ``(y_win, dy_win)`` of shapes ``[N, F, P_pos]`` / ``[N, F, P_vel]``,
            where ``[:, j]`` is the predicted reference at window step ``j`` in the
            same channel order (``ordered_pos_output_names`` /
            ``ordered_vel_output_names``) as ``self.y_des`` / ``self.dy_des``.
            Only meaningful for envs where :attr:`fm_active` is True (the buffers
            are stale/garbage for clip- and warmup-tracking envs).
        """
        n, f = self.num_envs, self.ctx.F
        y_win = torch.zeros(n, f, self.y_des.shape[1], device=self.device)
        dy_win = torch.zeros(n, f, self.dy_des.shape[1], device=self.device)

        # Assemble [N, F, nf, 13] world frames ([base] + ee) in self._wf_frames order.
        base13 = torch.cat(
            [self._pred_root_pos, self._pred_root_quat, self._pred_root_lin, self._pred_root_ang],
            dim=-1,
        )  # [N, F, 13]
        frames = base13[:, :, None, :]
        if self.ctx.num_extra_frames > 0:
            ee13 = torch.cat(
                [self._pred_frame_pos, self._pred_frame_quat,
                 self._pred_frame_lin, self._pred_frame_ang],
                dim=-1,
            )  # [N, F, K, 13]
            frames = torch.cat([frames, ee13], dim=2)  # [N, F, nf, 13]

        # Scatter the world-frame Cartesian channels into the output layout (same
        # flat index maps + reshape order as _write_world_des, with an extra F axis).
        y_win[:, :, self._wf_pos_flat] = frames[..., 0:3].reshape(n, f, -1)
        y_win[:, :, self._wf_ori_flat] = frames[..., 3:7].reshape(n, f, -1)
        dy_win[:, :, self._wf_lin_flat] = frames[..., 7:10].reshape(n, f, -1)
        dy_win[:, :, self._wf_ang_flat] = frames[..., 10:13].reshape(n, f, -1)

        # Joints: FM order -> clip output slots (same permutation as _update_command).
        y_win[:, :, self._joint_slots_pos] = self._pred_jp[:, :, self._fm_jperm]
        dy_win[:, :, self._joint_slots_vel] = self._pred_jv[:, :, self._fm_jperm]
        return y_win, dy_win

    # ------------------------------------------------------------------
    # Future window (per-env source-selected: FM prediction vs clip)
    # ------------------------------------------------------------------

    def get_future_world_base(
        self, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame future reference base pose+velocity, source-selected per env.

        Overrides the clip-only base implementation so the look-ahead reference
        (consumed by the ``future_base_pose`` / ``future_base_velocity`` policy
        observations) comes from whichever source actually drives each env's
        reference: the **flow-matching generator's** predicted future for envs
        currently tracking it (``_fm_mask & _pred_valid``), the **clip** for the
        rest (clip-tracking and FM-warmup envs).  This mirrors the current-step
        source selection in :meth:`_update_command`, so a deployed FM env never
        sees privileged clip look-ahead -- only the generator's own prediction
        (available at deployment from the same FM + tracker pair).

        The FM prediction stores ``F`` window steps anchored at the current base
        (steps ``0..N-1`` reproduce the recent past; novel future is ``N..F-1``);
        ``_pred_idx`` is the currently-tracked reference step (starts at ``N``, one
        ahead of "now" = ``N-1``), so future offset ``o`` reads prediction index
        ``_pred_idx + o`` (clamped to the predicted horizon ``F - 1``, the FM-side
        analogue of the clip's clamp-to-clip-end).

        Returns:
            ``(pos_w, quat_w, lin_w, ang_w)`` of shapes ``[N, K, 3]``,
            ``[N, K, 4]`` (xyzw), ``[N, K, 3]``, ``[N, K, 3]`` -- same layout and
            world frame as the base clip-only implementation.
        """
        pos_c, quat_c, lin_c, ang_c = super().get_future_world_base(offsets)

        # FM-predicted future at the same offsets (world frame, current-base anchored).
        idx = torch.clamp(self._pred_idx[:, None] + offsets[None, :], max=self.ctx.F - 1)  # [N, K]
        ar = self._arange_n[:, None].expand_as(idx)
        pos_f = self._pred_root_pos[ar, idx]    # [N, K, 3]
        quat_f = self._pred_root_quat[ar, idx]  # [N, K, 4] (xyzw)
        lin_f = self._pred_root_lin[ar, idx]    # [N, K, 3]
        ang_f = self._pred_root_ang[ar, idx]    # [N, K, 3]

        use_fm = (self._fm_mask & self._pred_valid)[:, None, None]
        return (
            torch.where(use_fm, pos_f, pos_c),
            torch.where(use_fm, quat_f, quat_c),
            torch.where(use_fm, lin_f, lin_c),
            torch.where(use_fm, ang_f, ang_c),
        )

    # ------------------------------------------------------------------
    # Debug visualization (extends the inherited clip ribbon)
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Toggle the clip ribbon (inherited) plus the generator-tracking layers.

        The FM-prediction ribbon and per-env source markers are created lazily
        in :meth:`_debug_vis_callback` (the sim may not be up yet); here we only
        hide them when debug viz is turned off.
        """
        super()._set_debug_vis_impl(debug_vis)
        if not debug_vis:
            for m in (self._fm_markers, self._source_markers,
                      self._ref_vel_markers, self._act_vel_markers):
                if m is not None:
                    m.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Draw the clip ribbon (cylinders) + FM prediction (spheres) + source markers.

        The inherited callback draws the clip reference ribbon for every env (the
        clip each env was reset on, advancing by step index). On top of that we
        overlay, per the cfg flags, the flow-matching generator's predicted
        trajectory and a per-env indicator of which reference is authoritative.
        """
        super()._debug_vis_callback(event)
        if self.cfg.debug_vis_show_fm_prediction:
            self._draw_fm_prediction_ribbon()
        if self.cfg.debug_vis_show_source:
            self._draw_source_markers()
        if self.cfg.debug_vis_show_base_velocity:
            self._draw_base_velocity_arrows()

    def _clip_ribbon_env_mask(self) -> torch.Tensor:
        """Draw the clip ribbon only where the env is NOT actively tracking the FM.

        FM-active envs (``_fm_mask & _pred_valid``) show the FM-prediction
        spheres instead; clip-tracking and FM-warmup envs (still following the
        clip) keep the clip ribbon.
        """
        return ~(self._fm_mask & self._pred_valid)

    def _draw_fm_prediction_ribbon(self) -> None:
        """Draw the FM generator's predicted world trajectory as per-frame spheres.

        Only the envs currently tracking the FM (``_fm_mask & _pred_valid``) are
        drawn; the predicted positions are read straight from the world-frame
        prediction buffers (subsampled over the F-step horizon). Sphere colors
        match the clip ribbon's per-frame colors, so the FM prediction (spheres)
        can be compared directly to the clip baseline (cylinders).
        """
        if self._fm_markers is None:
            self._fm_markers = self._create_fm_markers()
            if self._fm_markers is None:
                return
        if not self._fm_pred_specs:
            self._fm_markers.set_visibility(False)
            return

        use_fm = self._fm_mask & self._pred_valid
        ids = torch.nonzero(use_fm, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            self._fm_markers.set_visibility(False)
            return

        offs = self._fm_debug_offsets
        translations, marker_indices = [], []
        for proto_idx, (_name, _color, kind, k) in enumerate(self._fm_pred_specs):
            if kind == "base":
                pts = self._pred_root_pos[ids][:, offs]        # [V, Ns, 3]
            else:
                pts = self._pred_frame_pos[ids][:, offs, k]    # [V, Ns, 3]
            pts = pts.reshape(-1, 3)
            translations.append(pts)
            marker_indices.append(
                torch.full((pts.shape[0],), proto_idx, dtype=torch.long, device=self.device)
            )

        self._fm_markers.set_visibility(True)
        self._fm_markers.visualize(
            translations=torch.cat(translations, dim=0),
            marker_indices=torch.cat(marker_indices, dim=0),
        )

    def _draw_source_markers(self) -> None:
        """Float a colored marker above each robot showing its tracking source.

        Green => open-loop FM (history full, prediction live); blue => closed-loop
        FM (PD command active); yellow => assigned to the FM but still warming up
        (falling back to the clip); white => clip.
        """
        if self._source_markers is None:
            self._source_markers = self._create_source_markers()
            if self._source_markers is None:
                return

        use_fm = self._fm_mask & self._pred_valid
        warmup = self._fm_mask & ~self._pred_valid
        closed = use_fm & self._closed_loop_mask
        idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # 1 => clip
        idx[use_fm] = 0   # 0 => fm (open-loop)
        idx[warmup] = 2   # 2 => warmup
        idx[closed] = 3   # 3 => fm (closed-loop)

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
            print(f"[WARN GeneratorTrackingCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_fm_marker_radius)
        prototypes = {
            f"fm_{name}": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
            for name, color, _kind, _k in self._fm_pred_specs
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/generator_pred", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN GeneratorTrackingCommand] could not create FM markers: {exc}")
            return None

    def _create_source_markers(self):
        """Build the three-color (fm / clip / warmup) tracking-source indicator set."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN GeneratorTrackingCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_source_marker_radius)
        # Insertion order fixes the marker_indices mapping:
        # 0 => fm (open-loop), 1 => clip, 2 => warmup, 3 => fm (closed-loop).
        colors = {
            "fm": (0.10, 0.90, 0.20),
            "clip": (0.90, 0.90, 0.90),
            "warmup": (0.95, 0.85, 0.10),
            "closed": (0.10, 0.45, 0.95),
        }
        prototypes = {
            name: sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            for name, color in colors.items()
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/track_source", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN GeneratorTrackingCommand] could not create source markers: {exc}")
            return None

    def _draw_base_velocity_arrows(self) -> None:
        """Draw two world-frame base-velocity arrows per env, floated above the base.

        Mirrors the stock velocity-command arrows: a **green** arrow for the
        joystick-level velocity *command* (``target_vel`` = ``(vx, vy, wz)`` -- the
        signal the velocity-tracking reward grades and the FM is conditioned on,
        *not* the achieved reference velocity) and a **blue** arrow for the
        robot's actual base velocity. ``target_vel`` is stored in the robot's
        heading (base-yaw) frame, so it is rotated into the world frame by the
        base yaw before drawing; both arrows then point along their world heading
        and grow in length with speed, so the gap between them is the
        joystick-level tracking error at a glance. Drawn for every env (clip- and
        FM-tracking alike).
        """
        if not self.command_velocity_available:
            return  # library has no target_vel -> nothing meaningful to draw
        if self._ref_vel_markers is None:
            self._ref_vel_markers, self._act_vel_markers = self._create_vel_markers()
            if self._ref_vel_markers is None:
                return

        # Green: joystick-level command (target_vel, heading frame) rotated to the
        # world frame by the base yaw -- the exact signal the reward/FM consume,
        # not the achieved reference velocity. Blue: actual world-frame base vel.
        cmd_vel = self.get_command_velocity()                          # [N, 3] = (vx, vy, wz), heading frame
        cmd_xy = torch.zeros_like(cmd_vel)
        cmd_xy[:, :2] = cmd_vel[:, :2]
        base_yaw = yaw_quat(wp.to_torch(self.robot.data.root_quat_w))   # xyzw
        ref_lin_w = quat_apply(base_yaw, cmd_xy)                        # [N, 3], world frame
        act_lin_w = wp.to_torch(self.robot.data.root_lin_vel_w)

        base_pos = wp.to_torch(self.robot.data.root_pos_w).clone()
        base_pos[:, 2] += float(self.cfg.debug_vis_velocity_height)

        ref_scale, ref_quat = self._resolve_xy_velocity_to_arrow(ref_lin_w[:, :2], self._ref_vel_markers)
        act_scale, act_quat = self._resolve_xy_velocity_to_arrow(act_lin_w[:, :2], self._act_vel_markers)

        self._ref_vel_markers.set_visibility(True)
        self._act_vel_markers.set_visibility(True)
        self._ref_vel_markers.visualize(translations=base_pos, orientations=ref_quat, scales=ref_scale)
        self._act_vel_markers.visualize(translations=base_pos, orientations=act_quat, scales=act_scale)

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity_w: torch.Tensor, markers
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a world-frame XY base velocity to an x-arrow's ``(scale, quat)``.

        The arrow's x-scale grows with speed (``debug_vis_velocity_scale`` gain)
        and it is yawed to the velocity's world heading. The quaternion is wxyz
        in the world frame -- the convention the marker visualizer consumes --
        so no robot-orientation compose is needed (the input is already world).
        """
        default_scale = markers.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity_w.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity_w, dim=1) * float(self.cfg.debug_vis_velocity_scale)
        heading = torch.atan2(xy_velocity_w[:, 1], xy_velocity_w[:, 0])
        zeros = torch.zeros_like(heading)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading)  # wxyz, world frame
        return arrow_scale, arrow_quat

    def _create_vel_markers(self):
        """Build the green (reference) + blue (actual) base-velocity arrow marker sets."""
        try:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
        except Exception as exc:  # sim not up yet
            print(f"[WARN GeneratorTrackingCommand] could not import arrow markers: {exc}")
            return None, None
        try:
            ref = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/gen_ref_vel"))
            act = VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/gen_act_vel"))
            return ref, act
        except Exception as exc:
            print(f"[WARN GeneratorTrackingCommand] could not create velocity markers: {exc}")
            return None, None
