# Copyright 2026 Zachary Olkin. All rights reserved.

"""Closed-loop, robot-state-conditioned flow-matching tracking command (world-frame CLF).

Extends :class:`MotionSequenceCommand`: every env tracks the **flow-matching generator**,
which conditions on the **live robot state** history (the N most-recent measured steps, in
the FM's xyz+yaw ground frame), an exteroception token, and the future velocity command,
and predicts the F-step ego-anchored clip future. Because the FM is anchored on the robot's
own xyz+yaw pose each re-plan and predicts node 0 = "now" (offset ≈ 0), no re-centering
hack is needed -- the prediction is already in the robot's frame. This is the deploy
counterpart of the recorded ``ego_anchored_clip_future`` training data (each re-inference
== a snap).

**The reference is the FM and only the FM, from control step 0.** There is no clip warmup
fallback: before the N-step history window fills, the generator is sampled with the learned
``null_history`` embedding it was trained with (``null_cond_prob``), which is the
in-distribution way to cover the post-reset steps and the only one deployment has. What the
clip library still supplies is exactly two things, both unavoidable in simulation:

1. the **reset spawn pose**, and the terrain mesh built from the same library;
2. the **``target_vel`` command** -- the joystick stand-in that conditions the FM and feeds
   the policy's ``command_window``. On hardware this is a real joystick.

Everything else is generated. ``strict_fm_only`` fills every output channel the FM does not
write with NaN each step, so an accidental read of clip-derived data fails loudly instead of
quietly tracking the wrong thing.

The desired/measured outputs are in the world frame, so the inherited CLF and observations
work uniformly. All base/ee velocities are the **link** (actor-frame) twist (the FM is
conditioned on ``root_link_{lin,ang}_vel_w``; ``y_act`` comes from ``body_link_vel_w``).
"""

from __future__ import annotations

import os

import torch
import warp as wp

from isaaclab.envs.mdp import height_scan as mdp_height_scan
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import depth_obs_stack

from robot_rl.flow_matching.online import (
    BatchedHistory,
    build_cmd_tokens,
    build_extero_tokens,
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
                "No FM checkpoint: pass --fm_checkpoint <path/to/best.pt>. There is "
                "deliberately no default -- it must be a trained robot-state-conditioned "
                "(target_source='ego_anchored_clip_future') flow-matching .pt whose horizon "
                "covers this task's reference window, and a stale one would load and predict "
                "and be silently wrong. Same reasoning as --clip having no default."
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
        if self.cfg.tracking_mode == "body":
            raise ValueError(
                "FmTrackingCommand does not support tracking_mode='body'. The body CLF grades "
                "13 '@root' body links, but the generator predicts only the base, the joints "
                "and its end-effector POSITIONS -- so those channels would be NaN under "
                "strict_fm_only. Enabling it needs an FM recorded with --state_frames "
                "covering G1_BODY_TRACK_FRAMES, which also widens the state vector.\n"
                "Note this is about THIS task's CLF, which here is diagnostic only (no "
                "reward reads it). It says nothing about the teacher you are loading: a "
                "body-mode teacher runs fine on the default joint-mode tracker, because the "
                "actor's observation groups are identical in both modes."
            )
        self._check_snap_cadence()

        # The FM's extra frames must be exactly the CLF's end-effector output frames.
        ee_frames = self.resolve_ee_frames()
        if set(self.ctx.frame_names) != set(ee_frames):
            raise ValueError(
                f"FM checkpoint extra frames {self.ctx.frame_names} != command ee output "
                f"frames {ee_frames}; re-record/-train the FM with matching --state_frames."
            )

        # Rolling per-env history of the LIVE ROBOT state (N steps), in FM joint/frame order.
        self._history = BatchedHistory(n, self.ctx.N, dev)

        # World-frame index maps in FM frame order ([base] + FM ee), so the per-step write
        # aligns the FM frames with the output channels without a permutation.
        self._build_world_frame_maps([self.cfg.base_frame_name] + list(self.ctx.frame_names))

        # Joint slots (clip-output order) + the two joint permutations. These are DIFFERENT
        # maps and mixing them silently scrambles the reference:
        #   _fm_jperm        : FM order -> clip-OUTPUT slot order (writing y_des/dy_des)
        #   _robot_to_fm_jperm: robot order -> FM order (reading robot.data)
        #   _fm_to_robot_jperm: FM order -> ROBOT order (serving get_future_joint_state,
        #                       whose contract is the robot's own joint order)
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
        # Inverse of the above: for a permutation, argsort IS the inverse.
        self._fm_to_robot_jperm = torch.argsort(self._robot_to_fm_jperm)
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
        self._pred_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self._pred_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._fm_reinfer_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._last_step_seen = -1
        self._arange_n = torch.arange(n, device=dev)
        # Largest look-ahead offset validated against the horizon so far (see
        # :meth:`_check_horizon`); -1 = nothing checked yet.
        self._checked_max_offset = -1

        # The FM predicts NO orientation or twist for its extra frames (position only), so
        # those output channels are never written. Constant zeros keep the scatter in
        # ``_write_world_des`` uniform; ``strict_fm_only`` then NaNs them like any other
        # unwritten channel.
        self._ee_zero_quat = torch.zeros(n, max(k, 1), 4, device=dev)
        self._ee_zero_quat[..., 3] = 1.0
        self._ee_zero_vec = torch.zeros(n, max(k, 1), 3, device=dev)

        # Last applied reference, for history_source="reference_partial" (see
        # :meth:`_stash_applied_reference`). None until the first step has produced one.
        self._applied_jp: torch.Tensor | None = None
        self._applied_jv: torch.Tensor | None = None
        self._applied_frames: torch.Tensor | None = None
        if self.ctx.history_source == "reference_partial":
            print("[INFO FmTrackingCommand] history_source='reference_partial': the FM "
                  "conditions on its OWN last-applied joints/ee (measured root only).")

        self._build_strict_masks()

        # --- FM-prediction debug ribbon: map each ribbon frame (base built by the base
        # _post_init, re-ordered above) to its slot in the FM prediction buffers.
        self._fm_pred_specs: list[tuple[str, tuple[float, float, float], str, int]] = []
        for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
            if name == self.cfg.base_frame_name:
                self._fm_pred_specs.append((name, color, "base", -1))
            elif name in self.ctx.frame_names:
                self._fm_pred_specs.append((name, color, "frame", self.ctx.frame_names.index(name)))
        # Show EVERY predicted node in the FM ribbon (dense), independent of the
        # clip ribbon's debug_vis_stride subsampling.
        self._fm_debug_offsets = torch.arange(0, self.ctx.F, dtype=torch.long, device=dev)

    def _check_snap_cadence(self) -> None:
        """Fail if the tracker re-plans at a different rate than the recording snapped.

        The FM's horizon is sized from the re-plan period (see
        ``FlowMatchingConfig.future_len``), and the training targets were cut at the
        recording's ``fm_snap_every``. A tracker running a different period is both off the
        training distribution and possibly short of horizon, with nothing in the shapes to
        reveal it.
        """
        recorded = getattr(self.ctx, "fm_snap_every", None)
        if recorded is None:
            print("[WARN FmTrackingCommand] the FM checkpoint predates the fm_snap_every "
                  f"record; cannot verify it matches fm_update_every={self.cfg.fm_update_every}.")
            return
        if int(recorded) != int(self.cfg.fm_update_every):
            raise ValueError(
                f"fm_update_every={self.cfg.fm_update_every} but the FM was trained on data "
                f"snapped every {recorded} steps. The generator's horizon is sized from the "
                "re-plan period and its targets were cut at the snap cadence, so these must "
                "agree. Set commands.traj_ref.fm_update_every to the recorded value, or "
                "re-record + retrain at the cadence you want."
            )

    def _build_strict_masks(self) -> None:
        """Index the output channels the FM does NOT write, for ``strict_fm_only``.

        The FM writes the base + ee Cartesian channels (``_wf_*_flat``) and the joints
        (``_joint_slots_*``). Everything else in ``y_des``/``dy_des`` -- the COM block, the
        ``@root`` body-link blocks, any frame the checkpoint does not predict -- is
        clip-derived and must not be read here.
        """
        dev = self.device

        def _complement(width: int, written: list[torch.Tensor]) -> torch.Tensor:
            keep = torch.ones(width, dtype=torch.bool, device=dev)
            for idx in written:
                keep[idx] = False
            return torch.nonzero(keep, as_tuple=False).squeeze(-1)

        self._unwritten_pos = _complement(
            self.y_des.shape[1], [self._wf_pos_flat, self._wf_ori_flat, self._joint_slots_pos]
        )
        self._unwritten_vel = _complement(
            self.dy_des.shape[1], [self._wf_lin_flat, self._wf_ang_flat, self._joint_slots_vel]
        )

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
        """World-frame desired outputs from the FM alone, then measured outputs + CLF."""
        self._advance_step()          # clip step counter -- drives the target_vel look-ahead only
        # Sit out the reset-time resample, for the same reason the re-anchor snapshot does
        # (see ``MotionSequenceCommand._resample_command``): that call runs with a bogus
        # ``_step``, and ``_snapshot_and_infer`` is likewise guarded to once per
        # ``common_step_counter`` -- so a history row appended here would both be taken
        # around the reset AND stamp the guard, suppressing the real snapshot in the
        # ``compute()`` that follows. Everything below is harmless to run: it is overwritten
        # by that same following call.
        if not self._resampling:
            self._snapshot_and_infer()

        fm_base, fm_ee, fm_jp, fm_jv = self._fm_current_outputs()
        self._write_world_des(torch.cat([fm_base[:, None, :], fm_ee], dim=1))   # [N, nf, 13]
        self.y_des[:, self._joint_slots_pos] = fm_jp[:, self._fm_jperm]
        self.dy_des[:, self._joint_slots_vel] = fm_jv[:, self._fm_jperm]
        self._stash_applied_reference(fm_jp, fm_jv, fm_ee)
        if self.cfg.strict_fm_only:
            self.y_des[:, self._unwritten_pos] = torch.nan
            self.dy_des[:, self._unwritten_vel] = torch.nan

        # Measured outputs in the world frame (identity reference -> no alignment), then the
        # CLF. Deliberately AFTER the strict_fm_only fill: the graded channel subset should
        # contain only channels the FM writes, so if one of the NaNs reaches eta then ``v``
        # goes NaN and says so -- which is the whole point of the barrier. (Joint mode grades
        # the base, the world ee frames and the joints; the NaNs are the ``@root`` blocks.)
        self.compute_measured_output(self._zeros3, self._ident_quat)
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot, self.v = vdot, vcur

    def _write_global_base_des(self) -> None:
        """Disabled: there is no global clip base to grade against here.

        On the teacher this re-points ``y_des``'s base channels at the GLOBALLY-placed clip,
        so the base says WHERE to be while the links say WHAT SHAPE -- and the reward then
        reads the same reference the drift termination fires on.

        Neither half of that reasoning survives on the generator. The FM's prediction IS the
        reference, and it is anchored on the robot by construction; there is no
        globally-placed clip the robot is supposed to be at, and the drift termination that
        motivated the split is not in this task's termination set. Letting the base run would
        put clip data straight into the channel the CLF grades hardest -- exactly what
        ``strict_fm_only`` exists to catch.
        """

    def _stash_applied_reference(
        self, fm_jp: torch.Tensor, fm_jv: torch.Tensor, fm_ee: torch.Tensor
    ) -> None:
        """Keep this step's applied reference for a partially autoregressive history.

        Only used when the checkpoint says ``history_source="reference_partial"``; the next
        step's :meth:`_robot_snapshot` reads it in place of the measured joints/ee. Storing
        it here (in FM order, as ``_run_fm`` wants it) rather than re-deriving from ``y_des``
        avoids a round trip through the clip-output permutation.

        The one-step lag is the convention, not an oversight: the recorder reads ``y_des``
        pre-step and so captures step ``t-1``'s reference too. Autoregression cannot do
        better -- this step's reference is what we are in the middle of computing.
        """
        if self.ctx.history_source != "reference_partial":
            return
        self._applied_jp = fm_jp.clone()
        self._applied_jv = fm_jv.clone()
        # [N, K, 13] world frames -> the ee half only; the FM predicts position, and the
        # snapshot's frame layout is [pos(3), quat(4), lin(3), ang(3)].
        self._applied_frames = fm_ee.clone() if self.ctx.num_extra_frames > 0 else None

    # ------------------------------------------------------------------
    # Robot-state history snapshot + inference (once per control step)
    # ------------------------------------------------------------------
    def _robot_snapshot(self, ids: torch.Tensor | slice = slice(None)) -> dict[str, torch.Tensor]:
        """Snapshot one history row (FM joint/frame order, world frame) for ``ids``.

        The root pose/twist is ALWAYS the measured robot -- that is what re-grounds the
        generator in reality every re-plan, and it is also the frame everything else is
        expressed in. Under ``history_source="reference_partial"`` the joints and ee
        positions instead come from the reference applied last step (the generator's own
        output), making the model partially autoregressive on the posture channels.
        """
        d = self.robot.data
        snap = {
            "root_pos": wp.to_torch(d.root_pos_w)[ids].clone(),
            "root_quat": wp.to_torch(d.root_quat_w)[ids].clone(),           # xyzw
            "root_lin": wp.to_torch(d.root_link_lin_vel_w)[ids].clone(),
            "root_ang": wp.to_torch(d.root_link_ang_vel_w)[ids].clone(),
        }
        applied_jp = getattr(self, "_applied_jp", None)
        if self.ctx.history_source == "reference_partial" and applied_jp is not None:
            snap["jp"] = applied_jp[ids].clone()
            snap["jv"] = self._applied_jv[ids].clone()
            if self._ee_body_idx is not None:
                snap["frames"] = self._applied_frames[ids].clone()          # [.,K,13]
            return snap

        # Measured (the default, and the first step of a partially autoregressive run,
        # before any reference has been applied).
        snap["jp"] = wp.to_torch(d.joint_pos)[ids][:, self._robot_to_fm_jperm].clone()
        snap["jv"] = wp.to_torch(d.joint_vel)[ids][:, self._robot_to_fm_jperm].clone()
        if self._ee_body_idx is not None:
            fp = wp.to_torch(d.body_pos_w)[ids][:, self._ee_body_idx]
            fq = wp.to_torch(d.body_quat_w)[ids][:, self._ee_body_idx]
            fl = wp.to_torch(d.body_link_vel_w)[ids][:, self._ee_body_idx]  # [.,K,6]
            snap["frames"] = torch.cat([fp, fq, fl[..., :3], fl[..., 3:]], dim=-1).clone()  # [.,K,13]
        return snap

    def _read_extero(self) -> torch.Tensor:
        """Live exteroception reading for the sensor this FM checkpoint was trained on.

        The source comes from the CHECKPOINT (``ctx.extero_source``), not the task config,
        so a depth-trained FM cannot silently be fed a height scan (or vice versa) just
        because the env happens to have both sensors.

        Returns:
            ``(num_envs, K)`` for the height scan, or ``(num_envs, T, K)`` for depth --
            :func:`build_extero_tokens` flattens either.

        Raises:
            RuntimeError: If the scene lacks the sensor the checkpoint needs.
        """
        src = self.ctx.extero_source
        if src == "height_scan":
            return mdp_height_scan(self.env, SceneEntityCfg("height_scanner"))
        if src == "depth":
            return depth_obs_stack(self.env)
        raise RuntimeError(f"unknown extero_source {src!r} on the FM checkpoint.")

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

        # Envs whose window has not filled yet (post-reset) are sampled with the learned
        # null-history embedding the FM was trained with, so the generator drives from step 0
        # and no clip fallback is needed.
        null_hist = self._history.null_mask()
        reinfer = (self.env.episode_length_buf % max(int(self.cfg.fm_update_every), 1)) == 0
        infer_mask = reinfer | ~self._pred_valid
        self._fm_reinfer_mask = infer_mask
        # Advance the applied index on steps that did not re-plan. The cursor can never reach
        # the end of the horizon: re-planning every ``fm_update_every`` steps caps it at
        # ``fm_update_every - 1``, and :meth:`_check_horizon` guarantees F covers that plus
        # the largest look-ahead offset. Assert rather than clamp -- a clamp here would
        # silently freeze the tail of the observed window.
        advance = self._pred_valid & ~infer_mask
        self._pred_idx = torch.where(advance, self._pred_idx + 1, self._pred_idx)
        if bool((self._pred_idx >= self.ctx.F).any()):
            raise RuntimeError(
                f"FM prediction cursor ran past the horizon (F={self.ctx.F}) with "
                f"fm_update_every={self.cfg.fm_update_every}: the re-plan cadence is not "
                "firing. Check episode_length_buf and the reset latch."
            )
        if bool(infer_mask.any()):
            self._run_fm(infer_mask, null_hist)
        if not bool(self._pred_valid.all()):
            raise RuntimeError(
                "FmTrackingCommand has envs with no live FM prediction after inference; the "
                "reference has no clip fallback, so this must never happen."
            )

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
        # Exteroception tokens = the live reading at "now", from whichever sensor this FM
        # checkpoint was TRAINED on -- the same mdp functions the teacher obs uses, so
        # deployment conditions on exactly the training signal.
        height = build_extero_tokens(self.ctx, self._read_extero()[ids])
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
        # Node 0 = "now" (the anchor step, offset ≈ 0). Track from node 0 and advance one
        # per step, resetting to 0 at each re-plan -- exactly the recorded snap regime.
        self._pred_idx[ids] = 0
        self._pred_valid[ids] = True

    def _fm_current_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather the FM prediction at each env's current advance index (world frame).

        The ee half is ``[pos, identity quat, 0, 0]``: the FM predicts ee POSITION only, so
        the orientation/twist channels are placeholders that ``strict_fm_only`` then NaNs.
        """
        ar, idx = self._arange_n, self._pred_idx
        base = torch.cat([
            self._pred_root_pos[ar, idx], self._pred_root_quat[ar, idx],
            self._pred_root_lin[ar, idx], self._pred_root_ang[ar, idx],
        ], dim=-1)  # [N,13]
        k = self.ctx.num_extra_frames
        ee = torch.cat([
            self._pred_frame_pos[ar, idx], self._ee_zero_quat[:, :k],
            self._ee_zero_vec[:, :k], self._ee_zero_vec[:, :k],
        ], dim=-1)  # [N,K,13]
        return base, ee, self._pred_jp[ar, idx], self._pred_jv[ar, idx]

    # ------------------------------------------------------------------
    # Future window (always the FM prediction -- there is no clip source here)
    # ------------------------------------------------------------------
    def _check_horizon(self, offsets: torch.Tensor) -> None:
        """Raise unless the trained horizon covers ``offsets`` at maximum cursor staleness.

        Done lazily on the first look-ahead call rather than in ``_post_init``, because the
        observation manager (which owns the offsets, via ``reference_window``'s ``num_steps``
        / ``stride``) is built after the command manager. Re-checks only when a caller asks
        for a larger offset than any seen so far.

        The cursor is at most ``fm_update_every - 1`` nodes past the re-plan, so serving
        offset ``o`` needs node ``o + fm_update_every - 1``, i.e. ``F >= o +
        fm_update_every``. Nothing clamps: a short horizon is an error, not a frozen tail.
        """
        max_off = int(offsets.max().item())
        if max_off <= self._checked_max_offset:
            return
        required = max_off + int(self.cfg.fm_update_every)
        if required > self.ctx.F:
            raise ValueError(
                f"The FM checkpoint predicts F={self.ctx.F} nodes, but serving a look-ahead "
                f"offset of {max_off} steps at a re-plan period of {self.cfg.fm_update_every} "
                f"needs F >= {required} (the cursor is up to {self.cfg.fm_update_every - 1} "
                "nodes stale when the furthest sample is read). Re-record with "
                f"--ref_future_len {required} and retrain with --future_len {required}."
            )
        self._checked_max_offset = max_off

    def _future_indices(self, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(env_index, node_index)`` broadcast pair for gathering ``offsets`` per env."""
        self._check_horizon(offsets)
        idx = self._pred_idx[:, None] + offsets[None, :]        # [N, K]
        return self._arange_n[:, None].expand_as(idx), idx

    def get_future_world_base(
        self, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame future reference base pose+velocity from the FM prediction.

        Overrides the clip look-ahead entirely: in this command the reference IS the
        generator, so there is no per-env source to select.
        """
        ar, idx = self._future_indices(offsets)
        return (
            self._pred_root_pos[ar, idx], self._pred_root_quat[ar, idx],
            self._pred_root_lin[ar, idx], self._pred_root_ang[ar, idx],
        )

    def get_future_joint_state(self, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FM joint pos/vel ``([N, K, nj], [N, K, nj])`` at the future ``offsets``.

        Returned in the **robot's** joint order, matching the base-class contract (the clip
        implementation gathers ``_base_idx["joint_pos"]``, which is built by iterating
        ``robot.joint_names``). The prediction buffers are in FM order, so the permutation
        is mandatory -- and it is ``_fm_to_robot_jperm``, NOT the clip-output ``_fm_jperm``
        used when writing ``y_des``.
        """
        ar, idx = self._future_indices(offsets)
        perm = self._fm_to_robot_jperm
        return self._pred_jp[ar, idx][..., perm], self._pred_jv[ar, idx][..., perm]

    def get_future_world_frames(self, frame_names: list[str], offsets: torch.Tensor) -> torch.Tensor:
        """FM end-effector futures ``[N, K, F, 13]`` at the future ``offsets``.

        Layout matches the clip implementation -- ``[pos_w(3), quat_w(4 xyzw), lin_w(3),
        ang_w(3)]`` -- but the FM predicts POSITION only, so the trailing 10 channels are
        an identity quaternion and zeros. The one consumer, ``mdp.reference_window``, reads
        the position; anything reaching for the rest is reading a placeholder, which is why
        they are constants rather than a stale lift of clip data.

        Raises:
            ValueError: If a requested frame is not one the checkpoint predicts.
        """
        missing = [f for f in frame_names if f not in self.ctx.frame_names]
        if missing:
            raise ValueError(
                f"get_future_world_frames asked for {missing}, which the FM checkpoint does "
                f"not predict (it has {list(self.ctx.frame_names)}). There is no clip "
                "fallback in this command."
            )
        ar, idx = self._future_indices(offsets)
        sel = torch.tensor([self.ctx.frame_names.index(f) for f in frame_names],
                           dtype=torch.long, device=self.device)
        pos = self._pred_frame_pos[ar, idx][:, :, sel]                    # [N, K, F, 3]
        shape = pos.shape[:-1]
        quat = torch.zeros(*shape, 4, device=pos.device, dtype=pos.dtype)
        quat[..., 3] = 1.0
        zeros = torch.zeros(*shape, 3, device=pos.device, dtype=pos.dtype)
        return torch.cat([pos, quat, zeros, zeros], dim=-1)               # [N, K, F, 13]

    # ------------------------------------------------------------------
    # Debug visualization (FM-prediction ribbon)
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Toggle the inherited viz plus the FM-prediction markers."""
        super()._set_debug_vis_impl(debug_vis)
        if not debug_vis and self._fm_markers is not None:
            self._fm_markers.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Draw the inherited viz plus the FM prediction (spheres).

        Guarded independently of the base's own check: this callback's second half runs
        whether or not the base drew anything, and :meth:`_draw_fm_prediction_ribbon` reads
        ``self.device`` -- another env-backed property. See
        :meth:`MotionSequenceCommand._scene_alive`.
        """
        if not self._scene_alive():
            return
        super()._debug_vis_callback(event)
        if self.cfg.debug_vis_show_fm_prediction:
            self._draw_fm_prediction_ribbon()

    def _clip_ribbon_env_mask(self) -> torch.Tensor:
        """Never draw the inherited CLIP ribbon: no env here tracks the clip."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _draw_fm_prediction_ribbon(self) -> None:
        """Draw the FM's predicted world-frame trajectory as per-frame spheres."""
        if self._fm_markers is None:
            self._fm_markers = self._create_fm_markers()
            if self._fm_markers is None:
                return
        if not self._fm_pred_specs:
            self._fm_markers.set_visibility(False)
            return
        offs = self._fm_debug_offsets
        translations, marker_indices = [], []
        for proto_idx, (_name, _color, kind, k) in enumerate(self._fm_pred_specs):
            pts = (self._pred_root_pos[:, offs] if kind == "base"
                   else self._pred_frame_pos[:, offs, k]).reshape(-1, 3)
            translations.append(pts)
            marker_indices.append(torch.full((pts.shape[0],), proto_idx, dtype=torch.long, device=self.device))
        self._fm_markers.set_visibility(True)
        self._fm_markers.visualize(
            translations=torch.cat(translations, dim=0), marker_indices=torch.cat(marker_indices, dim=0),
        )

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
