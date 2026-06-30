# Copyright 2026 Zachary Olkin. All rights reserved.

"""Flow-matching generator-tracking command (world-frame CLF).

Extends :class:`MotionSequenceCommand` into a single, robot-decoupled mode:
every env runs the **flow-matching generator** autoregressively as a receding-
horizon reference source.  The FM conditions on its OWN past reference output
(seeded from the clip at reset and during the warmup) -- it never sees the
robot -- and each freshly sampled trajectory is re-anchored onto the robot root
(xy + z + heading onto the robot root, keeping only the FM's tilt).  The clip library
only supplies the reset spawn pose and the ``target_vel`` velocity command; the
clip reference is the warmup fallback until the history window fills.

The desired outputs ``y_des``/``dy_des`` and the measured outputs
``y_act``/``dy_act`` are expressed in the world frame (no stance-anchor
re-anchoring), so the inherited CLF reward, observations, and terminations work
uniformly.  The world frame is a single stationary inertial frame, valid for the
CLF's finite-difference ``vdot``.

Base-velocity convention: every base/ee velocity is the **link** (actor-frame)
twist.  The FM is trained and conditioned on ``root_link_{lin,ang}_vel_w`` (see
``record_data.py``) and the measured ``y_act`` base velocity comes from
``body_link_vel_w`` -- the same pelvis link frame.  No CoM velocities are used
anywhere, so the FM reference and the CLF's measured outputs share one frame
(no ``omega x r_com`` mismatch).
"""

from __future__ import annotations

import contextlib
import os

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
from .world_lift import lift_base_world_state, lift_orientation, lift_position, lift_vector


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
        """Set up the FM context, history buffer, prediction buffers, and world index maps."""
        super()._post_init()
        n, dev = self.num_envs, self.device

        if not self.command_velocity_available:
            raise RuntimeError(
                "GeneratorTrackingCommand requires a clip library with target_vel "
                "(the FM velocity command). Regenerate clips with a generator that writes it."
            )
        if not self.cfg.fm_checkpoint:
            raise ValueError(
                "GeneratorTrackingCommandCfg.fm_checkpoint must be set to a trained, "
                "clip-conditioned flow-matching .pt checkpoint (pass --fm_checkpoint "
                "<flow_matching_run>/best.pt at play time, or set the cfg default)."
            )
        if not os.path.isfile(self.cfg.fm_checkpoint):
            raise FileNotFoundError(
                f"GeneratorTrackingCommandCfg.fm_checkpoint '{self.cfg.fm_checkpoint}' does not "
                "exist. Pass an existing clip-conditioned FM checkpoint via --fm_checkpoint "
                "<flow_matching_run>/best.pt (paths are relative to the run cwd)."
            )
        if int(self.cfg.fm_update_every) < 1:
            raise ValueError(
                f"GeneratorTrackingCommandCfg.fm_update_every must be >= 1 (the FM re-plan "
                f"cadence in control steps), got {self.cfg.fm_update_every}."
            )

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

        # The PARC-style FM conditions on AND predicts the same end-effector frames
        # (``ctx.frame_names``), so each autoregressive history snapshot carries them
        # in that order -- assembled from the applied reference, never the robot.
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
        # Inverse perm: clip-output (joint_out) joint order -> FM (ctx) joint order,
        # used to express the applied/clip reference joints in the FM history layout
        # for the autoregressive history source.
        self._clip_to_fm_jperm = torch.argsort(self._fm_jperm)
        # Cached previous-step applied reference (FM history format), the
        # autoregressive history input; None until the first update.
        self._applied_ref: dict[str, torch.Tensor] | None = None

        # Per-step CLIP reference cache (parallel to the applied FM reference), for
        # obs-level FM-vs-clip comparison logging: the clip's current-step desired
        # outputs + world frames, populated in _update_command and served through the
        # normal obs interface via as_clip_reference(). None until the first update.
        self._clip_y_des: torch.Tensor | None = None
        self._clip_dy_des: torch.Tensor | None = None
        self._clip_wf: torch.Tensor | None = None
        self._serve_clip: bool = False

        # Per-step FM reference cache (symmetric to the clip cache above), so the FM
        # prediction can be served through the normal obs interface via
        # as_fm_reference() -- for plotting the FM alongside the clip whichever drives.
        # In the default mode this equals the applied reference; in drive_with_clip
        # mode (clip drives) it is the passive FM. None until the first update.
        self._fm_y_des: torch.Tensor | None = None
        self._fm_dy_des: torch.Tensor | None = None
        self._fm_wf: torch.Tensor | None = None
        self._serve_fm: bool = False

        # -- prediction buffers + cadence state (every env runs the FM) --
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
        # Per-env mask of which envs the FM re-inferred (re-planned) this step; used to
        # re-anchor the clip on the SAME steps/envs as the FM (set in _snapshot_and_infer).
        self._fm_reinfer_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        # Per-env FM-bucket mask: True where the FM (not the clip) drives the reference.
        # Re-rolled per reset (prob cfg.fm_fraction) in reset_on_motion_sequence; FM-bucket
        # envs are forced ego-anchored there too. Gates FM inference below, so _pred_valid
        # (and thus use_fm / fm_active) only ever becomes True for these envs. Allocated
        # all-False; the reset event sets it before the first post-reset inference.
        self._fm_mask = torch.zeros(n, dtype=torch.bool, device=dev)

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

        # Ego-re-anchor the CLIP onto the robot root, on TWO cadences for the two
        # clip-using buckets:
        #   - FM-bucket envs (``_fm_mask``): on the SAME steps the FM just re-planned
        #     (``_fm_reinfer_mask``), so the clip warmup fallback + as_clip_reference()
        #     obs stay re-anchored at identical instants to the FM (matches deploy).
        #     This fires regardless of ego_anchor_fraction -- the FM is always ego.
        #   - ego-anchored clip-bucket envs (``_ego_mask & ~_fm_mask``): on the inherited
        #     episode-step cadence (``ego_anchor_every``), like MotionSequenceCommand.
        # World-anchored clip envs (``~_ego_mask & ~_fm_mask``) are never re-anchored.
        fm_due = self._fm_mask & self._fm_reinfer_mask
        period = max(int(self.cfg.ego_anchor_every), 1)
        ego_due = self._ego_mask & ~self._fm_mask & ((self.env.episode_length_buf % period) == 0)
        self._reanchor_clip_ids(torch.nonzero(fm_due | ego_due, as_tuple=False).squeeze(-1))

        # Clip desired outputs (clip-world). Joints are frame-invariant; the
        # base/ee Cartesian channels are overwritten with world values below.
        self.get_desired_outputs()

        # Clip world reference (base + ee), lifted through the per-env placement
        # (frames in self._wf_frames == [base] + FM ee order). Used as the warmup
        # fallback until each env's FM history window fills.
        clip_frames = self._clip_world_frames()                      # [N, nf, 13]

        # Cache the CLIP-only reference (clip joints from get_desired_outputs above +
        # clip world base/ee) so as_clip_reference() can serve the clip's version of
        # every reference obs for FM-vs-clip diagnostics. Cloned here, before the FM
        # joints/frames overwrite y_des/dy_des below; mirrors _write_world_des's scatter.
        n = self.num_envs
        clip_y, clip_dy = self.y_des.clone(), self.dy_des.clone()     # clip joints already present
        clip_y[:, self._wf_pos_flat] = clip_frames[:, :, 0:3].reshape(n, -1)
        clip_y[:, self._wf_ori_flat] = clip_frames[:, :, 3:7].reshape(n, -1)
        clip_dy[:, self._wf_lin_flat] = clip_frames[:, :, 7:10].reshape(n, -1)
        clip_dy[:, self._wf_ang_flat] = clip_frames[:, :, 10:13].reshape(n, -1)
        self._clip_y_des, self._clip_dy_des, self._clip_wf = clip_y, clip_dy, clip_frames

        # FM world reference at the current advance index, in the same order.
        fm_base, fm_ee, fm_jp, fm_jv = self._fm_current_outputs()
        fm_frames = torch.cat([fm_base[:, None, :], fm_ee], dim=1)    # [N, nf, 13]
        # FM drives only the FM-bucket envs past their warmup. ``_pred_valid`` already
        # implies ``_fm_mask`` (inference is gated by it), but make the bucket explicit.
        use_fm = self._pred_valid & self._fm_mask

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

        # Cache this step's applied reference (FM history format) so the next step's
        # autoregressive history append feeds the FM its own output. Snapshotted from
        # the FM-selected reference NOW, before any drive_with_clip swap below, so the
        # FM stays autoregressive on its OWN output regardless of what drives the robot.
        self._applied_ref = self._reference_snapshot()

        # Cache the FM reference (symmetric to the clip cache) for as_fm_reference():
        # y_des/dy_des/_wf_world_frames currently hold the FM selection (FM where valid,
        # clip during warmup) -- the same world reference the FM just fed its history.
        self._fm_y_des, self._fm_dy_des = self.y_des.clone(), self.dy_des.clone()
        self._fm_wf = self._wf_world_frames.clone()

        # Diagnostic mode: hand the robot the CLIP instead of the FM. The FM keeps
        # running (cached above + history snapshotted above), but the applied reference
        # the policy tracks / CLF / rewards / obs see becomes the ego-anchored clip --
        # so we can record + plot the FM passively while the robot tracks the clip.
        if self.cfg.drive_with_clip and self._clip_y_des is not None:
            self.y_des, self.dy_des = self._clip_y_des.clone(), self._clip_dy_des.clone()
            self._wf_world_frames = self._clip_wf.clone()

        # Measured outputs in the world frame (identity reference -> no alignment).
        self.compute_measured_output(self._zeros3, self._ident_quat)

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot = vdot
        self.v = vcur

    # ------------------------------------------------------------------
    # History snapshot + flow-matching inference (once per control step)
    # ------------------------------------------------------------------

    def _snapshot_and_infer(self) -> None:
        """Append the current reference output to history and (re-)run FM inference.

        Guarded to run once per control step (Isaac Lab computes commands twice
        per step: once via ``command_manager.reset`` inside ``_reset_idx`` and
        once via ``command_manager.compute``). Freshly-reset envs (latched in
        :meth:`reset`) are re-seeded from the clip and get an invalidated history
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
            self._pred_valid = self._pred_valid & ~just_reset
            self._pred_idx = torch.where(just_reset, torch.zeros_like(self._pred_idx), self._pred_idx)

        # Build the history snapshot (FM joint/frame order) for this step:
        # autoregressive and robot-decoupled -- the history is the FM's OWN past
        # reference output. ``self._applied_ref`` holds the previous step's applied
        # reference (cached at the end of _update_command; the clip during the
        # warmup). Just-reset envs are (re-)seeded from the clip spawn pose so the
        # first window is clip references, not a stale prior episode.
        if self._applied_ref is None:
            self._applied_ref = self._clip_reference_snapshot(self._arange_n)
        if bool(just_reset.any()):
            rids = torch.nonzero(just_reset, as_tuple=False).squeeze(-1)
            seed = self._clip_reference_snapshot(rids)
            for key in self._applied_ref:
                self._applied_ref[key][rids] = seed[key]
        snap = {k: v.clone() for k, v in self._applied_ref.items()}

        # Invalidate reset envs' windows, then append (this snapshot = step 1 of the new episode).
        self._history.invalidate(just_reset)
        self._history.append(snap)

        # Decide which envs to (re-)infer vs advance the prediction index.
        valid = self._history.valid()
        self._pred_valid = self._pred_valid & valid          # lose validity if the window broke
        # Re-plan cadence on the PER-ENV episode clock (episode_length_buf), not the
        # global common_step_counter: re-plans land at deterministic episode steps from
        # each reset (every fm_update_every), so the post-reset phase is fixed and aligned
        # with the clip ego re-anchor -- instead of an arbitrary, per-env-staggered global
        # phase. (The once-per-step guard above still uses the global step.) Per-env bool.
        reinfer = (self.env.episode_length_buf % max(int(self.cfg.fm_update_every), 1)) == 0
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        # Only FM-bucket envs (``_fm_mask``) ever run inference, so clip-bucket envs pay
        # no FM cost and their ``_pred_valid`` stays False (-> use_fm False -> clip drives).
        infer_mask = valid & self._fm_mask & (reinfer | ~self._pred_valid | exhausted)
        self._fm_reinfer_mask = infer_mask         # re-anchor the clip on these same steps/envs
        advance = valid & ~infer_mask
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
        frames = self._history.stack("frames")[ids] if self.ctx.num_extra_frames > 0 else None

        cond, known = build_cond_states(self.ctx, rp, rq, rl, ra, jp, jv, frames=frames)
        # Commanded-velocity profile over the window -> strided command tokens.
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        cmd_tokens = build_cmd_tokens(self.ctx, fut_cmd)
        # Snap the prediction's current step onto the ROBOT root: planar position (xy),
        # HEIGHT (z), and heading (yaw), keeping only the FM's natural tilt (roll/pitch).
        # The conditioning is still the FM's own past output (the network never sees the
        # robot) -- only the world placement is re-anchored, so the autoregressive rollout
        # follows the robot (receding horizon) instead of drifting on its own prior
        # trajectory. Anchoring z (not just xy) keeps the FM's free-running height drift
        # (the autoregressive root z climbs/sinks unboundedly -- see the bench rollout)
        # from pulling the reference height off the robot.
        # The TEACHER's ego-anchor during training now matches this: world_lift.ego_placement
        # / MotionSequenceCommand._reanchor_ego_envs snap the clip's current-step base z onto
        # the robot root too (not the clip's NATURAL z), so the train-time ego reference height
        # and this deploy-time placement agree. A sinking robot drags the reference down with it
        # (no free recovery-height target) in both train and deploy, so the teacher must learn to
        # stay up / avoid states where the robot-anchored reference becomes infeasible.
        p_ref = rp[:, -1].clone()
        q_ref = rq[:, -1].clone()
        robot_p = wp.to_torch(self.robot.data.root_pos_w)[ids]
        robot_q = wp.to_torch(self.robot.data.root_quat_w)[ids]
        p_ref[:, :3] = robot_p[:, :3]                 # anchor xy + z onto the robot root
        dz_yaw = quat_mul(yaw_quat(robot_q), quat_conjugate(yaw_quat(q_ref)))  # world z-rot
        q_ref = quat_mul(dz_yaw, q_ref)               # FM tilt, re-yawed to the robot heading
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

    # ------------------------------------------------------------------
    # Autoregressive ('reference') history snapshots (robot-decoupled)
    # ------------------------------------------------------------------

    def _reference_snapshot(self) -> dict[str, torch.Tensor]:
        """This step's applied reference (clip-or-FM selected) in the FM history format.

        Reads the world-frame reference written by :meth:`_write_world_des`
        (``_wf_world_frames`` = ``[base] + ee`` in FM frame order) plus the applied
        joints (``_joint_slots_*`` in clip-output order -> FM order). Cached as the
        next step's autoregressive history input.
        """
        base = self._wf_world_frames[:, 0, :]  # [N, 13] = [pos(3), quat(4), lin(3), ang(3)]
        snap = {
            "root_pos": base[:, 0:3].clone(),
            "root_quat": base[:, 3:7].clone(),
            "root_lin": base[:, 7:10].clone(),
            "root_ang": base[:, 10:13].clone(),
            "jp": self.y_des[:, self._joint_slots_pos][:, self._clip_to_fm_jperm].clone(),
            "jv": self.dy_des[:, self._joint_slots_vel][:, self._clip_to_fm_jperm].clone(),
        }
        if self.ctx.num_extra_frames > 0:
            snap["frames"] = self._wf_world_frames[:, 1:, :].clone()  # [N, K, 13], FM ee order
        return snap

    def _clip_reference_snapshot(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Clip reference at each env's start step (FM history format) for ``ids``.

        Used to seed the autoregressive history at reset (and the warmup), so the
        first FM window conditions on clip references rather than a stale prior
        episode. Lifts the clip-world base + ee outputs through the per-env placement
        (the same transform the runtime applies) and reorders joints to FM order.
        """
        step = self.start_step[ids]
        y, dy = self.manager.gather_desired(self.clip_id[ids], step)
        wpos, wquat = self.clip_world_pos[ids], self.clip_world_quat[ids]
        base = lift_base_world_state(y, dy, torch.cat([wpos, wquat], dim=-1), self._base_idx)
        nj = len(self.ctx.joint_names)
        snap = {
            "root_pos": base[:, 0:3].clone(),
            "root_quat": base[:, 3:7].clone(),
            "root_lin": base[:, 7:10].clone(),
            "root_ang": base[:, 10:13].clone(),
            "jp": base[:, 13:13 + nj][:, self._clip_to_fm_jperm].clone(),
            "jv": base[:, 13 + nj:13 + 2 * nj][:, self._clip_to_fm_jperm].clone(),
        }
        if self.ctx.num_extra_frames > 0:
            ee = []
            for f in self._wf_frames[1:]:  # ctx.frame_names order
                fi = self._frame_output_indices(f)
                ee.append(torch.cat([
                    lift_position(wpos, wquat, y[:, fi["pos"]]),
                    lift_orientation(wquat, y[:, fi["ori"]]),
                    lift_vector(wquat, dy[:, fi["lin_vel"]]),
                    lift_vector(wquat, dy[:, fi["ang_vel"]]),
                ], dim=-1))
            snap["frames"] = torch.stack(ee, dim=1)  # [k, K, 13]
        return snap

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
    # Introspection for play-time logging / plots
    # ------------------------------------------------------------------

    @property
    def fm_active(self) -> torch.Tensor:
        """Per-env ``[N]`` bool: True where the FM prediction (not the clip) drives the reference.

        This is the exact authoritative-source mask used in :meth:`_update_command`
        (``_pred_valid``): True once an env is past its history warmup.  Flips
        False->True at the end of warmup and True->False on a window break -- the
        boundary the play plots mark with a transition line.

        In ``drive_with_clip`` mode the FM never drives the reference (the clip does),
        so this is all-False -- the FM is shown as a passive overlay, not a driver.
        """
        if self.cfg.drive_with_clip:
            return torch.zeros_like(self._pred_valid)
        return self._pred_valid

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
        past their warmup (``_pred_valid``), the **clip** for the rest (FM-warmup
        envs).  This mirrors the current-step source selection in
        :meth:`_update_command`, so a deployed FM env never sees privileged clip
        look-ahead -- only the generator's own prediction (available at deployment
        from the same FM + tracker pair).

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
        # Serve the look-ahead from the applied source: the clip (base impl) when the
        # clip drives (drive_with_clip) or under as_clip_reference(); the FM otherwise
        # or under as_fm_reference(). Keeps future_base_pose/velocity consistent with
        # the current-step y_des the policy tracks, and lets each comparison context
        # see its source's future.
        if not self._serving_fm():
            return super().get_future_world_base(offsets)
        pos_c, quat_c, lin_c, ang_c = super().get_future_world_base(offsets)

        # FM-predicted future at the same offsets (world frame, current-base anchored).
        idx = torch.clamp(self._pred_idx[:, None] + offsets[None, :], max=self.ctx.F - 1)  # [N, K]
        ar = self._arange_n[:, None].expand_as(idx)
        pos_f = self._pred_root_pos[ar, idx]    # [N, K, 3]
        quat_f = self._pred_root_quat[ar, idx]  # [N, K, 4] (xyzw)
        lin_f = self._pred_root_lin[ar, idx]    # [N, K, 3]
        ang_f = self._pred_root_ang[ar, idx]    # [N, K, 3]

        use_fm = (self._pred_valid & self._fm_mask)[:, None, None]
        return (
            torch.where(use_fm, pos_f, pos_c),
            torch.where(use_fm, quat_f, quat_c),
            torch.where(use_fm, lin_f, lin_c),
            torch.where(use_fm, ang_f, ang_c),
        )

    def _serving_fm(self) -> bool:
        """Whether the served reference / look-ahead should be the FM (vs the clip).

        Explicit comparison contexts win (:meth:`as_clip_reference` /
        :meth:`as_fm_reference`); otherwise it's the applied source -- the FM by
        default, the clip under ``drive_with_clip``.
        """
        if self._serve_clip:
            return False
        if self._serve_fm:
            return True
        return not self.cfg.drive_with_clip

    @contextlib.contextmanager
    def as_clip_reference(self):
        """Temporarily serve the CLIP reference through the normal obs interface.

        Swaps ``y_des``/``dy_des`` and the world-frame cache to the cached clip
        reference and routes :meth:`get_future_world_base` to the clip (base) impl,
        so the standard reference obs (``base_pose_error``, ``joint_*_ref``,
        ``ee_ref``, ``base_vel_ref``, ``future_base_pose``/``velocity``) computed
        inside the ``with`` block report what the policy WOULD see tracking the clip
        instead of the FM -- for obs-level FM-vs-clip comparison (e.g. in the play
        DataLogger). No-op-safe to call before the first update (clip cache is None).
        """
        if self._clip_y_des is None:
            yield
            return
        saved = (self.y_des, self.dy_des, self._wf_world_frames, self._serve_clip, self._serve_fm)
        self.y_des, self.dy_des = self._clip_y_des, self._clip_dy_des
        self._wf_world_frames = self._clip_wf
        self._serve_clip, self._serve_fm = True, False
        try:
            yield
        finally:
            (self.y_des, self.dy_des, self._wf_world_frames,
             self._serve_clip, self._serve_fm) = saved

    @contextlib.contextmanager
    def as_fm_reference(self):
        """Temporarily serve the FM reference through the normal obs interface.

        The mirror of :meth:`as_clip_reference`: swaps ``y_des``/``dy_des`` + the
        world-frame cache to the cached FM prediction and routes
        :meth:`get_future_world_base` to the FM, so the standard reference obs report
        what the policy WOULD see tracking the FM.  Used to record/plot the (passive)
        FM alongside the clip in ``drive_with_clip`` mode -- the inverse comparison.
        In the default mode the FM cache equals the applied reference, so this is a
        no-op view of ``y_des``.  No-op-safe before the first update (FM cache None).
        """
        if self._fm_y_des is None:
            yield
            return
        saved = (self.y_des, self.dy_des, self._wf_world_frames, self._serve_clip, self._serve_fm)
        self.y_des, self.dy_des = self._fm_y_des, self._fm_dy_des
        self._wf_world_frames = self._fm_wf
        self._serve_clip, self._serve_fm = False, True
        try:
            yield
        finally:
            (self.y_des, self.dy_des, self._wf_world_frames,
             self._serve_clip, self._serve_fm) = saved

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

        FM-active envs (``_pred_valid``) show the FM-prediction spheres instead;
        FM-warmup envs (still following the clip) keep the clip ribbon.
        """
        return ~self._pred_valid

    def _draw_fm_prediction_ribbon(self) -> None:
        """Draw the FM generator's predicted world trajectory as per-frame spheres.

        Only the envs currently tracking the FM (``_pred_valid``) are drawn; the
        predicted positions are read straight from the world-frame prediction
        buffers (subsampled over the F-step horizon). Sphere colors match the clip
        ribbon's per-frame colors, so the FM prediction (spheres) can be compared
        directly to the clip baseline (cylinders).
        """
        if self._fm_markers is None:
            self._fm_markers = self._create_fm_markers()
            if self._fm_markers is None:
                return
        if not self._fm_pred_specs:
            self._fm_markers.set_visibility(False)
            return

        use_fm = self._pred_valid
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

        Green => FM (history full, prediction live); yellow => still warming up
        (history not yet full, falling back to the clip).
        """
        if self._source_markers is None:
            self._source_markers = self._create_source_markers()
            if self._source_markers is None:
                return

        idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # 1 => warmup
        idx[self._pred_valid] = 0   # 0 => fm (prediction live)

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
        """Build the two-color (fm / warmup) tracking-source indicator set."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN GeneratorTrackingCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_source_marker_radius)
        # Insertion order fixes the marker_indices mapping: 0 => fm, 1 => warmup.
        colors = {
            "fm": (0.10, 0.90, 0.20),
            "warmup": (0.95, 0.85, 0.10),
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
