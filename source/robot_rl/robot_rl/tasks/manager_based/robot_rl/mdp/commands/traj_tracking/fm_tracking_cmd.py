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

**Both references are computed every step.** The command builds the clip reference and the
generator reference into separate caches and then APPLIES the one
:attr:`~.fm_tracking_cmd_cfg.FmTrackingCommandCfg.reference_source` names; the other stays
available through :meth:`FmTrackingCommand.as_clip_reference` /
:meth:`FmTrackingCommand.as_fm_reference`. That is what lets a diagnostic ask "what would the
OTHER source have said at this exact robot state" -- for the reference observations, the
world-frame outputs, and the whole look-ahead window -- without a second rollout. The barrier
is unchanged: with ``reference_source="fm"`` the applied ``y_des`` is bit-for-bit what it was
when only the generator was computed (the clip half writes into ``y_des`` first, but every
channel is then either overwritten by the generator or NaN-filled by ``strict_fm_only``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import warp as wp

from isaaclab.envs.mdp import height_scan as mdp_height_scan
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
    depth_obs_stack,
    fm_depth_groups,
)

from robot_rl.flow_matching.online import (
    BatchedHistory,
    build_cmd_tokens,
    build_extero_tokens,
    build_history_tokens,
    load_fm_context,
    sample_window,
)
from isaaclab.utils.math import quat_apply

from robot_rl.flow_matching.window import (
    REFERENCE_BLOCK_KEYS,
    expand_window_blocks,
    reanchor_window,
    resolve_reference_blocks,
    window_to_world,
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

        if self.cfg.reference_source not in ("fm", "clip"):
            raise ValueError(
                f"FmTrackingCommandCfg.reference_source must be 'fm' or 'clip', got "
                f"{self.cfg.reference_source!r}."
            )
        # Reference caches + the "which half is being built / served" flags. Set before any
        # early return so the accessors below are always safe to call.
        self._building: str | None = None
        self._serve_override: str | None = None
        self._y_des_clip = self.y_des.clone()
        self._dy_des_clip = self.dy_des.clone()
        self._clip_wf_frames = self._wf_world_frames.clone()
        self._y_des_fm = self.y_des.clone()
        self._dy_des_fm = self.dy_des.clone()
        self._fm_wf_frames = self._wf_world_frames.clone()
        # Allocated up front so :meth:`reset` is safe on the no-generator path too.
        self._pending_reset = torch.zeros(n, dtype=torch.bool, device=dev)
        # Offset of the node being served within the live plan: 0 on a re-plan, +1 per step
        # kept as a tensor so the look-ahead helpers take one code path.
        self._zero_offset = torch.zeros(1, dtype=torch.long, device=dev)

        if not self.command_velocity_available:
            raise RuntimeError(
                "FmTrackingCommand requires a clip library with target_vel (the FM velocity "
                "command). Regenerate clips with a generator that writes it."
            )
        # The generator is optional in exactly one configuration: the clip-driven control
        # run, where there is nothing for it to do. Everywhere else a missing checkpoint is
        # a hard error -- see the message below.
        self.fm_enabled = bool(self.cfg.fm_checkpoint) or self.cfg.reference_source == "fm"
        if not self.fm_enabled:
            print("[INFO FmTrackingCommand] reference_source='clip' with no fm_checkpoint: "
                  "the generator is not loaded. This is the pure-clip control; nothing here "
                  "can report on the FM.")
            return
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
        k = int(self.cfg.fm_update_every)
        if k < 1:
            raise ValueError(f"fm_update_every must be >= 1, got {k}.")

        # 0 means "keep the checkpoint's own step count" -- the cfg cannot default to None
        # without making the field un-overridable from hydra (see the cfg's docstring).
        steps = int(self.cfg.fm_num_sampling_steps) or None
        self.ctx = load_fm_context(
            self.cfg.fm_checkpoint, device=self.cfg.fm_device or dev,
            num_sampling_steps=steps,
            zero_noise=self.cfg.fm_zero_noise,
            guidance_scale=self.cfg.fm_guidance_scale,
        )

        # K must be 1 or >= N: the history is the previous plan's TAIL (cursor K-N .. K-1), so
        # all N rows come from one plan only at K >= N. Between 2 and N-1 the ring straddles
        # two plans, mixing two anchor frames in one history -- silently, since the shapes are
        # identical either way. K=1 is fine: every row is node 0 of its own plan, which is the
        # same statement with N frames that each coincide with the robot at that step.
        #
        # Checked HERE rather than with the other cfg validation above, because N is a property
        # of the checkpoint and ``self.ctx`` does not exist until it is loaded.
        if 1 < k < self.ctx.N:
            raise ValueError(
                f"fm_update_every={k} is between 2 and N-1 (N={self.ctx.N} history tokens on "
                "this checkpoint). The generator's history is a slice of ONE plan's tail, so K "
                f"must be either 1 or >= N. Use 1, or {self.ctx.N} or more."
            )
        if steps is not None:
            print(f"[INFO FmTrackingCommand] fm_num_sampling_steps={steps} overrides the "
                  f"checkpoint's; the flow is integrated in {steps} Euler steps.")
        if float(self.cfg.fm_guidance_scale) != 1.0:
            print(f"[INFO FmTrackingCommand] fm_guidance_scale={self.cfg.fm_guidance_scale}: "
                  "classifier-free guidance on the history branch (2x-batch forward per step).")
        if self.cfg.fm_zero_noise:
            print("[INFO FmTrackingCommand] fm_zero_noise=True: the flow integrates from x0=0, "
                  "so predictions are deterministic given the conditioning (diagnostic mode, "
                  "off the training distribution).")
        if self.ctx.cmd_dim <= 0:
            raise ValueError("FmTrackingCommand expects a command-conditioned FM checkpoint (cmd_dim>0).")
        if self.cfg.tracking_mode == "body" and self.cfg.reference_source == "fm":
            raise ValueError(
                "FmTrackingCommand does not support tracking_mode='body'. The body CLF grades "
                "13 '@root' body links, but the generator predicts only the base, the joints "
                "and its end-effector POSITIONS -- so those channels would be NaN under "
                "strict_fm_only. Enabling it needs an FM recorded with --state_frames "
                "covering G1_BODY_TRACK_FRAMES, which also widens the state vector.\n"
                "Note this is about THIS task's CLF, which here is diagnostic only (no "
                "reward reads it). It says nothing about the teacher you are loading: a "
                "body-mode teacher runs fine on the default joint-mode tracker, because the "
                "actor's observation groups are identical in both modes.\n"
                "With reference_source='clip' this is allowed: the APPLIED reference is the "
                "clip, which does carry those links -- only the passive FM cache is short of "
                "them, and nothing grades it."
            )
        # NOTE: the re-plan cadence is deliberately NOT checked against the recording's
        # fm_snap_every any more -- see FmTrackingCommandCfg.fm_update_every. What still
        # guards the horizon is _check_horizon, which is lazy (first look-ahead call).

        # The FM's extra frames must be exactly the CLF's end-effector output frames.
        ee_frames = self.resolve_ee_frames()
        if set(self.ctx.frame_names) != set(ee_frames):
            raise ValueError(
                f"FM checkpoint extra frames {self.ctx.frame_names} != command ee output "
                f"frames {ee_frames}; re-record/-train the FM with matching --state_frames."
            )

        # The live exteroception must be the width this generator conditions on. Checked ONCE
        # rather than per step -- the alternative is a shape error from inside the graph, at
        # which point the useful information (which cameras, which group, which resolution)
        # is gone. On the FIRST reading and not here, because the cameras are discovered by
        # walking the observation manager and it does not exist yet: IsaacLab's
        # ``ManagerBasedRLEnv.load_managers`` builds the COMMAND manager first (observations
        # need to know the commands), so anything a command term asks about an obs group at
        # construction time is asking too early.
        self._extero_checked = False

        # Rolling per-env history of the N reference-window nodes this command last
        # SERVED -- the generator's own output, not the robot's state. That is the
        # autoregression, and it is why nothing here needs a base-pose estimate.
        self._history = BatchedHistory(n, self.ctx.N, dev)

        # World-frame index maps in FM frame order ([base] + FM ee), so the per-step write
        # aligns the FM frames with the output channels without a permutation.
        self._build_world_frame_maps([self.cfg.base_frame_name] + list(self.ctx.frame_names))

        # Joint slots (clip-output order) + the one remaining permutation:
        #   _fm_jperm : ROBOT order -> clip-OUTPUT slot order, for writing y_des/dy_des.
        # The read-side permutations are gone: the window's joint block is robot order,
        # so the accessors serve it straight through (checked below).
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
        # The window's joint block is defined in the ROBOT's joint order (that is the
        # ``reference_window`` contract, so the blocks line up elementwise with the
        # proprioception terms), and the FM predicts window nodes -- so the checkpoint's
        # joint order has to BE the robot's. There is no permutation to apply any more;
        # check instead, because a mismatch would silently scramble every joint.
        robot_joints = list(self.robot.data.joint_names)
        if list(self.ctx.joint_names) != robot_joints:
            raise ValueError(
                "FM checkpoint joint order differs from the robot's, but the reference "
                "window's joint block is defined in ROBOT order.\n"
                f"  checkpoint: {list(self.ctx.joint_names)}\n"
                f"  robot:      {robot_joints}\n"
                "Re-record and retrain: the recorder's state_joint_names must be the "
                "robot's joint order."
            )

        # Which blocks this checkpoint predicts. A teacher can pack a SUBSET of the
        # window, and the generator must predict the same subset -- ``reference_window_obs``
        # checks that against the observation term. Everything the FM cannot supply
        # (velocities, ee frames, joints) then becomes an unwritten y_des channel; see
        # :meth:`_build_strict_masks`.
        self._fm_blocks = tuple(self.ctx.reference_blocks)
        if "base_pose" not in self._fm_blocks:
            raise ValueError(
                f"the FM checkpoint predicts {list(self._fm_blocks)}, with no 'base_pose'. "
                "This command lifts the prediction to the world for y_des, the diagnostic "
                "CLF and the debug ribbon, and base_pose is the only block that carries "
                "where the reference IS -- without it there is nothing to place. Train the "
                "generator on a window that includes it."
            )

        # ONE prediction buffer: the F reference-window nodes, in exactly the layout the
        # policy observes. No world-frame mirror -- ``window_to_world`` derives that on
        # demand for the diagnostic CLF and the debug ribbon, sim-side only.
        if self.cfg.history_from not in ("generator", "clip"):
            raise ValueError(
                f"history_from={self.cfg.history_from!r} must be 'generator' or 'clip'."
            )
        # Printed UNCONDITIONALLY. Which history the generator is eating changes the
        # result more than anything else on this command, and a message that appears only
        # in the non-default case makes "no message" ambiguous with "I did not look".
        if self.cfg.history_from == "clip":
            print("[INFO FmTrackingCommand] history_from='clip': TEACHER FORCING. The "
                  "generator conditions on the clip's nodes, not its own -- a diagnostic "
                  "for separating a weak model from a diverging closed loop. This is NOT "
                  "deployable; there is no clip on the robot.")
        else:
            print(f"[INFO FmTrackingCommand] history_from='generator': CLOSED LOOP -- the "
                  f"generator conditions on its own last {self.ctx.N} served nodes. "
                  f"re-plan K={self.cfg.fm_update_every}"
                  + (" (nodes re-anchored onto the current robot each step)"
                     if int(self.cfg.fm_update_every) > 1 else ""))
        # Last step's CLIP node, for history_from="clip". The history row must be the
        # PREVIOUS step's (see _push_served_node), and _advance_step has already moved the
        # clip on by the time the push happens -- so it is stashed a step ahead.
        self._prev_clip_node: torch.Tensor | None = None
        self._clip_node_valid = torch.zeros(n, dtype=torch.bool, device=dev)

        f, d, k = self.ctx.F, self.ctx.D, self.ctx.num_extra_frames
        self._pred_window = torch.zeros(n, f, d, device=dev)
        # Node currently being served within the window; 0 on a re-plan, +1 per step after.
        self._pred_idx = torch.zeros(n, dtype=torch.long, device=dev)
        # The robot pose the live prediction is expressed RELATIVE TO -- captured at the
        # re-plan. With fm_update_every > 1 the robot moves away from it between re-plans,
        # so every served node has to be re-anchored onto the robot NOW (see
        # :meth:`_reanchor_to_now`). At K=1 these are the current pose and it is a no-op.
        self._anchor_pos = torch.zeros(n, 3, device=dev)
        self._anchor_quat = torch.zeros(n, 4, device=dev)
        self._anchor_quat[:, 3] = 1.0
        # Served node of the PREVIOUS step, already re-anchored onto the robot as it was
        # then -- which is the frame the dataset's history rows use.
        self._prev_served_node: torch.Tensor | None = None
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

        self._build_strict_masks()
        self._setup_draw_selection()

        # --- FM-prediction debug ribbon: map each ribbon frame (base built by the base
        # _post_init, re-ordered above) to its slot in the FM prediction buffers.
        self._fm_pred_specs: list[tuple[str, tuple[float, float, float], str, int]] = []
        # The ee frames come either from the predicted ee_pos block or, when the window
        # drops it, from FK of the predicted joints (viz only -- see _fk_ee_positions).
        # Without either there is nothing to draw but the base, and plotting the
        # placeholder would put every foot at the world origin.
        self._fk_chain = self._build_fk_chain()
        draw_ee = "ee_pos" in self._fm_blocks or self._fk_chain is not None
        for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
            if name == self.cfg.base_frame_name:
                self._fm_pred_specs.append((name, color, "base", -1))
            elif name in self.ctx.frame_names and draw_ee:
                self._fm_pred_specs.append((name, color, "frame", self.ctx.frame_names.index(name)))
        # The ribbon draws EVERY predicted node (dense), independent of the clip
        # ribbon's debug_vis_stride subsampling. It indexes _pred_window directly
        # rather than through the cursor, so it needs no offset table.

    def _setup_draw_selection(self) -> None:
        """Validate and prepare best-of-N oracle selection (``cfg.fm_num_draws``).

        Builds one boolean channel mask per :data:`REFERENCE_BLOCK_KEYS` key present in this
        checkpoint's layout, through ``expand_window_blocks`` -- the same expansion the
        observation noise and scale tables use, so the score's channel grouping cannot drift
        from the layout on its own. Masks are built for EVERY present key, not just the
        weighted ones, so the log can report a magnitude for a key before it is turned on.

        Raises:
            ValueError: On ``fm_num_draws < 1``; on ``fm_num_draws > 1`` combined with
                ``fm_zero_noise`` (identical draws) or with no non-zero weight (nothing to
                select on); or on a weight naming a key this checkpoint does not predict.
        """
        n_draws = int(self.cfg.fm_num_draws)
        if n_draws < 1:
            raise ValueError(f"fm_num_draws must be >= 1, got {n_draws}.")
        self._n_draws = n_draws
        # Diagnostic-only samples, alongside the served one. Meaningless next to selection,
        # which already records its whole candidate set.
        self._n_diag = max(0, int(self.cfg.fm_diag_draws)) if n_draws == 1 else 0
        if int(self.cfg.fm_diag_draws) and n_draws > 1:
            print("[INFO FmTrackingCommand] fm_diag_draws ignored: fm_num_draws > 1 already "
                  "records every candidate draw.")
        total = n_draws if n_draws > 1 else 1 + self._n_diag
        self._n_viz_reject = (max(0, min(int(self.cfg.fm_viz_num_draws), total) - 1)
                              if total > 1 else 0)
        self._viz_draws: torch.Tensor | None = None
        self._select_key_masks: dict[str, torch.Tensor] = {}
        self._select_weights: dict[str, float] = {}
        self._last_draw_stats: dict[str, torch.Tensor] = {}
        self._replan_count = 0
        # --- Capture buffers for the offline stepper (see :meth:`fm_conditioning`).
        # The RAW inputs, before build_*_tokens: normalizing is part of what an offline
        # replay must reproduce, so storing the normalized form would hide a bug in it.
        # Every re-plan overwrites the resetting envs' rows, so a reader gets the
        # conditioning of the plan currently being served.
        self._cap_hist_nodes: torch.Tensor | None = None
        self._cap_extero: torch.Tensor | None = None
        self._cap_cmd: torch.Tensor | None = None
        self._cap_null = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cap_draws: torch.Tensor | None = None
        self._cap_scores: torch.Tensor | None = None
        self._cap_chosen = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # RNG for the diagnostic-only draws, kept off the env's shared stream (see
        # :meth:`_stash_single_draw`). Lazily created so no generator exists unless used.
        self._diag_gen: torch.Generator | None = None
        # Absolute node offsets 0..F-1, for the clip window the draws are scored against.
        self._fm_future_offsets = torch.arange(self.ctx.F, dtype=torch.long, device=self.device)
        if n_draws > 1 and self.cfg.fm_zero_noise:
            raise ValueError(
                "fm_num_draws > 1 with fm_zero_noise=True: integrating from x0 = 0 makes every "
                "draw identical, so best-of-N would be selecting among N copies of one window. "
                "Leave the noise Gaussian for the draws, or set fm_num_draws=1."
            )
        all_keys = sorted({k for keys in REFERENCE_BLOCK_KEYS.values() for k in keys})
        for key in all_keys:
            # One-hot over the keys -> the channels this key owns in the selected blocks.
            onehot = {k: (1.0 if k == key else 0.0) for k in all_keys}
            vec = torch.tensor(
                expand_window_blocks(self.ctx.nj, self.ctx.num_extra_frames, onehot,
                                     self._fm_blocks),
                device=self.device,
            )
            present = vec > 0
            if bool(present.any()):
                self._select_key_masks[key] = present
            weight = float(getattr(self.cfg, f"fm_select_w_{key}"))
            if weight == 0.0:
                continue
            if not bool(present.any()):
                raise ValueError(
                    f"fm_select_w_{key}={weight} weights the {key!r} channels, but this "
                    f"checkpoint's window carries blocks {list(self._fm_blocks)}, which hold "
                    f"none of them -- the score would silently ignore the weight. Set it to 0, "
                    f"or use a checkpoint that predicts that block."
                )
            self._select_weights[key] = weight
        if n_draws > 1 and not self._select_weights:
            raise ValueError(
                "fm_num_draws > 1 but every fm_select_w_* is 0, so all draws score identically "
                "and 'selection' would just take draw 0. Weight at least one key."
            )
        if n_draws == 1:
            # Nothing is SELECTED at one draw, but the single window is still scored against
            # the clip and stashed as a one-entry draw set: it costs one extra clip-window
            # build per re-plan and it makes a deploy-representative capture (one draw, and
            # with zero noise the deployable configuration) readable by exactly the same plots
            # and browser as an oracle capture, and comparable through the same score.
            if self._n_diag:
                print(f"[INFO FmTrackingCommand] fm_diag_draws={self._n_diag}: the served "
                      f"window stays the "
                      f"{'DETERMINISTIC (x0=0)' if self.cfg.fm_zero_noise else 'single Gaussian'}"
                      f" sample -- the physics is unchanged -- and {self._n_diag} extra Gaussian "
                      f"windows are recorded per re-plan for the diagnostics only (draw 0 = "
                      f"served).")
            return
        terms = " + ".join(f"{w:g}*RMS[{k}]" for k, w in self._select_weights.items())
        print(f"[INFO FmTrackingCommand] fm_num_draws={n_draws}: ORACLE best-of-N -- every "
              f"re-plan samples {n_draws} windows and serves the one closest to the CLIP, "
              f"score = {terms}. NOT deployable: it reads the clip at inference. Cost is "
              f"{n_draws}x the generator per re-plan.")
        if self.cfg.history_from == "clip":
            print("[INFO FmTrackingCommand]   NOTE history_from='clip': the SELECTED draw does "
                  "not feed the history, the clip does. Selection then changes only what is "
                  "served, not what the generator conditions on next.")
        if self._n_viz_reject:
            print(f"[INFO FmTrackingCommand]   drawing the chosen ribbon plus the next "
                  f"{self._n_viz_reject} by score, faded "
                  f"(opacity {float(self.cfg.debug_vis_fm_reject_opacity):g}).")

    def _build_strict_masks(self) -> None:
        """Index the output channels the FM does NOT write, for ``strict_fm_only``.

        Everything in ``y_des``/``dy_des`` the generator cannot supply -- the COM block,
        the ``@root`` body-link blocks, any frame the checkpoint does not predict -- is
        clip-derived and must not be read here.

        What it *can* supply is decided by its BLOCK SELECTION, which is why this is not a
        fixed list. A window without ``base_vel`` says nothing about how fast the reference
        base is moving; without ``ee_pos`` it places no end effector; without ``joint_vel``
        it has no joint rates. Those channels are filled with placeholders on the write
        path and NaN'd here, so a CLF that grades one of them fails loudly instead of
        being handed a zero that looks like a perfectly tracked reference.

        Unchanged from before the selection existed: an ee frame the FM DOES predict
        counts as written across all four of its channel groups, even though only its
        position is real (the orientation is an identity quaternion and the twist is
        zero). That is a deliberate looseness -- the ee block has always been
        position-only, and narrowing it here would silently change what an existing
        tracker's CLF grades.

        The ``_wf_*_flat`` scatter indices are frame-major (``[base] + ee``, in
        ``_wf_frames`` order), so the base's own channels are the leading entries and the
        ee frames' are the tail.
        """
        dev = self.device
        blocks = self._fm_blocks
        has_ee = "ee_pos" in blocks

        def _complement(width: int, written: list[torch.Tensor]) -> torch.Tensor:
            keep = torch.ones(width, dtype=torch.bool, device=dev)
            for idx in written:
                keep[idx] = False
            return torch.nonzero(keep, as_tuple=False).squeeze(-1)

        def _frames(flat: torch.Tensor, base_width: int) -> torch.Tensor:
            """The whole frame-major index list, or just the base's leading slice."""
            return flat if has_ee else flat[:base_width]

        # base_pose is required (checked in _post_init), so the base's pose channels are
        # always written.
        pos_written = [_frames(self._wf_pos_flat, 3), _frames(self._wf_ori_flat, 4)]
        if "joint_pos" in blocks:
            pos_written.append(self._joint_slots_pos)
        # The base's twist comes from base_vel; the ee twists are the zero placeholders,
        # written whenever the frames themselves are.
        vel_written: list[torch.Tensor] = []
        if "base_vel" in blocks:
            vel_written += [self._wf_lin_flat[:3], self._wf_ang_flat[:3]]
        if has_ee:
            vel_written += [self._wf_lin_flat[3:], self._wf_ang_flat[3:]]
        if "joint_vel" in blocks:
            vel_written.append(self._joint_slots_vel)

        self._unwritten_pos = _complement(self.y_des.shape[1], pos_written)
        self._unwritten_vel = _complement(self.dy_des.shape[1], vel_written)
        print(f"[INFO FmTrackingCommand] window blocks {list(blocks)}: the generator "
              f"writes {self.y_des.shape[1] - self._unwritten_pos.numel()}/"
              f"{self.y_des.shape[1]} position and "
              f"{self.dy_des.shape[1] - self._unwritten_vel.numel()}/"
              f"{self.dy_des.shape[1]} velocity output channels"
              + (" (the rest are NaN under strict_fm_only -- a CLF grading one of them "
                 "will produce a NaN v)" if self.cfg.strict_fm_only else ""))

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
        """Build BOTH references, apply the configured one, then measured outputs + CLF.

        The clip half runs first and the generator half second, each caching its own
        ``y_des``/``dy_des``/world-frame set; :meth:`_apply_source` then installs one of them.
        Order matters only in that the generator must overwrite the clip in ``y_des`` for
        ``reference_source="fm"`` to be byte-identical to the FM-only implementation -- which
        it is, because every channel the generator does not write is NaN-filled by
        ``strict_fm_only``.
        """
        self._build_clip_reference()
        if self.fm_enabled:
            self._build_fm_reference()
        self._apply_source(self.cfg.reference_source)

        # Measured outputs in the world frame (identity reference -> no alignment), then the
        # CLF. Deliberately AFTER the strict_fm_only fill: the graded channel subset should
        # contain only channels the FM writes, so if one of the NaNs reaches eta then ``v``
        # goes NaN and says so -- which is the whole point of the barrier. (Joint mode grades
        # the base, the world ee frames and the joints; the NaNs are the ``@root`` blocks.)
        self.compute_measured_output(self._zeros3, self._ident_quat)
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot, self.v = vdot, vcur

    def _build_clip_reference(self) -> None:
        """Run the inherited clip path and cache its world-frame outputs.

        Exactly ``MotionSequenceCommand._update_command`` minus the CLF (which grades the
        APPLIED reference and so runs once, in :meth:`_update_command`). ``_advance_step``
        lives here and is what drives the ``target_vel`` look-ahead the generator conditions
        on, so it runs whichever source is applied.
        """
        self._building = "clip"
        try:
            self._advance_step()
            self._reanchor_ego_envs()
            self._refresh_current_placement()
            self.get_desired_outputs()             # clip-world y_des/dy_des
            self._write_world_des(self._clip_world_frames())
        finally:
            self._building = None
        self._y_des_clip = self.y_des.clone()
        self._dy_des_clip = self.dy_des.clone()
        self._clip_wf_frames = self._wf_world_frames

    def _build_fm_reference(self) -> None:
        """Run generator inference and cache its world-frame outputs."""
        # Sit out the reset-time resample, for the same reason the re-anchor snapshot does
        # (see ``MotionSequenceCommand._resample_command``): that call runs with a bogus
        # ``_step``, and ``_snapshot_and_infer`` is likewise guarded to once per
        # ``common_step_counter`` -- so a history row appended here would both be taken
        # around the reset AND stamp the guard, suppressing the real snapshot in the
        # ``compute()`` that follows. Everything below is harmless to run: it is overwritten
        # by that same following call.
        if not self._resampling:
            self._snapshot_and_infer()

        self._building = "fm"
        try:
            fm_base, fm_ee, fm_joints = self._fm_current_outputs()
            self._write_world_des(torch.cat([fm_base[:, None, :], fm_ee], dim=1))   # [N, nf, 13]
            if "joint_pos" in fm_joints:
                self.y_des[:, self._joint_slots_pos] = fm_joints["joint_pos"][:, self._fm_jperm]
            if "joint_vel" in fm_joints:
                self.dy_des[:, self._joint_slots_vel] = fm_joints["joint_vel"][:, self._fm_jperm]
            if self.cfg.strict_fm_only:
                self.y_des[:, self._unwritten_pos] = torch.nan
                self.dy_des[:, self._unwritten_vel] = torch.nan
        finally:
            self._building = None
        self._y_des_fm = self.y_des.clone()
        self._dy_des_fm = self.dy_des.clone()
        self._fm_wf_frames = self._wf_world_frames

    def _apply_source(self, source: str) -> None:
        """Install one cached reference as the live ``y_des``/``dy_des`` + world-frame cache.

        Copies IN PLACE rather than rebinding, because ``get_desired_outputs`` writes into
        ``self.y_des`` with an indexed assignment -- it needs one stable tensor for the whole
        run, and rebinding would silently alias the live vector onto a cache.

        ``_wf_world_frames`` moves with them: it is what
        :meth:`~.motion_sequence_cmd.MotionSequenceCommand.reference_world_state` and the
        recorder read back, so leaving it on the other source would make the world-frame API
        disagree with ``y_des``.

        Args:
            source: ``"clip"`` or ``"fm"``.
        """
        y, dy, frames = ((self._y_des_fm, self._dy_des_fm, self._fm_wf_frames) if source == "fm"
                         else (self._y_des_clip, self._dy_des_clip, self._clip_wf_frames))
        self.y_des.copy_(y)
        self.dy_des.copy_(dy)
        self._wf_world_frames = frames

    # ------------------------------------------------------------------
    # Serving the OTHER source (diagnostics)
    # ------------------------------------------------------------------
    def _serving(self) -> str:
        """Which source the look-ahead accessors currently answer for."""
        return self._serve_override or self.cfg.reference_source

    @contextmanager
    def _serve(self, source: str):
        """Temporarily apply and serve ``source``, restoring the configured one after.

        Swaps BOTH halves of the reference surface, because consumers use both and would
        otherwise see a mix: ``y_des``/``dy_des`` (read by ``base_pose_error``,
        ``joint_pos_ref``, ``ee_ref_relative_to_base``, the CLF) and the three look-ahead
        accessors (read by ``mdp.reference_window``). Swapping only ``y_des`` -- the obvious
        version -- would silently pair one source's current step with the other's future.
        """
        if source not in ("clip", "fm"):
            raise ValueError(f"reference source must be 'clip' or 'fm', got {source!r}.")
        if source == "fm" and not self.fm_enabled:
            raise RuntimeError(
                "as_fm_reference() needs a generator, but this command was built with "
                "reference_source='clip' and no fm_checkpoint (the pure-clip control)."
            )
        prev_override, prev_source = self._serve_override, self._serving()
        self._serve_override = source
        self._apply_source(source)
        try:
            yield self
        finally:
            self._serve_override = prev_override
            self._apply_source(prev_source)

    def as_clip_reference(self):
        """Context manager serving the CLIP reference (outputs + look-ahead) throughout."""
        return self._serve("clip")

    def as_fm_reference(self):
        """Context manager serving the GENERATOR reference (outputs + look-ahead) throughout."""
        return self._serve("fm")

    @property
    def fm_active(self) -> torch.Tensor:
        """``[N]`` bool: whether the APPLIED reference is the generator (uniform per run).

        Per-env rather than a scalar because the diagnostics (``data_logger`` and the
        clip-to-FM transition markers in ``play_plots``) index it per env -- a hangover from
        when the two sources could be mixed within one rollout. They cannot be today: the
        source is a config field.
        """
        return torch.full((self.num_envs,), self.cfg.reference_source == "fm",
                          dtype=torch.bool, device=self.device)

    def predicted_outputs_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole predicted horizon in the ``y_des`` / ``dy_des`` channel layout.

        ``([N, F, P], [N, F, V])``, so a plot can fan every node of one prediction off the
        applied line using the same channel names as the tracked reference. Channels the
        generator does not write are NaN under ``strict_fm_only``, matching the applied
        vector rather than quietly reading zero.

        Diagnostic only -- it lifts the window to world through the robot's measured pose.

        Raises:
            RuntimeError: If this command was built without a generator.
        """
        if not self.fm_enabled:
            raise RuntimeError("predicted_outputs_window() needs a generator; none was loaded.")
        n, f, k = self.num_envs, self.ctx.F, self.ctx.num_extra_frames
        w = self._pred_world(torch.arange(f, device=self.device))
        y = torch.zeros(n, f, self.y_des.shape[1], device=self.device)
        dy = torch.zeros(n, f, self.dy_des.shape[1], device=self.device)
        zeros_ee = self._ee_zero_vec[:, None, :k].expand(n, f, k, 3)
        # [base] + ee, in _wf_frames order -- the same order _write_world_des scatters.
        # Anything the block selection leaves out stays at its placeholder and is NaN'd
        # below, exactly as on the live write path.
        ee_pos = w["ee_pos_w"][:, :, :k] if "ee_pos" in self._fm_blocks else zeros_ee
        pos = torch.cat([w["base_pos_w"][:, :, None, :], ee_pos], dim=2)
        y[:, :, self._wf_pos_flat] = pos.reshape(n, f, -1)
        quat = torch.cat([w["base_quat_w"][:, :, None, :],
                          self._ee_zero_quat[:, None, :k].expand(n, f, k, 4)], dim=2)
        y[:, :, self._wf_ori_flat] = quat.reshape(n, f, -1)
        if "base_vel" in self._fm_blocks:
            dy[:, :, self._wf_lin_flat] = torch.cat(
                [w["base_lin_vel_w"][:, :, None, :], zeros_ee], dim=2).reshape(n, f, -1)
            dy[:, :, self._wf_ang_flat] = torch.cat(
                [w["base_ang_vel_w"][:, :, None, :], zeros_ee], dim=2).reshape(n, f, -1)
        if "joint_pos" in self._fm_blocks:
            y[:, :, self._joint_slots_pos] = w["joint_pos"][..., self._fm_jperm]
        if "joint_vel" in self._fm_blocks:
            dy[:, :, self._joint_slots_vel] = w["joint_vel"][..., self._fm_jperm]
        if self.cfg.strict_fm_only:
            y[:, :, self._unwritten_pos] = torch.nan
            dy[:, :, self._unwritten_vel] = torch.nan
        return y, dy

    def _write_global_base_des(self) -> None:
        """Re-point the base channels at the GLOBAL clip -- for the CLIP half only.

        On the teacher this makes the base say WHERE to be while the links say WHAT SHAPE, so
        the reward reads the same reference the drift termination fires on. That reasoning
        holds for the clip reference here too, and the clip half must reproduce the teacher
        exactly or it is not a baseline.

        Neither half of it survives on the generator. The FM's prediction IS the reference,
        and it is anchored on the robot by construction; there is no globally-placed clip the
        robot is supposed to be at, and the drift termination that motivated the split is not
        in this task's termination set. Letting the base run there would put clip data
        straight into the channel the CLF grades hardest -- exactly what ``strict_fm_only``
        exists to catch.
        """
        if self._building == "fm":
            return
        super()._write_global_base_des()

    def _clip_history_node(self) -> torch.Tensor:
        """The CLIP's window node for the current step, anchored on the robot now.

        The teacher-forcing counterpart of the served node, in the SAME frame the closed loop
        pushes: the PLAN's frame, i.e. the robot as it was at the last re-plan.

        ``reference_window_obs`` anchors on the measured robot at THIS step, so the row is
        re-expressed onto ``_anchor_pos``/``_anchor_quat`` -- the exact inverse of
        :meth:`_reanchor_to_now`, and a no-op at ``fm_update_every=1`` where the two coincide.

        This used to return the robot-now row and claim it was "byte-for-byte the row the
        DATASET builds a history from" (``history_ref[h]`` against ``history_state[h]``). That
        was true until 3b3af11, which moved the recorder to substitute the PLAN anchor into
        history rows ``0..H-2`` -- so the claim silently inverted, and teacher forcing fed the
        generator per-step-anchored history it was never trained on. It measured 158/200 against
        closed_loop's 176/200: below the loop it is supposed to bound.

        Built with the CHECKPOINT's block selection, not the layout default: this row goes
        into the same history ring as the generator's own nodes, so an 82-wide clip node
        cannot be stacked against 41-wide predictions. (It could not, and this is the one
        path that finds out -- ``history_from="clip"`` is the only caller.)

        Forces the clip branch through ``_serve_override`` rather than
        :meth:`as_clip_reference`, deliberately: that context manager also swaps
        ``y_des``/``dy_des`` in place, and this runs mid-way through building the generator's
        reference, where clobbering the live output vector would be a real side effect. Only
        the look-ahead accessors need redirecting.
        """
        prev = self._serve_override
        self._serve_override = "clip"
        try:
            node = super().reference_window_obs(self._zero_offset, self._fm_blocks)
        finally:
            self._serve_override = prev
        # robot-now -> plan anchor. Computed HERE, at stash time, so the row carries the anchor
        # in force at the step it describes; a re-plan between the stash and the push (one step
        # later) must not retroactively change it.
        return reanchor_window(
            node,
            wp.to_torch(self.robot.data.root_pos_w)[:, None, :],
            wp.to_torch(self.robot.data.root_quat_w)[:, None, :],
            self._anchor_pos[:, None, :], self._anchor_quat[:, None, :],
            self.ctx.nj, self.ctx.num_extra_frames, self._fm_blocks,
        )[:, 0]

    def _push_served_node(self) -> None:
        """Push one row into the autoregressive history (see ``cfg.history_from``).

        Normally the node this command SERVED last step -- node 0 of the previous
        prediction, the reference the tracker was actually following. It is already
        expressed relative to the robot as it was at that step, which is exactly the per-row
        anchoring the dataset builds, so there is nothing to transform and no base pose to
        estimate.

        Under ``history_from="clip"`` it is instead the CLIP's node for the previous step:
        teacher forcing, to tell a weak model apart from a diverging loop.

        Either way the row is from the PREVIOUS step, which is the convention the dataset
        uses (the history block is the N rows BEFORE node 0). For the clip that means
        pushing the value stashed last step, not the one available now -- ``_advance_step``
        has already moved the clip on to the current step by the time this runs.

        Before the first inference (and for the first ``N`` steps after a reset) there is
        nothing real to push; zeros go in and :meth:`BatchedHistory.null_mask` tells the model
        to use its learned ``null_history`` embedding instead of reading them.
        """
        if self.cfg.history_from == "clip":
            stash = self._prev_clip_node
            node = (torch.zeros_like(self._pred_window[:, 0]) if stash is None
                    else torch.where(self._clip_node_valid[:, None], stash,
                                     torch.zeros_like(stash)))
            self._prev_clip_node = self._clip_history_node()
            self._clip_node_valid = torch.ones_like(self._pred_valid)
        else:
            stash = self._prev_served_node
            node = (torch.zeros_like(self._pred_window[:, 0]) if stash is None
                    else torch.where(self._pred_valid[:, None], stash, torch.zeros_like(stash)))
        self._history.append({"node": node})

    def _read_extero(self) -> torch.Tensor:
        """Live exteroception reading for the sensor this FM checkpoint was trained on.

        Three things pick themselves rather than being configured here, which is what stops
        a generator being fed something plausible and wrong:

        * the SOURCE comes from the CHECKPOINT (``ctx.extero_source``), so a depth-trained
          FM cannot be handed a height scan just because the env carries both sensors;
        * the CAMERAS come from the env, by group: its own pair when the task built one at
          the generator's resolution, else the policy's (``fm_depth_groups`` returns
          ``None`` = every group = the shared-pair behaviour);
        * the WIDTH is checked on the FIRST reading (:meth:`_check_extero_shape`), not
          per step and not at construction -- see ``_post_init``.

        Returns:
            ``(num_envs, K)`` for the height scan, or ``(num_envs, T, K)`` for depth --
            :func:`build_extero_tokens` flattens either.

        Raises:
            RuntimeError: If the scene lacks the sensor the checkpoint needs.
        """
        src = self.ctx.extero_source
        if src == "height_scan":
            reading = mdp_height_scan(self.env, SceneEntityCfg("height_scanner"))
        elif src == "depth":
            reading = depth_obs_stack(self.env, fm_depth_groups(self.env))
        else:
            raise RuntimeError(f"unknown extero_source {src!r} on the FM checkpoint.")
        if not self._extero_checked:
            self._check_extero_shape(reading)
            self._extero_checked = True
        return reading

    def _check_extero_shape(self, reading: torch.Tensor) -> None:
        """Fail on the first exteroception reading if it is not what the generator expects.

        A SHAPE check, deliberately: the full camera contract comparison lives on the deploy
        side (``deploy.depth_obs.assert_depth_specs_agree``) and ``robot_rl`` must not import
        ``transfer``. What this catches is the case that would otherwise reach the graph --
        a live reading of the wrong width, or the right total width split over the wrong
        number of cameras.

        Args:
            reading: The live reading from :meth:`_read_extero`, passed in rather than
                fetched here so the two cannot recurse.

        Raises:
            RuntimeError: On a sensor-count or per-sensor-width disagreement.
        """
        if reading.ndim == 2:                      # the height scan: one sensor row
            reading = reading.unsqueeze(1)
        count, dim = int(reading.shape[1]), int(reading.shape[2])
        want = (int(self.ctx.extero_sensor_count), int(self.ctx.extero_sensor_dim))
        if (count, dim) != want:
            groups = fm_depth_groups(self.env)
            raise RuntimeError(
                f"the FM checkpoint conditions on {want[0]} sensor row(s) of {want[1]} "
                f"values, but this env serves {count} of {dim} "
                f"(obs group(s): {groups or 'all'}). For depth that is a resolution "
                "mismatch: build the generator's own camera pair with "
                "g1_depth_env_cfg.add_fm_depth_pair(env_cfg, W, H) at the resolution the "
                "checkpoint states (robot_rl.flow_matching.online.read_fm_extero_meta reads "
                "it), which is what play.py does from --fm_checkpoint."
            )

    def _snapshot_and_infer(self) -> None:
        """Push the served node into history and re-run FM inference (once per step).

        Re-plans every ``fm_update_every`` (= K) control steps. At K=1 node 0 is always "now"
        and there is no cursor; at K>1 the cursor walks the live plan and every SERVED node is
        re-anchored onto the robot as it is now -- which costs the relative egomotion since the
        re-plan (see :meth:`_reanchor_to_now`). That odometry is on the serving path only. The
        HISTORY is pushed raw, in its plan's own frame, so the generator's own observation
        never depends on measured state.
        """
        step = int(getattr(self.env, "common_step_counter", self._last_step_seen + 1))
        if step == self._last_step_seen:
            return
        self._last_step_seen = step

        just_reset = self._pending_reset.clone()
        self._pending_reset[just_reset] = False
        if bool(just_reset.any()):
            self._pred_valid = self._pred_valid & ~just_reset
            self._pred_idx = torch.where(just_reset, torch.zeros_like(self._pred_idx),
                                         self._pred_idx)
            # The stashed clip node is from before the reset, so it describes a different
            # episode on a different tile. ``_history.invalidate`` null-masks those rows
            # anyway, but leaving stale data in the ring to be masked is the kind of thing
            # that stops being masked the moment someone changes the warmup.
            self._clip_node_valid = self._clip_node_valid & ~just_reset

        self._history.invalidate(just_reset)
        self._push_served_node()

        # Re-plan on cadence, or wherever there is no live prediction to walk (post-reset).
        reinfer = (self.env.episode_length_buf % max(int(self.cfg.fm_update_every), 1)) == 0
        infer_mask = reinfer | ~self._pred_valid
        # Advance the cursor only where we are NOT re-planning. _check_horizon guarantees F
        # covers (K-1) + max_offset, so this can never run off the end; assert rather than
        # clamp, because a clamp would silently freeze the tail of the observed window.
        advance = self._pred_valid & ~infer_mask
        self._pred_idx = torch.where(advance, self._pred_idx + 1, self._pred_idx)
        if bool((self._pred_idx >= self.ctx.F).any()):
            raise RuntimeError(
                f"FM prediction cursor ran past the horizon (F={self.ctx.F}) at "
                f"fm_update_every={self.cfg.fm_update_every}: the re-plan cadence is not "
                "firing. Check episode_length_buf and the reset latch."
            )

        # Envs whose window has not filled yet (post-reset, or the first steps of a run) are
        # sampled with the learned null-history embedding the FM was trained with, so the
        # generator drives from step 0 and no clip fallback is needed.
        null_hist = self._history.null_mask()
        self._fm_reinfer_mask = infer_mask
        if bool(infer_mask.any()):
            self._run_fm(infer_mask, null_hist)
        if not bool(self._pred_valid.all()):
            raise RuntimeError(
                "FmTrackingCommand has envs with no live FM prediction after inference; the "
                "reference has no clip fallback, so this must never happen."
            )
        # Stash the node being served THIS step, RAW -- in the frame of the plan it came
        # from, NOT re-anchored onto the robot now.
        #
        # That is what keeps odometry out of the FM's observation: the history says "this was
        # my plan", a statement about the model's own past output and nothing else. Since
        # inference only happens at a re-plan, and the N pushes before it all come from the
        # previous plan (cursor K-N .. K-1), the ring holds that plan's tail in ONE frame --
        # exactly ``prev_pred[:, K-N:K]``.
        #
        # At K=1 this is bit-identical to re-anchoring, because the cursor is 0 and the
        # plan's anchor IS the robot now, so the re-anchor was a no-op. It only diverges at
        # K>1, where re-anchoring per step used to difference each row against a DIFFERENT
        # robot pose -- cancelling the reference's motion and collapsing the N rows onto one
        # number (measured: corr 0.87-0.98 across rows).
        #
        # Needs K >= N for all N rows to come from one plan; ``_post_init`` enforces it.
        self._prev_served_node = self._pred_window[self._arange_n, self._pred_idx].clone()

    @torch.no_grad()
    def _run_fm(self, mask: torch.Tensor, null_hist: torch.Tensor | None = None) -> None:
        """Run batched FM inference for the masked envs and store the predicted window.

        ``null_hist`` is the per-env "history not yet complete" mask; those envs are sampled
        with the model's learned ``null_history`` embedding instead of their (incomplete,
        possibly pre-reset) window. ``stack_padded`` keeps the window N long even on the
        first steps of a run -- its content is discarded wherever ``null_hist`` is set.

        The result needs no lift: it IS the reference window, already relative to the robot
        at this step.
        """
        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        nodes = self._history.stack_padded("node")[ids]      # [V, N, D] emitted nodes
        history = build_history_tokens(self.ctx, nodes)
        # Exteroception tokens = the live reading at "now", from whichever sensor this FM
        # checkpoint was TRAINED on -- the same mdp functions the teacher obs uses, so
        # deployment conditions on exactly the training signal.
        extero_raw = self._read_extero()[ids]
        height = build_extero_tokens(self.ctx, extero_raw)
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        cmd_tokens = build_cmd_tokens(self.ctx, fut_cmd)
        null = None if null_hist is None else null_hist[ids]
        self._stash_conditioning(ids, nodes, extero_raw, fut_cmd, null)
        # Node 0 = "now", and the frame it is expressed in is the robot AT THIS INSTANT.
        # Both are needed to serve the window later: the cursor to index it, the anchor to
        # undo the staleness once the robot has moved on.
        #
        # Set BEFORE sampling, not after: best-of-N scores the clip window in the PLAN's
        # frame, which is this anchor. With the old ordering the score would have been taken
        # against the PREVIOUS re-plan's anchor -- exactly the mistake that made teacher
        # forcing measure below the loop it bounds (see _clip_history_node).
        self._pred_idx[ids] = 0
        self._anchor_pos[ids] = wp.to_torch(self.robot.data.root_pos_w)[ids].clone()
        self._anchor_quat[ids] = wp.to_torch(self.robot.data.root_quat_w)[ids].clone()
        if self._n_draws == 1:
            window = sample_window(self.ctx, history, height, cmd_tokens,
                                   history_null_mask=null)
            self._pred_window[ids] = window
            self._stash_single_draw(ids, window, history, height, cmd_tokens, null)
        else:
            self._pred_window[ids] = self._sample_best_of_n(ids, history, height, cmd_tokens, null)
        self._pred_valid[ids] = True

    def _stash_conditioning(self, ids: torch.Tensor, nodes: torch.Tensor,
                            extero_raw: torch.Tensor, fut_cmd: torch.Tensor,
                            null: torch.Tensor | None) -> None:
        """Keep this re-plan's RAW generator inputs, for the offline stepper.

        Together with the checkpoint these four are everything ``sample_window`` consumes, so a
        recording of them replays the generator with no simulator: change the sample count, the
        Euler steps or the guidance and re-sample the same conditioning. Raw rather than
        normalized on purpose -- the normalizers are part of the path under test.
        """
        if self._cap_hist_nodes is None:
            n = self.num_envs
            self._cap_hist_nodes = torch.zeros(n, *nodes.shape[1:], device=self.device)
            self._cap_extero = torch.zeros(n, *extero_raw.shape[1:], device=self.device)
            self._cap_cmd = torch.zeros(n, *fut_cmd.shape[1:], device=self.device)
        self._cap_hist_nodes[ids] = nodes
        self._cap_extero[ids] = extero_raw
        self._cap_cmd[ids] = fut_cmd
        self._cap_null[ids] = (torch.zeros_like(ids, dtype=torch.bool) if null is None else null)

    def fm_conditioning(self) -> dict[str, torch.Tensor]:
        """The generator's raw inputs for the plan each env is currently serving.

        Empty until the first inference. Keys: ``hist_nodes`` ``[N, H, D]``, ``extero``
        ``[N, ...]`` (the sensor this checkpoint was trained on, NOT necessarily the policy's
        cameras), ``cmd_future`` ``[N, F, 3]``, ``hist_null`` ``[N]``, plus the plan's
        ``anchor_pos``/``anchor_quat`` needed to lift any window to the world.
        """
        if self._cap_hist_nodes is None:
            return {}
        return {"hist_nodes": self._cap_hist_nodes, "extero": self._cap_extero,
                "cmd_future": self._cap_cmd, "hist_null": self._cap_null,
                "anchor_pos": self._anchor_pos, "anchor_quat": self._anchor_quat}

    def fm_draws(self) -> dict[str, torch.Tensor]:
        """Every draw of the plan each env is serving, with its score -- empty at ``n=1``.

        ``draws`` ``[N, n, F, D]`` in the window layout (lift with ``window_to_world`` against
        the anchor from :meth:`fm_conditioning`), ``scores`` ``[N, n]`` and ``chosen`` ``[N]``.
        Draw ``chosen[i]`` is the one that became the served window.
        """
        if self._cap_draws is None:
            return {}
        return {"draws": self._cap_draws, "scores": self._cap_scores, "chosen": self._cap_chosen}

    def _clip_future_window(self) -> torch.Tensor:
        """The CLIP's window over nodes ``0..F-1``, in the PLAN's frame -- ``[N, F, D]``.

        The scoring counterpart of :meth:`_clip_history_node`, and built the same two ways at
        once: ``super()`` for the clip's window builder, ``_serve_override`` for the
        look-ahead accessors underneath it (this class overrides both, and only redirecting
        one would quietly score the generator against itself).

        ``reference_window_obs`` anchors on the measured robot at THIS step, so the result is
        re-expressed onto ``_anchor_pos``/``_anchor_quat``. Called from ``_run_fm`` after the
        anchor is refreshed, those two coincide and the re-anchor is a no-op -- it is here so
        that stays true if the call ever moves, since a window scored against the wrong anchor
        measures how far the robot walked rather than how good the draw is.

        Note the clip runs out before the episode does unless the episode is bounded to
        ``clip_len``: past that the library's repeat-last padding freezes, and the oracle then
        selects the draw closest to a stationary reference.
        """
        prev = self._serve_override
        self._serve_override = "clip"
        try:
            win = super().reference_window_obs(self._fm_future_offsets, self._fm_blocks)
        finally:
            self._serve_override = prev
        f = win.shape[1]
        pos_now = wp.to_torch(self.robot.data.root_pos_w)[:, None, :].expand(-1, f, -1)
        quat_now = wp.to_torch(self.robot.data.root_quat_w)[:, None, :].expand(-1, f, -1)
        return reanchor_window(
            win, pos_now, quat_now,
            self._anchor_pos[:, None, :].expand(-1, f, -1),
            self._anchor_quat[:, None, :].expand(-1, f, -1),
            self.ctx.nj, self.ctx.num_extra_frames, self._fm_blocks,
        )

    def _draw_key_distances(
        self, draws: torch.Tensor, clip_win: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-key RMS distance from each draw to the clip window, in RAW units.

        Args:
            draws: ``[V, n_draws, F, D]`` sampled windows.
            clip_win: ``[V, F, D]`` the clip's window over the same nodes and anchor.

        Returns:
            ``{key: [V, n_draws]}`` for every key this checkpoint's layout carries -- weighted
            or not, so the log can state a magnitude for a key before it is weighted.
        """
        sq = (draws - clip_win[:, None]).pow(2)                      # [V, n, F, D]
        return {key: sq[..., chan].mean(dim=(-2, -1)).sqrt()
                for key, chan in self._select_key_masks.items()}

    @torch.no_grad()
    def _sample_best_of_n(
        self,
        ids: torch.Tensor,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None,
        null_hist: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample ``cfg.fm_num_draws`` windows and return the one closest to the clip.

        The draws come from ONE ``sample_window`` call at ``V * n_draws`` batch rather than a
        loop: the noise is drawn per batch row inside the sampler, so repeating each env's
        conditioning ``n_draws`` times gives that many independent draws of the same
        conditioning. ``repeat_interleave`` keeps an env's rows adjacent, which is what makes
        the ``view`` below split them correctly.

        Args:
            ids: ``[V]`` env indices being re-planned.
            history: History tokens for ``ids``.
            extero: Exteroception tokens for ``ids``.
            cmd_tokens: Command tokens for ``ids``, or None.
            null_hist: Per-env null-history mask for ``ids``, or None.

        Returns:
            ``[V, F, D]`` the selected window per env.
        """
        n, v = self._n_draws, int(ids.numel())

        def rep(t: torch.Tensor | None) -> torch.Tensor | None:
            """One copy of each env's conditioning per draw, that env's rows adjacent."""
            return None if t is None else t.repeat_interleave(n, dim=0)

        draws = sample_window(self.ctx, rep(history), rep(extero), rep(cmd_tokens),
                              history_null_mask=rep(null_hist))
        draws = draws.view(v, n, self.ctx.F, self.ctx.D)
        dists = self._draw_key_distances(draws, self._clip_future_window()[ids])
        score = sum(w * dists[key] for key, w in self._select_weights.items())   # [V, n]
        best = torch.argmin(score, dim=1)                                        # [V]
        if self._n_viz_reject:
            self._stash_viz_draws(ids, draws, score)
        # Every draw, for the offline stepper. Lazily sized: n_draws is known but D/F come
        # from the sampler, and this way the buffer cannot disagree with it.
        if self._cap_draws is None:
            self._cap_draws = torch.zeros(self.num_envs, *draws.shape[1:], device=self.device)
            self._cap_scores = torch.zeros(self.num_envs, n, device=self.device)
        self._cap_draws[ids] = draws
        self._cap_scores[ids] = score
        self._cap_chosen[ids] = best
        self._record_draw_selection(score, dists, best)
        return draws[torch.arange(v, device=score.device), best]

    @torch.no_grad()
    def _stash_single_draw(self, ids: torch.Tensor, window: torch.Tensor, history=None,
                           extero=None, cmd_tokens=None, null=None) -> None:
        """Record a served-only re-plan as a draw set: draw 0 served, then any diagnostic draws.

        At ``fm_num_draws == 1`` there is no selection, but the diagnostics should not have to
        care: this makes the deploy-representative configuration (one draw, and with
        ``fm_zero_noise`` the deterministic one) produce the same ``fm_draws`` / ``fm_scores`` /
        ``fm_chosen`` arrays an oracle capture does, so the x-z figures, the viser draws layer
        and the keyboard browser all read it, and its score is comparable to theirs.

        With ``cfg.fm_diag_draws`` set, N extra GAUSSIAN windows are sampled from the same
        conditioning and stored after it. They are never served and never fed to the history --
        the rollout is bit-for-bit what it would be without them -- so the plots can show the
        sampling spread around a deployable trajectory. ``fm_chosen`` stays 0, which is what
        makes the plots highlight the served curve and grey the rest.

        Every entry is scored against the clip; the scores are diagnostic throughout.
        """
        n_total = 1 + self._n_diag
        if self._cap_draws is None:
            self._cap_draws = torch.zeros(self.num_envs, n_total, *window.shape[1:],
                                          device=self.device)
            self._cap_scores = torch.zeros(self.num_envs, n_total, device=self.device)
        self._cap_draws[ids, 0] = window
        self._cap_chosen[ids] = 0
        if self._n_diag and history is not None:
            n, v = self._n_diag, int(ids.numel())

            def rep(t):
                """One copy of each env's conditioning per diagnostic draw."""
                return None if t is None else t.repeat_interleave(n, dim=0)

            # Force Gaussian noise for these regardless of how the SERVED sample is drawn:
            # under fm_zero_noise the context integrates from x0 = 0, and N copies of one
            # deterministic window would show no spread at all -- which is the opposite of
            # what they are for. Restored in `finally` so the next served sample is unaffected.
            #
            # And draw them from a DEDICATED generator. The sampler's default RNG is the global
            # torch stream that the whole env shares, so taking N extra samples from it shifts
            # every later draw in the rollout -- the depth noise and the camera DR among them --
            # which changes the observations and therefore the trajectory. Measured before this
            # was fixed: 4.8 cm of root divergence purely from the diagnostics being present.
            if self._diag_gen is None:
                self._diag_gen = torch.Generator(device=self.device)
                self._diag_gen.manual_seed(0xF00D)
            was_zero = self.ctx.zero_noise
            try:
                self.ctx.zero_noise = False
                extra = sample_window(self.ctx, rep(history), rep(extero), rep(cmd_tokens),
                                      history_null_mask=rep(null), generator=self._diag_gen)
            finally:
                self.ctx.zero_noise = was_zero
            self._cap_draws[ids, 1:] = extra.view(v, n, *window.shape[1:])
        if not self._select_weights:
            return
        clip_win = self._clip_future_window()[ids]
        dists = self._draw_key_distances(self._cap_draws[ids], clip_win)
        score = sum(w * dists[key] for key, w in self._select_weights.items())   # [V, n_total]
        self._cap_scores[ids] = score
        if self._n_viz_reject:
            self._stash_viz_draws(ids, self._cap_draws[ids], score)
        self._record_draw_selection(score, dists, torch.zeros_like(ids))

    def _stash_viz_draws(self, ids: torch.Tensor, draws: torch.Tensor, score: torch.Tensor) -> None:
        """Keep the best REJECTED draws for the debug ribbon (``cfg.fm_viz_num_draws``)."""
        v = score.shape[0]
        # argsort is best-first and its first column is the chosen draw, so the rejects to
        # draw are simply the next few.
        keep = torch.argsort(score, dim=1)[:, 1:1 + self._n_viz_reject]          # [V, M]
        if self._viz_draws is None:
            self._viz_draws = torch.zeros(self.num_envs, self._n_viz_reject,
                                          self.ctx.F, self.ctx.D, device=self.device)
        self._viz_draws[ids] = draws[torch.arange(v, device=score.device)[:, None], keep]

    def _record_draw_selection(
        self, score: torch.Tensor, dists: dict[str, torch.Tensor], best: torch.Tensor
    ) -> None:
        """Stash the selection stats, and print them every ``cfg.fm_select_log_every`` re-plans."""
        rows = torch.arange(score.shape[0], device=score.device)
        self._last_draw_stats = {
            "chosen": best.detach(),
            "score_best": score[rows, best].detach(),
            "score_worst": score.max(dim=1).values.detach(),
            **{f"rms_{key}": value[rows, best].detach() for key, value in dists.items()},
        }
        self._replan_count += 1
        every = int(self.cfg.fm_select_log_every)
        if every <= 0 or self._replan_count % every:
            return
        stats = self._last_draw_stats
        best_s = float(stats["score_best"].mean())
        worst_s = float(stats["score_worst"].mean())
        rms = "  ".join(f"{k[4:]} {float(v.mean()):.4f}"
                        for k, v in stats.items() if k.startswith("rms_"))
        print(f"[fm-draws] re-plan {self._replan_count}: chose {best.tolist()[:8]} of "
              f"{self._n_draws}  score {best_s:.4f} (worst {worst_s:.4f}, "
              f"spread {worst_s - best_s:.4f})  RMS[{rms}]")

    def draw_selection_stats(self) -> dict[str, torch.Tensor]:
        """Last re-plan's best-of-N stats (empty when ``fm_num_draws == 1``).

        ``chosen`` / ``score_best`` / ``score_worst`` plus ``rms_<key>`` for the chosen draw,
        each ``[num_envs]``. For a diagnostic to log per-step; the spread between best and
        worst is what bounds any selection rule.
        """
        return dict(self._last_draw_stats)

    def _pred_world(self, offsets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Lift the served nodes at ``offsets`` to the world frame -- SIMULATION ONLY.

        Lifts through the RE-PLAN anchor, which is the frame the prediction is actually
        expressed in. At ``fm_update_every=1`` that is the robot now and this is what it
        always was; at K>1 using the current pose instead would silently shear the whole
        prediction by however far the robot has walked since the re-plan.

        Exists for the things that genuinely need world coordinates and only exist in sim:
        the diagnostic CLF, the debug ribbon, ``fm_debug``'s overlays and the recorder.

        Args:
            offsets: Offsets from the CURRENT served node.

        Returns:
            The :func:`~robot_rl.flow_matching.window.window_to_world` mapping,
            ``[N, K, ...]`` -- carrying only the blocks the checkpoint predicts, so a
            caller reading e.g. ``base_lin_vel_w`` has to check it is there.
        """
        ar, idx = self._future_indices(offsets)
        return window_to_world(self._pred_window[ar, idx], self._anchor_pos[:, None, :],
                               self._anchor_quat[:, None, :], self.ctx.nj,
                               self.ctx.num_extra_frames, self._fm_blocks)

    def _reanchor_to_now(self, offsets: torch.Tensor) -> torch.Tensor:
        """Served nodes at ``offsets``, re-expressed relative to the robot RIGHT NOW.

        The prediction is anchored on the robot as it was at the re-plan, but the
        observation's contract is "relative to the robot's current actual base". Those
        coincide only at ``fm_update_every=1``; at K>1 they differ by up to ``K-1`` steps of
        egomotion, and serving the stale frame would hand the policy a window that is
        wrong by exactly the distance walked since the re-plan.

        Correcting it costs a world round trip through the MEASURED pose -- which is why
        K=1 was worth pinning while it lasted: this is the odometry dependency the
        representation otherwise does not have. It is exact in simulation.

        Goes through :func:`~robot_rl.flow_matching.window.reanchor_window` rather than
        lifting to the world and back, because only ``base_pose`` and ``base_vel``
        actually reference the anchor -- and because the round trip cannot be completed at
        all for a window that drops a block (the return leg needs every one of them).

        Args:
            offsets: Offsets from the current served node.

        Returns:
            ``[N, K, D]`` window nodes in the current robot frame.
        """
        ar, idx = self._future_indices(offsets)
        return reanchor_window(
            self._pred_window[ar, idx],
            self._anchor_pos[:, None, :], self._anchor_quat[:, None, :],
            wp.to_torch(self.robot.data.root_pos_w)[:, None, :],
            wp.to_torch(self.robot.data.root_quat_w)[:, None, :],
            self.ctx.nj, self.ctx.num_extra_frames, self._fm_blocks,
        )

    def _fm_current_outputs(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The served node lifted to the world frame, for ``y_des`` (diagnostic CLF + viz).

        Only what the checkpoint's block selection actually predicts is real; the rest is
        a placeholder that :meth:`_build_strict_masks` has already marked unwritten, so
        ``strict_fm_only`` NaNs it. Concretely: the ee half is
        ``[pos, identity quat, 0, 0]`` because the ee block is position-only, the ee rows
        are all-placeholder when ``ee_pos`` is not predicted, and the base twist is zero
        when ``base_vel`` is not.

        Returns:
            ``(base [N,13], ee [N,K,13], joints)`` where ``joints`` maps
            ``"joint_pos"`` / ``"joint_vel"`` to ``[N, nj]`` for whichever of the two the
            generator predicts.
        """
        w = self._pred_world(self._zero_offset)
        k = self.ctx.num_extra_frames
        zero3 = self._ee_zero_vec[:, 0]                                       # [N,3]
        base = torch.cat([
            w["base_pos_w"][:, 0], w["base_quat_w"][:, 0],
            w["base_lin_vel_w"][:, 0] if "base_vel" in self._fm_blocks else zero3,
            w["base_ang_vel_w"][:, 0] if "base_vel" in self._fm_blocks else zero3,
        ], dim=-1)                                                            # [N,13]
        ee_pos = (w["ee_pos_w"][:, 0] if "ee_pos" in self._fm_blocks
                  else self._ee_zero_vec[:, :k])
        ee = torch.cat([
            ee_pos, self._ee_zero_quat[:, :k],
            self._ee_zero_vec[:, :k], self._ee_zero_vec[:, :k],
        ], dim=-1)                                                            # [N,K,13]
        joints = {name: w[name][:, 0] for name in ("joint_pos", "joint_vel")
                  if name in self._fm_blocks}
        return base, ee, joints

    # ------------------------------------------------------------------
    # Future window (always the FM prediction -- there is no clip source here)
    # ------------------------------------------------------------------
    def _check_horizon(self, offsets: torch.Tensor) -> None:
        """Raise unless the trained horizon covers ``offsets``.

        Done lazily on the first look-ahead call rather than in ``_post_init``, because the
        observation manager (which owns the offsets, via ``reference_window``'s ``num_steps``
        / ``stride``) is built after the command manager. Re-checks only when a caller asks
        for a larger offset than any seen so far.

        The requirement is ``F >= max_offset + K``: at a re-plan period of K the cursor sits
        up to K-1 nodes past the re-plan when the furthest sample is read. Nothing clamps: a
        short horizon is an error, not a frozen tail.
        """
        max_off = int(offsets.max().item())
        if max_off <= self._checked_max_offset:
            return
        k = int(self.cfg.fm_update_every)
        need = max_off + k
        if need > self.ctx.F:
            raise ValueError(
                f"The FM checkpoint predicts F={self.ctx.F} nodes, but serving offset "
                f"{max_off} at a re-plan period of {k} needs F >= {need} (the cursor is up "
                f"to {k - 1} nodes past the re-plan when the furthest sample is read). "
                f"Re-record with --ref_future_len >= {need} and retrain with "
                f"--future_len {need}."
            )
        self._checked_max_offset = max_off

    def _future_indices(self, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(env_index, node_index)`` pair for gathering ``offsets`` from the served node.

        Offsets are relative to the CURSOR, not to the start of the window: node
        ``_pred_idx + offset``. The cursor is 0 on a re-plan and advances one node per step
        between re-plans, so "offset 0" always means "now" whatever the re-plan period.
        """
        self._check_horizon(offsets)
        idx = self._pred_idx[:, None] + offsets[None, :]        # [N, K]
        return self._arange_n[:, None].expand_as(idx), idx

    def reference_window_obs(self, offsets: torch.Tensor, blocks=None) -> torch.Tensor:
        """The predicted window, served straight through -- no world round-trip.

        This is the payoff of the shared representation: the generator emits exactly the
        vector ``mdp.reference_window`` feeds the policy, so on the FM branch this is a
        gather. The clip branch still goes through the inherited world-frame maths.

        Raises:
            ValueError: If the observation term asks for a different block selection than
                the checkpoint predicts. There is nothing to drop or reorder here -- the
                prediction goes into the observation verbatim -- so a disagreement means
                the policy would read one block's numbers under another's name.
        """
        if self._serving() == "clip":
            return super().reference_window_obs(offsets, blocks)
        wanted = resolve_reference_blocks(blocks)
        if wanted != self._fm_blocks:
            raise ValueError(
                f"the reference_window observation asks for blocks {list(wanted)} but the "
                f"FM checkpoint predicts {list(self._fm_blocks)}. The generator's output "
                "IS this observation, so the two must be the same vector. Retrain the "
                "generator against this task's window (train.py takes the selection from "
                "the recording, so re-record with this teacher), or set the term's "
                "`blocks` to the checkpoint's."
            )
        return self._reanchor_to_now(offsets)

    def get_future_world_base(
        self, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame future reference base pose+velocity for the SERVED source.

        Defaults to ``cfg.reference_source``; inside :meth:`as_clip_reference` /
        :meth:`as_fm_reference` it answers for that one instead. The clip branch is the
        inherited implementation verbatim; the FM branch lifts its window through the
        robot's measured pose, which only simulation has (see :meth:`_pred_world`).

        Raises:
            ValueError: On the FM branch, if the checkpoint does not predict ``base_vel``
                -- there is no velocity to return and a zero would be read as "the
                reference is standing still".
        """
        if self._serving() == "clip":
            return super().get_future_world_base(offsets)
        if "base_vel" not in self._fm_blocks:
            raise ValueError(
                "get_future_world_base returns the reference base VELOCITY, which this FM "
                f"checkpoint does not predict (its window packs {list(self._fm_blocks)}). "
                "Ask for the pose alone, or train a generator that includes base_vel."
            )
        w = self._pred_world(offsets)
        return w["base_pos_w"], w["base_quat_w"], w["base_lin_vel_w"], w["base_ang_vel_w"]

    def get_future_joint_state(self, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Future joint pos/vel ``([N, K, nj], [N, K, nj])`` for the SERVED source.

        In the **robot's** joint order on both branches. On the FM branch that needs no
        permutation any more: the window's joint block is defined in robot order and
        ``_post_init`` refuses a checkpoint that disagrees.

        A window may carry ``joint_pos`` without ``joint_vel`` (the g1 default does), and
        the two halves are asked for separately in practice -- the diagnostics pose a ghost
        from the positions alone. So a missing VELOCITY half comes back NaN rather than
        taking the positions down with it; a missing position half is fatal, since there is
        nothing useful left to return.

        Raises:
            ValueError: On the FM branch, if the checkpoint does not predict ``joint_pos``.
        """
        if self._serving() == "clip":
            return super().get_future_joint_state(offsets)
        if "joint_pos" not in self._fm_blocks:
            raise ValueError(
                "get_future_joint_state needs 'joint_pos', which this FM checkpoint does "
                f"not predict (its window packs {list(self._fm_blocks)}). There is no clip "
                "fallback in this command."
            )
        w = self._pred_world(offsets)
        jp = w["joint_pos"]
        jv = w["joint_vel"] if "joint_vel" in self._fm_blocks else torch.full_like(jp, torch.nan)
        return jp, jv

    def get_future_world_frames(self, frame_names: list[str], offsets: torch.Tensor) -> torch.Tensor:
        """Future end-effector state ``[N, K, F, 13]`` for the SERVED source.

        Layout matches the clip implementation -- ``[pos_w(3), quat_w(4 xyzw), lin_w(3),
        ang_w(3)]`` -- but the FM predicts POSITION only, so on that branch the trailing 10
        channels are an identity quaternion and zeros: placeholders, not a stale lift of
        clip data.

        Raises:
            ValueError: If a requested frame is not one the checkpoint predicts, or if the
                checkpoint's window has no ``ee_pos`` block at all.
        """
        if self._serving() == "clip":
            return super().get_future_world_frames(frame_names, offsets)
        if frame_names and "ee_pos" not in self._fm_blocks:
            raise ValueError(
                f"get_future_world_frames asked for {list(frame_names)}, but this FM "
                f"checkpoint's window packs {list(self._fm_blocks)} -- it predicts no end "
                "effector positions at all. There is no clip fallback in this command."
            )
        missing = [f for f in frame_names if f not in self.ctx.frame_names]
        if missing:
            raise ValueError(
                f"get_future_world_frames asked for {missing}, which the FM checkpoint does "
                f"not predict (it has {list(self.ctx.frame_names)}). There is no clip "
                "fallback in this command."
            )
        sel = torch.tensor([self.ctx.frame_names.index(f) for f in frame_names],
                           dtype=torch.long, device=self.device)
        pos = self._pred_world(offsets)["ee_pos_w"][:, :, sel]            # [N, K, F, 3]
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

        Also bails when ``_post_init`` did not finish. Isaac Lab registers this callback
        from ``CommandTerm.__init__``, i.e. BEFORE ``_post_init`` runs, so a config error in
        there (a stale FM checkpoint, say) leaves a half-built term that the callback then
        walks -- turning one legible ValueError into three stacked tracebacks with the real
        cause buried in the middle. ``_fm_pred_specs`` is set late in ``_post_init``, after
        the checkpoint load and every buffer the ribbon reads, so its absence is a good
        proxy for "not finished building".
        """
        if not self._scene_alive() or getattr(self, "_fm_pred_specs", None) is None:
            return
        super()._debug_vis_callback(event)
        if self.fm_enabled and self.cfg.debug_vis_show_fm_prediction:
            self._draw_fm_prediction_ribbon()

    def _clip_ribbon_env_mask(self) -> torch.Tensor | None:
        """Draw the inherited CLIP ribbon only when the clip is what the robot tracks.

        With ``reference_source="fm"`` no env tracks the clip, so the ribbon would be a
        distraction pointing at a reference nothing follows. With ``"clip"`` it is the
        applied reference and the FM ribbon becomes the overlay -- which is exactly the
        picture the diagnostic wants.
        """
        if self.cfg.reference_source == "clip":
            return None
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _fk_ee_positions(self, world: dict[str, torch.Tensor]) -> torch.Tensor:
        """End-effector positions implied by the predicted JOINTS -- viz only.

        A window that drops ``ee_pos`` has not lost the end effectors, only the explicit
        channels: the posture is still fully determined by ``joint_pos``, and the base
        pose places it in the world. Running FK recovers what the ee block used to carry
        -- indeed the ``fk_pose`` loss existed precisely to force the two to agree, so on
        a window that HAS both they are the same thing to within that loss.

        **Diagnostic only, deliberately.** Nothing feeds this back into ``y_des``, the CLF
        or the observation: those must show what the generator actually predicted, and a
        channel reconstructed from a different block is not that. It is here so the ribbon
        can draw a recognisable pose instead of a single base dot.

        Args:
            world: The :func:`window_to_world` mapping for the predicted window; must
                carry ``base_pos_w`` / ``base_quat_w`` / ``joint_pos``.

        Returns:
            ``[N, F, K, 3]`` world-frame ee positions, or a zero-width tensor when the FK
            chain is unavailable (no joints predicted, or the MJCF failed to load).
        """
        n, f = world["base_pos_w"].shape[:2]
        k = self.ctx.num_extra_frames
        if self._fk_chain is None or "joint_pos" not in world:
            return world["base_pos_w"].new_zeros(n, f, k, 3)
        q = world["joint_pos"].reshape(n * f, self.ctx.nj)
        rel, _rot, _jv, _jw = self._fk_chain.pose_and_jacobian(q)      # (n*f, K, 3), base frame
        rel = rel.reshape(n, f, k, 3)
        base_q = world["base_quat_w"][:, :, None, :].expand(n, f, k, 4).reshape(-1, 4)
        return (world["base_pos_w"][:, :, None, :]
                + quat_apply(base_q, rel.reshape(-1, 3)).reshape(n, f, k, 3))

    def _build_fk_chain(self):
        """An :class:`~motion_gen.fk.FKChain` for the ribbon, or ``None``.

        Built only when the ribbon would otherwise have nothing but the base to draw --
        i.e. the window predicts joints but not ee positions. Failure is non-fatal: the
        ribbon degrades to the base dot rather than taking the run down for a viz.
        """
        if "ee_pos" in self._fm_blocks or "joint_pos" not in self._fm_blocks:
            return None
        if not self.ctx.frame_names:
            return None
        try:
            from motion_gen.features import DEFAULT_MJCF                    # noqa: PLC0415
            from motion_gen.fk import FKChain                               # noqa: PLC0415
            chain = FKChain(DEFAULT_MJCF, list(self.ctx.joint_names),
                            list(self.ctx.frame_names),
                            base_link=self.cfg.base_frame_name).to(self.device)
        except Exception as exc:
            print(f"[WARN FmTrackingCommand] FM ribbon: could not build the FK chain for the "
                  f"ee frames ({exc}); drawing the predicted base only.")
            return None
        print(f"[INFO FmTrackingCommand] FM ribbon: this window predicts no ee_pos, so the "
              f"{len(self.ctx.frame_names)} ee frames are drawn from FK of the predicted "
              "joints (viz only -- y_des and the CLF still see only what was predicted).")
        return chain

    def _draw_fm_prediction_ribbon(self) -> None:
        """Draw the FM's predicted world-frame trajectory as per-frame spheres."""
        if self._fm_markers is None:
            self._fm_markers = self._create_fm_markers()
            if self._fm_markers is None:
                return
        if not self._fm_pred_specs:
            self._fm_markers.set_visibility(False)
            return
        # Lift the WHOLE predicted window directly, NOT through _pred_world/_future_indices.
        # Those exist for SERVING: they offset from the cursor and apply the serving horizon
        # check (F >= offset + fm_update_every). The ribbon wants absolute nodes 0..F-1, so
        # routing it through them (a) raises as soon as fm_update_every > 1 -- offset F-1
        # plus K exceeds F, which killed the callback every tick at K=12 -- and (b) poisons
        # _checked_max_offset with a much larger offset than the observation ever asks for,
        # so the real horizon check silently stops running.
        world = window_to_world(self._pred_window, self._anchor_pos[:, None, :],
                                self._anchor_quat[:, None, :], self.ctx.nj,
                                self.ctx.num_extra_frames, self._fm_blocks)
        if "ee_pos_w" not in world:
            world["ee_pos_w"] = self._fk_ee_positions(world)
        translations, marker_indices = self._ribbon_points(world, 0)
        # The rejected draws, faded: their own prototype block (appended after the chosen
        # ones in _create_fm_markers, hence the offset) plus a per-instance scale, which IS
        # available where per-instance colour is not.
        scales = None
        if self._n_viz_reject and self._viz_draws is not None:
            offset = len(self._fm_pred_specs)
            shrink = float(self.cfg.debug_vis_fm_reject_scale)
            n_chosen = sum(t.shape[0] for t in translations)
            for m in range(self._n_viz_reject):
                rej = window_to_world(self._viz_draws[:, m], self._anchor_pos[:, None, :],
                                      self._anchor_quat[:, None, :], self.ctx.nj,
                                      self.ctx.num_extra_frames, self._fm_blocks)
                if "ee_pos_w" not in rej:
                    rej["ee_pos_w"] = self._fk_ee_positions(rej)
                t, i = self._ribbon_points(rej, offset)
                translations += t
                marker_indices += i
            n_reject = sum(t.shape[0] for t in translations) - n_chosen
            scales = torch.cat([
                torch.ones(n_chosen, 3, device=self.device),
                torch.full((n_reject, 3), shrink, device=self.device),
            ], dim=0)
        self._fm_markers.set_visibility(True)
        self._fm_markers.visualize(
            translations=torch.cat(translations, dim=0),
            marker_indices=torch.cat(marker_indices, dim=0),
            scales=scales,
        )

    def _ribbon_points(
        self, world: dict[str, torch.Tensor], proto_offset: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """One ribbon's sphere positions and prototype indices, per drawn frame.

        Args:
            world: A :func:`window_to_world` mapping for one window.
            proto_offset: Added to each prototype index -- 0 for the chosen draw's opaque
                prototypes, ``len(_fm_pred_specs)`` for the faded reject block.

        Returns:
            ``(translations, marker_indices)`` lists, one entry per drawn frame.
        """
        translations, marker_indices = [], []
        for proto_idx, (_name, _color, kind, k) in enumerate(self._fm_pred_specs):
            pts = (world["base_pos_w"] if kind == "base"
                   else world["ee_pos_w"][:, :, k]).reshape(-1, 3)
            translations.append(pts)
            marker_indices.append(torch.full((pts.shape[0],), proto_idx + proto_offset,
                                             dtype=torch.long, device=self.device))
        return translations, marker_indices

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
        # A second, low-opacity copy of every frame's prototype for the non-chosen draws.
        # Appended AFTER the opaque block, and dict order is prototype-index order, so the
        # reject indices are exactly ``len(_fm_pred_specs) + i`` (see _ribbon_points).
        if self._n_viz_reject:
            opacity = float(self.cfg.debug_vis_fm_reject_opacity)
            prototypes.update({
                f"fmrej_{name}": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(color), opacity=opacity,
                    ),
                )
                for name, color, _kind, _k in self._fm_pred_specs
            })
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/fm_pred", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN FmTrackingCommand] could not create FM markers: {exc}")
            return None
