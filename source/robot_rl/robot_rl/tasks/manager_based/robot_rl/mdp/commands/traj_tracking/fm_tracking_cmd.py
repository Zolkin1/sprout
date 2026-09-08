# Copyright 2026 Zachary Olkin. All rights reserved.

"""Closed-loop flow-matching tracking command (world-frame CLF).

Extends :class:`MotionSequenceCommand`: every env tracks the **flow-matching generator**,
which conditions on an ``N``-row history of reference-window nodes, an exteroception token,
and the future velocity command, and predicts the F-step ego-anchored clip future. Because
the FM is anchored on the robot's own xyz+yaw pose each re-plan and predicts node 0 = "now"
(offset ≈ 0), no re-centering hack is needed -- the prediction is already in the robot's
frame. This is the deploy counterpart of the recorded ``ego_anchored_clip_future`` training
data (each re-inference == a snap).

**The history rows are NOT the measured robot state**, whatever an older version of this
docstring said. By default they are the nodes this command SERVED at the previous ``N``
steps -- the model's own output, which is what makes the loop autoregressive and what keeps
odometry off the generator's critical path. ``cfg.history_from`` selects the producer:
``"generator"`` (that default), ``"clip"`` (teacher forcing) or ``"measured"`` (the robot's
achieved state, which really is measured, and is off-distribution because of it).

**The reference is the FM and only the FM, from control step 0.** There is no clip warmup
fallback: before the N-step history window fills, the generator is sampled with the learned
``null_history`` embedding it was trained with (``null_cond_prob``), which is the
in-distribution way to cover the post-reset steps and the only one deployment has. What the
clip library still supplies is exactly two things, both unavoidable in simulation:

1. the **reset spawn pose**, and the terrain mesh built from the same library;
2. the **``target_vel`` command** -- the joystick stand-in that conditions the FM and feeds
   the policy's ``command_window``. On hardware this is a real joystick.

**On the FREE-FORM terrain sections neither of those exists**, because that library carries a
manifest and no clips at all. ``cfg.freeform_command`` (off by default, so every clip task is
untouched) replaces the second: the command is generated here -- a per-episode forward speed
after a standing hold, ``vy`` pinned to 0, and a yaw rate from a P controller on the bearing to
a point down the section's centreline -- and it is HELD across every window offset and across
the generator's whole horizon, which is what both deploy runtimes do. See
:func:`freeform_yaw_rate` for the law and for the distribution change steering on a staircase
is. The spawn is the reset event's problem
(``mdp.events.resets.reset_on_freeform_section``).

**And the CLIP HALF of every step is skipped there.** This class builds BOTH references every
step and applies one, which needs a clip to build the first from; with none, ``has_clips`` is
False and :meth:`FmTrackingCommand._advance_without_a_clip` runs in place of
:meth:`FmTrackingCommand._build_clip_reference`, keeping only the plan-anchor latch. The clip
step therefore never advances, ``as_clip_reference()`` is refused, and every manager gather
raises by name -- so a clip-consuming term left wired on that task fails loudly instead of
tracking a reference nobody wrote. The permission is per configuration
(:attr:`FmTrackingCommand.accepts_clip_free_library`): under ``reference_source="clip"`` a
library with no clips is refused at construction, here as on the base class.

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

from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.observations.height_scan import (
    fm_scan_sensor,
    height_scan_specs,
    scan_obs_stack,
)
from robot_rl.flow_matching.config import default_depth_lag_steps
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
    FM_DEPTH_OBS_GROUP,
    depth_obs_stack,
    depth_xyz_stack,
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
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_matrix

from motion_gen.plan_geometry import FOOT_BODIES

#: Colour of the ribbon marker for a plan node the plan's own kinematics call a STANCE.
#:
#: Deliberately not one of ``_DEBUG_FRAME_COLOR_MAP``'s per-frame colours: the point is to
#: read contact against the frame colour, so it has to be distinguishable from every one of
#: them. Drawn at 1.35x the swing radius as well, because a colour alone is hard to pick out
#: in a dense ribbon.
DEBUG_CONTACT_COLOR: tuple[float, float, float] = (0.15, 0.85, 0.35)

from robot_rl.flow_matching.window import (
    DEFAULT_WINDOW_ANCHOR,
    MEASURED_HISTORY_FALLBACK,
    assert_history_source_matches,
    measured_history_fallback_spans,
    reanchor_window,
    resolve_reference_blocks,
    window_to_world,
    world_to_window,
)

from .world_lift_isaac import yaw_from_quat

from .motion_sequence_cmd import MotionSequenceCommand
from .fm_tracking_cmd_cfg import FmTrackingCommandCfg


#: Body-frame lateral velocity on the free-form sections, always exactly this. The deployed
#: controller ships ``v_y_scale: 0.0`` and the training library measures ``|vy| max = 0.0000``
#: over all 3500 stair clips, so a non-zero channel here would train something the deployed
#: pair cannot ask for.
FREEFORM_VY: float = 0.0


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle (rad) to ``[-pi, pi]``, elementwise.

    Via ``atan2(sin, cos)`` rather than a modulo, matching
    ``transfer/sim/mjeval/runner.steer_yaw_rate``: that is what makes a heading either side of
    the target turn the SHORT way, and the two forms disagree at the branch.

    **Which sign an exactly-antiparallel input comes back with is not defined here** and
    nothing may depend on it: it follows the sign of the floating-point ``sin``, and
    ``float32(pi)`` rounds ABOVE pi, so ``wrap_to_pi(pi)`` measures -3.1415925 rather than
    +pi. Only the magnitude is meaningful there, and a robot pointed exactly backwards is a
    measure-zero case the controller resolves on the next step either way.

    Args:
        angle: Any shape.

    Returns:
        The same shape, every entry in ``[-pi, pi]``.
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def freeform_yaw_rate(
    pos_xy: torch.Tensor,
    yaw: torch.Tensor,
    origin_xy: torch.Tensor,
    heading: torch.Tensor,
    lookahead: float,
    kp: float,
    wz_max: float,
) -> torch.Tensor:
    """Proportional yaw rate aiming the robot at a point ``lookahead`` metres down the section.

    The vectorised twin of ``transfer/sim/mjeval/runner.steer_yaw_rate`` -- same law, same
    wrap, same clip -- differing only in that the section is an infinite straight CENTRELINE
    rather than a goal point, so the aim point recedes with the robot instead of being reached.
    Project the root onto the centreline, step ``lookahead`` further along it, take the bearing
    to that point, and drive the heading error to zero.

    **The effective CROSS-TRACK gain is ``kp / lookahead``, not ``kp``.** With the robot already
    pointing down the section and a signed lateral offset ``e``, the bearing error is
    ``atan2(-e, lookahead) ~= -e / lookahead``, so ``wz ~= -(kp / lookahead) * e``. The two
    knobs are therefore coupled: at the shipped 1.0 and 5.0 m that is 0.2 1/s, which is a weak
    restoring gain across a 1.219 m flight. Shorten the look-ahead before raising ``kp``.

    **This runs EVERYWHERE, the flight included, and that is a deliberate distribution change.**
    The generator was commanded ``|wz| max = 0.0000`` over all 3500 stair clips of the training
    library (measured; see ``mjeval/runner.steer_yaw_rate``, which states the same number and is
    opt-in there for this reason), and the generator is FROZEN during the fine-tune. So on a
    staircase this asks it to extrapolate in wz. It is on anyway because the free-form sections
    are metres long and nothing else in the loop steers: without it, a few degrees of
    accumulated heading error walks the robot off the side of the flight, and what gets measured
    is the drift rather than the terrain.

    Args:
        pos_xy: ``[N, 2]`` root world position.
        yaw: ``[N]`` root heading (rad), world frame.
        origin_xy: ``[N, 2]`` any point on the section's centreline, world frame.
        heading: ``[N]`` centreline direction (rad), world frame -- the sense the robot
            traverses in, so the aim point is always ahead.
        lookahead: Metres along the centreline past the robot's projection. Must be ``> 0``.
        kp: Proportional gain (1/s). ``0`` returns exactly zero.
        wz_max: Symmetric clip (rad/s).

    Returns:
        ``[N]`` commanded yaw rate (rad/s).

    Raises:
        ValueError: If ``lookahead <= 0``, where the aim point would sit on or behind the
            robot's own projection and the bearing would stop meaning anything.
    """
    if lookahead <= 0.0:
        raise ValueError(
            f"freeform_yaw_rate needs a positive look-ahead, got {lookahead}. At 0 the aim "
            "point is the robot's own projection onto the centreline and the bearing is "
            "undefined; negative aims it backwards."
        )
    direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)   # [N, 2]
    along = ((pos_xy - origin_xy) * direction).sum(dim=-1)                      # [N] downrange
    target = origin_xy + (along + lookahead)[:, None] * direction               # [N, 2]
    delta = target - pos_xy
    bearing = torch.atan2(delta[:, 1], delta[:, 0])
    return torch.clamp(kp * wrap_to_pi(bearing - yaw), min=-wz_max, max=wz_max)


class FmTrackingCommand(MotionSequenceCommand):
    """Track a robot-state-conditioned flow-matching generator in the world frame."""

    #: Synthetic camera lag for the GENERATOR, in control steps. Defaulted at CLASS level so
    #: an object that skips construction -- the test doubles in tests/ do -- still reads as
    #: "no lag" rather than raising AttributeError deep in the re-plan path.
    _depth_lag_steps: int = 0
    #: Ring buffer behind that lag. Immutable at class level on purpose: it is only ever
    #: appended to after __init__ rebinds it per instance, and a shared mutable default would
    #: leak frames between environments.
    _extero_buf: tuple = ()
    #: Optional per-env flow-matching noise provider; see :meth:`set_noise_source`.
    _noise_source = None

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
        if not self.has_clips:
            # Reached only under reference_source="fm" -- ``super()._post_init()`` above has
            # already refused every other configuration via ``accepts_clip_free_library``.
            # Printed because a library with no motion in it is the single most surprising
            # thing about this run, and it is otherwise invisible: the file loads, the terrain
            # meshes and the task starts.
            print(
                "[INFO FmTrackingCommand] the clip library carries a terrain manifest and ZERO "
                f"CLIPS ({self.cfg.path}). The clip half of every step is SKIPPED -- no clip "
                "reference is built, the clip step never advances, as_clip_reference() is "
                "refused and every manager gather raises. The generator is the only reference. "
                "This is the free-form fine-tune; the spawn, the velocity command and the "
                "episode bound must all come from somewhere other than a clip (see "
                "G1FinetuneFreeformEnvCfg)."
            )
            self._refuse_clip_phase_knobs()
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
        #: Nodes 0 and 1 of the prediction, for the finite-difference joint velocity
        #: the body FK Jacobian needs (see _write_fm_root_body_frames).
        self._first_two_offsets = torch.arange(2, dtype=torch.long, device=dev)

        # The free-form velocity command. Allocated BEFORE the early returns below, for the
        # same reason the reference caches are: the four command-velocity accessors are public
        # and must never reach an attribute that a skipped construction path never wrote.
        self._ff_cmd = torch.zeros(n, 3, device=dev)
        self._ff_speed = torch.zeros(n, device=dev)
        self._ff_origin_w = torch.zeros(n, 2, device=dev)
        self._ff_heading_w = torch.zeros(n, device=dev)
        self._ff_hold_steps = 0
        if self.freeform_command:
            self._init_freeform_command()

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
        # Synthetic camera lag for the GENERATOR. Resolved once: explicit cfg wins, else the
        # checkpoint's own trained distribution, else none (a pre-latency graph).
        _lag_cfg = getattr(self.cfg, "fm_depth_lag_steps", None)
        self._depth_lag_steps = (int(_lag_cfg) if _lag_cfg is not None
                                 else default_depth_lag_steps(
                                     getattr(self.ctx, "depth_latency_probs", None)))
        self._depth_lag_steps = max(self._depth_lag_steps, 0)
        self._extero_buf: list[torch.Tensor] = []
        if self._depth_lag_steps:
            print(f"[INFO FmTrackingCommand] FM camera lag {self._depth_lag_steps} step(s) = "
                  f"{20 * self._depth_lag_steps} ms; the generator is conditioned on a frame "
                  "that old, with the camera pose it was captured at. The teacher's own depth "
                  "observation is untouched.")

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

        if getattr(self.cfg, "tracking_mode", "joint") == "body" and self.cfg.reference_source == "fm":
            # Allowed, but only because the body links are DERIVED -- see _body_fk.
            # Everything the body CLF grades has to be written, or strict_fm_only NaNs it
            # and v goes NaN, so this is a hard dependency rather than a nicety.
            if "joint_pos" not in self._fm_blocks:
                raise ValueError(
                    "tracking_mode='body' with reference_source='fm' derives the 13 '@root' "
                    "body links by FK from the generator's predicted joint angles, but this "
                    "checkpoint does not predict joint_pos (blocks: "
                    f"{sorted(self._fm_blocks)}). There is no posture to run FK on."
                )
        if "base_pose" not in self._fm_blocks:
            raise ValueError(
                f"the FM checkpoint predicts {list(self._fm_blocks)}, with no 'base_pose'. "
                "This command lifts the prediction to the world for y_des, the diagnostic "
                "CLF and the debug ribbon, and base_pose is the only block that carries "
                "where the reference IS -- without it there is nothing to place. Train the "
                "generator on a window that includes it."
            )

        # For history_from="measured": the robot bodies whose MEASURED positions fill the
        # ee_pos block, in the same order ``reference_window_obs`` gathers the reference's,
        # and the node columns a robot cannot measure. Both resolved once, and only where
        # the checkpoint's layout actually carries them -- a node without ``ee_pos`` needs
        # no body lookup and one without ``base_vel`` has no linear half to fall back on.
        self._measured_ee_idx: torch.Tensor | None = None
        if "ee_pos" in self._fm_blocks:
            body_names = list(self.robot.data.body_names)
            self._measured_ee_idx = torch.tensor(
                [body_names.index(nm) for nm in self.resolve_ee_frames(None)],
                dtype=torch.long, device=dev,
            )
        self._measured_fallback_spans = measured_history_fallback_spans(
            self.ctx.nj, self.ctx.num_extra_frames, self._fm_blocks)

        # ONE prediction buffer: the F reference-window nodes, in exactly the layout the
        # policy observes. No world-frame mirror -- ``window_to_world`` derives that on
        # demand for the diagnostic CLF and the debug ribbon, sim-side only.
        if self.cfg.history_from not in ("generator", "clip", "measured"):
            raise ValueError(
                f"history_from={self.cfg.history_from!r} must be 'generator', 'clip' or "
                "'measured'."
            )
        # The arm has to be one the CHECKPOINT was fitted for. Identical row widths either
        # way, so a mismatch is a silently wrong conditioning rather than a load error.
        _ckpt_src = str(getattr(self.ctx, "history_source", None) or "reference")
        if self.cfg.allow_history_source_mismatch:
            # Loud even when it agrees: the flag being ON is the thing worth knowing, since
            # it disables the only check that can see a wrong-kind history.
            print("[WARN FmTrackingCommand] allow_history_source_mismatch=True: serving "
                  f"history_from={self.cfg.history_from!r} against a checkpoint fitted on "
                  f"history_source={_ckpt_src!r}. Legitimate only when these rollouts are "
                  "the training set that moves it onto this arm; anything measured under "
                  "the mismatch describes a model fed the wrong kind of history.")
        else:
            assert_history_source_matches(
                self.cfg.history_from, _ckpt_src,
                where=str(self.cfg.fm_checkpoint or "the FM checkpoint"))
        # Printed UNCONDITIONALLY. Which history the generator is eating changes the
        # result more than anything else on this command, and a message that appears only
        # in the non-default case makes "no message" ambiguous with "I did not look".
        if self.cfg.history_from == "clip":
            print("[INFO FmTrackingCommand] history_from='clip': TEACHER FORCING. The "
                  "generator conditions on the clip's nodes, not its own -- a diagnostic "
                  "for separating a weak model from a diverging closed loop. This is NOT "
                  "deployable; there is no clip on the robot.")
        elif self.cfg.history_from == "measured":
            print("[INFO FmTrackingCommand] history_from='measured': the generator "
                  f"conditions on the ROBOT'S OWN state over the last {self.ctx.N} steps, "
                  f"except {list(MEASURED_HISTORY_FALLBACK)} which no robot can measure and "
                  "which come from the served node. This is real but OFF-DISTRIBUTION -- "
                  "the model was fitted on reference rows, not achieved ones.")
            if self._seeding_enabled:
                print("[WARN FmTrackingCommand] seed_history_from_clip="
                      f"{self.cfg.seed_history_from_clip!r} with history_from='measured': "
                      "each episode OPENS on clip rows and then switches to achieved ones, "
                      "so the ring mixes two kinds of quantity for its first "
                      f"{self.ctx.N} steps. Deliberate for opening a '*_pure' clip in phase; "
                      "set 'never' for an arm that is measured throughout.")
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
        # Last step's MEASURED node, for history_from="measured". Stashed a step ahead for
        # the same reason the clip node is: the row pushed at step t must describe step t-1,
        # and ``robot.data`` at push time already holds the state after step t's physics.
        self._prev_measured_node: torch.Tensor | None = None
        self._measured_node_valid = torch.zeros(n, dtype=torch.bool, device=dev)

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
        #: Per-env copy of the exteroception row each plan was drawn from, or ``None``.
        #: Allocated only by :meth:`enable_extero_stash`, because it is one depth row per
        #: env and nothing in training would ever read it.
        self._extero_stash: torch.Tensor | None = None
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
        # A SECOND prototype per foot frame, in the contact colour, so a node the plan's own
        # kinematics call a stance is drawn differently from a swing node. The ribbon then
        # shows the flag the terrain-consistency reward gates on, and the two cannot drift:
        # both read `plan_foot_contact`, which is the single producer.
        self._fm_contact_specs: list[tuple[str, tuple[float, float, float], str, int]] = []
        self._fm_contact_proto: dict[int, int] = {}
        for i, (name, _c, kind, k) in enumerate(self._fm_pred_specs):
            if kind == "frame" and name in FOOT_BODIES:
                self._fm_contact_proto[i] = len(self._fm_pred_specs) + len(
                    self._fm_contact_specs)
                self._fm_contact_specs.append(
                    (f"{name}_contact", DEBUG_CONTACT_COLOR, kind, k))
        self._plan_collider_model = None
        self._plan_reward_collider_model = None
        # The ribbon draws EVERY predicted node (dense), independent of the clip
        # ribbon's debug_vis_stride subsampling. It indexes _pred_window directly
        # rather than through the cursor, so it needs no offset table.

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
        # The '@root' body links are FK of base+joints (_write_fm_root_body_frames), so in
        # body mode they ARE written -- by us rather than by the generator. Registering them
        # here is what stops strict_fm_only NaN-ing the channels the body CLF then grades.
        rf_maps = self._root_frame_maps()
        if getattr(self.cfg, "tracking_mode", "joint") == "body" and rf_maps is not None:
            pos_written += [rf_maps[0], rf_maps[1]]
        # The base's twist comes from base_vel; the ee twists are the zero placeholders,
        # written whenever the frames themselves are.
        vel_written: list[torch.Tensor] = []
        if "base_vel" in blocks:
            vel_written += [self._wf_lin_flat[:3], self._wf_ang_flat[:3]]
        if has_ee:
            vel_written += [self._wf_lin_flat[3:], self._wf_ang_flat[3:]]
        if "joint_vel" in blocks:
            vel_written.append(self._joint_slots_vel)
        if getattr(self.cfg, "tracking_mode", "joint") == "body" and rf_maps is not None:
            vel_written += [rf_maps[2], rf_maps[3]]

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
        out = super().reset(env_ids)
        if self.freeform_command:
            ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                   else env_ids)
            self._reset_freeform_command(ids)
        return out

    # ------------------------------------------------------------------
    # The free-form velocity command (cfg.freeform_command; off by default)
    #
    # The clip tasks read the command the library recorded, because the clip WAS generated
    # from it. The free-form terrain sections have no clips, so the command is manufactured
    # the way the deployed joystick supplies one: a per-episode forward speed, no lateral
    # channel, and a yaw rate that keeps the robot pointed down the section.
    #
    # Every accessor below is a view of ONE per-step buffer that :meth:`_update_freeform_command`
    # rewrites IN PLACE, matching how ``_apply_source`` keeps ``y_des`` a single stable tensor.
    # Clone before holding one across a step -- ``mdp.command_velocity`` already does.
    # ------------------------------------------------------------------
    #: Config fields that only mean something with a clip, as ``(field, forbidden value, why)``.
    #: Each one is checked at construction on a clip-free library rather than left to fail at
    #: the gather it eventually reaches: three of the four are read only at a RESET, so the
    #: failure would otherwise land after a simulator launch and a first episode, and one
    #: (``random_start_phase``) is read only by the clip reset event the free-form task
    #: disables -- so it would never fail at all, and would sit in ``params/env.yaml`` as a
    #: false statement about how the run started.
    #:
    #: **This list is NOT every clip-only field on the cfg, and must not be read as one.**
    #: Four more are likewise read only by the disabled clip reset and are deliberately left
    #: out, because the shipped free-form cfg inherits them at values a refusal would reject:
    #: ``ego_anchor_fraction`` (1.0 -- with no clip reset ``_ego_mask`` stays all-False, so
    #: the ego re-anchor is inert rather than wrong, and ``_reanchor_clip_ids`` returns on the
    #: empty id set before it can gather), ``random_start_before_terrain_end`` (True),
    #: ``random_start_min_remaining_s`` and ``randomize_heading`` -- the last three read only
    #: inside ``resets.sample_start_step`` / the clip placement. The line drawn is therefore
    #: "would refusing reject the shipped cfg", not "is this field about a clip": the four
    #: above are inert, and like ``random_start_phase`` they too sit in ``params/env.yaml``
    #: describing a start that never happens. Verified by walking the constructed cfg, not by
    #: reading the class.
    _CLIP_PHASE_KNOBS: tuple[tuple[str, object, str], ...] = (
        ("history_from", "generator",
         "the only other value is 'clip' -- teacher forcing, which feeds the generator the "
         "CLIP's window as history (_clip_history_node) and is a per-step clip gather"),
        ("seed_history_from_clip", "never",
         "seeding reads the clip's steps BEFORE the episode start; with no clip every episode "
         "opens on the learned null-history embedding, which is the deploy condition anyway"),
        ("random_start_phase", False,
         "there is no clip phase to start at"),
        ("random_start_before_terrain_start", False,
         "there is no clip whose terrain-pinned run a start could be placed before"),
    )

    def _refuse_clip_phase_knobs(self) -> None:
        """Refuse any clip-phase knob left on against a library with no clips.

        Every one of these is a claim about a motion that does not exist. Checked together and
        reported together, so a misconfigured run is fixed in one pass rather than one field per
        launch.

        Raises:
            ValueError: If any knob in :data:`_CLIP_PHASE_KNOBS` is not at its required value.
        """
        wrong = [(name, getattr(self.cfg, name, want), want, why)
                 for name, want, why in self._CLIP_PHASE_KNOBS
                 if getattr(self.cfg, name, want) != want]
        if not wrong:
            return
        lines = "\n".join(
            f"    commands.traj_ref.{name} = {got!r}  (must be {want!r}: {why})"
            for name, got, want, why in wrong
        )
        raise ValueError(
            "these clip-phase settings cannot be honoured against a CLIP-FREE library "
            f"({self.cfg.path}), which carries a terrain manifest and zero clips:\n"
            f"{lines}\n"
            "G1FinetuneFreeformEnvCfg sets all four; a run overriding one of them back on is "
            "asking for a clip that is not in the file."
        )

    @property
    def accepts_clip_free_library(self) -> bool:
        """A generator-driven run needs no clips: it needs the library's TERRAIN.

        Overrides :attr:`MotionSequenceCommand.accepts_clip_free_library`. True exactly when
        the generator IS the reference -- under ``reference_source="clip"`` this term serves
        the clip, so a library with none is as useless here as it is on the base class, and the
        pure-clip control run (``fm_checkpoint=""``) is that case.

        Note this permits the combination but does not make the whole TASK work on its own: a
        clip-free library also removes the spawn, the velocity command and the episode bound,
        and each of those has its own replacement wired in ``G1FinetuneFreeformEnvCfg``. What
        it does guarantee is that anything still reaching for a clip fails by name.
        """
        return getattr(self.cfg, "reference_source", None) == "fm"

    @property
    def freeform_command(self) -> bool:
        """Is the velocity command generated here rather than read from the clip library?

        Read through ``getattr`` so the bare instances the unit tests build (``object.__new__``
        with a ``SimpleNamespace`` cfg) answer False instead of raising -- the same idiom the
        ``tracking_mode`` reads in this file use.
        """
        return bool(getattr(self.cfg, "freeform_command", False))

    def _init_freeform_command(self) -> None:
        """Validate the free-form knobs and derive the opening hold in whole control steps.

        Called from :meth:`_post_init` only when the flag is set, so a clip task pays nothing
        and cannot be failed by a validation rule over fields it never touches.

        The hold rounds against ``env.step_dt``: at the shipped 0.5 s and a 0.02 s control step
        that is 25 held steps, the same count ``mjeval``'s ``time < hold_until`` produces.
        """
        lo = float(self.cfg.freeform_speed_min)
        hi = float(self.cfg.freeform_speed_max)
        if hi < lo:
            raise ValueError(
                f"freeform_speed_max ({hi}) is below freeform_speed_min ({lo}); the per-episode "
                "draw is uniform on [min, max] and an inverted band would silently give every "
                "episode the same speed."
            )
        if float(self.cfg.freeform_lookahead_m) <= 0.0:
            raise ValueError(
                f"freeform_lookahead_m must be > 0, got {self.cfg.freeform_lookahead_m}. "
                "See freeform_yaw_rate: at 0 the aim point is the robot's own projection."
            )
        if float(self.cfg.freeform_steer_kp) < 0.0:
            raise ValueError(
                f"freeform_steer_kp must be >= 0, got {self.cfg.freeform_steer_kp}. A negative "
                "gain steers AWAY from the centreline, which looks like a drifting policy."
            )
        if float(self.cfg.freeform_wz_max) < 0.0:
            raise ValueError(
                f"freeform_wz_max must be >= 0, got {self.cfg.freeform_wz_max}; it is a "
                "symmetric clip."
            )
        hold_s = float(self.cfg.freeform_cmd_hold_s)
        if hold_s < 0.0:
            raise ValueError(f"freeform_cmd_hold_s must be >= 0, got {hold_s}.")
        self._ff_hold_steps = int(round(hold_s / float(self.env.step_dt)))
        print(
            f"[INFO FmTrackingCommand] free-form velocity command: vx ~ U[{lo:.3f}, {hi:.3f}] "
            f"m/s held for the whole episode, vy = {FREEFORM_VY}, "
            f"wz = clip({self.cfg.freeform_steer_kp} * heading_err, "
            f"+/-{self.cfg.freeform_wz_max}) at a {self.cfg.freeform_lookahead_m} m look-ahead "
            f"(cross-track gain "
            f"{float(self.cfg.freeform_steer_kp) / float(self.cfg.freeform_lookahead_m):.3f} "
            f"1/s), after {self._ff_hold_steps} steps ({hold_s} s) of zero command. The "
            "library's target_vel is NOT read. STEERING ON A STAIRCASE IS OFF-DISTRIBUTION: "
            "the generator was commanded |wz| max = 0.0000 over all 3500 stair clips and is "
            "frozen during this fine-tune."
        )

    def _reset_freeform_command(self, env_ids: torch.Tensor) -> None:
        """Draw the episode's forward speed and resolve the section centreline.

        Reached from :meth:`reset`, which ``ManagerBasedRLEnv._reset_idx`` calls AFTER the
        event manager's reset terms -- so the robot has already been placed on its section by
        ``mdp.events.resets.reset_on_freeform_section``, which deliberately touches no command
        state and leaves exactly this to run here.

        The command is zeroed here as well as drawn: ``episode_length_buf`` is not cleared
        until every manager has been reset, so anything reading the command between now and
        the next :meth:`_update_command` would otherwise see the PREVIOUS episode's speed
        rather than the opening hold.

        Args:
            env_ids: ``[k]`` env indices being reset.
        """
        k = int(env_ids.numel())
        lo = float(self.cfg.freeform_speed_min)
        hi = float(self.cfg.freeform_speed_max)
        self._ff_speed[env_ids] = torch.empty(k, device=self.device).uniform_(lo, hi)
        origin_w, heading_w = self._default_freeform_section(env_ids)
        self._ff_origin_w[env_ids] = origin_w
        self._ff_heading_w[env_ids] = heading_w
        self._ff_cmd[env_ids] = 0.0

    def set_freeform_section(
        self, env_ids: torch.Tensor, origin_w: torch.Tensor, heading_w: torch.Tensor
    ) -> None:
        """Declare the straight centreline the steering controller aims down, per env.

        THE INTERFACE for the free-form terrain generator: a reset event that knows a
        section's own centreline calls this instead of leaving
        :meth:`_default_freeform_section` to guess one from the tile. Safe to call at any
        point after the env's reset; the controller reads these buffers fresh every step.

        Args:
            env_ids: ``[k]`` env indices.
            origin_w: ``[k, 2]`` ANY point on the centreline, world frame -- the controller
                projects onto the infinite line, so a section's start point and its midpoint
                give identical commands.
            heading_w: ``[k]`` centreline direction (rad), world frame, pointing the way the
                robot traverses. Sections run along their tile's +x, so this is the tile's
                ``frame_yaw`` lifted by the placement, and 0.0 on an axis-aligned manifest.

        Raises:
            RuntimeError: If the free-form command is off, where the buffers exist but nothing
                reads them -- a silent no-op is the wrong answer to a caller that meant it.
        """
        if not self.freeform_command:
            raise RuntimeError(
                "set_freeform_section() needs cfg.freeform_command=True; with the clip "
                "command in force nothing reads the centreline and this would be a no-op."
            )
        self._ff_origin_w[env_ids] = origin_w[..., :2].to(self._ff_origin_w.dtype)
        self._ff_heading_w[env_ids] = heading_w.to(self._ff_heading_w.dtype)

    def _default_freeform_section(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve each env's centreline from the tile's own manifest route.

        **The centreline is the ray the GENERATOR authored**: tile-local ``(start_x, start_y)``
        at heading ``start_yaw``, lifted into the world by the tile's ``corner`` and
        ``frame_yaw``. That is the definition ``motion_gen.terrain.freeform``'s own
        ``route_project`` / ``route_lookahead`` use, and those docstrings name this function as
        the consumer -- so the two must state one line or the terrain means something different
        to the thing that built it than to the thing that drives over it.

        ``start_yaw`` is NOT decoration. It is how the boxes family produces an angled approach
        to an axis-aligned box (``OverlayVolumeCfg`` has no yaw and the loader refuses a
        non-zero one, so the APPROACH is the only place an angle can live), and how the flat
        family gets a 3-18 deg walk instead of a straight one. Measured over 18 sections at the
        default config, an earlier version of this function that read
        ``(start_x, goal_y)`` at ``frame_yaw`` put the 5 m aim point up to 3.94 m (flat) /
        2.57 m (boxes) from the authored one -- i.e. it steered the angle straight back out,
        silently, and stairs (``start_yaw == 0``) was the one family where the bug was invisible.

        The spawn is jittered off this line on purpose
        (``mdp.events.resets.reset_on_freeform_section``'s ``pos_jitter_m``); the controller
        projects onto the infinite ray, so the jitter is an initial cross-track error to be
        driven out rather than a datum to be defended.

        ``reset_on_freeform_section`` deliberately leaves the command alone ("anything the
        clip-free command needs at reset belongs in that command's reset"), and it also never
        calls ``_commit_clip_reset``, so ``_mask_frame_yaw`` is NOT written on this path and
        the tile's heading has to come from the importer directly.

        The route lookup is imported HERE rather than at module scope: ``mdp.events.resets``
        already imports from this package (``world_lift_isaac``), and a module-scope import
        back the other way would make the pair order-dependent for the sake of one call per
        reset.

        Falls back to the env origin at heading 0 where the scene carries no route at all --
        a flat scene, a clip library, or a unit-test double. That is a placeholder, not a
        centreline: :meth:`set_freeform_section` is how a caller states a real one.

        Args:
            env_ids: ``[k]`` env indices being reset.

        Returns:
            ``(origin_w [k, 2], heading_w [k])``, world frame.
        """
        scene = getattr(self.env, "scene", None)
        importer = getattr(scene, "terrain", None) if scene is not None else None
        if not getattr(importer, "terrain_route", None):
            origin_w = (scene.env_origins[env_ids][:, :2] if scene is not None
                        else self._ff_origin_w[env_ids])
            return (origin_w.to(self._ff_origin_w.dtype),
                    torch.zeros_like(self._ff_heading_w[env_ids]))

        # THREE dots: this module sits at mdp.commands.traj_tracking, so ``..`` is
        # mdp.commands and ``..events`` does not exist. It was two, and because nothing
        # exercised the route branch (the tests reach the centreline through
        # set_freeform_section or through the no-route fallback) it would have raised
        # ModuleNotFoundError on the first reset of the first free-form run.
        from ...events.resets import freeform_route

        route = freeform_route(importer, env_ids)
        frame_yaw = route["frame_yaw"].to(self.device)
        local_x = route["start_x"].to(self.device)
        local_y = route["start_y"].to(self.device)
        cos_y, sin_y = torch.cos(frame_yaw), torch.sin(frame_yaw)
        corner = route["corner"].to(self.device)
        origin_w = torch.stack(
            [corner[:, 0] + cos_y * local_x - sin_y * local_y,
             corner[:, 1] + sin_y * local_x + cos_y * local_y], dim=-1)
        heading_w = frame_yaw + route["start_yaw"].to(self.device)
        return (origin_w.to(self._ff_origin_w.dtype),
                heading_w.to(self._ff_heading_w.dtype))

    def _update_freeform_command(self) -> None:
        """Rebuild ``(vx, 0, wz)`` for every env from its pose and its section centreline.

        Runs at the top of :meth:`_update_command`, before the clip half advances the step and
        before the generator reads its command horizon, so everything in one control step sees
        one command.

        During the opening hold the WHOLE triple is zero, not just ``vx``: the robot spawns at
        rest from a standing pose, and a yaw rate applied to a standing robot is as much a step
        input as a forward speed is. Matches ``mjeval``'s ``live_cmd = _ZERO_CMD``.
        """
        held = self.env.episode_length_buf < self._ff_hold_steps                # [N] bool
        root_xy = wp.to_torch(self.robot.data.root_pos_w)[:, :2]
        yaw = yaw_from_quat(wp.to_torch(self.robot.data.root_quat_w))
        wz = freeform_yaw_rate(
            root_xy, yaw, self._ff_origin_w, self._ff_heading_w,
            float(self.cfg.freeform_lookahead_m),
            float(self.cfg.freeform_steer_kp),
            float(self.cfg.freeform_wz_max),
        )
        zero = torch.zeros((), device=self._ff_cmd.device, dtype=self._ff_cmd.dtype)
        self._ff_cmd[:, 0] = torch.where(held, zero, self._ff_speed)
        self._ff_cmd[:, 1] = FREEFORM_VY
        self._ff_cmd[:, 2] = torch.where(held, zero, wz)

    # --- the four command-velocity accessors, overridden as one set -----------------
    #
    # All four are HELD: one command broadcast across every window offset and across the
    # generator's F-step horizon. The clip path instead serves target_vel sampled per offset,
    # which is a future no deployed runtime has -- both transfer/deploy/gen_tracker.py and the
    # C++ runtime hold the command from request time over the whole horizon. So this removes a
    # train/deploy mismatch, at the cost of the tracker's reference-group encoder seeing 13
    # copies of one row where it used to see 13 distinct ones.

    @property
    def command_velocity_available(self) -> bool:
        """True when a velocity command exists -- generated here, or carried by the library.

        Overriding the inherited property is what lets the free-form task run against a
        MANIFEST-ONLY ``.h5`` (terrain and nothing else): the command it needs is manufactured,
        so a library ``target_vel`` stops being a prerequisite. With the flag off this is
        exactly the inherited answer.
        """
        if self.freeform_command:
            return True
        return super().command_velocity_available

    def get_command_velocity(self) -> torch.Tensor:
        """Per-env commanded base velocity ``[N, 3]`` = ``(vx, 0, wz)`` at the current step."""
        if self.freeform_command:
            return self._ff_cmd
        return super().get_command_velocity()

    def get_command_velocity_horizon(self, num_future: int) -> torch.Tensor:
        """Commanded velocity over the next ``num_future`` steps, ``[N, num_future, 3]``.

        The generator's conditioning. Held: a broadcast VIEW of the current command, which is
        what the deployed runtimes supply ("the future command is not knowable; hold the one
        from request time").
        """
        if self.freeform_command:
            return self._ff_cmd[:, None, :].expand(-1, int(num_future), -1)
        return super().get_command_velocity_horizon(num_future)

    def get_future_command_velocity(self, offsets: torch.Tensor) -> torch.Tensor:
        """Commanded base velocity ``[N, K, 3]`` at the window ``offsets``.

        The policy's ``command_window`` observation. Held across the offsets, for the reason
        above -- so this returns ``len(offsets)`` copies of one row rather than a trajectory.
        """
        if self.freeform_command:
            return self._ff_cmd[:, None, :].expand(-1, int(offsets.numel()), -1)
        return super().get_future_command_velocity(offsets)

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

        The free-form command is rebuilt FIRST when it is in force, so the clip half, the
        generator's conditioning horizon and every observation in this step read one command.

        **The clip half is SKIPPED outright on a clip-free library**, and gated on
        :attr:`~.motion_sequence_cmd.MotionSequenceCommand.has_clips` rather than on
        ``freeform_command``: what makes the clip unreadable is the LIBRARY having none, not
        the command being manufactured, and the two are independent flags.
        """
        if self.freeform_command:
            self._update_freeform_command()
        if self.has_clips:
            self._build_clip_reference()
        else:
            self._advance_without_a_clip()
        if self.fm_enabled:
            # BEFORE the generator runs, and every step -- a frame can only be served `lag`
            # steps late if it was captured. Skipped entirely at lag 0, so the default costs
            # nothing.
            self._push_extero_lag()
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

    def _advance_without_a_clip(self) -> None:
        """The part of :meth:`_build_clip_reference` that is not about a clip, and only that.

        Runs in its place on a clip-free library. Exactly ONE of that method's five statements
        survives, and it survives because it is not clip machinery at all:
        :meth:`~.motion_sequence_cmd.MotionSequenceCommand._reanchor_ego_envs` is the step that
        latches ``_plan_anchor_pos`` / ``_plan_anchor_quat`` onto the robot's root on the snap
        cadence and publishes ``_reanchor_active``. It is the ONLY writer of the plan anchor on
        a running episode (the other is ``seed_reanchor_snapshot``, which the clip reset calls
        and which never runs here), so dropping it would silently leave that anchor at the
        origin for the whole run. Its own clip half -- ``_reanchor_clip_ids`` -- is masked by
        ``_ego_mask``, which nothing sets without a clip reset, so it is a no-op; and if a
        future config did set it, the manager's gather refuses by name rather than indexing an
        empty table.

        What is deliberately NOT run, and what each would have cost:

        * ``_advance_step`` -- it is what marches ``self._step``, and there is no clip for the
          step to index. Left un-run, ``_step`` stays at its reset value of 0 rather than
          walking to ``num_steps - 1``; with ``--like`` copying a real library's layout that
          bound is the other library's length, so a marching step would be a real number
          pointing at nothing.
        * ``_refresh_current_placement`` -- gathers the current frame's anchor mask per clip.
        * ``get_desired_outputs`` -- gathers ``y_des`` / ``dy_des`` per clip.
        * ``_write_world_des(self._clip_world_frames())`` -- lifts those through the placement.

        The three clip CACHES (``_y_des_clip``, ``_dy_des_clip``, ``_clip_wf_frames``) are
        therefore never written after ``_post_init`` and keep their zero-filled allocation.
        Nothing reads them: ``_apply_source`` only installs them for ``source="clip"``, which
        :meth:`_serve` now refuses here and which ``reference_source`` cannot be.
        """
        self._reanchor_ego_envs()

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
            # BEFORE _write_world_des, which is what lifts the '@root' bodies through the
            # anchored base (_lift_root_frames_to_world). Writing them after would leave
            # pelvis-relative values in channels the CLF reads as world -- finite, enormous,
            # and visible only as a reward pinned at 0.
            if getattr(self.cfg, "tracking_mode", "joint") == "body":
                self._write_fm_root_body_frames()
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
        if source == "clip" and not self.has_clips:
            # The caches this would install were allocated and never written (see
            # :meth:`_advance_without_a_clip`), so serving them would hand a diagnostic a
            # zero-filled reference that looks like a robot standing at the world origin. That
            # is precisely the silent answer this whole path is arranged to avoid.
            raise RuntimeError(
                "as_clip_reference() was asked of a CLIP-FREE library: it carries a terrain "
                "manifest and zero clips, so no clip reference is ever built and there is "
                "nothing to serve. The clip-vs-generator A/B (scripts/rsl_rl/fm_debug.py) "
                "needs a real clip library; on free-form terrain there is no clip to compare "
                "against by construction."
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

    @property
    def min_start_step(self) -> int:
        """Reserve the first ``N`` clip steps when the history is seeded from the clip.

        Under ``history_from="clip"`` a resetting env has its generator history written from
        the clip's OWN preceding steps (:meth:`_seed_clip_history`), which requires those
        steps to exist. The clip-step gather clamps only at the top, so a start inside the
        reserve would index backwards and quietly read the clip's END as history.

        Zero when nothing is seeded, which leaves the whole clip available to start on.
        """
        return self.ctx.N if self._seeding_enabled else 0

    @property
    def _seeding_enabled(self) -> bool:
        """Whether ANY env may have its history seeded from the clip this run.

        Resolves the ``"auto"`` default, under which seeding tracks teacher forcing.

        Raises:
            ValueError: On an unknown mode -- a typo must not silently disable the seed.
        """
        mode = self.cfg.seed_history_from_clip
        if mode not in ("auto", "never", "always", "on_terrain_start"):
            raise ValueError(
                f"seed_history_from_clip={mode!r} must be one of "
                "'auto', 'never', 'always', 'on_terrain_start'."
            )
        if mode == "never":
            return False
        if mode == "auto":
            return self.cfg.history_from == "clip"
        return True

    def _envs_needing_seed(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Narrow just-reset envs to the ones this run's mode says to seed.

        Under ``"on_terrain_start"`` that is the envs with no flat lead-in -- those whose
        start lands at or after the clip's ``terrain_start_step``. A flat clip carries
        ``terrain_start_step = -1`` (no terrain at all) and is never in this set, so it keeps
        the null embedding exactly as deploy does.

        Args:
            env_ids: Long indices of the envs that just reset.

        Returns:
            The subset to seed.
        """
        if self.cfg.seed_history_from_clip != "on_terrain_start":
            return env_ids
        t_start = self.manager.gather_terrain_start_step(self.clip_id[env_ids]).to(self.device)
        on_terrain = (t_start >= 0) & (self._step[env_ids] >= t_start)
        return env_ids[on_terrain]

    def _seed_clip_history(self, env_ids: torch.Tensor) -> None:
        """Fill the generator history of just-reset envs from the clip's preceding steps.

        Teacher forcing is supposed to hold the prediction in phase with the motion the
        reward grades. It cannot do that on the FIRST inference of an episode, because there
        is no history yet and the model falls back to the learned ``null_history`` embedding,
        which is phase-agnostic by construction. With ``random_start_phase`` the robot spawns
        at a definite gait phase, so that opening plan disagrees with both the robot and the
        reward for a whole re-plan period -- every episode.

        The rows are the clip's nodes for steps ``s0-N .. s0-1``, all expressed against the
        SPAWN pose. That is not an approximation of the recorded convention, it IS it: the
        recorder substitutes the plan anchor into history rows ``0..H-2``, and at a reset the
        plan anchor is the spawn pose, so a freshly re-planned history is exactly one anchor
        repeated across its rows.

        Closed-loop runs never reach this: on a robot there is no clip to read, so that arm
        keeps the null embedding and the two are deliberately allowed to differ here. Nor
        does ``history_from="measured"`` under the ``"auto"`` default -- and an explicit
        ``"always"`` / ``"on_terrain_start"`` there mixes CLIP rows into a ring that
        otherwise holds achieved ones, which ``_post_init`` warns about rather than refusing
        (it is the one way to open a ``*_pure`` stair clip in phase under that arm).

        Args:
            env_ids: Long indices of the envs that just reset.
        """
        if not self._seeding_enabled or env_ids.numel() == 0:
            return
        env_ids = self._envs_needing_seed(env_ids)
        if env_ids.numel() == 0:
            return
        n = self.ctx.N
        offsets = torch.arange(-n, 0, device=self.device, dtype=torch.long)
        # Seed only the envs that HAVE n preceding steps; the rest keep the null embedding.
        #
        # `min_start_step` reserves those steps whenever the start is drawn, but a start can
        # also be PINNED, and the eval harness pins every episode to clip step 0 on purpose
        # (reproducibility, and it is what makes cells pair on clip id). There is no history
        # before step 0 and nothing to invent, so those envs fall through to null -- which is
        # the honest answer for them and leaves the harness's existing numbers comparable.
        #
        # Filtering rather than raising: the clip-step gather clamps only at the TOP, so a
        # negative index would silently read the clip's END as its history. That is what must
        # not happen; a pinned start is not itself an error.
        seedable = self._step[env_ids] >= n
        if not bool(seedable.any()):
            return
        env_ids = env_ids[seedable]
        prev = self._serve_override
        self._serve_override = "clip"
        try:
            rows = super().reference_window_obs(offsets, self._fm_blocks)  # [N_env, n, D]
        finally:
            self._serve_override = prev
        rows = rows[env_ids]
        # robot-now -> plan anchor, exactly as _clip_history_node does it for the live path.
        # `reference_window_obs` anchors on the MEASURED robot, and the history ring holds
        # plan-anchored rows; at a reset the two normally coincide, but deriving the
        # transform instead of assuming it means this stays correct if the reset ordering
        # ever puts the anchor latch after the seed.
        src_p = wp.to_torch(self.robot.data.root_pos_w)[env_ids][:, None, :]
        src_q = wp.to_torch(self.robot.data.root_quat_w)[env_ids][:, None, :]
        rep = lambda t, d: t.expand(-1, n, -1).reshape(-1, 1, d)  # noqa: E731
        seeded = reanchor_window(
            rows.reshape(-1, rows.shape[-1])[:, None, :],
            rep(src_p, 3), rep(src_q, 4),
            rep(self._anchor_pos[env_ids][:, None, :], 3),
            rep(self._anchor_quat[env_ids][:, None, :], 4),
            self.ctx.nj, self.ctx.num_extra_frames, self._fm_blocks,
        )[:, 0].reshape(rows.shape)
        self._history.seed(env_ids, {"node": seeded})

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

    def _measured_history_node(self) -> torch.Tensor:
        """The ROBOT'S OWN state as a window node, in the plan's frame.

        The third history producer, beside the served node and the clip node, and built the
        same way all three are: :func:`world_to_window` with the plan anchor. What differs is
        only what is handed in as the "reference" -- here the robot's measured base pose,
        base velocity, joints and end-effector positions.

        The anchor is ``_anchor_pos`` / ``_anchor_quat``, the pose at the last re-plan, NOT
        the robot now. That is the frame every history row lives in (the recorder substitutes
        it into rows ``0..H-2``), and it is what the two other producers use. Under it the
        ``base_pose`` position half is the robot's displacement SINCE the re-plan -- at most
        ``fm_update_every`` steps of travel -- which is why this arm needs only short-horizon
        relative odometry on a robot rather than a global base position. Note that half is
        overwritten by the caller anyway, since this robot cannot supply it at all; it is
        computed here so the node is complete and the substitution is one explicit step.

        Joint order needs no permutation: the window's joint block is defined in the robot's
        order and ``_post_init`` refuses a checkpoint whose order differs.

        Returns:
            ``[N, D]`` window nodes in the checkpoint's own block layout.
        """
        d = self.robot.data
        root_pos = wp.to_torch(d.root_pos_w)[:, None, :]                 # [N, 1, 3]
        root_quat = wp.to_torch(d.root_quat_w)[:, None, :]               # [N, 1, 4]
        if self._measured_ee_idx is not None:
            ee_w = wp.to_torch(d.body_pos_w)[:, self._measured_ee_idx, :][:, None, :, :]
        else:
            ee_w = root_pos.new_zeros(root_pos.shape[0], 1, 0, 3)
        return world_to_window(
            self._anchor_pos[:, None, :], self._anchor_quat[:, None, :],
            root_pos, root_quat,
            wp.to_torch(d.root_link_lin_vel_w)[:, None, :],
            wp.to_torch(d.root_link_ang_vel_w)[:, None, :],
            wp.to_torch(d.joint_pos)[:, None, :], wp.to_torch(d.joint_vel)[:, None, :],
            ee_w, blocks=self._fm_blocks,
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

        Under ``history_from="measured"`` it is the ROBOT'S OWN state at the previous step
        (:meth:`_measured_history_node`), with the columns no robot can measure
        (:data:`MEASURED_HISTORY_FALLBACK`) taken from the node served at that same step.

        Whichever it is, the row is from the PREVIOUS step, which is the convention the
        dataset uses (the history block is the N rows BEFORE node 0). For the clip and the
        measured arms that means pushing the value stashed last step, not the one available
        now -- ``_advance_step`` has already moved the clip on to the current step by the
        time this runs, and ``robot.data`` already holds the state after this step's physics.

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
        elif self.cfg.history_from == "measured":
            stash = self._prev_measured_node
            served = self._prev_served_node
            # Both stashes describe step t-1 at this instant -- the measured one was taken
            # at the end of that step and ``_prev_served_node`` is set at the end of
            # ``_snapshot_and_infer``, so the substitution below pairs the SAME step. Doing
            # it at stash time instead would pair a measured row for step t with a served
            # node for step t-1, off by one control step of travel.
            valid = self._measured_node_valid & self._pred_valid
            if stash is None or served is None:
                # Nothing real to push: either the robot has no previous step or the
                # generator has served nothing, and 3 of the row's columns can only come
                # from the latter. A row that is part measurement and part zero is not a
                # row the model has ever seen, so the whole thing goes in as zeros and the
                # null-history embedding covers it -- exactly as the other two arms do.
                node = torch.zeros_like(self._pred_window[:, 0])
            else:
                node = torch.where(valid[:, None], stash, torch.zeros_like(stash))
                for a, b in self._measured_fallback_spans:
                    node[:, a:b] = torch.where(valid[:, None], served[:, a:b], node[:, a:b])
            self._prev_measured_node = self._measured_history_node()
            self._measured_node_valid = torch.ones_like(self._pred_valid)
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
        * the SENSORS come from the env, and for DEPTH that is no longer an opt-in: the
          policy reads the NEAR camera alone, so ``fm_depth_groups`` returning ``None``
          = "every group" would serve the generator one camera row where its checkpoint
          states two. Every path that runs or records the generator therefore installs
          ``g1_depth_env_cfg.install_fm_depth_group`` unconditionally, and a depth task
          that reaches the ``None`` arm is a FAULT rather than an unchanged task --
          :meth:`_check_extero_shape` is what names it. The SCAN is unaffected:
          ``fm_scan_sensor`` still falls back to ``height_scanner``, because there is one
          scanner and the policy and the generator read the same one;
        * the CHANNELS come from the checkpoint too (``ctx.extero_channels``): a generator
          trained on depth alone gets the depth image, one trained on 4 channels gets it
          with each pixel's xyz in the root's yaw frame beside it. Same reasoning as the
          source -- the artifact says what it eats;
        * the WIDTH is checked on the FIRST reading (:meth:`_check_extero_shape`), not
          per step and not at construction -- see ``_post_init``.

        Returns:
            ``(num_envs, T, K)`` -- :func:`build_extero_tokens` flattens it.

        Raises:
            RuntimeError: If the scene lacks the sensor the checkpoint needs.
        """
        src = self.ctx.extero_source
        if src == "height_scan":
            reading = scan_obs_stack(self.env, fm_scan_sensor(self.env))
        elif src == "depth":
            groups = fm_depth_groups(self.env)
            n_ch = int(getattr(self.ctx, "extero_channels", 1))
            reading = (depth_xyz_stack(self.env, groups, channels=n_ch) if n_ch > 1
                       else depth_obs_stack(self.env, groups))
        else:
            raise RuntimeError(f"unknown extero_source {src!r} on the FM checkpoint.")
        if not self._extero_checked:
            self._check_extero_shape(reading)
            self._extero_checked = True
        return reading

    def _push_extero_lag(self) -> None:
        """Record this step's exteroception so a later step can be served an older one.

        Buffers the FUSED reading (:meth:`_read_extero`), not the raw image: for a multi-
        channel row the xyz planes are already unprojected with the pose that was live when
        it was taken, so image and pose cannot come apart. Unprojecting an old image with the
        CURRENT pose would displace every pixel by the egomotion -- the registration error
        those channels exist to remove.
        """
        if not self._depth_lag_steps:
            return
        reading = self._read_extero().clone()
        # An env that just reset has nothing to be LATE FROM: every frame still in the ring
        # was taken during its previous episode, before the robot was teleported -- in the
        # eval, onto a different tile entirely. Serving one conditions the generator on
        # terrain that is not in front of it.
        #
        # This is not a small window. The reset also clears ``_pred_valid``, which forces an
        # immediate re-plan, so the FIRST plan of every episode was built from the stale
        # frame and then walked for ``fm_update_every`` steps before anything re-planned.
        # The families that start ON terrain are the ones with no room to absorb that.
        #
        # Refilling with the CURRENT reading (rather than dropping the ring) reproduces, per
        # env, exactly the semantics :meth:`_lagged_extero` documents for the start of a run:
        # the oldest held frame is repeated until real history exists. ``_pending_reset`` is
        # still set here -- ``_snapshot_and_infer`` consumes it later in the same step.
        if self._extero_buf and bool(self._pending_reset.any()):
            for frame in self._extero_buf:
                frame[self._pending_reset] = reading[self._pending_reset]
        self._extero_buf.append(reading)
        del self._extero_buf[:-(self._depth_lag_steps + 1)]

    def _lagged_extero(self) -> torch.Tensor:
        """The reading the generator conditions on: ``_depth_lag_steps`` old, or live.

        Early in an episode the buffer is short and the OLDEST held frame is repeated. Not
        zero-filled: a zero depth row is a valid reading -- a surface at the near plane --
        so zeros would condition the generator on a wall that is not there.

        Returns:
            ``(num_envs, T, K)``, as :meth:`_read_extero`.
        """
        if not self._depth_lag_steps or not self._extero_buf:
            return self._read_extero()
        return self._extero_buf[max(len(self._extero_buf) - 1 - self._depth_lag_steps, 0)]

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
            if self.ctx.extero_source == "height_scan":
                served = fm_scan_sensor(self.env)
                fix = (f"this env reads scanner {served!r}. Build the generator's own dense "
                       "scanner with g1_fm_scan_env_cfg.add_fm_height_scan(env_cfg, size=..., "
                       "resolution=...) at the geometry the checkpoint states "
                       "(robot_rl.flow_matching.online.fm_scan_geometry reads it), which is "
                       "what play.py does from --fm_checkpoint.")
            elif fm_depth_groups(self.env) is None:
                # The common cause, and the one this env cannot distinguish from a
                # deliberate one-camera checkpoint: no generator group at all. The policy
                # reads the NEAR camera alone, so the fallback to "every group" serves ONE
                # row against a checkpoint that states two.
                fix = (f"this env has no {FM_DEPTH_OBS_GROUP!r} obs group, so every group was "
                       "searched and the generator was served the POLICY's cameras -- and the "
                       "policy reads the near camera ALONE. Install the generator's own group "
                       "with g1_depth_env_cfg.install_fm_depth_group(env_cfg) before gym.make "
                       "(record_data.py / play.py / augment_registration.py do); it covers "
                       "every camera on the scene, and at the policy's own resolution it "
                       "re-observes them rather than casting a second pair. If the resolutions "
                       "differ, pass it: install_fm_depth_group(env_cfg, W, H) at what the "
                       "checkpoint states (read_fm_extero_meta reads it).")
            else:
                fix = (f"obs group(s): {fm_depth_groups(self.env)} -- the generator's own group "
                       "IS installed, so this is a per-camera RESOLUTION mismatch (or a "
                       "checkpoint trained on a different number of cameras). Re-install it at "
                       "the resolution the checkpoint states with "
                       "g1_depth_env_cfg.install_fm_depth_group(env_cfg, W, H) "
                       "(read_fm_extero_meta reads it), which is what play.py does from "
                       "--fm_checkpoint.")
            raise RuntimeError(
                f"the FM checkpoint conditions on {want[0]} sensor row(s) of {want[1]} "
                f"values, but this env serves {count} of {dim}. {fix}"
            )
        # A width match is NOT a geometry match: two scans of the same ray count can differ in
        # extent or resolution, which is invisible to every shape assertion and would hand the
        # convolution a correctly-sized image of the wrong ground. Compare the spec when the
        # env can produce one.
        spec = self.ctx.extero_spec
        if spec and self.ctx.extero_source == "height_scan":
            served = fm_scan_sensor(self.env)
            live = height_scan_specs(self.env, served) if served else []
            if live:
                keys = ("resolution", "size", "ordering", "height", "width", "offset_pos",
                        "ray_alignment", "sensor_offset")
                want_geom = {k: spec[0].get(k) for k in keys if k in spec[0]}
                got_geom = {k: live[0].get(k) for k in keys if k in spec[0]}
                if want_geom != got_geom:
                    raise RuntimeError(
                        f"scanner {served!r} has the right ray count but different geometry "
                        f"from the one this checkpoint trained on:\n  checkpoint: {want_geom}\n"
                        f"  env       : {got_geom}\n"
                        "Same width, different ground -- the image would be built from the "
                        "wrong geometry with every tensor the right size."
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
            # Same for the measured stash, and for a sharper reason: it is the robot's pose
            # on the PREVIOUS tile, so pushing it would hand the generator a metres-wide
            # step in base_pose that reads as a fall rather than as a reset.
            self._measured_node_valid = self._measured_node_valid & ~just_reset

        self._history.invalidate(just_reset)
        self._push_served_node()
        # AFTER the push: it appends a row (zeros for a reset env, whose stash was just
        # cleared), so seeding first would have that row shift the oldest seeded one out.
        # Seeding last overwrites the whole ring for those envs and clears their mask.
        if bool(just_reset.any()):
            self._seed_clip_history(torch.nonzero(just_reset, as_tuple=False).squeeze(-1))

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
        extero_row = self._lagged_extero()[ids]
        if self._extero_stash is not None:
            # The row the generator ACTUALLY conditioned on, kept per env for diagnostics.
            # Re-reading the sensor afterwards is not the same thing: this is the LAGGED
            # frame, and by the time anything downstream looks, the live one has moved on.
            # Opt-in (see enable_extero_stash) so training allocates nothing.
            self._extero_stash[ids] = extero_row.detach()
        height = build_extero_tokens(self.ctx, extero_row)
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        cmd_tokens = build_cmd_tokens(self.ctx, fut_cmd)
        self._pred_window[ids] = sample_window(
            self.ctx, history, height, cmd_tokens,
            history_null_mask=(None if null_hist is None else null_hist[ids]),
            noise=(None if self._noise_source is None else self._noise_source(ids)),
        )
        # Node 0 = "now", and the frame it is expressed in is the robot AT THIS INSTANT.
        # Both are needed to serve the window later: the cursor to index it, the anchor to
        # undo the staleness once the robot has moved on.
        self._pred_idx[ids] = 0
        self._anchor_pos[ids] = wp.to_torch(self.robot.data.root_pos_w)[ids].clone()
        self._anchor_quat[ids] = wp.to_torch(self.robot.data.root_quat_w)[ids].clone()
        self._pred_valid[ids] = True

    def set_noise_source(self, source) -> None:
        """Take the flow's initial noise from ``source`` instead of the global RNG.

        Off by default, and there is no config field for it: training and every normal
        rollout want an independent draw per re-plan, which is what ``sample_trajectory``
        does when handed no ``noise``.

        What this exists for is a controlled COMPARISON of two samples at one state. The
        flow's only stochasticity is ``x0``, so fixing it per env makes the whole plan a
        deterministic function of the conditioning -- which is what lets the best-of-K
        headroom harness (``scripts/rsl_rl/eval/headroom*``) hand K envs K different chunks
        at an identical state, and then hand all of them the SAME draw on every subsequent
        re-plan so the ranking cannot be dominated by which branch got lucky afterwards.

        This is NOT ``fm_zero_noise``, which replaces the draw with zeros and is off the
        training distribution. Every draw served here is meant to be an ordinary Gaussian
        sample; only WHICH one is under external control.

        Args:
            source: Callable taking the ``[V]`` long tensor of env ids about to be sampled
                and returning either ``None`` (fall back to the global RNG for this call) or
                a ``[V, F, D]`` tensor in the same env order.  ``None`` uninstalls.
        """
        self._noise_source = source

    def enable_extero_stash(self) -> None:
        """Start keeping the exteroception row each plan was drawn from.

        A diagnostic hook, off by default: it holds one depth row per env, which is the
        widest array in the observation, and only the evaluation harness's flagged-plan dump
        ever reads it. Calling it twice is harmless.

        What it buys is the ability to answer "did the image look like a box, or did the
        image look fine and the plan go wrong" -- the one split a plan-space classifier
        cannot make on its own, because by the time a plan is classified the frame that
        produced it is two lags stale and no longer recoverable from the sensor.
        """
        if self._extero_stash is None:
            row = self._lagged_extero()
            self._extero_stash = torch.zeros_like(row)

    def window_anchor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The FM command's plan anchor is the pose latched when the plan was drawn.

        Overrides :meth:`MotionSequenceCommand.window_anchor`, whose plan anchor advances on
        the ``fm_snap_every`` cadence. Here the authority is ``_anchor_pos`` / ``_anchor_quat``
        -- the pose the CURRENT prediction was made against -- so the observation frame and
        the prediction frame cannot drift apart even if the two cadences are configured
        differently.

        **Only when a generator is loaded.** In the pure-clip control
        (``reference_source='clip'`` with no ``fm_checkpoint``) ``_post_init`` returns before
        allocating those buffers, because there is no prediction to anchor -- so reaching for
        them raised ``AttributeError: no attribute '_anchor_pos'``, and a ``window_anchor='plan'``
        checkpoint could not run clip control on this task at all. With no prediction there is
        no prediction frame to stay aligned with, so the base class's plan anchor is the
        correct authority.

        Returns:
            ``([N, 3], [N, 4])`` world position and quaternion (xyzw).
        """
        if (getattr(self.cfg, "window_anchor", DEFAULT_WINDOW_ANCHOR) == "plan"
                and self.fm_enabled):
            return self._anchor_pos, self._anchor_quat
        return super().window_anchor()

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

    @property
    def _body_fk(self):
        """Batched FK over the 13 tracked ``@root`` body frames, built on first use.

        The body CLF grades ``<body>@root:*`` -- each tracked link's pose and twist relative
        to the base -- and the generator does not predict them. It does not need to: those
        poses are pure forward kinematics of the base and joint angles it DOES predict, so
        deriving them adds no information, it only moves a computation that the clip did
        offline into the loop.

        ``FKChain`` reports base-frame poses, which is exactly what ``@root`` means, so
        nothing has to be composed or rotated on the way in. It is built lazily because the
        parse is CPU work and only body mode needs it.
        """
        if getattr(self, "_body_fk_chain", None) is None:
            from motion_gen.features import DEFAULT_MJCF  # noqa: PLC0415
            from motion_gen.fk import FKChain  # noqa: PLC0415
            from motion_gen.joint_names import (  # noqa: PLC0415
                G1_BODY_TRACK_FRAMES, strip_root_marker,
            )
            frames = [strip_root_marker(f) for f in (self.root_body_frames or ())]
            if not frames:
                frames = list(G1_BODY_TRACK_FRAMES)
            self._body_fk_chain = FKChain(
                DEFAULT_MJCF, list(self.ctx.joint_names), frames,
                base_link=self.cfg.base_frame_name,
            ).to(self.device)
        return self._body_fk_chain

    def _write_fm_root_body_frames(self) -> None:
        """Fill the ``@root`` body channels by FK from the generator's own prediction.

        Pose comes straight from :attr:`_body_fk`. The TWIST needs joint velocities, and this
        generator predicts only ``joint_pos`` -- so they are finite-differenced across the
        predicted window, ``qdot ~ (q[1] - q[0]) / dt``, and pushed through the same FK
        Jacobian. That is the same standing the poses have: a quantity the generator implies
        rather than one it states. It inherits the difference's noise, which is the honest
        cost of not having ``joint_vel`` in the block selection.

        Writes BASE-FRAME values into the ``@root`` slots. The lift to world is done for us
        by :meth:`~.motion_sequence_cmd.MotionSequenceCommand._write_world_des`, which calls
        ``_lift_root_frames_to_world`` through the anchored base -- so this MUST run before
        it. Called after, the CLF reads pelvis-relative poses as world ones: finite,
        enormous, and visible only as ``clf_reward`` pinned at 0.
        """
        maps = self._root_frame_maps()
        if maps is None:
            return
        pos_idx, ori_idx, vlin_idx, vang_idx = maps
        w = self._pred_world(self._first_two_offsets)
        q = w["joint_pos"]                                  # [N, 2, nj] in FM joint order
        # pose_and_jacobian returns a ROTATION MATRIX, not a quaternion.
        #
        # The component order is WXYZ, and that is measured rather than reasoned: the '@root'
        # ori slots are gathered by the names ori_x/y/z/w, which reads as xyzw, but FK'ing a
        # clip's own joint angles and comparing against its stored channels matches to 0.0
        # under wxyz and to 1.13 under xyzw. Getting this wrong is not a crash -- it is a
        # plausible wrong attitude, worth ~100 in v_root_body_ori while the positions stay
        # exact, which is precisely how it presented.
        pose_pos, pose_rot, jac_lin, jac_ang = self._body_fk.pose_and_jacobian(q[:, 0])
        n, nb = pose_pos.shape[0], pose_pos.shape[1]
        pose_quat = quat_from_matrix(pose_rot.reshape(-1, 3, 3)).reshape(n, nb, 4)

        # Jacobian x joint velocity reproduces the clip's own stored '@root' twists exactly
        # (validated offline to 0.0), so the convention is right; the finite difference is
        # the only approximation, and it is there because this checkpoint predicts no
        # joint_vel.
        qdot = (q[:, 1] - q[:, 0]) / self.manager.dt        # [N, nj]
        lin = torch.einsum("nbij,nj->nbi", jac_lin, qdot)
        ang = torch.einsum("nbij,nj->nbi", jac_ang, qdot)

        self.y_des[:, pos_idx] = pose_pos.reshape(n, -1)
        self.y_des[:, ori_idx] = pose_quat.reshape(n, -1)
        self.dy_des[:, vlin_idx] = lin.reshape(n, -1)
        self.dy_des[:, vang_idx] = ang.reshape(n, -1)

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
        if getattr(self.cfg, "window_anchor", DEFAULT_WINDOW_ANCHOR) == "plan":
            # The prediction is ALREADY in the plan frame -- that is the frame the generator
            # was trained to emit -- and under this convention the observation is that frame.
            # So there is nothing to undo: serve the nodes as predicted. This is the whole
            # odometry saving, and its deploy twin is GenTrackerPolicy._serve_nodes.
            ar, idx = self._future_indices(offsets)
            # `_pred_window`, NOT `_pred`. The deploy twin (GenTrackerPolicy._serve_nodes)
            # calls its buffer `_pred`, and this line was written from it -- but on this class
            # no such attribute exists, so the plan arm raised AttributeError the first time
            # the observation manager sized the term. It went unnoticed because the anchor A/B
            # ran on the TEACHER (MotionSequenceCommand), which does not take this branch.
            return self._pred_window[ar, idx]
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
        translations, marker_indices = [], []
        contact = self.plan_foot_contact()
        for proto_idx, (name, _color, kind, k) in enumerate(self._fm_pred_specs):
            pts = (world["base_pos_w"] if kind == "base"
                   else world["ee_pos_w"][:, :, k]).reshape(-1, 3)
            translations.append(pts)
            idx = torch.full((pts.shape[0],), proto_idx, dtype=torch.long, device=self.device)
            alt = self._fm_contact_proto.get(proto_idx)
            if alt is not None and contact is not None:
                # `contact["contact"]` is [N, F, 2] in FOOT_BODIES order, and the ribbon's
                # points are node-major within each env, matching the reshape above.
                j = FOOT_BODIES.index(name)
                idx = torch.where(contact["contact"][:, :, j].reshape(-1),
                                  torch.full_like(idx, alt), idx)
            marker_indices.append(idx)
        self._fm_markers.set_visibility(True)
        self._fm_markers.visualize(
            translations=torch.cat(translations, dim=0), marker_indices=torch.cat(marker_indices, dim=0),
        )

    def plan_collider_model(self):
        """The feet-only collider model the plan's contact flags are derived from.

        Built lazily and cached: parsing the MJCF and resolving the FK chain is the whole
        setup cost, and this is read once per debug draw (and, once the terrain-consistency
        reward is wired, once per re-plan).

        FEET ONLY, and surface-sampled. The contact flag reads the sole, so nothing else is
        needed here -- and keeping it to the feet is what makes the flag cheap enough to
        compute for a debug ribbon at full env count.

        Returns:
            A :class:`~motion_gen.plan_geometry.PlanColliderModel`, or ``None`` when the
            window predicts no joints (there is then no posture to run FK on).
        """
        if self._plan_collider_model is not None:
            return self._plan_collider_model
        if "joint_pos" not in self._fm_blocks:
            return None
        try:
            from motion_gen.features import DEFAULT_MJCF                    # noqa: PLC0415
            from motion_gen.plan_geometry import PlanColliderModel          # noqa: PLC0415
            self._plan_collider_model = PlanColliderModel(
                DEFAULT_MJCF, list(self.ctx.joint_names), body_names=list(FOOT_BODIES),
                base_link=self.cfg.base_frame_name, dtype=torch.float32,
            ).to(self.device)
        except Exception as exc:
            print(f"[WARN FmTrackingCommand] could not build the plan collider model "
                  f"({exc}); the ribbon will not mark contacts.")
            return None
        return self._plan_collider_model

    def plan_reward_collider_model(self):
        """The WHOLE-BODY collider model the terrain-consistency REWARD is graded on.

        Not :meth:`plan_collider_model`, and the difference is not an optimisation.  That one
        is feet-only because the contact flag reads the sole and a debug ribbon has to be cheap
        at full env count.  The PENETRATION term sums over every collider point with a per-body
        weight -- feet 1.0, everything else 10.0 -- so grading it on a feet-only model silently
        charges nothing for a knee, a torso or a hand inside the terrain.

        That was measured, not reasoned: against the offline grader over 3933 real re-plans the
        feet-only reward agreed on `contact_penalty` to 1e-6 (it reads soles either way) and
        under-read `penetration_sum_m` by 13-17x on exactly the windows where the robot had
        fallen -- the windows a terrain reward exists to charge.  The offline grader builds the
        full model, and the critic was fitted on ITS numbers, so this is the model that makes
        the live reward the same quantity.

        Built lazily and cached separately, so a run that only draws the ribbon never pays for
        it.

        Returns:
            A :class:`~motion_gen.plan_geometry.PlanColliderModel` over every body carrying
            collision geometry, or ``None`` when the window predicts no joints.
        """
        if self._plan_reward_collider_model is not None:
            return self._plan_reward_collider_model
        if "joint_pos" not in self._fm_blocks:
            return None
        try:
            from motion_gen.features import DEFAULT_MJCF                    # noqa: PLC0415
            from motion_gen.plan_geometry import PlanColliderModel          # noqa: PLC0415
            # `body_names=None` = every body with collision geometry, matching
            # `inspect_terrain_reward._plan_model`. Refused rather than defaulted if it
            # fails: a feet-only fallback here is precisely the bug this method exists for.
            self._plan_reward_collider_model = PlanColliderModel(
                DEFAULT_MJCF, list(self.ctx.joint_names), body_names=None,
                base_link=self.cfg.base_frame_name, dtype=torch.float32,
            ).to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"could not build the whole-body collider model for the terrain reward "
                f"({exc}). Falling back to the feet-only model would under-read the "
                "penetration term by 13-17x on fallen plans while reporting the same "
                "column name.") from exc
        return self._plan_reward_collider_model

    def plan_foot_contact(self):
        """Contact flags for the CURRENT plan, from sole-corner speed alone.

        The single producer shared by the debug ribbon and the terrain-consistency reward, so
        the flag drawn is the flag graded. See
        :func:`~motion_gen.plan_geometry.plan_contact` for why the criterion is velocity and
        not proximity: gating a terrain reward on nearness to the terrain would be a
        tautology.

        Returns:
            The :func:`~motion_gen.plan_geometry.plan_contact` mapping over ``[N, F, 2]``, or
            ``None`` when no collider model could be built.
        """
        model = self.plan_collider_model()
        if model is None:
            return None
        from motion_gen.plan_geometry import (                              # noqa: PLC0415
            plan_collider_points, plan_contact)
        base_pos, rot, joint_pos = self.plan_world_pose()
        pts = plan_collider_points(base_pos, rot, joint_pos, model)
        # The plan's nodes are contiguous CONTROL steps, so the node spacing IS the env's
        # control period -- read off the env rather than restated, since a decimation change
        # would otherwise leave a stale dt scaling every speed the contact test thresholds.
        return plan_contact(pts, model, dt=float(self._env.step_dt))

    def plan_world_pose(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The current plan lifted into the world: base position, base rotation, joints.

        One producer for the lift, shared by the contact flags and by the terrain-consistency
        rewards, so the two can never place the same plan differently. The anchor frame is
        YAW-ONLY -- ``window_to_world`` drops the anchor's roll and pitch and keeps the
        reference's -- and restating that at a call site is precisely the frame detail that
        goes wrong silently.

        Returns:
            ``(base_pos_w (N, F, 3), base_rot_w (N, F, 3, 3), joint_pos (N, F, nj))``. The
            rotation as a MATRIX because that is what the collider FK consumes.
        """
        world = window_to_world(self._pred_window, self._anchor_pos[:, None, :],
                                self._anchor_quat[:, None, :], self.ctx.nj,
                                self.ctx.num_extra_frames, self._fm_blocks)
        n, f = world["base_pos_w"].shape[:2]
        rot = matrix_from_quat(world["base_quat_w"].reshape(n * f, 4)).reshape(n, f, 3, 3)
        return world["base_pos_w"], rot, world["joint_pos"]

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
                radius=radius * (1.35 if name.endswith("_contact") else 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
            for name, color, _kind, _k in self._fm_pred_specs + self._fm_contact_specs
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/fm_pred", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN FmTrackingCommand] could not create FM markers: {exc}")
            return None
