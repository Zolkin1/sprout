# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-step tracking error inside the MuJoCo loop: the latch, the reward, and the trace.

``transfer.deploy.tracking_error`` defines WHAT the error is; this is the thing that reads it
once per control step against the reference the tracker was actually served, and keeps the three
running quantities every consumer of this stack needs:

* ``first_violation_step`` -- the LATCH.  It does not end the episode; the verdict is decided by
  first-event ordering against the goal, so one run yields both the strict and the loose number
  and a threshold change is an offline relabel rather than a re-collection.
* ``chunk_rewards`` -- the dense per-chunk tracking reward the critic's quality head regresses,
  accumulated at the re-plan boundary so it is defined at the rate decisions happen.
* ``trace`` -- optionally every row, which is what makes the thresholds fittable at all.  Off by
  default: it is ``(steps, 40)`` float32 per episode and a sweep runs tens of thousands.

**Two loops drive this and neither owns the other.**  ``mjeval.runner.run_episode`` folds it
through :class:`~mjeval.accumulate.EpisodeAccumulator`; ``live_select._chunk`` runs its own loop
with snapshot/restore and folds it directly.  Hence a small standalone object rather than more
state on the accumulator -- and hence :meth:`reset`, because a restored branch has to rewind the
latch along with everything else or a rejected candidate's stumble would follow the episode.

**The reference is the SERVED node, and the pairing is exact where it matters.**
``run_episode`` calls ``create_obs`` and then ``robot.failed()`` before any physics advance, so
at :meth:`fold` -- and at the terminal fold after a FALL -- the reference and the robot state
describe the same instant.  After a success or a timeout the terminal frame sits one control
step past the last serve; that is the same one-step skew ``_fold_extremes`` already carries for
``first_fall_step`` and it is not corrected here, because inventing a reference for a step the
policy never planned would be worse than a known one-step lag on the last frame of an episode
that did not fail.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_HERE)
# This directory, `transfer/sim`, and `transfer` -- the same bootstrap `runner` uses, so the
# deploy runtime is reached by the tree's one convention rather than a second one.
for _d in (_HERE, _SIM_DIR, os.path.dirname(_SIM_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from deploy.tracker_session import convert_joint_order                   # noqa: E402
from deploy.tracking_error import (                                      # noqa: E402
    BOOTSTRAP_SPEC,
    NUM_FIXED_CHANNELS,
    VIOLATION_CHECK_NAMES,
    TrackingErrorSpec,
    chunk_reward,
    step_reward,
    tracking_error,
    violates,
    violation_checks,
)


def _xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo's WXYZ base quaternion as the XYZW the deploy runtime uses everywhere.

    Args:
        quat_wxyz: ``[4]`` from ``mj_data.qpos[3:7]``.

    Returns:
        ``[4]`` xyzw. Reordered, never reinterpreted.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


class TrackingProbe:
    """Folds the served-reference tracking error for one episode.

    **The opening command hold is LATCHED but not SCORED**, the same split
    ``accumulate.fold``'s ``scored`` argument makes.  The reason here is not the one that
    applies to its velocity statistics, and the difference is worth stating because the obvious
    guess is backwards.  The hold is a SPAWN TRANSIENT, not a near-perfect standing interval:
    the generator's history ring is empty at reset, so its first plans are poor and the reward
    climbs through the hold rather than sitting at 1.0.  Measured over 25 steps at
    ``CMD_HOLD_S = 0.5``::

        tile 20 seed  2   hold 0.834  (0.653 -> 0.903)   scored 0.788
        tile 20 seed  3   hold 0.755  (0.486 -> 0.916)   scored 0.601   (fall)
        tile 20 seed 10   hold 0.796  (0.629 -> 0.906)   scored 0.545   (stumble)
        tile 75 seed  0   hold 0.849  (0.609 -> 0.884)   scored 0.892
        tile 75 seed  1   hold 0.834  (0.630 -> 0.899)   scored 0.888

    So the hold sits near 0.80 whatever the episode goes on to do, and folding it in drags every
    episode TOWARD 0.80 -- up on the bad ones, down on the good ones.  That compresses exactly
    the contrast the reward exists to express, and it compresses it hardest on SHORT episodes,
    where the 25 steps are a larger share: 5% of a 450-step stairs run against 16% of a 154-step
    jump_off_front one.  A violation during the hold still latches, because a robot that cannot
    stand while settling has failed.

    **The Isaac side does NOT do this**, and the difference is deliberate: the command hold is
    a MuJoCo-only artifact (its courses spawn standing and then command full speed; nothing in
    ``scripts/rsl_rl`` has a ``cmd_hold_s``). Isaac has the analogous empty-history ramp and
    keeps it, for the reason ``eval/accumulate._fold_tracking`` records. The consequence is
    that ``track_reward_mean`` is not comparable ACROSS the two harnesses -- only within one.

    Args:
        spec: Thresholds and kernel widths.  Defaults to the un-fitted
            :data:`~transfer.deploy.tracking_error.BOOTSTRAP_SPEC`; anything PUBLISHING a
            number should pass a fitted one and refuse a bootstrap.
        keep_trace: Store every error row.  Needed to fit the thresholds and to plot an
            episode; off for a sweep, where it is the largest per-episode allocation.
        hold_until_s: Absolute sim time the opening hold ends, i.e. the driver's own
            ``cmd_hold_s``.  Read off ``robot.mj_data.time`` inside :meth:`fold` rather than
            taken as a per-call flag, so a caller cannot forget it and quietly publish an
            inflated mean.
    """

    def __init__(self, spec: TrackingErrorSpec = BOOTSTRAP_SPEC, *,
                 keep_trace: bool = False, hold_until_s: float = 0.0) -> None:
        self.spec = spec
        self.keep_trace = bool(keep_trace)
        self.hold_until_s = float(hold_until_s)
        #: Cached joint permutation; the two name lists do not change within a run.
        self._joint_names_isaac: list[str] | None = None
        self.reset()

    def reset(self) -> None:
        """Rewind every running quantity, for a new episode or a restored branch.

        Called by ``live_select``'s snapshot/restore as well as at episode start: a candidate
        chunk that stumbled and was REJECTED must not leave its latch behind, or the first
        rejected candidate would condemn the episode.
        """
        #: Step index of the first bound violation, or -1.
        self.first_violation_step: int = -1
        #: Same, per check, so a report can say WHICH threshold is doing the work.
        self.first_check_step: dict[str, int] = dict.fromkeys(VIOLATION_CHECK_NAMES, -1)
        #: Per-chunk mean tracking reward, earliest first.
        self.chunk_rewards: list[float] = []
        #: Every error row, when ``keep_trace``.
        self.trace: list[np.ndarray] = []
        #: The CONTROL-STEP index each trace row belongs to.  Kept because the trace is
        #: SHORTER than the episode -- no row exists before the generator's first plan -- so a
        #: plot or a join that assumed row *i* was step *i* would be silently shifted by
        #: however many steps opened the episode without a reference.
        self.step_indices: list[int] = []
        #: Whether each trace row was SCORED, i.e. fell outside the opening hold.  Stored so a
        #: plot can shade the unscored region rather than leaving the reader to wonder why the
        #: first half-second is flat at 1.0.
        self.scored_flags: list[bool] = []
        #: Steps folded, and the rows of the chunk currently being accumulated.
        self.n_folded: int = 0
        #: Steps that fed the reward statistics -- the hold does not.
        self.n_scored: int = 0
        self._pending: list[np.ndarray] = []
        self._sum_reward = 0.0
        self._n_reward = 0

    # ------------------------------------------------------------------
    @property
    def violated(self) -> bool:
        """Whether the latch has fired."""
        return self.first_violation_step >= 0

    @property
    def mean_step_reward(self) -> float:
        """Mean per-step tracking reward over the SCORED steps, or NaN before any.

        Scored means "outside the opening command hold" -- see the class docstring for why the
        hold is excluded and why including it biases short episodes upward.
        """
        return self._sum_reward / self._n_reward if self._n_reward else float("nan")

    def error_row(self, robot: Any, policy: Any, *,
                  lin_vel_w: np.ndarray | None = None,
                  ang_vel_w: np.ndarray | None = None) -> np.ndarray | None:
        """The current step's error row, or ``None`` before the first served window.

        Args:
            robot: The MuJoCo ``Robot``, read only.
            policy: The loaded ``GenTrackerPolicy``.
            lin_vel_w: Measured base linear velocity in world, if the caller already has it
                (``EpisodeAccumulator.fold`` does).  Recomputed via ``mj_jacBody`` otherwise.
            ang_vel_w: Measured base angular velocity in world, likewise.

        Returns:
            ``[NUM_FIXED_CHANNELS + nj]``, or ``None`` when no reference has been served yet --
            which is every step before the generator's first plan, and is a real state rather
            than an error.
        """
        ref = policy.served_reference_w()
        if ref is None or "base_lin_vel_w" not in ref or "joint_pos" not in ref:
            # A window without base_vel or joint_pos cannot fill the row. Not raised: a
            # generator predicting only base_pose is a legitimate configuration, it just has
            # nothing for this instrument to measure.
            return None
        if lin_vel_w is None or ang_vel_w is None:
            lin_vel_w, ang_vel_w = robot.get_base_world_velocity(policy.base_frame_name)

        if self._joint_names_isaac is None:
            self._joint_names_isaac = policy.get_joint_names()
        qpos = robot.mj_data.qpos
        act_joint = convert_joint_order(np.asarray(qpos[7:], dtype=np.float64),
                                        robot.joint_names, self._joint_names_isaac)

        return tracking_error(
            ref["base_pos_w"], ref["base_quat_w"], ref["base_lin_vel_w"], ref["base_ang_vel_w"],
            ref["joint_pos"],
            np.asarray(qpos[:3], dtype=np.float64), _xyzw(qpos[3:7]),
            np.asarray(lin_vel_w, dtype=np.float64), np.asarray(ang_vel_w, dtype=np.float64),
            act_joint,
        )

    def fold(self, robot: Any, policy: Any, step_index: int, *,
             is_replan: bool = False,
             lin_vel_w: np.ndarray | None = None,
             ang_vel_w: np.ndarray | None = None) -> np.ndarray | None:
        """Fold one control step: latch, reward, and (optionally) the trace.

        Args:
            robot: The MuJoCo ``Robot``.
            policy: The loaded ``GenTrackerPolicy``.
            step_index: The CONTROL-step index of this step -- the caller's own step counter,
                not a count of folded rows.  It is what the latch is stamped with, so it has to
                be on the same clock as ``first_fall_step`` and as every other index in an
                episode row; and it is what aligns a trace row with the pose captured at the
                same step.
            is_replan: Whether a FRESH plan was produced this step.  The chunk boundary, and
                therefore where a per-chunk reward is closed.  The caller already computes this
                edge for the plan classifier, so it is passed rather than re-detected -- and
                that edge is now advanced unconditionally in ``run_episode``, which it was not
                always (it used to live inside the classifier branch).
            lin_vel_w: Measured base linear velocity in world, if already to hand.
            ang_vel_w: Measured base angular velocity in world, likewise.

        Returns:
            The error row, or ``None`` when no reference has been served yet.
        """
        # The chunk that just ENDED closes on the step a new plan appears, not on the step
        # after: the rows accumulated so far are exactly the ones the previous chunk drove.
        if is_replan and self._pending:
            self._close_chunk()

        err = self.error_row(robot, policy, lin_vel_w=lin_vel_w, ang_vel_w=ang_vel_w)
        if err is None:
            return None

        # Outside the opening hold? See the class docstring: standing at a zero command against
        # a near-stationary reference scores ~1.0 and says nothing, and it is a larger share of
        # a short episode than a long one.
        scored = float(robot.mj_data.time) >= self.hold_until_s

        self.n_folded += 1
        if self.keep_trace:
            self.trace.append(err.astype(np.float32))
            self.step_indices.append(int(step_index))
            self.scored_flags.append(scored)

        if scored:
            self.n_scored += 1
            self._pending.append(err)
            self._sum_reward += float(step_reward(err, self.spec))
            self._n_reward += 1

        # The LATCH runs on every step, hold included: a robot that cannot stand while settling
        # has failed, and this is the same split `_fold_extremes` makes for `first_fall_step`.
        if self.first_violation_step < 0 and bool(violates(err, self.spec)):
            self.first_violation_step = int(step_index)
        for check, hit in violation_checks(err, self.spec).items():
            if self.first_check_step[check] < 0 and bool(hit):
                self.first_check_step[check] = int(step_index)
        return err

    def finish(self) -> None:
        """Close the chunk still in flight, so a truncated episode's last chunk is not lost.

        An episode ends mid-chunk far more often than not -- a fall is not synchronised to the
        re-plan cadence -- and dropping the partial chunk would systematically discard the steps
        immediately before a failure, which are the informative ones.
        """
        if self._pending:
            self._close_chunk()

    def _close_chunk(self) -> None:
        """Reduce the pending rows to one chunk reward."""
        rows = np.stack(self._pending, axis=0)
        self.chunk_rewards.append(float(chunk_reward(rows, self.spec)))
        self._pending = []

    def trace_array(self) -> np.ndarray:
        """The stored trace as ``[steps, NUM_FIXED_CHANNELS + nj]`` float32.

        Returns:
            The rows, or an empty ``[0, NUM_FIXED_CHANNELS]`` array when nothing was kept --
            shaped so a caller can concatenate without special-casing an empty episode.
        """
        if not self.trace:
            return np.zeros((0, NUM_FIXED_CHANNELS), dtype=np.float32)
        return np.stack(self.trace, axis=0)

    def step_index_array(self) -> np.ndarray:
        """The control-step index of each stored trace row, ``[steps]`` int64.

        The trace does not start at step 0 -- nothing is folded before the generator's first
        plan -- so this is what any plot or join against the per-step pose has to use as its x.

        Returns:
            The indices, index-aligned with :meth:`trace_array`.
        """
        return np.asarray(self.step_indices, dtype=np.int64)
