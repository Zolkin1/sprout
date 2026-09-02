# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-episode metric accumulation for the MuJoCo terrain eval.

One instance per EPISODE, where ``scripts/rsl_rl/eval/accumulate.py`` carries every env at
once -- IsaacLab steps a thousand robots together, MuJoCo steps one.  That removes the whole
done-mask/reset-hook apparatus that shapes the Isaac file: nothing here can fold two episodes
into one buffer, because a buffer only ever sees one.  What survives from it, deliberately, is
the ``fold`` / ``fold_terminal`` split: the loop folds before each physics advance, so the
frame the episode ENDS on -- for a fall, exactly the frame that crossed the threshold -- is
never folded by the loop and is picked up once at the end.

The other inherited rule is that every quantity is stored as a SUM of squares (or a running
count) and divided only in :meth:`row`.  A mean of per-episode RMSEs is biased low by Jensen,
and the reducer forms ``sqrt(mse_vx + mse_vy)`` at the join, which needs the squares.

numpy only.  The MuJoCo ids this needs (actuator indices, and the dof addresses behind them)
are ctor arguments or are read straight off ``robot.mj_model`` as plain arrays, so nothing
here imports ``mujoco`` or calls a name lookup in the hot loop.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# `terrains` and `schema` are siblings reached by bare name, the way `terrains` itself reaches
# `generate_stairs_scene`: this directory is imported both as a package (`mjeval.accumulate`)
# and as a bare module by the runner, and only the path insert works in both.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import terrains  # noqa: E402
import tracking  # noqa: E402

#: Standard gravity used for the cost of transport, matching the Isaac harness's constant so
#: the two numbers are comparable without a unit conversion.
GRAVITY: float = 9.81


def yaw_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """The pure-yaw part of a quaternion: same heading, zero roll and pitch.

    Written out here rather than imported from ``isaaclab.utils.math`` because this package
    runs in ``env_transfer``, which has no IsaacLab -- and it is the SAME construction the
    generator's level frame uses (``yaw_quat(root_quat_w)``), so the eval measures velocity in
    the frame the command was issued in.

    Args:
        quat_wxyz: Root quaternion in MuJoCo's ``(w, x, y, z)`` order.

    Returns:
        ``(w, 0, 0, z)``, unit norm, rotating the yaw-aligned frame into world.
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Express a world-frame vector in the frame ``quat_wxyz`` rotates into world.

    ``q^-1 * v * q`` in closed form; the cross term's sign is what distinguishes it from the
    forward rotation, and getting it backwards is silent on any symmetric test vector.

    Args:
        quat_wxyz: Unit quaternion in ``(w, x, y, z)`` order.
        vec: World-frame vector, length 3.

    Returns:
        The same vector's components in the quaternion's own frame.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w, u = float(q[0]), q[1:]
    return v * (2.0 * w * w - 1.0) - 2.0 * w * np.cross(u, v) + 2.0 * u * float(u @ v)


def _mean(total: float, count: int) -> float:
    """A mean that is NaN rather than a division error when nothing was counted.

    Args:
        total: Running sum.
        count: How many terms went into it.

    Returns:
        ``total / count``, or NaN when ``count`` is zero.
    """
    return float(total / count) if count > 0 else float("nan")


class EpisodeAccumulator:
    """Folds one episode's per-step state into the single row :mod:`schema` stores.

    **The velocity frame is split on purpose, and the split is the point of this class.**
    The commanded velocity ``(vx, vy, wz)`` is issued in the root's YAW frame -- gravity-aligned
    z, zero roll and pitch -- so that is the only frame in which "did it track the command" is
    a well-posed question.  ``gen_tracker_runner``'s CSV instead reports ``vx_b``/``vy_b`` by
    rotating the world velocity through the inverse of the FULL root quaternion, which bleeds
    vertical velocity into the downrange channel whenever the torso pitches.  That error is
    systematic and it is worst exactly on stairs and box jumps, where the torso pitches most
    and where this eval spends its entire budget.  ``wz_b`` has the same defect from the other
    direction: it is ``qvel[5]``, the BODY-local angular z (MuJoCo stores a free joint's
    angular velocity in the body frame), while the commanded ``wz`` is a world-z yaw rate.

    So the headline columns ``vel_mse_{vx,vy,wz}`` use the yaw frame, with ``wz`` taken as the
    world-frame angular z -- which is yaw-invariant, hence the same number in either frame.
    ``vel_mse_vx_body`` / ``vel_mse_vy_body`` keep the full body frame ONLY so the numbers stay
    comparable with the Isaac harness, whose ``fold`` reads ``root_lin_vel_b``; they are the
    contaminated pair and are not the headline.
    """

    def __init__(self, course: terrains.Course, cmd: np.ndarray, dt: float,
                 act_ids: Sequence[int] | np.ndarray, effort_limits: np.ndarray,
                 total_mass: float, tracking_spec=None, hold_until_s: float = 0.0) -> None:
        """Allocate this episode's running sums.

        Args:
            course: The authored course being run; supplies the goal rectangle, the
                approach/on-course split at ``obstacle_x_m`` and the traversal gate.
            cmd: The commanded ``(vx, vy, wz)`` held for the whole episode, in the yaw frame.
            dt: Control step, seconds.  The fold cadence, so it is also the work integrand's
                and the action rate's denominator.
            act_ids: MuJoCo actuator indices in the POLICY's joint order, as
                ``gen_tracker_runner`` resolves them (``f"{isaac_name}_pos"``).
            effort_limits: Per-joint effort limit, same order as ``act_ids``, from the
                tracker's own metadata.  Non-positive entries would make the utilisation
                infinite, so they are replaced by 1.0 and the channel reads as raw torque.
            total_mass: Robot mass (kg), the cost of transport's denominator.
            tracking_spec: A ``deploy.tracking_error.TrackingErrorSpec`` to turn on the STRICT
                tracking checks, or ``None`` to leave them off.  Off means the columns come
                back -1 / NaN, which a reader must be able to tell from "measured and never
                fired" -- the same rule the Isaac accumulator's `attach_tracking` follows.
            hold_until_s: End of the opening command hold; its steps are LATCHED but not
                SCORED (see ``mjeval.tracking.TrackingProbe``).
        """
        self.course = course
        self.cmd = np.asarray(cmd, dtype=np.float64).reshape(3)
        self.dt = float(dt)
        self.total_mass = float(total_mass)
        self._act_ids = np.asarray(act_ids, dtype=np.int64)
        limits = np.asarray(effort_limits, dtype=np.float64).reshape(-1)
        self._effort = np.where(limits > 0.0, limits, 1.0)
        # Resolved on the first fold: the ctor is handed ids, not a model, and the dof address
        # behind each actuator is a two-hop lookup through the model's own arrays.
        self._dof_ids: np.ndarray | None = None

        self._steps = 0
        # Steps that actually fed the velocity/effort statistics. Separate from `_steps`
        # because the opening standing hold advances the clock without being scored, and
        # dividing a sum of 250 squares by 275 steps is a quietly wrong mean.
        self._n_scored = 0
        # Commanded-velocity tracking. Squares, never RMSE (see the module docstring).
        self._vel_sq = np.zeros(3)          # yaw frame: the headline
        self._vel_sq_body = np.zeros(2)     # full body frame: the Isaac-comparable pair
        self._vx_sq_pre = 0.0
        self._vx_sq_post = 0.0
        self._n_pre = 0
        self._n_post = 0
        # Extremes.
        self._min_base_z = float("inf")
        self._max_base_z = float("-inf")
        self._max_tilt = 0.0
        self._max_x = float("-inf")
        self._min_target_dist = float("inf")
        self._first_fall_step = -1
        self._target_step = -1
        # The STRICT tracking checks, against the SERVED reference node. `robot.failed()`
        # above is reference-FREE and says nothing about a robot that stumbles down a flight
        # and barely catches itself.
        self.track = (tracking.TrackingProbe(tracking_spec, hold_until_s=float(hold_until_s))
                      if tracking_spec is not None else None)
        # Effort, smoothness, travel.
        self._torque_util_max = 0.0
        self._torque_sq = 0.0
        self._n_torque = 0
        self._work_j = 0.0
        self._act_rate_sum = 0.0
        self._act_rate_max = 0.0
        self._dist_m = 0.0
        self._prev_action: np.ndarray | None = None
        self._prev_xy: np.ndarray | None = None
        # What KIND of motion each re-plan produced, when a classifier is attached.
        self._plan_counts = {"flat": 0, "jump": 0, "stairs": 0}
        # Inference cost, weighted by the call counts pop_avg_infer_ms reports.
        self._fm_ms_sum = 0.0
        self._fm_calls = 0
        self._trk_ms_sum = 0.0
        self._trk_calls = 0
        # Terminal frame, filled by fold_terminal.
        self._end_base_z = float("nan")
        self._end_tilt = float("nan")

    # ------------------------------------------------------------------
    def _resolve_dof_ids(self, robot: Any) -> np.ndarray:
        """Map this episode's actuators to their qvel addresses, once.

        ``actuator_trnid[:, 0]`` is the driven joint and ``jnt_dofadr`` is that joint's first
        dof, which for the G1's hinges is its only one -- so the result indexes ``qvel``
        directly and in the policy's joint order, which is what makes ``tau * qvel`` a
        per-joint product rather than two vectors that merely have the same length.

        Args:
            robot: The MuJoCo ``Robot``.

        Returns:
            Integer dof addresses, one per entry of ``act_ids``.
        """
        model = robot.mj_model
        joints = np.asarray(model.actuator_trnid, dtype=np.int64)[self._act_ids, 0]
        return np.asarray(model.jnt_dofadr, dtype=np.int64)[joints]

    def _fold_extremes(self, robot: Any, step_index: int) -> None:
        """Update every running EXTREME from the current physics state.

        Shared by :meth:`fold` and :meth:`fold_terminal` so the terminal frame reaches the
        extremes, the fall crossing and the goal rectangle on exactly the same terms as any
        other frame.  The running SUMS deliberately do not go through here: their divisor is
        the folded step count, and advancing one without the other biases every mean.

        Args:
            robot: The MuJoCo ``Robot``.
            step_index: Index to stamp on a first crossing -- the number of steps folded so
                far, which for the terminal call is the terminal frame's own index.
        """
        qpos = robot.mj_data.qpos
        base_z = float(qpos[2])
        self._min_base_z = min(self._min_base_z, base_z)
        self._max_base_z = max(self._max_base_z, base_z)
        self._max_tilt = max(self._max_tilt, max(robot.tilt_from_upright()))
        self._max_x = max(self._max_x, float(qpos[0]))

        # Distance to the goal RECTANGLE, not to its centre: a robot stopped at the band's
        # edge has arrived for scoring purposes, and measuring it against the centre would
        # report it as half a lane short.  Zero inside.
        x, y = float(qpos[0]), float(qpos[1])
        self._min_target_dist = min(self._min_target_dist, self.course.goal_distance(x, y))
        if self._target_step < 0 and self.course.in_goal(x, y):
            self._target_step = step_index
        if self._first_fall_step < 0 and robot.failed():
            self._first_fall_step = step_index

    # ------------------------------------------------------------------
    def fold(self, robot: Any, policy: Any, plan_class: str | None = None,
             cmd: Any = None, scored: bool = True, is_replan: bool = False) -> None:
        """Fold one control step into the running sums.

        Called once per control step BEFORE the physics advance, so every folded frame is a
        state the policy actually acted on.

        Args:
            scored: False during the opening standing hold, when the command is zero by
                construction.  The EXTREMES still fold -- a fall while settling is a fall, and
                a robot that cannot stand should be visible -- but the velocity, effort and
                smoothness statistics do not, because a commanded zero met by a standing robot
                is a near-perfect tracking score that says nothing about the policy.  Folding it
                would dilute every episode's error, and dilute the SHORT ones most, which are
                the falls -- so the bias would land differently on an arm that falls more.
            cmd: The command actually issued THIS step, or ``None`` to use the episode's
                constant.  Only the steering controller varies it, and the velocity error must
                be scored against what was issued rather than against the constant it started
                from -- otherwise every steered step reads as a yaw-tracking failure of exactly
                the size the steering asked for.
            robot: The MuJoCo ``Robot``, read only.
            policy: The loaded ``GenTrackerPolicy``; supplies the base frame name, the raw
                action and the inference timers.
            plan_class: ``"flat"``, ``"jump"``, ``"stairs"`` when this step carried a fresh
                re-plan that a classifier labelled, else ``None``.  ``None`` means no
                classification THIS step -- most steps are not re-plans -- and is not a
                fourth class.
            is_replan: Whether a fresh plan appeared this step.  Passed separately from
                ``plan_class``, which is ``None`` on every step when no classifier is
                attached -- deriving the edge from it would put the whole episode in one
                chunk on the clip arm and on every un-classified run.
        """
        data = robot.mj_data
        qpos = data.qpos
        self._fold_extremes(robot, self._steps)
        # Before the `scored` early-out: a violation while settling is still a violation, and
        # the probe makes the latched/scored split itself from the sim clock.
        if self.track is not None:
            self.track.fold(robot, policy, self._steps, is_replan=bool(is_replan))
        if not scored:
            # The hold still costs a step of the clock and of the episode's length, so the
            # counter advances; only the statistics above sit it out.
            self._steps += 1
            return

        self._n_scored += 1

        # -- commanded-velocity tracking, in both frames --------------------------------
        lin_w, ang_w = robot.get_base_world_velocity(policy.base_frame_name)
        quat = np.asarray(qpos[3:7], dtype=np.float64)
        lin_yaw = rotate_inverse(yaw_quat(quat), lin_w)
        lin_body = rotate_inverse(quat, lin_w)
        live = self.cmd if cmd is None else np.asarray(cmd, dtype=float)
        err = np.array([lin_yaw[0] - live[0], lin_yaw[1] - live[1],
                        float(ang_w[2]) - live[2]])
        self._vel_sq += err * err
        err_body = np.array([lin_body[0] - live[0], lin_body[1] - live[1]])
        self._vel_sq_body += err_body * err_body
        # Approach vs on-course. The approach is the honest tracking number: past the first
        # riser or box face a deviation from the commanded velocity is the terrain doing its
        # job, not the tracker failing to do its own.
        if float(qpos[0]) < self.course.obstacle_x_m:
            self._vx_sq_pre += float(err[0] * err[0])
            self._n_pre += 1
        else:
            self._vx_sq_post += float(err[0] * err[0])
            self._n_post += 1

        # -- effort ----------------------------------------------------------------------
        if self._dof_ids is None:
            self._dof_ids = self._resolve_dof_ids(robot)
        tau = np.asarray(data.actuator_force, dtype=np.float64)[self._act_ids]
        qvel = np.asarray(data.qvel, dtype=np.float64)[self._dof_ids]
        self._torque_util_max = max(self._torque_util_max,
                                    float(np.max(np.abs(tau) / self._effort)))
        self._torque_sq += float(tau @ tau)
        self._n_torque += tau.size
        self._work_j += self.dt * float(np.sum(np.abs(tau * qvel)))

        # -- smoothness ------------------------------------------------------------------
        action = np.asarray(policy._trk.action_isaac, dtype=np.float64).reshape(-1)
        if self._prev_action is not None:
            rate = float(np.linalg.norm(action - self._prev_action)) / self.dt
            self._act_rate_sum += rate
            self._act_rate_max = max(self._act_rate_max, rate)
        self._prev_action = action.copy()

        # -- travel ----------------------------------------------------------------------
        xy = np.array([float(qpos[0]), float(qpos[1])])
        if self._prev_xy is not None:
            self._dist_m += float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy = xy

        if plan_class is not None:
            self._plan_counts[plan_class] += 1

        # -- inference cost ---------------------------------------------------------------
        # pop_avg_infer_ms DRAINS its window, so it has to be read every step or it reports
        # whatever happened since some other consumer last looked. Weighting by the counts it
        # returns makes the episode mean a mean over CALLS: the generator runs once per
        # `fm_update_every` steps and contributes `fm_n == 0` on the rest.
        pop = getattr(policy, "pop_avg_infer_ms", None)
        if pop is not None:
            stats = pop()
            fm_n, trk_n = int(stats["fm_n"]), int(stats["trk_n"])
            self._fm_ms_sum += float(stats["fm_ms"]) * fm_n
            self._fm_calls += fm_n
            self._trk_ms_sum += float(stats["trk_ms"]) * trk_n
            self._trk_calls += trk_n

        self._steps += 1

    def fold_terminal(self, robot: Any, policy: Any = None) -> None:
        """Fold the frame the episode ENDED on into the extremes, and stamp the end state.

        The loop folds before each physics advance and stops after one, so without this the
        final state is never seen -- and for a fall that is precisely the frame that crossed
        the threshold, which would leave ``first_fall_step`` at -1 on a robot lying on the
        ground.  ``self._steps`` is the terminal frame's own index, one past the last folded
        step, so a crossing found here is stamped consistently with one found in the loop.

        Args:
            robot: The MuJoCo ``Robot``, in its final state.
            policy: The loaded policy, for the strict checks' served reference. ``None`` skips
                them on this frame -- the running latch is unaffected, so an existing caller
                that does not pass it loses only the terminal frame's crossing rather than the
                whole column.
        """
        self._fold_extremes(robot, self._steps)
        # And the strict checks, for the same reason the fall crossing is folded here: on a
        # fall the loop breaks BEFORE `fold`, so the crossing frame is only ever seen here.
        if self.track is not None and policy is not None:
            self.track.fold(robot, policy, self._steps)
        if self.track is not None:
            self.track.finish()
        self._end_base_z = float(robot.mj_data.qpos[2])
        self._end_tilt = float(max(robot.tilt_from_upright()))

    # ------------------------------------------------------------------
    @property
    def traversed(self) -> bool:
        """Did the root actually go over (or down) the course rather than around it?

        Delegates to :meth:`terrains.Course.traversed`; these lanes are finite in y on an
        infinite floor, so the goal rectangle alone can be entered by walking beside the
        course.
        """
        return self.course.traversed(self._max_base_z, self._min_base_z)

    def row(self, episode_id: int, terrain_idx: int, speed_idx: int, arm: int, seed: int,
            verdict: int | str, failure_code: int | str) -> dict[str, Any]:
        """Reduce the episode to the one row :mod:`schema` stores.

        Args:
            episode_id: Unique within the run.
            terrain_idx: Index into :func:`terrains.catalogue`.
            speed_idx: Index into ``course.speeds``.
            arm: Which tracker ran (the generator is shared).
            seed: The trial's seed, recorded even though every cell is deterministic --
                ``--zero_noise`` integrates the flow from ``x0 = 0`` and MuJoCo is
                deterministic, so a repeat is bit-identical and the arms pair exactly.
            verdict: A :data:`terrains.VERDICT_CODES` key or its code.
            failure_code: A :data:`terrains.FAILURE_CODES` key (``Robot.failure_reason()``'s
                own string) or its code.

        Returns:
            One dict carrying exactly the keys in ``schema.MJ_EPISODE_COLUMNS``.

        Raises:
            KeyError: If the produced keys and the schema's disagree in either direction.
                Failing here names the offending column; failing at write time names a dtype.
        """
        import schema  # noqa: PLC0415  -- see the module docstring's path note

        steps = self._steps
        finite = steps > 0 or math.isfinite(self._end_base_z)
        work = self._work_j
        travel = self._dist_m
        # Same guard as the Isaac accumulate: below a millimetre the ratio is the numerical
        # noise in the numerator divided by the numerical noise in the denominator.
        denom = self.total_mass * GRAVITY * travel
        cot = float(work / denom) if travel > 1e-6 else float("nan")

        out: dict[str, Any] = {
            "episode_id": int(episode_id),
            "arm": int(arm),
            "seed": int(seed),
            "terrain_idx": int(terrain_idx),
            "terrain_class": int(self.course.class_code),
            "variant": int(self.course.variant),
            "speed_idx": int(speed_idx),
            "cmd_vx": float(self.cmd[0]),
            "cmd_vy": float(self.cmd[1]),
            "cmd_wz": float(self.cmd[2]),
            "steps_survived": int(steps),
            "verdict": int(terrains.VERDICT_CODES[verdict] if isinstance(verdict, str)
                           else verdict),
            "failure_code": int(terrains.FAILURE_CODES[failure_code]
                                if isinstance(failure_code, str) else failure_code),
            "first_fall_step": int(self._first_fall_step),
            # The STRICT tracking checks. -1 / NaN when they were not turned on at all, which
            # `schema.derive_success` distinguishes from "measured and never fired".
            "first_violation_step": int(self.track.first_violation_step
                                        if self.track is not None else -1),
            "first_lateral_step": int(self.track.first_check_step["lateral"]
                                      if self.track is not None else -1),
            "first_height_step": int(self.track.first_check_step["height"]
                                     if self.track is not None else -1),
            "first_tilt_step": int(self.track.first_check_step["tilt"]
                                   if self.track is not None else -1),
            "track_reward_mean": (float(self.track.mean_step_reward)
                                  if self.track is not None else float("nan")),
            "reached_target": int(self._target_step >= 0),
            "time_to_target_s": (float(self._target_step * self.dt)
                                 if self._target_step >= 0 else float("nan")),
            "traversed": int(self.traversed),
            "progress_x_m": float(self._max_x) if finite else float("nan"),
            "progress_frac": (float(self._max_x / self.course.course_length_m)
                              if finite else float("nan")),
            "distance_m": float(travel),
            "min_dist_to_target_m": (float(self._min_target_dist) if finite
                                     else float("nan")),
            "vel_mse_vx": _mean(self._vel_sq[0], self._n_scored),
            "vel_mse_vy": _mean(self._vel_sq[1], self._n_scored),
            "vel_mse_wz": _mean(self._vel_sq[2], self._n_scored),
            "vel_mse_vx_body": _mean(self._vel_sq_body[0], self._n_scored),
            "vel_mse_vy_body": _mean(self._vel_sq_body[1], self._n_scored),
            "vel_mse_vx_pre": _mean(self._vx_sq_pre, self._n_pre),
            "vel_mse_vx_post": _mean(self._vx_sq_post, self._n_post),
            "min_base_height_w": float(self._min_base_z) if finite else float("nan"),
            "max_base_height_w": float(self._max_base_z) if finite else float("nan"),
            "max_tilt_rad": float(self._max_tilt),
            "end_base_height_w": float(self._end_base_z),
            "end_tilt_rad": float(self._end_tilt),
            # One fewer difference than steps: the first fold has no predecessor.
            "action_rate_mean": _mean(self._act_rate_sum, max(self._n_scored - 1, 0)),
            "action_rate_max": float(self._act_rate_max),
            "torque_util_max": float(self._torque_util_max),
            "torque_rms": math.sqrt(_mean(self._torque_sq, self._n_torque)),
            "mech_work_j": float(work),
            "cost_of_transport": cot,
            "n_plans": int(sum(self._plan_counts.values())),
            "n_plans_flat": int(self._plan_counts["flat"]),
            "n_plans_jump": int(self._plan_counts["jump"]),
            "n_plans_stairs": int(self._plan_counts["stairs"]),
            "fm_ms_mean": _mean(self._fm_ms_sum, self._fm_calls),
            "trk_ms_mean": _mean(self._trk_ms_sum, self._trk_calls),
        }

        # The schema may spell its columns as names or as (name, dtype) pairs; either way the
        # names are the contract, and both directions of disagreement are a bug worth naming.
        cols = schema.MJ_EPISODE_COLUMNS
        want = {c if isinstance(c, str) else c[0] for c in cols}
        missing, extra = want - set(out), set(out) - want
        if missing or extra:
            raise KeyError(
                f"accumulate.row does not match schema.MJ_EPISODE_COLUMNS: "
                f"missing {sorted(missing)}, unexpected {sorted(extra)}")
        return out
