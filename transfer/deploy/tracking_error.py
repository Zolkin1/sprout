# Copyright 2026 Zachary Olkin. All rights reserved.

"""How far the robot is from the reference it was served, in one place.

Three things read this quantity and they must not be allowed to disagree:

* the **failure latch** -- an episode that leaves a box around the served reference, or tilts
  away from it, has stumbled, and the old fall test (pelvis low, or 70 deg from world upright)
  says nothing about that;
* the **dense tracking reward** the failure critic's quality head regresses -- the same errors,
  through an exponential kernel instead of a threshold;
* the **recorders**, which store the row so the thresholds and the kernel widths stay an
  OFFLINE knob rather than a property of whichever collection ran that week.

So there is one producer, :func:`tracking_error`, and the other two are pure functions of its
output.  A second copy of this algebra would be a second place to get the frames wrong, in
code whose entire job is to decide what counts as a failure.

**THREE of the six channels can end an episode: lateral, height, tilt.  Downrange and yaw are
measured and rewarded and are never fatal, and both exclusions are mechanical rather than
cautious.**

* **Downrange** is excluded because the generator legitimately moves the reference along the
  direction of travel to MATCH THE TERRAIN -- a re-plan that lands the reference on the next
  tread jumps it forward, and the tracker then cannot be anything but behind it.  That error is
  the plan doing its job, so bounding it would end episodes for the one thing the generator is
  supposed to do on stairs.  Measured on tile 20: downrange excursions reach 0.30-0.45 m on
  episodes whose lateral and height stay under 0.09 m and whose tilt never passes 15 deg.
* **Yaw** is excluded because it is the channel the anchor divides out in FULL, so it is purely
  chunk-local drift -- what a soft term handles better than a hard bound.

**And the surviving checks are not the same kind of quantity.**  Under ``window_anchor="plan"``
the served node is expressed against the robot's own root at the last re-plan, so:

* ``base_pose[0:3]`` is ``R_yawT (p_ref - p_anchor)`` -- POSITION error is re-zeroed at every
  re-plan and has to accumulate inside one chunk (0.24 s at ``fm_update_every=12``) to fire.
  0.2 m in 0.24 s is 0.83 m/s of relative divergence: a stumble, not drift.
* ``base_pose[3:9]`` is ``rot6d(R_yawT R_ref)``, and the anchor contributes YAW ONLY -- see
  ``robot_rl.flow_matching.window``, which states it outright: *"the yaw frame drops the
  anchor's tilt, never the reference's ... so its absolute tilt against gravity"*.  TILT error
  is therefore NOT re-zeroed.  A robot pitched 20 deg at the re-plan against an upright
  reference starts the chunk already two thirds of the way to a 30 deg bound, so this check
  behaves like an attitude cone and fires far more readily than the position checks.

**Frames.**  Everything is expressed in the REFERENCE's yaw frame and signed as
``actual - reference``, so ``+downrange`` means the robot is ahead of where it should be and the
row reads as "how the robot is displaced from its reference".  Differences are taken in world
and then rotated, never the other way round.  Quaternions are xyzw, matching the whole deploy
runtime.

This module is numpy-only and imports nothing but :mod:`transfer.deploy.quat_math_np`, so it
runs in ``env_transfer`` -- ``mjeval`` without ``--critic_ckpt`` is torch-free and must stay
that way.  The torch twin used inside IsaacLab is
``robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error``, held to this one by
``transfer/deploy/tests/test_tracking_error_parity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np

from .quat_math_np import matrix_from_quat, quat_apply_inverse, yaw_quat

#: Fixed channels of an error row, before the per-joint block.  Named rather than sliced by
#: magic index: the row is written to disk and read back by three consumers, and a silently
#: reordered channel is a plausible number in the wrong column.
ERROR_CHANNEL_NAMES: tuple[str, ...] = (
    "d_downrange", "d_lateral", "d_height",     # reference yaw frame, actual - reference
    "theta_tilt", "theta_yaw",                  # rad
    "dv_x", "dv_y", "dv_z",                     # reference yaw frame, actual - reference
    "dw_x", "dw_y", "dw_z",
)

#: Width of the fixed part; a full row is ``NUM_FIXED_CHANNELS + num_joints``.
NUM_FIXED_CHANNELS: int = len(ERROR_CHANNEL_NAMES)

_POS = slice(0, 3)
_TILT = 3
_YAW = 4
_LIN = slice(5, 8)
_ANG = slice(8, 11)
_JOINT = slice(NUM_FIXED_CHANNELS, None)


def error_channel_names(num_joints: int) -> tuple[str, ...]:
    """Every channel of an error row, in order.

    Args:
        num_joints: Robot joint count; the row's trailing block is one channel per joint, in
            the robot's ISAAC joint order.

    Returns:
        ``NUM_FIXED_CHANNELS + num_joints`` names.
    """
    return ERROR_CHANNEL_NAMES + tuple(f"dq_{i}" for i in range(int(num_joints)))


@dataclass(frozen=True)
class TrackingErrorSpec:
    """The thresholds and kernel widths, together, so a checkpoint carries what it used.

    Split into a failure half and a reward half because they answer different questions and are
    fitted differently.  The WIDTHS can be sized from a population -- where episodes typically
    sit.  The BOUNDS cannot: the only population available is "episodes the old metric passed",
    which is the metric being replaced, so a quantile over it says what the policy currently
    does and not what is acceptable.  Those are chosen by sorting on the peak excursion and
    watching the motions (``transfer/sim/watch_tracking.py``).

    **The defaults are a BOOTSTRAP and are meant to be replaced.**  ``provenance["bootstrap"]``
    marks them, and consumers that publish a number refuse a bootstrap spec rather than
    reporting one -- the same rule ``eval/report.py`` already applies to a bootstrap goal
    region, and for the same reason: a threshold nobody measured produces a confident wrong
    number instead of an error.  ``scripts/rsl_rl/fit_tracking_thresholds.py`` writes a real one.

    **There is deliberately no downrange bound and no yaw bound.**  Not an omission and not a
    value of ``inf`` -- the fields do not exist, so nothing can quietly start reading one.  See
    the module docstring: a terrain-matched re-plan moves the reference downrange on purpose,
    and yaw is divided out by the anchor.

    Attributes:
        max_lateral_m: Bound on ``|d_lateral|``, in the reference's yaw frame.
        max_height_m: Bound on ``|d_height|``, gravity-aligned.
        max_tilt_rad: Bound on the angle between the robot's and the reference's body-z axes.
            NOT re-zeroed at a re-plan -- see the module docstring.
        sigma_pos_m: Width of the position reward kernel (on the 3-vector's norm).
        sigma_ori_rad: Width of the orientation kernel; tilt and yaw share it and enter as a
            sum of squares, so yaw is graded here even though it cannot fail.
        sigma_lin_vel: Width of the linear-velocity kernel (m/s).
        sigma_ang_vel: Width of the angular-velocity kernel (rad/s).
        sigma_joint_rad: Width of the joint kernel, applied to the ROOT-MEAN-SQUARE joint error
            so it does not scale with the joint count.
        weights: ``(position, orientation, velocity, joint)``, summing to 1 so a step reward is
            bounded in ``(0, 1]`` and a normalized return needs no running scale.
        gamma: Per-CHUNK discount, not per control step.  The critic's decisions happen at
            re-plans, so that is the rate the return is defined at.
        provenance: Free-form record of where these numbers came from.  ``{"bootstrap": True}``
            on the defaults.
    """

    max_lateral_m: float = 0.2
    #: **0.12, not 0.20, and it is a MEASURED choice rather than a fitted one.**  Watching the
    #: two-box course showed proper box jump-downs peaking at 0.130-0.182 m against the old
    #: 0.20 bound -- nowhere near it -- while stairs-down-off-a-box dismounts came close, so
    #: the bound is being used to separate a correct dismount from the wrong MODE rather than
    #: to catch a fall.  The cost was measured on the real opening data set before the change
    #: (`awr_vm30/inc_001`, 2000 parent episodes, DR on): clean arrivals 21.70% -> 16.85% of
    #: all episodes, and 95.7% -> 93.4% of windows survive a cut at the latch.  Ground-start
    #: clean arrivals do collapse, 4.18% -> 0.46%, so at this bound every clean success in an
    #: opening set is a box-top dismount; that scarcity predates the change and is what
    #: box-top spawns exist for.
    #:
    #: **The re-anchoring objection does not apply.**  `base_height_below_termination`'s
    #: docstring rejects a reference-relative height bound as a FALL detector, because ego
    #: re-anchoring re-datums the reference every re-plan and a slow sink never accumulates.
    #: That is right, and it is a different use: this bound is chunk-local by construction, so
    #: 0.12 m means "within one re-plan the robot went 12 cm from where the plan put it",
    #: which is a plan-vs-robot disagreement and not a sink.
    #:
    #: Still `bootstrap: True` -- nothing has been FITTED, `fit_tracking_thresholds.py` still
    #: does not exist, and publishing consumers must keep refusing it.
    max_height_m: float = 0.12
    max_tilt_rad: float = float(np.deg2rad(30.0))

    sigma_pos_m: float = 0.10
    sigma_ori_rad: float = float(np.deg2rad(15.0))
    sigma_lin_vel: float = 0.5
    sigma_ang_vel: float = 1.0
    sigma_joint_rad: float = 0.15

    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.10, 0.40)
    gamma: float = 0.95

    provenance: dict[str, Any] = field(default_factory=lambda: {"bootstrap": True})

    def __post_init__(self) -> None:
        """Refuse a spec that cannot mean what it says.

        Raises:
            ValueError: On a non-positive width or bound, or weights that do not sum to 1.  The
                weight check is not pedantry: the whole reason a step reward is bounded in
                ``(0, 1]`` -- and therefore that the quality head's support can be DECLARED
                rather than fitted -- is that they do.
        """
        for name in ("max_lateral_m", "max_height_m",
                     "max_tilt_rad", "sigma_pos_m", "sigma_ori_rad", "sigma_lin_vel",
                     "sigma_ang_vel", "sigma_joint_rad"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if len(self.weights) != 4:
            raise ValueError(f"weights must be (pos, ori, vel, joint), got {self.weights}")
        if any(w < 0.0 for w in self.weights):
            raise ValueError(f"weights must be non-negative, got {self.weights}")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError(
                f"weights must sum to 1 so a step reward is bounded in (0, 1] -- the quality "
                f"head's support is DECLARED on that basis. Got {self.weights}, "
                f"sum {sum(self.weights)!r}.")
        if not 0.0 < float(self.gamma) < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma!r}")

    @property
    def is_bootstrap(self) -> bool:
        """Whether these numbers are the un-fitted placeholders."""
        return bool(self.provenance.get("bootstrap", False))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict, for stamping into a checkpoint or a manifest."""
        d = asdict(self)
        d["weights"] = list(self.weights)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrackingErrorSpec":
        """Rebuild a spec from :meth:`to_dict`.

        Args:
            d: The mapping.

        Returns:
            The spec.

        Raises:
            ValueError: On an unrecognised key.  Deliberately strict: a renamed field that fell
                back to its default would silently score an artifact under different numbers
                than it was trained with.
        """
        known = {f for f in cls.__dataclass_fields__}                     # noqa: SLF001
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"TrackingErrorSpec got unknown keys {sorted(unknown)}; expected a subset of "
                f"{sorted(known)}. Re-stamp the artifact rather than dropping them.")
        kw = dict(d)
        if "weights" in kw:
            kw["weights"] = tuple(float(v) for v in kw["weights"])
        return cls(**kw)


#: The un-fitted spec, for code that has to construct one before Phase 0 has run.
BOOTSTRAP_SPEC = TrackingErrorSpec()


def tracking_error(
    ref_pos_w: np.ndarray,
    ref_quat_w: np.ndarray,
    ref_lin_vel_w: np.ndarray,
    ref_ang_vel_w: np.ndarray,
    ref_joint_pos: np.ndarray,
    act_pos_w: np.ndarray,
    act_quat_w: np.ndarray,
    act_lin_vel_w: np.ndarray,
    act_ang_vel_w: np.ndarray,
    act_joint_pos: np.ndarray,
) -> np.ndarray:
    """One error row: how the robot is displaced from the reference it was served.

    Every argument is WORLD frame and describes the SAME instant -- the served node at the
    current cursor, and the robot as it was when the policy acted on it.  Indexing the plan at
    node 0 instead of ``GenTrackerPolicy.served_node_index`` compares against a reference up to
    ``fm_update_every - 1`` steps stale and puts a sawtooth in every channel that is the
    indexing rather than the tracking.

    Args:
        ref_pos_w: ``[..., 3]`` reference base position.
        ref_quat_w: ``[..., 4]`` reference base orientation, xyzw.
        ref_lin_vel_w: ``[..., 3]`` reference base linear velocity.
        ref_ang_vel_w: ``[..., 3]`` reference base angular velocity.
        ref_joint_pos: ``[..., nj]`` reference joint angles, ISAAC order.
        act_pos_w: ``[..., 3]`` measured base position.
        act_quat_w: ``[..., 4]`` measured base orientation, xyzw.
        act_lin_vel_w: ``[..., 3]`` measured base linear velocity, world.
        act_ang_vel_w: ``[..., 3]`` measured base angular velocity, world.
        act_joint_pos: ``[..., nj]`` measured joint angles, ISAAC order.

    Returns:
        ``[..., NUM_FIXED_CHANNELS + nj]`` float64, laid out as
        :func:`error_channel_names` says.  Angles in radians, everything else in SI.
    """
    ref_p = np.asarray(ref_pos_w, dtype=np.float64)
    ref_q = np.asarray(ref_quat_w, dtype=np.float64)
    act_p = np.asarray(act_pos_w, dtype=np.float64)
    act_q = np.asarray(act_quat_w, dtype=np.float64)

    # The reference's HEADING frame. Yaw-only for the same reason the window itself is
    # yaw-only: rolling the error frame with the reference's own tilt would mix a vertical
    # displacement into `d_downrange` exactly when the reference is pitched, which on stairs is
    # always.
    ref_yaw = yaw_quat(ref_q)

    dp = quat_apply_inverse(ref_yaw, act_p - ref_p)
    dv = quat_apply_inverse(ref_yaw, np.asarray(act_lin_vel_w, np.float64)
                            - np.asarray(ref_lin_vel_w, np.float64))
    dw = quat_apply_inverse(ref_yaw, np.asarray(act_ang_vel_w, np.float64)
                            - np.asarray(ref_ang_vel_w, np.float64))

    tilt = tilt_between(ref_q, act_q)
    yaw = _yaw_error(ref_q, act_q)
    dq = np.asarray(act_joint_pos, np.float64) - np.asarray(ref_joint_pos, np.float64)

    return np.concatenate(
        [dp, tilt[..., None], yaw[..., None], dv, dw, dq], axis=-1)


def tilt_between(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Angle between two orientations' body-z axes, in radians.

    Yaw-independent by construction -- a pure heading change moves neither z axis -- so this
    and :func:`_yaw_error` do not double-count, and neither needs a frame to be resolved in.
    This is the same quantity ``mdp.terminations.base_pose_error_termination`` calls ``tilt``,
    which is what makes the two comparable; the difference is only the datum, which there is
    the reference and in the legacy fall tests is world upright.

    Args:
        q_a: ``[..., 4]`` xyzw.
        q_b: ``[..., 4]`` xyzw.

    Returns:
        ``[...]`` angle in ``[0, pi]``.
    """
    za = matrix_from_quat(np.asarray(q_a, np.float64))[..., :, 2]
    zb = matrix_from_quat(np.asarray(q_b, np.float64))[..., :, 2]
    return np.arccos(np.clip(np.sum(za * zb, axis=-1), -1.0, 1.0))


def _yaw_error(ref_quat: np.ndarray, act_quat: np.ndarray) -> np.ndarray:
    """Signed heading difference ``act - ref``, wrapped to ``(-pi, pi]``.

    Args:
        ref_quat: ``[..., 4]`` xyzw.
        act_quat: ``[..., 4]`` xyzw.

    Returns:
        ``[...]`` radians.
    """
    def _yaw(q: np.ndarray) -> np.ndarray:
        """Heading of an xyzw quaternion."""
        q = np.asarray(q, np.float64)
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    d = _yaw(act_quat) - _yaw(ref_quat)
    return (d + np.pi) % (2.0 * np.pi) - np.pi


#: The channels that can end an episode, in report order.  DOWNRANGE AND YAW ARE ABSENT BY
#: DESIGN -- see the module docstring; both are measured, both are rewarded, neither is fatal.
VIOLATION_CHECK_NAMES: tuple[str, ...] = ("lateral", "height", "tilt")


def violation_checks(err: np.ndarray,
                    spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> dict[str, np.ndarray]:
    """Which check fired, for a report that has to say more than "it failed".

    Three checks, named individually rather than pooled into a "box": they behave completely
    differently -- lateral and height are chunk-local and re-zeroed at every re-plan, tilt is
    measured against gravity and is not -- so a single rate hides which threshold is doing the
    work and which one to move first.

    Args:
        err: ``[..., n]`` rows from :func:`tracking_error`.
        spec: The bounds.

    Returns:
        ``{"lateral", "height", "tilt"}`` -> bool arrays; a row can set several.
    """
    err = np.asarray(err, dtype=np.float64)
    return {
        "lateral": np.abs(err[..., 1]) > float(spec.max_lateral_m),
        "height": np.abs(err[..., 2]) > float(spec.max_height_m),
        "tilt": np.asarray(err[..., _TILT] > float(spec.max_tilt_rad)),
    }


def violates(err: np.ndarray, spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> np.ndarray:
    """Whether an error row is outside the failure bounds.

    The union of :func:`violation_checks`, and it goes through that function rather than
    restating the three tests so the predicate and the per-check breakdown cannot disagree about
    what failed.

    Args:
        err: ``[..., n]`` rows from :func:`tracking_error`.
        spec: The bounds.

    Returns:
        ``[...]`` bool.
    """
    checks = violation_checks(err, spec)
    out = checks[VIOLATION_CHECK_NAMES[0]]
    for name in VIOLATION_CHECK_NAMES[1:]:
        out = out | checks[name]
    return np.asarray(out)


#: Reward terms, in weight order -- ``spec.weights`` is indexed by this.
REWARD_TERM_NAMES: tuple[str, ...] = ("pos", "ori", "vel", "joint")


def step_reward_terms(err: np.ndarray,
                      spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> dict[str, np.ndarray]:
    """The four kernels separately, each in ``(0, 1]``, UNWEIGHTED.

    Exists for the viewer and the threshold fit: a single blended scalar cannot say whether a
    step scored badly because the base drifted or because the arms did, and that is the whole
    question when choosing weights.  :func:`step_reward` is the weighted sum of exactly these,
    so the breakdown cannot disagree with the number the critic regresses.

    Args:
        err: ``[..., n]`` rows from :func:`tracking_error`.
        spec: The kernel widths.

    Returns:
        ``{"pos", "ori", "vel", "joint"}`` -> ``[...]``, each in ``(0, 1]``.
    """
    err = np.asarray(err, dtype=np.float64)
    pos_sq = np.sum(err[..., _POS] ** 2, axis=-1)
    ori_sq = err[..., _TILT] ** 2 + err[..., _YAW] ** 2
    lin_sq = np.sum(err[..., _LIN] ** 2, axis=-1)
    ang_sq = np.sum(err[..., _ANG] ** 2, axis=-1)
    # RMS rather than a sum, so the width does not have to be re-fitted if the robot's joint
    # count ever changes.
    joint_ms = np.mean(err[..., _JOINT] ** 2, axis=-1)
    return {
        "pos": np.exp(-pos_sq / spec.sigma_pos_m ** 2),
        "ori": np.exp(-ori_sq / spec.sigma_ori_rad ** 2),
        "vel": np.exp(-(lin_sq / spec.sigma_lin_vel ** 2 + ang_sq / spec.sigma_ang_vel ** 2)),
        "joint": np.exp(-joint_ms / spec.sigma_joint_rad ** 2),
    }


def step_reward(err: np.ndarray,
                spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> np.ndarray:
    """DeepMimic-style tracking reward for one control step, in ``(0, 1]``.

    Four exponential kernels over exactly the blocks the generator emits -- base pose, base
    twist, joint angles -- and nothing else.  No end effectors and no joint velocities, because
    the live selection (``REFERENCE_WINDOW_BLOCKS``) does not carry them, so a term reading
    either would be grading against something the model never predicted.  Deliberately not the
    CLF: the CLF grades body-link positions in its own anchored frame through a Lyapunov
    matrix, which is the right object for shaping an RL policy and the wrong one for a scalar
    a critic has to regress.

    Bounded above by 1 because the weights sum to 1 and each kernel is at most 1.  That is what
    lets the quality head declare its support instead of fitting a running scale.

    Args:
        err: ``[..., n]`` rows from :func:`tracking_error`.
        spec: The kernel widths and weights.

    Returns:
        ``[...]`` float64 in ``(0, 1]``.
    """
    terms = step_reward_terms(err, spec)
    return sum(float(w) * terms[name]
               for w, name in zip(spec.weights, REWARD_TERM_NAMES))


def chunk_reward(err: np.ndarray,
                 spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> np.ndarray:
    """Mean :func:`step_reward` over a chunk's control steps, in ``(0, 1]``.

    The MEAN rather than a sum, so the value is independent of ``fm_update_every`` and a chunk
    reward stays commensurate with the per-step one it is built from.

    Args:
        err: ``[..., K, n]`` -- the chunk's rows, control steps on the second-to-last axis.
        spec: The kernel widths and weights.

    Returns:
        ``[...]`` float64.
    """
    return np.mean(step_reward(err, spec), axis=-1)


def normalized_return(chunk_rewards: Sequence[float] | np.ndarray,
                      spec: TrackingErrorSpec = BOOTSTRAP_SPEC,
                      *, failed: bool, tail_chunks: int = 1) -> float:
    """``G = (1 - gamma) * sum gamma^i R_i  +  tail``, in ``[0, 1]``.

    The ``(1 - gamma)`` factor is not cosmetic.  It is what puts a return on the same scale as
    a single step's reward, so the quality head's support can be ``[0, 1]`` -- DECLARED, not
    fitted -- and so the measured and bootstrapped halves of an advantage are in the same units.

    **How the episode ENDED decides the tail, and getting this wrong confounds length with
    quality.**  Two terminal cases, and only one of them is absorbing:

    * ``failed=True`` -- the latch fired or the robot fell.  Tail zero: nothing good follows,
      so a run that fails early is worth much less than one that fails late.  This is what
      makes the quality head correlate with survival instead of competing with it.
    * ``failed=False`` -- the robot ARRIVED, or the budget ran out.  Tail
      ``gamma^T * R_{T-1}``: the motion is assumed to continue at the quality it ended on.
      Absorbing here instead would punish arriving QUICKLY -- twelve chunks of flawless
      tracking followed by a goal would score 0.46 against a slow run's 0.9, which is a
      statement about episode length and nothing else.  With the bootstrap, flawless tracking
      scores 1.0 however long it took.

    Args:
        chunk_rewards: ``[T]`` per-chunk rewards, earliest first, already truncated at the
            terminal event (for a latched episode, at the latch).
        spec: Supplies ``gamma``.
        tail_chunks: How many trailing chunk rewards the non-failure tail averages. ``1``
            (the default) is what every existing artifact was built under; the tail is
            stamped nowhere, so changing the DEFAULT would silently re-price every critic
            already fitted. Larger values trade a little bias for variance -- the tail
            carries a median 52% of ``G`` on a survived episode, so resting it on one
            measurement is what makes it worth averaging. MUST match whatever built the
            quality head's target.
        failed: Whether the truncation point is a FAILURE.  Keyword-only and with no default:
            the two answers differ by more than a factor of two on a short episode, and a
            silent default would pick one of them for every caller that forgot.

    Returns:
        The normalized return, in ``[0, 1]``.
    """
    r = np.asarray(chunk_rewards, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return 0.0
    g = float(spec.gamma)
    head = (1.0 - g) * float(np.sum(r * g ** np.arange(r.size, dtype=np.float64)))
    # The tail averages the last `tail_chunks` rewards, clamped to the episode's own
    # length so a short one averages what it has rather than reaching past its start.
    k = max(1, min(int(tail_chunks), r.size))
    tail = 0.0 if failed else float(g ** r.size * float(np.mean(r[-k:])))
    return head + tail
