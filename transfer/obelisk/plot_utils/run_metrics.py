# Copyright 2026 Zachary Olkin. All rights reserved.

"""ONE fall criterion, applied to both stacks.

The question this exists for is "does the obelisk stack fall where MuJoCo sim2sim does not",
and the fastest way to get a wrong answer is to score the two runs with two implementations.
So the criterion lives here, takes the two quantities BOTH stacks record per tick
(``root_pos`` and ``base_quat_xyzw`` -- ``ControllerDebug``/``est_state`` on one side,
``deploy.replay_stream``'s input fields on the other), and neither caller restates it.

It is the criterion from :meth:`transfer.sim.robot.Robot.failure_reason`, transcribed rather
than imported because importing it drags in mujoco and pygame at module scope, and the
obelisk side must be readable on a box with neither. :func:`assert_matches_runner` checks the
transcription against the runner's own verdict whenever a ``sim_meta.yaml`` is at hand, so the
two cannot drift silently.

Two properties worth keeping in mind when reading a result:

* **Terrain-relative.** ``fall_height`` is measured above the scene's LOWEST terrain surface,
  not above zero. An absolute ``z < 0.2`` scored every stairs-down run as fallen before it
  began, because tile20's bottom slab sits at -0.777 m.
* **Roll and pitch bound SEPARATELY**, on the pelvis's up-axis rather than on Euler angles --
  yaw-independent by construction, and Euler degenerates near pitch = 90 deg, which is exactly
  where a falling robot ends up.
"""

from __future__ import annotations

import math

import numpy as np

#: Metres above the scene's lowest terrain surface below which the pelvis counts as down.
DEFAULT_FALL_HEIGHT = 0.2

#: Radians from upright, in roll or pitch separately, beyond which the robot counts as tipped.
DEFAULT_FALL_TILT_RAD = math.radians(70.0)


def tilt_from_upright(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute roll and pitch away from upright, per sample.

    From the body's UP axis (the third column of the rotation matrix), so the result is
    independent of heading and has no Euler degeneracy near pitch = 90 deg.

    Args:
        quat_xyzw: ``[N, 4]`` base orientation, ``(x, y, z, w)``.

    Returns:
        ``(|roll|, |pitch|)``, each ``[N]``, in radians.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Third column of R(q): the body z-axis expressed in the world.
    up = np.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
                  axis=-1)
    return np.abs(np.arctan2(up[:, 1], up[:, 2])), np.abs(np.arctan2(-up[:, 0],
                                                                     np.hypot(up[:, 1], up[:, 2])))


def failure_reason(base_pos: np.ndarray, base_quat_xyzw: np.ndarray, terrain_min_z: float = 0.0,
                   fall_height: float = DEFAULT_FALL_HEIGHT,
                   fall_tilt_rad: float = DEFAULT_FALL_TILT_RAD) -> np.ndarray:
    """Per-sample ``"height"`` / ``"tilt"`` / ``""``.

    Args:
        base_pos: ``[N, 3]`` base world position. Only z is read.
        base_quat_xyzw: ``[N, 4]`` base orientation.
        terrain_min_z: Lowest terrain surface in the scene (see
            :func:`terrain_min_surface_z`). 0 for a flat world.
        fall_height: Metres above ``terrain_min_z`` the pelvis must stay.
        fall_tilt_rad: Roll/pitch bound, applied to each separately.

    Returns:
        ``[N]`` array of strings.
    """
    z = np.asarray(base_pos, dtype=np.float64)[:, 2]
    roll, pitch = tilt_from_upright(base_quat_xyzw)
    out = np.full(z.shape, "", dtype=object)
    out[(roll > fall_tilt_rad) | (pitch > fall_tilt_rad)] = "tilt"
    out[z < terrain_min_z + fall_height] = "height"   # last: height wins when both fire
    return out


def verdict(base_pos: np.ndarray, base_quat_xyzw: np.ndarray, t: np.ndarray,
            terrain_min_z: float = 0.0, use_height: bool = True,
            **kwargs) -> tuple[str, float | None, int | None]:
    """The run's outcome: why it ended, when, and at which sample.

    Args:
        base_pos: ``[N, 3]`` base world position.
        base_quat_xyzw: ``[N, 4]`` base orientation.
        t: ``[N]`` time, seconds, zeroed at the start of the scored window.
        terrain_min_z: Lowest terrain surface.
        use_height: Whether the height half of the criterion is meaningful. FALSE when the
            base POSITION is not trustworthy -- a hardware run with no odometry, or an obelisk
            clip run where ``root_pos`` is never written (the plan anchor makes it optional,
            so it can be identically zero). Scoring height there reports "fell" for every
            tick of a healthy run, which is worse than not scoring it.
        **kwargs: Passed to :func:`failure_reason`.

    Returns:
        ``(reason, time_of_fall, index_of_fall)``; ``("", None, None)`` if it survived.
    """
    reasons = failure_reason(base_pos, base_quat_xyzw, terrain_min_z, **kwargs)
    if not use_height:
        reasons = np.where(reasons == "height", "", reasons)
    hit = np.flatnonzero(reasons != "")
    if hit.size == 0:
        return "", None, None
    i = int(hit[0])
    return str(reasons[i]), float(t[i]), i


def first_measured_sample(base_pos: np.ndarray) -> int:
    """Index of the first sample whose base position is a MEASUREMENT rather than a zero.

    Arming does not wait for the estimator, so the first few ticks of a bag carry an
    all-zero pose -- measured here, three of them. Those ticks have no base position at all,
    and scoring them reports a pelvis on the floor at t=0 for a robot that is standing
    perfectly well. That is not a conservative failure: it hides the real verdict behind a
    fake one at the very first sample, which is where nobody looks.

    Args:
        base_pos: ``[N, 3]`` base world position.

    Returns:
        The first index with a non-zero position, or ``0`` if there is none (an
        odometry-free bag, where :func:`base_position_is_usable` then turns the height check
        off entirely).
    """
    p = np.asarray(base_pos, dtype=np.float64)
    nz = np.flatnonzero(np.any(np.abs(p) > 1e-9, axis=-1))
    return int(nz[0]) if nz.size else 0


def base_position_is_usable(base_pos: np.ndarray) -> bool:
    """Whether the base HEIGHT carries a real measurement.

    The C++ deploy runtime is odometry-free by design: under the ``plan`` anchor ``root_pos``
    reaches only the diagnostics, so a bag can legitimately contain an all-zero base position
    beside a robot that walked several metres. Reading that as "the pelvis is on the floor" is
    the whole reason this check exists.

    Judged on the Z CHANNEL specifically, not on the vector. One bag here had real x and y
    with z identically zero, and a whole-vector test called that usable and then scored every
    tick as a fall.
    """
    p = np.asarray(base_pos, dtype=np.float64)
    return bool(p.size and np.isfinite(p[:, 2]).all() and np.any(np.abs(p[:, 2]) > 1e-9))


def terrain_min_surface_z(robot_name: str, scene_name: str) -> float:
    """Lowest terrain surface in a MuJoCo scene, or 0.0 if it cannot be loaded.

    Delegates to :class:`transfer.sim.robot.Robot`, which is the definition. Imported lazily:
    that module pulls in mujoco and pygame at module scope, and this file has to stay readable
    on a box with neither.
    """
    try:
        import numpy as _np

        from sim.robot import Robot
        r = Robot(robot_name, scene_name, _np.ones(3))
        return float(r.terrain_min_z)
    except Exception as exc:                       # noqa: BLE001 -- a scoring convenience
        print(f"[run_metrics] no terrain datum for {robot_name}/{scene_name} ({exc}); "
              "using 0.0, so the height check is against a flat floor.")
        return 0.0


def assert_matches_runner(meta: dict, reason: str, t_fall: float | None,
                          tol_s: float = 0.25) -> None:
    """Check this criterion against the MuJoCo runner's own verdict.

    The runner evaluates the same rule inside its loop and stamps the answer into
    ``sim_meta.yaml``. Re-deriving it here from the recorded stream is what makes the obelisk
    side comparable -- and a disagreement means the transcription has drifted, which would
    silently corrupt every comparison built on it.

    Args:
        meta: The runner's ``sim_meta.yaml``.
        reason: This module's verdict.
        t_fall: This module's fall time, or None.
        tol_s: Allowed disagreement in the fall INSTANT. The runner checks before its physics
            step and this scores the recorded sample, so a tick or two of slack is expected;
            the VERDICT itself must match exactly.

    Raises:
        AssertionError: On a disagreement.
    """
    survived_meta = meta.get("survived")
    if survived_meta is None:
        return                                     # Ctrl-C: neither survival nor a fall
    assert bool(survived_meta) == (reason == ""), (
        f"the shared criterion says {'survived' if not reason else reason!r} but the runner "
        f"recorded survived={survived_meta}. The transcription in run_metrics has drifted "
        "from Robot.failure_reason.")
    runner_t = meta.get("time_to_fall_s")
    if runner_t is not None and t_fall is not None:
        assert abs(runner_t - t_fall) <= tol_s, (
            f"fall time disagrees: runner {runner_t:.2f}s vs {t_fall:.2f}s")
