# Copyright 2026 Zachary Olkin. All rights reserved.

"""ONE velocity-command schedule sampler, shared by every clip family.

A clip's command schedule is a list of ``(duration_s, vx, vy, wz)`` segments. The robot's motion is
generated to TRACK that schedule, and the RL sees the same schedule as ``target_vel`` -- so the two
agree by construction rather than by two code paths happening to sample the same distribution.

Families differ only in what they pass in, not in which sampler they call:

* **flat** -- the full command space: forward / backward / sidestep / turn-while-walking / diagonal,
  a standing lead-in, and pauses. Nothing overrides it, so the motion tracks it end to end.
* **terrain (transition / box)** -- forward only (``side_vy = turn_max = 0``, no pauses): a traversal
  is authored along the tile's +x, and a lateral or yaw command would walk the robot off it. The
  motion tracks the schedule too, EXCEPT where the terrain overrides it -- the last ``ramp_s`` of the
  approach and the terrain interaction itself. That override belongs to the consumer
  (``mb_terrain_segments``), not here: this module has no concept of terrain.
* **pure stairs** -- no schedule at all. The motion is a fixed looped replay that cannot track a
  command, so a varying command would be misaligned for 100% of the clip; those clips carry one
  constant command equal to the stride's achieved speed instead.

Kept numpy-only and free of every heavy dependency ON PURPOSE: it is imported from the
``motionbricks`` conda env (which generates the motion) as well as from ``env_isaaclab`` (which
writes the clip), and those two cannot share a process.
"""

from __future__ import annotations

import numpy as np

# Mode weights matching the historical flat distribution: forward over-weighted 3:1 against each of
# the other four. Passing ``vx_range`` alone (terrain) collapses to "forward" regardless.
FLAT_MODE_WEIGHTS: dict[str, float] = {
    "forward": 3.0, "backward": 1.0, "sidestep": 1.0, "turn": 1.0, "diagonal": 1.0,
}
FORWARD_ONLY: dict[str, float] = {"forward": 1.0}

Schedule = list[tuple[float, float, float, float]]


def _seg_command(rng: np.random.Generator, *, vx_range, back_vx_range, side_vy: float,
                 turn_max: float, run_vx_range, run_frac: float, mode_weights: dict[str, float],
                 turn_while_forward_frac: float) -> tuple[float, float, float]:
    """One MOVING segment's ``(vx, vy, wz)``.

    Speeds come from explicit bands, never from a scale times a fraction: the old form could not
    express a floor (which is why the flat family had nothing below 0.45 m/s) and it left a hole
    between the walk and run bands. A mode whose band is degenerate is skipped by giving it zero
    weight, so "terrain has no sidestep" needs no branch.
    """
    modes = [m for m, w in mode_weights.items() if w > 0.0]
    weights = np.array([mode_weights[m] for m in modes], dtype=np.float64)
    mode = str(rng.choice(modes, p=weights / weights.sum()))
    lo, hi = float(vx_range[0]), float(vx_range[1])
    vx = vy = wz = 0.0
    if mode == "forward":
        band = (run_vx_range if (run_vx_range is not None and rng.random() < run_frac) else (lo, hi))
        vx = float(rng.uniform(float(band[0]), float(band[1])))
        if turn_max > 0.0 and rng.random() < turn_while_forward_frac:
            wz = float(rng.uniform(-1.0, 1.0)) * turn_max
    elif mode == "backward":
        b_lo, b_hi = (float(v) for v in back_vx_range)
        vx = -float(rng.uniform(b_lo, b_hi))
    elif mode == "sidestep":
        vy = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.4, 1.0)) * side_vy
    elif mode == "turn":
        vx = float(rng.uniform(lo, lo + 0.7 * (hi - lo)))
        wz = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.5, 1.0)) * turn_max
    elif mode == "diagonal":
        vx = float(rng.uniform(lo, lo + 0.8 * (hi - lo)))
        vy = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.3, 0.7)) * side_vy
    return vx, vy, wz


def sample_command_schedule(
    rng: np.random.Generator,
    *,
    duration: float,
    seg_dur_range,
    vx_range,
    back_vx_range=None,
    run_vx_range=None,
    run_frac: float = 0.4,
    side_vy: float = 0.0,
    turn_max: float = 0.0,
    stand_frac: float = 0.0,
    lead_in_s: float = 0.0,
    turn_while_forward_frac: float = 0.3,
    mode_weights: dict[str, float] | None = None,
) -> Schedule:
    """Sample a schedule totalling EXACTLY ``duration`` seconds.

    Args:
        rng: Seeded generator; the schedule is a pure function of it and these arguments.
        duration: Total schedule length (s).
        seg_dur_range: ``(min, max)`` seconds per moving segment.
        vx_range: Forward band ``(min, max)`` m/s. Also the base band for turn / diagonal.
        back_vx_range: Backward SPEED band (magnitudes). ``None`` disables the backward mode.
        run_vx_range: Optional second forward band, used on ``run_frac`` of forward segments.
            Make it contiguous with ``vx_range`` to cover a wide range with no hole.
        run_frac: Fraction of forward segments drawn from ``run_vx_range``.
        side_vy / turn_max: Lateral and yaw scales. ``0`` disables their modes.
        stand_frac: Probability that a segment is a full stop instead of a move.
        lead_in_s: A standing segment prepended, so a clip starts from a settled stand.
        turn_while_forward_frac: Chance of a yaw command superimposed on a forward segment.
        mode_weights: Relative mode weights; defaults to forward-only when no lateral/yaw/backward
            band is given, else the flat distribution.

    Returns:
        ``[(duration_s, vx, vy, wz), ...]`` summing to ``duration``.

    The FINAL segment is stretched or trimmed so the total is exact. Downstream fits to the clip
    length either way, so an overshoot is generated motion thrown away and an undershoot is
    repeat-last padding; pinning the total makes every clip full-length real motion. The trim is
    bounded by the segment band, so the last segment can land anywhere in ``(0.5, max)`` s -- wider
    than the band itself -- but never zero or negative: the loop exits with at least 0.5 s left.
    """
    if mode_weights is None:
        rich = (back_vx_range is not None) or side_vy > 0.0 or turn_max > 0.0
        mode_weights = dict(FLAT_MODE_WEIGHTS) if rich else dict(FORWARD_ONLY)
    mode_weights = {m: w for m, w in mode_weights.items()
                    if not (m == "backward" and back_vx_range is None)
                    and not (m == "sidestep" and side_vy <= 0.0)
                    and not (m == "turn" and turn_max <= 0.0)
                    and not (m == "diagonal" and side_vy <= 0.0)}
    if not any(w > 0 for w in mode_weights.values()):
        raise ValueError("no command mode is enabled: give at least a forward vx_range")

    duration = float(duration)
    lo_s, hi_s = (float(v) for v in seg_dur_range)
    segs: Schedule = []
    t = 0.0
    if lead_in_s > 0.0:
        segs.append((float(lead_in_s), 0.0, 0.0, 0.0))
        t = float(lead_in_s)
    while t < duration - 0.5:
        if stand_frac > 0.0 and rng.random() < stand_frac:
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            vx, vy, wz = _seg_command(
                rng, vx_range=vx_range, back_vx_range=back_vx_range, side_vy=side_vy,
                turn_max=turn_max, run_vx_range=run_vx_range, run_frac=run_frac,
                mode_weights=mode_weights, turn_while_forward_frac=turn_while_forward_frac)
        dur = float(rng.uniform(lo_s, hi_s))
        segs.append((dur, vx, vy, wz))
        t += dur
    if not segs:                                        # duration <= 0.5 s: one segment covers it
        segs.append((duration, *_seg_command(
            rng, vx_range=vx_range, back_vx_range=back_vx_range, side_vy=side_vy,
            turn_max=turn_max, run_vx_range=run_vx_range, run_frac=run_frac,
            mode_weights=mode_weights, turn_while_forward_frac=turn_while_forward_frac)))
        return segs
    d, vx, vy, wz = segs[-1]
    segs[-1] = (d + (duration - t), vx, vy, wz)
    return segs



def hold_constant(schedule: Schedule, windows, dt: float = 0.02) -> Schedule:
    """Force the command CONSTANT across each ``(t0, t1)`` window, keeping the value it had at ``t0``.

    A terrain clip's approach and exit are stretches of steady walking whose whole contract is that the
    command and the motion agree -- the approach is now an exact looped stride at the commanded speed, so
    a command that CHANGES mid-approach asks for something the single loop cannot express and the clip
    ends up showing the RL one velocity while commanding another.  The interaction is the opposite case:
    the terrain overrides the command there by design, so that is exactly where a change belongs.

    Confining changes to the interaction keeps the command varied within a clip (which is the point of
    sampling one at all) without ever contradicting the segments that must track it.

    Args:
        schedule: the sampled schedule.
        windows: iterable of ``(t0, t1)`` second spans to hold constant.
        dt: resampling period used to rebuild the segment list.

    Returns a schedule of the same total duration.
    """
    total = total_duration(schedule)
    n = max(1, int(round(total / float(dt))))
    grid = rasterize(schedule, n, float(dt)).astype(np.float64)
    for w0, w1 in windows:
        a = max(0, min(n, int(round(float(w0) / float(dt)))))
        b = max(0, min(n, int(round(float(w1) / float(dt)))))
        if b > a:
            grid[a:b] = grid[a]
    out: Schedule = []
    start = 0
    for i in range(1, n + 1):
        if i == n or np.any(np.abs(grid[i] - grid[start]) > 1e-6):
            out.append(((i - start) * float(dt), *(float(v) for v in grid[start])))
            start = i
    # Rounding can leave the total a frame short/long; the last segment absorbs it, as in the sampler.
    drift = total - sum(d for d, *_ in out)
    if out and abs(drift) > 1e-9:
        d, vx, vy, wz = out[-1]
        out[-1] = (d + drift, vx, vy, wz)
    return out


def cmd_at(schedule: Schedule, t_now: float) -> np.ndarray:
    """The ``(vx, vy, wz)`` command in effect at ``t_now`` seconds (saturating at the end)."""
    acc = 0.0
    for dur, vx, vy, wz in schedule:
        acc += float(dur)
        if float(t_now) < acc:
            return np.array([vx, vy, wz], dtype=np.float32)
    _d, vx, vy, wz = schedule[-1]
    return np.array([vx, vy, wz], dtype=np.float32)


def rasterize(schedule: Schedule, n_steps: int, dt: float) -> np.ndarray:
    """Per-step ``[n_steps, 3]`` command, the form a clip's ``target_vel`` needs."""
    return np.stack([cmd_at(schedule, i * float(dt)) for i in range(int(n_steps))], axis=0)


def total_duration(schedule: Schedule) -> float:
    """Sum of the segment durations (s)."""
    return float(sum(float(d) for d, *_ in schedule))


def change_times(schedule: Schedule) -> np.ndarray:
    """Interior segment boundaries -- the times the command CHANGES.

    Used to relax a pose pin through a transition, and to report how much command variety a library
    actually carries.
    """
    if len(schedule) <= 1:
        return np.zeros(0)
    return np.cumsum([float(d) for d, *_ in schedule])[:-1]


def as_json(schedule: Schedule) -> list[list[float]]:
    """JSON-safe form, for storing a schedule in a plan / request file."""
    return [[float(d), float(vx), float(vy), float(wz)] for d, vx, vy, wz in schedule]


def from_json(raw) -> Schedule:
    """Inverse of :func:`as_json`."""
    return [(float(d), float(vx), float(vy), float(wz)) for d, vx, vy, wz in raw]
