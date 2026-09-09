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

# NOTE: ``hold_constant`` used to live here -- it forced a schedule constant across named TIME
# windows, and existed for exactly one caller: the terrain path, which laid its command on an
# absolute clip timeline and then had to pin it flat across the approach and the exit. That timeline
# was the bug (pass A reserved a fixed 12 s for a traversal that is really 2-6 s), and the terrain
# command is now a list of distance-indexed HOLDS laid onto the spliced piece boundaries, so there
# is no window to hold anything across. Deleted rather than kept: it only makes sense against a
# representation that no longer exists.

from __future__ import annotations

import numpy as np

# Mode weights matching the historical flat distribution: forward over-weighted 3:1 against each of
# the other four. Passing ``vx_range`` alone (terrain) collapses to "forward" regardless.
Schedule = list[tuple[float, float, float, float]]


class StratifiedSpeeds:
    """A queue of forward-speed draws whose union tiles ``vx_range`` evenly at FINITE sample size.

    The distribution being sampled is uniform either way -- a draw is uniform WITHIN its stratum, so
    nothing here quantizes a command and every value is a distinct real. Stratification only fixes how
    a finite library REALISES that uniform: with i.i.d. draws a 3000-clip flat family still leaves
    visible ripples across 23 bins, and the whole point of this step is a flat empirical marginal.

    It is the same device ``_clip_speeds`` already uses for stairs ("evenly spaced ... so a tile's
    clips tile the speed axis"), applied to the flat family's forward command.

    The queue is shared across the whole flat batch rather than per clip: a clip has ~4 moving
    segments and there are 23 strata, so a per-clip queue would be mostly refills and no better than
    i.i.d. Over a batch it is ~480 draws per stratum.
    """

    def __init__(self, rng, vx_range, width: float):
        lo, hi = float(min(vx_range)), float(max(vx_range))
        n = max(1, int(np.ceil((hi - lo) / max(float(width), 1e-6))))
        self._edges = np.linspace(lo, hi, n + 1)
        self._rng = rng
        self._q: list[int] = []

    @property
    def n_strata(self) -> int:
        return len(self._edges) - 1

    def _refill(self) -> None:
        order = np.arange(self.n_strata)
        self._rng.shuffle(order)
        self._q = [int(i) for i in order]

    def __call__(self) -> float:
        if not self._q:
            self._refill()
        i = self._q.pop()
        return float(self._rng.uniform(self._edges[i], self._edges[i + 1]))


def _seg_command(rng: np.random.Generator, *, vx_range, vy_range, wz_range,
                 mass_stand: float, mass_sidestep: float, vx_draw=None) -> tuple:
    """One segment's ``(vx, vy, wz)``, drawn CONTINUOUSLY over the driver's whole envelope.

    **The support is what a driver or high-level policy will ever send, not what the solved library
    can serve.** The motion is then whichever solved stride is nearest (``flat_plan.nearest_stride``)
    and the command recorded is the continuous value, never the stride's. A generator only ever
    commanded speeds it had motions for cannot learn to SELECT; drive it and it needs exactly those
    velocities, which no real driver produces.

    This replaced a mode/band sampler whose structure was visible in the output as cliffs, because
    each mode drew its forward speed from a DIFFERENT sub-band:

    * ``turn`` drew ``vx = U(lo, lo + 0.7*(hi-lo))`` and ``diagonal`` ``U(lo, lo + 0.8*(hi-lo))``,
      putting hard edges at 0.73 and 0.82 m/s and confining every turn and every diagonal to walk
      speed -- so ``jog_turn``, ``run_turn`` and ``run_forward x vy`` were solved and never commanded;
    * ``run_frac = 0.4`` mixed a 0.9-wide walk band at density 0.67/(m/s) against a 2.0-wide run band
      at 0.20, a 3.3x step at exactly vx = 1.0;
    * ``sidestep`` set ``vx = 0``, so of the 21.3% of flat time at zero only 8.5% was a true stand and
      the other 12.8% was a mass nobody chose.

    Measured, the marginal came out 10.3 : 1 max/min over 0.1 m/s bins with 3.0-3.6 entirely empty.

    So: draw the SUPPORT by declared mass, then the speed continuously over ONE range, then the
    channel. ``forward`` / ``turn`` / ``diagonal`` stop being speed-selectors and become channel
    selectors, which is the whole fix.

    At most ONE of ``vy`` / ``wz`` is non-zero -- the user's constraint, and it costs nothing: zero of
    the solved library's 233 cells carry both.
    """
    r = float(rng.random())
    if r < mass_stand:
        return 0.0, 0.0, 0.0                                  # an exact stop
    if r < mass_stand + mass_sidestep:
        return 0.0, float(rng.uniform(*vy_range)), 0.0        # pure lateral, the vx = 0 lateral case
    vx = float(vx_draw()) if vx_draw is not None else float(rng.uniform(*vx_range))
    pick = int(rng.integers(0, 3))                            # straight / lateral / yaw
    if pick == 1:
        return vx, float(rng.uniform(*vy_range)), 0.0
    if pick == 2:
        return vx, 0.0, float(rng.uniform(*wz_range))
    return vx, 0.0, 0.0


def sample_command_schedule(
    rng: np.random.Generator,
    *,
    duration: float,
    seg_dur_range,
    vx_range,
    vy_range=(0.0, 0.0),
    wz_range=(0.0, 0.0),
    mass_stand: float = 0.0,
    mass_sidestep: float = 0.0,
    lead_in_s: float = 0.0,
    vx_draw=None,
) -> Schedule:
    """Sample a schedule totalling EXACTLY ``duration`` seconds.

    Args:
        rng: Seeded generator; the schedule is a pure function of it and these arguments.
        duration: Total schedule length (s).
        seg_dur_range: ``(min, max)`` seconds per segment.
        vx_range: The forward envelope, ONE continuous range -- what a driver will ever ask for, not
            what the solved library can serve.
        vy_range / wz_range: The lateral and yaw envelopes. ``(0, 0)`` disables that channel, which is
            how the terrain families stay forward-only without a branch.
        mass_stand: Declared probability of an exact stop (``vx == vy == wz == 0``).
        mass_sidestep: Declared probability of a pure lateral segment at ``vx = 0``. Declared, because
            it used to be an EMERGENT 12.8% of flat time -- the old ``sidestep`` mode set ``vx = 0``
            as a side effect, so nobody had chosen how much mass sat at zero.
        lead_in_s: A standing segment prepended, so a clip starts from a settled stand.
        vx_draw: Optional zero-arg callable returning one forward speed, for variance reduction (see
            :class:`StratifiedSpeeds`). It must sample the same distribution as ``vx_range``; it
            changes the REALISATION, never the target.

    Returns:
        ``[(duration_s, vx, vy, wz), ...]`` summing to ``duration``.

    The FINAL segment is stretched or trimmed so the total is exact. Downstream fits to the clip
    length either way, so an overshoot is generated motion thrown away and an undershoot is
    repeat-last padding; pinning the total makes every clip full-length real motion. The trim is
    bounded by the segment band, so the last segment can land anywhere in ``(0.5, max)`` s -- wider
    than the band itself -- but never zero or negative: the loop exits with at least 0.5 s left.
    """
    if float(max(vx_range)) <= float(min(vx_range)) and not (mass_stand > 0 or mass_sidestep > 0):
        raise ValueError("vx_range is degenerate and no stand / sidestep mass is declared")

    kw = dict(vx_range=vx_range, vy_range=vy_range, wz_range=wz_range,
              mass_stand=float(mass_stand), mass_sidestep=float(mass_sidestep), vx_draw=vx_draw)
    duration = float(duration)
    lo_s, hi_s = (float(v) for v in seg_dur_range)
    segs: Schedule = []
    t = 0.0
    if lead_in_s > 0.0:
        segs.append((float(lead_in_s), 0.0, 0.0, 0.0))
        t = float(lead_in_s)
    while t < duration - 0.5:
        vx, vy, wz = _seg_command(rng, **kw)
        dur = float(rng.uniform(lo_s, hi_s))
        segs.append((dur, vx, vy, wz))
        t += dur
    if not segs:                                        # duration <= 0.5 s: one segment covers it
        return [(duration, *_seg_command(rng, **kw))]
    drift = duration - sum(d for d, *_ in segs)
    d0, vx, vy, wz = segs[-1]
    segs[-1] = (max(0.5, d0 + drift), vx, vy, wz)
    return segs


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


def terrain_interaction_schedule(rng, cell: tuple, band: tuple, ref_speed: float, n_seg: int,
                          sign: float, vy_range=(0.0, 0.0), wz_range=(0.0, 0.0)) -> list:
    """``[(fraction_of_replay, vx, vy, wz)]`` -- the command while the terrain traversal plays.

    The motion is a single fixed replay and nothing here changes it; this is ``target_vel`` only. What
    it buys is the one thing a constant command cannot: the generator sees the SAME motion asked for
    at several commands, which is what teaches it that a command selects a reference rather than
    describing one.

    **vx is constrained; vy and wz are not, and that asymmetry is the whole design.**

    ``vx`` is drawn from the intersection of the played reference's VORONOI CELL
    (:func:`terrain_place.terrain_speed_cell`) with the family's own band. Outside the cell the command
    would point at a DIFFERENT solved motion, which teaches the opposite of the intended lesson; the
    band keeps a stair clip from being commanded faster than any stair clip is ever generated.

    ``vy`` and ``wz`` are free, because on a traversal they are unconstrained BY CONSTRUCTION: the
    replay is world-pinned (``anchor_mask == 2``), so a lateral or yaw command cannot move the robot
    off the tile no matter what it says. Before this they were identically zero on every terrain clip
    -- ``tvel[:, 1:]`` was never written -- so the generator had never once been asked to turn while a
    staircase was under it, and could not have learned that the answer is to keep climbing. It has to
    see them to learn to ignore them.

    The selector compares FORWARD speed only, so vy and wz cannot move which reference is nearest and
    need no cell. And when a diagonal or turning stair gait is eventually solved, it slots in here
    without changing this contract: it arrives as another candidate reference, ``terrain_speed_cell``
    starts splitting on it, and the command that used to be ignorable starts selecting.

    At most ONE of ``vy`` / ``wz`` is non-zero per segment -- the same rule the flat sampler follows,
    and it matches the solved library exactly (zero of its cells carry both).

    Segment boundaries are drawn uniformly over the replay: the motion is identical throughout, so no
    frame is a better place for a change than any other, and a fixed fraction would be a landmark.
    """
    lo = max(float(cell[0]), float(min(band)))
    hi = min(float(cell[1]), float(max(band)))
    n = max(1, int(n_seg))
    degenerate = not (hi > lo)

    def _lat():
        """At most one of (vy, wz), drawn from its own range; either may be zero."""
        pick = rng.integers(0, 3)                     # 0 = straight, 1 = lateral, 2 = yaw
        if pick == 1 and max(map(abs, vy_range)) > 0:
            return float(rng.uniform(*vy_range)), 0.0
        if pick == 2 and max(map(abs, wz_range)) > 0:
            return 0.0, float(rng.uniform(*wz_range))
        return 0.0, 0.0

    cuts = sorted(float(x) for x in rng.uniform(0.0, 1.0, size=n - 1)) if n > 1 else []
    out, prev = [], 0.0
    for c in list(cuts) + [1.0]:
        vx = (float(sign) * abs(float(ref_speed)) if degenerate
              else float(sign) * float(rng.uniform(lo, hi)))
        vy, wz = _lat()
        out.append((float(c) - prev, vx, vy, wz))
        prev = float(c)
    out = [r for r in out if r[0] > 1e-6]
    return out or [(1.0, float(sign) * abs(float(ref_speed)), 0.0, 0.0)]
