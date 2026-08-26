# Copyright 2026 Zachary Olkin. All rights reserved.

"""The flat-clip plan: which strides a clip loops, and which transients Motion Bricks must in-between.

A flat clip is ``[transient][loop][transient][loop]...``.  Steady stretches are EXACT looped replay of
optimized traj-opt strides (velocity exact by construction); Motion Bricks is used only for the
transients between them, with full freedom over position and timing.

This module is the cross-environment contract, mirroring ``raw_schema`` / ``terrain_plan`` /
``command_schedule``: numpy + stdlib only and free of intra-package imports, so the Motion Bricks conda
env can load it by path while ``motion_gen`` imports it normally.  Three passes read it:

* **A** (``motion_gen``) samples the plan -- it needs the stride bank to know periods and velocities;
* **B** (Motion Bricks env) generates one segment per transient, from the plan's phases alone;
* **C** (``motion_gen``) replays the loops, splices in the generated transients, writes the clip.

**Everything is decided in pass A, nothing is discovered later.**  Both phases of every transient are
inputs: ``phase_out`` is where we leave the outgoing loop, ``phase_in`` where we enter the incoming one.
That is only possible because a loop's LENGTH is ours to choose -- pass C picks a frame count that lands
the loop exactly on the next ``phase_out`` (:func:`loop_frames_for`), absorbing the requested hold
duration to within one gait period.  The alternative -- letting the phase fall out of whatever length
the hold happened to be -- would make the context window unknowable in pass A, and pass B needs it.

``phase_in`` is SAMPLED rather than fixed at 0: frame 0 is the traj-opt closure frame, so always
entering there would give every loop in the library the same hand-off pose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Loop:
    """An exact looped replay of bank stride ``stride``, entered at integer bank frame ``phase0``."""

    stride: int
    phase0: int
    phase_out: int                 # the phase its LAST frame must land on (the next transient's context)
    hold_s: float                  # requested duration; pass C rounds to land on phase_out
    # The CONTINUOUS commanded velocity this loop serves, which is NOT the stride's own achieved
    # velocity: the stride is the nearest one the bank has, so the two differ by the snap error.  The
    # clip records THIS as target_vel.  Recording the achieved velocity instead makes tracking error
    # zero by construction but collapses the command distribution onto the bank's ~0.2 m/s lattice --
    # measured on g1_mb_v6, the flat family had 73 distinct commanded vx with a p95 gap of 0.2000, so
    # the teacher never saw a command between grid points.
    cmd_vel: tuple = (0.0, 0.0, 0.0)
    kind: str = "loop"


@dataclass
class Transient:
    """A Motion-Bricks in-betweening from one loop's phase to the next one's.

    ``stride_out < 0`` marks a transient with no outgoing loop to be seeded from, which starts Motion
    Bricks from its own standing reset instead.  Plans no longer produce one: a clip OPENS on a loop, so
    every transient has a real outgoing gait to be seeded from and starts at the speed it is leaving
    rather than from a standing start.  The branch is kept because it is the only way to begin a clip
    from rest if that is ever wanted deliberately.
    """

    stride_out: int
    phase_out: int
    stride_in: int
    phase_in: int
    key: str                       # segment filename stem; also how pass C finds the generated motion
    cmd_vel: tuple = (0.0, 0.0, 0.0)   # the command it is heading TO (see Loop.cmd_vel)
    kind: str = "transient"


@dataclass
class FlatClipPlan:
    """One clip: an ordered piece list plus the total length it must hit exactly."""

    name: str
    n_frames: int                  # exact clip length in control frames
    dt: float
    seed: int
    pieces: list = field(default_factory=list)     # Loop | Transient, alternating, ending on a Loop

    @property
    def transients(self) -> list:
        return [p for p in self.pieces if isinstance(p, Transient)]

    @property
    def loops(self) -> list:
        return [p for p in self.pieces if isinstance(p, Loop)]


def loop_frames_for(phase0: int, phase_out: int, period: int, want: int, *, min_frames: int) -> int:
    """Frame count for a loop entered at ``phase0`` whose LAST frame must sit on ``phase_out``.

    Playing ``n`` frames from ``phase0`` ends on phase ``(phase0 + n - 1) % period``, so ``n`` is pinned
    modulo the period: ``n = (phase_out - phase0 + 1) mod period`` plus any number of whole periods.  We
    take the admissible ``n`` closest to ``want`` that is at least ``min_frames`` -- so the requested hold
    is honoured to within one gait cycle while the hand-off phase is honoured exactly.

    ``min_frames`` exists because a loop shorter than a period (or than the seam blends on either side)
    is not a steady stretch in any measurable sense.
    """
    period = max(1, int(period))
    base = (int(phase_out) - int(phase0) + 1) % period
    if base == 0:
        base = period                                  # a full cycle, not zero frames
    k = max(0, round((int(want) - base) / period))
    n = base + k * period
    while n < int(min_frames):
        n += period
    return int(n)


def transient_context_phase(phase_out: int, n: int, step: float) -> float:
    """Bank-frame position the outgoing loop's ``n``-frame context window STARTS at.

    The window is Motion Bricks' 4 context frames, spaced ``step = mb_dt / bank_dt`` bank frames apart
    and ENDING on ``phase_out`` -- the exact frame the loop hands off at, so the generated motion
    continues the gait rather than restarting it.
    """
    return float(phase_out) - (int(n) - 1) * float(step)


def to_json(plans: list, path: str) -> None:
    """Write a list of :class:`FlatClipPlan` (piece types tagged by their ``kind`` field)."""
    out = [{**asdict(p), "pieces": [asdict(x) for x in p.pieces]} for p in plans]
    with open(path, "w") as f:
        json.dump({"version": 1, "plans": out}, f, indent=1)


def from_json(path: str) -> list:
    """Read back what :func:`to_json` wrote."""
    with open(path) as f:
        d = json.load(f)
    if int(d.get("version", 0)) != 1:
        raise ValueError(f"{path}: unsupported flat-plan version {d.get('version')}")
    plans = []
    for p in d["plans"]:
        pieces = []
        for x in p["pieces"]:
            k = x.pop("kind")
            if "cmd_vel" in x:
                x["cmd_vel"] = tuple(float(c) for c in x["cmd_vel"])
            pieces.append(Loop(**x, kind=k) if k == "loop" else Transient(**x, kind=k))
        plans.append(FlatClipPlan(name=p["name"], n_frames=int(p["n_frames"]), dt=float(p["dt"]),
                                  seed=int(p["seed"]), pieces=pieces))
    return plans


def sample_plan(name: str, rng, *, bank_periods, bank_vel, schedule, n_frames: int, dt: float,
                seed: int, blend_n: int = 12) -> FlatClipPlan:
    """Build one clip plan from a CONTINUOUS command schedule: a loop per segment, in-betweenings between.

    The clip OPENS ON A LOOP, so it starts at a steady commanded speed.  It used to open on a transient
    with ``stride_out = -1``, i.e. Motion Bricks' own standing reset, which made every one of the 3000
    flat clips begin from rest and spend its first second initiating gait.  A standing start is now just
    a plan whose first segment happens to command zero -- the bank has true zero-velocity strides -- so
    it costs a leading loop like any other speed instead of a special case.

    It still ENDS on a loop: a clip that stops mid-in-betweening hands the tracker and the flow-matching
    model a truncated motion.

    Args:
        bank_periods: ``[N]`` period (frames) per stride.
        bank_vel: ``[N,3]`` achieved velocity per stride; only used to snap a command to a stride.
        schedule: ``[(duration_s, vx, vy, wz)]`` -- the continuous commands the clip serves.  Sampling
            these rather than picking strides uniformly is what keeps the recorded command off the
            bank's velocity lattice (see :attr:`Loop.cmd_vel`).
        rng: numpy Generator, used only for the hand-off phases.
        blend_n: seam blend length, which sets the floor on a usable loop.
    """
    min_hold = max(int(2 * blend_n + 1), 1)
    pieces: list = []
    used = 0
    prev_stride, prev_phase = -1, 0
    for j, (dur, vx, vy, wz) in enumerate(schedule):
        cmd = (float(vx), float(vy), float(wz))
        s_in = nearest_stride(bank_vel, cmd)
        period = int(bank_periods[s_in])
        p_in = int(rng.integers(0, max(1, period)))
        p_out = int(rng.integers(0, max(1, period)))
        n_loop = loop_frames_for(p_in, p_out, period, int(round(float(dur) / dt)),
                                 min_frames=max(min_hold, period))
        # A transient's own length is Motion Bricks' choice and is not known yet, so reserve its WORST
        # case for EVERY transient, not just the next one.  Over-reserving is harmless -- pass C hands
        # the slack to the final loop, i.e. more steady motion -- whereas under-reserving overruns.
        cost = n_loop + (_MAX_TRANSIENT_FRAMES if pieces else 0)
        if used + cost > n_frames and pieces:
            break
        if pieces:                                    # every loop but the first is entered via one
            pieces.append(Transient(stride_out=prev_stride, phase_out=prev_phase, stride_in=s_in,
                                    phase_in=p_in, key=f"{name}_t{len(pieces)}", cmd_vel=cmd))
        pieces.append(Loop(stride=s_in, phase0=p_in, phase_out=p_out, hold_s=n_loop * dt, cmd_vel=cmd))
        used += cost
        prev_stride, prev_phase = s_in, p_out
    return FlatClipPlan(name=name, n_frames=int(n_frames), dt=float(dt), seed=int(seed), pieces=pieces)


def nearest_stride(bank_vel, v) -> int:
    """Index of the bank stride whose achieved velocity is closest to ``v``, per-channel normalized.

    An unweighted Euclidean metric over ``(vx, vy, wz)`` trades 1 rad/s of yaw against 1 m/s of speed,
    which picks visibly wrong strides once the matched stride IS the motion.  Normalizing by each
    channel's spread across the bank makes the three commensurate.
    """
    import numpy as _np
    bv = _np.asarray(bank_vel, _np.float64)
    scale = _np.ptp(bv, axis=0)
    scale[scale < 1e-6] = 1.0
    return int(_np.argmin(_np.linalg.norm((bv - _np.asarray(v, _np.float64)) / scale, axis=1)))


def plan_from_holds(name: str, holds, rng, *, bank_periods, bank_vel, n_frames: int, dt: float,
                    seed: int, blend_n: int = 12) -> FlatClipPlan:
    """A plan whose steady phases are PRESCRIBED -- one loop per hold, in-betweenings between them.

    :func:`sample_plan` draws its own velocities, which is right for the flat family.  A distractor's
    velocities instead come from its command profile (the box is fitted to the motion that profile
    produces, so the two cannot be chosen independently), so the caller supplies
    ``holds = [(duration_s, (vx, vy, wz))]`` and this picks the nearest bank stride for each.

    Args:
        holds: ordered steady phases; a zero-velocity hold resolves to the bank's standing stride.
        rng: numpy Generator, used only for the hand-off phases.
    """
    min_hold = max(int(2 * blend_n + 1), 1)
    pieces: list = []
    prev_stride, prev_phase = -1, 0
    for j, (hold_s, v) in enumerate(holds):
        cmd = (float(v[0]), float(v[1]), float(v[2]))
        s_in = nearest_stride(bank_vel, cmd)
        period = int(bank_periods[s_in])
        p_in = int(rng.integers(0, max(1, period)))
        p_out = int(rng.integers(0, max(1, period)))
        if pieces:                                    # opens on a loop, like the flat plans
            pieces.append(Transient(stride_out=prev_stride, phase_out=prev_phase, stride_in=s_in,
                                    phase_in=p_in, key=f"{name}_t{j}", cmd_vel=cmd))
        n_loop = loop_frames_for(p_in, p_out, period, int(round(float(hold_s) / dt)),
                                 min_frames=max(min_hold, period))
        pieces.append(Loop(stride=s_in, phase0=p_in, phase_out=p_out, hold_s=n_loop * dt, cmd_vel=cmd))
        prev_stride, prev_phase = s_in, p_out
    return FlatClipPlan(name=name, n_frames=int(n_frames), dt=float(dt), seed=int(seed), pieces=pieces)


# Motion Bricks predicts at most 16 tokens of 4 frames at 30 fps = 2.13 s, which is 107 frames at the
# 50 Hz control rate.  Used only to keep a sampled plan from over-filling its clip.
_MAX_TRANSIENT_FRAMES = 107
