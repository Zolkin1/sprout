# Copyright 2026 Zachary Olkin. All rights reserved.

"""Distractor command PROFILES -- the third cross-environment contract, alongside ``raw_schema`` /
``terrain_plan``.

A distractor clip is flat locomotion with a box placed where that motion makes it coherent (walk up and
STOP short of it, steer AROUND it, walk BETWEEN walls) -- the robot never touches the box, so there is no
terrain segment and none of the terrain plan/splice machinery applies. What the two environments must agree
on is only the COMMAND PROFILE: Motion Bricks generates the motion for it (in its own env), and the
assembler fits the box to the result + writes that same profile as the RL-visible ``target_vel``.

A profile is a small JSON-able dict so it survives the trip:

    {"kind": "stop",  "vx": 1.4, "t_stop": 2.1, "ramp_s": 0.6, "secs": 6.0}
    {"kind": "turn",  "vx": 1.2, "wz": -0.55, "secs": 6.0}
    {"kind": "straight", "vx": 1.3, "secs": 6.0}

:func:`profile_cmd` gives the continuous ``t -> (vx, vy, wz)`` the clip machinery wants; :func:`profile_segs`
gives the piecewise-constant ``[(duration, vx, vy, wz)]`` schedule the Motion Bricks driver wants (a stop's
linear ramp is discretized, since the MB command schedule is piecewise constant). Both read the same dict,
so the generated motion and the recorded command cannot drift apart.

Like the other contracts this module is numpy-only with no intra-package imports, so the Motion Bricks env
can load it straight from its source file (``mb_env.load_schema_module``).
"""

from __future__ import annotations

_RAMP_STEPS = 6          # segments a stop's linear decel is discretized into for the MB schedule


def profile_cmd(spec: dict):
    """A profile dict -> continuous command ``cmd(t_seconds) -> (vx, vy, wz)``."""
    kind = str(spec["kind"])
    vx = float(spec.get("vx", 0.0))
    if kind == "straight":
        return lambda _t: (vx, 0.0, 0.0)
    if kind == "turn":
        wz = float(spec.get("wz", 0.0))
        return lambda _t: (vx, 0.0, wz)
    if kind == "stop":
        t_stop, ramp_s = float(spec["t_stop"]), max(1e-6, float(spec["ramp_s"]))
        t0 = t_stop - ramp_s

        def cmd(t: float):
            if t <= t0:
                return (vx, 0.0, 0.0)
            if t >= t_stop:
                return (0.0, 0.0, 0.0)
            return (vx * (1.0 - (t - t0) / ramp_s), 0.0, 0.0)

        return cmd
    raise ValueError(f"unknown distractor profile kind {kind!r}")


def profile_segs(spec: dict) -> list[tuple[float, float, float, float]]:
    """A profile dict -> piecewise-constant ``[(duration_s, vx, vy, wz)]`` for the Motion Bricks driver.

    Durations sum to ``spec["secs"]``. A ``stop`` becomes hold -> ``_RAMP_STEPS`` decel steps -> zero, which
    is the same deceleration the continuous profile describes, sampled at the segment resolution.
    """
    kind = str(spec["kind"])
    secs = float(spec["secs"])
    vx = float(spec.get("vx", 0.0))
    if kind == "straight":
        return [(secs, vx, 0.0, 0.0)]
    if kind == "turn":
        return [(secs, vx, 0.0, float(spec.get("wz", 0.0)))]
    if kind == "stop":
        t_stop, ramp_s = float(spec["t_stop"]), max(1e-6, float(spec["ramp_s"]))
        t0 = max(0.0, t_stop - ramp_s)
        segs: list[tuple[float, float, float, float]] = []
        if t0 > 0.0:
            segs.append((t0, vx, 0.0, 0.0))
        step = min(ramp_s, max(0.0, secs - t0)) / _RAMP_STEPS
        for i in range(_RAMP_STEPS):
            if step <= 0.0:
                break
            frac = (i + 0.5) / _RAMP_STEPS                  # mid-step value of the linear decel
            segs.append((step, vx * (1.0 - frac), 0.0, 0.0))
        tail = secs - sum(d for d, *_ in segs)
        if tail > 0.0:
            segs.append((tail, 0.0, 0.0, 0.0))
        return segs
    raise ValueError(f"unknown distractor profile kind {kind!r}")


def request_key(spec: dict, index: int) -> str:
    """Stable, readable name for a profile's generated motion (used as the ``.npz`` stem)."""
    kind = str(spec["kind"])
    return f"{kind}_{index:04d}"
