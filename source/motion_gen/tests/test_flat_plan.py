# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the flat-clip plan, above all its phase arithmetic.

The design decides everything in pass A so pass B (a different conda env) needs nothing but the plan.
That hinges on one trick: a loop's LENGTH is free, so it can be chosen to land the loop's last frame on
a hand-off phase picked up front. Get that arithmetic wrong and the generated transient is seeded from a
pose the assembled clip never actually reaches -- which would show up only as a seam artefact.
"""

from __future__ import annotations

import numpy as np
import pytest

from motion_gen.motionbricks.flat_plan import (
    FlatClipPlan,
    Loop,
    Transient,
    from_json,
    loop_frames_for,
    sample_plan,
    to_json,
    transient_context_phase,
)

DT = 0.02


def _phase_after(phase0: int, n: int, period: int) -> int:
    """Phase of the LAST of ``n`` frames played from ``phase0`` -- what loop_frames_for must control."""
    return (phase0 + n - 1) % period


@pytest.mark.parametrize("period", [29, 40, 43, 107])
@pytest.mark.parametrize("phase0", [0, 7, 28])
@pytest.mark.parametrize("phase_out", [0, 1, 19])
def test_loop_length_lands_on_the_requested_handoff_phase(period, phase0, phase_out):
    p0, po = phase0 % period, phase_out % period
    n = loop_frames_for(p0, po, period, want=150, min_frames=period)
    assert _phase_after(p0, n, period) == po
    assert n >= period


def test_loop_length_is_the_admissible_one_closest_to_the_request():
    period = 40
    for want in (60, 100, 137, 200):
        n = loop_frames_for(5, 11, period, want=want, min_frames=period)
        assert _phase_after(5, n, period) == 11
        assert abs(n - want) <= period // 2 + 1, (want, n)


def test_loop_length_respects_the_floor():
    """A loop shorter than a period, or than the blends on either side, is not a steady stretch."""
    n = loop_frames_for(0, 5, 40, want=1, min_frames=90)
    assert n >= 90 and _phase_after(0, n, 40) == 5


def test_a_full_cycle_is_never_zero_frames():
    """phase_out == phase0-1 means 'go all the way round', not 'play nothing'."""
    n = loop_frames_for(10, 9, 40, want=40, min_frames=1)
    assert n == 40 and _phase_after(10, n, 40) == 9


def test_context_window_ends_exactly_on_the_handoff_phase():
    step = (1.0 / 30.0) / 0.02
    start = transient_context_phase(phase_out=17, n=4, step=step)
    taps = start + np.arange(4) * step
    assert taps[-1] == pytest.approx(17.0)


def test_json_round_trip():
    plan = FlatClipPlan(name="c0", n_frames=1000, dt=DT, seed=3, pieces=[
        Transient(stride_out=-1, phase_out=0, stride_in=4, phase_in=7, key="c0_t0"),
        Loop(stride=4, phase0=7, phase_out=19, hold_s=1.6),
        Transient(stride_out=4, phase_out=19, stride_in=9, phase_in=2, key="c0_t2"),
        Loop(stride=9, phase0=2, phase_out=5, hold_s=2.4),
    ])
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plans.json")
        to_json([plan], p)
        back = from_json(p)
    assert len(back) == 1
    b = back[0]
    assert b.name == "c0" and b.n_frames == 1000 and b.seed == 3
    assert [type(x).__name__ for x in b.pieces] == ["Transient", "Loop", "Transient", "Loop"]
    assert b.transients[0].stride_out == -1 and b.loops[1].stride == 9


def test_sampled_plan_alternates_and_ends_on_a_loop():
    periods = np.array([40, 43, 31, 55])
    vel = np.zeros((4, 3), np.float32)
    rng = np.random.default_rng(0)
    plan = sample_plan("c", rng, bank_periods=periods, bank_vel=vel, n_frames=1000, dt=DT, seed=0)
    kinds = [p.kind for p in plan.pieces]
    assert kinds[0] == "transient" and kinds[-1] == "loop"
    assert kinds == ["transient", "loop"] * (len(kinds) // 2)
    # Every transient's context phase must be the phase the PREVIOUS loop hands off at.
    for i in range(2, len(plan.pieces), 2):
        t, prev = plan.pieces[i], plan.pieces[i - 1]
        assert t.stride_out == prev.stride and t.phase_out == prev.phase_out
    assert plan.pieces[0].stride_out == -1                      # the opening one has no outgoing loop


def test_sampled_plan_leaves_room_for_its_transients():
    periods = np.array([40, 43, 31, 55])
    vel = np.zeros((4, 3), np.float32)
    for seed in range(8):
        plan = sample_plan("c", np.random.default_rng(seed), bank_periods=periods, bank_vel=vel,
                           n_frames=1000, dt=DT, seed=seed)
        loop_frames = sum(int(round(p.hold_s / DT)) for p in plan.loops)
        assert loop_frames < 1000, (seed, loop_frames)
        assert len(plan.loops) >= 1


def test_sampled_phases_are_in_range():
    periods = np.array([29, 107])
    vel = np.zeros((2, 3), np.float32)
    plan = sample_plan("c", np.random.default_rng(5), bank_periods=periods, bank_vel=vel,
                       n_frames=2000, dt=DT, seed=5)
    for lp in plan.loops:
        per = int(periods[lp.stride])
        assert 0 <= lp.phase0 < per and 0 <= lp.phase_out < per
