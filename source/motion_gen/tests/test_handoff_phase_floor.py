# Copyright 2026 Zachary Olkin. All rights reserved.

"""A loop may not hand off at a gait phase whose context window wraps the stride's closure seam.

This is the root cause of every root-speed blow-up the library had, and it is not what it looked
like. It looked like Motion Bricks occasionally diverging on a hard velocity change; it is actually
the model being handed a HISTORY that does not describe the motion it is continuing.

``transient_context_phase`` walks back ``(nfpt - 1) * step`` bank frames from the hand-off phase to
find the context window's start -- 4 frames at the model's 30 fps against the bank's 50 Hz, so
``step = 5/3`` and the window spans exactly 5.0 frames. Below that the start phase goes negative, the
window wraps the closure seam, and the sampled root barely advances or advances backwards. Motion
Bricks then has a near-stationary history and a target at speed, and reconciles them by sprinting.

Measured over the quick build's 46 flat transients, the separation was total: 5 of the 7 transients
with ``phase_out < 5`` peaked over 6 m/s (p50 9.80, max 19.51) and 0 of the 39 above it did (p50
1.95, max 4.39). |dv| correlated +0.05 with the peak; ``phase_out / period`` correlated -0.42.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "source" / "motion_gen" / "motion_gen").is_dir():
        _REPO = _p
        break
else:                                                          # pragma: no cover - layout guard
    raise RuntimeError("could not locate the checkout above this test")
sys.path.insert(0, str(_REPO / "source" / "motion_gen"))

from motion_gen.motionbricks import flat_plan as fp  # noqa: E402

_NFPT, _STEP = 4, (1.0 / 30.0) / 0.02          # the shipped Motion Bricks / bank rates


def test_the_floor_is_the_context_window_span_and_not_a_guess():
    """``MIN_HANDOFF_PHASE`` must equal the frames the context window reaches back over."""
    assert fp.MIN_HANDOFF_PHASE == int(np.ceil((_NFPT - 1) * _STEP)) == 5


def test_a_phase_below_the_floor_really_does_wrap():
    """The mechanism, checked directly: the context start phase goes negative."""
    for below in range(fp.MIN_HANDOFF_PHASE):
        assert fp.transient_context_phase(below, _NFPT, _STEP) < 0
    for ok in range(fp.MIN_HANDOFF_PHASE, fp.MIN_HANDOFF_PHASE + 6):
        assert fp.transient_context_phase(ok, _NFPT, _STEP) >= 0


def test_draw_handoff_phase_never_returns_a_wrapping_phase():
    rng = np.random.default_rng(0)
    for period in (28, 46, 64, 88, 106):
        got = [fp.draw_handoff_phase(rng, period) for _ in range(2000)]
        assert min(got) >= fp.MIN_HANDOFF_PHASE, (period, min(got))
        assert max(got) == period - 1, "the rest of the circle must stay reachable"


def test_a_period_too_short_for_a_safe_band_falls_back_rather_than_hanging():
    """No stride in the shipped bank is this short (the minimum is 28), but a caller should not get
    an empty range if one ever is."""
    rng = np.random.default_rng(0)
    for period in (1, 3, 5, 6):
        got = [fp.draw_handoff_phase(rng, period) for _ in range(50)]
        assert all(0 <= g < period for g in got), (period, got)


def _bank(n=6):
    """A synthetic bank: distinct periods and speeds, enough for a plan to alternate."""
    periods = np.array([28, 36, 46, 64, 88, 106], np.int64)[:n]
    vel = np.stack([np.linspace(0.2, 3.0, n), np.zeros(n), np.zeros(n)], axis=1)
    return periods, vel


@pytest.mark.parametrize("planner", ["sample_plan", "plan_from_holds", "plan_from_distance_holds"])
def test_every_planner_honours_the_floor(planner):
    """All three author hand-off phases, so a floor in one of them would leave the others broken."""
    periods, vel = _bank()
    rng = np.random.default_rng(4)
    for _ in range(40):
        if planner == "sample_plan":
            sched = [(2.0, float(v), 0.0, 0.0) for v in (0.4, 1.6, 2.4, 0.8)]
            plan = fp.sample_plan("t", rng, bank_periods=periods, bank_vel=vel, schedule=sched,
                                  n_frames=2000, dt=0.02, seed=0)
        elif planner == "plan_from_holds":
            holds = [(2.0, (v, 0.0, 0.0)) for v in (0.4, 1.6, 2.4)]
            plan = fp.plan_from_holds("t", holds, rng, bank_periods=periods, bank_vel=vel,
                                      n_frames=2000, dt=0.02, seed=0)
        else:
            holds = [(3.0, (v, 0.0, 0.0)) for v in (0.4, 1.6, 2.4)]
            plan = fp.plan_from_distance_holds("t", holds, rng, bank_periods=periods, bank_vel=vel,
                                               dt=0.02, seed=0)
        for lp in plan.loops:
            assert lp.phase_out >= fp.MIN_HANDOFF_PHASE, (planner, lp.phase_out)
        # and every transient is seeded from a loop's hand-off phase, so it inherits the floor
        for tr in plan.transients:
            assert tr.phase_out >= fp.MIN_HANDOFF_PHASE, (planner, tr.phase_out)


def test_an_explicitly_PINNED_hand_off_phase_is_still_honoured():
    """The terrain path pins the phase its boundary transient was seeded from; a floor must not
    silently move it, or pass A and pass B stop agreeing."""
    periods, vel = _bank()
    plan = fp.plan_from_distance_holds(
        "t", [(3.0, (0.4, 0.0, 0.0)), (3.0, (1.6, 0.0, 0.0))], np.random.default_rng(0),
        bank_periods=periods, bank_vel=vel, dt=0.02, seed=0, phase_out_last=2)
    assert plan.loops[-1].phase_out == 2, "an explicit pin must win over the floor"
