# Copyright 2026 Zachary Olkin. All rights reserved.

"""A terrain clip's lead-up is ordinary flat machinery, and its command budget has to fund a LOOP.

Step 8's structural claim is that a terrain clip is not a special case: its approach and its exit are
`flat_plan`'s own loop / in-betweening alternation, budgeted in metres instead of seconds, with one
extra transient at each end that targets the terrain instead of a stride.

What is pinned here is the part that was found by measurement rather than by design -- the budget. A
hold has to pay for its incoming in-betweening AND for a whole gait cycle before it buys any steady
motion at all, because ``loop_frames_for`` will not emit less than one period and ``_loop_piece``
refuses to round a sub-cycle loop up to a whole one. Getting that wrong does not fail loudly: it
produces a lead-up made entirely of in-betweenings, which is exactly the motion the exact-loop
pipeline exists to avoid.
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
sys.path.insert(0, str(_REPO / "scripts" / "rsl_rl"))

from motion_gen.motionbricks import flat_plan as fp  # noqa: E402


# A small synthetic bank: three strides at 0.2 / 1.0 / 3.0 m/s with realistic periods.
_BANK_VEL = np.array([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], np.float64)
_BANK_PERIODS = np.array([60, 45, 30], np.int64)
_DT = 0.02


# ------------------------------------------------------------------ transient_travel_bound
def test_the_transient_reserve_is_not_a_constant():
    """A flat scalar reserve is wrong by an order of magnitude across the band.

    That is the whole reason the bound is computed per hold: 0.2 m/s and 3.5 m/s in-betweenings do
    not cover remotely the same distance, and reserving one number for both either starves the slow
    end or forbids variation at the fast one.
    """
    slow = fp.transient_travel_bound((0.2, 0, 0), (0.4, 0, 0), a_max=2.0, floor_s=0.35, slack=1.3)
    fast = fp.transient_travel_bound((3.5, 0, 0), (2.0, 0, 0), a_max=2.0, floor_s=0.35, slack=1.3)
    assert fast > 5 * slow, (slow, fast)
    # 0.2 -> 0.4 m/s: the token FLOOR binds (0.533 s) at a 0.3 m/s mean -> 0.16 m.
    # 3.5 -> 2.0 m/s: |dv|/a_max binds (0.75 s x 1.3 slack) at a 2.75 m/s mean -> 2.68 m.
    assert slow == pytest.approx(0.16, abs=0.01), slow
    assert fast == pytest.approx(2.68, abs=0.01), fast


def test_the_transient_bound_respects_the_models_own_horizon():
    """Clipped to Motion Bricks' 6..16 tokens, because that is what actually limits it."""
    tiny = fp.transient_travel_bound((1.0, 0, 0), (1.0, 0, 0), a_max=2.0, floor_s=0.0, slack=1.0)
    huge = fp.transient_travel_bound((3.6, 0, 0), (-1.0, 0, 0), a_max=0.01, floor_s=0.0, slack=99.0)
    assert tiny == pytest.approx(fp._KEPT_S_MIN * 1.0, rel=1e-9)
    assert huge == pytest.approx(fp._KEPT_S_MAX * 0.5 * (3.6 + 1.0), rel=1e-9)


# ------------------------------------------------------------------ plan_from_distance_holds
def _plan(holds, seed=0, **kw):
    return fp.plan_from_distance_holds(
        "t", holds, np.random.default_rng(seed), bank_periods=_BANK_PERIODS, bank_vel=_BANK_VEL,
        dt=_DT, seed=seed, **kw)


def test_a_distance_plan_alternates_and_ends_on_loops():
    plan = _plan([(4.0, (1.0, 0, 0)), (5.0, (3.0, 0, 0)), (4.0, (0.2, 0, 0))])
    kinds = [p.kind for p in plan.pieces]
    assert kinds == ["loop", "transient", "loop", "transient", "loop"]
    # Every transient hands off from the loop before it to the loop after it -- if these drift the
    # generated motion starts at a gait phase the clip does not contain.
    loops = [p for p in plan.pieces if p.kind == "loop"]
    trs = [p for p in plan.pieces if p.kind == "transient"]
    for j, tr in enumerate(trs):
        assert (tr.stride_out, tr.phase_out) == (loops[j].stride, loops[j].phase_out)
        assert (tr.stride_in, tr.phase_in) == (loops[j + 1].stride, loops[j + 1].phase0)


def test_the_distance_is_the_statement_and_the_duration_the_consequence():
    """``hold_m`` is what was asked for; ``hold_s`` is derived at the SELECTED stride's own speed."""
    plan = _plan([(6.0, (3.0, 0, 0))])
    loop = plan.pieces[0]
    assert loop.hold_m == pytest.approx(6.0)
    # 6 m at the nearest stride's 3.0 m/s is ~2 s, honoured to within one gait cycle.
    assert abs(loop.hold_s - 2.0) <= _BANK_PERIODS[2] * _DT


def test_both_ends_can_be_pinned_because_the_terrain_decides_them():
    """The approach's last hand-off phase and the exit's first entry phase are not free choices."""
    plan = _plan([(4.0, (1.0, 0, 0)), (4.0, (1.0, 0, 0))], phase_in0=7, phase_out_last=11)
    loops = [p for p in plan.pieces if p.kind == "loop"]
    assert loops[0].phase0 == 7 and loops[-1].phase_out == 11


def test_an_empty_hold_list_is_refused():
    with pytest.raises(ValueError):
        _plan([])


# ------------------------------------------------------------------ the budget
def test_a_hold_must_fund_a_whole_gait_cycle_before_it_buys_any_steady_motion():
    """``loop_frames_for`` will not emit fewer than one period, so a shorter budget buys a WHOLE one.

    Measured: at 3.0 m/s one cycle of the matching stride is 0.6 s = 1.8 m, so a 1.5 m ``min_hold_m``
    is not a floor at all there -- and a clip whose holds were sized by it lost every exact loop on
    its lead-up. Pinned as the inequality rather than as a number, since the periods are the bank's.
    """
    import mb_plan_library as pl

    args = type("A", (), {"dt": _DT, "approach_a_max": 2.0, "approach_min_transient_s": 0.35,
                          "approach_hop_slack": 1.3})()
    bank = type("B", (), {"achieved_vel": _BANK_VEL, "period_frames": _BANK_PERIODS})()
    for cmd in (0.2, 1.0, 3.0):
        cyc = pl._cycle_travel(bank, cmd, _DT)
        mins = pl._hold_minimums([cmd], min_hold=0.0, args=args, bank=bank)
        assert mins[0] >= cyc - 1e-9, (cmd, cyc, mins)
    # ...and the fast end really is above a 1.5 m flat floor, which is why the flat floor failed.
    assert pl._cycle_travel(bank, 3.0, _DT) > 1.5


def test_the_exit_reserves_its_terrain_transient_and_the_approach_does_not():
    """Asymmetric BY CONSTRUCTION, and the asymmetry cost a clip its loops before it was found.

    An approach's first hold opens the clip, so nothing precedes it. An exit's first hold is entered
    from the terrain, via the exit in-betweening -- budget it like an approach's and the transient
    eats the whole hold. (The approach's mirror of this is the approach zone, added to its LAST hold.)
    """
    import mb_plan_library as pl

    args = type("A", (), {"dt": _DT, "approach_a_max": 2.0, "approach_min_transient_s": 0.35,
                          "approach_hop_slack": 1.3})()
    bank = type("B", (), {"achieved_vel": _BANK_VEL, "period_frames": _BANK_PERIODS})()
    cmds = [1.0, 3.0]
    approach = pl._hold_minimums(cmds, 1.5, args, bank, lead_in_from=None)
    exit_ = pl._hold_minimums(cmds, 1.5, args, bank, lead_in_from=0.0)
    assert exit_[0] > approach[0], (approach, exit_)
    assert exit_[1] == pytest.approx(approach[1])       # only the FIRST hold differs


def test_every_interior_hold_reserves_its_own_incoming_transient():
    import mb_plan_library as pl

    args = type("A", (), {"dt": _DT, "approach_a_max": 2.0, "approach_min_transient_s": 0.35,
                          "approach_hop_slack": 1.3})()
    bank = type("B", (), {"achieved_vel": _BANK_VEL, "period_frames": _BANK_PERIODS})()
    mins = pl._hold_minimums([0.2, 3.0, 0.2], 1.5, args, bank)
    # The 0.2 -> 3.0 hop is the expensive one; its hold must be the largest of the three.
    assert mins[1] > mins[0] and mins[1] > mins[2], mins


def test_split_dist_honours_every_minimum_and_the_total():
    import mb_plan_library as pl

    rng = np.random.default_rng(0)
    mins = [1.5, 3.0, 2.0]
    for _ in range(50):
        shares = pl._split_dist(rng, 12.0, mins)
        assert sum(shares) == pytest.approx(12.0)
        assert all(s >= m - 1e-9 for s, m in zip(shares, mins)), (shares, mins)


def test_the_split_is_not_the_same_place_every_time():
    """A fixed split would put every clip's command change at one distance from the terrain -- a
    landmark the model could key on instead of the depth image."""
    import mb_plan_library as pl

    rng = np.random.default_rng(0)
    first = [pl._split_dist(rng, 12.0, [1.5, 3.0])[0] for _ in range(20)]
    assert np.std(first) > 0.2, first
