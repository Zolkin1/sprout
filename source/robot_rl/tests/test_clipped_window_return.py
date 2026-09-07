# Copyright 2026 Zachary Olkin. All rights reserved.

"""The reward/value spec: one clip constant, one window length, one discount.

    r_t = -min(p_t, c)                  live chunk
    r_t = -c                            every chunk after termination
    G_t = sum_{i<N} gamma^i r_{t+i}     fixed window, no bootstrap
    G_t / (c * sum_{i<N} gamma^i)       normalized into [-1, 0]

The clip constant is doing three jobs -- termination penalty, survival reward, and outlier
control for the regression -- so the tests are organised around checking each of them holds
rather than around the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching.advantage import clipped_window_return as cwr


def _episode(penalties, failed, uid=0):
    """One episode's arrays from a list of per-chunk penalties."""
    p = np.asarray(penalties, dtype=np.float64)
    n = p.size
    return dict(penalty=p, valid=np.ones(n, bool), episode_uid=np.full(n, uid, np.int64),
                replan_index=np.arange(n), failed=np.full(n, bool(failed)))


def _g(penalties, failed, *, c=2.0, window=3, gamma=1.0, uid=0):
    """The return and validity for one episode."""
    e = _episode(penalties, failed, uid)
    return cwr(e["penalty"], e["valid"], e["episode_uid"], e["replan_index"], e["failed"],
               clip_c=c, window=window, gamma=gamma)


# -----------------------------------------------------------------------------------------
# the arithmetic and the declared range
# -----------------------------------------------------------------------------------------
def test_a_perfect_surviving_plan_scores_exactly_zero():
    """Zero penalty is the top of the range, and the range top is 0 by construction."""
    g, ok = _g([0.0] * 6, failed=False)
    assert np.allclose(g[ok], 0.0)


def test_a_fully_saturated_plan_scores_exactly_minus_one():
    """Any penalty at or above the clip is the bottom of the range."""
    for p in (2.0, 5.0, 47.5):
        g, ok = _g([p] * 6, failed=False, c=2.0)
        assert np.allclose(g[ok], -1.0), p


def test_the_return_stays_inside_the_declared_range():
    """[-1, 0] is what the MSE target's support relies on; nothing may leave it."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(0.0, 60.0, 12)
        for failed in (False, True):
            g, ok = _g(p, failed=failed, c=2.5, window=4, gamma=0.9)
            assert np.all(g[ok] <= 1e-12) and np.all(g[ok] >= -1.0 - 1e-12)


def test_the_discount_and_normalisation_are_the_stated_formula():
    """Checked against the closed form, not against another implementation."""
    c, n, gamma = 2.0, 3, 0.9
    p = np.array([0.5, 1.0, 3.0, 0.0, 0.0, 0.0])
    g, ok = _g(p, failed=False, c=c, window=n, gamma=gamma)
    disc = gamma ** np.arange(n)
    r = -np.minimum(p, c)
    want = float((disc * r[:n]).sum() / (c * disc.sum()))
    assert ok[0]
    assert g[0] == pytest.approx(want, rel=1e-12)


# -----------------------------------------------------------------------------------------
# job 1: the clip constant IS the termination penalty
# -----------------------------------------------------------------------------------------
def test_dying_earlier_is_strictly_worse_and_in_proportion():
    """Compared at the SAME row, so this measures the death and not the position.

    Two otherwise identical episodes with zero penalty while alive, dying after 1, 2 and 3
    chunks. With a 3-chunk window, row 0 sees 2, 1 and 0 dead chunks respectively.
    """
    got = []
    for alive in (1, 2, 3):
        g, ok = _g([0.0] * alive, failed=True, c=2.0, window=3, gamma=1.0)
        assert ok[0]
        got.append(g[0])
    assert got[0] < got[1] < got[2], got
    assert got[2] == pytest.approx(0.0)          # nothing dead inside the window
    assert got[0] == pytest.approx(-2.0 / 3.0)   # 2 of 3 chunks dead


def test_a_chunk_of_death_and_a_catastrophic_chunk_are_the_same():
    """Deliberate: the clip is what makes them comparable rather than one swamping the other."""
    dead, _ = _g([0.0], failed=True, c=2.0, window=2, gamma=1.0)
    awful, _ = _g([0.0, 99.0], failed=False, c=2.0, window=2, gamma=1.0)
    assert dead[0] == pytest.approx(awful[0])


# -----------------------------------------------------------------------------------------
# job 2: the clip constant IS the survival reward
# -----------------------------------------------------------------------------------------
def test_surviving_longer_is_better_with_no_bonus_to_pick():
    """Padding the dead chunks with -c is the survival reward, up to an offset."""
    short, _ = _g([0.0] * 2, failed=True, c=2.0, window=4, gamma=1.0)
    long, _ = _g([0.0] * 4, failed=True, c=2.0, window=4, gamma=1.0)
    assert short[0] < long[0]
    assert long[0] == pytest.approx(0.0)


# -----------------------------------------------------------------------------------------
# job 3: outlier control
# -----------------------------------------------------------------------------------------
def test_the_clip_bounds_the_influence_of_a_through_the_wall_plan():
    """Measured max penalty is 47.5 against a p95 of 1.72; unclipped it would dominate."""
    mild, _ = _g([1.0] * 4, failed=False, c=2.0, window=3, gamma=1.0)
    wild, _ = _g([47.5] * 4, failed=False, c=2.0, window=3, gamma=1.0)
    assert mild[0] == pytest.approx(-0.5)
    assert wild[0] == pytest.approx(-1.0), "at most one clip-unit worse, not 24x"


# -----------------------------------------------------------------------------------------
# the boundary decision
# -----------------------------------------------------------------------------------------
def test_a_window_past_a_NON_failure_end_is_invalid_rather_than_invented():
    """Padding a surviving episode with -c would price arriving as if it were dying."""
    g, ok = _g([0.0] * 5, failed=False, window=3)
    assert ok.tolist() == [True, True, True, False, False]
    assert np.allclose(g[~ok], 0.0), "an invalid row must not carry a number"


def test_a_window_past_a_FAILURE_end_is_padded_and_stays_valid():
    """Those are the rows nearest the outcome, and the -c fill is exactly the spec."""
    g, ok = _g([0.0] * 5, failed=True, window=3)
    assert bool(ok.all())
    assert g[-1] == pytest.approx(-2.0 / 3.0)


def test_a_missing_chunk_in_the_middle_is_a_GAP_not_a_shift():
    """The fit drops post-latch windows, so the table has holes; indices must be honoured."""
    p = np.array([0.0, 0.0, 0.0, 0.0])
    e = dict(penalty=p, valid=np.ones(4, bool), episode_uid=np.zeros(4, np.int64),
             replan_index=np.array([0, 1, 3, 4]), failed=np.zeros(4, bool))
    g, ok = cwr(e["penalty"], e["valid"], e["episode_uid"], e["replan_index"], e["failed"],
                clip_c=2.0, window=2, gamma=1.0)
    # Row at index 1 needs chunk 2, which is absent, and the episode did not fail -> invalid.
    assert ok.tolist() == [True, False, True, False]


def test_window_one_is_just_the_clipped_reward():
    """The degenerate case has to be the reward itself, or the normalisation is wrong."""
    p = np.array([0.0, 1.0, 2.0, 5.0])
    g, ok = _g(p, failed=False, c=2.0, window=1, gamma=1.0)
    assert bool(ok.all())
    assert np.allclose(g, -np.minimum(p, 2.0) / 2.0)


def test_malformed_constants_and_lengths_are_refused():
    """Each of the three constants has a range, and a length mismatch would broadcast."""
    e = _episode([0.0] * 4, False)
    args = (e["penalty"], e["valid"], e["episode_uid"], e["replan_index"], e["failed"])
    with pytest.raises(ValueError, match="clip_c"):
        cwr(*args, clip_c=0.0, window=2, gamma=1.0)
    with pytest.raises(ValueError, match="window"):
        cwr(*args, clip_c=1.0, window=0, gamma=1.0)
    with pytest.raises(ValueError, match="gamma"):
        cwr(*args, clip_c=1.0, window=2, gamma=1.5)
    with pytest.raises(ValueError, match="agree in length"):
        cwr(e["penalty"][:-1], *args[1:], clip_c=1.0, window=2, gamma=1.0)


def test_episodes_are_independent():
    """One episode's death cannot leak into another's window."""
    p = np.zeros(8)
    uid = np.array([0] * 4 + [1] * 4, np.int64)
    ridx = np.tile(np.arange(4), 2)
    failed = np.array([True] * 4 + [False] * 4)
    g, ok = cwr(p, np.ones(8, bool), uid, ridx, failed, clip_c=2.0, window=3, gamma=1.0)
    assert bool(ok[:4].all()), "the failed episode is padded throughout"
    assert ok[4:].tolist() == [True, True, False, False]
    assert np.allclose(g[4:6], 0.0), "the survivor is unaffected by the other's death"
