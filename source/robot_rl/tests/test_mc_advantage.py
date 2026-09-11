# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Monte-Carlo advantage: no bootstrap in either half.

``combined_n_step_advantage`` differences against the critic's own predictions at ``t + n``;
``mc_advantage`` differences against what actually happened -- the realized normalized return
and the episode's own outcome. Same decomposition of ``V``, different target, so the two label
the same quantity and can be compared directly.

The property worth protecting is the one that makes the estimator meaningful at all: a
perfectly calibrated critic must score exactly zero advantage everywhere. If it does not, the
label carries a constant offset and every percentile cut downstream is shifted.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching import advantage as adv


def _table(w: int = 24, seed: int = 0):
    """A small window table: 4 episodes of 6 re-plans, half of them failing."""
    rng = np.random.default_rng(seed)
    uid = np.repeat(np.arange(4), 6)
    ridx = np.tile(np.arange(6), 4)
    failed = np.repeat(np.array([True, False, True, False]), 6)
    g = rng.uniform(0.0, 1.0, w)
    g_hat = rng.uniform(0.0, 1.0, w)
    p_fail = rng.uniform(0.0, 1.0, w)
    return uid, ridx, failed, g, g_hat, p_fail


def test_a_perfectly_calibrated_critic_scores_zero_advantage():
    """The defining property. A constant offset here shifts every percentile cut downstream."""
    uid, ridx, failed, g, _g_hat, _p = _table()
    a, ok = adv.mc_advantage(g, g, failed.astype(np.float64), failed, w_fail=0.5)
    assert bool(ok.all())
    assert np.allclose(a, 0.0, atol=1e-12)


def test_w_fail_one_is_the_survival_half_alone_and_needs_no_return():
    """``P(fail|s) - 1{failed}``: positive when the critic hedged and the episode survived."""
    uid, ridx, failed, g, g_hat, p_fail = _table()
    a, ok = adv.mc_advantage(np.zeros_like(g), np.zeros_like(g), p_fail, failed,
                             w_fail=1.0, valid=np.zeros(len(g), dtype=bool))
    assert bool(ok.all()), "the survival half survives an absent return"
    assert np.allclose(a, p_fail - failed.astype(np.float64))
    assert (a[~failed] > 0).sum() > 0 and (a[failed] < 0).sum() > 0


def test_w_fail_zero_is_the_quality_half_alone():
    """``G_t - G_hat(s_t)``: positive when the episode beat the critic's expectation."""
    uid, ridx, failed, g, g_hat, p_fail = _table()
    a, ok = adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=0.0)
    assert np.allclose(a, g - g_hat)


def test_the_two_halves_are_a_convex_combination():
    """``w_track = 1 - w_fail``, matching combined_n_step_advantage's own decomposition."""
    uid, ridx, failed, g, g_hat, p_fail = _table()
    q, _ = adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=0.0)
    s, _ = adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=1.0)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        a, _ = adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=w)
        assert np.allclose(a, (1.0 - w) * q + w * s, atol=1e-12), w


def test_a_row_with_no_measured_return_loses_its_QUALITY_half_only():
    """It must not be charged for a return that was never measured, nor silently dropped."""
    uid, ridx, failed, g, g_hat, p_fail = _table()
    valid = np.ones(len(g), dtype=bool)
    valid[[0, 7, 13]] = False
    a, ok = adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=0.5, valid=valid)
    surv = p_fail - failed.astype(np.float64)
    assert np.allclose(a[~valid], 0.5 * surv[~valid]), "quality half dropped, survival kept"
    assert bool(ok.all()), "still usable at w_fail > 0"

    # ...but at w_fail = 0 the row has nothing left, so it is flagged invalid.
    a0, ok0 = adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=0.0, valid=valid)
    assert not bool(ok0[~valid].any())
    assert np.allclose(a0[~valid], 0.0)


def test_it_bootstraps_nothing_so_it_ignores_the_episode_layout():
    """No ``n``, no look-ahead: shuffling the rows cannot change any row's advantage.

    This is the concrete difference from the n-step form, which reads ``s_{t+n}`` and so
    depends on ``episode_uid`` / ``replan_index`` entirely.
    """
    uid, ridx, failed, g, g_hat, p_fail = _table()
    a, _ = adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=0.5)
    perm = np.random.default_rng(1).permutation(len(g))
    a_perm, _ = adv.mc_advantage(g[perm], g_hat[perm], p_fail[perm], failed[perm], w_fail=0.5)
    assert np.allclose(a_perm, a[perm])


def test_malformed_inputs_are_refused():
    """A length mismatch would broadcast into a plausible wrong label."""
    uid, ridx, failed, g, g_hat, p_fail = _table()
    with pytest.raises(ValueError, match="agree in length"):
        adv.mc_advantage(g[:-1], g_hat, p_fail, failed)
    with pytest.raises(ValueError, match="w_fail must be"):
        adv.mc_advantage(g, g_hat, p_fail, failed, w_fail=1.5)


def test_the_labeller_offers_the_estimator_and_defaults_to_n_step():
    """Default unchanged, so every existing artifact reproduces bit for bit."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[3] / "scripts" / "flow_matching"
           / "label_advantage.py").read_text()
    assert 'choices=("n_step", "mc")' in src
    assert 'default="n_step"' in src
    assert "adv_mod.mc_advantage(" in src
