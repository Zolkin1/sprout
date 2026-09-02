# Copyright 2026 Zachary Olkin. All rights reserved.

"""The advantage estimators must agree with the recursion they claim to invert.

Every function in ``robot_rl.flow_matching.critic_eval`` is a small piece of arithmetic over a
flat window table, which is exactly the shape of code that is silently wrong: a discount
applied once too often, a terminal closed on the critic instead of the outcome, an episode
boundary crossed by a backward recursion.  None of those raise, and all of them produce
finite, plausible advantages.

So the tests are identities rather than smoke:

* the TD residuals of the TRUE value function are identically zero, which pins the discount,
  the ``(1 - gamma)`` factor and the terminal together -- get any one wrong and the residual
  is non-zero somewhere;
* ``GAE(lambda = 1)`` equals the Monte-Carlo advantage and ``GAE(lambda = 0)`` the one-chunk
  TD one, which pins the backward accumulation and its truncation at the episode boundary;
* the variance decomposition sums to the total;
* reliability returns 1 on a duplicated estimate and ~0 on an independent one.

Row order is DELIBERATELY shuffled in the recursion tests: the functions promise to sort a
view internally and write back in the caller's order, and a version that assumed sorted input
would pass every one of these on sorted rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching import advantage as adv_mod
from robot_rl.flow_matching import critic_eval as ce


def _toy(n_ep: int = 7, seed: int = 0):
    """A small window table with ragged episodes and mixed outcomes.

    Args:
        n_ep: Episodes.
        seed: RNG seed.

    Returns:
        ``(uid, replan_index, reward, failed)``, rows in a SHUFFLED order.
    """
    rng = np.random.default_rng(seed)
    uid, ri, rew, fail = [], [], [], []
    for e in range(n_ep):
        m = int(rng.integers(2, 9))
        f = bool(e % 3 == 0)
        uid += [e] * m
        ri += list(range(m))
        rew += list(rng.uniform(0.1, 1.0, size=m))
        fail += [f] * m
    order = rng.permutation(len(uid))
    return (np.asarray(uid)[order], np.asarray(ri)[order],
            np.asarray(rew)[order], np.asarray(fail)[order])


def test_the_td_residual_of_the_true_value_is_zero():
    """Handed ``G`` itself as the value, every residual must vanish.

    This is the identity that pins the whole quality-head arm: ``G`` satisfies
    ``G_t = (1-g) R_t + g G_{t+1}`` with the tail :func:`advantage.normalized_returns` uses, so
    a TD residual built with the same convention is zero to floating point. A wrong discount,
    a missing ``(1 - gamma)`` or a terminal closed on the critic all break it.
    """
    uid, ri, rew, fail = _toy()
    ok = np.ones(uid.size, dtype=bool)
    g, g_ok = adv_mod.normalized_returns(rew, ok, uid, ri, fail, gamma=0.95)
    assert g_ok.all()
    delta, term = ce.td_residuals(g, rew, uid, ri, fail, gamma=0.95, normalized=True)
    assert np.abs(delta).max() < 1e-12, f"max |delta| = {np.abs(delta).max()}"
    # One terminal per episode, and it is the LAST chunk of each.
    assert int(term.sum()) == int(np.unique(uid).size)


def test_gae_spans_td_to_monte_carlo():
    """``lam = 0`` is the one-chunk TD advantage; ``lam = 1`` is ``G - V``.

    Both ends are checked against quantities built independently of the GAE recursion, so the
    backward accumulation and its truncation at the episode boundary are pinned rather than
    compared against themselves.
    """
    uid, ri, rew, fail = _toy(seed=3)
    ok = np.ones(uid.size, dtype=bool)
    g, _ = adv_mod.normalized_returns(rew, ok, uid, ri, fail, gamma=0.95)
    rng = np.random.default_rng(1)
    v = g + rng.normal(0.0, 0.05, size=g.size)      # a critic that is wrong, but not by much
    delta, _ = ce.td_residuals(v, rew, uid, ri, fail, gamma=0.95, normalized=True)

    a0 = ce.gae_advantage(delta, uid, ri, gamma=0.95, lam=0.0)
    assert np.allclose(a0, delta, atol=1e-12)

    a1 = ce.gae_advantage(delta, uid, ri, gamma=0.95, lam=1.0)
    assert np.allclose(a1, g - v, atol=1e-10), np.abs(a1 - (g - v)).max()


def test_the_survival_arm_reduces_to_the_n_step_advantage():
    """With a zero reward and ``gamma = 1``, the one-chunk TD is ``advantage.n_step_advantage``.

    The two are written independently -- one as a value difference, the other as a probability
    difference with an explicit terminal bootstrap -- and the sign convention between them is
    the single easiest thing in this module to get backwards.
    """
    uid, ri, _, fail = _toy(seed=5)
    rng = np.random.default_rng(2)
    p = rng.uniform(0.0, 1.0, size=uid.size)
    delta, _ = ce.td_residuals(-p, np.zeros(p.size), uid, ri, fail,
                               gamma=1.0, normalized=False)
    ref, _ = adv_mod.n_step_advantage(p, uid, ri, fail, n=1)
    assert np.allclose(delta, ref, atol=1e-12), np.abs(delta - ref).max()


def test_variance_decomposition_sums_to_the_total():
    """Between plus within is the population variance, exactly."""
    rng = np.random.default_rng(7)
    grp = rng.integers(0, 12, size=500)
    x = rng.normal(size=500) + grp * 0.3
    d = ce.variance_decomposition(x, grp)
    assert abs(d["between"] + d["within"] - d["total"]) < 1e-12
    assert abs(d["total"] - float(np.var(x))) < 1e-12
    assert 0.0 <= d["frac_within"] <= 1.0


def test_variance_decomposition_is_all_between_when_a_group_is_constant():
    """A quantity constant within each group has no within-group variance at all."""
    grp = np.repeat(np.arange(10), 5)
    x = grp.astype(np.float64)
    d = ce.variance_decomposition(x, grp)
    assert d["within"] == pytest.approx(0.0, abs=1e-12)
    assert d["frac_within"] == pytest.approx(0.0, abs=1e-12)


def test_reliability_is_one_on_a_duplicate_and_zero_on_noise():
    """The coefficient must read as the signal fraction it claims to be."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=4000)
    assert ce.reliability(a, a.copy())["pearson"] == pytest.approx(1.0, abs=1e-12)
    assert abs(ce.reliability(a, rng.normal(size=4000))["pearson"]) < 0.05


def test_reliability_recovers_a_known_signal_fraction():
    """With equal independent errors, ``corr`` is the true signal variance fraction."""
    rng = np.random.default_rng(13)
    t = rng.normal(0.0, 1.0, size=200000)
    e = 1.0                                   # so signal fraction = 1 / (1 + 1) = 0.5
    r = ce.reliability(t + rng.normal(0, e, t.size), t + rng.normal(0, e, t.size))
    assert r["pearson"] == pytest.approx(0.5, abs=0.02)


def test_awr_weights_do_not_overflow_at_a_sharp_temperature():
    """A small ``beta`` must collapse the ESS, not produce NaNs.

    Without the max-shift before the exponential this returns ``inf`` and every downstream
    effective sample size comes back NaN -- which reads as a broken pipeline rather than as
    the weight collapse it actually is.
    """
    rng = np.random.default_rng(17)
    a = rng.normal(size=5000)
    w = ce.awr_weights(a, beta=0.01, w_max=None)
    assert np.isfinite(w).all()
    ess = ce.effective_sample_size(w) / w.size
    assert 0.0 < ess < 0.05


def test_awr_weights_are_uniform_in_the_hot_limit():
    """As ``beta`` grows the weights flatten and the ESS returns to the row count."""
    rng = np.random.default_rng(19)
    a = rng.normal(size=5000)
    w = ce.awr_weights(a, beta=1e6, w_max=None)
    assert ce.effective_sample_size(w) / w.size == pytest.approx(1.0, abs=1e-3)


def test_normalized_progress_uses_the_surviving_length():
    """Progress is measured against the rows PRESENT, not against a recorded episode length.

    The fit drops censored episodes and every window at or after the first bad step, so a
    failed episode's last surviving chunk must still sit at 1.0 -- otherwise every failure is
    pushed into a lower progress bucket than every success and any metric reported against
    progress is confounded with the outcome.
    """
    uid = np.array([0, 0, 0, 1, 1])
    ri = np.array([0, 1, 2, 0, 5])          # episode 1's rows are NOT contiguous in index
    p = ce.normalized_progress(uid, ri)
    assert np.allclose(p, [0.0, 0.5, 1.0, 0.0, 1.0])


def test_clustered_bootstrap_is_wider_than_a_row_level_one():
    """Resampling clusters must widen the interval when rows within a cluster agree.

    A row-level interval on 40 clusters of 50 identical rows would be ~sqrt(50) too tight; the
    clustered one has to see through the replication.
    """
    rng = np.random.default_rng(23)
    per = rng.normal(0.0, 1.0, size=40)
    cluster = np.repeat(np.arange(40), 50)
    values = np.repeat(per, 50)
    _, lo, hi = ce.clustered_bootstrap_ci(values, cluster, n_boot=500, seed=0)
    naive = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    assert (hi - lo) / 2 > 3 * naive


def test_spearman_ranks_ties_by_sorted_runs_not_by_scanning():
    """``value_gates.spearman`` must average tied ranks in ``O(n log n)``, not ``O(u n)``.

    The tie loop it used to run scanned the whole array once per DISTINCT value, which on a
    continuous score is quadratic: at the 294k rows of a full rollout table it is ~9e10
    element comparisons and the call does not return, while at the few thousand rows every
    previous caller passed it merely looked slow. This pins the replacement against a
    reference implementation of average ranks on the cases that separate them -- heavy ties,
    a binary target, and one distinct value among thousands of identical ones.
    """
    rng = np.random.default_rng(0)

    def reference(x, y):
        """Average-rank Spearman, written the slow obvious way."""
        def rank(v):
            out = np.empty(v.size, dtype=np.float64)
            for i, vi in enumerate(v):
                out[i] = 1 + (v < vi).sum() + 0.5 * ((v == vi).sum() - 1)
            return out
        ra, rb = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
        return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

    from robot_rl.flow_matching import value_gates as vg

    for a, b in ((rng.normal(size=400), rng.normal(size=400)),
                 (rng.integers(0, 4, 400).astype(float), rng.integers(0, 3, 400).astype(float)),
                 (rng.normal(size=400), (rng.random(400) < 0.3).astype(float)),
                 (np.r_[np.zeros(399), 1.0], rng.normal(size=400))):
        assert vg.spearman(a, b) == pytest.approx(reference(a, b), abs=1e-12)

    # And it must actually be fast: the quadratic form takes ~8 s at this size.
    import time
    x, y = rng.normal(size=100_000), rng.normal(size=100_000)
    t0 = time.time()
    vg.spearman(x, y)
    assert time.time() - t0 < 1.0


def test_the_terminal_is_closed_on_the_last_REAL_reward_not_the_placeholder():
    """The identity must survive the one-row shift that ``chunk_rewards_from_track_err`` leaves.

    That builder stores each chunk's reward on the row BEFORE it, so every episode's last row
    carries ``reward = 0.0`` with ``valid = False``. ``normalized_returns`` drops those rows
    before recursing; a residual computed over ALL rows instead closes the terminal on that
    zero. The error is largest on SUCCESSFUL episodes, whose tail is the last reward rather
    than an absorbing zero -- so it is confounded with the outcome, not noise.

    The all-valid version of this test passes either way, which is exactly why it needed its
    own case.
    """
    uid, ri, rew, fail = _toy(seed=9)
    # Reproduce the real layout: the last row of every episode is a placeholder.
    ok = np.ones(uid.size, dtype=bool)
    order = np.lexsort((ri, uid))
    us = uid[order]
    starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
    ends = np.r_[starts[1:], us.size]
    for s, e in zip(starts, ends):
        rows = order[s:e]
        ok[rows[-1]] = False
        rew[rows[-1]] = 0.0

    g, g_ok = adv_mod.normalized_returns(rew, ok, uid, ri, fail, gamma=0.95)
    assert (g_ok == ok).all(), "the builder and the mask must agree on which rows are real"

    delta, _ = ce.td_residuals(g, rew, uid, ri, fail, gamma=0.95, normalized=True, valid=ok)
    assert np.abs(delta[ok]).max() < 1e-12, f"max |delta| = {np.abs(delta[ok]).max()}"

    # And the unmasked call must be visibly WRONG, so this test cannot pass vacuously.
    bad, _ = ce.td_residuals(g, rew, uid, ri, fail, gamma=0.95, normalized=True)
    assert np.abs(bad[ok]).max() > 1e-3

    # lam = 1 must still be exactly G - V under the mask.
    rng = np.random.default_rng(4)
    v = g + rng.normal(0.0, 0.05, size=g.size)
    d2, _ = ce.td_residuals(v, rew, uid, ri, fail, gamma=0.95, normalized=True, valid=ok)
    a1 = ce.gae_advantage(d2, uid, ri, gamma=0.95, lam=1.0, valid=ok)
    assert np.allclose(a1[ok], (g - v)[ok], atol=1e-10)


def test_clustered_bootstrap_accepts_row_indices_as_values():
    """A multi-column statistic is expressed by resampling INDICES, so no cast may happen.

    An AUC gap needs two predictions and a label per row, which cannot be packed into the
    single ``values`` array. The idiom is to pass row indices and close over the columns --
    and a ``float64`` cast inside the helper turns those indices into floats, which numpy
    then refuses as an index. That failure is at least loud; the reason it is pinned is that
    the helper is the only source of confidence intervals in the report, so a signature that
    silently forbids the multi-column case leaves every interval unwritten.
    """
    rng = np.random.default_rng(31)
    n, n_tiles = 600, 12
    cluster = rng.integers(0, n_tiles, size=n)
    label = rng.random(n) < 0.3
    good = rng.normal(size=n) + label * 1.5
    bad = rng.normal(size=n)

    from robot_rl.flow_matching import value_gates as vg

    def gap(ix):
        """AUC of the informative score minus AUC of the uninformative one."""
        return vg.auc(good[ix], label[ix]) - vg.auc(bad[ix], label[ix])

    pt, lo, hi = ce.clustered_bootstrap_ci(np.arange(n), cluster, statistic=gap,
                                           n_boot=200, seed=0)
    assert lo <= pt <= hi
    assert pt > 0.1, "the informative score must beat the uninformative one"
