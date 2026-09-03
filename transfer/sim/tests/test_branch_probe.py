# Copyright 2026 Zachary Olkin. All rights reserved.

"""The branch probe's statistics, against synthetic data with known answers.

``branch_probe.py`` exists to answer whether the critic can order SIBLING chunks branched from
one state -- the question every state-averaged number (AUC, ECE, lead time) is silent on. Its
output will be used to decide whether the whole best-of-K premise is sound, so its estimators
are validated here on data whose answer is known by construction, in the discipline
``eval/seed_headroom.py`` records the hard way: *"Validate any estimator here against synthetic
data with a known answer before quoting it"* -- where three were, and two of the three failed.

What is pinned:

* a PERFECT ranker scores tau +1 and best-pick 1.0; an INVERTED one -1 and ~0;
* a ranker with NO within-group information scores tau ~0 and best-pick at the chance line,
  which for K siblings is *not* 1/K when ties in the target are possible -- so the null is
  measured by permutation rather than assumed;
* the within-group permutation null is a REAL null (it moves with the data and returns the
  chance line on a signal-free input), unlike the global label shuffle used elsewhere in this
  tree, which returns 0.5 analytically for any input and so cannot fail;
* groups the target cannot separate are EXCLUDED rather than counted as misses -- with a binary
  survival target most groups may be all-survive or all-fail, and scoring those as failures to
  rank would manufacture a null result out of missing data;
* the tie-in-target case, because a K=4 group with two equally-best siblings is common on a
  binary label and a naive ``argmax == argmax`` test would call a correct pick wrong.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loaded BY PATH: `transfer/sim` is not a package, and a bare import would depend on whether
# another suite has already put that directory on `sys.path`.
_spec = importlib.util.spec_from_file_location(
    "_branch_probe_under_test", os.path.join(_SIM, "branch_probe.py"))
_bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bp)

_K = 4


def _groups(n_groups, k=_K):
    """Group ids for ``n_groups`` groups of ``k`` siblings."""
    return np.repeat(np.arange(n_groups), k)


def test_a_perfect_ranker_scores_one_and_an_inverted_one_scores_minus_one():
    """The two ends of the scale, so the sign convention cannot be backwards."""
    rng = np.random.default_rng(0)
    gid = _groups(300)
    real = rng.uniform(0, 1, gid.size)
    # lower_is_better: a perfect risk score is the NEGATIVE of the realized quality.
    tau, n = _bp._within_group_tau(-real, real, gid, lower_is_better=True)
    assert tau == pytest.approx(1.0), tau
    assert n == 300
    rate, nr = _bp._pick_rate(-real, real, gid, lower_is_better=True)
    assert rate == pytest.approx(1.0) and nr == 300
    tau_bad, _ = _bp._within_group_tau(real, real, gid, lower_is_better=True)
    assert tau_bad == pytest.approx(-1.0)
    # ...and the SAME data read as a quality score (higher is better) flips to +1, which is
    # what the orientation flag exists to make explicit rather than a reader's job.
    tau_q, _ = _bp._within_group_tau(real, real, gid, lower_is_better=False)
    assert tau_q == pytest.approx(1.0)
    rate_bad, _ = _bp._pick_rate(real, real, gid, lower_is_better=True)
    assert rate_bad < 0.05, rate_bad


def test_a_signal_free_ranker_lands_on_the_chance_line_the_permutation_null_predicts():
    """No within-group information must score tau ~0 and best-pick at the measured null.

    The chance line is NOT 1/K in general: with ties in the target, several siblings can be
    "a best", so a random pick is right more often than 1/K. That is why the report measures
    the null by permutation instead of drawing a 0.25 reference line.
    """
    rng = np.random.default_rng(1)
    gid = _groups(4000)
    real = rng.uniform(0, 1, gid.size)
    noise = rng.uniform(0, 1, gid.size)          # independent of `real`
    tau, _ = _bp._within_group_tau(noise, real, gid, lower_is_better=True)
    assert abs(tau) < 0.03, tau
    rate, _ = _bp._pick_rate(noise, real, gid, lower_is_better=True)
    assert abs(rate - 1.0 / _K) < 0.03, rate     # continuous target -> no ties -> 1/K

    # ...and with a BINARY target the chance line moves well above 1/K, because ties make
    # several siblings simultaneously best.
    binary = (rng.uniform(0, 1, gid.size) < 0.5).astype(float)
    rate_b, n_b = _bp._pick_rate(noise, binary, gid, lower_is_better=True)
    assert rate_b > 1.0 / _K + 0.1, (rate_b, "a binary target's chance line is not 1/K")
    assert n_b < gid.size / _K, "all-equal groups were not excluded"


def test_groups_the_target_cannot_separate_are_excluded_not_scored_as_misses():
    """An all-survive or all-fail group carries no ranking information.

    Counting it as a miss would turn missing data into a null result -- and on a binary
    survival label most groups may be exactly that, which is why the report prints the
    informative-group count beside every statistic.
    """
    gid = _groups(3)
    # group 0 all-fail, group 1 all-survive, group 2 mixed
    real = np.array([0, 0, 0, 0,  1, 1, 1, 1,  1, 0, 0, 0], dtype=float)
    pred = np.array([.1, .2, .3, .4] * 3)
    rate, n = _bp._pick_rate(pred, real, gid, lower_is_better=True)
    assert n == 1, f"only the mixed group informs; got n={n}"
    assert rate == pytest.approx(1.0), "the lowest-risk sibling IS the survivor here"
    tau, ntau = _bp._within_group_tau(pred, real, gid, lower_is_better=True)
    assert ntau == 1 and tau == pytest.approx(1.0), (tau, ntau)


def test_a_tie_for_best_counts_as_a_hit():
    """Picking any of several equally-best siblings is a correct pick, not a wrong one."""
    gid = _groups(1)
    real = np.array([1.0, 1.0, 0.0, 0.0])          # two siblings tie for best
    pred = np.array([0.9, 0.1, 0.5, 0.4])          # argmin risk -> index 1, a joint best
    rate, n = _bp._pick_rate(pred, real, gid, lower_is_better=True)
    assert n == 1 and rate == pytest.approx(1.0)
    # ...and picking a non-best is a miss.
    rate2, _ = _bp._pick_rate(np.array([0.1, 0.9, 0.5, 0.4]), real, gid, lower_is_better=True)
    assert rate2 == pytest.approx(1.0), "index 0 also ties for best here"
    rate3, _ = _bp._pick_rate(np.array([0.5, 0.6, 0.1, 0.4]), real, gid, lower_is_better=True)
    assert rate3 == pytest.approx(0.0), "index 2 is not a best and must count as a miss"


def test_nan_predictions_and_targets_are_dropped_pairwise():
    """A branch that fell inside its own chunk has no critic read; it must not poison a group."""
    gid = _groups(2)
    real = np.array([1., 0., 0., 0.,  1., 0., 0., 0.])
    pred = np.array([.1, .2, np.nan, .4,  np.nan, np.nan, np.nan, np.nan])
    rate, n = _bp._pick_rate(pred, real, gid, lower_is_better=True)
    assert n == 1, "the all-NaN group must be dropped, not scored"
    assert rate == pytest.approx(1.0)
    tau, ntau = _bp._within_group_tau(pred, real, gid, lower_is_better=True)
    assert ntau == 1 and np.isfinite(tau)


def test_the_within_group_permutation_null_is_a_real_null():
    """It must return the chance line on signal-free input AND move on real signal.

    Contrast the GLOBAL label shuffle this tree uses for AUC, which is vacuous: under a
    permutation of every label, E[AUC] = 0.5 analytically for any score vector, so it reports
    0.500 even for a provably broken estimator. A within-group permutation preserves the group
    structure and destroys only the within-group association -- the actual hypothesis.
    """
    rng = np.random.default_rng(2)
    gid = _groups(1500)
    real = rng.uniform(0, 1, gid.size)

    def null_of(pred):
        out = []
        for _ in range(60):
            sh = pred.copy()
            for g in np.unique(gid):
                m = np.where(gid == g)[0]
                sh[m] = pred[rng.permutation(m)]
            out.append(_bp._pick_rate(sh, real, gid, lower_is_better=True)[0])
        return float(np.mean(out))

    perfect = -real
    assert _bp._pick_rate(perfect, real, gid, lower_is_better=True)[0] == pytest.approx(1.0)
    # The null of a PERFECT ranker is still the chance line -- that is what makes it a null.
    assert abs(null_of(perfect) - 1.0 / _K) < 0.03, null_of(perfect)
    # And a signal-free ranker sits ON its own null, i.e. the test has no false positive here.
    noise = rng.uniform(0, 1, gid.size)
    obs = _bp._pick_rate(noise, real, gid, lower_is_better=True)[0]
    assert abs(obs - null_of(noise)) < 0.04, (obs, null_of(noise))


def test_partial_signal_lands_strictly_between_the_ends():
    """A ranker that is right more often than chance but not always must read as such."""
    rng = np.random.default_rng(3)
    gid = _groups(4000)
    real = rng.uniform(0, 1, gid.size)
    # risk = -quality corrupted by noise of comparable scale
    pred = -real + rng.normal(0, 0.55, gid.size)
    tau, _ = _bp._within_group_tau(pred, real, gid, lower_is_better=True)
    rate, _ = _bp._pick_rate(pred, real, gid, lower_is_better=True)
    assert 0.1 < tau < 0.9, tau
    assert 1.0 / _K + 0.03 < rate < 0.97, rate


def test_the_continuation_tags_cannot_collide_with_a_branch_key():
    """CRN is only a control if a sibling's continuation is not its own branch draw again.

    The branch key is ``[seed, replan, cand]`` with ``cand < k``; the continuation keys carry
    a large tag in a later position. Pinned because a collision would silently turn the
    variance-reduction control into a duplicated treatment.
    """
    assert _bp._CONT_TAG != _bp._CONT_TAG_B
    assert _bp._CONT_TAG > 1000 and _bp._CONT_TAG_B > 1000, "tags must not alias a cand index"
    src = open(os.path.join(_SIM, "branch_probe.py")).read()
    # the shared stream must NOT be keyed by `cand`
    assert "np.random.default_rng([seed, start_replan, r, tag_seed])" in src, (
        "the continuation key changed; it must carry no `cand`, or siblings stop being paired")
    # The branch key must carry `cand` -- that difference IS the treatment -- and under
    # --same_noise it must NOT, since the control's whole content is that every sibling was
    # handed one draw. Both halves are asserted, because the control silently keeping `cand`
    # would make it a second treatment arm reported as a control.
    assert "[seed, n_replans] if same_noise else [seed, n_replans, cand]" in src, (
        "the branch key changed; it must carry `cand` normally and drop it under "
        "--same_noise, or the control stops controlling for anything")
