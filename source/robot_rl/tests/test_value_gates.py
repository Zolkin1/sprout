# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the critic's gates, and in particular pin what each one REFUSES to say.

A single aggregate AUC is the obvious gate and it is close to uninformative here: most states
in a rollout sit on flat approach where nothing has been decided, so a critic that is right
about those and blind at the riser scores well. These three ask sharper questions, and each
has a way of looking healthy while being wrong -- that is what these tests are for.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching import value_gates as vg


# ----------------------------------------------------------------------------------------
# AUC. A constant critic has earned 0.5, not 1.0.
# ----------------------------------------------------------------------------------------

def test_perfect_and_inverted_separation():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([False, False, True, True])
    assert vg.auc(s, y) == pytest.approx(1.0)
    assert vg.auc(-s, y) == pytest.approx(0.0)


def test_a_constant_critic_scores_exactly_one_half():
    """Ties must take mid-ranks. Without that a critic that outputs one number scores 1.0.

    That is not a hypothetical: an untrained head with a saturated bias does exactly this,
    and it would pass an aggregate-AUC gate outright.
    """
    y = np.array([True, False, True, False, False])
    assert vg.auc(np.full(5, 0.42), y) == pytest.approx(0.5)


def test_auc_is_nan_when_a_class_is_missing():
    """Not 0.5. "Nothing to separate" and "separated no better than chance" are different
    claims, and a phase bin holding only survivors is the first."""
    assert np.isnan(vg.auc(np.array([0.1, 0.9]), np.array([False, False])))
    assert np.isnan(vg.auc(np.array([0.1, 0.9]), np.array([True, True])))


# ----------------------------------------------------------------------------------------
# Calibration -- the check that does not presuppose predictability.
# ----------------------------------------------------------------------------------------

def test_a_perfectly_calibrated_but_UNDISCRIMINATING_critic_passes():
    """The whole point of gating on calibration.

    Here every state gets the same 0.3 and 30% of them fail. No critic could separate these
    episodes -- AUC is exactly 0.5 -- and yet the critic is telling the truth about the rate.
    That is a working critic in a region where discrimination is impossible, and a gate that
    only measured AUC would throw it away.
    """
    rng = np.random.default_rng(0)
    p = np.full(4000, 0.3)
    y = rng.random(4000) < 0.3
    cal = vg.calibration(p, y)
    assert cal["ece"] < 0.02
    assert vg.auc(p, y) == pytest.approx(0.5)


def test_an_overconfident_critic_is_caught():
    """Predicts 0.9 on states that fail a third of the time."""
    rng = np.random.default_rng(1)
    p = np.full(2000, 0.9)
    y = rng.random(2000) < 0.33
    cal = vg.calibration(p, y)
    assert cal["ece"] > 0.5
    assert cal["mce"] > 0.5


def test_empty_bins_carry_nan_and_do_not_drag_the_error():
    p = np.concatenate([np.full(100, 0.05), np.full(100, 0.95)])
    y = np.concatenate([np.zeros(100, bool), np.ones(100, bool)])
    cal = vg.calibration(p, y, n_bins=10)
    assert np.isnan(cal["observed"][5])            # nothing landed mid-range
    assert cal["n"].sum() == 200
    assert cal["ece"] < 0.06                        # unaffected by the empty bins


# ----------------------------------------------------------------------------------------
# Discrimination BY PHASE.
# ----------------------------------------------------------------------------------------

def test_the_healthy_signature_is_a_RISING_curve():
    """Chance early, sharp late -- which is what a real critic should look like.

    Flat at 0.5 at the START is expected and correct: nothing has happened yet on the flat
    approach, so nothing is predictable, and a critic claiming otherwise on the first chunk
    has learned which tile it is on rather than what state it is in.
    """
    rng = np.random.default_rng(2)
    n_ep, k = 400, 10
    failed = rng.random(n_ep) < 0.4
    ph, p, y = [], [], []
    for e in range(n_ep):
        for i in range(k):
            frac = i / (k - 1)
            # Signal strength grows with phase; pure noise at frac = 0.
            p.append(frac * (0.9 if failed[e] else 0.1) + (1 - frac) * rng.random())
            ph.append(frac)
            y.append(failed[e])
    out = vg.auc_by_phase(np.array(p), np.array(y), np.array(ph), n_bins=5)
    assert out["auc"][0] < 0.65, "should be near chance on the approach"
    assert out["auc"][-1] > 0.95, "should be sharp at the decisive moment"
    assert np.all(np.diff(out["auc"]) > -0.05), "should rise monotonically here"


def test_a_critic_that_only_learned_the_FAMILY_is_flat_and_high():
    """The failure aggregate AUC cannot see.

    This critic knows each episode's outcome from chunk zero -- it has memorised which tile
    is hard -- so its aggregate AUC is 1.0 and it carries no information about any STATE. The
    phase curve shows it immediately: high and flat, including at phase 0, where nothing has
    happened yet.
    """
    rng = np.random.default_rng(3)
    n_ep, k = 200, 10
    failed = rng.random(n_ep) < 0.4
    p = np.repeat(np.where(failed, 0.9, 0.1), k)
    y = np.repeat(failed, k)
    ph = np.tile(np.linspace(0, 1, k), n_ep)
    out = vg.auc_by_phase(p, y, ph, n_bins=5)
    assert vg.auc(p, y) == pytest.approx(1.0)      # aggregate says "perfect"
    assert out["auc"][0] > 0.95, "and the phase curve exposes it: sharp before anything happened"
    assert np.ptp(out["auc"]) < 0.05


def test_a_phase_bin_with_one_class_is_nan_not_chance():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([False, False, True, True])
    ph = np.array([0.05, 0.05, 0.95, 0.95])        # each bin holds one class only
    out = vg.auc_by_phase(p, y, ph, n_bins=2)
    assert np.isnan(out["auc"]).all()


# ----------------------------------------------------------------------------------------
# Lead time -- the gate that decides whether anything is attributable.
# ----------------------------------------------------------------------------------------

def _episode(uid, p, failed):
    return (np.full(len(p), uid), np.arange(len(p)), np.asarray(p), np.full(len(p), failed))


def test_lead_time_measures_a_SUSTAINED_rise_not_a_spike():
    """A single spike far out is noise, and a first-touch definition would sell it as warning.

    Here P(fail) touches 0.9 at chunk 1, falls back for six chunks, then rises for good at
    chunk 8 of 10. The honest lead is 1, not 8.
    """
    ep, ix, p, y = _episode(0, [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.95], True)
    out = vg.lead_time(p, ep, ix, y, tau=0.5)
    assert out["lead"].tolist() == [1.0]


def test_no_warning_at_all_reads_as_NaN_not_as_zero():
    """A critic that never rises has not given a zero-chunk warning; it has given none.

    The distinction matters because a median over zeros would read as "warns at the last
    moment", which is a much weaker claim than "does not warn".
    """
    ep, ix, p, y = _episode(0, [0.1] * 8, True)
    assert np.isnan(vg.lead_time(p, ep, ix, y, tau=0.5)["lead"][0])


def test_lead_can_be_measured_against_the_FALL_rather_than_the_END():
    """The sharper question, and the one that says whether a chunk is attributable.

    This episode runs 20 chunks but crossed the fall predicate at chunk 10. P(fail) rises for
    good at chunk 6. Against the END that is 14 chunks of warning, which flatters the critic;
    against the FALL it is 4, which is the number that matters -- it is the window in which a
    chunk could still have been the cause.
    """
    p = np.array([0.1] * 6 + [0.9] * 14)
    ep, ix, y = np.zeros(20, int), np.arange(20), np.ones(20, bool)
    assert vg.lead_time(p, ep, ix, y, tau=0.5)["lead"][0] == pytest.approx(13.0)
    ref = np.full(20, 10.0)
    assert vg.lead_time(p, ep, ix, y, tau=0.5, reference=ref)["lead"][0] == pytest.approx(4.0)


def test_a_critic_that_only_fires_AFTER_the_fall_gets_zero_lead():
    """The red flag. If V only rises once the base is already past recovery, the advantage
    arrives after the causal chunk and nothing is attributable to any action."""
    p = np.array([0.1] * 12 + [0.9] * 8)
    ep, ix, y = np.zeros(20, int), np.arange(20), np.ones(20, bool)
    ref = np.full(20, 10.0)
    assert vg.lead_time(p, ep, ix, y, tau=0.5, reference=ref)["lead"][0] == 0.0


def test_only_failures_get_a_lead_time():
    ep = np.concatenate([np.zeros(5, int), np.ones(5, int)])
    ix = np.tile(np.arange(5), 2)
    p = np.concatenate([np.full(5, 0.9), np.full(5, 0.9)])
    y = np.concatenate([np.ones(5, bool), np.zeros(5, bool)])
    assert vg.lead_time(p, ep, ix, y)["episode"].tolist() == [0]


def test_value_traces_sort_by_index_and_carry_the_outcome():
    ep = np.array([7, 7, 7]); ix = np.array([2, 0, 1]); p = np.array([0.3, 0.1, 0.2])
    tr = vg.value_traces(p, ep, ix, np.array([True] * 3))
    assert tr[7]["index"].tolist() == [0, 1, 2]
    assert tr[7]["p_fail"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert tr[7]["failed"] is True


def test_auc_within_stratum_isolates_DIFFICULTY_from_discrimination():
    """A critic that reads tile difficulty alone must collapse toward 0.5 within a tile.

    Two tiles, one easy (mostly successes) and one hard (mostly failures). A score that is
    CONSTANT per tile (pure difficulty, zero within-tile signal) separates the pooled sample
    almost perfectly -- easy-tile scores are all low, hard-tile scores all high -- but must
    read as chance once every comparison is confined to one tile.
    """
    import numpy as np
    from robot_rl.flow_matching.value_gates import auc, auc_within_stratum

    rng = np.random.default_rng(0)
    tile = np.r_[np.zeros(200, int), np.ones(200, int)]
    failed = np.r_[rng.random(200) < 0.1, rng.random(200) < 0.9]      # easy vs hard
    score = np.where(tile == 0, 0.1, 0.9) + rng.normal(0, 1e-6, 400)  # pure difficulty

    assert auc(score, failed) > 0.85                      # confounded: looks highly predictive
    assert abs(auc_within_stratum(score, failed, tile) - 0.5) < 0.05   # correctly collapses


def test_auc_within_stratum_matches_plain_auc_on_ONE_stratum():
    """With a single group the pooled statistic must reduce to ordinary AUC."""
    import numpy as np
    from robot_rl.flow_matching.value_gates import auc, auc_within_stratum

    rng = np.random.default_rng(1)
    score = rng.normal(size=300)
    failed = rng.random(300) < 0.3
    tile = np.zeros(300, dtype=int)
    assert auc_within_stratum(score, failed, tile) == auc(score, failed)


def test_auc_within_stratum_finds_REAL_signal_when_present():
    """The opposite failure mode: a genuinely predictive score must NOT be washed out just
    because it is computed within strata."""
    import numpy as np
    from robot_rl.flow_matching.value_gates import auc_within_stratum

    rng = np.random.default_rng(2)
    tile = rng.integers(0, 5, 400)
    failed = rng.random(400) < 0.3
    score = np.where(failed, rng.normal(0.8, 0.1, 400), rng.normal(0.2, 0.1, 400))
    assert auc_within_stratum(score, failed, tile) > 0.85


# --- difference gates ----------------------------------------------------------------
# An advantage never uses a value LEVEL, so a gate on levels cannot say whether one will
# work. These pin the level/difference distinction, which is the whole reason the gate exists.


def test_a_perfect_predictor_scores_one_on_both_rho_and_ratio():
    """The trivial fixed point: predicting the target exactly gets every difference right."""
    rng = np.random.default_rng(0)
    ep = np.repeat(np.arange(40), 25)
    ix = np.tile(np.arange(25), 40)
    g = rng.normal(size=ep.size)
    out = vg.difference_gates(g.copy(), g, ep, ix, horizons=(1, 4))
    for n, r in out.items():
        assert r["rho"] == pytest.approx(1.0, abs=1e-9), (n, r)
        assert r["ratio"] == pytest.approx(1.0, abs=1e-9), (n, r)


def test_a_level_accurate_but_difference_noisy_value_is_caught():
    """The case the level gates cannot see, and the one the real critic is in.

    A predictor that tracks a slow trend plus independent per-step noise correlates strongly
    with the LEVEL and barely at all with the step-to-step CHANGE -- and its predicted change
    is far larger than the real one, which is exactly what inflates an advantage built from
    it.
    """
    rng = np.random.default_rng(1)
    ep = np.repeat(np.arange(60), 30)
    ix = np.tile(np.arange(30), 60)
    trend = np.tile(np.linspace(0.0, 1.0, 30), 60)          # the slow, predictable part
    g = trend + 0.02 * rng.normal(size=ep.size)
    pred = trend + 0.30 * rng.normal(size=ep.size)          # right level, wrong differences
    assert vg.spearman(pred, g) > 0.6, "fixture must be level-accurate"
    out = vg.difference_gates(pred, g, ep, ix, horizons=(1,))
    assert out[1]["rho"] < 0.25, out[1]
    assert out[1]["ratio"] > 3.0, ("the inflation of the predicted change is the number that "
                                   "diagnoses this, and it did not fire")


def test_differences_are_taken_WITHIN_an_episode():
    """A pair straddling a boundary compares two unrelated trajectories.

    Built so the across-boundary pairing would be strongly ANTI-correlated: each episode
    ramps up, so the last row of one against the first of the next is a large negative step
    where every within-episode step is positive.
    """
    ep = np.repeat(np.arange(30), 10)
    ix = np.tile(np.arange(10), 30)
    # Steps must VARY, or every within-episode difference is the same number and Spearman on
    # a constant is undefined -- which is the honest answer, and not what this test is about.
    step = np.random.default_rng(2).uniform(0.5, 1.5, 10)
    g = np.tile(np.cumsum(step), 30)
    out = vg.difference_gates(g.copy(), g, ep, ix, horizons=(1,))
    # 30 episodes x 9 within-episode pairs; a leaky implementation would find 299.
    assert out[1]["n_pairs"] == 30 * 9, out[1]["n_pairs"]
    assert out[1]["rho"] == pytest.approx(1.0, abs=1e-9)


def test_a_horizon_longer_than_every_episode_is_omitted_not_nan():
    """Reported as absent rather than as a NaN that a reader might average."""
    ep = np.repeat(np.arange(5), 4)
    ix = np.tile(np.arange(4), 5)
    g = np.arange(20.0)
    out = vg.difference_gates(g.copy(), g, ep, ix, horizons=(1, 8))
    assert 1 in out and 8 not in out, sorted(out)
