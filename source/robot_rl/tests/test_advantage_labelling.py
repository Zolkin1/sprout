# Copyright 2026 Zachary Olkin. All rights reserved.

"""The advantage label: the n-step arithmetic, the per-family binarization, and the join.

The join is the one worth a test that would otherwise never be written.  ``fit_value.py``
drops censored episodes, so the label sidecar is SHORTER than the window file it describes;
a positional join therefore shifts every label by the number of censored windows ahead of
it, silently, into an array of exactly the right shape and dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching.advantage import (
    LABEL_NEGATIVE, LABEL_NULL, LABEL_POSITIVE, binarize, label_from_name, n_step_advantage,
)
from robot_rl.flow_matching.rollout_dataset import join_labels


def _episode(uid: int, p: list[float], failed: bool):
    """Build the four parallel arrays for one episode."""
    n = len(p)
    return (np.array(p), np.full(n, uid), np.arange(n), np.full(n, failed))


def test_interior_advantage_is_the_n_step_difference():
    """``A = P(fail|s_t) - P(fail|s_{t+n})`` wherever the look-ahead lands inside."""
    p, uid, ridx, failed = _episode(0, [0.5, 0.4, 0.3, 0.2, 0.1, 0.05], failed=False)
    a, boot = n_step_advantage(p, uid, ridx, failed, n=2)
    assert a[0] == pytest.approx(0.5 - 0.3)
    assert a[3] == pytest.approx(0.2 - 0.05)
    assert not boot[:4].any()


def test_the_tail_bootstraps_on_the_REALIZED_outcome():
    """Past the last recorded re-plan the future value is the outcome, not a prediction.

    Those rows are the ones nearest the terminal and so the most informative; dropping them
    would discard exactly the chunks whose consequence is known.
    """
    p, uid, ridx, failed = _episode(0, [0.2, 0.3, 0.9], failed=True)
    a, boot = n_step_advantage(p, uid, ridx, failed, n=2)
    assert boot[1] and boot[2] and not boot[0]
    assert a[2] == pytest.approx(0.9 - 1.0)     # it failed: V(s_{t+n}) = -1
    p2, uid2, ridx2, ok = _episode(1, [0.2, 0.3, 0.1], failed=False)
    a2, _ = n_step_advantage(p2, uid2, ridx2, ok, n=2)
    assert a2[2] == pytest.approx(0.1 - 0.0)    # it arrived: V(s_{t+n}) = 0


def test_episodes_do_not_leak_into_each_other():
    """The look-ahead is per episode; row t+n of a DIFFERENT episode is not a future."""
    p = np.array([0.5, 0.5, 0.9, 0.9])
    uid = np.array([0, 0, 1, 1])
    ridx = np.array([0, 1, 0, 1])
    failed = np.array([False, False, False, False])
    a, boot = n_step_advantage(p, uid, ridx, failed, n=1)
    # Episode 0's last row bootstraps on ITS outcome (0.5 - 0.0), not on episode 1's 0.9.
    assert a[1] == pytest.approx(0.5)
    assert boot[1] and boot[3]


def test_rows_need_not_arrive_sorted():
    """The table is grouped internally, so a shuffled file gives the same answer."""
    p, uid, ridx, failed = _episode(7, [0.5, 0.4, 0.3, 0.2], failed=False)
    a_sorted, _ = n_step_advantage(p, uid, ridx, failed, n=1)
    perm = np.array([2, 0, 3, 1])
    a_shuf, _ = n_step_advantage(p[perm], uid[perm], ridx[perm], failed[perm], n=1)
    assert a_shuf == pytest.approx(a_sorted[perm])


def test_a_family_with_no_spread_is_null_not_negative():
    """A family that barely ever fails carries no preference, so it must not be thresholded.

    This is the ``flat`` case. Its advantages are a cluster of near-zeros; a blind percentile
    would label the lower third negative and teach the model that competent flat walking is
    bad.
    """
    rng = np.random.default_rng(0)
    flat = rng.normal(0.0, 1e-3, 500)                    # no spread
    stairs = rng.normal(0.0, 0.30, 500)                  # real spread
    adv = np.r_[flat, stairs]
    fam = np.r_[np.zeros(500, int), np.ones(500, int)]
    labels, report = binarize(adv, fam, percentile=35.0, null_spread=0.02)
    assert (labels[:500] == LABEL_NULL).all()
    assert report[0]["null"] and not report[1]["null"]
    # The family WITH spread is split near the requested percentile.
    assert 0.25 < float((labels[500:] == LABEL_NEGATIVE).mean()) < 0.45


def test_the_threshold_is_per_family():
    """A family whose advantages are uniformly higher must not be labelled positive wholesale.

    A global percentile does exactly that, which is why the threshold is within family.
    """
    easy = np.linspace(0.4, 0.6, 400)      # every value above every value of `hard`
    hard = np.linspace(-0.6, -0.4, 400)
    labels, _ = binarize(np.r_[easy, hard], np.r_[np.zeros(400, int), np.ones(400, int)],
                         percentile=50.0, null_spread=0.0)
    for half in (labels[:400], labels[400:]):
        assert (half == LABEL_POSITIVE).mean() == pytest.approx(0.5, abs=0.01)


def test_the_absolute_floor_is_OFF_by_default_and_vetoes_when_set():
    """The floor is opt-in, because at 0.0 it labels half of a CLEAN episode negative.

    Over a successful episode the advantages telescope to a small total spread across ~25
    chunks, so an individual chunk's SIGN is mostly noise. The within-family percentile is
    the calibrated statement; the floor is a veto for a family believed hopeless.
    """
    adv = np.linspace(-0.5, -0.1, 200)
    fam = np.zeros(200, int)
    default, _ = binarize(adv, fam, percentile=35.0, null_spread=0.0)
    assert (default == LABEL_POSITIVE).mean() == pytest.approx(0.65, abs=0.01)
    vetoed, _ = binarize(adv, fam, percentile=35.0, null_spread=0.0, absolute_floor=0.0)
    assert (vetoed == LABEL_NEGATIVE).all()


def test_join_is_on_the_KEY_not_the_POSITION():
    """The sidecar is shorter than the window file, and the missing rows are interior.

    Positionally this join is off by one from row 1 onwards; on the key it is exact, and the
    unmatched row comes back ``null`` rather than inheriting a neighbour's label.
    """
    uid = np.array([0, 0, 0, 1, 1])
    ridx = np.array([0, 1, 2, 0, 1])
    # Row 1 (episode 0, re-plan 1) was censored and has no label.
    side = {"label": np.array([LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_POSITIVE,
                               LABEL_NEGATIVE], dtype=np.int8),
            "episode_uid": np.array([0, 0, 1, 1]),
            "replan_index": np.array([0, 2, 0, 1])}
    got = join_labels(side, uid, ridx)
    assert got.tolist() == [LABEL_POSITIVE, LABEL_NULL, LABEL_NEGATIVE,
                            LABEL_POSITIVE, LABEL_NEGATIVE]


def test_join_refuses_a_sidecar_whose_key_is_not_unique():
    """Two labels for one chunk means the join is not a function; whichever won would be luck."""
    side = {"label": np.array([0, 1], dtype=np.int8),
            "episode_uid": np.array([3, 3]), "replan_index": np.array([1, 1])}
    with pytest.raises(ValueError, match="duplicate"):
        join_labels(side, np.array([3]), np.array([1]))


def test_label_names_round_trip_and_an_unknown_one_raises():
    """A silent default here would make the deliverable and its control indistinguishable."""
    assert label_from_name("negative") == LABEL_NEGATIVE
    assert label_from_name("positive") == LABEL_POSITIVE
    assert label_from_name("null") == LABEL_NULL
    with pytest.raises(ValueError):
        label_from_name("good")
