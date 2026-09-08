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
    LABEL_NEGATIVE, LABEL_NULL, LABEL_POSITIVE, binarize, combined_n_step_advantage,
    label_from_name, n_step_advantage, normalized_returns,
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


# ----------------------------------------------------------------------------------------
# The QUALITY target: undoing the recording's one-row shift, and the return it feeds.
# ----------------------------------------------------------------------------------------
def _spec():
    """The shared tracking contract, via the torch twin robot_rl is allowed to import."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import TrackingErrorSpec

    return TrackingErrorSpec()


def _blocks(uid, ridx, lateral):
    """Per-window track_err_prev blocks with a chosen lateral error per row."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import NUM_FIXED_CHANNELS

    nj, k = 29, 12
    b = np.zeros((len(uid), k, NUM_FIXED_CHANNELS + nj))
    b[:, :, 1] = np.asarray(lateral)[:, None]
    return b


def test_the_reward_shift_reads_the_NEXT_rows_block():
    """Row i's action is scored by row i+1's block, and the last row has no reward.

    A window is written at the re-plan that PRODUCES its fm_plan, whose consequences have not
    happened yet. Getting this shift wrong is an off-by-one-CHUNK: it fits, it trains, and it
    yields a plausible weak critic rather than an error.
    """
    from robot_rl.flow_matching import advantage as adv
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import chunk_reward
    import torch

    uid = np.array([0, 0, 0, 1, 1])
    ridx = np.array([0, 1, 2, 0, 1])
    lateral = np.array([0.0, 0.01, 0.02, 0.0, 0.05])   # row 0 of each episode is all-zero
    blocks = _blocks(uid, ridx, lateral)
    r, ok = adv.chunk_rewards_from_track_err(blocks, uid, ridx, _spec())

    per_row = chunk_reward(torch.as_tensor(blocks, dtype=torch.float64), _spec()).numpy()
    # episode 0: rows 0,1 get rows 1,2's blocks; row 2 has none.
    assert ok.tolist() == [True, True, False, True, False]
    assert r[0] == pytest.approx(per_row[1])
    assert r[1] == pytest.approx(per_row[2])
    # A larger error must score LOWER, or the shift is inverted.
    assert r[0] > r[1]
    # ...and the all-zero row-0 block is never used as a reward.
    assert not np.isclose(r[0], per_row[0])


def test_the_return_is_length_independent_on_arrival_and_decays_on_failure():
    """G = 1.0 for flawless tracking however long it took; a failure discounts toward 0.

    Absorbing on a NON-failure would punish arriving quickly, making a short flawless run
    score far below a long one for reasons entirely about episode length.
    """
    from robot_rl.flow_matching import advantage as adv

    for m in (2, 5, 40):
        uid = np.zeros(m, dtype=np.int64)
        ridx = np.arange(m)
        r, ok = np.ones(m), np.ones(m, dtype=bool)
        g_ok, _ = adv.normalized_returns(r, ok, uid, ridx, np.zeros(m, bool), gamma=0.95)
        assert np.allclose(g_ok, 1.0), f"arrival at m={m} did not score 1.0: {g_ok}"
        g_bad, _ = adv.normalized_returns(r, ok, uid, ridx, np.ones(m, bool), gamma=0.95)
        assert (g_bad < 1.0).all()
        # Nearer the failure is worth less.
        assert g_bad[-1] < g_bad[0]


def test_the_return_stays_in_the_declared_support_and_flags_unusable_episodes():
    """G in [0, 1] -- what lets the quality head DECLARE its support instead of fitting it."""
    from robot_rl.flow_matching import advantage as adv

    rng = np.random.default_rng(0)
    m = 30
    uid = np.zeros(m, dtype=np.int64)
    ridx = np.arange(m)
    r = rng.random(m)
    ok = np.ones(m, dtype=bool)
    for failed in (True, False):
        g, gok = adv.normalized_returns(r, ok, uid, ridx, np.full(m, failed), gamma=0.9)
        assert bool(((g[gok] >= 0.0) & (g[gok] <= 1.0)).all())

    # A one-window episode has no next block, so no reward and therefore no target -- flagged
    # rather than zero-filled, since 0 is the worst possible return.
    g, gok = adv.normalized_returns(np.zeros(1), np.zeros(1, bool), np.zeros(1, dtype=np.int64),
                                    np.zeros(1, dtype=np.int64), np.zeros(1, bool), gamma=0.95)
    assert not gok.any()

    with pytest.raises(ValueError, match="gamma"):
        adv.normalized_returns(r, ok, uid, ridx, np.zeros(m, bool), gamma=1.0)


def test_the_combined_advantage_at_a_long_horizon_IS_the_monte_carlo_one():
    """``w_fail = 0`` with ``n`` past the longest episode must equal ``G - G_hat`` exactly.

    That is the identity the filtered-BC label rests on: ``A_MC = G_t - V(s_t)`` is only an
    advantage if its ``G`` is the quantity the baseline was fitted to predict, which is
    :func:`normalized_returns`'. Two things used to break it, and both are invisible in the
    output -- the reward sum included each episode's placeholder final row, and the
    non-failure tail closed on ``G_hat[-1]`` (a prediction) where ``normalized_returns``
    closes on ``r[-1]`` (the last measurement). They agreed on FAILED episodes, where the
    tail is 0 either way, so half the data looked correct.
    """
    rng = np.random.default_rng(5)
    uid, ri, rew, fail, ok = [], [], [], [], []
    for e in range(9):
        m = int(rng.integers(3, 12))
        f = bool(e % 3 == 0)
        uid += [e] * m
        ri += list(range(m))
        rew += list(rng.uniform(0.1, 1.0, size=m - 1)) + [0.0]   # the placeholder last row
        fail += [f] * m
        ok += [True] * (m - 1) + [False]
    order = rng.permutation(len(uid))                            # rows need not arrive sorted
    uid = np.asarray(uid)[order]
    ri = np.asarray(ri)[order]
    rew = np.asarray(rew)[order]
    fail = np.asarray(fail)[order]
    ok = np.asarray(ok)[order]
    gam = 0.95

    g, g_ok = normalized_returns(rew, ok, uid, ri, fail, gamma=gam)
    q = g + rng.normal(0.0, 0.05, size=g.size)                   # a plausible baseline
    a, _ = combined_n_step_advantage(np.zeros(g.size), q, rew, uid, ri, fail, 500,
                                     gamma=gam, w_fail=0.0, valid=ok)
    assert np.abs(a[g_ok] - (g - q)[g_ok]).max() < 1e-12

    # And the survival control arm is untouched by any of it.
    p = rng.uniform(0.0, 1.0, size=g.size)
    s1, _ = combined_n_step_advantage(p, q, rew, uid, ri, fail, 4, gamma=gam,
                                      w_fail=1.0, valid=ok)
    s2, _ = n_step_advantage(p, uid, ri, fail, 4)
    assert np.abs(s1 - s2).max() == 0.0


def test_the_cut_can_be_stratified_beyond_the_family():
    """``strata`` must make the positive rate flat in the stratum, not just in the family.

    A Monte-Carlo advantage drifts with position in the episode, so a family-only percentile
    keeps disproportionately many chunks from wherever the drift is high. The synthetic case
    here is that drift and nothing else: one family, an advantage that rises with the bin.
    """
    rng = np.random.default_rng(6)
    n_per = 4000
    bins = np.repeat(np.arange(5), n_per)
    adv = rng.normal(size=bins.size) * 0.1 + bins * 0.5      # a strong positional trend
    fam = np.zeros(bins.size, dtype=np.int64)

    flat, _ = binarize(adv, fam, percentile=50.0, null_spread=0.0)
    strat, rep = binarize(adv, fam, strata=bins, percentile=50.0, null_spread=0.0)

    rate_flat = [float((flat[bins == b] == LABEL_POSITIVE).mean()) for b in range(5)]
    rate_str = [float((strat[bins == b] == LABEL_POSITIVE).mean()) for b in range(5)]
    # Family-only: the trend decides everything -- the low bins are all negative.
    assert rate_flat[0] < 0.02 and rate_flat[-1] > 0.98
    # Stratified: every bin keeps its own half.
    assert all(abs(r - 0.5) < 0.02 for r in rate_str), rate_str
    # And the report says which grouping produced it, rather than looking family-keyed.
    assert all(isinstance(k, str) and "stratum=" in k for k in rep)


def test_the_success_tail_can_average_several_chunks_and_defaults_to_one():
    """``tail_chunks`` must average the last k rewards on a success, and change nothing else.

    Three properties, because each can break independently:

    * the DEFAULT is 1, reproducing every artifact built before the argument existed -- the
      tail is stamped in no old checkpoint, so a changed default would score a fitted critic
      against a target it never saw;
    * a FAILED episode is untouched at any k (its tail is absorbing at 0);
    * the tail really is the mean of the last k, which is checkable in closed form: the last
      surviving chunk's ``G`` is ``(1-g) R_last + g * mean(last k)``.
    """
    uid = np.zeros(6, dtype=np.int64)
    ri = np.arange(6)
    rew = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.0])   # last row is the placeholder
    ok = np.array([True] * 5 + [False])
    g = 0.9

    base, _ = normalized_returns(rew, ok, uid, ri, np.zeros(6, bool), gamma=g)
    k1, _ = normalized_returns(rew, ok, uid, ri, np.zeros(6, bool), gamma=g, tail_chunks=1)
    assert np.array_equal(base, k1), "the default must be tail_chunks=1"

    k3, ok3 = normalized_returns(rew, ok, uid, ri, np.zeros(6, bool), gamma=g, tail_chunks=3)
    want_tail = float(np.mean([0.6, 0.8, 0.9]))
    assert k3[4] == pytest.approx((1 - g) * 0.9 + g * want_tail)
    # Direction, stated so it cannot be misread: on a RISING reward trace the average pulls
    # the tail DOWN toward the earlier, worse chunks (0.767 against the final 0.9); on a
    # falling one it lifts it. The purpose is variance, not a directional correction.
    assert k3[4] < k1[4]
    assert k1[4] == pytest.approx(0.9), "at k=1 the last chunk's G IS its own reward"

    # A failure is absorbing at 0 regardless of k.
    f1, _ = normalized_returns(rew, ok, uid, ri, np.ones(6, bool), gamma=g, tail_chunks=1)
    f3, _ = normalized_returns(rew, ok, uid, ri, np.ones(6, bool), gamma=g, tail_chunks=3)
    assert np.array_equal(f1, f3)

    # An episode shorter than k averages what it has rather than reaching past its start.
    su, sr, so = np.zeros(3, np.int64), np.array([0.4, 0.8, 0.0]), np.array([True, True, False])
    short, _ = normalized_returns(sr, so, su, np.arange(3), np.zeros(3, bool),
                                  gamma=g, tail_chunks=9)
    assert short[1] == pytest.approx((1 - g) * 0.8 + g * float(np.mean([0.4, 0.8])))


def test_the_combined_advantage_tracks_the_same_tail():
    """``combined_n_step_advantage`` must use the SAME tail, or ``A_MC`` stops being one.

    Its baseline is a head fitted to ``normalized_returns``' target, so the two tails have to
    agree for ``G - G_hat`` to be a difference of like quantities. At a long horizon the
    identity is exact, which is what this checks -- at ``tail_chunks=3`` on both sides.
    """
    rng = np.random.default_rng(11)
    uid, ri, rew, fail, ok = [], [], [], [], []
    for e in range(7):
        m = int(rng.integers(4, 11))
        f = bool(e % 2)
        uid += [e] * m
        ri += list(range(m))
        rew += list(rng.uniform(0.1, 1.0, size=m - 1)) + [0.0]
        fail += [f] * m
        ok += [True] * (m - 1) + [False]
    uid, ri = np.asarray(uid), np.asarray(ri)
    rew, fail, ok = np.asarray(rew), np.asarray(fail), np.asarray(ok)

    g, g_ok = normalized_returns(rew, ok, uid, ri, fail, gamma=0.95, tail_chunks=3)
    q = g + rng.normal(0.0, 0.05, size=g.size)
    a, _ = combined_n_step_advantage(np.zeros(g.size), q, rew, uid, ri, fail, 500,
                                     gamma=0.95, w_fail=0.0, valid=ok, tail_chunks=3)
    assert np.abs(a[g_ok] - (g - q)[g_ok]).max() < 1e-12


def test_label_advantage_writes_under_quality_only():
    """The sidecar write must not be nested inside the G2 gate's `else`.

    It was, and the failure was silent in the worst way: a ``--quality_only`` fit skips the G2
    gate (that gate bands by ``P(fail|s_t)`` and the survival head is untrained), and the write
    went with it -- so the run printed every statistic, returned 0, and produced no
    ``advantage.npz``.  A pipeline reading it then either refused or, if a previous run had left
    one, trained on a STALE label with nothing anywhere saying so.

    Checked structurally rather than by running the script, which needs a fitted
    ``predictions.npz``: the ``np.savez`` must not be a descendant of the ``if quality_only``
    statement in either direction.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[3] / "scripts/flow_matching/label_advantage.py"
    tree = ast.parse(src.read_text())

    def is_quality_only_test(node: ast.If) -> bool:
        return isinstance(node.test, ast.Name) and node.test.id == "quality_only"

    gates = [n for n in ast.walk(tree) if isinstance(n, ast.If) and is_quality_only_test(n)]
    assert gates, "the `if quality_only:` gate is gone; re-check where the write sits"

    def saves(node) -> bool:
        return any(isinstance(c, ast.Attribute) and c.attr == "savez"
                   for c in ast.walk(node))

    for gate in gates:
        for branch, name in ((gate.body, "body"), (gate.orelse, "orelse")):
            for stmt in branch:
                assert not saves(stmt), (
                    f"np.savez is inside the `if quality_only` {name}; a quality_only fit "
                    "would write no advantage.npz while still returning 0")

    # And it must actually still be there at all.
    assert any(isinstance(n, ast.Attribute) and n.attr == "savez" for n in ast.walk(tree)), \
        "label_advantage no longer writes a sidecar"
