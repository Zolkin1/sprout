# Copyright 2026 Zachary Olkin. All rights reserved.

"""The split-continuation selection gain, against synthetic data with known answers.

``oracle_gain.py`` answers "what is CHOOSING a chunk worth at one re-plan", and its output is
meant to bound best-of-K, filtered BC and advantage conditioning at the per-decision level.
A wrong estimator here would not look wrong: the failure mode is a large, clean, plausible
number, because the obvious estimator -- pick the best sibling and report how good it was --
returns the spread of the outcome NOISE and returns it confidently.  So every claim the tool
makes is pinned here against a closed form, in the discipline ``eval/seed_headroom.py``
records the hard way: *"Validate any estimator here against synthetic data with a known answer
before quoting it"*, where three estimators were validated that way and two of the three
failed.

What is pinned:

* the honest arm returns ZERO when the siblings carry no signal, while the hindsight arm
  returns ``sigma * E[max of K]`` -- the two ends of the only distinction this file exists to
  make;
* the honest arm recovers ``rho * sigma_mu * E[max of K]`` on data built to have exactly that
  answer, at three signal-to-noise ratios;
* the perfect-predictor extrapolation ``gain / sqrt(r)`` recovers the TRUE oracle gain, which
  is what licenses quoting it at all (and only under the bivariate-normal model it assumes);
* the score can only come from the column the selection did NOT read -- pinned structurally,
  by making the scoring column constant and requiring an exact zero;
* the within-group permutation null returns zero on signal-free data and rejects on signal;
* the loader DROPS a group that lacks two continuations for every sibling rather than
  analysing a partial one, REFUSES a sibling-0-only file outright, and computes a group's
  position in its episode from the pre-drop re-plans.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loaded BY PATH: `transfer/sim` is not a package, and a bare import would depend on whether
# another suite has already put that directory on `sys.path`.
_spec = importlib.util.spec_from_file_location(
    "_oracle_gain_under_test", os.path.join(_SIM, "oracle_gain.py"))
_og = importlib.util.module_from_spec(_spec)
# Registered BEFORE exec: `@dataclass` resolves its own module out of `sys.modules` to test
# for `KW_ONLY`, and a module that is not there yet fails with an AttributeError inside
# dataclasses rather than anywhere near the cause.
sys.modules[_spec.name] = _og
_spec.loader.exec_module(_og)

_K = 4
#: E[max of 4 i.i.d. standard normals].
_EMAX = 1.0293753730039641


def _synth(n, k, sig_mu, sig_eps, rng):
    """Two independent noisy draws of K siblings whose true means differ by a known amount."""
    mu = rng.normal(0.0, sig_mu, size=(n, k))
    return (mu,
            mu + rng.normal(0.0, sig_eps, size=(n, k)),
            mu + rng.normal(0.0, sig_eps, size=(n, k)))


def test_the_honest_arm_is_zero_without_sibling_signal_and_hindsight_is_not():
    """The whole point of two columns, in one test.

    With ``sig_mu = 0`` no selector can do better than random, so the honest gain must be 0.
    The hindsight arm -- which scores the draw it selected with -- reports a quarter of a
    standard deviation of pure luck as if it were an improvement.
    """
    rng = np.random.default_rng(0)
    _mu, ya, yb = _synth(40000, _K, 0.0, 0.25, rng)
    a = _og.arms(ya, yb, rng)
    assert abs(float(a["gain"].mean())) < 0.006, a["gain"].mean()
    assert float(a["hindsight"].mean()) == pytest.approx(0.25 * _EMAX, abs=0.01)


@pytest.mark.parametrize(("sig_mu", "sig_eps"), [(0.30, 0.30), (0.10, 0.40), (0.40, 0.10)])
def test_the_honest_gain_recovers_its_closed_form(sig_mu, sig_eps):
    """``E[gain] = rho * sig_mu * E[max K]`` with ``rho = sig_mu / sqrt(sig_mu^2+sig_eps^2)``.

    Selecting on one noisy draw regresses the selected sibling's true mean toward zero by
    exactly ``rho``, and the held-out draw pays out that mean.  Three ratios, so a formula
    that happened to match at one signal level cannot pass.
    """
    rng = np.random.default_rng(1)
    _mu, ya, yb = _synth(40000, _K, sig_mu, sig_eps, rng)
    rho = sig_mu / np.sqrt(sig_mu ** 2 + sig_eps ** 2)
    got = float(_og.arms(ya, yb, rng)["gain"].mean())
    assert got == pytest.approx(rho * sig_mu * _EMAX, abs=0.012), got


@pytest.mark.parametrize(("sig_mu", "sig_eps"), [(0.30, 0.30), (0.10, 0.40), (0.40, 0.10)])
def test_the_perfect_predictor_extrapolation_recovers_the_true_ceiling(sig_mu, sig_eps):
    """``gain / sqrt(r)`` must return ``sig_mu * E[max K]``, the gain of a PERFECT predictor.

    This is the one arm of the report that is a model rather than a measurement, so what it
    can be held to is exactness under its own model.  ``r`` is the within-group demeaned
    correlation of the two columns and must equal ``rho^2``; a selector reading one draw
    correlates ``sqrt(r) = rho`` with the conditional mean, and the gain is linear in that.
    """
    rng = np.random.default_rng(2)
    _mu, ya, yb = _synth(40000, _K, sig_mu, sig_eps, rng)
    rho = sig_mu / np.sqrt(sig_mu ** 2 + sig_eps ** 2)
    r = _og.reliability(ya, yb)
    assert r == pytest.approx(rho ** 2, abs=0.02), r
    gain = float(_og.arms(ya, yb, rng)["gain"].mean())
    assert gain / np.sqrt(r) == pytest.approx(sig_mu * _EMAX, abs=0.03)


def test_the_score_can_only_come_from_the_column_the_selection_did_not_read():
    """The guardrail, pinned STRUCTURALLY rather than by a tolerance.

    Make the scoring column constant within each group.  Then the honest gain is
    algebraically zero however strong the selection column's signal is, because the score and
    the baseline are the same number -- and an implementation that leaked the selection column
    into the score would return the hindsight value instead, which here is large.
    """
    rng = np.random.default_rng(3)
    ya = rng.normal(0.0, 1.0, size=(4000, _K))
    yb = np.tile(rng.normal(0.0, 1.0, size=(4000, 1)), (1, _K))
    a = _og.arms(ya, yb, rng)
    assert float(np.abs(a["gain_ab"]).max()) == 0.0
    assert float(a["hindsight"].mean()) > 0.4          # the wrong answer is nowhere near 0
    # The mirror arm is zero for a different reason and it is worth being exact about which:
    # selecting on the constant column is a full TIE, so the pick is uniform and its
    # EXPECTATION over that draw is the group mean of A -- exactly the baseline. Under the
    # integrated tie-break that is an algebraic zero; under the random one it is a mean-zero
    # draw with the spread of A, which is why the signed mean is the thing to test and the
    # mean absolute value is not.
    assert float(np.abs(_og.arms(ya, yb, rng, "expected")["gain_ba"]).max()) == 0.0
    assert float(a["gain_ba"].mean()) == pytest.approx(0.0, abs=0.05)


def test_identical_siblings_give_exactly_zero_which_is_the_same_noise_control():
    """A ``--same_noise`` run has bit-identical siblings, so the gain must be EXACTLY 0.

    Not "small": every sibling's outcome is the same number, so max, mean and any pick
    coincide in floating point.  A control that came back at 1e-9 would mean the two columns
    are not what they claim, which invalidates every treatment number.
    """
    rng = np.random.default_rng(4)
    same = rng.normal(0.0, 1.0, size=(500, 1)) @ np.ones((1, _K))
    a = _og.arms(same, same.copy(), rng)
    assert float(np.abs(a["gain"]).max()) == 0.0
    assert float(np.abs(a["spread"]).max()) == 0.0


def test_a_deterministic_binary_group_gives_exactly_one_minus_one_over_k():
    """``{0,0,0,1}`` in both columns: the pick is right every time, the baseline is 1/K."""
    rng = np.random.default_rng(5)
    det = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (500, 1))
    a = _og.arms(det, det.copy(), rng)
    assert float(a["gain"].mean()) == pytest.approx(1.0 - 1.0 / _K, abs=0.0)


def test_hindsight_and_any_of_k_are_the_same_number_on_one_column():
    """They are reported as two rows and are algebraically identical on column A.

    Pinned so that seeing them agree to the last digit is never mistaken for a copy-paste
    bug, and so a future edit cannot silently make one of them mean something else.
    """
    rng = np.random.default_rng(6)
    _mu, ya, yb = _synth(2000, _K, 0.3, 0.3, rng)
    a = _og.arms(ya, yb, rng)
    hind_a = ya.max(axis=1) - ya.mean(axis=1)
    assert np.allclose(a["any_of_k"], hind_a, atol=0.0)
    # 1e-12 rather than 0: the two sides sum in a different ORDER (mean of a sum against a
    # sum of means), which is a last-bit difference and not a definitional one.
    assert float(a["hindsight"].mean()) == pytest.approx(
        0.5 * (hind_a.mean() + (yb.max(axis=1) - yb.mean(axis=1)).mean()), abs=1e-12)


def test_the_within_group_permutation_null_is_a_real_null():
    """Zero on signal-free data, and it must REJECT when there is signal.

    A null that cannot fire is the failure that retired the Isaac best-of-K harness -- its
    control matched its treatment, so the instrument had no resolving power. Both directions
    are asserted here for that reason.
    """
    rng = np.random.default_rng(7)
    _mu, ya, yb = _synth(4000, _K, 0.0, 0.25, rng)
    p_null, mean_null, _sd = _og.permutation_p(ya, yb, reps=200, rng=rng, tie_break="random")
    assert abs(mean_null) < 0.006, mean_null
    assert p_null > 0.01, p_null
    _mu, ya, yb = _synth(4000, _K, 0.30, 0.30, rng)
    p_sig, _m, _s = _og.permutation_p(ya, yb, reps=200, rng=rng, tie_break="random")
    assert p_sig <= 0.01, p_sig


def test_the_tie_break_draw_only_changes_the_variance_not_the_estimand():
    """``expected`` is the same estimator with the uniform tie draw integrated out."""
    rng = np.random.default_rng(8)
    det = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), (4000, 1))
    yb = (det + np.random.default_rng(9).normal(0, 0.1, det.shape))
    g_rand = float(_og.arms(det, yb, rng, "random")["gain"].mean())
    g_exp = float(_og.arms(det, yb, rng, "expected")["gain"].mean())
    assert g_rand == pytest.approx(g_exp, abs=0.01)
    # And on a full tie the integrated version is EXACTLY zero, which is the claim the
    # "all siblings agree" stratum rests on.
    tie = np.zeros((100, _K))
    assert float(np.abs(_og.arms(tie, yb[:100], rng, "expected")["gain"]).max()) == 0.0


# ======================================================================================
# the loader
# ======================================================================================
def _write_branches(path, groups, *, k=_K, same_noise=False, second_seed_scope="all_siblings",
                    tile=6, speed=1.6, klass="stairs_up"):
    """Write a minimal ``branches.h5``.

    Args:
        path: Output directory.
        groups: ``[(seed, replan, [(branch_verdict, cont_verdict, cont_verdict_b, g_real,
            g_real_b), ...]), ...]``, one tuple per sibling.
        k: Siblings per group, stamped into the attrs.
        same_noise: The control flag.
        second_seed_scope: The design stamp the loader reads.
        tile: Tile id.
        speed: Command speed.
        klass: Course class, so the loader never needs a clip library.
    """
    import h5py

    os.makedirs(path, exist_ok=True)
    cols: dict[str, list] = {n: [] for n in
                             ("seed", "replan", "cand", "branch_verdict", "cont_verdict",
                              "cont_verdict_b", "terminal_in_branch", "g_real", "g_real_b")}
    for seed, replan, sibs in groups:
        for cand, (bv, cv, cvb, ga, gb) in enumerate(sibs):
            cols["seed"].append(seed)
            cols["replan"].append(replan)
            cols["cand"].append(cand)
            cols["branch_verdict"].append(bv)
            cols["cont_verdict"].append(cv)
            cols["cont_verdict_b"].append(cvb)
            cols["terminal_in_branch"].append(bv != "")
            cols["g_real"].append(ga)
            cols["g_real_b"].append(gb)
    with h5py.File(os.path.join(path, "branches.h5"), "w") as f:
        for name, vals in cols.items():
            if name in ("branch_verdict", "cont_verdict", "cont_verdict_b"):
                f.create_dataset(name, data=np.array(vals, dtype=h5py.string_dtype()))
            elif name == "terminal_in_branch":
                f.create_dataset(name, data=np.array(vals, dtype=np.uint8))
            else:
                f.create_dataset(name, data=np.array(vals, dtype=np.float64))
        f.attrs["k"] = k
        f.attrs["tile"] = tile
        f.attrs["speed"] = speed
        f.attrs["same_noise"] = same_noise
        f.attrs["klass"] = klass
        f.attrs["second_seed_scope"] = second_seed_scope
    return path


def _sib(ga, gb, *, ok=True, terminal=""):
    """One sibling row: two rolled continuations, or a partial / terminal one."""
    if terminal:
        return (terminal, terminal, "", float("nan"), float("nan"))
    if not ok:
        return ("", "success", "", ga, float("nan"))
    return ("", "success", "success", ga, gb)


def test_a_group_missing_a_second_continuation_is_dropped_and_counted(tmp_path):
    """The guardrail: exclude a partial group rather than mix it in.

    A group with three of four siblings rolled twice is a different estimator -- the baseline
    is over a different K -- and averaging it in would move the headline with no error.
    """
    d = _write_branches(str(tmp_path / "run"), [
        (0, 0, [_sib(0.5, 0.6), _sib(0.4, 0.4), _sib(0.7, 0.6), _sib(0.3, 0.5)]),
        (0, 4, [_sib(0.5, 0.6), _sib(0.4, 0.4), _sib(0.7, 0.6, ok=False), _sib(0.3, 0.5)]),
    ])
    cell = _og.load_cell(d, outcome="continuous", include_terminal=False)
    assert cell.y_a.shape == (1, _K)
    assert cell.drops["partial_second_continuation"] == 1
    assert cell.n_groups_total == 2


def test_a_sibling_zero_only_file_is_refused_rather_than_analysed(tmp_path):
    """The pre-change design cannot support a selection, and says so instead of returning one.

    Its B column has one entry per group, so ranking on it ranks nothing.  A tool that
    accepted it would report a gain computed from a single sibling as if it were K.
    """
    d = _write_branches(str(tmp_path / "old"), [
        (0, 0, [_sib(0.5, 0.6), _sib(0.4, 0.4, ok=False), _sib(0.7, 0.6, ok=False),
                _sib(0.3, 0.5, ok=False)]),
    ], second_seed_scope="")
    with pytest.raises(SystemExit, match="ONE sibling per group"):
        _og.load_cell(d, outcome="continuous", include_terminal=False)


def test_terminal_siblings_are_dropped_by_default_and_includable_on_the_binary_outcome(tmp_path):
    """The stated bias arm.

    A sibling that reached the goal inside its own chunk has no continuation to roll and no
    continuous outcome, so the headline drops its whole group -- which removes exactly the
    decisive states.  On the BINARY outcome its result carries no continuation variance at
    all, so ``include_terminal`` can restore the group exactly rather than by imputation.
    """
    d = _write_branches(str(tmp_path / "run"), [
        (0, 0, [_sib(0.5, 0.6), _sib(0.4, 0.4), _sib(0.7, 0.6), _sib(0.3, 0.5)]),
        (0, 4, [_sib(0, 0, terminal="success"), _sib(0.4, 0.4), _sib(0.7, 0.6),
                _sib(0.3, 0.5)]),
        (0, 8, [_sib(0, 0, terminal="violation"), _sib(0.4, 0.4), _sib(0.7, 0.6),
                _sib(0.3, 0.5)]),
    ])
    strict = _og.load_cell(d, outcome="binary", include_terminal=False)
    assert strict.y_a.shape == (1, _K)
    loose = _og.load_cell(d, outcome="binary", include_terminal=True)
    assert loose.y_a.shape == (3, _K)
    # The terminal sibling's two columns are its own verdict, and identical -- zero variance.
    assert loose.y_a[1, 0] == 1.0 and loose.y_b[1, 0] == 1.0
    assert loose.y_a[2, 0] == 0.0 and loose.y_b[2, 0] == 0.0
    assert loose.degenerate.tolist() == [False, True, True]
    # The continuous outcome has no such repair: those groups stay dropped either way.
    assert _og.load_cell(d, outcome="continuous", include_terminal=True).y_a.shape == (1, _K)


def test_the_episode_position_is_computed_before_the_drop(tmp_path):
    """A group's place in its episode is a property of the EPISODE, not of what survived.

    Computing it after dropping would move a mid-episode group to "late" whenever the later
    re-plans happened to be the partial ones -- and the late bucket is the one a reader looks
    at to ask whether selection matters more as the robot gets into trouble.
    """
    d = _write_branches(str(tmp_path / "run"), [
        (0, 0, [_sib(0.5, 0.6)] * _K),
        (0, 4, [_sib(0.5, 0.6)] * _K),
        (0, 8, [_sib(0.5, 0.6, ok=False)] * _K),
    ])
    cell = _og.load_cell(d, outcome="continuous", include_terminal=False)
    assert cell.replan.tolist() == [0, 4]
    # Denominator 8, the episode's last PROBED re-plan, not 4, the last surviving one.
    assert cell.ep_frac.tolist() == [0.0, 0.5]


def test_cells_are_pooled_but_a_disagreeing_k_is_refused(tmp_path):
    """A gain over K is not comparable across K, so pooling two of them is refused."""
    a = _write_branches(str(tmp_path / "a"), [(0, 0, [_sib(0.5, 0.6)] * _K)])
    b = _write_branches(str(tmp_path / "b"), [(0, 0, [_sib(0.5, 0.6)] * 2)], k=2,
                        tile=20, speed=0.8, klass="stairs_down")
    c = _write_branches(str(tmp_path / "c"), [(0, 0, [_sib(0.5, 0.6)] * _K)],
                        tile=40, speed=1.6, klass="box")
    ca = _og.load_cell(a, outcome="continuous", include_terminal=False)
    cb = _og.load_cell(b, outcome="continuous", include_terminal=False)
    cc = _og.load_cell(c, outcome="continuous", include_terminal=False)
    st = _og.stack([ca, cc])
    assert st.y_a.shape == (2, _K)
    assert sorted(set(st.cell.tolist())) == ["t40_s1.6", "t6_s1.6"]
    # The bootstrap cluster carries the CELL, not just the seed: every cell is collected at
    # seed_offset 0, so seed alone would pool two different episodes into one cluster and
    # then widen nothing.
    assert np.unique(st.episode).size == 2
    with pytest.raises(SystemExit, match="disagree on K"):
        _og.stack([ca, cb])


def test_the_bootstrap_cluster_is_the_episode_and_it_widens_the_interval(tmp_path):
    """Group-level resampling understates the interval when groups share an episode.

    Built so the correlation is total: every group in an episode carries that episode's own
    gain.  A group bootstrap then sees n independent draws where there are only a handful, and
    returns an interval that is too tight by construction.
    """
    rng = np.random.default_rng(11)
    ep = np.repeat(np.arange(8), 25)
    v = np.repeat(rng.normal(0.0, 1.0, 8), 25)
    lo_g, hi_g = _og.bootstrap_ci(v, reps=500, rng=rng)
    lo_c, hi_c = _og.bootstrap_ci(v, reps=500, rng=rng, cluster=ep.astype(str))
    assert (hi_c - lo_c) > 3 * (hi_g - lo_g)


def test_the_two_reliability_statistics_separate_the_way_they_are_documented_to():
    """Pooled-Pearson against mean-within-group-Spearman, on data built to separate them.

    The earlier label-ceiling number (0.268) is the mean within-group Spearman, and this file
    reports a pooled demeaned Pearson beside it.  They are different statistics on the same
    data, and the report says so; this pins the mechanism rather than the number.  Half the
    groups are pure continuation noise with a small spread, half are near-perfectly reliable
    with a large one -- so weighting a group by its own spread (Pearson) must land far above
    weighting every group equally (Spearman).
    """
    rng = np.random.default_rng(21)
    noise = rng.normal(0.0, 0.02, size=(300, _K))
    junk_a, junk_b = noise, rng.normal(0.0, 0.02, size=(300, _K))
    mu = rng.normal(0.0, 1.0, size=(300, _K))
    good_a = mu + rng.normal(0.0, 0.01, size=(300, _K))
    good_b = mu + rng.normal(0.0, 0.01, size=(300, _K))
    ya = np.vstack([junk_a, good_a])
    yb = np.vstack([junk_b, good_b])
    pooled = _og.reliability(ya, yb)
    mean_rho, n = _og.reliability_spearman(ya, yb)
    assert n == 600
    assert pooled > 0.95, pooled            # the reliable groups carry all the variance
    assert mean_rho < 0.7, mean_rho         # the junk groups get an equal vote


def test_an_empty_same_noise_control_is_a_fail_not_a_pass():
    """A blocking check that no data reached asserts nothing.

    The control's whole job is to prove the two columns are what they claim, so "n = 0, gain
    trivially zero" must not print PASS -- that is the shape of a threshold computed from an
    empty control group, which this tree has been bitten by before.
    """
    assert _og.control_verdict({"n_groups": 0}).startswith("FAIL")
    assert _og.control_verdict({"n_groups": 12, "mean_gain": 0.0,
                                "mean_spread_a": 0.0}) == "PASS"
    assert _og.control_verdict({"n_groups": 12, "mean_gain": 0.02,
                                "mean_spread_a": 0.1}).startswith("FAIL")
    # A gain that rounds to zero while the siblings DID differ is still a fail: under
    # --same_noise the siblings are bit-identical, so a non-zero spread means the branch
    # restore is not exact and every treatment number is invalid.
    assert _og.control_verdict({"n_groups": 12, "mean_gain": 0.0,
                                "mean_spread_a": 1e-9}).startswith("FAIL")
