# Copyright 2026 Zachary Olkin. All rights reserved.

"""AWR's plumbing: the per-sample weight, and the float join that feeds it.

Three things here are silent when wrong, and none of them raises:

* a weight applied to the WRONG axis -- scaling one of a window's ``F*D`` entries rather
  than the window -- still produces a finite loss that falls;
* ``flow_mse`` computed on the weighted tensor makes a weighted run's curve incomparable
  with the unweighted one it exists to be compared against, while looking entirely normal;
* a float sidecar joined POSITIONALLY, or with its misses filled by ``0.0``, gives every
  unlabelled row ``exp(0) = 1`` -- average influence over the update, from a row the critic
  never scored.

So each is tested as an identity against a hand-computed answer, not as a smoke test.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_rl.flow_matching.advantage import LABEL_NULL, binarize, group_ids  # noqa: E402
from robot_rl.flow_matching.flow import flow_matching_loss  # noqa: E402
from robot_rl.flow_matching.rollout_dataset import join_labels, join_values  # noqa: E402


class _StubFlowModel(torch.nn.Module):
    """A deterministic stand-in: one linear map, so ``err`` varies across the batch."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x_tau, tau, history, extero, cmd_tokens=None,
                history_null_mask=None, advantage=None):
        """Return a velocity of the right shape that depends on the input."""
        return self.lin(x_tau)


def _batch(b: int = 5, f: int = 3, d: int = 4):
    """One fixed (model, inputs) tuple for the loss tests."""
    torch.manual_seed(0)
    model = _StubFlowModel(d)
    x1 = torch.randn(b, f, d)
    hist = torch.randn(b, 2, d)
    ext = torch.randn(b, 6)
    return model, x1, hist, ext


def _loss(model, x1, hist, ext, w=None, seed: int = 7):
    """Run the loss under a pinned RNG so two calls share their tau/x0 draw."""
    torch.manual_seed(seed)
    return flow_matching_loss(model, x1, hist, ext, sample_weight=w,
                              blocks=["base_pose"])


def test_the_weighted_loss_is_the_weighted_mean_of_the_per_sample_losses():
    """The identity that pins BOTH the axis the weight acts on and the normalization."""
    model, x1, hist, ext = _batch()
    b = x1.shape[0]
    # A one-hot weight isolates one sample's own loss, because the reduction is
    # self-normalized: sum(w * l) / sum(w) with w = e_i is exactly l_i.
    per = []
    for i in range(b):
        w = torch.zeros(b)
        w[i] = 1.0
        per.append(float(_loss(model, x1, hist, ext, w)["loss"]))
    per = np.array(per)

    w = torch.tensor([3.0, 1.0, 0.5, 2.0, 0.25])
    got = float(_loss(model, x1, hist, ext, w)["loss"])
    want = float((w.numpy() * per).sum() / w.numpy().sum())
    assert got == pytest.approx(want, rel=1e-5), (got, want)

    # And uniform weights must reproduce the plain mean, i.e. the unweighted arm.
    flat = float(_loss(model, x1, hist, ext, torch.ones(b))["loss"])
    assert flat == pytest.approx(float(per.mean()), rel=1e-5)
    assert flat == pytest.approx(float(_loss(model, x1, hist, ext)["loss"]), rel=1e-5)


def test_flow_mse_is_reported_unweighted():
    """A weighted run's curve has to stay comparable with an unweighted one's."""
    model, x1, hist, ext = _batch()
    plain = _loss(model, x1, hist, ext)
    skewed = _loss(model, x1, hist, ext,
                   torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0]))
    assert float(skewed["flow_mse"]) == pytest.approx(float(plain["flow_mse"]), rel=1e-6)
    # ... while the quantity actually optimised DID move, or the weight did nothing.
    assert float(skewed["loss"]) != pytest.approx(float(plain["loss"]), rel=1e-3)


def test_an_all_zero_weight_batch_does_not_return_nan():
    """Reachable at a sharp beta once the clip has moved the mass onto a few rows."""
    model, x1, hist, ext = _batch()
    out = _loss(model, x1, hist, ext, torch.zeros(x1.shape[0]))
    assert np.isfinite(float(out["loss"])), "an empty-weight batch produced a NaN loss"


def test_the_weight_scales_the_window_not_its_entries():
    """Doubling every weight must leave the loss unchanged -- it is self-normalized."""
    model, x1, hist, ext = _batch()
    w = torch.tensor([3.0, 1.0, 0.5, 2.0, 0.25])
    a = float(_loss(model, x1, hist, ext, w)["loss"])
    b = float(_loss(model, x1, hist, ext, w * 10.0)["loss"])
    assert a == pytest.approx(b, rel=1e-6)


def test_combining_a_sample_weight_with_the_pinned_axis_loss_is_refused():
    """`pinned_axis_flow_loss` reduces internally, so the weight would vanish silently."""
    model, x1, hist, ext = _batch()
    mask = torch.ones(x1.shape[0], x1.shape[1], dtype=torch.bool)
    with pytest.raises(ValueError, match="pinned-axis"):
        flow_matching_loss(model, x1, hist, ext, sample_weight=torch.ones(x1.shape[0]),
                           anchor_mask=mask, pinned_axis_weight=5.0, blocks=["base_pose"])


# --- the float join ------------------------------------------------------------------


def _sidecar(uids, ridx, adv):
    return {"episode_uid": np.asarray(uids), "replan_index": np.asarray(ridx),
            "advantage": np.asarray(adv, dtype=np.float32),
            "label": np.zeros(len(uids), dtype=np.int8)}


def test_a_missing_row_comes_back_nan_and_flagged_not_zero():
    """0.0 is an ordinary advantage; as a filler it means weight exp(0) = 1."""
    side = _sidecar([1, 1, 2], [0, 1, 0], [0.5, -0.5, 0.25])
    uid = np.array([1, 1, 2, 9])       # 9 is not in the sidecar
    ridx = np.array([0, 1, 0, 0])
    vals, hit = join_values(side, "advantage", uid, ridx)
    assert hit.tolist() == [True, True, True, False]
    assert np.isnan(vals[3]), "a miss was filled rather than flagged"
    assert vals[:3].tolist() == pytest.approx([0.5, -0.5, 0.25])


def test_the_join_is_by_key_not_by_position():
    """The sidecar is SHORTER than the file (censored episodes are dropped upstream)."""
    side = _sidecar([2, 3], [0, 0], [1.0, 2.0])
    uid = np.array([1, 2, 3])          # episode 1 has no sidecar row
    ridx = np.array([0, 0, 0])
    vals, hit = join_values(side, "advantage", uid, ridx)
    assert hit.tolist() == [False, True, True]
    # Positionally this would have read 1.0 into episode 1 and 2.0 into episode 2.
    assert vals[1] == pytest.approx(1.0) and vals[2] == pytest.approx(2.0)


def test_the_value_join_and_the_label_join_agree_row_for_row():
    """They must resolve the same rows, or a weight lands on a different chunk."""
    rng = np.random.default_rng(0)
    uid = rng.integers(0, 50, 400)
    ridx = rng.integers(0, 30, 400)
    key = np.unique(np.stack((uid, ridx), 1), axis=0)
    keep = key[rng.permutation(len(key))[: len(key) // 2]]
    side = {"episode_uid": keep[:, 0], "replan_index": keep[:, 1],
            "advantage": rng.normal(size=len(keep)).astype(np.float32),
            "label": np.ones(len(keep), dtype=np.int8)}
    labs = join_labels(side, uid, ridx)
    _, hit = join_values(side, "advantage", uid, ridx)
    assert ((labs != LABEL_NULL) == hit).all()


def test_a_duplicate_key_is_refused_by_the_value_join_too():
    """Otherwise whichever row won would be arbitrary."""
    side = _sidecar([1, 1], [0, 0], [1.0, 2.0])
    with pytest.raises(ValueError, match="duplicate"):
        join_values(side, "advantage", np.array([1]), np.array([0]))


def test_a_missing_column_names_itself():
    """Asking for a column the sidecar predates must not read as 'all rows missing'."""
    side = _sidecar([1], [0], [1.0])
    with pytest.raises(KeyError, match="stratum"):
        join_values(side, "stratum", np.array([1]), np.array([0]))


# --- the grouping AWR standardizes over ------------------------------------------------


def test_group_ids_reproduces_the_cells_binarize_cuts_within():
    """AWR centers per cell, so its partition must be binarize's, not an approximation."""
    rng = np.random.default_rng(1)
    n = 600
    fam = rng.integers(0, 3, n)
    strata = rng.integers(0, 4, n)
    adv = rng.normal(size=n)
    _, per_group = binarize(adv, fam, strata=strata, percentile=70.0, null_spread=0.0)
    gid = group_ids(fam, strata)
    assert len(per_group) == np.unique(gid).size == 12
    # Every cell binarize reported must be exactly one group id, and vice versa.
    for f in range(3):
        for s in range(4):
            sel = (fam == f) & (strata == s)
            assert np.unique(gid[sel]).size == 1
            assert per_group[f"fam={f}|stratum={s}"]["n"] == int(sel.sum())


def test_group_ids_without_strata_is_the_family_alone():
    """The historical grouping, which pre-stratum sidecars were labelled under."""
    fam = np.array([7, 7, 3, 3, 9])
    gid = group_ids(fam)
    assert np.unique(gid).size == 3
    assert (gid[0] == gid[1]) and (gid[2] == gid[3]) and gid[0] != gid[2]


def test_global_normalization_lets_a_low_weight_batch_contribute_less():
    """The two reductions differ in exactly the way that decides AWR's sharp limit.

    Under `batch` a batch is self-normalized, so its step is full size however small its
    weights are -- `beta -> 0` converges to best-of-BATCH, not to filtered BC. Under
    `global` the same batch contributes proportionally less, which is what makes the sharp
    limit filtered BC at a shrinking percentile.
    """
    model, x1, hist, ext = _batch()
    b = x1.shape[0]
    small = torch.full((b,), 0.01, dtype=torch.float64)
    ones = torch.ones(b, dtype=torch.float64)
    torch.manual_seed(7)
    batch_small = float(flow_matching_loss(model, x1, hist, ext, sample_weight=small,
                                           weight_reduction="batch",
                                           blocks=["base_pose"])["loss"])
    torch.manual_seed(7)
    batch_ones = float(flow_matching_loss(model, x1, hist, ext, sample_weight=ones,
                                          weight_reduction="batch",
                                          blocks=["base_pose"])["loss"])
    assert batch_small == pytest.approx(batch_ones, rel=1e-6), (
        "batch normalization must be invariant to a constant weight scale")
    torch.manual_seed(7)
    glob_small = float(flow_matching_loss(model, x1, hist, ext, sample_weight=small,
                                          weight_reduction="global",
                                          blocks=["base_pose"])["loss"])
    assert glob_small == pytest.approx(0.01 * batch_ones, rel=1e-5), (
        "global normalization must scale with the weights, or it is batch normalization")


def test_an_unknown_weight_reduction_is_refused():
    """A typo must not silently fall through to one of the two arms."""
    model, x1, hist, ext = _batch()
    with pytest.raises(ValueError, match="weight_reduction"):
        flow_matching_loss(model, x1, hist, ext, sample_weight=torch.ones(x1.shape[0]),
                           weight_reduction="mean", blocks=["base_pose"])


def test_a_trailing_singleton_axis_on_the_weight_is_refused():
    """`(B, 1)` broadcasts into a `(B, B)` outer product and returns a plausible number."""
    model, x1, hist, ext = _batch()
    with pytest.raises(ValueError, match="1-D"):
        flow_matching_loss(model, x1, hist, ext,
                           sample_weight=torch.ones(x1.shape[0], 1), blocks=["base_pose"])


def test_the_weight_does_not_silently_compose_with_the_auxiliary_terms():
    """They are added as UNWEIGHTED batch means, so the total would look weighted and not be."""
    model, x1, hist, ext = _batch()
    with pytest.raises(ValueError, match="auxiliary"):
        flow_matching_loss(model, x1, hist, ext, sample_weight=torch.ones(x1.shape[0]),
                           vel_weight=1.0, blocks=["base_pose"])


# --- the arm stamp -------------------------------------------------------------------


def _load_finetune():
    """Import ``finetune_generator`` by path -- ``scripts/`` is not a package."""
    import importlib.util
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "flow_matching"
    spec = importlib.util.spec_from_file_location(
        "finetune_generator", str(here / "finetune_generator.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    """The three fields the arm rule reads."""

    def __init__(self, filter_label="", unconditioned=False, awr_beta=0.0):
        self.filter_label = filter_label
        self.unconditioned = unconditioned
        self.awr_beta = awr_beta


@pytest.mark.parametrize("args,plain", [
    (_Args(), False),                              # advantage-conditioned: grows an embedding
    (_Args(filter_label="positive"), True),        # label-filtered BC
    (_Args(unconditioned=True), True),             # BC on a pre-filtered rollout
    (_Args(awr_beta=1.0), True),                   # AWR -- the arm that was missed
])
def test_every_plain_arm_is_recognised_as_plain(args, plain):
    """The stamp and the architecture come from ONE rule, or the artifact lies about itself.

    ``_save`` used to restate this rule instead of sharing it, which is why every AWR
    checkpoint written before the fix carried ``advantage_cond=True`` over plain weights:
    `load_fm_context` then builds the conditioned model and `load_state_dict` dies on a
    missing ``advantage_emb.weight``, at LOAD time, in whatever consumer picks it up -- so
    the training run looks entirely successful and the failure surfaces hours later.
    """
    fg = _load_finetune()
    assert fg.is_plain_arm(args) is plain


def test_a_sharper_awr_beta_is_still_a_plain_arm():
    """Any positive beta selects AWR, so the boundary is > 0 rather than a magic value."""
    fg = _load_finetune()
    assert fg.is_plain_arm(_Args(awr_beta=1e-6)) is True
    assert fg.is_plain_arm(_Args(awr_beta=0.0)) is False


def test_the_help_text_renders():
    """argparse %-formats every help string, so one bare ``%`` breaks --help for ALL flags.

    This has bitten three times in this file: ``"0.3-0.7% of ordinary..."``,
    ``"keeps 30% of EVERY cell"`` and ``"(~6% overshoot)"`` each parse their ``% o`` / ``% of``
    as a ``%o`` conversion and raise ``TypeError: %o format: an integer is required``. The
    failure is in `format_help`, so it takes down every flag's help at once and is invisible
    until someone runs ``--help`` -- which no training run does.
    """
    fg = _load_finetune()
    import argparse
    import io
    import contextlib
    # Rebuild the parser the way `main` does, without running anything.
    src = pathlib.Path(fg.__file__).read_text() if hasattr(fg, "__file__") else ""
    del src
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        old = sys.argv
        try:
            sys.argv = ["finetune_generator.py", "--help"]
            fg.main()
        finally:
            sys.argv = old
    out = buf.getvalue()
    assert "--awr_beta" in out and "--ckpt_every" in out, out[:400]
