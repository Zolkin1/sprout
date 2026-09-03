# Copyright 2026 Zachary Olkin. All rights reserved.

"""The aliasing probe's feature embedding must not manufacture its own signal.

Both defects these cover were SILENT -- the probe printed healthy dimension counts and
returned plausible statistics in each case:

* ``_block_embed`` re-standardized every retained PCA component to unit variance, so a block
  whose effective rank was below the ceiling had float round-off divided by its own size. The
  fix is a SINGLE scalar per block rather than a variance floor: a floor was tried first and
  proved dataset-dependent, leaving a 110-sigma point on the second row-set it met.
  Measured on v4_big, ``cmd_future`` (192 dims of a command that barely moves inside a
  re-plan window, effective rank 10) put one window 236 sigma out. That both made the global
  PCA's leading component "the direction isolating that one row" and gave six of
  ``z_partial``'s 64 dimensions to noise -- which blurs every neighbourhood and biases the
  local-variance statistic TOWARD the no-separation answer the probe exists to test.
* ``_zscore`` pooled its mean and std over non-finite values too. The height scan carries
  ``-inf`` for a missed ray by construction, and one of those makes a column's mean ``-inf``,
  every centred value NaN, and the finite-guard then zeroes the WHOLE COLUMN -- so a
  privileged block can vanish while the run still reports its width.

Numpy + sklearn only; no Isaac, no rollout data.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

# Derived from the package root rather than counted in ``..`` hops -- moving this file must
# not silently point the import somewhere else.
_PROBE = (pathlib.Path(__file__).resolve().parents[3]
          / "scripts" / "flow_matching" / "advantage_aliasing_probe.py")


def _probe():
    """Import the probe module by path (``scripts/`` is not an importable package)."""
    spec = importlib.util.spec_from_file_location("advantage_aliasing_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ap():
    return _probe()


def _rank_deficient(n: int = 20_000, d: int = 192, rank: int = 10,
                    seed: int = 0) -> np.ndarray:
    """A block with a KNOWN effective rank, plus float32 round-off below it.

    Deliberately shaped like the real ``cmd_future`` -- 192 raw columns that are linear
    combinations of 10 underlying signals -- because the amplification GROWS with the sample,
    and a small synthetic understates it. At this size the unfixed path reaches ~55 sigma
    against the real data's 236; at ``n=2000, d=40`` it only reaches 23, which is high but
    not obviously broken.

    Casting to float32 is what supplies the sub-rank noise. It is not added by hand, so the
    test exercises the real mechanism rather than a stand-in for it.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, rank))
    mixing = rng.normal(size=(rank, d))
    return (latent @ mixing).astype(np.float32)


def _participation_ratio(emb: np.ndarray) -> float:
    """How many directions a block ACTUALLY carries, as opposed to how many columns it has."""
    v = emb.var(axis=0)
    return float(v.sum() ** 2 / np.square(v).sum())


def test_block_embed_does_not_amplify_a_degenerate_direction(ap):
    """The core promise: no direction is scaled up to meet a variance quota.

    Asserted as a magnitude rather than a column count on purpose. A count is a property of
    whatever threshold is in force, and the threshold is precisely what proved
    dataset-dependent -- the same block survived a 1e-4 floor on one row-set and still reached
    110 sigma on another.
    """
    raw = _rank_deficient()
    emb = ap._block_embed(raw, 16, seed=0, name="synthetic")

    assert np.abs(emb).max() < 12.0, (
        f"max |z| {np.abs(emb).max():.1f} -- a degenerate direction was amplified")


def test_block_embed_reports_the_blocks_TRUE_width(ap):
    """A rank-10 block must behave as ~10-dimensional, however many columns it carries.

    This is what the single-scalar scaling buys and whitening destroyed: under whitening every
    block looks exactly `n_components`-dimensional, so a block that is really 10 directions of
    signal plus 6 of round-off is indistinguishable from one with 16 of signal.
    """
    emb = ap._block_embed(_rank_deficient(rank=10), 16, seed=0, name="rank10")
    pr = _participation_ratio(emb)

    assert 8.0 < pr < 12.0, (
        f"participation ratio {pr:.1f} -- expected ~10, the block's true rank")


def test_block_embed_gives_every_block_the_same_variance_budget(ap):
    """Equal say per source is the whole point, and it must survive the rank change.

    A rank-10 block and a full-rank one must arrive with the same TOTAL variance, or whichever
    has more real directions quietly dominates every distance -- the failure this function
    exists to prevent, reintroduced from the other side.
    """
    rng = np.random.default_rng(7)
    thin = ap._block_embed(_rank_deficient(rank=4), 16, seed=0, name="thin")
    full = ap._block_embed(rng.normal(size=(20_000, 192)).astype(np.float32), 16,
                           seed=0, name="full")

    assert np.isclose(thin.var(axis=0).sum(), thin.shape[1], rtol=0.05)
    assert np.isclose(full.var(axis=0).sum(), full.shape[1], rtol=0.05)


def test_the_unfixed_embedding_would_have_been_visibly_wrong(ap):
    """Whitening every component of the SAME block manufactures a huge outlier.

    Without this the fix above could be silently reverted and the suite would stay green: the
    assertion has to show the defect is reachable, not merely that the fixed path is tidy.
    """
    from sklearn.decomposition import PCA

    raw = _rank_deficient()
    z = ap._zscore(raw)
    emb_all = PCA(n_components=16, svd_solver="randomized",
                  random_state=0).fit_transform(z)
    whitened = ap._zscore(emb_all)          # what the old code returned

    assert np.abs(whitened).max() > 30.0, (
        f"max |z| only {np.abs(whitened).max():.1f} -- the synthetic block no longer "
        "reproduces the round-off amplification, so this test has stopped guarding anything")


def test_block_embed_keeps_a_full_rank_block_whole(ap):
    """A block with no degenerate directions must lose nothing -- the guard is not a haircut."""
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(2000, 40)).astype(np.float32)
    emb = ap._block_embed(raw, 16, seed=0, name="full_rank")

    assert emb.shape[1] == 16
    assert _participation_ratio(emb) > 14.0, (
        "a full-rank block must read as nearly its full width")


def test_zscore_survives_a_missed_ray_sentinel(ap):
    """One ``-inf`` must not zero its whole column -- the height scan has them by design."""
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(500, 8)).astype(np.float32)
    raw[3, 2] = -np.inf                      # a missed ray, exactly as the scan stores it

    z = ap._zscore(raw)

    assert np.isfinite(z).all(), "non-finite leaked through the standardization"
    # The column still carries its signal: it correlates with the clean version of itself
    # over the rows that were never sentinels.
    clean = np.delete(np.arange(raw.shape[0]), 3)
    assert abs(np.corrcoef(z[clean, 2], raw[clean, 2])[0, 1]) > 0.99, (
        "column 2 lost its content -- this is the silent whole-block zeroing")
    assert z[:, 2].std() > 0.5, "column 2 collapsed toward constant"


def test_zscore_leaves_a_constant_column_at_zero(ap):
    """A zero-variance column is centred, never divided by its own (zero) spread."""
    raw = np.zeros((100, 3), dtype=np.float32)
    raw[:, 0] = 7.0
    raw[:, 1] = np.arange(100)

    z = ap._zscore(raw)

    assert np.allclose(z[:, 0], 0.0)
    assert np.isclose(z[:, 1].std(), 1.0, atol=1e-5)
