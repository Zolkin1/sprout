# Copyright 2026 Zachary Olkin. All rights reserved.

"""The knot action space and its factored exploration width.

Both exist because the full-width action could not be searched: measured over 840 iterations
the residual stayed ~97% exploration noise and the 2728 per-dimension stds moved 1.5-2.2%
with a spread across dimensions of 0.0025. What these tests pin is the geometry that
replaced it -- that the basis is a genuine interpolation, that a zero action still serves the
generator's own plan bit-for-bit, and that the 18-parameter std presents itself to PPO as the
action's true width.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_rl.flow_matching.knot_basis import (
    DEFAULT_NUM_KNOTS,
    knot_basis,
    knot_positions,
    knot_sample_matrix,
)
from robot_rl.rl.knot_distribution import GroupedKnotGaussian, channel_groups

F, D = 62, 44
BLOCKS = ("base_pose", "base_vel", "joint_pos")
NJ = 29


# ----------------------------------------------------------------------------------
# The basis
# ----------------------------------------------------------------------------------
def test_the_knots_span_the_whole_window() -> None:
    """Endpoints included, so every node is an interpolation and none is extrapolated.

    The command-token positions ``(0, 5, ..., 60)`` stop one node short of a 62-node window,
    and clamping that node measured 0.0485 of unexplained RMS against ~0.021 elsewhere.
    """
    k = knot_positions(F, DEFAULT_NUM_KNOTS)
    assert k[0] == 0.0
    assert k[-1] == F - 1
    assert np.all(np.diff(k) > 0)


@pytest.mark.parametrize("kind", ["catmull", "linear"])
def test_a_constant_knot_vector_gives_a_constant_window(kind: str) -> None:
    """Rows sum to 1 -- which is what makes a zero action serve the generator's own plan."""
    b = knot_basis(F, DEFAULT_NUM_KNOTS, kind)
    np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-12)
    u = np.full((DEFAULT_NUM_KNOTS, D), 0.37)
    np.testing.assert_allclose(b @ u, 0.37, atol=1e-12)


@pytest.mark.parametrize("kind", ["catmull", "linear"])
def test_the_basis_has_full_rank(kind: str) -> None:
    """Rank-deficient, some knots would be unreachable and the action width a lie."""
    b = knot_basis(F, DEFAULT_NUM_KNOTS, kind)
    assert np.linalg.matrix_rank(b) == DEFAULT_NUM_KNOTS


def test_the_last_node_is_exactly_the_last_knot() -> None:
    b = knot_basis(F, DEFAULT_NUM_KNOTS)
    np.testing.assert_allclose(b[F - 1, :-1], 0.0, atol=1e-12)
    assert b[F - 1, -1] == pytest.approx(1.0)


def test_catmull_overshoots_and_linear_does_not() -> None:
    """Recorded because the ORDER of tanh and interpolation depends on it.

    Cubic has negative lobes, so an interpolated node can exceed the largest knot -- worst
    row L1 1.2499. The env therefore lifts FIRST and squashes second, which keeps the
    per-channel cap binding exactly; squashing the knots first would let the window exceed
    the cap it is supposed to respect.
    """
    cat = np.abs(knot_basis(F, DEFAULT_NUM_KNOTS, "catmull")).sum(axis=1).max()
    lin = np.abs(knot_basis(F, DEFAULT_NUM_KNOTS, "linear")).sum(axis=1).max()
    assert cat == pytest.approx(1.2499, abs=1e-3)
    assert lin == pytest.approx(1.0, abs=1e-12)


def test_a_zero_action_lifts_to_a_zero_residual() -> None:
    """The whole reason the head is zero-initialised: the served plan must start unchanged."""
    b = torch.as_tensor(knot_basis(F, DEFAULT_NUM_KNOTS), dtype=torch.float32)
    u = torch.zeros(4, DEFAULT_NUM_KNOTS, D)
    lifted = torch.einsum("fk,bkd->bfd", b, u)
    assert lifted.shape == (4, F, D)
    assert float(torch.tanh(lifted).abs().max()) == 0.0


def test_more_knots_than_nodes_is_refused() -> None:
    with pytest.raises(ValueError, match="rank-deficient"):
        knot_positions(8, 9)


def test_the_sampler_is_an_interpolation_not_a_transpose() -> None:
    """``S`` reads the window's local value at each knot; rows must therefore sum to 1.

    ``B.T`` does not (it would scale the feature) and the pseudo-inverse is a least-squares
    deconvolution (it would hand the token a fitted coefficient instead of a local value).
    """
    s = knot_sample_matrix(F, DEFAULT_NUM_KNOTS)
    assert s.shape == (DEFAULT_NUM_KNOTS, F)
    np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(s >= 0.0)


# ----------------------------------------------------------------------------------
# The channel grouping
# ----------------------------------------------------------------------------------
def test_the_channel_groups_come_from_the_window_layout() -> None:
    """Derived, not hardcoded, so a layout change cannot leave this describing the old one."""
    idx, names = channel_groups(NJ, 0, BLOCKS)
    assert names == ["base_pos", "base_ori", "base_lin_vel", "base_ang_vel", "joint_pos"]
    assert len(idx) == D
    counts = [idx.count(i) for i in range(len(names))]
    assert counts == [3, 6, 3, 3, NJ], "position 3, rot6d 6, two velocities 3, joints nj"


# ----------------------------------------------------------------------------------
# The factored std
# ----------------------------------------------------------------------------------
def _dist(init_std=None) -> GroupedKnotGaussian:
    idx, names = channel_groups(NJ, 0, BLOCKS)
    return GroupedKnotGaussian(
        DEFAULT_NUM_KNOTS * D, DEFAULT_NUM_KNOTS, idx, names,
        init_std=init_std or {"base_pos": 0.05, "base_ori": 0.05, "base_lin_vel": 0.10,
                              "base_ang_vel": 0.10, "joint_pos": 0.15},
    )


def test_the_std_is_eighteen_numbers_not_five_hundred() -> None:
    """5 groups + 13 knots. The point: 18 parameters share every sample of the iteration.

    At full width there were 2728 of them with ~3 samples each per iteration, and they moved
    1.5-2.2% in 850 iterations with a spread across dimensions of 0.0025 -- i.e. they never
    differentiated at all.
    """
    d = _dist()
    assert sum(p.numel() for p in d.parameters()) == 5 + DEFAULT_NUM_KNOTS


def test_it_presents_the_actions_true_width_to_ppo() -> None:
    """``ChunkPPO`` reads this attribute's width to check its ``K * D`` layout.

    Were it 18 wide, the layout assertion would fire and refuse the run; were it silently
    reshaped, the ratio would slice knots into the wrong groups with every shape checking out.
    """
    d = _dist()
    assert tuple(d.log_std_param.shape) == (DEFAULT_NUM_KNOTS * D,)


def test_each_group_starts_at_exactly_its_own_width() -> None:
    d = _dist()
    idx, names = channel_groups(NJ, 0, BLOCKS)
    sigma = torch.exp(d.log_std_param).detach().view(DEFAULT_NUM_KNOTS, D)
    want = {"base_pos": 0.05, "base_ori": 0.05, "base_lin_vel": 0.10,
            "base_ang_vel": 0.10, "joint_pos": 0.15}
    for g, name in enumerate(names):
        cols = [i for i, v in enumerate(idx) if v == g]
        got = sigma[:, cols]
        assert float(got.min()) == pytest.approx(want[name], rel=1e-6)
        assert float(got.max()) == pytest.approx(want[name], rel=1e-6)


def test_the_knot_factor_is_gauge_pinned() -> None:
    """``log s + log h`` is over-parameterised; mean-centring ``h`` makes both readable.

    Without it, adding ``c`` to every ``log h`` and subtracting it from every ``log s``
    leaves sigma unchanged, so the two factors drift together and neither logged value means
    anything on its own.
    """
    d = _dist()
    assert torch.allclose(d.knot_std, torch.ones(DEFAULT_NUM_KNOTS), atol=1e-6)
    with torch.no_grad():
        d.log_knot_scale.add_(0.75)                 # a pure gauge shift
    assert torch.allclose(d.knot_std, torch.ones(DEFAULT_NUM_KNOTS), atol=1e-6)
    sigma = torch.exp(d.log_std_param).detach()
    assert float(sigma.max()) == pytest.approx(0.15, rel=1e-5), "sigma must not have moved"


def test_a_missing_group_is_refused_rather_than_defaulted() -> None:
    """The measured correction differs ~2.7x across groups, so a silent default is a bug."""
    idx, names = channel_groups(NJ, 0, BLOCKS)
    with pytest.raises(ValueError, match="missing channel group"):
        GroupedKnotGaussian(DEFAULT_NUM_KNOTS * D, DEFAULT_NUM_KNOTS, idx, names,
                            init_std={"base_pos": 0.05})


def test_a_scalar_parameterisation_is_refused() -> None:
    """The factors ADD in log space; in scalar space they would multiply."""
    idx, names = channel_groups(NJ, 0, BLOCKS)
    with pytest.raises(ValueError, match="log space"):
        GroupedKnotGaussian(DEFAULT_NUM_KNOTS * D, DEFAULT_NUM_KNOTS, idx, names,
                            init_std=0.1, std_type="scalar")


def test_sampling_and_entropy_have_the_batch_shape() -> None:
    d = _dist()
    d.update(torch.zeros(6, DEFAULT_NUM_KNOTS * D))
    assert tuple(d.sample().shape) == (6, DEFAULT_NUM_KNOTS * D)
    assert tuple(d.entropy.shape) == (6,)
    assert tuple(d.log_prob(d.sample()).shape) == (6,)


# ----------------------------------------------------------------------------------
# The mean/noise estimator the diagnostics report
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("mean_scale, want", [(0.0, 0.0), (1.0, 0.5), (10.0, 0.99)])
def test_the_mean_energy_estimator_recovers_a_known_split(mean_scale, want) -> None:
    """``1 - E[(du)^2] / (2 E[u^2])`` on i.i.d. draws about a CONSTANT mean.

    This is the primary "is it learning" diagnostic, so its arithmetic is pinned against
    cases with a known answer: pure noise reads 0, equal mean and noise power reads 0.5.
    """
    g = torch.Generator().manual_seed(0)
    mean = mean_scale * torch.ones(1, DEFAULT_NUM_KNOTS, D)
    u = mean + torch.randn(4096, DEFAULT_NUM_KNOTS, D, generator=g)
    up = mean + torch.randn(4096, DEFAULT_NUM_KNOTS, D, generator=g)
    sq = u.pow(2).mean()
    var = ((u - up).pow(2).mean() * 0.5).clamp_max(sq)
    assert float((sq - var) / sq) == pytest.approx(want, abs=0.02)
