# Copyright 2026 Zachary Olkin. All rights reserved.

"""FSQ bottleneck: the lattice, the id packing, and the bound that must not be doubled."""

from __future__ import annotations

import math

import pytest
import torch

from robot_rl.token_planner.fsq import FSQ, codes_to_id, id_to_codes


LEVELS = (8, 8, 8, 5, 5)


def test_vocab_size_is_the_product_of_the_levels():
    q = FSQ(LEVELS)
    assert q.vocab_size == math.prod(LEVELS) == 12800
    assert q.latent_dim == 5


def test_every_token_id_round_trips_through_codes():
    """``codes_to_id`` and ``id_to_codes`` must be exact inverses over the WHOLE vocabulary.

    Not a sample: the packing is mixed-radix, so an off-by-one in the basis shows up only on
    the ids where a digit carries, and those are exactly the ones a random draw is least
    likely to hit.
    """
    q = FSQ(LEVELS)
    ids = torch.arange(q.vocab_size)
    codes = id_to_codes(ids, q.levels)
    assert codes.shape == (q.vocab_size, len(LEVELS))
    for i, lv in enumerate(LEVELS):
        assert int(codes[:, i].min()) == 0
        assert int(codes[:, i].max()) == lv - 1
    assert torch.equal(codes_to_id(codes, q.levels), ids)


def test_dim_zero_is_the_fastest_varying_digit():
    """The documented convention, pinned so a basis reversal cannot pass silently."""
    q = FSQ(LEVELS)
    zero = torch.zeros(len(LEVELS), dtype=torch.long)
    first = zero.clone()
    first[0] = 1
    second = zero.clone()
    second[1] = 1
    assert int(codes_to_id(first, q.levels)) == 1
    assert int(codes_to_id(second, q.levels)) == LEVELS[0]


def test_quantized_latents_land_on_the_lattice_and_span_it():
    """A large sweep of raw inputs must produce every level on every axis, at the right values.

    The reachable set on an ``l``-level axis is ``{-1, ..., 1}`` in ``l`` even steps; if the
    encoder's bound were applied twice the extreme levels would be unreachable and this
    would fail on the ``min``/``max`` rather than on the spacing.
    """
    q = FSQ(LEVELS)
    z = torch.linspace(-6.0, 6.0, 2001).unsqueeze(-1).repeat(1, len(LEVELS))
    latent, ids = q(z)
    codes = id_to_codes(ids, q.levels)
    for i, lv in enumerate(LEVELS):
        seen = torch.unique(codes[:, i])
        assert seen.numel() == lv, f"axis {i} reached {seen.numel()} of {lv} levels"
        expect = torch.linspace(-1.0, 1.0, lv)
        assert torch.allclose(torch.unique(latent[:, i]), expect, atol=1e-6)


def test_id_to_latent_agrees_with_the_forward_pass():
    q = FSQ(LEVELS)
    z = torch.randn(64, 7, len(LEVELS)) * 2.0
    latent, ids = q(z)
    assert torch.allclose(q.id_to_latent(ids), latent, atol=1e-6)


def test_the_gradient_is_straight_through_to_the_tanh():
    """``d latent / d z`` must be exactly ``d tanh / d z`` -- the rounding contributes nothing.

    Also the check that the range clamp does not eat gradients: it can only bind where the
    rounding already landed on an end level, and PyTorch's ``clamp`` passes the gradient at
    the boundary itself, so a saturated input must still get ``1 - tanh(z)^2``.
    """
    for scale in (0.5, 3.0, 8.0):          # 8.0 saturates tanh, so every code is an end one
        z = (torch.randn(64, len(LEVELS)) * scale).requires_grad_()
        latent, _ = FSQ(LEVELS).quantize(z)
        latent.sum().backward()
        assert torch.allclose(z.grad, 1.0 - torch.tanh(z.detach()) ** 2, atol=1e-6)


def test_an_out_of_range_id_is_refused_rather_than_wrapped():
    """A wrapped id would decode a plausible trajectory for a token nothing emitted."""
    q = FSQ(LEVELS)
    with pytest.raises(ValueError, match="out of range"):
        q.id_to_latent(torch.tensor([q.vocab_size]))
    with pytest.raises(ValueError, match="out of range"):
        q.id_to_latent(torch.tensor([-1]))


def test_a_degenerate_level_table_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        FSQ([])
    with pytest.raises(ValueError, match=">= 2"):
        FSQ([8, 1, 5])


def test_the_ablation_level_table_works_too():
    q = FSQ((7, 5, 5, 5, 5))
    assert q.vocab_size == 4375
    ids = torch.arange(q.vocab_size)
    assert torch.equal(codes_to_id(id_to_codes(ids, q.levels), q.levels), ids)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_the_lattice_does_not_depend_on_the_input_dtype(dtype):
    """The dequantized latent is a CONSTANT of the level table, not of the caller's dtype.

    It was not: :meth:`FSQ.quantize` computed the lattice in ``z``'s dtype while
    ``id_to_latent`` used fp32, so under bf16 autocast the decoder was trained against one
    lattice and served another -- measured at 3.3e-3 in latent units and 1.0 mm / 20.2 mm p99
    of base position on a live checkpoint, with the reported metrics (computed outside
    autocast) moving 0.04%. Nothing had the wrong shape, so no number in a run could show it.
    """
    q = FSQ(LEVELS)
    z = torch.randn(4096, len(LEVELS)) * 2.0
    latent_ref, ids_ref = q(z)
    latent, ids = q(z.to(dtype))
    assert latent.dtype == torch.float32, "the lattice must come back in fp32"
    # The IDS may differ, because casting z is a real change to the encoder's output; what
    # may not differ is the latent a given id maps to.
    assert torch.equal(q.id_to_latent(ids), latent)
    assert torch.equal(q.id_to_latent(ids_ref), latent_ref)


def test_a_double_tanh_would_lose_the_end_codes_of_the_eight_level_axes():
    """The failure the module docstring is about, measured rather than asserted.

    Kept as a test because the number in that docstring was wrong once: the 5-level axes
    SURVIVE a double tanh (``half * (1 - tanh 1) = 0.477`` still rounds to 0), so the
    collapse is 12800 -> 5400 and not the 1944 a first estimate claimed.
    """
    q = FSQ(LEVELS)
    z = torch.linspace(-60.0, 60.0, 20001).unsqueeze(-1).repeat(1, len(LEVELS))
    reachable_ok = [int(torch.unique(id_to_codes(q(z)[1], q.levels)[:, i]).numel())
                    for i in range(len(LEVELS))]
    assert reachable_ok == list(LEVELS)
    # The bug: a squash applied before FSQ's own.
    reachable_bad = [int(torch.unique(id_to_codes(q(torch.tanh(z))[1], q.levels)[:, i]).numel())
                     for i in range(len(LEVELS))]
    assert reachable_bad == [6, 6, 6, 5, 5]
    assert math.prod(reachable_bad) == 5400
