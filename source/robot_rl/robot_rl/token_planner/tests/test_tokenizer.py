# Copyright 2026 Zachary Olkin. All rights reserved.

"""The trajectory tokenizer: shapes, the chunk-local property, and the refusals.

These are structural tests -- an untrained autoencoder reconstructs nothing, so nothing here
asserts a reconstruction quality. What they do pin is every place a wrong answer would be
SHAPE-CORRECT and therefore silent: a token count that does not match the horizon, a decode
path that disagrees with the encode path, a chunk whose length the conv stack happily
consumes anyway.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.token_planner.config import TokenizerConfig
from robot_rl.token_planner.tokenizer import (
    TrajectoryDecoder,
    TrajectoryEncoder,
    TrajectoryTokenizer,
    _stride_schedule,
)


def small_cfg(**kw) -> TokenizerConfig:
    """A cheap config for the structural tests.

    Args:
        **kw: Field overrides.

    Returns:
        A config with a small trunk, so a test builds in milliseconds.
    """
    base = dict(
        node_dim=44, chunk_len=16, downsample=4, channels=32,
        num_blocks_high=2, num_blocks_low=1, fsq_levels=[4, 4, 3],
    )
    base.update(kw)
    return TokenizerConfig(**base)


def test_the_shipped_config_is_the_flow_matching_horizon():
    """The chunk IS one plan, so ``H`` must be the generator's ``future_len`` and ``D``
    must divide it. The shipped generator is ``future_len=64``, which admits ``D=4``; the
    config default's older sibling, 62, does not (``2 x 31``), and that mismatch is a
    construction-time error rather than a silent pad.
    """
    cfg = TokenizerConfig()
    assert cfg.chunk_len == 64
    assert cfg.chunk_len % cfg.downsample == 0
    assert cfg.num_tokens == 16
    assert cfg.vocab_size == 12800
    assert cfg.latent_dim == 5
    assert cfg.token_seconds == pytest.approx(0.08)


def test_forward_shapes_and_token_range():
    cfg = small_cfg()
    tk = TrajectoryTokenizer(cfg)
    x = torch.randn(5, cfg.chunk_len, cfg.node_dim)
    recon, ids = tk(x)
    assert recon.shape == x.shape
    assert ids.shape == (5, cfg.num_tokens)
    assert ids.dtype == torch.int64
    assert int(ids.min()) >= 0 and int(ids.max()) < cfg.vocab_size


def test_decode_ids_reproduces_the_forward_reconstruction_exactly():
    """The planner decodes from IDS while training tokenizes from NODES. If those two paths
    disagree, every closed-loop number is measured on a decoder the tokenizer was not
    trained as -- and the disagreement would be a small numerical one, so it would look
    like model error rather than a bug.
    """
    cfg = small_cfg()
    tk = TrajectoryTokenizer(cfg).eval()
    x = torch.randn(4, cfg.chunk_len, cfg.node_dim)
    with torch.no_grad():
        recon, ids = tk(x)
        again = tk.decode_ids(ids)
    assert torch.equal(recon, again)
    assert torch.equal(tk.encode_to_ids(x), ids)


def test_the_autoencoder_is_chunk_local():
    """Two chunks in one batch must not influence each other.

    Chunk-locality is what makes "tokenize a training chunk" and "decode a generated
    sequence" the same operation. A normalization that pooled over the batch, or a
    LayerNorm applied over TIME rather than channels, would break it while every shape
    stayed right -- and the only symptom would be a deploy-time reconstruction error that
    training never measured.
    """
    cfg = small_cfg()
    tk = TrajectoryTokenizer(cfg).eval()
    a = torch.randn(1, cfg.chunk_len, cfg.node_dim)
    b = torch.randn(1, cfg.chunk_len, cfg.node_dim)
    with torch.no_grad():
        alone, ids_alone = tk(a)
        together, ids_together = tk(torch.cat([a, b], dim=0))
    assert torch.allclose(alone[0], together[0], atol=1e-6)
    assert torch.equal(ids_alone[0], ids_together[0])


def test_a_downsample_that_does_not_divide_the_chunk_is_refused():
    """It would otherwise be padded silently by the conv stack, making the tokens at the
    seam mean something different from the tokens inside it."""
    with pytest.raises(ValueError, match="does not divide chunk_len"):
        TrajectoryTokenizer(small_cfg(chunk_len=62, downsample=4))
    # The message must name the usable divisors rather than only complaining.
    with pytest.raises(ValueError, match=r"\[1, 2, 31, 62\]"):
        TokenizerConfig(chunk_len=62, downsample=4).validate()


def test_a_chunk_of_the_wrong_length_or_width_is_refused():
    """A conv stack consumes any length, so this is the only place it can be caught."""
    cfg = small_cfg()
    enc = TrajectoryEncoder(cfg)
    with pytest.raises(ValueError, match="expected nodes of shape"):
        enc(torch.randn(2, cfg.chunk_len + 1, cfg.node_dim))
    with pytest.raises(ValueError, match="expected nodes of shape"):
        enc(torch.randn(2, cfg.chunk_len, cfg.node_dim + 1))
    dec = TrajectoryDecoder(cfg)
    with pytest.raises(ValueError, match="expected latents of shape"):
        dec(torch.randn(2, cfg.num_tokens + 1, cfg.latent_dim))
    tk = TrajectoryTokenizer(cfg)
    with pytest.raises(ValueError, match="token"):
        tk.decode_ids(torch.zeros(2, cfg.num_tokens + 1, dtype=torch.long))


@pytest.mark.parametrize("downsample,expect", [(1, []), (2, [2]), (4, [2, 2]),
                                               (8, [2, 2, 2]), (3, [3]), (6, [6])])
def test_the_stride_schedule_factors_powers_of_two_and_nothing_else(downsample, expect):
    assert _stride_schedule(downsample) == expect


@pytest.mark.parametrize("downsample", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("upsample_mode", ["nearest", "transpose"])
def test_every_downsample_round_trips_to_the_original_length(downsample, upsample_mode):
    """The decoder's output length must be ``H`` for every ``(D, upsample_mode)`` pair.

    A transposed convolution's output length depends on ``(kernel, stride, padding)`` in a
    way that is easy to get right for one stride and wrong for another, so this is
    parametrized rather than checked once at the shipped value.
    """
    cfg = small_cfg(chunk_len=16, downsample=downsample, upsample_mode=upsample_mode)
    tk = TrajectoryTokenizer(cfg)
    recon, ids = tk(torch.randn(2, cfg.chunk_len, cfg.node_dim))
    assert recon.shape == (2, cfg.chunk_len, cfg.node_dim)
    assert ids.shape == (2, cfg.chunk_len // downsample)


def test_gradients_reach_the_encoder_through_the_bottleneck():
    """The straight-through estimator is the only path; if it were dropped the encoder
    would simply never train and the loss would plateau at the decoder's best constant."""
    cfg = small_cfg()
    tk = TrajectoryTokenizer(cfg)
    recon, _ = tk(torch.randn(3, cfg.chunk_len, cfg.node_dim))
    recon.pow(2).mean().backward()
    named = dict(tk.named_parameters())
    for name in ("encoder.node_in.weight", "encoder.latent_out.weight",
                 "decoder.latent_in.weight", "decoder.node_out.weight"):
        g = named[name].grad
        assert g is not None and float(g.abs().sum()) > 0.0, f"no gradient at {name}"


def test_the_quantizer_holds_no_parameters():
    """FSQ's whole appeal here: no codebook to go stale, so utilisation is a property of the
    encoder's output distribution rather than of the optimiser's luck."""
    tk = TrajectoryTokenizer(small_cfg())
    assert sum(p.numel() for p in tk.quantizer.parameters()) == 0


def test_the_shipped_size_is_in_the_target_band():
    """The spec's parameter target is 2-5M; the shipped default must land near it, because
    a tokenizer that is much larger starts to be where the modelling happens rather than
    the planner."""
    n = sum(p.numel() for p in TrajectoryTokenizer(TokenizerConfig()).parameters())
    assert 2e6 <= n <= 6e6, f"{n / 1e6:.2f}M parameters"


def test_the_reconstruction_metrics_cannot_be_built_positionally():
    """``std`` and ``mean`` are both ``(D,)`` float tensors, so swapping them is shape-silent.

    Measured, before the accumulator was made keyword-only: passing them the other way round
    reported a ``base_pos`` RMSE of 1.0 where the truth was 0.04 -- a 25x error that still
    looks like a number, in the one place the gating decision is read from.
    """
    from robot_rl.token_planner.metrics import ReconAccumulator, channel_group_labels

    names, labels = channel_group_labels(29, 0, ["base_pose", "base_vel", "joint_pos"])
    with pytest.raises(TypeError):
        ReconAccumulator(names, labels, torch.ones(44), torch.zeros(44), 64, 12800, 1, ["a"])


def test_the_accumulator_refuses_an_out_of_range_family_and_leaves_its_input_alone():
    """It used to `clamp_` in place: an illegal code was folded into the last family, and the
    caller's own array was rewritten under it."""
    from robot_rl.token_planner.metrics import ReconAccumulator, channel_group_labels

    names, labels = channel_group_labels(4, 0, ["base_pose"])
    acc = ReconAccumulator(group_names=names, labels=labels, std=torch.ones(9),
                           mean=torch.zeros(9), chunk_len=4, vocab_size=16,
                           num_families=3, family_names=["a", "b", "c"])
    t = torch.randn(4, 4, 9)
    fam = torch.tensor([0, 1, 99, 2])
    with pytest.raises(ValueError, match="family code out of range"):
        acc.update(t, t, torch.zeros(4, 1, dtype=torch.long), fam)
    assert fam.tolist() == [0, 1, 99, 2], "the caller's tensor must not be mutated"
