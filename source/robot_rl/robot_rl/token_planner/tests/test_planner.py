# Copyright 2026 Zachary Olkin. All rights reserved.

"""The autoregressive planner: the attention decomposition, causality, and RL's interfaces.

Three families of test, and each covers a failure that is invisible at the shapes:

* **the decomposition** -- ``_run_layer`` and the prefix cache reproduce a masked full pass
  through a stock pre-norm layer. Attention over a cache is arithmetic, not a mask, so if the
  op order or the norm placement is wrong the model still trains, just as a different one.
* **causality** -- an output slot must not see the tokens after it. Teacher forcing hides
  this completely: a leaky mask yields a model with a beautiful training loss that cannot
  generate, and the loss curve is the only thing that looks different.
* **the RL contract** -- ``sample`` and ``logprobs`` must agree on the same tokens, entropy
  must be the real entropy, and the draws must be reproducible from an explicit generator.
  All three are what a PPO ratio is built from, and all three are silently wrong-able.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig
from robot_rl.token_planner.planner import TokenPlanner, _run_layer, causal_output_mask
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer


def small_planner(**kw) -> TokenPlanner:
    """A cheap planner with the shipped topology but tiny widths.

    Args:
        **kw: :class:`PlannerConfig` overrides.

    Returns:
        The planner in eval mode (dropout off, so every comparison is deterministic).
    """
    tok_cfg = TokenizerConfig(
        node_dim=12, chunk_len=16, downsample=4, channels=16,
        num_blocks_high=1, num_blocks_low=1, fsq_levels=[4, 4, 3],
    )
    cfg = PlannerConfig(
        tokenizer=tok_cfg.to_dict(), history_dim=12, num_history_states=3, future_len=16,
        extero_encoder="cnn", extero_sensor_count=2, extero_channels=5,
        extero_sensor_dim=5 * 6 * 8, extero_image_hw=[[6, 8], [6, 8]],
        extero_cnn_channels=[8, 16], extero_cnn_strides=[1, 2],
        extero_camera_intrinsics=[[4.0, 5.0, 3.5, 2.5], [7.0, 9.0, 3.5, 2.5]],
        cmd_dim=3, cmd_stride=5, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        dropout=0.0,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return TokenPlanner(cfg, TrajectoryTokenizer(tok_cfg)).eval()





def conditioning(model: TokenPlanner, batch: int = 3, seed: int = 0):
    """Random conditioning of the shapes the planner expects.

    Args:
        model: The planner, for its widths.
        batch: Batch size.
        seed: RNG seed.

    Returns:
        ``(history, extero, cmd_tokens)``.
    """
    g = torch.Generator().manual_seed(seed)
    c = model.cfg
    return (
        torch.randn(batch, c.num_history_states, c.history_dim, generator=g),
        torch.randn(batch, c.extero_sensor_count, c.extero_sensor_dim, generator=g),
        torch.randn(batch, len(range(0, c.future_len, c.cmd_stride)), c.cmd_dim, generator=g),
    )


# ----------------------------------------------------------------------
# The attention decomposition
# ----------------------------------------------------------------------
def test_run_layer_matches_a_stock_pre_norm_layer():
    """``_run_layer`` must BE ``nn.TransformerEncoderLayer(norm_first=True)``.

    There are three separate dropout modules on the way through and two norms whose order
    distinguishes pre- from post-norm. At the tested ``dropout=0`` a mix-up is invisible, and
    a post-norm layer would train perfectly well -- just not as the architecture the config
    says it is.
    """
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True).eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert torch.allclose(_run_layer(layer, x, x), layer(x), atol=1e-6)
    # And with a mask, since that is how the output block is actually run.
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        assert torch.allclose(
            _run_layer(layer, x, x, mask), layer(x, src_mask=mask), atol=1e-6)


def test_the_prefix_cache_reproduces_a_masked_full_pass():
    """The cached decomposition must equal running the whole sequence with a block mask.

    That equality is the licence for the optimisation: the prefix is encoded once per plan
    instead of once per token step. If it did not hold, training (which uses the cache) and
    any masked reference would be two different models.
    """
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    tokens = torch.randint(0, m.vocab_size, (hist.shape[0], m.num_tokens))
    with torch.no_grad():
        have = m(tokens, hist, extero, cmd)

        # The reference: one sequence [prefix | outputs], one (S, S) mask in which the prefix
        # attends only to itself and the outputs attend to the prefix and causally to
        # themselves -- exactly the block structure the cache expresses arithmetically.
        p_tok = m.prefix(hist, extero, cmd, None)
        o_tok = m._embed_outputs(tokens[:, :-1], tokens.shape[0])
        seq = torch.cat([p_tok, o_tok], dim=1)
        pl, n = m.prefix_len, m.num_tokens
        mask = torch.zeros(pl + n, pl + n, dtype=torch.bool)
        mask[:pl, pl:] = True                                   # prefix cannot see outputs
        mask[pl:, pl:] = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        x = seq
        for layer in m.encoder.layers:
            x = _run_layer(layer, x, x, mask)
        want = m.head(m.head_norm(x[:, pl:]))
    assert torch.allclose(have, want, atol=1e-5)


def test_the_mask_shape_and_polarity():
    """``True`` means FORBIDDEN in PyTorch, and inverting the triangle silently makes the
    model anti-causal -- which still trains, just on information it will not have at
    generation time."""
    n, pl = 4, 7
    mask = causal_output_mask(n, pl)
    # N query rows, P + N key columns: the prefix, then the output block itself.
    assert mask.shape == (n, pl + n)
    assert mask.dtype == torch.bool
    assert not mask[:, :pl].any(), "the prefix must always be attendable"
    tri = mask[:, pl:]
    # Row i may attend to output columns 0..i and no further.
    for i in range(n):
        assert not tri[i, :i + 1].any(), f"row {i} must see outputs 0..{i}"
        assert bool(tri[i, i + 1:].all()) or i == n - 1, f"row {i} must not see the future"
    assert not tri.diagonal().any(), "a slot must be able to see itself"


# ----------------------------------------------------------------------
# Causality
# ----------------------------------------------------------------------
def test_a_later_token_cannot_change_an_earlier_logit():
    """The single test teacher forcing cannot do for itself.

    With a leaky mask the model reads the answer and the training loss looks excellent, while
    generation -- which has no future tokens to read -- collapses. Nothing but this shows the
    difference before a closed-loop run.
    """
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    a = torch.randint(0, m.vocab_size, (hist.shape[0], m.num_tokens))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 1) % m.vocab_size            # only the LAST target changes
    with torch.no_grad():
        la, lb = m(a, hist, extero, cmd), m(b, hist, extero, cmd)
    assert torch.equal(la, lb), "changing token N-1 must change nothing (it is never an input)"

    b = a.clone()
    b[:, 2] = (b[:, 2] + 1) % m.vocab_size
    with torch.no_grad():
        lb = m(b, hist, extero, cmd)
    assert torch.allclose(la[:, :3], lb[:, :3], atol=1e-6), "positions 0..2 must not see t_2"
    assert not torch.allclose(la[:, 3:], lb[:, 3:], atol=1e-6), "positions 3+ must see it"


# ----------------------------------------------------------------------
# The RL contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_sample_and_logprobs_agree_on_the_same_tokens(temperature):
    """The consistency the spec requires, and what a PPO ratio is built on.

    Not bit-identical, and cannot be: ``sample`` attends over a growing key set while
    ``logprobs`` attends over the whole block at once, so the two sum the same numbers in a
    different order in float32. The agreement is to ~1e-6 on values of magnitude ~9, i.e.
    pure round-off; a causality or off-by-one defect would be orders larger.
    """
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    g = torch.Generator().manual_seed(7)
    tokens, logp, ent = m.sample(hist, extero, cmd, temperature=temperature, generator=g)
    logp2, ent2 = m.logprobs(tokens, hist, extero, cmd, temperature=temperature)
    assert torch.allclose(logp, logp2, atol=1e-4)
    assert torch.allclose(ent, ent2, atol=1e-4)


def test_logprobs_are_per_token_and_are_real_log_probabilities():
    """Per token, never summed -- PPO needs to know WHICH decision an advantage belongs to."""
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    tokens = torch.randint(0, m.vocab_size, (hist.shape[0], m.num_tokens))
    logp, ent, logits = m.logprobs(tokens, hist, extero, cmd, return_logits=True)
    assert logp.shape == ent.shape == (hist.shape[0], m.num_tokens)
    assert logits.shape == (hist.shape[0], m.num_tokens, m.vocab_size)
    assert bool((logp <= 0.0).all())
    # The gathered value must be the log-softmax of the returned logits at that token.
    ref = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(logp, ref, atol=1e-6)
    # Entropy must be the real entropy, and bounded by a uniform over V.
    p = torch.softmax(logits, dim=-1)
    assert torch.allclose(ent, -(p * torch.log_softmax(logits, -1)).sum(-1), atol=1e-5)
    assert bool((ent <= torch.log(torch.tensor(float(m.vocab_size))) + 1e-4).all())


def test_sampling_is_reproducible_from_an_explicit_generator():
    """Rollouts have to be replayable; a draw off the global RNG is not."""
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    a = m.sample(hist, extero, cmd, generator=torch.Generator().manual_seed(3))[0]
    b = m.sample(hist, extero, cmd, generator=torch.Generator().manual_seed(3))[0]
    c = m.sample(hist, extero, cmd, generator=torch.Generator().manual_seed(4))[0]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_greedy_decoding_is_deterministic_and_takes_the_argmax():
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    a, logp, _ = m.sample(hist, extero, cmd, greedy=True)
    b, _, _ = m.sample(hist, extero, cmd, greedy=True)
    assert torch.equal(a, b)
    logits = m(a, hist, extero, cmd)
    assert torch.equal(a, logits.argmax(-1))
    # The reported log-probability is still the density of what was emitted.
    assert torch.allclose(
        logp, torch.log_softmax(logits, -1).gather(-1, a.unsqueeze(-1)).squeeze(-1), atol=1e-4)


def test_a_non_positive_temperature_is_refused():
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    for fn in (lambda t: m.sample(hist, extero, cmd, temperature=t),
               lambda t: m.logprobs(torch.zeros(3, m.num_tokens, dtype=torch.long),
                                    hist, extero, cmd, temperature=t)):
        with pytest.raises(ValueError, match="temperature"):
            fn(0.0)


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------
def test_the_head_is_tied_to_the_token_embedding():
    m = small_planner()
    assert m.head.weight is m.token_emb.weight
    assert m.head.bias is None


def test_the_tokenizer_is_frozen_and_decoding_matches_it():
    """A planner that fine-tuned its own tokenizer would be emitting tokens whose meaning
    drifts under it, so plans generated before the drift decode to something else after."""
    m = small_planner()
    assert not any(p.requires_grad for p in m.tokenizer.parameters())
    tokens = torch.randint(0, m.vocab_size, (2, m.num_tokens))
    assert torch.equal(m.decode(tokens), m.tokenizer.decode_ids(tokens))
    assert m.decode(tokens).shape == (2, m.cfg.tokenizer_cfg.chunk_len,
                                      m.cfg.tokenizer_cfg.node_dim)


def test_gradients_reach_the_trunk_but_not_the_tokenizer():
    m = small_planner().train()
    hist, extero, cmd = conditioning(m)
    tokens = torch.randint(0, m.vocab_size, (hist.shape[0], m.num_tokens))
    m(tokens, hist, extero, cmd).square().mean().backward()
    named = dict(m.named_parameters())
    for name in ("token_emb.weight", "bos", "out_pos", "out_type",
                 "prefix.history_in.0.weight", "encoder.layers.0.linear1.weight"):
        assert named[name].grad is not None and float(named[name].grad.abs().sum()) > 0.0, name
    assert all(p.grad is None for p in m.tokenizer.parameters())


def test_a_horizon_that_disagrees_with_the_tokenizer_is_refused():
    """The planner would emit a valid token sequence that decodes to a plan of the wrong
    length, and the shape error would surface far away, inside the tracker."""
    tok = TokenizerConfig(chunk_len=16, downsample=4, node_dim=12)
    with pytest.raises(ValueError, match="chunk IS one plan"):
        PlannerConfig(tokenizer=tok.to_dict(), future_len=32).validate()
    with pytest.raises(ValueError, match="nothing can interpret them"):
        PlannerConfig(tokenizer={}).validate()


def test_the_wrong_token_count_is_refused():
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    with pytest.raises(ValueError, match="token"):
        m(torch.zeros(3, m.num_tokens + 1, dtype=torch.long), hist, extero, cmd)


def test_the_null_history_path_changes_the_prediction():
    """The learned null row must actually be reachable through the planner, not only through
    the prefix module -- it covers the first N control steps after a reset."""
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    tokens = torch.randint(0, m.vocab_size, (hist.shape[0], m.num_tokens))
    with torch.no_grad():
        a = m(tokens, hist, extero, cmd, None)
        b = m(tokens, hist, extero, cmd, torch.ones(hist.shape[0], dtype=torch.bool))
    assert not torch.allclose(a, b)


def test_a_generator_on_the_wrong_device_is_refused_by_name():
    """``torch.multinomial``'s own error names neither the argument nor the fix, and it only
    fires on GPU -- so a rollout script that works on a laptop dies on the training box."""
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    if not torch.cuda.is_available():
        pytest.skip("needs a second device to have a wrong one")
    g = torch.Generator(device="cuda").manual_seed(0)
    with pytest.raises(ValueError, match="generator is on"):
        m.sample(hist, extero, cmd, generator=g)


def test_sampling_in_train_mode_is_refused():
    """The per-token log-probability is only exact against ``logprobs`` in eval mode.

    The shipped dropout is 0.1, so two independent draws over the same tokens disagree by up
    to 0.087 nats -- a ~9% error in a PPO importance ratio, in the direction of noise. A
    rollout collected from a model left in train mode is the easy way to cause that, and it
    would look like a noisy advantage rather than a bug.
    """
    m = small_planner(dropout=0.1).train()
    hist, extero, cmd = conditioning(m)
    with pytest.raises(RuntimeError, match="requires eval mode"):
        m.sample(hist, extero, cmd, greedy=True)
    m.eval()
    m.sample(hist, extero, cmd, greedy=True)      # no raise


def test_a_supplied_prefix_cache_gives_the_same_plan_as_re_encoding():
    """The prefix is a function of the conditioning alone, so handing back the cache
    ``forward`` just built must change nothing -- it is 18% of a validation batch."""
    m = small_planner()
    hist, extero, cmd = conditioning(m)
    tokens = torch.randint(0, m.vocab_size, (hist.shape[0], m.num_tokens))
    with torch.no_grad():
        m(tokens, hist, extero, cmd)              # populates _last_prefix_cache
        a, la, _ = m.sample(hist, extero, cmd, greedy=True)
        b, lb, _ = m.sample(hist, extero, cmd, greedy=True,
                            prefix_cache=m._last_prefix_cache)
    assert torch.equal(a, b)
    assert torch.allclose(la, lb, atol=1e-6)
    with pytest.raises(ValueError, match="one per layer"):
        m.sample(hist, extero, cmd, greedy=True, prefix_cache=m._last_prefix_cache[:-1])


def test_the_ema_does_not_average_the_frozen_tokenizer():
    """Every loader prefers ``model_ema``, so an averaged tokenizer is the one that SHIPS.

    Harmless for decoding, and not harmless for anything that re-ENCODES, because FSQ rounds
    to a lattice and a drifted encoder crosses cell boundaries.
    """
    from robot_rl.flow_matching.ema import WeightEMA

    m = small_planner().train()
    ema = WeightEMA(m, decay=0.9, freeze=("tokenizer.",))
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.5)
    hist, extero, cmd = conditioning(m)
    tokens = torch.randint(0, m.vocab_size, (hist.shape[0], m.num_tokens))
    for _ in range(20):
        opt.zero_grad()
        m(tokens, hist, extero, cmd).square().mean().backward()
        opt.step()
        ema.update(m)
    live = m.state_dict()
    tok_keys = [k for k in ema.shadow if k.startswith("tokenizer.")]
    assert tok_keys, "the tokenizer must be in the state dict at all"
    for k in tok_keys:
        assert torch.equal(ema.shadow[k], live[k]), f"{k} drifted from the frozen weights"
    # And the trunk IS averaged, or the freeze would be doing nothing useful.
    moved = [k for k in ema.shadow
             if k.startswith("encoder.") and not torch.equal(ema.shadow[k], live[k])]
    assert moved, "the trunk should have been averaged away from the live weights"


def test_decode_returns_a_real_rotation():
    """A decoder emitting six free numbers produces something only APPROXIMATELY a rotation.

    The flow-matching generator projects at exactly this point
    (``online.sample_window``), which is what lets the tracker, the world lift and the
    torch-free deploy port all assume the served window is a valid pose. Without it the
    autoregressive history also drifts off the manifold it was trained on -- and nothing
    downstream checks, because the block is the right WIDTH either way.
    """
    from robot_rl.flow_matching.normalizer import Normalizer

    m = small_planner()
    d = m.cfg.tokenizer_cfg.node_dim
    norm = Normalizer(torch.zeros(d), torch.ones(d), torch.ones(d, dtype=torch.bool))
    tokens = torch.randint(0, m.vocab_size, (4, m.num_tokens))
    raw = m.decode(tokens, norm, project_rotations=False)
    fixed = m.decode(tokens, norm)
    assert fixed.shape == raw.shape

    # base_pose is first in the packing, so its rotation half is columns 3:9.
    def gram_error(w):
        """Departure from orthonormality of the two rot6d columns."""
        a, b = w[..., 3:6], w[..., 6:9]
        return max(float((a.norm(dim=-1) - 1).abs().max()),
                   float((b.norm(dim=-1) - 1).abs().max()),
                   float((a * b).sum(-1).abs().max()))

    assert gram_error(raw) > 1e-3, "untrained decode should be off the manifold to begin with"
    assert gram_error(fixed) < 1e-5, "the projection must put it back on"
    # Everything OUTSIDE the rotation block is untouched -- the projection is not a filter.
    assert torch.equal(raw[..., :3], fixed[..., :3])
    assert torch.equal(raw[..., 9:], fixed[..., 9:])
