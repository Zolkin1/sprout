# Copyright 2026 Zachary Olkin. All rights reserved.

"""One-way conditioning: the prefix attends to itself, the future attends to everything.

Under full self-attention the conditioning tokens attend to the future tokens being denoised,
so the conditioning's own representation is a function of the noise level -- when it should be
a fixed statement about the robot's past, what it can see, and what it was asked to do. The
dependency should run one way.

The mode is implemented as a decomposition rather than as an attention mask, for two reasons:
a mask makes ``nn.TransformerEncoder`` compute the same ``S x S`` attention it always did (so
it buys no speed), and with a 2-D mask the fused inference kernel materialises
``(B, nhead, S, S)`` inside ``MultiheadAttention.merge_masks`` -- gigabytes at Isaac's env
counts. So the masked pass appears here ONLY as the reference the decomposition is checked
against.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.flow_matching.model import FlowMatchingTransformer

#: ``num_history_states=3`` so the prefix (3 + 2 extero + 2 cmd = 7) differs from the future
#: (6): with P == F the length assertions below would pass under either mode.
_KW = dict(state_dim=12, future_len=6, num_history_states=3, cmd_dim=3, cmd_stride=4,
           d_model=32, nhead=4, num_layers=3, dim_feedforward=64,
           extero_sensor_dim=8, extero_sensor_count=2)
_B = 3


def _model(mode: str = "prefix", **over) -> FlowMatchingTransformer:
    """A small model in the given attention mode, in eval (dropout is 0.0 regardless)."""
    kw = dict(_KW)
    kw.update(over)
    return FlowMatchingTransformer(attention_mode=mode, **kw).eval()


def _inputs(model: FlowMatchingTransformer, batch: int = _B, seed: int = 0):
    """``(x_tau, tau, history, extero, cmd_tokens)`` for a forward pass."""
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, model.future_len, model.state_dim, generator=g),
        torch.rand(batch, generator=g),
        torch.randn(batch, model.num_history_states, model.history_dim, generator=g),
        torch.randn(batch, model.extero_sensor_count, model.extero_sensor_dim, generator=g),
        torch.randn(batch, len(model.cmd_steps), model.cmd_dim, generator=g),
    )


def _masked_reference(model: FlowMatchingTransformer, x, tau, hist, ext, cmd) -> torch.Tensor:
    """The same computation via a prefix MASK on the stock encoder.

    ``mask[i, j] = True`` means "row i may not attend to column j" -- torch's bool convention
    for ``attn_mask`` is the opposite of ``scaled_dot_product_attention``'s, which is exactly
    the sort of thing this reference is here to pin.

    ``is_causal=False`` is passed explicitly: left as ``None``, ``_detect_is_causal_mask``
    builds a full ``(S, S)`` comparison mask and reduces it with ``.all()`` on every call,
    forcing a device-to-host sync.
    """
    p, s, f = model.prefix_len, model.sequence_len, model.future_len
    tokens = torch.cat([
        model._embed_prefix(hist, ext, cmd, None),
        model._embed_future(x) + model._time_embedding(tau),   # tau on the FUTURE only
    ], dim=1)
    mask = torch.zeros(s, s, dtype=torch.bool)
    mask[:p, p:] = True
    out = model.encoder(tokens, mask=mask, is_causal=False)
    return model.output_head(out[:, -f:, :])


##
# The gate
##

def test_prefix_attention_matches_a_masked_full_pass():
    """The hand-written decomposition must equal the stock encoder under a prefix mask.

    This is what proves the layer arithmetic: the post-norm placement
    (``norm1(x + attn)``, not ``x + norm1(attn)``), the three separate dropout modules,
    the fused in/out projections, and the mask semantics. Everything else in this file
    assumes this holds.
    """
    torch.manual_seed(0)
    m = _model()
    args = _inputs(m)
    with torch.no_grad():
        got, ref = m(*args), _masked_reference(m, *args)
    assert got.shape == (_B, m.future_len, m.state_dim)
    # float32 through 3 layers; the observed gap is ~7e-7.
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_prefix_and_future_lengths_partition_the_sequence():
    """The prefix is the LEADING contiguous block, which is why no reordering is needed."""
    m = _model()
    assert m.prefix_len == (m.num_history_states + m.extero_seq_tokens + len(m.cmd_steps))
    assert m.prefix_len + m.future_len == m.sequence_len


##
# What the mode is actually for
##

def test_the_cached_prefix_does_not_depend_on_the_flow_time():
    """The property the whole change exists to establish.

    Under ``prefix`` attention the conditioning cannot see the noised tokens and the flow-time
    embedding never reaches it, so its representation is identical at every Euler step. That
    is what makes it computable once per re-plan.
    """
    torch.manual_seed(0)
    m = _model()
    x, _, hist, ext, cmd = _inputs(m)
    with torch.no_grad():
        cache = m.encode_prefix(hist, ext, cmd)
        # Identical output at both ends of the flow-time range, and for different noise.
        a = m.decode_future(x, torch.zeros(_B), cache)
        b = m.decode_future(x, torch.ones(_B), cache)
        recache = m.encode_prefix(hist, ext, cmd)
    assert all(torch.equal(u, v) for u, v in zip(cache, recache))
    assert not torch.allclose(a, b, atol=1e-6), "tau must still reach the FUTURE tokens"


def test_full_attention_really_does_leak_the_noise_level_into_the_conditioning():
    """The bug, demonstrated -- otherwise the fix above is a claim about nothing.

    With full attention the conditioning rows attend to the noised future AND receive the
    flow-time embedding, so their representation moves with tau.
    """
    torch.manual_seed(0)
    m = _model("full")
    x, _, hist, ext, cmd = _inputs(m)
    p = m.prefix_len
    with torch.no_grad():
        pre, fut = m._embed_prefix(hist, ext, cmd, None), m._embed_future(x)
        rows = [m.encoder(torch.cat([pre, fut], dim=1) + m._time_embedding(t))[:, :p]
                for t in (torch.zeros(_B), torch.ones(_B))]
    assert not torch.allclose(rows[0], rows[1], atol=1e-6)


def test_one_cache_reused_across_steps_equals_re_encoding_each_step():
    """Bit-identical, which is what licenses hoisting the encode out of the Euler loop."""
    torch.manual_seed(0)
    m = _model()
    x, tau, hist, ext, cmd = _inputs(m)
    with torch.no_grad():
        reused = m.decode_future(x, tau, m.encode_prefix(hist, ext, cmd))
        fresh = m(x, tau, hist, ext, cmd)
    assert torch.equal(reused, fresh)


def test_the_future_still_depends_on_every_conditioning_stream():
    """One-way must not mean disconnected: history, extero and cmd all still reach the output."""
    torch.manual_seed(0)
    m = _model()
    x, tau, hist, ext, cmd = _inputs(m)
    with torch.no_grad():
        base = m(x, tau, hist, ext, cmd)
        for label, args in (("history", (x, tau, hist + 1.0, ext, cmd)),
                            ("extero", (x, tau, hist, ext + 1.0, cmd)),
                            ("cmd", (x, tau, hist, ext, cmd + 1.0))):
            assert not torch.allclose(m(*args), base, atol=1e-6), f"{label} does not reach it"


def test_the_null_history_branch_reaches_the_cache():
    """The CFG branch conditions through the prefix, so the mask must change the cache."""
    torch.manual_seed(0)
    m = _model()
    _, _, hist, ext, cmd = _inputs(m)
    with torch.no_grad():
        plain = m.encode_prefix(hist, ext, cmd, None)
        nulled = m.encode_prefix(hist, ext, cmd, torch.ones(_B, dtype=torch.bool))
    assert not torch.allclose(plain[0], nulled[0], atol=1e-6)


##
# Who attends to whom, at the attention call itself
##

@pytest.mark.parametrize("mode", ["full", "prefix"])
def test_every_attention_call_sees_the_right_query_and_key_lengths(mode):
    """A sharper statement than a sequence-length check: it pins WHO attends to whom.

    Read as the whole (query, key) shape of a forward pass:

    * ``full``   -- ``L`` calls of ``(S, S)``: everything attends to everything.
    * ``prefix`` -- ``L`` calls of ``(P, P)`` (the conditioning, on its own) followed by ``L``
      of ``(F, S)``: the future is the only query set, and it sees the whole sequence.

    The QUERY column is what distinguishes the modes; the key column cannot, since
    ``P + F == S`` by definition.
    """
    m = _model(mode)
    seen: list[tuple[int, int]] = []
    for layer in m.encoder.layers:
        layer.self_attn.register_forward_pre_hook(
            lambda _m, inp: seen.append((inp[0].shape[1], inp[1].shape[1])))
    with torch.no_grad():
        m(*_inputs(m))

    n_layers, p, f, s = len(m.encoder.layers), m.prefix_len, m.future_len, m.sequence_len
    if mode == "full":
        assert seen == [(s, s)] * n_layers
    else:
        assert seen == [(p, p)] * n_layers + [(f, s)] * n_layers


def test_encode_prefix_attends_only_within_the_prefix():
    """The conditioning pass must never see a future row, at any layer."""
    m = _model()
    seen: list[tuple[int, int]] = []
    for layer in m.encoder.layers:
        layer.self_attn.register_forward_pre_hook(
            lambda _m, inp: seen.append((inp[0].shape[1], inp[1].shape[1])))
    _, _, hist, ext, cmd = _inputs(m)
    with torch.no_grad():
        m.encode_prefix(hist, ext, cmd)
    assert seen == [(m.prefix_len, m.prefix_len)] * len(m.encoder.layers)


##
# The samplers -- where the cache is actually hoisted out of the Euler loop
##

def _uncached_euler(model, hist, ext, cmd, noise, num_steps, null_mask=None) -> torch.Tensor:
    """``sample_trajectory``'s integration, written out with a full ``forward`` per step.

    The independent reference for the hoist: if encoding the conditioning once and reusing it
    is not equivalent to re-encoding it every step, these two disagree.
    """
    x, dt = noise.clone(), 1.0 / num_steps
    with torch.no_grad():
        for i in range(num_steps):
            tau = torch.full((x.shape[0],), i * dt, dtype=x.dtype)
            x = x + dt * model(x, tau, hist, ext, cmd, history_null_mask=null_mask)
    return x


def test_sample_trajectory_hoists_the_encode_without_changing_the_result():
    """The hoist must be exactly an optimisation.

    The ONNX parity test cannot catch a bad hoist -- both sides of it hoist -- so this is
    what does: one cached prefix across all steps against a full forward per step.
    """
    from robot_rl.flow_matching.flow import sample_trajectory

    torch.manual_seed(0)
    m = _model()
    _, _, hist, ext, cmd = _inputs(m)
    noise = torch.randn(_B, m.future_len, m.state_dim, generator=torch.Generator().manual_seed(7))
    got = sample_trajectory(m, hist, ext, cmd, m.future_len, m.state_dim, num_steps=4,
                            zero_noise=True)
    ref = _uncached_euler(m, hist, ext, cmd, torch.zeros_like(noise), 4)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["full", "prefix"])
def test_the_encode_happens_once_per_sample_not_once_per_step(mode):
    """Count the attention calls: the conditioning pass must not scale with ``num_steps``.

    ``prefix`` at ``num_steps=4`` should be ``L`` conditioning calls plus ``4 * L`` future
    calls; ``full`` is ``4 * L`` calls over the whole sequence and has no way to be otherwise.
    """
    from robot_rl.flow_matching.flow import sample_trajectory

    torch.manual_seed(0)
    m = _model(mode)
    _, _, hist, ext, cmd = _inputs(m)
    seen: list[tuple[int, int]] = []
    for layer in m.encoder.layers:
        layer.self_attn.register_forward_pre_hook(
            lambda _m, inp: seen.append((inp[0].shape[1], inp[1].shape[1])))
    sample_trajectory(m, hist, ext, cmd, m.future_len, m.state_dim, num_steps=4, zero_noise=True)

    n_layers, p, f, s = len(m.encoder.layers), m.prefix_len, m.future_len, m.sequence_len
    if mode == "prefix":
        assert seen.count((p, p)) == n_layers, "the conditioning was encoded more than once"
        assert seen.count((f, s)) == 4 * n_layers
    else:
        assert seen == [(s, s)] * (4 * n_layers)


##
# Classifier-free guidance -- untested before this change, and this change restructures it
##

def test_guided_sampling_matches_an_uncached_two_branch_reference():
    """CFG builds the prefix once at 2B (conditional half + null half). Verify that.

    The guidance branch had NO test coverage at all, and prefix mode reorganises exactly it,
    so a mistake here would have gone unnoticed.
    """
    from robot_rl.flow_matching.flow import sample_trajectory

    torch.manual_seed(0)
    m = _model()
    _, _, hist, ext, cmd = _inputs(m)
    w, steps = 2.5, 4
    got = sample_trajectory(m, hist, ext, cmd, m.future_len, m.state_dim, num_steps=steps,
                            zero_noise=True, guidance_scale=w)

    # The same integration with the two branches run separately, un-cached.
    x, dt = torch.zeros(_B, m.future_len, m.state_dim), 1.0 / steps
    null = torch.ones(_B, dtype=torch.bool)
    with torch.no_grad():
        for i in range(steps):
            tau = torch.full((_B,), i * dt)
            v_cond = m(x, tau, hist, ext, cmd, history_null_mask=None)
            v_null = m(x, tau, hist, ext, cmd, history_null_mask=null)
            x = x + dt * (v_null + w * (v_cond - v_null))
    torch.testing.assert_close(got, x, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["full", "prefix"])
def test_guidance_scale_of_one_is_a_no_op(mode):
    """``w = 1`` reduces to ``v_cond``, so it must equal the unguided path exactly."""
    from robot_rl.flow_matching.flow import sample_trajectory

    torch.manual_seed(0)
    m = _model(mode)
    _, _, hist, ext, cmd = _inputs(m)
    kw = dict(future_len=m.future_len, state_dim=m.state_dim, num_steps=4, zero_noise=True)
    plain = sample_trajectory(m, hist, ext, cmd, **kw)
    guided = sample_trajectory(m, hist, ext, cmd, guidance_scale=1.0, **kw)
    torch.testing.assert_close(plain, guided, atol=1e-6, rtol=1e-6)


##
# Guard rails
##

def test_full_attention_refuses_to_hand_out_a_cache():
    """There is nothing stable to cache when the conditioning can see the future."""
    with pytest.raises(RuntimeError, match="attention_mode='prefix'"):
        _, _, hist, ext, cmd = _inputs(_model("full"))
        _model("full").encode_prefix(hist, ext, cmd)


def test_a_cache_of_the_wrong_depth_is_refused():
    """One entry per layer -- a short cache would silently skip layers' conditioning."""
    m = _model()
    x, tau, hist, ext, cmd = _inputs(m)
    cache = m.encode_prefix(hist, ext, cmd)
    with pytest.raises(ValueError, match="one per layer"):
        m.decode_future(x, tau, cache[:-1])


def test_unknown_attention_mode_is_refused():
    with pytest.raises(ValueError, match="unknown attention_mode"):
        _model("causal")


def test_the_two_modes_share_one_parameter_set():
    """No parameters distinguish them -- which is WHY the config flag is load-bearing.

    ``load_state_dict`` cannot catch a mislabelled checkpoint the way it catches a wrong
    ``extero_encoder``, so ``FlowMatchingConfig.from_dict``'s default is the only guard.
    """
    full, prefix = _model("full"), _model("prefix")
    assert set(dict(full.named_parameters())) == set(dict(prefix.named_parameters()))
    # ...and a checkpoint from either loads into the other without complaint. Silently.
    prefix.load_state_dict(full.state_dict())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
