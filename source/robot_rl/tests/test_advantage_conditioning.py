# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the advantage label, and above all pin that adding it changes NOTHING at first.

The whole method rests on fine-tuning an EXISTING generator rather than retraining one, so
the conversion has to be provably inert: a checkpoint with the label embedding bolted on must
produce bit-identical output to the checkpoint it came from, under every label. If it does
not, the fine-tune starts from a model that is already different and nothing downstream can
attribute a change to the conditioning.

Zero-initialising the embedding is what buys that, and it only works because the label is
added to the FLOW-TIME embedding rather than introduced as a new prefix token. An extra token
is not a no-op at any initialisation -- it is another key in the attention softmax, so it
changes every other token's output -- and it would move `prefix_len`, and with it the exported
graph's shapes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: E402

D, F, N, T, K, CMD = 16, 12, 3, 1, 24, 3
NEG, POS, NULL = 0, 1, 2


def _model(advantage_cond, mode="prefix", seed=0):
    torch.manual_seed(seed)
    return FlowMatchingTransformer(
        state_dim=D, future_len=F, num_history_states=N, extero_sensor_dim=K,
        extero_sensor_count=T, cmd_dim=CMD, cmd_stride=4, d_model=32, nhead=4,
        num_layers=2, dim_feedforward=64, attention_mode=mode,
        advantage_cond=advantage_cond).eval()


def _inputs(b=4):
    g = torch.Generator().manual_seed(7)
    return dict(
        x_tau=torch.randn(b, F, D, generator=g),
        tau=torch.rand(b, generator=g),
        history=torch.randn(b, N, D, generator=g),
        extero=torch.randn(b, T * K, generator=g),
        cmd_tokens=torch.randn(b, len(range(0, F, 4)), CMD, generator=g),
    )


@pytest.mark.parametrize("mode", ["prefix", "full"])
@pytest.mark.parametrize("label", [NEG, POS, NULL])
def test_a_converted_checkpoint_is_BIT_IDENTICAL_under_every_label(mode, label):
    """The conversion must be inert, or the fine-tune does not start where it claims to."""
    plain, cond = _model(False, mode), _model(True, mode)
    # Same weights, plus the zero embedding: exactly what converting a checkpoint does.
    missing, unexpected = cond.load_state_dict(plain.state_dict(), strict=False)
    assert unexpected == [], unexpected
    assert set(missing) == {"advantage_emb.weight"}, missing
    x = _inputs()
    with torch.no_grad():
        a = plain(**x)
        b = cond(**x, advantage=torch.full((4,), label, dtype=torch.long))
    assert torch.equal(a, b), f"{mode}/{label}: conversion is not a no-op"


def test_the_embedding_starts_at_exactly_zero():
    assert torch.count_nonzero(_model(True).advantage_emb.weight) == 0


def test_the_label_MATTERS_once_the_embedding_is_trained():
    """Inert at init is only useful if it stops being inert. Perturb and the output must move."""
    m = _model(True)
    with torch.no_grad():
        m.advantage_emb.weight.normal_(0.0, 0.5)
    x = _inputs()
    with torch.no_grad():
        pos = m(**x, advantage=torch.full((4,), POS, dtype=torch.long))
        neg = m(**x, advantage=torch.full((4,), NEG, dtype=torch.long))
    assert not torch.allclose(pos, neg)


def test_the_labels_are_INDEPENDENT_of_each_other():
    """Three separate rows, so `null` is its own branch and not an average of the other two.

    `null` has to be a real third option -- it is what signal-free data is labelled, and what
    classifier-free guidance pushes away from -- so tying it to the others would collapse the
    dropout the guidance depends on.
    """
    m = _model(True)
    with torch.no_grad():
        m.advantage_emb.weight.normal_(0.0, 0.5)
    x = _inputs(1)
    with torch.no_grad():
        out = [m(**x, advantage=torch.tensor([lb])) for lb in (NEG, POS, NULL)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.allclose(out[i], out[j])


def test_the_prefix_is_UNTOUCHED_by_the_label():
    """The label rides the flow-time embedding, so the conditioning encoding cannot see it.

    That is what makes the prefix cache SHARED across label values -- classifier-free guidance
    then costs one extra decode instead of a second full encode -- and it is why the ONNX
    graph's prefix hoisting needs no change.
    """
    m = _model(True)
    with torch.no_grad():
        m.advantage_emb.weight.normal_(0.0, 0.5)
    x = _inputs()
    with torch.no_grad():
        c1 = m.encode_prefix(x["history"], x["extero"], x["cmd_tokens"])
        _ = m(**x, advantage=torch.full((4,), POS, dtype=torch.long))
        c2 = m.encode_prefix(x["history"], x["extero"], x["cmd_tokens"])
    for a, b in zip(c1, c2):
        assert torch.equal(a, b)


def test_a_label_on_an_UNCONDITIONED_model_is_refused():
    """Dropping it would leave a 'conditioned' run that never saw the label -- and would
    train, and export, and look entirely normal."""
    m = _model(False)
    with pytest.raises(ValueError, match="advantage_cond=False"):
        m(**_inputs(), advantage=torch.zeros(4, dtype=torch.long))


def test_omitting_the_label_MEANS_null_rather_than_meaning_nothing():
    """``None`` must resolve to the null ROW, not skip the addition.

    Two reasons, and the second is the one that bites. (a) For guidance the conditional and
    unconditional passes should differ by an embedding VALUE and nothing else; if one of them
    is missing a whole term, that asymmetry lands in the guidance direction as an artifact.
    (b) An UNLABELLED fine-tune -- the control arm, `--labels ""` -- would otherwise never add
    the embedding at all, so it would never send it gradient, and the resulting checkpoint
    would claim to be conditioned while carrying an untouched zero table.
    """
    m = _model(True)
    with torch.no_grad():
        m.advantage_emb.weight.normal_(0.0, 0.5)
    x = _inputs()
    with torch.no_grad():
        omitted = m(**x)
        explicit_null = m(**x, advantage=torch.full((4,), NULL, dtype=torch.long))
        positive = m(**x, advantage=torch.full((4,), POS, dtype=torch.long))
    assert torch.equal(omitted, explicit_null)
    assert not torch.allclose(omitted, positive)


def test_the_null_index_buffer_stays_OUT_of_the_state_dict():
    """It is a constant, so persisting it would break the one-new-key conversion contract."""
    assert "advantage_null_idx" not in _model(True).state_dict()


def test_a_shape_1_index_broadcasts_over_the_batch():
    """The ONNX export bakes the label as a shape-``(1,)`` CONSTANT, and that must be enough.

    ``nn.Embedding`` on a ``(1,)`` index returns ``(1, d_model)``, which broadcasts against
    the ``(B, d_model)`` flow-time embedding -- so one constant serves any batch and the
    graph needs no int64 arithmetic on an input.  That is not cosmetic: onnxruntime keeps
    int64 work on the HOST, and a single host node makes the whole graph ineligible for CUDA
    Graph capture, which the deploy runtime depends on.
    """
    m = _model(True)
    torch.nn.init.normal_(m.advantage_emb.weight, std=0.5)
    kw = _inputs(b=4)
    per_row = m(**kw, advantage=torch.full((4,), POS, dtype=torch.long))
    broadcast = m(**kw, advantage=torch.tensor([POS]))
    assert torch.equal(per_row, broadcast)


def test_the_conversion_adds_exactly_one_state_dict_key():
    """Anything else missing would mean weights silently failed to transfer.

    The fine-tune asserts this at load time (``finetune_advantage.convert``); pinning it here
    means a future architecture change that quietly renames a parameter fails in a unit test
    rather than in an overnight run.
    """
    plain, cond = _model(False), _model(True)
    missing, unexpected = cond.load_state_dict(plain.state_dict(), strict=False)
    assert set(missing) == {"advantage_emb.weight"}
    assert not unexpected


# --------------------------------------------------------------------------------------
# prefix_token injection. The additive variant was MEASURED INADEQUATE on real data (the
# label moved the served window by 0.3-0.7% of ordinary sampling noise across three
# configurations, including one whose whole model demonstrably moved), so the label gets its
# own TOKEN -- with its own attention weights at every layer -- as the structural alternative.
# --------------------------------------------------------------------------------------

def _tok_model(seed=0):
    """A prefix_token-injection model."""
    torch.manual_seed(seed)
    return FlowMatchingTransformer(
        state_dim=D, future_len=F, num_history_states=N, extero_sensor_dim=K,
        extero_sensor_count=T, cmd_dim=CMD, cmd_stride=4, d_model=32, nhead=4,
        num_layers=2, dim_feedforward=64, attention_mode="prefix",
        advantage_cond=True, advantage_inject="prefix_token").eval()


def test_the_token_variant_GROWS_prefix_len_by_exactly_one():
    """It is one extra conditioning token -- and that is the cost the additive form avoided.

    `prefix_len` feeds the exported graph's shapes, so this is why a time_embed checkpoint and
    a prefix_token one are not interchangeable artifacts and the mode has to ride in the file.
    """
    assert _tok_model().prefix_len == _model(True).prefix_len + 1
    assert _tok_model().sequence_len == _model(True).sequence_len + 1


def test_the_token_CHANGES_the_output_between_labels():
    """The whole point: a differently-valued token must produce a different prediction."""
    m = _tok_model()
    x = _inputs()
    with torch.no_grad():
        pos = m(**x, advantage=torch.full((4,), POS, dtype=torch.long))
        neg = m(**x, advantage=torch.full((4,), NEG, dtype=torch.long))
    assert not torch.allclose(pos, neg)


def test_the_token_reaches_the_PREFIX_CACHE_unlike_the_additive_form():
    """This is the structural difference, stated as a test.

    Under `time_embed` the label rides the flow-time embedding, so `encode_prefix` is
    label-INDEPENDENT and one cache serves every label. Under `prefix_token` the label IS a
    conditioning token, so the cache itself differs -- which is exactly what lets it modulate
    how the future tokens attend to the rest of the conditioning, and also why CFG over the
    label costs a second encode here.
    """
    x = _inputs()
    add = _model(True)
    with torch.no_grad():
        torch.nn.init.normal_(add.advantage_emb.weight, std=0.5)
        c_pos = add.encode_prefix(x["history"], x["extero"], x["cmd_tokens"],
                                  advantage=torch.full((4,), POS, dtype=torch.long))
        c_neg = add.encode_prefix(x["history"], x["extero"], x["cmd_tokens"],
                                  advantage=torch.full((4,), NEG, dtype=torch.long))
    for a, b in zip(c_pos, c_neg):
        assert torch.equal(a, b), "time_embed: the prefix must NOT depend on the label"

    tok = _tok_model()
    with torch.no_grad():
        t_pos = tok.encode_prefix(x["history"], x["extero"], x["cmd_tokens"],
                                  advantage=torch.full((4,), POS, dtype=torch.long))
        t_neg = tok.encode_prefix(x["history"], x["extero"], x["cmd_tokens"],
                                  advantage=torch.full((4,), NEG, dtype=torch.long))
    assert not torch.allclose(t_pos[0], t_neg[0]), \
        "prefix_token: the prefix MUST depend on the label"


def test_omitting_the_label_still_means_null_under_the_token():
    """Same convention as the additive path: `None` is the null ROW, not a missing token.

    Otherwise the unconditioned call would produce a DIFFERENT SEQUENCE LENGTH from a
    conditioned one, which would be a far worse failure than a wrong value.
    """
    m = _tok_model()
    x = _inputs()
    with torch.no_grad():
        omitted = m(**x)
        explicit = m(**x, advantage=torch.full((4,), NULL, dtype=torch.long))
    assert torch.equal(omitted, explicit)


def test_the_token_rows_do_NOT_start_at_zero():
    """A zero token still changes the attention softmax while carrying no signal to separate
    the rows -- the worst of both worlds, so the token variant starts at a real scale."""
    m = _tok_model()
    assert m.advantage_emb.weight.norm() > 0.1


def test_an_unrecognised_injection_mode_is_refused():
    """A silent fallback would train one architecture while the config claimed another."""
    with pytest.raises(ValueError, match="advantage_inject"):
        FlowMatchingTransformer(
            state_dim=D, future_len=F, num_history_states=N, extero_sensor_dim=K,
            extero_sensor_count=T, cmd_dim=CMD, cmd_stride=4, d_model=32, nhead=4,
            num_layers=2, dim_feedforward=64, advantage_cond=True,
            advantage_inject="both")


def test_sample_trajectory_ROUTES_the_label_into_the_cached_prefix():
    """The hoisted `encode_prefix` must receive the label, or the token path serves `null`.

    REGRESSION. `sample_trajectory` encodes the conditioning ONCE outside the Euler loop
    (that is the whole point of `prefix` attention). The first version of the token variant
    did not pass `advantage` to that call, so under `prefix_token` both arms encoded a `null`
    prefix and a positive-conditioned sample came out BIT-IDENTICAL to a negative-conditioned
    one -- served windows differing by exactly 0.0000, which reads as "the model ignored the
    label" rather than as a plumbing bug. The unit tests above did not catch it because they
    call `model(...)` directly; only the sampler hoists.
    """
    from robot_rl.flow_matching.flow import sample_trajectory

    m = _tok_model()
    x = _inputs()
    noise = torch.randn(4, F, D, generator=torch.Generator().manual_seed(0))
    kw = dict(future_len=F, state_dim=D, num_steps=3, noise=noise)
    with torch.no_grad():
        pos = sample_trajectory(m, x["history"], x["extero"], x["cmd_tokens"],
                                advantage=torch.full((4,), POS, dtype=torch.long), **kw)
        neg = sample_trajectory(m, x["history"], x["extero"], x["cmd_tokens"],
                                advantage=torch.full((4,), NEG, dtype=torch.long), **kw)
    assert not torch.allclose(pos, neg), \
        "the label did not reach the cached prefix: both arms sampled identically"
