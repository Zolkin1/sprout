# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the critic's architecture, and above all pin the claim that makes it cheap.

The value model borrows the generator's encoder and adds tokens beside its conditioning. That
is only sound because the generator runs `prefix` attention, under which its conditioning
attends ONLY to itself -- so tokens added alongside cannot perturb it and `encode_prefix`
returns bit-for-bit what the generator computes. If that ever stopped holding, the critic
would be silently reading a different representation from the one the generator acts on, and
every gap measured between them would be meaningless.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: E402
from robot_rl.flow_matching.value import FlowValueModel, ValueInputs  # noqa: E402

D, F, N, T, K, CMD = 16, 12, 3, 1, 24, 3
# The real height scan's grid (176 rays = 11 x 16). Not a smaller toy: at (4, 6)
# a stride-2 trunk's native grid is (1, 2) and the encoder rightly REFUSES a
# larger pooled grid rather than inventing tokens, so a toy size would be
# testing the refusal instead of the model.
SCAN_HW, P_DIM, P_STEPS, CTX = (11, 16), 8, 5, 9


def _fm():
    return FlowMatchingTransformer(
        state_dim=D, future_len=F, num_history_states=N, extero_sensor_dim=K,
        extero_sensor_count=T, cmd_dim=CMD, cmd_stride=4, d_model=32, nhead=4,
        num_layers=2, dim_feedforward=64, attention_mode="prefix").eval()


def _inputs(b=4, *, scan=True, proprio=True, ctx=True):
    g = torch.Generator().manual_seed(0)
    return ValueInputs(
        history=torch.randn(b, N, D, generator=g),
        extero=torch.randn(b, T * K, generator=g),
        cmd_tokens=torch.randn(b, len(range(0, F, 4)), CMD, generator=g),
        history_null=torch.zeros(b, dtype=torch.bool),
        scan=torch.randn(b, *SCAN_HW, generator=g) if scan else None,
        proprio=torch.randn(b, P_STEPS, P_DIM, generator=g) if proprio else None,
        context=torch.randn(b, CTX, generator=g) if ctx else None,
    )


def _model(fm, **kw):
    base = dict(scan_hw=SCAN_HW, proprio_dim=P_DIM, proprio_steps=P_STEPS, context_dim=CTX,
                scan_grid=(2, 3), scan_channels=(4, 8))
    base.update(kw)
    return FlowValueModel(fm, **base).eval()


# ----------------------------------------------------------------------------------------
# THE claim.
# ----------------------------------------------------------------------------------------

def test_the_generators_prefix_is_UNCHANGED_by_the_value_tokens():
    """Bit-for-bit, which is what makes borrowing the encoder exact rather than approximate.

    Under `prefix` attention the conditioning is not in the query set, so tokens added to the
    query stream cannot enter its keys. If this failed, the critic would be scoring a
    representation the generator never computes -- and every V_priv/V_onboard gap, which is
    supposed to measure what the generator cannot SEE, would instead be measuring that the
    two models were reading different inputs.
    """
    fm = _fm()
    x = _inputs()
    with torch.no_grad():
        alone = fm.encode_prefix(x.history, x.extero, x.cmd_tokens, x.history_null)
        m = _model(fm)
        _ = m(x)
        after = fm.encode_prefix(x.history, x.extero, x.cmd_tokens, x.history_null)
    for a, b in zip(alone, after):
        assert torch.equal(a, b)


def test_full_attention_is_REFUSED():
    """Under `full` the conditioning attends to the future and to the flow time, so there is
    no fixed encoding to borrow -- and adding tokens would change what it means."""
    fm = FlowMatchingTransformer(state_dim=D, future_len=F, num_history_states=N,
                                 extero_sensor_dim=K, extero_sensor_count=T, cmd_dim=CMD,
                                 cmd_stride=4, d_model=32, nhead=4, num_layers=2,
                                 attention_mode="full")
    with pytest.raises(ValueError, match="prefix-attention"):
        FlowValueModel(fm)


# ----------------------------------------------------------------------------------------
# The readout token.
# ----------------------------------------------------------------------------------------

def test_the_value_comes_from_the_READOUT_token_alone():
    """Position 0 of the query stream, and nothing else reaches the head."""
    fm = _fm()
    m = _model(fm)
    x = _inputs()
    with torch.no_grad():
        toks = m.value_tokens(x)
    assert toks.shape[1] == 1 + 2 * 3 + P_STEPS + 1     # readout + 2x3 scan + proprio + ctx
    assert torch.allclose(toks[:, 0], (m.readout + m.readout_type).expand_as(toks[:, :1])[:, 0])


def test_the_readout_is_trainable_so_its_QUERY_can_learn_where_to_look():
    """The whole argument for a readout over mean-pooling: the weights of the sum are the
    token's own query, and gradient descent points it at whatever predicts failure."""
    fm = _fm()
    m = _model(fm)
    out = m(_inputs())
    out.sum().backward()
    assert m.readout.grad is not None and m.readout.grad.abs().sum() > 0


def test_the_frozen_trunk_gets_no_gradient():
    fm = _fm()
    m = _model(fm, freeze_trunk=True)
    m(_inputs()).sum().backward()
    assert all(p.grad is None for p in fm.parameters())
    assert m.quality_head[0].weight.grad is not None


# ----------------------------------------------------------------------------------------
# The streams. Each absence is silent unless refused.
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("drop,match", [
    ("scan", "as if the ground were flat"),
    ("proprio", "proprio tokens"),
    ("ctx", "goal offsets"),
])
def test_a_missing_stream_is_REFUSED_not_zero_filled(drop, match):
    """Zero-filling would score every state against a flat floor, or a goal at the origin --
    plausible numbers, right shapes, wrong model."""
    fm = _fm()
    m = _model(fm)
    x = _inputs(**{drop: False})
    with pytest.raises(ValueError, match=match):
        m(x)


def test_the_onboard_variant_builds_with_no_privileged_streams():
    fm = _fm()
    m = FlowValueModel(fm, scan_hw=None, proprio_dim=0, proprio_steps=0, context_dim=CTX)
    x = _inputs(scan=False, proprio=False)
    assert m(x).shape == (4,)
    assert m.value_tokens(x).shape[1] == 2          # readout + context only














def test_the_critic_is_ONE_readout_and_ONE_scalar_head() -> None:
    """The architecture the paper describes, pinned so the description stays true.

    This class used to carry a survival head predicting ``P(fail | s)`` on its OWN readout
    token, plus a distributional (two-hot) form of the value head selected by
    ``quality_kind``, plus an unfitted non-privileged variant. None of them produced a
    reported result: the fine-tune loop passes ``--quality_only``, which drops the survival
    loss entirely, and ``--quality_kind mse``.

    They are deleted rather than documented-as-unused, because an architecture section cannot
    describe a head without a reader asking what it was for, and the honest answer was "a
    different experiment".

    Deleting the survival READOUT was not cosmetic and this test is also the record of why:
    the value readout attends to the rest of the stream, so removing a token changes the
    representation it reads. Every critic fitted before the cut is a different function, which
    is why those checkpoints do not load and why numbers measured with one do not describe
    this model.
    """
    from robot_rl.flow_matching import value as V

    gone_methods = ("p_fail", "heads", "combined_value", "value")
    present = [n for n in gone_methods if hasattr(V.FlowValueModel, n)]
    assert not present, f"the survival interface is back: {present}"

    gone_module = ("two_hot", "quality_loss")
    present = [n for n in gone_module if hasattr(V, n)]
    assert not present, f"the two-hot machinery is back: {present}"

    import inspect
    params = set(inspect.signature(V.FlowValueModel.__init__).parameters)
    for knob in ("quality_bins", "quality_kind", "quality_range"):
        assert knob not in params, (
            f"{knob} is back in the ctor; the head is a scalar regression with no options")

    m = _model(_fm())
    heads = [n for n, _ in m.named_children() if n.endswith("head")]
    assert heads == ["quality_head"], f"expected exactly one head, got {heads}"
    assert m.quality_head[-1].out_features == 1, (
        "the head must emit ONE scalar; a wider output is the distributional form returning")
    # The TOKEN, not its type embedding: every stream has a `*_type`, so `readout_type`
    # legitimately survives alongside the single `readout` query.
    readouts = [n for n, _ in m.named_parameters()
                if "readout" in n and not n.endswith("_type")]
    assert readouts == ["readout"], f"expected exactly one readout token, got {readouts}"


def test_the_head_reads_the_READOUT_token_and_the_index_is_pinned() -> None:
    """The riskiest line in the single-head cut, and nothing else guards it.

    ``quality_logits`` reads ``_encode(x)[:, 0]``. It read index 1 while a survival readout
    occupied 0, and the cut moved it. Get that index wrong and the head reads a scan patch or
    a proprio token instead of its own query: finite, plausible, differentiable, and silently
    predicting from the wrong representation.

    A review found this unpinned -- reverting the index to ``[:, 1]`` left the whole suite
    green, because the parameter-count assertions above cannot see it and
    ``test_the_readout_is_trainable`` cannot either (under ``prefix`` attention every token
    attends to the readout, so its gradient stays non-zero whichever token the head reads).

    Checked by construction rather than by value: make the readout token identifiable, then
    assert the head's input is the row ``value_tokens`` put first.
    """
    torch.manual_seed(0)
    m = _model(_fm())
    x = _inputs()
    with torch.no_grad():
        tokens = m.value_tokens(x)
        encoded = m._encode(x)
        # The readout is parts[0] on the QUERY side, and `_encode` preserves the query order,
        # so the head's input must be column 0 of the encoded stream.
        taken = m.quality_head[0](encoded[:, 0])
        assert torch.allclose(m.quality_logits(x), m.quality_head[1:](taken), atol=0), (
            "quality_logits does not read the encoded readout at index 0")
    assert tokens.shape[1] == encoded.shape[1], (
        "the encoder changed the token count; the head's index is no longer meaningful")

    import inspect

    src = inspect.getsource(m.quality_logits)
    assert "[:, 0]" in src, (
        "the head's token index is not 0 in the source; value_tokens puts the readout first, "
        "so any other index reads a scan patch or a proprio token")
