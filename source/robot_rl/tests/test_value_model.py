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
    assert m.head[0].weight.grad is not None


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


def test_p_fail_and_value_are_consistent_and_bounded():
    fm = _fm()
    m = _model(fm)
    x = _inputs()
    with torch.no_grad():
        p, v = m.p_fail(x), m.value(x)
    assert torch.all((p >= 0) & (p <= 1))
    assert torch.allclose(v, -p)
