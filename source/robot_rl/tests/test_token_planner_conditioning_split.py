# Copyright 2026 Zachary Olkin. All rights reserved.

"""``split_conditioning`` must invert exactly how ``_run_fm`` packs the stash.

The RL policy samples from a flat row the command term built for the plan it drew. That row is
``[history | extero | cmd | null_flag]``, and the env slices it back apart. A swapped order or
an off-by-one here would still RUN -- every shape stays right -- and would condition the
policy on scrambled inputs. That is the same class of failure as the raw-vs-normalized bug
that had a planner emitting coherent downstairs plans on flat ground: plausible output, wrong
input, nothing raising.

No simulator: the split reads only widths off the context, so a stub context exercises it.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from robot_rl.tasks.manager_based.robot_rl.envs.token_planner_env import (  # noqa: E402
    TokenPlannerEnv,
)

N, D, SENSORS, SENSOR_DIM, CMD_TOK, CMD_DIM = 4, 44, 2, 3900, 13, 3


def _env(cmd_dim: int = CMD_DIM):
    """A stub carrying only what :meth:`split_conditioning` reads."""
    ctx = types.SimpleNamespace(
        N=N, D=D, extero_sensor_count=SENSORS, extero_sensor_dim=SENSOR_DIM,
        num_cmd_tokens=CMD_TOK, cmd_dim=cmd_dim)
    return types.SimpleNamespace(planner_ctx=ctx)


def _pack(hist, extero, cmd, null):
    """Flatten exactly as ``FmTrackingCommand._run_fm`` builds ``_cond_stash``."""
    parts = [hist.flatten(1), extero.flatten(1)]
    if cmd is not None:
        parts.append(cmd.flatten(1))
    parts.append(null.unsqueeze(-1).float())
    return torch.cat(parts, dim=-1)


def test_the_split_inverts_the_packing_exactly() -> None:
    """Round trip on distinguishable values, so a swapped order cannot pass."""
    b = 5
    torch.manual_seed(0)
    hist = torch.randn(b, N, D)
    extero = torch.randn(b, SENSORS * SENSOR_DIM)
    cmd = torch.randn(b, CMD_TOK, CMD_DIM)
    null = torch.tensor([True, False, True, False, True])
    cond = _pack(hist, extero, cmd, null)

    h2, e2, c2, n2 = TokenPlannerEnv.split_conditioning(_env(), cond)
    assert torch.equal(h2, hist), "history did not survive the round trip"
    assert torch.equal(e2, extero), "exteroception did not survive the round trip"
    assert torch.equal(c2, cmd), "command tokens did not survive the round trip"
    assert torch.equal(n2, null), "the null-history flag did not survive the round trip"


def test_a_swapped_order_would_be_caught() -> None:
    """The round-trip test is only meaningful if a wrong packing actually fails it."""
    b = 3
    hist = torch.full((b, N, D), 1.0)
    extero = torch.full((b, SENSORS * SENSOR_DIM), 2.0)
    cmd = torch.full((b, CMD_TOK, CMD_DIM), 3.0)
    null = torch.zeros(b, dtype=torch.bool)
    # extero and history swapped: same total width, different meaning
    wrong = torch.cat([extero, hist.flatten(1), cmd.flatten(1), null.unsqueeze(-1).float()], -1)
    h2, e2, _, _ = TokenPlannerEnv.split_conditioning(_env(), wrong)
    assert not torch.equal(h2, hist), (
        "a history/extero swap produced the right history, so this split cannot detect one"
    )
    assert float(h2.mean()) == pytest.approx(2.0), "the swap should surface as extero's value"


def test_a_wrong_width_is_refused_rather_than_silently_sliced() -> None:
    """One extra column would shift every block by one and still produce valid shapes."""
    b = 2
    good = _pack(torch.zeros(b, N, D), torch.zeros(b, SENSORS * SENSOR_DIM),
                 torch.zeros(b, CMD_TOK, CMD_DIM), torch.zeros(b, dtype=torch.bool))
    with pytest.raises(ValueError, match="conditioning width"):
        TokenPlannerEnv.split_conditioning(_env(), torch.cat([good, good[:, :1]], dim=-1))


def test_an_uncommanded_checkpoint_yields_no_command_tokens() -> None:
    """``cmd_dim == 0`` must give ``None``, not a zero-width tensor the model would embed."""
    b = 2
    hist = torch.randn(b, N, D)
    extero = torch.randn(b, SENSORS * SENSOR_DIM)
    null = torch.zeros(b, dtype=torch.bool)
    cond = _pack(hist, extero, None, null)
    _, _, c2, _ = TokenPlannerEnv.split_conditioning(_env(cmd_dim=0), cond)
    assert c2 is None


def test_the_null_flag_is_the_last_column() -> None:
    """It is the one input that cannot be reconstructed from the rows, so it must survive.

    Before the first plan, three of a history row's columns can come from nowhere, and the
    model is told to use its learned ``null_history`` embedding instead of reading them. A
    consumer that loses this flag feeds the previous episode's rows -- from a different tile
    -- as though they were real history.
    """
    b = 4
    cond = _pack(torch.zeros(b, N, D), torch.zeros(b, SENSORS * SENSOR_DIM),
                 torch.zeros(b, CMD_TOK, CMD_DIM),
                 torch.tensor([True, True, False, False]))
    _, _, _, n2 = TokenPlannerEnv.split_conditioning(_env(), cond)
    assert n2.tolist() == [True, True, False, False]
