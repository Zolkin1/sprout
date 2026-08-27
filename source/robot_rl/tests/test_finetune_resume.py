# Copyright 2026 Zachary Olkin. All rights reserved.

"""The two mechanics that let a trained teacher resume into the fine-tune task.

Both fail quietly if they are wrong.  A mis-padded critic loads and predicts nonsense; a
warm-up that does not actually freeze the actor looks exactly like one that does, except the
policy has moved.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

_SCRIPT = (pathlib.Path(__file__).resolve().parents[3]
           / "scripts" / "rsl_rl" / "pad_critic_for_finetune.py")


def _load_pad_module():
    """Import the padding script by path; it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("pad_critic_for_finetune", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture(scope="module")
def pad():
    """The script's ``pad_critic`` helper."""
    if not _SCRIPT.exists():
        pytest.skip(f"{_SCRIPT} not found")
    return _load_pad_module().pad_critic


def _critic(width: int) -> dict[str, torch.Tensor]:
    """A minimal critic state dict with the layer naming the script expects."""
    torch.manual_seed(0)
    return {"mlp.0.weight": torch.randn(8, width), "mlp.0.bias": torch.randn(8)}


def test_padding_preserves_the_old_block_and_zeroes_the_rest(pad):
    """Existing inputs keep their columns; the appended ones start at zero."""
    state = _critic(1961)
    out, old_width = pad(state, 2533)
    assert old_width == 1961
    assert out["mlp.0.weight"].shape == (8, 2533)
    assert torch.equal(out["mlp.0.weight"][:, :1961], state["mlp.0.weight"])
    assert bool((out["mlp.0.weight"][:, 1961:] == 0).all())
    assert torch.equal(out["mlp.0.bias"], state["mlp.0.bias"])


def test_the_appended_columns_are_inert(pad):
    """The padded layer computes the same thing whatever arrives on the new inputs.

    Random-vs-zero padding is the check that separates "the new columns contribute nothing"
    from "the outputs happen to look close": if they were contributing, the two would differ.
    """
    state = _critic(1961)
    out, _ = pad(state, 2533)
    x = torch.randn(4, 1961)
    ref = torch.nn.functional.linear(x, state["mlp.0.weight"], state["mlp.0.bias"])
    got = [
        torch.nn.functional.linear(
            torch.cat([x, tail], dim=1), out["mlp.0.weight"], out["mlp.0.bias"]
        )
        for tail in (torch.zeros(4, 572), torch.randn(4, 572) * 100.0)
    ]
    assert torch.equal(got[0], got[1])                      # inert, exactly
    assert torch.allclose(ref, got[0], rtol=1e-5, atol=1e-5)  # same function, up to reassoc


def test_padding_refuses_to_shrink_or_no_op(pad):
    """Shrinking would need to know WHICH inputs to drop; a no-op means something is off."""
    state = _critic(1961)
    with pytest.raises(ValueError, match="shrink"):
        pad(state, 1000)
    with pytest.raises(ValueError, match="already"):
        pad(state, 1961)


def test_padding_refuses_an_unrecognised_layer_name(pad):
    """Padding a tensor found by guesswork would corrupt the critic silently."""
    with pytest.raises(KeyError, match="mlp.0.weight"):
        pad({"trunk.0.weight": torch.randn(8, 10)}, 20)


def test_warmup_predicate_counts_this_run_not_the_resumed_iteration():
    """The warm-up must not be skipped just because the checkpoint resumes at 20000."""
    from robot_rl.rl.aux_ppo import AuxPPO

    obj = AuxPPO.__new__(AuxPPO)          # no PPO construction: the predicate is pure
    obj.critic_warmup_iterations = 30
    obj._updates_done = 0
    assert obj._in_critic_warmup
    obj._updates_done = 29
    assert obj._in_critic_warmup
    obj._updates_done = 30
    assert not obj._in_critic_warmup

    obj.critic_warmup_iterations = 0       # stock PPO: never warming
    obj._updates_done = 0
    assert not obj._in_critic_warmup


def test_the_finetune_runner_cuts_kl_and_entropy_against_pretraining():
    """The fine-tune's departures ride the task's agent cfg, not a CLI flag that can be lost."""
    pytest.importorskip("isaaclab")
    from robot_rl.tasks.manager_based.robot_rl.g1.agents.rsl_rl_ppo_cfg import (
        FinetunePPORunnerCfg, PPORunnerCfg,
    )
    ft, pre = FinetunePPORunnerCfg().algorithm, PPORunnerCfg().algorithm
    assert 2.0 <= pre.desired_kl / ft.desired_kl <= 4.0
    assert ft.entropy_coef < pre.entropy_coef
    assert ft.critic_warmup_iterations > 0
    assert pre.critic_warmup_iterations == 0
