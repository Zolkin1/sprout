# Copyright 2026 Zachary Olkin. All rights reserved.

"""Resuming a trained teacher into the fine-tune task.

The critic keeps the teacher's width under the FM-graded design, so there is no checkpoint
surgery to test any more -- what remains is the warm-up, which fails quietly if it is wrong: a
warm-up that does not actually freeze the actor looks exactly like one that does, except the
policy has moved.
"""

from __future__ import annotations

import pytest


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
