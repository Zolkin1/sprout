# Copyright 2026 Zachary Olkin. All rights reserved.

"""Config for the token-planner RL env, kept apart from the class for the usual reason.

Hydra composes an ``env_cfg_entry_point`` module BEFORE ``AppLauncher`` runs, so a cfg module
that reaches a concrete ``ManagerBasedRLEnv`` binds pip's ``usd-core`` instead of Kit's and
segfaults Kit a second into startup. See :mod:`.adjuster_env_cfg`, which says it at length --
this file exists for the same reason and nothing more.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .adjuster_env_cfg import AdjusterEnvCfg


@configclass
class TokenPlannerEnvCfg(AdjusterEnvCfg):
    """The adjuster env's fields, minus the residual, plus how the policy samples.

    Inherits ``tracker_run_dir`` / ``tracking_spec_path`` / ``command_name`` / the steering
    pair verbatim: the frozen tracker, the fitted tracking spec and the plan cadence are the
    same requirements whether the agent corrects a plan or emits one.

    Attributes:
        roll_temperature: Softmax temperature the ROLLOUT samples at. The stored per-token
            log-probabilities are under this tempered distribution, so the PPO ratio and the
            reference KL must both be told about it; it is recorded in every checkpoint for
            that reason.
        zero_residual: Inherited and MEANINGLESS here -- there is no residual to zero. The
            env refuses it rather than ignoring it, because "the control arm" is exactly what
            someone would reach for it to mean, and here the control arm is instead the
            reference planner sampling normally.
    """

    roll_temperature: float = 1.0
