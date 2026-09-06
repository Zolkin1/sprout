# Copyright 2026 Zachary Olkin. All rights reserved.

"""Config for the token-ADJUSTER RL env, kept apart from the class for the usual reason.

Hydra composes an ``env_cfg_entry_point`` module BEFORE ``AppLauncher`` runs, so a cfg module
that reaches a concrete ``ManagerBasedRLEnv`` binds pip's ``usd-core`` instead of Kit's and
segfaults Kit a second into startup. See :mod:`.adjuster_env_cfg`, which says it at length.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .adjuster_env_cfg import AdjusterEnvCfg


@configclass
class TokenAdjusterEnvCfg(AdjusterEnvCfg):
    """The adjuster env's fields, with a DISCRETE action drawn from a residual codebook.

    Attributes:
        residual_tokenizer_path: A ``train_tokenizer.py --residual`` checkpoint. The env owns
            it because the ACTION is only meaningful through it -- an id names a point in the
            tokenizer's own normalized residual space, so the env cannot turn one into metres
            without it, and a mismatched one would still decode, to a correction rescaled per
            channel by a factor nothing downstream would notice.
        residual_alpha: Scalar gain on the decoded correction before it is composed onto the
            plan. ``0.0`` makes the agent EXACTLY the identity (bit-identically, by
            ``compose_window_residual``'s zero-residual property), which is how RL starts
            from a supervised arm that has not earned the right to steer -- the measured
            state of stage C. Ramp it up once the critic has something to say.
        roll_temperature: Softmax temperature the ROLLOUT samples at. The stored per-token
            log-probabilities are under this tempered distribution, so the PPO ratio and the
            reference KL must both be told about it.
        zero_residual: Inherited and meaningful HERE, unlike on the token planner: it zeroes
            the correction after decoding, so the plan still travels the identical projection
            and clamp path. It is the bare-generator control, bit-identical by construction --
            and is exactly what ``residual_alpha = 0`` also produces, which is why the two are
            checked against each other rather than both being trusted.
    """

    residual_tokenizer_path: str = ""
    residual_alpha: float = 1.0
    roll_temperature: float = 1.0
