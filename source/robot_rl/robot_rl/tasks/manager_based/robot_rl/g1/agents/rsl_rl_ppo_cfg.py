# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL runner configs for the G1 offline-clip teacher.

Two runners: a plain MLP PPO and a sagittal-symmetry-augmented variant.  The clips are
full episodic motions with no gait-phase observation, so the *episodic* augmentation is
the correct one -- there is no phase clock to negate under reflection.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .symmetry_functions import symmetric_data_augmentation_episodic

# Shared PPO hyperparameters. Declared once so the symmetric variant differs from the
# baseline in exactly one field (``symmetry_cfg``) rather than by a copied literal.
_PPO_KWARGS = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.008,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default MLP PPO runner -- what the teacher is trained with."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1"
    empirical_normalization = False
    logger = "wandb"
    wandb_project = "robot_rl"
    actor = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        obs_normalization=False,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(**_PPO_KWARGS)
    resume = False
    resume_path = None
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }


@configclass
class SymmetricEpisodicPPORunnerCfg(PPORunnerCfg):
    """As :class:`PPORunnerCfg`, plus sagittal-reflection data augmentation."""

    algorithm = RslRlPpoAlgorithmCfg(
        **_PPO_KWARGS,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=symmetric_data_augmentation_episodic,
        ),
    )
