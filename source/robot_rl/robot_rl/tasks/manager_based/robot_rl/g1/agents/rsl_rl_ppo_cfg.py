# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL runner configs for the G1 offline-clip teacher.

Two runners: the default PPO and a sagittal-symmetry-augmented variant.  The clips are
full episodic motions with no gait-phase observation, so the *episodic* augmentation is
the correct one -- there is no phase clock to negate under reflection.

The actor is a two-branch model: proprioception (history-stacked) goes straight to the
trunk while the reference window is compressed by its own encoder.  Both the model and
the algorithm are in-repo classes, reached through rsl-rl's ``class_name`` ->
:func:`rsl_rl.utils.resolve_callable` indirection, which accepts ``"module:Class"``.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .symmetry_functions import symmetric_data_augmentation_episodic


@configclass
class RefEncoderModelCfg(RslRlMLPModelCfg):
    """Actor model whose ``reference`` observation group goes through an MLP encoder.

    See :class:`robot_rl.rl.ref_encoder_model.RefEncoderMLPModel`.  ``hidden_dims`` (from
    the base class) sizes the trunk that consumes the direct groups plus the encoder
    latent.
    """

    class_name: str = "robot_rl.rl.ref_encoder_model:RefEncoderMLPModel"

    encoder_obs_groups: list[str] = ["reference"]
    """Observation groups routed through the encoder instead of straight to the trunk."""

    encoder_dims: list[int] = [1024, 512, 512]
    """Encoder widths; the LAST entry is the latent width, the rest are hidden layers."""

    reconstruction: bool = False
    """Build the decoder for the auxiliary reconstruction loss.  Off: this is the ablation."""

    reconstruction_dims: list[int] | None = None
    """Decoder hidden widths.  ``None`` mirrors the encoder (``reversed(encoder_dims[:-1])``)."""


@configclass
class AuxPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus the optional reference-reconstruction term.

    See :class:`robot_rl.rl.aux_ppo.AuxPPO`.  With the coefficient at 0 (the default) the
    algorithm is arithmetically identical to stock PPO.
    """

    class_name: str = "robot_rl.rl.aux_ppo:AuxPPO"

    reference_reconstruction_coeff: float = 0.0
    """Weight of the reconstruction loss.  Non-zero requires ``actor.reconstruction=True``."""

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
    """Default PPO runner -- what the teacher is trained with.

    The actor takes two observation groups: ``policy`` (proprioception + exteroception,
    straight to the trunk) and ``reference`` (the look-ahead window, through the
    encoder).  The critic is a plain MLP over its single privileged group.

    To ablate the auxiliary reconstruction loss, no code change is needed -- ``train.py``
    passes unrecognised arguments to Hydra::

        python scripts/rsl_rl/train.py --task=G1-teacher-clf-terrain ... \\
            agent.actor.reconstruction=true \\
            agent.algorithm.reference_reconstruction_coeff=0.1

    Both must move together; ``AuxPPO`` raises if the coefficient is non-zero without a
    decoder.
    """

    num_steps_per_env = 24
    max_iterations = 10000
    # A checkpoint is ~98 MB, so this is the dominant consumer of log storage: at the old
    # 200 a full run wrote ~10 GB and the archive had reached 3500 checkpoints / 92 GB.
    # 400 halves that while keeping the exposure on a crash under an hour (~55 min at the
    # ~8.3 s/iter measured at 8192 envs), which matters more than the disk -- do NOT push
    # this much higher just to save space. Thin FINISHED runs instead; keeping the last few
    # plus a milestone every 2000 recovers ~85% without touching crash resilience.
    save_interval = 400
    experiment_name = "g1"
    empirical_normalization = False
    logger = "wandb"
    wandb_project = "robot_rl"
    actor = RefEncoderModelCfg(
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
    algorithm = AuxPpoAlgorithmCfg(**_PPO_KWARGS)
    resume = False
    resume_path = None
    obs_groups = {
        "actor": ["policy", "reference"],
        "critic": ["critic"],
    }


@configclass
class SymmetricEpisodicPPORunnerCfg(PPORunnerCfg):
    """As :class:`PPORunnerCfg`, plus sagittal-reflection data augmentation.

    Still broken -- the reflector has no handler for the reference observation terms.
    See the note in ``g1/__init__.py``.
    """

    algorithm = AuxPpoAlgorithmCfg(
        **_PPO_KWARGS,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=symmetric_data_augmentation_episodic,
        ),
    )
