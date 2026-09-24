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

    critic_warmup_iterations: int = 0
    """Fit the value function alone for this many iterations before the actor may move.

    ``0`` is stock PPO.  Non-zero is for RESUMING into a changed observation, where the value
    target moves before the policy has done anything wrong; see
    :attr:`robot_rl.rl.aux_ppo.AuxPPO._in_critic_warmup`.  Counted from the start of THIS run,
    not from the resumed iteration number."""

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
    # 20000, which is what the tracker that every result on disk was measured with actually ran
    # (`2026-08-20_16-36-10_mode11_decor_plan_v1`). The repo said 10000 and that run passed
    # `--max_iterations 20000` on the command line, so the pairing lived only in a shell
    # history -- and a naive re-launch trained for half as long with nothing saying so.
    max_iterations = 20000
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
class FinetunePPORunnerCfg(PPORunnerCfg):
    """PPO for resuming a trained teacher into the generator-served observation.

    Three departures from :class:`PPORunnerCfg`, all of them about the RESUME rather than
    about the task being harder:

    * ``critic_warmup_iterations`` -- the value target moves the moment the observation does
      (different states, and shorter episodes so different returns), and until the critic has
      caught up the advantages are noise. Fit it first.
    * ``desired_kl`` cut from the pre-training 0.01 -- a converged policy released into a new
      observation can take a very large step before anything registers as wrong, and the
      adaptive controller only reacts AFTER the KL is measured. A tighter target makes it
      react sooner.
    * ``entropy_coef`` cut from 0.008 -- the bonus is there to keep a policy exploring while
      it is still learning a task. This one already has the task; leaving the bonus at full
      strength lets it win against a weak surrogate signal and inflate the action std, which
      is the "entropy goes up during fine-tuning" failure.

    These live HERE, on the agent config the task names, rather than as CLI overrides. A flag
    that has to be retyped is a flag that eventually is not -- the eval harness lost a whole
    grid to exactly that (see the eval README on ``stage_grid`` forwarding).
    """

    load_critic = False
    """Skip restoring ``critic_state_dict``; ``play.py`` honours this.

    The free-form tasks build a DIFFERENT critic group from any teacher the actor came from
    -- a clip-free library has no clip terms to observe, so the critic is 1961 wide here
    against 2533 in the `model_*_padded.pt` checkpoints -- while the ACTOR's groups are
    identical. Loading the critic therefore couples `play.py` to a width that has nothing to
    do with what it is about to run, and it failed outright: `size mismatch for mlp.0.weight
    ... [1024, 2533] vs [1024, 1961]`, before a single frame was drawn.

    `record_rollouts.py` already passes `load_cfg={"critic": False}` unconditionally and
    documents the same numbers, so this makes the viewer agree with the recorder rather than
    be the one entry point that cannot open these tasks. Actor weights are still loaded
    STRICTLY, so a genuine policy mismatch fails as loudly as before.

    A plain attribute rather than a field of the rsl-rl base cfg, which has no such knob --
    so it cannot be set from the hydra CLI and has to live here.
    """

    algorithm = AuxPpoAlgorithmCfg(**{
        **_PPO_KWARGS,
        # Critic-only iterations before the actor is released. Long enough for the value loss
        # to come off its spike, short enough that the frozen policy is still collecting data
        # worth fitting to.
        "critic_warmup_iterations": 30,
        # 3x tighter than pre-training's 0.01.
        "desired_kl": 0.0033,
        # 4x below pre-training's 0.008.
        "entropy_coef": 0.002,
    })


@configclass
class FmTrackerPlayRunnerCfg(PPORunnerCfg):
    """Play-only runner for the flow-matching tracker: **the critic is vestigial**.

    Nothing trains in that task -- the policy is the teacher, already trained -- so the
    value function is never evaluated, and ``play.py`` does not load its weights. Its
    observation set is pointed at ``policy`` purely so rsl-rl can still construct one; the
    tracker env has no privileged ``critic`` group to give it (see
    :class:`~..g1_tracker_env_cfg.G1TrackerObservationsCfg`).

    That is what makes this task teacher-agnostic. The actor's two groups have the same
    layout under either ``tracking_mode``, so with the critic out of the shape contract, a
    body-mode teacher and a joint-mode teacher both load here unchanged -- where before, a
    body-mode checkpoint was rejected on the critic's width alone.
    """

    load_critic = False
    """Skip restoring ``critic_state_dict`` (``play.py`` honours this).

    A plain attribute rather than a field of the rsl-rl base cfg, which has no such knob.
    """

    obs_groups = {
        "actor": ["policy", "reference"],
        "critic": ["policy"],
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
