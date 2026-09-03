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


#: Action width the adjuster's PPO constants below were scaled for: F * D on the shipped
#: measured-history generator (62 nodes x 44 channels). The deployed nojumpoff one is F=64, so
#: the true width can be 2816 -- a 3% shift, immaterial next to the 94x rescale itself, which
#: is why these are constants rather than something resolved from a checkpoint the cfg cannot
#: see. ``test_adjuster_ppo_cfg`` pins the RATIOS rather than the numbers.
_ADJUSTER_ACTION_DIM = 62 * 44
#: The teacher's, for the ratios below.
_TEACHER_ACTION_DIM = 29


@configclass
class AdjusterModelCfg(RslRlMLPModelCfg):
    """The residual actor over the generator's frozen encoder.

    See :class:`robot_rl.rl.adjuster_model.WindowAdjusterModel`. ``hidden_dims`` sizes a trunk
    the model REPLACES with its shared per-token head, so it is deliberately minimal --
    ``rsl_rl.modules.MLP`` rejects an empty list, and raising it only allocates a larger module
    to throw away.
    """

    class_name: str = "robot_rl.rl.adjuster_model:WindowAdjusterModel"
    plan_obs_group: str = "fm_plan"
    cond_obs_group: str = "fm_cond"
    #: Borrowed layers kept frozen; the rest become trainable copies warm-started from the
    #: generator. 6 of 8 leaves ~6.4M trainable against a 26.8M borrowed trunk.
    num_frozen_layers: int = 6


@configclass
class AdjusterPPORunnerCfg(PPORunnerCfg):
    """PPO for the residual adjuster, with three constants rescaled by the action dimension.

    **The three rescales are mandatory, and none of them is guarded anywhere in rsl-rl.**
    ``GaussianDistribution``'s ``entropy``, ``log_prob`` and ``kl_divergence`` all sum over the
    action dimension, and the adaptive-LR controller compares ``torch.mean(kl)`` directly
    against ``desired_kl``. Going 29 -> 2728 dims multiplies both statistics by ~94x:

    * ``entropy_coef`` 0.008 -> 8.5e-5. Left alone the entropy term is ~31 against a surrogate
      of O(0.01-0.1) and the action std runs away -- this repo's documented "entropy goes up
      during fine-tuning" failure, 94x more likely.
    * ``desired_kl`` 0.01 -> 0.9. Left alone the per-dimension trust region is 94x tighter, the
      controller divides the learning rate down to its HARDCODED ``max(1e-5, ...)`` floor, and
      the policy never moves -- with no symptom but a small LR.
``num_steps_per_env`` 24 -> 8 because a step is 0.24 s: 8 agent steps is 1.92 s of
    experience, four times the teacher's 0.48 s, where 24 would be nearly 6 s per iteration.

    ``critic_warmup_iterations`` 30 -> 100: the critic starts from scratch here (the teacher's
    weights are for a different observation), and the actor's zero-initialised head means
    warm-up data comes from the UNMODIFIED generator -- exactly the baseline the value function
    should be anchored on.

    ``num_mini_batches`` and ``gamma`` are deliberately left at the teacher's values.

    ``TODO(adjuster-gamma)``: 0.99 is inherited, not chosen, and it does NOT mean here what it
    means for the teacher -- gamma is a per-STEP discount and a step is 12 control steps long,
    so 0.99 per 0.24 s is a far weaker discount per unit time than 0.99 per 0.02 s
    (equivalently 0.99958 per control step). Over a <=100-decision episode the return is close
    to undiscounted: a reward ten decisions out weighs 0.90 and one a hundred out still 0.37.
    Time-matching the teacher would be ``0.99 ** 12 = 0.886``. Worth a sweep once the run
    trains at all, anchored on something measured -- the failure critic's median lead time on a
    fall is 2.52 s, about ten decisions, so credit needs to reach at least that far.

    ``TODO(adjuster-minibatches)``: kept at 4, which is a memory bet rather than a compute one
    (``num_learning_epochs x batch_size`` is invariant, so the total work does not change).
    ESTIMATED, not measured: at 1024 envs x 8 steps the batch is 8192 samples, so a 4-way split
    is 2048 per minibatch, and the backward over 8 transformer layers of 62 tokens at
    ``d_model`` 512 / ``dim_feedforward`` 2048 retains roughly ``(512 + 2048 + 512)`` floats per
    sample-token per layer -- about 1.6 GB per layer, ~12 GB over the stack. Fine on talos
    (96 GB); raise it toward 16 if this is ever run on a 12 GB card, where it would present as
    an OOM inside ``update()`` AFTER a rollout that already succeeded. Note all EIGHT layers
    count, not just the two trainable ones: gradients flow through the frozen layers to reach
    the input embedding, so their activations are retained too.

    """

    num_steps_per_env = 8
    max_iterations = 2000
    # ~35-42 s/iteration at 1024 envs, so this is ~30-45 min of crash exposure -- the same
    # trade PPORunnerCfg.save_interval documents, at a much longer iteration.
    save_interval = 50
    experiment_name = "g1_adjuster"
    actor = AdjusterModelCfg(
        hidden_dims=[1],
        activation="elu",
        # MEASURED, not chosen. 0.5 was shipped first on the reasoning that the physical
        # smallness belongs in the residual's cap table rather than in sigma -- but the caps
        # are deliberately LOOSE (base position is capped at the full 0.2 m violation
        # threshold), so sigma 0.5 in capped units is ~0.1 m RMS of re-drawn reference noise
        # per node, well above the 0.06 m half-width `REFERENCE_WINDOW_NOISE` says the tracker
        # was hardened to. A 16-env smoke triple, all with the actor frozen by critic warm-up
        # so each measures only its initialisation:
        #
        #   init_std   residual_l2   trackability   goal_reached   tracking_violation
        #   1e-4              -0.000          0.178          0.320                0.555
        #   0.15              -0.077          0.135          0.219                0.688
        #   0.5               -0.330          0.034          0.172                0.812
        #
        # Monotone in all four columns, and `residual_l2` is a clean low-noise dose, so the
        # dose-RESPONSE is solid even though the absolute rates are not: ~50-100 completed
        # episodes per arm puts a standard error of roughly 0.05-0.07 on `goal_reached`.
        #
        # 0.15 puts the initial residual at ~0.03 m RMS on base position -- the noise the
        # tracker already tolerates -- and it still costs ~24% of trackability against the bare
        # generator. That is the price of having any exploration at all, and it is a genuine
        # trade rather than a solved value: too little and PPO has no gradient signal, too much
        # and the value function is anchored on a distribution far worse than the baseline the
        # policy has to beat. TODO(adjuster-init-std): revisit once a real run trains, against
        # whether the early dip recovers.
        #
        # Do NOT push it much lower: KL ~ A * delta^2 / (2 sigma^2), so a tiny sigma shrinks
        # the permitted mean step and strangles the adaptive LR controller instead of limiting
        # the residual -- which is what the cap table is for.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15, std_type="log"),
        obs_normalization=False,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = AuxPpoAlgorithmCfg(**{
        **_PPO_KWARGS,
        "entropy_coef": _PPO_KWARGS["entropy_coef"] * _TEACHER_ACTION_DIM / _ADJUSTER_ACTION_DIM,
        "desired_kl": _PPO_KWARGS["desired_kl"] * _ADJUSTER_ACTION_DIM / _TEACHER_ACTION_DIM,
        # num_mini_batches and gamma stay at the teacher's values -- see the TODOs above.
        "critic_warmup_iterations": 100,
    })
    obs_groups = {
        # The actor sees the generator's conditioning and the generator's plan -- and nothing
        # privileged, so the same network could run on a robot.
        "actor": ["fm_cond", "fm_plan"],
        "critic": ["critic"],
    }
