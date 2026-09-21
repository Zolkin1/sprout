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


#: Action width the adjuster's PPO constants below were scaled for: F * D on the shipped
#: measured-history generator (62 nodes x 44 channels). The deployed nojumpoff one is F=64, so
#: the true width can be 2816 -- a 3% shift, immaterial next to the 94x rescale itself, which
#: is why these are constants rather than something resolved from a checkpoint the cfg cannot
#: see. ``test_adjuster_ppo_cfg`` pins the RATIOS rather than the numbers.
_ADJUSTER_ACTION_DIM = 62 * 44
#: The teacher's, for the ratios below.
_TEACHER_ACTION_DIM = 29


@configclass
class ChunkPpoAlgorithmCfg(AuxPpoAlgorithmCfg):
    """:class:`AuxPpoAlgorithmCfg` with the importance ratio factored along the chunk.

    See :class:`robot_rl.rl.chunk_ppo.ChunkPPO`. ``chunk_steps`` / ``chunk_dim`` default to
    the actor's own ``(num_nodes, node_dim)``, and a mismatch against the policy's output
    width is refused rather than reshaped.
    """

    class_name: str = "robot_rl.rl.chunk_ppo:ChunkPPO"

    chunk_steps: int | None = None
    """``T``, look-ahead nodes per chunk. ``None`` reads it off the actor."""

    chunk_dim: int | None = None
    """``D``, channels per node. ``None`` reads it off the actor."""

    ratio_mode: str = "per_step"
    """``"per_step"`` (T ratios of D dims) or ``"gspo"`` (one geometric-mean ratio).

    ``gspo`` needs a much SMALLER ``clip_param``, not a larger one -- its ``1/N``
    normalization divides the log-ratio spread by ``N`` rather than by ``sqrt(N)``.
    """


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
    #: ``K``, the action's knot count. MUST equal the env cfg's ``num_knots`` -- the two are
    #: separate config objects, and the cross-check is the action-width assertion in
    #: ``AdjusterEnv._residual_from_action``, which names the mismatch rather than letting a
    #: 13-knot actor drive a 21-knot lift.
    num_knots: int = 13
    #: Interpolation for the knot lift; must likewise match the env cfg's ``knot_kind``.
    knot_kind: str = "catmull"
    #: Per channel-group initial exploration width, which turns the std into 18 factored
    #: numbers (``log sigma[g,j] = log s[g] + log h[j]``) instead of one per action dim.
    #:
    #: DIFFERENT PER GROUP on purpose, and the sizes come from the offline correction fit:
    #: the correction the generator needs is 0.19-0.52 of each channel's cap depending on
    #: the group, so a single width was over-exploring some and under-exploring others.
    #: Base position and orientation get the smallest, because those are the channels the
    #: tracking-violation bound is measured on (0.2 m lateral / height, 30 deg tilt) and
    #: noise there converts directly into failures -- the frozen-actor dose response showed
    #: goal-reached falling 0.32 -> 0.22 -> 0.17 as a single shared width went 1e-4 -> 0.15
    #: -> 0.5.
    #:
    #: These are a STARTING POINT to be replaced by the dose response, not a fitted result.
    #: Set to ``None`` to fall back to the stock per-dimension Gaussian.
    group_init_std: dict | None = None


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

    # 24, the rsl_rl default, up from 8. ~1 episode per env per iteration (~26 decisions),
    # so ~24k decisions per iteration at 1024 envs instead of 8k. Together with the knot
    # action this is the samples-per-dimension fix: at full width and 8 steps it was ~3
    # samples per action dimension per iteration, which is why neither the mean nor the
    # per-dimension stds could move; at 572 dims and 24 steps it is ~42, a 14x improvement.
    # Rollout storage grows to ~2 GB, which fits.
    num_steps_per_env = 24
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
        num_knots=13,
        knot_kind="catmull",
        # `distribution_cfg`'s init_std above is what the stock per-dimension Gaussian would
        # use and is kept only so the fallback stays valid; THIS is what the model builds.
        group_init_std={
            "base_pos": 0.05,
            "base_ori": 0.05,
            "base_lin_vel": 0.10,
            "base_ang_vel": 0.10,
            "joint_pos": 0.15,
        },
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        # ON, unlike the actor's. This group is un-normalized nowhere else in the repo
        # only because the teacher's happened to be O(1) by construction; here it mixes
        # metres, radians, normalized window units and binary contacts, and the critic
        # trains on ~1/12 the samples a per-control-step one would. Running per-channel
        # mean/std costs one buffer and removes a whole class of conditioning defect --
        # the goal_geometry scaling above was one instance found by hand.
        obs_normalization=True,
    )
    algorithm = ChunkPpoAlgorithmCfg(**{
        **_PPO_KWARGS,
        # ALL THREE 94x rescales are reverted, because the statistics they bound are now
        # per-STEP rather than per-chunk. `ChunkPPO` factors the importance ratio along the
        # chunk's time axis -- one 44-dim Gaussian ratio per look-ahead node, each clipped
        # separately under one shared chunk advantage -- which is the regime these defaults
        # were tuned for. See robot_rl/rl/chunk_ppo.py for the derivation and for why
        # neither a wider clip band nor a smaller desired_kl fixes the joint ratio.
        "clip_param": 0.2,
        "desired_kl": 0.01,
        # ZERO. Measured on the first steered run, and the number that settles it is the
        # LOSS ratio rather than the gradient one I originally reasoned from:
        #
        #     Mean surrogate loss   -0.0345
        #     Mean entropy loss     -1059      (= 2728 dims x (log sigma + 1.419))
        #     entropy contribution  -coef * entropy = -1.6e-4 * -1059 = +0.169
        #
        # The entropy term entered the loss about 5x larger than the objective PPO is meant
        # to maximise, so the optimiser spent the run inflating sigma: 0.150 -> 0.164 over
        # 110 iterations, with the residual penalty growing alongside it and the mean return
        # falling monotonically from +0.70 to -1.52 from the moment the actor was released.
        # Success never moved off the bare generator's rate.
        #
        # I set 1.6e-4 by matching the teacher's ratio of entropy to surrogate GRADIENT on
        # log_std and never compared the two as they enter the loss; the 5:1 would have been
        # visible immediately. Zero is what the original design called for.
        #
        # Exploration does not come from this term -- sigma is a learned parameter, and the
        # surrogate adapts it in BOTH directions (up when high-advantage actions landed far
        # from the mean, down when they landed near it). What the bonus buys is insurance
        # against premature COLLAPSE, which is a different failure, slower, and visible in
        # `Policy/mean_noise_std`. TODO(adjuster-entropy-coef): if sigma collapses, ~2e-5
        # puts entropy at roughly 5-10% of the surrogate's gradient instead of 56%.
        "entropy_coef": 0.0,
        # 0.96, finally applied. A step here is 0.24 s, so 0.99 is a ~24 s horizon against a
        # ~6.2 s episode (~26 decisions) -- the critic was effectively predicting a
        # near-constant return. 0.96 is ~6 s, matching the episode. This was the original
        # design intent and had been inherited from the teacher, unexamined, since v1;
        # `TODO(adjuster-gamma)` above is now closed.
        #
        # It composes with the goal becoming a TERMINAL REWARD rather than a bootstrapped
        # truncation: with nothing bootstrapped, gamma no longer sets the arrival value's
        # scale, so the two changes had to land together.
        "gamma": 0.96,
        # 12, not 4, and this is a MEMORY requirement rather than a tuning choice. The
        # binding cost in `update()` is the recomputed conditioning prefix, which scales with
        # MINIBATCH SIZE: 8 layers x B x 128 prefix tokens x 512 floats. Tripling
        # `num_steps_per_env` to 24 tripled B at 4 minibatches and the run OOM'd at 91.7 GB
        # of a 93 GB card, having allocated ~26 GB for that cache alone. 12 restores the
        # 2048-sample minibatch the full-width runs proved out (~8.6 GB), while keeping the
        # 3x larger rollout the samples-per-dimension fix needs.
        #
        # It is NOT a FLOP change and not a different optimisation: `num_learning_epochs *
        # (num_steps_per_env * num_envs / num_mini_batches) * num_mini_batches` is invariant,
        # so the same samples are seen the same number of times in smaller groups.
        "num_mini_batches": 12,
        # 30, not the 100 this started at. The critic is a third of its old width, with
        # `window_residual` (the majority of it) gone and input normalization on, so it has
        # far less to fit and better-conditioned inputs to fit it from. What warm-up buys is
        # unchanged and still wanted -- a garbage critic turns advantages into noise and the
        # entropy term, which has no such problem, wins -- so this is shortened rather than
        # dropped. `explained_variance` (logged every iteration) is the number that says
        # whether 30 was enough: if it is still climbing when the actor is released, raise it.
        "critic_warmup_iterations": 30,
    })
    obs_groups = {
        # The actor sees the generator's conditioning and the generator's plan -- and nothing
        # privileged, so the same network could run on a robot.
        "actor": ["fm_cond", "fm_plan"],
        "critic": ["critic"],
    }


@configclass
class AdjusterFreeformPPORunnerCfg(AdjusterPPORunnerCfg):
    """PPO for the free-form arm: the FULL per-node action, and the off-policy arm's discount.

    Four changes from the clip-terrain arm. Three are about matching the thing being compared
    against rather than about tuning; the fourth is a memory requirement that follows from the
    env count:

    * **`gamma` 0.96 -> 0.95.** 0.96 was fitted to the clip task's ~6.2 s episode. 0.95 is the
      offline reducer's own per-chunk discount (`advantage.bootstrapped_window_return`), and a
      chunk is a chunk on both sides. It must equal `AdjusterEnvCfg.progress_gamma`, because
      that is what scales the arrival bonus so the two returns are the same object -- pinned by
      `test_adjuster_freeform_cfg.py`.
    * **the action is FULL WIDTH again**, 62 x 44 = 2728. `knot_basis(F, F)` is exactly the
      identity (verified: `max|B - I| = 0.0`), so this is one config field rather than a second
      code path, and `ChunkPPO`'s `T` becomes 62. This is the arm the paper describes as "PPO
      on the full continuous action space", so the parameterisation is the claim.

      The risk is measured and is the one to watch: at 1024 envs x 8 steps the full-width run
      held ~96-97% exploration variance over 840 iterations, because one scalar advantage
      against 2728 dims is ~3 samples per dimension per iteration. At 2048 envs x 24 steps it
      is ~18, six times better sampled, and `group_init_std` keeps the grouped std (5 + 62 = 67
      learned numbers, not 2728). `Diagnostics/residual_mean_energy_frac` is the go/no-go;
      `num_knots = 13` on both this and the env cfg is the documented fallback.
    The env count is NOT set here -- `RslRlOnPolicyRunnerCfg` has no such field, so assigning
    one would be inert and would read as a decision. It lives on the env cfg's `scene.num_envs`
    (2048, matching the off-policy collect) and `--num_envs` still overrides it.

    `entropy_coef = 0.0`, `desired_kl = 0.01` and `clip_param = 0.2` are inherited unchanged:
    `ChunkPPO` factors the ratio per node, so they are per-STEP statistics and the stock values
    are the regime they were tuned for.
    """

    experiment_name = "g1_adjuster_freeform"
    actor = AdjusterModelCfg(
        hidden_dims=[1],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15, std_type="log"),
        obs_normalization=False,
        # FULL WIDTH: one knot per node, so the lift is the identity and the action is the
        # whole 62 x 44 window. Must match `AdjusterEnvCfg.num_knots`; the width assertion in
        # `AdjusterEnv._residual_from_action` catches a mismatch rather than applying it.
        num_knots=62,
        knot_kind="catmull",
        group_init_std={
            "base_pos": 0.05,
            "base_ori": 0.05,
            "base_lin_vel": 0.10,
            "base_ang_vel": 0.10,
            "joint_pos": 0.15,
        },
    )
    algorithm = ChunkPpoAlgorithmCfg(**{
        **_PPO_KWARGS,
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.0,
        # The offline reducer's per-chunk discount. See the class docstring.
        "gamma": 0.95,
        # 24, NOT the 12 this class used to inherit, and it is arithmetic rather than a tune.
        # The binding cost in `update()` is the recomputed conditioning prefix, which scales
        # with MINIBATCH SIZE -- and 12 was sized for the clip arm's 1024 envs, where it gives
        # 1024 * 24 / 12 = 2048 samples per minibatch, the figure that class's comment calls
        # "proved out (~8.6 GB)". At this arm's 2048 envs the same 12 gives 4096, double that.
        #
        # MEASURED: it OOM'd inside `update()` at iteration 1, after a rollout that had already
        # succeeded -- 90.94 GB in use of a 93.09 GB card, asking for 4.03 more with 2.09 free.
        # 2048 * 24 / 24 = 2048 restores the proved-out minibatch exactly.
        #
        # It is NOT a FLOP change and not a different optimisation: `num_learning_epochs *
        # (num_steps_per_env * num_envs / num_mini_batches) * num_mini_batches` is invariant, so
        # the same samples are seen the same number of times in smaller groups.
        "num_mini_batches": 24,
        "critic_warmup_iterations": 30,
    })


@configclass
class AdjusterFreeformPlayPPORunnerCfg(AdjusterFreeformPPORunnerCfg):
    """Watch the free-form adjuster. Same actor, no critic restored.

    ``play.py`` never evaluates the value function, so skipping the critic load lets a
    checkpoint drive the robot even if the critic group later moves. The actor is still loaded
    STRICTLY, so a genuine policy mismatch fails loudly rather than driving a fresh network.
    """

    load_critic = False


@configclass
class AdjusterPlayPPORunnerCfg(AdjusterPPORunnerCfg):
    """Watch an adjuster checkpoint. Same actor, no critic restored.

    ``play.py`` never evaluates the value function, so skipping the critic load is what lets
    a checkpoint from BEFORE a critic-group change still drive the robot -- and this task's
    critic group has moved (4519 -> 1631 channels) while its actor has not. Without this the
    load fails on the critic's width alone and says nothing about the policy.

    The actor is still loaded STRICTLY, so a genuine policy mismatch fails loudly.
    """

    load_critic = False
    """Skip restoring ``critic_state_dict`` (``play.py`` honours this).

    A plain attribute rather than a field of the rsl-rl base cfg, which has no such knob.
    """
