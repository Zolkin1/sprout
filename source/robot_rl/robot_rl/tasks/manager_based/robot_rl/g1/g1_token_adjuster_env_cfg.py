# Copyright 2026 Zachary Olkin. All rights reserved.

"""The G1 token-ADJUSTER RL task: the generator plans, a discrete policy corrects it.

Inherits the adjuster task wholesale -- the same frozen tracker, the same clip library, the
same goal regions, the same plan cadence -- and changes three things: what the action IS, how
the plan's fall is detected, and which history arm the generator serves. Each of those is
argued at its site below.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.envs import TokenAdjusterEnvCfg

from .g1_adjuster_env_cfg import G1AdjusterEnvCfg


#: Weight that keeps the residual-magnitude terms COMPUTED without pricing them.
#:
#: Not 0: ``RewardManager.compute`` skips zero-weight terms, so the diagnostic would be a
#: structural 0.0. At 1e-8 against a dense reward of order 1 and a -10 fall, the contribution
#: is 9 orders below the smallest thing that matters, while the term still runs and logs.
RESIDUAL_DIAGNOSTIC_WEIGHT: float = -1e-8


@configclass
class G1TokenAdjusterRewardCfg:
    """The adjuster's reward set, with the residual penalties kept but ZEROED.

    A full interval of perfect tracking is worth 1.0, a fall costs ten of them, and running
    the clock out costs two. Reaching the goal is a TRUNCATION rather than a rewarded event,
    so its payoff is the accumulated trackability plus the ``-2`` it avoids -- which is what
    makes standing still lose without a separate progress term.

    The two residual penalties are RETAINED at weight zero rather than removed, and they are
    one of the two places the correction's MAGNITUDE can be bounded.

    .. admonition:: The bound is on magnitude, NOT on the policy's shape

        The obvious regularizer -- a KL toward the supervised adjuster -- is wrong here, and
        the reason is specific to this stack rather than general. In LLM RL the reference is
        PEAKED, so a KL to it is a genuine trust region against reward hacking. This reference
        is not: stage C measured its predictive entropy at ~4.3 nats/token, because the
        conditional over residual tokens is broad and its mode is "no correction". A KL toward
        a broad reference therefore penalizes SHARPENING -- which is the one thing RL has to
        do here, namely commit to the correction that works.

        So the supervised model is the INITIALISATION and the sampling distribution, and
        nothing else. What is bounded instead is how far the served correction moves the plan:
        either the zero-code KL in the loss (``-log pi(ZERO_CODE)``, which anchors magnitude
        without caring how peaked the rest of the distribution is) or these terms, which price
        the decoded correction directly.

    .. admonition:: The weight is -1e-8, not 0, and that is not a rounding choice

        IsaacLab's ``RewardManager.compute`` SKIPS a zero-weight term entirely -- it never
        calls the function -- while ``reset`` still emits ``Episode_Reward/residual_l2`` from
        an all-zero accumulator. So a weight of exactly 0 does not mean "measured but not
        priced"; it means the one quantity saying how hard the agent is pushing reads 0.0
        forever, which is indistinguishable from the "does nothing" failure this whole stage
        exists to escape. :data:`RESIDUAL_DIAGNOSTIC_WEIGHT` is small enough to be inert
        against a dense reward of order 1 and a fall at -10, and non-zero so the term runs.

    Set it to a real magnitude to make these the bound instead of the loss-side anchor. On
    the continuous adjuster the same terms at ``-2.0 / -1.0`` were measured cancelling the
    entire dense reward, which is what a logged term catches and an absent one does not.
    """

    trackability: RewTerm = RewTerm(func=mdp.served_track_reward, weight=1.0)
    residual_l2: RewTerm = RewTerm(
        func=mdp.plan_residual_l2, weight=RESIDUAL_DIAGNOSTIC_WEIGHT,
        params={"command_name": "traj_ref"},
    )
    residual_rate_l2: RewTerm = RewTerm(
        func=mdp.plan_residual_rate_l2, weight=RESIDUAL_DIAGNOSTIC_WEIGHT,
        params={"command_name": "traj_ref"},
    )
    failed: RewTerm = RewTerm(
        func=mdp.episode_ended_by, weight=-10.0,
        params={"term_keys": ["tracking_violation"]},
    )
    ran_out_of_time: RewTerm = RewTerm(
        func=mdp.episode_ended_by, weight=-2.0,
        params={"term_keys": ["clip_end", "time_out"]},
    )


@configclass
class G1TokenAdjusterEnvCfg(G1AdjusterEnvCfg, TokenAdjusterEnvCfg):
    """The token-adjuster RL task."""

    def __post_init__(self) -> None:
        """Run the adjuster's setup, then swap in the discrete-correction MDP."""
        super().__post_init__()
        self.rewards = G1TokenAdjusterRewardCfg()
        # THE ABSOLUTE FALL DETECTOR, ALONGSIDE the reference-relative one, for the reason
        # the token-planner task states: `tracking_violation` is reference-RELATIVE, so it
        # cannot see a robot that tracks a bad plan faithfully all the way to the floor. That
        # gap is if anything wider here -- the agent is actively editing the reference the
        # violation is measured against, so making the reference follow the fall is a
        # gradient the relative term alone would reward.
        #
        # 55 deg rather than the function's 70: 70 is past the point of no return, so it
        # fires long after the episode is decided and leaves a stretch of doomed steps still
        # collecting dense trackability.
        self.terminations.base_fallen = DoneTerm(
            func=mdp.base_fallen,
            params={"min_height": 0.5, "roll_limit_deg": 55.0, "pitch_limit_deg": 55.0},
            time_out=False,
        )
        # It must COST the same as the relative fall, or it is an escape hatch: a termination
        # with no reward term ends the episode with no penalty, which beats the -2 for
        # running the clock out and would be the cheapest way off a hard tile.
        self.rewards.failed.params["term_keys"] = ["tracking_violation", "base_fallen"]
        # The GENERATOR is the flow-matching one here -- it is the thing being corrected --
        # so the flow stash stays available and `fm_plan` with it. The policy reads the plan
        # through `env.plan_window()` rather than through the observation group, but the group
        # costs nothing and the critic is welcome to it.
        self.commands.traj_ref.generator_kind = "flow_matching"
        # `measured`, matching what the residual dataset was BUILT under. The residual is
        # defined as the error of a generator conditioned a particular way; serving a
        # different history arm at RL time would make the plan being corrected a different
        # object from the one the codebook and the reference policy were fitted against, at
        # exactly the same width. `assert_history_source_matches` refuses a checkpoint that
        # disagrees, which is the backstop rather than the plan.
        self.commands.traj_ref.history_from = "measured"
        # The policy conditions on the SAME rows the runtime built for this re-plan. Set on
        # the CFG, not via `enable_cond_stash()`, which runs after the observation manager
        # has probed shapes.
        self.commands.traj_ref.stash_conditioning = True
        # x0 = 0, matching BOTH ends of this arm. The residual dataset was built by sampling
        # the generator at zero noise, so the plans the adjuster was fitted to correct are
        # the deterministic ones; and the deployed C++ controller runs zero-noise too. Left
        # stochastic, RL would train the policy to correct plans drawn from a distribution
        # neither its initialisation nor its deployment ever sees -- at exactly the same
        # shape, since a plan is a plan.
        self.commands.traj_ref.fm_zero_noise = True
