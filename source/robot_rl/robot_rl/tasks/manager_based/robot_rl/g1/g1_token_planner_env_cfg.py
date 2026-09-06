# Copyright 2026 Zachary Olkin. All rights reserved.

"""The tokenized-planner RL task: PPO over whole plans, tracker frozen inside the step.

A thin specialisation of the adjuster task, and thin on purpose. The scene, cameras, depth
terms, terminations, goal geometry, steering, episode budget and -- critically -- the REWARD
SET are the adjuster's, unchanged. What differs is who acts: the adjuster corrects the plan
the generator drew, while here the tokenized planner IS the policy and its ``N`` token ids
are the action.

Two deliberate departures, both forced:

* **The two residual reward terms are gone.** They price a correction against the plan the
  generator drew, and there is no such reference when the agent emits the plan outright.
  :class:`~robot_rl.tasks.manager_based.robot_rl.envs.token_planner_env.TokenPlannerEnv`
  refuses them rather than letting them price zeros, which would read as a well-behaved
  agent rather than as a meaningless term.
* **``generator_kind='token_planner'``**, because the action is decoded through the
  checkpoint's own frozen tokenizer and a flow checkpoint carries none.

Everything the adjuster cfg says about the three paths with no defaults (``--clip``,
``--fm_checkpoint``, ``--tracker_run``) applies here verbatim.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.envs import TokenPlannerEnvCfg

from .g1_adjuster_env_cfg import G1AdjusterEnvCfg


@configclass
class G1TokenPlannerRewardCfg:
    """The adjuster's reward set minus the residual penalties.

    The three that remain are exactly what the RL spec asks for and their relative sizes carry
    the same meaning they do on the adjuster: a full interval of perfect tracking is worth
    1.0, a fall costs ten of them, and running out of time costs two. Reaching the goal is a
    TRUNCATION rather than a rewarded event, so its payoff is the accumulated trackability
    plus the ``-2`` it avoids -- which is what makes standing still lose without needing a
    separate progress term.

    These weights are a STARTING POINT. The first run's job is to calibrate them from the
    ``Episode_Reward/*`` logs, not to confirm them.
    """

    trackability: RewTerm = RewTerm(func=mdp.served_track_reward, weight=1.0)
    failed: RewTerm = RewTerm(
        func=mdp.episode_ended_by, weight=-10.0,
        params={"term_keys": ["tracking_violation"]},
    )
    ran_out_of_time: RewTerm = RewTerm(
        func=mdp.episode_ended_by, weight=-2.0,
        params={"term_keys": ["clip_end", "time_out"]},
    )


@configclass
class G1TokenPlannerEnvCfg(G1AdjusterEnvCfg, TokenPlannerEnvCfg):
    """The token-planner RL task."""

    def __post_init__(self) -> None:
        """Run the adjuster's setup, then swap in the token-planner MDP."""
        super().__post_init__()
        self.rewards = G1TokenPlannerRewardCfg()
        # THE ABSOLUTE FALL DETECTOR, BACK, ALONGSIDE the reference-relative one. The
        # adjuster task dropped it because `tracking_violation` fires ~0.20 s earlier on the
        # failures it shares -- but it is reference-RELATIVE, so it cannot see a robot that
        # tracks a bad plan faithfully all the way to the floor. This one is
        # trajectory-independent: world height, and roll/pitch from world upright.
        #
        # 55 deg rather than the 70 the function defaults to: 70 is past the point of no
        # return for a humanoid, so it fires long after the episode is decided and gives the
        # policy a stretch of already-doomed steps still collecting dense trackability.
        #
        # `min_height` stays at 0.5. It is ABSOLUTE world z, so the question is whether a
        # descending tile can put an UPRIGHT pelvis under it: measured on this library the
        # deepest walkable surface is +0.130 m, so an upright pelvis is never below ~0.92 m.
        # 0.5 clears that with margin and still sits well above a fallen one.
        self.terminations.base_fallen = DoneTerm(
            func=mdp.base_fallen,
            params={"min_height": 0.5, "roll_limit_deg": 55.0, "pitch_limit_deg": 55.0},
            time_out=False,
        )
        # It must COST the same as the relative fall, or it is an escape hatch: a
        # termination with no reward term ends the episode with no penalty, which beats the
        # -2 for running the clock out and would be the cheapest way out of a hard tile.
        self.rewards.failed.params["term_keys"] = ["tracking_violation", "base_fallen"]
        # NO FLOW FIELD. The adjuster observes the generator's final Euler-step velocity as
        # a hint about where the plan was heading; a categorical planner has no such
        # intermediate -- its state between decode steps is a distribution over tokens, not a
        # direction in window space -- and `sample_window` refuses `return_flow` rather than
        # returning zeros. So the stash is off and the observation term goes with it.
        self.commands.traj_ref.stash_flow = False
        # `fm_plan_window` returns the plan CONCATENATED WITH ITS FLOW, so it goes entirely
        # rather than being trimmed -- and it costs nothing here: the policy conditions on
        # `plan_conditioning()` (the rows the runtime actually built), not on an observation
        # group, and the frozen tracker reads only the `reference` group.
        # The whole GROUP, not the term inside it: nulling the term leaves an empty group
        # that the observation manager then fails to concatenate.
        if hasattr(self.observations, "fm_plan"):
            self.observations.fm_plan = None
        # The adjuster task serves `measured` history, but the planner was fitted on
        # `history_source='reference'` and the two are the same WIDTH -- so serving the
        # adjuster's arm would feed the model a kind of history it was never trained on,
        # silently. `generator` is the runtime's own served output: reference-kind, and the
        # arm that actually deploys. `assert_history_source_matches` refuses the mismatch,
        # which is how this was caught rather than merely run.
        self.commands.traj_ref.history_from = "generator"
        # The action is decoded through the checkpoint's own frozen tokenizer.
        self.commands.traj_ref.generator_kind = "token_planner"
        # The policy must sample from the SAME rows the runtime built for this re-plan, so
        # the stash is mandatory here where it is opt-in elsewhere. Set on the CFG, not via
        # `enable_cond_stash()`, which runs after the observation manager has probed shapes.
        self.commands.traj_ref.stash_conditioning = True
        # Sampled, not greedy: an on-policy method needs the action to come from the
        # distribution its log-probabilities describe, and greedy has none.
        self.commands.traj_ref.planner_greedy = False
        self.commands.traj_ref.planner_temperature = float(self.roll_temperature)
