# Copyright 2026 Zachary Olkin. All rights reserved.

"""The token-planner RL task's config: the parts that would ruin a run without a sound.

* **The termination flags ARE the reward design.** Arrival carries no reward term, so
  ``gamma * V(s)`` on the one truncation is the entire mechanism that makes finishing beat
  running the clock out. Flip one flag and the return stops distinguishing arriving from
  failing -- while the dense reward keeps rising.
* **The two residual reward terms must be GONE, not zeroed.** They price a correction against
  the plan the generator drew, and this agent emits the plan outright; left in, they would
  price zeros and read as a well-behaved agent.
* **The history arm must match what the checkpoint was fitted on.** ``measured`` and
  ``reference`` rows are the same WIDTH, so serving the wrong one is silent.
* **The flow field must be off.** A categorical planner has no velocity field; the adjuster's
  observation of one cannot be served and the env must not require its stash.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab", reason="needs the full Isaac install")

from robot_rl.tasks.manager_based.robot_rl.g1.g1_token_planner_env_cfg import (  # noqa: E402
    G1TokenPlannerEnvCfg,
)


@pytest.fixture(scope="module")
def cfg() -> G1TokenPlannerEnvCfg:
    """The composed task cfg."""
    return G1TokenPlannerEnvCfg()


def test_only_the_goal_bootstraps_its_value(cfg) -> None:
    """The whole mechanism for rewarding arrival, since no ending carries a bonus."""
    t = cfg.terminations
    assert t.goal_reached.time_out is True, (
        "goal_reached must be a truncation: it is the only ending that collects gamma*V(s), "
        "and that asymmetry is what makes arriving worth more than running the clock out"
    )
    for name in ("clip_end", "time_out", "tracking_violation"):
        assert getattr(t, name).time_out is False, (
            f"{name} must terminate, not truncate -- as a truncation it would collect a "
            "positive bootstrap and the return would stop telling failing from arriving"
        )


def test_the_reward_set_is_the_three_the_spec_names(cfg) -> None:
    """Trackability, fall, timeout -- and the timeout must cost less than the fall."""
    r = cfg.rewards
    assert r.trackability.weight > 0.0
    assert r.failed.weight < 0.0
    assert r.ran_out_of_time.weight < 0.0
    assert r.failed.weight < r.ran_out_of_time.weight, (
        "a fall must cost strictly more than a timeout, or standing still until the deadline "
        "becomes the safe play"
    )


def test_the_residual_reward_terms_are_absent(cfg) -> None:
    """There is no reference plan to price a correction against."""
    for banned in ("residual_l2", "residual_rate_l2"):
        assert not hasattr(cfg.rewards, banned), (
            f"{banned} prices a residual against the plan the generator drew; this agent's "
            "tokens ARE the plan, so the term would price zeros and read as good behaviour"
        )


def test_the_generator_is_a_token_planner(cfg) -> None:
    """The action is decoded through the checkpoint's own frozen tokenizer."""
    assert cfg.commands.traj_ref.generator_kind == "token_planner"


def test_the_history_arm_matches_supervised_training(cfg) -> None:
    """``measured`` and ``reference`` rows are the same width, so a mismatch is silent."""
    assert cfg.commands.traj_ref.history_from == "generator", (
        "the planner was fitted on history_source='reference'; 'generator' is the "
        "reference-kind arm (the runtime's own served output) and the one that deploys. "
        "'measured' is a different KIND of history at identical width"
    )


def test_the_conditioning_stash_is_on_and_the_flow_stash_is_off(cfg) -> None:
    """The policy samples from the stash; the flow field does not exist for this generator."""
    assert cfg.commands.traj_ref.stash_conditioning is True, (
        "the policy must sample from the rows the runtime actually built for this re-plan"
    )
    assert cfg.commands.traj_ref.stash_flow is False


def test_the_flow_observation_group_is_gone(cfg) -> None:
    """``fm_plan_window`` returns the plan CONCATENATED with its flow, so it goes entirely."""
    assert getattr(cfg.observations, "fm_plan", None) is None


def test_the_critic_group_survives_and_can_see_the_goal_and_the_robot(cfg) -> None:
    """The critic must predict the fall term and the arrival bootstrap; both need this group.

    The planner's own conditioning row cannot: it carries no measured robot state (its
    history rows are the REFERENCE) and no goal, and its depth channels are unprojected in a
    level frame precisely so that tilt does not move static terrain.
    """
    critic = getattr(cfg.observations, "critic", None)
    assert critic is not None, "the privileged critic group was removed"
    names = {k for k in vars(critic) if not k.startswith("_")}
    for needed in ("projected_gravity", "joint_pos", "goal_geometry"):
        assert needed in names, (
            f"the critic cannot see '{needed}'. Without projected_gravity it cannot predict "
            "the tilt that fires the fall penalty; without goal_geometry it cannot predict "
            "the truncation whose bootstrap is arrival's entire payoff"
        )


def test_the_policy_samples_rather_than_taking_the_argmax(cfg) -> None:
    """An on-policy method needs the action drawn from the distribution it scores."""
    assert cfg.commands.traj_ref.planner_greedy is False, (
        "greedy decoding has no distribution, so its log-probabilities do not describe how "
        "the action was chosen and the PPO ratio is meaningless"
    )
