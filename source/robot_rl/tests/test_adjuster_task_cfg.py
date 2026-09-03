# Copyright 2026 Zachary Olkin. All rights reserved.

"""The adjuster task's config, and the constants that would ruin a run without a sound.

Most of what a cfg test can catch is caught by construction. These are the ones that are not:

* **``entropy_coef`` and ``desired_kl`` are per-DIMENSION quantities compared against SUMMED
  statistics.** ``GaussianDistribution``'s entropy, log-prob and KL all sum over the action
  dimension, and the adaptive-LR controller compares ``mean(kl)`` to ``desired_kl``. At 2728
  action dims instead of 29 both are ~94x larger, so the teacher's values give either a
  runaway action std or a learning rate pinned at rsl-rl's hardcoded 1e-5 floor -- and neither
  shows up as anything but a bad run. rsl-rl guards neither. The RATIOS are asserted rather
  than the numbers, so re-deriving them for a different generator's F stays legal.
* **The termination flags ARE the reward design.** Only a truncation collects ``gamma * V(s)``,
  so goal-as-truncation while every other ending terminates is the entire mechanism for
  rewarding arrival. Flip one and the return silently stops distinguishing arriving from
  falling.
* **The actor's groups must all be deployable.** A privileged group here would train a policy
  that cannot run on a robot -- and it would train perfectly well.
* **The critic's new terms must be appended LAST**, or a resumed critic is permuted rather than
  zero-padded.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab", reason="needs the full Isaac install")

from robot_rl.tasks.manager_based.robot_rl.g1.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    AdjusterPPORunnerCfg,
    PPORunnerCfg,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_adjuster_env_cfg import (  # noqa: E402
    EPISODE_LENGTH_S,
    GOAL_SLACK_S,
    G1AdjusterEnvCfg,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg import (  # noqa: E402
    G1TeacherObservationsCfg,
)

#: The teacher's action width, which the rescales below are relative to.
TEACHER_ACTION_DIM = 29


def _terms(group) -> list[str]:
    return [k for k in vars(group) if not k.startswith("_")]


# ----------------------------------------------------------------------------------
# The two invisible run-killers
# ----------------------------------------------------------------------------------
def test_entropy_coef_is_rescaled_per_action_dimension() -> None:
    """Left at the teacher's value the entropy bonus swamps the surrogate and sigma diverges."""
    teacher = PPORunnerCfg().algorithm.entropy_coef
    adjuster = AdjusterPPORunnerCfg().algorithm.entropy_coef
    cfg = G1AdjusterEnvCfg()
    del cfg  # constructed only to prove the module imports cleanly beside the agent cfg

    # Per-dimension, the two should be within a small factor of each other. 4x of slack, so a
    # deliberate retune stays legal and a forgotten rescale (94x) does not.
    ratio = (adjuster * 2728) / (teacher * TEACHER_ACTION_DIM)
    assert 0.25 <= ratio <= 4.0, (
        f"entropy_coef is {adjuster:.3e}; per action dimension that is {ratio:.1f}x the "
        f"teacher's {teacher}. rsl-rl SUMS entropy over the action dimension, so at ~94x the "
        "teacher's width the bonus dwarfs the surrogate loss and the action std runs away."
    )


def test_desired_kl_is_rescaled_per_action_dimension() -> None:
    """Left at the teacher's value the LR collapses to rsl-rl's hardcoded floor, silently."""
    teacher = PPORunnerCfg().algorithm.desired_kl
    adjuster = AdjusterPPORunnerCfg().algorithm.desired_kl
    ratio = (adjuster / 2728) / (teacher / TEACHER_ACTION_DIM)
    assert 0.25 <= ratio <= 4.0, (
        f"desired_kl is {adjuster}; per action dimension that is {ratio:.2f}x the teacher's "
        f"{teacher}. rsl-rl compares mean(kl) -- summed over dimensions -- against this, so "
        "an unscaled value divides the learning rate down to its 1e-5 floor and the policy "
        "never moves."
    )


def test_gamma_and_the_minibatch_count_are_inherited_on_purpose() -> None:
    """Both are the teacher's values, and both carry a TODO rather than a justification.

    This is a deliberately weak assertion, and saying so is the point: unlike ``entropy_coef``
    and ``desired_kl`` above -- which MUST be rescaled or the run fails silently -- these two
    are open questions, not settled decisions. Pinning a band around an untuned placeholder
    would dress a guess up as a requirement.

    What is worth recording is that ``gamma`` does NOT mean here what it means for the teacher:
    it is a per-STEP discount and a step is 12 control steps long, so 0.99 is a much weaker
    discount per unit time than the teacher's 0.99 and the return is close to undiscounted over
    an episode. See ``TODO(adjuster-gamma)``.
    """
    adjuster, teacher = AdjusterPPORunnerCfg().algorithm, PPORunnerCfg().algorithm
    assert adjuster.gamma == teacher.gamma
    assert adjuster.num_mini_batches == teacher.num_mini_batches
    # The consequence, stated so a future change to either value is made with it in view.
    assert adjuster.gamma ** 100 > 0.3, (
        "at ~100 decisions per episode this gamma leaves the return nearly undiscounted; that "
        "is a known open item, not a claim that it is right"
    )


# ----------------------------------------------------------------------------------
# The termination flags are the reward design
# ----------------------------------------------------------------------------------
def test_only_the_goal_bootstraps_its_value() -> None:
    """The whole mechanism for rewarding arrival, since neither ending carries a bonus."""
    term = G1AdjusterEnvCfg().terminations
    assert term.goal_reached.time_out is True, (
        "goal_reached must be a truncation: rsl-rl adds gamma*V(s) only to those, and that "
        "asymmetry is what makes arriving worth more than running the clock out"
    )
    for name in ("clip_end", "time_out", "tracking_violation"):
        assert getattr(term, name).time_out is False, (
            f"{name} must terminate, not truncate -- flagged as a truncation it would collect "
            "a positive value bootstrap and the return would stop distinguishing failing from "
            "arriving"
        )


def test_the_absolute_fall_termination_is_absent() -> None:
    """Replaced by the reference-relative check, which is measured to fire 0.20 s earlier."""
    names = _terms(G1AdjusterEnvCfg().terminations)
    assert "base_fallen" not in names
    assert "tracking_violation" in names


def test_the_deadline_has_slack_over_the_clips_own_duration() -> None:
    """Without it, arriving is a race and hurrying is the reliable strategy."""
    cfg = G1AdjusterEnvCfg()
    assert cfg.terminations.clip_end.params["slack_s"] == GOAL_SLACK_S
    assert GOAL_SLACK_S >= 1.0
    # And the episode must outlast the longest clip plus that slack, or `time_out` becomes the
    # binding deadline and the slack buys nothing. Longest last-usable step is 948 control
    # steps; 948 + 3.0/0.02 = 1098, against 24.0/0.02 = 1200.
    assert EPISODE_LENGTH_S / 0.02 > 948 + GOAL_SLACK_S / 0.02


# ----------------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------------
def test_the_actor_sees_only_deployable_groups() -> None:
    """A privileged group would train a policy that cannot run on a robot -- and train well."""
    actor = AdjusterPPORunnerCfg().obs_groups["actor"]
    assert actor == ["fm_cond", "fm_plan"]
    assert "critic" not in actor


def test_the_critic_terms_are_appended_last() -> None:
    """So a resumed critic is zero-padded rather than permuted."""
    teacher = _terms(G1TeacherObservationsCfg.CriticCfg())
    adjuster = _terms(G1AdjusterEnvCfg().observations.critic)
    assert adjuster[: len(teacher)] == teacher, (
        "the teacher's critic terms must keep their positions; pad_critic_for_finetune.py "
        "zero-pads a widened first layer and cannot repair a permutation"
    )
    assert adjuster[len(teacher):] == ["window_residual", "goal_geometry"]


def test_the_plan_groups_are_noise_free() -> None:
    """They are the generator's own output, not a sensor."""
    obs = G1AdjusterEnvCfg().observations
    assert obs.fm_cond.enable_corruption is False
    assert obs.fm_plan.enable_corruption is False


# ----------------------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------------------
def test_each_spec_path_is_stated_once_on_the_env_cfg() -> None:
    """And is NOT copied into the termination params, which is the point.

    An earlier version fanned the paths out in ``__post_init__``. That looked tidier and was
    wrong: hydra applies CLI overrides to the ALREADY-COMPOSED cfg, so the fan-out would copy
    the DEFAULT (empty) and a ``--tracking_spec`` flag would never reach the term -- which
    fails loudly here, but would have been a confusing failure blamed on the flag. The terms
    fall back to these fields at construction instead, which is after any override.
    """
    cfg = G1AdjusterEnvCfg()
    assert cfg.terminations.tracking_violation.params["spec_path"] == ""
    assert cfg.terminations.goal_reached.params["spec_path"] == ""
    # And the fields the terms fall back to exist on the cfg, by the names they look up.
    assert hasattr(cfg, "tracking_spec_path")
    assert hasattr(cfg, "goal_regions_path")


def test_the_loop_is_closed_and_conditioned_on_the_robot() -> None:
    """``fm`` reference with ``measured`` history: the prefix is the adjuster's state encoder.

    Under ``generator`` the history rows are the model's own past output -- a plan, not a robot
    -- so an adjuster asked to correct for where the robot actually is would have no channel
    carrying it.
    """
    cmd = G1AdjusterEnvCfg().commands.traj_ref
    assert cmd.reference_source == "fm"
    assert cmd.history_from == "measured"
    assert cmd.window_anchor == "plan"


def test_no_artifact_path_has_a_default() -> None:
    """A default path to an artifact is a thing this repo has watched rot silently."""
    cfg = G1AdjusterEnvCfg()
    assert cfg.tracker_run_dir == ""
    assert cfg.tracking_spec_path == ""
    assert cfg.goal_regions_path == ""
    assert cfg.commands.traj_ref.fm_checkpoint == ""
    assert cfg.commands.traj_ref.path == ""


def test_the_task_is_registered_with_its_own_env_class() -> None:
    """The stock env would run the adjuster's action through the joint action term."""
    import gymnasium as gym

    import robot_rl  # noqa: F401

    spec = gym.registry["G1-adjuster-fm-terrain-depth"]
    assert spec.entry_point.endswith("adjuster_env:AdjusterEnv")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith("AdjusterPPORunnerCfg")
