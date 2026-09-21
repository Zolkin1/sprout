# Copyright 2026 Zachary Olkin. All rights reserved.

"""The PPO arm's MDP is the off-policy arms' MDP, and the departures are the stated ones.

``G1-adjuster-ft-freeform`` exists to answer one question -- what does swapping AWR for PPO buy
-- and it only answers it if everything EXCEPT the algorithm matches
``G1-teacher-clf-terrain-depth-finetune-freeform``, the task `configs/finetune/boxes.yaml`
names. Every difference is either listed here as deliberate or is a bug.

The whole file runs on CONSTRUCTED cfgs, never on the class, because the difference between the
two has cost this repo runs before: `G1TeacherEnvCfg.__post_init__` overwrites weights and
`__post_init__` chains mutate terms by name, so a value read off a declaration is not the value
the run gets.
"""

from __future__ import annotations

import pytest

from robot_rl.tasks.manager_based.robot_rl.g1.agents.rsl_rl_ppo_cfg import (
    AdjusterFreeformPPORunnerCfg,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_adjuster_env_cfg import (
    FREEFORM_EPISODE_LENGTH_S,
    G1AdjusterFreeformEnvCfg,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (
    G1FinetuneFreeformEnvCfg,
)


@pytest.fixture(scope="module")
def cfgs():
    """The PPO arm and the task it must match, both constructed once."""
    return G1AdjusterFreeformEnvCfg(), G1FinetuneFreeformEnvCfg()


# ==============================================================================================
# What must be IDENTICAL
# ==============================================================================================
def test_the_terminations_are_the_finetune_tasks_exactly(cfgs) -> None:
    """Same terms, same funcs, same params, same ``time_out`` flags.

    The flags are the half that decides the RETURN rather than the episode: `time_out=True`
    bootstraps `gamma * V(s)` and `False` does not, which is what makes a failure absorbing and
    a budget end continue -- the two boundary conditions the offline reducer's tail convention
    also encodes. A flag flipped here would change the target while leaving every episode
    length identical.
    """
    a, b = cfgs
    ta, tb = vars(a.terminations), vars(b.terminations)
    assert set(ta) == set(tb)
    for k in ta:
        if ta[k] is None or tb[k] is None:
            assert (ta[k] is None) and (tb[k] is None), f"{k} exists on only one side"
            continue
        assert ta[k].func is tb[k].func, k
        assert ta[k].params == tb[k].params, k
        assert ta[k].time_out == tb[k].time_out, k


def test_the_failure_bound_is_BOOTSTRAP_SPEC_not_a_fitted_one(cfgs) -> None:
    """A fitted spec would make the arms incomparable, and loudly.

    `rollout_outcomes.assert_tracking_bounds_agree` refuses to compare two recordings made under
    different tracking bounds -- the same parent measured 0.2440 under a latched 0.20 m bound
    and 0.0002 under a terminating 0.12 m one. The off-policy arms terminate on
    `fm_tracking_violation_termination`, which reads `BOOTSTRAP_SPEC`, so this one must too.
    """
    a, _ = cfgs
    assert a.terminations.tracking_violation.params == {"command_name": "traj_ref"}, (
        "the violation term takes a spec path, so this arm's failure bounds are not the "
        "off-policy arms'")


def test_the_world_DR_is_untouched(cfgs) -> None:
    """The teacher's own randomisation is part of the transition kernel, not of an algorithm."""
    a, b = cfgs
    ea, eb = vars(a.events), vars(b.events)
    assert set(ea) == set(eb)
    for k in ea:
        assert (ea[k] is None) == (eb[k] is None), k
        if ea[k] is not None:
            assert ea[k].func is eb[k].func, k
            assert ea[k].params == eb[k].params, k
            assert ea[k].mode == eb[k].mode, k


def test_the_trackers_observation_corruption_stays_ON(cfgs) -> None:
    """Including the reference-window jitter the CLIP-terrain arm silences.

    That arm turned it off because a ~0.006 m learned correction was drowning in +-0.06 m of
    jitter, and on its own terms that was right. Here it would be an MDP difference: the tracker
    is frozen in both arms, so `REFERENCE_WINDOW_NOISE` is part of the transition kernel the
    off-policy collect also runs under. Removing it for one arm makes them different problems.
    """
    a, b = cfgs
    for group in ("policy", "reference"):
        assert getattr(a.observations, group).enable_corruption is True, group
    assert a.observations.reference.reference_window.noise is not None, (
        "the reference-window noise was silenced; the off-policy collect has it")
    assert (a.observations.reference.reference_window.noise
            == b.observations.reference.reference_window.noise)


def test_the_episode_budget_is_the_collects(cfgs) -> None:
    """40 s, what `run_finetune_loop.sh` passes, not the teacher's 20."""
    a, _ = cfgs
    assert a.episode_length_s == FREEFORM_EPISODE_LENGTH_S == 40.0


def test_the_command_is_the_free_form_one_and_the_history_arm_is_generator(cfgs) -> None:
    """`measured` is not a preference here, it is a refusal at generator load.

    The ladder's parent is a `reference`-history checkpoint, and
    `assert_history_source_matches` kills the run. So the clip-terrain arm's pin must NOT be
    inherited or re-stated.
    """
    a, b = cfgs
    ca, cb = a.commands.traj_ref, b.commands.traj_ref
    assert ca.history_from == "generator" == cb.history_from
    for f in ("freeform_command", "tracking_mode", "reference_source", "window_anchor",
              "seed_history_from_clip", "random_start_phase", "fm_update_every",
              "freeform_cmd_hold_s", "freeform_steer_kp", "freeform_approach_kp"):
        assert getattr(ca, f) == getattr(cb, f), f


def test_the_clip_only_start_machinery_is_off(cfgs) -> None:
    """`stand_start` and `cmd_hold_s` are read by the CLIP reset, which this task disables.

    Inert, so easy to leave on -- except `seed_history_from_stand`, which is NOT: it is applied
    in `_snapshot_and_infer` regardless of the command source, so it would change the opening
    plan. The off-policy collect opens on the learned null-history embedding.
    """
    a, _ = cfgs
    c = a.commands.traj_ref
    assert c.stand_start is False
    assert c.seed_history_from_stand is False
    assert a.terminations.clip_end is None
    assert a.events.reset_on_clip is None


def test_the_spawn_is_the_manifest_sections(cfgs) -> None:
    """Including `spawn_from="all"`: the raised areas are what put the second half of a
    traverse in the data at all, and every EVAL overrides it to `ground`."""
    a, b = cfgs
    assert a.events.reset_on_section.params == b.events.reset_on_section.params
    assert a.events.reset_on_section.params["spawn_from"] == "all"


# ==============================================================================================
# The ONE deliberate departure
# ==============================================================================================
def test_the_three_generator_side_collect_corruptions_are_OFF(cfgs) -> None:
    """The stated departure, pinned in both directions so it cannot drift either way.

    These are collect-only overrides in `run_finetune_loop.sh`, not cfg values, and they perturb
    the GENERATOR rather than the world. `fm_perturb_fraction` in particular exists to
    manufacture the behavioural spread AWR reweights, which is a job PPO's own Gaussian does.

    Note the gate defaults to 1.0, not 0: with both inner fractions at 0 that changes nothing,
    but it READS as "perturbation on", so the cfg states it.
    """
    a, _ = cfgs
    c = a.commands.traj_ref
    assert c.fm_perturb_fraction == 0.0
    assert c.fm_upper_camera_noise_fraction == 0.0
    assert c.fm_speed_mismatch_fraction == 0.0
    assert c.fm_history_null_fraction == 0.0


# ==============================================================================================
# What the ADJUSTER adds, and the two numbers that must agree across cfg objects
# ==============================================================================================
def test_the_reward_is_the_progress_reward_and_only_that(cfgs) -> None:
    """Three terms at weight 1.0. Not one of the clip arm's five survives."""
    a, _ = cfgs
    assert set(vars(a.rewards)) == {"progress_terrain", "progress_velocity", "progress_success"}
    for name, term in vars(a.rewards).items():
        assert term.weight == 1.0, f"{name} is weighted; the reward being matched has no weights"
        assert term.params == {"command_name": "traj_ref"}, name


def test_the_critic_can_see_the_deadline_and_the_plan_anchor_drift(cfgs) -> None:
    """Two channels the free-form base and the bootstrap respectively require.

    `tracking_error_global` at the `plan_anchor` datum: the base re-points it by name, so a
    critic group without it is an AttributeError at composition -- and under
    `history_from="generator"` it is the only channel carrying how far the tracker has fallen
    behind the plan. `goal_geometry`'s fourth channel is the deadline, which a bootstrapped
    `time_out` truncation needs the critic to see.
    """
    a, _ = cfgs
    crit = a.observations.critic
    assert crit.tracking_error_global.params["datum"] == "plan_anchor"
    gg = crit.goal_geometry.params
    assert gg["slack_s"] == 0.0, "clip_end is disabled here, so there is no slack to be in step with"
    assert gg["step_scale"] == pytest.approx(1.0 / (FREEFORM_EPISODE_LENGTH_S / 0.02))


def test_the_discount_is_stated_once_and_reaches_both_cfgs(cfgs) -> None:
    """`progress_success` scales the bonus by `1 - gamma`; PPO discounts by `gamma`.

    They are two config objects and one number. If they disagreed, the arrival bonus would be
    mis-scaled relative to the return it is meant to enter -- and nothing else would look wrong.
    """
    a, _ = cfgs
    assert a.progress_gamma == AdjusterFreeformPPORunnerCfg().algorithm.gamma == 0.95


def test_the_action_is_the_full_per_node_residual(cfgs) -> None:
    """One knot per plan node, so the lift is the identity and the action is the whole window.

    Pinned across both cfgs because `AdjusterEnv._residual_from_action` reads the env's and the
    actor emits the model's; a mismatch is caught there, but at simulator-launch time.
    """
    a, _ = cfgs
    assert a.num_knots == AdjusterFreeformPPORunnerCfg().actor.num_knots == 62


def test_the_knot_basis_at_one_knot_per_node_is_exactly_the_identity() -> None:
    """Which is what makes `num_knots = 62` a config field rather than a second code path."""
    import numpy as np

    from robot_rl.flow_matching.knot_basis import knot_basis

    b = knot_basis(62, 62, "catmull")
    np.testing.assert_array_equal(b, np.eye(62))


def test_steering_is_the_commands_not_the_adjusters(cfgs) -> None:
    """`AdjusterEnv._install_steering` scans for a `mdp.GoalReached` INSTANCE, and this task
    uses the `freeform_goal_reached` FUNCTION -- so a non-zero gain raises at construction."""
    a, b = cfgs
    assert a.steer_kp == 0.0
    assert a.commands.traj_ref.freeform_steer_kp == b.commands.traj_ref.freeform_steer_kp > 0.0


# ==============================================================================================
# The PLAY variant: watchable, and behaviourally the trained task
# ==============================================================================================
def test_play_differs_from_training_only_in_what_is_WATCHED() -> None:
    """A play cfg that quietly changed the task would show something that was never scored.

    So the allowed differences are enumerated: fewer envs on one terrain row, the randomisation
    off, and three debug draws. Everything that decides BEHAVIOUR -- the command, the
    terminations, the frozen pair's settings, the speed band -- must be the training task's.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_adjuster_env_cfg import (
        G1AdjusterFreeformEnvCfgPlay,
    )

    play, train = G1AdjusterFreeformEnvCfgPlay(), G1AdjusterFreeformEnvCfg()

    assert play.scene.num_envs < train.scene.num_envs
    assert play.scene.terrain.num_rows == 1
    for group in ("policy", "reference"):
        assert getattr(play.observations, group).enable_corruption is False
    assert play.commands.traj_ref.debug_vis is True
    assert play.scene.terrain_rays.debug_vis is True
    assert play.scene.terrain_base_patch.debug_vis is True

    # Decoration STAYS ON, unlike the clip-terrain play cfg. The generator was fitted on
    # decorated scenes and misreads a bare vertical surface as a box, so watching it without
    # decoration is watching a different input distribution.
    assert play.scene.terrain.render_decor is True

    # ...and nothing behavioural moved.
    assert play.episode_length_s == train.episode_length_s
    assert play.progress_gamma == train.progress_gamma
    assert play.num_knots == train.num_knots
    assert play.steer_kp == train.steer_kp
    for f in ("history_from", "freeform_command", "tracking_mode", "freeform_speed_min",
              "freeform_speed_max", "fm_update_every", "fm_perturb_fraction"):
        assert getattr(play.commands.traj_ref, f) == getattr(train.commands.traj_ref, f), f
    ta, tb = vars(play.terminations), vars(train.terminations)
    assert set(ta) == set(tb)
    for k in ta:
        assert (ta[k] is None) == (tb[k] is None), k
        if ta[k] is not None:
            assert ta[k].func is tb[k].func and ta[k].time_out == tb[k].time_out, k
