# Copyright 2026 Zachary Olkin. All rights reserved.

"""The fine-tune task grades the GENERATOR, and its terminations use the right datums.

Two things here are easy to get wrong and silent when wrong. If ``reference_source`` slips
back to ``"clip"`` the task still runs, still trains, and quietly optimises the thing the
deployed policy is not judged on. And if a termination is measured against the wrong datum it
either fires on healthy terrain motion or never fires at all -- both of which look like a
training problem rather than a config one.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab", reason="the env configs import IsaacLab")

from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (  # noqa: E402
    G1TeacherTerrainDepthEnvCfg,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (  # noqa: E402
    G1FinetuneTerrainDepthEnvCfg,
)


@pytest.fixture(scope="module")
def cfgs():
    """The fine-tune config and the teacher it derives from."""
    return G1FinetuneTerrainDepthEnvCfg(), G1TeacherTerrainDepthEnvCfg()


def _terms(group) -> list[str]:
    """Active term names of a manager config, in declaration order."""
    return [k for k, v in vars(group).items()
            if not k.startswith("_") and v is not None and hasattr(v, "func")]


def test_the_generator_is_the_graded_reference(cfgs):
    """``reference_source="fm"`` is the single field the whole design rests on.

    It carries the CLF (and so the reward), the actor's window, the critic's copy of it and
    ``reference_world_state``. Back at ``"clip"`` the task trains against the motion the robot
    will not be given at deploy, with nothing to show that it did.
    """
    finetune, _ = cfgs
    assert finetune.commands.traj_ref.reference_source == "fm"


def test_the_generator_runs_autoregressively(cfgs):
    """Teacher forcing is deliberately off: it phase-locks to a clip nothing now grades."""
    finetune, _ = cfgs
    assert finetune.commands.traj_ref.history_from == "generator"


def test_episodes_open_on_flat_where_the_clip_allows(cfgs):
    """The null-history opening is a flat-ground prior, and hardware starts on flat."""
    finetune, _ = cfgs
    assert finetune.commands.traj_ref.random_start_before_terrain_start is True


def test_history_is_seeded_only_where_the_episode_must_start_on_terrain(cfgs):
    """Seeding everywhere would deny every episode the opening deploy actually gets.

    ``on_terrain_start`` confines it to the clips with no flat lead-in -- the ``*_pure`` stair
    families and the few ``jump_off*`` clips whose terrain begins at step 0 -- which are the
    ones that would otherwise have to be dropped.
    """
    finetune, _ = cfgs
    assert finetune.commands.traj_ref.seed_history_from_clip == "on_terrain_start"


def test_the_critic_and_actor_observations_are_the_teachers(cfgs):
    """No observation edits: with ``reference_source="fm"`` both groups follow it already.

    This is what keeps a resumed teacher loading with no checkpoint surgery -- the critic's
    input width is unchanged. An earlier design appended a residual block to the critic and
    needed a padded checkpoint; if that ever comes back, this test is where it shows up.
    """
    finetune, teacher = cfgs
    for group in ("policy", "critic", "reference"):
        assert (getattr(finetune.observations, group).to_dict()
                == getattr(teacher.observations, group).to_dict()), f"{group} obs changed"


def test_the_clip_graded_failures_are_gone_and_clip_end_stays(cfgs):
    """``base_pose_error`` / ``ee_pose_error`` measure a clip that is no longer tracked.

    ``clip_end`` is NOT one of them: it is a truncation that reads only the clip's LENGTH, and
    it is the only thing stopping an episode when the motion runs out. Without it the robot is
    commanded the last frame's velocity off the end of a finite tile.
    """
    finetune, _ = cfgs
    terms = _terms(finetune.terminations)
    assert "base_pose_error" not in terms
    assert "ee_pose_error" not in terms
    assert "clip_end" in terms
    assert finetune.terminations.clip_end.time_out is True


def test_height_is_absolute_and_tilt_is_reference_relative(cfgs):
    """The two datums differ on purpose, and each is wrong the other way round.

    A world-upright cone cannot work on terrain -- climbing and jumping pitch the base
    deliberately. A reference-RELATIVE height cannot work either: ego re-anchoring re-datums
    the reference onto the robot every re-plan, so that error resets toward zero and a slow
    sink escapes it.
    """
    finetune, _ = cfgs
    height = finetune.terminations.base_height
    tilt = finetune.terminations.base_tilt_error

    assert height.func.__name__ == "base_height_below_termination"
    assert "command_name" not in (height.params or {})      # absolute: no reference at all
    assert height.params["min_height"] == pytest.approx(0.3)
    assert height.time_out is False

    assert tilt.func.__name__ == "fm_base_tilt_error_termination"
    assert tilt.params["command_name"] == "traj_ref"         # relative to the served reference
    assert tilt.params["max_tilt_rad"] == pytest.approx(1.047, abs=1e-3)   # 60 deg
    assert tilt.time_out is False


def test_the_velocity_command_still_comes_from_the_clip(cfgs):
    """The joystick track is the clip's ``target_vel``, whatever the reference source is.

    ``command_window`` reads the library directly rather than the served reference, so this
    holds by construction -- pinned because it is a premise of the design, not an accident.
    """
    finetune, teacher = cfgs
    assert (finetune.observations.reference.command_window.to_dict()
            == teacher.observations.reference.command_window.to_dict())


def test_the_task_is_registered():
    """Reachable by id, with an rsl_rl agent cfg carrying the fine-tune hyperparameters."""
    import gymnasium as gym

    import robot_rl.tasks  # noqa: F401

    spec = gym.spec("G1-teacher-clf-terrain-depth-finetune")
    assert "g1_finetune_env_cfg:G1FinetuneTerrainDepthEnvCfg" in spec.kwargs["env_cfg_entry_point"]
    assert "FinetunePPORunnerCfg" in spec.kwargs["rsl_rl_cfg_entry_point"]
