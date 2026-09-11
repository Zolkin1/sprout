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


def test_a_slack_world_upright_backstop_covers_the_height_terms_blind_spot():
    """The absolute height term cannot see a fall on RAISED ground, so a backstop is needed.

    `base_height_below_termination` is world z below 0.3 m, which a collapse on the
    up-into-down course's 0.90 m spawn platform -- or on a 0.40 m box top -- never reaches.
    Before `base_upright` such a fall was caught only by the reference-relative cone, which is
    a comparison against a PREDICTION and therefore goes quiet exactly when the generator's
    plan has also fallen apart.

    It has to be SLACK. The termination cfg's own docstring records why a tight cone around
    world upright cannot work on terrain -- climbing and jumping pitch the base deliberately --
    so this pins both directions: nothing fires at a deliberate terrain pitch, and a
    face-plant does.
    """
    import math

    from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (
        G1Box2FinetuneEnvCfg,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations import (
        base_orientation_from_upright,
    )

    cfg = G1Box2FinetuneEnvCfg()
    term = getattr(cfg.terminations, "base_upright", None)
    assert term is not None, "the world-upright backstop is gone"
    assert term.params["roll_limit_deg"] == 70.0
    assert term.params["pitch_limit_deg"] == 70.0
    # It must NOT replace either of the other two -- it is a third opinion, not a swap.
    assert cfg.terminations.base_height is not None
    assert cfg.terminations.base_tilt_error is not None
    # And `base_height` specifically must survive under that NAME: `record_rollouts` resolves
    # the fall term by name (`base_fallen` on the tracker, `base_height` here) and refuses a
    # task carrying neither.
    assert cfg.terminations.base_height.params["min_height"] == 0.3

    class _Asset:
        def __init__(self, quat):
            self.data = type("D", (), {"root_quat_w": quat})()

    class _Env:
        def __init__(self, quat):
            self.scene = {"robot": _Asset(quat)}

    import numpy as np
    import warp as wp

    def fires(pitch_deg: float) -> bool:
        """Whether the backstop trips at a pure pitch of `pitch_deg`.

        The quaternion is XYZW, which is what `root_quat_w` carries, and it is handed over as
        a WARP array because the termination reads it through `wp.to_torch` -- so this
        exercises the shipped function rather than a re-derivation of its arithmetic.
        """
        h = math.radians(pitch_deg) / 2.0
        q = np.array([[0.0, math.sin(h), 0.0, math.cos(h)]], dtype=np.float32)
        return bool(base_orientation_from_upright(
            _Env(wp.array(q, dtype=wp.float32)), 70.0, 70.0)[0])

    # A stair climb is ~20-30 deg and a box jump ~40: healthy terrain motion, must not fire.
    for healthy in (0.0, 20.0, 30.0, 40.0, 55.0):
        assert not fires(healthy), f"the backstop fires at a deliberate {healthy} deg pitch"
    # A face-plant does.
    for fallen in (75.0, 90.0, 120.0):
        assert fires(fallen), f"a {fallen} deg pitch is not caught"
