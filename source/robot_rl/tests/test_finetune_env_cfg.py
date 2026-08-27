# Copyright 2026 Zachary Olkin. All rights reserved.

"""The fine-tune task changes the OBSERVATION and nothing that is graded.

That separation is the whole premise of ``G1-teacher-clf-terrain-depth-finetune``: the actor
is shown the generator's reference window while the reward, the critic, the terminations and
every DR term keep grading the clip, exactly as in teacher training.  If any of those move,
the reward stops describing the motion it was chosen for and the fine-tune is measuring
something nobody designed.

Nothing here needs a simulator -- these are config identities.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("isaaclab", reason="the env configs import IsaacLab")

from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (  # noqa: E402
    G1TeacherTerrainDepthEnvCfg,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (  # noqa: E402
    FINETUNE_NOISE_SCALE,
    G1FinetuneTerrainDepthEnvCfg,
    _scaled_reference_noise,
)


@pytest.fixture(scope="module")
def cfgs():
    """The fine-tune config and the teacher it must agree with everywhere but the window."""
    return G1FinetuneTerrainDepthEnvCfg(), G1TeacherTerrainDepthEnvCfg()


@pytest.mark.parametrize(
    "attr",
    ["rewards", "terminations", "events"],
)
def test_the_graded_managers_are_the_teachers(cfgs, attr):
    """Reward, termination and DR configs are untouched.

    These are what the fine-tune must NOT move: every one of them grades the clip, and the
    point of serving the generator only at observation time is that they carry on doing so.
    """
    finetune, teacher = cfgs
    assert getattr(finetune, attr).to_dict() == getattr(teacher, attr).to_dict()


@pytest.mark.parametrize("group", ["policy", "critic"])
def test_the_non_reference_obs_groups_are_the_teachers(cfgs, group):
    """Proprioception/exteroception and the privileged critic are untouched.

    The reference lives in its own group, which is what makes swapping it a local change --
    if it did not, this test would be the one to notice.
    """
    finetune, teacher = cfgs
    assert (getattr(finetune.observations, group).to_dict()
            == getattr(teacher.observations, group).to_dict())


def test_the_reference_window_is_served_by_the_generator(cfgs):
    """The one term that moves, and the one that deliberately does not.

    ``command_window`` stays on the clip: it is the joystick command, which at deploy comes
    from the operator rather than the generator.
    """
    finetune, teacher = cfgs
    assert finetune.observations.reference.reference_window.func.__name__ == "fm_reference_window"
    assert teacher.observations.reference.reference_window.func.__name__ == "reference_window"
    assert finetune.observations.reference.command_window.func.__name__ == "command_window"


def test_the_graded_reference_is_still_the_clip(cfgs):
    """``reference_source="clip"`` is the safety property, not an oversight.

    With it, ``y_des`` -- and so the CLF, the critic's error vectors and the clip-graded
    terminations -- is the clip.  Flipping it to ``"fm"`` would re-point all of them at the
    generator, which is a far larger change than serving one observation term.
    """
    finetune, _ = cfgs
    assert finetune.commands.traj_ref.reference_source == "clip"


def test_teacher_forcing_is_on(cfgs):
    """``history_from="clip"`` keeps the prediction phase-locked to the graded motion.

    The closed-loop arm is one field away and is deliberately not the default: with the
    generator drifting on its own history, the reward stops describing what the policy saw.
    """
    finetune, _ = cfgs
    assert finetune.commands.traj_ref.history_from == "clip"


def test_the_reference_window_layout_matches_the_teacher(cfgs):
    """Width, clip and post-noise scale are shared, so a checkpoint moves between the tasks.

    Only the noise MAGNITUDE is allowed to differ, and only through
    :data:`FINETUNE_NOISE_SCALE`.  Anything else here moving would change the observation's
    shape or units, and the resumed teacher would be reading a different vector.
    """
    finetune, teacher = cfgs
    ref_f = finetune.observations.reference.reference_window
    ref_t = teacher.observations.reference.reference_window

    assert (np.asarray(ref_f.noise.n_half).shape
            == np.asarray(ref_t.noise.n_half).shape)
    assert np.allclose(np.asarray(ref_f.scale, dtype=float),
                       np.asarray(ref_t.scale, dtype=float))
    assert ref_f.clip == ref_t.clip


def test_the_reference_noise_is_the_teachers_at_the_default_scale(cfgs):
    """At ``FINETUNE_NOISE_SCALE = 1.0`` the noise IS the teacher's, channel for channel.

    Reducing it is a defensible follow-up -- the generator now supplies for real the error
    this term was faking -- but it would change the reference AND the jitter hardening in one
    run, so the default keeps it fixed and leaves that as its own sweep.
    """
    finetune, teacher = cfgs
    n_f = np.asarray(finetune.observations.reference.reference_window.noise.n_half, dtype=float)
    n_t = np.asarray(teacher.observations.reference.reference_window.noise.n_half, dtype=float)
    assert FINETUNE_NOISE_SCALE == 1.0
    assert np.allclose(n_f, n_t)
    assert (finetune.observations.reference.reference_window.noise.magnitudes["joint_pos"]
            == teacher.observations.reference.reference_window.noise.magnitudes["joint_pos"])


def test_the_noise_scale_knob_actually_scales(cfgs):
    """The sweep lever works, and scales every channel by the same factor.

    Pinned separately from the default so turning the knob down later cannot silently
    reshape the vector instead of rescaling it.
    """
    _, teacher = cfgs
    halved = _scaled_reference_noise(0.5)
    n_h = np.asarray(halved.n_half, dtype=float)
    n_t = np.asarray(teacher.observations.reference.reference_window.noise.n_half, dtype=float)
    assert n_h.shape == n_t.shape
    nonzero = n_t != 0.0
    assert nonzero.any()
    assert np.allclose(n_h[nonzero] / n_t[nonzero], 0.5)
    assert halved.magnitudes["joint_pos"] == pytest.approx(0.05)


def test_the_task_is_registered():
    """The env is reachable by id, and carries an rsl_rl agent cfg so train.py can run it."""
    import gymnasium as gym

    import robot_rl.tasks  # noqa: F401  (registers)

    spec = gym.spec("G1-teacher-clf-terrain-depth-finetune")
    assert "g1_finetune_env_cfg:G1FinetuneTerrainDepthEnvCfg" in spec.kwargs["env_cfg_entry_point"]
    assert spec.kwargs["rsl_rl_cfg_entry_point"]
