# Copyright 2026 Zachary Olkin. All rights reserved.

"""What counts as a fall in MuJoCo sim2sim.

This is scoring code, and wrong scoring is the quietest kind of wrong: it does not crash, it
publishes a number. The bug it exists to prevent already happened -- ``Robot.failed()`` was
``qpos[2] < 0.2`` in WORLD coordinates, which assumes the ground is at zero. On
``g1_perceptive_tile20_scene`` the bottom slab is at -0.777 m, so a healthy robot standing there has
a pelvis at ~0.0 and was scored as fallen the instant it arrived. Every stairs-down run was a
failure before it started, and the conclusion drawn from those runs ("stairs down does not
work") was measuring the scoring, not the policy.

Two conditions now, and each covers the other's blind spot:

* HEIGHT, relative to the scene's lowest terrain surface. One datum per scene, so it is loose
  on multi-level terrain -- on tile20 it lands at -0.707 m.
* TILT, roll and pitch separately. This is what catches a collapse on an UPPER level, which
  stays metres above the scene minimum but cannot stay upright.

The tilt bound deliberately does NOT include yaw: yaw is heading, and the eval command
profiles turn on purpose (``smooth_ramp_running`` yaws at t=10-14 s). A yaw bound would score
a turn as a fall, which is why ``test_a_pure_yaw_is_never_a_fall`` exists.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")

_SIM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

from robot import Robot  # noqa: E402

#: The limit the runner defaults to, in degrees.
_TILT_DEG = 70.0


def _robot(scene: str) -> Robot:
    """A robot in ``scene``, standing at the keyframe pose.

    Args:
        scene: Scene name, e.g. ``tile20_scene``.

    Returns:
        The constructed :class:`Robot`.
    """
    return Robot(robot_name="g1_perceptive", scene_name=scene, joystick_scaling=np.ones(3),
                 input_function=lambda t: np.zeros(3))


def _set_orientation(robot: Robot, axis: str, degrees: float) -> None:
    """Rotate the free joint about one world axis, leaving position alone.

    Args:
        robot: Target robot.
        axis: ``"roll"`` (x), ``"pitch"`` (y) or ``"yaw"`` (z).
        degrees: Rotation angle.
    """
    half = np.deg2rad(degrees) / 2.0
    unit = {"roll": (1, 0, 0), "pitch": (0, 1, 0), "yaw": (0, 0, 1)}[axis]
    robot.mj_data.qpos[3:7] = [np.cos(half), *(np.sin(half) * np.asarray(unit))]


#: (scene, expected lowest terrain surface). Read off the MJCF geometry by hand, so a scene
#: edit that moves the ground has to be acknowledged here rather than silently rescoring runs.
#: `basic_scene`'s floor is an infinite PLANE at 0, which is what keeps flat results identical
#: to the historical absolute 0.2 m bound.
_SCENE_DATUM = [
    ("basic_scene", 0.0),
    ("terrain_scene", 0.0),
    ("tile0_scene", -0.130),
    ("tile26_scene", -0.450),
    ("tile20_scene", -0.907),
]


@pytest.mark.parametrize("scene,expected", _SCENE_DATUM)
def test_the_terrain_datum_is_read_from_the_model(scene, expected):
    """Every scene's lowest surface, from the geometry rather than a per-scene constant."""
    assert _robot(scene).terrain_min_z == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("scene", [s for s, _ in _SCENE_DATUM])
def test_standing_at_the_keyframe_is_never_a_fall(scene):
    """THE regression. A robot standing still, on any scene, has not fallen.

    On tile20 this failed before the datum was made terrain-relative -- which is the whole
    reason stairs-down looked broken.
    """
    robot = _robot(scene)
    assert not robot.failed(), (
        f"{scene}: standing at the keyframe scores as a fall (pelvis "
        f"{robot.mj_data.qpos[2]:+.3f} m, threshold "
        f"{robot.terrain_min_z + robot.fall_height:+.3f} m)"
    )


def test_flat_scoring_is_unchanged():
    """A flat scene must reduce to the historical absolute threshold, or old runs moved."""
    robot = _robot("basic_scene")
    assert robot.terrain_min_z == pytest.approx(0.0)
    assert robot.terrain_min_z + robot.fall_height == pytest.approx(0.2)


@pytest.mark.parametrize("axis", ["roll", "pitch"])
@pytest.mark.parametrize("degrees,fallen", [(0.0, False), (60.0, False), (69.0, False),
                                           (71.0, True), (90.0, True), (180.0, True)])
def test_tilt_is_bounded_on_both_axes(axis, degrees, fallen):
    """Roll and pitch each trip at the limit, and the transition is where it is claimed."""
    robot = _robot("basic_scene")
    robot.mj_data.qpos[2] = 0.79          # well clear of the height condition
    _set_orientation(robot, axis, degrees)
    assert robot.failed() is fallen, f"{axis} {degrees} deg: expected fallen={fallen}"


@pytest.mark.parametrize("degrees", [0.0, 60.0, 71.0, 90.0, 180.0, -140.0])
def test_a_pure_yaw_is_never_a_fall(degrees):
    """The power check on the tilt bound: spinning about gravity is not falling over.

    If this fails, the bound was written on Euler angles or on the wrong matrix column, and
    every eval run that turns would be scored as a fall partway through.
    """
    robot = _robot("basic_scene")
    robot.mj_data.qpos[2] = 0.79
    _set_orientation(robot, "yaw", degrees)
    roll, pitch = robot.tilt_from_upright()
    assert roll == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)
    assert not robot.failed()


def test_the_height_condition_still_bites_where_it_should():
    """Upright but on the floor is still a fall -- the tilt check must not mask it."""
    robot = _robot("tile20_scene")
    robot.mj_data.qpos[2] = robot.terrain_min_z + 0.5 * robot.fall_height
    assert robot.tilt_from_upright() == (pytest.approx(0.0), pytest.approx(0.0))
    assert robot.failed()
