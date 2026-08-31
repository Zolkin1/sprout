# Copyright 2026 Zachary Olkin. All rights reserved.

"""The per-CLIP course: does it reproduce an Isaac episode's definition, exactly.

``isaac_tile_course`` builds one course per TILE, which forces a reduction of the tile's ~50
clips to one number per field -- and any two fields reduced differently describe a trajectory
no clip ever ran.  Measured: a ``max`` duration beside a ``median`` endpoint gave tile 6 zero
successes in six reference episodes, an infeasible course rather than a policy result.
``isaac_clip_course`` takes every number from ONE clip, as Isaac does.

Two of the things pinned here were silently wrong when first written, and both failed as a
plausible locomotion outcome rather than as an error:

* **the joint ORDER.**  ``y_des``'s joint columns are the ROBOT's per-limb order, while
  ``Spawn.joint_pos_isaac`` is the POLICY's interleaved order.  They agree in 2 of 29
  positions.  A positional copy puts every joint at another joint's angle and still stands up.
* **the flight DATUM.**  ``Course.traversed`` tests ``max_base_z >= SPAWN_BASE_Z +
  TRAVERSAL_FRAC * flight_height_m``, i.e. a RISE from the spawn pelvis.  Passing the goal's
  height above the spawn GROUND inflates it by a pelvis height (1.612 against 0.826 on clip
  300), no episode is ever marked traversed, and every one times out *after walking through
  the goal band* -- which reads as the policy failing to arrive.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# transfer/sim and transfer/ both on the path: pytest imports transfer/sim/__init__.py while
# collecting here, and that file's `from sim.robot import Robot` only resolves with transfer/
# on the path. Same bootstrap every other test in this directory uses.
_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(_SIM))
for _p in (_SIM, os.path.dirname(_SIM), os.path.join(_SIM, "mjeval"),
           os.path.join(_ROOT, "source", "robot_rl"),
           os.path.join(_ROOT, "source", "motion_gen"),
           os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LIB = os.path.join(_ROOT, "clips", "g1_mb_v5_body_mode11_decor.h5")

pytestmark = pytest.mark.skipif(not os.path.exists(LIB),
                               reason="needs the mode11 clip library")

#: ``(clip_id, tile)`` -- one clip per feature family, all buildable as courses.
CASES = [(300, 6), (1000, 20), (2000, 40)]


@pytest.mark.parametrize("clip_id,tile", CASES)
def test_every_field_comes_from_the_named_clip(clip_id, tile):
    """Budget, goal, command and start must be THAT clip's, not a tile aggregate."""
    import goal_anchor as ga
    import terrains

    ep = ga.clip_episode(LIB, clip_id, max_offset=50)
    c = terrains.isaac_clip_course(LIB, clip_id)
    assert c.params["terrain_id"] == tile
    assert c.params["clip_id"] == clip_id
    assert c.budget_s == pytest.approx(ep.duration_s + terrains.CMD_HOLD_S)
    assert c.params["clip_command"][0] == pytest.approx(list(ep.command[0]))
    # The goal is the clip's endpoint relative to the clip's start, both in the tile frame.
    sx, sy = c.params["spawn_tile_xy"]
    assert c.target_xy[0] == pytest.approx(ep.goal_xyz[0] - sx)
    assert c.target_xy[1] == pytest.approx(ep.goal_xyz[1] - sy)


@pytest.mark.parametrize("clip_id,tile", CASES)
def test_flight_height_is_a_rise_not_an_absolute_height(clip_id, tile):
    """The clip course's flight must agree with the tile course's net terrain change.

    Two independent derivations of the same physical quantity: the tile course takes it from
    the walkable surfaces, this one from the clip's own pelvis trajectory.  Agreement to a few
    cm is what says the DATUM is right -- and the datum is exactly what was wrong.
    """
    import terrains

    c = terrains.isaac_clip_course(LIB, clip_id)
    t = terrains.isaac_tile_course(LIB, tile, speeds=(0.8,))
    assert c.flight_height_m == pytest.approx(t.flight_height_m, abs=0.05), (
        f"clip course says {c.flight_height_m:+.3f} m of flight, tile course says "
        f"{t.flight_height_m:+.3f}. If the clip figure is larger by roughly a pelvis height "
        "(~0.79 m) the datum has reverted to the spawn GROUND; `traversed` measures a rise "
        "from the spawn PELVIS, so that makes every episode time out inside the goal band.")


def test_the_spawn_converts_joint_order_by_name():
    """A positional copy of the clip's joints must not be mistaken for the policy's order."""
    import terrains
    from deploy.tracker_session import convert_joint_order
    from runner import clip_spawn

    c = terrains.isaac_clip_course(LIB, 300)
    clip_names = list(c.params["clip_joint_names"])
    # The policy's order, stated here rather than loaded, so this test needs no ONNX artifact.
    policy_names = [
        "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
        "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
        "left_knee_joint", "right_knee_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_elbow_joint", "right_elbow_joint",
        "left_wrist_roll_joint", "right_wrist_roll_joint",
        "left_wrist_pitch_joint", "right_wrist_pitch_joint",
        "left_wrist_yaw_joint", "right_wrist_yaw_joint"]
    assert set(policy_names) == set(clip_names)
    n_moved = sum(a != b for a, b in zip(clip_names, policy_names))
    assert n_moved >= 20, (
        f"only {n_moved} of 29 positions differ between the clip and policy orders. This test "
        "exists because they differ in 27; if they now agree, the conversion is a no-op and "
        "this file is no longer testing anything.")

    sp = clip_spawn(c, policy_names)
    orig = np.asarray(c.params["clip_start_joint_pos"], dtype=np.float64)
    # Converting back must recover the clip's own vector exactly.
    back = convert_joint_order(sp.joint_pos_isaac, policy_names, clip_names)
    assert np.abs(orig - back).max() == 0.0
    # And the converted vector must NOT equal the positional copy, or nothing was reordered.
    assert np.abs(sp.joint_pos_isaac - orig).max() > 1e-6


def test_the_spawn_carries_the_velocity_half():
    """Isaac writes root AND joint velocities at reset; a clip starts mid-gait, not at rest."""
    import terrains
    from runner import clip_spawn

    c = terrains.isaac_clip_course(LIB, 300)
    sp = clip_spawn(c, list(c.params["clip_joint_names"]))
    assert sp.base_lin_vel_w is not None and sp.joint_vel_isaac is not None
    assert np.linalg.norm(sp.base_lin_vel_w) > 0.05, (
        "clip 300 starts at 0.298 m/s; a zero here means the velocity half was dropped, "
        "which places the robot at the right pose and dead still -- a different initial "
        "condition that looks correct in a viewer.")
    assert np.abs(sp.joint_vel_isaac).max() > 0.1


def test_clip_spawn_refuses_a_tile_course_and_a_wrong_joint_set():
    """Both refusals guard a silent, well-shaped, wrong pose."""
    import terrains
    from runner import clip_spawn

    tile = terrains.isaac_tile_course(LIB, 6, speeds=(0.8,))
    with pytest.raises(ValueError, match="no clip start state"):
        clip_spawn(tile, ["a"])
    clip = terrains.isaac_clip_course(LIB, 300)
    bad = list(clip.params["clip_joint_names"])[:-1] + ["not_a_joint"]
    with pytest.raises(ValueError, match="NAME SETS differ"):
        clip_spawn(clip, bad)
