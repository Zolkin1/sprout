# Copyright 2026 Zachary Olkin. All rights reserved.

"""The RViz ribbon's arithmetic, without a ROS graph.

``build_plan_geometry`` is the whole reason the marker node can be trusted: it is the inverse
of the representation the generator emits, and if it is wrong the drawing is confidently wrong
in exactly the way someone would then go and debug in the controller. The properties pinned
here are the ones that mistake would break.

Imports the node module for the pure function only -- it pulls in rclpy at module scope, so
the whole file skips on a box without ROS (every dev machine that is not the robot).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANSFER = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_TRANSFER, os.path.join(_HERE, "..", "g1_control")):
    if p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

viz = pytest.importorskip("g1_control.fm_plan_viz_node")

sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
from deploy.quat_math_np import quat_from_axis_angle, yaw_quat  # noqa: E402
from deploy.window_np import world_to_window  # noqa: E402

BLOCKS = ["base_pose", "base_vel", "joint_pos"]
NJ = 29


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """A quaternion from intrinsic roll/pitch/yaw, built from the helpers that exist here."""
    from deploy.quat_math_np import quat_mul

    qx = quat_from_axis_angle(np.array([roll, 0.0, 0.0]))
    qy = quat_from_axis_angle(np.array([0.0, pitch, 0.0]))
    qz = quat_from_axis_angle(np.array([0.0, 0.0, yaw]))
    return quat_mul(quat_mul(qz, qy), qx)


def _window(ref_pos, ref_quat, anchor_pos, anchor_quat, joints=None):
    """Build window nodes the forward way, so the test inverts a REAL encoding."""
    n = len(ref_pos)
    joints = np.zeros((n, NJ)) if joints is None else joints
    return world_to_window(
        root_pos_w=np.asarray(anchor_pos, dtype=np.float64),
        root_quat_w=np.asarray(anchor_quat, dtype=np.float64),
        ref_pos_w=np.asarray(ref_pos, dtype=np.float64),
        ref_quat_w=np.asarray(ref_quat, dtype=np.float64),
        ref_lin_w=np.zeros((n, 3)),
        ref_ang_w=np.zeros((n, 3)),
        joint_pos=joints,
        joint_vel=np.zeros((n, NJ)),
        ee_pos_w=None,
        blocks=BLOCKS,
    )


def test_a_zero_position_block_puts_every_node_on_the_anchor():
    """The degenerate case, and the one that catches a dropped anchor translation."""
    anchor_pos = np.array([3.0, -2.0, 0.8])
    anchor_quat = np.array([0.0, 0.0, 0.0, 1.0])
    nodes = _window([anchor_pos] * 4, [anchor_quat] * 4, anchor_pos, anchor_quat)
    geom = viz.build_plan_geometry(
        nodes.reshape(-1), BLOCKS, nodes.shape[1], NJ, 0, anchor_pos, anchor_quat, None)
    assert np.allclose(geom["root_pos"], anchor_pos)


def test_the_lift_round_trips_the_forward_map():
    """Encode a known world path, decode it, and require the original back."""
    anchor_pos = np.array([1.0, 2.0, 0.75])
    anchor_quat = _rpy(0.0, 0.0, 0.6)
    ref_pos = np.array([[1.0 + 0.1 * i, 2.0 + 0.02 * i, 0.75] for i in range(8)])
    ref_quat = np.array([_rpy(0.0, 0.0, 0.6 + 0.01 * i) for i in range(8)])
    nodes = _window(ref_pos, ref_quat, anchor_pos, anchor_quat)

    geom = viz.build_plan_geometry(
        nodes.reshape(-1), BLOCKS, nodes.shape[1], NJ, 0, anchor_pos, anchor_quat, None)
    assert np.allclose(geom["root_pos"], ref_pos, atol=1e-9)
    # Quaternions come back sign-normalised, so compare up to sign.
    dots = np.abs(np.sum(geom["root_quat"] * ref_quat, axis=-1))
    assert np.allclose(dots, 1.0, atol=1e-9)


def test_the_anchor_contributes_heading_only():
    """THE property this drawing exists to expose.

    The window's frame is the anchor's YAW, never its roll or pitch. Lifting through the full
    rotation instead tilts the entire ribbon by however far the robot happens to be leaning --
    which looks like a generator that wants the robot to walk into the floor, and would send
    someone to debug the wrong half of the stack.
    """
    anchor_pos = np.zeros(3)
    upright = _rpy(0.0, 0.0, 0.0)
    leaning = _rpy(0.25, -0.15, 0.0)   # roll and pitch, NO yaw
    ref_pos = np.array([[float(i) * 0.1, 0.0, 0.8] for i in range(6)])
    ref_quat = np.array([upright] * 6)

    # Same world path, encoded against a leaning anchor: the encoding uses yaw only, so the
    # decoded path must be identical to the upright case.
    up = _window(ref_pos, ref_quat, anchor_pos, upright)
    lean = _window(ref_pos, ref_quat, anchor_pos, leaning)
    g_up = viz.build_plan_geometry(
        up.reshape(-1), BLOCKS, up.shape[1], NJ, 0, anchor_pos, upright, None)
    g_lean = viz.build_plan_geometry(
        lean.reshape(-1), BLOCKS, lean.shape[1], NJ, 0, anchor_pos, leaning, None)
    assert np.allclose(g_up["root_pos"], g_lean["root_pos"], atol=1e-9)
    assert np.allclose(g_lean["root_pos"], ref_pos, atol=1e-9)

    # And a yaw-only anchor DOES rotate it -- the same test in the other direction, so a lift
    # that ignored the anchor entirely would not pass by accident.
    yawed = _rpy(0.0, 0.0, 0.9)
    assert not np.allclose(yaw_quat(yawed), upright)
    y = _window(ref_pos, ref_quat, anchor_pos, yawed)
    g_y = viz.build_plan_geometry(
        y.reshape(-1), BLOCKS, y.shape[1], NJ, 0, anchor_pos, yawed, None)
    assert np.allclose(g_y["root_pos"], ref_pos, atol=1e-9)
    # ... and that the encoded block is genuinely different, i.e. the agreement above is the
    # lift undoing the rotation rather than there being no rotation to undo.
    assert not np.allclose(y, up)


def test_a_window_without_base_pose_draws_nothing_rather_than_guessing():
    """A joints-only window has no world coordinate at all; returning {} is the honest answer."""
    nodes = np.zeros((4, NJ))
    geom = viz.build_plan_geometry(
        nodes.reshape(-1), ["joint_pos"], NJ, NJ, 0, np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), None)
    assert geom == {}


def test_an_empty_trajectory_is_not_a_crash():
    """A plan message can arrive before the generator has produced anything."""
    assert viz.build_plan_geometry(
        np.zeros(0), BLOCKS, 44, NJ, 0, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), None) == {}


def test_feet_are_detected_by_name():
    """The foot/other split drives the colour, and is the only place a name is interpreted."""
    assert viz._is_foot("left_ankle_roll_link")
    assert viz._is_foot("right_foot")
    assert not viz._is_foot("left_wrist_yaw_link")
