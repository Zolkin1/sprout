# Copyright 2026 Zachary Olkin. All rights reserved.

"""The registration augmentation must move each node only along the axes its mask frees.

Numpy only -- no Isaac. These are the invariants that make the augmentation teach the anchor
mask rather than "every axis is world-fixed", and a sign slip in the terrain-frame rotation
would satisfy neither.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "rsl_rl"))


def _shift():
    """Import the helper without pulling in Isaac (the module launches an app at import)."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "scripts", "rsl_rl", "augment_registration.py")
    src = open(path).read()
    ns: dict = {"np": np}
    # Take only the two pure helpers; the rest of the module needs a SimulationApp.
    for fn in ("def quat_mul_xyzw", "def apply_free_axis_shift"):
        i = src.index(fn)
        j = src.index("\n\n\n", i)
        exec(compile(src[i:j], "<helper>", "exec"), ns)
    ns["_DOF_DOWNRANGE"], ns["_DOF_LATERAL"], ns["_DOF_YAW"] = 0, 1, 5
    exec(compile(src[src.index("def apply_free_axis_shift"):
                     src.index("\n\n\n", src.index("def apply_free_axis_shift"))],
                 "<helper>", "exec"), ns)
    return ns["apply_free_axis_shift"]


def _nodes(n=1, F=4):
    """One window of world-frame nodes with identity orientation and zero velocity."""
    ref = np.zeros((n, F, 13))
    ref[..., 0] = np.linspace(1.0, 2.0, F)          # downrange-ish spread
    ref[..., 1] = 0.3
    ref[..., 2] = 0.8
    ref[..., 6] = 1.0                                # quat w (xyzw)
    return ref


@pytest.mark.parametrize("tyaw", [0.0, 0.7])
def test_all_free_mask_is_a_no_op_relative_to_the_robot(tyaw):
    """Mask 35 (flat): the node takes the WHOLE displacement, so the robot-relative pose is
    unchanged -- augmenting flat ground must teach nothing, in any terrain frame."""
    f = _shift()
    ref = _nodes()
    robot = np.array([[0.0, 0.0, 0.9]])
    d_dr, d_lat, d_yaw = np.array([0.15]), np.array([-0.2]), np.array([0.1])
    mask = np.full((1, ref.shape[1]), 0b100011, np.uint8)   # downrange|lateral|yaw free
    out = f(ref, mask, robot, np.array([tyaw]), d_dr, d_lat, d_yaw)
    # robot moves by the same displacement; relative offset must be preserved
    c, s = np.cos(d_yaw[0]), np.sin(d_yaw[0])
    dx = d_dr[0] * np.cos(tyaw) - d_lat[0] * np.sin(tyaw)
    dy = d_dr[0] * np.sin(tyaw) + d_lat[0] * np.cos(tyaw)
    rel0 = ref[0, :, :2] - robot[0, :2]
    rel1 = out[0, :, :2] - (robot[0, :2] + np.array([dx, dy]))
    rot = np.stack([rel0[:, 0] * c - rel0[:, 1] * s, rel0[:, 0] * s + rel0[:, 1] * c], -1)
    assert np.allclose(rel1, rot, atol=1e-9)


def test_pinned_mask_moves_only_laterally():
    """Mask 2 (stairs/box): lateral is FREE so the node follows sideways, but downrange and
    yaw are PINNED -- that residual IS the correction the generator must learn."""
    f = _shift()
    ref = _nodes()
    robot = np.array([[0.0, 0.0, 0.9]])
    out = f(ref, np.full((1, ref.shape[1]), 0b000010, np.uint8), robot, np.array([0.0]),
            np.array([0.15]), np.array([-0.2]), np.array([0.1]))
    assert np.allclose(out[0, :, 0], ref[0, :, 0], atol=1e-9)          # downrange unmoved
    assert np.allclose(out[0, :, 1], ref[0, :, 1] - 0.2, atol=1e-9)    # lateral followed
    assert np.allclose(out[0, :, 3:7], ref[0, :, 3:7], atol=1e-9)      # yaw unmoved


def test_fully_pinned_mask_does_not_move_at_all():
    """Mask 0 (the un-regenerated distractors): nothing is free, so nothing moves."""
    f = _shift()
    ref = _nodes()
    out = f(ref, np.zeros((1, ref.shape[1]), np.uint8), np.array([[0.0, 0.0, 0.9]]),
            np.array([0.0]), np.array([0.15]), np.array([-0.2]), np.array([0.1]))
    assert np.allclose(out, ref, atol=1e-12)


def test_a_transition_window_is_mixed_node_by_node():
    """The point of doing this PER NODE: one window's flat approach takes the full shift
    while its terrain nodes take only the lateral part."""
    f = _shift()
    ref = _nodes(F=4)
    mask = np.array([[0b100011, 0b100011, 0b000010, 0b000010]], np.uint8)
    out = f(ref, mask, np.array([[0.0, 0.0, 0.9]]), np.array([0.0]),
            np.array([0.15]), np.array([0.0]), np.array([0.0]))
    assert np.allclose(out[0, :2, 0], ref[0, :2, 0] + 0.15, atol=1e-9)   # flat: moved
    assert np.allclose(out[0, 2:, 0], ref[0, 2:, 0], atol=1e-9)          # terrain: pinned
