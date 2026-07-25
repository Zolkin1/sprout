# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the Motion Bricks qpos layout + name-based joint reindex."""

from __future__ import annotations

import numpy as np
import pytest

from motion_gen.motion_matching.features import G1_JOINT_NAMES
from motion_gen.motionbricks.joint_map import reindex_to, split_qpos


def test_split_qpos_shapes_and_slices():
    q = np.arange(2 * 36, dtype=np.float32).reshape(2, 36)
    root_pos, root_quat, joints = split_qpos(q)
    assert root_pos.shape == (2, 3)
    assert root_quat.shape == (2, 4)
    assert joints.shape == (2, 29)
    np.testing.assert_array_equal(root_pos, q[:, 0:3])
    np.testing.assert_array_equal(root_quat, q[:, 3:7])
    np.testing.assert_array_equal(joints, q[:, 7:36])


def test_split_qpos_rejects_bad_shape():
    with pytest.raises(ValueError):
        split_qpos(np.zeros((4, 35)))


def test_reindex_identity_for_matching_order():
    idx = reindex_to(list(G1_JOINT_NAMES), list(G1_JOINT_NAMES))
    np.testing.assert_array_equal(idx, np.arange(len(G1_JOINT_NAMES)))


def test_reindex_default_target_is_g1():
    # Motion Bricks' g1.xml order == G1_JOINT_NAMES, so the reindex is the identity.
    idx = reindex_to(list(G1_JOINT_NAMES))
    np.testing.assert_array_equal(idx, np.arange(len(G1_JOINT_NAMES)))


def test_reindex_permutation_reorders_to_target():
    src = list(reversed(G1_JOINT_NAMES))
    idx = reindex_to(src, list(G1_JOINT_NAMES))
    reordered = [src[i] for i in idx]
    assert reordered == list(G1_JOINT_NAMES)


def test_reindex_missing_joint_raises():
    src = list(G1_JOINT_NAMES)[:-1] + ["not_a_real_joint"]
    with pytest.raises(ValueError):
        reindex_to(src, list(G1_JOINT_NAMES))
