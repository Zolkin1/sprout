# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the flat-clip -> slope rotation (motion_matching.slope_rotate).

Pure torch (no IsaacLab).  A rigid +y-axis pitch about the step-0 ground pivot must:
keep the grounded step-0 foot on the plane, tilt the pelvis orientation by the pitch,
turn a forward walk into a climb, rotate velocities, and leave joints untouched.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from motion_gen import features as F
from motion_gen.motion_matching.slope_rotate import (
    rotate_clip_to_slope,
    slope_pitch_for_climb,
    upright_slope_clip,
)

# Minimal channel layout: pelvis (pos+ori) + one foot (pos) + one joint.
POS_NAMES = [
    "pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
    "pelvis_link:ori_w", "pelvis_link:ori_x", "pelvis_link:ori_y", "pelvis_link:ori_z",
    "left_ankle_roll_link:pos_x", "left_ankle_roll_link:pos_y", "left_ankle_roll_link:pos_z",
    "joint:j0",
]
VEL_NAMES = [
    "pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
    "pelvis_link:ori_x", "pelvis_link:ori_y", "pelvis_link:ori_z",
    "joint:j0",
]


def _p(name):
    return POS_NAMES.index(name)


def _v(name):
    return VEL_NAMES.index(name)


def _synthetic():
    """A forward (+x) walk at constant pelvis height; step-0 foot at the pivot origin."""
    T = 6
    y = torch.zeros(T, len(POS_NAMES))
    x = torch.arange(T, dtype=torch.float32) * 0.1
    y[:, _p("pelvis_link:pos_x")] = x
    y[:, _p("pelvis_link:pos_z")] = 0.78
    y[:, _p("pelvis_link:ori_w")] = 1.0  # identity (wxyz-by-name)
    y[:, _p("left_ankle_roll_link:pos_x")] = x     # foot tracks pelvis x
    y[:, _p("left_ankle_roll_link:pos_z")] = 0.0   # on the ground
    y[:, _p("joint:j0")] = 0.5
    dy = torch.zeros(T, len(VEL_NAMES))
    dy[:, _v("pelvis_link:pos_x")] = 1.0  # forward linear velocity
    dy[:, _v("joint:j0")] = 0.3
    return y, dy


def _Ry(phi):
    c, s = math.cos(phi), math.sin(phi)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_step0_grounded_foot_stays_on_plane():
    """A foot at the pivot xy on z=0 is fixed by the rotation (stays grounded)."""
    y, dy = _synthetic()
    yr, _ = rotate_clip_to_slope(y, dy, POS_NAMES, VEL_NAMES,
                                 slope_pitch_for_climb(0.2), pivot_xy=(0.0, 0.0))
    foot0 = yr[0, [_p("left_ankle_roll_link:pos_x"), _p("left_ankle_roll_link:pos_y"),
                    _p("left_ankle_roll_link:pos_z")]]
    assert torch.allclose(foot0, torch.zeros(3), atol=1e-6)


def test_forward_walk_climbs():
    """Under the climb pitch a forward walk gains height monotonically."""
    y, dy = _synthetic()
    yr, dyr = rotate_clip_to_slope(y, dy, POS_NAMES, VEL_NAMES,
                                   slope_pitch_for_climb(0.25), pivot_xy=(0.0, 0.0))
    pz = yr[:, _p("pelvis_link:pos_z")]
    assert bool((pz[1:] - pz[:-1] > 0).all())          # strictly climbing
    assert float(pz[-1]) > float(pz[0])
    # forward linear velocity picks up a +z (climbing) component.
    assert float(dyr[0, _v("pelvis_link:pos_z")]) > 1e-3


def test_pelvis_orientation_tilted_by_pitch():
    """Identity pelvis orientation becomes the pitch quaternion (wxyz)."""
    y, dy = _synthetic()
    pitch = slope_pitch_for_climb(0.2)
    yr, _ = rotate_clip_to_slope(y, dy, POS_NAMES, VEL_NAMES, pitch, pivot_xy=(0.0, 0.0))
    h = 0.5 * pitch
    want_wxyz = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0])
    got_wxyz = yr[0, [_p("pelvis_link:ori_w"), _p("pelvis_link:ori_x"),
                      _p("pelvis_link:ori_y"), _p("pelvis_link:ori_z")]]
    assert torch.allclose(got_wxyz, want_wxyz, atol=1e-6)


def test_position_matches_Ry_about_pivot():
    """Pelvis positions equal R_y(pitch) about the ground pivot, elementwise."""
    y, dy = _synthetic()
    pitch = 0.2  # raw +y rotation
    yr, _ = rotate_clip_to_slope(y, dy, POS_NAMES, VEL_NAMES, pitch, pivot_xy=(0.0, 0.0))
    R = _Ry(pitch)
    pel = y[:, [_p("pelvis_link:pos_x"), _p("pelvis_link:pos_y"), _p("pelvis_link:pos_z")]]
    want = (R @ pel.T).T  # pivot at origin
    got = yr[:, [_p("pelvis_link:pos_x"), _p("pelvis_link:pos_y"), _p("pelvis_link:pos_z")]]
    assert torch.allclose(got, want, atol=1e-5)


def test_joints_untouched():
    """Joint channels (gait) pass through the rotation unchanged."""
    y, dy = _synthetic()
    yr, dyr = rotate_clip_to_slope(y, dy, POS_NAMES, VEL_NAMES,
                                   slope_pitch_for_climb(0.3), pivot_xy=(0.0, 0.0))
    assert torch.equal(yr[:, _p("joint:j0")], y[:, _p("joint:j0")])
    assert torch.equal(dyr[:, _v("joint:j0")], dy[:, _v("joint:j0")])


def test_does_not_mutate_inputs():
    """The rotation returns new tensors and leaves the inputs untouched."""
    y, dy = _synthetic()
    y0, dy0 = y.clone(), dy.clone()
    rotate_clip_to_slope(y, dy, POS_NAMES, VEL_NAMES, 0.15, pivot_xy=(0.0, 0.0))
    assert torch.equal(y, y0) and torch.equal(dy, dy0)

