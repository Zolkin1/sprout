# Copyright 2026 Zachary Olkin. All rights reserved.

"""The foot envelope must be an over-estimate, never an under-estimate.

Every terrain narrowing is sized by this number. Too large and a tile is narrowed less than
it could be, which costs variety. Too small and the robot walks off the edge of a surface
that was fitted to a footprint the motion does not actually stay inside -- so the failures
are not symmetric, and the tests are written accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from motion_gen.terrain.footprint import (EGO_ANCHOR_ALLOWANCE_M, FOOT_BOX_HALF,
                                          foot_corners, tile_footprint)


def _quat_z(angle: float) -> np.ndarray:
    """Unit quaternion about +z, w first."""
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def test_the_foot_box_matches_the_clip_generator():
    """These constants are duplicated from ``clip_output``; they must not drift apart.

    ``contact`` in every clip library is derived from that copy, so a divergence would mean
    the terrain is sized against a different foot than the one the contacts describe.
    """
    from motion_gen import clip_output
    assert FOOT_BOX_HALF == clip_output._FOOT_BOX_HALF
    from motion_gen.terrain.footprint import FOOT_BOX_CENTER
    assert FOOT_BOX_CENTER == clip_output._FOOT_BOX_CENTER


def test_an_axis_aligned_foot_spans_its_own_half_width():
    """At zero yaw the corners are exactly the box, so this is the easy case to get right."""
    corners = foot_corners(np.zeros((1, 3)), _quat_z(0.0)[None])
    assert corners.shape == (1, 8, 3)
    assert corners[0, :, 1].max() == pytest.approx(FOOT_BOX_HALF[1])
    assert corners[0, :, 1].min() == pytest.approx(-FOOT_BOX_HALF[1])


def test_a_yawed_foot_reaches_further_than_its_half_width():
    """The reason ankle_roll_link:pos_y alone under-reports the envelope by 0.10-0.13 m.

    A foot is 0.203 m long and 0.065 m wide, and its box is offset 0.036 m FORWARD of the
    ankle because the toe is longer than the heel. Yawed 45 degrees the far toe corner swings
    to 0.1204 m off the ankle -- 3.7x the 0.033 m half-width -- so a tread sized to the
    half-width is too narrow by nearly 9 cm a side.

    0.1204 = (0.03592 + 0.101555 + 0.032735) / sqrt(2), i.e. the toe corner rotated. The
    forward offset is easy to forget and makes the answer 27% larger than the box alone.
    """
    straight = foot_corners(np.zeros((1, 3)), _quat_z(0.0)[None])[0, :, 1].max()
    yawed = foot_corners(np.zeros((1, 3)), _quat_z(np.pi / 4)[None])[0, :, 1].max()
    assert yawed > straight * 3.0, "a 45 deg yaw must widen the lateral envelope substantially"
    assert yawed == pytest.approx(0.12036, abs=5e-5)


def test_the_envelope_adds_the_ego_anchor_allowance_on_both_sides():
    """The runtime re-anchors the clip laterally, so the baked footprint is not the extent."""
    pos = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    quat = np.repeat(_quat_z(0.0)[None], 2, axis=0)
    fp = tile_footprint(pos, quat, terrain_id=7, n_clips=2)
    assert fp.y_max == pytest.approx(0.5 + FOOT_BOX_HALF[1])
    assert fp.y_min == pytest.approx(-0.5 - FOOT_BOX_HALF[1])
    assert fp.y_max_sized == pytest.approx(fp.y_max + EGO_ANCHOR_ALLOWANCE_M)
    assert fp.y_min_sized == pytest.approx(fp.y_min - EGO_ANCHOR_ALLOWANCE_M)
    assert fp.width_sized == pytest.approx(fp.width + 2 * EGO_ANCHOR_ALLOWANCE_M)


def test_the_sized_envelope_is_never_narrower_than_the_raw_one():
    """The property that makes it safe to narrow terrain to this number."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos = rng.normal(scale=0.4, size=(50, 3))
        ang = rng.uniform(-np.pi, np.pi, size=50)
        quat = np.stack([np.cos(ang / 2), np.zeros(50), np.zeros(50), np.sin(ang / 2)], -1)
        fp = tile_footprint(pos, quat, terrain_id=0, n_clips=1)
        assert fp.y_min_sized < fp.y_min
        assert fp.y_max_sized > fp.y_max
        # and the raw envelope already contains every ankle, before the foot box
        assert fp.y_min <= pos[:, 1].min()
        assert fp.y_max >= pos[:, 1].max()


def test_no_samples_is_an_error_not_an_empty_envelope():
    """An empty envelope reads as 'this tile needs no surface', the worst possible answer."""
    with pytest.raises(ValueError, match="no foot samples"):
        tile_footprint(np.zeros((0, 3)), np.zeros((0, 4)), terrain_id=3, n_clips=0)
