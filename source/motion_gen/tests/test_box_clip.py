# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the box/terrain clip splice core (align+concat, geometry transform, mask stamp)."""

from __future__ import annotations

import math

import torch

from motion_gen.terrain.box_clip import (
    align_and_concat,
    build_anchor_mask_t,
    transform_boxes,
)


def _quat_yaw(theta: float) -> torch.Tensor:
    return torch.tensor([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])


def test_align_and_concat_continuity_and_heading():
    T0, T1 = 6, 6
    seg0_p = torch.stack([torch.tensor([k * 0.1, 0.0, 0.8]) for k in range(T0)])
    seg0_q = _quat_yaw(0.0).unsqueeze(0).repeat(T0, 1)
    seg0_j = torch.zeros(T0, 21)
    # seg1: faces -y and travels -y, parked far from seg0 -- alignment must fix BOTH.
    seg1_p = torch.stack([torch.tensor([5.0, -k * 0.1, 0.8]) for k in range(T1)])
    seg1_q = _quat_yaw(-math.pi / 2).unsqueeze(0).repeat(T1, 1)
    seg1_j = torch.full((T1, 21), 0.3)

    rp, rq, jt, bounds, tfs = align_and_concat(
        [(seg0_p, seg0_q, seg0_j), (seg1_p, seg1_q, seg1_j)])

    # seg0 kept whole; seg1 lost its duplicated seam frame
    assert bounds == [(0, T0), (T0, T0 + T1 - 1)]
    assert rp.shape[0] == T0 + T1 - 1
    # no teleport across the join
    steps = (rp[1:] - rp[:-1]).norm(dim=1)
    assert float(steps.max()) < 0.15
    # heading normalized to the approach: aligned seg1 keeps moving +x, not -y
    assert float(rp[T0, 0] - rp[T0 - 1, 0]) > 0.05
    assert abs(float(rp[-1, 1])) < 1e-4
    # joints pass through (seg1 frame 0 was the dropped seam)
    assert torch.allclose(jt[bounds[1][0]], seg1_j[1])
    assert len(tfs) == 2


def test_transform_boxes_rotate_translate():
    boxes = [{"pos": [1.0, 0.0, 0.2], "size": [0.4, 0.6, 0.4], "yaw": 0.0}]
    out = transform_boxes(boxes, _quat_yaw(math.pi / 2), torch.tensor([0.5, 0.0, 0.0]))
    p = out[0]["pos"]
    assert abs(p[0] - 0.5) < 1e-5 and abs(p[1] - 1.0) < 1e-5 and abs(p[2] - 0.2) < 1e-5
    assert abs(out[0]["yaw"] - math.pi / 2) < 1e-5
    assert out[0]["size"] == [0.4, 0.6, 0.4]        # extents unchanged


def test_build_anchor_mask_t():
    m = build_anchor_mask_t(10, [(0, 4), (4, 7), (7, 10)], [35, 2, 35])
    assert m.dtype == torch.uint8
    assert m.tolist() == [35, 35, 35, 35, 2, 2, 2, 35, 35, 35]


def test_yaw_normalize_axis_aligns_box_rigidly():
    import numpy as np

    from motion_gen.terrain.box_clip import _quat_yaw_wxyz, yaw_normalize
    T = 4
    rp = torch.stack([torch.tensor([1.0 + k * 0.1, 0.3, 0.8]) for k in range(T)])
    rq = _quat_yaw_wxyz(math.radians(-13)).unsqueeze(0).repeat(T, 1)
    boxes = [{"pos": [1.5, 0.5, 0.2], "size": [0.8, 0.45, 0.4], "yaw": math.radians(78)}]

    rp2, rq2, b2 = yaw_normalize(rp, rq, boxes, _quat_yaw_wxyz(math.radians(78)))
    # box becomes axis-aligned (its arbitrary approach-heading yaw removed)
    assert abs(b2[0]["yaw"]) < 1e-5
    # rotation is about the origin -> rigid (norms preserved for clip + box)
    assert torch.allclose(rp.norm(dim=1), rp2.norm(dim=1), atol=1e-5)
    assert abs(np.linalg.norm(b2[0]["pos"][:2]) - np.linalg.norm([1.5, 0.5])) < 1e-5


def test_yaw_normalize_facing_cancels_the_heading_not_the_box_yaw():
    """A solved BOX carries yaw=0 whichever way the robot walked, so aligning by it is a no-op.

    That is why the box families need ``align="facing"``: without it the interaction keeps its raw source
    heading (~-91 deg for every jump reference) and travels along world y, while the tile is laid out
    along world x -- transposing the tile's axes and handing the anchor mask's free-y the takeoff
    distance instead of the lateral offset.
    """
    from motion_gen.terrain.box_clip import _quat_yaw_wxyz, yaw_normalize
    T = 4
    facing = math.radians(-90)         # the library's jump references sit within ~4 deg of this
    rp = torch.stack([torch.tensor([0.0, -0.2 * k, 0.8]) for k in range(T)])   # walks along -y
    rq = _quat_yaw_wxyz(facing).unsqueeze(0).repeat(T, 1)
    boxes = [{"pos": [0.0, -0.6, 0.2], "size": [0.8, 0.45, 0.4], "yaw": 0.0}]  # ahead of the robot, in -y

    same, _, _ = yaw_normalize(rp, rq, boxes, rq[0])                            # align="box" -> no-op
    assert torch.allclose(same, rp, atol=1e-6)

    rot, rq2, b2 = yaw_normalize(rp, rq, boxes, rq[0], align="facing")
    assert abs(rot[-1, 0] - 0.6) < 1e-4 and abs(rot[-1, 1]) < 1e-4              # travel is now +x
    assert b2[0]["pos"][0] > 0.5 and abs(b2[0]["pos"][1]) < 1e-4                # the box came with it
    assert abs(abs(b2[0]["yaw"]) - abs(facing)) < 1e-6                          # ...rotated a quarter turn


def test_yaw_normalize_rejects_an_unknown_align():
    import pytest

    from motion_gen.terrain.box_clip import _quat_yaw_wxyz, yaw_normalize
    rp, rq = torch.zeros(2, 3), _quat_yaw_wxyz(0.0).unsqueeze(0).repeat(2, 1)
    with pytest.raises(ValueError, match="align"):
        yaw_normalize(rp, rq, [], rq[0], align="tread")
