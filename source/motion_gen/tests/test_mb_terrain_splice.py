# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the Motion-Bricks terrain splice: weld vs ease, boundary matching, footprint recentering."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motion_gen.motionbricks.terrain_splice import (
    footprint_recenter,
    match_arrival,
    match_overlap,
    overlays_for,
    splice_terrain,
)

_DT = 0.02


def _seg(T: int, x0: float = 0.0, z: float = 0.8, joints: float = 0.0, vx: float = 0.6, yaw: float = 0.0):
    """A straight-line segment at constant height with a constant joint vector."""
    P = torch.zeros(T, 3)
    P[:, 0] = x0 + torch.arange(T, dtype=torch.float32) * vx * _DT
    P[:, 2] = z
    Q = torch.zeros(T, 4)
    Q[:, 0] = np.cos(yaw / 2.0)
    Q[:, 3] = np.sin(yaw / 2.0)
    J = torch.full((T, 29), float(joints))
    return P, Q, J


def _terrain(T: int = 40, z0: float = 1.0, dz: float = 0.15, joints: float = 1.0):
    """A rising terrain replay, offset in xy and yawed so the splice has real work to do."""
    P, Q, J = _seg(T, x0=5.0, z=z0, joints=joints, vx=0.5, yaw=0.6)
    P = P.clone()
    P[:, 1] = 2.0
    P[:, 2] = z0 + torch.arange(T, dtype=torch.float32) / max(1, T - 1) * dz
    return P, Q, J


def test_weld_makes_the_seams_exactly_continuous():
    terrain = _terrain()
    approach = _seg(30, z=0.8, joints=0.0)
    exit_seg = _seg(30, z=0.8, joints=2.0)
    Wp, Wq, Wj, bounds, diag = splice_terrain(
        terrain, approach, exit_seg, approach_surf_z=0.15, exit_surf_z=0.90, blend_n=12,
        approach_seam="weld", exit_seam="weld", dt=_DT)
    tP = terrain[0]
    na, nt = bounds[0][1], bounds[1][1] - bounds[1][0]
    assert bounds == [(0, 29), (29, 29 + nt), (29 + nt, 29 + nt + 29)]
    assert Wp.shape[0] == 29 + nt + 29
    # The terrain segment is untouched, and the welded seam is continuous: the frame before the boundary is
    # one walking step away from it, not a jump (the duplicated boundary frame was dropped).
    np.testing.assert_allclose(Wp[na:na + nt].numpy(), tP.numpy(), atol=1e-5)
    assert torch.linalg.norm(Wp[na] - Wp[na - 1]).item() < 0.05
    assert torch.linalg.norm(Wp[na + nt] - Wp[na + nt - 1]).item() < 0.05
    # A weld reports how far off its surface the boundary pose put the segment.
    assert diag["approach_surface_err"] == pytest.approx(tP[0, 2].item() - 0.8 - 0.15, abs=1e-5)
    assert diag["approach_seam"] == "weld" and diag["exit_seam"] == "weld"


def test_ease_grounds_each_segment_on_its_own_surface():
    terrain = _terrain()
    approach = _seg(40, z=0.8, joints=0.0)
    exit_seg = _seg(40, z=0.8, joints=2.0)
    Wp, _Wq, _Wj, bounds, diag = splice_terrain(
        terrain, approach, exit_seg, approach_surf_z=0.15, exit_surf_z=0.90, blend_n=12,
        approach_seam="ease", exit_seam="ease", dt=_DT)
    assert diag["approach_surface_err"] == 0.0 and diag["exit_surface_err"] == 0.0
    # Away from the eased seam each segment keeps its own height + surface offset.
    assert Wp[0, 2].item() == pytest.approx(0.8 + 0.15, abs=1e-5)
    assert Wp[-1, 2].item() == pytest.approx(0.8 + 0.90, abs=1e-5)
    # The ease pulls the frames adjacent to the terrain onto its boundary.
    na, nt = bounds[0][1], bounds[1][1] - bounds[1][0]
    assert abs(Wp[na - 1, 2].item() - Wp[na, 2].item()) < abs(Wp[0, 2].item() - Wp[na, 2].item())
    assert abs(Wp[na + nt, 2].item() - Wp[na + nt - 1, 2].item()) < 0.05


def test_tile_frame_segment_is_not_lifted_again():
    terrain = _terrain()
    exit_seg = _seg(30, z=1.15, joints=1.0)              # already in tile coordinates, on the 0.90 surface
    Wp, _Wq, _Wj, bounds, _diag = splice_terrain(
        terrain, None, exit_seg, approach_surf_z=0.15, exit_surf_z=0.90, blend_n=12,
        exit_seam="ease", exit_frame="tile", dt=_DT)
    assert bounds[0] == (0, 0)                           # no approach slot
    assert Wp[-1, 2].item() == pytest.approx(1.15, abs=1e-5)


def test_omitting_both_sides_returns_the_terrain_unchanged():
    terrain = _terrain()
    Wp, Wq, Wj, bounds, diag = splice_terrain(terrain, None, None, approach_surf_z=0.0, exit_surf_z=0.0,
                                              blend_n=12, dt=_DT)
    np.testing.assert_allclose(Wp.numpy(), terrain[0].numpy())
    np.testing.assert_allclose(Wj.numpy(), terrain[2].numpy())
    assert bounds[0] == (0, 0) and bounds[2] == (Wp.shape[0], Wp.shape[0])
    assert "approach_joint_res" not in diag


def test_invalid_seam_or_frame_is_rejected():
    terrain = _terrain()
    with pytest.raises(ValueError, match="approach_seam"):
        splice_terrain(terrain, None, None, approach_surf_z=0, exit_surf_z=0, blend_n=0,
                       approach_seam="squish")
    with pytest.raises(ValueError, match="exit_frame"):
        splice_terrain(terrain, None, None, approach_surf_z=0, exit_surf_z=0, blend_n=0,
                       exit_frame="world")


def test_match_arrival_finds_the_frame_closest_to_the_boundary_pose():
    T = 40
    P, Q, J = _seg(T, z=0.8)
    J = J.clone()
    J[25] = 1.0                                          # frame 25 matches the target joints exactly
    tgt = torch.ones(29)
    assert match_arrival((P, Q, J), tgt, z_target=0.8, search=20) == 25
    assert match_arrival((P, Q, J), tgt, z_target=0.8, search=1) == T - 1     # search=1 -> the last frame


def test_match_arrival_weighs_height_against_joints():
    T = 20
    P, Q, J = _seg(T, z=0.8)
    P = P.clone()
    P[10, 2] = 0.5                                       # a 0.3 m height error at frame 10 must lose
    J = J.clone()
    J[10] = 1.0
    tgt = torch.ones(29)
    assert match_arrival((P, Q, J), tgt, z_target=0.8, search=20, w_z=100.0) != 10


def test_match_overlap_skips_the_re_predicted_seed_frames():
    P, Q, J = _seg(20, x0=3.0, z=1.1, joints=0.5)
    j = 4
    tgt_p, tgt_q, tgt_j = P[j], Q[j], J[j]
    assert match_overlap((P, Q, J), tgt_p, tgt_q, tgt_j, search=12) == j


def test_footprint_recenter_uses_the_geometry_as_the_datum():
    boxes = [{"pos": [2.0, 0.0, 0.1], "size": [1.0, 1.4, 0.2], "yaw": 0.0}]
    Wp = torch.zeros(5, 3)
    Wp[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out, corner, size_x, size_y = footprint_recenter(Wp, boxes, margin=1.0)
    # datum = min(motion, box) - margin  ->  x: min(1.0, 1.5) - 1 = 0.0 ; y: -0.7 - 1 = -1.7
    assert corner[0] == pytest.approx(0.0) and corner[1] == pytest.approx(-1.7)
    assert out[:, 0].min().item() == pytest.approx(1.0)
    assert size_x == pytest.approx((5.0 - 1.0) + 2.0)
    assert size_y == pytest.approx(1.4 + 2.0)
    ov = overlays_for(boxes, corner)
    assert ov[0]["pos_xy"] == pytest.approx([2.0, 1.7])
    assert ov[0]["base_z"] == pytest.approx(0.0) and ov[0]["walkable_top"]


def test_overlays_for_passes_support_variant_through():
    """Support-structure keys authored on a box reach the manifest verbatim; absent keys stay absent."""
    boxes = [{"pos": [2.0, 0.0, 0.1], "size": [1.0, 1.4, 0.2], "yaw": 0.0,
              "variant": "floating", "slab_thickness": 0.05},
             {"pos": [4.0, 0.0, 0.1], "size": [1.0, 1.4, 0.2], "yaw": 0.0}]
    ov = overlays_for(boxes, (0.0, 0.0))
    assert ov[0]["variant"] == "floating" and ov[0]["slab_thickness"] == pytest.approx(0.05)
    assert "variant" not in ov[1] and "slab_thickness" not in ov[1]
