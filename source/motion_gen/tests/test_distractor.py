# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for distractor-box placement geometry (occupancy heatmap -> cold-cell box placement)."""

from __future__ import annotations

import numpy as np

from motion_gen.terrain.distractor import (
    distractor_layout,
    occupancy_grid,
    place_boxes,
)


def test_occupancy_grid_marks_point_and_dilates():
    origin = np.array([0.0, 0.0])
    cell, shape = 0.1, (11, 11)
    pts = np.array([[0.5, 0.5]])                       # -> center cell (5, 5)
    g0 = occupancy_grid(pts, origin, cell, shape, radius=0.0)
    assert g0[5, 5] and g0.sum() == 1
    g1 = occupancy_grid(pts, origin, cell, shape, radius=0.2)   # ~2-cell disk
    assert g1[5, 5] and g1[5, 7] and g1[3, 5]
    assert not g1[0, 0]                                 # far corner stays clear


def test_place_boxes_lands_in_cold_band():
    shape, cell, origin = (20, 20), 0.1, np.array([0.0, 0.0])
    cold = np.zeros(shape, dtype=bool)
    cold[:, 12:] = True                                # cold band on the +y side (y >= 1.2 m)
    boxes = place_boxes(cold, box_xy=(0.3, 0.3), cell=cell, origin=origin, n_boxes=1, min_sep=0.5)
    assert len(boxes) == 1
    _bx, by = boxes[0]
    assert by >= 1.4 - 1e-9                             # eroded by the box half-extent (2 cells)


def test_place_boxes_none_when_no_room():
    cold = np.zeros((10, 10), dtype=bool)              # nothing cold
    assert place_boxes(cold, (0.3, 0.3), 0.1, np.array([0.0, 0.0]), 3, 0.5) == []


def _straight_walk(T=40, length=4.0, foot_offset=0.1):
    x = np.linspace(0.0, length, T)
    pelvis = np.stack([x, np.zeros(T)], 1)
    lfoot = np.stack([x, np.full(T, foot_offset)], 1)
    rfoot = np.stack([x, np.full(T, -foot_offset)], 1)
    body = np.concatenate([pelvis, lfoot, rfoot], 0)
    return body, pelvis


def _cell(bx, by, origin, cell=0.1):
    return int(round((bx - origin[0]) / cell)), int(round((by - origin[1]) / cell))


def test_layout_straight_walk_box_off_path_but_in_scan():
    body, pelvis = _straight_walk()
    boxes, (origin, shape, heat, cold) = distractor_layout(
        [body], [pelvis], cell=0.1, tile_margin=2.0, robot_radius=0.25,
        scan_fwd_offset=0.75, scan_reach=0.9, box_xy=(0.3, 0.3), n_boxes=1, min_sep=0.5,
    )
    assert len(boxes) == 1
    bx, by = boxes[0]
    i, j = _cell(bx, by, origin)
    assert cold[i, j] and heat[i, j] == 0              # shared negative space AND perceivable
    assert bx > -0.5                                   # forward-biased: not placed behind the start


def test_layout_shared_negative_space_across_group():
    """Two clips (walk drifting +y / -y) share cold space; every box avoids BOTH footprints."""
    b1, p1 = _straight_walk(foot_offset=0.1)
    b1, p1 = b1 + np.array([0.0, 0.2]), p1 + np.array([0.0, 0.2])   # drift +y
    b2, p2 = _straight_walk(foot_offset=0.1)
    b2, p2 = b2 + np.array([0.0, -0.2]), p2 + np.array([0.0, -0.2])  # drift -y
    boxes, (origin, shape, heat, cold) = distractor_layout(
        [b1, b2], [p1, p2], cell=0.1, tile_margin=2.0, robot_radius=0.25,
        scan_fwd_offset=0.75, scan_reach=1.0, box_xy=(0.3, 0.3), n_boxes=2, min_sep=0.6,
    )
    assert 1 <= len(boxes) <= 2
    for bx, by in boxes:
        i, j = _cell(bx, by, origin)
        assert heat[i, j] == 0 and cold[i, j]          # occupied by NEITHER clip, perceivable
