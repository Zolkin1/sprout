# Copyright 2026 Zachary Olkin. All rights reserved.

"""Distractor-box placement over flat clips.

Put a box in the shared NEGATIVE space of a group of flat clips -- the cells their swept
footprints leave empty -- while still inside perception range of the path, so the box shows up in
the height scan but the motion never touches it (breaking the "box in view => jump on it" reflex;
see the terrain plan). The negative TYPE is EMERGENT from the motion + this one placement rule:
straight walks leave the sides cold (walk-past), decelerate-to-stop clips leave the space ahead
cold (stop-before), turns leave the arc apex cold (turn-around).

Clips are stored clip-world (each starts at the origin facing +x), so their footprints are already
aligned to a common frame and directly comparable -- the heatmap over a clip group is the shared
occupancy, and cold+perceivable cells are the shared negative space in view.

Pure numpy + scipy geometry; the CLI (``scripts/rsl_rl/generate_distractor_clips.py``) extracts
foot/pelvis xy from a flat clip library and composes these. All scan-coupled sizes (``scan_reach``,
the placement band) are caller knobs -- the scan pattern will change, so nothing is baked in here.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def grid_frame(points_xy: np.ndarray, cell: float, margin: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Axis-aligned grid covering ``points_xy`` [P,2] padded by ``margin`` (m).

    Returns ``(origin[2], shape=(H, W))`` where ``origin`` is the world xy of cell (0,0)'s center,
    ``H`` indexes x (forward), ``W`` indexes y (lateral), at resolution ``cell`` (m).
    """
    lo = points_xy.min(0) - margin
    hi = points_xy.max(0) + margin
    shape = (int(np.ceil((hi[0] - lo[0]) / cell)) + 1, int(np.ceil((hi[1] - lo[1]) / cell)) + 1)
    return lo.astype(np.float64), shape


def _disk(radius_cells: int) -> np.ndarray:
    """Boolean disk structuring element of radius ``radius_cells`` (>=0)."""
    r = int(radius_cells)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def occupancy_grid(points_xy: np.ndarray, origin: np.ndarray, cell: float,
                   shape: tuple[int, int], radius: float) -> np.ndarray:
    """[H,W] bool: cells within ``radius`` (m) of ANY point in ``points_xy`` [P,2].

    Rasterizes the points then dilates by a disk of ``radius`` -- one primitive serves both a
    swept-footprint occupancy (feet+pelvis over all frames, dilated by the robot half-width) and a
    perception mask (the pelvis path dilated by the scan reach).
    """
    grid = np.zeros(shape, dtype=bool)
    ij = np.floor((np.asarray(points_xy, dtype=np.float64) - origin) / cell + 0.5).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    ij = ij[ok]
    if len(ij):
        grid[ij[:, 0], ij[:, 1]] = True
    rad = int(round(radius / cell))
    if rad > 0:
        grid = ndimage.binary_dilation(grid, structure=_disk(rad))
    return grid


def _heading_tangent(pelvis_xy: np.ndarray) -> np.ndarray:
    """Unit forward direction [T,2] per frame, from the pelvis-path tangent (finite diff).

    Falls back to the previous valid tangent where the robot is stationary (near-zero step), and to
    +x if the whole path is a point -- so a standing/stop clip still has a sensible facing.
    """
    d = np.diff(pelvis_xy, axis=0, append=pelvis_xy[-1:])          # [T,2]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    unit = np.divide(d, n, out=np.zeros_like(d), where=n > 1e-6)
    last = np.array([1.0, 0.0])
    for t in range(len(unit)):
        if n[t, 0] <= 1e-6:
            unit[t] = last
        else:
            last = unit[t]
    return unit


def perceivable_grid(pelvis_xy: np.ndarray, origin: np.ndarray, cell: float, shape: tuple[int, int],
                     fwd_offset: float, reach: float) -> np.ndarray:
    """[H,W] bool: cells the FORWARD scan can see along the path.

    The scan is offset ``fwd_offset`` ahead of the pelvis along the heading (path tangent); a cell is
    perceivable if it falls within ``reach`` of that forward scan-center at some frame. Forward-biased
    (unlike a symmetric disk) so boxes BEHIND the robot -- never scanned -- aren't proposed. ``fwd_offset``
    / ``reach`` are knobs approximating the height-scan footprint (the scan pattern will change).
    """
    heading = _heading_tangent(pelvis_xy)
    scan_center = pelvis_xy + fwd_offset * heading                # shift each path point forward
    return occupancy_grid(scan_center, origin, cell, shape, reach)


def cold_perceivable(heatmap: np.ndarray, perceivable: np.ndarray, max_count: int = 0) -> np.ndarray:
    """[H,W] bool: perceivable cells whose occupancy count <= ``max_count`` (shared negative space)."""
    return perceivable & (heatmap <= max_count)


def place_boxes(cold: np.ndarray, box_xy: tuple[float, float], cell: float, origin: np.ndarray,
                n_boxes: int, min_sep: float, order: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Place up to ``n_boxes`` axis-aligned boxes of full extent ``box_xy`` (m) into ``cold`` cells.

    A box may sit where its WHOLE footprint lands on cold cells (erode ``cold`` by the box
    half-extent), and centers are greedily picked spread by at least ``min_sep`` (m). ``order`` (an
    optional permutation of the candidate list, for run-to-run variety without a live RNG) reorders
    the greedy pick; default = raster order. Returns ``(x, y)`` box-center world positions.
    """
    half_i = int(np.ceil(box_xy[0] / 2.0 / cell))
    half_j = int(np.ceil(box_xy[1] / 2.0 / cell))
    se = np.ones((2 * half_i + 1, 2 * half_j + 1), dtype=bool)
    fit = ndimage.binary_erosion(cold, structure=se, border_value=0)   # centers where the box fits
    cand = np.argwhere(fit)
    if order is not None and len(cand):
        cand = cand[np.asarray(order)[: len(cand)]]
    sep2 = (min_sep / cell) ** 2
    chosen: list[tuple[int, int]] = []
    for i, j in cand:
        if all((i - ci) ** 2 + (j - cj) ** 2 >= sep2 for ci, cj in chosen):
            chosen.append((int(i), int(j)))
            if len(chosen) >= n_boxes:
                break
    return [(float(origin[0] + i * cell), float(origin[1] + j * cell)) for i, j in chosen]


def distractor_layout(body_pts: list[np.ndarray], pelvis_paths: list[np.ndarray], *,
                      cell: float, tile_margin: float, robot_radius: float,
                      scan_fwd_offset: float, scan_reach: float,
                      box_xy: tuple[float, float], n_boxes: int, min_sep: float,
                      max_count: int = 0, order: np.ndarray | None = None):
    """End-to-end for one clip GROUP -> box (x,y) centers in the shared cold+perceivable space.

    ``body_pts[c]`` = [P,2] feet+pelvis xy over clip ``c`` (occupancy); ``pelvis_paths[c]`` = [T,2]
    pelvis xy (heading + perception). Occupancy = every clip's footprint dilated by ``robot_radius``;
    perception = the forward scan (``scan_fwd_offset`` ahead, ``scan_reach`` wide) unioned over frames
    and clips. A box goes in cells that are cold for EVERY clip yet perceivable by SOME frame. Grid
    spans the body points + forward scan-centers + ``tile_margin``. Returns ``(boxes, debug)`` where
    ``debug = (origin, shape, heatmap, cold)`` for viz/tests.
    """
    scan_centers = [pp + scan_fwd_offset * _heading_tangent(pp) for pp in pelvis_paths]
    allpts = np.concatenate(list(body_pts) + scan_centers, axis=0)
    origin, shape = grid_frame(allpts, cell, tile_margin)
    heat = np.zeros(shape, dtype=np.int32)
    for bp in body_pts:
        heat += occupancy_grid(bp, origin, cell, shape, robot_radius).astype(np.int32)
    perc = np.zeros(shape, dtype=bool)
    for pp in pelvis_paths:
        perc |= perceivable_grid(pp, origin, cell, shape, scan_fwd_offset, scan_reach)
    cold = cold_perceivable(heat, perc, max_count)
    boxes = place_boxes(cold, box_xy, cell, origin, n_boxes, min_sep, order=order)
    return boxes, (origin, shape, heat, cold)
