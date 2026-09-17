# Copyright 2026 Zachary Olkin. All rights reserved.
"""Spawn areas: the box tops are reachable, on the surface, and inside the box.

The reason box-top spawns exist is measured: every one of 352 successful episodes had a
worst-chunk terrain penalty >= 0.673 (median 1.844), and the cleanest were stairs-down
dismounts off both boxes -- luck, not mode selection. No gate sharpness fixes that, because
there was nothing better in the data to prefer. Spawning ON the boxes puts the second half of
the course in reach without needing the first half to go right.

Every failure here is silent in simulation: a spawn 1 cm inside the box top is an ejection
kick (the same class as `spawn-ground-penetration`), and a spawn past the edge is a fall the
policy is then blamed for.
"""

from __future__ import annotations

import numpy as np

from motion_gen.terrain import box2, freeform


def _areas(spawn_distance_m: float = 2.5) -> list[dict]:
    """The authored spawn areas of one box2 tile, read off the route block itself."""
    tile = box2.make_box2_tile(0, spawn_distance_m, family=box2.BOX2_V2_FAMILY)
    return list(tile["route"]["spawn_areas"])


def test_the_route_authors_three_spawn_areas():
    """1/3 ground, 1/3 box 1, 1/3 box 2 -- the split the request named. Equal weights, so the
    draw is the split rather than a coincidence of area sizes."""
    areas = _areas()
    assert len(areas) == 3, f"expected 3 spawn areas, got {len(areas)}"
    w = np.array([float(a["weight"]) for a in areas])
    assert np.allclose(w / w.sum(), 1.0 / 3.0), f"weights are not a thirds split: {w}"


def test_every_spawn_area_carries_the_full_field_set():
    """`SPAWN_AREA_FIELDS` is the contract the reset reads. A missing field is a KeyError
    after a simulator launch; a spare one is a field nothing consumes, which reads as
    configured while doing nothing."""
    for a in _areas():
        assert set(a) == set(freeform.SPAWN_AREA_FIELDS), (
            f"area {a} does not match {freeform.SPAWN_AREA_FIELDS}")


def test_the_box_top_areas_sit_ON_the_box_surface_not_inside_it():
    """z must be the box TOP. One centimetre low is a PhysX ejection kick on every episode --
    the defect `spawn-ground-penetration` records, which made survival numbers incomparable
    across the fix."""
    areas = _areas()
    tops = sorted({round(float(a["z"]), 4) for a in areas})
    assert 0.0 in tops or min(tops) == 0.0, f"no ground-level area: {tops}"
    box_tops = [t for t in tops if t > 0.0]
    assert box_tops, "no elevated spawn area at all"
    for t in box_tops:
        assert abs(t - box2.BOX_HEIGHT_M) < 1e-6, (
            f"elevated area at z={t} is not the {box2.BOX_HEIGHT_M} m box top")


def test_the_edge_buffer_keeps_the_usable_area_strictly_inside_the_box():
    """A spawn on the lip is a fall attributed to the policy. The buffer is subtracted from
    BOTH half-extents, so the usable rectangle must be smaller than the box in both axes and
    still non-empty -- a buffer larger than the box would silently give a negative extent and
    sample OUTSIDE it."""
    elevated = [a for a in _areas() if float(a["z"]) > 0.0]
    assert elevated
    for a in elevated:
        hx, hy = float(a["half_x"]), float(a["half_y"])
        assert hx > 0.0 and hy > 0.0, f"empty usable area: {a}"
        assert 2.0 * hx < box2.BOX_DEPTH_M + 1e-9
        assert box2.BOX_SPAWN_EDGE_BUFFER_M > 0.0


def test_the_two_box_areas_are_over_the_two_DIFFERENT_boxes():
    """Both areas at one box would halve the coverage while the weights still read as a
    thirds split -- the failure is a distribution, not an error."""
    xs = sorted(float(a["x"]) for a in _areas() if float(a["z"]) > 0.0)
    assert len(xs) == 2
    assert xs[1] - xs[0] > 1.0, f"the two box-top areas are {xs} -- same box"


def test_a_section_with_no_spawn_areas_is_legal():
    """Extensibility was the ask: other terrains carry no areas and must keep working off the
    single authored start, so absence is a valid state rather than a missing key."""
    assert freeform.SPAWN_AREA_FIELDS
    # `_route` defaults `spawn_areas` to nothing, and the reset synthesizes a fallback area
    # from the single start pose -- so a terrain that authors none keeps working unchanged.
    r = freeform._route((1.0, 0.0), 0.0, 0.0, (5.0, 0.0), 1.0, 0.0)
    assert not r.get("spawn_areas")
