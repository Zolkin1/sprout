# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the jump-on BOX tiles-then-clips phase-A sampler + fixed tile geometry.

Locks the box-family reuse invariant (the box analog of the stair transition tiles): ``sample_box_tiles``
produces a tile population with contiguous ids, in-range source indices, variant weights honored, and
lateral offsets bounded to the corridor; ``box_tile_box`` authors ONE fixed box shared by all of a tile's
clips -- its near face sits ``approach_len`` down the flat run-up and its top is ``sz`` (base at z=0).

``box_tile_boxes`` adds the walkable TOP PLATFORM flush with that box, on the side away from the jump
face, so a mount can walk ON after landing and a dismount can walk IN before dropping off -- the box-tile
analog of a stair tile's approach / recovery platforms.
"""

from __future__ import annotations

import numpy as np
import pytest

from motion_gen.terrain.box_clip import box_tile_box, box_tile_boxes
from motion_gen.terrain.terrain_sampler import BoxTileSpec, sample_box_tiles


def test_ids_contiguous_and_count():
    tiles = sample_box_tiles(np.random.default_rng(0), n_tiles=5, n_sources=2,
                             variant_fracs=None, nav_width_range=(1.0, 1.4))
    assert len(tiles) == 5
    assert all(isinstance(t, BoxTileSpec) for t in tiles)
    assert [t.terrain_id for t in tiles] == list(range(5))       # contiguous 0..N-1


def test_source_idx_in_range():
    tiles = sample_box_tiles(np.random.default_rng(1), n_tiles=40, n_sources=3,
                             variant_fracs=None, nav_width_range=(1.0, 1.4))
    assert all(0 <= t.source_idx < 3 for t in tiles)


def test_single_source_always_index_zero():
    # A single played jump source -> every tile must reference it (index 0).
    tiles = sample_box_tiles(np.random.default_rng(2), n_tiles=8, n_sources=1,
                             variant_fracs=None, nav_width_range=(1.0, 1.4))
    assert all(t.source_idx == 0 for t in tiles)


def test_variant_fracs_none_is_all_solid():
    tiles = sample_box_tiles(np.random.default_rng(3), n_tiles=12, n_sources=1,
                             variant_fracs=None, nav_width_range=(1.0, 1.4))
    assert all(t.variant == "solid" for t in tiles)


def test_variant_fracs_forced_single_variant():
    # A single-variant weight dict must force every tile to that variant (aliases resolved upstream).
    tiles = sample_box_tiles(np.random.default_rng(4), n_tiles=12, n_sources=1,
                             variant_fracs={"floating_corner_legs": 1.0}, nav_width_range=(1.0, 1.4))
    assert all(t.variant == "floating_corner_legs" for t in tiles)


def test_nav_width_within_range():
    lo, hi = 1.0, 1.6
    tiles = sample_box_tiles(np.random.default_rng(5), n_tiles=50, n_sources=2,
                             variant_fracs=None, nav_width_range=(lo, hi))
    assert all(lo - 1e-9 <= t.nav_width <= hi + 1e-9 for t in tiles)


def test_y_offset_bounded_to_corridor():
    frac = 0.3
    tiles = sample_box_tiles(np.random.default_rng(6), n_tiles=100, n_sources=2,
                             variant_fracs=None, nav_width_range=(1.0, 1.4), y_offset_max_frac=frac)
    # |offset| <= frac * (nav / 2): the box stays inside its own corridor.
    assert all(abs(t.y_offset) <= frac * (t.nav_width / 2.0) + 1e-9 for t in tiles)


def test_zero_offset_frac_centers_box():
    tiles = sample_box_tiles(np.random.default_rng(7), n_tiles=20, n_sources=1,
                             variant_fracs=None, nav_width_range=(1.0, 1.4), y_offset_max_frac=0.0)
    assert all(abs(t.y_offset) < 1e-12 for t in tiles)


def test_footprint_scale_expand_only():
    # Default (1,1) -> no scaling.
    tiles = sample_box_tiles(np.random.default_rng(8), n_tiles=20, n_sources=1,
                             variant_fracs=None, nav_width_range=(1.0, 1.4))
    assert all(t.sx_scale == 1.0 and t.sy_scale == 1.0 for t in tiles)
    # A range expands (>= 1) and stays within [max(1,lo), hi]; a <1 lo is clamped to 1.
    tiles = sample_box_tiles(np.random.default_rng(9), n_tiles=60, n_sources=1,
                             variant_fracs=None, nav_width_range=(1.0, 1.4),
                             footprint_scale_range=(0.5, 1.8))
    assert all(1.0 <= t.sx_scale <= 1.8 and 1.0 <= t.sy_scale <= 1.8 for t in tiles)
    assert any(t.sx_scale > 1.01 for t in tiles)             # actually varies


def test_box_tile_box_geometry():
    sx, sy, sz = 0.8, 0.45, 0.47
    approach_len, yoff = 6.0, 0.12
    box = box_tile_box(box_size=(sx, sy, sz), approach_len=approach_len, y_offset=yoff)
    assert box["size"] == [sx, sy, sz]
    assert box["yaw"] == 0.0
    # center: near face at approach_len (=> x = approach_len + sx/2), lateral = y_offset, base at z=0.
    assert abs(box["pos"][0] - (approach_len + sx / 2.0)) < 1e-9
    assert abs(box["pos"][1] - yoff) < 1e-9
    assert abs(box["pos"][2] - sz / 2.0) < 1e-9
    # the near face sits exactly approach_len down the run-up; the top is the landing surface at z = sz.
    assert abs((box["pos"][0] - sx / 2.0) - approach_len) < 1e-9
    assert abs((box["pos"][2] + sz / 2.0) - sz) < 1e-9


def test_box_tile_box_default_centered():
    box = box_tile_box(box_size=(1.0, 0.5, 0.3), approach_len=4.0)
    assert box["pos"][1] == 0.0                                  # default y_offset -> corridor centerline


# ---- box_tile_boxes: the jump box + the walkable TOP PLATFORM flush with it ----------------------

_BOX = (0.8, 0.45, 0.47)


def _span_x(b: dict) -> tuple[float, float]:
    """``(x_min, x_max)`` of one authored box."""
    return b["pos"][0] - b["size"][0] / 2.0, b["pos"][0] + b["size"][0] / 2.0


def test_box_tile_boxes_without_a_platform_matches_box_tile_box():
    only = box_tile_boxes(box_size=_BOX, approach_len=6.0, y_offset=0.12)
    assert len(only) == 1
    assert only[0] == box_tile_box(box_size=_BOX, approach_len=6.0, y_offset=0.12)


def test_box_tile_boxes_platform_is_flush_and_adjacent():
    sx, sy, sz = _BOX
    for sign in (1.0, -1.0):
        boxes = box_tile_boxes(box_size=_BOX, approach_len=6.0, platform_len=1.5, platform_sign=sign)
        box, plat = boxes
        assert len(boxes) == 2
        assert box["size"] == [sx, sy, sz]                       # the jump footprint is untouched...
        assert plat["size"] == [1.5, sy, sz]                     # ...and the platform tops out with it
        assert abs((box["pos"][2] + sz / 2.0) - (plat["pos"][2] + plat["size"][2] / 2.0)) < 1e-9
        blo, bhi = _span_x(box)
        plo, phi = _span_x(plat)
        assert abs(min(blo, plo) - 6.0) < 1e-9                   # structure starts at approach_len
        if sign > 0:
            assert abs(plo - bhi) < 1e-9                         # platform butts onto the far (+x) face
        else:
            assert abs(phi - blo) < 1e-9                         # ...or onto the near (-x) face
        assert min(bhi, phi) - max(blo, plo) < 1e-9              # never overlapping


def test_box_tile_boxes_jump_box_is_index_zero():
    # place_episodic_on_tile anchors the played interaction onto boxes[0], so the platform must not be it.
    for sign in (1.0, -1.0):
        boxes = box_tile_boxes(box_size=_BOX, approach_len=3.0, platform_len=2.0, platform_sign=sign)
        assert boxes[0]["size"][0] == _BOX[0]
        assert boxes[1]["size"][0] == 2.0


def test_box_tile_boxes_nav_width_widens_both_expand_only():
    wide = box_tile_boxes(box_size=_BOX, approach_len=3.0, platform_len=1.0, nav_width=1.2)
    assert [b["size"][1] for b in wide] == [1.2, 1.2]
    # Expand-only: a corridor narrower than the solved box must not shrink the landing pad.
    narrow = box_tile_boxes(box_size=_BOX, approach_len=3.0, platform_len=1.0, nav_width=0.2)
    assert [b["size"][1] for b in narrow] == [_BOX[1], _BOX[1]]
    # Widening is symmetric about the centerline, so the anchor placement (a box-CENTER align) is unmoved.
    assert wide[0]["pos"][:2] == box_tile_boxes(box_size=_BOX, approach_len=3.0,
                                                platform_len=1.0)[0]["pos"][:2]


# ---- travel-frame box extents (what the tile must be authored from) -----------------------------

class _FakeSource:
    """Minimal stand-in for an ingested episodic MotionSource: a heading and a solved box."""

    def __init__(self, facing_deg: float, size, box_yaw: float = 0.0):
        import torch
        th = np.radians(facing_deg) / 2.0
        self.root_quat = torch.tensor([[np.cos(th), 0.0, 0.0, np.sin(th)]], dtype=torch.float32)
        self.terrain_meta = {"terrain": {"boxes": [{"pos": [0, 0, 0], "size": list(size),
                                                    "yaw": box_yaw}]}}


def test_episodic_box_extents_swap_when_the_source_walks_along_y():
    """Every jump reference in the library faces ~-91 deg, so its travel-frame depth is the source's sy.

    Authoring the tile from the raw solved ``size`` would lay the box out 90 degrees to the motion: the
    robot would cross the lateral width and have the depth to either side of it.
    """
    from motion_gen.terrain.terrain_place import episodic_box_extents
    src = _FakeSource(-90.0, (0.80, 0.45, 0.47))
    assert episodic_box_extents(src, align="box") == pytest.approx([0.80, 0.45, 0.47])
    assert episodic_box_extents(src, align="facing") == pytest.approx([0.45, 0.80, 0.47])


def test_episodic_box_extents_are_the_aabb_of_the_rotated_footprint():
    from motion_gen.terrain.terrain_place import episodic_box_extents
    # Facing +x already: nothing to rotate, so the travel frame IS the source frame.
    assert episodic_box_extents(_FakeSource(0.0, (0.80, 0.45, 0.47)), align="facing") \
        == pytest.approx([0.80, 0.45, 0.47])
    # An off-axis heading inflates both footprint dims (axis-aligned bound), never the height.
    sx, sy, sz = episodic_box_extents(_FakeSource(-91.0, (0.80, 0.45, 0.47)), align="facing")
    assert 0.45 < sx < 0.47 and 0.79 < sy < 0.81 and sz == pytest.approx(0.47)
