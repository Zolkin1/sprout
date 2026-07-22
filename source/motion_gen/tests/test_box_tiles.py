# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the jump-on BOX tiles-then-clips phase-A sampler + fixed tile geometry.

Locks the box-family reuse invariant (the box analog of the stair transition tiles): ``sample_box_tiles``
produces a tile population with contiguous ids, in-range source indices, variant weights honored, and
lateral offsets bounded to the corridor; ``box_tile_box`` authors ONE fixed box shared by all of a tile's
clips -- its near face sits ``approach_len`` down the flat run-up and its top is ``sz`` (base at z=0).
"""

from __future__ import annotations

import numpy as np

from motion_gen.motion_matching.box_clip import box_tile_box
from motion_gen.motion_matching.terrain_sampler import BoxTileSpec, sample_box_tiles


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
