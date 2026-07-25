# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the isaaclab-free terrain tile sampler (``terrain_sampler``).

Locks the tiles-then-clips phase-A policy: pure tiles are long + platform-less, transition tiles are
flat -> single staircase -> flat, ids are contiguous, and single-staircase transitions are allowed
(the ``num_transitions_range=(1, 1)`` path the extraction relaxed from the old ``>= 2`` clamp).
"""

from __future__ import annotations

import numpy as np

from motion_gen.motion_matching.terrain_sampler import sample_stair_tiles, sample_terrain_specs


_COMMON = dict(
    hd_grid=[(0.15, 0.19)], width_range=(0.8, 1.2), num_steps_range=(3, 7),
    lead_in_range=(1.5, 4.0), end_platform=1.0, inter_flat_range=(3.0, 5.0),
    nav_width_range=(1.0, 1.4), flat_width_margin=0.3,
)


def test_single_staircase_transition_allowed():
    # The extraction relaxed the old max(2,...) clamp; (1, 1) must now yield exactly one flight.
    specs = sample_terrain_specs(np.random.default_rng(1), n_terrain=0, n_transition=20,
                                 num_transitions_range=(1, 1), transition_length=16.0, **_COMMON)
    assert specs and all(len(s.stair_steps) == 1 for s in specs)


def test_pure_tiles_are_long_and_platformless():
    tiles = sample_stair_tiles(np.random.default_rng(2), n_pure=3, n_transition=0,
                               pure_length=20.0, transition_length=16.0, **_COMMON)
    assert len(tiles) == 3
    for t in tiles:
        assert t.ttype == "terrain"
        assert t.lead_in == 0.0 and t.flat_after == [0.0]        # no platforms
        assert len(t.stair_steps) == 1
        # ~ pure_length / step_depth steps (long enough for a 20s-ish looped descent)
        assert t.stair_steps[0] == max(2, round(20.0 / t.step_depth))
        assert t.stair_steps[0] > 50


def test_transition_tiles_single_flight_with_platforms():
    tiles = sample_stair_tiles(np.random.default_rng(3), n_pure=0, n_transition=5,
                               pure_length=20.0, transition_length=16.0, **_COMMON)
    assert len(tiles) == 5
    for t in tiles:
        assert t.ttype == "transition"
        assert len(t.stair_steps) == 1                            # single staircase
        assert t.lead_in >= 1.5 - 1e-6                            # real flat lead-in
        assert t.flat_after[-1] >= 1.0 - 1e-6                     # top platform >= end_platform


def test_ids_contiguous_across_merged_population():
    tiles = sample_stair_tiles(np.random.default_rng(4), n_pure=2, n_transition=3,
                               pure_length=20.0, transition_length=16.0, **_COMMON)
    assert [t.terrain_id for t in tiles] == list(range(5))        # contiguous 0..N-1
    assert sum(t.ttype == "terrain" for t in tiles) == 2
    assert sum(t.ttype == "transition" for t in tiles) == 3


def test_floor_frac_controls_stair_ground_plane():
    # Default (floor_frac=1.0): every stair tile keeps its ground plane.
    all_floor = sample_stair_tiles(np.random.default_rng(10), n_pure=4, n_transition=12,
                                   pure_length=20.0, transition_length=16.0, **_COMMON)
    assert all(t.has_floor for t in all_floor)
    # floor_frac=0.0: NO tile keeps it -- pure AND transition.
    none_floor = sample_stair_tiles(np.random.default_rng(11), n_pure=4, n_transition=12,
                                    pure_length=20.0, transition_length=16.0, floor_frac=0.0, **_COMMON)
    assert not any(t.has_floor for t in none_floor)
    # A partial fraction over many tiles yields a mix within BOTH families.
    mixed = sample_stair_tiles(np.random.default_rng(12), n_pure=30, n_transition=30,
                               pure_length=20.0, transition_length=16.0, floor_frac=0.5, **_COMMON)
    for tt in ("terrain", "transition"):
        flags = [t.has_floor for t in mixed if t.ttype == tt]
        assert any(flags) and not all(flags)                 # both grounded and floorless present per family


def test_pure_length_scales_with_target():
    short = sample_stair_tiles(np.random.default_rng(5), n_pure=1, n_transition=0,
                               pure_length=10.0, transition_length=16.0, **_COMMON)[0]
    long = sample_stair_tiles(np.random.default_rng(5), n_pure=1, n_transition=0,
                              pure_length=30.0, transition_length=16.0, **_COMMON)[0]
    # A longer target (future faster stair velocity => more distance in 20s) grows the tile.
    assert long.stair_steps[0] > short.stair_steps[0]
