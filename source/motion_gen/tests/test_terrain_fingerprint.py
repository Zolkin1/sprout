# Copyright 2026 Zachary Olkin. All rights reserved.

"""A rewritten terrain manifest must not report an unchanged fingerprint.

``fingerprint_clip_config`` folds in the source library, the generator config and the
seed -- not the terrain. It is also imported in exactly one place and called in none, so
no library on disk carries a ``clip_fingerprint`` attribute and the existing check has
never had anything to compare. Both halves matter for terrain decoration, which rewrites
tile geometry while leaving every motion byte-identical.
"""

from __future__ import annotations

import json

import pytest

from motion_gen.clips.clip_fingerprint import (
    fingerprint_terrain_manifest,
    verify_terrain_fingerprint,
)

TILE = {
    "terrain_id": 0, "type": "transition", "size_x": 22.86, "size_y": 4.4589,
    "has_floor": True, "frame_yaw": 0.0, "family": "family_stairs_up.h5",
    "overlays": [
        {"kind": "box", "pos_xy": [6.05, 2.229], "size_xyz": [10.1, 4.4589, 0.13],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0},
        {"kind": "box", "pos_xy": [11.21, 1.866], "size_xyz": [0.22, 0.955, 0.13],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0},
    ],
}


def _tile(**over):
    t = json.loads(json.dumps(TILE))
    t.update(over)
    return t


def test_the_same_manifest_gives_the_same_hash():
    assert fingerprint_terrain_manifest([TILE])["hash"] == \
           fingerprint_terrain_manifest([_tile()])["hash"]


def test_tile_order_and_key_order_do_not_matter():
    """A re-serialisation that reorders nothing meaningful must not look like new terrain."""
    a, b = _tile(terrain_id=0), _tile(terrain_id=1)
    assert fingerprint_terrain_manifest([a, b])["hash"] == \
           fingerprint_terrain_manifest([b, a])["hash"]
    shuffled = {k: v for k, v in reversed(list(TILE.items()))}
    assert fingerprint_terrain_manifest([shuffled])["hash"] == \
           fingerprint_terrain_manifest([TILE])["hash"]


def test_widening_a_tread_changes_the_hash():
    """The decoration transform, in miniature: geometry moves, motions do not."""
    wide = _tile()
    wide["overlays"][1]["size_xyz"][1] = 4.4589
    before, after = fingerprint_terrain_manifest([TILE]), fingerprint_terrain_manifest([wide])
    assert before["hash"] != after["hash"]
    ok, msg = verify_terrain_fingerprint(before, after)
    assert not ok and "TERRAIN MISMATCH" in msg
    assert "GROUND is not" in msg


def test_adding_a_wall_changes_the_hash_and_the_overlay_count():
    decorated = _tile()
    decorated["overlays"].append(
        {"kind": "obstacle", "pos_xy": [11.0, 0.3], "size_xyz": [8.0, 0.2, 2.5],
         "base_z": 0.0, "walkable_top": False, "yaw": 0.0})
    before, after = fingerprint_terrain_manifest([TILE]), fingerprint_terrain_manifest([decorated])
    assert before["num_overlays"] == 2 and after["num_overlays"] == 3
    assert before["hash"] != after["hash"]


def test_splitting_a_tile_changes_the_hash_and_the_tile_count():
    before = fingerprint_terrain_manifest([TILE])
    after = fingerprint_terrain_manifest([TILE, _tile(terrain_id=140)])
    assert (before["num_tiles"], after["num_tiles"]) == (1, 2)
    assert before["hash"] != after["hash"]


def test_an_empty_manifest_is_a_legitimate_flat_only_library():
    for manifest in (None, []):
        fp = fingerprint_terrain_manifest(manifest)
        assert fp["num_tiles"] == 0 and len(fp["hash"]) == 64
    assert fingerprint_terrain_manifest(None)["hash"] == fingerprint_terrain_manifest([])["hash"]


def test_a_match_is_a_match():
    fp = fingerprint_terrain_manifest([TILE])
    ok, msg = verify_terrain_fingerprint(fp, fp)
    assert ok and "terrain match" in msg


@pytest.mark.parametrize("recorded,current", [(None, {"hash": "x"}), ({"hash": "x"}, None)])
def test_an_absent_fingerprint_is_reported(recorded, current):
    ok, msg = verify_terrain_fingerprint(recorded, current)
    assert not ok and msg


def test_the_shipped_libraries_fingerprint_and_differ():
    """Two real libraries with different terrain must not collide."""
    h5py = pytest.importorskip("h5py")
    import pathlib

    from motion_gen.features import _REPO_ROOT

    paths = [_REPO_ROOT / "clips" / n
             for n in ("g1_mb_v5_body.h5", "g1_mb_v5_small_body.h5")]
    if not all(pathlib.Path(p).is_file() for p in paths):
        pytest.skip("clip libraries not present")
    fps = []
    for p in paths:
        with h5py.File(p, "r") as f:
            fps.append(fingerprint_terrain_manifest(json.loads(f.attrs["terrain_manifest"])))
    assert fps[0]["num_tiles"] == 140 and fps[0]["num_overlays"] == 1260
    assert fps[0]["hash"] != fps[1]["hash"]
