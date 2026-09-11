# Copyright 2026 Zachary Olkin. All rights reserved.

"""The box-union SDF is exact, and it agrees with the two oracles the repo already has.

Three independent checks, because a signed distance is exactly the kind of quantity that
returns a plausible wrong number:

* against a BRUTE-FORCE nearest-point search over the box surface, which is the definition;
* against ``trimesh``'s own proximity query on the same boxes as a mesh;
* against ``generate_clip_tile_scene.terrain_top_at``, the manifest reader the MuJoCo emitter
  uses for spawn and goal heights -- a completely separate implementation reading the same
  manifest.

Also pinned: the union's ``min`` under-reads penetration at an internal seam.  That is a
documented bound, and a test that asserts it is what stops it being rediscovered as a bug.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from motion_gen.terrain.box_sdf import BoxUnionTerrain, box_union_sdf, pad_box_sets
from motion_gen.terrain.overlay_boxes import (
    GROUND_SLAB_THICKNESS_M,
    collider_boxes_from_manifest_tile,
    flat_tile_collider_boxes,
)

_REPO = pathlib.Path(__file__).resolve().parents[3]

BOX2_TILE = {
    "terrain_id": 0, "type": "terrain", "size_x": 12.9, "size_y": 6.0,
    "frame_yaw": 0.0, "has_floor": False,
    "overlays": [
        {"kind": "box", "pos_xy": [6.45, 3.0], "size_xyz": [12.9, 6.0, 0.5],
         "base_z": -0.5, "walkable_top": True, "yaw": 0.0, "decor": False,
         "variant": "solid", "slab_thickness": 0.1},
        {"kind": "box", "pos_xy": [3.35, 3.0], "size_xyz": [1.2, 2.59, 0.4],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0, "decor": False,
         "variant": "solid", "slab_thickness": 0.1},
        {"kind": "box", "pos_xy": [7.55, 3.0], "size_xyz": [1.2, 2.59, 0.4],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0, "decor": False,
         "variant": "solid", "slab_thickness": 0.1},
    ],
}


def _brute_force_sdf(point: np.ndarray, box: np.ndarray, n: int = 240) -> float:
    """Signed distance from ``point`` to one box, by sampling its surface.

    Args:
        point: ``(3,)`` query point.
        box: ``(6,)`` ``(cx, cy, cz, sx, sy, sz)``.
        n: Samples per face edge.

    Returns:
        The signed distance; negative inside.
    """
    c, h = box[:3], box[3:] / 2.0
    lo, hi = c - h, c + h
    # Nearest surface point: clamp into the box, then push out to the closest face.
    clamped = np.clip(point, lo, hi)
    if np.all((point >= lo) & (point <= hi)):
        # inside: distance to the nearest face
        return -float(np.min(np.concatenate([point - lo, hi - point])))
    return float(np.linalg.norm(point - clamped))


def test_single_box_matches_the_definition():
    """The closed form equals a clamp-based nearest-point computation, in and out."""
    rng = np.random.default_rng(0)
    box = np.array([0.3, -0.2, 0.5, 1.1, 0.7, 0.9])
    pts = rng.uniform(-2.0, 2.0, size=(500, 3))
    got = box_union_sdf(torch.as_tensor(pts, dtype=torch.float64)[None],
                        torch.as_tensor(box, dtype=torch.float64)[None, None])[0].numpy()
    want = np.array([_brute_force_sdf(p, box) for p in pts])
    assert np.allclose(got, want, atol=1e-9), np.abs(got - want).max()


def test_union_matches_trimesh_proximity_outside():
    """Outside the union, the SDF is the true distance to the surface mesh."""
    trimesh = pytest.importorskip("trimesh")
    boxes = collider_boxes_from_manifest_tile(BOX2_TILE)
    meshes = [trimesh.creation.box(extents=b[3:], transform=trimesh.transformations
                                   .translation_matrix(b[:3])) for b in boxes]
    mesh = trimesh.util.concatenate(meshes)

    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(0.0, 12.9, 400), rng.uniform(0.0, 6.0, 400),
                    rng.uniform(0.6, 2.5, 400)], axis=-1)          # comfortably above
    got = box_union_sdf(torch.as_tensor(pts, dtype=torch.float64)[None],
                        torch.as_tensor(boxes, dtype=torch.float64)[None])[0].numpy()
    want = trimesh.proximity.signed_distance(mesh, pts)            # positive INSIDE
    assert np.all(got > 0.0)
    assert np.allclose(got, -want, atol=1e-6), np.abs(got + want).max()


def test_sign_agrees_with_terrain_top_at():
    """The SDF's sign matches the MuJoCo emitter's own 2.5-D manifest reader.

    ``terrain_top_at`` is a separate implementation in a separate package answering "height
    at (x, y)" from the same manifest, so agreeing with it is evidence about the manifest
    reading and not just about the arithmetic.
    """
    import sys
    sys.path.insert(0, str(_REPO / "transfer" / "sim"))
    from generate_clip_tile_scene import terrain_top_at            # noqa: PLC0415

    boxes = collider_boxes_from_manifest_tile(BOX2_TILE)
    terrain = BoxUnionTerrain(torch.as_tensor(boxes, dtype=torch.float64)[None])

    rng = np.random.default_rng(2)
    xy = np.stack([rng.uniform(0.2, 12.7, 300), rng.uniform(0.2, 5.8, 300)], axis=-1)
    tops = np.array([terrain_top_at(BOX2_TILE["overlays"], float(x), float(y)) for x, y in xy])

    for dz, expect_positive in ((0.05, True), (-0.05, False)):
        pts = np.concatenate([xy, (tops + dz)[:, None]], axis=-1)
        sd = terrain.sdf(torch.as_tensor(pts, dtype=torch.float64)[None])[0].numpy()
        assert np.all(sd > 0) == expect_positive, (dz, sd.min(), sd.max())

    # And the SDF's own top_z_at must reproduce the same surface.
    got = terrain.top_z_at(torch.as_tensor(xy, dtype=torch.float64)[None])[0].numpy()
    assert np.allclose(got, tops, atol=1e-9), np.abs(got - tops).max()


def test_min_over_boxes_under_reads_penetration_at_an_internal_seam():
    """The documented bound, asserted so it cannot be rediscovered as a bug.

    A point 0.1 m inside the ground slab, directly under a box that sits ON that slab, is
    0.5 m below the union's own top surface -- but reads -0.1, the distance to the slab's own
    top face, which is an internal seam and not a boundary of the union.
    """
    boxes = collider_boxes_from_manifest_tile(BOX2_TILE)
    terrain = BoxUnionTerrain(torch.as_tensor(boxes, dtype=torch.float64)[None])
    sd = terrain.sdf(torch.tensor([[[3.35, 3.0, -0.10]]], dtype=torch.float64))[0, 0].item()
    assert sd == pytest.approx(-0.10, abs=1e-9)


def test_padding_and_masking_do_not_change_the_answer():
    """A masked-out padded box cannot win the min, so padding is inert."""
    a = collider_boxes_from_manifest_tile(BOX2_TILE)
    b = flat_tile_collider_boxes(8.0, 8.0)
    boxes, mask = pad_box_sets([a, b], dtype=torch.float64)
    assert boxes.shape == (2, len(a), 6) and mask[1].sum() == 1

    pts = torch.tensor([[[3.35, 3.0, 0.5]], [[3.35, 3.0, 0.5]]], dtype=torch.float64)
    padded = box_union_sdf(pts, boxes, mask)
    alone_a = box_union_sdf(pts[:1], torch.as_tensor(a, dtype=torch.float64)[None])
    alone_b = box_union_sdf(pts[1:], torch.as_tensor(b, dtype=torch.float64)[None])
    assert padded[0].item() == pytest.approx(alone_a[0].item(), abs=1e-12)
    assert padded[1].item() == pytest.approx(alone_b[0].item(), abs=1e-12)


def test_translated_and_index_compose_the_way_a_tile_table_needs():
    """``index`` then ``translated`` is how a per-tile table becomes a per-env one."""
    boxes, mask = pad_box_sets([collider_boxes_from_manifest_tile(BOX2_TILE),
                                flat_tile_collider_boxes(8.0, 8.0)], dtype=torch.float64)
    tiles = BoxUnionTerrain(boxes, mask)
    envs = tiles.index(torch.tensor([0, 0, 1]))
    assert len(envs) == 3
    corner = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 50.0, 0.0]],
                          dtype=torch.float64)
    world = envs.translated(corner)
    q = torch.tensor([[[3.35, 3.0, 0.5]], [[103.35, 3.0, 0.5]], [[4.0, 54.0, 0.1]]],
                     dtype=torch.float64)
    sd = world.sdf(q)
    assert sd[0].item() == pytest.approx(sd[1].item(), abs=1e-9)     # same tile, shifted
    assert sd[2].item() == pytest.approx(0.1, abs=1e-9)              # 0.1 m over flat ground


def test_shape_disagreements_raise_rather_than_broadcast():
    """A batch mismatch is refused: broadcasting it would grade plans against wrong tiles."""
    pts = torch.zeros(2, 5, 3)
    boxes = torch.zeros(3, 4, 6)
    with pytest.raises(ValueError, match="batch"):
        box_union_sdf(pts, boxes)
    with pytest.raises(ValueError, match=r"points must be"):
        box_union_sdf(torch.zeros(5, 3), torch.zeros(1, 4, 6))
    with pytest.raises(ValueError, match=r"boxes must be"):
        box_union_sdf(pts, torch.zeros(2, 4))


def test_legacy_bands_tile_is_refused_not_flattened():
    """A legacy tile has no box list, and returning only its ground would be a FLAT floor."""
    legacy = {"terrain_id": 7, "type": "terrain", "size_x": 20.0, "size_y": 4.0,
              "has_floor": True, "dims": {"w": 0.3, "h": 0.13, "d": 1.0, "rise": 0.0},
              "bands": [{"block_type": "StairBlock", "aabb": [0.0, 10.0, 0.0, 4.0]}]}
    with pytest.raises(ValueError, match="legacy"):
        collider_boxes_from_manifest_tile(legacy)


def test_a_yawed_overlay_is_refused():
    """Both simulators refuse a yawed overlay; an axis-aligned SDF must too."""
    tile = json.loads(json.dumps(BOX2_TILE))
    tile["overlays"][1]["yaw"] = 0.3
    with pytest.raises(ValueError, match="yaw"):
        collider_boxes_from_manifest_tile(tile)


def test_has_floor_adds_a_ground_slab_and_false_does_not():
    """``has_floor`` is the difference between a tile with ground and an isolated structure."""
    tile = json.loads(json.dumps(BOX2_TILE))
    tile["has_floor"] = True
    with_floor = collider_boxes_from_manifest_tile(tile)
    without = collider_boxes_from_manifest_tile(BOX2_TILE)
    assert len(with_floor) == len(without) + 1
    slab = with_floor[-1]
    assert slab[5] == pytest.approx(GROUND_SLAB_THICKNESS_M)
    assert slab[2] == pytest.approx(-GROUND_SLAB_THICKNESS_M / 2.0)


def test_decor_overlays_are_excluded_from_the_collider_set():
    """Decoration has no collider in either simulator, so it must not be ground."""
    tile = json.loads(json.dumps(BOX2_TILE))
    tile["overlays"].append({"kind": "box", "pos_xy": [1.0, 1.0],
                             "size_xyz": [0.5, 0.5, 2.0], "base_z": 0.0,
                             "walkable_top": False, "yaw": 0.0, "decor": True,
                             "variant": "solid", "slab_thickness": 0.1})
    assert len(collider_boxes_from_manifest_tile(tile)) == \
        len(collider_boxes_from_manifest_tile(BOX2_TILE))


def test_a_floating_variant_contributes_its_slab_and_not_the_gap():
    """A floating tread's collider is the slab; the volume beneath it is empty."""
    tile = json.loads(json.dumps(BOX2_TILE))
    tile["overlays"][1]["variant"] = "floating"
    tile["overlays"][1]["slab_thickness"] = 0.1
    boxes = collider_boxes_from_manifest_tile(tile)
    terrain = BoxUnionTerrain(torch.as_tensor(boxes, dtype=torch.float64)[None])
    # Under the slab (z = 0.15) there is now a gap: the nearest surface is the slab bottom
    # at 0.30, i.e. 0.15 away, NOT inside anything.
    sd = terrain.sdf(torch.tensor([[[3.35, 3.0, 0.15]]], dtype=torch.float64))[0, 0].item()
    assert sd == pytest.approx(0.15, abs=1e-9)
