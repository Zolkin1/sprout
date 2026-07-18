# Copyright 2026 Zachary Olkin. All rights reserved.

"""Unit tests for overlay volumes (standalone boxes) on composite sub-terrains.

Exercises :class:`OverlayVolumeCfg` / :func:`build_overlay`, the overlay merge in
:func:`composite_terrain`, and the box-tile branch of :func:`assemble_clip_terrain`.
Isaac sim is not required (pure trimesh + numpy + the IsaacLab utils import).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import (
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    OverlayVolumeCfg,
    build_overlay,
    composite_terrain,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
    assemble_clip_terrain,
)

_FLAT = {"standing": 1.0}


def test_build_overlay_box_geometry():
    ov = OverlayVolumeCfg(kind="box", size_xyz=(0.6, 0.8, 0.4), pos_xy=(3.0, 1.0), base_z=0.0)
    o = build_overlay(ov)
    assert o.aabb == (2.7, 3.3, 0.6, 1.4, 0.0, 0.4)
    assert o.top_z == pytest.approx(0.4)
    assert o.walkable_top is True
    # A solid box of the requested extents, sitting on the ground (zmin == base_z).
    assert np.allclose(o.mesh.bounding_box.extents, (0.6, 0.8, 0.4))
    assert float(o.mesh.bounds[0][2]) == pytest.approx(0.0)


def test_build_overlay_floating_overhead():
    """An overhead bar floats: base_z > 0 lifts its bottom face off the ground."""
    o = build_overlay(OverlayVolumeCfg(kind="overhead", size_xyz=(0.3, 1.0, 0.1),
                                       pos_xy=(2.0, 0.5), base_z=1.2, walkable_top=False))
    assert float(o.mesh.bounds[0][2]) == pytest.approx(1.2)
    assert o.top_z == pytest.approx(1.3)
    assert o.walkable_top is False


def test_build_overlay_rejects_nonpositive_extent():
    with pytest.raises(ValueError):
        build_overlay(OverlayVolumeCfg(kind="box", size_xyz=(0.0, 0.8, 0.4), pos_xy=(0.0, 0.0)))


def test_composite_merges_overlays():
    ov = OverlayVolumeCfg(kind="box", size_xyz=(0.6, 0.8, 0.4), pos_xy=(3.0, 1.0))
    sub = CompositeSubTerrainCfg(
        size=(6.0, 2.0), blocks=[FlatBlockCfg(size_x=6.0, skill_probs=_FLAT)], overlays=[ov],
    )
    meshes, _origin, meta = composite_terrain(0.0, sub)
    assert len(meshes) == 2                      # flat ground + box
    assert len(meta["blocks"]) == 1              # ground block metadata unchanged
    assert len(meta["overlays"]) == 1
    m = meta["overlays"][0]
    assert m["kind"] == "box"
    assert m["pos_xy"] == (3.0, 1.0)
    assert m["top_z"] == pytest.approx(0.4)


def test_composite_without_overlays_is_empty_list():
    sub = CompositeSubTerrainCfg(size=(6.0, 2.0), blocks=[FlatBlockCfg(size_x=6.0, skill_probs=_FLAT)])
    _m, _o, meta = composite_terrain(0.0, sub)
    assert meta["overlays"] == []


def _fake_box_lib(manifest):
    """Minimal stand-in for a loaded clip library carrying a box manifest."""
    return types.SimpleNamespace(
        attrs={"terrain_manifest": manifest},
        terrain_id=torch.tensor([0], dtype=torch.int32),
        y_des=torch.zeros(1, 5, 4),
        meta={"pos_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y",
                                   "pelvis_link:pos_z", "j0"]},
    )


def test_assemble_box_tile_from_manifest():
    manifest = [{
        "terrain_id": 0, "type": "box", "size_x": 6.0, "size_y": 2.0,
        "overlays": [{"kind": "box", "pos_xy": [3.0, 1.0],
                      "size_xyz": [0.6, 0.8, 0.4], "base_z": 0.0, "walkable_top": True}],
    }]
    mesh, _origins, layout = assemble_clip_terrain(_fake_box_lib(manifest), num_rows=1)

    assert layout.num_tiles == 1
    assert int(layout.tile_terrain_id[0, 0]) == 0
    assert layout.terrain_type_code[0] == 4                       # "box"
    assert layout.terrain_overlays[0][0]["kind"] == "box"
    # The box raises the terrain to its top height.
    assert float(mesh.bounds[1][2]) == pytest.approx(0.4)
    # Box world (x, y) = tile corner + local pos_xy, and lands within the tile.
    corner = layout.tile_corner[0, 0]
    ov = layout.terrain_overlays[0][0]
    bx = float(corner[0]) + ov["pos_xy"][0]
    by = float(corner[1]) + ov["pos_xy"][1]
    bmin, bmax = mesh.bounds
    assert bmin[0] <= bx <= bmax[0]
    assert bmin[1] <= by <= bmax[1]
