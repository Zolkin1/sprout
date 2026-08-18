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
    OVERLAY_KINDS,
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    OverlayVolumeCfg,
    build_overlay,
    composite_terrain,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.overlay import overlay_sub_boxes
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
    _overlays_from_manifest,
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


def _fake_lib(manifest, terrain_id=(0,)):
    """Minimal stand-in for a loaded clip library carrying an arbitrary terrain manifest."""
    return types.SimpleNamespace(
        attrs={"terrain_manifest": manifest},
        terrain_id=torch.tensor(list(terrain_id), dtype=torch.int32),
        y_des=torch.zeros(len(terrain_id), 5, 4),
        meta={"pos_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y",
                                   "pelvis_link:pos_z", "j0"]},
    )


def test_assemble_pure_stair_tile_from_overlays():
    """A pure-stair 'terrain' tile carrying explicit tread overlays (the tiles-then-clips schema,
    where the generator emits geometry as the source of truth) assembles with NO legacy 'dims' key."""
    treads = [{"kind": "box", "pos_xy": [0.5 + 0.2 * i, 1.0],
               "size_xyz": [0.2, 1.0, 0.45 - 0.15 * i],          # solid columns; tops 0.45, 0.30, 0.15
               "base_z": 0.0, "walkable_top": True} for i in range(3)]
    manifest = [{"terrain_id": 0, "type": "terrain", "size_x": 6.0, "size_y": 2.0, "overlays": treads}]
    mesh, _origins, layout = assemble_clip_terrain(_fake_lib(manifest), num_rows=1)
    assert layout.terrain_type_code[0] == 1                       # "terrain" (pure stairs)
    assert len(layout.terrain_overlays[0]) == 3                   # all treads carried as overlays
    assert float(mesh.bounds[1][2]) == pytest.approx(0.45)        # mesh top == highest tread


def test_assemble_transition_tile_from_overlays():
    """A 'transition' tile (flat -> flight -> platform) built from explicit overlays; the clip is
    already tile-framed so the loader adds no front-pad shift."""
    ovs = [{"kind": "box", "pos_xy": [1.0, 1.0], "size_xyz": [2.0, 1.0, 0.30],   # top platform
            "base_z": 0.0, "walkable_top": True},
           {"kind": "box", "pos_xy": [2.2, 1.0], "size_xyz": [0.2, 0.9, 0.30],
            "base_z": 0.0, "walkable_top": True},
           {"kind": "box", "pos_xy": [2.4, 1.0], "size_xyz": [0.2, 0.9, 0.15],
            "base_z": 0.0, "walkable_top": True}]
    manifest = [{"terrain_id": 0, "type": "transition", "size_x": 6.0, "size_y": 2.0, "overlays": ovs}]
    mesh, _origins, layout = assemble_clip_terrain(_fake_lib(manifest), num_rows=1)
    assert layout.terrain_type_code[0] == 2                       # "transition"
    assert len(layout.terrain_overlays[0]) == 3
    assert float(mesh.bounds[1][2]) == pytest.approx(0.30)        # mesh top == platform/top step


def test_assemble_pure_stair_tile_no_floor():
    """has_floor=False on a pure-stair 'terrain' tile drops the flat ground plane too: the solid
    staircase stands isolated (built from overlays only), keeping its tread heights."""
    treads = [{"kind": "box", "pos_xy": [0.5 + 0.2 * i, 1.0],
               "size_xyz": [0.2, 1.0, 0.45 - 0.15 * i],
               "base_z": 0.0, "walkable_top": True} for i in range(3)]
    grounded = [{"terrain_id": 0, "type": "terrain", "size_x": 6.0, "size_y": 2.0,
                 "overlays": treads, "has_floor": True}]
    floorless = [{"terrain_id": 0, "type": "terrain", "size_x": 6.0, "size_y": 2.0,
                  "overlays": treads, "has_floor": False}]
    mg, _og, _lg = assemble_clip_terrain(_fake_lib(grounded), num_rows=1)
    mf, _of, lf = assemble_clip_terrain(_fake_lib(floorless), num_rows=1)
    assert lf.terrain_type_code[0] == 1                         # still a pure-stair tile
    assert len(lf.terrain_overlays[0]) == 3                     # treads kept
    assert float(mf.bounds[1][2]) == pytest.approx(0.45)        # same top tread height
    assert (mf.bounds[1][0] - mf.bounds[0][0]) < (mg.bounds[1][0] - mg.bounds[0][0])   # no ground plane
    assert len(mf.vertices) < len(mg.vertices)


def test_assemble_transition_tile_no_floor():
    """has_floor=False -> the transition tile drops the flat ground plane: same stairs+platforms and
    the same top height, but built from overlays only (no full-footprint ground quad)."""
    ovs = [{"kind": "box", "pos_xy": [1.0, 1.0], "size_xyz": [2.0, 1.0, 0.30],
            "base_z": 0.0, "walkable_top": True},
           {"kind": "box", "pos_xy": [2.2, 1.0], "size_xyz": [0.2, 0.9, 0.30],
            "base_z": 0.0, "walkable_top": True}]
    grounded = [{"terrain_id": 0, "type": "transition", "size_x": 6.0, "size_y": 2.0,
                 "overlays": ovs, "has_floor": True}]
    floorless = [{"terrain_id": 0, "type": "transition", "size_x": 6.0, "size_y": 2.0,
                  "overlays": ovs, "has_floor": False}]
    mg, _og, _lg = assemble_clip_terrain(_fake_lib(grounded), num_rows=1)
    mf, _of, lf = assemble_clip_terrain(_fake_lib(floorless), num_rows=1)
    assert lf.terrain_type_code[0] == 2                          # still a transition tile
    assert len(lf.terrain_overlays[0]) == len(ovs)              # stairs + platforms kept
    assert float(mf.bounds[1][2]) == pytest.approx(0.30)        # same top height
    # No ground plane -> the mesh spans only the box extents, not the full 6 m footprint.
    assert (mf.bounds[1][0] - mf.bounds[0][0]) < (mg.bounds[1][0] - mg.bounds[0][0])
    assert len(mf.vertices) < len(mg.vertices)


def test_assemble_legacy_stair_tile_from_dims_still_works():
    """The legacy dims/bands StairBlock schema (no 'overlays') still routes through the recompute
    path -- the tiles-then-clips overlay support must not break older libraries."""
    manifest = [{"terrain_id": 0, "type": "terrain", "size_x": 6.0, "size_y": 2.0,
                 "dims": {"w": 1.0, "h": 0.15, "d": 0.3},
                 "bands": [{"aabb": [0.0, 3.0], "block_type": "StairBlock"}]}]
    mesh, _origins, layout = assemble_clip_terrain(_fake_lib(manifest), num_rows=1)
    assert layout.terrain_type_code[0] == 1
    assert not layout.terrain_overlays.get(0)                     # legacy path carries no overlays
    assert float(mesh.bounds[1][2]) > 0.0                         # the staircase rose above ground


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


# --- box support-structure variants (floating top / center leg / corner legs) ---

def test_overlay_variant_top_z_invariant():
    """Every variant keeps the walkable top + aabb of the equivalent solid box."""
    size, pos, base_z = (0.6, 0.6, 0.4), (2.0, 2.0), 0.0
    solid = build_overlay(OverlayVolumeCfg(size_xyz=size, pos_xy=pos, base_z=base_z, variant="solid"))
    for variant in ("floating", "floating_center_leg", "floating_corner_legs"):
        o = build_overlay(OverlayVolumeCfg(size_xyz=size, pos_xy=pos, base_z=base_z, variant=variant))
        assert o.top_z == pytest.approx(solid.top_z)          # landing height unchanged
        assert o.aabb == solid.aabb                            # bounding footprint x height unchanged
        # The mesh top face reaches exactly top_z (the slab/box top).
        assert float(o.mesh.bounds[1][2]) == pytest.approx(o.top_z)


def test_overlay_sub_box_counts():
    args = ("solid", (0.6, 0.6, 0.4), (0.0, 0.0), 0.0, 0.1, (0.1, 0.1), 0.05)
    assert len(overlay_sub_boxes(*args)) == 1
    assert len(overlay_sub_boxes("floating", *args[1:])) == 1
    assert len(overlay_sub_boxes("floating_center_leg", *args[1:])) == 2
    assert len(overlay_sub_boxes("floating_corner_legs", *args[1:])) == 5


def test_overlay_floating_has_gap_below_slab():
    """A floating box's mesh sits ONLY near the top (a thin slab), not down to base_z."""
    o = build_overlay(OverlayVolumeCfg(size_xyz=(0.6, 0.6, 0.4), pos_xy=(0.0, 0.0),
                                       base_z=0.0, variant="floating", slab_thickness=0.1))
    zmin = float(o.mesh.bounds[0][2])
    assert zmin == pytest.approx(0.3)                          # top_z (0.4) - slab (0.1); gap [0, 0.3) below


def test_overlay_corner_legs_positions():
    """Four legs sit inset from the footprint corners; slab spans the full top."""
    subs = overlay_sub_boxes("floating_corner_legs", (1.0, 1.0, 0.5), (0.0, 0.0), 0.0, 0.1, (0.1, 0.1), 0.05)
    legs = subs[1:]
    assert len(legs) == 4
    off = 1.0 / 2.0 - 0.05 - 0.1 / 2.0                          # 0.4
    for cx, cy, _cz, _sx, _sy, _sz in legs:
        assert abs(abs(cx) - off) < 1e-9 and abs(abs(cy) - off) < 1e-9


def test_overlay_variant_round_trips_through_manifest():
    """A variant overlay survives the composite metadata + assemble round-trip."""
    sub = CompositeSubTerrainCfg(
        size=(4.0, 4.0), blocks=[FlatBlockCfg(size_x=4.0, skill_probs=_FLAT)],
        overlays=[OverlayVolumeCfg(size_xyz=(0.6, 0.6, 0.4), pos_xy=(2.0, 2.0),
                                   variant="floating_corner_legs", slab_thickness=0.08,
                                   leg_size=(0.09, 0.09), leg_inset=0.04)],
    )
    _meshes, _origin, meta = composite_terrain(0.0, sub)
    ov = meta["overlays"][0]
    assert ov["variant"] == "floating_corner_legs"
    assert ov["slab_thickness"] == pytest.approx(0.08)
    assert tuple(ov["leg_size"]) == pytest.approx((0.09, 0.09))
    assert ov["leg_inset"] == pytest.approx(0.04)
    assert ov["top_z"] == pytest.approx(0.4)


##
# The manifest contract: fields this loader cannot represent are REFUSED, not reinterpreted.
##


def _one_overlay_entry(**overrides):
    """A minimal one-overlay manifest entry, with ``overrides`` applied to the overlay."""
    ov = {"kind": "box", "pos_xy": [1.0, 2.0], "size_xyz": [1.0, 1.0, 1.0],
          "base_z": 0.0, "walkable_top": True}
    ov.update(overrides)
    return {"terrain_id": 7, "type": "box", "size_x": 4.0, "size_y": 4.0, "overlays": [ov]}


def test_overlay_yaw_zero_or_absent_is_accepted():
    """The only yaw any generator authors today, in both spellings."""
    assert len(_overlays_from_manifest(_one_overlay_entry(yaw=0.0))) == 1
    entry = _one_overlay_entry()
    entry["overlays"][0].pop("yaw", None)
    assert len(_overlays_from_manifest(entry)) == 1


def test_overlay_nonzero_yaw_is_refused():
    """A yawed overlay must RAISE rather than be silently built axis-aligned.

    ``OverlayVolumeCfg`` has no yaw field and ``build_overlay`` only translates, while
    ``transfer/sim/generate_clip_tile_scene.py`` emits a rotation quaternion for the same
    manifest entry -- so a yawed box would be a different box in each consumer, with no
    error anywhere.  Supporting yaw properly means widening ``overlay_sub_boxes``' return
    contract, which every renderer depends on.
    """
    with pytest.raises(ValueError, match="yaw"):
        _overlays_from_manifest(_one_overlay_entry(yaw=0.7853981633974483))


def test_overlay_unknown_kind_is_refused():
    """``kind`` is a Literal that ``@configclass`` does not enforce at runtime.

    Without this check ``kind="wall"`` is accepted and drawn as a plain box -- which is
    how a new shape would look correct in Isaac and be missing in MuJoCo.
    """
    with pytest.raises(ValueError, match="kind"):
        _overlays_from_manifest(_one_overlay_entry(kind="wall"))


@pytest.mark.parametrize("kind", sorted(OVERLAY_KINDS))
def test_every_declared_overlay_kind_is_accepted(kind):
    """The refusal must be exactly the complement of the declared Literal."""
    assert _overlays_from_manifest(_one_overlay_entry(kind=kind))[0].kind == kind


##
# Every clip must have a tile: the complement of resets.py's "tile has no clips" raise.
##


def _box_manifest(terrain_id=0):
    return [{"terrain_id": terrain_id, "type": "box", "size_x": 4.0, "size_y": 4.0,
             "overlays": [{"kind": "box", "pos_xy": [2.0, 2.0], "size_xyz": [0.6, 0.6, 0.4],
                           "base_z": 0.0, "walkable_top": True}]}]


def test_clip_on_an_undescribed_terrain_is_refused():
    """A clip tagged with an id no manifest entry claims must RAISE at assembly.

    It used to be dropped silently: no tile is built for the id, so no env lands on it,
    ``sample_terrain_clips`` never asks for it, and the clip vanishes from training and
    from every recording with no message. Renumbering ids in an offline rewrite stage is
    exactly how that orphan appears.
    """
    lib = _fake_lib(_box_manifest(terrain_id=0), terrain_id=(0, 99))
    with pytest.raises(ValueError, match=r"terrain id\(s\) \[99\]"):
        assemble_clip_terrain(lib, num_flat_tiles=0)


def test_flat_clips_are_exempt_from_the_coverage_check():
    """``terrain_id == -1`` is flat ground, which the manifest deliberately never describes."""
    lib = _fake_lib(_box_manifest(terrain_id=0), terrain_id=(0, -1))
    mesh, _origins, layout = assemble_clip_terrain(lib, num_flat_tiles=1)
    assert mesh is not None
    assert -1 in layout.terrain_type_code
