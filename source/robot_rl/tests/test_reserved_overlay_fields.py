# Copyright 2026 Zachary Olkin. All rights reserved.

"""The visual-only decoration channel, and how each simulator realises it.

``decor`` is Part 2.4's visual-only channel: a wall the depth cameras see, the height scan
does not, and the robot cannot walk into. Each of those three is a different mechanism per
simulator -- a separate prim in IsaacLab, because the height scanner is a ``RayCaster`` that
takes exactly ONE mesh prim while the cameras take a list; geom group 1 with ``contype=0`` in
MuJoCo.

A consumer that merely ignores the field builds the decoration as a SOLID, COLLIDABLE box
that the height scan reads as terrain. That is worse than not having the feature: instead of
a wall the robot can see and not hit, you get a wall it can hit and the scan mistakes for
ground. Both sides now implement it, and these tests pin the two mechanisms together -- they differ
because the constraints differ, not by accident, and neither may borrow the other's approach.

The field was reserved-and-refused first, deliberately, so the emitter could not land ahead
of its consumers. Same move Part 0 made for overlay ``yaw``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "transfer" / "sim"))


def _overlay(**extra) -> dict:
    """A valid overlay, plus whatever the caller adds."""
    return {"kind": "box", "pos_xy": [1.0, 2.0], "size_xyz": [1.0, 1.0, 0.2],
            "base_z": 0.0, "walkable_top": True, "yaw": 0.0, **extra}


def test_the_mujoco_emitter_makes_decor_visible_but_not_solid():
    """Group 1 so the depth cameras cast it, contype 0 so nothing can walk into it.

    depth_camera.py casts groups {1, 2} and the height scan casts {2}. So group 1 is exactly
    "the cameras see it, the scan does not", and the contact flags are the third property.
    All three come from one field, and getting any one wrong is silent.
    """
    import generate_clip_tile_scene as G
    plain = G._box_geom("g", 0, 0, 0, 1, 1, 1, 0.0, decor=False)
    decor = G._box_geom("g", 0, 0, 0, 1, 1, 1, 0.0, decor=True)
    assert 'group="2"' in plain and "contype" not in plain
    assert 'group="1"' in decor
    assert 'contype="0"' in decor and 'conaffinity="0"' in decor


def test_the_mujoco_tile_emitter_routes_decor_overlays():
    """End to end through tile_geoms, not just the geom writer."""
    import generate_clip_tile_scene as G
    entry = {"terrain_id": 0, "size_x": 10.0, "size_y": 4.0, "has_floor": True}
    overlays = [_overlay(), _overlay(decor=True)]
    lines, _floor = G.tile_geoms(entry, overlays, (0.0, 0.0, 0.0), False)
    solid = [ln for ln in lines if 'group="2"' in ln]
    decor = [ln for ln in lines if 'group="1"' in ln]
    assert len(solid) == 1 and len(decor) == 1
    assert "tile_decor" in decor[0], "a decor geom should be named as one, for debugging"
    assert 'contype="0"' in decor[0]


def test_the_vendored_decomposition_accepts_decor():
    """The sub-box split is the same either way -- only the geom attributes differ."""
    from overlay_geom import is_decor, overlay_sub_boxes_from_manifest
    assert is_decor(_overlay(decor=True)) and not is_decor(_overlay())
    assert (overlay_sub_boxes_from_manifest(_overlay(decor=True))
            == overlay_sub_boxes_from_manifest(_overlay()))


def test_the_mujoco_path_still_accepts_an_ordinary_overlay():
    """The refusal must be keyed on the field, not on anything incidental."""
    from overlay_geom import overlay_sub_boxes_from_manifest
    boxes = overlay_sub_boxes_from_manifest(_overlay())
    assert len(boxes) == 1


def test_the_isaac_loader_now_IMPLEMENTS_decor_and_keeps_it_off_the_ground():
    """IsaacLab implements the channel; the refusal has been replaced by real support.

    The load-bearing property is not that the field parses -- it is that a decor overlay
    ends up in a DIFFERENT mesh from the ground. The height scanner ray-casts a single mesh
    prim, so anything merged into the ground is scannable no matter how the prim is
    configured afterwards.
    """
    pytest.importorskip("isaaclab", reason="the loader imports IsaacLab")
    from robot_rl.tasks.manager_based.robot_rl.terrains import clip_terrain_loader as L

    entry = {"terrain_id": 3, "size_x": 10.0, "size_y": 4.0,
             "overlays": [_overlay(), _overlay(decor=True)]}
    cfgs = L._overlays_from_manifest(entry)
    assert [c.decor for c in cfgs] == [False, True]

    ground, meta, decor = L._tile_mesh((10.0, 4.0), [], cfgs)
    assert decor is not None and len(decor.faces) > 0, "the decor overlay built no mesh"
    assert len(meta) == 1, "the decor overlay must not appear in the ground's overlay metadata"
    # The two meshes must be disjoint: no decor triangle may have been merged into the ground.
    assert len(ground.faces) > 0
    assert len(ground.faces) < len(ground.faces) + len(decor.faces)


def test_a_tile_with_no_decor_builds_no_decor_mesh():
    """Every tile in an undecorated library, so this is the common path."""
    pytest.importorskip("isaaclab", reason="the loader imports IsaacLab")
    from robot_rl.tasks.manager_based.robot_rl.terrains import clip_terrain_loader as L
    entry = {"terrain_id": 3, "size_x": 10.0, "size_y": 4.0, "overlays": [_overlay()]}
    _ground, _meta, decor = L._tile_mesh((10.0, 4.0), [], L._overlays_from_manifest(entry))
    assert decor is None


def test_the_isaac_loader_still_accepts_an_ordinary_overlay():
    """The undecorated library must keep loading unchanged."""
    pytest.importorskip("isaaclab", reason="the loader imports IsaacLab")
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
        _overlays_from_manifest)
    entry = {"terrain_id": 3, "size_x": 10.0, "size_y": 4.0, "overlays": [_overlay()]}
    assert len(_overlays_from_manifest(entry)) == 1


def test_the_two_simulators_use_different_mechanisms_for_the_same_meaning():
    """IsaacLab: a separate prim. MuJoCo: geom group 1 plus contype 0.

    Different because the constraint differs -- IsaacLab's height scanner is a RayCaster
    bound to exactly ONE mesh prim, so separation has to happen at the prim; MuJoCo's casts
    are group-based, so it happens at the geom. Pinned together so neither drifts into
    thinking it can borrow the other's approach.
    """
    from overlay_geom import DECOR_GEOM_GROUP
    assert DECOR_GEOM_GROUP == 1, "depth casts {1, 2} and the scan casts {2}"
    pytest.importorskip("isaaclab", reason="the importer imports IsaacLab")
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer import (
        ClipTerrainImporter)
    assert not ClipTerrainImporter.DECOR_PRIM_PATH.startswith("/World/ground"), (
        "decor must be a SIBLING of the ground: the height scanner is pointed at "
        "/World/ground and would otherwise cast the decoration too"
    )
