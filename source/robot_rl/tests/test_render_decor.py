# Copyright 2026 Zachary Olkin. All rights reserved.

"""``render_decor=False`` must drop decoration and NOTHING else.

The teacher trains with visual-only decoration off; the recorder and everything with the
generator in the loop keep it on. One clip library serves both, so the flag has to be a pure
render-time filter: if it moved so much as one ground vertex, a decor-off teacher and a
decor-on recording would be walking on different worlds -- and nothing downstream could tell,
because ``terrain_fingerprint`` hashes the MANIFEST and both runs hash the same.

So the assertions here are equalities, not smoke: the walkable mesh byte-identical, the tile
corners and origins byte-identical, the overlay metadata unchanged. Separately, the PHYSICAL
variation the decoration pass shares a manifest with -- flight widths, landing widths, the box
support variants -- must still reach the ground, since that is a rewrite of what the robot
stands on and is not decoration at all.

Manifests are built here rather than read from a library: the shipped one is 6.8 GB, and a
synthetic tile can carry the case that matters (decor and walkable overlays on the same tile).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab", reason="the loader imports IsaacLab utils")

from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (  # noqa: E402
    assemble_clip_terrain,
)


def _walkable(pos_xy, size_xyz, *, variant: str = "solid", base_z: float = 0.0) -> dict:
    """One walkable overlay (a tread, a landing platform, a jump-on box)."""
    return {"kind": "box", "pos_xy": list(pos_xy), "size_xyz": list(size_xyz),
            "base_z": base_z, "walkable_top": True, "yaw": 0.0, "variant": variant}


def _wall(pos_xy, size_xyz) -> dict:
    """One decoration overlay -- what ``decorate.add_walls`` writes."""
    return {"kind": "box", "pos_xy": list(pos_xy), "size_xyz": list(size_xyz),
            "base_z": 0.0, "walkable_top": False, "yaw": 0.0, "decor": True}


def _pillar(pos_xy, size_xyz) -> dict:
    """A decoration CYLINDER -- the kind only the decor path can mesh."""
    return {"kind": "cylinder", "pos_xy": list(pos_xy), "size_xyz": list(size_xyz),
            "base_z": 0.0, "walkable_top": False, "yaw": 0.0, "decor": True}


def _manifest(flight_w: float = 1.2, support: str = "solid") -> list[dict]:
    """Three tiles, each carrying walkable geometry AND decoration.

    ``flight_w`` / ``support`` are the PHYSICAL knobs the decoration pass rewrites alongside
    the walls (tread width and the box support variant), so varying them is how the tests
    below prove the flag left the ground alone rather than left everything alone.
    """
    treads = [_walkable((0.6 + 0.3 * i, 2.0), (0.3, flight_w, 0.15 * (i + 1))) for i in range(4)]
    return [
        # A transition tile: flat ground + a flight + a landing, walled on two sides.
        {"terrain_id": 0, "type": "transition", "size_x": 8.0, "size_y": 4.0,
         "has_floor": True, "frame_yaw": 0.0,
         "overlays": treads + [_walkable((3.0, 2.0), (2.0, flight_w, 0.6)),
                               _wall((4.0, 0.1), (8.0, 0.2, 1.5)),
                               _wall((4.0, 3.9), (8.0, 0.2, 1.5))]},
        # A box tile: the support variant is PHYSICAL (what holds the top up), not decoration.
        {"terrain_id": 1, "type": "box", "size_x": 6.0, "size_y": 4.0,
         "has_floor": True, "frame_yaw": 0.0,
         "overlays": [_walkable((3.0, 2.0), (1.2, 1.2, 0.45), variant=support),
                      _pillar((1.0, 1.0), (0.3, 0.3, 2.0))]},
        # A floorless pure-stair tile: with no ground plane, dropping decoration must not
        # leave the tile empty of the treads that ARE its ground.
        {"terrain_id": 2, "type": "terrain", "size_x": 5.0, "size_y": 3.0,
         "has_floor": False, "frame_yaw": 0.0,
         "overlays": treads + [_wall((2.5, 0.1), (5.0, 0.2, 1.5))]},
    ]


#: One room variant for the flat tiles -- ``decorate.make_flat_rooms``' output shape.
#: Positions are relative to the tile CENTRE, which is what the loader adds back.
_FLAT_ROOMS = [[{"kind": "box", "pos_xy": [0.0, -10.0], "size_xyz": [20.0, 0.2, 2.0],
                 "base_z": 0.0, "walkable_top": False, "yaw": 0.0, "decor": True},
                {"kind": "box", "pos_xy": [-10.0, 0.0], "size_xyz": [0.2, 20.0, 2.0],
                 "base_z": 0.0, "walkable_top": False, "yaw": 0.0, "decor": True}],
               []]


def _fake_lib(manifest: list[dict], *, flat_decor: list | None = None):
    """Minimal stand-in for a loaded clip library (manifest + one clip per tile + one flat)."""
    tids = [int(m["terrain_id"]) for m in manifest] + [-1]
    attrs: dict = {"terrain_manifest": manifest}
    if flat_decor is not None:
        attrs["flat_tile_decor"] = flat_decor
    return types.SimpleNamespace(
        attrs=attrs,
        terrain_id=torch.tensor(tids, dtype=torch.int32),
        y_des=torch.zeros(len(tids), 5, 4),
        meta={"pos_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y",
                                   "pelvis_link:pos_z", "j0"]},
    )


def _build(manifest, *, render_decor: bool, flat_decor=None, n_flat: int = 2):
    """Assemble once. Returns ``(mesh, origins, layout)``."""
    return assemble_clip_terrain(
        _fake_lib(manifest, flat_decor=flat_decor),
        num_rows=2, num_flat_tiles=n_flat, render_decor=render_decor,
    )


def _bytes(mesh) -> tuple[bytes, bytes]:
    """The mesh as raw vertex + face bytes, for an exact comparison."""
    return (np.asarray(mesh.vertices).tobytes(), np.asarray(mesh.faces).tobytes())


##
# (a) decoration is dropped, and the walkable ground is byte-identical
##


def test_decoration_is_built_when_the_flag_is_on():
    """The control: without this the rest of the file could pass on a manifest with no decor."""
    _mesh, _origins, layout = _build(_manifest(), render_decor=True)
    assert layout.decor_mesh is not None
    assert len(layout.decor_mesh.faces) > 0


def test_decoration_is_dropped_when_the_flag_is_off():
    """``decor_mesh is None`` is what makes ``create_decor_prim`` fall back to the placeholder."""
    _mesh, _origins, layout = _build(_manifest(), render_decor=False)
    assert layout.decor_mesh is None


def test_the_walkable_ground_is_byte_identical():
    """The whole contract. A decor-off teacher and a decor-on recording share one world."""
    on_mesh, _on_o, _on_l = _build(_manifest(), render_decor=True)
    off_mesh, _off_o, _off_l = _build(_manifest(), render_decor=False)
    assert _bytes(on_mesh) == _bytes(off_mesh)


def test_the_ground_mesh_did_not_merely_shrink_to_the_decor():
    """Guards the degenerate way the equality above could hold: both meshes empty.

    The decoration is a large fraction of the vertex count on these tiles, so an
    implementation that dropped the WALKABLE overlays instead would still produce two equal
    meshes -- equal and wrong.
    """
    off_mesh, _o, _l = _build(_manifest(), render_decor=False)
    # The four treads, the landing and the box all reach above the ground plane.
    assert float(off_mesh.bounds[1][2]) == pytest.approx(0.6)
    assert len(off_mesh.faces) > 0


def test_the_overlay_metadata_is_unchanged():
    """``terrain_overlays`` feeds the Phase-3 reset; decoration was never in it either way."""
    _m_on, _o_on, on = _build(_manifest(), render_decor=True)
    _m_off, _o_off, off = _build(_manifest(), render_decor=False)
    assert on.terrain_overlays == off.terrain_overlays
    for tile in on.terrain_overlays.values():
        assert all(not ov.get("decor", False) for ov in tile)


##
# (b) the flat-tile rooms live in a different attribute, and must go too
##


def test_flat_rooms_are_built_when_the_flag_is_on():
    """The flat rooms are the one decor source that is NOT in ``terrain_manifest``."""
    _mesh, _origins, layout = _build(_manifest(), render_decor=True, flat_decor=_FLAT_ROOMS)
    assert layout.decor_mesh is not None


def test_flat_rooms_are_dropped_when_the_flag_is_off():
    """With no tile decoration at all, the rooms alone must not resurrect the decor mesh."""
    bare = [dict(m, overlays=[o for o in m["overlays"] if not o.get("decor")])
            for m in _manifest()]
    _m_on, _o_on, on = _build(bare, render_decor=True, flat_decor=_FLAT_ROOMS)
    assert on.decor_mesh is not None, "the fixture must actually carry rooms"
    _m_off, _o_off, off = _build(bare, render_decor=False, flat_decor=_FLAT_ROOMS)
    assert off.decor_mesh is None


def test_flat_rooms_do_not_touch_the_flat_ground():
    """A room is walls on a bare plane: the plane is the same plane with or without them."""
    on_mesh, _o, _l = _build(_manifest(), render_decor=True, flat_decor=_FLAT_ROOMS)
    off_mesh, _o2, _l2 = _build(_manifest(), render_decor=False, flat_decor=_FLAT_ROOMS)
    assert _bytes(on_mesh) == _bytes(off_mesh)


@pytest.mark.parametrize("bad,match", [
    ({"kind": "wedge"}, "kind="),
    ({"yaw": 0.785}, "yaw="),
], ids=["unknown_kind", "nonzero_yaw"])
def test_a_malformed_flat_room_is_refused_at_both_flag_settings(bad, match):
    """``flat_tile_decor`` content is validated whether or not it is going to be drawn.

    ``_flat_decor_overlays`` is the only place a flat room passes through
    ``_overlays_from_manifest``, which is the only thing that refuses an unknown ``kind`` or a
    non-zero ``yaw``. An early return on ``render_decor`` there -- the shape this had when it
    landed -- puts that validation behind the flag: a malformed library is ACCEPTED by the
    decor-off teacher and then raises hours later on the recorder or the eval, which run with
    decoration on. Manifest-borne decor never had that asymmetry, so the two decor sources
    must agree here.

    The rooms are therefore built and validated unconditionally and ``_tile_mesh`` does the
    dropping -- which is also the only correct place, since a ``wall_collidable`` room
    (``decor=False``) is walkable, so dropping it here would move the ground mesh.
    """
    rooms = [[dict(_FLAT_ROOMS[0][0], **bad)], []]
    for render_decor in (True, False):
        with pytest.raises(ValueError, match=match):
            _build(_manifest(), render_decor=render_decor, flat_decor=rooms)


##
# (c) the PHYSICAL variation the decoration pass shares a manifest with survives
##


@pytest.mark.parametrize("support", ["solid", "floating", "floating_center_leg",
                                     "floating_corner_legs"])
def test_the_box_support_variant_still_reaches_the_ground(support):
    """Support structure is what holds the walkable top up -- ground, not decoration."""
    solid_mesh, _o, _l = _build(_manifest(support="solid"), render_decor=False)
    mesh, _o2, _l2 = _build(_manifest(support=support), render_decor=False)
    if support == "solid":
        assert _bytes(mesh) == _bytes(solid_mesh)
    else:
        assert _bytes(mesh) != _bytes(solid_mesh), (
            f"{support!r} produced the same ground as 'solid' -- the support variant was lost"
        )
    # The walkable top face is at base_z + sz for EVERY variant.
    assert float(mesh.bounds[1][2]) == pytest.approx(0.6)


def test_the_flight_width_still_reaches_the_ground():
    """Tread / landing WIDTH is a manifest rewrite of the walkable mesh, not decoration."""
    narrow, _o, _l = _build(_manifest(flight_w=0.86), render_decor=False)
    wide, _o2, _l2 = _build(_manifest(flight_w=1.96), render_decor=False)
    assert _bytes(narrow) != _bytes(wide)


def test_the_physical_variation_is_identical_with_decor_on():
    """Each physical arm must be the same ground whether or not decoration is drawn."""
    for kwargs in ({"flight_w": 0.86}, {"flight_w": 1.96},
                   {"support": "floating_corner_legs"}):
        on, _o, _l = _build(_manifest(**kwargs), render_decor=True)
        off, _o2, _l2 = _build(_manifest(**kwargs), render_decor=False)
        assert _bytes(on) == _bytes(off), kwargs


##
# (d) placement: every tile sits where it sat
##


def test_tile_corners_and_origins_are_unchanged():
    """Clip placement is ``clip_frame_origin(tile_corner)``, so a shifted corner moves motion.

    The centring offset is taken from the GROUND mesh's bounds, and the decor mesh is moved by
    that same offset rather than by its own -- so a decor mesh that reached past the ground
    (walls do, by half their thickness) could otherwise re-centre the whole field.
    """
    _m_on, on_origins, on = _build(_manifest(), render_decor=True, flat_decor=_FLAT_ROOMS)
    _m_off, off_origins, off = _build(_manifest(), render_decor=False, flat_decor=_FLAT_ROOMS)
    assert torch.equal(on_origins, off_origins)
    assert torch.equal(on.tile_corner, off.tile_corner)
    assert torch.equal(on.tile_footprint, off.tile_footprint)
    assert torch.equal(on.tile_terrain_id, off.tile_terrain_id)
    assert torch.equal(on.tile_clip_x_offset, off.tile_clip_x_offset)
    assert on.clip_size_y == off.clip_size_y
    assert on.terrain_type_code == off.terrain_type_code
    assert on.terrain_frame_yaw == off.terrain_frame_yaw
    assert (on.num_rows, on.num_tiles) == (off.num_rows, off.num_tiles)


##
# The cfg side: the default, and the sidecar refusal
##


def test_the_cfg_default_is_todays_behaviour():
    """Every existing consumer -- viewers, the recorder, sim2sim emitters -- keeps decoration."""
    import inspect

    from robot_rl.tasks.manager_based.robot_rl.terrains import clip_terrain_loader as L
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer_cfg import (
        ClipTerrainImporterCfg)

    assert ClipTerrainImporterCfg().render_decor is True
    assert inspect.signature(L.assemble_clip_terrain).parameters["render_decor"].default is True


def test_the_teacher_trains_without_decoration_and_plays_with_it():
    """The one asymmetry that is deliberate: the ``-play`` tasks are what the recorder records.

    ``record_data.py`` documents recording from a PLAY task, and the generator's own difficulty
    is that it has barely seen a vertical surface -- so the recorder needs the walls the
    12k-iteration teacher run does not.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_terrain_env_cfg import (
        G1TeacherTerrainEnvCfg, G1TeacherTerrainEnvCfgPlay)

    assert G1TeacherTerrainEnvCfg().scene.terrain.render_decor is False
    assert G1TeacherTerrainEnvCfgPlay().scene.terrain.render_decor is True


def test_a_clutter_sidecar_with_decor_off_is_refused():
    """Schema 1 adds ONLY decor boxes, so the pair is a contradiction, not a no-op.

    Left unchecked the run is byte-identical to one with no sidecar while the merge still
    prints "+N visual-only boxes" -- a clutter A/B that silently compares nothing.
    """
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer import (
        assert_sidecar_is_rendered)

    with pytest.raises(ValueError, match="render_decor=False"):
        assert_sidecar_is_rendered({"schema": 1, "per_terrain_id": {}}, False, "clutter.json")
    # A sidecar with no schema key is schema 1 by the importer's own default.
    with pytest.raises(ValueError, match="render_decor=False"):
        assert_sidecar_is_rendered({"per_terrain_id": {}}, False, "clutter.json")


def test_a_ground_variant_sidecar_with_decor_off_is_allowed():
    """Schema 2 edits the WALKABLE ground, which this flag does not touch."""
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer import (
        assert_sidecar_is_rendered)

    assert_sidecar_is_rendered({"schema": 2, "per_terrain_id": {}}, False, "variant.json")
    assert_sidecar_is_rendered({"schema": 1, "per_terrain_id": {}}, True, "clutter.json")
