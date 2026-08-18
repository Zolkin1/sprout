# Copyright 2026 Zachary Olkin. All rights reserved.

"""``generate_clip_tile_scene.py`` must emit the tile IsaacLab would build, not one like it.

The whole point of a single-tile scene is that it is the same staircase the policy trained
on, so "close enough" is the one thing it may not be.  This casts a height field over both
sides -- MuJoCo's ``mj_ray`` against the emitted scene, trimesh's ray intersector against
``clip_terrain_loader``'s own tile mesh -- and requires them to agree everywhere.

Everywhere except the box FACE PLANES: at a point exactly on a vertical face the two casters
break the tie differently (trimesh takes the slab top, MuJoCo the floor below), which is a
measure-zero disagreement about a shared edge and not a geometry error.  On the shipped
stairs_up tile that is 447 of 165924 sampled points, every one of them on a face; the rest
agree to 1e-8 m.  The test samples off the faces and asserts a face-plane point is the ONLY
thing it had to exclude.

The tile is synthesised here rather than read from ``clips/``: the fixture is then hermetic,
fast, and covers the shapes that matter (a leading tread flush with the approach platform, an
off-centre flight, a yawed overlay) rather than whichever library happens to be on disk.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SIM = os.path.join(_ROOT, "transfer", "sim")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

mujoco = pytest.importorskip("mujoco")
trimesh = pytest.importorskip("trimesh")

from generate_clip_tile_scene import (  # noqa: E402
    solid_overlays,
    spawn_point,
    terrain_top_at,
    tile_geoms,
)
from generate_stairs_scene import scene_xml  # noqa: E402

# Isaac's own tile builder. Needs isaaclab importable (not IsaacSim running).
_loader = pytest.importorskip(
    "robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader"
)

#: A transition tile shaped like the shipped ``family_stairs_up`` one: an approach platform,
#: a flight whose leading tread is FLUSH with it, and a landing -- with the flight off the
#: tile centreline, which is the offset a tile-centred spawn would hide.
TILE = {
    "terrain_id": 7, "type": "transition", "size_x": 14.0, "size_y": 4.0,
    "has_floor": True, "frame_yaw": 0.0, "family": "synthetic_stairs_up.h5",
    "overlays": [
        {"kind": "box", "pos_xy": [3.0, 2.0], "size_xyz": [6.0, 4.0, 0.13],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0},
        {"kind": "box", "pos_xy": [6.11, 1.6], "size_xyz": [0.22, 0.95, 0.13],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0},
        {"kind": "box", "pos_xy": [6.33, 1.6], "size_xyz": [0.22, 0.95, 0.26],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0},
        {"kind": "box", "pos_xy": [9.5, 2.0], "size_xyz": [6.0, 4.0, 0.26],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0},
    ],
}


def _isaac_mesh(entry: dict):
    """The tile mesh ``clip_terrain_loader`` builds from this entry, in tile-local coords."""
    footprint = (float(entry["size_x"]), float(entry["size_y"]))
    blocks = ([_loader.FlatBlockCfg(size_x=footprint[0], skill_probs={"standing": 1.0})]
              if entry.get("has_floor", True) else [])
    mesh, _meta = _loader._tile_mesh(footprint, blocks,
                                     _loader._overlays_from_manifest(entry))
    return mesh


def _mujoco_model(entry: dict, tmpdir: str, origin):
    """Write the emitted scene next to the robot assets and load it.

    The scene ``<include>``s the robot model, and MuJoCo resolves that relative to the
    scene file, so it cannot live in a bare tmp dir without copying the meshes too.
    """
    geoms, floor = tile_geoms(entry, solid_overlays(entry), origin, infinite_floor=False)
    xml = scene_xml("g1_29j.xml", "pytest fixture", "\n".join(geoms), floor)
    path = os.path.join(_SIM, "robots", "g1", "_pytest_tile_scene.xml")
    with open(path, "w") as f:
        f.write(xml)
    try:
        model = mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)
    return model


def _face_planes(entry: dict) -> tuple[set[float], set[float]]:
    """The x and y planes of every overlay face, where the two casters may legitimately tie."""
    fx, fy = set(), set()
    for ov in entry["overlays"]:
        (px, py), (sx, sy, _sz) = ov["pos_xy"], ov["size_xyz"]
        fx |= {round(px - sx / 2, 6), round(px + sx / 2, 6)}
        fy |= {round(py - sy / 2, 6), round(py + sy / 2, 6)}
    return fx, fy


@pytest.fixture(scope="module")
def height_fields():
    """``(local_xy, isaac_z, mujoco_z)`` over a grid spanning the tile."""
    ovs = solid_overlays(TILE)
    sx, sy, _why = spawn_point(TILE, ovs, approach=2.0)
    sz = terrain_top_at(ovs, sx, sy)
    model = _mujoco_model(TILE, "", (sx, sy, sz))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    xs = np.unique(np.concatenate([np.arange(0.3, 13.7, 0.05), np.arange(5.8, 6.8, 0.01)]))
    ys = np.unique(np.concatenate([np.arange(0.3, 3.7, 0.05), np.arange(1.0, 2.2, 0.01)]))
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    mesh = _isaac_mesh(TILE)
    origins = np.column_stack([grid, np.full(len(grid), 50.0)])
    dirs = np.tile([0.0, 0.0, -1.0], (len(grid), 1))
    loc, ray_idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    iso = np.full(len(grid), np.nan)
    iso[ray_idx] = loc[:, 2]

    gg = np.zeros(6, dtype=np.uint8)
    gg[2] = 1
    gid = np.array([-1], dtype=np.int32)
    mj = np.full(len(grid), np.nan)
    for i, (px, py) in enumerate(grid):
        dist = mujoco.mj_ray(model, data, np.array([px - sx, py - sy, 50.0]),
                             np.array([0.0, 0.0, -1.0]), gg, 1, -1, gid)
        if dist >= 0:
            mj[i] = 50.0 - dist + sz          # world z -> tile-local z
    return grid, iso, mj


def test_the_emitted_scene_has_no_holes(height_fields):
    """A floored tile is ground everywhere over its footprint -- a hole is a missing geom.

    Asserted on the MuJoCo side only. The trimesh intersector drops the occasional ray that
    passes exactly through a shared triangle EDGE (the floor quad's diagonal costs it one ray
    in ~70k here), which says nothing about the mesh; ``mj_ray`` has no such gap, so the
    emitted scene is held to the stricter standard.
    """
    _grid, iso, mj = height_fields
    assert np.isfinite(mj).all(), "the emitted scene has holes over the tile footprint"
    missed = int((~np.isfinite(iso)).sum())
    assert missed < 0.001 * iso.size, (
        f"the Isaac tile mesh missed {missed}/{iso.size} rays -- far more than the stray "
        "shared-edge miss, so this is the mesh and not the intersector"
    )


def test_height_fields_agree_off_the_face_planes(height_fields):
    """The geometry is the same geometry, to mesh precision."""
    grid, iso, mj = height_fields
    fx, fy = _face_planes(TILE)
    off = np.array([not (any(abs(p[0] - v) < 1e-6 for v in fx)
                         or any(abs(p[1] - v) < 1e-6 for v in fy)) for p in grid])
    off &= np.isfinite(iso) & np.isfinite(mj)
    assert off.sum() > 0.9 * len(grid), "excluded far more than the face planes"
    assert np.abs(iso[off] - mj[off]).max() < 1e-6


def test_every_disagreement_is_a_face_plane(height_fields):
    """The exclusion above must not be hiding a real difference somewhere else."""
    grid, iso, mj = height_fields
    fx, fy = _face_planes(TILE)
    bad = (np.abs(iso - mj) > 1e-6) & np.isfinite(iso) & np.isfinite(mj)
    for point in grid[bad]:
        assert (any(abs(point[0] - v) < 1e-6 for v in fx)
                or any(abs(point[1] - v) < 1e-6 for v in fy)), \
            f"height fields differ at {point}, which is not on any overlay face"


def test_the_climb_starts_at_the_second_tread(height_fields):
    """The leading tread is FLUSH with the approach platform on tiles shaped like this.

    Counting step-shaped overlays says three risers on the shipped tile; the ground says two.
    A scene that got this wrong would still look like a staircase.
    """
    ovs = solid_overlays(TILE)
    sx, sy, _why = spawn_point(TILE, ovs, approach=2.0)
    assert terrain_top_at(ovs, sx, sy) == pytest.approx(0.13)          # on the approach slab
    assert terrain_top_at(ovs, 6.11, 1.6) == pytest.approx(0.13)       # leading tread: flush
    assert terrain_top_at(ovs, 6.33, 1.6) == pytest.approx(0.26)       # the first real riser


def test_the_spawn_lands_on_the_flight_not_the_tile_centre():
    """The flight is off-centre; a tile-centred spawn would silently start the robot beside it."""
    ovs = solid_overlays(TILE)
    sx, sy, _why = spawn_point(TILE, ovs, approach=2.0)
    assert sy == pytest.approx(1.6)                     # the flight, not 2.0 (the tile centre)
    assert sx == pytest.approx(6.11 - 0.11 - 2.0)       # 2 m before the leading tread's face


def test_an_overlay_this_cannot_reproduce_is_refused():
    """A floating overlay is a slab on legs; its bounding box is not the same terrain."""
    entry = dict(TILE, overlays=[dict(TILE["overlays"][0], variant="floating")])
    with pytest.raises(ValueError, match=r"cannot reproduce exactly"):
        solid_overlays(entry)


def test_the_manifest_is_the_shared_source():
    """Both sides are built from the same dicts -- this pins that they still parse alike."""
    mine = solid_overlays(TILE)
    theirs = _loader._overlays_from_manifest(TILE)
    assert len(mine) == len(theirs)
    for ov, cfg in zip(mine, theirs):
        assert tuple(ov["size_xyz"]) == pytest.approx(tuple(cfg.size_xyz))
        assert tuple(ov["pos_xy"]) == pytest.approx(tuple(cfg.pos_xy))
        assert float(ov.get("base_z", 0.0)) == pytest.approx(cfg.base_z)


def test_the_json_round_trip_does_not_change_the_geometry():
    """The manifest travels as a JSON string in the .h5; floats must survive it."""
    assert solid_overlays(json.loads(json.dumps(TILE))) == solid_overlays(TILE)
