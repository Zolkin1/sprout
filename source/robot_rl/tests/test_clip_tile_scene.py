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
    OVERLAY_KINDS as emitter_kinds,
)
from generate_clip_tile_scene import (  # noqa: E402
    emittable_overlays,
    spawn_point,
    terrain_top_at,
    tile_geoms,
)
from generate_stairs_scene import scene_xml  # noqa: E402
from overlay_geom import OVERLAY_VARIANTS as emitter_variants  # noqa: E402
from overlay_geom import overlay_sub_boxes as vendored_sub_boxes  # noqa: E402
from overlay_geom import overlay_sub_boxes_from_manifest as vendored_from_manifest  # noqa: E402

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
    geoms, floor = tile_geoms(entry, emittable_overlays(entry), origin, infinite_floor=False)
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
    ovs = emittable_overlays(TILE)
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
    ovs = emittable_overlays(TILE)
    sx, sy, _why = spawn_point(TILE, ovs, approach=2.0)
    assert terrain_top_at(ovs, sx, sy) == pytest.approx(0.13)          # on the approach slab
    assert terrain_top_at(ovs, 6.11, 1.6) == pytest.approx(0.13)       # leading tread: flush
    assert terrain_top_at(ovs, 6.33, 1.6) == pytest.approx(0.26)       # the first real riser


def test_the_spawn_lands_on_the_flight_not_the_tile_centre():
    """The flight is off-centre; a tile-centred spawn would silently start the robot beside it."""
    ovs = emittable_overlays(TILE)
    sx, sy, _why = spawn_point(TILE, ovs, approach=2.0)
    assert sy == pytest.approx(1.6)                     # the flight, not 2.0 (the tile centre)
    assert sx == pytest.approx(6.11 - 0.11 - 2.0)       # 2 m before the leading tread's face


#: The same tile with a railing post beside the flight -- narrow in x like a tread, but
#: ``walkable_top=False``. This is the shape every planned wall / railing / clutter overlay
#: has, and before the ``walkable`` filter it hijacked the whole geometry query path.
TILE_WITH_RAILING = dict(
    TILE,
    overlays=TILE["overlays"] + [
        {"kind": "obstacle", "pos_xy": [0.5, 0.3], "size_xyz": [0.1, 0.1, 1.1],
         "base_z": 0.0, "walkable_top": False, "yaw": 0.0},
    ],
)


def test_a_non_walkable_overlay_is_not_mistaken_for_the_first_riser():
    """A railing post is tread-narrow in x; only ``walkable_top`` tells it apart.

    Without the filter ``first_step`` picks the post, so ``spawn_point`` returns the post's
    centreline minus the approach -- measured at 2.05 m OUTSIDE the tile, over the void --
    and the emitter happily writes that scene.
    """
    ovs = emittable_overlays(TILE_WITH_RAILING)
    sx, sy, _why = spawn_point(TILE_WITH_RAILING, ovs, approach=2.0)
    assert sy == pytest.approx(1.6)                     # the flight, not the post at y=0.3
    assert sx == pytest.approx(6.11 - 0.11 - 2.0)       # unchanged by the post


def test_terrain_top_ignores_non_walkable_volumes():
    """Standing height comes from ground, not from the top of a 1.1 m post."""
    ovs = emittable_overlays(TILE_WITH_RAILING)
    assert terrain_top_at(ovs, 0.5, 0.3) == pytest.approx(0.13)   # the approach platform
    assert terrain_top_at(ovs, 6.11, 1.6) == pytest.approx(0.13)  # the leading tread


def test_a_spawn_outside_the_tile_is_refused():
    """The post-condition that pins ``first_step``'s predicate.

    Derived from one overlay's geometry, so a misidentified riser puts the spawn where the
    tile does not exist. An approach longer than the run-up triggers the same check.
    """
    with pytest.raises(ValueError, match="outside the tile footprint"):
        spawn_point(TILE, emittable_overlays(TILE), approach=99.0)


#: One overlay, ``floating``: a 0.08 m top slab at z in [0.52, 0.60] with an OPEN GAP below.
#: As ``solid`` the same entry is a pillar filling z in [0, 0.60]. The two have an identical
#: top face by design, which is exactly why a top-down height field cannot tell them apart.
_FLOATING_TILE = {
    "terrain_id": 21, "type": "box", "size_x": 14.0, "size_y": 4.0,
    "has_floor": True, "frame_yaw": 0.0, "family": "synthetic.h5",
    "overlays": [
        {"kind": "box", "pos_xy": [7.0, 2.0], "size_xyz": [1.0, 1.0, 0.6], "base_z": 0.0,
         "walkable_top": True, "yaw": 0.0, "variant": "floating", "slab_thickness": 0.08},
    ],
}


def _first_hit_x(entry: dict, z: float) -> tuple[float, float]:
    """Distance to the first surface along +x at height ``z``, in (isaac, mujoco) metres.

    A HORIZONTAL ray, deliberately: it is the cheapest probe that separates a floating slab
    from a solid pillar, and the pair of them is what the top-down height field cannot do.
    """
    start = np.array([2.0, 2.0, z])
    mesh = _isaac_mesh(entry)
    loc, _idx, _tri = mesh.ray.intersects_location(
        start[None, :], np.array([[1.0, 0.0, 0.0]]), multiple_hits=False)
    iso = float(loc[0, 0] - start[0]) if len(loc) else float("inf")

    model = _mujoco_model(entry, "", (0.0, 0.0, 0.0))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gg = np.zeros(6, dtype=np.uint8)
    gg[2] = 1
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, start, np.array([1.0, 0.0, 0.0]), gg, 1, -1, gid)
    return iso, (float(dist) if dist >= 0 else float("inf"))


def test_a_floating_overlay_is_a_slab_on_a_gap_in_both_sims():
    """The emitter must build the gap, not the silhouette.

    Before ``overlay_sub_boxes`` reached the MuJoCo side this tile could not be emitted at
    all here, and ``compare_sim2sim`` emitted it as a solid pillar -- so the depth cameras saw
    two different worlds while every top-down check agreed.
    """
    iso, mj = _first_hit_x(_FLOATING_TILE, z=0.25)          # through the gap
    assert iso == mj == float("inf"), (
        f"a ray through the gap under a floating slab hit something: isaac={iso}, mujoco={mj}"
    )
    iso_top, mj_top = _first_hit_x(_FLOATING_TILE, z=0.56)  # through the slab itself
    assert iso_top == pytest.approx(mj_top, abs=1e-6)
    assert iso_top == pytest.approx(4.5, abs=1e-6)          # 6.5 - 2.0, the slab's -x face


def test_the_gap_probe_would_catch_a_solid_pillar():
    """The discriminator has to discriminate, or the test above proves nothing."""
    solid = dict(_FLOATING_TILE,
                 overlays=[dict(_FLOATING_TILE["overlays"][0], variant="solid")])
    iso, mj = _first_hit_x(solid, z=0.25)
    assert iso == pytest.approx(4.5, abs=1e-6)              # the pillar blocks the ray
    assert mj == pytest.approx(4.5, abs=1e-6)


def test_an_unknown_overlay_variant_is_refused():
    """A new variant must not be silently flattened to its bounding box."""
    entry = dict(TILE, overlays=[dict(TILE["overlays"][0], variant="cantilever")])
    with pytest.raises(ValueError, match="variant='cantilever'"):
        emittable_overlays(entry)


def test_the_emitter_and_the_loader_agree_on_the_overlay_variants():
    """The vendored decomposition and the IsaacLab one must know the same variants."""
    from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.overlay import overlay_sub_boxes

    for variant in emitter_variants:
        assert overlay_sub_boxes(variant, (1.0, 1.0, 1.0), (0.0, 0.0), 0.0, 0.1, (0.1, 0.1), 0.05)


@pytest.mark.parametrize("variant", ["solid", "floating", "floating_center_leg",
                                     "floating_corner_legs"])
@pytest.mark.parametrize("sz,slab", [(0.4, 0.1), (0.13, 0.05), (0.06, 0.2), (1.0, 0.03)])
def test_the_vendored_decomposition_matches_the_isaaclab_one(variant, sz, slab):
    """``transfer/sim/overlay_geom.py`` is a port, held to the original by this test.

    ``transfer/`` runs under ``env_transfer``, which has no IsaacLab, so the MuJoCo emitters
    cannot import ``blocks/overlay.py``. Same situation as ``deploy/quat_math_np.py`` vs
    ``torch_math/quat_math.py``, and the same remedy: two copies, one parity test.

    The ``sz=0.06, slab=0.2`` case is the slab-taller-than-the-box clamp; ``sz=0.13`` is a
    real shipped tread height.
    """
    from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.overlay import overlay_sub_boxes

    args = (variant, (0.9, 1.3, sz), (2.5, -1.25), 0.07, slab, (0.11, 0.09), 0.04)
    assert vendored_sub_boxes(*args) == overlay_sub_boxes(*args)


def test_the_vendored_manifest_reader_matches_the_isaaclab_loader():
    """And the manifest-dict entry point defaults its keys the same way the loader does."""
    from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.overlay import overlay_sub_boxes

    ov = {"kind": "box", "pos_xy": [3.0, 1.5], "size_xyz": [0.8, 1.1, 0.45],
          "base_z": 0.12, "walkable_top": True, "variant": "floating_corner_legs",
          "slab_thickness": 0.06, "leg_size": [0.12, 0.1], "leg_inset": 0.03}
    cfg = _loader._overlays_from_manifest({"terrain_id": 1, "overlays": [ov]})[0]
    expected = overlay_sub_boxes(cfg.variant, tuple(cfg.size_xyz), tuple(cfg.pos_xy),
                                 cfg.base_z, cfg.slab_thickness, tuple(cfg.leg_size),
                                 cfg.leg_inset)
    assert vendored_from_manifest(ov) == expected

    # ...and with every optional key absent, where the two default sets must also agree.
    bare = {"pos_xy": [1.0, 2.0], "size_xyz": [0.5, 0.5, 0.3]}
    bare_cfg = _loader._overlays_from_manifest({"terrain_id": 1, "overlays": [bare]})[0]
    assert vendored_from_manifest(bare) == overlay_sub_boxes(
        bare_cfg.variant, tuple(bare_cfg.size_xyz), tuple(bare_cfg.pos_xy), bare_cfg.base_z,
        bare_cfg.slab_thickness, tuple(bare_cfg.leg_size), bare_cfg.leg_inset)


def test_an_unknown_overlay_kind_is_refused():
    """A new shape name must not be silently flattened into a box."""
    entry = dict(TILE, overlays=[dict(TILE["overlays"][0], kind="cylinder")])
    with pytest.raises(ValueError, match="kind='cylinder'"):
        emittable_overlays(entry)


def test_the_emitter_and_the_loader_agree_on_the_overlay_kinds():
    """One vocabulary, two modules -- ``transfer/`` cannot import the IsaacLab-side one."""
    from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import OVERLAY_KINDS as isaac_kinds

    assert set(emitter_kinds) == set(isaac_kinds)


def test_the_manifest_is_the_shared_source():
    """Both sides are built from the same dicts -- this pins that they still parse alike."""
    mine = emittable_overlays(TILE)
    theirs = _loader._overlays_from_manifest(TILE)
    assert len(mine) == len(theirs)
    for ov, cfg in zip(mine, theirs):
        assert tuple(ov["size_xyz"]) == pytest.approx(tuple(cfg.size_xyz))
        assert tuple(ov["pos_xy"]) == pytest.approx(tuple(cfg.pos_xy))
        assert float(ov.get("base_z", 0.0)) == pytest.approx(cfg.base_z)


def test_the_json_round_trip_does_not_change_the_geometry():
    """The manifest travels as a JSON string in the .h5; floats must survive it."""
    assert emittable_overlays(json.loads(json.dumps(TILE))) == emittable_overlays(TILE)
