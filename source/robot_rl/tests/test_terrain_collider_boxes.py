# Copyright 2026 Zachary Olkin. All rights reserved.

"""The RL importer's collider boxes must equal the ones read straight from the manifest.

There are two paths to a tile's collision geometry and the terrain-consistency reward depends
on them agreeing:

* the RL loader builds them from the ``OverlayVolumeCfg`` / block CFGS it assembled the mesh
  from (``clip_terrain_loader._tile_collider_boxes``);
* ``motion_gen.terrain.overlay_boxes.collider_boxes_from_manifest_tile`` builds them from the
  manifest dict, which is what the offline inspector and the viser viewer read.

If those diverge, the reward grades a plan against different geometry in Isaac than the tool
that verified it -- and both would look right. So they are checked against each other on real
library manifests rather than each being checked against itself.

The loader path deliberately does NOT go through ``composite_terrain``'s overlay metadata,
because that drops ``variant``: a ``floating`` tread would come back a solid pillar and the
reward would charge a robot for walking under it. That is what the floating case pins.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from motion_gen import features as F
from motion_gen.terrain.overlay_boxes import collider_boxes_from_manifest_tile

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import (
    FlatBlockCfg,
    OverlayVolumeCfg,
    StairBlockCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
    _tile_collider_boxes,
)

_TRIVIAL = {"flat": 1.0}


def _sorted(a: np.ndarray) -> np.ndarray:
    """Row-sorted copy, so the two builders may emit boxes in any order."""
    return np.array(sorted(a.tolist()))


def _overlay_cfgs(entry: dict) -> list[OverlayVolumeCfg]:
    """The manifest's overlays as the cfgs the RL loader would have built."""
    return [OverlayVolumeCfg(
        kind=str(o["kind"]), size_xyz=tuple(o["size_xyz"]), pos_xy=tuple(o["pos_xy"]),
        base_z=float(o.get("base_z", 0.0)),
        walkable_top=bool(o.get("walkable_top", True)),
        decor=bool(o.get("decor", False)), variant=str(o.get("variant", "solid")),
        slab_thickness=float(o.get("slab_thickness", 0.1)),
        leg_size=tuple(o.get("leg_size", (0.1, 0.1))),
        leg_inset=float(o.get("leg_inset", 0.05)),
    ) for o in entry["overlays"]]


def _libraries():
    """Clip libraries on this machine that carry a terrain manifest."""
    out = []
    for name in ("box2_train.h5", "box2_v2_train.h5", "g1_mb_v5_body_mode11_decor_peek.h5"):
        path = F._REPO_ROOT / "clips" / name
        if path.exists():
            out.append(path)
    return out


@pytest.mark.parametrize("lib", _libraries(), ids=lambda p: p.name)
def test_the_two_builders_agree_on_every_tile_of_a_real_library(lib):
    """The load-bearing check: same tile, same boxes, from cfgs and from the manifest."""
    with h5py.File(lib, "r") as f:
        raw = f.attrs["terrain_manifest"]
    manifest = json.loads(raw if isinstance(raw, str) else raw.decode())
    checked = 0
    for entry in manifest:
        if "overlays" not in entry:
            continue                                   # legacy schema; refused by both
        want = collider_boxes_from_manifest_tile(entry)
        footprint = (float(entry["size_x"]), float(entry["size_y"]))
        blocks = ([FlatBlockCfg(size_x=footprint[0], skill_probs=_TRIVIAL)]
                  if entry.get("has_floor", True) else [])
        got = _tile_collider_boxes(footprint, blocks, _overlay_cfgs(entry))
        assert got is not None, entry["terrain_id"]
        assert np.allclose(_sorted(got), _sorted(want), atol=1e-12), entry["terrain_id"]
        checked += 1
    assert checked > 0, f"{lib.name} exercised no overlay tiles"


def test_a_floating_overlay_contributes_its_SLAB_and_not_a_solid_pillar():
    """The reason the loader reads the cfgs and not ``composite_terrain``'s metadata.

    That metadata carries no ``variant``, so a floating tread would decompose as one solid box
    and the reward would charge a plan for passing through the gap the tread floats over.
    """
    ov = OverlayVolumeCfg(kind="box", size_xyz=(1.0, 1.0, 0.5), pos_xy=(0.0, 0.0),
                          base_z=0.0, variant="floating", slab_thickness=0.1)
    got = _tile_collider_boxes((4.0, 4.0), [], [ov])
    assert got.shape == (1, 6)
    assert got[0, 2] == pytest.approx(0.45)            # slab centre, top face still at 0.5
    assert got[0, 5] == pytest.approx(0.1)             # ...and it is a SLAB, not 0.5 thick


def test_decor_overlays_are_not_collision_geometry():
    """Decoration has no collider in either simulator, so it must not become ground."""
    solid = OverlayVolumeCfg(kind="box", size_xyz=(1.0, 1.0, 0.4), pos_xy=(1.0, 1.0))
    decor = OverlayVolumeCfg(kind="box", size_xyz=(1.0, 1.0, 2.0), pos_xy=(3.0, 3.0),
                             decor=True)
    with_decor = _tile_collider_boxes((8.0, 8.0), [], [solid, decor])
    without = _tile_collider_boxes((8.0, 8.0), [], [solid])
    assert np.allclose(_sorted(with_decor), _sorted(without))


def test_a_legacy_block_schema_is_REFUSED_not_given_a_flat_floor():
    """A flat floor under a staircase is the defect that retired the FM's terrain loss."""
    assert _tile_collider_boxes(
        (20.0, 4.0), [StairBlockCfg(size_x=20.0, skill_probs=_TRIVIAL)], []) is None


def test_has_floor_decides_whether_a_ground_slab_exists():
    """An isolated staircase must have a void beside it, not an implied floor."""
    ov = OverlayVolumeCfg(kind="box", size_xyz=(1.0, 1.0, 0.4), pos_xy=(2.0, 2.0))
    with_floor = _tile_collider_boxes((8.0, 8.0),
                                      [FlatBlockCfg(size_x=8.0, skill_probs=_TRIVIAL)], [ov])
    without = _tile_collider_boxes((8.0, 8.0), [], [ov])
    assert len(with_floor) == len(without) + 1
    assert with_floor[-1][2] < 0.0                     # the slab sits below z = 0
