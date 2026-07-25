#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Interactive viser view of TERRAIN geometry only -- no robot, no Isaac Sim.

A quick way to verify that terrain tiles (flat ground + overlay volumes such as
standalone boxes to jump onto / off / over) are assembled correctly before any
clips or RL training exist.  Two input modes:

* ``--lib clips/foo.h5`` -- load a terrain-aware clip library and render the mesh
  :func:`assemble_clip_terrain` builds from its embedded ``terrain_manifest``.
* ``--cfg pkg.module:ATTR`` -- import a ``list[CompositeSubTerrainCfg]`` (or a
  zero-arg callable returning one) and render it directly.  Works with no library,
  so terrain can be eyeballed while the box motions are still being generated.

With neither flag a small built-in demo of box tiles is shown.

Ground is drawn grey; overlay volumes are colour-coded by kind (box=orange,
obstacle=red, overhead=blue).

Run under the full-stack env (viser + isaaclab):

    PYTHONPATH=source/robot_rl:source/motion_gen \\
        env_isaaclab/bin/python scripts/rsl_rl/view_terrain_viser.py --cfg demo
    # then open the printed http://localhost:8080 URL
"""

from __future__ import annotations

import argparse
import importlib
import time

import numpy as np
import trimesh
import viser

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import (
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    OverlayVolumeCfg,
    composite_terrain,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.overlay import overlay_sub_boxes
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
    assemble_clip_terrain,
)

_GROUND_COLOR = (0.62, 0.62, 0.66)
_KIND_COLOR = {
    "box": (0.93, 0.55, 0.15),        # orange -- jump onto / off / over
    "obstacle": (0.90, 0.25, 0.25),   # red    -- hurdle
    "overhead": (0.30, 0.55, 0.95),   # blue   -- crouch under
}
_TRIVIAL_SKILL = "standing"


def _add_mesh(server: viser.ViserServer, name: str, mesh: trimesh.Trimesh,
              color: tuple[float, float, float], opacity: float = 1.0) -> None:
    """Add one trimesh to the scene as a flat-shaded solid."""
    server.scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, np.float32),
        faces=np.asarray(mesh.faces, np.int64),
        color=color, opacity=opacity, side="double", flat_shading=True,
    )


def _add_overlay_boxes(server: viser.ViserServer, prefix: str,
                       overlays: list[dict], offset_xy: tuple[float, float]) -> None:
    """Draw each overlay volume as colour-coded solid box(es) at world coords.

    ``overlays`` are the per-overlay metadata dicts emitted by
    :func:`composite_terrain` (``pos_xy`` / ``size_xyz`` / ``base_z`` / ``variant``
    in the tile's local frame); ``offset_xy`` shifts the tile's local origin into
    world.  Multi-part variants (floating top + support legs) are drawn as their
    actual sub-boxes via the shared :func:`overlay_sub_boxes`, so the colored redraw
    matches the grey mesh instead of painting one solid box over a floating one.
    """
    for j, ov in enumerate(overlays):
        subs = overlay_sub_boxes(
            str(ov.get("variant", "solid")),
            tuple(float(v) for v in ov["size_xyz"]),
            (float(ov["pos_xy"][0]), float(ov["pos_xy"][1])),
            float(ov["base_z"]),
            float(ov.get("slab_thickness", 0.1)),
            tuple(ov.get("leg_size", (0.1, 0.1))),
            float(ov.get("leg_inset", 0.05)),
        )
        color = _KIND_COLOR.get(str(ov.get("kind", "box")), _GROUND_COLOR)
        for k, (cx, cy, cz, sx, sy, sz) in enumerate(subs):
            box = trimesh.creation.box(extents=(sx, sy, sz))
            box.apply_translation((offset_xy[0] + cx, offset_xy[1] + cy, cz))
            _add_mesh(server, f"{prefix}/overlay_{j}_{k}", box, color)


def _render_library(server: viser.ViserServer, lib_path: str, num_rows: int) -> trimesh.Trimesh:
    """Render the terrain a clip library's manifest assembles into."""
    from motion_gen.clips.clip_io import load_clip_library

    lib = load_clip_library(lib_path, "cpu")
    mesh, _origins, layout = assemble_clip_terrain(lib, num_rows=num_rows)
    _add_mesh(server, "/terrain", mesh, _GROUND_COLOR)
    # Colour the overlay volumes on top (row 0 -- other rows are replicas).
    for ti in range(layout.num_tiles):
        tid = int(layout.tile_terrain_id[0, ti])
        corner = layout.tile_corner[0, ti].numpy()
        overlays = layout.terrain_overlays.get(tid, [])
        _add_overlay_boxes(server, f"/lib_tile_{ti}", overlays, (float(corner[0]), float(corner[1])))
    print(f"[view_terrain_viser] library '{lib_path}': {layout.num_tiles} tiles x {num_rows} rows")
    return mesh


def _render_subterrains(server: viser.ViserServer, subs: list[CompositeSubTerrainCfg]) -> trimesh.Trimesh:
    """Render a list of composite sub-terrains laid out left-to-right along x."""
    all_meshes: list[trimesh.Trimesh] = []
    xo = 0.0
    for i, sub in enumerate(subs):
        meshes, _origin, meta = composite_terrain(0.0, sub)
        tile = trimesh.util.concatenate(meshes)
        tile.apply_translation((xo, 0.0, 0.0))
        _add_mesh(server, f"/tile_{i}/ground", tile, _GROUND_COLOR)
        _add_overlay_boxes(server, f"/tile_{i}", meta.get("overlays", []), (xo, 0.0))
        all_meshes.append(tile)
        xo += float(sub.size[0]) + 2.0   # gap between tiles
    print(f"[view_terrain_viser] {len(subs)} sub-terrain tiles")
    return trimesh.util.concatenate(all_meshes)


def _demo_subterrains() -> list[CompositeSubTerrainCfg]:
    """Box tiles: jump-over (low), jump-onto (small), and jump-on-walk-across (long).

    A "walk across the top" box is not a new kind -- it is a normal ``box`` overlay
    with a *large* footprint (long in x, the walking direction) and ``walkable_top``.
    The tile grows so the whole approach + on-top walk + dismount fits.
    """
    def _tile(box_xyz: tuple[float, float, float], kind: str = "box",
              tile: tuple[float, float] = (6.0, 3.0)) -> CompositeSubTerrainCfg:
        fx, fy = tile
        return CompositeSubTerrainCfg(
            size=(fx, fy),
            blocks=[FlatBlockCfg(size_x=fx, skill_probs={_TRIVIAL_SKILL: 1.0})],
            overlays=[OverlayVolumeCfg(
                kind=kind, size_xyz=box_xyz,
                pos_xy=(fx / 2.0, fy / 2.0), base_z=0.0, walkable_top=(kind != "obstacle"),
            )],
        )
    return [
        _tile((0.35, 1.0, 0.30), kind="obstacle"),            # low hurdle -> jump over
        _tile((0.70, 1.0, 0.45)),                             # small box  -> jump onto / off
        _tile((3.0, 1.4, 0.45), tile=(9.0, 3.0)),             # long box   -> jump on, walk across, off
    ]


def _load_cfg(spec: str) -> list[CompositeSubTerrainCfg]:
    """Resolve ``--cfg`` into a list of :class:`CompositeSubTerrainCfg`.

    ``"demo"`` yields the built-in demo; otherwise ``spec`` is ``module:attr``
    naming a list (or a zero-arg callable returning one).
    """
    if spec == "demo":
        return _demo_subterrains()
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--cfg must be 'demo' or 'module:attr'; got '{spec}'.")
    obj = getattr(importlib.import_module(mod_name), attr)
    subs = obj() if callable(obj) else obj
    return list(subs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--lib", help="Terrain-aware clip library .h5 to assemble + render.")
    src.add_argument("--cfg", help="'demo' or 'module:attr' -> list[CompositeSubTerrainCfg].")
    p.add_argument("--num-rows", type=int, default=1, help="Tile-set replicas for --lib.")
    p.add_argument("--port", type=int, default=8080, help="Viser server port.")
    args = p.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    if args.lib:
        mesh = _render_library(server, args.lib, args.num_rows)
    else:
        mesh = _render_subterrains(server, _load_cfg(args.cfg or "demo"))

    # A ground grid sized to the whole field, sitting at z=0.
    bmin, bmax = mesh.bounds
    wx, wy = float(bmax[0] - bmin[0]), float(bmax[1] - bmin[1])
    cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
    server.scene.add_grid("/grid", width=wx + 2.0, height=wy + 2.0, cell_size=1.0,
                          position=(cx, cy, 0.0))

    print(f"[view_terrain_viser] serving on http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
