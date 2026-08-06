#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Inspect an RL clip-terrain library IN ISAAC SIM -- the exact mesh + tile distribution training sees.

Builds ONLY the terrain: instantiates the real :class:`ClipTerrainImporter` on the given library (the
same class the training scene uses) and opens the GUI, so you can visually confirm the terrain imported
correctly (box sizes/placement, walls, flat patches) BEFORE a training run. No robot, no policy, no FM --
just the ground mesh + the env->tile assignment.

It also prints the training tile distribution TWO ways:
  * per-FAMILY env share -- envs are placed round-robin over tiles, so this is the actual fraction of
    training time spent on each source family (jump boxes vs distractors vs flat). Families come from the
    ``family`` tag ``compose_terrain_library.py`` stamps on each manifest tile (both jump + distractor
    tiles are terrain-type ``box``, so type alone can't separate them);
  * per-tile counts (how many distinct tiles each family contributes).

Run under the full Isaac env (env_isaaclab), WITH a display. ``--viz kit`` opens the Kit GUI window
(omitting ``--viz`` runs headless -> no window):

    PYTHONPATH=$PWD/source/robot_rl:$PWD/source/motion_gen \
        <env_isaaclab py> scripts/rsl_rl/view_terrain_isaac.py --lib clips/g1_train_box_v1.h5 --viz kit

Headless distribution check only (no window; prints the split and exits):

    ... view_terrain_isaac.py --lib clips/g1_train_box_v1.h5 --viz none --max_steps 3
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--lib", required=True, help="Composed clip-terrain library .h5 to inspect.")
parser.add_argument("--num_envs", type=int, default=300,
                    help="Envs to distribute over tiles (drives the reported env share; not the mesh).")
parser.add_argument("--num_rows", type=int, default=1,
                    help="Replica strips of the WHOLE tile set along y (each an identical copy, for env "
                         "capacity). Default 1 = show each terrain once; the training scene uses 10.")
parser.add_argument("--env_spacing", type=float, default=2.5)
parser.add_argument("--flat_buffer", type=float, default=None,
                    help="Flat-tile margin (m) per side beyond the flat clips' bbox; None = the "
                         "cfg default. Exposed because this margin BOUNDS the survival metric -- "
                         "a robot that drifts off the platform reads as base_fallen -- and the "
                         "measured series is not monotonic (2 m 176/200, 10 m 188, 20 m 194, "
                         "40 m 189), which an edge effect alone cannot explain. Render two values "
                         "and compare the tile extents AND the env->tile assignment printed below: "
                         "if origins move, each margin is a different set of initial conditions "
                         "rather than one variable swept. See TODO(flat-buffer).")
parser.add_argument("--textured", action="store_true",
                    help="Use the training marble MDL material (needs nucleus); default = plain gray.")
parser.add_argument("--max_steps", type=int, default=0, help="If >0, step this many then exit (headless smoke).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs after the Kit app (and pxr) are up ----
import sys
from collections import Counter

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from motion_gen.clips.clip_io import load_clip_library
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer import ClipTerrainImporter
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer_cfg import ClipTerrainImporterCfg


def _family_by_terrain_id(lib_path: str) -> dict[int, str]:
    """{terrain_id: family} from the manifest (family stamped by compose_terrain_library)."""
    lib = load_clip_library(lib_path, device="cpu")
    out: dict[int, str] = {}
    for m in lib.attrs.get("terrain_manifest", []):
        out[int(m["terrain_id"])] = str(m.get("family", m.get("type", "terrain")))
    return out


def _report_distribution(importer: ClipTerrainImporter, lib_path: str) -> None:
    """Print per-family env share (actual training fraction) + per-tile family counts."""
    fam_of = _family_by_terrain_id(lib_path)
    env_tid = importer.env_terrain_id.cpu().tolist()                       # [num_envs], -1 = flat
    env_fam = Counter("flat" if t < 0 else fam_of.get(t, f"tid{t}") for t in env_tid)
    tile_ids = sorted({t for t in env_tid if t >= 0})
    tile_fam = Counter(fam_of.get(t, f"tid{t}") for t in tile_ids)
    # Actual assembled tile counts (per replica row) from the layout -- cfg.num_flat_tiles is None
    # (derived at assembly), so read the built grid, not the cfg.
    tt = importer._tile_terrain_id                                         # [rows, cols] terrain ids (-1=flat)
    row0 = tt[0] if tt.dim() > 1 else tt
    n_flat_tiles = int((row0 == -1).sum())
    n_terrain_tiles = int((row0 >= 0).sum())
    n = len(env_tid)

    print("\n" + "=" * 74)
    print(f"[view_terrain_isaac] {lib_path}")
    print(f"  per replica row: {n_terrain_tiles} terrain tiles + {n_flat_tiles} flat tiles "
          f"({n_terrain_tiles + n_flat_tiles} total); {n} envs over {args.num_rows} rows")
    print("  --- env share (round-robin over tiles == training-time fraction) ---")
    for fam, c in sorted(env_fam.items(), key=lambda kv: -kv[1]):
        print(f"    {fam:44s}: {c:5d} envs ({100 * c / max(1, n):5.1f}%)")
    print("  --- distinct tiles per family ---")
    for fam, c in sorted(tile_fam.items(), key=lambda kv: -kv[1]):
        print(f"    {fam:44s}: {c:4d} tiles")
    print("=" * 74 + "\n")
    sys.stdout.flush()   # Kit closes the process hard on exit; flush or the block-buffered report is lost


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device=args.device))

    # Dome light so the mesh is lit in the GUI.
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    visual = None
    if args.textured:
        visual = sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True, texture_scale=(0.25, 0.25),
        )
    else:
        visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.57, 0.60))

    terrain_cfg = ClipTerrainImporterCfg(
        prim_path="/World/ground",
        clip_library_path=args.lib,
        num_rows=args.num_rows,
        max_init_terrain_level=None,          # use all replica rows (no difficulty curriculum)
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        **({} if args.flat_buffer is None else {"flat_buffer": args.flat_buffer}),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        ),
        visual_material=visual,
        debug_vis=True,                       # env-origin markers show where robots spawn (the distribution)
    )
    importer = ClipTerrainImporter(terrain_cfg)   # assembles + imports the mesh; sets env->tile arrays

    _report_distribution(importer, args.lib)

    # Tile geometry + env placement, because flat_buffer changes BOTH and the survival metric
    # is sensitive to how much platform sits beyond where the robot walks. Printed rather than
    # only drawn so two margins can be diffed headlessly.
    import torch as _torch
    _eo = importer.env_origins.reshape(-1, 3)
    _to = importer.terrain_origins.reshape(-1, 3) if importer.terrain_origins is not None else _eo
    _span = (_to.amax(0) - _to.amin(0)).tolist()
    _gap = None
    if _to.shape[0] > 1:
        _u = _torch.unique(_to[:, 0]).sort().values
        if _u.numel() > 1:
            _gap = float((_u[1:] - _u[:-1]).min())
    print(f"  --- geometry at flat_buffer={terrain_cfg.flat_buffer} ---")
    print(f"    tile origins       : {_to.shape[0]}, span xyz "
          f"[{_span[0]:.1f}, {_span[1]:.1f}, {_span[2]:.1f}] m"
          + (f", min x-pitch {_gap:.1f} m" if _gap else ""))
    print(f"    env origins        : {_eo.shape[0]}, span xy "
          f"[{(_eo.amax(0) - _eo.amin(0))[0]:.1f}, {(_eo.amax(0) - _eo.amin(0))[1]:.1f}] m")
    print(f"    env origin[0]      : {[round(v, 3) for v in _eo[0].tolist()]}")
    print("=" * 74 + "\n")

    # Aim the camera at the tile-grid center.
    origins = importer.terrain_origins if importer.terrain_origins is not None else importer.env_origins
    center = origins.reshape(-1, 3).mean(0).cpu().tolist()
    sim.set_camera_view(eye=(center[0] - 8.0, center[1] - 14.0, 12.0), target=(center[0], center[1], 0.0))

    sim.reset()
    print("[view_terrain_isaac] terrain up. Orbit in the GUI; Ctrl-C to quit.", flush=True)
    steps = 0
    while simulation_app.is_running():
        sim.step()                            # render on -> viewport updates (no-op when headless)
        steps += 1
        if args.max_steps and steps >= args.max_steps:   # >0 only for the headless distribution smoke
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
