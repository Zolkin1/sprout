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
parser.add_argument("--goals", default=None, metavar="goal_regions.json",
                    help="Draw each tile's GOAL REGION as a translucent cyan gate -- the near "
                         "face the robot has to cross, spanning the lateral and height bands "
                         "the spec declares. The region is a downrange HALF-PLANE, so "
                         "everything past the gate is inside it. Free-form (manifest) tiles "
                         "only: a clip library's goal is the clip's own terminal pose and "
                         "there is no clip here to ask.")
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


def _draw_goal_gates(importer: ClipTerrainImporter, spec_path: str):
    """Draw one goal gate per free-form tile, and return the marker set (or ``None``).

    What is drawn is exactly what ``goal_region.in_goal`` tests, read through the same code
    the runtime reads it through -- ``freeform_route`` for the route row and the tile frame,
    ``GoalRegionSpec`` for the tolerances -- so a gate in the wrong place here means the
    SCORING is in the wrong place, which is the only reason to look at it.

    Three things about the geometry are easy to get wrong and are worth knowing while looking:

    * The region is a downrange **half-plane**, not a box. The gate is the near FACE; every
      point past it, to the end of the tile, is inside.
    * The datum is a **PELVIS height**, not the walkable surface. ``route.goal_z`` is the
      surface, and the anchor adds the robot's standing height -- the same quantity the spawn
      adds to ``start_z`` -- so the band is centred where a standing robot's base would be.
      Drawn against the surface it would look half a metre too low and be correct.
    * Per-family overrides ARE honoured here (``spec.for_family``), unlike the in-env gate in
      ``MotionSequenceCommand``, which draws the default set only.

    Args:
        importer: The assembled :class:`ClipTerrainImporter`.
        spec_path: Path to a ``goal_regions.json``.

    Returns:
        The :class:`VisualizationMarkers`, kept alive by the caller, or ``None`` if no tile
        carries a route.
    """
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.utils.math import quat_from_angle_axis

    from robot_rl.assets.robots.g1_21j import G1_CFG
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (
        _DEBUG_GOAL_COLOR,
        _DEBUG_GOAL_FACE_M,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.events.resets import freeform_route

    def _normalise_family(name: str) -> str:
        """A manifest family name as the goal spec keys it."""
        try:
            from eval.sampler import normalize_family  # noqa: PLC0415

            return normalize_family(name)
        except Exception:
            n = name[:-3] if name.endswith(".h5") else name
            return n[len("family_"):] if n.startswith("family_") else n
    from robot_rl.tasks.manager_based.robot_rl.mdp.goal_region import GoalRegionSpec

    spec = GoalRegionSpec.load(spec_path)
    fam_of = _family_by_terrain_id(args.lib)

    # One representative env per distinct tile: envs are round-robin over tiles, so a per-env
    # draw would stack dozens of identical gates on each tile.
    dev = importer.env_terrain_id.device
    tids = importer.env_terrain_id.cpu().tolist()
    first_env: dict[int, int] = {}
    for i, t in enumerate(tids):
        if t >= 0 and t not in first_env:
            first_env[int(t)] = i
    if not first_env:
        print("[view_terrain_isaac] --goals: no tile carries a route block (a clip library's "
              "goal is the clip's own terminal pose); nothing drawn.")
        return None

    env_ids = torch.tensor(sorted(first_env.values()), dtype=torch.long, device=dev)
    route = freeform_route(importer, env_ids)
    yaw = route["frame_yaw"]
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    gx, gy = route["goal_x"], route["goal_y"]
    gz = route["goal_z"]
    corner = route["corner"]

    # The standing height, off the asset cfg -- the same number `default_root_pose` carries and
    # that `reset_on_freeform_section` adds to `start_z`. Read rather than restated: a goal
    # datum that disagreed with the spawn's by even a few cm would put the whole band off.
    stand_z = float(G1_CFG.init_state.pos[2])
    anchor = torch.stack((corner[:, 0] + cos_y * gx - sin_y * gy,
                          corner[:, 1] + sin_y * gx + cos_y * gy,
                          corner[:, 2] + gz + stand_z), dim=-1)

    # Per-tile tolerances, so a spec with per-family overrides draws each family's own band.
    order = sorted(first_env)
    # NORMALISED. `GoalRegionSpec.by_family` is keyed by `sampler.normalize_family`
    # ("stairs_up"), while a manifest carries the raw source name ("family_stairs_up.h5"), so
    # an un-normalised lookup misses every override and silently falls back to `default` --
    # while the banner below still announces that overrides exist.
    tols = [spec.for_family(_normalise_family(fam_of.get(t, ""))) for t in order]
    margin = torch.tensor([float(t.approach_margin_m) for t in tols], device=dev)
    width = torch.tensor([float(t.half_width_m) for t in tols], device=dev)
    height = torch.tensor([float(t.height_tol_m) for t in tols], device=dev)

    # The boundary sits `approach_margin_m` SHORT of the anchor, along the TERRAIN frame's
    # downrange axis -- not along world x, which is the same thing only on tiles facing that way.
    centre = anchor.clone()
    centre[:, 0] -= margin * cos_y
    centre[:, 1] -= margin * sin_y

    # A height tolerance of 0 means the test is OFF, not "a zero-thickness band" -- so draw a
    # full-height gate there rather than an invisible one, which would read as a missing gate.
    face_h = torch.where(height > 0.0, 2.0 * height, torch.full_like(height, 4.0))
    scales = torch.stack(
        (torch.full_like(width, _DEBUG_GOAL_FACE_M), 2.0 * width, face_h), dim=-1)

    markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/terrain_goal_regions",
        markers={"goal_gate": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=_DEBUG_GOAL_COLOR, opacity=0.25))},
    ))
    world_z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(yaw.shape[0], 3)
    markers.set_visibility(True)
    markers.visualize(translations=centre,
                      orientations=quat_from_angle_axis(yaw, world_z),
                      scales=scales)

    t0 = tols[0]
    print(f"  --- goal regions from {spec_path} ---")
    print(f"    gates drawn        : {len(order)} (one per tile)")
    print(f"    default tolerances : margin {t0.approach_margin_m:.2f} m, half-width "
          f"{t0.half_width_m:.2f} m, height {t0.height_tol_m:.2f} m"
          + ("  [height test OFF]" if t0.height_tol_m <= 0 else ""))
    if spec.by_family:
        print(f"    per-family overrides: {', '.join(sorted(spec.by_family))}")
    print(f"    pelvis datum       : surface + {stand_z:.3f} m standing height")
    # Per tile while the list is short enough to read: with an ascending and a descending
    # variant of one course the INTERESTING thing is that their gates sit at different
    # heights, which a single summary row hides.
    if len(order) <= 12:
        for i, tid in enumerate(order):
            print(f"    tile {tid:<3d} {fam_of.get(tid, '?'):<16s}: gate "
                  f"{[round(v, 3) for v in centre[i].tolist()]}  "
                  f"half-width {float(width[i]):.2f}  height +-{float(height[i]):.2f}")
    else:
        zs = centre[:, 2]
        print(f"    gate heights       : {float(zs.min()):.3f} .. {float(zs.max()):.3f} m")
    print("=" * 74 + "\n")
    sys.stdout.flush()
    return markers


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

    # After `sim.reset()`: the marker prims need the stage up, and `visualize` writes through
    # a point instancer that a reset would otherwise clear.
    # The reference is HELD, not discarded: dropping it lets the marker object be collected
    # and the gates disappear from the stage a few frames later, which reads as a goal region
    # that was never drawn.
    goal_markers = _draw_goal_gates(importer, args.goals) if args.goals else None  # noqa: F841

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
