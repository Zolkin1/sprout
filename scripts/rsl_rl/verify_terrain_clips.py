# Copyright 2026 Zachary Olkin. All rights reserved.

"""Standalone check that a clip library's terrains + clips load and align.

Builds the IsaacLab terrain grid from a clip library's embedded
``terrain_manifest`` (one cell per terrain + a flat cell), assigns each env a
clip whose ``terrain_id`` matches its cell, and draws that clip's pelvis + feet
reference as markers on the terrain -- WITHOUT a robot or a policy. Use it to
confirm terrain + clip loading and placement before starting a training run: the
feet markers should ride the treads of each cell's staircase.

Prints a per-terrain summary and LOUD banners for any terrain with no clips or
any clip whose terrain has no cell.

Examples::

    # headless summary + id-mismatch check (fast)
    ./isaaclab.sh -p scripts/rsl_rl/verify_terrain_clips.py \
        --clip_path clips/g1_terrain_v1.h5 --headless --num_envs 64

    # static ribbons on the terrain, block outlines on
    ./isaaclab.sh -p scripts/rsl_rl/verify_terrain_clips.py \
        --clip_path clips/g1_terrain_v1.h5 --debug_vis

    # animate the reference sweeping through each clip
    ./isaaclab.sh -p scripts/rsl_rl/verify_terrain_clips.py \
        --clip_path clips/g1_terrain_v1.h5 --animate --loop --debug_vis
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -- argparse ---------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--clip_path", type=str, required=True, help="Clip library .h5 (terrain + clips).")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Probe placements (default: one per grid cell = num terrains + flat).")
parser.add_argument("--num_rows", type=int, default=1, help="Replica rows of the tile set.")
parser.add_argument("--num_flat_tiles", type=int, default=None,
                    help="Number of flat patches (default: one per flat clip).")
parser.add_argument("--start_step", type=int, default=0, help="Clip step where the (static) ribbon begins.")
parser.add_argument("--stride", type=int, default=4, help="Draw every Nth clip step in the ribbon.")
parser.add_argument("--window", type=int, default=None,
                    help="Ribbon length in steps (default: whole clip; the animate window when --animate).")
parser.add_argument("--animate", action="store_true", default=False,
                    help="Sweep the ribbon window through the clip each frame.")
parser.add_argument("--fps", type=float, default=30.0, help="Animation step rate (Hz) when --animate.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop the animation.")
parser.add_argument("--marker_radius", type=float, default=0.03, help="Reference-point sphere radius (m).")
parser.add_argument("--show_hands", action="store_true", default=False, help="Also draw wrist frames.")
parser.add_argument("--debug_vis", action="store_true", default=False, help="Per-block outline debug viz.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -- imports that require the running app -----------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: E402

from motion_gen.clips.clip_io import (  # noqa: E402
    group_clips_by_terrain,
    load_clip_library,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer_cfg import (  # noqa: E402
    ClipTerrainImporterCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (  # noqa: E402
    clip_frame_origin,
    default_num_flat_tiles,
)

# Reference-frame colors (pelvis blue, L-ankle teal, R-ankle orange, wrists grey).
_FRAMES: list[tuple[str, tuple[float, float, float]]] = [
    ("pelvis_link", (0.0, 0.47, 0.74)),
    ("left_ankle_roll_link", (0.0, 0.60, 0.53)),
    ("right_ankle_roll_link", (0.93, 0.46, 0.20)),
]
_HAND_FRAMES: list[tuple[str, tuple[float, float, float]]] = [
    ("left_wrist_yaw_link", (0.55, 0.55, 0.55)),
    ("right_wrist_yaw_link", (0.75, 0.75, 0.30)),
]
_BAR = "!" * 78


def _frames(names: list[str]) -> list[tuple[str, tuple[float, float, float]]]:
    """Frames to draw, filtered to those present in the clip output layout."""
    want = _FRAMES + (_HAND_FRAMES if args_cli.show_hands else [])
    return [(f, c) for f, c in want if f"{f}:pos_x" in names]


def _assign_clips(env_tid: torch.Tensor, groups: dict[int, torch.Tensor]) -> torch.Tensor:
    """Pick, per env, a random clip whose terrain_id matches the env's cell.

    Raises loudly if any env's terrain id has no clips.
    """
    missing = sorted({int(t) for t in env_tid.tolist() if int(t) not in groups or groups[int(t)].numel() == 0})
    if missing:
        print(f"\n{_BAR}\n!!  ABORT: terrain id(s) {missing} have a grid cell but NO clips in the\n"
              f"!!  library -- envs on those cells cannot be assigned a matching clip.\n{_BAR}\n", flush=True)
        raise SystemExit(1)
    # Deterministic round-robin per terrain so every distinct clip is shown when
    # a terrain has >=1 env (vs random, which could repeat and hide clips).
    out = torch.empty(env_tid.shape[0], dtype=torch.long)
    counters: dict[int, int] = {}
    for i, t in enumerate(env_tid.tolist()):
        g = groups[int(t)]
        k = counters.get(int(t), 0)
        out[i] = int(g[k % g.numel()])
        counters[int(t)] = k + 1
    return out


def _print_summary(lib, importer, env_tid) -> None:
    """Per-terrain table + loud flags for id mismatches between clips and tiles."""
    manifest = {int(m["terrain_id"]): m for m in lib.attrs["terrain_manifest"]}
    groups = group_clips_by_terrain(lib)

    def _step_hd(m: dict) -> str:
        """``riser/tread`` for a stair tile, whichever manifest schema it uses.

        LEGACY tiles carry ``dims`` (the generator re-derived the geometry RL-side); tiles-then-clips tiles
        carry explicit ``overlays`` instead, so the step size is measured off the narrow tread boxes -- the
        same width test (``size_x <= 1.5 * min``) the rest of the pipeline uses to tell treads from platforms.
        """
        if "dims" in m:
            return f"{m['dims']['h']:.2f}/{m['dims']['d']:.2f}"
        ov = m.get("overlays") or []
        if not ov:
            return "-"
        sx = [o["size_xyz"][0] for o in ov]
        treads = [o for o in ov if o["size_xyz"][0] <= 1.5 * min(sx)]
        tops = sorted({round(o["base_z"] + o["size_xyz"][2], 4) for o in treads})
        if len(tops) < 2:
            return f"box h={max(o['base_z'] + o['size_xyz'][2] for o in ov):.2f}"
        return f"{tops[1] - tops[0]:.2f}/{min(o['size_xyz'][0] for o in treads):.2f}"
    layout = importer._layout
    cell_ids = set(int(t) for t in torch.unique(layout.tile_terrain_id).tolist())
    clip_ids = set(int(t) for t in groups)
    fps = sorted({tuple(round(v, 2) for v in row) for row in layout.tile_footprint.reshape(-1, 2).tolist()})

    print(f"\n=== terrain-clip verify: {args_cli.clip_path} ===")
    print(f"{layout.num_rows} row(s) x {layout.num_tiles} tiles  |  {lib.num_clips} clips, "
          f"{lib.num_steps} steps @ dt={lib.meta['dt']}  |  tile footprints (x,y) = {fps}")
    print(f"{'terrain':>8} {'type':>11} {'#clips':>7} {'#envs':>6} {'size_x':>7} {'size_y':>7} {'step h/d':>12}")
    env_tid_list = env_tid.tolist()
    for tid in sorted(cell_ids):
        g = groups.get(tid)
        nclip = 0 if g is None else int(g.numel())
        nenv = sum(1 for t in env_tid_list if t == tid)
        if tid == -1:
            print(f"{tid:>8} {'flat':>11} {nclip:>7} {nenv:>6} {'-':>7} {'-':>7} {'-':>12}")
        else:
            m = manifest[tid]
            print(f"{tid:>8} {m['type']:>11} {nclip:>7} {nenv:>6} {m['size_x']:>7.2f} {m['size_y']:>7.2f} "
                  f"{_step_hd(m):>12}")

    no_clip = sorted(cell_ids - clip_ids)
    no_cell = sorted(clip_ids - cell_ids)
    if no_clip:
        print(f"\n{_BAR}\n!!  WARNING: terrain tile(s) {no_clip} have NO clips in the library.\n{_BAR}\n", flush=True)
    if no_cell:
        print(f"\n{_BAR}\n!!  WARNING: clip terrain id(s) {no_cell} have NO tile -- those clips\n"
              f"!!  cannot be shown / assigned. (Manifest vs per-clip terrain_id mismatch.)\n{_BAR}\n", flush=True)
    if not no_clip and not no_cell:
        print("[verify] terrain ids: clips <-> tiles consistent.\n", flush=True)


def main() -> None:
    """Build terrain, assign + place clips, draw reference markers, run the sim."""
    sim = SimulationContext(SimulationCfg())
    device = sim.device

    lib = load_clip_library(args_cli.clip_path, device="cpu")
    # Tile count = manifest terrains + flat tiles (default: the library's flat-terrain
    # count from generator_config, matching the generation split); size num_envs to one
    # env per tile per replica row unless overridden.
    n_terrain_tiles = len(lib.attrs["terrain_manifest"])
    num_flat_tiles = args_cli.num_flat_tiles if args_cli.num_flat_tiles is not None else default_num_flat_tiles(lib)
    num_tiles = n_terrain_tiles + num_flat_tiles
    num_envs = int(args_cli.num_envs or args_cli.num_rows * num_tiles)

    importer_cfg = ClipTerrainImporterCfg(
        num_envs=num_envs, prim_path="/World/ground",
        clip_library_path=args_cli.clip_path, num_rows=args_cli.num_rows,
        num_flat_tiles=num_flat_tiles, collision_group=-1, debug_vis=args_cli.debug_vis,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                     "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True, texture_scale=(0.25, 0.25),
        ),
    )
    importer = importer_cfg.class_type(importer_cfg)

    # Sky + camera framing the whole layout.
    sky_cfg = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )
    sky_cfg.func("/World/skyLight", sky_cfg)
    corners = importer._tile_corner.reshape(-1, 3).cpu()
    fps = importer._tile_footprint.reshape(-1, 2).cpu()
    ext = float(torch.stack([(corners[:, 0] + fps[:, 0]).max() - corners[:, 0].min(),
                             (corners[:, 1] + fps[:, 1]).max() - corners[:, 1].min()]).max())
    sim.set_camera_view(eye=(0.5 * ext, 0.5 * ext, 0.4 * ext), target=(0.0, 0.0, 0.0))

    # Per-env clip assignment (match the tile's terrain) + placement.
    env_tid = importer.env_terrain_id.cpu()
    groups = group_clips_by_terrain(lib)
    _print_summary(lib, importer, env_tid)
    clip_id = _assign_clips(env_tid, groups).to(device)                     # [E]

    # Clip frame origin: x-corner on the tile corner, centerline on the tile
    # centerline (terrain clips sit on the corridor centerline size_y/2; flat
    # clips ~0). z = tile ground; the clip's baked z rides the treads.
    tid_list = env_tid.tolist()
    names = lib.meta["pos_output_names"]
    type_code = torch.tensor([importer.terrain_type_code.get(t, 1) for t in tid_list], device=device)
    world_pos = clip_frame_origin(importer.env_cell_corner, importer.env_clip_x_offset)    # [E,3]

    # Flat clips (terrain_id -1) WANDER (they sample lateral/yaw -> turning walks
    # up to tens of metres), and their bbox is not centered on the clip's local
    # origin, so placing that origin at the cell center overhangs the edge.  The
    # flat cell is sized to the max flat bbox, so CENTER each flat clip's bbox in
    # its cell instead -> it always fits.  (Terrain clips keep the corner +
    # centerline placement that aligns their stairs.)
    px, py = names.index("pelvis_link:pos_x"), names.index("pelvis_link:pos_y")
    pel_xy = lib.y_des[clip_id.cpu()][:, :, [px, py]]                        # [E, T, 2] cpu
    bbox_center = (0.5 * (pel_xy.amax(dim=1) + pel_xy.amin(dim=1))).to(device)   # [E, 2]
    is_flat = type_code == 0
    world_pos[is_flat, 0] = importer.env_cell_center[is_flat, 0] - bbox_center[is_flat, 0]
    world_pos[is_flat, 1] = importer.env_cell_center[is_flat, 1] - bbox_center[is_flat, 1]

    # Reference-marker prototypes (one sphere per frame color).
    frames = _frames(names)
    frame_idx = torch.tensor(
        [[names.index(f"{f}:pos_{a}") for a in ("x", "y", "z")] for f, _ in frames], device="cpu",
    )                                                                       # [F, 3]
    markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/World/clip_ref",
        markers={f: sim_utils.SphereCfg(radius=args_cli.marker_radius,
                                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c))
                 for f, c in frames},
    ))
    F = len(frames)
    T = lib.num_steps
    E = num_envs
    y_des = lib.y_des                                                       # [N, T, P] on cpu

    def _draw(step0: int) -> None:
        """Draw the ribbon window starting at ``step0`` for every env."""
        win = args_cli.window or T
        steps = torch.arange(step0, min(step0 + win, T), args_cli.stride)   # [S]
        if steps.numel() == 0:
            return
        rel = y_des[clip_id.cpu()][:, steps][:, :, frame_idx]               # [E, S, F, 3] cpu clip-world
        rel = rel.to(device)
        pts = world_pos[:, None, None, :] + rel                            # identity heading -> translation
        # Group points by frame so each uses its color prototype.
        trans = pts.permute(2, 0, 1, 3).reshape(-1, 3)                      # [F*E*S, 3]
        idx = torch.arange(F, device=device).repeat_interleave(E * steps.numel())
        markers.visualize(translations=trans, marker_indices=idx)

    _draw(args_cli.start_step)
    sim.reset()

    # Headless has no viewer, so exit after a few steps (build + summary check).
    if args_cli.headless:
        for _ in range(5):
            sim.step()
        print("[verify] headless build OK -- terrain grid + clip placement constructed.", flush=True)
        return

    steps_per_frame = max(1, int(round(1.0 / (args_cli.fps * float(sim.get_physics_dt())))))
    cur, tick = args_cli.start_step, 0
    while simulation_app.is_running() and not sim.is_stopped():
        sim.step()
        if args_cli.animate:
            tick += 1
            if tick % steps_per_frame == 0:
                cur += args_cli.stride
                if cur >= T:
                    cur = args_cli.start_step if args_cli.loop else T - 1
                _draw(cur)


if __name__ == "__main__":
    main()
    simulation_app.close()
