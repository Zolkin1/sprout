#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Write the WALL-FIELD fine-tune library: one large flat tile per section, walls scattered on it.

    python scripts/rsl_rl/make_walls_terrain.py --tiles 24 --seed 0

The course teaches ONE thing: stop reading a vertical face as a climbable box. Every goal is
reachable in a straight line without crossing a wall, so the policy is never asked to route
around anything -- this stack has no planner and grading a failure it cannot act on would be a
mistake. See ``motion_gen.terrain.wall_field`` for the corridor invariant, why the walls are
axis-aligned, and what collidable walls change.

**A HOLDOUT file is written too, from a different seed rather than a different grid**, matching
``make_varied_terrain.py``: a tile index selects geometry only, so the equivalent held-out thing
is unseen geometry -- same distributions, same tile count, a seed the training library never
drew from.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "source" / "motion_gen"), str(_ROOT / "source" / "robot_rl"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from motion_gen.clips.clip_fingerprint import fingerprint_terrain_manifest  # noqa: E402
from motion_gen.clips.clip_io import ClipLibraryWriter  # noqa: E402
from motion_gen.terrain.freeform import GENERATOR_TAG, validate_manifest  # noqa: E402
from motion_gen.terrain.wall_field import (  # noqa: E402
    COMMANDED_SPEED_RANGE_MPS,
    WALLS_FAMILY,
    WallFieldCfg,
    family_blocks,
    generate_wall_manifest,
    generator_config,
    goal_region_spec,
)

from make_finetune_terrain import DEFAULT_LIKE  # noqa: E402
from terrain_library import library_meta, verify  # noqa: E402

#: Seed offset for the holdout library. Large and fixed, so a holdout can never coincide with a
#: training seed someone picked by hand.
HOLDOUT_SEED_OFFSET: int = 100_000


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="clips", help="Directory for the .h5 files.")
    p.add_argument("--prefix", default="ft_walls", help="Output file stem.")
    p.add_argument("--tiles", type=int, default=24, help="Wall-field tiles.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for the training library.")
    #: Defaults to the shared library rather than None: `library_meta` refuses None, because
    #: a clip-free library still has to name one output per robot joint (the trajectory command
    #: resolves them at construction). Imported rather than restated so it moves in one place.
    p.add_argument("--like", default=DEFAULT_LIKE,
                   help="A library to copy the channel layout from.")
    p.add_argument("--goals", default="logs/ft_walls/goal_regions.json",
                   help="Where to write the DECLARED goal-region spec. Shared by both roles -- "
                        "it is a tolerance, not geometry. Empty string skips it.")
    p.add_argument("--room_x", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Room interior length range (m).")
    p.add_argument("--room_y", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Room interior width range (m).")
    p.add_argument("--near_wall_frac", type=float, default=None,
                   help="Fraction of tiles whose goal sits close to the far wall.")
    p.add_argument("--walls", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Inclusive range on the wall count per tile.")
    p.add_argument("--wall_height", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Wall height range (m).")
    p.add_argument("--bearing_max", type=float, default=None,
                   help="Max |goal bearing| off the spawn heading (deg).")
    p.add_argument("--visual_only_walls", action="store_true",
                   help="Emit the walls in the visual-only decor channel instead of as solid "
                        "terrain. The cameras see them either way; this drops the physics AND "
                        "hides them from the height scan. See the module docstring.")
    p.add_argument("--no_holdout", action="store_true", help="Write only the training library.")
    p.add_argument("--no-verify", dest="no_verify", action="store_true",
                   help="Skip the read-back check.")
    return p.parse_args()


def build_cfg(args: argparse.Namespace, seed: int) -> WallFieldCfg:
    """The course config for one role, with the CLI overrides applied.

    Args:
        args: Parsed command line.
        seed: The seed for this role.

    Returns:
        The config.
    """
    cfg = WallFieldCfg(seed=seed, wall_collidable=not args.visual_only_walls)
    if args.room_x is not None:
        cfg.room_x_range_m = (float(args.room_x[0]), float(args.room_x[1]))
    if args.room_y is not None:
        cfg.room_y_range_m = (float(args.room_y[0]), float(args.room_y[1]))
    if args.near_wall_frac is not None:
        cfg.near_wall_fraction = float(args.near_wall_frac)
    if args.walls is not None:
        cfg.n_walls_range = (int(args.walls[0]), int(args.walls[1]))
    if args.wall_height is not None:
        cfg.wall_height_range_m = (float(args.wall_height[0]), float(args.wall_height[1]))
    if args.bearing_max is not None:
        cfg.goal_bearing_max_deg = float(args.bearing_max)
    return cfg


def write_wall_library(path: str, manifest: list[dict], meta: dict, role: str,
                       n_tiles: int, cfg: WallFieldCfg) -> None:
    """Write one manifest-only wall-field library.

    Args:
        path: Output file.
        manifest: The terrain manifest.
        meta: Channel-layout metadata.
        role: ``"train"`` or ``"holdout"``, recorded in the file.
        n_tiles: Tiles in the manifest.
        cfg: The course config the manifest was drawn from.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    gen_cfg = generator_config(n_tiles, cfg)
    gen_cfg["wall_field"]["role"] = role
    # No synthesised flat CLIP tiles; every tile here is a manifest tile.
    gen_cfg["counts"] = {"flat": 0}
    gen_cfg["seed"] = int(cfg.seed)
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as writer:
        writer.set_attr("terrain_manifest", manifest)
        writer.set_attr("generator", GENERATOR_TAG)
        writer.set_attr("generator_config", gen_cfg)


def main() -> int:
    """Write the training library, and the held-out-geometry one, and check both."""
    args = parse_args()
    if args.tiles <= 0:
        print("REFUSING: --tiles is 0; there would be no terrain.")
        return 2
    meta = library_meta(args.like)
    out_dir = pathlib.Path(args.out_dir)
    lo, hi = COMMANDED_SPEED_RANGE_MPS
    probe = build_cfg(args, args.seed)
    print(f"[walls] {args.tiles} wall-field tile(s), commanded speed {lo}-{hi} m/s per episode")
    print(f"[walls] walls are {'SOLID' if probe.wall_collidable else 'VISUAL-ONLY'}"
          f" ({'kind=obstacle, decor=False' if probe.wall_collidable else 'decor=True'})")
    print(f"[walls] protected corridor +-{probe.corridor_half_width_m:.2f} m "
          f"about the straight spawn->goal line")
    print(f"[walls] family blocks {family_blocks(args.tiles)}")

    if args.goals:
        spec = goal_region_spec(args.tiles, probe)
        os.makedirs(os.path.dirname(os.path.abspath(args.goals)) or ".", exist_ok=True)
        with open(args.goals, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2)
        d = spec["default"]
        print(f"[walls] goal regions -> {args.goals}  "
              f"(half_width {d['half_width_m']:.2f} m, height band {d['height_tol_m']:.2f} m)")

    roles = [("train", args.seed)]
    if not args.no_holdout:
        roles.append(("holdout", args.seed + HOLDOUT_SEED_OFFSET))
    for role, seed in roles:
        cfg = build_cfg(args, seed)
        manifest = generate_wall_manifest(args.tiles, cfg)
        validate_manifest(manifest, families=(WALLS_FAMILY,))
        path = str(out_dir / f"{args.prefix}_{role}.h5")
        write_wall_library(path, manifest, meta, role, args.tiles, cfg)
        print(f"\n[walls] {role} (seed {seed}): {len(manifest)} tile(s) -> {path}")
        print(f"[walls]   terrain fingerprint {fingerprint_terrain_manifest(manifest)}")
        short = 0
        n_near = sum(1 for t in manifest if t["wall_field"]["near_wall_goal"])
        for t in manifest:
            r, w = t["route"], t["wall_field"]
            if w["n_walls_placed"] < w["n_walls_requested"]:
                short += 1
            heights = [x["height_m"] for x in w["walls"]]
            span = (f"{min(heights):.2f}-{max(heights):.2f}" if heights else "-")
            kind = (f"NEAR-WALL (gap {w['goal_wall_gap_m']:.2f} m)" if w["near_wall_goal"]
                    else "open floor")
            print(f"[walls]   tile {t['terrain_id']:>2}: room "
                  f"{w['room_x_m']:.1f} x {w['room_y_m']:.1f} m, "
                  f"{w['n_walls_placed']:>2} interior wall(s) h {span} m, {kind}")
            print(f"[walls]      goal {w['goal_distance_m']:.1f} m at "
                  f"{w['route_bearing_deg']:+.1f} deg, spawn err "
                  f"{w['spawn_heading_error_deg']:+.1f} deg, tile "
                  f"{t['size_x']:.1f} x {t['size_y']:.1f} m")
        print(f"[walls]   near-wall goals: {n_near}/{len(manifest)} "
              f"({100.0 * n_near / len(manifest):.0f}%)")
        if short:
            # Not a failure: the corridor and the min-gap are hard constraints and a dense draw
            # can run out of room. Printed because a library that is quietly half-empty would
            # look identical in every downstream number.
            print(f"[walls]   NOTE: {short}/{len(manifest)} tile(s) placed fewer walls than "
                  f"drawn -- the corridor left no room. Lower --walls, or widen "
                  f"--room_x / --room_y, to raise the density.")
        if not args.no_verify:
            verify(path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
