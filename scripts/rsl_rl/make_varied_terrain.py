#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Write the MULTI-TERRAIN fine-tune library: varied boxes + varied up-into-down, one file.

    python scripts/rsl_rl/make_varied_terrain.py --boxes 20 --updown 20 --seed 0

One library holding BOTH course families, because the fine-tune loop takes a single ``--clip``
and a multi-terrain arm means a robot can land on either course from one recording run.  A
mixed-family library is nothing new -- the in-distribution routes are 75 tiles across four
families -- and ``terrain_id`` is contiguous over the concatenation so a family occupies one
block and a report can reduce per family by id range.

**A HOLDOUT file is written too, from a different seed rather than a different grid.**  On the
pinned courses the holdout was an interior point of the initial-condition grid, because the
tile axis WAS the initial condition.  Here a tile index selects geometry only, so the
equivalent held-out thing is unseen geometry: same distributions, same tile counts, a seed the
training library never drew from.  That tests generalisation over the terrain distribution,
which is what this library varies.
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
from motion_gen.terrain.varied_courses import (  # noqa: E402
    BOXES_FAMILY,
    COMMANDED_SPEED_RANGE_MPS,
    UPDOWN_FAMILY,
    VariedCoursesCfg,
    family_blocks,
    generate_varied_manifest,
    generator_config,
    goal_region_spec,
)

from terrain_library import library_meta, verify  # noqa: E402

#: Seed offset for the holdout library.  Large and fixed, so a holdout can never coincide with
#: a training seed someone picked by hand.
HOLDOUT_SEED_OFFSET: int = 100_000


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="clips", help="Directory for the .h5 files.")
    p.add_argument("--prefix", default="varied", help="Output file stem.")
    p.add_argument("--boxes", type=int, default=20, help="Varied box tiles.")
    p.add_argument("--updown", type=int, default=20, help="Varied up-into-down tiles.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for the training library.")
    p.add_argument("--like", default=None,
                   help="A library to copy the channel layout from.")
    p.add_argument("--goals", default="logs/varied/goal_regions.json",
                   help="Where to write the DECLARED goal-region spec. Written once and "
                        "shared by both roles -- it is a tolerance, not geometry. Empty "
                        "string skips it.")
    p.add_argument("--no_holdout", action="store_true",
                   help="Write only the training library.")
    p.add_argument("--no-verify", dest="no_verify", action="store_true",
                   help="Skip the read-back check.")
    return p.parse_args()


def write_varied_library(path: str, manifest: list[dict], meta: dict, role: str,
                         n_boxes: int, n_updown: int, cfg: VariedCoursesCfg) -> None:
    """Write one manifest-only multi-terrain library.

    Args:
        path: Output file.
        manifest: The terrain manifest.
        meta: Channel-layout metadata.
        role: ``"train"`` or ``"holdout"``, recorded in the file.
        n_boxes: Varied box tiles in the manifest.
        n_updown: Varied up-into-down tiles in the manifest.
        cfg: The course config the manifest was drawn from.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    gen_cfg = generator_config(n_boxes, n_updown, cfg)
    gen_cfg["varied_courses"]["role"] = role
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
    if args.boxes + args.updown == 0:
        print("REFUSING: --boxes and --updown are both 0; there would be no terrain.")
        return 2
    meta = library_meta(args.like)
    out_dir = pathlib.Path(args.out_dir)
    lo, hi = COMMANDED_SPEED_RANGE_MPS
    print(f"[varied] {args.boxes} box + {args.updown} up-into-down tiles, "
          f"commanded speed {lo}-{hi} m/s continuous per episode")
    print(f"[varied] family blocks {family_blocks(args.boxes, args.updown)}")

    if args.goals:
        spec = goal_region_spec(args.boxes, args.updown, VariedCoursesCfg(seed=args.seed))
        os.makedirs(os.path.dirname(os.path.abspath(args.goals)) or ".", exist_ok=True)
        with open(args.goals, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2)
        # Both roles share it: the band is a scoring TOLERANCE and every tile of every family
        # here is built to it, which is what lets one number serve the whole library.
        d = spec["default"]
        print(f"[varied] goal regions -> {args.goals}  "
              f"(half_width {d['half_width_m']:.2f} m, height band "
              f"{d['height_tol_m']:.2f} m)")

    roles = [("train", args.seed)]
    if not args.no_holdout:
        roles.append(("holdout", args.seed + HOLDOUT_SEED_OFFSET))
    for role, seed in roles:
        cfg = VariedCoursesCfg(seed=seed)
        manifest = generate_varied_manifest(args.boxes, args.updown, cfg)
        validate_manifest(manifest, families=(BOXES_FAMILY, UPDOWN_FAMILY))
        path = str(out_dir / f"{args.prefix}_{role}.h5")
        write_varied_library(path, manifest, meta, role, args.boxes, args.updown, cfg)
        print(f"\n[varied] {role} (seed {seed}): {len(manifest)} tile(s) -> {path}")
        print(f"[varied]   terrain fingerprint {fingerprint_terrain_manifest(manifest)}")
        for t in manifest:
            r = t["route"]
            if BOXES_FAMILY == t["family"]:
                b = t["varied_boxes"]
                geom = (f"{b['n_boxes']} box(es) at {b['approach_deg']:+.1f} deg, "
                        + " + ".join(f"{x['height_m']:.2f}h x {x['depth_m']:.2f}d "
                                     f"x {x['width_m']:.2f}w" for x in b["boxes"]))
                areas = b["spawn_areas"]
            else:
                u = t["varied_updown"]
                geom = (f"{u['steps']} riser(s) x {u['rise_m']:.3f} m "
                        f"(+{u['flight_height_m']:.2f} m), corridor {u['width_m']:.2f} m, "
                        f"platform {u['platform_depth_m']:.2f} m, "
                        f"walls {'+'.join(u['wall_sides'])}")
                areas = u["spawn_areas"]
            print(f"[varied]   tile {t['terrain_id']:>2} [{t['family']}]: {geom}")
            print(f"[varied]      tile {t['size_x']:.2f} x {t['size_y']:.2f} m, "
                  f"goal x={r['goal_x']:.2f} +-{r['goal_half_width']:.2f} m, "
                  f"{areas} spawn area(s)")
        if not args.no_verify:
            verify(path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
