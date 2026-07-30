#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Cut a small ``_small`` sibling out of a clip library, so play scripts load fast.

``play.py`` defaults to ``<name>_small.h5`` when it exists (override with ``--full_clip``), so
every library you intend to PLAY should have one. The composers
(``compose_terrain_library.py`` / ``mb_assemble_library.py``) now emit one automatically; this
script backfills the libraries that predate that, or re-cuts one after regenerating a parent.

Runs with NO IsaacLab, and streams with h5py -- a 2.8 GB parent is subset in well under a second
because only the selected clip rows are read.

    # default subset (10 tiles x 3 clips + 6 flat)
    python scripts/rsl_rl/make_clip_subset.py --lib clips/g1_mb_v5.h5

    # bigger / smaller
    python scripts/rsl_rl/make_clip_subset.py --lib clips/g1_mb_v5.h5 \
        --num-tiles 20 --clips-per-tile 5 --flat-clips 10

    # backfill everything that lacks one
    python scripts/rsl_rl/make_clip_subset.py --all clips/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "source", "motion_gen"))

from motion_gen.clips.subset import (  # noqa: E402
    DEFAULT_CLIPS_PER_TILE,
    DEFAULT_FLAT_CLIPS,
    DEFAULT_NUM_TILES,
    check_small_is_current,
    is_small_path,
    small_path_for,
    write_small_library,
)


def _one(path: str, args: argparse.Namespace) -> int:
    """Subset a single library; returns 0 on success (or a benign skip)."""
    out = small_path_for(path)
    if os.path.exists(out) and not args.force:
        ok, why = check_small_is_current(out, path)
        state = "current" if ok else f"STALE ({why})"
        if ok:
            print(f"[skip] {out} exists and is {state}; --force to re-cut")
            return 0
        print(f"[re-cut] {out} is {state}")
    try:
        info = write_small_library(
            path, out_path=out, num_tiles=args.num_tiles,
            clips_per_tile=args.clips_per_tile, flat_clips=args.flat_clips, seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        print(f"[FAIL] {path}: {exc}")
        return 1
    print(f"[ok]   {info['out']}  {info['num_clips']} clips over {info['num_tiles']} tiles, "
          f"{info['bytes'] / 1e6:.1f} MB")
    return 0


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--lib", help="Clip library (.h5) to subset.")
    src.add_argument("--all", metavar="DIR",
                     help="Subset every non-_small .h5 in DIR that lacks a current sibling.")
    p.add_argument("--num-tiles", type=int, default=DEFAULT_NUM_TILES,
                   help=f"Terrain tiles to keep (default {DEFAULT_NUM_TILES}). Spread evenly over "
                        "the tile ids, so each terrain FAMILY is represented.")
    p.add_argument("--clips-per-tile", type=int, default=DEFAULT_CLIPS_PER_TILE,
                   help=f"Clips kept per tile (default {DEFAULT_CLIPS_PER_TILE}).")
    p.add_argument("--flat-clips", type=int, default=DEFAULT_FLAT_CLIPS,
                   help=f"Flat (no-tile) clips to keep (default {DEFAULT_FLAT_CLIPS}).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the within-tile pick.")
    p.add_argument("--force", action="store_true", help="Re-cut even if a current sibling exists.")
    args = p.parse_args()

    if args.lib:
        return _one(args.lib, args)

    libs = [f for f in sorted(glob.glob(os.path.join(args.all, "*.h5"))) if not is_small_path(f)]
    if not libs:
        print(f"[FAIL] no non-_small .h5 found in {args.all}")
        return 1
    print(f"[info] {len(libs)} candidate librar{'y' if len(libs) == 1 else 'ies'} in {args.all}")
    rc = 0
    for f in libs:
        rc |= _one(f, args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
