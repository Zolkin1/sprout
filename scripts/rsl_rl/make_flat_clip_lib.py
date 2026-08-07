# Copyright 2026 Zachary Olkin. All rights reserved.

"""Cut a FLAT-GROUND-ONLY clip library out of a full terrain library.

The clip library is the spine: it carries both the reference motions and the terrain they
were generated on, so subsetting the clips subsets the world the tracker walks in. Keeping
only the flat clips therefore gives a task with no stairs and no boxes -- the easiest case,
and the right place to debug a generator before asking it to climb anything.

Usage::

    python scripts/rsl_rl/make_flat_clip_lib.py \\
        --src clips/g1_mb_v5_body.h5 --out clips/g1_mb_v5_flat_body.h5

Then pass ``--clip clips/g1_mb_v5_flat_body.h5`` to play/record exactly as usual; the
terrain importer reads the terrain from the same file, so nothing else changes.

.. note::
    Flat TILES are not flat CLIPS. Each flat tile hosts many flat clips (the reset samples
    any flat clip onto any flat tile), and the importer sizes the tile count from
    ``generator_config.counts.flat`` -- see ``clip_terrain_loader.default_num_flat_tiles``.
    :func:`~motion_gen.clips.clip_io.write_clip_subset` rewrites that field to the number of
    kept flat CLIPS, which is correct for a 3-clip diagnostic subset and badly wrong here: a
    3000-clip flat library would ask for 3000 flat tiles. This script fixes it back up to a
    tile count afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from motion_gen.clips.clip_io import _FLAT_TERRAIN_ID, write_clip_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Source clip library (.h5).")
    p.add_argument("--out", required=True, help="Destination flat-only library (.h5).")
    p.add_argument("--max-clips", type=int, default=None,
                   help="Keep at most this many flat clips (evenly spaced, so the sample "
                        "spans the library rather than taking a correlated prefix). "
                        "Default: all of them.")
    p.add_argument("--num-tiles", type=int, default=None,
                   help="Flat tiles to build. Default: the source library's "
                        "generator_config.counts.flat, capped at the kept clip count "
                        "(every tile needs at least one clip or the terrain reset aborts).")
    p.add_argument("--seed", type=int, default=0, help="Unused today; reserved.")
    return p.parse_args()


def main() -> None:
    """Write the flat-only subset and repair its flat-tile count."""
    args = parse_args()
    with h5py.File(args.src, "r") as f:
        terrain_id = np.asarray(f["terrain_id"][:])
        raw = f.attrs.get("generator_config")
        gcfg = json.loads(raw) if isinstance(raw, str) else (raw or {})
    src_tiles = int((gcfg.get("counts") or {}).get("flat", 0)) or None

    flat = np.flatnonzero(terrain_id == _FLAT_TERRAIN_ID)
    if flat.size == 0:
        raise SystemExit(
            f"{args.src} has no flat clips (terrain_id == {_FLAT_TERRAIN_ID}); there is "
            "nothing to cut a flat-only library from."
        )
    if args.max_clips is not None and args.max_clips < flat.size:
        # Evenly spaced rather than the first N: clips are written grouped by family and
        # tile, so a prefix would be a handful of tiles rather than a spread of the library.
        flat = flat[np.linspace(0, flat.size - 1, args.max_clips).round().astype(int)]
        flat = np.unique(flat)

    print(f"[INFO] {args.src}: {terrain_id.size} clips -> keeping {flat.size} flat")
    write_clip_subset(args.src, args.out, flat.tolist())

    tiles = args.num_tiles if args.num_tiles is not None else (src_tiles or flat.size)
    tiles = max(1, min(int(tiles), int(flat.size)))
    with h5py.File(args.out, "r+") as f:
        raw = f.attrs.get("generator_config")
        g = json.loads(raw) if isinstance(raw, str) else (raw or {})
        g.setdefault("counts", {})
        # write_clip_subset set this to the kept CLIP count; the importer wants TILES.
        g["counts"]["flat"] = tiles
        g["flat_only_subset"] = True
        f.attrs["generator_config"] = json.dumps(g)
        n = int(f.attrs["num_clips"])
        manifest = json.loads(f.attrs["terrain_manifest"])

    print(f"[INFO] wrote {args.out}: {n} clips, {tiles} flat tiles, "
          f"{len(manifest)} non-flat tiles (expected 0), "
          f"{os.path.getsize(args.out) / 2**30:.2f} GB")
    if manifest:
        raise SystemExit(f"expected no non-flat tiles, got {len(manifest)} -- "
                         "the flat selection is wrong.")


if __name__ == "__main__":
    main()
