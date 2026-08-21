#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Cut the FLAT clips out of a terrain clip library, for the blind flat teacher.

``G1-teacher-clf-flat-blind`` is the deploy-isolation policy: no exteroception at all, so it
has to be trained on ground it cannot fail to see. "Flat only" is a property of the LIBRARY,
not of the env config -- the clip ``.h5`` carries both the references and the terrain -- so
this is where that property is created.

Flat clips are the ones with ``terrain_id == -1``. They have no ``terrain_manifest`` entry by
design: the importer SYNTHESISES their tiles, all sharing that id, so any flat clip can land
on any flat tile. :func:`~motion_gen.clips.clip_io.write_clip_subset` handles the consequence
-- it rewrites ``generator_config.counts.flat`` to the kept count, which is what the importer
sizes the flat tile count from, and filters the manifest to the tiles the kept clips actually
stand on (for an all-flat subset, none).

The ``flat_tile_decor`` attribute rides along untouched, so the flat rooms survive. A blind
policy cannot see them, but the library stays usable for a camera run on the same ground.

Run from the repo root::

    PYTHONPATH=$PWD/source/robot_rl:$PWD/source/motion_gen \\
        <env_isaaclab py> scripts/rsl_rl/make_flat_clips.py \\
            --in clips/g1_mb_v5_body_mode11_decor.h5 \\
            --out clips/g1_mb_v5_body_mode11_flat.h5
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from motion_gen.clips.clip_io import write_clip_subset


def main() -> int:
    """Write the flat-only subset. Returns a process exit code."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True, help="source clip library .h5")
    p.add_argument("--out", dest="dst", required=True, help="output .h5 (must not exist)")
    p.add_argument("--max_clips", type=int, default=0,
                   help="keep at most this many flat clips (0 = all); evenly spaced, so a "
                        "smaller library still spans the whole source distribution")
    args = p.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} exists; refusing to overwrite a clip library")

    with h5py.File(args.src, "r") as f:
        terrain_id = f["terrain_id"][:]
        total = len(terrain_id)
    flat = np.flatnonzero(terrain_id == -1)
    if not len(flat):
        raise SystemExit(f"{args.src} has no flat clips (terrain_id == -1); nothing to cut")
    if args.max_clips and args.max_clips < len(flat):
        flat = flat[np.linspace(0, len(flat) - 1, args.max_clips).astype(int)]

    print(f"[flat] {args.src}: {total} clips, {len(flat)} flat -> keeping {len(flat)}")
    write_clip_subset(args.src, args.dst, [int(i) for i in flat])

    # Post-condition, because the whole point of the file is that it contains no terrain: a
    # single non-flat clip would put a blind policy on stairs and it would simply fall.
    with h5py.File(args.dst, "r") as f:
        kept = f["terrain_id"][:]
        manifest = json.loads(f.attrs.get("terrain_manifest", "[]"))
        decor = f.attrs.get("flat_tile_decor")
    bad = np.flatnonzero(kept != -1)
    if len(bad):
        raise SystemExit(f"{args.dst} still holds {len(bad)} non-flat clips; subset is wrong")
    print(f"[flat] wrote {args.dst}: {len(kept)} clips, all terrain_id -1, "
          f"{len(manifest)} manifest tiles, flat_tile_decor "
          f"{'preserved' if decor else 'ABSENT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
