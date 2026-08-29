#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Cut named FAMILIES out of a clip library, tiles and all.

``train.py --skip_families`` is a flow-matching flag: it drops recorded WINDOWS at FM train
time.  The RL side has no equivalent, because on that side a family is not a label on a
window -- it is a set of clips, and the ``ClipTerrainImporter`` meshes a tile for each of
them.  Dropping a family therefore means cutting the LIBRARY, which removes the clips and
their terrain together, so the policy is never spawned on ground it is not meant to see.

The family of a clip is its tile's ``family`` in the library's ``terrain_manifest``, joined
through ``terrain_id`` -- the same join :func:`record_data._clip_family_codes` uses to write
the per-window ``clip_family`` tag.  NOTE the manifest spells families WITH the source file
suffix (``family_jump_off.h5``), while the recorder strips it; this script matches either.

Clips whose tile has no manifest entry (flat clips, ``terrain_id == -1``) belong to no family
and are always KEPT -- a family filter is a statement about terrain, and flat ground is not
one of the families being named.

:func:`write_clip_subset` rewrites the manifest to the tiles the kept clips stand on and
fixes up ``generator_config.counts``, so the importer builds exactly the reduced terrain.

Run::

    python scripts/rsl_rl/make_family_subset_clips.py \\
        --in clips/g1_mb_v5_body_mode11_decor.h5 \\
        --out clips/g1_mb_v5_body_mode11_decor_nojumpoff_nodistractor.h5 \\
        --drop family_jump_off,family_distractor
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from motion_gen.clips.clip_io import write_clip_subset


def _norm(name: str) -> str:
    """Family name without the manifest's source-file suffix."""
    return name[:-3] if name.endswith(".h5") else name


def main() -> int:
    """Write the family-filtered subset. Returns a process exit code."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True, help="source clip library .h5")
    p.add_argument("--out", dest="dst", required=True, help="output .h5 (must not exist)")
    p.add_argument("--drop", required=True,
                   help="comma-separated families to remove, with or without the .h5 suffix")
    args = p.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} exists; refusing to overwrite a clip library")
    drop = {_norm(t.strip()) for t in args.drop.split(",") if t.strip()}

    with h5py.File(args.src, "r") as f:
        terrain_id = f["terrain_id"][:]
        manifest = json.loads(f.attrs.get("terrain_manifest", "[]"))
    tid2fam = {int(e["terrain_id"]): _norm(str(e.get("family") or "unknown"))
               for e in manifest if "terrain_id" in e}
    fams = np.array([tid2fam.get(int(t), "unknown") for t in terrain_id])

    present = {f for f in set(fams.tolist()) if f != "unknown"}
    unknown = sorted(drop - present)
    if unknown:
        raise SystemExit(
            f"--drop named {unknown}, which this library does not contain. Its families are "
            f"{sorted(present)}. A typo here would silently drop nothing.")

    keep = np.flatnonzero(~np.isin(fams, list(drop)))
    if not len(keep):
        raise SystemExit("the filter would drop every clip; nothing left")
    for fam in sorted(drop):
        print(f"[family] dropping {fam}: {int((fams == fam).sum())} clip(s)")
    print(f"[family] {args.src}: {len(terrain_id)} clips -> keeping {len(keep)}")

    write_clip_subset(args.src, args.dst, [int(i) for i in keep])

    # Post-condition: the whole point is that the named families are GONE, tiles included.
    with h5py.File(args.dst, "r") as f:
        kept_tid = f["terrain_id"][:]
        kept_man = json.loads(f.attrs.get("terrain_manifest", "[]"))
    kept_map = {int(e["terrain_id"]): _norm(str(e.get("family") or "unknown"))
                for e in kept_man if "terrain_id" in e}
    left = sorted({kept_map.get(int(t), "unknown") for t in kept_tid} & drop)
    if left:
        raise SystemExit(f"{args.dst} still holds families {left}; subset is wrong")
    print(f"[family] wrote {args.dst}: {len(kept_tid)} clips, {len(kept_man)} manifest tiles, "
          f"families {sorted({kept_map.get(int(t), 'unknown') for t in kept_tid})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
