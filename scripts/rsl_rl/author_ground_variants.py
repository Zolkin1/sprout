#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Author one terrain-variant sidecar: vary the GROUND a recorded motion was walked on.

The generator has been measured deciding what KIND of motion to produce from things that are
not the terrain -- a floor around a staircase flips stairs-down into jump-down, and flips a
slow controlled climb into the faster gait. This writes one variant of the ground so those
features stop being constant, and `augment_registration.py --visual_only --clutter` re-renders
the recording's depth through it.

Deliberately a separate script from ``author_clutter.py``, and a separate sidecar schema. The
two passes answer different questions -- "ignore scenery beside you" versus "the terrain is
the terrain regardless of its surroundings" -- and mixing them in one artifact would make a
result attributable to neither.

**Every edit is vetted against the recorded feet.** These edits take ground away, so what has
to be proved is the mirror of the clutter veto: nothing the robot stood on stopped existing,
and nothing it stood on is now buried. A tile whose draw fails backs off to a safer one rather
than being dropped, because the tiles that fail are the stair transitions and dropping them
would bias the set toward the easy ground.

    python scripts/rsl_rl/author_ground_variants.py \\
        --clip clips/lib.h5 --occupancy <run>/clutter/occupancy.npz --seed 1 --out v1.json
    python scripts/rsl_rl/author_ground_variants.py --clip clips/lib.h5 \\
        --occupancy ... --verify v1.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "source", "motion_gen"))

from motion_gen.terrain.ground_variants import (  # noqa: E402
    SCHEMA,
    VariantCfg,
    apply_edits,
    check_feet,
    plan_tile,
)

#: Families whose ground may be edited. Flat tiles are excluded: they have no terrain to be
#: confused about, and they carry no manifest entry to edit in the first place.
TERRAIN_FAMILIES = ("stairs", "jump")


def load_feet(path: str) -> dict[int, np.ndarray]:
    """Per-tile recorded foot positions from an occupancy file.

    Args:
        path: ``build_clutter_occupancy.py`` output.

    Returns:
        ``terrain_id -> (N, 3)`` tile-local foot xyz.

    Raises:
        SystemExit: If the file predates the foot arrays.
    """
    z = np.load(path)
    feet = {int(k.split("_", 1)[1]): z[k] for k in z.files if k.startswith("feet_")}
    if not feet:
        raise SystemExit(
            f"{path} carries no 'feet_*' arrays. It was built before build_clutter_occupancy "
            "recorded foot heights; re-run it. Without them nothing can check that an edit "
            "left the robot with ground to stand on, and that check is the whole safety "
            "argument for editing the ground at all.")
    return feet


def is_terrain(tile: dict) -> bool:
    """Whether this tile's family is one whose ground may be edited."""
    fam = str(tile.get("family", ""))
    return any(k in fam for k in TERRAIN_FAMILIES)


def main() -> int:
    """Author or verify one variant sidecar."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Clip library .h5 carrying the manifest.")
    p.add_argument("--occupancy", required=True,
                   help="build_clutter_occupancy.py output, for the recorded feet.")
    p.add_argument("--seed", type=int, default=0, help="Variant seed; one variant per run.")
    p.add_argument("--out", default=None, help="Sidecar .json to write.")
    p.add_argument("--verify", default=None,
                   help="Re-check an existing sidecar against the library and the feet, and "
                        "write nothing.")
    p.add_argument("--floor_modes", default="keep,flush,none")
    p.add_argument("--wall_modes", default="none,one,both,end")
    p.add_argument("--keep_decor", action="store_true",
                   help="Leave the library's own decoration in place. Off by default: the "
                        "side walls are a candidate cue, so the pass strips them and re-adds "
                        "walls as a variable rather than a constant.")
    a = p.parse_args()

    import h5py
    with h5py.File(a.clip, "r") as f:
        raw = f.attrs["terrain_manifest"]
        man = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    feet_by_tile = load_feet(a.occupancy)
    tiles = [t for t in man if is_terrain(t) and int(t["terrain_id"]) in feet_by_tile]
    print(f"[variants] {len(tiles)} terrain tile(s) with recorded feet, of {len(man)}")

    if a.verify:
        with open(a.verify) as fh:
            doc = json.load(fh)
        by_id = {int(t["terrain_id"]): t for t in man}
        bad = 0
        for k, blk in doc["per_terrain_id"].items():
            tile = by_id[int(k)]
            why = check_feet(tile, blk, feet_by_tile.get(int(k), np.zeros((0, 3), np.float32)))
            if why:
                bad += 1
                print(f"  tile {k}: " + "; ".join(why))
        print(f"[verify] {len(doc['per_terrain_id'])} tile(s), {bad} inadmissible")
        return 1 if bad else 0

    if not a.out:
        raise SystemExit("--out is required unless --verify.")
    cfg = VariantCfg(
        seed=a.seed,
        floor_modes=tuple(s.strip() for s in a.floor_modes.split(",") if s.strip()),
        wall_modes=tuple(s.strip() for s in a.wall_modes.split(",") if s.strip()),
        strip_existing_decor=not a.keep_decor,
    )
    per, backed, stats = {}, 0, collections.defaultdict(collections.Counter)
    for t in tiles:
        tid = int(t["terrain_id"])
        blk, history = plan_tile(t, feet_by_tile[tid], cfg)
        if history:
            backed += 1
        per[str(tid)] = blk
        fam = str(t.get("family", "")).replace("family_", "").replace(".h5", "")
        stats[fam][blk["set"].get("variant_floor", "keep")] += 1
        # Applied here as well as at load, so a block that cannot be applied fails while
        # the tile it belongs to is still in hand rather than inside the importer.
        apply_edits(list(t.get("overlays", [])), blk)

    doc = {"schema": SCHEMA, "source": os.path.basename(a.clip), "seed": a.seed,
           "floor_modes": list(cfg.floor_modes), "wall_modes": list(cfg.wall_modes),
           "strip_existing_decor": cfg.strip_existing_decor,
           "per_terrain_id": per}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"[variants] seed={a.seed}: {len(per)} tile(s), {backed} needed a safer draw")
    print("  floor mode by family:")
    for fam in sorted(k for k in stats if not k.startswith("_")):
        print(f"    {fam:20s} {dict(stats[fam])}")
    print(f"[variants] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
