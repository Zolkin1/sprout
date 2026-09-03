# Copyright 2026 Zachary Olkin. All rights reserved.

"""Cut a library containing only the named terrain FAMILIES out of a full one.

The clip library is the spine: it carries both the reference motions and the terrain they
were generated on, so subsetting the clips subsets the world the robot walks in. Selecting
families therefore selects which terrain a run ever sees.

**Filter here, not at the reset.** There is no family filter on the RL side, and adding one
would leave envs allocated to tiles with no clips -- ``mdp.events.resets.sample_terrain_clips``
raises there, so it is a hard failure rather than a quiet one, but it is still the wrong place.
Cutting the library keeps every runtime path unchanged: the importer, both MuJoCo emitters,
every viewer and the eval sampler all rebuild the world from the manifest they are handed.

The adjuster's first training set is the two stair transitions and the two box transitions --
the four the eval harness identified as the ones that matter on hardware::

    python scripts/rsl_rl/make_family_clip_lib.py \\
        --src clips/g1_mb_v5_body_mode11_decor.h5 \\
        --out clips/g1_mb_v5_adjuster4.h5 \\
        --families stairs_up stairs_down jump_on jump_off_front

Measured on that source: 3750 clips over 75 tiles, out of 10 000 over 200.

What is deliberately left out of that set, and why:

* ``stairs_up_pure`` / ``stairs_dn_pure`` -- not needed yet, and terrain completion is
  degenerate on both (they are pinned to their last frame).
* ``flat`` -- its goal regions are not right yet, which matters because arrival is the
  objective. Its fitted goal half-width is also 3.33 m against 1.5 m for the four kept
  families, so it is the one family a default-tolerance lookup would badly misjudge.
* ``distractor`` -- boxes the robot walks around; unresolved whether they help.
* ``jump_off`` -- jumps BACKWARDS, which no forward-facing camera can see. Its
  ``goal_travel_m`` is negative (it spawns inside its own goal region ~80% of the time), and
  the generator is trained with it dropped.

.. note::
    FLAT is not a family in the manifest -- flat clips carry the ``_FLAT_TERRAIN_ID`` sentinel
    and no manifest entry at all, so they read as ``unknown``. Ask for them by name and this
    says so rather than silently keeping none; use ``make_flat_clip_lib.py`` for a flat subset,
    which also repairs the flat-TILE count that a plain subset would leave meaning clips.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from motion_gen.clips.clip_io import write_clip_subset  # noqa: E402

#: The adjuster's v1 training set: the two stair transitions and the two box transitions.
DEFAULT_FAMILIES = ("stairs_up", "stairs_down", "jump_on", "jump_off_front")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Source clip library (.h5).")
    p.add_argument("--out", required=True, help="Destination subset library (.h5).")
    p.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES),
                   help="Families to keep, NORMALISED (no 'family_' prefix, no '.h5'). "
                        f"Default: {' '.join(DEFAULT_FAMILIES)}")
    p.add_argument("--max-clips", type=int, default=None,
                   help="Keep at most this many clips, evenly spaced across the selection so "
                        "the sample spans every kept family rather than taking a correlated "
                        "prefix of one. Default: all of them.")
    return p.parse_args()


def _family_of_tile(src: str) -> tuple[dict[int, str], list[str]]:
    """Map each manifest tile id to its NORMALISED family name.

    Args:
        src: Path to the source library.

    Returns:
        ``({terrain_id: family}, [sorted family names])``.

    Raises:
        SystemExit: If the library carries no terrain manifest, which means it has no
            families to select on (a flat-only library is the usual case).
    """
    with h5py.File(src, "r") as f:
        raw = f.attrs.get("terrain_manifest")
    if not raw:
        raise SystemExit(
            f"{src} carries no terrain_manifest, so it has no families to select on. A "
            "flat-only library legitimately has none -- see make_flat_clip_lib.py."
        )
    manifest = json.loads(raw) if isinstance(raw, str) else raw

    by_tile: dict[int, str] = {}
    for entry in manifest:
        if "terrain_id" not in entry:
            continue
        name = str(entry.get("family") or "unknown")
        # The composer stamps the generating file name; strip both decorations so the CLI
        # takes the same spelling the eval harness uses (`sampler.normalize_family`).
        name = name[:-3] if name.endswith(".h5") else name
        name = name[len("family_"):] if name.startswith("family_") else name
        by_tile[int(entry["terrain_id"])] = name
    return by_tile, sorted(set(by_tile.values()))


def main() -> None:
    """Write the family subset and report what it spans."""
    args = parse_args()
    by_tile, present = _family_of_tile(args.src)

    wanted = list(dict.fromkeys(args.families))
    missing = [f for f in wanted if f not in present]
    if missing:
        # Refuse rather than keep nothing: a typo would otherwise produce an empty library,
        # and "0 clips" reads as a property of the source rather than of the command line.
        raise SystemExit(
            f"family/families {missing} are not in {args.src}. It has: {present}. "
            "(Flat clips carry no manifest entry, so 'flat' is never listed here -- use "
            "make_flat_clip_lib.py for those.)"
        )

    with h5py.File(args.src, "r") as f:
        terrain_id = np.asarray(f["terrain_id"][:]).astype(int)
        total = int(f.attrs["num_clips"])

    keep_tiles = {t for t, fam in by_tile.items() if fam in wanted}
    keep = np.flatnonzero(np.isin(terrain_id, sorted(keep_tiles)))
    if keep.size == 0:
        raise SystemExit(
            f"no clips sit on the {len(keep_tiles)} tile(s) of {wanted}; nothing to write."
        )

    per_family: dict[str, int] = {}
    for idx in keep:
        fam = by_tile.get(int(terrain_id[idx]), "unknown")
        per_family[fam] = per_family.get(fam, 0) + 1

    if args.max_clips is not None and args.max_clips < keep.size:
        # Evenly spaced, not a prefix: clips are written grouped by family and tile, so the
        # first N would be one or two families rather than a spread of the selection.
        keep = np.unique(keep[np.linspace(0, keep.size - 1, args.max_clips).round().astype(int)])

    print(f"[INFO] {args.src}: {total} clips over {len(by_tile)} tiles")
    print(f"[INFO] keeping {keep.size} clips over {len(keep_tiles)} tiles: "
          + ", ".join(f"{k} {v}" for k, v in sorted(per_family.items())))
    write_clip_subset(args.src, args.out, keep.tolist())

    with h5py.File(args.out, "r+") as f:
        n = int(f.attrs["num_clips"])
        tiles = len(json.loads(f.attrs["terrain_manifest"]))
        # Record the selection IN the artifact. A subset library is otherwise
        # indistinguishable from a small full one, and which families a run trained on is
        # exactly the kind of thing that becomes unrecoverable a month later.
        f.attrs["family_subset"] = json.dumps(sorted(wanted))
    print(f"[INFO] wrote {args.out}: {n} clips, {tiles} tiles, "
          f"{os.path.getsize(args.out) / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
