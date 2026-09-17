# Copyright 2026 Zachary Olkin. All rights reserved.
"""What a terrain library's tiles VARY, read off the library itself.

``terrain_id`` is a tile INDEX, not a value, so every reducer needs a list saying what tile
``k`` was built at -- and that list used to be typed on the command line.  The failure that
makes is silent and total: box2_quick has 3.00 m at tile 1 while the full grid has it at
tile 3, so reducing a quick run against the full grid's list relabels every cell and pairs it
against the wrong control.  The library already records the answer in its own
``generator_config``; this reads it, so nothing types it again.

It also answers the two questions an eval has to ask before it spends a GPU hour:

* ``--name`` / ``--values``: the axis, for the reducer and the banner.
* ``--check_speeds``: are the commanded speeds inside the band this course DECLARES?  A speed
  outside it is not a harder cell, it is a different distribution -- and the up-into-down
  course's floor is 0.6 where box2's is 0.5, so one shared speed list is not automatically
  legal on both.

Usage::

    python scripts/rsl_rl/terrain_axis.py --clip clips/box2_v2spawn_quick_train.h5 --values
    python scripts/rsl_rl/terrain_axis.py --clip clips/updown_p020_r030_train.h5 --check_speeds 0.83 1.17
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import h5py

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from motion_gen.clips.clip_io import read_commanded_speed_range  # noqa: E402

#: ``generator_config`` block -> the key holding the per-tile axis, in TILE ORDER.
#:
#: One entry per course, and a course missing from here has no scalar axis rather than an
#: unknown one -- ``varied`` varies its whole geometry per tile and ``indist`` is 75 real
#: library tiles, so for those the tile INDEX is the only honest label and the reducer says so
#: instead of inventing a distance.
AXES: dict[str, tuple[str, ...]] = {
    "box2": ("spawn_distances_m",),
    "step_course": ("approach_distances_m",),
}


def read_axis(path: str) -> tuple[str, list[float]]:
    """The per-tile axis a library declares.

    Args:
        path: A clip library ``.h5``.

    Returns:
        ``(name, values)`` in tile order.  ``("tile", [])`` when the course declares no scalar
        axis, which is a fact about the course rather than a defect.

    Raises:
        SystemExit: If the file carries no ``generator_config``, or the declared axis is
            shorter than the manifest -- a list that does not cover every tile would label
            some cells and silently drop the rest.
    """
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("generator_config")
        man = f.attrs.get("terrain_manifest")
    if raw is None:
        raise SystemExit(f"{path} carries no generator_config; it was not written by a "
                         "terrain builder in this tree.")
    cfg = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    n_tiles = len(json.loads(man)) if man is not None else 0
    for block, keys in AXES.items():
        blk = cfg.get(block) or {}
        for k in keys:
            if k in blk:
                vals = [float(v) for v in blk[k]]
                if len(vals) < n_tiles:
                    raise SystemExit(
                        f"{path} declares {block}.{k} with {len(vals)} value(s) for "
                        f"{n_tiles} tile(s); every tile needs a label or its cells are "
                        "reduced under another tile's.")
                return k, vals
    return "tile", []


def main() -> int:
    """Print the axis, or check the speeds against the declared band."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="the terrain library")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", action="store_true", help="print the axis NAME")
    g.add_argument("--values", action="store_true",
                   help="print the per-tile values, space separated, in tile order")
    g.add_argument("--check_speeds", type=float, nargs="+",
                   help="exit non-zero unless every speed is inside the declared band")
    a = p.parse_args()
    name, vals = read_axis(a.clip)
    if a.name:
        print(name)
        return 0
    if a.values:
        # Empty is a legal answer and the caller must tolerate it: a course with no scalar
        # axis gets no `--distances`, and the reducer then labels cells by tile index.
        print(" ".join(f"{v:g}" for v in vals))
        return 0

    rng = read_commanded_speed_range(a.clip)
    if rng is None:
        print(f"REFUSING: {a.clip} declares no commanded_speed_range_mps, so nothing here can "
              "say whether these speeds are in distribution. Rebuild it with its builder.",
              file=sys.stderr)
        return 2
    lo, hi = rng
    bad = [s for s in a.check_speeds if not (lo <= s <= hi)]
    if bad:
        print(f"REFUSING: {bad} m/s fall outside {a.clip}'s declared band {lo}-{hi} m/s. A "
              "speed outside the band is a different distribution, not a harder cell -- pass "
              "SPEEDS inside it, or widen the course's band in its generator and rebuild.",
              file=sys.stderr)
        return 2
    print(f"[axis] {len(a.check_speeds)} speed(s) inside {lo}-{hi} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
