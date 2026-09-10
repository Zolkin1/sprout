# Copyright 2026 Zachary Olkin. All rights reserved.
"""Author free-form ROUTES on the ORIGINAL clip-library tiles, changing no geometry.

The point is an in-distribution measurement: run the SAME formula the box2 and up-into-down
arms use -- stand at the start, steer along a route, reach a declared goal -- on the terrain
the teacher and the original generator actually trained on. So the tiles are copied byte for
byte out of the source library's manifest and only a ``route`` block is added.

**Four families, and only four.** ``stairs_up``, ``stairs_down``, ``jump_on`` and
``jump_off_front`` are the transitions with real flat ground on at least one side; ``flat`` has
no terrain to cross, the two ``*_pure`` families are pinned to their last frame (so terrain
completion is degenerate on them), ``distractor`` is skipped by the FM by default, and
``jump_off`` jumps BACKWARDS where no forward camera can see.

**Where the route comes from, and why not the obvious place.** Averaging each tile's clips
does not work: within one tile the 50 clips start up to 3.0 m apart and, on
``jump_off_front``, end 2.4 m apart, so a mean or median names a pose no clip ever held. What
this does instead is pick, per tile, the clip whose pelvis starts FURTHEST BACK -- the most
flat lead-in, i.e. the closest thing that library has to "standing before the feature" -- and
take that same clip's terminal pelvis pose as the goal. Pairing them from one clip keeps the
route a path something actually walked.

**The spawn is the clip's LOCATION in a STANDING pose**, not its first frame:
``reset_on_freeform_section`` places the robot standing at the route's start. On
``jump_off_front`` and ``stairs_down`` that start is up on the platform, which is what makes
the descent unavoidable rather than something a robot can walk around.

**The goal needs the HEIGHT test on the ascending families.** ``freeform_goal_reached`` is
downrange-and-lateral with no height unless ``height_tol_m`` is set, and a ``stairs_up`` tile
has 0.90 m of open floor either side of a 2.25 m flight, so a robot that never climbed can
pass the line. The two descending families do not need it -- their goal IS at floor level.
This script records the recommendation per family in the library's own config block; the
termination's parameter still has to be set on the run.

Usage::

    python scripts/rsl_rl/make_indist_routes.py \\
        --source clips/g1_mb_v5_body_mode11_decor.h5 --out_dir clips --prefix indist
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_gen.clips.clip_fingerprint import fingerprint_terrain_manifest  # noqa: E402
from motion_gen.clips.clip_io import ClipLibraryWriter  # noqa: E402
from motion_gen.terrain.box2 import COMMANDED_SPEED_RANGE_MPS  # noqa: E402
from motion_gen.terrain.freeform import (  # noqa: E402
    FREEFORM_SPAWN_JITTER_M,
    GENERATOR_TAG,
    spawn_area,
)

from terrain_library import library_meta  # noqa: E402

#: The families in scope, and whether their goal sits ABOVE the spawn -- which decides
#: whether the run needs `terminations.goal_reached.params.height_tol_m` set.
FAMILIES: dict[str, bool] = {
    "family_stairs_up.h5": True,
    "family_stairs_down.h5": False,
    "family_jump_on.h5": True,
    "family_jump_off_front.h5": False,
}

#: Where the spawn goes, as a distance from the tile's BACK edge (m).  A window, not a
#: point: the surface is scanned inside it for a patch flat enough to stand the jitter on.
SPAWN_BACK_MIN_M: float = 1.0
SPAWN_BACK_MAX_M: float = 5.0

#: Where the goal goes, as a distance back from the tile's FAR edge (m).
GOAL_FROM_FAR_M: float = 2.0

#: Half-window the spawn surface must be flat over (m), so the +/-0.25 m spawn jitter cannot
#: straddle a riser and drop the robot onto a step edge at reset.
SPAWN_FLAT_HALF_M: float = 0.5

#: Lateral half-width of the goal band (m).  Deliberately NARROW -- the widest walkable
#: feature on these tiles is 2.25 m and some are 0.81 m, so a wide band overhangs the tile
#: floor beside the feature.  `freeform_goal_reached`'s own `min_half_width` floor still
#: clamps it up at run time, which is why the height test matters more than this number.
GOAL_HALF_WIDTH_M: float = 0.75


def _walkable(tile: dict) -> list[tuple[float, float, float, float, float]]:
    """Non-decor overlays as ``(x0, x1, y0, y1, top)`` boxes in tile-local metres."""
    out = []
    for o in tile.get("overlays", []):
        if o.get("decor"):
            continue
        cx, cy = (float(v) for v in o["pos_xy"])
        sx, sy, sz = (float(v) for v in o["size_xyz"])
        out.append((cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2,
                    float(o.get("base_z", 0.0)) + sz))
    return out


def surface_z(tile: dict, x: float, y: float) -> float | None:
    """Height of the highest walkable surface under ``(x, y)``, or ``None`` if there is none.

    ``None`` rather than 0.0 on a miss, and that distinction is the whole point of this
    function: a tile with ``has_floor`` false has genuine holes, and standing a robot at 0.0
    over one drops it. The first version of this script assumed 0.0 everywhere and spawned
    `stairs_down` off the back of its raised platform.

    Args:
        tile: A manifest tile.
        x / y: Tile-local position (m).

    Returns:
        The surface height, or ``None``.
    """
    tops = [t for (x0, x1, y0, y1, t) in _walkable(tile)
            if x0 <= x <= x1 and y0 <= y <= y1]
    if tops:
        return max(tops)
    return 0.0 if bool(tile.get("has_floor", True)) else None


def _feature_y(tile: dict) -> float:
    """Lateral centre of the tile's dominant walkable structure.

    The WIDEST non-decor overlay, because that is the approach/landing platform or the
    flight itself; the tile's own centre is not it -- these tiles are up to 8.2 m wide with
    a 0.8 m feature offset inside them.
    """
    w = _walkable(tile)
    if not w:
        return float(tile.get("size_y", 0.0)) / 2.0
    x0, x1, y0, y1, _ = max(w, key=lambda b: (b[3] - b[2]) * (b[1] - b[0]))
    return (y0 + y1) / 2.0


def _spawn_x(tile: dict, y: float) -> tuple[float, float] | None:
    """An x in the back window whose surface is FLAT over +/- :data:`SPAWN_FLAT_HALF_M`.

    Returns:
        ``(x, z)``, or ``None`` when the window has no such patch -- which is a tile this
        script must refuse rather than place a robot on a step edge.
    """
    best = None
    xs = np.arange(SPAWN_BACK_MIN_M, SPAWN_BACK_MAX_M + 1e-9, 0.05)
    for x in xs:
        z = surface_z(tile, float(x), y)
        if z is None:
            continue
        lo = surface_z(tile, float(x) - SPAWN_FLAT_HALF_M, y)
        hi = surface_z(tile, float(x) + SPAWN_FLAT_HALF_M, y)
        if lo is None or hi is None:
            continue
        if abs(lo - z) < 1e-6 and abs(hi - z) < 1e-6:
            # Prefer the FURTHEST BACK flat patch: most run-up before the feature.
            best = (float(x), float(z))
            break
    return best


def route_for_tile(tile: dict) -> dict:
    """The route block for one tile, derived from its GEOMETRY alone.

    Spawn 1-5 m from the back edge on the first flat patch found, goal 2 m back from the far
    edge, both at whatever surface height is actually there -- so a tile whose traverse starts
    on a raised platform (`stairs_down`, `jump_off_front`) spawns UP THERE, which is what makes
    its descent unavoidable rather than something to walk around.

    Clip poses are deliberately not used. Within one tile the 50 clips start up to 3.0 m apart
    and, on `jump_off_front`, end 2.4 m apart, so no per-tile clip pose is canonical -- and the
    earliest-start heuristic that replaced a mean put spawns off the back of the platform.

    Args:
        tile: A manifest tile.

    Returns:
        The ``route`` dict.

    Raises:
        ValueError: If the back window holds no flat patch, or the goal has no surface. Both
            mean this tile cannot carry the formula, and a placed-anyway robot would fall at
            reset or be scored against a goal in mid-air.
    """
    y = _feature_y(tile)
    got = _spawn_x(tile, y)
    if got is None:
        raise ValueError(
            f"tile {tile.get('terrain_id')}: no flat patch {SPAWN_BACK_MIN_M}-"
            f"{SPAWN_BACK_MAX_M} m from the back edge at y={y:.2f}")
    sx, sz = got
    gx = float(tile["size_x"]) - GOAL_FROM_FAR_M
    gz = surface_z(tile, gx, y)
    if gz is None:
        raise ValueError(
            f"tile {tile.get('terrain_id')}: no surface under the goal at x={gx:.2f}")
    return {
        "start_x": sx, "start_y": y, "start_z": sz, "start_yaw": 0.0,
        "goal_x": gx, "goal_y": y, "goal_z": gz,
        "goal_half_width": GOAL_HALF_WIDTH_M,
        # The area's own z is the SURFACE the robot stands on -- `reset_on_freeform_section`
        # reads `area[:, 2]`, and a wrong one here is a fall or a PhysX ejection kick on every
        # episode. Extents are FREEFORM_SPAWN_JITTER_M because a declared area replaces
        # `pos_jitter_m`, and the flat-patch search above is what makes that jitter safe.
        "spawn_areas": [spawn_area(sx, y, sz,
                                   FREEFORM_SPAWN_JITTER_M[0], FREEFORM_SPAWN_JITTER_M[1])],
    }


def main() -> int:
    """Write the manifest-only library and report each tile's route."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="the ORIGINAL clip library to copy tiles from")
    p.add_argument("--out_dir", default="clips")
    p.add_argument("--prefix", default="indist")
    p.add_argument("--like", default=None,
                   help="channel layout donor; defaults to --source, which is the right one")
    args = p.parse_args()

    with h5py.File(args.source, "r") as lib:
        man = json.loads(lib.attrs["terrain_manifest"])
        tiles = man if isinstance(man, list) else man["tiles"]
        tid = lib["terrain_id"][...]

        keep, report, refused = [], [], []
        for t in tiles:
            fam = str(t.get("family", ""))
            if fam not in FAMILIES:
                continue
            ids = np.flatnonzero(tid == int(t["terrain_id"]))
            if ids.size == 0:
                print(f"[indist] tile {t['terrain_id']} ({fam}) has no clips; skipped")
                continue
            out = dict(t)                                   # geometry copied VERBATIM
            try:
                out["route"] = route_for_tile(t)
            except ValueError as e:
                refused.append(str(e))
                continue
            keep.append(out)
            report.append((int(t["terrain_id"]), fam, out["route"], int(ids.size)))

    for msg in refused:
        print(f"[indist] REFUSED {msg}")
    if not keep:
        raise SystemExit(f"{args.source} carries none of {sorted(FAMILIES)}")

    # terrain_id is REINDEXED 0..n-1, and that is not cosmetic: the reset and the reward both
    # index a dense table by it, and the source ids are sparse once three families are
    # dropped. The original id travels in the tile so a row can still be traced back.
    for new, t in enumerate(keep):
        t["source_terrain_id"] = int(t["terrain_id"])
        t["terrain_id"] = new

    meta = library_meta(args.like or args.source)
    out = os.path.join(args.out_dir, f"{args.prefix}_train.h5")
    os.makedirs(args.out_dir or ".", exist_ok=True)
    with ClipLibraryWriter(out, num_clips=0, meta=meta) as w:
        w.set_attr("terrain_manifest", keep)
        w.set_attr("generator", GENERATOR_TAG)
        w.set_attr("generator_config", {
            "seed": 0, "counts": {"flat": 0},
            "indist_routes": {
                "source": os.path.basename(args.source),
                "families": sorted(FAMILIES),
                # The band, under the key `clip_io.read_commanded_speed_range` reads. These
                # are REAL library tiles, so the band is the one the fine-tune's other courses
                # use -- the comparison this set exists for is "did the OOD fine-tune cost
                # anything in distribution", and a different command band would answer a
                # different question.
                "commanded_speed_range_mps": [float(v) for v in COMMANDED_SPEED_RANGE_MPS],
                "goal_above_spawn": {k: v for k, v in FAMILIES.items()},
                "goal_half_width_m": GOAL_HALF_WIDTH_M,
                # The route is derived from GEOMETRY, not from clip poses -- recorded so a
                # reader knows the spawn is a flat patch in the back window and not a clip's
                # start (which varies 3.0 m within a tile and put spawns off the platform).
                "spawn_back_window_m": [SPAWN_BACK_MIN_M, SPAWN_BACK_MAX_M],
                "spawn_flat_half_m": SPAWN_FLAT_HALF_M,
                "goal_from_far_edge_m": GOAL_FROM_FAR_M,
                # The recommendation, recorded where a reader will find it. The termination's
                # parameter still has to be SET on the run; nothing here can enforce it.
                "needs_height_tol_for": [k for k, v in FAMILIES.items() if v],
            },
        })
    fp = fingerprint_terrain_manifest(keep)
    print(f"[indist] wrote {out}: {len(keep)} tiles, fingerprint {fp['short']}")
    print(f"[indist] {'tile':>5} {'src':>4} {'family':>16} {'start(x,y,z)':>22} "
          f"{'goal(x,y,z)':>22}  clip")
    for new, (src, fam, r, n) in enumerate(report):
        print(f"[indist] {new:>5} {src:>4} "
              f"{fam.replace('family_','').replace('.h5',''):>16} "
              f"({r['start_x']:6.2f},{r['start_y']:6.2f},{r['start_z']:5.2f}) "
              f"({r['goal_x']:6.2f},{r['goal_y']:6.2f},{r['goal_z']:5.2f})  "
              f"dz {r['goal_z'] - r['start_z']:+5.2f}")
    print(f"[indist] set terminations.goal_reached.params.height_tol_m on runs covering "
          f"{[k.replace('family_','').replace('.h5','') for k, v in FAMILIES.items() if v]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
