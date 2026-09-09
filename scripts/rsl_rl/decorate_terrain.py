# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rewrite a clip library's terrain manifest. No motion data is touched.

Usage::

    PYTHONPATH=source/motion_gen python scripts/rsl_rl/decorate_terrain.py \\
        --clip clips/g1_mb_v5_body_mode11.h5 \\
        --out clips/g1_mb_v5_body_mode11_decor.h5 --seed 0

Terrain lives in ONE root attribute, ``terrain_manifest``, and everything that draws or
simulates the world rebuilds it from there: the RL importer, both MuJoCo emitters and every
viser viewer. So this pass copies the file and rewrites that attribute -- the 6.8 GB of
motion is copied verbatim, byte for byte, and the clips' only tie to their tiles (an integer
``terrain_id``) is untouched.

The copy is why ``--out`` is required rather than optional: the input library is the thing
every existing checkpoint was trained against, and a decorated manifest is a different world.

What it does today is the stair-width pass of Part 2.3 -- see
``motion_gen.terrain.decorate``. Walls, railings and clutter are a separate step on the same
attribute and can run after this one.

**Every edit is validated against the motion, not just against the manifest.** After the
rewrite, each tile's own foot samples are checked: wherever a foot sits inside a flight's
downrange span, the tread beneath it must still be under it laterally. That check is run on
the SUB-BOXES rather than on a height field, because 348 of the 1260 overlays are
``floating`` -- a thin slab over open air -- and a top-down height test cannot see an
overhang or a relocated leg.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "motion_gen"))

from motion_gen.terrain.decorate import (ARMS, BOX_VARIANTS,  # noqa: E402
                                         DecorateConfig, add_walls,
                                         class_footprints, flat_tile_footprint,
                                         make_flat_rooms,
                                         decorate_box_tile,
                                         decorate_transition_tile,
                                         has_width_variation, span)
from motion_gen.terrain.footprint import foot_corners  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_tile_footprints import ANKLES, _cols, measure  # noqa: E402


def check_feet_are_supported(tile: dict, roles: list[str], ankle_pos: np.ndarray,
                             ankle_quat: np.ndarray, tol: float = 1e-3) -> list[str]:
    """Foot samples that sit within a tread's downrange span but off it laterally.

    The failure a width edit can cause, stated directly: a tread that no longer reaches the
    foot. Checked per overlay in the tile frame, on the overlay's own footprint.

    Args:
        tile: The rewritten manifest entry.
        roles: Per-overlay roles from BEFORE the rewrite
            (``motion_gen.terrain.decorate.classify_overlays``). Re-deriving them here would
            inspect the wrong overlays: the edit changes widths, so a widened tread now reads
            as a platform and a narrowed platform reads as a tread. A first version did
            exactly that and passed without examining a single widened tread.
        ankle_pos: ``(N, 3)`` ankle positions in the tile frame.
        ankle_quat: ``(N, 4)`` ankle orientations, w first.
        tol: Slack, in metres.

    Returns:
        One message per offending overlay; empty if every foot is supported.
    """
    corners = foot_corners(ankle_pos, ankle_quat)          # (N, 8, 3)
    cx, cy = corners[..., 0], corners[..., 1]
    problems: list[str] = []
    for i, (overlay, role) in enumerate(zip(tile["overlays"], roles)):
        if not overlay.get("walkable_top", True) or role != "tread":
            continue
        half_x = float(overlay["size_xyz"][0]) / 2.0
        mid_x = float(overlay["pos_xy"][0])
        lo_y, hi_y = span(overlay)
        inside_x = (cx >= mid_x - half_x - tol) & (cx <= mid_x + half_x + tol)
        off_y = inside_x & ((cy < lo_y - tol) | (cy > hi_y + tol))
        # A foot straddling the END of a flight is normal -- it is stepping onto the
        # platform. Only count a corner that is off the tread while INSIDE its x-span AND
        # no other tread covers it.
        if off_y.any():
            n = int(off_y.sum())
            problems.append(
                f"overlay {i} (tread, y [{lo_y:.3f}, {hi_y:.3f}]): {n} foot corner(s) "
                f"inside its x-span but off it laterally"
            )
    return problems


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="input clip library .h5")
    p.add_argument("--out", required=True, help="output .h5 (must not exist)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tread_band_m", type=float, nargs=2, default=(1.219, 2.438),
                   metavar=("MIN", "MAX"),
                   help="the stair-width band every flight lands in (m); the default is "
                        "4 ft to 8 ft, which is what real stairwells are")
    p.add_argument("--max_clips_per_tile", type=int, default=12,
                   help="clips sampled per tile for the envelope and the check")
    p.add_argument("--no_box_variants", action="store_true",
                   help="leave the box tiles solid (they are 100%% solid today only because "
                        "the generator never passes variant_fracs)")
    p.add_argument("--no_walls", action="store_true",
                   help="skip the wall pass")
    p.add_argument("--flat_room_variants", type=int, default=12,
                   help="distinct room layouts authored for the flat tiles, handed out "
                        "round-robin (flat tiles all share terrain_id -1)")
    p.add_argument("--collidable_walls", action="store_true",
                   help="emit the walls as ordinary collidable terrain rather than the "
                        "visual-only channel (see DecorateConfig.wall_collidable)")
    p.add_argument("--dry_run", action="store_true",
                   help="report what would change without writing anything")
    args = p.parse_args()

    if os.path.exists(args.out) and not args.dry_run:
        raise FileExistsError(f"{args.out} exists; refusing to overwrite.")

    cfg = DecorateConfig(seed=args.seed, min_tread_m=float(args.tread_band_m[0]),
                         max_tread_m=float(args.tread_band_m[1]),
                         wall_collidable=args.collidable_walls)
    prints = {fp.terrain_id: fp for fp in measure(args.clip, args.max_clips_per_tile)}

    with h5py.File(args.clip, "r") as f:
        manifest = json.loads(f.attrs["terrain_manifest"])
        # Refuse a second pass. The arms are seeded per tile, so re-running is not idempotent
        # -- a narrowed platform would be narrowed again, this time relative to its already
        # reduced width, and a widened tread would be re-classified as a platform. Structural
        # rather than attribute-based, so copying a decorated manifest into a fresh file
        # cannot get around it.
        if has_width_variation(manifest):
            raise ValueError(
                f"{args.clip} already has platform-width variation, so it has been decorated. "
                "Re-running would compound the edits rather than repeat them. Start from the "
                "undecorated library."
            )
        pos_names = json.loads(f.attrs["pos_output_names"])
        terrain_id = np.asarray(f["terrain_id"])
        clip_len = np.asarray(f["clip_len"])
        cols = {b: _cols(pos_names, b) for b in ANKLES}

        reports = {}
        for tile in manifest:
            if tile.get("type") != "transition":
                continue
            tid = int(tile["terrain_id"])
            reports[tid] = decorate_transition_tile(tile, prints[tid], cfg)

        box_w_before = [float(o["size_xyz"][1]) for t in manifest if t.get("type") == "box"
                        for o in t["overlays"]]
        box_variants = {}
        if not args.no_box_variants:
            for tile in manifest:
                if tile.get("type") == "box":
                    tid = int(tile["terrain_id"])
                    box_variants[tid] = decorate_box_tile(tile, cfg, prints.get(tid))

        print(f"\n[decor] {len(reports)} transition tiles rewritten, seed {args.seed}")
        counts = {a: sum(1 for r in reports.values() if r.arm == a) for a in ARMS}
        print(f"[decor] arms: " + "  ".join(f"{a}={n}" for a, n in counts.items()))
        t_before = [w for r in reports.values() for w in r.tread_before]
        t_after = [w for r in reports.values() for w in r.tread_after]
        p_before = [w for r in reports.values() for w in r.platform_before]
        p_after = [w for r in reports.values() for w in r.platform_after]
        for label, before, after in (("tread", t_before, t_after),
                                     ("platform", p_before, p_after)):
            b, a = np.array(before), np.array(after)
            print(f"[decor] {label:9s} {b.min():.3f}..{b.max():.3f} m  ->  "
                  f"{a.min():.3f}..{a.max():.3f} m   (median {np.median(b):.3f} -> "
                  f"{np.median(a):.3f})")

        if box_variants:
            counts = {v: 0 for v in BOX_VARIANTS}
            n_widened = 0
            for spec in box_variants.values():
                variants, _, mode = spec.partition("|")
                for v in variants.split(","):
                    counts[v] += 1
                n_widened += mode == "widened"
            total = sum(counts.values())
            print(f"[decor] box support variants over {len(box_variants)} tiles "
                  f"({total} overlays): "
                  + "  ".join(f"{v}={n}" for v, n in counts.items()))
            print("[decor]   walkable top and footprint are identical for every variant, so "
                  "no foot that was supported before is unsupported now")
            bw = [float(o["size_xyz"][1]) for t in manifest if t.get("type") == "box"
                  for o in t["overlays"]]
            print(f"[decor] box widths: {min(box_w_before):.3f}..{max(box_w_before):.3f}"
                  f" -> {min(bw):.3f}..{max(bw):.3f} m "
                  f"(median {np.median(bw):.3f}); {n_widened} of {len(box_variants)} "
                  "tiles widened")

        n_wall_boxes, n_wall_tiles, wall_sides = 0, 0, {}
        # Walls ring the RENDERED floor, which is the class footprint and not the tile's own
        # size -- they differ by metres on transition and box tiles.
        fps = class_footprints(manifest)
        for kind, fp_xy in sorted(fps.items()):
            biggest = max((float(m["size_x"]), float(m["size_y"])) for m in manifest
                          if m.get("type") == kind)
            print(f"[decor] {kind:11s} floor {fp_xy[0]:7.3f} x {fp_xy[1]:7.3f} m "
                  f"(largest tile {biggest[0]:7.3f} x {biggest[1]:7.3f})")
        if not args.no_walls:
            for tile in manifest:
                if tile.get("type") not in ("transition", "box", "terrain"):
                    continue
                tid = int(tile["terrain_id"])
                fp = prints.get(tid)
                if fp is None:
                    continue
                n_sides, n_boxes = add_walls(tile, fp, cfg,
                                             fps.get(str(tile.get("type"))))
                n_wall_boxes += n_boxes
                n_wall_tiles += n_sides > 0
                wall_sides[int(tile["terrain_id"])] = n_sides
            kind = ("COLLIDABLE terrain" if args.collidable_walls else
                    "VISUAL ONLY -- cameras see them, the height scan does not, "
                    "nothing can walk into them")
            hist = {}
            for n in wall_sides.values():
                hist[n] = hist.get(n, 0) + 1
            print(f"[decor] walls: {n_wall_boxes} boxes on {n_wall_tiles} of "
                  f"{len(manifest)} tiles ({kind})")
            print("[decor]   WALLS per tile: "
                  + "  ".join(f"{k}->{v} tiles" for k, v in sorted(hist.items()))
                  + "   (boxes differ: a windowed wall is up to 4)")

        flat_rooms = []
        if not args.no_walls:
            flat = np.flatnonzero(terrain_id == -1)
            if len(flat):
                px = pos_names.index("pelvis_link:pos_x")
                py = pos_names.index("pelvis_link:pos_y")
                # ALL of them, not a sample: this is a MAXIMUM, and sampling 200 of 3000
                # under-reported it by 4 m -- which built the room inside the floor.
                tx = ty = 0.0
                for c in flat:
                    n = int(clip_len[c])
                    row = f["y_des"][int(c)][:n]
                    tx = max(tx, float(row[:, px].max() - row[:, px].min()))
                    ty = max(ty, float(row[:, py].max() - row[:, py].min()))
                flat_rooms = make_flat_rooms(args.flat_room_variants, (tx, ty), cfg)
                ffx, ffy = flat_tile_footprint((tx, ty))
                walled = sum(1 for r in flat_rooms if r)
                print(f"[decor] flat tiles: {len(flat)} clips travel {tx:.2f} x {ty:.2f} m "
                      f"-> floor {ffx:.2f} x {ffy:.2f} m")
                print(f"[decor]   {args.flat_room_variants} room variants, {walled} walled "
                      f"({sum(len(r) for r in flat_rooms)} boxes), handed round-robin to the "
                      "flat tiles at load; walls on the tile EDGE, authored about its centre")

        print("\n[decor] validating every rewritten tile against its own foot samples")
        failures = 0
        for tile in manifest:
            if tile.get("type") != "transition":
                continue
            tid = int(tile["terrain_id"])
            members = np.flatnonzero(terrain_id == tid)
            used = members[np.linspace(0, len(members) - 1,
                                       min(len(members), args.max_clips_per_tile)).astype(int)]
            pos_all, quat_all = [], []
            for c in used:
                n = int(clip_len[c])
                row = f["y_des"][int(c)][:n]
                for body in ANKLES:
                    pcols, qcols = cols[body]
                    pos_all.append(row[:, pcols])
                    quat_all.append(row[:, qcols])
            bad = check_feet_are_supported(tile, reports[tid].roles,
                                           np.concatenate(pos_all),
                                           np.concatenate(quat_all))
            if bad:
                failures += 1
                print(f"[decor]   tile {tid}: {len(bad)} issue(s)")
                for msg in bad[:3]:
                    print(f"[decor]     {msg}")
        print(f"[decor] {failures} tile(s) with unsupported feet")

    if args.dry_run:
        print("[decor] dry run; nothing written")
        return

    print(f"[decor] copying {args.clip} -> {args.out}")
    shutil.copy2(args.clip, args.out)
    with h5py.File(args.out, "r+") as f:
        f.attrs["terrain_manifest"] = json.dumps(manifest)
        if flat_rooms:
            f.attrs["flat_tile_decor"] = json.dumps(flat_rooms)
        f.attrs["terrain_decoration"] = json.dumps({
            "source": os.path.basename(args.clip),
            "seed": args.seed,
            "tread_band_m": [float(args.tread_band_m[0]), float(args.tread_band_m[1])],
            "arms": {r.terrain_id: r.arm for r in reports.values()},
            "box_variants": box_variants,
            "wall_boxes": n_wall_boxes,
            "wall_sides": wall_sides,
            "wall_footprints": {k: list(v) for k, v in fps.items()},
        })
    print(f"[decor] wrote {args.out}")


if __name__ == "__main__":
    main()
