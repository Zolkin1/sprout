#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Author visual clutter for a library, as a SIDECAR, vetted against a closed-loop recording.

One variant per run. The output is a few hundred KB of JSON, not a copy of the library: only
the manifest's decoration changes, and the point of this pass is to produce SEVERAL variants
-- a tile's clutter is identical in every window on it, and with a couple of hundred tiles a
single variant is memorisable as "terrain_id 37 has a wall at x=12" rather than learnable as
"a wall is not terrain". Copying 6.3 GB per variant to change one attribute would put the
cost in exactly the wrong place.

**Variant 0 is the library as it stands, and it costs nothing.** The undecorated recording
already exists; it goes into the training mix unchanged. Everything here is added ON TOP of
that, so the model keeps seeing the distribution it was trained on.

**The veto is the recording, not the clips** -- see ``build_clutter_occupancy.py`` for why,
and ``motion_gen.terrain.clutter`` for the two tiers and the approach zone. Nothing is placed
here that the recorded robot ever occupied.

    python scripts/rsl_rl/build_clutter_occupancy.py --data <rec>.h5 --out occ.npz
    for v in 1 2 3; do
      python scripts/rsl_rl/author_clutter.py --clip <lib>.h5 --occupancy occ.npz \\
          --seed $v --out clutter_v$v.json
    done
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "source", "motion_gen"))

from motion_gen.terrain.clutter import (  # noqa: E402
    APPROACH_ZONE_M, BODY_RADIUS_M, CLEARANCE_M, INTERACTABLE, ClutterConfig,
    TileOccupancy, aabb,
    approach_zone_clear, clears_occupancy, clutter_for_flat, clutter_for_tile,
)

#: ``terrain_id`` every flat tile shares; they have no manifest entry of their own.
_FLAT_ID: int = -1

#: Beyond this the far camera cannot resolve anything useful, so a box out here is scenery
#: rather than a distractor. Used only to REPORT how much of a variant is in play.
FAR_CAMERA_REACH_M: float = 8.0


def load_occupancy(path: str) -> tuple[dict[int, TileOccupancy], dict[int, tuple[float, float]],
                                       dict[int, np.ndarray]]:
    """Read the per-``terrain_id`` occupancy, tile sizes and raw path points."""
    z = np.load(path)
    out: dict[int, TileOccupancy] = {}
    sizes: dict[int, tuple[float, float]] = {}
    pts: dict[int, np.ndarray] = {}
    for k in z["terrain_ids"]:
        k = int(k)
        cell, ox, oy, n, sx, sy = z[f"meta_{k}"]
        out[k] = TileOccupancy(k, float(cell), (float(ox), float(oy)), z[f"grid_{k}"], int(n))
        sizes[k] = (float(sx), float(sy))
        pts[k] = z[f"pts_{k}"] if f"pts_{k}" in z.files else np.zeros((0, 2), np.float32)
    return out, sizes, pts


def overlapping(boxes: list[dict], pts: np.ndarray, pad: float) -> list[dict]:
    """Boxes whose padded footprint contains a recorded path point.

    The INDEPENDENT check. Placement vets against a rasterised grid, which discretises at the
    cell size and is only as continuous as the densify pass made it; this vets against the
    recorded points themselves, so a bug in either one cannot pass both. It is the check that
    would have caught clutter sitting in a hole between two samples -- a real failure, seen on
    pure-stair, transition and box tiles before the path was densified.

    Args:
        boxes: Authored overlays for one tile.
        pts: ``(N, 2)`` recorded positions in that tile's frame.
        pad: Clearance added around each box (m).

    Returns:
        The offending boxes.
    """
    if not len(pts):
        return []
    bad = []
    for b in boxes:
        x0, y0, x1, y1 = aabb(b)
        if ((pts[:, 0] > x0 - pad) & (pts[:, 0] < x1 + pad)
                & (pts[:, 1] > y0 - pad) & (pts[:, 1] < y1 + pad)).any():
            bad.append(b)
    return bad


def nearest_pose_distance(b: dict, occ: TileOccupancy) -> float:
    """Distance from a box to the closest cell the robot was recorded in (m).

    Reported rather than enforced: the veto already guarantees a positive distance, and what
    this adds is whether a box is close enough to ever matter. A wall 30 m away is safe and
    useless.
    """
    ij = np.argwhere(occ.grid)
    if not len(ij):
        return float("inf")
    ox, oy = occ.origin
    cx = ox + (ij[:, 0] + 0.5) * occ.cell
    cy = oy + (ij[:, 1] + 0.5) * occ.cell
    x0, y0, x1, y1 = aabb(b)
    dx = np.maximum(np.maximum(x0 - cx, cx - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - cy, cy - y1), 0.0)
    return float(np.min(np.hypot(dx, dy)))


def verify(clip: str, occupancy: str, sidecar: str, show: int = 15) -> int:
    """Report how close an authored sidecar comes to the paths, and any true intersections.

    Placement already vets and re-checks, so this is for AFTER the fact -- to settle whether
    something seen in a 3-D view is a real overlap or a projection, and to say which tile it
    is on so the question can be answered rather than argued.

    Args:
        clip: The library the sidecar was authored against.
        occupancy: The occupancy the veto used.
        sidecar: The sidecar to check.
        show: How many of the closest volumes to list.

    Returns:
        ``0`` if nothing intersects, ``1`` otherwise.
    """
    occ, sizes, raw = load_occupancy(occupancy)
    doc = json.load(open(sidecar))
    pad = BODY_RADIUS_M + CLEARANCE_M
    with h5py.File(clip, "r") as f:
        man = json.loads(f.attrs["terrain_manifest"])
        fam = {int(m["terrain_id"]): str(m.get("family", "?")).replace(".h5", "") for m in man}
        rows, bad = [], 0
        for t, boxes in doc["per_terrain_id"].items():
            k = int(t)
            pts = raw.get(k)
            if pts is None or not len(pts):
                continue
            for b in boxes:
                x0, y0, x1, y1 = aabb(b)
                dx = np.maximum(np.maximum(x0 - pts[:, 0], pts[:, 0] - x1), 0.0)
                dy = np.maximum(np.maximum(y0 - pts[:, 1], pts[:, 1] - y1), 0.0)
                d = float(np.min(np.hypot(dx, dy)))
                rows.append((d, fam.get(k, "?"), k, b["clutter_class"], b["kind"]))
                bad += d < pad
    rows.sort()
    print(f"[verify] {sidecar}: {len(rows)} terrain volumes, veto distance {pad:.2f} m")
    print(f"  {'dist':>7s} {'family':<22s} {'tid':>5s} {'class':<12s} kind")
    for d, fm, k, c, kd in rows[:show]:
        flag = "  <-- INSIDE THE VETO" if d < pad else ""
        print(f"  {d:7.3f} {fm:<22s} {k:5d} {c:<12s} {kd}{flag}")
    print(f"[verify] {bad} volume(s) inside the veto distance")
    return 1 if bad else 0


def main() -> int:
    """Author one clutter variant and write it as a sidecar."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Clip library (read only, never written).")
    p.add_argument("--occupancy", required=True, help=".npz from build_clutter_occupancy.py")
    p.add_argument("--out", default=None,
                   help="Sidecar .json to write. Not needed with --verify, which "
                        "writes nothing.")
    p.add_argument("--verify", default=None, metavar="SIDECAR.json",
                   help="Re-check an existing sidecar instead of authoring one: report every "
                        "volume that comes within the veto distance of a recorded pose or a "
                        "clip path, and list the closest ones by tile. For settling whether "
                        "something seen in the viewer is a real intersection or a projection "
                        "-- a 2.5 m pillar a metre to the side reads as 'on the path' from "
                        "most angles.")
    p.add_argument("--seed", type=int, default=1,
                   help="Variant id. Every tile draws from its own stream seeded by this, so "
                        "two variants share no placement and either is reproducible alone.")
    p.add_argument("--flat_variants", type=int, default=0,
                   help="Flat-tile variants to author. 0 = match the library's own count, so "
                        "the round-robin over flat tiles keeps its period. Flat tiles are "
                        "30%% of training and carry no manifest entry, so they are authored "
                        "separately -- and they are the case that most needs it: at the "
                        "shipped buffer NO flat clip has a wall within camera reach.")
    p.add_argument("--min_visible_frac", type=float, default=0.25,
                   help="Refuse to write if fewer than this fraction of boxes land within "
                        "the far camera's reach of the recorded path. A variant the cameras "
                        "never see is a re-render that costs hours and changes nothing -- "
                        "the failure this repo already made once, with the flat rooms at a "
                        "20 m buffer.")
    a = p.parse_args()

    if a.verify:
        return verify(a.clip, a.occupancy, a.verify)
    if not a.out:
        p.error("--out is required when authoring (omit it only with --verify)")

    with h5py.File(a.clip, "r") as f:
        manifest = json.loads(f.attrs["terrain_manifest"])
        raw_flat = f.attrs["flat_tile_decor"] if "flat_tile_decor" in f.attrs else None
    existing_flat = (json.loads(raw_flat) if isinstance(raw_flat, str)
                     else list(raw_flat or [])) if raw_flat is not None else []
    occ, sizes, raw = load_occupancy(a.occupancy)
    cfg = ClutterConfig(seed=a.seed)

    per_tile: dict[str, list[dict]] = {}
    stats: dict[str, int] = {}
    dists: list[float] = []
    for tile in manifest:
        tid = int(tile["terrain_id"])
        if tid not in occ:
            continue                      # no recorded pose on this tile: nothing to vet against
        boxes = clutter_for_tile(tile, occ[tid], cfg)
        if not boxes:
            continue
        per_tile[str(tid)] = boxes
        for b in boxes:
            stats[b["clutter_class"]] = stats.get(b["clutter_class"], 0) + 1
            dists.append(nearest_pose_distance(b, occ[tid]))

    # V1 / V5: re-assert the vetoes on the written result rather than trusting the sampler.
    # The check is cheap and the failure -- a box on top of the robot -- is invisible in every
    # shape test downstream.
    bad = [(t, b["clutter_class"]) for t, bs in per_tile.items() for b in bs
           if not clears_occupancy(b, occ[int(t)])]
    bad_zone = [(t, b["clutter_class"]) for t, bs in per_tile.items() for b in bs
                if b["clutter_class"] in INTERACTABLE and not approach_zone_clear(b, occ[int(t)])]
    if bad or bad_zone:
        raise SystemExit(f"grid veto violated after authoring: {bad[:5]} {bad_zone[:5]}")

    # The independent pass: same question, different evidence.
    pad = BODY_RADIUS_M + CLEARANCE_M
    hits = {t: overlapping(bs, raw.get(int(t), np.zeros((0, 2), np.float32)), pad)
            for t, bs in per_tile.items()}
    n_hit = sum(len(v) for v in hits.values())
    for t, bs in list(hits.items()):
        if bs:
            keep = [b for b in per_tile[t] if b not in bs]
            per_tile[t] = keep
            for b in bs:
                stats[b["clutter_class"]] -= 1
    if n_hit:
        print(f"[clutter] overlap check against the raw path: DROPPED {n_hit} box(es) the "
              "grid let through (the grid discretises; the points do not)")

    # FLAT tiles. They share terrain_id -1 and any flat clip can be drawn onto any of them,
    # so the veto is the POOLED occupancy of every flat clip and the output is a list of
    # variants handed out round-robin by tile index -- the same mechanism the library's own
    # flat rooms already use.
    flat_variants: list[list[dict]] = []
    if _FLAT_ID in occ:
        n_var = a.flat_variants or max(1, len(existing_flat))
        fsx, fsy = sizes[_FLAT_ID]
        for v in range(n_var):
            boxes = clutter_for_flat(occ[_FLAT_ID], fsx, fsy, cfg, v)
            flat_variants.append(boxes)
            for b in boxes:
                stats[b["clutter_class"]] = stats.get(b["clutter_class"], 0) + 1
        n_flat = sum(len(v) for v in flat_variants)
        print(f"[clutter] flat tiles ({fsx:.0f} x {fsy:.0f} m): {n_flat} volumes over "
              f"{n_var} variants, vetted against {occ[_FLAT_ID].n_samples} pooled poses")
        # The same independent check for the flat variants, done in the tile-CORNER frame
        # the occupancy lives in -- the variants themselves are centre-relative.
        fpts = raw.get(_FLAT_ID, np.zeros((0, 2), np.float32))
        for i, boxes in enumerate(flat_variants):
            shifted = [{**b, "pos_xy": [b["pos_xy"][0] + fsx / 2.0, b["pos_xy"][1] + fsy / 2.0]}
                       for b in boxes]
            bad_f = overlapping(shifted, fpts, pad)
            if bad_f:
                idx = {id(x) for x in bad_f}
                flat_variants[i] = [b for b, sh in zip(boxes, shifted) if id(sh) not in idx]
                print(f"[clutter] flat variant {i}: dropped {len(bad_f)} overlapping box(es)")
        # Flat distances are measured in the same frame the veto used (tile corner), so the
        # centre-relative output has to be shifted back to be comparable.
        for boxes in flat_variants:
            for b in boxes:
                shifted = {**b, "pos_xy": [b["pos_xy"][0] + fsx / 2.0,
                                           b["pos_xy"][1] + fsy / 2.0]}
                dists.append(nearest_pose_distance(shifted, occ[_FLAT_ID]))
    else:
        print("[clutter] no flat occupancy in this recording; flat tiles get nothing")

    d = np.asarray(dists) if dists else np.zeros(0)
    visible = float((d < FAR_CAMERA_REACH_M).mean()) if len(d) else 0.0
    print(f"[clutter] variant seed={a.seed}: {sum(len(v) for v in per_tile.values())} boxes "
          f"on {len(per_tile)} of {len(manifest)} tiles")
    for k in sorted(stats):
        print(f"    {k:<14s} {stats[k]:5d}")
    if len(d):
        print(f"[clutter] distance from the recorded path: p10 {np.percentile(d, 10):.2f} m, "
              f"median {np.median(d):.2f} m, p90 {np.percentile(d, 90):.2f} m")
        print(f"[clutter] within the far camera's {FAR_CAMERA_REACH_M:.0f} m reach: "
              f"{visible:.1%} of boxes")
    if visible < a.min_visible_frac:
        raise SystemExit(
            f"only {visible:.1%} of boxes are within {FAR_CAMERA_REACH_M:.0f} m of the "
            f"recorded path, under --min_visible_frac {a.min_visible_frac:.0%}. Re-rendering "
            "this would burn hours to change almost no pixels.")

    doc = {"schema": 1, "source": os.path.basename(a.clip), "seed": a.seed,
           "occupancy": os.path.basename(a.occupancy),
           "approach_zone_m": APPROACH_ZONE_M, "body_radius_m": BODY_RADIUS_M,
           "counts": stats, "visible_frac": visible,
           "has_flat_decor_in_source": bool(existing_flat),
           "per_terrain_id": per_tile, "flat_variants": flat_variants}
    with open(a.out, "w") as fh:
        json.dump(doc, fh)
    print(f"[clutter] wrote {a.out} ({os.path.getsize(a.out) / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
