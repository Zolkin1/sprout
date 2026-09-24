# Copyright 2026 Zachary Olkin. All rights reserved.

"""Report, per tile, the lateral extent its clips' feet actually cover.

Usage::

    PYTHONPATH=source/motion_gen python scripts/rsl_rl/measure_tile_footprints.py \\
        --clip clips/g1_mb_v5_body_mode11.h5 --out /tmp/footprints.json

This is the measurement every terrain edit is sized against, and it is a SEPARATE step from
making those edits on purpose: the numbers are worth looking at before anything is rewritten,
and the decoration pass then consumes them rather than recomputing them differently.

What it prints per tile: how wide the tile is, how wide its motion actually is, and how much
of the tile is therefore unused. On a transition tile the treads are ~1 m in a 3-8 m tile, so
the interesting column is the one showing how far a landing could be narrowed -- or how far a
tread could grow -- before it stops covering the feet.

Read ``motion_gen.terrain.footprint`` for why a foot is not a point and why the envelope
carries a 0.30 m ego-anchor allowance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "motion_gen"))

from motion_gen.terrain.footprint import (EGO_ANCHOR_ALLOWANCE_M,  # noqa: E402
                                          TileFootprint, tile_footprint)

#: The two bodies whose boxes touch the ground.
ANKLES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _cols(names: list[str], body: str) -> tuple[list[int], list[int]]:
    """Position and orientation column indices for ``body``'s world block."""
    idx = {n: i for i, n in enumerate(names)}
    return ([idx[f"{body}:pos_{a}"] for a in "xyz"],
            [idx[f"{body}:ori_{a}"] for a in "wxyz"])


def measure(clip_path: str, max_clips_per_tile: int | None) -> list[TileFootprint]:
    """Measure every tile in ``clip_path``.

    Args:
        clip_path: The clip library.
        max_clips_per_tile: Cap on clips sampled per tile, or ``None`` for all. A cap is a
            speed/coverage trade and is REPORTED when it binds -- a silently truncated
            envelope is one that reads as narrower than the motion really is.

    Returns:
        One envelope per tile carrying clips, ordered by ``terrain_id``.
    """
    with h5py.File(clip_path, "r") as f:
        pos_names = json.loads(f.attrs["pos_output_names"])
        terrain_id = np.asarray(f["terrain_id"])
        clip_len = np.asarray(f["clip_len"])
        cols = {b: _cols(pos_names, b) for b in ANKLES}

        out: list[TileFootprint] = []
        tiles = sorted({int(t) for t in terrain_id.tolist() if int(t) >= 0})
        for tid in tiles:
            members = np.flatnonzero(terrain_id == tid)
            used = members
            if max_clips_per_tile is not None and len(members) > max_clips_per_tile:
                used = members[np.linspace(0, len(members) - 1,
                                           max_clips_per_tile).astype(int)]
            pos_all, quat_all = [], []
            # The TRAVERSAL feet, kept separately. `anchor_mask_t == 2` marks the world-pinned
            # terrain replay -- the frames whose feet are actually on the flight. Sizing a flight
            # from the whole clip widens it to carry the approach and exit, which happen on the
            # flat PLATFORMS and wander ~1 m between clips; see TileFootprint.tread_envelope.
            tread_pos, tread_quat = [], []
            has_mask = "anchor_mask_t" in f
            for c in used:
                # clip_len bounds the real frames; the tail is padding that repeats the last
                # sample, which would weight the envelope towards wherever the clip ended.
                n = int(clip_len[c])
                row = f["y_des"][int(c)][:n]
                on_terrain = (np.asarray(f["anchor_mask_t"][int(c)][:n]) == 2) if has_mask else None
                for body in ANKLES:
                    pcols, qcols = cols[body]
                    pos_all.append(row[:, pcols])
                    quat_all.append(row[:, qcols])
                    if on_terrain is not None and bool(on_terrain.any()):
                        tread_pos.append(row[on_terrain][:, pcols])
                        tread_quat.append(row[on_terrain][:, qcols])
            fp = tile_footprint(np.concatenate(pos_all), np.concatenate(quat_all),
                                tid, len(members),
                                tread_pos=(np.concatenate(tread_pos) if tread_pos else None),
                                tread_quat=(np.concatenate(tread_quat) if tread_quat else None))
            out.append(fp)
            if len(used) < len(members):
                print(f"[footprint]   tile {tid}: sampled {len(used)} of {len(members)} clips")
    return out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="clip library .h5")
    p.add_argument("--out", default=None, help="write the envelopes here as JSON")
    p.add_argument("--max_clips_per_tile", type=int, default=None,
                   help="cap clips sampled per tile (default: all)")
    args = p.parse_args()

    with h5py.File(args.clip, "r") as f:
        manifest = {int(m["terrain_id"]): m for m in json.loads(f.attrs["terrain_manifest"])}

    prints = measure(args.clip, args.max_clips_per_tile)
    print(f"\n[footprint] {len(prints)} tiles from {args.clip}")
    print(f"[footprint] envelopes carry +/-{EGO_ANCHOR_ALLOWANCE_M} m of ego-anchor allowance\n")
    header = (f"{'tile':>5s} {'type':<11s} {'clips':>6s} {'size_y':>8s} "
              f"{'foot y span':>22s} {'sized':>8s} {'unused':>8s}")
    print(header)
    print("-" * len(header))
    by_type: dict[str, list[float]] = {}
    for fp in prints:
        entry = manifest.get(fp.terrain_id, {})
        kind = str(entry.get("type", "?"))
        size_y = float(entry.get("size_y", float("nan")))
        unused = size_y - fp.width_sized
        by_type.setdefault(kind, []).append(unused)
        print(f"{fp.terrain_id:5d} {kind:<11s} {fp.n_clips:6d} {size_y:8.3f} "
              f"[{fp.y_min:+7.3f},{fp.y_max:+7.3f}] {fp.width_sized:8.3f} {unused:8.3f}")

    print("\n[footprint] unused width (tile size_y minus the sized envelope), by type:")
    for kind, vals in sorted(by_type.items()):
        arr = np.array(vals)
        print(f"[footprint]   {kind:<11s} n={len(arr):3d}  min {arr.min():+7.3f}  "
              f"median {np.median(arr):+7.3f}  max {arr.max():+7.3f} m")
    negative = [fp.terrain_id for fp, in zip(prints)
                if float(manifest.get(fp.terrain_id, {}).get("size_y", 0.0)) < fp.width_sized]
    if negative:
        print(f"[footprint] {len(negative)} tile(s) whose sized envelope EXCEEDS the tile: "
              f"{negative}. Those cannot be narrowed and must not be widened blindly.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump([fp.__dict__ for fp in prints], handle, indent=1)
        print(f"[footprint] wrote {args.out}")


if __name__ == "__main__":
    main()
