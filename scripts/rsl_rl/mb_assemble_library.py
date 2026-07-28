#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""PASS C (env_isaaclab): splice every planned clip against the generated segments -> per-family libraries.

Consumes ``jobs.json`` from pass A (``mb_plan_library.py``) plus the segment bank pass B wrote, and emits one
``.h5`` per terrain family, ready for ``compose_terrain_library.py``.

The unit of work is a TILE, not a clip, because a tile is a reuse unit: many clips share one terrain_id and
the RL importer builds ONE mesh per terrain_id from ONE manifest entry. That only works if every clip on a
tile agrees on the tile's origin and footprint -- and per-clip they do not, because
``terrain_splice.footprint_recenter`` datums each clip on the min of ITS motion and the geometry (an approach
that walks further back than the platform shifts the origin). So each tile gets a **re-datum pass**: one
common origin + one footprint covering every clip, applied to the clips and to the manifest they share.
Without it the mesh matches one clip of the tile and every other clip is offset against it.

Clips are spliced in a fork pool (one worker per tile) and a failing tile or clip is SKIPPED and counted --
a library build must not die on one unreachable geometry. Example::

    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" \
    env_isaaclab/bin/python scripts/rsl_rl/mb_assemble_library.py \
        --work-dir mb_raw/lib_v1 --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --out-dir clips/_mb_families
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mb_terrain_clips import clip_from_plan, make_builder                    # noqa: E402

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library    # noqa: E402
from motion_gen.motion_matching.gen_parallel import default_workers, unit_pool  # noqa: E402
from motion_gen.motionbricks import load_terrain_plan, overlays_for          # noqa: E402
from motion_gen.world_lift import _ANCHOR_MASK_BOX                           # noqa: E402

_CONTACT_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Read-only state shared with the forked workers via copy-on-write (set in main() BEFORE the pool).
_SHARED: dict = {}


def _redatum_tile(recs: list[dict], boxes: list[dict], margin: float, geom: str) -> tuple[float, float]:
    """Put every clip of one tile on a COMMON origin + footprint, in place. Returns ``(size_x, size_y)``.

    Each rec arrives recentred on its own ``corner`` (= its own ``lo - margin``). Recover each clip's extent
    in the pre-recentre frame, take the union, and shift the clips that were datumed further in so all of
    them share the tile's origin. The manifest + overlays are then rebuilt once for that origin, giving the
    single geometry the RL importer will mesh for this terrain_id.

    ``geom`` is the tile's family geometry (``"stairs"`` / ``"box"``), which decides whether the platform
    widening below applies.
    """
    los = [(r["corner"][0] + margin, r["corner"][1] + margin) for r in recs]
    his = [(lo[0] + r["size_x"] - 2.0 * margin, lo[1] + r["size_y"] - 2.0 * margin)
           for lo, r in zip(los, recs)]
    LO = (min(p[0] for p in los), min(p[1] for p in los))
    HI = (max(p[0] for p in his), max(p[1] for p in his))
    CORNER = (LO[0] - margin, LO[1] - margin)
    size_x, size_y = (HI[0] - LO[0]) + 2.0 * margin, (HI[1] - LO[1]) + 2.0 * margin
    overlays = overlays_for(boxes, CORNER)
    # Stair platforms are widened to the tile width by the per-clip path; redo it on the common footprint.
    # STAIRS ONLY, and keyed on the geometry rather than on "more than one overlay": a box tile also has two
    # overlays now (the jump box + its top platform), but its widths are authored deliberately -- stretching
    # the platform across the tile would lay a box-high slab over the whole footprint, margin included.
    if geom == "stairs" and len(overlays) > 1:
        min_sx = min(o["size_xyz"][0] for o in overlays)
        for o in overlays:
            if o["size_xyz"][0] > 1.5 * min_sx:
                o["size_xyz"][1] = size_y
                o["pos_xy"][1] = size_y / 2.0
    # Shift EVERY position channel, not just the root: ``y_des`` carries one column per constraint frame
    # (pelvis + both ankles + wrists), all of them world positions in the tile frame. Moving only the root
    # would leave the feet behind -- the clip would still look self-consistent in a root-only plot while its
    # feet sat metres from the treads. (Same convention as generate_box_clips / generate_distractor_clips.)
    x_ch, y_ch = _SHARED["x_ch"], _SHARED["y_ch"]
    for r in recs:
        dx, dy = r["corner"][0] - CORNER[0], r["corner"][1] - CORNER[1]
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            r["y"][:, x_ch] += dx
            r["y"][:, y_ch] += dy
        r["manifest"] = dict(r["manifest"])
        r["manifest"].update(size_x=float(size_x), size_y=float(size_y), overlays=overlays)
        r["corner"] = CORNER
    return float(size_x), float(size_y)


def _tile_worker(i: int):
    """Splice every clip of tile ``i`` and re-datum them onto one footprint. Returns the tile's records."""
    S = _SHARED
    args, work, builder, dev = S["args"], S["work"], S["builder"], S["dev"]
    key, jobs = S["tiles"][i]
    family, tid = key
    recs, boxes, geom, margin = [], None, "stairs", float(args.footprint_margin)
    for j in jobs:
        try:
            plan = load_terrain_plan(os.path.join(work, j["plan"]))
            # The SEAM treatment is chosen here, independently of how the segment was generated: 'ease'
            # inertializes the join, 'weld' snaps it rigidly. Pinned generation + eased seams is the
            # measured best pair -- the pin buys the boundary POSE, the ease keeps the feet on the surface
            # (welding a pinned stairs_up approach leaves its feet ~0.09 m inside the platform).
            pin = {"ease": "soft", "weld": "hard", "auto": None}[args.seams]
            job_args = SimpleNamespace(
                source=args.source, seam_blend_s=args.seam_blend_s, weld_search=args.weld_search,
                arrival_search=args.arrival_search, approach_pin=pin, exit_pin=pin,
                approach_raw=os.path.join(work, "segments", f"{j['approach_key']}.npz"),
                exit_raw=os.path.join(work, "segments", f"{j['exit_key']}.npz"),
            )
            for side in ("approach_raw", "exit_raw"):
                if not os.path.exists(getattr(job_args, side)):
                    raise FileNotFoundError(f"missing {side}: {getattr(job_args, side)} "
                                            f"(pass B did not generate it)")
            rec = clip_from_plan(plan, job_args, builder, dev)
        except (Exception, SystemExit) as e:
            print(f"[mb_asm][SKIP] {family} tile {tid} clip {j['clip']}: {type(e).__name__}: {e}", flush=True)
            continue
        rec["family"], rec["clip"] = family, int(j["clip"])
        recs.append(rec)
        boxes, geom = plan.boxes, str(plan.meta.get("geom", "stairs"))
    if not recs:
        print(f"[mb_asm][SKIP] {family} tile {tid}: no clip survived", flush=True)
        return []
    sx, sy = _redatum_tile(recs, boxes, margin, geom)
    print(f"[mb_asm] {family} tile {tid}: {len(recs)}/{len(jobs)} clips, footprint {sx:.1f}x{sy:.1f} m",
          flush=True)
    return recs


def _write_family(path: str, recs: list[dict], builder, dt: float, skill: str) -> None:
    """Write one family's clips to ``path``, padding to a common length (repeat-last, as clip_len records)."""
    T = max(int(r["n"]) for r in recs)

    def _pad(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        return x if n >= T else torch.cat([x, x[-1:].expand((T - n,) + tuple(x.shape[1:]))], dim=0)

    meta = {"dt": dt, "clip_length_s": T * dt, "num_steps": T,
            "pos_output_names": list(builder.pos_names), "vel_output_names": list(builder.vel_names),
            "contact_frames": _CONTACT_FRAMES, "ref_frame": "pelvis_link", "skill_list": [skill]}
    writer = ClipLibraryWriter(path, len(recs), meta)
    writer.set_attr("generator", f"motionbricks_terrain_{skill}_v1")
    writer.set_attr("terrain_manifest", [r["manifest"] for r in recs])   # one entry per clip; consumers dedup
    for i, r in enumerate(recs):
        writer.write_batch(i, {
            "y_des": _pad(r["y"]).unsqueeze(0), "dy_des": _pad(r["dy"]).unsqueeze(0),
            "contact": _pad(r["contact"]).unsqueeze(0),
            "skill_id": torch.zeros(1, T, dtype=torch.uint8),
            "stride_idx": torch.full((1, T), -1, dtype=torch.int32),
            "target_vel": _pad(r["target_vel"]).unsqueeze(0),
            "terrain_id": torch.tensor([int(r["manifest"]["terrain_id"])], dtype=torch.int32),
            "terrain_type": r["manifest"]["type"],
            "anchor_mask": torch.tensor([_ANCHOR_MASK_BOX], dtype=torch.uint8),
            "anchor_mask_t": _pad(r["anchor_mask_t"]).unsqueeze(0),
            "clip_len": torch.tensor([int(r["n"])], dtype=torch.int32),
        })
    writer.close()
    lib = load_clip_library(path)
    y = torch.as_tensor(lib.y_des)
    n_tiles = len({int(m["terrain_id"]) for m in lib.attrs["terrain_manifest"]})
    print(f"[mb_asm] wrote {path}: {lib.num_clips} clips x {lib.num_steps} steps over {n_tiles} tiles, "
          f"finite={bool(torch.isfinite(y).all())}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True, help="Pass-A/B work dir (jobs.json, plans/, segments/).")
    p.add_argument("--source", required=True, help="Traj-opt library root (channel layout).")
    p.add_argument("--out-dir", required=True, help="Directory for the per-family .h5 files.")
    p.add_argument("--families", default=None, help="Comma-separated subset to assemble (default: all).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0, help="Parallel tile workers (0 = all cores - 2).")
    p.add_argument("--seams", default="ease", choices=["ease", "weld", "auto"],
                   help="How both seams join the replay: 'ease' (default) inertializes them, 'weld' snaps "
                        "rigidly, 'auto' follows how each segment was generated. Pinned generation + eased "
                        "seams is the measured best combination.")
    p.add_argument("--seam-blend-s", type=float, default=0.25, help="Seam ease window (s) for eased seams.")
    p.add_argument("--weld-search", type=int, default=12)
    p.add_argument("--arrival-search", type=int, default=60,
                   help="Trailing approach frames searched for the in-betweened arrival. Must cover "
                        "--pin-tail, which is longer now that the arrival time is emergent.")
    args = p.parse_args()

    work = args.work_dir
    with open(os.path.join(work, "jobs.json")) as f:
        doc = json.load(f)
    cfg, jobs = doc["config"], doc["jobs"]
    args.footprint_margin = float(cfg.get("footprint_margin", 1.0))
    dt = float(cfg.get("dt", 0.02))
    want = {f.strip() for f in args.families.split(",")} if args.families else None
    if want:
        jobs = [j for j in jobs if j["family"] in want]
    if not jobs:
        raise SystemExit(f"no jobs to assemble in {work}/jobs.json")

    by_tile: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for j in jobs:
        by_tile[(j["family"], int(j["terrain_id"]))].append(j)
    tiles = sorted(by_tile.items())
    os.makedirs(args.out_dir, exist_ok=True)

    builder = make_builder(args, args.device)
    _SHARED.update(args=args, work=work, builder=builder, dev=args.device, tiles=tiles,
                   x_ch=[i for i, n in enumerate(builder.pos_names) if n.endswith(":pos_x")],
                   y_ch=[i for i, n in enumerate(builder.pos_names) if n.endswith(":pos_y")])
    n_workers = default_workers(len(tiles), args.workers)
    print(f"[mb_asm] {len(jobs)} clips over {len(tiles)} tiles, workers={n_workers}")
    recs = unit_pool(_tile_worker, range(len(tiles)), n_workers)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_family[r["family"]].append(r)
    for skill, frecs in sorted(by_family.items()):
        _write_family(os.path.join(args.out_dir, f"family_{skill}.h5"), frecs, builder, dt, skill)
    got, want_n = sum(len(v) for v in by_family.values()), len(jobs)
    print(f"[mb_asm] done: {got}/{want_n} clips assembled"
          + (f" ({want_n - got} SKIPPED)" if got < want_n else "")
          + f" -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
