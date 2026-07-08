#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate a flat motion-matching reference clip (stand -> run -> walk -> stand).

Builds the motion database (traj-opt strides + filtered bones-seed clips), runs the
:class:`~motion_gen.motion_matching.mm_clip_generator.MotionMatchClipGenerator` over a
scripted skill/speed schedule, and writes a standard ``.h5`` clip (same format as
``generate_clips.py``) plus loads it back for a sanity summary.

Runs with NO IsaacLab -- pure motion_gen.  Example::

    python scripts/rsl_rl/generate_mm_clips.py \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --dataset /home/zolkin/AmberLab/datasets/bones-seed \
        --out clips/g1_mm_flat_v1.h5 --device cpu --bones-limit 200
"""

from __future__ import annotations

import argparse

import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.motion_matching import features as F
from motion_gen.motion_matching.bones_ingest import ingest_bones, load_cache
from motion_gen.motion_matching.mm_clip_generator import MotionMatchClipGenerator
from motion_gen.motion_matching.motion_db import MotionDB
from motion_gen.motion_matching.trajopt_ingest import ingest_trajopt
from motion_gen.multiskill_manager import MultiSkillManager

_CONTACT_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt stride library root.")
    p.add_argument("--out", required=True, help="Output .h5 clip path.")
    p.add_argument("--dataset", default=None, help="bones-seed dataset dir (live ingest; omit to skip bones).")
    p.add_argument("--bones-cache", default=None,
                   help="Precomputed bones cache .h5 (from precompute_bones_cache.py); "
                        "preferred over --dataset (no CSV parse / FK at run time).")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--clip-length", type=float, default=9.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bones-limit", type=int, default=None, help="Cap bones clips (default: all that pass the filter).")
    p.add_argument("--bones-package", default="Locomotion")
    p.add_argument("--bones-style", default=None, help="Comma list of styles (default: any).")
    p.add_argument("--bones-movement", default=None, help="Comma list of content_type_of_movement (e.g. walking,jogging).")
    p.add_argument("--bones-body", default=None, help="Comma list of content_body_position (e.g. standing).")
    p.add_argument("--index", dest="index", action="store_true", default=True, help="Build PCA->KD-tree index (default).")
    p.add_argument("--no-index", dest="index", action="store_false", help="Use exact brute-force search instead.")
    p.add_argument("--n-pca", type=int, default=48, help="PCA dims for the KD-tree index.")
    p.add_argument("--skills", default="standing,running,walk_forward",
                   help="Traj-opt skills to ingest into the DB.")
    p.add_argument("--run-vx", type=float, default=2.5)
    p.add_argument("--walk-vx", type=float, default=1.0)
    p.add_argument("--w-vel", type=float, default=3.0)
    p.add_argument("--db-out", default=None, help="Optional .npz to cache the built DB.")
    args = p.parse_args()

    dev = args.device
    skills = [s for s in args.skills.split(",") if s]
    manager = MultiSkillManager(args.source, dev, env=None)
    cfg = F.FeatureConfig(dt=args.dt)
    fk = F.make_fk(cfg, device=dev)

    print(f"[mm] ingesting traj-opt strides for skills={skills} ...")
    sources = ingest_trajopt(manager, dt=args.dt, skills=skills, device=dev)

    def _csv_set(s):
        return None if not s else [x.strip() for x in s.split(",") if x.strip()]

    bfilters = {"package": args.bones_package}
    if args.bones_movement:
        bfilters["content_type_of_movement"] = _csv_set(args.bones_movement)
    if args.bones_body:
        bfilters["content_body_position"] = _csv_set(args.bones_body)

    if args.bones_cache:
        print(f"[mm] loading bones cache {args.bones_cache} (filters={bfilters}, style={args.bones_style}) ...")
        bsrc, _ = load_cache(args.bones_cache, filters=bfilters, styles=_csv_set(args.bones_style),
                             limit=args.bones_limit, device=dev)
        sources += bsrc
        print(f"[mm]   {len(bsrc)} bones clips from cache")
    elif args.dataset:
        print(f"[mm] live-ingesting bones-seed (filters={bfilters}, limit={args.bones_limit}) ...")
        sources += ingest_bones(args.dataset, filters=bfilters, styles=_csv_set(args.bones_style),
                                limit=args.bones_limit, dt=args.dt, device=dev)

    mode = "streaming PCA->KD-tree" if args.index else "dense brute-force"
    print(f"[mm] building DB from {len(sources)} sources ({mode}) ...")
    db = MotionDB(cfg, dev).build(sources, fk, index=args.index, n_pca=args.n_pca, w_vel=args.w_vel)
    print(f"[mm] DB frames = {db.owner.shape[0]}")
    if args.db_out:
        db.save(args.db_out)

    gen = MotionMatchClipGenerator(manager, db, sources, cfg, dt=args.dt, device=dev, w_vel=args.w_vel)
    schedule = [
        {"skill": "standing", "vx": 0.0, "wz": 0.0, "duration_s": 1.5},
        {"skill": "running", "vx": args.run_vx, "wz": 0.0, "duration_s": 3.0},
        {"skill": "walk_forward", "vx": args.walk_vx, "wz": 0.0, "duration_s": 3.0},
        {"skill": "standing", "vx": 0.0, "wz": 0.0, "duration_s": 1.5},
    ]
    print("[mm] generating clip: stand -> run -> walk -> stand ...")
    batch = gen.generate(schedule, clip_length_s=args.clip_length)
    T = batch["y_des"].shape[1]

    meta = {
        "dt": args.dt, "clip_length_s": args.clip_length, "num_steps": int(T),
        "pos_output_names": list(manager.pos_output_names),
        "vel_output_names": list(manager.vel_output_names),
        "contact_frames": _CONTACT_FRAMES,
        "ref_frame": "pelvis_link",             # root-driven clips: pelvis is the anchor
        "skill_list": [s.name for s in manager.skills],
    }
    writer = ClipLibraryWriter(args.out, 1, meta)
    writer.set_attr("source_library_fingerprint", manager.library_fingerprint)
    writer.set_attr("generator", "motion_matching_v1")
    writer.write_batch(0, batch)
    writer.close()
    print(f"[mm] wrote {args.out}")

    if gen.warnings:
        print(f"[mm] {len(gen.warnings)} transition warning(s) (surfaced, not smoothed):")
        for w in gen.warnings:
            print("   -", w)

    lib = load_clip_library(args.out)
    y = torch.as_tensor(lib.y_des)
    print(f"[mm] loaded back: y_des {tuple(y.shape)} finite={bool(torch.isfinite(y).all())}, "
          f"MM transition frames={int((torch.as_tensor(lib.stride_idx)[0] == -1).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
