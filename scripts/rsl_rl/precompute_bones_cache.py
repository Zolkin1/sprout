#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Precompute the bones-seed motion-matching feature cache (once).

Filters a broad scope of the bones-seed dataset (default: all ``Locomotion``, any
style), and for every clip parses -> resamples to ``dt`` (50 Hz) -> finite-difference
velocities -> FK per-frame feature block.  Streams everything to ONE chunked HDF5 so
``generate_mm_clips.py`` can filter down + build a motion DB at run time with no CSV
parsing or FK -- and without recomputing anything.

Runs with NO IsaacLab.  Example (one-time, ~20-40 min for all Locomotion)::

    python scripts/rsl_rl/precompute_bones_cache.py \
        --dataset /home/zolkin/AmberLab/datasets/bones-seed \
        --out clips/bones_cache_locomotion.h5 --device cpu

Cache layout (all per-frame arrays concatenated over clips, fp16, chunked):
  joint_pos[N,21] root_pos[N,3] root_quat[N,4] root_lin_vel[N,3] root_ang_vel[N,3]
  feat[N,pfd]                      # raw per-frame feature block (motion_gen features layout)
  clip_start[C] clip_len[C]        # per-clip frame ranges
  meta_<field>[C]                  # filename/package/category/style/movement/body strings
  attrs: dt, src_fps, include_hands, pfd, feature_frames(json), n_clips, n_frames,
         metadata_csv, feature_signature
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import torch

from motion_gen.motion_matching import features as F
from motion_gen.motion_matching import ingest_common as IC
from motion_gen.motion_matching.bones_ingest import clip_to_state, filter_rows, load_metadata

_STATE = ["joint_pos", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel"]
_META_COLS = {
    "filename": "filename", "package": "package", "category": "category",
    "style": "content_uniform_style", "movement": "content_type_of_movement",
    "body": "content_body_position", "is_mirror": "is_mirror",
}


def _csv_set(s: str | None):
    return None if not s else [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="bones-seed dataset dir.")
    p.add_argument("--out", required=True, help="Output cache .h5 path.")
    p.add_argument("--metadata-csv", default=None)
    p.add_argument("--package", default="Locomotion", help="Package filter (default Locomotion).")
    p.add_argument("--styles", default=None, help="Comma list of styles (default: any).")
    p.add_argument("--movement", default=None, help="Comma list of content_type_of_movement (default: any).")
    p.add_argument("--body", default=None, help="Comma list of content_body_position (default: any).")
    p.add_argument("--include-mirror", action="store_true", help="Keep mirrored clips too.")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--src-fps", type=float, default=120.0)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--flush-clips", type=int, default=400, help="Clips buffered per HDF5 write.")
    p.add_argument("--limit", type=int, default=None, help="Cap clips (debug).")
    args = p.parse_args()

    cfg = F.FeatureConfig(dt=args.dt)
    fk = F.make_fk(cfg, device=args.device)
    pfd = F.per_frame_feature_dim(cfg)
    meta_path = args.metadata_csv or os.path.join(args.dataset, "metadata", "seed_metadata_v004.csv")

    filters = {"package": args.package}
    if args.movement:
        filters["content_type_of_movement"] = _csv_set(args.movement)
    if args.body:
        filters["content_body_position"] = _csv_set(args.body)
    rows = filter_rows(load_metadata(meta_path), filters=filters, styles=_csv_set(args.styles),
                       exclude_mirror=not args.include_mirror)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[precompute] {len(rows)} clips pass the filter -> {args.out}")

    dims = {"joint_pos": 21, "root_pos": 3, "root_quat": 4, "root_lin_vel": 3,
            "root_ang_vel": 3, "feat": pfd}
    f = h5py.File(args.out, "w")
    ds = {k: f.create_dataset(k, shape=(0, d), maxshape=(None, d), dtype="float16",
                              chunks=(4096, d), compression="lzf") for k, d in dims.items()}
    clip_start, clip_len, meta_cols = [], [], {k: [] for k in _META_COLS}

    buf = {k: [] for k in dims}
    n_frames = 0

    def flush():
        nonlocal n_frames
        if not buf["feat"]:
            return
        for k in dims:
            block = np.concatenate(buf[k], axis=0).astype("float16")
            n = ds[k].shape[0]
            ds[k].resize(n + block.shape[0], axis=0)
            ds[k][n:] = block
            buf[k].clear()
        n_frames = ds["feat"].shape[0]

    kept = 0
    for i, r in enumerate(rows):
        rel = r.get("move_g1_path", "").strip()
        if not rel:
            continue
        st = clip_to_state(os.path.join(args.dataset, rel), args.src_fps, args.dt, args.device)
        if st is None or st[0].shape[0] < args.min_frames:
            continue
        joint_pos, root_pos, root_quat, root_lin_vel, root_ang_vel = st
        joint_vel = IC.central_diff(joint_pos, args.dt)
        feat = F.compute_frame_features(joint_pos, joint_vel, root_quat, root_lin_vel,
                                        root_ang_vel, fk, cfg)
        Fi = feat.shape[0]
        clip_start.append(n_frames + sum(b.shape[0] for b in buf["feat"]))
        clip_len.append(Fi)
        for mk, col in _META_COLS.items():
            meta_cols[mk].append(str(r.get(col, "")))
        for k, v in (("joint_pos", joint_pos), ("root_pos", root_pos), ("root_quat", root_quat),
                     ("root_lin_vel", root_lin_vel), ("root_ang_vel", root_ang_vel), ("feat", feat)):
            buf[k].append(v.detach().cpu().numpy())
        kept += 1
        if kept % args.flush_clips == 0:
            flush()
            print(f"[precompute] {kept} clips, {ds['feat'].shape[0]} frames ...", flush=True)
    flush()

    # per-clip table
    f.create_dataset("clip_start", data=np.asarray(clip_start, dtype="int64"))
    f.create_dataset("clip_len", data=np.asarray(clip_len, dtype="int32"))
    strdt = h5py.string_dtype(encoding="utf-8")
    for mk in _META_COLS:
        f.create_dataset(f"meta_{mk}", data=np.asarray(meta_cols[mk], dtype=object), dtype=strdt)
    f.attrs["dt"] = json.dumps(args.dt)
    f.attrs["src_fps"] = json.dumps(args.src_fps)
    f.attrs["include_hands"] = json.dumps(cfg.include_hands)
    f.attrs["pfd"] = json.dumps(pfd)
    f.attrs["feature_frames"] = json.dumps(cfg.frames())
    f.attrs["feature_signature"] = json.dumps({"frames": cfg.frames(), "include_hands": cfg.include_hands, "dt": args.dt})
    f.attrs["n_clips"] = len(clip_start)
    f.attrs["n_frames"] = int(ds["feat"].shape[0])
    f.attrs["metadata_csv"] = os.path.basename(meta_path)
    f.close()
    print(f"[precompute] wrote {args.out}: {len(clip_start)} clips, {n_frames} frames, pfd={pfd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
