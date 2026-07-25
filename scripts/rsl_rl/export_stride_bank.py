#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage 0 (env_isaaclab): export the clean STEADY-tier strides as a numpy "stride bank".

Phase-2 Motion Bricks generation drives the model with *our* optimized traj-opt strides as
velocity-indexed keyframes. Those strides live in the clean steady-state library
(``trajectories/skills/library_v2``) and are ingested via
:func:`motion_gen.motion_matching.clean_ingest.ingest_clean_library` -- which needs ``motion_gen``
(env_isaaclab), NOT the Motion Bricks conda env. So, mirroring the Stage-1/Stage-2 ``RawMotion``
contract, this stage writes a self-contained, numpy-only ``.npz`` that the Motion Bricks-side
``mb_encode.py`` loads without importing ``motion_gen``.

Each stride becomes a MuJoCo ``qpos [Fi, 36] = [root_pos(3), root_quat_wxyz(4), joints(29)]`` (the
joints already in ``G1_JOINT_NAMES`` order == Motion Bricks' ``g1.xml`` order), tagged with its
steady command ``(vx, vy, wz)`` and periodic/episodic wrap. Strides have different cycle lengths, so
the bank stores one flat ``qpos_all`` plus per-stride offsets (pickle-free). Example::

    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" \
    env_isaaclab/bin/python scripts/rsl_rl/export_stride_bank.py \
        --library trajectories/skills/library_v2 --hf-repo zolkin/robot_rl \
        --out mb_raw/stride_bank_v2.npz --dt 0.02
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from motion_gen.motion_matching.clean_ingest import ingest_clean_library


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default="trajectories/skills/library_v2",
                   help="Clean steady-stride library folder (local, or relative to --hf-repo root).")
    p.add_argument("--hf-repo", default="zolkin/robot_rl", help="HuggingFace repo id (blank to use a local path).")
    p.add_argument("--out", required=True, help="Output stride-bank .npz path.")
    p.add_argument("--dt", type=float, default=0.02, help="Stride sampling period (s); must match the RL control dt.")
    p.add_argument("--skills", default=None, help="Optional comma-separated skill whitelist (manifest 'skill' column).")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on the number of strides.")
    args = p.parse_args()

    skills = [s.strip() for s in args.skills.split(",")] if args.skills else None
    sources = ingest_clean_library(
        args.library, hf_repo=(args.hf_repo or None), dt=args.dt, device="cpu",
        skills=skills, limit=args.limit,
    )
    if not sources:
        raise SystemExit(f"no clean strides ingested from {args.library}")

    qpos_chunks: list[np.ndarray] = []
    offsets = [0]
    steady_vel = np.zeros((len(sources), 3), dtype=np.float32)
    periodic = np.zeros(len(sources), dtype=np.uint8)
    names: list[str] = []
    for i, s in enumerate(sources):
        # MuJoCo qpos layout MB's converter expects: [root_pos(3), root_quat_wxyz(4), joints(29)].
        qpos = torch.cat([s.root_pos, s.root_quat, s.joint_pos], dim=-1).cpu().numpy().astype(np.float32)
        if qpos.shape[1] != 36:
            raise ValueError(f"stride {s.name}: expected qpos [Fi,36], got {qpos.shape}")
        qpos_chunks.append(qpos)
        offsets.append(offsets[-1] + qpos.shape[0])
        steady_vel[i] = (s.steady_vel.cpu().numpy() if s.steady_vel is not None else 0.0)
        periodic[i] = 1 if s.wrap == "periodic" else 0
        names.append(s.name)

    qpos_all = np.concatenate(qpos_chunks, axis=0)
    meta = {"dt": args.dt, "library": args.library, "hf_repo": args.hf_repo,
            "n_strides": len(sources), "joint_order": "G1_JOINT_NAMES", "names": names}
    np.savez_compressed(
        args.out,
        qpos_all=qpos_all,
        offsets=np.asarray(offsets, dtype=np.int64),
        steady_vel=steady_vel,
        periodic=periodic,
        meta=np.array(json.dumps(meta)),
    )
    vmin, vmax = steady_vel.min(axis=0), steady_vel.max(axis=0)
    print(f"[stride_bank] wrote {args.out}: {len(sources)} strides, {qpos_all.shape[0]} total frames, dt={args.dt}")
    print(f"[stride_bank]   steady_vel range: vx[{vmin[0]:.2f},{vmax[0]:.2f}] "
          f"vy[{vmin[1]:.2f},{vmax[1]:.2f}] wz[{vmin[2]:.2f},{vmax[2]:.2f}]; "
          f"periodic={int(periodic.sum())}/{len(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
