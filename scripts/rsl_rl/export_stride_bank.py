#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage 0 (env_isaaclab): export the clean STEADY-tier strides as a numpy "stride bank".

Phase-2 Motion Bricks generation drives the model with *our* optimized traj-opt strides as
velocity-indexed keyframes. Those strides live in the clean steady-state library
(``trajectories/skills/library_v2``) and are ingested via
:func:`motion_gen.ingest.clean_ingest.ingest_clean_library` -- which needs ``motion_gen``
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

from motion_gen.ingest.clean_ingest import ingest_clean_library
from motion_gen.motionbricks.stride_bank import closure_residual, load_stride_bank, save_stride_bank


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
    contact_chunks: list = []
    steady_vel = np.zeros((len(sources), 3), dtype=np.float32)
    periodic = np.zeros(len(sources), dtype=np.uint8)
    names: list[str] = []
    for i, s in enumerate(sources):
        # MuJoCo qpos layout MB's converter expects: [root_pos(3), root_quat_wxyz(4), joints(29)].
        qpos = torch.cat([s.root_pos, s.root_quat, s.joint_pos], dim=-1).cpu().numpy().astype(np.float32)
        if qpos.shape[1] != 36:
            raise ValueError(f"stride {s.name}: expected qpos [Fi,36], got {qpos.shape}")
        qpos_chunks.append(qpos)
        contact_chunks.append(None if s.contact is None
                              else s.contact.cpu().numpy().astype(np.uint8))
        steady_vel[i] = (s.steady_vel.cpu().numpy() if s.steady_vel is not None else 0.0)
        periodic[i] = 1 if s.wrap == "periodic" else 0
        names.append(s.name)

    meta = {"dt": args.dt, "library": args.library, "hf_repo": args.hf_repo,
            "n_strides": len(sources), "joint_order": "G1_JOINT_NAMES", "names": names}
    n_ct = sum(c is not None for c in contact_chunks)
    if n_ct and n_ct != len(contact_chunks):
        raise SystemExit(f"[stride_bank] only {n_ct}/{len(contact_chunks)} strides carry a solver "
                         f"contact schedule -- refusing to write a bank where it is present for some "
                         f"strides and silently absent for others")
    save_stride_bank(args.out, qpos_chunks, steady_vel, periodic, meta,
                     contact=(contact_chunks if n_ct else None))

    n_frames = sum(c.shape[0] for c in qpos_chunks)
    vmin, vmax = steady_vel.min(axis=0), steady_vel.max(axis=0)
    print(f"[stride_bank] wrote {args.out}: {len(sources)} strides, {n_frames} total frames, dt={args.dt}")
    print(f"[stride_bank]   solver contact schedule: "
          f"{'carried' if n_ct else 'ABSENT (source npz had no contact_flags)'}")
    print(f"[stride_bank]   steady_vel range: vx[{vmin[0]:.2f},{vmax[0]:.2f}] "
          f"vy[{vmin[1]:.2f},{vmax[1]:.2f}] wz[{vmin[2]:.2f},{vmax[2]:.2f}]; "
          f"periodic={int(periodic.sum())}/{len(sources)}")

    # Two properties the exact looped replay depends on, checked HERE so a bad stride is visible when
    # the bank is built rather than as an unexplained artefact in a trained policy.
    bank = load_stride_bank(args.out)
    gaps = np.array([closure_residual(s) for s in bank.strides])
    err = np.abs(bank.achieved_vel - bank.steady_vel)
    bad = int((gaps[:, 0] > 1e-4).sum())
    print(f"[stride_bank]   period closure (frame F-1 vs 0): joint gap max={gaps[:, 0].max():.2e} rad, "
          f"height max={gaps[:, 1].max():.2e} m  -> {bad} stride(s) do NOT close")
    print(f"[stride_bank]   achieved vs commanded velocity: vx p50={np.median(err[:, 0]):.4f} "
          f"p95={np.percentile(err[:, 0], 95):.4f} max={err[:, 0].max():.4f} m/s; "
          f"period {bank.period_s.min():.2f}-{bank.period_s.max():.2f} s")
    if bad:
        worst = int(np.argmax(gaps[:, 0]))
        print(f"[stride_bank]   WARNING: looping an unclosed stride steps at every period seam; "
              f"worst is {bank.names[worst]} ({gaps[worst, 0]:.3e} rad)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
