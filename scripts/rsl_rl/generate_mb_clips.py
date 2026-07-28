#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage 2 driver: ingest Motion Bricks raw motions into a flat clip library (drop-in beside the DP).

Reads the raw ``.npz`` files written by ``scripts/rsl_rl/mb_encode.py`` (Stage 1,
Motion Bricks env), resamples each to the RL control dt + reindexes joints
(``motion_gen.motionbricks.mb_ingest``), and writes them through the SAME ``MultiSkillManager``
output-name layout + ``ClipOutputBuilder`` + ``ClipLibraryWriter`` that ``generate_mm_clips.py`` uses --
so the resulting ``.h5`` is byte-compatible with the DP flat clips (only the motion source differs).

``--rl-vels`` picks which of the raw motion's two command channels becomes the RL-visible ``target_vel``:

* ``raw`` (default) -- the CONTINUOUS commanded velocity. The RL sees a rich velocity distribution and
  learns to track the NEAREST available reference; command and motion differ by the snap error (reported).
* ``snapped`` -- the velocity the motion was actually generated at. Command and motion agree exactly, but
  the RL only ever sees the discrete stride-bank velocities.

Clip conversion is embarrassingly parallel, so it fans across a fork pool. NO IsaacLab. Example::

    PYTHONPATH=source/motion_gen:source/robot_rl \
    env_isaaclab/bin/python scripts/rsl_rl/generate_mb_clips.py \
        --raw-dir mb_raw/flat_lib_v1 \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --out clips/g1_flat_mb_v1.h5 --clip-length 9.0 --device cpu
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.clip_output import ClipOutputBuilder
from motion_gen.gen_parallel import default_workers, unit_pool
from motion_gen.motionbricks import fit_to_length, load_raw_motion, raw_to_world, resample_rows
from motion_gen.multiskill_manager import MultiSkillManager
from motion_gen.world_lift import anchor_mask_from_free

_CONTACT_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Read-only state shared with the forked workers via copy-on-write (set in main() BEFORE the pool).
_SHARED: dict = {}


def _convert(raw, dt: float, T: int, builder, rl_vels: str, dev):
    """One raw motion -> the per-clip arrays the writer wants, plus the snap error for reporting.

    Returns ``(y, dy, contact, target_vel, clip_len, snap_err[T,3], pelvis_z_mean)``. ``target_vel`` is the
    channel selected by ``rl_vels``; a raw motion with no snapped channel (e.g. a free-run capture,
    which never snapped) reports zero error and uses its single command for both.
    """
    Wp, Wq, Wj, tvel_raw = raw_to_world(raw, dt, device=dev)
    if raw.target_vel_snapped is None:
        tvel_snap = tvel_raw
    else:                                    # same clock as tvel_raw, without redoing the pose conversion
        tvel_snap = torch.as_tensor(resample_rows(raw.target_vel_snapped, raw.fps, 1.0 / dt),
                                    dtype=torch.float32, device=dev)
    tvel = tvel_raw if rl_vels == "raw" else tvel_snap
    err = (tvel_raw - tvel_snap).abs()

    Wp, clen = fit_to_length(Wp, T)
    Wq, _ = fit_to_length(Wq, T)
    Wj, _ = fit_to_length(Wj, T)
    tvel, _ = fit_to_length(tvel, T)
    err, _ = fit_to_length(err, T)
    y, dy, contact = builder.build(Wp, Wq, Wj, dt)
    return y, dy, contact, tvel, clen, err[:clen], float(Wp[:clen, 2].mean())


def _worker(i: int):
    """Convert raw clip ``i`` in a forked worker; returns a one-element list (``unit_pool`` flattens)."""
    S = _SHARED
    raw = load_raw_motion(S["raw_paths"][i])
    y, dy, contact, tvel, clen, err, pz = _convert(raw, S["dt"], S["T"], S["builder"], S["rl_vels"], S["dev"])
    return [{"i": i, "y": y, "dy": dy, "contact": contact, "target_vel": tvel, "clip_len": clen,
             "snap_err": err.numpy(), "pelvis_z": pz, "src_frames": raw.num_frames}]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="Directory of Stage-1 raw motion .npz files.")
    p.add_argument("--source", required=True, help="Traj-opt library root -- ONLY for the robot "
                                                   "pos/vel output-name layout (data comes from the raw motions).")
    p.add_argument("--out", required=True, help="Output .h5 clip library path.")
    p.add_argument("--dt", type=float, default=0.02, help="RL control dt (clip sampling period).")
    p.add_argument("--clip-length", type=float, default=9.0, help="Clip length (s); clips pad/truncate to this.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--flat-anchor", default="x,y,yaw",
                   help="Comma-separated FREE root DOFs (of x,y,z,roll,pitch,yaw) for the RL anchor mask.")
    p.add_argument("--rl-vels", default="raw", choices=["raw", "snapped"],
                   help="Which command channel becomes the RL-visible target_vel: 'raw' (default) the "
                        "continuous commanded velocity, or 'snapped' the one the motion was generated at.")
    p.add_argument("--workers", type=int, default=0, help="Parallel conversion workers (0 = all cores - 2).")
    args = p.parse_args()

    # Ingest RawMotion clips only; skip underscore-prefixed sidecars (e.g. mb_encode's _keyframes.npz).
    raw_paths = sorted(g for g in glob.glob(os.path.join(args.raw_dir, "*.npz"))
                       if not os.path.basename(g).startswith("_"))
    if not raw_paths:
        raise SystemExit(f"no raw .npz found under {args.raw_dir}")
    dev = args.device
    manager = MultiSkillManager(args.source, dev, env=None)          # robot output-names + skill_list only
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT_FRAMES, device=dev)
    anchor = int(anchor_mask_from_free([x.strip() for x in args.flat_anchor.split(",") if x.strip()]))
    T = max(1, int(round(args.clip_length / args.dt)))
    N = len(raw_paths)

    meta = {
        "dt": args.dt, "clip_length_s": args.clip_length, "num_steps": int(T),
        "pos_output_names": list(builder.pos_names),                 # builder is authoritative (29-joint layout)
        "vel_output_names": list(builder.vel_names),
        "contact_frames": _CONTACT_FRAMES,
        "ref_frame": "pelvis_link",
        "skill_list": [s.name for s in manager.skills],
    }
    n_workers = default_workers(N, args.workers)
    print(f"[mb_clips] {N} raw clips -> {args.out}  (T={T}, P={len(builder.pos_names)}, "
          f"V={len(builder.vel_names)}, rl_vels={args.rl_vels}, workers={n_workers})")

    _SHARED.update(raw_paths=raw_paths, dt=args.dt, T=T, builder=builder, rl_vels=args.rl_vels, dev=dev)
    recs = unit_pool(_worker, range(N), n_workers)

    writer = ClipLibraryWriter(args.out, N, meta)
    writer.set_attr("generator", "motionbricks_v1")
    writer.set_attr("generator_config", {"rl_vels": args.rl_vels, "raw_dir": os.path.abspath(args.raw_dir)})
    snap_err = []
    for r in sorted(recs, key=lambda d: d["i"]):
        writer.write_batch(r["i"], {
            "y_des": r["y"].unsqueeze(0), "dy_des": r["dy"].unsqueeze(0), "contact": r["contact"].unsqueeze(0),
            "target_vel": r["target_vel"].unsqueeze(0),
            "terrain_id": torch.tensor([-1], dtype=torch.int32),
            "terrain_type": "flat",
            "anchor_mask": torch.tensor([anchor], dtype=torch.uint8),
            "clip_len": torch.tensor([r["clip_len"]], dtype=torch.int32),
        })
        snap_err.append(r["snap_err"])
    writer.close()

    # SNAP ERROR: with --rl-vels raw the RL is commanded a velocity the reference only approximates. Report
    # the gap so the divergence is a known quantity before training, not a surprise in the reward.
    e = np.concatenate(snap_err, axis=0) if snap_err else np.zeros((1, 3), np.float32)
    p50, p95, emax = (np.percentile(e, 50, axis=0), np.percentile(e, 95, axis=0), e.max(axis=0))
    print(f"[mb_clips] snap error |raw - snapped| over {e.shape[0]} real frames "
          f"({'RL sees raw' if args.rl_vels == 'raw' else 'RL sees snapped'}):")
    for j, ax in enumerate(("vx", "vy", "wz")):
        print(f"[mb_clips]   {ax}: p50={p50[j]:.3f} p95={p95[j]:.3f} max={emax[j]:.3f}")

    lib = load_clip_library(args.out)
    y = torch.as_tensor(lib.y_des)
    tv = torch.as_tensor(lib.target_vel)
    vmin, vmax = tv.amin(dim=(0, 1)), tv.amax(dim=(0, 1))
    print(f"[mb_clips] wrote {args.out}: y_des {tuple(y.shape)} finite={bool(torch.isfinite(y).all())} "
          f"dt={lib.meta['dt']} ref_frame={lib.meta['ref_frame']}")
    print(f"[mb_clips]   target_vel span: vx[{vmin[0]:+.2f},{vmax[0]:+.2f}] vy[{vmin[1]:+.2f},{vmax[1]:+.2f}] "
          f"wz[{vmin[2]:+.2f},{vmax[2]:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
