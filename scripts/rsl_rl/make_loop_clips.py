#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Build a clip library that is NOTHING but exact looped replay of optimized strides -- no Motion Bricks.

This is the steady-state half of flat clip generation, isolated. In the full pipeline a flat clip is
``[stand][transient][loop][transient][loop]...``: Motion Bricks in-betweens the TRANSIENTS while every
steady stretch is an exact replay of a traj-opt stride, looped. Steady velocity is then exact by
construction rather than being whatever duration the model happened to choose.

Isolating it matters because it is the floor everything else stands on: if a pure-loop library does not
track its own command to ~1e-3 m/s, no amount of splicing downstream will help, and the failure would
otherwise be invisible under the transients' much larger error. So this driver exists to be MEASURED
(``mb_qc.py``), not to be trained on.

One clip per bank stride by default, each looped to ``--clip-length``. Because the replay is the
stride's own frames, ``target_vel`` is written from the stride's MEASURED per-period twist (the SE(2)
log, see :mod:`motion_gen.motionbricks.stride_bank`) -- so command and motion agree by construction and
any residual `mb_qc` reports is a real defect rather than a bookkeeping gap.

Runs in ``env_isaaclab`` (needs ``motion_gen`` + torch, no simulator)::

    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" \
    env_isaaclab/bin/python scripts/rsl_rl/make_loop_clips.py \
        --stride-bank mb_raw/gate_phase0/stride_bank_v3.npz \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --out mb_raw/gate_phase0/flat_loops.h5 --clip-length 20.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter
from motion_gen.clip_output import ClipOutputBuilder
from motion_gen.library_metadata import read_library_metadata
from motion_gen.motionbricks.stride_bank import load_stride_bank
from motion_gen.terrain.box_clip import loop_frames
from motion_gen.world_lift import anchor_mask_from_free

_CONTACT_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _split(qpos: np.ndarray, dev: str):
    """``qpos[F,36]`` -> torch ``(root_pos[F,3], root_quat[F,4] wxyz, joints[F,29])``."""
    q = torch.as_tensor(np.asarray(qpos, np.float32), dtype=torch.float32, device=dev)
    return q[:, 0:3], q[:, 3:7], q[:, 7:36]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stride-bank", required=True, help="Stride bank .npz (export_stride_bank.py).")
    p.add_argument("--source", required=True, help="Traj-opt library root (robot output-name layout).")
    p.add_argument("--out", required=True, help="Output clip library .h5.")
    p.add_argument("--clip-length", type=float, default=20.0, help="Seconds per clip.")
    p.add_argument("--dt", type=float, default=0.02, help="Clip control period (s); must match the bank's.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="Only the first N strides (a quick check).")
    p.add_argument("--start-phase", default="zero", choices=["zero", "random"],
                   help="Gait phase each clip enters its stride at. 'random' exercises the entry-phase "
                        "path the real pipeline uses when a transient hands off mid-cycle.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flat-anchor", default="x,y,yaw",
                   help="Free axes of the per-clip anchor mask (flat clips are ego-anchored).")
    args = p.parse_args()

    bank = load_stride_bank(args.stride_bank)
    if abs(bank.dt - args.dt) > 1e-9:
        # A silent rate mismatch would rescale every velocity in the library, so refuse rather than warn.
        raise SystemExit(f"bank dt {bank.dt} != --dt {args.dt}; the replay is the bank's own frames, so "
                         f"the clip control period must be the bank's sampling period")
    idx = list(range(len(bank)))[:args.limit] if args.limit else list(range(len(bank)))
    if not idx:
        raise SystemExit(f"no strides in {args.stride_bank}")

    dev = args.device
    lib_meta = read_library_metadata(args.source)
    builder = ClipOutputBuilder(lib_meta.pos_output_names, lib_meta.vel_output_names,
                                contact_frames=_CONTACT_FRAMES, device=dev)
    t_frames = max(1, int(round(args.clip_length / args.dt)))
    anchor = int(anchor_mask_from_free([x.strip() for x in args.flat_anchor.split(",") if x.strip()]))
    rng = np.random.default_rng(args.seed)

    meta = {
        "dt": args.dt, "clip_length_s": args.clip_length, "num_steps": int(t_frames),
        "pos_output_names": list(builder.pos_names),
        "vel_output_names": list(builder.vel_names),
        "contact_frames": _CONTACT_FRAMES,
        "ref_frame": "pelvis_link",
        "skill_list": lib_meta.skill_list,
    }
    print(f"[loop_clips] {len(idx)} strides -> {args.out}  (T={t_frames}, P={len(builder.pos_names)}, "
          f"V={len(builder.vel_names)}, start_phase={args.start_phase})")

    writer = ClipLibraryWriter(args.out, len(idx), meta)
    writer.set_attr("generator", "loop_replay_v1")
    writer.set_attr("generator_config", {"stride_bank": os.path.abspath(args.stride_bank),
                                         "start_phase": args.start_phase, "seed": args.seed})
    for out_i, i in enumerate(idx):
        period = int(bank.period_frames[i])
        start = int(rng.integers(0, period)) if args.start_phase == "random" else 0
        rp, rq, jt = _split(bank.strides[i], dev)
        lrp, lrq, ljt, _ = loop_frames(rp, rq, jt, None, t_frames, start=start)
        y, dy, contact = builder.build(lrp, lrq, ljt, args.dt)
        # The stride's OWN measured twist, so command and motion agree by construction; anything mb_qc
        # then reports as tracking error is a real defect, not a bookkeeping gap.
        tvel = torch.zeros(t_frames, 3, dtype=torch.float32, device=dev)
        tvel[:] = torch.as_tensor(bank.achieved_vel[i], dtype=torch.float32, device=dev)
        writer.write_batch(out_i, {
            "y_des": y.unsqueeze(0), "dy_des": dy.unsqueeze(0), "contact": contact.unsqueeze(0),
            "target_vel": tvel.unsqueeze(0),
            "terrain_id": torch.tensor([-1], dtype=torch.int32),
            "terrain_type": "flat",
            "anchor_mask": torch.tensor([anchor], dtype=torch.uint8),
            "clip_len": torch.tensor([t_frames], dtype=torch.int32),
            "stride_idx": torch.full((1, t_frames), i, dtype=torch.int32),
        })
    writer.close()

    v = bank.achieved_vel[idx]
    print(f"[loop_clips] wrote {args.out}: {len(idx)} clips")
    print(f"[loop_clips]   velocity coverage: vx[{v[:,0].min():+.2f},{v[:,0].max():+.2f}] "
          f"vy[{v[:,1].min():+.2f},{v[:,1].max():+.2f}] wz[{v[:,2].min():+.2f},{v[:,2].max():+.2f}]")
    print(f"[loop_clips]   periods {bank.period_s[idx].min():.2f}-{bank.period_s[idx].max():.2f} s "
          f"({args.clip_length / bank.period_s[idx].max():.1f}-{args.clip_length / bank.period_s[idx].min():.1f} "
          f"cycles per clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
