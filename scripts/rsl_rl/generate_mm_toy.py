#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vanilla motion matching toy clip (bones-seed only; no skills / traj-opt / handoff).

Drives a scripted ``(vx, vy, omega)`` command through vanilla motion matching over the
bones-seed database and writes a standard ``.h5`` clip.  Runs with NO IsaacLab.

    python scripts/rsl_rl/generate_mm_toy.py \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --bones-cache clips/bones_cache_locomotion.h5 \
        --bones-movement walking,jogging --bones-body standing \
        --out clips/g1_mm_toy.h5 --device cpu
"""

from __future__ import annotations

import argparse

import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.motion_matching.bones_ingest import load_cache
from motion_gen.motion_matching.clip_output import ClipOutputBuilder
from motion_gen.motion_matching.vanilla import VanillaConfig, VanillaMatcher
from motion_gen.multiskill_manager import MultiSkillManager

_CONTACT = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _toy_command(t: float):
    """Scripted (vx, vy, omega): stand -> walk -> turn-while-walking -> walk -> stop."""
    if t < 1.0:
        return (0.0, 0.0, 0.0)
    if t < 4.0:
        return (1.2, 0.0, 0.0)
    if t < 6.0:
        return (1.0, 0.0, 0.5)     # forward + turn left
    if t < 8.0:
        return (1.2, 0.0, 0.0)
    return (0.0, 0.0, 0.0)


def _csv_set(s):
    return None if not s else [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt library (only for the output channel layout).")
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--clip-length", type=float, default=9.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bones-package", default="Locomotion")
    p.add_argument("--bones-style", default=None)
    p.add_argument("--bones-movement", default=None)
    p.add_argument("--bones-body", default=None)
    p.add_argument("--bones-limit", type=int, default=None)
    p.add_argument("--search-interval", type=float, default=0.1)
    args = p.parse_args()

    dev = args.device
    cfg = VanillaConfig(dt=args.dt, search_interval_s=args.search_interval)
    filters = {"package": args.bones_package}
    if args.bones_movement:
        filters["content_type_of_movement"] = _csv_set(args.bones_movement)
    if args.bones_body:
        filters["content_body_position"] = _csv_set(args.bones_body)

    print(f"[toy] loading bones cache (filters={filters}, style={args.bones_style}) ...")
    sources, _ = load_cache(args.bones_cache, filters=filters, styles=_csv_set(args.bones_style),
                            limit=args.bones_limit, device=dev)
    print(f"[toy] {len(sources)} clips; building vanilla matcher (KD-tree over {sum(s.n for s in sources)} frames) ...")
    matcher = VanillaMatcher(sources, cfg, device=dev)

    T = int(round(args.clip_length / args.dt))
    print(f"[toy] matching {T} steps against the toy command ...")
    root_pos, root_quat, joints = matcher.generate(_toy_command, T)

    manager = MultiSkillManager(args.source, dev, env=None)
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT, device=dev)
    y, dy, contact = builder.build(root_pos, root_quat, joints, args.dt)
    tvel = torch.tensor([[*_toy_command(i * args.dt)] for i in range(T)], dtype=torch.float32, device=dev)
    batch = {
        "y_des": y.unsqueeze(0), "dy_des": dy.unsqueeze(0), "contact": contact.unsqueeze(0),
        "skill_id": torch.zeros(1, T, dtype=torch.uint8, device=dev),
        "stride_idx": torch.full((1, T), -1, dtype=torch.int32, device=dev),
        "target_vel": tvel.unsqueeze(0), "terrain_id": torch.tensor([-1], dtype=torch.int32, device=dev),
        "terrain_type": "flat",
    }
    meta = {
        "dt": args.dt, "clip_length_s": args.clip_length, "num_steps": int(T),
        "pos_output_names": list(manager.pos_output_names),
        "vel_output_names": list(manager.vel_output_names),
        "contact_frames": _CONTACT, "ref_frame": "pelvis_link",
        "skill_list": [s.name for s in manager.skills],
    }
    writer = ClipLibraryWriter(args.out, 1, meta)
    writer.set_attr("generator", "vanilla_motion_matching")
    writer.write_batch(0, batch)
    writer.close()
    lib = load_clip_library(args.out)
    yy = torch.as_tensor(lib.y_des)
    print(f"[toy] wrote {args.out}: y_des {tuple(yy.shape)} finite={bool(torch.isfinite(yy).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
