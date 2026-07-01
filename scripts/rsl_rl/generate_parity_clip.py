#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Bake a single deterministic parity clip for IsaacLab <-> MuJoCo comparison.

Authors one fixed velocity profile -- ``stand -> ramp to a walk speed -> hold ->
ramp to a run speed -> hold -> ramp down to stand`` -- and stitches the matching
strides into a single dense clip stored as a one-clip HDF5 library (the same
format ``generate_clips.py`` produces, so the runtime tracks it open-loop).

The exact same clip can then be played in IsaacLab (clip-only, FM-free, via
``G1-teacher-clf-parity-play``) and in the MuJoCo sim2sim runner (``--clip``),
giving a deterministic reference for checking sim2sim parity.

Example::

    python scripts/rsl_rl/generate_parity_clip.py \\
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \\
        --out clips/g1_parity_v1.h5 --dt 0.02

The ramps run at ``--accel`` m/s^2 (matching the training-time clip ramp), so the
per-segment hold durations (``--walk_s`` / ``--run_s``) are the *flat* hold
lengths and the ramp time between segments is ``delta_v / accel``.  ``--source``
must be the same stride library the tracker/FM trained on and must cover the run
speed.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
    MultiSkillManager,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.scripted_velocity_sampler import (
    ScriptedVelocitySampler,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_clip_generator import (
    OfflineClipGenerator,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_io import (
    ClipLibraryWriter,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_fingerprint import (
    fingerprint_clip_config,
)


@dataclass
class _Bucket:
    """Minimal VelocityBucketCfg stand-in (same attribute names)."""

    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    heading: tuple[float, float] | None = None


def _library_derived_buckets(manager: MultiSkillManager) -> dict[str, _Bucket]:
    """Per-skill (min, max) velocity buckets from the library's own coverage.

    Mirrors ``generate_clips.py`` so the offline skill selection
    (``bucket_for_velocity``) picks stand / walk / run from the commanded speed
    exactly as the random library generator does.
    """
    buckets: dict[str, _Bucket] = {}
    cond = manager._global_conditioning  # [T, >=3] (vx, vy, wz, ...)
    for s in manager.skills:
        off = manager._skill_traj_offset[manager.skill_name_to_idx[s.name]]
        block = cond[off:off + s.num_trajectories, :3]
        lo = block.min(dim=0).values.tolist()
        hi = block.max(dim=0).values.tolist()
        buckets[s.name] = _Bucket(
            lin_vel_x=(lo[0], hi[0]), lin_vel_y=(lo[1], hi[1]), ang_vel_z=(lo[2], hi[2]),
        )
    return buckets


def _build_keyframes(args: argparse.Namespace) -> tuple[list[tuple[float, tuple[float, float, float]]], float]:
    """Build the forward-only ``(t, (vx, vy, wz))`` schedule from segment knobs.

    Ramps run at ``args.accel`` (m/s^2); holds are the flat segment lengths.

    Returns:
        ``(keyframes, clip_length_s)``.
    """
    a = float(args.accel)
    kf: list[tuple[float, tuple[float, float, float]]] = [(0.0, (0.0, 0.0, 0.0))]
    t = 0.0

    def add(dt: float, vx: float) -> None:
        nonlocal t
        t += float(dt)
        kf.append((t, (float(vx), 0.0, 0.0)))

    add(args.stand_s, 0.0)                               # initial stand
    add(args.walk_speed / a, args.walk_speed)            # ramp up to walk
    add(args.walk_s, args.walk_speed)                    # hold walk
    add((args.run_speed - args.walk_speed) / a, args.run_speed)  # ramp up to run
    add(args.run_s, args.run_speed)                      # hold run
    add(args.run_speed / a, 0.0)                         # ramp down to stand
    if args.end_stand_s > 0:
        add(args.end_stand_s, 0.0)                       # final stand
    return kf, t


def main() -> int:
    """Parse args, build the scripted pipeline, and write the one-clip library."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Source stride library root (per-skill subfolders).")
    p.add_argument("--out", default="clips/g1_parity_v1.h5", help="Output .h5 clip library path.")
    p.add_argument("--dt", type=float, default=0.02, help="Control-step dt (s); must equal the env step_dt.")
    p.add_argument("--accel", type=float, default=2.0, help="Ramp acceleration between segments (m/s^2).")
    p.add_argument("--stand_s", type=float, default=3.0, help="Initial stand duration (s).")
    p.add_argument("--walk_speed", type=float, default=1.0, help="Walk forward speed (m/s).")
    p.add_argument("--walk_s", type=float, default=5.0, help="Walk hold duration (s).")
    p.add_argument("--run_speed", type=float, default=3.0, help="Run forward speed (m/s).")
    p.add_argument("--run_s", type=float, default=3.0, help="Run hold duration (s).")
    p.add_argument("--end_stand_s", type=float, default=2.0, help="Final stand duration after ramp-down (s).")
    p.add_argument("--seed", type=int, default=0, help="Seed (for the fingerprint; profile is deterministic).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hf-repo", default=None, help="Optional HuggingFace repo id for the source library.")
    p.add_argument("--no-ground-projection", action="store_true",
                   help="Disable pinning stance landings to the ground plane (allows z drift).")
    p.add_argument("--ground-height", type=float, default=0.0, help="Ground plane height for projection (m).")
    p.add_argument("--seam-blend-steps", type=int, default=8,
                   help="Cross-fade the new stride's first N steps with the old stride at each seam.")
    args = p.parse_args()

    device = torch.device(args.device)
    manager = MultiSkillManager(args.source, device, env=None, hf_repo=args.hf_repo)
    skill_list = [s.name for s in manager.skills]
    buckets = _library_derived_buckets(manager)
    # Uniform skill probs: unused for sampling (the scripted profile drives the
    # command), but the base sampler requires a positive-sum vector.
    skill_probs = [1.0 / len(skill_list)] * len(skill_list)

    keyframes, clip_length_s = _build_keyframes(args)

    sampler = ScriptedVelocitySampler(
        buckets, skill_list, skill_probs, keyframes=keyframes,
        command_dt=args.dt, max_acc=args.accel, device=device, seed=args.seed,
    )
    gen = OfflineClipGenerator(
        manager, sampler, dt=args.dt, clip_length_s=clip_length_s,
        project_to_ground=not args.no_ground_projection, ground_height=args.ground_height,
        seam_blend_steps=args.seam_blend_steps, device=device,
    )

    gen_config = {
        "num_clips": 1,
        "clip_length_s": clip_length_s,
        "dt": args.dt,
        "profile": "parity_stand_walk_run_stand",
        "accel": args.accel,
        "keyframes": [[t, list(v)] for t, v in keyframes],
        "skill_list": skill_list,
        "project_to_ground": not args.no_ground_projection,
        "ground_height": args.ground_height,
        "seam_blend_steps": args.seam_blend_steps,
        "velocity_buckets": {
            name: {"lin_vel_x": list(b.lin_vel_x), "lin_vel_y": list(b.lin_vel_y),
                   "ang_vel_z": list(b.ang_vel_z), "heading": b.heading}
            for name, b in buckets.items()
        },
    }
    clip_fp = fingerprint_clip_config(manager.library_fingerprint, gen_config, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    batch = gen.generate_batch(1)
    writer = ClipLibraryWriter(args.out, 1, batch["meta"])
    writer.set_attr("source_library_fingerprint", manager.library_fingerprint)
    writer.set_attr("clip_fingerprint", clip_fp)
    writer.set_attr("generator_config", gen_config)
    writer.write_batch(0, batch)
    writer.close()

    seg = [f"stand {args.stand_s}s", f"walk {args.walk_speed} m/s {args.walk_s}s",
           f"run {args.run_speed} m/s {args.run_s}s", f"stand {args.end_stand_s}s"]
    print(
        f"[generate_parity_clip] source={args.source} fp={manager.library_fingerprint['short']} "
        f"clip_fp={clip_fp['short']} -> {args.out}\n"
        f"  1 clip x {batch['meta']['num_steps']} steps @ dt={args.dt}s "
        f"({clip_length_s:.2f}s): " + " -> ".join(seg) + "\n"
        f"  (ramps @ {args.accel} m/s^2)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
