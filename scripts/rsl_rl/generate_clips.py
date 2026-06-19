#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate a dense motion-clip library by stitching per-stride references.

Samples a velocity profile per clip (the same rules the live sim uses) and
stitches the matching strides into fixed long world-anchored clips, then
writes them to a single HDF5 file the runtime tracks open-loop.  Flat
ground only (v1): pure velocity nearest-neighbour selection, no terrain.

Example::

    python scripts/rsl_rl/generate_clips.py \\
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \\
        --out clips/g1_flat_v1.h5 \\
        --num-clips 10000 --clip-length 20.0 --dt 0.02 --seed 0

Velocity buckets and skill mix default to the source library's own per-skill
velocity coverage and trajectory-count proportions; override with a JSON
``--config`` containing ``velocity_buckets`` and/or ``skill_probs``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
    MultiSkillManager,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_velocity_sampler import (
    OfflineVelocitySampler,
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
    """Per-skill (min, max) velocity buckets from the library's own coverage."""
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


def _default_skill_probs(manager: MultiSkillManager) -> list[float]:
    """Trajectory-count-proportional skill probabilities, in manager order."""
    counts = [float(s.num_trajectories) for s in manager.skills]
    total = sum(counts)
    return [c / total for c in counts]


def main() -> int:
    """Parse args, build the pipeline, generate, and write the clip library."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Source stride library root (per-skill subfolders).")
    p.add_argument("--out", required=True, help="Output .h5 clip library path.")
    p.add_argument("--num-clips", type=int, default=10000)
    p.add_argument("--clip-length", type=float, default=20.0, help="Clip length (s).")
    p.add_argument("--dt", type=float, default=0.02, help="Control-step dt (s).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512, help="Clips generated per rollout batch.")
    p.add_argument("--resample-min", type=float, default=2.0, help="Min seconds between velocity resamples.")
    p.add_argument("--resample-max", type=float, default=5.0, help="Max seconds between velocity resamples.")
    p.add_argument("--max-acc", type=float, default=2.0, help="Velocity ramp max acceleration.")
    p.add_argument("--transition-prob", type=float, default=None,
                   help="Per-resample probability of switching to a DIFFERENT skill mid-clip. "
                        "0.0 = lock each clip to its start skill -- walk/run/stand still vary across "
                        "the library and velocity still resamples within that one skill, but there are "
                        "no in-clip skill transitions. None (default) = draw a skill i.i.d. at every "
                        "velocity resample (frequent transitions).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hf-repo", default=None, help="Optional HuggingFace repo id for the source library.")
    p.add_argument("--config", default=None, help="JSON file overriding velocity_buckets / skill_probs.")
    p.add_argument("--exclude-skills", default="", help="Comma-separated skills to never generate (e.g. walk_backwards).")
    p.add_argument("--forward-only", action="store_true",
                   help="Zero lateral (vy) and yaw (wz) velocity in every bucket so all clips go "
                        "straight (no strafing / turning). Combine with --exclude-skills to isolate "
                        "a single gait, e.g. --exclude-skills running,standing,walk_backwards for "
                        "a trivial forward-walk-only validation library.")
    p.add_argument("--no-ground-projection", action="store_true",
                   help="Disable pinning stance landings to the ground plane (allows z drift).")
    p.add_argument("--ground-height", type=float, default=0.0, help="Ground plane height for projection (m).")
    p.add_argument("--seam-blend-steps", type=int, default=8,
                   help="Cross-fade the new stride's first N steps with the old stride at each "
                        "seam to smooth transitions (0 disables).")
    args = p.parse_args()

    device = torch.device(args.device)
    manager = MultiSkillManager(args.source, device, env=None, hf_repo=args.hf_repo)
    skill_list = [s.name for s in manager.skills]

    # Buckets + skill mix: library-derived defaults, optional JSON override.
    buckets = _library_derived_buckets(manager)
    skill_probs = _default_skill_probs(manager)
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        for name, b in cfg.get("velocity_buckets", {}).items():
            buckets[name] = _Bucket(**b)
        if "skill_probs" in cfg:
            sp = cfg["skill_probs"]
            skill_probs = [float(sp[name]) for name in skill_list] if isinstance(sp, dict) else list(sp)

    # Exclude skills: zero their sampling probability (renormalised) and mark
    # them ineligible for stride selection in the generator.
    exclude_skills = [s for s in args.exclude_skills.split(",") if s]
    bad = [s for s in exclude_skills if s not in skill_list]
    if bad:
        raise SystemExit(f"--exclude-skills not in library {skill_list}: {bad}")
    if exclude_skills:
        skill_probs = [0.0 if name in exclude_skills else p for name, p in zip(skill_list, skill_probs)]
        if sum(skill_probs) <= 0:
            raise SystemExit("--exclude-skills removed every skill; nothing left to sample.")

    # Forward-only: clamp lateral (vy) and yaw (wz) to zero in every bucket so the
    # sampler only ever commands straight-line motion. Captured in gen_config below
    # (built from `buckets`) so the fingerprint reflects it.
    if args.forward_only:
        buckets = {
            name: _Bucket(lin_vel_x=b.lin_vel_x, lin_vel_y=(0.0, 0.0),
                          ang_vel_z=(0.0, 0.0), heading=None)
            for name, b in buckets.items()
        }

    sampler = OfflineVelocitySampler(
        buckets, skill_list, skill_probs,
        resampling_time_range=(args.resample_min, args.resample_max),
        max_acc=args.max_acc, command_dt=args.dt,
        skill_transition_prob=args.transition_prob, device=device, seed=args.seed,
    )
    gen = OfflineClipGenerator(
        manager, sampler, dt=args.dt, clip_length_s=args.clip_length,
        exclude_skills=exclude_skills, project_to_ground=not args.no_ground_projection,
        ground_height=args.ground_height, seam_blend_steps=args.seam_blend_steps, device=device,
    )

    # Generator config that fully determines the clips (for the fingerprint).
    gen_config = {
        "num_clips": args.num_clips,
        "clip_length_s": args.clip_length,
        "dt": args.dt,
        "resampling_time_range": [args.resample_min, args.resample_max],
        "max_acc": args.max_acc,
        "skill_transition_prob": args.transition_prob,
        "skill_list": skill_list,
        "skill_probs": skill_probs,
        "exclude_skills": exclude_skills,
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

    # Probe one tiny batch for layout metadata (output names, T), then stream.
    probe = gen.generate_batch(1)
    writer = ClipLibraryWriter(args.out, args.num_clips, probe["meta"])
    writer.set_attr("source_library_fingerprint", manager.library_fingerprint)
    writer.set_attr("clip_fingerprint", clip_fp)
    writer.set_attr("generator_config", gen_config)
    writer.write_batch(0, probe)  # reuse the probe clip as clip 0

    print(
        f"[generate_clips] source={args.source} fp={manager.library_fingerprint['short']} "
        f"clip_fp={clip_fp['short']} -> {args.out}\n"
        f"  {args.num_clips} clips x {probe['meta']['num_steps']} steps @ dt={args.dt}s "
        f"(P={len(probe['meta']['pos_output_names'])}, V={len(probe['meta']['vel_output_names'])})\n"
        f"  skill_probs={dict(zip(skill_list, [round(x, 3) for x in skill_probs]))}"
    )

    written = 1
    while written < args.num_clips:
        n = min(args.batch_size, args.num_clips - written)
        batch = gen.generate_batch(n)
        writer.write_batch(written, batch)
        written += n
        print(f"  generated {written}/{args.num_clips}", end="\r", flush=True)
    writer.close()
    print(f"\n[generate_clips] wrote {written} clips to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
