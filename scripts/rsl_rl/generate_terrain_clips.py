#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate a terrain-aware motion-clip library (flat / terrain / transition).

Given a TOTAL number of terrains, the requested fractions split them into
``flat`` (no terrain), ``terrain`` (terrain-only, spawn at the base and climb),
and ``transition`` (flat lead-in with standing + walking, then approach and
climb -- no running).  Each non-flat terrain has UNIFORM dimensions sampled
from the library's own discrete stair grid and hosts ``--clips-per-terrain``
clips at sampled velocities, so the total is ``num_terrains * clips_per_terrain``.

Stair geometry reuses the terrain "block" system, and each clip re-anchors its
stance foot onto the tread it rests on (the same snapping the online sim uses),
so the reference feet land on the steps.  A terrain manifest describing every
terrain is embedded in the ``.h5`` for viser / plotting.

Run under IsaacLab's launcher (imports pull in omni), e.g.::

    ./isaaclab.sh -p scripts/rsl_rl/generate_terrain_clips.py \\
        --source hf/trajectories/retargeted/<merged_with_stairs> \\
        --out clips/g1_terrain_v1.h5 \\
        --num-terrains 64 --frac-flat 0.34 --frac-terrain 0.33 --frac-transition 0.33 \\
        --clips-per-terrain 64 --clip-length 12.0 --dt 0.02 --device cpu --seed 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
    orchestrate,
    sample_terrain_specs,
    split_counts,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.config.stair_dim_scan import (
    scan_stair_step_dims,
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


def _probs(skill_list: list[str], weights: dict[str, float]) -> list[float]:
    """Skill-probability vector in manager order (unlisted skills -> 0)."""
    vec = [float(weights.get(name, 0.0)) for name in skill_list]
    total = sum(vec)
    if total <= 0:
        raise SystemExit(f"skill weights {weights} zero out every skill in {skill_list}")
    return [v / total for v in vec]


def _make_sampler(buckets, skill_list, probs, dt, seed):
    """OfflineVelocitySampler sharing the common buckets (differ only in probs)."""
    return OfflineVelocitySampler(
        buckets, skill_list, probs, resampling_time_range=(1.5, 4.0),
        max_acc=2.0, command_dt=dt, device="cpu", seed=seed,
    )


def main() -> int:
    """Parse args, build the pipeline, orchestrate, and write the library."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Source stride library root (per-skill subfolders, incl. a stair skill).")
    p.add_argument("--out", required=True, help="Output .h5 clip library path.")
    p.add_argument("--num-terrains", type=int, default=64, help="Total terrains to allocate across the three types.")
    p.add_argument("--frac-flat", type=float, default=0.34)
    p.add_argument("--frac-terrain", type=float, default=0.33)
    p.add_argument("--frac-transition", type=float, default=0.33)
    p.add_argument("--clips-per-terrain", type=int, default=64, help="Clips per terrain (and per flat terrain slot).")
    p.add_argument("--clip-length", type=float, default=12.0, help="Clip length (s).")
    p.add_argument("--dt", type=float, default=0.02, help="Control-step dt (s).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stair-skill", default=None, help="Stair skill subfolder name for the step-dim scan (default: the auto-detected terrain-aware skill).")
    p.add_argument("--stair-width-max", type=float, default=3.0, help="Upper bound (m) when the stair library declares terrain_width as an open range (e.g. {min: 1.0}); tread widths are sampled per terrain in [range_min, this].")
    p.add_argument("--num-steps-min", type=int, default=4)
    p.add_argument("--num-steps-max", type=int, default=8)
    p.add_argument("--lead-in-min", type=float, default=1.5, help="[transition] flat lead-in (start platform) FLOOR (m).")
    p.add_argument("--lead-in-max", type=float, default=4.0, help="[transition] flat lead-in (start platform) CAP (m); excess slack is redistributed to the other flat bands.")
    p.add_argument("--end-platform", type=float, default=1.0, help="[transition] top-platform FLOOR after the last staircase (m), at exit_z.")
    p.add_argument("--size-y", type=float, default=1.2, help="Corridor width floor (m); actual width = max(nav_width, stair_width, this).")
    # -- multi-transition + nav knobs --
    p.add_argument("--transition-length", type=float, default=16.0, help="[transition] FIXED total length (m) of every transition terrain; flats absorb the slack.")
    p.add_argument("--num-transitions-min", type=int, default=2, help="[transition] min staircases per terrain (floored at 2).")
    p.add_argument("--num-transitions-max", type=int, default=3, help="[transition] max staircases per terrain.")
    p.add_argument("--inter-flat-min", type=float, default=3.0, help="[transition] inter-stair flat FLOOR (m); absorbs slack to hit --transition-length.")
    p.add_argument("--inter-flat-max", type=float, default=5.0, help="[transition] ignored — inter-stair flats are set by the fixed-length fill.")
    p.add_argument("--nav-width-min", type=float, default=1.5, help="[transition] min corridor width (m); the flat bands span this, stairs are centered narrower.")
    p.add_argument("--nav-width-max", type=float, default=3.0, help="[transition] max corridor width (m).")
    p.add_argument("--flat-width-margin", type=float, default=0.5, help="[transition] keep the spawn corner this far (m) from the corridor y-edge.")
    p.add_argument("--nav-y-kp", type=float, default=1.5, help="[transition] lateral position gain for goal-directed nav.")
    p.add_argument("--nav-y-kd", type=float, default=0.3, help="[transition] lateral damping gain (unused offline; lat vel = 0).")
    p.add_argument("--nav-heading-kp", type=float, default=0.5, help="[transition] heading gain toward the goal bearing.")
    p.add_argument("--nav-vy-clamp", type=float, default=0.4, help="[transition] |vy| clamp on the nav lateral command (m/s).")
    p.add_argument("--nav-wz-clamp", type=float, default=0.5, help="[transition] |wz| clamp on the nav yaw command (rad/s).")
    p.add_argument("--transition", choices=("raw", "linear", "quadratic"), default="linear",
                   help="Skill-transition (seam) strategy for ALL clips incl. the flat->stair commit: 'linear' cross-fade (default, online parity), 'quadratic' ease-in cross-fade (holds the old stride longer), or 'raw' hard cut.")
    p.add_argument("--transition-blend-end-phi", type=float, default=0.3,
                   help="Cross-fade completion length as a fraction of a STEP (half-cycle), in (0, 1]; e.g. 0.3 settles 30%% of the way into the step. Applies to 'linear'/'quadratic'.")
    p.add_argument("--blend-pace", choices=("old", "new", "longer"), default="old",
                   help="Which stride's step DURATION the cross-fade window is a fraction of at a cross-gait (walk<->stair) seam: 'old' leaving stride (default), 'new' incoming, or 'longer' of the two.")
    p.add_argument("--approach-threshold", type=float, default=0.15, help="Stance-to-stair distance (m) to commit the climb.")
    p.add_argument("--debug-interp", action="store_true",
                   help="Log per-step skill-transition state (blend_active/blend_alpha) into the .h5 for scripts/rsl_rl/plot_clip_frames.py.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--hf-repo", default=None, help="Optional HuggingFace repo id for the source library.")
    args = p.parse_args()

    device = torch.device(args.device)
    manager = MultiSkillManager(args.source, device, env=None, hf_repo=args.hf_repo)
    skill_list = [s.name for s in manager.skills]

    # Locate the stair (terrain-aware) skill by conditioning (read straight from
    # each trajectory's conditioner.terrain), not by name.  A flat-only library
    # is fine -- it just yields flat clips (terrain/transition need a stair skill).
    terrain_skill_names = [s.name for s in manager.skills if s.has_terrain]
    has_terrain_lib = bool(terrain_skill_names)
    stair_skill = terrain_skill_names[0] if has_terrain_lib else None

    counts = split_counts(
        args.num_terrains,
        {"flat": args.frac_flat, "terrain": args.frac_terrain, "transition": args.frac_transition},
    )
    if not has_terrain_lib and counts["terrain"] + counts["transition"] > 0:
        print(
            f"[generate_terrain_clips] WARNING: library '{args.source}' has no "
            "terrain-aware skill (no trajectory declares conditioner.terrain != "
            "'flat'); generating FLAT clips only and ignoring the terrain/transition "
            "fractions. Provide a library with a stair skill to generate terrain clips."
        )
        counts = {"flat": args.num_terrains, "terrain": 0, "transition": 0}

    # Velocity buckets.  Flat clips use the full library coverage; terrain /
    # transition clips climb straight, so those samplers use forward-only
    # buckets (vy = wz = 0) to avoid lateral / yaw drift into the stair anchor.
    # (The generator's own bucketing tables come from the flat sampler and are
    # fixed when its sampler is swapped per batch; the samplers only differ in
    # which velocities they emit.)
    buckets = _library_derived_buckets(manager)
    default_probs = {s.name: float(s.num_trajectories) for s in manager.skills}
    # Flat clips: library proportions, but never the stair skill.
    flat_probs = _probs(skill_list, {k: v for k, v in default_probs.items() if k != stair_skill})
    samplers = {"flat": _make_sampler(buckets, skill_list, flat_probs, args.dt, args.seed + 1)}

    hd_grid, width_range, specs, walk_names = [], (0.0, 0.0), [], []
    if has_terrain_lib:
        # Discrete (step_height, step_depth) grid from the library.  The scan
        # folder defaults to the auto-detected terrain skill (its name is the
        # subfolder name), overridable with --stair-skill.
        scan_folder = args.stair_skill or stair_skill
        hd_grid = scan_stair_step_dims(args.source, hf_repo=args.hf_repo, skill_folder=scan_folder)
        stair_idx = manager.skill_name_to_idx[stair_skill]
        off = manager._skill_traj_offset[stair_idx]
        cnt = manager.skills[stair_idx].num_trajectories
        # Tread-width range declared by the stair trajectories (terrain_width).
        # A fixed width has lo == hi (built at that width on every terrain); an
        # open range (e.g. {min: 1.0}) has hi == +inf, capped by --stair-width-max
        # and sampled per terrain by sample_terrain_specs.
        w_lo = float(manager._terrain_dim_lo[off:off + cnt, 0].min())
        w_hi = float(manager._terrain_dim_hi[off:off + cnt, 0].max())
        w_hi = min(w_hi, args.stair_width_max)
        w_lo = min(w_lo, w_hi)
        width_range = (w_lo, w_hi)
        span = f"[{w_lo:.3f}, {w_hi:.3f}]" if w_hi > w_lo else f"{w_lo:.3f} (fixed)"
        print(f"[generate_terrain_clips] stair skill '{stair_skill}': {len(hd_grid)} (h,d) pairs, tread width {span}")

        fwd_buckets = {
            name: _Bucket(lin_vel_x=b.lin_vel_x, lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
            for name, b in buckets.items()
        }
        # Command vx spans the UNION of the walking + stair vx ranges.  The stair
        # skill's own coverage may be a single speed (e.g. 0.4), which would pin
        # the commanded vx.  Instead we sample the command from the wider walking
        # + stair range and let the nearest-neighbour trajectory selection pick
        # the closest available speed WITHIN each skill (stairs clamp to their one
        # speed; the flat lead-in walks at the nearest walk speed) -- so the robot
        # sees the whole commanded range while climbing at whatever the stair
        # library actually has.  Only the STAIR skill's bucket is widened (the
        # sampled command vx uses the active skill's bucket): standing stays 0 and
        # running is not used on terrain, so walking coverage comes via walk skills.
        walk_names = [
            s.name for s in manager.skills
            if s.name not in (stair_skill, "running") and buckets[s.name].lin_vel_x[1] > 0.0
        ]
        stair_vx = buckets[stair_skill].lin_vel_x
        lo = min([stair_vx[0]] + [buckets[n].lin_vel_x[0] for n in walk_names])
        hi = max([stair_vx[1]] + [buckets[n].lin_vel_x[1] for n in walk_names])
        union_vx = (max(0.0, float(lo)), float(hi))   # forward-only command range
        fwd_buckets[stair_skill] = _Bucket(lin_vel_x=union_vx, lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
        print(
            f"[generate_terrain_clips] terrain/transition command vx range = "
            f"[{union_vx[0]:.2f}, {union_vx[1]:.2f}] (union of walking {walk_names} + stairs; "
            f"stair library vx = [{stair_vx[0]:.2f}, {stair_vx[1]:.2f}])"
        )
        # Terrain-only: forward union-range velocities; Transition: + some standing (no running).
        samplers["terrain"] = _make_sampler(fwd_buckets, skill_list, _probs(skill_list, {stair_skill: 1.0}), args.dt, args.seed + 2)
        samplers["transition"] = _make_sampler(
            fwd_buckets, skill_list, _probs(skill_list, {stair_skill: 0.85, "standing": 0.15}), args.dt, args.seed + 3)

        rng = np.random.default_rng(args.seed)
        specs = sample_terrain_specs(
            rng, counts["terrain"], counts["transition"], hd_grid, width_range,
            num_steps_range=(args.num_steps_min, args.num_steps_max),
            lead_in_range=(args.lead_in_min, args.lead_in_max),
            end_platform=args.end_platform,
            num_transitions_range=(args.num_transitions_min, args.num_transitions_max),
            inter_flat_range=(args.inter_flat_min, args.inter_flat_max),
            nav_width_range=(args.nav_width_min, args.nav_width_max),
            flat_width_margin=args.flat_width_margin,
            transition_length=args.transition_length,
        )

    # One generator; its sampler is swapped per batch and terrain context bound
    # per terrain.  Exclude the stair skill from the FLAT rollout's selection
    # (velocity-bucketed) so flat clips never grab a stair stride; the terrain
    # path selects stair strides directly and is unaffected.
    generator = OfflineClipGenerator(
        manager, samplers["flat"], dt=args.dt, clip_length_s=args.clip_length,
        exclude_skills=[stair_skill] if has_terrain_lib else [],
        transition=args.transition,
        transition_blend_end_phi=args.transition_blend_end_phi,
        blend_pace=args.blend_pace,
        approach_threshold=args.approach_threshold,
        nav_gains=(args.nav_y_kp, args.nav_y_kd, args.nav_heading_kp),
        nav_vy_clamp=(-args.nav_vy_clamp, args.nav_vy_clamp),
        nav_wz_clamp=(-args.nav_wz_clamp, args.nav_wz_clamp),
        debug_interp=args.debug_interp,
        device=device,
    )

    total_clips = args.num_terrains * args.clips_per_terrain
    print(
        f"[generate_terrain_clips] terrains {counts} x {args.clips_per_terrain} clips = {total_clips} total"
    )

    gen_config = {
        "num_terrains": args.num_terrains,
        "fracs": {"flat": args.frac_flat, "terrain": args.frac_terrain, "transition": args.frac_transition},
        "counts": counts,
        "clips_per_terrain": args.clips_per_terrain,
        "clip_length_s": args.clip_length,
        "dt": args.dt,
        "stair_skill": stair_skill,
        "num_steps_range": [args.num_steps_min, args.num_steps_max],
        "lead_in_range": [args.lead_in_min, args.lead_in_max],
        "end_platform": args.end_platform,
        "size_y": args.size_y,
        "transition_length": args.transition_length,
        "num_transitions_range": [args.num_transitions_min, args.num_transitions_max],
        "inter_flat_range": [args.inter_flat_min, args.inter_flat_max],
        "nav_width_range": [args.nav_width_min, args.nav_width_max],
        "flat_width_margin": args.flat_width_margin,
        "nav_gains": [args.nav_y_kp, args.nav_y_kd, args.nav_heading_kp],
        "nav_vy_clamp": args.nav_vy_clamp,
        "nav_wz_clamp": args.nav_wz_clamp,
        "transition": args.transition,
        "transition_blend_end_phi": args.transition_blend_end_phi,
        "blend_pace": args.blend_pace,
        "approach_threshold": args.approach_threshold,
        "skill_list": skill_list,
        "hd_grid": [list(hd) for hd in hd_grid],
        "width_range": list(width_range),
        "stair_width_max": args.stair_width_max,
    }
    clip_fp = fingerprint_clip_config(manager.library_fingerprint, gen_config, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # Probe a flat clip for layout metadata (output names, T), then size the writer.
    generator.clear_terrain_context()
    generator.sampler = samplers["flat"]
    probe = generator.generate_batch(1)
    writer = ClipLibraryWriter(args.out, total_clips, probe["meta"])
    writer.set_attr("source_library_fingerprint", manager.library_fingerprint)
    writer.set_attr("clip_fingerprint", clip_fp)
    writer.set_attr("generator_config", gen_config)

    manifest = orchestrate(
        generator=generator, samplers=samplers, writer=writer, counts=counts,
        specs=specs, clips_per_terrain=args.clips_per_terrain, size_y_floor=args.size_y,
        seed=args.seed, device=device,
        # Approach slow-down uses the forward-walking skill(s) only (masked by skill).
        approach_skills=tuple(walk_names) or None,
    )
    writer.set_attr("terrain_manifest", manifest)
    writer.close()
    print(f"\n[generate_terrain_clips] wrote {total_clips} clips ({len(manifest)} terrains) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
