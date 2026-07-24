#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate VARIED flat reference clips via the shared locomotion DP matcher (parallelized).

Uses the SAME motion DB as the terrain/distractor generators (:func:`build_locomotion_matcher`: flat
bones + clean library_v2 strides + jump lead-ins), restricted to ``{flat, clean}`` so no elevated
lead-in leaks onto flat ground.  For each clip a randomized ``(vx, vy, wz)`` schedule -- covering
forward / backward / sidestep / turn / diagonal -- becomes a piecewise velocity command, and one DP
solve plays the matched clips' OWN root motion (natural velocity, no foot-pin; cuts inertialized).

The clip solves are independent over the shared read-only matcher, so they fan across a fork pool
(``--workers``); each clip is seeded by ``(--seed, clip_index)`` (reproducible, order-independent).
Records library_v2's fingerprint as ``source_library_fingerprint``.  NO IsaacLab.  Example::

    python scripts/rsl_rl/generate_mm_clips.py \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --bones-cache clips/bones_cache_locomotion.h5 \
        --out clips/g1_mm_flat_v1.h5 --device cpu --bones-limit 1500
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.motion_matching.clip_output import ClipOutputBuilder
from motion_gen.motion_matching.gen_parallel import default_workers, unit_pool
from motion_gen.motion_matching.locomotion_db import (
    APPROACH_FILTERS,
    build_locomotion_matcher,
    clean_library_fingerprint,
)
from motion_gen.multiskill_manager import MultiSkillManager
from motion_gen.world_lift import anchor_mask_from_free

_CONTACT_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Read-only state shared with the forked workers via copy-on-write (set in main() BEFORE the pool).
_SHARED: dict = {}


def _seg_vel(rng, args):
    """Random ``(vx, vy, wz, label)`` for one MOVING segment, sampled to cover the whole traj-opt range:
    forward / backward / sidestep / turn-while-walking / diagonal.  vx = forward (+) / backward (-),
    vy = lateral (sidestep), wz = yaw rate (turn)."""
    mode = str(rng.choice(["forward", "forward", "forward", "backward", "sidestep", "turn", "diagonal"]))
    vx = vy = wz = 0.0
    if mode == "forward":
        vx = float(rng.uniform(0.45, 1.0)) * (args.run_vx if rng.random() < 0.4 else args.walk_vx)
        wz = float(rng.uniform(-1.0, 1.0)) * args.turn_max if rng.random() < 0.3 else 0.0
    elif mode == "backward":
        vx = -float(rng.uniform(0.4, 0.9)) * args.back_vx
    elif mode == "sidestep":
        vy = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.4, 1.0)) * args.side_vy
    elif mode == "turn":
        vx = float(rng.uniform(0.2, 0.7)) * args.walk_vx
        wz = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.5, 1.0)) * args.turn_max
    elif mode == "diagonal":
        vx = float(rng.uniform(0.3, 0.8)) * args.walk_vx
        vy = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.3, 0.7)) * args.side_vy
    return vx, vy, wz, mode


def _schedule(rng, args):
    """A randomized ``{label, vx, vy, wz, duration_s}`` schedule: clean standing start, then mostly-moving
    segments (occasional pause) covering the forward/backward/lateral/turn command space."""
    segs = [{"label": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 1.0}]
    t = 1.0
    while t < args.clip_length - 0.5:
        if rng.random() < 0.12:
            vx = vy = wz = 0.0
            lab = "stand"
        else:
            vx, vy, wz, lab = _seg_vel(rng, args)
        dur = float(rng.uniform(1.5, 3.0))
        segs.append({"label": lab, "vx": vx, "vy": vy, "wz": wz, "duration_s": dur})
        t += dur
    return segs


def _cmd_from_schedule(sched):
    """Piecewise ``(vx, vy, wz)`` command over time from the segment schedule (the DP steering signal)."""
    segs, t0 = [], 0.0
    for s in sched:
        segs.append((t0, t0 + s["duration_s"], float(s["vx"]), float(s["vy"]), float(s["wz"])))
        t0 += s["duration_s"]

    def cmd(t):
        for a, b, vx, vy, wz in segs:
            if t < b:
                return (vx, vy, wz)
        return (segs[-1][2], segs[-1][3], segs[-1][4])
    return cmd


def _flat_worker(i: int):
    """Worker: generate flat clip ``i`` via one DP solve over the shared matcher (inherited by fork),
    restricted to the flat/clean categories.  Returns a one-element ``[record]`` (the fork-pool contract)."""
    S = _SHARED
    args, matcher, builder, T, dt = S["args"], S["matcher"], S["builder"], S["T"], S["args"].dt
    rng = np.random.default_rng([S["seed"], i])                # per-clip stream -> order-independent
    sched = _schedule(rng, args)
    cmd = _cmd_from_schedule(sched)
    Wp, Wq, Wj = matcher.generate(cmd, T, allowed_categories=APPROACH_FILTERS["flat"])
    y, dy, contact = builder.build(Wp, Wq, Wj, dt)
    tvel = torch.tensor([list(cmd(k * dt)) for k in range(T)], dtype=torch.float32)
    return [{"y_des": y.unsqueeze(0).cpu(), "dy_des": dy.unsqueeze(0).cpu(), "contact": contact.unsqueeze(0).cpu(),
             "target_vel": tvel.unsqueeze(0), "_sched": sched, "_n_cuts": int(getattr(matcher, "n_cuts", 0))}]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt library root -- used ONLY for the robot "
                                                   "pos/vel output-names (the clip DATA comes from the shared DB).")
    p.add_argument("--out", required=True, help="Output .h5 clip path.")
    p.add_argument("--bones-cache", required=True, help="Precomputed bones cache .h5 (flat locomotion source).")
    p.add_argument("--clean-source", default="trajectories/skills/library_v2",
                   help="Clean steady-stride library (the shared clip source of truth: fwd/back/run + "
                        "sidestep/diagonal/turn).")
    p.add_argument("--clean-hf-repo", default="zolkin/robot_rl")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--clip-length", type=float, default=9.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bones-limit", type=int, default=None, help="Cap flat bones clips (default: all).")
    p.add_argument("--n-clips", type=int, default=24,
                   help="Number of VARIED flat clips (each a randomized speed/turn schedule).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the randomized schedules (per-clip stream).")
    p.add_argument("--turn-max", type=float, default=0.8, help="Max |yaw-rate| (rad/s) for turning segments.")
    p.add_argument("--run-vx", type=float, default=2.5, help="Max forward (running) speed sampled (m/s).")
    p.add_argument("--walk-vx", type=float, default=1.0, help="Nominal walking speed (m/s).")
    p.add_argument("--back-vx", type=float, default=1.2, help="Max backward speed sampled (m/s).")
    p.add_argument("--side-vy", type=float, default=1.0, help="Max lateral (sidestep) speed sampled (m/s).")
    p.add_argument("--cut-penalty", type=float, default=3.0, help="DP additive cost per cut.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel clip-solve workers (0 = all cores - 2); DP solves fan across a fork pool.")
    p.add_argument("--flat-anchor", default="x,y,yaw",
                   help="Comma-separated FREE root DOFs (of x,y,z,roll,pitch,yaw) for the RL anchor mask.")
    args = p.parse_args()

    dev = args.device
    manager = MultiSkillManager(args.source, dev, env=None)   # robot output-names + skill_list only
    print(f"[mm] building shared locomotion DB (flat/clean; no lead-ins) ...")
    matcher = build_locomotion_matcher(
        bones_cache=args.bones_cache, clean_source=args.clean_source, clean_hf_repo=args.clean_hf_repo,
        bones_limit=args.bones_limit, include_leadins=False, dt=args.dt, device=dev,
        cut_penalty=args.cut_penalty, w_traj=1.0, w_track=0.0, log=lambda m: print(m.replace("[locomotion_db]", "[mm]")))
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT_FRAMES, device=dev)
    anchor = int(anchor_mask_from_free([x.strip() for x in args.flat_anchor.split(",") if x.strip()]))
    T = max(1, int(round(args.clip_length / args.dt)))
    N = int(args.n_clips)

    # --- fan the independent clip solves across a fork pool (shared read-only matcher via COW) ---
    _SHARED.update(matcher=matcher, builder=builder, args=args, T=T, seed=args.seed)
    n_workers = default_workers(N, args.workers)
    print(f"[mm] generating {N} varied flat clips via DP across {n_workers} worker(s) (T={T}) ...")
    records = unit_pool(_flat_worker, range(N), n_workers)

    meta = {
        "dt": args.dt, "clip_length_s": args.clip_length, "num_steps": int(T),
        "pos_output_names": list(manager.pos_output_names),
        "vel_output_names": list(manager.vel_output_names),
        "contact_frames": _CONTACT_FRAMES,
        "ref_frame": "pelvis_link",
        "skill_list": [s.name for s in manager.skills],
    }
    writer = ClipLibraryWriter(args.out, N, meta)
    writer.set_attr("source_library_fingerprint", clean_library_fingerprint(args.clean_source, args.clean_hf_repo))
    writer.set_attr("generator", "motion_matching_dp_v1")
    n_cuts_total = 0
    for i, b in enumerate(records):
        writer.write_batch(i, {
            "y_des": b["y_des"], "dy_des": b["dy_des"], "contact": b["contact"], "target_vel": b["target_vel"],
            "terrain_id": torch.tensor([-1], dtype=torch.int32),
            "terrain_type": "flat",
            "anchor_mask": torch.tensor([anchor], dtype=torch.uint8),
            "clip_len": torch.tensor([T], dtype=torch.int32),
        })
        n_cuts_total += b["_n_cuts"]
        print(f"[mm]   clip {i}: "
              + " -> ".join(f"{s['label']}(vx{s['vx']:+.1f},vy{s['vy']:+.1f},wz{s['wz']:+.1f})" for s in b["_sched"])
              + f"  [cuts={b['_n_cuts']}]")
    writer.close()
    print(f"[mm] wrote {args.out}  (mean {n_cuts_total / max(1, N):.1f} cuts/clip)")

    lib = load_clip_library(args.out)
    y = torch.as_tensor(lib.y_des)
    print(f"[mm] loaded back: y_des {tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
