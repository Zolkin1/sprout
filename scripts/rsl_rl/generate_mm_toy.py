#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Motion-matching toy clip (bones-seed only; no skills / traj-opt / handoff).

Drives a scripted ``(vx, vy, omega)`` command through motion matching over the bones-seed
database and writes a standard ``.h5`` clip.  Runs with NO IsaacLab.  Two algorithms,
selectable with ``--method`` (they share the feature + playback, so only the SEARCH differs):

* ``vanilla`` -- the greedy, real-time-style matcher (default).
* ``beam``    -- offline beam-search DP over a precomputed candidate graph (global search;
  the graph is cached like the feature cache, keyed by the filter set + params).

    python scripts/rsl_rl/generate_mm_toy.py \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --bones-cache clips/bones_cache_locomotion.h5 \
        --method beam --out clips/g1_mm_toy.h5 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.motion_matching.beam import BeamConfig, BeamMatcher
from motion_gen.motion_matching.bones_ingest import load_cache
from motion_gen.motion_matching.dp import DPConfig, DPMatcher
from motion_gen.motion_matching.clip_output import ClipOutputBuilder
from motion_gen.motion_matching.graph import (
    CandidateGraph,
    GraphConfig,
    build_candidate_graph,
    graph_cache_key,
    graph_cache_path,
)
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig, VanillaMatcher
from motion_gen.multiskill_manager import MultiSkillManager

_CONTACT = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Default velocity profile: piecewise-constant (vx fwd, vy left, wz yaw-rate), each held
# for ``t`` seconds.  stand -> walk -> turn-while-walking -> walk -> stop.
DEFAULT_SCHEDULE = [
    {"t": 3.0, "vx": -1.0},
    {"t": 3.0, "vx": 1.0}, #$1.2},
    {"t": 2.0, "vx": 0.0, "vy": 1.0, "wz": 0.0},
    {"t": 4.0, "vx": 3.5, "vy": 0.6},
    {"t": 1.0, "vx": 0.0},
]


def _make_command(schedule: list[dict]):
    """Build ``command(t)->(vx,vy,wz)`` + total duration from a list of segments.

    Each segment is ``{t, vx, vy, wz}``; by default it HOLDS ``(vx,vy,wz)`` constant for ``t``
    seconds (a step change from the previous segment = a velocity JUMP).  Add ``"to":
    {vx,vy,wz}`` to make the segment linearly RAMP from ``(vx,vy,wz)`` to ``to`` across its
    duration.  (A ramp keeps the regime signal ~0 throughout; a jump dips it around the step.)"""
    def _v(d):
        return (float(d.get("vx", 0.0)), float(d.get("vy", 0.0)), float(d.get("wz", 0.0)))

    edges = []            # (t_start, t_end, v0, v1)
    acc = 0.0
    for s in schedule:
        dur = float(s["t"]); v0 = _v(s)
        v1 = _v(s["to"]) if isinstance(s.get("to"), dict) else v0
        edges.append((acc, acc + dur, v0, v1))
        acc += dur
    total = acc

    def command(t: float):
        for t0, t1, v0, v1 in edges:
            if t < t1:
                if v0 == v1 or t1 <= t0:
                    return v0
                a = (t - t0) / (t1 - t0)
                return tuple((1.0 - a) * c0 + a * c1 for c0, c1 in zip(v0, v1))
        return edges[-1][3]                                  # hold the last segment's end value

    return command, total


def _csv_set(s):
    # ';'-separated (NOT ','): some bones-seed labels contain a comma, e.g. "running, turning".
    return None if not s else [x.strip() for x in s.split(";") if x.strip()]


def _load_schedule(spec: str | None) -> list[dict]:
    """Parse ``--schedule``: a JSON list (inline) or ``@path.json``; None -> default toy."""
    if not spec:
        return DEFAULT_SCHEDULE
    if spec.startswith("@"):
        with open(spec[1:]) as fh:
            return json.load(fh)
    return json.loads(spec)


def _build_or_load_graph(matcher, args, filters, clean_paths=None):
    """Load the cached candidate graph for this DB/config, or build+cache it.  Shared by the
    beam and dp branches (both use the identical postural eps-ball graph).  When
    ``--include-clean``, the clean clips are tagged STEADY + given a periodic wrap edge, and the
    clean-library identity is folded into the cache key (distinct from the bones-only graph)."""
    gcfg = GraphConfig(k_min=args.graph_k_min, k_max=args.graph_k_max, k_target=args.graph_k_target,
                       same_clip_cap=args.graph_same_clip_cap, dt=args.dt)
    include_clean = bool(getattr(args, "include_clean", False))
    steady_mask = matcher._src_steady if include_clean else None
    periodic_mask = matcher._src_periodic if include_clean else None
    key = graph_cache_key(args.bones_cache, filters=filters, styles=_csv_set(args.bones_style),
                          exclude_mirror=True, exclude_name_substrings=_csv_set(args.exclude_name),
                          limit=args.bones_limit, pose_weights=matcher.pose_weights(), cfg=gcfg,
                          include_clean=include_clean, clean_sources=clean_paths)
    cache_file = args.graph_cache or str(graph_cache_path(key))
    graph = None if args.rebuild_graph else CandidateGraph.load(cache_file, expected_key=key)
    if graph is None:
        print(f"[toy] building candidate graph (postural eps-ball, k in "
              f"[{args.graph_k_min},{args.graph_k_max}]{', +clean' if include_clean else ''}) ...")
        cost_feat = matcher._norm[:, matcher._traj_dim:]         # full 27-D pose (jump_cost)
        select_feat = cost_feat[:, POSE_POSTURAL_IDX]           # postural (feet+hand pos) selection
        graph = build_candidate_graph(select_feat, cost_feat, matcher.feat_raw, matcher.owner,
                                      matcher.src_start, matcher.src_len, matcher.yaw_all, gcfg,
                                      steady_owner_mask=steady_mask, periodic_owner_mask=periodic_mask,
                                      key=key)
        graph.save(cache_file)
        print(f"[toy] cached candidate graph -> {cache_file}")
    else:
        print(f"[toy] loaded cached candidate graph <- {cache_file}")
    return graph


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt library (only for the output channel layout).")
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--clip-length", type=float, default=None,
                   help="Clip length (s); default = the schedule's total duration.")
    p.add_argument("--schedule", default=None,
                   help='Velocity profile as JSON (inline or @file): a list of segments '
                        '{"t": seconds, "vx":, "vy":, "wz":} held piecewise-constant. '
                        'vx=forward, vy=left, wz=yaw-rate (rad/s). Omit for the default toy.')
    p.add_argument("--device", default="cpu")
    p.add_argument("--method", choices=("vanilla", "beam", "dp"), default="vanilla",
                   help="Search algorithm: 'vanilla' (greedy), 'beam' (approx global), or "
                        "'dp' (exact backward DP, local-velocity only).")
    p.add_argument("--bones-package", default="Locomotion")
    p.add_argument("--bones-category", default=None,
                   help="';'-separated bones-seed 'category' values to keep, e.g. "
                        "'Basic Locomotion Neutral;Baseline' (default: any category in the cache).")
    p.add_argument("--bones-style", default=None)
    # ';'-separated (some labels contain a comma, e.g. "running, turning").  Default set covers
    # the toy command: walk/jog/run + turning + a way to decelerate/stop ("standing idle" +
    # start/stop "transition").  Turning + stop data are REQUIRED for the toy to turn and to
    # come to rest; the run labels give the sprint segment the speed it needs.
    p.add_argument("--bones-movement",
                   default="walking;jogging;running;running, turning;turning;standing idle;transition")
    p.add_argument("--bones-body", default="standing")
    p.add_argument("--bones-limit", type=int, default=None)
    # 'jump'/'roll' drop stunt clips mislabeled as running (jump_form_box_to_safety_roll,
    # avoid_obstacle_jump_run) whose pelvis leaps to ~1.3m / rolls to the floor.
    p.add_argument("--exclude-name", default="injured;limp;drunk;jump;roll",
                   help="';'-separated substrings; drop any clip whose filename contains "
                        "one (case-insensitive), e.g. 'injured;jump'.")
    # --- vanilla (greedy) knobs ---
    p.add_argument("--search-interval", type=float, default=0.1)
    p.add_argument("--min-dwell", type=float, default=0.4,
                   help="[vanilla] Min seconds to commit to a clip after a jump (anti-thrash).")
    p.add_argument("--inertia-ratio", type=float, default=0.8,
                   help="[vanilla] Jump only if the best match beats continuing by this ratio (<1=stickier).")
    # --- beam (offline global) knobs ---
    p.add_argument("--beam-width", type=int, default=128, help="[beam] beam width B.")
    p.add_argument("--min-cut-gap", type=float, default=0.25,
                   help="[beam] cooldown (s) between cuts (anti-thrash).")
    p.add_argument("--cut-penalty", type=float, default=3.0, help="[beam] additive cost per cut.")
    p.add_argument("--w-track", type=float, default=1.0,
                   help="[beam] local velocity-window tracking weight (smoothness/gait).")
    p.add_argument("--w-pos", type=float, default=2.0,
                   help="[beam] GLOBAL position-tracking weight (accumulated world pos vs integrated ref).")
    p.add_argument("--w-traj", type=float, default=0.0,
                   help="[beam] local 1s-trajectory match weight (superseded by --w-pos; off by default).")
    p.add_argument("--w-tier", type=float, default=0.0,
                   help="clean STEADY-tier weight for the ACTIVE method (vanilla/beam/dp); needs "
                        "--include-clean; 0 = off (bones-only behavior).")
    p.add_argument("--track-window", type=int, default=10,
                   help="[beam] bidirectional velocity-match half-window W (frames).")
    # --- clean steady-state trajectory library (STEADY tier) ---
    p.add_argument("--include-clean", action="store_true",
                   help="Fold the clean periodic steady-state trajectory library into the DB (STEADY tier).")
    p.add_argument("--clean-source", default="trajectories/skills/library_v1",
                   help="Clean library path (local dir, or HF-relative when --clean-hf-repo is set).")
    p.add_argument("--clean-hf-repo", default="zolkin/robot_rl",
                   help="HuggingFace repo for the clean library (downloads to ./hf/<path>).")
    p.add_argument("--clean-skills", default=None,
                   help="';'-separated subset of clean skills to include (default: all).")
    p.add_argument("--regime-ramp", type=float, default=0.5,
                   help="Steady->neutral ramp half-width (s) for the tier cost (~1 s around a change).")
    p.add_argument("--soft-tier-penalty", type=float, default=2.0,
                   help="Magnitude of the per-frame VARIED penalty during steady regions.")
    p.add_argument("--tier-k", type=int, default=16,
                   help="[vanilla] nearest neighbours to re-rank when the tier penalty is on.")
    p.add_argument("--tier-nn-k", type=int, default=1,
                   help="Reward the k clean clips nearest the command by velocity (k=1 = single "
                        "NN; velocity ties broken toward the stillest clip, e.g. stand over shuffle).")
    # --- dp (exact backward DP) knobs ---
    p.add_argument("--dp-cut-penalty", type=float, default=3.0, help="[dp] additive cost per cut.")
    p.add_argument("--dp-w-track", type=float, default=1.0, help="[dp] velocity-tracking weight.")
    p.add_argument("--dp-w-traj", type=float, default=0.0,
                   help="[dp] future-trajectory (standard MM lookahead) weight; velocity DP=1/0, trajectory DP=0/1.")
    p.add_argument("--dp-track-window", type=int, default=10, help="[dp] velocity-match half-window (frames).")
    p.add_argument("--graph-k-max", type=int, default=50, help="[beam] max jump fan-out per frame.")
    p.add_argument("--graph-k-min", type=int, default=25, help="[beam] min jump fan-out per frame (floor).")
    p.add_argument("--graph-k-target", type=int, default=40,
                   help="[beam] eps auto-calibrated so the median frame gets ~this many candidates.")
    p.add_argument("--graph-same-clip-cap", type=int, default=2,
                   help="[beam] max candidates allowed from a frame's own clip.")
    p.add_argument("--graph-cache", default=None,
                   help="[beam] explicit graph .npz path (default: auto content-addressed cache).")
    p.add_argument("--rebuild-graph", action="store_true",
                   help="[beam] force a rebuild even if a cached graph matches.")
    p.add_argument("--print-clips", action="store_true",
                   help="Print which bones clip is played over which time interval.")
    args = p.parse_args()

    dev = args.device
    cfg = VanillaConfig(dt=args.dt, search_interval_s=args.search_interval,
                        min_dwell_s=args.min_dwell, inertia_ratio=args.inertia_ratio,
                        w_tier=args.w_tier, soft_tier_penalty=args.soft_tier_penalty,
                        regime_ramp_s=args.regime_ramp, tier_k=args.tier_k,
                        tier_nn_k=args.tier_nn_k)
    filters = {"package": args.bones_package}
    if args.bones_category:
        filters["category"] = _csv_set(args.bones_category)
    if args.bones_movement:
        filters["content_type_of_movement"] = _csv_set(args.bones_movement)
    if args.bones_body:
        filters["content_body_position"] = _csv_set(args.bones_body)

    command, total = _make_command(_load_schedule(args.schedule))
    clip_len = args.clip_length if args.clip_length is not None else total
    T = int(round(clip_len / args.dt))

    print(f"[toy] loading bones cache (filters={filters}, style={args.bones_style}, "
          f"exclude_name={args.exclude_name}) ...")
    sources, _ = load_cache(args.bones_cache, filters=filters, styles=_csv_set(args.bones_style),
                            exclude_name_substrings=_csv_set(args.exclude_name),
                            limit=args.bones_limit, device=dev)
    clean_paths = None
    if args.include_clean:
        from motion_gen.motion_matching.clean_ingest import ingest_clean_library
        print(f"[toy] ingesting clean library {args.clean_source} (hf_repo={args.clean_hf_repo}) ...")
        clean = ingest_clean_library(args.clean_source, hf_repo=args.clean_hf_repo, dt=args.dt,
                                     device=dev, skills=_csv_set(args.clean_skills))
        print(f"[toy] + {len(clean)} clean STEADY clips (periodic)")
        sources = sources + clean
        base = (Path("hf") / args.clean_source) if args.clean_hf_repo else Path(args.clean_source)
        clean_paths = [str(base / "manifest.csv")]     # library identity for the graph cache key
    nframes = sum(s.n for s in sources)
    if args.method == "vanilla":
        print(f"[toy] {len(sources)} clips; building vanilla matcher (KD-tree over {nframes} frames) ...")
        matcher = VanillaMatcher(sources, cfg, device=dev)
        generator_name = "vanilla_motion_matching"
    elif args.method == "beam":
        print(f"[toy] {len(sources)} clips; building beam matcher DB ({nframes} frames) ...")
        bcfg = BeamConfig(dt=args.dt, beam_width=args.beam_width, min_cut_gap_s=args.min_cut_gap,
                          cut_penalty=args.cut_penalty, w_track=args.w_track, w_pos=args.w_pos,
                          w_traj=args.w_traj, w_tier=args.w_tier, track_window=args.track_window,
                          soft_tier_penalty=args.soft_tier_penalty, regime_ramp_s=args.regime_ramp,
                          tier_nn_k=args.tier_nn_k)
        matcher = BeamMatcher(sources, cfg, bcfg, device=dev)
        matcher.attach_graph(_build_or_load_graph(matcher, args, filters, clean_paths))
        generator_name = "beam_motion_matching"
    else:  # dp -- exact backward DP, local-velocity only
        print(f"[toy] {len(sources)} clips; building DP matcher DB ({nframes} frames) ...")
        dcfg = DPConfig(dt=args.dt, cut_penalty=args.dp_cut_penalty, w_track=args.dp_w_track,
                        w_traj=args.dp_w_traj, track_window=args.dp_track_window,
                        w_tier=args.w_tier, soft_tier_penalty=args.soft_tier_penalty,
                        regime_ramp_s=args.regime_ramp, tier_nn_k=args.tier_nn_k)
        matcher = DPMatcher(sources, cfg, dcfg, device=dev)
        matcher.attach_graph(_build_or_load_graph(matcher, args, filters, clean_paths))
        generator_name = "dp_motion_matching"

    print(f"[toy] [{args.method}] matching {T} steps ({clip_len:.1f}s) against the velocity profile ...")
    root_pos, root_quat, joints = matcher.generate(command, T, debug=args.print_clips)
    for w in getattr(matcher, "warnings", []) or []:
        print(f"[toy][{args.method}][WARN] {w}")
    if args.method in ("beam", "dp"):
        print(f"[toy] {args.method} cuts: {getattr(matcher, 'n_cuts', '?')}")

    if args.print_clips:
        # Collapse consecutive same-source steps into (t_start, t_end, source, local frames).
        segs = []
        for step, _jump, o, l, _bs, _os in matcher.dbg:
            if segs and segs[-1]["o"] == o:
                segs[-1]["s1"] = step; segs[-1]["l1"] = l
            else:
                segs.append({"s0": step, "s1": step, "o": o, "l0": l, "l1": l})
        print(f"\n[toy] clip usage ({len(segs)} segments):")
        print(f"  {'t_start':>8}  {'t_end':>7}  {'dur':>5}   {'frames':>9}   clip")
        for s in segs:
            t0, t1 = s["s0"] * args.dt, (s["s1"] + 1) * args.dt
            print(f"  {t0:8.2f}  {t1:7.2f}  {t1 - t0:5.2f}   [{s['l0']:>4}-{s['l1']:<4}] {matcher.src_names[s['o']]}")
        print()

    manager = MultiSkillManager(args.source, dev, env=None)
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT, device=dev)
    y, dy, contact = builder.build(root_pos, root_quat, joints, args.dt)
    tvel = torch.tensor([[*command(i * args.dt)] for i in range(T)], dtype=torch.float32, device=dev)
    batch = {
        "y_des": y.unsqueeze(0), "dy_des": dy.unsqueeze(0), "contact": contact.unsqueeze(0),
        "skill_id": torch.zeros(1, T, dtype=torch.uint8, device=dev),
        "stride_idx": torch.full((1, T), -1, dtype=torch.int32, device=dev),
        "target_vel": tvel.unsqueeze(0), "terrain_id": torch.tensor([-1], dtype=torch.int32, device=dev),
        "terrain_type": "flat",
    }
    meta = {
        "dt": args.dt, "clip_length_s": clip_len, "num_steps": int(T),
        "pos_output_names": list(manager.pos_output_names),
        "vel_output_names": list(manager.vel_output_names),
        "contact_frames": _CONTACT, "ref_frame": "pelvis_link",
        "skill_list": [s.name for s in manager.skills],
    }
    writer = ClipLibraryWriter(args.out, 1, meta)
    writer.set_attr("generator", generator_name)
    writer.write_batch(0, batch)
    writer.close()
    lib = load_clip_library(args.out)
    yy = torch.as_tensor(lib.y_des)
    print(f"[toy] wrote {args.out}: y_des {tuple(yy.shape)} finite={bool(torch.isfinite(yy).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
