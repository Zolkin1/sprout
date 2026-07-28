#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""DEPRECATED: part of the motion-matching / DP clip generator, retained for ablation studies
only -- a matcher research one-off (vanilla vs beam vs DP).  Not on the live pipeline; see
``source/motion_gen/motion_gen/motion_matching/README.md``.

Batch motion-matching experiment for investigating solutions.

Runs BOTH matchers (``vanilla`` greedy + ``beam`` offline global, sharing the cached v2
candidate graph) on several velocity-command schedules -- a simple forward proof-of-concept
plus N reproducible random schedules -- and, per solution, writes:

* a viewable ``.h5`` clip,
* a diagnostic figure (velocity tracking, top-down world path vs commanded, clip-switching
  timeline, and position-error / cost over time),

plus a ``summary.csv`` of switching + cost metrics across every (schedule, method).

The DB + vanilla KD-tree are built ONCE and reused across all solves (fast).  Everything
lands in ``--out-dir`` (default ``experiments/mm/``).  Runs with NO IsaacLab.

    python scripts/rsl_rl/mm_experiment.py \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --bones-cache clips/bones_cache_locomotion.h5 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from motion_gen.clips.clip_io import ClipLibraryWriter                     # noqa: E402
from motion_gen.world_lift import _ANCHOR_MASK_FLAT                        # noqa: E402
from motion_gen.motion_matching.beam import BeamConfig, BeamMatcher        # noqa: E402
from motion_gen.motion_matching.bones_ingest import load_cache            # noqa: E402
from motion_gen.motion_matching.dp import DPConfig, DPMatcher             # noqa: E402
from motion_gen.motion_matching.clip_output import ClipOutputBuilder      # noqa: E402
from motion_gen.motion_matching.graph import (                            # noqa: E402
    STEADY, CandidateGraph, GraphConfig, build_candidate_graph, graph_cache_key, graph_cache_path)
from motion_gen.motion_matching.playback import play_frame_path           # noqa: E402
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig  # noqa: E402
from motion_gen.multiskill_manager import MultiSkillManager               # noqa: E402

_CONTACT = ["left_ankle_roll_link", "right_ankle_roll_link"]
_FILTERS = {"package": "Locomotion",
            "content_type_of_movement": ["walking", "jogging", "running", "running, turning",
                                         "turning", "standing idle", "transition"],
            "content_body_position": ["standing"]}
_EXCLUDE = ["injured", "limp", "drunk", "jump", "roll"]


# ---- schedules -------------------------------------------------------------------------
def _schedules(n_random: int, seed: int) -> list[tuple[str, list]]:
    scheds = [("forward_poc", [{"t": 1.0, "vx": 0.0}, {"t": 4.0, "vx": 1.0}, {"t": 1.0, "vx": 0.0}])]
    rng = random.Random(seed)
    for i in range(n_random):
        segs = []
        for _ in range(rng.randint(4, 6)):
            if rng.random() < 0.15:                       # occasional stand
                segs.append({"t": round(rng.uniform(1.0, 2.0), 1), "vx": 0.0, "vy": 0.0, "wz": 0.0})
            else:
                segs.append({"t": round(rng.uniform(1.5, 3.0), 1),
                             "vx": round(rng.uniform(-1.2, 2.0), 2),
                             "vy": round(rng.uniform(-1.0, 1.0), 2),
                             "wz": round(rng.uniform(-0.7, 0.7), 2)})
        scheds.append((f"rand{i + 1}", segs))
    return scheds


def _make_command(schedule: list[dict]):
    segs = [(float(s["t"]), float(s.get("vx", 0.0)), float(s.get("vy", 0.0)), float(s.get("wz", 0.0)))
            for s in schedule]
    edges, acc = [], 0.0
    for dur, vx, vy, wz in segs:
        acc += dur; edges.append((acc, vx, vy, wz))

    def command(t):
        for end, vx, vy, wz in edges:
            if t < end:
                return (vx, vy, wz)
        return (edges[-1][1], edges[-1][2], edges[-1][3])
    return command, acc


# ---- analysis --------------------------------------------------------------------------
def _yaw(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _cdiff(a, dt):
    d = np.zeros_like(a); d[1:-1] = (a[2:] - a[:-2]) / (2 * dt); d[0] = (a[1] - a[0]) / dt
    d[-1] = (a[-1] - a[-2]) / dt; return d


def _analyze(bm, command, T, frame_path, jumped, method):
    dt = bm.cfg.dt
    frame_path = np.asarray(frame_path)
    T = len(frame_path)                              # beam may truncate on a dead-end
    jumped = list(jumped)[:T]
    rp, rq, jj, _ = play_frame_path(torch.as_tensor(frame_path), jumped, bm, bm.cfg)
    px = rp[:, 0].numpy(); py = rp[:, 1].numpy()
    yaw = _yaw(rq.numpy())
    vwx, vwy = _cdiff(px, dt), _cdiff(py, dt)
    cy, sy = np.cos(yaw), np.sin(yaw)
    act = np.stack([cy * vwx + sy * vwy, -sy * vwx + cy * vwy, _cdiff(yaw, dt)], -1)   # [T,3] body
    tgt = np.array([command(t * dt) for t in range(T)])
    # commanded integrated world path
    cx = cyy = cth = 0.0; cmd = np.zeros((T, 2))
    for t in range(T):
        vx, vy, wz = tgt[t]; c, s = np.cos(cth), np.sin(cth)
        cx += (c * vx - s * vy) * dt; cyy += (s * vx + c * vy) * dt; cth += wz * dt; cmd[t] = (cx, cyy)
    actual = np.stack([px - px[0], py - py[0]], 1)
    pos_err = np.linalg.norm(actual - cmd, axis=1)
    owner = bm.owner.numpy()[np.asarray(frame_path)]
    cuts = [t for t in range(1, T) if jumped[t]]
    seg_lens = np.diff([0] + cuts + [T])
    # beam cost decomposition (under the beam objective), for both methods (comparable)
    match = np.array([float(bm._match_cost(np.array([int(frame_path[t])]), t)[0]) for t in range(T)])
    pn = bm._norm[:, bm._traj_dim:].cpu().numpy().astype(np.float64)
    jump_c = sum(float(np.linalg.norm(pn[int(frame_path[t])] - pn[int(frame_path[t - 1])])) for t in cuts)
    m = {
        "method": method, "T": T, "n_cuts": len(cuts),
        "mean_dwell_s": float(seg_lens.mean() * dt), "min_dwell_s": float(seg_lens.min() * dt),
        "travel_ratio": float(np.sqrt((np.diff(actual, axis=0) ** 2).sum(1)).sum()
                              / max(1e-9, np.sqrt((np.diff(cmd, axis=0) ** 2).sum(1)).sum())),
        "final_gap_m": float(pos_err[-1]), "mean_pos_err_m": float(pos_err.mean()),
        "vel_MAE": float(np.abs(act - tgt).mean()),
        "mae_vx": float(np.abs(act[:, 0] - tgt[:, 0]).mean()),
        "mae_vy": float(np.abs(act[:, 1] - tgt[:, 1]).mean()),
        "mae_wz": float(np.abs(act[:, 2] - tgt[:, 2]).mean()),
        "sum_match_cost": float(match.sum()), "sum_jump_cost": float(jump_c),
        "cut_penalty_total": float(bm.bcfg.cut_penalty * len(cuts)),
    }
    series = dict(t=np.arange(T) * dt, act=act, tgt=tgt, actual=actual, cmd=cmd,
                  pos_err=pos_err, owner=owner, cuts=cuts, match=match)
    return rp, rq, jj, m, series


def _figure(name, method, s, m, path):
    t = s["t"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{name} — {method}   (cuts={m['n_cuts']}, mean dwell {m['mean_dwell_s']:.2f}s, "
                 f"travel {m['travel_ratio']:.2f}, final gap {m['final_gap_m']:.2f}m, MAE {m['vel_MAE']:.3f})")
    # velocity tracking
    a = ax[0, 0]
    for i, (lbl, col) in enumerate([("vx", "tab:blue"), ("vy", "tab:orange"), ("wz", "tab:green")]):
        a.plot(t, s["act"][:, i], color=col, lw=1.3, label=f"{lbl} act")
        a.plot(t, s["tgt"][:, i], color=col, lw=1.0, ls="--", alpha=0.7, label=f"{lbl} cmd")
    for c in s["cuts"]:
        a.axvline(c * (t[1] - t[0]), color="k", alpha=0.06)
    a.set_title("body-frame velocity: actual vs command"); a.set_xlabel("t (s)"); a.legend(fontsize=7, ncol=3)
    # world path
    a = ax[0, 1]
    a.plot(s["cmd"][:, 0], s["cmd"][:, 1], "r--", label="commanded")
    a.plot(s["actual"][:, 0], s["actual"][:, 1], "b-", label="actual")
    a.plot(0, 0, "ko", ms=4); a.set_aspect("equal", "datalim")
    a.set_title("world root path (top-down)"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.legend(fontsize=8)
    # switching timeline
    a = ax[1, 0]
    a.step(t, s["owner"], where="post", lw=1.0)
    for c in s["cuts"]:
        a.axvline(c * (t[1] - t[0]), color="r", alpha=0.5, lw=0.8)
    a.set_title(f"clip-switching timeline ({m['n_cuts']} cuts)"); a.set_xlabel("t (s)"); a.set_ylabel("source clip id")
    # position error + per-step match cost
    a = ax[1, 1]
    a.plot(t, s["pos_err"], "m-", label="position error (m)")
    a.set_xlabel("t (s)"); a.set_ylabel("position error (m)", color="m"); a.tick_params(axis="y", labelcolor="m")
    a2 = a.twinx(); a2.plot(t, s["match"], color="gray", lw=0.8, alpha=0.7, label="match cost/step")
    a2.set_ylabel("match cost / step", color="gray"); a2.tick_params(axis="y", labelcolor="gray")
    for c in s["cuts"]:
        a.axvline(c * (t[1] - t[0]), color="r", alpha=0.4, lw=0.8)
    a.set_title("tracking cost over time (position error + per-step match cost)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--out-dir", default="experiments/mm")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    # clean steady-state library (STEADY tier) -- compare {vanilla,beam,dp} x {no-clean, clean}
    p.add_argument("--include-clean", action="store_true")
    p.add_argument("--clean-source", default="trajectories/skills/library_v1")
    p.add_argument("--clean-hf-repo", default="zolkin/robot_rl")
    p.add_argument("--w-tier", type=float, default=0.0, help="STEADY-tier weight (needs --include-clean).")
    p.add_argument("--regime-ramp", type=float, default=0.5)
    p.add_argument("--soft-tier-penalty", type=float, default=2.0)
    p.add_argument("--tier-k", type=int, default=16)
    p.add_argument("--tier-nn-k", type=int, default=1)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device

    scheds = _schedules(args.n_random, args.seed)
    with open(os.path.join(args.out_dir, "schedules.json"), "w") as fh:
        json.dump({n: s for n, s in scheds}, fh, indent=2)
    print("[exp] schedules:")
    for n, s in scheds:
        print(f"  {n}: {s}")

    print("[exp] loading sources + building DB (once) ...")
    sources, _ = load_cache(args.bones_cache, filters=_FILTERS, exclude_name_substrings=_EXCLUDE, device=dev)
    clean_paths = None
    if args.include_clean:
        from motion_gen.motion_matching.clean_ingest import ingest_clean_library
        clean = ingest_clean_library(args.clean_source, hf_repo=args.clean_hf_repo, dt=args.dt, device=dev)
        print(f"[exp] + {len(clean)} clean STEADY clips")
        sources = sources + clean
        base = (Path("hf") / args.clean_source) if args.clean_hf_repo else Path(args.clean_source)
        clean_paths = [str(base / "manifest.csv")]
    tier_kw = dict(w_tier=args.w_tier, soft_tier_penalty=args.soft_tier_penalty,
                   regime_ramp_s=args.regime_ramp, tier_nn_k=args.tier_nn_k)
    vcfg = VanillaConfig(dt=args.dt, tier_k=args.tier_k, **tier_kw)
    bcfg = BeamConfig(dt=args.dt, **tier_kw)
    bm = BeamMatcher(sources, vcfg, bcfg, device=dev)
    gcfg = GraphConfig(dt=args.dt)
    steady_mask = bm._src_steady if args.include_clean else None
    periodic_mask = bm._src_periodic if args.include_clean else None
    key = graph_cache_key(args.bones_cache, filters=_FILTERS, styles=None, exclude_mirror=True,
                          exclude_name_substrings=_EXCLUDE, limit=None,
                          pose_weights=bm.pose_weights(), cfg=gcfg,
                          include_clean=args.include_clean, clean_sources=clean_paths)
    graph = CandidateGraph.load(graph_cache_path(key), expected_key=key)
    if graph is None:
        print("[exp] building candidate graph (cache miss) ...")
        cost_feat = bm._norm[:, bm._traj_dim:]
        graph = build_candidate_graph(cost_feat[:, POSE_POSTURAL_IDX], cost_feat, bm.feat_raw,
                                      bm.owner, bm.src_start, bm.src_len, bm.yaw_all, gcfg,
                                      steady_owner_mask=steady_mask, periodic_owner_mask=periodic_mask,
                                      key=key)
        graph.save(graph_cache_path(key))
    bm.attach_graph(graph)
    print("[exp] building vanilla full-feature KD-tree ...")
    bm._tree = cKDTree(bm._norm[bm._cand_idx].cpu().numpy().astype("float64"))
    print("[exp] building DP matcher (shares the same graph) ...")
    dm = DPMatcher(sources, vcfg, DPConfig(dt=args.dt, **tier_kw), graph=graph, device=dev)

    manager = MultiSkillManager(args.source, dev, env=None)
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT, device=dev)
    base_meta = {"dt": args.dt, "pos_output_names": list(manager.pos_output_names),
                 "vel_output_names": list(manager.vel_output_names), "contact_frames": _CONTACT,
                 "ref_frame": "pelvis_link", "skill_list": [s.name for s in manager.skills]}

    rows = []
    for name, sched in scheds:
        command, total = _make_command(sched)
        T = int(round(total / args.dt))
        bm._prepare(command, T)                     # sets _target/_ref_xy for cost eval + beam
        for method in ("vanilla", "beam", "dp"):
            mm = dm if method == "dp" else bm       # analysis/playback context for this method
            if method == "vanilla":
                fp, jm = bm._select_path(command, T); fp = fp.numpy()
            elif method == "beam":
                fp, jm = bm._beam_search(T)
            else:  # exact backward DP
                dm._prepare(command, T)
                fp, jm = dm._reconstruct(dm._dp_backward(T), T)
            rp, rq, jj, m, s = _analyze(mm, command, T, fp, jm, method)
            m = {"schedule": name, "clean_frac": float((graph.tier[np.asarray(fp)] == STEADY).mean()),
                 **m}
            rows.append(m)
            # clip
            y, dy, contact = builder.build(rp, rq, jj, args.dt)
            tvel = torch.tensor([[*command(i * args.dt)] for i in range(T)], dtype=torch.float32, device=dev)
            batch = {"y_des": y.unsqueeze(0), "dy_des": dy.unsqueeze(0), "contact": contact.unsqueeze(0),
                     "skill_id": torch.zeros(1, T, dtype=torch.uint8, device=dev),
                     "stride_idx": torch.full((1, T), -1, dtype=torch.int32, device=dev),
                     "target_vel": tvel.unsqueeze(0), "terrain_id": torch.tensor([-1], dtype=torch.int32),
                     "terrain_type": "flat",
                     "anchor_mask": torch.tensor([_ANCHOR_MASK_FLAT], dtype=torch.uint8)}
            meta = {**base_meta, "clip_length_s": total, "num_steps": int(T)}
            out_h5 = os.path.join(args.out_dir, f"{name}_{method}.h5")
            w = ClipLibraryWriter(out_h5, 1, meta); w.set_attr("generator", f"{method}_mm"); w.write_batch(0, batch); w.close()
            _figure(name, method, s, m, os.path.join(args.out_dir, f"{name}_{method}.png"))
            print(f"[exp] {name:12s} {method:8s} cuts={m['n_cuts']:3d} travel={m['travel_ratio']:.2f} "
                  f"gap={m['final_gap_m']:.2f} MAE={m['vel_MAE']:.3f}")

    cols = ["schedule", "method", "clean_frac", "n_cuts", "mean_dwell_s", "min_dwell_s", "travel_ratio",
            "final_gap_m", "mean_pos_err_m", "vel_MAE", "mae_vx", "mae_vy", "mae_wz",
            "sum_match_cost", "sum_jump_cost", "cut_penalty_total", "T"]
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols); wcsv.writeheader()
        for r in rows:
            wcsv.writerow({c: r.get(c) for c in cols})
    print(f"\n[exp] wrote {len(rows)} solutions + figures + summary.csv to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
