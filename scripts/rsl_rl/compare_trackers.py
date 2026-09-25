#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compare the trained teacher against NVIDIA's SONIC tracker on the same clips and terrain.

Staged like ``compare_sim2sim.py``, and for the same reason: the Isaac stages need a GPU and two
minutes of start-up each, while plotting and viewing should be re-runnable in a second without
holding the simulator.

    prep     -- list the clip library's terrain families and write cases.json
    rollout  -- one recorded bundle per (policy, reference source), all on the same tiles+clips
    plot     -- tracking / survival metrics + figures + summary.csv
    viser    -- hand off to view_tracker_comparison.py (scrub both robots side by side)

What makes the comparison controlled, rather than merely seeded:

* ``--clip_pick`` pins each terrain tile to a fixed clip, so the tile -> clip mapping cannot
  drift when the two policies terminate at different steps (a shared ``--seed`` does not survive
  that: the reset RNG streams diverge on the first termination).
* ``--capture_families`` selects the same tiles for every run from the scene, not from the policy.
* ``--no_dr`` drops the startup domain randomization on both sides, so the dynamics are identical.
* Terrain, tracking command, terminations and episode length are inherited from the teacher play
  env by the SONIC env, so "survived" means the same thing for both.

What is NOT identical, and is reported with the results: SONIC drives its own G1 (its gains,
defaults, action scale and collision geometry), so a gap is policy *and* body. It also has no
exteroception at all -- no height scan, no root-position target -- while the teacher gets a
176-ray scan.

Usage::

    P=source/robot_rl:source/motion_gen
    PYTHONPATH=$P python scripts/rsl_rl/compare_trackers.py --stage prep  --clip clips/g1_mb_v5.h5
    PYTHONPATH=$P python scripts/rsl_rl/compare_trackers.py --stage rollout \\
        --clip clips/g1_mb_v5.h5 --teacher_run logs/rsl_rl/g1/<run> \\
        --sonic_dir ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release \\
        --fm_checkpoint logs/flow_matching/<run>/<fm.pt>
    PYTHONPATH=$P python scripts/rsl_rl/compare_trackers.py --stage plot
    PYTHONPATH=$P python scripts/rsl_rl/compare_trackers.py --stage viser
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

# (policy, reference) -> (task id, extra CLI). One bundle per combination.
_RUNS = {
    ("teacher", "clip"): ("G1-teacher-clf-terrain-play", ()),
    ("teacher", "fm"): ("G1-tracker-clf-fm-terrain-play", ()),
    ("sonic", "clip"): ("G1-sonic-terrain-play", ()),
    ("sonic", "fm"): ("G1-sonic-fm-terrain-play", ()),
}


def bundle_path(out_dir: str, policy: str, reference: str, pick: int) -> str:
    """Where a given run's bundle lives."""
    return os.path.join(out_dir, f"{policy}_{reference}_pick{pick}.npz")


# --------------------------------------------------------------------------- prep

def stage_prep(args: argparse.Namespace) -> int:
    """Report the library's terrain families and record the comparison plan."""
    import h5py

    with h5py.File(args.clip, "r") as f:
        raw = f.attrs["terrain_manifest"]
        manifest = json.loads(raw) if isinstance(raw, str) else list(raw)
        terrain_id = f["terrain_id"][:]
    counts: dict[str, int] = {}
    for tile in manifest:
        fam = str(tile.get("family", tile.get("type", "?")))
        counts[fam] = counts.get(fam, 0) + 1
    n_flat = int((terrain_id == -1).sum())

    print(f"[prep] {args.clip}: {len(manifest)} terrain tiles, {len(terrain_id)} clips "
          f"({n_flat} on flat ground)")
    for fam, n in sorted(counts.items()):
        print(f"         {fam:28s} {n:4d} tiles")
    if n_flat:
        # Flat ground has no manifest entry (no geometry), so it has no tile COUNT -- it is one
        # pseudo-tile at terrain id -1 that debug_episodes resolves from the live scene under
        # family 'flat'. Printing it as "0 tiles" read as "flat will be skipped"; it is not, and
        # flat is the one regime where SONIC is in-distribution, so it must not look optional.
        print(f"         {'flat':28s}    1 pseudo-tile (terrain id -1), {n_flat} clips")
        counts["flat"] = 1
    families = args.families or ("all" if not n_flat else "all")
    cases = {
        "clip": os.path.abspath(args.clip),
        "families": families,
        "per_family": args.per_family,
        "clip_picks": args.clip_picks,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "ego_anchor_fraction": args.ego_anchor_fraction,
        "family_tile_counts": counts,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "cases.json"), "w") as fh:
        json.dump(cases, fh, indent=2)
    print(f"[prep] wrote {os.path.join(args.out_dir, 'cases.json')}")
    print(f"[prep] plan: families={families} x {args.per_family} tiles x "
          f"{args.clip_picks} clip pick(s) x {len(_select_runs(args))} runs")
    return 0


def _select_runs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The (policy, reference) combinations this invocation covers."""
    policies = ["teacher", "sonic"] if args.policies == "both" else [args.policies]
    refs = ["clip", "fm"] if args.references == "both" else [args.references]
    return [(p, r) for p in policies for r in refs]


# --------------------------------------------------------------------------- rollout

def stage_rollout(args: argparse.Namespace) -> int:
    """Record one bundle per (policy, reference, clip pick) via ``debug_episodes.py``."""
    cases = _load_cases(args)
    runs = _select_runs(args)
    todo = [(p, r, k) for k in range(cases["clip_picks"]) for (p, r) in runs]
    print(f"[rollout] {len(todo)} run(s): {[f'{p}/{r}/pick{k}' for p, r, k in todo]}")

    for policy, reference, pick in todo:
        task, extra = _RUNS[(policy, reference)]
        out = bundle_path(args.out_dir, policy, reference, pick)
        if os.path.exists(out) and not args.overwrite:
            print(f"[rollout] {out} exists; skipping (pass --overwrite to redo)")
            continue
        cmd = [
            sys.executable, "-u", os.path.join(_HERE, "debug_episodes.py"),
            "--task", task,
            "--clip", cases["clip"],
            "--clip_device", args.clip_device,
            "--num_envs", str(cases["num_envs"]),
            "--steps", str(cases["steps"]),
            "--capture_families", cases["families"],
            "--per_family", str(cases["per_family"]),
            "--clip_pick", str(pick),
            "--no_dr",
            "--ego_anchor_fraction", str(cases["ego_anchor_fraction"]),
            "--seed", str(cases["seed"]),
            "--out", out,
            *extra,
        ]
        if policy == "sonic":
            if not args.sonic_dir:
                raise SystemExit("[rollout] --sonic_dir is required for the SONIC runs.")
            cmd += ["--policy", "sonic", "--sonic_dir", args.sonic_dir]
        else:
            if not (args.teacher_run or args.checkpoint):
                raise SystemExit("[rollout] --teacher_run or --checkpoint is required for the "
                                 "teacher runs.")
            cmd += ["--checkpoint", _teacher_checkpoint(args)]
        if reference == "fm":
            if not args.fm_checkpoint:
                raise SystemExit("[rollout] --fm_checkpoint is required for the FM runs.")
            cmd += ["--fm_checkpoint", args.fm_checkpoint]

        print(f"\n[rollout] === {policy}/{reference}/pick{pick} -> {out}")
        print("  " + " ".join(cmd))
        log = out.replace(".npz", ".log")
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=_REPO, stdout=fh, stderr=subprocess.STDOUT)
        if rc != 0 or not os.path.exists(out):
            raise SystemExit(f"[rollout] {policy}/{reference}/pick{pick} failed (rc={rc}); "
                             f"see {log}")
        print(f"[rollout] ok -> {out}")
    return 0


def _teacher_checkpoint(args: argparse.Namespace) -> str:
    """Resolve the teacher checkpoint from ``--checkpoint`` or the newest one in ``--teacher_run``."""
    if args.checkpoint:
        return args.checkpoint
    run = args.teacher_run
    cands = sorted(f for f in os.listdir(run) if f.startswith("model_") and f.endswith(".pt"))
    if not cands:
        raise SystemExit(f"[rollout] no model_*.pt in {run}; pass --checkpoint explicitly.")
    return os.path.join(run, cands[-1])


# --------------------------------------------------------------------------- metrics

def _load_bundle(path: str) -> tuple[dict, dict]:
    """Load a recorder bundle as ``(arrays, meta)``."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files if k != "meta_json"}, json.loads(str(d["meta_json"]))


def episode_metrics(b: dict, meta: dict, ep: int) -> dict:
    """Tracking, velocity and survival metrics for one captured env.

    Metrics are computed over the FIRST episode only -- up to the first ``done`` -- because after
    that the env has been reset onto a new phase of the clip and the two policies are no longer
    on comparable timelines.

    Args:
        b: Bundle arrays.
        meta: Bundle metadata.
        ep: Index into the captured envs.

    Returns:
        A flat dict of scalars, ready for a CSV row.
    """
    dt = float(meta["dt"])
    done = b["done"][:, ep].astype(bool)
    end = int(np.argmax(done)) + 1 if done.any() else len(done)
    sl = slice(0, end)

    # Reference tracking: the CLF frames (pelvis + both feet) in the world.
    ref, act = b["ref_pos"][sl, ep], b["body_pos"][sl, ep]           # [t, F, 3]
    per_frame = np.linalg.norm(ref - act, axis=-1)                    # [t, F]
    frames = list(meta["frames"])

    # Joint tracking: the reference is in CLIP channel order, the measurement in DOF order.
    perm = [meta["ref_joint_names"].index(n) for n in meta["joint_names"]]
    jerr = b["joint_pos"][sl, ep] - b["ref_joint_pos"][sl, ep][:, perm]

    # Velocity tracking against the command the clip carries.
    cmd_v, ach_v = b["cmd_vel"][sl, ep], b["ach_vel"][sl, ep]
    ach_xyw = ach_v[:, [0, 1, 3]]                                     # (vx, vy, wz)
    nv = min(cmd_v.shape[1], 3)

    terminated = bool(b["terminated"][:end, ep].any())
    row = {
        "terrain_id": int(b["terrain_id"][ep]),
        "clip_id": int(b["clip_id"][ep]) if "clip_id" in b else -1,
        "survived_steps": end,
        "survived_s": end * dt,
        "terminated": int(terminated),
        "timed_out": int(bool(b["time_out"][:end, ep].any())),
        # Mean and worst-case pelvis error are the headline tracking numbers.
        "pelvis_err_mean_m": float(per_frame[:, 0].mean()),
        "pelvis_err_max_m": float(per_frame[:, 0].max()),
        "root_z_err_mean_m": float(np.abs(ref[:, 0, 2] - act[:, 0, 2]).mean()),
        # MPJPE over the frames the clip library carries, in mm, to line up with SONIC's own
        # reported metric family (theirs is over 14 bodies; ours is over these 3).
        "mpjpe_mm": float(per_frame.mean() * 1000.0),
        "joint_err_mean_rad": float(np.abs(jerr).mean()),
        "joint_err_rmse_rad": float(np.sqrt((jerr ** 2).mean())),
    }
    for i, f in enumerate(frames[:per_frame.shape[1]]):
        row[f"{f}_err_mean_m"] = float(per_frame[:, i].mean())
    for i, name in enumerate(("vx", "vy", "wz")[:nv]):
        row[f"vel_{name}_rmse"] = float(np.sqrt(((cmd_v[:, i] - ach_xyw[:, i]) ** 2).mean()))
    return row


def _family_of(meta: dict, terrain_id: int) -> str:
    """Family name for a terrain id, from the bundle's manifest copy."""
    if terrain_id < 0:
        return "flat"
    for tile in meta.get("terrain_manifest", []) or []:
        if int(tile.get("terrain_id", -2)) == terrain_id:
            return str(tile.get("family", tile.get("type", "?")))
    return "?"


def stage_plot(args: argparse.Namespace) -> int:
    """Compute metrics for every bundle, write summary.csv, and draw the figures."""
    rows, bundles = [], {}
    for policy, reference in _select_runs(args):
        for pick in range(_load_cases(args)["clip_picks"]):
            path = bundle_path(args.out_dir, policy, reference, pick)
            if not os.path.exists(path):
                print(f"[plot] missing {path}; skipping")
                continue
            b, meta = _load_bundle(path)
            bundles[(policy, reference, pick)] = (b, meta)
            for ep in range(int(meta["num_captured"])):
                row = episode_metrics(b, meta, ep)
                row.update(policy=policy, reference=reference, pick=pick, episode=ep,
                           family=_family_of(meta, row["terrain_id"]))
                rows.append(row)
    if not rows:
        raise SystemExit(f"[plot] no bundles in {args.out_dir}; run --stage rollout first.")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    cols = (["policy", "reference", "pick", "episode", "family", "terrain_id", "clip_id"]
            + [k for k in rows[0] if k not in
               ("policy", "reference", "pick", "episode", "family", "terrain_id", "clip_id")])
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[plot] wrote {csv_path} ({len(rows)} episodes)")

    _print_summary(rows)
    _figures(rows, bundles, args.out_dir)
    return 0


def _print_summary(rows: list[dict]) -> None:
    """Print a per-policy and per-family table to stdout."""
    keys = ("survived_s", "pelvis_err_mean_m", "mpjpe_mm", "joint_err_rmse_rad", "vel_vx_rmse")
    print("\n=== per (policy, reference) ===")
    print(f"{'policy/ref':22s} {'n':>3s} " + " ".join(f"{k:>20s}" for k in keys))
    combos = sorted({(r["policy"], r["reference"]) for r in rows})
    for policy, reference in combos:
        sub = [r for r in rows if r["policy"] == policy and r["reference"] == reference]
        vals = " ".join(f"{np.mean([r.get(k, np.nan) for r in sub]):>20.4f}" for k in keys)
        print(f"{policy + '/' + reference:22s} {len(sub):>3d} {vals}")

    print("\n=== per family (mean survived_s / mean pelvis err m) ===")
    fams = sorted({r["family"] for r in rows})
    head = " ".join(f"{p + '/' + r:>18s}" for p, r in combos)
    print(f"{'family':30s} {head}")
    for fam in fams:
        cells = []
        for policy, reference in combos:
            sub = [r for r in rows if r["family"] == fam and r["policy"] == policy
                   and r["reference"] == reference]
            cells.append(f"{np.mean([r['survived_s'] for r in sub]):8.2f}s/"
                         f"{np.mean([r['pelvis_err_mean_m'] for r in sub]):7.3f}" if sub
                         else f"{'-':>18s}")
        print(f"{fam:30s} " + " ".join(f"{c:>18s}" for c in cells))


def _figures(rows: list[dict], bundles: dict, out_dir: str) -> None:
    """Draw the comparison figures next to the summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combos = sorted({(r["policy"], r["reference"]) for r in rows})
    fams = sorted({r["family"] for r in rows})
    colors = {c: f"C{i}" for i, c in enumerate(combos)}

    # 1. per-family survival + tracking error bars
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 1.6 * len(fams)), 8.0), sharex=True)
    width = 0.8 / max(len(combos), 1)
    x = np.arange(len(fams))
    for j, combo in enumerate(combos):
        surv = [np.mean([r["survived_s"] for r in rows
                         if r["family"] == f and (r["policy"], r["reference"]) == combo] or [np.nan])
                for f in fams]
        err = [np.mean([r["pelvis_err_mean_m"] for r in rows
                        if r["family"] == f and (r["policy"], r["reference"]) == combo] or [np.nan])
               for f in fams]
        lbl = f"{combo[0]}/{combo[1]}"
        axes[0].bar(x + j * width, surv, width, label=lbl, color=colors[combo])
        axes[1].bar(x + j * width, err, width, label=lbl, color=colors[combo])
    axes[0].set_ylabel("survived [s]")
    axes[0].set_title("teacher vs SONIC: survival and pelvis tracking error, per terrain family")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].set_ylabel("mean pelvis |ref - actual| [m]")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_xticks(x + 0.4 - width / 2)
    axes[1].set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
    p = os.path.join(out_dir, "per_family.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {p}")

    # 2. per-episode time series: velocity tracking + reference error, both policies overlaid
    picks = sorted({k[2] for k in bundles})
    for pick in picks:
        avail = [(p, r) for (p, r, kk) in bundles if kk == pick]
        if not avail:
            continue
        any_meta = bundles[(avail[0][0], avail[0][1], pick)][1]
        n_ep = int(any_meta["num_captured"])
        for ep in range(n_ep):
            fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), sharex=True)
            fam = tid = None
            for combo in avail:
                b, meta = bundles[(combo[0], combo[1], pick)]
                if ep >= int(meta["num_captured"]):
                    continue
                dt = float(meta["dt"])
                done = b["done"][:, ep].astype(bool)
                end = int(np.argmax(done)) + 1 if done.any() else len(done)
                t = np.arange(end) * dt
                c = colors[combo]
                lbl = f"{combo[0]}/{combo[1]}"
                tid = int(b["terrain_id"][ep])
                fam = _family_of(meta, tid)
                axes[0].plot(t, b["cmd_vel"][:end, ep, 0], lw=1.0, ls=":", color=c, alpha=0.7)
                axes[0].plot(t, b["ach_vel"][:end, ep, 0], lw=1.4, color=c, label=lbl)
                err = np.linalg.norm(b["ref_pos"][:end, ep, 0] - b["body_pos"][:end, ep, 0],
                                     axis=-1)
                axes[1].plot(t, err, lw=1.4, color=c, label=lbl)
                axes[2].plot(t, b["root_pos"][:end, ep, 2], lw=1.4, color=c, label=lbl)
                axes[2].plot(t, b["ref_root_pos"][:end, ep, 2], lw=1.0, ls=":", color=c,
                             alpha=0.7)
            axes[0].set_ylabel("vx [m/s]")
            axes[0].set_title(f"episode {ep} (family {fam}, tile {tid}, pick {pick}) -- "
                              f"dotted = reference/command")
            axes[0].legend(fontsize=8, loc="upper right")
            axes[1].set_ylabel("pelvis |ref-act| [m]")
            axes[2].set_ylabel("root z [m]")
            axes[2].set_xlabel("time [s]")
            for a in axes:
                a.grid(alpha=0.3)
            p = os.path.join(out_dir, f"episode{ep:02d}_pick{pick}.png")
            fig.savefig(p, dpi=110, bbox_inches="tight")
            plt.close(fig)
    print(f"[plot] wrote per-episode figures to {out_dir}")


def _load_cases(args: argparse.Namespace) -> dict:
    """Read cases.json, falling back to the CLI defaults if prep was skipped."""
    path = os.path.join(args.out_dir, "cases.json")
    if os.path.exists(path):
        with open(path) as fh:
            cases = json.load(fh)
        cases.setdefault("ego_anchor_fraction", args.ego_anchor_fraction)
        if args.clip:
            cases["clip"] = os.path.abspath(args.clip)
        return cases
    if not args.clip:
        raise SystemExit(f"[compare] no {path} and no --clip; run --stage prep first.")
    return {"clip": os.path.abspath(args.clip), "families": args.families or "all",
            "per_family": args.per_family, "clip_picks": args.clip_picks, "steps": args.steps,
            "num_envs": args.num_envs, "seed": args.seed,
            "ego_anchor_fraction": args.ego_anchor_fraction}


def stage_viser(args: argparse.Namespace) -> int:
    """Hand off to the two-robot viser viewer."""
    cmd = [sys.executable, os.path.join(_HERE, "view_tracker_comparison.py"),
           "--out-dir", args.out_dir, "--port", str(args.port)]
    print("[viser] " + " ".join(cmd))
    return subprocess.call(cmd, cwd=_REPO)


def main() -> int:
    """Parse arguments and dispatch to the requested stage."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--stage", required=True, choices=("prep", "rollout", "plot", "viser"))
    p.add_argument("--out-dir", default="logs/tracker_compare")
    p.add_argument("--clip", default=None, help="Clip library .h5 (terrain mesh + clips).")
    p.add_argument("--clip_device", default="cpu", help="Hold the clip library here.")
    p.add_argument("--families", default=None,
                   help="Comma-separated terrain families, or 'all' (default).")
    p.add_argument("--per_family", type=int, default=2,
                   help="Tiles captured per family (the 'two clips of each terrain type').")
    p.add_argument("--clip_picks", type=int, default=1,
                   help="Distinct pinned clips per tile; each is a separate rollout pass.")
    p.add_argument("--steps", type=int, default=1000, help="Control steps per rollout.")
    p.add_argument("--num_envs", type=int, default=64,
                   help="Envs to simulate. Must be enough for the importer to instantiate a tile "
                        "of every requested family; only the captured ones are recorded.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ego_anchor_fraction", type=float, default=1.0,
                   help="Pinned for every run. The cfg default (0.5) re-places the clip onto the "
                        "robot root for a RANDOM half of the envs every ego_anchor_every steps, so "
                        "the two policies would be handed DIFFERENT references -- a larger effect "
                        "than the policy difference being measured (0.069 vs 0.247 m mean pelvis "
                        "error on the same clip and tile). 1.0 re-anchors EVERY env, which is "
                        "deterministic and is the regime a policy with no root-position target "
                        "can work in; note that pelvis error is then LOCAL tracking error, and "
                        "absolute progress must be read from the velocity-tracking columns. Use "
                        "0.0 for world-anchored absolute error.")
    p.add_argument("--policies", default="both", choices=("both", "teacher", "sonic"))
    p.add_argument("--references", default="both", choices=("both", "clip", "fm"))
    p.add_argument("--teacher_run", default=None, help="Teacher run dir (newest model_*.pt).")
    p.add_argument("--checkpoint", default=None, help="Explicit teacher checkpoint .pt.")
    p.add_argument("--fm_checkpoint", default=None, help="Flow-matching checkpoint (FM runs).")
    p.add_argument("--sonic_dir", default=None, help="SONIC release ONNX directory.")
    p.add_argument("--overwrite", action="store_true", help="Re-record existing bundles.")
    p.add_argument("--port", type=int, default=8080, help="viser port.")
    args = p.parse_args()

    return {"prep": stage_prep, "rollout": stage_rollout,
            "plot": stage_plot, "viser": stage_viser}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
