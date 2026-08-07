#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play the SAME clips in Isaac Lab and in MuJoCo, and compare the two aggregates.

The sim2sim question no single-clip run answers: does the tracker behave the same in both
simulators, over a clip set large enough to mean something? Both sides track a pre-made clip
with the generator OFF, so what is being measured is the policy plus the physics -- not the
FM. That makes it the right baseline to establish before any FM number is trusted.

**What makes the comparison sound.** Three things have to agree, and each is enforced here
rather than left to the caller:

1. **The same clips.** Isaac Lab otherwise samples ``clip_id`` uniformly at every reset, so
   two runs never see the same set (``mdp.events.resets`` used a bare ``torch.randint``).
   This pins env ``i`` to clip ``ids[i]`` via ``pin_clip_ids`` and passes the identical list
   to MuJoCo as ``--clip_id``.
2. **The same start state.** ``pin_start_step=0`` on the Isaac side, ``--clip_start_time 0``
   on the MuJoCo side, so both spawn on the clip's first frame.
3. **The same anchoring.** Clip tracking needs ``ego_anchor_fraction=1.0`` with
   ``ego_anchor_every=1``; MuJoCo's ``--clip_ego_anchor_every`` defaults to 10. Mismatching
   them means the two sims track differently anchored references, which reads as a physics
   gap. Both are pinned to 1 below and the run manifest records it.

**What is compared.** Only quantities both sides compute the same way:

* ``success`` -- the episode ended without a fall. Isaac: ended via ``time_out`` (the tracker
  task has no clip-graded terminations, so a fall is the only way out). MuJoCo:
  ``survival.survived``.
* ``vel_rmse_{vx,vy,wz}`` -- RMS of measured base velocity against the clip's commanded
  ``target_vel``. Isaac: ``sqrt(vel_track_*_mse)`` per env. MuJoCo:
  ``command_velocity_rms.*``.
* ``orient_rmse`` / ``joint_rmse`` -- base orientation and joint-position error against the
  reference. Isaac: the per-env group means of the tracked set. MuJoCo:
  ``reference_tracking.{roll,pitch,yaw}`` and ``joint_tracking.rms.*``, reduced the same way.

**Base position is deliberately NOT compared.** Matched ego anchoring re-anchors the
reference onto the robot's root every step, which zeroes base position error by construction
-- measured 1e-17 m. A column of zeros would read as "both sims track position perfectly"
when the quantity is not being tested at all. Orientation and joints survive the re-anchor.

Anything each simulator reports in its own units and nothing else does (Isaac's Lyapunov
``V``, MuJoCo's torque utilisation) is also left out of the comparison table.

**Aggregate, not paired.** This reports the DISTRIBUTION per simulator, side by side. Both
sides are per-clip underneath -- ``tracking_per_env.csv`` and one
``metrics_summary.json`` per clip -- so pairing clip-by-clip is a later reduction over the
same artifacts, not another set of runs.

Stages (each writes what the next reads, so a slow half can be re-run alone)::

    # everything, 100 clips
    python scripts/rsl_rl/clip_ab_sweep.py --clips 100

    # just one side
    python scripts/rsl_rl/clip_ab_sweep.py --stage isaac  --out logs/clip_ab/<run>
    python scripts/rsl_rl/clip_ab_sweep.py --stage mujoco --out logs/clip_ab/<run> --jobs 6

    # re-aggregate + re-plot without re-running anything (IsaacLab-free)
    python scripts/rsl_rl/clip_ab_sweep.py --stage report --out logs/clip_ab/<run>

    # print the resolved commands and exit
    python scripts/rsl_rl/clip_ab_sweep.py --clips 4 --dry_run

Runtime for 100 clips: Isaac ~2 min (one pinned run, all clips at once), MuJoCo ~19 min
serial or ~4 min at ``--jobs 6`` (clip playback has no generator, so it is ~11 s per 20 s
clip). ``--stage report`` needs neither simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
PYTHON = sys.executable

#: The tracker task. Deliberately the TRACKER and not the teacher: the teacher env
#: terminates on clip-graded tracking error, which MuJoCo has no equivalent for, so
#: "success" would mean different things in the two sims. The tracker env terminates only on
#: a fall, which is exactly what MuJoCo detects.
TASK = "G1-tracker-clf-fm-terrain-depth-play"

#: Defaults matching the current assets; every one is overridable.
DEFAULT_LIB = "clips/g1_mb_v5_flat_body.h5"
DEFAULT_TEACHER_RUN = "2026-08-01_08-49-20_depth_v5_refslim"
DEFAULT_CHECKPOINT = "model_12000.pt"
DEFAULT_MJ_RUN = "xl_k12_pinned"

def _dotted(d: dict, path: str):
    """Fetch ``a.b.c`` out of nested dicts, or ``None``."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _rms(vals) -> float:
    """RMS over a list of per-axis / per-joint RMS values."""
    v = [float(x) for x in vals if x is not None]
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")


#: Isaac -> MuJoCo metric mapping. ``isaac`` keys are columns of ``tracking_per_env.csv``
#: (mean SQUARED error, so square-rooted here); ``mujoco`` keys are either a dotted path into
#: ``metrics_summary.json`` (already RMS) or a callable reducing that dict to one number.
#:
#: BASE POSITION IS DELIBERATELY ABSENT. Clip tracking re-anchors the reference onto the
#: robot's root every step (ego_anchor_every=1, matched on both sides), which zeroes the base
#: position error by construction -- measured 1e-17 m in MuJoCo. Including it would put a
#: column of zeros in the table and read as "both sims track position perfectly", when in
#: fact the quantity is not being tested at all. Orientation and joints survive the
#: re-anchor, so they are what discriminates.
COMPARED: tuple[tuple[str, str, object], ...] = (
    ("vel_rmse_vx", "vel_track_forward_vx_mse", "command_velocity_rms.vx"),
    ("vel_rmse_vy", "vel_track_lateral_vy_mse", "command_velocity_rms.vy"),
    ("vel_rmse_wz", "vel_track_yaw_wz_mse", "command_velocity_rms.wz"),
    ("orient_rmse", "pos_error_orientations",
     lambda d: _rms([_dotted(d, f"reference_tracking.{a}") for a in ("roll", "pitch", "yaw")])),
    ("joint_rmse", "pos_error_joints",
     lambda d: _rms(list((_dotted(d, "joint_tracking.rms") or {}).values()))),
)


def choose_clips(lib: pathlib.Path, n: int, seed: int, min_len: int) -> list[int]:
    """Pick ``n`` clip indices from ``lib``, deterministically.

    Sorted so the list reads the same however it was drawn, and filtered to clips at least
    ``min_len`` steps long -- a shorter clip runs out mid-episode and the tail is padding, so
    both sims would be scored on a frozen reference.

    Args:
        lib: Clip library ``.h5``.
        n: How many clips to pick.
        seed: RNG seed for the draw.
        min_len: Reject clips shorter than this many steps.

    Returns:
        Sorted clip indices.

    Raises:
        ValueError: If fewer than ``n`` clips clear ``min_len``.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(lib, "r") as f:
        clip_len = np.asarray(f["clip_len"][:]).reshape(-1)
    ok = np.nonzero(clip_len >= min_len)[0]
    if ok.size < n:
        raise ValueError(
            f"{lib.name}: only {ok.size} of {clip_len.size} clips are >= {min_len} steps, "
            f"need {n}. Lower --clips, or --min_clip_len (longest clip is {clip_len.max()})."
        )
    rng = np.random.default_rng(seed)
    return sorted(int(c) for c in rng.choice(ok, size=n, replace=False))


def isaac_argv(args, ids_path: pathlib.Path, num_envs: int) -> list[str]:
    """The ``play.py`` argv for the pinned clip-tracking rollout.

    NO ``--fm_checkpoint`` by default. With ``reference_source=clip`` and an empty
    ``fm_checkpoint`` the generator is not loaded AT ALL, which matters twice over: MuJoCo's
    clip-only mode does not load ``fm.onnx`` either, so loading it here would make the two
    sims do different work; and the FM's sampling consumes RNG, so its mere presence
    perturbs the observation-noise draws. Passing one anyway (``--fm_checkpoint``) runs the
    27M generator every step only to discard its output -- measured at several minutes of
    pure overhead for 100 envs.

    Args:
        args: Parsed CLI namespace.
        ids_path: JSON file holding the pinned clip ids.
        num_envs: One env per clip.

    Returns:
        The argv list.
    """
    argv = [
        PYTHON, str(REPO / "scripts" / "rsl_rl" / "play.py"),
        "--task", TASK,
        "--load_run", args.teacher_run,
        "--checkpoint", str(REPO / "logs" / "rsl_rl" / "g1" / args.teacher_run / args.checkpoint),
        "--clip", args.lib, "--clip_device", "cpu",
        "--num_envs", str(num_envs), "--max_steps", str(args.max_steps),
        "--seed", str(args.seed), "--headless",
    ]
    if args.fm_checkpoint:
        argv += ["--fm_checkpoint", str(REPO / args.fm_checkpoint)]
    return argv + [
        # The clip drives the tracker; with no checkpoint above, no generator is built.
        "env.commands.traj_ref.reference_source=clip",
        # Clip tracking is only in-distribution when the clip is re-anchored on the robot
        # every step -- see the module docstring.
        "env.commands.traj_ref.ego_anchor_fraction=1.0",
        "env.commands.traj_ref.ego_anchor_every=1",
        # The whole point: state the clips instead of sampling them.
        f"env.events.reset_on_clip.params.pin_clip_ids={ids_path}",
        "env.events.reset_on_clip.params.pin_start_step=0",
    ]


def mujoco_argv(args, clip_id: int, log_dir: pathlib.Path) -> list[str]:
    """The ``gen_tracker_runner.py`` argv for one clip.

    No ``--clip_handoff_time``, so this is clip-ONLY playback: the runner does not even load
    ``fm.onnx``, which is what makes it fast enough to sweep.

    Args:
        args: Parsed CLI namespace.
        clip_id: Clip index to play.
        log_dir: Where this run writes its CSV + metrics (unique per clip, since the
            runner's default log path is shared and concurrent runs would collide).

    Returns:
        The argv list.
    """
    return [
        PYTHON, "gen_tracker_runner.py",
        "--load_run", args.mj_run,
        "--robot_name", args.robot, "--scene", args.scene,
        "--clip", args.lib, "--clip_id", str(clip_id),
        # Matches the Isaac side's ego_anchor_every=1.
        "--clip_ego_anchor", "--clip_ego_anchor_every", "1",
        "--clip_start_time", "0.0",
        "--total_time", f"{args.clip_seconds:.4f}",
        "--headless", "--no_viz", "--num_threads", str(args.mj_threads),
        "--log", "--log_dir", str(log_dir),
    ]


def stage_isaac(args, ids: list[int], out: pathlib.Path) -> int:
    """Run the pinned Isaac rollout and copy its artifacts out of the shared plot dir."""
    ids_path = out / "clip_ids.json"
    argv = isaac_argv(args, ids_path, len(ids))
    print("[isaac] " + " ".join(argv))
    if args.dry_run:
        return 0
    log_path = out / "isaac.log"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(REPO / "source" / "robot_rl"), str(REPO / "source" / "motion_gen"),
         os.environ.get("PYTHONPATH", "")])}
    started = time.time()
    with open(log_path, "w") as f:
        rc = subprocess.call(argv, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env)
    # play.py writes into a per-task plot dir that EVERY run against this task overwrites,
    # so copy immediately rather than reading it later -- AND only take files this run
    # actually wrote. A run that dies during startup leaves the previous run's artifacts
    # sitting there, and copying those silently reports someone else's numbers as ours
    # (observed: a 200-env sweep's CSV picked up by a failed 4-env run).
    src = REPO / "logs" / "rsl_rl" / "g1" / args.teacher_run / "plots" / TASK
    got, stale = [], []
    for name in ("stats.txt", "tracking_stats.csv", "tracking_per_env.csv"):
        p = src / name
        if not p.is_file():
            continue
        if p.stat().st_mtime < started:
            stale.append(name)
            continue
        shutil.copyfile(p, out / f"isaac_{name}")
        got.append(name)
    print(f"[isaac] rc={rc}, copied {got or 'NOTHING'} from {src}")
    if stale:
        print(f"[isaac] IGNORED {stale} -- older than this run, i.e. left by a previous one.")
    if rc != 0:
        print(f"[isaac] FAILED -- see {log_path}")
    elif "tracking_per_env.csv" not in got:
        print("[isaac] WARNING: no fresh tracking_per_env.csv, so the Isaac distribution "
              "will be empty. Either play_plots did not run, or it predates the per-env "
              "columns.")
    return rc


def stage_mujoco(args, ids: list[int], out: pathlib.Path) -> int:
    """Run every clip through MuJoCo, up to ``--jobs`` at a time."""
    sim_dir = REPO / "transfer" / "sim"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(REPO / "source" / "robot_rl"), str(REPO / "source" / "motion_gen"),
         str(REPO), os.environ.get("PYTHONPATH", "")])}
    if args.dry_run:
        print("[mujoco] " + " ".join(mujoco_argv(args, ids[0], out / "mj" / f"clip_{ids[0]:05d}")))
        print(f"[mujoco] ... and {len(ids) - 1} more like it")
        return 0
    running: list[tuple[int, subprocess.Popen, object]] = []
    failed, done, t0 = [], 0, time.time()

    def reap_finished() -> int:
        """Harvest every job that has exited; returns how many were collected."""
        nonlocal done
        harvested = 0
        for entry in list(running):
            cid, proc, fh = entry
            if proc.poll() is None:
                continue
            fh.close()
            running.remove(entry)
            done += 1
            harvested += 1
            if proc.returncode != 0:
                failed.append(cid)
            if done % max(1, len(ids) // 10) == 0 or done == len(ids):
                print(f"[mujoco] {done}/{len(ids)} clips, {time.time() - t0:.0f}s elapsed, "
                      f"{len(failed)} failed")
        return harvested

    def wait_for_slot(limit: int) -> None:
        """Block until fewer than ``limit`` jobs are running."""
        while len(running) >= limit:
            if not reap_finished():
                time.sleep(0.2)

    for cid in ids:
        wait_for_slot(args.jobs)
        log_dir = out / "mj" / f"clip_{cid:05d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = open(log_dir / "run.log", "w")
        running.append((cid, subprocess.Popen(
            mujoco_argv(args, cid, log_dir), cwd=sim_dir, stdout=fh,
            stderr=subprocess.STDOUT, env=env), fh))
    wait_for_slot(1)
    print(f"[mujoco] finished {done}/{len(ids)} in {time.time() - t0:.0f}s; "
          f"{len(failed)} failed{': ' + str(failed[:10]) if failed else ''}")
    return 1 if failed else 0


def collect_isaac(out: pathlib.Path, ids: list[int]) -> dict[str, np.ndarray]:
    """Per-clip Isaac metrics from the copied ``tracking_per_env.csv``.

    Env ``i`` was pinned to ``ids[i]``, so row ``i`` IS clip ``ids[i]``.
    """
    path = out / "isaac_tracking_per_env.csv"
    if not path.is_file():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    grab = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan
                               for r in rows])
    cols: dict[str, np.ndarray] = {}
    if rows and "survived" in rows[0]:
        cols["success"] = grab("survived")
    for name, isaac_key, _mj in COMPARED:
        if rows and isaac_key in rows[0]:
            cols[name] = np.sqrt(grab(isaac_key))      # MSE -> RMSE
    n = min([len(ids)] + [v.shape[0] for v in cols.values()]) if cols else 0
    cols = {k: v[:n] for k, v in cols.items()}
    cols["clip_id"] = np.asarray(ids[:n], dtype=float)
    return cols


def collect_mujoco(out: pathlib.Path, ids: list[int]) -> dict[str, np.ndarray]:
    """Per-clip MuJoCo metrics from each run's ``metrics_summary.json``."""
    keep, rows = [], []
    for cid in ids:
        p = out / "mj" / f"clip_{cid:05d}" / "metrics_summary.json"
        if not p.is_file():
            continue
        try:
            rows.append(json.loads(p.read_text()))
            keep.append(cid)
        except json.JSONDecodeError:
            continue
    if not rows:
        return {}
    cols: dict[str, np.ndarray] = {"clip_id": np.asarray(keep, dtype=float)}
    surv = [_dotted(r, "survival.survived") for r in rows]
    cols["success"] = np.array([np.nan if s is None else float(bool(s)) for s in surv])
    for name, _isaac, mj_key in COMPARED:
        pick = mj_key if callable(mj_key) else (lambda r, k=mj_key: _dotted(r, k))
        vals = [pick(r) for r in rows]
        cols[name] = np.array([np.nan if v is None else float(v) for v in vals])
    return cols


def _summarise(col: np.ndarray) -> dict:
    """Median / mean / IQR of a metric, ignoring nan."""
    v = col[np.isfinite(col)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max())}


def stage_report(args, ids: list[int], out: pathlib.Path) -> int:
    """Aggregate both sides, print the table, write JSON + a distribution figure."""
    isaac, mj = collect_isaac(out, ids), collect_mujoco(out, ids)
    if not isaac and not mj:
        print(f"[report] nothing to aggregate under {out}. Run --stage isaac / mujoco first.")
        return 1
    summary = {"clips_requested": len(ids), "task": TASK,
               "isaac": {}, "mujoco": {}, "library": args.lib}
    for side, cols in (("isaac", isaac), ("mujoco", mj)):
        if not cols:
            continue
        s = cols.get("success")
        if s is not None:
            fin = s[np.isfinite(s)]
            summary[side]["success"] = {
                "n": int(fin.size), "up": int(fin.sum()),
                "rate": float(fin.mean()) if fin.size else float("nan")}
        for name, _i, _m in COMPARED:
            if name in cols:
                summary[side][name] = _summarise(cols[name])

    w = 22
    print("\n" + "=" * 78)
    print(f"  Isaac Lab vs MuJoCo, {len(ids)} clips, generator OFF (clip tracking)")
    print("=" * 78)
    for side in ("isaac", "mujoco"):
        d = summary.get(side, {}).get("success")
        if d:
            print(f"  {side:<8s} success {d['up']:4d}/{d['n']:<4d} = {100 * d['rate']:5.1f}%")
        else:
            print(f"  {side:<8s} success        (no data)")
    print("-" * 78)
    print(f"  {'metric':<{w}s} {'isaac median':>14s} {'mujoco median':>15s} "
          f"{'isaac IQR':>18s}")
    for name, _i, _m in COMPARED:
        a, b = summary.get("isaac", {}).get(name), summary.get("mujoco", {}).get(name)
        f = lambda d, k: f"{d[k]:.4f}" if d and d.get("n") else "-"
        iqr = (f"{a['p25']:.4f}-{a['p75']:.4f}" if a and a.get("n") else "-")
        print(f"  {name:<{w}s} {f(a, 'median'):>14s} {f(b, 'median'):>15s} {iqr:>18s}")
    print("=" * 78)
    print("  Aggregate distributions per simulator, NOT paired per clip. Both sides are\n"
          "  per-clip underneath, so pairing is a later reduction over the same files.")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_per_clip(out, isaac, mj)
    fig = _plot(out, isaac, mj, len(ids))
    print(f"\n[report] wrote {out / 'summary.json'}")
    print(f"[report] wrote {out / 'per_clip.csv'}")
    if fig:
        print(f"[report] wrote {fig}")
    return 0


def _write_per_clip(out: pathlib.Path, isaac: dict, mj: dict) -> None:
    """Long-form per-clip rows for both sims, so pairing later needs no re-run."""
    names = ["success"] + [n for n, _i, _m in COMPARED]
    with open(out / "per_clip.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sim", "clip_id", *names])
        for side, cols in (("isaac", isaac), ("mujoco", mj)):
            if not cols:
                continue
            cid = cols["clip_id"]
            for i in range(cid.shape[0]):
                wr.writerow([side, int(cid[i])] + [
                    (f"{cols[n][i]:.6f}" if n in cols and np.isfinite(cols[n][i]) else "")
                    for n in names])


def _plot(out: pathlib.Path, isaac: dict, mj: dict, n_clips: int):
    """Success bars + one distribution panel per compared metric."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[report] matplotlib unavailable; skipping the figure.")
        return None
    metrics = [n for n, _i, _m in COMPARED
               if (n in isaac and np.isfinite(isaac[n]).any())
               or (n in mj and np.isfinite(mj[n]).any())]
    ncol = 1 + len(metrics)
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.4), squeeze=False)
    ax = axes[0][0]
    rates, labels, colors = [], [], []
    for side, cols, c in (("isaac", isaac, "#3b6ea5"), ("mujoco", mj, "#c8641e")):
        s = cols.get("success")
        if s is None:
            continue
        fin = s[np.isfinite(s)]
        rates.append(100.0 * fin.mean() if fin.size else 0.0)
        labels.append(f"{side}\n{int(fin.sum())}/{fin.size}")
        colors.append(c)
    ax.bar(range(len(rates)), rates, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("success rate (%)")
    ax.set_title(f"survived, {n_clips} clips")
    ax.grid(axis="y", alpha=0.3)

    for j, name in enumerate(metrics):
        ax = axes[0][j + 1]
        data, ticks = [], []
        for side, cols in (("isaac", isaac), ("mujoco", mj)):
            v = cols.get(name)
            v = v[np.isfinite(v)] if v is not None else np.array([])
            if v.size:
                data.append(v)
                ticks.append(side)
        if data:
            parts = ax.violinplot(data, showmedians=True, showextrema=False)
            for k, body in enumerate(parts["bodies"]):
                body.set_facecolor("#3b6ea5" if ticks[k] == "isaac" else "#c8641e")
                body.set_alpha(0.65)
            ax.set_xticks(range(1, len(ticks) + 1))
            ax.set_xticklabels(ticks)
        ax.set_title(name, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Isaac Lab vs MuJoCo, clip tracking (generator off)", fontsize=11)
    fig.tight_layout()
    path = out / "clip_ab.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=["all", "isaac", "mujoco", "report"])
    p.add_argument("--out", default=None,
                   help="Run directory (default logs/clip_ab/<timestamp>). Reused by "
                        "--stage, which is how a half can be re-run alone.")
    p.add_argument("--clips", type=int, default=100, help="How many clips to play.")
    p.add_argument("--lib", default=DEFAULT_LIB, help="Clip library (repo-relative).")
    p.add_argument("--seed", type=int, default=42, help="Clip draw + rollout seed.")
    p.add_argument("--min_clip_len", type=int, default=990,
                   help="Reject shorter clips: the tail is repeat-last padding, so both sims "
                        "would be scored against a frozen reference.")
    p.add_argument("--max_steps", type=int, default=1100,
                   help="Isaac rollout length; must EXCEED the 1000-step episode or no "
                        "episode completes and success reads 0 whatever the quality.")
    p.add_argument("--clip_seconds", type=float, default=19.96,
                   help="MuJoCo run length. Defaults just under the 999-step clip so the "
                        "reference never runs into its padding.")
    p.add_argument("--teacher_run", default=DEFAULT_TEACHER_RUN, help="Isaac policy run dir.")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Policy .pt in that run.")
    p.add_argument("--fm_checkpoint", default="",
                   help="Usually leave EMPTY. With reference_source=clip an empty checkpoint "
                        "means no generator is built at all, which is what MuJoCo's "
                        "clip-only mode does too. Passing one runs the generator every step "
                        "purely to discard it, and its RNG draws perturb the obs noise.")
    p.add_argument("--mj_run", default=DEFAULT_MJ_RUN,
                   help="Run dir holding exported/tracker.onnx for MuJoCo.")
    p.add_argument("--robot", default="g1_29j")
    p.add_argument("--scene", default="basic_scene")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 3),
                   help="Concurrent MuJoCo runs. Each uses --mj_threads ORT threads.")
    p.add_argument("--mj_threads", type=int, default=2,
                   help="onnxruntime threads per MuJoCo run; keep jobs*threads under the "
                        "core count or the runs fight each other.")
    p.add_argument("--dry_run", action="store_true", help="Print commands and exit.")
    args = p.parse_args()

    # ABSOLUTE, always. The MuJoCo runner is launched with cwd=transfer/sim, so a relative
    # --out would make its --log_dir resolve under THAT directory: the driver would create
    # the per-clip dirs here, the runner would write its CSV somewhere else entirely, and the
    # report would find 100 empty directories and call it "no data" -- with every run having
    # exited 0. Observed exactly once, which is once more than enough.
    out = (pathlib.Path(args.out).resolve() if args.out else
           REPO / "logs" / "clip_ab" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "clip_ids.json"

    if ids_path.is_file():
        ids = json.loads(ids_path.read_text())
        print(f"[setup] reusing {len(ids)} pinned clips from {ids_path}")
    else:
        ids = choose_clips(REPO / args.lib, args.clips, args.seed, args.min_clip_len)
        ids_path.write_text(json.dumps(ids))
        print(f"[setup] drew {len(ids)} clips (seed {args.seed}) -> {ids_path}")
    # The manifest is the record of what made the two runs comparable.
    (out / "manifest.json").write_text(json.dumps({
        "task": TASK, "library": args.lib, "clips": ids, "seed": args.seed,
        "teacher_run": args.teacher_run, "checkpoint": args.checkpoint,
        "fm_checkpoint": args.fm_checkpoint, "mj_run": args.mj_run,
        "robot": args.robot, "scene": args.scene,
        "isaac_max_steps": args.max_steps, "mujoco_seconds": args.clip_seconds,
        "matched": {"start_step": 0, "ego_anchor_every": 1, "ego_anchor_fraction": 1.0,
                    "reference_source": "clip", "generator": "off"},
    }, indent=2))

    rc = 0
    if args.stage in ("all", "isaac"):
        rc |= stage_isaac(args, ids, out)
    if args.stage in ("all", "mujoco"):
        rc |= stage_mujoco(args, ids, out)
    if args.stage in ("all", "report") and not args.dry_run:
        rc |= stage_report(args, ids, out)
    print(f"\n[done] {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
