# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reduce the two-box rollouts to the initial-condition grid they were collected on.

One row per ``(approach distance, commanded speed)`` cell, which is the shape the sweep was
designed in: distance is a TILE (``terrain_id`` indexes
:data:`motion_gen.terrain.box2.TRAIN_SPAWN_DISTANCES_M`) and speed is a per-run command, so a
cell is recoverable from the episode table plus the run it came from.

**Success here is ``reached_goal``, and it is not the same question as "did it survive".**  An
episode can end upright, on the far side of nothing, having timed out; and a fall can happen
after crossing the goal line.  Both are reported, because on this obstacle the interesting
failure is neither -- it is the robot committing to the wrong MODE at the second box.

**Timeout fraction is a DIAGNOSTIC on the harness, not on the policy.**  The episode budget has
to be non-binding, or the slow x far cells are scored on the clock: at the 20 s default, tile 4
at 0.50 m/s needed 20.3 s at PERFECT tracking and scored 0.011 against 0.075 once the budget
moved to 40 s, with its timeout fraction going 0.172 -> 0.004.  So the column is printed beside
every rate, and a cell above :data:`TIMEOUT_WARN` is called out rather than left to be noticed.

Usage::

    python scripts/rsl_rl/box2_ic_report.py logs/box2/rollouts
    python scripts/rsl_rl/box2_ic_report.py logs/box2/rollouts --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys

import h5py
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from motion_gen.terrain.box2 import (  # noqa: E402
    HOLDOUT_SPAWN_DISTANCE_M,
    TRAIN_SPAWN_DISTANCES_M,
)

#: A cell whose episodes time out more often than this is being scored partly on the clock, and
#: the rate below it is not a statement about the policy.  0.05 is well above the 0.004-0.015
#: a non-binding budget leaves and well below the 0.172 the binding one produced.
TIMEOUT_WARN: float = 0.05

#: ``train_v1.17`` / ``holdout_v1.00`` -> the commanded speed.
_SPEED_RE = re.compile(r"_v(?P<speed>[0-9.]+)$")


def run_speed(name: str) -> float | None:
    """The commanded speed a run directory's name declares.

    Args:
        name: The run directory's basename.

    Returns:
        The speed in m/s, or ``None`` when the name does not carry one.
    """
    m = _SPEED_RE.search(name)
    return float(m.group("speed")) if m else None


def load_cells(root: str) -> list[dict]:
    """One record per ``(run, tile)`` cell under ``root``.

    Args:
        root: Directory holding one subdirectory per recording run.

    Returns:
        Cell dicts, sorted by (holdout last, speed, distance).

    Raises:
        SystemExit: If no episode table is found -- an empty report reads exactly like a
            finished sweep with nothing in it.
    """
    cells: list[dict] = []
    for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
        tables = glob.glob(os.path.join(run_dir, "*.episodes.h5"))
        if not tables:
            continue
        name = os.path.basename(run_dir)
        holdout = name.startswith("holdout")
        dists = ((HOLDOUT_SPAWN_DISTANCE_M,) if holdout else TRAIN_SPAWN_DISTANCES_M)
        with h5py.File(tables[0], "r") as h:
            keep = ~h["censored"][:].astype(bool)
            tid = h["terrain_id"][:][keep]
            ok = h["reached_goal"][:][keep].astype(bool)
            trunc = h["truncated"][:][keep].astype(bool)
            # The FAILURE ending is `terminated`, not the accumulator's derived fall flag:
            # measured over 5000 episodes, `terminated` is 0.869 and every one of those has
            # the tilt and/or absolute-height term latched (`terminated & neither` = 0.000),
            # while `first_fall_step >= 0` is a stricter 0.237. Using the latter left the three
            # categories summing to ~0.32 and the rest unaccounted for.
            failed = h["terminated"][:][keep].astype(bool)
            steps = h["steps_survived"][:][keep]
        for t in sorted(np.unique(tid)):
            if t < 0:
                continue
            m = tid == t
            cells.append({
                "run": name, "holdout": holdout, "speed": run_speed(name),
                "tile": int(t),
                "distance_m": dists[int(t)] if int(t) < len(dists) else None,
                "n": int(m.sum()),
                "reached_goal": float(ok[m].mean()),
                "timeout": float((trunc[m] & ~ok[m]).mean()),
                "failed": float(failed[m].mean()),
                "steps_median": float(np.median(steps[m])),
            })
    if not cells:
        raise SystemExit(f"no *.episodes.h5 under {root!r}. An empty report is "
                         "indistinguishable from a finished sweep, so this refuses.")
    return sorted(cells, key=lambda c: (c["holdout"], c["speed"] or 0.0, c["distance_m"] or 0.0))


def format_report(cells: list[dict]) -> str:
    """The grid, as a table plus its marginals.

    Args:
        cells: :func:`load_cells` output.

    Returns:
        The report text.
    """
    train = [c for c in cells if not c["holdout"]]
    hold = [c for c in cells if c["holdout"]]
    speeds = sorted({c["speed"] for c in train if c["speed"] is not None})
    dists = sorted({c["distance_m"] for c in train if c["distance_m"] is not None})
    by = {(c["speed"], c["distance_m"]): c for c in train}

    L = ["=" * 78, "TWO-BOX INITIAL-CONDITION GRID -- reached_goal", "=" * 78, ""]
    L.append("  approach distance (m) across, commanded speed (m/s) down")
    L.append("  " + " " * 8 + "".join(f"{d:>9.2f}" for d in dists) + f"{'row':>11}")
    for s in speeds:
        row = [by.get((s, d)) for d in dists]
        got = [c for c in row if c]
        rate = (sum(c["reached_goal"] * c["n"] for c in got) / sum(c["n"] for c in got)
                if got else float("nan"))
        L.append(f"  {s:>6.2f}  " + "".join(f"{c['reached_goal']:>9.3f}" if c else f"{'-':>9}"
                                            for c in row) + f"{rate:>11.3f}")
    col = []
    for d in dists:
        got = [by[(s, d)] for s in speeds if (s, d) in by]
        col.append(sum(c["reached_goal"] * c["n"] for c in got) / sum(c["n"] for c in got)
                   if got else float("nan"))
    L.append(f"  {'col':>6}  " + "".join(f"{v:>9.3f}" for v in col))
    n_tr = sum(c["n"] for c in train)
    pooled = sum(c["reached_goal"] * c["n"] for c in train) / n_tr if n_tr else float("nan")
    L.append(f"\n  pooled over {n_tr} training episodes: {pooled:.4f}")

    if hold:
        n_h = sum(c["n"] for c in hold)
        ph = sum(c["reached_goal"] * c["n"] for c in hold) / n_h
        d = hold[0]["distance_m"]
        L.append(f"  HELD OUT ({d} m, {hold[0]['speed']} m/s), {n_h} episodes: {ph:.4f}"
                 "   -- off the training grid on BOTH axes")

    L += ["", "-- per cell " + "-" * 65,
          f"  {'run':<16}{'tile':>5}{'dist':>7}{'n':>7}{'reached':>9}{'failed':>8}"
          f"{'timeout':>9}{'sum':>7}{'steps':>8}"]
    for c in cells:
        flag = "  <- TIMEOUT-LIMITED" if c["timeout"] > TIMEOUT_WARN else ""
        # reached + failed + timeout must account for the episode; printed so a category
        # that stops adding up is visible in the table rather than inferred from it.
        total = c["reached_goal"] + c["failed"] + c["timeout"]
        L.append(f"  {c['run']:<16}{c['tile']:>5}{c['distance_m'] or 0:>7.2f}{c['n']:>7}"
                 f"{c['reached_goal']:>9.3f}{c['failed']:>8.3f}{c['timeout']:>9.3f}"
                 f"{total:>7.3f}{c['steps_median']:>8.0f}{flag}")
    hot = [c for c in cells if c["timeout"] > TIMEOUT_WARN]
    if hot:
        L.append(f"\n  !! {len(hot)} cell(s) time out above {TIMEOUT_WARN:.0%}: their rates are "
                 "partly the episode budget, not the policy. Raise env.episode_length_s and "
                 "re-record those cells.")
    else:
        L.append(f"\n  every cell times out below {TIMEOUT_WARN:.0%}, so the budget is "
                 "non-binding and these rates are about the policy.")
    L.append("=" * 78)
    return "\n".join(L)


def main() -> int:
    """Print the grid, and optionally write it as JSON."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="directory holding the per-run rollout subdirectories")
    p.add_argument("--json", default="", help="also write the cells here")
    a = p.parse_args()
    cells = load_cells(a.root)
    print(format_report(cells))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(cells, f, indent=2)
        print(f"\n[box2] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
