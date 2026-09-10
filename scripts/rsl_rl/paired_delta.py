# Copyright 2026 Zachary Olkin. All rights reserved.
"""The PAIRED delta, which is the stronger statement the figure's level intervals cannot make.

``record_rollouts --seed`` defaults to the agent config's fixed seed, so both arms draw the same
reset stream -- and every env's first episode was measured to open at a bit-identical downrange
in both arms (max |diff| 0.00e+00).  So the design is paired at the ENV level, not merely at the
cell level, and the uncertainty on the DIFFERENCE is much smaller than the two level intervals
suggest: whatever the DR draw and the spawn jitter did to env i, they did to both arms.

Reported three ways, because they answer different questions:

* the unpaired cell bootstrap on each LEVEL -- what the bar chart's error bars are;
* the paired cluster bootstrap on the DELTA -- resample cells, then envs within them,
  differencing within env;
* exact McNemar over discordant env pairs -- distribution-free, and the count of discordant
  pairs is the honest sample size for a paired binary comparison.

The episode selection and the success RULE come from the same places the figure and the report
use (:func:`awr_buffer.first_per_env`, :func:`rollout_outcomes.terminal_reward`), so all three
readouts describe one recording the same way.

Usage::

    python scripts/rsl_rl/paired_delta.py --out paired.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
from math import exp, lgamma, log

import h5py
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_p = str(_REPO_ROOT / "scripts" / "flow_matching")
if _p not in sys.path:
    sys.path.insert(0, _p)

import rollout_outcomes as ro  # noqa: E402
from awr_buffer import first_per_env  # noqa: E402

#: Bootstrap resamples.
N_BOOT: int = 20000

#: The paper figure's clusters, matching ``fig_reduce.DEFAULT_ARMS``.
CLUSTERS: tuple[tuple[str, str, str, range | None], ...] = (
    ("Two boxes (OOD)", "eval_parent_prog_s3", "eval_prog_s3_i5", None),
    ("Stairs up-into-down (OOD)", "eval_parent_updn_s3", "eval_updn_s3_i5", None),
    ("Stairs down (in-distribution)", "eval_parent_indist", "eval_indist_i5", range(20, 40)),
)


def load(root: str, tag: str, tiles: range | None = None, *,
         metric: str = "terminal_reward", strict: bool = True) -> dict:
    """``{(tile, speed): {env: success}}`` over each env's FIRST episode.

    Args:
        root: Directory holding the eval run directories.
        tag: The run directory under ``root``.
        tiles: Restrict to these ``terrain_id`` values.
        metric: Success rule; ``terminal_reward`` or ``reached_goal``.
        strict: Count a tracking violation as a failure as well as a fall.

    Returns:
        Per cell, the per-env outcome of that env's first completed episode.
    """
    out: dict = {}
    for d in sorted(glob.glob(os.path.join(root, tag, "train_v*"))):
        if not os.path.isdir(d):
            continue
        speed = float(d.rsplit("train_v", 1)[1])
        f = glob.glob(d + "/*.episodes.h5")
        if not f:
            continue
        with h5py.File(f[0], "r") as h:
            keep = ~h["censored"][:].astype(bool)
            uid_all = h["episode_uid"][:].astype(np.int64)
            env_all = h["env_id"][:].astype(np.int64)
            keep &= np.isin(uid_all, np.asarray(
                first_per_env(uid_all[keep], env_all[keep], 1), dtype=np.int64))
            cols = {k: h[k][:][keep] for k in h if isinstance(h[k], h5py.Dataset)}
        ok = (cols["reached_goal"].astype(bool) if metric == "reached_goal"
              else ro.terminal_reward(cols, strict=strict) >= 0.0)
        for i in range(ok.size):
            t = int(cols["terrain_id"][i])
            if tiles is not None and t not in tiles:
                continue
            out.setdefault((t, speed), {})[int(cols["env_id"][i])] = float(ok[i])
    return out


def report(name: str, cells_a: dict, cells_b: dict, *, n_boot: int = N_BOOT,
           seed: int = 0) -> dict:
    """Paired delta with a cluster bootstrap and exact McNemar.

    Args:
        name: Cluster label, for the printed block.
        cells_a: :func:`load` output for the control arm.
        cells_b: :func:`load` output for the fine-tuned arm.
        n_boot: Bootstrap resamples.
        seed: RNG seed, so a reported interval is reproducible independently of the order
            clusters happen to be reduced in.

    Returns:
        The cluster's estimates.
    """
    rng = np.random.default_rng(seed)
    cells = sorted(set(cells_a) & set(cells_b))
    per_cell = []
    for c in cells:
        envs = sorted(set(cells_a[c]) & set(cells_b[c]))
        per_cell.append((np.array([cells_a[c][e] for e in envs]),
                         np.array([cells_b[c][e] for e in envs])))
    n_env = sum(a.size for a, _ in per_cell)
    delta = float(np.mean([b.mean() - a.mean() for a, b in per_cell]))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(per_cell), len(per_cell))
        acc = 0.0
        for j in idx:
            a, b = per_cell[j]
            k = rng.integers(0, a.size, a.size)      # resample ENVS, differencing within env
            acc += (b[k] - a[k]).mean()
        boots[i] = acc / len(idx)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Exact McNemar over discordant pairs, pooled across cells.
    b_win = sum(int(((b - a) > 0).sum()) for a, b in per_cell)
    a_win = sum(int(((b - a) < 0).sum()) for a, b in per_cell)
    m = a_win + b_win
    k = min(a_win, b_win)
    if m:
        lb = -m * log(2.0)
        tail = sum(exp(lgamma(m + 1) - lgamma(i + 1) - lgamma(m - i + 1) + lb)
                   for i in range(k + 1))
        p = min(2.0 * tail, 1.0)
    else:
        p = 1.0
    print(f"\n{name}")
    print(f"  {len(cells)} cells, {n_env} paired envs")
    print(f"  paired delta   {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"(width {hi - lo:.4f})")
    print(f"  McNemar        {b_win} envs improved, {a_win} regressed, "
          f"{n_env - m} unchanged   p={p:.3g}")
    return {"delta": delta, "lo": float(lo), "hi": float(hi), "n_env": n_env,
            "n_cells": len(cells), "improved": b_win, "regressed": a_win,
            "unchanged": n_env - m, "mcnemar_p": p}


def main() -> int:
    """Report every cluster's paired delta."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="logs/box2/quick")
    p.add_argument("--metric", default="terminal_reward",
                   choices=("terminal_reward", "reached_goal"))
    p.add_argument("--failure_rule", choices=("strict", "loose"), default="strict")
    p.add_argument("--n_boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="paired.json")
    a = p.parse_args()
    kw = dict(metric=a.metric, strict=a.failure_rule == "strict")
    out = {}
    for nm, pt, ft, tiles in CLUSTERS:
        out[nm] = report(nm, load(a.root, pt, tiles, **kw), load(a.root, ft, tiles, **kw),
                         n_boot=a.n_boot, seed=a.seed)
    out["_meta"] = {"metric": a.metric, "failure_rule": a.failure_rule,
                    "n_boot": a.n_boot, "seed": a.seed,
                    "selection": "each env's first completed episode"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
