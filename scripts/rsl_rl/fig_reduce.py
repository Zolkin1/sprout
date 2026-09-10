# Copyright 2026 Zachary Olkin. All rights reserved.
"""Reduce paired quick-eval runs into the paper figure's numbers, with clustered intervals.

Two quantities, each with the reduction the measurement actually requires.

SUCCESS is scored over each env's FIRST episode, through the SAME
:func:`awr_buffer.first_per_env` the training buffer and ``box2_ic_report`` use -- pooling every
recorded episode re-weights toward envs whose episodes were SHORT, because an env only starts a
second while the run waits for the slowest.  On one in-distribution tile that meant 241
episodes against ~21 on a healthy tile, biasing the pooled rate by ~0.15.  The success RULE
comes from :func:`rollout_outcomes.terminal_reward`, so the figure and the report cannot report
different numbers for the same recording.

TERRAIN PENALTY is ``contact_penalty * w_c + penetration_sum_m * w_p`` with **the weights the
rows were GRADED under**, read from the chunk group's own ``spec`` stamp rather than restated
here: the two terms' p95s differ by a factor of 640, so equal weights make the "total" a report
on penetration alone (to within 0.16%), and a hardcoded pair would silently misreport a
recording graded under different ones.

It is then EXPOSURE-STANDARDISED over ``goal_downrange_m``, because the penalty is a
per-re-plan average and a better arm survives longer -- so its re-plans are drawn from deeper,
harder course and an unmatched comparison INVERTS the conclusion.  Direct standardisation: mean
per downrange bin, then averaged over bins weighted by the POOLED (both-arm) exposure there, so
neither arm's own survival decides how much a bin counts.

INTERVALS are a cluster bootstrap over (tile, speed) CELLS.  A binomial interval over pooled
episodes has an actual coverage of 14-70% against a nominal 95% here, because episodes sharing
a tile share its geometry.  The cell count is the real limit on precision -- 6 for box2, 15 for
up-into-down -- and it is reported so a reader can see it.

Usage::

    python scripts/rsl_rl/fig_reduce.py --out fig_numbers.json
    python scripts/rsl_rl/fig_reduce.py --arm "Two boxes:eval_parent:eval_ft" --out x.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
from math import comb

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

#: Deep-penetration thresholds (m).  ``max_penetration_m`` is the RAW geometric depth --
#: ``(-sd - penetration_tol_m).clamp_min(0)`` maxed over charged points, with the per-body
#: weights entering only the SUM -- so these are real centimetres of a limb inside the terrain,
#: already net of the 5 mm uncharged tolerance.
DEEP_THRESH_M: tuple[float, ...] = (0.03, 0.04, 0.05, 0.06)

#: Bootstrap resamples, and the downrange bin width (m) the penalty is standardised over.
N_BOOT: int = 20000
BIN_M: float = 1.0

#: The paper figure's clusters: ``label -> (parent run, fine-tuned run, tile filter)``.  Every
#: fine-tuned arm is ITERATION 5 of its own loop.
#:
#: The third is IN-DISTRIBUTION -- routes authored over the original clip-library tiles, the
#: terrain the generator and the tracker were both trained on -- while the first two are
#: out-of-distribution courses.  ``stairs_down`` is tiles 20-39 of that 75-tile library, a
#: contiguous family block read off the library's own ``terrain_manifest`` rather than assumed.
DEFAULT_ARMS: dict[str, tuple[str, str, range | None]] = {
    "Two boxes": ("eval_parent_prog_s3", "eval_prog_s3_i5", None),
    "Stairs up-into-down": ("eval_parent_updn_s3", "eval_updn_s3_i5", None),
    "Stairs down (in-dist.)": ("eval_parent_indist", "eval_indist_i5", range(20, 40)),
}


def load(root: str, tag: str, tiles: range | None = None, *,
         metric: str = "terminal_reward", strict: bool = True,
         rounds: int = 1) -> dict:
    """Per-cell first-episode outcomes and per-chunk penalties for one eval run.

    Args:
        root: Directory holding the eval run directories.
        tag: The run directory under ``root``.
        tiles: Restrict to these ``terrain_id`` values.
        metric: Success rule; one of ``box2_ic_report.METRICS``.
        strict: Count a tracking violation as a failure as well as a fall.
        rounds: Episodes to keep per env (``0`` pools everything).

    Returns:
        ``{(tile, speed): {...}}``.

    Raises:
        SystemExit: If a run directory holds no episode table, or a table carries no chunk
            group -- an empty cluster reads exactly like a finished run with no effect in it.
    """
    cells: dict[tuple, dict] = {}
    for d in sorted(glob.glob(os.path.join(root, tag, "train_v*"))):
        if not os.path.isdir(d):        # a `train_v0.83.log` sits beside the directory
            continue
        speed = float(d.rsplit("train_v", 1)[1])
        f = glob.glob(d + "/*.episodes.h5")
        if not f:
            continue
        with h5py.File(f[0], "r") as h:
            if "chunks" not in h:
                raise SystemExit(
                    f"{f[0]} carries no `chunks` group, so it was recorded without "
                    "--terrain_reward and the penalty half of the figure cannot be built "
                    "from it.")
            keep = ~h["censored"][:].astype(bool)
            uid_all = h["episode_uid"][:].astype(np.int64)
            if rounds > 0 and "env_id" in h:
                env_all = h["env_id"][:].astype(np.int64)
                keep &= np.isin(uid_all, np.asarray(
                    first_per_env(uid_all[keep], env_all[keep], rounds), dtype=np.int64))
            cols = {k: h[k][:][keep] for k in h if isinstance(h[k], h5py.Dataset)}
            tile = cols["terrain_id"].astype(np.int64)
            uid = cols["episode_uid"].astype(np.int64)
            ok = (cols["reached_goal"].astype(bool) if metric == "reached_goal"
                  else ro.terminal_reward(cols, strict=strict) >= 0.0)
            c = h["chunks"]
            spec = json.loads(c.attrs.get("spec", "{}"))
            w_c = float(spec.get("contact_weight", 1.0))
            w_p = float(spec.get("penetration_weight", 1.0))
            con = w_c * c["contact_penalty"][:].astype(np.float64)
            pnt = w_p * c["penetration_sum_m"][:].astype(np.float64)
            mx = c["max_penetration_m"][:].astype(np.float64)
            dr = c["goal_downrange_m"][:].astype(np.float64)
            cuid = c["episode_uid"][:].astype(np.int64)
        # Chunks follow the KEPT episodes, so a discarded episode's re-plan rows leave the
        # penalty too -- scoring success on one sample and the penalty on another would make
        # the two halves of the figure describe different runs.
        uid_to_tile = dict(zip(uid.tolist(), tile.tolist()))
        ctile = np.array([uid_to_tile.get(int(u), -1) for u in cuid], np.int64)
        for t in np.unique(tile):
            if tiles is not None and int(t) not in tiles:
                continue
            m = tile == t
            cm = ctile == t
            cells[(int(t), speed)] = {
                "goal": ok[m].astype(float), "pen": (con + pnt)[cm], "dr": dr[cm],
                "con": con[cm], "pnt": pnt[cm], "weights": (w_c, w_p),
                **{f"deep{int(round(v * 100)):02d}": (mx[cm] >= v).astype(float)
                   for v in DEEP_THRESH_M}}
    if not cells:
        raise SystemExit(f"no eval cells under {os.path.join(root, tag)!r}. An empty cluster "
                         "is indistinguishable from a finished run with no effect in it.")
    return cells


def standardised_penalty(pen_a, dr_a, pen_b, dr_b) -> tuple[float, float]:
    """Both arms' mean penalty, reweighted to ONE shared exposure distribution.

    Args:
        pen_a: Arm A's per-chunk penalties.
        dr_a: Arm A's per-chunk downrange-to-goal.
        pen_b: Arm B's per-chunk penalties.
        dr_b: Arm B's per-chunk downrange-to-goal.

    Returns:
        ``(mean_a, mean_b)``; nan when the shared support is empty.
    """
    if pen_a.size == 0 or pen_b.size == 0:
        return float("nan"), float("nan")
    lo = max(dr_a.min(), dr_b.min())
    hi = min(dr_a.max(), dr_b.max())
    if not hi > lo:
        return float("nan"), float("nan")
    edges = np.arange(lo, hi + BIN_M, BIN_M)
    ia = np.clip(np.digitize(dr_a, edges) - 1, 0, len(edges) - 2)
    ib = np.clip(np.digitize(dr_b, edges) - 1, 0, len(edges) - 2)
    ka = (dr_a >= lo) & (dr_a <= hi)
    kb = (dr_b >= lo) & (dr_b <= hi)
    num_a = num_b = den = 0.0
    for k in range(len(edges) - 1):
        sa, sb = ka & (ia == k), kb & (ib == k)
        if not (sa.any() and sb.any()):
            continue
        # Pooled exposure in this bin is the common weight, so neither arm's own survival
        # decides how much the bin counts.
        w = float(sa.sum() + sb.sum())
        num_a += w * float(pen_a[sa].mean())
        num_b += w * float(pen_b[sb].mean())
        den += w
    if den == 0.0:
        return float("nan"), float("nan")
    return num_a / den, num_b / den


def _sign_p(diff: np.ndarray) -> float:
    """Exact two-sided sign test over paired per-cell deltas, ties and nans dropped.

    Args:
        diff: Per-cell differences.

    Returns:
        The two-sided p-value; 1.0 when nothing is testable.
    """
    d = diff[np.isfinite(diff)]
    k, m = int((d > 0).sum()), int((d != 0).sum())
    if m == 0:
        return 1.0
    return min(sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m * 2, 1.0)


def boot(cells_a: dict, cells_b: dict, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimates and 95% cell-clustered bootstrap intervals for both arms.

    Args:
        cells_a: :func:`load` output for the control arm.
        cells_b: :func:`load` output for the fine-tuned arm.
        n_boot: Bootstrap resamples.
        seed: RNG seed, so a reported interval is reproducible.

    Returns:
        A flat dict of estimates, interval bounds and per-cell counts.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(set(cells_a) & set(cells_b))
    ga = [cells_a[k]["goal"] for k in keys]
    gb = [cells_b[k]["goal"] for k in keys]
    series = {}
    deep_fields = tuple(f"deep{int(round(v * 100)):02d}" for v in DEEP_THRESH_M)
    for field in ("pen", "con", "pnt", *deep_fields):
        va, vb = [], []
        for k in keys:
            ma, mb = standardised_penalty(cells_a[k][field], cells_a[k]["dr"],
                                          cells_b[k][field], cells_b[k]["dr"])
            va.append(ma)
            vb.append(mb)
        series[field] = (np.array(va), np.array(vb))
    pa, pb = series["pen"]

    n = len(keys)
    out = {"n_cells": n, "n_tiles": len({k[0] for k in keys}),
           "n_episodes_a": int(sum(v.size for v in ga)),
           "n_episodes_b": int(sum(v.size for v in gb))}
    # Point estimates: the cell-averaged rate, matching the bootstrap's own statistic.
    out["succ_a"] = float(np.mean([v.mean() for v in ga]))
    out["succ_b"] = float(np.mean([v.mean() for v in gb]))
    for field, (va, vb) in series.items():
        out[f"{field}_a"] = float(np.nanmean(va))
        out[f"{field}_b"] = float(np.nanmean(vb))
        out[f"{field}_cells_down"] = int((vb < va).sum())
        # Lower is better for every one of these, so the test is on `a - b`.
        out[f"{field}_sign_p"] = _sign_p(va - vb)

    extra: dict = {}
    sa, sb = np.empty(n_boot), np.empty(n_boot)
    qa, qb = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # Resample the CELL, then the episodes within it: both levels of the design.
        sa[i] = np.mean([ga[j][rng.integers(0, ga[j].size, ga[j].size)].mean() for j in idx])
        sb[i] = np.mean([gb[j][rng.integers(0, gb[j].size, gb[j].size)].mean() for j in idx])
        qa[i] = np.nanmean(pa[idx])
        qb[i] = np.nanmean(pb[idx])
        for field, (va, vb) in series.items():
            if field == "pen":
                continue
            extra.setdefault(field + "_a", np.empty(n_boot))[i] = np.nanmean(va[idx])
            extra.setdefault(field + "_b", np.empty(n_boot))[i] = np.nanmean(vb[idx])
    for nm, arr in (list(extra.items())
                    + [("succ_a", sa), ("succ_b", sb), ("pen_a", qa), ("pen_b", qb)]):
        out[nm + "_lo"], out[nm + "_hi"] = (float(np.percentile(arr, 2.5)),
                                            float(np.percentile(arr, 97.5)))
    # Paired sign test over cells, on success.
    d = np.array([gb[j].mean() - ga[j].mean() for j in range(n)])
    k, m = int((d > 0).sum()), int((d != 0).sum())
    out["cells_up"], out["cells_tested"] = k, m
    out["sign_p"] = _sign_p(d)
    return out


def parse_arm(text: str) -> tuple[str, str, str, range | None]:
    """One ``label:parent:finetuned[:lo-hi]`` cluster spec.

    Args:
        text: The spec.

    Returns:
        ``(label, parent_tag, finetuned_tag, tile_range)``.

    Raises:
        SystemExit: On a spec that does not carry at least a label and two run tags.
    """
    parts = text.split(":")
    if len(parts) < 3:
        raise SystemExit(f"--arm needs 'label:parent:finetuned[:lo-hi]', got {text!r}")
    tiles = None
    if len(parts) > 3 and parts[3]:
        lo, hi = (int(v) for v in parts[3].split("-"))
        tiles = range(lo, hi + 1)
    return parts[0], parts[1], parts[2], tiles


def main() -> int:
    """Reduce every cluster and write the figure's numbers."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="logs/box2/quick",
                   help="directory holding the eval run directories")
    p.add_argument("--arm", action="append", default=[],
                   help="'label:parent:finetuned[:lo-hi]', repeatable. Default: the paper's "
                        "three clusters (see DEFAULT_ARMS).")
    p.add_argument("--metric", default="terminal_reward",
                   choices=("terminal_reward", "reached_goal"),
                   help="Success rule; matches box2_ic_report's flag of the same name.")
    p.add_argument("--failure_rule", choices=("strict", "loose"), default="strict",
                   help="strict is what Isaac now terminates on; loose is fall-only and is "
                        "what a recording made before that termination must be scored under.")
    p.add_argument("--rounds", type=int, default=1,
                   help="episodes kept per env, first-completed first (0 pools everything)")
    p.add_argument("--n_boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="fig_numbers.json")
    a = p.parse_args()

    arms = ({lbl: (pt, ft, tl) for lbl, pt, ft, tl in (parse_arm(s) for s in a.arm)}
            if a.arm else DEFAULT_ARMS)
    strict = a.failure_rule == "strict"
    kw = dict(metric=a.metric, strict=strict, rounds=a.rounds)
    res: dict = {}
    for name, (pt, ft, tiles) in arms.items():
        res[name] = boot(load(a.root, pt, tiles, **kw), load(a.root, ft, tiles, **kw),
                         n_boot=a.n_boot, seed=a.seed)
        r = res[name]
        print(f"\n=== {name} ===  {r['n_cells']} cells / {r['n_tiles']} tiles, "
              f"n={r['n_episodes_a']} vs {r['n_episodes_b']} episodes")
        print(f"  success  parent {r['succ_a']:.4f} "
              f"[{r['succ_a_lo']:.4f}, {r['succ_a_hi']:.4f}]"
              f"   fine-tuned {r['succ_b']:.4f} "
              f"[{r['succ_b_lo']:.4f}, {r['succ_b_hi']:.4f}]")
        for f_, lbl in ([("pen", "penalty TOTAL"), ("con", "  contact term"),
                         ("pnt", "  penetration ")]
                        + [(f"deep{int(round(v * 100)):02d}",
                            f"deep>={int(round(v * 100))}cm rate") for v in DEEP_THRESH_M]):
            print(f"  {lbl}  parent {r[f_ + '_a']:.4f} "
                  f"[{r[f_ + '_a_lo']:.4f}, {r[f_ + '_a_hi']:.4f}]"
                  f"   fine-tuned {r[f_ + '_b']:.4f} "
                  f"[{r[f_ + '_b_lo']:.4f}, {r[f_ + '_b_hi']:.4f}]"
                  f"   cells down {r[f_ + '_cells_down']}/{r['n_cells']}"
                  f"  p={r[f_ + '_sign_p']:.3g}")
        print(f"  cells up {r['cells_up']}/{r['cells_tested']}, sign p={r['sign_p']:.4g}")

    res["_meta"] = {
        "deep_thresholds_m": list(DEEP_THRESH_M), "penetration_tol_m": 0.005,
        "n_boot": a.n_boot, "seed": a.seed, "downrange_bin_m": BIN_M,
        "metric": a.metric, "failure_rule": a.failure_rule, "rounds": a.rounds,
        "success": f"{a.metric}, first {a.rounds} episode(s) per env, cell-averaged",
        "penalty": ("exposure-standardised over goal_downrange_m, cell-averaged, with the "
                    "weights read from each chunk group's own spec stamp"),
        "interval": "95% cluster bootstrap over (tile, speed) cells",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
