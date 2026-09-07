# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pair two two-box rollout sets on ``(distance, speed)`` and report the difference.

**The pairing is the point.**  Success on this course varies far more across the initial
condition than between any two generators -- 0.003 to 0.260 across the full grid against
deltas of order 0.01 -- so an unpaired comparison of two pooled rates is dominated by
whichever cells each arm happened to run.  Cells are matched by ``(approach distance,
commanded speed)``, which is the same course in both arms.

**Two intervals, and they disagree on purpose.**  The naive pooled binomial one is printed
because it is what everyone computes first, and the TILE-CLUSTERED bootstrap beside it is the
one to believe: episodes within a tile share a course and are correlated, and a binomial
interval over pooled episodes has a measured actual coverage of 14-70% against a nominal 95%
in this repo.  With 5 tiles (or 2, under ``box2_quick``) the clustered interval is wide, and
that width is the honest resolution rather than a defect of the estimator.

The sign test over cells is reported as a third view: it uses only the DIRECTION of each
cell's delta, so it is unaffected by one cell with a large swing.

Usage::

    python scripts/rsl_rl/box2_compare.py logs/box2/rollouts logs/box2/eval_fbc \\
        --labels parent fbc
    python scripts/rsl_rl/box2_compare.py logs/box2/rollouts logs/box2/quick/awr_b1.0 \\
        --labels parent awr --distances_b 1.5 3.0
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from box2_ic_report import load_cells  # noqa: E402


def sign_test(deltas: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign test on the non-zero deltas.

    Args:
        deltas: Per-cell differences.

    Returns:
        ``(n_positive, n_nonzero, p)``.
    """
    nz = deltas[deltas != 0.0]
    n, pos = int(nz.size), int((nz > 0).sum())
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(max(pos, n - pos), n + 1))
    return pos, n, min(1.0, 2.0 * tail / 2 ** n)


def clustered_ci(deltas: np.ndarray, tiles: np.ndarray, n_boot: int = 20000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over TILES, resampling whole clusters.

    Args:
        deltas: Per-cell differences.
        tiles: The tile (approach distance) each cell belongs to.
        n_boot: Bootstrap draws.
        seed: RNG seed.

    Returns:
        The 2.5th and 97.5th percentiles of the resampled mean delta.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(tiles)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        out[i] = np.mean(np.concatenate([deltas[tiles == t] for t in pick]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    """Print the paired comparison."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="baseline rollout directory")
    p.add_argument("b", help="comparison rollout directory")
    p.add_argument("--labels", nargs=2, default=("A", "B"))
    p.add_argument("--distances_a", type=float, nargs="+", default=None,
                   help="tile-order approach distances for A (see box2_ic_report)")
    p.add_argument("--distances_b", type=float, nargs="+", default=None)
    a = p.parse_args()

    A = {(c["speed"], c["distance_m"]): c
         for c in load_cells(a.a, tuple(a.distances_a) if a.distances_a else None)
         if not c["holdout"]}
    B = {(c["speed"], c["distance_m"]): c
         for c in load_cells(a.b, tuple(a.distances_b) if a.distances_b else None)
         if not c["holdout"]}
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit(
            f"no (speed, distance) cell is present in both runs.\n  {a.labels[0]}: "
            f"{sorted(A)}\n  {a.labels[1]}: {sorted(B)}\nIf one used a smaller library, its "
            "tile indices mean different distances -- pass --distances_a/--distances_b.")
    la, lb = a.labels

    print("=" * 84)
    print(f"TWO-BOX PAIRED COMPARISON -- {lb} against {la}, {len(keys)} matched cells")
    print("=" * 84)
    print(f"{'speed':>6}{'dist':>6}{'n_' + la:>9}{'n_' + lb:>9}{la:>9}{lb:>9}{'delta':>9}"
          f"{'track ' + la:>11}{'track ' + lb:>11}")
    d = np.array([B[k]["reached_goal"] - A[k]["reached_goal"] for k in keys])
    dt = np.array([B[k]["track_reward_mean"] - A[k]["track_reward_mean"] for k in keys])
    tiles = np.array([k[1] for k in keys])
    for i, k in enumerate(keys):
        x, y = A[k], B[k]
        print(f"{k[0]:>6.2f}{k[1]:>6.2f}{x['n']:>9}{y['n']:>9}{x['reached_goal']:>9.3f}"
              f"{y['reached_goal']:>9.3f}{d[i]:>+9.3f}{x['track_reward_mean']:>11.3f}"
              f"{y['track_reward_mean']:>11.3f}")

    na = sum(A[k]["n"] for k in keys)
    nb = sum(B[k]["n"] for k in keys)
    pa = sum(A[k]["reached_goal"] * A[k]["n"] for k in keys) / na
    pb = sum(B[k]["reached_goal"] * B[k]["n"] for k in keys) / nb
    ta = sum(A[k]["track_reward_mean"] * A[k]["n"] for k in keys) / na
    tb = sum(B[k]["track_reward_mean"] * B[k]["n"] for k in keys) / nb
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    pos, nnz, psign = sign_test(d)
    lo, hi = clustered_ci(d, tiles)

    print(f"\n  pooled reached_goal   {la} {pa:.4f} ({na} ep)   {lb} {pb:.4f} ({nb} ep)   "
          f"delta {pb - pa:+.4f}")
    print(f"  pooled track_reward   {la} {ta:.4f}   {lb} {tb:.4f}   delta {tb - ta:+.4f}"
          f"   ({int((dt > 0).sum())}/{len(keys)} cells up)")
    print(f"\n  mean per-cell delta   {d.mean():+.4f}   median {np.median(d):+.4f}")
    print(f"  sign test over cells  {pos}/{nnz} positive, p = {psign:.4f}")
    print(f"  naive binomial 95%    [{pb - pa - 1.96 * se:+.4f}, {pb - pa + 1.96 * se:+.4f}]"
          "   <- OPTIMISTIC, episodes within a tile are correlated")
    print(f"  TILE-CLUSTERED 95%    [{lo:+.4f}, {hi:+.4f}]   over {len(np.unique(tiles))} "
          "tiles  <- believe this one")
    verdict = ("resolved" if lo > 0 or hi < 0 else "NOT resolved")
    print(f"\n  => the success difference is {verdict} at this sample size.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
