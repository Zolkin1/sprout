#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reduce two arms' ``live_select.py`` runs into a seed-paired per-cell table.

``live_select`` writes one ``live_select.json`` per invocation and no reducer of its own --
it prints a per-tile success rate and stops. That is enough for a single arm and not enough
for an A/B, where what matters is the PAIRED difference: ``live_select`` keys each draw on
``[episode_seed, replan_index]``, so the same ``(tile, speed, episode)`` sees the same noise
tape in every arm and a per-episode disagreement is attributable to the arms.

**The test is exact McNemar on the DISCORDANT pairs, not a clustered interval.** A handful
of authored tiles is a handful of clusters, and a percentile bootstrap over three of them
cannot resolve anything -- the sign-flip p floors at ``2/2**3``. The discordant counts are
the honest handle at this sample size, and they are what this prints.

Usage::

    python report_measured_pairs.py --a 'logs/value/ftmeas_s*_parent' \\
                                    --b 'logs/value/ftmeas_s*_finetune' \\
                                    --label_a parent --label_b finetune
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import comb


def sign_p(wins: int, n: int) -> float:
    """Exact two-sided binomial sign test on ``wins`` of ``n`` discordant pairs.

    Args:
        wins: Pairs the B arm won.
        n: Discordant pairs.

    Returns:
        The two-sided p, or 1.0 when nothing disagreed.
    """
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(patterns: list[str]) -> dict:
    """``(tile, speed) -> {episode: verdict}`` over every run matching the patterns.

    Args:
        patterns: Glob patterns for run DIRECTORIES (each holding a ``live_select.json``).

    Returns:
        The nested mapping.
    """
    out: dict = {}
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            p = os.path.join(d, "live_select.json")
            if not os.path.isfile(p):
                continue
            data = json.load(open(p))
            speed = float(data["args"]["speed"])
            for r in data["rows"]:
                out.setdefault((int(r["tile"]), speed), {})[int(r["episode"])] = r["verdict"]
    return out


def main() -> int:
    """Print the paired table.

    Returns:
        0 on success, 2 when either arm produced nothing.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", nargs="+", required=True, help="glob(s) for arm A's run dirs")
    p.add_argument("--b", nargs="+", required=True, help="glob(s) for arm B's run dirs")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    args = p.parse_args()

    A, B = load(args.a), load(args.b)
    if not A or not B:
        print(f"[report] nothing to reduce: {args.label_a} has {len(A)} cell(s), "
              f"{args.label_b} has {len(B)}")
        return 2

    print(f"{'tile':>5} {'speed':>6} {'n':>4} {args.label_a:>9} {args.label_b:>9} "
          f"{'delta':>7} {'won/disc':>9} {'p':>7}")
    print("-" * 64)
    sa = sb = n = won = lost = 0
    for key in sorted(set(A) & set(B)):
        seeds = sorted(set(A[key]) & set(B[key]))
        if not seeds:
            continue
        a = [A[key][s] == "success" for s in seeds]
        b = [B[key][s] == "success" for s in seeds]
        w = sum(1 for i in range(len(seeds)) if b[i] and not a[i])
        l = sum(1 for i in range(len(seeds)) if a[i] and not b[i])   # noqa: E741
        ar, br = sum(a) / len(a), sum(b) / len(b)
        print(f"{key[0]:>5} {key[1]:>6.1f} {len(seeds):>4} {ar:>9.3f} {br:>9.3f} "
              f"{br - ar:>+7.3f} {f'{w}/{w + l}':>9} {sign_p(w, w + l):>7.3f}")
        sa += sum(a); sb += sum(b); n += len(seeds); won += w; lost += l
    print("-" * 64)
    if n:
        print(f"POOLED n={n}  {args.label_a} {sa / n:.3f}  {args.label_b} {sb / n:.3f}  "
              f"delta {(sb - sa) / n:+.3f}  won {won}/{won + lost} discordant  "
              f"p={sign_p(won, won + lost):.3f}")
        print("Read the discordant counts, not a clustered interval: a few tiles is a few "
              "clusters and the sign-flip p floors out.")
    only = (set(A) ^ set(B))
    if only:
        print(f"\n{len(only)} cell(s) present for one arm only, excluded: {sorted(only)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
