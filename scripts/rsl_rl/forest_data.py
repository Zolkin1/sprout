# Copyright 2026 Zachary Olkin. All rights reserved.

"""Paired deltas of every ladder arm against AWR, grouped, for the forest plot.

AWR IS THE ZERO LINE, so every row is `arm - AWR` measured WITHIN its own group: the boxes-strict
arms against boxes-strict AWR, the boxes-loose arms against boxes-LOOSE AWR, and so on. That is
the comparison the ablation is actually about -- "what does using this update instead of AWR
cost" -- and it is not the same as differencing two arms' vs-parent deltas, which would carry the
parent's own noise into every row twice.

Each row's interval is the 95% CLUSTER-T over the per-(tile, speed)-cell paired deltas --
`mean +- t(C-1) * SD/sqrt(C)`, the cell being the unit of independence -- plus exact McNemar over
discordant env pairs, via `paired_delta.report`. A single-stage cluster bootstrap rides along as
a robustness check. It is NOT the two-stage bootstrap this file used to describe: resampling
cells AND envs within them double-counts the within-cell variance and inflated every interval in
this campaign by 14-29%. Nothing is restated here; this only chooses the pairings.

Usage::

    python forest_data.py groups.json out.json        # uses this checkout
    W=<other-worktree> python forest_data.py ...      # or point it elsewhere
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

def _checkout_root() -> str:
    """The checkout this script lives in.

    Walks up looking for the two source trees rather than counting ``..`` hops, which break
    silently when a file moves between packages. ``W`` still overrides, because the reductions
    are routinely pointed at a DIFFERENT checkout than the one holding the script -- the ladder
    arms ran from a pinned worktree.

    Returns:
        Absolute path to the checkout root.
    """
    here = pathlib.Path(__file__).resolve()
    for p in here.parents:
        if (p / "scripts/rsl_rl").is_dir() and (p / "source").is_dir():
            return str(p)
    raise SystemExit(f"cannot locate a checkout root above {here}; set W=<worktree>")


W = os.environ.get("W") or _checkout_root()

sys.path.insert(0, os.path.join(W, "scripts/rsl_rl"))
sys.path.insert(0, os.path.join(W, "scripts/rsl_rl/eval"))

import paired_delta as pd  # noqa: E402

METRIC = os.environ.get("METRIC", "terminal_reward")

spec = json.load(open(sys.argv[1]))
out = []
for g in spec["groups"]:
    ref = g["reference"]
    cells_ref = pd.load(os.path.dirname(ref), os.path.basename(ref), metric=METRIC, strict=True)
    if not cells_ref:
        print(f"  !! {g['label']}: reference has no episodes, skipping group")
        continue
    for arm in g["arms"]:
        name, path = arm["label"], arm["dir"]
        if path == ref:
            # AWR against itself is the zero line by construction, not a measurement. Emitted
            # with no interval so the plot can draw the reference without implying one.
            out.append({"group": g["label"], "arm": name, "delta": 0.0,
                        "lo": 0.0, "hi": 0.0, "reference": True,
                        "n_cells": len(cells_ref), "n_env": None, "mcnemar_p": None})
            continue
        cells = pd.load(os.path.dirname(path), os.path.basename(path), metric=METRIC, strict=True)
        if not cells:
            print(f"  !! {g['label']}/{name}: no episodes, skipping arm")
            continue
        r = pd.report(f"{g['label']} / {name} vs AWR", cells_ref, cells)
        r.update({"group": g["label"], "arm": name, "reference": False})
        out.append(r)

json.dump({"metric": METRIC, "rows": out}, open(sys.argv[2], "w"), indent=1)
print(f"\nwrote {sys.argv[2]}  ({len(out)} rows)")
