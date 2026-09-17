# Copyright 2026 Zachary Olkin. All rights reserved.

"""Contact-penalty deltas against AWR, in the same row shape the forest plot's success panel uses.

Runs `fig_reduce` once per arm with that arm's own block AWR as the "parent" slot, so `con_delta`
is `arm - AWR` on shared (tile, speed) cells with the exposure standardisation applied -- the same
head-to-head form used for every terrain comparison in this campaign.

**THE TWO PANELS PAIR AT DIFFERENT LEVELS AND THE CAPTION HAS TO SAY SO.** Success is paired per
ENVIRONMENT (`paired_delta`), because an env's first episode is identifiable across arms. The
contact penalty is a per-re-plan quantity standardised over downrange, so it pairs per CELL.
Both intervals are 95% cluster-t over per-cell values, but they are not the same kind of object
and should not be compared for width.

Usage::

    python forest_contact.py groups.json out.json     # uses this checkout
    W=<other-worktree> python forest_contact.py ...   # or point it elsewhere
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

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
spec = json.load(open(sys.argv[1]))
rows = []
for g in spec["groups"]:
    ref = g["reference"]
    for arm in g["arms"]:
        name, path = arm["label"], arm["dir"]
        if path == ref:
            rows.append({"group": g["label"], "arm": name, "delta": 0.0, "lo": 0.0, "hi": 0.0,
                         "reference": True})
            continue
        with tempfile.TemporaryDirectory() as td:
            # fig_reduce addresses arms as directories under one --root, so link the pair in.
            os.symlink(ref, os.path.join(td, "eval_ref"))
            os.symlink(path, os.path.join(td, "eval_arm"))
            out = os.path.join(td, "n.json")
            r = subprocess.run(
                [os.path.join(W, "env_isaaclab/bin/python"),
                 os.path.join(W, "scripts/rsl_rl/fig_reduce.py"),
                 "--root", td, "--arm", "x:eval_ref:eval_arm",
                 "--metric", "terminal_reward", "--out", out],
                capture_output=True, text=True, cwd=W)
            if r.returncode != 0:
                print(f"  !! {g['label']}/{name}: fig_reduce failed\n{r.stdout[-400:]}")
                continue
            d = json.load(open(out))["x"]
        rows.append({"group": g["label"], "arm": name, "reference": False,
                     # The CLUSTER-T delta, matching the success panel. `con_delta_lo/hi` carry
                     # the single-stage bootstrap and are the check, not the headline -- one
                     # method across the whole table rather than a per-panel switch.
                     "delta": d["con_delta_t"], "lo": d["con_delta_t_lo"],
                     "hi": d["con_delta_t_hi"],
                     "boot_lo": d["con_delta_lo"], "boot_hi": d["con_delta_hi"],
                     "a": d["con_a"], "b": d["con_b"],
                     "cells_down": d["con_cells_down"], "sign_p": d["con_sign_p"],
                     "n_cells": d["n_cells"]})
        print(f"  {g['label']:15s} {name:12s} contact {d['con_a']:.4f} -> {d['con_b']:.4f}  "
              f"delta {d['con_delta_t']:+.4f} "
              f"t[{d['con_delta_t_lo']:+.4f}, {d['con_delta_t_hi']:+.4f}] "
              f"boot[{d['con_delta_lo']:+.4f}, {d['con_delta_hi']:+.4f}]")

json.dump({"metric": "contact", "rows": rows}, open(sys.argv[2], "w"), indent=1)
print(f"\nwrote {sys.argv[2]}  ({len(rows)} rows)")
