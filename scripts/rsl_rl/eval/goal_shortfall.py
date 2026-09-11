# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where do the robots that SURVIVED but never reached the goal actually end up?

Spatial success is ``no failure fired AND the base entered the goal region``.  When it comes
in well under the survival rate, the difference is a population of episodes that stayed
upright and did not arrive -- and those are not one thing.  A robot that stopped 30 cm short
of an approach margin did the task; one that walked off the side of the flight did not.  The
first is a tolerance that is too tight, the second is a real failure, and averaging them
hides both.

So every miss is attributed to the limb of ``goal_region.in_goal`` that actually excluded it:

* ``short``      -- downrange never reached ``-approach_margin``; laterally and vertically fine.
* ``lateral``    -- veered outside ``half_width``.
* ``height``     -- ended outside ``height_tol`` (walked off an elevated tile, typically).
* ``multiple``   -- more than one limb.

Only ``short`` is a candidate for widening the margin, and the sweep at the end says what
widening would buy -- BEFORE anyone changes a tolerance and re-runs.

**The sweep is an UPPER BOUND, not a prediction.**  ``in_goal`` is evaluated per step and
needs all three limbs true AT THE SAME INSTANT, but the table stores the terminal lateral and
height against the furthest downrange ever reached.  An episode that was laterally fine at its
furthest point and drifted afterwards counts here and would not count in a re-run.  Re-running
with a wider margin is the only way to get the real number; this says whether that is worth
doing.

Imports no simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import report as eval_report  # noqa: E402
import schema  # noqa: E402


def classify(d: dict, tol: np.ndarray) -> dict[str, np.ndarray]:
    """Which limb of the goal test excluded each episode.

    Args:
        d: Episode columns.
        tol: ``[n, 3]`` of ``(approach_margin_m, half_width_m, height_tol_m)``.

    Returns:
        Boolean masks, one per limb, plus the derived categories.
    """
    short = d["max_goal_downrange_m"] < -tol[:, 0]
    lateral = np.abs(d["goal_lateral_m"]) > tol[:, 1]
    height = np.abs(d["goal_height_err_m"]) > tol[:, 2]
    n_bad = short.astype(int) + lateral.astype(int) + height.astype(int)
    return {"short": short & (n_bad == 1), "lateral": lateral & (n_bad == 1),
            "height": height & (n_bad == 1), "multiple": n_bad > 1,
            "none": n_bad == 0, "short_any": short}


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cell_dir", help="a cell directory holding episodes.h5")
    p.add_argument("--goal_regions", required=True)
    p.add_argument("--plan", default="", help="plan.json, for family names")
    args = p.parse_args()

    d, meta = schema.read_episode_table(os.path.join(args.cell_dir, "episodes.h5"))
    spec = json.loads(open(args.goal_regions).read())
    codes = meta.get("family_codes") or {}
    name_of = {int(v): k for k, v in codes.items()}

    fams = np.array([name_of.get(int(c), "?") for c in d["family_code"]])
    tol = np.zeros((len(fams), 3), dtype=np.float64)
    for i, f in enumerate(fams):
        t = (spec.get("by_family") or {}).get(f) or spec["default"]
        tol[i] = (t["approach_margin_m"], t["half_width_m"], t["height_tol_m"])

    # The SAME partition report.py uses, so 'failure' means one thing across both. A local
    # list would drift the moment a termination term is added.
    _, fail_terms = eval_report.term_partition(meta)
    bits = d["termination_bits"]
    terms = list(meta.get("termination_terms") or [])
    failed = np.zeros(bits.shape, dtype=bool)
    for name in fail_terms:
        if name in terms:
            failed |= (bits & (1 << terms.index(name))) != 0
    arrived = d["reached_goal"].astype(bool)

    alive_miss = (~failed) & (~arrived)
    n = len(arrived)
    print(f"cell {os.path.basename(args.cell_dir)}: {n} episodes")
    print(f"  survived (no failure)      {(~failed).mean():.3f}")
    print(f"  arrived                    {arrived.mean():.3f}")
    print(f"  SURVIVED BUT NEVER ARRIVED {alive_miss.mean():.3f}  ({alive_miss.sum()} eps)"
          "   <- the population in question")

    if not alive_miss.any():
        return 0
    cls = classify(d, tol)
    print("\n  why they missed (single-limb attribution; only 'short' is a tolerance issue):")
    for key in ("short", "lateral", "height", "multiple", "none"):
        m = cls[key] & alive_miss
        if not m.any():
            continue
        short_by = -(d["max_goal_downrange_m"][m] + tol[m, 0])
        extra = (f"  median {np.median(short_by):+.2f} m beyond the margin"
                 if key in ("short",) else "")
        print(f"    {key:9s} {m.sum():5d}  ({m.sum() / alive_miss.sum():.1%} of misses)"
              f"{extra}")

    print("\n  how close did they get (min distance to the region, 0 = inside):")
    md = d["min_dist_to_goal_m"][alive_miss]
    for q in (10, 25, 50, 75, 90):
        print(f"    p{q:<3d} {np.percentile(md, q):6.2f} m")

    print("\n  MARGIN SWEEP -- success if approach_margin were widened (UPPER BOUND):")
    # SPATIAL success, not the arrival rate: `got` below excludes episodes that failed, so
    # comparing it against a base that counts arrived-then-fell makes the sweep read
    # negative at the CURRENT margin, where by construction it must read zero.
    base = float((arrived & ~failed).mean())
    print(f"    (spatial success today = {base:.3f}; the row at the current margin shows "
          "this method's own optimism)")
    for extra_m in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        wide = tol.copy()
        wide[:, 0] += extra_m
        would = (d["max_goal_downrange_m"] >= -wide[:, 0]) & \
                (np.abs(d["goal_lateral_m"]) <= wide[:, 1]) & \
                (np.abs(d["goal_height_err_m"]) <= wide[:, 2])
        got = (arrived | (would & ~failed)) & ~failed
        print(f"    margin {tol[0, 0] + extra_m:4.2f} m   success <= {got.mean():.3f}"
              f"   ({got.mean() - base:+.3f})")
    print("\n  Read the sweep as a ceiling: in_goal needs all three limbs true at the SAME "
          "step,\n  and this pairs the furthest downrange with the TERMINAL lateral/height.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
