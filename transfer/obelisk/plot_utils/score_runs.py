# Copyright 2026 Zachary Olkin. All rights reserved.

"""Score a set of obelisk runs into one table, so configurations are compared as distributions.

Outcomes are not repeatable run to run: over four IDENTICAL local runs, three climbed and one
fell. A single run therefore cannot compare two configurations, so this prints the fall rate
and the tilt distribution over a whole set.

The box a run came from is taken from its DIRECTORY (``log/bags`` local, ``log/bags_thor``
remote), never inferred from the data. It was briefly inferred from the cross-box clock offset,
which mislabelled two rows the moment SNTP stepped the offset near zero.

    python plot_utils/score_runs.py log/bags/2026-08-20_1[78]-*  log/bags_thor/*
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bag_reader import read_bag          # noqa: E402
from bag_run import Run                  # noqa: E402

#: Past this tilt the robot is on the floor, not recovering. The sim2sim fall check uses 70
#: deg; this is lower because we want "went over", not "was declared fallen".
FELL_DEG = 45.0


def _tilt_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    """Angle between the base's z axis and world up, per sample.

    Args:
        quat_xyzw: ``[n, 4]`` base orientation.

    Returns:
        ``[n]`` tilt in degrees.
    """
    x, y = quat_xyzw[:, 0], quat_xyzw[:, 1]
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def score(bag: str) -> dict | None:
    """One run's configuration and outcome.

    Args:
        bag: Path to a bag directory.

    Returns:
        A dict of fields, or None if the bag is unusable (torn, or never armed).
    """
    try:
        data = read_bag(bag)
        run = Run(data)
    except BaseException:
        return None
    ticks = [t for t in run.ticks if getattr(t, "armed", False)]
    if len(ticks) < 200:
        return None

    def col(name):
        return np.array([getattr(t, name, np.nan) for t in ticks], dtype=float)

    infer = col("fm_infer_ms")
    _, first = np.unique(col("fm_plan_id"), return_index=True)
    per_plan = infer[np.sort(first)]
    per_plan = per_plan[np.isfinite(per_plan)]
    est = [m for _t, m in data.get("/obelisk/g1/est_state", [])]
    if not est or per_plan.size == 0:
        return None
    qb = np.array([list(e.q_base) for e in est], dtype=float)
    tilt = _tilt_deg(qb[:, 3:7])
    cmd = np.array([list(t.command_effective) for t in ticks])
    leads = {int(p.lead_ticks) for _t, p in run.plans if hasattr(p, "lead_ticks")}
    return {
        "name": os.path.basename(bag.rstrip("/")),
        "box": "thor" if "bags_thor" in bag else "local",
        # TF32 is not recorded anywhere, so it is READ OFF the generator's cost: the same
        # graph measures ~14 ms (desktop) or ~21 ms (Thor) with it on, and ~15 / ~35 with it
        # off. The two bands do not overlap on either machine.
        "fm_ms": float(np.median(per_plan)),
        "lead": max(leads) if leads else 0,
        "cmd": "const" if len(np.unique(np.round(cmd, 2))) <= 4 else "joy",
        "clamped": int(np.nanmax(col("fm_offsets_clamped"))),
        "x_travel": float(qb[:, 0].max() - qb[:, 0].min()),
        "tilt_p99": float(np.percentile(tilt, 99)),
        "tilt_max": float(tilt.max()),
        "fell": bool(tilt.max() > FELL_DEG),
    }


def main() -> None:
    """Print one row per bag, then a per-group summary."""
    paths: list[str] = []
    for pattern in sys.argv[1:]:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        sys.exit("usage: score_runs.py <bag dirs or globs>")
    rows = [r for r in (score(p) for p in paths if os.path.isdir(p)) if r]
    if not rows:
        sys.exit("no usable runs (torn bags, or none armed)")

    hdr = f'{"run":<20}{"box":<7}{"tf32":<6}{"lead":>5}{"cmd":>6}{"clamp":>7}{"x_m":>7}{"p99":>8}{"max":>8}'
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tf = "off" if (r["fm_ms"] > 28 or (r["box"] == "local" and r["fm_ms"] > 14.4)) else "on"
        print(f'{r["name"][11:]:<20}{r["box"]:<7}{tf:<6}{r["lead"]:>5}{r["cmd"]:>6}'
              f'{r["clamped"]:>7}{r["x_travel"]:>7.2f}{r["tilt_p99"]:>8.1f}{r["tilt_max"]:>8.1f}'
              f'{"  FELL" if r["fell"] else ""}')

    print()
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        tf = "off" if (r["fm_ms"] > 28 or (r["box"] == "local" and r["fm_ms"] > 14.4)) else "on"
        groups.setdefault((r["box"], tf, r["lead"], r["cmd"]), []).append(r)
    print(f'{"group":<30}{"n":>3}{"fell":>6}{"tilt p99 (median of runs)":>28}')
    print("-" * 67)
    for (box, tf, lead, cmd), rs in sorted(groups.items()):
        fell = sum(r["fell"] for r in rs)
        ok = [r["tilt_p99"] for r in rs if not r["fell"]]
        med = f"{np.median(ok):.1f}" if ok else "n/a"
        print(f'{f"{box} tf32={tf} lead={lead} {cmd}":<30}{len(rs):>3}{f"{fell}/{len(rs)}":>6}{med:>28}')


if __name__ == "__main__":
    main()
