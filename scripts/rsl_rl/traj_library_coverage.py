#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""What the SOLVED traj-opt library contains, and what the clip recipe asks for that it lacks.

The clip library's command distribution is deliberately wider than the solved library (see
``motion_gen.command_schedule._seg_command``): the command is the driver's envelope, the motion is
whichever solved cell is nearest, and the gap between them is what teaches a generator to SELECT
rather than to memorise. That makes the gap a thing to MEASURE and aim, not to close -- and nothing
measured it. This does.

Two questions, and the second is the useful one:

1. what did the solver actually converge on? The ``status`` column is free text and nothing in the
   ingest chain reads it, so a skill where three quarters of the rows failed looks exactly like one
   where none did (``ingest.clean_ingest.solve_converged`` is the shared test);
2. given a recipe, where does the command support land relative to those cells -- and which cells,
   ordered by how much commanded PROBABILITY MASS they would serve, are missing?

Reads ``manifest.csv`` and nothing else: no h5py, no torch, no IsaacLab, no clip library. That is
deliberate -- it runs under a bare ``python3`` against a solved library that has not been ingested
yet, which is when its answer is worth having::

    python3 scripts/rsl_rl/traj_library_coverage.py hf/trajectories/skills/library_v2
    python3 scripts/rsl_rl/traj_library_coverage.py hf/trajectories/skills/library_v2 \
        --recipe configs/clip_gen/g1_mb_v6.yaml --requests 25
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re

# Substrings that mark a solver row as CONVERGED. Mirrors
# ``motion_gen.ingest.clean_ingest._CONVERGED_MARKERS`` -- duplicated on purpose, because importing
# it would drag in numpy/torch and cost this script its "runs under a bare python3" property. The
# test pins the two lists equal.
CONVERGED_MARKERS = ("locally optimal", "optimal solution found")

# The channels a command has, and the recipe keys that bound each. Kept as data so a recipe change
# does not need a code change here.
_CHANNELS = (("vel_x", "flat_vx_range"), ("vel_y", "flat_vy_range"), ("vel_yaw", "flat_wz_range"))


def converged(status: str) -> bool:
    """Whether a manifest row's solver ``status`` reports a converged solve."""
    s = (status or "").strip().lower()
    return any(m in s for m in CONVERGED_MARKERS)


def read_manifest(path: str) -> list[dict]:
    """Rows of ``<path>/manifest.csv``, or of ``path`` itself if it names the csv."""
    csv_path = path if path.endswith(".csv") else os.path.join(path, "manifest.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"no manifest.csv at {csv_path}")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float | None:
    """A float column, or ``None`` where the solver left it blank.

    Blank is NOT zero and must not be read as zero: ``vel_x`` is empty on essentially every box row,
    and treating that as 0 m/s is what made the box reference pick degenerate to "take the first"
    (see ``ingest.terrain_ingest._measured_steady_vel``).
    """
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def _fmt_cells(vals: list[float]) -> str:
    """A compact rendering of the distinct values on one axis."""
    u = sorted({round(v, 3) for v in vals})
    if not u:
        return "-"
    if len(u) == 1:
        return f"{u[0]:g}"
    steps = {round(b - a, 3) for a, b in zip(u, u[1:])}
    if len(steps) == 1 and len(u) > 2:
        return f"{u[0]:g} .. {u[-1]:g} step {steps.pop():g}"
    return ", ".join(f"{v:g}" for v in u)


def summarise(rows: list[dict]) -> dict:
    """Per-skill convergence and the cells each skill covers."""
    out: dict[str, dict] = {}
    for r in rows:
        sk = (r.get("skill") or "?").strip()
        d = out.setdefault(sk, {"n": 0, "ok": 0, "vx": [], "vy": [], "wz": [],
                                "box_height": [], "tread": []})
        d["n"] += 1
        if not converged(r.get("status", "")):
            continue
        d["ok"] += 1                                    # only CONVERGED rows contribute cells
        for key, dst in (("vel_x", "vx"), ("vel_y", "vy"), ("vel_yaw", "wz"),
                         ("box_height", "box_height"), ("tread", "tread")):
            v = _f(r, key)
            if v is not None:
                d[dst].append(v)
    return out


def print_coverage(summary: dict) -> None:
    """The table: what converged, and what cells that leaves."""
    print(f"\n{'skill':18s} {'converged':>11}  {'vx cells':38s} {'vy':16s} {'wz':16s}")
    print("-" * 105)
    tot_n = tot_ok = 0
    for sk, d in sorted(summary.items()):
        tot_n += d["n"]; tot_ok += d["ok"]
        frac = f"{d['ok']}/{d['n']}"
        flag = "  <-- " + f"{(1 - d['ok'] / d['n']) * 100:.0f}% FAILED" if d["ok"] < d["n"] else ""
        print(f"{sk:18s} {frac:>11}  {_fmt_cells(d['vx']):38s} "
              f"{_fmt_cells(d['vy']):16s} {_fmt_cells(d['wz']):16s}{flag}")
    print("-" * 105)
    print(f"{'TOTAL':18s} {f'{tot_ok}/{tot_n}':>11}"
          f"   ({(1 - tot_ok / max(tot_n, 1)) * 100:.0f}% of rows did not converge and are "
          f"EXCLUDED from every cell above)")
    for sk, d in sorted(summary.items()):
        if d["box_height"]:
            print(f"  {sk}: box heights {_fmt_cells(d['box_height'])}")
        if d["tread"]:
            print(f"  {sk}: treads {_fmt_cells(d['tread'])}")


# --------------------------------------------------------------------------- the recipe half
def read_recipe_ranges(path: str) -> dict:
    """The three command envelopes from a clip recipe, without a yaml dependency.

    A deliberately small parser: these are three flat ``key: [lo, hi]`` lines, and importing yaml
    would cost the script its bare-python3 property for no gain. Raises if a key is missing rather
    than defaulting, because a silently-defaulted envelope would make this report describe a recipe
    nobody ran.
    """
    text = open(path).read()
    out = {}
    for _col, key in _CHANNELS:
        m = re.search(rf"^{key}:\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]", text, re.M)
        if not m:
            raise SystemExit(f"{path}: no `{key}:` line -- is this a v6-era recipe?")
        out[key] = (float(m.group(1)), float(m.group(2)))
    return out


def nearest_cell(cells: list[tuple], target: tuple, scale: tuple) -> tuple:
    """The solved cell nearest ``target`` under a per-channel-normalised metric.

    The SAME shape of metric ``flat_plan.nearest_stride`` uses, so this report and the generator
    agree about which cell a command would actually select. An unweighted distance would trade
    1 rad/s of yaw against 1 m/s of speed and pick visibly wrong cells.
    """
    def d(c):
        return math.sqrt(sum(((a - b) / s) ** 2 for a, b, s in zip(c, target, scale)))
    return min(cells, key=d)


def report_requests(rows: list[dict], ranges: dict, n_req: int, seed: int, draws: int) -> None:
    """Rank the cells the recipe would ask for and the library lacks, by commanded probability mass.

    Monte Carlo over the recipe's own envelope rather than a grid: the envelope is continuous and the
    channels combine freely, so a grid would be a different distribution from the one the clips are
    actually drawn from. Each draw is snapped to its nearest solved cell and the SNAP ERROR recorded;
    a cell's "mass" is the share of draws that land on it, and the request list is the places where
    that share is large and the error is too.
    """
    cells = sorted({(round(_f(r, "vel_x") or 0.0, 3), round(_f(r, "vel_y") or 0.0, 3),
                     round(_f(r, "vel_yaw") or 0.0, 3))
                    for r in rows if converged(r.get("status", "")) and _f(r, "vel_x") is not None})
    if not cells:
        raise SystemExit("no converged rows carry a vel_x -- nothing to snap against")
    scale = tuple(max(1e-6, max(c[i] for c in cells) - min(c[i] for c in cells)) for i in range(3))
    rng = random.Random(seed)
    lo_x, hi_x = ranges["flat_vx_range"]
    lo_y, hi_y = ranges["flat_vy_range"]
    lo_w, hi_w = ranges["flat_wz_range"]

    hits: dict[tuple, list[float]] = {}
    for _ in range(draws):
        vx = rng.uniform(lo_x, hi_x)
        # At most one of vy / wz, matching the sampler exactly (_seg_command).
        pick = rng.randrange(3)
        vy = rng.uniform(lo_y, hi_y) if pick == 1 else 0.0
        wz = rng.uniform(lo_w, hi_w) if pick == 2 else 0.0
        tgt = (vx, vy, wz)
        c = nearest_cell(cells, tgt, scale)
        err = math.dist(c[:2], tgt[:2])              # planar snap error, m/s
        hits.setdefault(c, []).append(err)

    print(f"\n\nWHAT THE RECIPE ASKS FOR ({draws} draws over "
          f"vx{tuple(ranges['flat_vx_range'])} vy{tuple(ranges['flat_vy_range'])} "
          f"wz{tuple(ranges['flat_wz_range'])})")
    print(f"  {len(cells)} converged command cells to snap against\n")
    allerr = sorted(e for v in hits.values() for e in v)
    def pct(p):
        return allerr[min(len(allerr) - 1, int(p * len(allerr)))]
    print(f"  planar snap error over all draws: p50 {pct(0.5):.3f}  p95 {pct(0.95):.3f}  "
          f"max {allerr[-1]:.3f} m/s")
    print("  (this is a DESIGNED disagreement, not an error to drive to zero -- it is what teaches")
    print("   the generator to select. What follows is where it is largest, i.e. worth solving.)\n")

    ranked = sorted(hits.items(), key=lambda kv: -sum(kv[1]))
    print(f"  {'mass':>6} {'p50 err':>8} {'max err':>8}   nearest solved cell (vx, vy, wz)")
    for c, errs in ranked[:n_req]:
        print(f"  {len(errs) / draws:6.1%} {sorted(errs)[len(errs)//2]:8.3f} {max(errs):8.3f}   "
              f"({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})")
    print(f"\n  The rows above absorb the most commanded time. A cell with a LARGE p50 error is one")
    print(f"  the library is serving from far away -- solving a neighbour there buys the most.")


def main() -> int:
    """Parse args, read the manifest, print coverage and (with --recipe) the request list."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("library", help="Solved traj-opt library directory (or its manifest.csv).")
    p.add_argument("--recipe", default=None,
                   help="Clip recipe .yaml; adds the command-support analysis and request list.")
    p.add_argument("--requests", type=int, default=20, help="Rows in the request list.")
    p.add_argument("--draws", type=int, default=20000, help="Monte Carlo draws over the envelope.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = read_manifest(args.library)
    print(f"solved library: {args.library}\n  {len(rows)} manifest rows")
    print_coverage(summarise(rows))
    if args.recipe:
        report_requests(rows, read_recipe_ranges(args.recipe), args.requests, args.seed, args.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
