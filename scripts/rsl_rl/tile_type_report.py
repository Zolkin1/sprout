#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reduce multi-terrain rollouts PER TILE TYPE.

    python scripts/rsl_rl/tile_type_report.py logs/ft_multi/rollouts

``box2_ic_report`` reduces to the two-box initial-condition grid, where ``terrain_id`` indexes
an approach DISTANCE. On a multi-terrain library it indexes a FAMILY instead -- boxes,
up-into-down, the entry stairs in each direction, and the four copied in-distribution families
-- so the same episodes need a different reduction key. This is that reduction, and nothing
else: the success rules, the first-per-env correction and the strict/loose failure rule all
come from ``box2_ic_report.load_cells``, so the two reports cannot disagree about what an
episode was.

**A POOLED RATE OVER A MIX IS NOT A MEASUREMENT OF THE POLICY.** It moves when the MIX moves,
with no policy change at all: raise the entry stairs from 3 tiles to 5 and the pooled number
shifts by the difference between that family's rate and the average, for free. That is the
whole reason the multi-terrain config needs this report -- config 4 can only be compared with
configs 1-3 per TYPE, because those libraries hold one type each. The pooled row is printed
last and labelled, rather than first and headline.

**The interval is TILE-CLUSTERED.** Episodes on one tile share its geometry, so they are not
independent draws, and a binomial interval over pooled episodes is too narrow -- which matters
most exactly where the tile count is smallest. The entry stairs have 5 tiles per direction and
all 5 are the SAME building, so the cluster bootstrap there is resampling 5 identical
geometries: the interval it reports is about the spawn draw and the flow noise, not about
terrain variation, and it will be narrower than a box family's for that reason.

**The family map comes from the library the rollouts were RECORDED on**, resolved from the
episode table's own ``clip_library_path`` stamp rather than from a flag, because reducing a
recording by a different library's map silently mislabels every episode -- the ids all resolve,
the table is full, and every rate is attributed to the wrong terrain.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import h5py
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "source" / "motion_gen"), str(_ROOT / "source" / "robot_rl"),
           str(_HERE), str(_HERE / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from box2_ic_report import METRICS, load_cells  # noqa: E402
from stats import cluster_bootstrap  # noqa: E402

#: Columns reported per family: ``(cell key, label, is a rate?)``.
_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("success", "success", True),
    ("reached_goal", "reached", True),
    ("arrived_before_failing", "arr_b4_fail", True),
    ("failed", "failed", True),
    ("timeout", "timeout", True),
)


def family_map(lib_path: str) -> dict[int, str]:
    """``{terrain_id: family}`` from a library's manifest.

    Args:
        lib_path: The ``.h5``.

    Returns:
        The map.

    Raises:
        SystemExit: If the file is missing or carries no manifest -- a flat-only library has
            none, and reducing per tile type on one is a question with no answer rather than
            an empty table.
    """
    if not os.path.isfile(lib_path):
        raise SystemExit(f"clip library not found: {lib_path}")
    with h5py.File(lib_path, "r") as f:
        raw = f.attrs.get("terrain_manifest")
        if raw is None:
            raise SystemExit(
                f"{lib_path} carries no terrain_manifest, so its tiles have no families and "
                "there is nothing to break down. Use box2_ic_report for a single-type library.")
        manifest = json.loads(raw if isinstance(raw, str) else raw.tobytes().decode())
    return {int(t["terrain_id"]): str(t.get("family") or "unknown")
            for t in manifest if "terrain_id" in t}


def library_from_recording(root: str) -> str:
    """The clip library the rollouts under ``root`` were recorded on.

    Read from the episode table's own provenance rather than taken as a flag: the family map
    is only correct for the library the ids were assigned by, and a wrong one mislabels every
    episode while every id still resolves.

    Args:
        root: Directory holding the per-run rollout subdirectories.

    Returns:
        The stamped path.

    Raises:
        SystemExit: If no table is found, or the tables disagree about which library they ran
            on -- which means the directory holds two different experiments.
    """
    found: set[str] = set()
    for t in sorted(glob.glob(os.path.join(root, "*", "*.episodes.h5"))):
        with h5py.File(t, "r") as h:
            p = h.attrs.get("clip_library_path")
            if p:
                found.add(str(p))
    if not found:
        raise SystemExit(
            f"no *.episodes.h5 under {root!r} carries a clip_library_path stamp; pass --lib.")
    if len(found) > 1:
        raise SystemExit(
            "the recordings under this root ran on DIFFERENT clip libraries:\n  "
            + "\n  ".join(sorted(found))
            + "\nOne family map cannot be correct for both. Reduce them separately.")
    return found.pop()


def by_family(cells: list[dict], fam_of: dict[int, str]) -> dict[tuple[str, str], dict]:
    """Group per-tile cells into per-``(run, family)`` records with clustered intervals.

    Args:
        cells: :func:`box2_ic_report.load_cells` output, one per ``(run, tile)``.
        fam_of: ``{terrain_id: family}``.

    Returns:
        ``{(run, family): record}``.

    Raises:
        SystemExit: If a tile in the recording has no family in the map, which means the map
            came from the wrong library.
    """
    missing = sorted({c["tile"] for c in cells if c["tile"] not in fam_of})
    if missing:
        raise SystemExit(
            f"terrain id(s) {missing} appear in the recording but not in the library's "
            "manifest. The family map is from the wrong library -- every rate it produced "
            "would be attributed to the wrong terrain.")

    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cells:
        groups.setdefault((c["run"], fam_of[c["tile"]]), []).append(c)

    out: dict[tuple[str, str], dict] = {}
    for key, got in groups.items():
        n = np.array([g["n"] for g in got], dtype=np.int64)
        tiles = np.array([g["tile"] for g in got])
        rec: dict = {"n_tiles": len(got), "n_episodes": int(n.sum()),
                     "tiles": sorted(int(t) for t in tiles)}
        for col, _label, is_rate in _COLUMNS:
            vals = np.array([g[col] for g in got], dtype=float)
            if not is_rate:
                rec[col] = float(np.average(vals, weights=n))
                continue
            # Episode-weighted point estimate, tile-clustered interval. The bootstrap resamples
            # TILES, so a family whose tiles agree gets a narrow interval and one whose tiles
            # disagree gets a wide one -- which is the thing a pooled binomial cannot say.
            point, lo, hi, method = cluster_bootstrap(vals, tiles, n, binary=True)
            rec[col] = {"value": float(np.average(vals, weights=n)),
                        "cluster_value": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                        "method": method}
        rec["steps_median"] = float(np.average(
            [g["steps_median"] for g in got], weights=n))
        out[key] = rec
    return out


def format_report(recs: dict[tuple[str, str], dict], *, metric: str, strict: bool,
                  rounds: int, lib_path: str) -> str:
    """The per-tile-type table.

    Args:
        recs: :func:`by_family` output.
        metric: The rule ``success`` was reduced under.
        strict: Whether a tracking violation counted as a failure.
        rounds: The per-env episode cap, ``0`` for pooled.
        lib_path: The library the family map came from.

    Returns:
        The report text.
    """
    rule = ("arrival BEFORE the first failure" if metric == "terminal_reward"
            else "arrival at any point, even after failing")
    red = (f"first {rounds} episode(s) per env" if rounds > 0
           else "POOLED (biased -- see box2_ic_report)")
    L = ["=" * 100, f"PER TILE TYPE -- {metric}", "=" * 100, "",
         f"  success   = {rule}",
         f"  failure   = {'the fall OR the tracking bounds' if strict else 'the fall alone'}",
         f"  reduction = {red}",
         f"  families  = {lib_path}",
         "  interval  = 95% tile-clustered bootstrap (episodes on one tile are not "
         "independent draws)", ""]

    for run in sorted({r for r, _ in recs}):
        fams = sorted(f for r, f in recs if r == run)
        L.append(f"  run {run}")
        L.append(f"    {'family':<26}{'tiles':>6}{'eps':>7}{'success':>10}{'95% CI':>17}"
                 f"{'reached':>9}{'failed':>8}{'timeout':>9}{'steps':>8}")
        tot_n = 0
        tot_ok = 0.0
        for f in fams:
            r = recs[(run, f)]
            s = r["success"]
            L.append(
                f"    {f:<26}{r['n_tiles']:>6}{r['n_episodes']:>7}{s['value']:>10.3f}"
                f"{f'[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]':>17}"
                f"{r['reached_goal']['value']:>9.3f}{r['failed']['value']:>8.3f}"
                f"{r['timeout']['value']:>9.3f}{r['steps_median']:>8.0f}")
            tot_n += r["n_episodes"]
            tot_ok += s["value"] * r["n_episodes"]
        # LAST and LABELLED. It is a property of the library's tile mix as much as of the
        # policy, so it is not the headline and must not be compared across libraries.
        L.append(f"    {'-' * 92}")
        L.append(f"    {'pooled over this mix':<26}{'':>6}{tot_n:>7}"
                 f"{tot_ok / max(tot_n, 1):>10.3f}"
                 f"{'(mix-dependent)':>17}")
        L.append("")

    L += ["  The pooled row moves with the TILE MIX, not only with the policy: changing a",
          "  family's tile count shifts it by that family's distance from the average, with no",
          "  policy change at all. Compare the multi-terrain arm against the single-terrain",
          "  ones PER FAMILY -- those libraries hold one type each, so their pooled row and",
          "  their family row are the same number.", "=" * 100]
    return "\n".join(L)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="directory holding the per-run rollout subdirectories")
    p.add_argument("--lib", default=None,
                   help="Clip library for the family map. Defaults to the one the episode "
                        "tables say they were recorded on, which is the only one that is "
                        "correct -- override only to reduce a copy under a different path.")
    p.add_argument("--json", default="", help="also write the per-family records here")
    p.add_argument("--metric", choices=METRICS, default="terminal_reward",
                   help="Success rule; matches box2_ic_report's flag of the same name.")
    p.add_argument("--failure_rule", choices=("strict", "loose"), default="strict",
                   help="'strict' counts a tracking violation as a failure as well as a fall.")
    p.add_argument("--rounds", type=int, default=1,
                   help="Episodes to keep per env; 0 pools everything (biased).")
    return p.parse_args()


def main() -> int:
    """Reduce and print."""
    args = parse_args()
    lib = args.lib or library_from_recording(args.root)
    cells = load_cells(args.root, metric=args.metric,
                       strict=args.failure_rule == "strict", rounds=args.rounds)
    recs = by_family(cells, family_map(lib))
    print(format_report(recs, metric=args.metric, strict=args.failure_rule == "strict",
                        rounds=args.rounds, lib_path=lib))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({f"{r}|{f}": v for (r, f), v in recs.items()}, fh, indent=2)
        print(f"[tile-type] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
