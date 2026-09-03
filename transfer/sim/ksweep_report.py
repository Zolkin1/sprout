#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Read a set of ``live_select.py`` runs as a K sweep: paired on CELL, with the controls.

``live_select.py`` writes one ``live_select.json`` per (tile, speed) cell, holding every
episode of every K arm it ran. This reduces a directory of them to the numbers a best-of-K
claim actually needs, and it exists because three of those numbers are easy to get wrong:

* **The unit of analysis is the CELL, not the episode.** Forty episodes of one tile at one
  speed share a geometry and an approach; they are not forty independent draws. Every interval
  here is a percentile bootstrap over cells, and every arm-to-arm comparison is PAIRED within
  cell before aggregation. Pooling episodes instead reports an interval that is too tight by
  roughly the square root of the episodes per cell.
* **A K arm needs its own control, and "K=1" is not it.** ``live_select`` keys each candidate's
  flow draw as ``[seed, replan, cand]`` while the K=1 arm lets ``policy._rng`` advance
  naturally from the episode seed -- so a K>1 arm runs a DIFFERENT noise tape than K=1 and
  differs from it for that reason alone, worth a measured +0.056 on the same-noise arm. The
  control that isolates the selector is ``--random_select``: it rolls and scores all K
  candidates exactly as usual and then commits a UNIFORMLY RANDOM one, so it differs from the
  real arm in the choice rule and nothing else.

      critic-pick(K) - random-pick(K)   = what the critic's RANKING is worth
      random-pick(K) - K=1              = what the re-keying is worth, and should be ~0
      critic-pick(K) - K=1              = what a sweep usually reports, which is the sum

* **Success is not the objective on its own.** A selector can buy survival by picking slow,
  conservative chunks, and a success rate cannot see it. Time-to-goal is reported beside every
  arm, paired per EPISODE over the episodes that succeeded in both arms being compared -- not
  as a difference of per-arm means, which would be a comparison between different episode sets.

The tile -> family map is read from the clip library's own ``terrain_manifest``, never from a
range table here: tiles 40-59 are ``jump_on`` and a hardcoded "40-44 are stairs" has already
appeared once in a written-up table.

Usage::

    python ksweep_report.py --sweep ../../logs/value/talos_fetch/sweep_v3 \\
        --sweep ../../logs/value/talos_fetch/sweep_v4_k8 \\
        --clip_lib ../../clips/g1_mb_v5_body_mode11_decor.h5 \\
        --out ../../logs/value/ksweep_v3v4 --only_nosteer
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if not os.path.isfile(os.path.join(REPO_ROOT, "scripts", "rsl_rl", "plot_style.py")):
    raise RuntimeError(f"{__file__} derived REPO_ROOT={REPO_ROOT}, which has no "
                       "scripts/rsl_rl/plot_style.py -- the file moved and this derivation "
                       "did not follow it.")
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval"),
           os.path.join(REPO_ROOT, "scripts", "rsl_rl")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: Control steps per re-plan x the control period, i.e. how much sim time one re-plan is.
#: The deployed tracker's ``fm_update_every`` is 12 at 50 Hz. It is a CLI default rather than a
#: constant because `live_select.json` records neither, so a sweep run with a different
#: cadence must be told; `n_replans` itself is reported alongside so the raw unit is visible.
REPLAN_SECONDS = 12 * 0.02


def tile_families(clip_lib: str) -> dict[int, str]:
    """``{tile: family}`` from the library's own manifest.

    Raises:
        SystemExit: If the library cannot be read.  Guessing the map is what produced a
            published table calling the ``jump_on`` tiles "stairs".
    """
    import h5py  # noqa: PLC0415

    if not os.path.isfile(clip_lib):
        raise SystemExit(f"--clip_lib {clip_lib} not readable; the tile -> family map comes "
                         "from the library's terrain_manifest and is not guessed here.")
    with h5py.File(clip_lib, "r") as f:
        raw = f.attrs.get("terrain_manifest")
    if raw is None:
        raise SystemExit(f"{clip_lib} carries no terrain_manifest")
    man = json.loads(raw)
    out = {}
    for i, entry in enumerate(man):
        tid = int(entry.get("terrain_id", i))
        out[tid] = str(entry.get("family", "")).removeprefix("family_").removesuffix(".h5")
    return out


def load_sweeps(dirs: list[str], *, only_nosteer: bool) -> tuple[list[dict], dict]:
    """Every episode row from every ``live_select.json`` under ``dirs``.

    Args:
        dirs: Sweep directories, each holding one subdirectory per cell.
        only_nosteer: Keep cells run with ``steer_kp == 0`` only.  The steering arm is a
            different policy, so pooling it with the flat-command arm would average two.

    Returns:
        ``(rows, meta)``.  Each row gains ``cell``, ``speed``, ``arm`` and ``sweep``.
    """
    rows: list[dict] = []
    meta: dict = {"sweeps": {}, "critics": set(), "arms": set()}
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*", "live_select.json")))
        if not found:
            raise SystemExit(f"no */live_select.json under {d}")
        n_before = len(rows)
        for p in found:
            with open(p) as fh:
                blob = json.load(fh)
            a = blob["args"]
            if only_nosteer and float(a.get("steer_kp", 0.0)) != 0.0:
                continue
            # The choice RULE is part of the arm's identity: a --random_select run and a real
            # run write identical schemas, and pooling them would average the control into
            # the treatment.
            # `select_on` is part of the arm's identity too, and it defaults to "critic" on
            # every file written before that flag existed. A measured-quality arm carries no
            # critic at all, so labelling it "critic" would pool two different SELECTORS
            # under one name -- the same failure the random/same_noise split exists to avoid,
            # and harder to spot because the schemas are identical.
            rule = "random" if a.get("random_select") else (
                "same_noise" if a.get("same_noise") else
                str(a.get("select_on") or "critic"))
            meta["critics"].add(os.path.basename(str(a.get("critic_ckpt", "")) or "none"))
            for r in blob["rows"]:
                k = int(r["k"])
                rows.append({**r, "speed": float(a["speed"]),
                             "cell": f"t{int(r['tile'])}_s{float(a['speed']):g}",
                             # K=1 has no choice to make, so the rule cannot apply to it and
                             # labelling it "random" would create a phantom control arm.
                             "arm": f"K{k}" if k == 1 else f"K{k}_{rule}",
                             "sweep": os.path.basename(os.path.normpath(d))})
        meta["sweeps"][os.path.basename(os.path.normpath(d))] = len(rows) - n_before
    meta["critics"] = sorted(meta["critics"])
    meta["arms"] = sorted({r["arm"] for r in rows})
    return rows, meta


def check_shared_arms(rows: list[dict]) -> list[str]:
    """Warn where two sweeps ran the SAME arm on the same cell, and whether they agree.

    Pooling K=4 from one sweep with K=8 from another is only legitimate if their shared arm
    reproduces -- that is the whole licence for the comparison, and it is cheap to check
    rather than assume.
    """
    seen: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in rows:
        seen[(r["cell"], r["arm"], r["sweep"])].append(r["verdict"])
    by_arm: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    for (cell, arm, sweep), v in seen.items():
        by_arm[(cell, arm)][sweep] = v
    notes, n_same, n_diff = [], 0, 0
    for (cell, arm), per in sorted(by_arm.items()):
        if len(per) < 2:
            continue
        vals = list(per.values())
        if all(v == vals[0] for v in vals[1:]):
            n_same += 1
        else:
            n_diff += 1
            notes.append(f"   {cell} {arm}: sweeps DISAGREE " +
                         ", ".join(f"{s}={sum(1 for x in v if x == 'success')}/{len(v)}"
                                   for s, v in per.items()))
    if n_same or n_diff:
        head = (f"   shared arms across sweeps: {n_same} reproduce EXACTLY, {n_diff} do not")
        notes.insert(0, head)
        if n_diff:
            notes.append("   a disagreeing shared arm means the two sweeps are not on one "
                         "noise tape, and an arm taken from each is not paired.")
    return notes


def per_cell(rows: list[dict], *, replan_seconds: float) -> dict:
    """``{(cell, arm): {...}}`` -- success rate, n, and per-episode outcome maps."""
    acc: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "wins": 0, "success_replans": {}, "verdicts": defaultdict(int)})
    for r in rows:
        a = acc[(r["cell"], r["arm"])]
        a["n"] += 1
        a["verdicts"][r["verdict"]] += 1
        if r["verdict"] == "success":
            a["wins"] += 1
            # Time-to-goal, keyed by EPISODE so the comparison can be paired on it. A mean
            # over each arm's own successes compares different episode sets.
            a["success_replans"][int(r["episode"])] = int(r["n_replans"])
    out = {}
    for key, a in acc.items():
        out[key] = {"n": a["n"], "wins": a["wins"], "success_rate": a["wins"] / max(a["n"], 1),
                    "success_replans": a["success_replans"],
                    "verdicts": dict(a["verdicts"]),
                    "mean_time_to_goal_s": (replan_seconds
                                            * float(np.mean(list(a["success_replans"].values())))
                                            if a["success_replans"] else float("nan"))}
    return out


def boot_ci(v: np.ndarray, *, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile 95% CI for ``mean(v)`` over the CELLS in ``v``.

    **A single cell gets NaN, not a zero-width interval.** Resampling one value returns that
    value every time, so the bootstrap prints ``[x, x]`` -- which reads as certainty where it
    means "no data about the between-cell spread". Same failure mode as a normal interval at
    p = 0 or 1, which `fm_noise_attribution_plots` uses Wilson to avoid.
    """
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if v.size < 2:
        return (float("nan"), float("nan"))
    m = v[rng.integers(0, v.size, size=(reps, v.size))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def contrast(cells: dict, arm_a: str, arm_b: str, *, reps: int, rng: np.random.Generator,
             replan_seconds: float, subset: set[str] | None = None) -> dict:
    """``arm_a - arm_b``, paired within cell, with a cell-clustered interval.

    Time-to-goal is paired per EPISODE within each cell -- only episodes that succeeded in
    BOTH arms contribute -- and then averaged over cells, so a selector that survives by
    slowing down shows up rather than hiding behind a per-arm mean over different episodes.
    """
    names = sorted({c for (c, a) in cells if a in (arm_a, arm_b)
                    and (subset is None or c in subset)})
    paired = [c for c in names if (c, arm_a) in cells and (c, arm_b) in cells]
    if not paired:
        return {"n_cells": 0, "arm_a": arm_a, "arm_b": arm_b}
    da = np.array([cells[(c, arm_a)]["success_rate"] - cells[(c, arm_b)]["success_rate"]
                   for c in paired])
    ra = np.array([cells[(c, arm_a)]["success_rate"] for c in paired])
    rb = np.array([cells[(c, arm_b)]["success_rate"] for c in paired])
    dt_cell, n_pairs = [], 0
    for c in paired:
        sa = cells[(c, arm_a)]["success_replans"]
        sb = cells[(c, arm_b)]["success_replans"]
        both = sorted(set(sa) & set(sb))
        n_pairs += len(both)
        if both:
            dt_cell.append(replan_seconds * float(np.mean([sa[e] - sb[e] for e in both])))
        else:
            dt_cell.append(float("nan"))
    dt_cell = np.array(dt_cell, dtype=np.float64)
    wins = int((da > 0).sum())
    losses = int((da < 0).sum())
    return {"arm_a": arm_a, "arm_b": arm_b, "n_cells": len(paired),
            "rate_a": float(ra.mean()), "rate_b": float(rb.mean()),
            "delta": float(da.mean()), "ci": list(boot_ci(da, reps=reps, rng=rng)),
            "cells_better": wins, "cells_worse": losses,
            "cells_tied": len(paired) - wins - losses,
            "sign_test_p": _sign_p(wins, losses),
            "delta_time_to_goal_s": float(np.nanmean(dt_cell)) if dt_cell.size else float("nan"),
            "ci_time_to_goal_s": list(boot_ci(dt_cell, reps=reps, rng=rng)),
            "n_paired_success_episodes": n_pairs}


def _sign_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test on the cells that moved."""
    n = wins + losses
    if n == 0:
        return float("nan")
    from math import comb  # noqa: PLC0415

    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def stage_report(args) -> int:
    """Load, contrast, print, write."""
    rows, meta = load_sweeps(args.sweep, only_nosteer=args.only_nosteer)
    fams = tile_families(args.clip_lib)
    for r in rows:
        r["family"] = fams.get(int(r["tile"]), "unknown")
    cells = per_cell(rows, replan_seconds=args.replan_seconds)
    cell_family = {}
    for r in rows:
        cell_family[r["cell"]] = r["family"]

    all_cells = sorted({c for (c, _a) in cells})
    fam_of = lambda c: cell_family.get(c, "unknown")                          # noqa: E731
    groups = {"ALL": set(all_cells)}
    for fam in sorted({fam_of(c) for c in all_cells}):
        groups[fam] = {c for c in all_cells if fam_of(c) == fam}
    # `jump_on` reported separately AND excluded from a "no jump_on" aggregate, because a cell
    # whose base rate is 1.000 has nothing to select away and contributes a structural zero.
    if "jump_on" in groups:
        groups["ALL_no_jump_on"] = {c for c in all_cells if fam_of(c) != "jump_on"}

    rng = np.random.default_rng(args.seed)
    arms = meta["arms"]
    ks = sorted({int(a.split("_")[0][1:]) for a in arms})
    print("\n=== K sweep ===")
    print(f"   sweeps      {meta['sweeps']}")
    print(f"   critics     {meta['critics']}")
    print(f"   arms        {arms}")
    print(f"   cells       {len(all_cells)}  families "
          f"{ {f: len(g) for f, g in groups.items() if f not in ('ALL', 'ALL_no_jump_on')} }")
    print(f"   one re-plan = {args.replan_seconds:.3f} s of sim time")
    for note in check_shared_arms(rows):
        print(note)

    # ---- per-arm levels, cell-averaged ------------------------------------------------
    print("\n-- success rate, averaged over cells (not over episodes) --")
    print(f"   {'group':>18s} {'cells':>5s} " + " ".join(f"{a:>14s}" for a in arms))
    levels: dict = {}
    for gname, gset in groups.items():
        line, lv = [], {}
        for a in arms:
            vals = [cells[(c, a)]["success_rate"] for c in sorted(gset) if (c, a) in cells]
            lv[a] = {"n_cells": len(vals),
                     "rate": float(np.mean(vals)) if vals else float("nan"),
                     "ci": list(boot_ci(np.array(vals), reps=args.boot, rng=rng))}
            line.append(f"{lv[a]['rate']:>8.3f}({lv[a]['n_cells']:>3d})" if vals
                        else f"{'--':>14s}")
        levels[gname] = lv
        print(f"   {gname:>18s} {len(gset):>5d} " + " ".join(line))

    # ---- the contrasts ----------------------------------------------------------------
    # Built from the arms PRESENT rather than from a fixed list of rule names: a new
    # selection rule (`--select_on measured` was one) otherwise lands in the levels table and
    # silently in no contrast at all, which is a blank section rather than an error.
    rules = sorted({a.split("_", 1)[1] for a in arms if "_" in a})
    pairs = []
    for k in ks:
        if k == 1:
            continue
        present = [r for r in rules if f"K{k}_{r}" in arms]
        # Every rule against its own control first, then against no selection. `random` is
        # the control for any rule, so it is the SECOND element wherever it exists.
        for r in present:
            if r != "random" and f"K{k}_random" in arms:
                pairs.append((f"K{k}_{r}", f"K{k}_random"))
        for r in present:
            pairs.append((f"K{k}_{r}", "K1"))
    # And each rule against itself at the previous K, to read the K trend.
    prev_k = None
    for k in ks:
        if k == 1:
            prev_k = k
            continue
        for r in rules:
            if prev_k and prev_k != 1 and f"K{k}_{r}" in arms and f"K{prev_k}_{r}" in arms:
                pairs.append((f"K{k}_{r}", f"K{prev_k}_{r}"))
        prev_k = k

    # ---- the DECOMPOSITION, on cells every arm reached ---------------------------------
    # Each contrast above is paired on its own cells, which is right for that contrast and
    # WRONG for reading them as a decomposition: the control arms cover fewer cells than the
    # headline, so `critic - K1` came out +0.098 over 27 cells beside a `critic - random` of
    # +0.167 and a `random - K1` of +0.008 over 9 -- three correct numbers that do not add up,
    # which is an invitation to misread. Restricted to the intersection they must.
    decomp: dict = {}
    for k in ks:
        rd = f"K{k}_random"
        if rd not in arms:
            continue
        # Every rule that has BOTH a control arm and a no-selection arm decomposes the same
        # way, so this is written over the rules present rather than for the critic alone.
        for cr in [f"K{k}_{r}" for r in rules
                   if r != "random" and f"K{k}_{r}" in arms]:
            common = {c for c in all_cells
                      if all((c, a) in cells for a in (cr, rd, "K1"))}
            if not common:
                continue
            rule = cr.split("_", 1)[1]
            print(f"\n-- decomposition of {cr} at K={k}, on the {len(common)} cells all "
                  "three arms reached (these ADD UP) --")
            parts = {f"{rule} ranking ({rule} - random)": (cr, rd),
                     "re-keying (random - K=1)": (rd, "K1"),
                     f"TOTAL ({rule} - K=1)": (cr, "K1")}
            decomp[cr] = {}
            for lbl, (a, b) in parts.items():
                c = contrast(cells, a, b, reps=args.boot, rng=rng,
                             replan_seconds=args.replan_seconds, subset=common)
                decomp[cr][lbl] = c
                print(f"   {lbl:>34s} {c['delta']:>+8.3f} [{c['ci'][0]:>+7.3f},"
                      f"{c['ci'][1]:>+7.3f}]  {c['cells_better']:>2d}/{c['cells_worse']:>2d}/"
                      f"{c['cells_tied']:>2d}  p={c['sign_test_p']:.3f}  "
                      f"dTTG {c['delta_time_to_goal_s']:>+7.3f} s "
                      f"[{c['ci_time_to_goal_s'][0]:>+6.3f},"
                      f"{c['ci_time_to_goal_s'][1]:>+6.3f}]")
            vals = [decomp[cr][lbl]["delta"] for lbl in parts]
            print(f"   {'check: ranking + re-keying':>34s} {vals[0] + vals[1]:>+8.3f}  vs "
                  f"TOTAL {vals[2]:>+8.3f}   residual {vals[0] + vals[1] - vals[2]:+.6f}")
            sn = f"K{k}_same_noise"
            if sn in arms:
                c = contrast(cells, sn, rd, reps=args.boot, rng=rng,
                             replan_seconds=args.replan_seconds, subset=common)
                if c["n_cells"]:
                    print(f"   {'aside: same_noise - random':>34s} {c['delta']:>+8.3f} "
                          f"[{c['ci'][0]:>+7.3f},{c['ci'][1]:>+7.3f}]  <- same_noise re-keys "
                          "AND removes the signal, so it is not the control for a ranking")

    print("\n-- paired contrasts, delta over cells, 95% cell-clustered bootstrap --")
    print(f"   {'contrast':>30s} {'cells':>5s} {'A':>6s} {'B':>6s} {'delta':>8s} "
          f"{'95% CI':>20s} {'w/l/t':>10s} {'sign p':>7s} {'dTTG (s)':>9s} {'CI':>18s}")
    out_contrasts: dict = {}
    for gname in ("ALL", "ALL_no_jump_on"):
        if gname not in groups:
            continue
        print(f"   [{gname}]")
        out_contrasts[gname] = []
        for a, b in pairs:
            c = contrast(cells, a, b, reps=args.boot, rng=rng,
                         replan_seconds=args.replan_seconds, subset=groups[gname])
            if not c["n_cells"]:
                continue
            out_contrasts[gname].append(c)
            print(f"   {a + ' - ' + b:>30s} {c['n_cells']:>5d} {c['rate_a']:>6.3f} "
                  f"{c['rate_b']:>6.3f} {c['delta']:>+8.3f} "
                  f"[{c['ci'][0]:>+7.3f},{c['ci'][1]:>+7.3f}] "
                  f"{c['cells_better']:>3d}/{c['cells_worse']:>2d}/{c['cells_tied']:>2d} "
                  f"{c['sign_test_p']:>7.3f} {c['delta_time_to_goal_s']:>+9.3f} "
                  f"[{c['ci_time_to_goal_s'][0]:>+7.3f},{c['ci_time_to_goal_s'][1]:>+7.3f}]")
    # per family, headline contrast only
    print("\n-- by family, the contrast a sweep usually reports --")
    for gname, gset in sorted(groups.items()):
        if gname in ("ALL", "ALL_no_jump_on"):
            continue
        for a, b in pairs:
            if b != "K1" or not a.endswith("_critic"):
                continue
            c = contrast(cells, a, b, reps=args.boot, rng=rng,
                         replan_seconds=args.replan_seconds, subset=gset)
            if c["n_cells"]:
                print(f"   {gname:>18s} {a + ' - ' + b:>22s} n={c['n_cells']:>3d} "
                      f"{c['rate_b']:.3f} -> {c['rate_a']:.3f}  {c['delta']:>+7.3f} "
                      f"[{c['ci'][0]:>+7.3f},{c['ci'][1]:>+7.3f}]  "
                      f"dTTG {c['delta_time_to_goal_s']:>+7.3f} s")
            out_contrasts.setdefault(gname, []).append(c)

    os.makedirs(args.out, exist_ok=True)
    payload = {"meta": {**meta, "sweep_dirs": args.sweep, "clip_lib": args.clip_lib,
                        "only_nosteer": args.only_nosteer, "seed": args.seed,
                        "bootstrap_reps": args.boot,
                        "replan_seconds": args.replan_seconds},
               "cells": {f"{c}|{a}": {k: v for k, v in d.items()
                                      if k != "success_replans"}
                         for (c, a), d in sorted(cells.items())},
               "cell_family": cell_family, "levels": levels, "contrasts": out_contrasts,
               "decomposition": decomp}
    with open(os.path.join(args.out, "ksweep.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[ksweep] wrote {args.out}/ksweep.json")
    return 0


def main() -> int:
    """Entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="append", default=[], required=True,
                   help="a directory of <cell>/live_select.json. Repeatable; arms are pooled "
                        "across sweeps and any arm two sweeps share is checked for exact "
                        "reproduction, which is what licenses taking one arm from each.")
    p.add_argument("--clip_lib", required=True,
                   help="the library whose terrain_manifest supplies tile -> family. Not "
                        "optional and not guessed: a range table here got tiles 40-59 "
                        "(jump_on) written up as stairs.")
    p.add_argument("--out", required=True)
    p.add_argument("--only_nosteer", action="store_true",
                   help="drop cells run with --steer_kp > 0. The steering arm is a different "
                        "policy and pooling the two averages two experiments.")
    p.add_argument("--replan_seconds", type=float, default=REPLAN_SECONDS,
                   help="sim seconds per re-plan, for time-to-goal. live_select.json records "
                        "neither the cadence nor the control period, so it is stated here; "
                        "the default is the deployed tracker's fm_update_every=12 at 50 Hz.")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    return stage_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
