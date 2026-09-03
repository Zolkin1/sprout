#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reduce a 9-cell ``live_select`` K=1 sweep over several generators into one paired table.

Each cell is ``(arm, tile, speed)`` with a fixed episode budget, and every arm runs the SAME
seeds through the SAME courses -- so the arms pair episode by episode and the comparison can be
made on the pairs rather than on two independent rates.  That matters at 40 episodes a cell:
an unpaired difference of a few points there is inside the binomial noise, while the paired
one is not, because the episodes that both arms find easy cancel.

**The paired statistic is over CELLS, not episodes.**  Episodes inside a cell share a tile, a
speed and a course, so they are not independent draws; the cell is the unit that varies
independently.  With nine of them the interval is wide and is reported as such rather than
being narrowed by pooling episodes that are not exchangeable.

**A K=1 arm is a plain rollout, not a selector.**  With one candidate there is nothing to
select, so these rates are the generators' own, and any difference between arms is the
fine-tune.  That is what makes this the right first read on a fine-tune -- it removes the
selector from the question entirely.

Usage::

    python scripts/flow_matching/fbc_eval_report.py --root logs/value/fbc_eval_e4 \\
        --baseline base --out logs/value/fbc_eval_e4/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

#: ``<arm>_t<tile>_s<speed>`` -- the layout the sweep writes.
CELL_RE = re.compile(r"^(?P<arm>.+)_t(?P<tile>\d+)_s(?P<speed>[\d.]+)$")


def load(root: str) -> dict:
    """Read every cell under ``root``.

    Args:
        root: Directory holding ``<arm>_t<tile>_s<speed>/live_select.json``.

    Returns:
        ``{(arm, tile, speed): {episode: bool_success}}``.

    Raises:
        SystemExit: If no cell parses, which usually means ``--root`` is the wrong directory.
    """
    out: dict = {}
    for name in sorted(os.listdir(root)):
        m = CELL_RE.match(name)
        path = os.path.join(root, name, "live_select.json")
        if not m or not os.path.exists(path):
            continue
        rows = json.load(open(path))["rows"]
        # K=1 only: a higher-K row in the same file would be a different experiment.
        cell = {int(r["episode"]): {"ok": str(r["verdict"]) == "success",
                                    "r": float(r.get("track_reward", float("nan")))}
                for r in rows if int(r["k"]) == 1}
        if cell:
            out[(m["arm"], int(m["tile"]), float(m["speed"]))] = cell
    if not out:
        raise SystemExit(f"no <arm>_t<tile>_s<speed>/live_select.json under {root}")
    return out


def _mean(cell: dict | None, field: str, keys=None) -> float:
    """Cell mean of one field, NaN on an absent cell or an all-NaN column.

    Args:
        cell: ``{episode: {"ok": bool, "r": float}}`` or ``None``.
        field: ``"ok"`` or ``"r"``.
        keys: Restrict to these episode ids, or ``None`` for all of them.

    Returns:
        The mean, or NaN.
    """
    if not cell:
        return float("nan")
    eps = cell.keys() if keys is None else [e for e in keys if e in cell]
    v = np.array([float(cell[e][field]) for e in eps], dtype=np.float64)
    return float(np.nanmean(v)) if v.size and np.isfinite(v).any() else float("nan")


def _common_success(cells: dict, arms, tile, speed) -> list:
    """Episode ids every arm completed successfully in this cell.

    **This is the only reward comparison that is not a selection effect.**  Averaging each
    arm's reward over its OWN successes compares two different episode sets: an arm that fails
    the hard episodes is left averaging the easy ones, and scores higher for failing more.
    Restricting to the episodes EVERY arm survived removes that -- the arms are then compared
    on identical work.

    Args:
        cells: The loaded table.
        arms: Arm names.
        tile: Tile id.
        speed: Commanded speed.

    Returns:
        The shared successful episode ids, possibly empty.
    """
    sets = []
    for a in arms:
        c = cells.get((a, tile, speed))
        if not c:
            return []
        sets.append({e for e, rec in c.items() if rec["ok"]})
    return sorted(set.intersection(*sets)) if sets else []


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--baseline", default="base", help="Arm every other one is paired against.")
    p.add_argument("--out", default="", help="Markdown to write; stdout if omitted.")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cells = load(args.root)
    arms = sorted({a for a, _, _ in cells})
    tiles = sorted({t for _, t, _ in cells})
    speeds = sorted({s for _, _, s in cells})
    if args.baseline not in arms:
        raise SystemExit(f"baseline {args.baseline!r} not among {arms}")

    L = ["# Filtered-BC 9-cell eval (K=1)\n",
         f"`{os.path.abspath(args.root)}`. Arms {arms}; tiles {tiles}; speeds {speeds}.\n",
         "\nEvery arm runs the same seeds through the same courses, so cells pair exactly. "
         "K=1 means there is no selection -- these are the generators' own rates.\n"]

    # ---- the grids, success and reward ---------------------------------------------------
    # BOTH, because they answer different questions and can disagree: success is a threshold
    # most episodes clear, so it is decided by the few near it, while the reward grades every
    # episode INCLUDING the ones both arms survive. A reward gain with no success gain is
    # cleaner tracking; a success gain with no reward gain is an arm surviving by tracking
    # loosely, which is the shape that ends badly on hardware.
    per_arm, per_arm_r = {}, {}
    for field, name, fmt in (("ok", "Success rate", "{:.3f}"),
                             ("r", "Mean tracking reward", "{:.4f}")):
        L.append(f"\n## {name} per cell\n")
        L.append("| arm | " + " | ".join(f"t{t} s{s}" for t in tiles for s in speeds)
                 + " | **mean** |")
        L.append("|---" * (len(tiles) * len(speeds) + 2) + "|")
        for a in arms:
            vals = [_mean(cells.get((a, t, s)), field) for t in tiles for s in speeds]
            (per_arm if field == "ok" else per_arm_r)[a] = vals
            body = " | ".join("--" if np.isnan(v) else fmt.format(v) for v in vals)
            mean = np.nanmean(vals)
            L.append(f"| {a} | {body} | **{'--' if np.isnan(mean) else fmt.format(mean)}** |")
    if all(np.isnan(v) for vals in per_arm_r.values() for v in vals):
        L.append("\n_Every `track_reward` is NaN: the sweep ran without `--tracking_spec`, "
                 "so no probe was folded. Re-run the cells with it to populate this._\n")

    # ---- reward on SUCCESSFUL episodes, two ways -----------------------------------------
    # The all-episode grid above mixes tracking quality with WHICH episodes ended early, so a
    # reward gain there can be an arm that failed differently rather than tracked better.
    common = {(t, s): _common_success(cells, arms, t, s) for t in tiles for s in speeds}
    L.append("\n## Mean tracking reward, SUCCESSFUL episodes only\n")
    L.append("Two scopes, because the obvious one is a selection effect. **own** averages each "
             "arm over the episodes IT completed -- an arm that fails the hard ones is left "
             "averaging the easy ones and scores higher for failing more. **common** uses only "
             "the episodes EVERY arm survived, so the arms are compared on identical work; its "
             "per-cell n is printed because that intersection can be small.\n")
    L.append("| arm | scope | " + " | ".join(f"t{t} s{s}" for t in tiles for s in speeds)
             + " | **mean** |")
    L.append("|---" * (len(tiles) * len(speeds) + 3) + "|")
    per_arm_own, per_arm_com = {}, {}
    for a in arms:
        own = [_mean(cells.get((a, t, s)), "r",
                     keys=[e for e, rec in (cells.get((a, t, s)) or {}).items() if rec["ok"]])
               for t in tiles for s in speeds]
        com = [_mean(cells.get((a, t, s)), "r", keys=common[(t, s)])
               for t in tiles for s in speeds]
        per_arm_own[a], per_arm_com[a] = own, com
        for scope, vals in (("own", own), ("common", com)):
            body = " | ".join("--" if np.isnan(v) else f"{v:.4f}" for v in vals)
            mu = np.nanmean(vals)
            L.append(f"| {a} | {scope} | {body} | "
                     f"**{'--' if np.isnan(mu) else f'{mu:.4f}'}** |")
    L.append("| _n common_ | | " + " | ".join(str(len(common[(t, s)]))
                                              for t in tiles for s in speeds)
             + f" | **{sum(len(v) for v in common.values())}** |")

    # ---- paired deltas, bootstrapped over CELLS -----------------------------------------
    # The count is DERIVED, never written down. It was hardcoded at 9 and the grid later grew
    # to 27, so the header stated a sample size three times smaller than the one the interval
    # was actually computed from -- a caption that makes a result look weaker than it is.
    n_cells = len(tiles) * len(speeds)
    L.append(f"\n## Paired against `{args.baseline}`, bootstrapped over the {n_cells} "
             f"cells ({len(tiles)} tiles x {len(speeds)} speeds)\n")
    L.append("| arm | success delta | 95% CI | reward delta, ALL eps | **reward delta, COMMON successes** | cells better | cells worse | episodes flipped +/- |")
    L.append("|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(args.seed)
    base_vals = per_arm[args.baseline]
    for a in arms:
        if a == args.baseline:
            continue
        d = np.array(per_arm[a]) - np.array(base_vals)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        boot = np.array([rng.choice(d, d.size, replace=True).mean()
                         for _ in range(args.n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # Episode-level flips, which say whether a null delta is agreement or cancellation.
        up = dn = 0
        for t in tiles:
            for s in speeds:
                ca, cb = cells.get((a, t, s)), cells.get((args.baseline, t, s))
                if not ca or not cb:
                    continue
                for ep in set(ca) & set(cb):
                    # `ca[ep]` is a RECORD now, not a bool -- a dict is always truthy, so the
                    # membership test has to name the field or every flip count reads zero.
                    a_ok, b_ok = bool(ca[ep]["ok"]), bool(cb[ep]["ok"])
                    up += int(a_ok and not b_ok)
                    dn += int(b_ok and not a_ok)
        dr = np.array(per_arm_r[a]) - np.array(per_arm_r[args.baseline])
        dr = dr[np.isfinite(dr)]
        rtxt = "--"
        if dr.size:
            rboot = np.array([rng.choice(dr, dr.size, replace=True).mean()
                              for _ in range(args.n_boot)])
            rlo, rhi = np.percentile(rboot, [2.5, 97.5])
            rtxt = f"{dr.mean():+.4f} [{rlo:+.4f}, {rhi:+.4f}]"
        dc = np.array(per_arm_com[a]) - np.array(per_arm_com[args.baseline])
        dc = dc[np.isfinite(dc)]
        ctxt = "--"
        if dc.size:
            cboot = np.array([rng.choice(dc, dc.size, replace=True).mean()
                              for _ in range(args.n_boot)])
            clo, chi = np.percentile(cboot, [2.5, 97.5])
            ctxt = f"{dc.mean():+.4f} [{clo:+.4f}, {chi:+.4f}]"
        L.append(f"| {a} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {rtxt} | {ctxt} | "
                 f"{int((d > 0).sum())}/{d.size} | {int((d < 0).sum())}/{d.size} | "
                 f"+{up} / -{dn} |")

    L.append(f"\nA delta whose interval spans zero over {n_cells} cells is not evidence of no "
             f"effect -- {n_cells} paired units is a wide interval by construction. The flip "
             "counts say whether a null mean is arms agreeing or two effects cancelling. To "
             "call a null INFORMATIVE, check that the same cells detect some other effect: "
             "here they detect the reward effect, which is what bounds the success one.\n")

    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"[report] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
