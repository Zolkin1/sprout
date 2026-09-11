# Copyright 2026 Zachary Olkin. All rights reserved.

"""Statistics for the mjeval steering sweep, in the eval harness's own idiom.

Reads the ``live_select.json`` written by every cell of the sweep (9 tiles x 3 speeds x
{steer, no-steer} x {K=1, K=4} x 40 episodes) and answers three questions with the SAME
machinery ``scripts/rsl_rl/eval/report.py`` uses -- ``stats.paired_delta``, a tile-clustered
bootstrap with a whole-tile sign-flip permutation -- so these numbers are method-comparable to
the Isaac harness's rather than a second invention.

**The pairing is stronger here than in the Isaac harness, and that is a property of the
setup rather than a cleverer test.**  ``paired_delta``'s docstring says it is "deliberately not
episode-paired: episode *i* of a clip in one cell is a different draw from episode *i* in
another".  In mjeval's live selector the episode seed IS the flow-noise seed
(``policy._rng = default_rng(seed)``), the spawn is byte-identical, the command is constant and
MuJoCo is deterministic -- so episode *i* of the steer arm and episode *i* of the no-steer arm
run the same noise tape from the same state, and only diverge because the command differs.
Pairing on ``(tile, speed, seed)`` therefore removes the draw's contribution to the contrast,
which is the dominant variance component here (see below).

**What "the effect of the seed" means in this design.**  Within one cell NOTHING varies except
the noise draw -- no DR, no depth noise, no clip, one spawn, one command.  So the entire
within-cell success variance IS the seed's doing, and asking "how big is the seed effect" in
the usual sense is answered by construction.  The question that is NOT answered by construction,
and that :func:`seed_structure` measures, is whether a draw is good *in general* or only
good *here*: if some seeds succeed across many different cells and others fail across many, the
noise has a shared quality axis and per-seed rates are over-dispersed relative to Binomial; if
the draw only interacts with the state it met, per-seed rates are Binomial and the dispersion
index sits at 1.  The second is the interesting outcome for best-of-K -- it means a candidate
cannot be pre-screened as "a good draw" independently of the state it will be executed from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _d in (os.path.join(_ROOT, "scripts", "rsl_rl", "eval"),
           os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import stats as hstats  # noqa: E402  the eval harness's own estimators

#: Tile-id ranges -> family, read off ``g1_mb_v5_body_mode11_decor.h5``'s ``terrain_manifest``
#: (contiguous blocks, one per family file).  A RANGE rather than a per-tile dict on purpose:
#: the first version of this listed only the nine phase-1 tiles, so every phase-2 tile fell
#: through to "?" and silently vanished from the per-family breakdown while the pooled numbers
#: stayed right -- the kind of hole that is invisible unless a family total is checked against
#: the episode count.
FAMILY_RANGES = ((0, 19, "stairs_up"), (20, 39, "stairs_down"), (40, 59, "jump_on"),
                 (60, 74, "jump_off"), (75, 89, "jump_off_front"),
                 (90, 109, "distractor"), (110, 124, "stairs_dn_pure"),
                 (125, 139, "stairs_up_pure"))


def family_of(tile: int) -> str:
    """Family name for a tile id.

    Args:
        tile: Terrain id from the clip library's manifest.

    Returns:
        The family name, or ``"unknown"`` outside every known block.
    """
    for lo, hi, name in FAMILY_RANGES:
        if lo <= tile <= hi:
            return name
    return "unknown"


class _FamilyMap(dict):
    """``FAMILY[tile]`` kept working as a mapping, now backed by the ranges."""

    def get(self, tile, default=None):        # noqa: D102
        return family_of(int(tile))

    def __getitem__(self, tile):              # noqa: D105
        return family_of(int(tile))


FAMILY = _FamilyMap()


def load_cells(root: str) -> list[dict]:
    """Read every finished cell of the sweep into flat episode records.

    Args:
        root: The sweep directory holding ``t<tile>_s<speed>_<arm>/live_select.json``.

    Returns:
        One dict per EPISODE: tile, family, speed, steer, k, seed, success.
    """
    out = []
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d, "live_select.json")
        if not os.path.isfile(path):
            continue
        try:
            blob = json.load(open(path))
        except json.JSONDecodeError:          # a cell still being written
            continue
        a = blob.get("args", {})
        speed = float(a.get("speed", float("nan")))
        steer = float(a.get("steer_kp", 0.0)) > 0.0
        for r in blob["rows"]:
            out.append({"tile": int(r["tile"]), "family": FAMILY.get(int(r["tile"]), "?"),
                        "speed": speed, "steer": steer, "k": int(r["k"]),
                        "seed": int(r["episode"]), "success": r["verdict"] == "success"})
    return out


def paired(rows: list[dict], k: int, *, split: str | None = None, boot: int = 4000,
           seed: int = 0) -> dict:
    """Steer-minus-no-steer, paired on ``(tile, speed, seed)`` and clustered on tile.

    Args:
        rows: Episode records from :func:`load_cells`.
        k: Which K arm to contrast.
        split: Optional grouping key (``"family"`` or ``"speed"``); ``None`` pools everything.
        boot: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        ``{group: Estimate.as_dict()}``.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["k"] != k:
            continue
        groups["all" if split is None else r[split]].append(r)

    out = {}
    for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        arms = {True: {}, False: {}}
        for r in rs:
            arms[r["steer"]][(r["tile"], r["speed"], r["seed"])] = r["success"]
        common = sorted(set(arms[True]) & set(arms[False]))
        if not common:
            continue
        # `paired_delta` groups by its "clip" key and pairs on it; one episode per key here,
        # so the group mean IS that episode's 0/1 outcome and the pairing is episode-level.
        pair_ids = np.array([f"{t}_{s}_{e}" for t, s, e in common])
        tiles = np.array([t for t, _s, _e in common])
        va = np.array([float(arms[False][c]) for c in common])
        vb = np.array([float(arms[True][c]) for c in common])
        est = hstats.paired_delta(pair_ids, tiles, va, pair_ids, tiles, vb,
                                  boot=boot, seed=seed)
        d = est.as_dict()
        d["mean_nosteer"], d["mean_steer"] = float(va.mean()), float(vb.mean())
        out[g] = d
    return out


def seed_structure(rows: list[dict], *, boot: int = 4000, seed: int = 0,
                  one_arm: bool = True) -> dict:
    """Is a noise draw good IN GENERAL, or only good where it landed?

    Each cell fixes everything but the draw, so a seed's outcome in one cell is one Bernoulli
    trial at that cell's own rate.  Pooling a seed over all cells and comparing the spread of
    per-seed rates against the Binomial expectation therefore tests exactly one thing: whether
    seed identity carries any quality that transfers across states.

    The null is generated by permuting seed labels WITHIN each cell, which holds every cell's
    success rate fixed and destroys only the correspondence between cells -- so a significant
    result cannot be manufactured by some cells being easier than others.

    **``one_arm`` is not a convenience, it is the whole validity of the test.**  A seed appears
    at the same ``(tile, speed)`` in four cells -- steer/no-steer x K=1/K=4 -- and those four
    run the SAME noise tape from the SAME spawn, so they are near-duplicates rather than
    independent states.  Pooling them while permuting labels independently per cell destroys
    that legitimate correlation in the null but keeps it in the observed statistic, which
    manufactures over-dispersion out of replication alone: measured here, pooling all cells
    gives an index of 3.71 against 1.30 for the one-arm restriction on the same data.  So the
    default keeps ONE cell per ``(tile, speed)`` and the pooled number is reported only as the
    inflated comparison it is.

    Args:
        rows: Episode records.
        boot: Permutation replicates.
        seed: RNG seed.
        one_arm: Keep only the no-steer K=1 cells, so each seed meets each state once.

    Returns:
        Observed and null between-seed variance, a dispersion index, and a permutation p.
    """
    if one_arm:
        rows = [r for r in rows if not r["steer"] and r["k"] == 1]
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["tile"], r["speed"], r["steer"], r["k"])][r["seed"]] = r["success"]
    seeds = sorted({s for c in cells.values() for s in c})
    idx = {s: i for i, s in enumerate(seeds)}

    mat = np.full((len(seeds), len(cells)), np.nan)
    for j, c in enumerate(cells.values()):
        for s, ok in c.items():
            mat[idx[s], j] = float(ok)

    per_seed = np.nanmean(mat, axis=1)
    observed = float(np.nanvar(per_seed))

    rng = np.random.default_rng(seed)
    null = np.empty(boot)
    for b in range(boot):
        shuf = mat.copy()
        for j in range(shuf.shape[1]):
            col = shuf[:, j]
            ok = ~np.isnan(col)
            v = col[ok]
            rng.shuffle(v)
            col[ok] = v
        null[b] = float(np.nanvar(np.nanmean(shuf, axis=1)))

    p = float((null >= observed).mean())

    # EFFECT SIZE, in the same units as the Isaac spec's sigma_seed (per-cell success rate).
    # The observed between-seed variance is (true seed variance + within-cell sampling noise);
    # the permuted null IS that sampling noise with the seed effect removed, so the difference
    # is the seed component. No simulator-floor term appears because this sim is run-to-run
    # reproducible (measured 12/12 identical verdicts across two concurrent processes), i.e.
    # sigma_run = 0 -- the term the Isaac spec has to estimate from same-seed repeats.
    sigma_seed = float(np.sqrt(max(0.0, observed - float(null.mean()))))

    # Interval by resampling CELLS (the independent unit here: one state each), not seeds --
    # the question is how well this seed ranking generalises to other states.
    rng2 = np.random.default_rng(seed + 1)
    ncell = mat.shape[1]
    boot_sigma, boot_index = [], []
    for _ in range(min(boot, 2000)):
        cols = rng2.integers(0, ncell, ncell)
        m = mat[:, cols]
        obs_b = float(np.nanvar(np.nanmean(m, axis=1)))
        shuf = m.copy()
        for j in range(shuf.shape[1]):
            col = shuf[:, j]
            ok = ~np.isnan(col)
            v = col[ok]
            rng2.shuffle(v)
            col[ok] = v
        null_b = float(np.nanvar(np.nanmean(shuf, axis=1)))
        boot_sigma.append(np.sqrt(max(0.0, obs_b - null_b)))
        boot_index.append(obs_b / null_b if null_b > 0 else np.nan)
    lo_s, hi_s = np.nanpercentile(boot_sigma, [2.5, 97.5])
    lo_i, hi_i = np.nanpercentile(boot_index, [2.5, 97.5])

    return {"n_seeds": len(seeds), "n_cells": len(cells),
            "grand_mean": float(np.nanmean(mat)),
            "var_between_seeds": observed,
            "null_var_mean": float(null.mean()),
            "dispersion_index": float(observed / null.mean()) if null.mean() > 0 else float("nan"),
            "dispersion_ci": [float(lo_i), float(hi_i)],
            "sigma_seed": sigma_seed, "sigma_seed_ci": [float(lo_s), float(hi_s)],
            "sigma_run": 0.0,
            "permutation_p": p}


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, help="sweep_v1 directory")
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json_out", default="")
    args = p.parse_args()

    rows = load_cells(args.sweep)
    if not rows:
        raise SystemExit(f"no finished cells under {args.sweep}")
    cells = {(r["tile"], r["speed"], r["steer"], r["k"]) for r in rows}
    print(f"[sweep] {len(rows):,} episodes over {len(cells)} cells, "
          f"{len({r['tile'] for r in rows})} tiles, "
          f"speeds {sorted({r['speed'] for r in rows})}\n")

    report = {"n_episodes": len(rows), "n_cells": len(cells)}

    def show(title, res):
        print(title)
        print(f"  {'group':<16} {'no-steer':>9} {'steer':>7} {'delta':>8} "
              f"{'95% CI':>18} {'perm p':>8} {'tiles':>6}")
        for g, e in res.items():
            lo, hi = e.get("ci_lo"), e.get("ci_hi")
            ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None and np.isfinite(lo) else "     --      "
            pp = e.get("extra", {}).get("permutation_p", float("nan"))
            print(f"  {str(g):<16} {e['mean_nosteer']:>9.3f} {e['mean_steer']:>7.3f} "
                  f"{e['value']:>+8.3f} {ci:>18} {pp:>8.3f} {e.get('n_tiles', 0):>6}")
        print()

    for k in (1, 4):
        report[f"k{k}_overall"] = paired(rows, k, boot=args.boot, seed=args.seed)
        show(f"STEERING EFFECT, K={k} (paired on tile/speed/seed, tile-clustered)",
             report[f"k{k}_overall"])
        report[f"k{k}_family"] = paired(rows, k, split="family", boot=args.boot, seed=args.seed)
        show(f"  by family, K={k}", report[f"k{k}_family"])
        report[f"k{k}_speed"] = paired(rows, k, split="speed", boot=args.boot, seed=args.seed)
        show(f"  by speed, K={k}", report[f"k{k}_speed"])

    ss = seed_structure(rows, boot=args.boot, seed=args.seed, one_arm=True)
    pooled = seed_structure(rows, boot=args.boot, seed=args.seed, one_arm=False)
    report["seed_structure"] = ss
    report["seed_structure_pooled_INFLATED"] = pooled
    print("SEED STRUCTURE -- does a draw transfer across states?")
    print(f"  {ss['n_seeds']} seeds x {ss['n_cells']} cells (no-steer, K=1: each seed meets "
          f"each state once), grand mean {ss['grand_mean']:.3f}")
    print(f"  between-seed variance {ss['var_between_seeds']:.5f} against a "
          f"within-cell-permuted null of {ss['null_var_mean']:.5f}")
    print(f"  dispersion index {ss['dispersion_index']:.3f} "
          f"[{ss['dispersion_ci'][0]:.2f},{ss['dispersion_ci'][1]:.2f}]   "
          f"permutation p {ss['permutation_p']:.4f}")
    print(f"  sigma_seed {ss['sigma_seed']:.4f} "
          f"[{ss['sigma_seed_ci'][0]:.4f},{ss['sigma_seed_ci'][1]:.4f}] in per-cell "
          f"success-rate units, against sigma_run = 0 (this sim is run-to-run reproducible)")
    print(f"  [pooling all {pooled['n_cells']} cells instead gives "
          f"{pooled['dispersion_index']:.3f} -- INFLATED, see seed_structure's docstring: the")
    print("   four cells sharing a (tile, speed) run one noise tape and are not independent.]")
    print("  index ~1 = the draw only interacts with the state it met (a candidate cannot be")
    print("  pre-screened as 'a good draw'); >1 = seed identity carries transferable quality.\n")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"[sweep] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
