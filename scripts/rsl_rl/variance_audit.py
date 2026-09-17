# Copyright 2026 Zachary Olkin. All rights reserved.
"""What is inside one eval arm's episode count, and where its variance lives.

Answers four questions the bar chart cannot:

1. **What the count is made of** -- envs x speeds x tiles, and the per-cell count.
2. **Whether the two arms are spawn-MATCHED.** `record_rollouts --seed` defaults to the agent
   config's fixed seed, so both arms draw the same reset stream; if env i's first episode opens
   at the same place in both, the comparison is paired at the ENV level and the unpaired
   interval the figure shows is conservative.
3. **The SPAWN AREA split.** The two-box course declares three spawn areas -- ground, box-1
   top, box-2 top -- at equal weight, so a third of the episodes start already on an obstacle.
   Those are different tasks with different success rates, and the headline rate averages them.
4. **A variance decomposition** of the per-episode outcome: how much sits between cells,
   between spawn areas, and how much is irreducible per-episode Bernoulli spread.
"""

import glob
import json
import argparse
import os
import pathlib
import sys

import h5py
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_p = str(_REPO_ROOT / "scripts" / "flow_matching")
if _p not in sys.path:
    sys.path.insert(0, _p)

import rollout_outcomes as ro  # noqa: E402
from awr_buffer import first_per_env  # noqa: E402


def load_first(root: str, tag: str, tiles: range | None = None, *,
               metric: str = "terminal_reward", strict: bool = True) -> dict:
    """First episode per env, with its opening downrange, per (tile, speed) cell.

    Args:
        root: Directory holding the eval run directories.
        tag: The run directory under ``root``.
        tiles: Restrict to these ``terrain_id`` values.
        metric: Success rule; ``terminal_reward`` or ``reached_goal``.
        strict: Count a tracking violation as a failure as well as a fall.

    Returns:
        ``{(tile, speed, env): (success, opening downrange)}``.
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(root, tag, "train_v*"))):
        if not os.path.isdir(d):
            continue
        speed = float(d.rsplit("train_v", 1)[1])
        f = glob.glob(d + "/*.episodes.h5")
        if not f:
            continue
        with h5py.File(f[0], "r") as h:
            # The SAME selection and success rule the figure and the report use, so this
            # diagnostic describes their sample rather than a neighbouring one.
            keep = ~h["censored"][:].astype(bool)
            uid_all = h["episode_uid"][:].astype(np.int64)
            env_all = h["env_id"][:].astype(np.int64)
            keep &= np.isin(uid_all, np.asarray(
                first_per_env(uid_all[keep], env_all[keep], 1), dtype=np.int64))
            cols = {k: h[k][:][keep] for k in h if isinstance(h[k], h5py.Dataset)}
            c = h["chunks"]
            cuid = np.asarray(c["episode_uid"][...], np.int64)
            cidx = np.asarray(c["replan_index"][...], np.int64)
            cdr = np.asarray(c["goal_downrange_m"][...], float)
        env = cols["env_id"].astype(np.int64)
        tile = cols["terrain_id"].astype(np.int64)
        uid = cols["episode_uid"].astype(np.int64)
        goal = (cols["reached_goal"].astype(float) if metric == "reached_goal"
                else (ro.terminal_reward(cols, strict=strict) >= 0.0).astype(float))
        first = np.arange(goal.size)
        # The OPENING downrange of each episode: the first re-plan is made at the spawn, so
        # this is where the robot started, in the goal's own frame.
        m0 = cidx == 0
        open_dr = dict(zip(cuid[m0].tolist(), cdr[m0].tolist()))
        for i in first:
            if tiles is not None and int(tile[i]) not in tiles:
                continue
            out[(int(tile[i]), speed, int(env[i]))] = (
                float(goal[i]), open_dr.get(int(uid[i]), float("nan")))
    return out


def audit(name: str, A: dict, B: dict, n_areas: int) -> dict:
    """Print the breakdown for one cluster and return its numbers."""
    keys = sorted(set(A) & set(B))
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    envs = {k[2] for k in keys}
    speeds = sorted({k[1] for k in keys})
    tids = sorted({k[0] for k in keys})
    print(f"  episodes/arm     {len(keys)}  =  {len(envs)} envs x {len(speeds)} speeds"
          f"   (tiles {len(tids)}, speeds {speeds})")
    print(f"  per (tile,speed) {len(keys) / (len(tids) * len(speeds)):.1f} envs on average")

    # --- 2. spawn matching between arms -----------------------------------------------------
    da = np.array([A[k][1] for k in keys])
    db = np.array([B[k][1] for k in keys])
    ok = np.isfinite(da) & np.isfinite(db)
    exact = float(np.mean(np.abs(da[ok] - db[ok]) < 1e-9))
    print(f"  spawn-matched    {exact * 100:.1f}% of envs open at an IDENTICAL downrange in "
          f"both arms (max |diff| {np.abs(da[ok] - db[ok]).max():.2e} m)")

    # --- 3. spawn-area split ----------------------------------------------------------------
    # The areas are far apart downrange, so a k-means-free split at the gaps recovers them.
    print(f"  spawn areas declared by the course: {n_areas}")
    ga = np.array([A[k][0] for k in keys])
    gb = np.array([B[k][0] for k in keys])
    per_tile_areas = {}
    for t in tids:
        sel = np.array([k[0] == t for k in keys])
        d = da[sel & ok]
        if d.size < 10:
            continue
        edges = np.histogram_bin_edges(d, bins=60)
        cnt, _ = np.histogram(d, bins=edges)
        # Cut wherever an empty bin separates populated ones: the areas are metres apart.
        cuts = [i for i in range(1, len(cnt)) if cnt[i - 1] > 0 and cnt[i] == 0]
        groups = 1 + sum(1 for i in cuts if cnt[i:].sum() > 0)
        per_tile_areas[t] = min(groups, n_areas)
    if per_tile_areas:
        print(f"  spawn clusters found in the data, per tile: "
              f"{sorted(set(per_tile_areas.values()))}")
    # Rate by opening-downrange TERCILE, which is the area split when there are three.
    qs = np.nanquantile(da[ok], [1 / 3, 2 / 3]) if n_areas == 3 else np.nanquantile(da[ok], [0.5])
    grp = np.digitize(da, qs)
    print(f"  success by opening position (near the goal -> far):")
    for g in range(len(qs) + 1):
        sel = (grp == g) & ok
        if not sel.any():
            continue
        print(f"    group {g}  n={int(sel.sum()):5d}  open_dr "
              f"[{da[sel].min():7.2f},{da[sel].max():7.2f}] m   "
              f"parent {ga[sel].mean():.4f}  fine-tuned {gb[sel].mean():.4f}")

    # --- 4. variance decomposition ----------------------------------------------------------
    print("  variance of the per-episode outcome (fine-tuned arm):")
    tot = float(gb.var())
    cellmean = {}
    for k, v in zip(keys, gb):
        cellmean.setdefault((k[0], k[1]), []).append(v)
        
    cm = np.array([np.mean(v) for v in cellmean.values()])
    wts = np.array([len(v) for v in cellmean.values()], float)
    between_cell = float(np.average((cm - gb.mean()) ** 2, weights=wts))
    within_cell = tot - between_cell
    print(f"    total            {tot:.5f}")
    print(f"    between (tile,speed) cells {between_cell:.5f}  "
          f"({between_cell / tot * 100:.1f}%)")
    print(f"    within cells     {within_cell:.5f}  ({within_cell / tot * 100:.1f}%)"
          f"   <- per-episode: DR draw + FM noise + spawn jitter + simulator")
    # Within a cell, how much does the OPENING POSITION explain? That is the spawn-area share
    # of the within-cell spread, i.e. structure the figure pools over.
    res = []
    for (t, s), _ in cellmean.items():
        sel = np.array([(k[0], k[1]) == (t, s) for k in keys]) & ok
        if sel.sum() < 20:
            continue
        d = da[sel]
        q = np.nanquantile(d, [1 / 3, 2 / 3]) if n_areas == 3 else np.nanquantile(d, [0.5])
        g = np.digitize(d, q)
        y = gb[sel]
        mu = y.mean()
        bg = sum((y[g == j].size) * (y[g == j].mean() - mu) ** 2 for j in np.unique(g)) / y.size
        res.append((bg, y.var()))
    if res:
        bg = float(np.mean([r[0] for r in res]))
        wc = float(np.mean([r[1] for r in res]))
        print(f"    of that, opening position explains {bg / wc * 100:.1f}% "
              f"(spawn area; the rest is per-episode)")
    return {"n": len(keys), "envs": len(envs), "speeds": len(speeds), "tiles": len(tids),
            "spawn_matched_frac": exact, "var_total": tot,
            "var_between_cell": between_cell, "var_within_cell": within_cell}


#: ``label -> (parent, fine-tuned, tile filter, declared spawn areas)``, matching
#: ``fig_reduce.DEFAULT_ARMS``. The area count is the COURSE's, from its terrain generator.
CLUSTERS: tuple[tuple[str, str, str, range | None, int], ...] = (
    ("Two boxes (OOD)", "eval_parent_prog_s3", "eval_prog_s3_i5", None, 3),
    ("Stairs up-into-down (OOD)", "eval_parent_updn_s3", "eval_updn_s3_i5", None, 2),
    ("Stairs down (in-distribution)", "eval_parent_indist", "eval_indist_i5",
     range(20, 40), 1),
)


def main() -> int:
    """Audit every cluster's episode count and variance."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="logs/box2/quick")
    p.add_argument("--metric", default="terminal_reward",
                   choices=("terminal_reward", "reached_goal"))
    p.add_argument("--failure_rule", choices=("strict", "loose"), default="strict")
    p.add_argument("--out", default="variance_audit.json")
    a = p.parse_args()
    kw = dict(metric=a.metric, strict=a.failure_rule == "strict")
    out = {}
    for nm, pt, ft, tl, na in CLUSTERS:
        out[nm] = audit(nm, load_first(a.root, pt, tl, **kw),
                        load_first(a.root, ft, tl, **kw), na)
    out["_meta"] = {"metric": a.metric, "failure_rule": a.failure_rule,
                    "selection": "each env's first completed episode"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
