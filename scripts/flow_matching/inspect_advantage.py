# Copyright 2026 Zachary Olkin. All rights reserved.

"""Ask what the advantage label actually rewards, on the two-box course.

Filtered BC and AWR both assume the label points at the decision.  On a course whose failure
is a MODE CHOICE -- mount the box or not -- that assumption has two ways to fail, and this
measures both against the realized outcome rather than against itself.

**1. WHERE the label puts its magnitude.**  If only a handful of chunks decide an episode, the
label has to concentrate on them.  It does not: measured on the v2 labelling, ``|advantage|``
is 0.166 approaching the first box against 0.199 seven to nine metres out, which is flat
ground with nothing to decide.  The cause is structural rather than statistical --
``advantage.binarize`` takes its percentile INSIDE each (family x progress x speed) cell, so
exactly 30% of every cell is positive, the pure-approach cells included.  Stratifying is what
stops the keep-set skewing towards fast commands; the cost is that it cannot concentrate.

**2. WHETHER the label prefers the right MODE.**  ``chunk_reward`` is a kernel on
``track_err_prev``, so at ``--w_fail 0`` the advantage is improvement in expected TRACKING
return.  Climbing a 0.40 m box tracks worse than walking flat whether or not it is correct, so
the label can penalise the mode the course requires.  Measured at the first box: the label
prefers not-climbing at ``corr = -0.285`` while the OUTCOME prefers it at only ``-0.050`` --
a 5.7x over-penalty, and 7.9% of climbers labelled positive against 39.7% of non-climbers for
realized success rates of 9.5% and 13.2%.

**The outcome column is the point.**  Every number here is paired with what actually happened
on the same windows, because "the label dislikes climbing" is only a defect if climbing works
better than the label says.

Usage::

    python scripts/flow_matching/inspect_advantage.py \\
        --rollout logs/box2/merged/box2_train \\
        --labels logs/box2/critic_v2_loose/adv_p70_w0_strat.npz \\
        --predictions logs/box2/critic_v2_loose/predictions.npz \\
        --json out.json --dump_pairs pairs.json
"""

from __future__ import annotations

import argparse
import json
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

from motion_gen.terrain import box2 as B  # noqa: E402
from robot_rl.flow_matching.advantage import LABEL_NULL  # noqa: E402

#: ``base_pose[0:3]`` is the reference base position in the plan's yaw frame, so channel 2 is
#: the planned rise above the anchor -- the mode signal, in metres, needing no FK.
_Z_CHANNEL = 2


def landmarks() -> dict[str, float]:
    """Metres-to-goal of each course landmark, derived from the terrain module.

    Returns:
        Landmark name -> distance still to travel when the base is at it.
    """
    g = B.TARGET_LEAD_M - B.GOAL_DEPTH_M / 2.0        # goal line, past box 2's far edge
    return {
        "box2_far": g,
        "box2_near": g + B.BOX_DEPTH_M,
        "box1_far": g + B.BOX_DEPTH_M + B.BOX_GAP_M,
        "box1_near": g + 2 * B.BOX_DEPTH_M + B.BOX_GAP_M,
    }


def join(side: dict, uid: np.ndarray, ridx: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                 np.ndarray]:
    """Expand a sidecar onto the window rows on ``(episode_uid, replan_index)``.

    Args:
        side: The advantage ``.npz``.
        uid: ``[W]`` window-file episode ids.
        ridx: ``[W]`` window-file re-plan indices.

    Returns:
        ``(advantage, label, hit)`` -- NaN / ``LABEL_NULL`` / False where the sidecar has no
        row.  A miss is flagged rather than filled: 0.0 is an ordinary advantage.
    """
    k = lambda u, r: (u.astype(np.int64) << np.int64(20)) + r.astype(np.int64)  # noqa: E731
    ks, kf = k(side["episode_uid"], side["replan_index"]), k(uid, ridx)
    order = np.argsort(ks)
    pos = np.clip(np.searchsorted(ks[order], kf), 0, max(ks.size - 1, 0))
    hit = ks[order][pos] == kf
    adv = np.full(kf.size, np.nan)
    lab = np.full(kf.size, LABEL_NULL, dtype=np.int64)
    adv[hit] = side["advantage"][order][pos[hit]]
    lab[hit] = side["label"].astype(np.int64)[order][pos[hit]]
    return adv, lab, hit


def planned_climb(f: h5py.File, rows: np.ndarray, block: int = 2000) -> np.ndarray:
    """The peak planned rise above the plan anchor, per window, in metres.

    Read in blocks because the window table is tens of GB and only one channel is needed.

    Args:
        f: The open window file.
        rows: Row indices, ASCENDING (h5py fancy indexing requires it).
        block: Rows per read.

    Returns:
        ``[len(rows)]`` metres.
    """
    out = np.empty(rows.size)
    for i in range(0, rows.size, block):
        j = rows[i:i + block]
        out[i:i + block] = f["fm_plan"][j][:, :, _Z_CHANNEL].max(axis=1)
    return out


def main() -> int:
    """Print the two diagnostics and optionally dump matched pairs."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", required=True, help="rollout stem (no .h5)")
    p.add_argument("--labels", required=True, help="advantage .npz")
    p.add_argument("--predictions", default="",
                   help="predictions.npz -- enables the p_fail-MATCHED mode comparison, "
                        "which is the controlled version of it. Without one the mode numbers "
                        "are correlational within a downrange band.")
    p.add_argument("--band", type=float, nargs=2, default=None,
                   help="metres-to-goal window for the mode test; defaults to the first "
                        "box's approach, [box1_near, box1_near + 1.1]")
    p.add_argument("--max_rows", type=int, default=20000,
                   help="cap on windows read for the mode test (fm_plan is the expensive read)")
    p.add_argument("--pairs", type=int, default=10, help="matched pairs to dump")
    p.add_argument("--p_band", type=float, default=0.05,
                   help="how close two chunks' p_fail must be to be a pair")
    p.add_argument("--json", default="", help="write the summary here")
    p.add_argument("--dump_pairs", default="",
                   help="write the matched pairs' chunk geometry here, for a viewer")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    lm = landmarks()
    with h5py.File(a.rollout + ".h5", "r") as f, \
            h5py.File(a.rollout + ".episodes.h5", "r") as ef:
        n = int(f.attrs.get("num_windows_written", f["episode_uid"].shape[0]))
        uid, ridx = f["episode_uid"][:n], f["replan_index"][:n]
        adv, lab, hit = join(dict(np.load(a.labels)), uid, ridx)
        d = -f["goal_downrange_m"][:n]          # stored as robot MINUS goal, so negate
        ok = hit & (lab != LABEL_NULL)
        reached = {int(u): bool(r) for u, r in
                   zip(ef["episode_uid"][:], ef["reached_goal"][:].astype(bool))}
        succ = np.array([reached.get(int(u), False) for u in uid])

        out: dict = {"n_labelled": int(ok.sum()), "landmarks": lm, "bands": [], "mode": {}}
        print(f"[adv] {int(ok.sum())} labelled windows;  landmarks (metres to goal): "
              + "  ".join(f"{k}={v:.2f}" for k, v in lm.items()))

        # --- 1. concentration -------------------------------------------------------
        print("\nWHERE THE LABEL PUTS ITS MAGNITUDE")
        print(f"  {'metres to goal':>16}{'n':>8}{'mean|A|':>10}{'sd A':>9}{'%pos':>8}"
              f"{'success':>9}   note")
        edges = [0.0, 1.0, lm["box2_near"], 2.9, lm["box1_far"], lm["box1_near"], 7.0, 9.0,
                 1e9]
        notes = {lm["box2_near"]: "APPROACH BOX 2", lm["box1_near"]: "APPROACH BOX 1"}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (d >= lo) & (d < hi)
            if m.sum() < 50:
                continue
            row = {"lo": float(lo), "hi": float(min(hi, 99.0)), "n": int(m.sum()),
                   "mean_abs_adv": float(np.abs(adv[m]).mean()),
                   "sd_adv": float(adv[m].std()),
                   "frac_positive": float((lab[m] == 1).mean()),
                   "success": float(succ[m].mean())}
            out["bands"].append(row)
            print(f"  {f'[{lo:.1f},{min(hi, 99):.1f})':>16}{row['n']:>8}"
                  f"{row['mean_abs_adv']:>10.4f}{row['sd_adv']:>9.4f}"
                  f"{100 * row['frac_positive']:>7.1f}%{100 * row['success']:>8.1f}%   "
                  f"{notes.get(lo, '')}")
        flat = [r for r in out["bands"] if r["lo"] >= 7.0]
        dec = [r for r in out["bands"] if abs(r["lo"] - lm["box1_near"]) < 1e-6]
        if flat and dec:
            print(f"\n  => decisive band |A| {dec[0]['mean_abs_adv']:.4f} against "
                  f"{flat[0]['mean_abs_adv']:.4f} on flat approach. A label that pointed at "
                  "the decision would be LARGER here, not smaller.")

        # --- 2. mode ----------------------------------------------------------------
        lo, hi = (a.band if a.band else (lm["box1_near"], lm["box1_near"] + 1.1))
        band = ok & (d >= lo) & (d < hi)
        idx = np.flatnonzero(band)
        rng = np.random.default_rng(a.seed)
        if idx.size > a.max_rows:
            idx = np.sort(rng.choice(idx, size=a.max_rows, replace=False))
        climb = planned_climb(f, idx)
        av, lv, sv = adv[idx], lab[idx], succ[idx]
        hi_c, lo_c = climb >= np.percentile(climb, 70), climb <= np.percentile(climb, 30)
        r_adv = float(np.corrcoef(climb, av)[0, 1])
        r_suc = float(np.corrcoef(climb, sv.astype(float))[0, 1])
        print(f"\nWHETHER THE LABEL PREFERS THE RIGHT MODE  "
              f"({lo:.2f}-{hi:.2f} m to goal, {idx.size} windows)")
        print(f"  planned climb: p10 {np.percentile(climb, 10):.3f}  "
              f"p50 {np.percentile(climb, 50):.3f}  p90 {np.percentile(climb, 90):.3f} m")
        print(f"  {'group':<22}{'n':>7}{'mean adv':>11}{'%labelled +':>13}"
              f"{'REALIZED success':>18}")
        for tag, m in (("top-30% climbers", hi_c), ("bottom-30% climbers", lo_c)):
            print(f"  {tag:<22}{int(m.sum()):>7}{av[m].mean():>+11.4f}"
                  f"{100 * (lv[m] == 1).mean():>12.1f}%{100 * sv[m].mean():>17.1f}%")
        print(f"\n  corr(climb, ADVANTAGE) = {r_adv:+.4f}")
        print(f"  corr(climb, SUCCESS)   = {r_suc:+.4f}")
        if r_suc != 0:
            print(f"  => the label's preference is {abs(r_adv / r_suc):.1f}x the outcome's. "
                  "Above ~1 it is penalising a mode harder than the results justify.")
        out["mode"] = {"band": [float(lo), float(hi)], "n": int(idx.size),
                       "corr_climb_advantage": r_adv, "corr_climb_success": r_suc,
                       "climb_p50": float(np.percentile(climb, 50)),
                       "hi_climb": {"n": int(hi_c.sum()), "mean_adv": float(av[hi_c].mean()),
                                    "frac_pos": float((lv[hi_c] == 1).mean()),
                                    "success": float(sv[hi_c].mean())},
                       "lo_climb": {"n": int(lo_c.sum()), "mean_adv": float(av[lo_c].mean()),
                                    "frac_pos": float((lv[lo_c] == 1).mean()),
                                    "success": float(sv[lo_c].mean())}}

        # --- 3. matched pairs, for a viewer ----------------------------------------
        if a.dump_pairs:
            pf = None
            if a.predictions:
                pr = dict(np.load(a.predictions))
                # HELD-OUT ROWS ONLY. `fit_value` writes predictions OUT OF FOLD, so at
                # `num_folds = 1` every training row carries a constant 0.0 -- and matching
                # two chunks on that is matching on a fill value, which looks like a perfect
                # match and controls for nothing. Restricting to `val | val_tile` is what
                # makes "matched on p_fail" mean anything.
                real = pr["val"].astype(bool) | pr["val_tile"].astype(bool)
                pfa, _, phit = join({"episode_uid": pr["episode_uid"][real],
                                     "replan_index": pr["replan_index"][real],
                                     "advantage": pr["p_V_priv"][real],
                                     "label": np.zeros(int(real.sum()), np.int8)},
                                    uid, ridx)
                pf = np.where(phit, pfa, np.nan)
                print(f"[adv] p_fail available on {int(np.isfinite(pf).sum())} of {n} rows "
                      f"(held-out only; train rows carry no out-of-fold prediction)")
            pairs = []
            # SPREAD THE CLIMBERS ACROSS DIFFICULTY. Taking the highest-advantage ones
            # returns a set that is entirely `p_fail = 0.000` -- the label's favourite
            # climbers are its easiest states, so a viewer built from them shows ten copies
            # of one situation and says nothing about the hard end, which is the end that
            # decides the course.
            cl = idx[hi_c]
            if pf is not None and np.isfinite(pf[cl]).any():
                q = pf[cl]
                order = np.argsort(np.where(np.isfinite(q), q, np.inf))
                take = np.linspace(0, int(np.isfinite(q).sum()) - 1,
                                   num=min(a.pairs * 3, int(np.isfinite(q).sum())))
                hi_rows = cl[order][np.unique(take.astype(int))]
            else:
                hi_rows = cl[np.argsort(av[hi_c])[::-1]]
            lo_rows = idx[lo_c]
            used = set()
            for hr in hi_rows:
                if len(pairs) >= a.pairs:
                    break
                cand = lo_rows
                if pf is not None and np.isfinite(pf[hr]):
                    cand = lo_rows[np.abs(pf[lo_rows] - pf[hr]) <= a.p_band]
                cand = [c for c in cand if int(c) not in used]
                if not cand:
                    continue
                # NEAREST p_fail, not highest advantage. Maximising the partner's advantage
                # inside the band picks the HARDEST state in it -- the two are correlated --
                # so every climber ended up paired against a strictly harder flat chunk
                # (0.598 vs 0.677, 0.734 vs 0.814, ...), which flatters the climber on the
                # outcome column the pair exists to compare.
                if pf is not None and np.isfinite(pf[hr]):
                    lr = int(min(cand, key=lambda c: abs(pf[c] - pf[hr])))
                else:
                    lr = int(max(cand, key=lambda c: adv[c]))
                used.add(lr)
                rows = np.sort(np.array([int(hr), lr]))
                traj = f["fm_plan"][rows][:, :, :3]
                order = [0, 1] if rows[0] == int(hr) else [1, 0]
                pairs.append({
                    "climber": {"xyz": traj[order[0]].tolist(), "adv": float(adv[hr]),
                                "label": int(lab[hr]), "p_fail": (None if pf is None else
                                                                  float(pf[hr])),
                                "success": bool(succ[hr]),
                             "to_goal_m": float(d[hr]),
                                "climb": float(traj[order[0]][:, 2].max())},
                    "flat": {"xyz": traj[order[1]].tolist(), "adv": float(adv[lr]),
                             "label": int(lab[lr]), "p_fail": (None if pf is None else
                                                               float(pf[lr])),
                             "success": bool(succ[lr]),
                             "to_goal_m": float(d[lr]),
                             "climb": float(traj[order[1]][:, 2].max())}})
            with open(a.dump_pairs, "w") as fh:
                json.dump({"pairs": pairs, "landmarks": lm, "summary": out["mode"]}, fh)
            print(f"\n[adv] wrote {len(pairs)} matched pair(s) -> {a.dump_pairs}")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[adv] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
