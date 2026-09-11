#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where does the advantage put its blame, relative to the fall?

Re-indexes every chunk of every FAILED episode by its distance from the end -- ``k = 0`` is the
last chunk before termination, ``k = 1`` the one before that -- and plots the median and
interquartile range of the cross-fit ``A_MC`` against ``k``, fall on the right.

**The survived-episode curve is drawn beside it, and the plot is not readable without it.**
Aligned to their own last chunk, surviving episodes answer the question the failed curve cannot
ask alone: is a negative advantage near ``k = 0`` about FAILING, or about being near the end of
any episode at all?  The tail assumption alone gives the last chunks of every episode a
distinctive value (``G`` there is almost entirely the tail), so a control that shares that
structure and differs only in outcome is what isolates the fall.

What the shape means, stated before it is looked at:

* **negative near the fall** -- the realized return came in below what the critic expected from
  that state, which is the label saying "this execution was bad".  With an absorbing tail,
  ``G`` at ``k = 0`` is nearly zero by construction, so the depth there is mostly a statement
  about ``V(s_t)``: how much the critic still expected when it was already over.
* **near zero far from the fall** -- past roughly ``log(eps)/log(gamma)`` chunks the discount
  has hidden the terminal, so any offset out there is NOT the terminal penalty and would mean
  the critic mis-prices those states for some other reason.
* **the transition** -- a sharp drop confined to the last few chunks means blame lands on the
  immediate lead-up; a ramp beginning ~20 chunks out means it is spread over the discount
  horizon.  That is Monte-Carlo smearing, expected, and worth knowing rather than fixing.

At ``gamma = 0.95`` the tail retains 1% of its weight at ``k = 90``, so the horizon is set by
how many chunks the episodes actually have, not by the discount.

Usage::

    python scripts/flow_matching/plot_advantage_vs_fall.py \\
        --predictions logs/value/v4/predictions.npz \\
        --out logs/value/critic_eval_v4/advantage_vs_fall
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(os.path.dirname(_HERE), "rsl_rl"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def distance_from_end(uid: np.ndarray, ri: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Chunks between each row and its episode's LAST VALID one.

    Measured against the last row that carries a usable return, not the last row present:
    ``chunk_rewards_from_track_err`` leaves a placeholder on every episode's final row, so
    counting from there would put ``k = 0`` on a window with no target and shift every episode
    by one against the others.

    Args:
        uid: ``[W]`` episode ids.
        ri: ``[W]`` re-plan index within the episode.
        valid: ``[W]`` whether the row carries a usable ``G``.

    Returns:
        ``[W]`` int64 ``k``; ``-1`` on rows that are invalid or whose episode has none.
    """
    k = np.full(uid.shape[0], -1, dtype=np.int64)
    _, inv = np.unique(uid, return_inverse=True)
    last = np.full(inv.max() + 1 if inv.size else 0, -1, dtype=np.int64)
    np.maximum.at(last, inv[valid], ri[valid])
    good = valid & (last[inv] >= 0)
    k[good] = last[inv][good] - ri[good]
    return k


def _profile(a: np.ndarray, k: np.ndarray, k_max: int, min_n: int) -> dict:
    """Median, IQR and count of ``a`` at each distance ``k``.

    Args:
        a: ``[n]`` advantages.
        k: ``[n]`` distance from the episode's last valid chunk.
        k_max: Largest ``k`` to report.
        min_n: Drop a ``k`` with fewer rows than this -- a quartile over a handful of
            episodes is noise wearing the shape of a trend, and the far end of this axis is
            exactly where the episode count collapses.

    Returns:
        Arrays ``k``, ``median``, ``q25``, ``q75``, ``n``.
    """
    ks, med, lo, hi, cnt = [], [], [], [], []
    for kk in range(0, k_max + 1):
        m = k == kk
        if m.sum() < min_n:
            continue
        v = a[m]
        ks.append(kk)
        med.append(float(np.median(v)))
        lo.append(float(np.percentile(v, 25)))
        hi.append(float(np.percentile(v, 75)))
        cnt.append(int(m.sum()))
    return {"k": np.asarray(ks), "median": np.asarray(med), "q25": np.asarray(lo),
            "q75": np.asarray(hi), "n": np.asarray(cnt)}


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="predictions.npz from fit_value.py")
    p.add_argument("--out", required=True, help="Output path WITHOUT extension.")
    p.add_argument("--variant", default="V_priv", choices=("V_priv", "V_onboard"))
    p.add_argument("--k_max", type=int, default=40, help="Chunks back from the end to plot.")
    p.add_argument("--min_n", type=int, default=50,
                   help="Drop a k with fewer rows than this.")
    p.add_argument("--scope", default="all", choices=("all", "unseen_tiles"),
                   help="`unseen_tiles` restricts to tiles NEITHER fold trained on. The whole "
                        "table is already cross-fit, so `all` is honest; the restriction is a "
                        "check that the shape is not a property of the training tiles.")
    args = p.parse_args()

    d = np.load(args.predictions, allow_pickle=True)
    uid, ri = d["episode_uid"], d["replan_index"]
    failed = d["failed"].astype(bool)
    g, gok = d["g"], d["g_valid"].astype(bool)
    q = d[f"q_{args.variant}"]
    gamma = float(d["gamma"]) if "gamma" in d.files else 0.95
    sel = gok if args.scope == "all" else (gok & d["val_tile"].astype(bool))

    a_mc = g - q                       # cross-fit: both terms are out-of-fold readings
    k = distance_from_end(uid, ri, sel)
    ok = sel & (k >= 0)

    prof = {}
    for name, m in (("failed", ok & failed), ("survived", ok & ~failed)):
        prof[name] = _profile(a_mc[m], k[m], args.k_max, args.min_n)
        n_ep = int(np.unique(uid[m]).size)
        pr = prof[name]
        print(f"[plot] {name}: {int(m.sum())} rows over {n_ep} episodes; "
              f"median A_MC at k=0 {pr['median'][0]:+.4f}, "
              f"k=10 {pr['median'][10] if pr['k'].size > 10 else float('nan'):+.4f}, "
              f"k={pr['k'][-1]} {pr['median'][-1]:+.4f}")

    # Where does the failed curve come back to the survived one? Reported rather than eyeballed
    # off the figure: it is the number the "how far does blame reach" question actually wants.
    kf, ks_ = prof["failed"], prof["survived"]
    n = min(kf["k"].size, ks_["k"].size)
    gap = kf["median"][:n] - ks_["median"][:n]
    within = np.flatnonzero(np.abs(gap) < 0.02)
    reach = int(kf["k"][within[0]]) if within.size else None
    print(f"[plot] failed-vs-survived median gap at k=0: {gap[0]:+.4f}; "
          f"first k where |gap| < 0.02: {reach}"
          + (f" ({reach * 0.24:.1f} s before the fall)" if reach is not None else ""))

    # THE FAR END IS A LENGTH-BIASED SUBSAMPLE, and it is the one part of this plot that
    # cannot be read as a trend. Only episodes with at least k chunks appear at k, and failed
    # episodes are much shorter than survived ones -- so large k compares the LONG failures
    # against a survivor population that is typical.
    for name, m in (("failed", ok & failed), ("survived", ok & ~failed)):
        lens = np.bincount(np.unique(uid[m], return_inverse=True)[1])
        print(f"[plot] {name} episode length (valid chunks): p25 {np.percentile(lens, 25):.0f}"
              f"  median {np.median(lens):.0f}  p75 {np.percentile(lens, 75):.0f}"
              f"  max {lens.max()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import plot_style as ps  # noqa: PLC0415

    ps.apply_paper_style()
    fig, (ax, axg, axn) = plt.subplots(
        3, 1, figsize=(ps.DOUBLE_COL[0], ps.DOUBLE_COL[1] * 1.75), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.6, 1]})

    for name, colour in (("failed", ps.COLORS["red"]), ("survived", ps.COLORS["blue"])):
        pr = prof[name]
        ax.plot(pr["k"], pr["median"], color=colour, label=name.capitalize())
        ax.fill_between(pr["k"], pr["q25"], pr["q75"], color=colour, alpha=0.2, linewidth=0)
        axn.plot(pr["k"], pr["n"], color=colour)

    # THE GAP IS THE READABLE STATISTIC. Large k means EARLY in the episode for both arms, and
    # the critic hedges early -- so both curves drift upward with k for a reason that has
    # nothing to do with failing. Differencing them removes that shared drift.
    ax.plot(kf["k"][:n], np.zeros(n) + np.nan)          # keep the colour cycle aligned
    axg.plot(kf["k"][:n], gap, color=ps.COLORS["black"])
    axg.axhline(0.0, color=ps.COLORS["grey"], lw=1.2, ls=":", zorder=0)
    axg.set_ylabel(r"failed $-$ survived")

    i_min = int(np.argmin(prof["failed"]["median"]))
    k_min = int(prof["failed"]["k"][i_min])
    for a in (ax, axg):
        a.axvline(k_min, color=ps.COLORS["grey"], lw=1.2, ls="--", zorder=0)
    ax.annotate(rf"trough $k={k_min}$ ({k_min * 0.24:.1f}\,s)",
                xy=(k_min, prof["failed"]["median"][i_min]),
                xytext=(0.42, 0.12), textcoords="axes fraction", fontsize=13,
                arrowprops={"arrowstyle": "->", "color": ps.COLORS["grey"], "lw": 1.5})

    ax.axhline(0.0, color=ps.COLORS["black"], lw=1.2, ls=":", zorder=0)
    ax.set_ylabel(r"$A_{\mathrm{MC}} = G_t - \hat{G}(s_t)$")
    ax.legend(loc="lower left")
    ax.invert_xaxis()                  # the fall on the RIGHT
    axn.set_yscale("log")
    axn.set_ylabel(r"episodes")
    axn.set_xlabel(r"chunks $k$ before the last chunk ($1$ chunk $= 0.24$\,s)")
    fig.tight_layout()
    ps.save_figure(fig, args.out)
    plt.close(fig)

    np.savez(args.out + ".npz", gamma=gamma, scope=args.scope,
             **{f"{n}_{c}": prof[n][c] for n in prof for c in prof[n]})
    print(f"[plot] wrote {args.out}.png / .svg / .npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
