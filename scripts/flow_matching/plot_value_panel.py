# Copyright 2026 Zachary Olkin. All rights reserved.

"""The three critic gates, for both variants and both scopes, on one page.

Rows are ``(variant, scope)`` and columns are the three questions, so a column compares like
with like down the page:

1. **P(fail) against time-to-terminal** -- do the failing and surviving families separate,
   and how long before the end?  This is the qualitative form of the lead-time number.
2. **Reliability** -- predicted P(fail) against the observed failure frequency in each bin.
   Testable even where discrimination is impossible, which is why the gate list starts here.
3. **AUC by phase** -- discrimination as a function of position in the episode, never in
   aggregate: a critic that is sharp only at the end scores well in aggregate while being
   useless for credit assignment.

**The scope split is the measurement, not a formality.**  ``seen tiles`` holds out EPISODES
from terrain the critic trained on; ``unseen tiles`` holds out the terrain itself.  A gap
between them is the critic reading tile identity rather than rollout state -- and on this data
that gap is concentrated in the EARLY phase, which is exactly where an advantage estimate
needs it least to be tile-driven.

Every prediction plotted is OUT OF FOLD (``fit_value.py --folds 2``), so no window is scored
by a model that saw its episode.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_style  # noqa: E402
from robot_rl.flow_matching import value_gates  # noqa: E402


def _traces(ax, p, uid, ridx, failed, mask, rng, C, n_max=25, k=12, dt=0.02):
    """Per-episode P(fail) against time-to-terminal, successes and failures overlaid.

    Time runs backwards from each episode's own last re-plan, so episodes of different length
    are comparable at the only instant they share: the end.
    """
    eps = np.unique(uid[mask])
    fail_eps = [e for e in eps if failed[uid == e][0]]
    ok_eps = [e for e in eps if not failed[uid == e][0]]
    for group, colour in ((fail_eps, C["red"]), (ok_eps, C["teal"])):
        pick = rng.choice(group, size=min(n_max, len(group)), replace=False) if group else []
        for e in pick:
            sel = mask & (uid == e)
            if sel.sum() < 2:
                continue
            order = np.argsort(ridx[sel])
            y = p[sel][order]
            # Negative x: 0 is the terminal re-plan, -2 is two chunks before it.
            x = (np.arange(len(y)) - (len(y) - 1)) * k * dt
            ax.plot(x, y, color=colour, alpha=0.35, linewidth=1.5)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1)
    ax.set_ylim(-0.02, 1.02)


def _reliability(ax, p, failed, C, n_bins=10):
    """Predicted P(fail) against observed frequency, with the bin populations annotated."""
    cal = value_gates.calibration(p, failed, n_bins=n_bins)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1.5)
    ax.plot(cal["predicted"], cal["observed"], marker="o", color=C["blue"], linewidth=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.04, 0.92, rf"ECE {cal['ece']:.3f}", transform=ax.transAxes, fontsize=14)
    return cal


def _phase(ax, p, failed, phase, C, n_bins=5):
    """AUC within equal-width phase bins, 0 = first chunk of an episode, 1 = last."""
    by = value_gates.auc_by_phase(p, failed, phase, n_bins=n_bins)
    mid = 0.5 * (by["bin_lo"] + by["bin_hi"])
    ax.plot(mid, by["auc"], marker="s", color=C["orange"], linewidth=3, markersize=9)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.5)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(0, 1)
    return by


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fit_dir", help="directory holding predictions.npz")
    p.add_argument("--out", default="", help="output path stem (default: <fit_dir>/gate_panel)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traces", type=int, default=25, help="episodes drawn per outcome group.")
    args = p.parse_args()

    d = np.load(os.path.join(args.fit_dir, "predictions.npz"))
    uid, ridx = d["episode_uid"], d["replan_index"]
    failed = d["failed"].astype(bool)
    scopes = {"seen tiles": d["val"].astype(bool), "unseen tiles": d["val_tile"].astype(bool)}
    variants = [k[2:] for k in d.files if k.startswith("p_")]

    # Phase: position within the episode, 0 at the first chunk and 1 at the last.
    last = {int(u): int(ridx[uid == u].max()) for u in np.unique(uid)}
    phase = np.array([ridx[i] / max(last[int(uid[i])], 1) for i in range(uid.size)])

    plot_style.apply_paper_style()
    import matplotlib.pyplot as plt  # noqa: PLC0415
    C = plot_style.COLORS
    rows = [(v, s) for v in sorted(variants) for s in scopes]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15.0, 4.0 * len(rows)))
    rng = np.random.default_rng(args.seed)

    for r, (variant, scope) in enumerate(rows):
        pv = d[f"p_{variant}"]
        m = scopes[scope]
        _traces(axes[r][0], pv, uid, ridx, failed, m, rng, C, n_max=args.traces)
        cal = _reliability(axes[r][1], pv[m], failed[m], C)
        by = _phase(axes[r][2], pv[m], failed[m], phase[m], C)
        auc = value_gates.auc(pv[m], failed[m])

        axes[r][0].set_ylabel(rf"\textbf{{{plot_style.tex_escape(variant)}}}" "\n"
                              rf"{plot_style.tex_escape(scope)}" "\n" r"$P(\mathrm{fail})$")
        axes[r][0].text(0.04, 0.92, rf"AUC {auc:.3f}", transform=axes[r][0].transAxes,
                        fontsize=14)
        if r == 0:
            axes[r][0].set_title(r"traces vs time-to-terminal", fontsize=17)
            axes[r][1].set_title(r"reliability", fontsize=17)
            axes[r][2].set_title(r"AUC by phase", fontsize=17)
        if r == len(rows) - 1:
            axes[r][0].set_xlabel(r"time to terminal (s)")
            axes[r][1].set_xlabel(r"predicted $P(\mathrm{fail})$")
            axes[r][2].set_xlabel(r"phase (0 = first chunk, 1 = last)")
        axes[r][1].set_ylabel(r"observed failure rate")
        axes[r][2].set_ylabel(r"AUC")
        print(f"{variant:11s} {scope:13s} AUC {auc:.3f}  ECE {cal['ece']:.3f}  "
              f"phase {np.array2string(by['auc'], precision=3, floatmode='fixed')}")

    from matplotlib.lines import Line2D  # noqa: PLC0415
    axes[0][0].legend(handles=[Line2D([], [], color=C["red"], label=r"failed"),
                               Line2D([], [], color=C["teal"], label=r"survived")],
                      loc="upper left", fontsize=13)
    fig.tight_layout()
    stem = args.out or os.path.join(args.fit_dir, "gate_panel")
    plot_style.save_figure(fig, stem)
    print(f"\n[panel] wrote {stem}.png / .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
