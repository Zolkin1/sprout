# Copyright 2026 Zachary Olkin. All rights reserved.

"""Look at the critic before believing any statistic about it.

Three panels, and the first is the one that matters:

* **left -- ``P(fail)`` against time-to-terminal**, successes and failures overlaid. The
  question is WHERE the two families separate. A critic that is useful separates them some
  way before the end, at the mistake; one that separates only in the last chunk or two is
  reporting the consequence, and an advantage built on it arrives after the chunk that caused
  the failure. Plotted against time-to-terminal rather than absolute time so episodes of
  different lengths are comparable at the moment that matters.
* **centre -- calibration.** Predicted against observed, which is testable even in regions
  where no critic could separate individual episodes.
* **right -- AUC by phase, on BOTH validation scopes.** The healthy signature RISES from
  ~0.5 early to high near the decisive moment. A curve that is flat and high has two readings
  and this panel alone cannot separate them -- the critic memorised which tile is hard, or the
  depth image genuinely shows the terrain ahead and the outcome really is partly predictable
  before anything happens. **The two curves are what decide it.** Held-out EPISODES come from
  tiles the critic trained on; held-out TILES it has never seen. The vertical gap between them
  is the memorisation, and the shape of the unseen-tile curve is the signal that survives it.

Everything defaults to the UNSEEN-tile scope, because that is the honest one and the seen-tile
numbers read a good deal better than the critic deserves.

Usage::

    python scripts/flow_matching/plot_value_traces.py <fit dir> [--variant V_priv]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _d in (str(_HERE.parent / "rsl_rl"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import plot_style  # noqa: E402


def _traces(ax, pf, uid, ix, failed, mask, args, C):
    """Per-episode P(fail) against time-to-terminal, successes and failures overlaid.

    The question this panel answers is WHERE the two families separate. A critic that
    separates them only in the last chunk or two is reporting the consequence; an advantage
    built on it arrives after the chunk that caused the failure.
    """
    rng = np.random.default_rng(args.seed)
    eps = np.unique(uid[mask])
    for cls, colour, lbl in ((True, C["red"], "failure"), (False, C["teal"], "success")):
        pool = [e for e in eps if bool(failed[uid == e][0]) == cls]
        rng.shuffle(pool)
        for n, e in enumerate(pool[:args.max_traces // 2]):
            m = uid == e
            o = np.argsort(ix[m])
            ax.plot((ix[m][o] - ix[m][o].max()) * 0.24, pf[m][o], color=colour, lw=1.6,
                    alpha=0.45, label=lbl if n == 0 else None)
    ax.axhline(0.5, color="0.5", lw=1.2, ls=":")
    ax.set_xlabel(r"time to terminal (s)")
    ax.set_ylabel(r"$P(\mathrm{fail})$")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.text(0.03, 0.03, args.scope.replace("_", " "), transform=ax.transAxes, fontsize=13,
            color="0.35")


def _series(g, key):
    """Pull a gate array out of the JSON, turning its nulls back into NaN."""
    return np.array([np.nan if v is None else v for v in g[key]], dtype=float)


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fit_dir", help="directory written by fit_value.py")
    p.add_argument("--variant", default="V_onboard", choices=("V_priv", "V_onboard"))
    p.add_argument("--scope", default="val_UNSEEN_tiles",
                   choices=("val_UNSEEN_tiles", "val_seen_tiles"),
                   help="which held-out set the traces and calibration come from; the phase "
                        "panel always shows both")
    p.add_argument("--max_traces", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # The traces need per-window predictions; the other two panels need only the gates. A
    # fit that produced gates.json but not predictions.npz is still worth looking at, so the
    # traces panel is dropped rather than the whole figure.
    npz = os.path.join(args.fit_dir, "predictions.npz")
    z = np.load(npz) if os.path.isfile(npz) else None
    with open(os.path.join(args.fit_dir, "gates.json")) as f:
        all_gates = json.load(f)["results"][args.variant]
    gates = all_gates[args.scope]

    plot_style.apply_paper_style()
    import matplotlib.pyplot as plt  # noqa: PLC0415
    C = plot_style.COLORS
    n_panels = 3 if z is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.7 * n_panels, 4.0))
    axes = list(np.atleast_1d(axes))
    if z is None:
        print(f"NOTE: {npz} is absent, so the TRACE panel -- the one worth looking at first "
              "-- is missing. Re-run fit_value.py to produce it.")
        axes = [None] + axes

    # --- traces, against time-to-terminal -------------------------------------------------
    if z is not None:
        pf = z[f"p_{args.variant}"]
        uid, ix, failed = z["episode_uid"], z["replan_index"], z["failed"]
        mask = z["val_tile"] if args.scope == "val_UNSEEN_tiles" else z["val"]
        _traces(axes[0], pf, uid, ix, failed, mask, args, C)
    # --- calibration ----------------------------------------------------------------------
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="0.5", lw=1.5, ls=":")
    ax.plot(_series(gates["calibration"], "predicted"),
            _series(gates["calibration"], "observed"),
            color=C["blue"], lw=3, marker="o", ms=7)
    ax.set_xlabel(r"predicted $P(\mathrm{fail})$")
    ax.set_ylabel(r"observed failure rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.92, rf"ECE {gates['calibration']['ece']:.3f}", transform=ax.transAxes,
            fontsize=15, va="top")

    # --- AUC by phase, BOTH scopes ---------------------------------------------------------
    ax = axes[2]
    ax.axhline(0.5, color="0.5", lw=1.5, ls=":")
    for scope, colour, lbl in (("val_seen_tiles", C["grey"], "seen tiles"),
                               ("val_UNSEEN_tiles", C["orange"], "UNSEEN tiles")):
        g = all_gates[scope]["auc_by_phase"]
        lo, hi = _series(g, "bin_lo"), _series(g, "bin_hi")
        ax.plot(0.5 * (lo + hi), _series(g, "auc"), color=colour, lw=3, marker="s", ms=8,
                label=lbl)
    ax.set_xlabel(r"phase (0 = first chunk, 1 = last)")
    ax.set_ylabel(r"AUC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    gap = all_gates["val_seen_tiles"]["auc"] - all_gates["val_UNSEEN_tiles"]["auc"]
    ax.text(0.04, 0.06, rf"gap {gap:+.3f}", transform=ax.transAxes, fontsize=14,
            color=C["red"] if gap > 0.10 else "0.35")

    fig.tight_layout()
    out = os.path.join(args.fit_dir, f"value_{args.variant}_{args.scope}")
    plot_style.save_figure(fig, out)
    print(f"wrote {out}.png")
    print("left: where the two families separate -- a critic that separates only in the last "
          "chunk is reporting the consequence, not the mistake.")
    print("right: the vertical gap between the curves IS the memorisation; the shape of the "
          "UNSEEN curve is what survives it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
