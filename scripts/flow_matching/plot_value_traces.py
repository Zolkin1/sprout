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
* **right -- AUC by phase.** The healthy signature RISES from ~0.5 early to high near the
  decisive moment. **Flat and high is the failure this panel exists to catch**: a critic that
  has memorised which tile is hard scores a perfect aggregate AUC and is already sharp at
  phase 0, before anything has happened.

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


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fit_dir", help="directory written by fit_value.py")
    p.add_argument("--variant", default="V_priv", choices=("V_priv", "V_onboard"))
    p.add_argument("--max_traces", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    z = np.load(os.path.join(args.fit_dir, "predictions.npz"))
    with open(os.path.join(args.fit_dir, "gates.json")) as f:
        gates = json.load(f)["results"][args.variant]
    pf = z[f"p_{args.variant}"]
    uid, ix, val, failed = z["episode_uid"], z["replan_index"], z["val"], z["failed"]

    plot_style.apply_paper_style()
    import matplotlib.pyplot as plt  # noqa: PLC0415
    C = plot_style.COLORS
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    # --- traces, against time-to-terminal -------------------------------------------------
    ax = axes[0]
    rng = np.random.default_rng(args.seed)
    eps = np.unique(uid[val])
    for cls, colour, lbl in ((True, C["red"], "failure"), (False, C["teal"], "success")):
        pool = [e for e in eps if bool(failed[uid == e][0]) == cls]
        rng.shuffle(pool)
        for n, e in enumerate(pool[:args.max_traces // 2]):
            m = uid == e
            o = np.argsort(ix[m])
            t = (ix[m][o] - ix[m][o].max()) * 0.24     # 0 at the terminal chunk
            ax.plot(t, pf[m][o], color=colour, lw=1.6, alpha=0.45,
                    label=lbl if n == 0 else None)
    ax.axhline(0.5, color="0.5", lw=1.2, ls=":")
    ax.set_xlabel(r"time to terminal (s)")
    ax.set_ylabel(r"$P(\mathrm{fail})$")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    # --- calibration ----------------------------------------------------------------------
    ax = axes[1]
    cal = gates["calibration"]
    pred = np.array([np.nan if v is None else v for v in cal["predicted"]], dtype=float)
    obs = np.array([np.nan if v is None else v for v in cal["observed"]], dtype=float)
    ax.plot([0, 1], [0, 1], color="0.5", lw=1.5, ls=":")
    ax.plot(pred, obs, color=C["blue"], lw=3, marker="o", ms=7)
    ax.set_xlabel(r"predicted $P(\mathrm{fail})$")
    ax.set_ylabel(r"observed failure rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ece = gates["calibration"]["ece"]
    ax.text(0.04, 0.92, rf"ECE {ece:.3f}", transform=ax.transAxes, fontsize=15, va="top")

    # --- AUC by phase ----------------------------------------------------------------------
    ax = axes[2]
    byph = gates["auc_by_phase"]
    lo = np.array(byph["bin_lo"], dtype=float)
    hi = np.array(byph["bin_hi"], dtype=float)
    a = np.array([np.nan if v is None else v for v in byph["auc"]], dtype=float)
    ax.axhline(0.5, color="0.5", lw=1.5, ls=":")
    ax.plot(0.5 * (lo + hi), a, color=C["orange"], lw=3, marker="s", ms=8)
    ax.set_xlabel(r"phase (0 = first chunk, 1 = last)")
    ax.set_ylabel(r"AUC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)
    flat = np.nanmax(a) - np.nanmin(a) < 0.1 and np.nanmin(a) > 0.7
    if flat:
        ax.text(0.04, 0.10, "FLAT and HIGH:\ncheck held-out tiles", transform=ax.transAxes,
                fontsize=14, va="bottom", color=C["red"])

    fig.tight_layout()
    out = os.path.join(args.fit_dir, f"value_{args.variant}")
    plot_style.save_figure(fig, out)
    print(f"wrote {out}.png")
    if flat:
        print("NOTE: the phase curve is flat and high -- already sharp at phase 0, before "
              "anything has happened. That has TWO readings and this panel cannot separate "
              "them: the critic has memorised which tile is hard, or the depth image "
              "genuinely shows the terrain ahead and the outcome really is partly "
              "predictable before anything happens. HELD-OUT TILES decide it -- under "
              "memorisation the unseen-tile score collapses, under real perception it holds. "
              "fit_value.py reports both and their gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
