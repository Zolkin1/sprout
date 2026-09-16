# Copyright 2026 Zachary Olkin. All rights reserved.

"""Forest plot of the fine-tune algorithm ablation, with AWR as the zero line.

    python scripts/rsl_rl/plot_forest.py forest_rows.json --out logs/paper/forest

One row per arm, grouped by (course, termination setting), showing `arm - AWR` as a paired delta
with its 95% cluster-bootstrap interval. AWR sits at 0 in every group by construction and is drawn
as a marker without an interval, because it is a reference rather than a measurement.

**WHY WITHIN-GROUP PAIRING, not a difference of two vs-parent deltas.** Every row here is measured
directly against its OWN group's AWR arm, on shared (tile, speed) cells with envs resampled inside
them. Differencing two arms' vs-parent numbers instead would carry the parent's own sampling noise
into every row twice and would not be paired at the env level at all -- the design that makes
these intervals as tight as they are.

The interval is the honest part of the figure: a row whose interval crosses 0 has not been shown
to differ from AWR, however suggestive its point estimate. `arrived` on boxes-strict is exactly
that case, and the plot should let a reader see it rather than rank it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_style  # noqa: E402


def main() -> None:
    """Render the forest plot."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", help="json from forest_data.py")
    ap.add_argument("--out", default="logs/paper/forest")
    ap.add_argument("--contact", default="",
                    help="second json, keyed the same way, holding contact-penalty deltas")
    ap.add_argument("--xlabel", default="Success rate (rel. to AWR)")
    ap.add_argument("--xlabel2", default="Mean contact penalty (rel. to AWR)")
    a = ap.parse_args()

    plot_style.apply_paper_style()
    data = json.loads(pathlib.Path(a.rows).read_text())
    rows = data["rows"]

    # Groups in the order the data file gives them; arms in order within each group. Top group
    # is drawn topmost, which means plotting from the bottom up.
    groups: list[str] = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])

    ys, labels, seps, ticks = [], [], [], []
    y = 0.0
    for gi, g in enumerate(reversed(groups)):
        grows = [r for r in rows if r["group"] == g]
        for r in reversed(grows):
            ys.append((y, r))
            labels.append(r["arm"])
            y += 1.0
        ticks.append((y - len(grows) / 2.0 - 0.5, g))
        if gi < len(groups) - 1:
            seps.append(y - 0.5)
            y += 0.8

    contact = {}
    if a.contact:
        for r in json.loads(pathlib.Path(a.contact).read_text())["rows"]:
            contact[(r["group"], r["arm"])] = r

    npanel = 2 if contact else 1
    fig, axes = plt.subplots(1, npanel, sharey=True,
                             figsize=(plot_style.SINGLE_COL[0] * (1.15 * npanel + 0.55),
                                      0.42 * y + 1.4))
    axes = np.atleast_1d(axes)

    # ONE COLOUR for every arm. An earlier version coloured by whether the interval excluded
    # zero, which reads as a verdict the figure should not be issuing on the reader's behalf --
    # the interval is already on the page and says it better. Uniform colour also stops the two
    # panels from disagreeing about an arm when one crosses zero and the other does not.
    col = plot_style.COLORS["blue"]
    for ax, (src, xlabel) in zip(axes, [(None, a.xlabel), (contact, a.xlabel2)][:npanel]):
        ax.axvline(0.0, color="0.35", lw=1.4, ls=(0, (5, 3)), zorder=1)
        for pos, r in ys:
            row = r if src is None else src.get((r["group"], r["arm"]))
            if row is None:
                continue
            if r.get("reference"):
                ax.plot([0.0], [pos], marker="D", ms=7, color="0.35", zorder=4)
                continue
            # `lo`/`hi` are the CLUSTER-T interval, which `paired_delta.report` and
            # `fig_reduce` now both emit as the headline; `boot_lo`/`boot_hi` carry the
            # single-stage bootstrap and are the robustness check, not drawn.
            ax.plot([row["lo"], row["hi"]], [pos, pos], color=col, lw=2.2,
                    solid_capstyle="round", zorder=3)
            ax.plot([row["delta"]], [pos], marker="o", ms=6.5, color=col, zorder=4)
        ax.set_xlabel(xlabel)
        # Title above the panel it describes. `label_position` moves the text; `tick_top` moves
        # the numbers with it, and without the second call the ticks stay at the bottom and the
        # axis reads as though it belonged to whatever is below.
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.xaxis.grid(True, color="0.90", lw=1.0)
        ax.set_axisbelow(True)
        for sp in seps:
            ax.axhline(sp, color="0.85", lw=1.0, zorder=0)

    ax = axes[0]
    ax.set_yticks([p for p, _ in ys])
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, y - 0.2)

    # Group names on the right, so they do not fight the arm labels for the left margin.
    for pos, name in ticks:
        axes[-1].annotate(name, xy=(1.03, pos), xycoords=("axes fraction", "data"),
                          va="center", ha="left", fontsize="small", color="0.30", rotation=90,
                          annotation_clip=False)
    fig.subplots_adjust(right=0.88)

    n = {r["n_cells"] for r in rows if r.get("n_cells")}
    fig.text(0.5, 0.005,
             plot_style.tex_escape(
                 "paired against AWR within each block; 95% cluster bootstrap over "
                 "(tile, speed) cells"),
             ha="center", va="bottom", fontsize="small", color="0.35")

    caption = (
        "Fine-tune algorithm ablation as differences from AWR. Each row is one arm's paired "
        "success-rate difference from the AWR arm of its own block, with a 95% cluster bootstrap "
        "interval over (tile, speed) cells resampling envs within them; AWR is the zero line and "
        "is drawn without an interval because it is the reference. Blocks differ in course and in "
        "whether the tracking-bound termination was active during collection. Grey intervals "
        "include zero. Pairing is within block and at the environment level, not a difference of "
        "two vs-parent deltas, which would carry the pre-trained arm's noise into every row.")
    for f in plot_style.save_figure(fig, a.out, caption=caption):
        print(f"wrote {f}")
    for pos, r in ys:
        if r.get("reference"):
            print(f"  {r['group']:16s} {r['arm']:12s}  reference")
        else:
            print(f"  {r['group']:16s} {r['arm']:12s} {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  p={r['mcnemar_p']:.3g}")


if __name__ == "__main__":
    main()
