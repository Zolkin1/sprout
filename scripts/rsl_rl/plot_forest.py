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

#: One colour per METHOD, keyed by the arm label, so a method reads the same in every block and
#: stays the same method across figures. Colour here encodes IDENTITY, not a verdict -- an
#: earlier version coloured by whether the interval excluded zero, which made the figure state a
#: conclusion the interval already carries, and made the two panels disagree about an arm
#: whenever one crossed zero and the other did not. AWR is black rather than one of the hues
#: because it is the reference rather than a measurement: its "interval" is the zero it defines.
#: The zero RULE stays grey, so the marker reads against it instead of merging into it.
METHOD_COLOURS: dict[str, str] = {
    "AWR": plot_style.COLORS["black"],
    "Arrived": plot_style.COLORS["blue"],
    "Filtered BC": plot_style.COLORS["orange"],
    "Plain BC": plot_style.COLORS["teal"],
}
#: An arm not in the map still plots, in a colour none of the named ones already uses.
_SPARE = [c for c in plot_style.COLOR_CYCLE if c not in set(METHOD_COLOURS.values())]

#: What each arm is CALLED on the figure. The json files keep the canonical arm name, so
#: renaming a method is a plotting change and costs no re-reduction -- `forest_contact.py` runs
#: `fig_reduce` once per arm, which is the expensive half of this figure. It also means the
#: colour map above and the group keys in `forest_groups.json` stay keyed on one spelling.
DISPLAY_NAMES: dict[str, str] = {
    "Arrived": "Arrival Filt.",
    "Filtered BC": "Filt. BC",
}

#: Same idea for the block labels down the right-hand edge. `--exclude` still matches the
#: CANONICAL name, because that is what `forest_groups.json` and both json files are keyed on --
#: matching the display name would mean the flag stopped working the moment a block was renamed.
GROUP_DISPLAY: dict[str, str] = {
    "Multi, strict": "Multi-terrain, strict",
}


def main() -> None:
    """Render the forest plot."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", help="json from forest_data.py")
    ap.add_argument("--out", default="logs/paper/forest")
    ap.add_argument("--contact", default="",
                    help="second json, keyed the same way, holding contact-penalty deltas")
    ap.add_argument("--exclude", action="append", default=[], metavar="GROUP",
                    help="group to leave out; repeat the flag per group (names contain commas)")
    ap.add_argument("--xlabel", default="Success rate (rel. to AWR)")
    ap.add_argument("--xlabel2", default="Mean contact penalty (rel. to AWR)")
    a = ap.parse_args()

    plot_style.apply_paper_style()
    data = json.loads(pathlib.Path(a.rows).read_text())
    rows = data["rows"]

    # Dropping a block is a DISPLAY choice, so it happens here rather than by editing
    # `forest_groups.json` -- the arms stay computed and reappear by removing the flag.
    drop = {g.strip() for g in a.exclude if g.strip()}
    if drop:
        missing = drop - {r["group"] for r in rows}
        if missing:
            raise SystemExit(f"--exclude names no such group: {sorted(missing)}")
        rows = [r for r in rows if r["group"] not in drop]

    # Groups in the order the data file gives them; arms in order within each group. Top group
    # is drawn topmost, which means plotting from the bottom up.
    groups: list[str] = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])

    ys, labels, arms, seps, ticks = [], [], [], [], []
    y = 0.0
    for gi, g in enumerate(reversed(groups)):
        grows = [r for r in rows if r["group"] == g]
        for r in reversed(grows):
            ys.append((y, r))
            labels.append(DISPLAY_NAMES.get(r["arm"], r["arm"]))
            arms.append(r["arm"])
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

    # Colour by METHOD (see METHOD_COLOURS), assigned once over the arms actually present so
    # every block draws a given method the same and the eye can track it down the figure.
    seen: list[str] = []
    for _, r in ys:
        if r["arm"] not in seen:
            seen.append(r["arm"])
    colour = {arm: METHOD_COLOURS.get(arm, _SPARE[i % len(_SPARE)])
              for i, arm in enumerate(seen)}
    for ax, (src, xlabel) in zip(axes, [(None, a.xlabel), (contact, a.xlabel2)][:npanel]):
        ax.axvline(0.0, color="0.35", lw=1.4, ls=(0, (5, 3)), zorder=1)
        for pos, r in ys:
            row = r if src is None else src.get((r["group"], r["arm"]))
            if row is None:
                continue
            col = colour[r["arm"]]
            if r.get("reference"):
                ax.plot([0.0], [pos], marker="D", ms=7, color=col, zorder=4)
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
    # Keyed off `arms`, not the tick TEXT: the text is now a display name and would miss the
    # colour map, which silently falls back to black rather than raising.
    for lab, arm in zip(ax.get_yticklabels(), arms):
        lab.set_color(colour[arm])
    ax.set_ylim(-0.8, y - 0.2)

    # Group names on the right, so they do not fight the arm labels for the left margin.
    for pos, name in ticks:
        axes[-1].annotate(GROUP_DISPLAY.get(name, name),
                          xy=(1.03, pos), xycoords=("axes fraction", "data"),
                          va="center", ha="left", fontsize="small", color="0.30", rotation=90,
                          annotation_clip=False)
    fig.subplots_adjust(right=0.88)

    caption = (
        "Fine-tune algorithm ablation as differences from AWR. Each row is one arm's paired "
        "success-rate difference from the AWR arm of its own block. Intervals are 95% cluster-t, "
        "mean $\\pm\\,t_{C-1}\\,\\mathrm{SD}/\\sqrt{C}$ over the per-cell paired deltas, "
        "taking the (tile, speed) cell as the unit of independence; a single-stage cluster "
        "bootstrap agrees with these to within 0.001 everywhere. AWR is the zero line and is "
        "drawn without an interval because it is the reference. Colour identifies the method, "
        "not the outcome. Blocks differ in course and in whether the tracking-bound termination "
        "was active during collection. Pairing is within block and at the environment level, not "
        "a difference of two vs-parent deltas, which would carry the pre-trained arm's noise "
        "into every row.")
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
