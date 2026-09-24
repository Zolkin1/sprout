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
    # The same update with the conditioning perturbations off. Its own hue rather than a muted
    # AWR: grey read as "de-emphasised" next to four saturated arms, which is the opposite of
    # what an ablation row should say.
    "AWR no-perturb": plot_style.COLORS["magenta"],
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
    "AWR no-perturb": "AWR (no pert.)",
    "Arrived": "Arrival Filt.",
    "Filtered BC": "Filt. BC",
}

#: Same idea for the block labels down the right-hand edge. `--exclude` still matches the
#: CANONICAL name, because that is what `forest_groups.json` and both json files are keyed on --
#: matching the display name would mean the flag stopped working the moment a block was renamed.
GROUP_DISPLAY: dict[str, str] = {
    "Multi, strict": "Multi-terrain, strict",
}

#: Row geometry, in row-units, converted to inches by ROW_IN. The block name is its own row above
#: its arms (the meta-analysis subgroup convention) rather than rotated text down the right edge:
#: at this figure's printed size that text came out both tiny and sideways.
ARM_ROW, BLOCK_GAP, ROW_IN = 1.0, 0.8, 0.38

#: Block names read down the RIGHT edge, rotated, so they never compete with the arm labels for
#: the left margin. They were unreadable before not because of the placement but because the
#: figure was declared at 9.79 in and printed in a ~3.4 in column -- at that 36% they were
#: sub-3 pt. Declared 1:1 (PAGE_W_IN) this size is what it says it is.
BLOCK_LABEL_PT = 12.5

#: Where the arm names sit, as a fraction of the FIRST PANEL's width, measured from its left
#: spine. Anchoring them to the panel rather than to the label strip is what keeps them a fixed
#: distance from the data: pinned to the strip's right edge instead, the visible gap is whatever
#: `tight_layout` happens to leave as `wspace`, which changes with the block count and the font.
ARM_LABEL_X = -0.015

#: Gap between columns, as a fraction of the mean axes width, set EXPLICITLY. `tight_layout`
#: derives one `wspace` from the widest thing it has to fit in any gap and then applies it to all
#: of them, so the arm labels hanging off panel 1 pushed the two PANELS 0.91 in apart. Owning the
#: value keeps the label gap and the panel gap from being the same number by construction.
PANEL_WSPACE = 0.07

#: Both label sets hang off the LEFT of panel 1, and there is no separate label axes. A third
#: column was tried and is worse: `tight_layout` derives ONE `wspace` from the widest thing in
#: any gap, so labels sitting in the strip-to-panel gap were reproduced as an identical gap
#: BETWEEN the two panels (measured 0.91 in in both). With two panels there is a single gap,
#: `PANEL_WSPACE` owns it, and the labels land in the figure's LEFT MARGIN -- which
#: `tight_layout` sizes from panel 1's tight bbox, overhanging text included.
#:
#: Clearance between the rotated block name and the longest arm label, in panel-width fractions.
BLOCK_LABEL_PAD = 0.035

#: IEEE two-column text width. The figure is DECLARED at the width it is printed at, because
#: LaTeX scales whatever it is given: at the previous 9.79 in declaration every label on the page
#: rendered at 73% of its intended size, which no amount of bolding recovers.
PAGE_W_IN = 7.16

#: LaTeX preamble per font choice. Matplotlib's `usetex` default is Computer Modern, and IEEE
#: classes typeset in Times -- at the SAME point size Computer Modern's strokes are lighter, so
#: the figure reads thinner than the body text beside it. That is a TYPEFACE difference, not a
#: weight one, and bolding to compensate overshoots: it lands heavier than the body and reads as
#: emphasis on labels that are not emphatic.
FONT_PREAMBLE: dict[str, str] = {
    "cm": r"\usepackage{amsmath} \usepackage{amssymb}",
    # `amssymb` and `newtxmath` both define \Bbbk, and loading the second one second is a hard
    # LaTeX error ("Command \Bbbk already defined") -- matplotlib surfaces it as a failed
    # render with no figure written. Let the earlier definition go before newtx redefines it.
    "times": (r"\usepackage{amsmath} \usepackage{amssymb}"
              r" \let\Bbbk\relax \usepackage{newtxtext,newtxmath}"),
}

#: `plot_style` is built for DRAW-BIG-THEN-DOWNSCALE: 16-20 pt text and 4 pt lines that come out
#: right once LaTeX shrinks a ~10 in figure into a column. This figure declares the width it is
#: printed at instead, so those sizes would land on the page at face value -- twice the body
#: text. These are 1:1 print sizes, near the 8 pt that IEEE figure text usually runs at.
PRINT_RC: dict[str, float] = {
    "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.labelpad": 3,
}
#: Interval bar and point marker, in points, matched to the text above.
BAR_LW, DOT_MS, REF_MS = 2.1, 5.8, 6.3


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
    ap.add_argument("--font", choices=sorted(FONT_PREAMBLE), default="times",
                    help="typeface: 'times' matches an IEEE document, 'cm' is matplotlib's "
                         "Computer Modern default (lighter strokes at the same size)")
    ap.add_argument("--arm_rotation", type=float, default=0.0, metavar="DEG",
                    help="tilt the arm labels; the row pitch grows to keep them from "
                         "overlapping, so the figure gets TALLER as it gets narrower")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"),
                    help="clip the success panel; arms outside get an edge caret, not silence")
    ap.add_argument("--xlim2", type=float, nargs=2, metavar=("LO", "HI"),
                    help="same for the contact panel")
    ap.add_argument("--width", type=float, default=PAGE_W_IN,
                    help="figure width in inches; declare it at the width it is PRINTED at")
    ap.add_argument("--xlabel", default="Success rate (rel. to AWR)")
    ap.add_argument("--xlabel2", default="Mean contact penalty (rel. to AWR)")
    a = ap.parse_args()

    tex = plot_style.apply_paper_style()
    plt.rcParams.update(PRINT_RC)
    if tex:
        plt.rcParams["text.latex.preamble"] = FONT_PREAMBLE[a.font]
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

    # A tilted label eats VERTICAL space (w*sin + h*cos), so the rows have to be pushed apart or
    # it writes over its neighbours. Measured on a throwaway figure because the real one's height
    # depends on the answer -- and measured rather than assumed, since the widest label changes
    # with the data. At 45 deg the horizontal and vertical extents are equal by construction, so
    # the trade is poor for a label much wider than it is tall; 90 deg is where it pays off.
    row_in = ROW_IN
    if a.arm_rotation:
        import math  # noqa: PLC0415

        probe = plt.figure(figsize=(1, 1))
        pr = probe.canvas.get_renderer()
        rad = math.radians(a.arm_rotation)
        need = 0.0
        for r in rows:
            t = probe.text(0.5, 0.5, DISPLAY_NAMES.get(r["arm"], r["arm"]),
                           fontsize=plt.rcParams["ytick.labelsize"])
            bb = t.get_window_extent(renderer=pr)
            w, h = bb.width / probe.dpi, bb.height / probe.dpi
            need = max(need, w * abs(math.sin(rad)) + h * abs(math.cos(rad)))
        plt.close(probe)
        row_in = max(ROW_IN, need / ARM_ROW * 1.08)   # 8% breathing room between rows
        if row_in > ROW_IN:
            print(f"[plot_forest] arm_rotation={a.arm_rotation:g} needs row pitch "
                  f"{row_in:.3f} in (was {ROW_IN:.3f}) -- figure grows taller")

    ys, labels, arms, seps, ticks = [], [], [], [], []
    y = 0.0
    for gi, g in enumerate(reversed(groups)):
        grows = [r for r in rows if r["group"] == g]
        for r in reversed(grows):
            ys.append((y, r))
            labels.append(DISPLAY_NAMES.get(r["arm"], r["arm"]))
            arms.append(r["arm"])
            y += ARM_ROW
        ticks.append((y - len(grows) / 2.0 - 0.5, g))
        if gi < len(groups) - 1:
            seps.append(y - 0.5)
            y += BLOCK_GAP

    contact = {}
    if a.contact:
        for r in json.loads(pathlib.Path(a.contact).read_text())["rows"]:
            contact[(r["group"], r["arm"])] = r

    npanel = 2 if contact else 1
    fig, axes = plt.subplots(1, npanel, sharey=True,
                             figsize=(a.width, row_in * y + 1.1))
    axes = np.atleast_1d(axes)

    # Colour by METHOD (see METHOD_COLOURS), assigned once over the arms actually present so
    # every block draws a given method the same and the eye can track it down the figure.
    seen: list[str] = []
    for _, r in ys:
        if r["arm"] not in seen:
            seen.append(r["arm"])
    colour = {arm: METHOD_COLOURS.get(arm, _SPARE[i % len(_SPARE)])
              for i, arm in enumerate(seen)}
    lims = [a.xlim, a.xlim2]
    for ax, (src, xlabel, lim) in zip(
            axes, [(None, a.xlabel, lims[0]), (contact, a.xlabel2, lims[1])][:npanel]):
        ax.axvline(0.0, color="0.35", lw=0.9, ls=(0, (4, 3)), zorder=1)
        for pos, r in ys:
            row = r if src is None else src.get((r["group"], r["arm"]))
            if row is None:
                continue
            col = colour[r["arm"]]
            if r.get("reference"):
                ax.plot([0.0], [pos], marker="D", ms=REF_MS, color=col, zorder=4)
                continue
            # `lo`/`hi` are the CLUSTER-T interval, which `paired_delta.report` and
            # `fig_reduce` now both emit as the headline; `boot_lo`/`boot_hi` carry the
            # single-stage bootstrap and are the robustness check, not drawn.
            ax.plot([row["lo"], row["hi"]], [pos, pos], color=col, lw=BAR_LW,
                    solid_capstyle="round", zorder=3)
            ax.plot([row["delta"]], [pos], marker="o", ms=DOT_MS, color=col, zorder=4)
        if lim:
            ax.set_xlim(*lim)
            # An arm off the edge must not read as an arm with no data. A caret at the boundary
            # in the arm's own colour says "continues past here"; the number is in the table and
            # the caption. Without it a clipped row is indistinguishable from a missing one --
            # and this figure already has genuinely missing arms.
            lo, hi = lim
            for pos, r in ys:
                row = r if src is None else src.get((r["group"], r["arm"]))
                if row is None or r.get("reference"):
                    continue
                d = row["delta"]
                if d < lo or d > hi:
                    # Just INSIDE the panel, not on the boundary: drawn exactly on the edge with
                    # clip_on=False the caret lands on top of the arm's own tick label.
                    inset = 0.022 * (hi - lo)
                    left = d < lo
                    ax.plot([lo + inset if left else hi - inset], [pos],
                            marker="<" if left else ">", ms=DOT_MS + 1.5,
                            color=colour[r["arm"]], zorder=5)
                    # Print the value the caret points at. A caret alone says "off this way"
                    # without saying how far, and the difference between -0.06 and -0.45 is the
                    # whole point of the row. Decimals follow the axis range so a contact
                    # penalty does not get rounded to "+0.09" on a 0.03-wide axis.
                    dec = 3 if (hi - lo) < 0.1 else 2
                    ax.text(lo + 2.4 * inset if left else hi - 2.4 * inset, pos,
                            f"${d:+.{dec}f}$", color=colour[r["arm"]],
                            fontsize=BLOCK_LABEL_PT - 2.0, va="center",
                            ha="left" if left else "right", zorder=5)
        ax.set_xlabel(xlabel)
        # Title above the panel it describes. `label_position` moves the text; `tick_top` moves
        # the numbers with it, and without the second call the ticks stay at the bottom and the
        # axis reads as though it belonged to whatever is below.
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.xaxis.grid(True, color="0.90", lw=0.6)
        ax.set_axisbelow(True)
        for sp in seps:
            ax.axhline(sp, color="0.85", lw=0.6, zorder=0)

    ax = axes[0]
    ax.set_ylim(-0.8, y - BLOCK_GAP + 0.3)
    ax.set_yticks([p for p, _ in ys])
    ax.tick_params(axis="y", length=0, labelleft=False)

    # Arm names hang off panel 1 (see ARM_LABEL_X); `get_yaxis_transform` is x-in-axes-fraction,
    # y-in-data, so a row lands on the row it labels. `clip_on=False` lets them render outside
    # the axes they are anchored to.
    # `rotation_mode="anchor"` rotates about the anchor point rather than the text's centre, so
    # a tilted label's right end stays put against the panel instead of drifting away from it.
    texts = [ax.text(ARM_LABEL_X, pos, lab, transform=ax.get_yaxis_transform(),
                     va="center", ha="right", clip_on=False,
                     rotation=a.arm_rotation, rotation_mode="anchor",
                     fontsize=plt.rcParams["ytick.labelsize"], color=colour[arm])
             for (pos, _), lab, arm in zip(ys, labels, arms)]

    # Block names sit just clear of the WIDEST arm label, measured rather than guessed: the
    # widest one changes with the data and with the font, and a fixed offset either collides or
    # leaves a gap. One draw is needed before any text has an extent.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    panel_px = ax.get_window_extent().width
    widest = max(t.get_window_extent(renderer=rend).width for t in texts) / panel_px
    for pos, name in ticks:
        ax.text(ARM_LABEL_X - widest - BLOCK_LABEL_PAD, pos, GROUP_DISPLAY.get(name, name),
                transform=ax.get_yaxis_transform(), rotation=90,
                va="center", ha="center", clip_on=False,
                fontsize=BLOCK_LABEL_PT, color="0.15")

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
        "into every row."
        + (" The axis is clipped to resolve the arms that sit near zero; an arm falling outside "
           "it is drawn as a caret at that edge with its value printed beside it, so a clipped "
           "row is never mistaken for a missing one." if (a.xlim or a.xlim2) else ""))
    # exact_size: the default tight bbox crops to ink, so the file would come out narrower than
    # `--width` and LaTeX would scale it back up -- reintroducing exactly the shrink this is
    # meant to remove.
    # Lay out first, then set the column gap, then save WITHOUT letting save_figure re-run
    # tight_layout -- it would recompute wspace from the label overhang and undo this.
    fig.tight_layout()
    fig.subplots_adjust(wspace=PANEL_WSPACE)
    for f in plot_style.save_figure(fig, a.out, caption=caption, exact_size=True, tight=False):
        print(f"wrote {f}")
    for pos, r in ys:
        if r.get("reference"):
            print(f"  {r['group']:16s} {r['arm']:12s}  reference")
        else:
            print(f"  {r['group']:16s} {r['arm']:12s} {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  p={r['mcnemar_p']:.3g}")


if __name__ == "__main__":
    main()
