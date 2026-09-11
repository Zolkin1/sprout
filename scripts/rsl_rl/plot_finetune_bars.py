#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""The AWR fine-tune result figure: success and terrain penalty, per terrain, per arm.

    python scripts/rsl_rl/plot_finetune_bars.py fig_numbers.json --out logs/paper/finetune_bars

Two panels, both clustered by TERRAIN and coloured by ARM (pre-trained vs fine-tuned), reading
the reduction `fig_reduce.py` produces.  Imports no IsaacLab and no torch.

**What the error bars are, and why not a binomial.**  They are a 95% cluster bootstrap over
``(tile, speed)`` cells, resampling the cell and then the episodes within it.  A binomial
interval over pooled episodes has an actual coverage of 14-70% against a nominal 95% on this
data, because episodes sharing a tile share its geometry -- so a pooled interval would be
several times too narrow and would keep getting narrower with more robots per tile.  The number
of CELLS is therefore the real limit on precision, and it is printed on the panel.

**Panel B is the requested mean penalty, and it is the weaker of the two panels.**  The total is
``68.0 * contact_penalty + 0.107 * penetration_sum_m`` -- the spec's own weights, which put each
term near 1.0 at its own p95 -- exposure-standardised over ``goal_downrange_m`` so a
longer-surviving arm is not charged for reaching deeper course.  Even so its ``penetration_sum_m``
half is a heavy-tailed MEAN: on the two-box course it rises while the rate of deep penetration
FALLS on every cell, so a few extreme plans move the mean against the direction every other
reduction agrees on.  ``--panel_b`` selects a better-behaved alternative; see its help.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import plot_style  # noqa: E402

#: The panel-B choices, as ``(json field, axis label, whether lower is better)``.
PANEL_B = {
    "penalty": ("pen", "Mean terrain penalty", True),
    "contact": ("con", "Mean contact penalty", True),
    "penetration": ("pnt", "Mean penetration penalty", True),
    # The deep-penetration RATE, at several thresholds. `max_penetration_m` is raw geometric
    # depth net of the 5 mm uncharged tolerance -- the per-body weights enter only the sum --
    # so these are real centimetres of a limb inside the terrain. The arm ordering is the same
    # at every threshold, which is what makes the choice presentational rather than a result.
    **{f"deep{c}": (f"deep{c:02d}",
                    rf"Deep penetration rate ($\geq${c} cm)", True)
       for c in (3, 4, 5, 6)},
}

#: Short, WRAPPED tick labels.  Matplotlib does not wrap tick text, and at the paper's 16 pt
#: serif two full terrain names in a 3.5 inch panel overlap into each other -- which is legible
#: in a draft and not in a submission.  Wrapping is explicit so the break point is a choice.
TERRAIN_LABELS = {
    "Two boxes": "Two\nboxes",
    "Stairs up-into-down": "Up-into-\ndown",
    "Stairs down (in-dist.)": "Stairs\ndown",
}

#: Clusters measured on the terrain the generator was TRAINED on, rather than on an
#: out-of-distribution course.  They are separated on the axis because the two kinds of claim
#: are different -- one is "it learns a course it has never seen", the other "it does not pay
#: for that on the terrain it already knew" -- and three undifferentiated clusters invite a
#: reader to take all three as OOD.
IN_DISTRIBUTION = ("Stairs down (in-dist.)",)

#: Course-class labels for the mode panel's clusters.
KLASS_LABELS = {"two_box_course": "Two boxes", "up_into_down": "Up-into-down"}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    """Point estimate and a 95% Wilson score interval for ``k`` of ``n``.

    **A binomial interval IS right here, unlike panels (a) and (b).**  Those pool episodes
    that share a tile, so their spread is clustered and a binomial reads several times too
    narrow.  A mode probe is N independent flow-noise draws from ONE fixed initial condition
    on ONE tile -- nothing is shared to cluster on -- so the trials really are exchangeable
    Bernoulli draws.  Wilson rather than normal because these rates sit at the ends of the
    scale, where the normal interval runs past 0 and 1 (one arm here is 50/50).

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile; the default is two-sided 95%.

    Returns:
        ``(p, lo, hi)``; ``(nan, nan, nan)`` when ``n`` is 0.
    """
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(centre - half, 0.0), min(centre + half, 1.0)


#: Arm labels and colours.  Pre-trained is the muted one and fine-tuned the saturated one, so
#: the comparison reads in greyscale as well as in colour.
ARMS = (("a", "Pre-trained", plot_style.COLORS["grey"]),
        ("b", "Fine-tuned", plot_style.COLORS["blue"]))


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("numbers", help="fig_numbers.json from fig_reduce.py")
    p.add_argument("--out", default="logs/paper/finetune_bars", help="output path, no extension")
    p.add_argument("--panel_b", default="penalty", choices=sorted(PANEL_B),
                   help="what the right panel shows. 'penalty' is the spec-weighted total (the "
                        "quantity the fine-tune optimises, but a heavy-tailed mean); 'deep3'.."
                        "'deep6' are the fraction of plans putting a limb that many cm into the "
                        "terrain, which is the tail measure and moves consistently on every "
                        "cell at every threshold; 'contact' and 'penetration' are the total's "
                        "two halves.")
    p.add_argument("--mode_json", nargs="*", default=(),
                   help="Reduced mode-probe numbers from `mode_probe.py --stage report "
                        "--json`, one file per (course, speed). Adds panel (c): the human "
                        "MODE labels, clustered by course and speed.")
    p.add_argument("--both_mode_panels", action="store_true",
                   help="Draw BOTH mode metrics, as panels (c) and (d) on a second row. The "
                        "two courses dissociate -- the two-box gain is selection and the "
                        "staircase gain is execution -- so either metric alone misreads one "
                        "of them. Off by default: one row of three is the figure that fits a "
                        "page, and the dropped panel is one flag away.")
    p.add_argument("--mode_metric", default="correct", choices=("correct", "survival"),
                   help="Which mode metric panel (c) shows; ignored under "
                        "--both_mode_panels. "
                        "'correct' is the fraction of trials whose FIRST mode was the right "
                        "one -- the selection question. 'survival' is the fraction that then "
                        "reached the goal, conditioned on having chosen right -- the execution "
                        "question. The two dissociate: on the two-box course the gain is "
                        "selection, on the staircase selection is near ceiling and the gain is "
                        "execution.")
    p.add_argument("--no_counts", action="store_true",
                   help="omit the cell/episode annotation.")
    return p.parse_args()


def _bars(ax, res: dict, terrains: list[str], field: str, ylabel: str) -> None:
    """One clustered bar panel with cluster-bootstrap error bars.

    Args:
        ax: The axes.
        res: The reduction, keyed by terrain name.
        terrains: Terrain order, left to right.
        field: The json field stem, e.g. ``"pen"``.
        ylabel: Axis label.
    """
    width = 0.36
    for j, (arm, label, colour) in enumerate(ARMS):
        xs = [i + (j - 0.5) * width for i in range(len(terrains))]
        ys = [res[t][f"{field}_{arm}"] for t in terrains]
        # Asymmetric by construction: a percentile bootstrap interval is not centred on the
        # estimate, and forcing it to be would misreport the direction of the uncertainty.
        lo = [res[t][f"{field}_{arm}"] - res[t][f"{field}_{arm}_lo"] for t in terrains]
        hi = [res[t][f"{field}_{arm}_hi"] - res[t][f"{field}_{arm}"] for t in terrains]
        ax.bar(xs, ys, width, label=label, color=colour, edgecolor="black", linewidth=1.2,
               zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=1.6,
                    capsize=5, capthick=1.6, zorder=3)
    ax.set_xticks(range(len(terrains)))
    ax.set_xticklabels([TERRAIN_LABELS.get(t, t) for t in terrains])
    # The cluster is the terrain, so ticks between the clusters would imply a scale that is
    # not there; the label sits under its pair and the tick mark itself is dropped.
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(terrains) - 0.4)
    ax.yaxis.grid(True, color="0.85", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    split = [i for i, t in enumerate(terrains) if t in IN_DISTRIBUTION]
    if split and 0 < split[0] < len(terrains):
        ax.axvline(split[0] - 0.5, color="0.55", linewidth=1.2, linestyle=(0, (4, 3)),
                   zorder=1)


def _mode_panel(ax, files: list[str], metric: str) -> list[str]:
    """Panel (c): the human mode labels, clustered by course and grouped by speed.

    The speed axis is the finding, not decoration.  On the two-box course the parent reads the
    box correctly in 50 of 50 trials at 1.17 m/s and misreads it as a staircase in 38 of 50 at
    2.50 -- so the mode failure the fine-tune repairs is SPEED-INDUCED, and a panel at either
    speed alone would say something false: at 1.17 that there is no effect, at 2.50 that mode
    selection is broken in general.

    Args:
        ax: The axes.
        files: The reduced json files, one per (course, speed).
        metric: ``"correct"`` or ``"survival"``; see the CLI help.

    Returns:
        Cluster labels, so the caller can annotate counts.
    """
    recs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            recs.append(json.load(fh))
    # Course first, speed within it, so a cluster pair reads left-to-right as slow then fast.
    recs.sort(key=lambda r: (KLASS_LABELS.get(r["klass"], r["klass"]), r["speed"]))
    labels = [f"{KLASS_LABELS.get(r['klass'], r['klass'])}\n{r['speed']:.2f} m/s" for r in recs]

    width = 0.36
    for j, (arm, label, colour) in enumerate(ARMS):
        xs, ys, lo, hi = [], [], [], []
        for i, r in enumerate(recs):
            a = r["arms"].get({"a": "parent", "b": "finetuned"}[arm])
            if a is None:
                continue
            k, n = ((a["correct"], a["n"]) if metric == "correct"
                    else (a["surv_success"], a["surv_n"]))
            pt, l, h = wilson(int(k), int(n))
            xs.append(i + (j - 0.5) * width)
            ys.append(pt)
            lo.append(pt - l)
            hi.append(h - pt)
        ax.bar(xs, ys, width, label=label, color=colour, edgecolor="black", linewidth=1.2,
               zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=1.6,
                    capsize=5, capthick=1.6, zorder=3)
    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Correct first mode" if metric == "correct"
                  else r"Reached goal $\mid$ correct mode")
    ax.set_xlim(-0.6, len(recs) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, color="0.85", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for i in range(1, len(recs)):
        if recs[i]["klass"] != recs[i - 1]["klass"]:
            ax.axvline(i - 0.5, color="0.55", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    return [f"{KLASS_LABELS.get(r['klass'], r['klass'])} @ {r['speed']:.2f}: "
            f"{r['arms'][next(iter(r['arms']))]['n']} trials/arm" for r in recs]


def main() -> int:
    """Draw and save the figure."""
    args = parse_args()
    with open(args.numbers, encoding="utf-8") as fh:
        res = json.load(fh)
    meta = res.pop("_meta", {})
    terrains = [k for k in res]
    # The penalty weights live PER ARM (`fig_reduce` stamps each one from that chunk group's
    # own `spec`), not in `_meta` -- reading them off `_meta` printed "contact None,
    # penetration None" into the caption, which would have gone into the paper.
    #
    # They must AGREE across arms: the caption states one pair of weights, so two arms reduced
    # under different ones would be mislabelled rather than merely unlabelled. A `spec` differs
    # between arms exactly when their recordings were graded under different reward
    # calibrations, which is a comparison that should not be plotted together at all.
    _wk = ("contact_weight", "penetration_weight")
    _seen = {tuple(res[k].get(w) for w in _wk) for k in terrains}
    if len(_seen) > 1:
        raise SystemExit(
            "REFUSING: these arms were graded under DIFFERENT penalty weights, so one caption "
            "cannot describe them:\n"
            + "\n".join(f"      {dict(zip(_wk, tuple(res[k].get(w) for w in _wk)))}  <- {k}"
                         for k in terrains)
            + "\n  Re-grade the arms against one reward spec before plotting them together.")
    weights = dict(zip(_wk, next(iter(_seen)))) if _seen else {}
    field, ylabel, _ = PANEL_B[args.panel_b]

    plot_style.apply_paper_style()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    base_w, base_h = plot_style.figsize(cols=2)
    n_mode = len(args.mode_json)
    mode_note: list[str] = []
    if n_mode and args.both_mode_panels:
        # Two rows: the episode metrics on top, both mode metrics below. The mode row is the
        # wider one (four cells against three terrains), so the width follows IT.
        fig_w = base_w * max(len(terrains), n_mode) * 2.0 / 4.0
        fig, grid = plt.subplots(2, 2, figsize=(fig_w, base_h * 2.0))
        axes = [grid[0][0], grid[0][1], grid[1][0], grid[1][1]]
        _bars(axes[0], res, terrains, "succ", "Success rate")
        _bars(axes[1], res, terrains, field, ylabel)
        mode_note = _mode_panel(axes[2], list(args.mode_json), "correct")
        _mode_panel(axes[3], list(args.mode_json), "survival")
    elif n_mode:
        # One row of three. Width follows the CLUSTER count rather than the panel count: a
        # wrapped tick label is as wide as its own bar pair, so a panel cannot be squeezed
        # without its labels colliding.
        widths = [len(terrains), len(terrains), n_mode]
        fig, grid = plt.subplots(1, 3, figsize=(base_w * sum(widths) / 4.0, base_h),
                                 gridspec_kw={"width_ratios": widths})
        axes = list(np.atleast_1d(grid))
        _bars(axes[0], res, terrains, "succ", "Success rate")
        _bars(axes[1], res, terrains, field, ylabel)
        mode_note = _mode_panel(axes[2], list(args.mode_json), args.mode_metric)
    else:
        fig_w = base_w * (1.0 + 0.20 * max(len(terrains) - 2, 0))
        fig, grid = plt.subplots(1, 2, figsize=(fig_w, base_h))
        axes = list(np.atleast_1d(grid))
        _bars(axes[0], res, terrains, "succ", "Success rate")
        _bars(axes[1], res, terrains, field, ylabel)
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, None)
    for ax, tag in zip(axes, "abcd"):
        ax.set_title(f"({tag})", loc="left")
    axes[0].legend(loc="upper left", frameon=True)

    if not args.no_counts:
        # The cell count is the precision limit, so it belongs on the figure rather than only in
        # a caption that may not travel with it.
        note = " | ".join(f"{t}: {res[t]['n_cells']} cells / {res[t]['n_tiles']} tiles, "
                          f"{res[t]['n_episodes_a']} ep/arm" for t in terrains)
        if mode_note:
            note += "\n(c) one tile, noise draws only: " + " | ".join(mode_note)
        fig.text(0.5, -0.14, note, ha="center", va="top", fontsize=10, color="0.3")

    # Name the two regimes once, under the panel that has room, rather than in the caption
    # alone -- a figure travels without its caption more often than not.
    split = [i for i, t in enumerate(terrains) if t in IN_DISTRIBUTION]
    if split and 0 < split[0] < len(terrains):
        mid_ood = (split[0] - 1) / 2.0
        mid_id = (split[0] + len(terrains) - 1) / 2.0
        for ax in axes[:2]:
            for x, lbl in ((mid_ood, "out-of-distribution"), (mid_id, "in-distribution")):
                ax.annotate(lbl, xy=(x, 1.015), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=12, color="0.35",
                            annotation_clip=False)

    fig.tight_layout()
    caption = (
        "AWR fine-tuning of the flow-matching generator on two out-of-distribution courses. "
        f"(a) success rate; (b) {ylabel.lower()}. Bars are cell-averaged; error bars are 95% "
        "cluster bootstrap intervals over (tile, speed) cells, resampling the cell and then the "
        "episodes within it -- a pooled binomial interval has 14-70% actual coverage here, "
        "because episodes sharing a tile share its geometry. Success is scored over each "
        "environment's first episode, the design whose membership cannot depend on episode "
        "duration. The penalty is exposure-standardised over downrange position, so a "
        "longer-surviving arm is not charged for reaching deeper course."
        + (f" Penalty weights: contact {weights['contact_weight']:g}, penetration "
           f"{weights['penetration_weight']:g}."
           if weights.get("contact_weight") is not None else ""))
    paths = plot_style.save_figure(fig, args.out, caption=caption)
    for p in paths:
        print(f"wrote {p}")
    print(f"\npanel (b) = {args.panel_b}: " + ", ".join(
        f"{t} {res[t][field + '_a']:.4f} -> {res[t][field + '_b']:.4f} "
        f"({res[t][field + '_cells_down']}/{res[t]['n_cells']} cells down)" for t in terrains))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
