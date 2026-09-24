#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""The AWR fine-tune result figure: success and terrain penalty, per terrain, per arm.

    python scripts/rsl_rl/plot_finetune_bars.py fig_numbers.json --out logs/paper/finetune_bars

Two panels, both clustered by TERRAIN and coloured by ARM (pre-trained vs fine-tuned), reading
the reduction `fig_reduce.py` produces.  Imports no IsaacLab and no torch.

**THE INTERVALS.** 95% cluster-t over ``(tile, speed)`` cells: one value per cell, then
``mean +- t(C-1) * SD / sqrt(C)``. Episodes sharing a tile share its geometry, so the CELL is the
unit of independence and C -- not the episode count -- is the sample size. A single-stage cluster
bootstrap (resampling cells, contents intact) agrees with this to within 0.0008 on every arm
measured, and is the robustness check rather than the headline.

Do NOT resample cells and then episodes within them: a cell's own value already carries
within-cell sampling noise, so the second stage counts it twice. Measured here, that inflated
widths by 14-29% and flipped one arm's conclusion.

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
    # The SAME contact statistic, plotted as `1 - x` so the panel reads the way (a) does.
    # A linear flip is the only faithful way to turn this one upward: it preserves every gap
    # exactly, and because it is monotone the per-cell ordering, the sign test and the
    # percentile intervals all carry over unchanged. The two flips a reader might reach for
    # instead do NOT: `contact_reward` is the legacy hover kernel (it measures contact
    # COVERAGE and falls under this fine-tune), and a zero-penalty "clean plan" rate also
    # falls -- the gain here is fewer SEVERE contacts, not more perfect plans.
    "contact_score": ("con", r"Contact score ($1-$penalty)", False),
}

#: Short, WRAPPED tick labels.  Matplotlib does not wrap tick text, and at the paper's 16 pt
#: serif two full terrain names in a 3.5 inch panel overlap into each other -- which is legible
#: in a draft and not in a submission.  Wrapping is explicit so the break point is a choice.
TERRAIN_LABELS = {
    "Two boxes": "Boxes",
    "Stairs up-into-down": "Stairs",
    "Stairs down (in-dist.)": "Stairs\ndown",
    "Varied multi-terrain": "Multi-\nterrain",
    "In-distribution": "In-\ndist.",
}

#: Clusters measured on the terrain the generator was TRAINED on, rather than on an
#: out-of-distribution course.  They are separated on the axis because the two kinds of claim
#: are different -- one is "it learns a course it has never seen", the other "it does not pay
#: for that on the terrain it already knew" -- and three undifferentiated clusters invite a
#: reader to take all three as OOD.
IN_DISTRIBUTION = ("Stairs down (in-dist.)", "In-distribution")

#: Course-class labels for the mode panel's clusters.
#: The heading over each (c) cluster names the COURSE, in the same words as TERRAIN_LABELS --
#: (c) sits beside (a) and (b) on one figure and one course under two names reads as two
#: courses. The TICK under each bar pair names the specific tile instead ("Two boxes",
#: "Stairs (wide)"), so the two lines read general-over-specific rather than repeating.
#:
#: `stairs_up` is the entry staircase of the MULTI-TERRAIN course -- the class says what the
#: geometry is, the heading says which course it was probed on, and in this campaign the only
#: stairs_up probe is that one. A stairs_up probe rolled on a different course would need its
#: own key rather than this label.
KLASS_LABELS = {"two_box_course": "Boxes", "up_into_down": "Stairs",
                "stairs_up": "Multi-terrain", "stairs_down": "Multi-terrain",
                "box": "Box"}


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
                        "--json`, one file per probe. Adds panel (c): the human MODE labels. "
                        "Write `path=Label` to name a cluster explicitly (a literal newline "
                        "in the label wraps it); otherwise the label is the course class, "
                        "with the TILE ID appended whenever two probes would otherwise get "
                        "the same one -- several tiles of a single course is the normal "
                        "shape now, and two identically-labelled clusters is a figure that "
                        "cannot be read.")
    p.add_argument("--both_mode_panels", action="store_true",
                   help="Draw BOTH mode metrics, as panels (c) and (d) on a second row. The "
                        "two courses dissociate -- the two-box gain is selection and the "
                        "staircase gain is execution -- so either metric alone misreads one "
                        "of them. Off by default: one row of three is the figure that fits a "
                        "page, and the dropped panel is one flag away.")
    p.add_argument("--mode_scope", default="all", choices=("all", "first"),
                   help="What the human actually graded, which sets panel (c)'s axis label. "
                        "'all' (default) is a session run with `mode_probe --stage label "
                        "--whole`, where the operator watches the traverse and the trial "
                        "counts as correct only if EVERY mode decision was right. 'first' is "
                        "the truncated label window, where only the opening decision is "
                        "judged. The stored counts are identical -- `correct` over `n` either "
                        "way -- so nothing but the reader can tell them apart, which is why "
                        "it is stated rather than inferred.")
    p.add_argument("--mode_metric", default="correct", choices=("correct", "survival"),
                   help="Which mode metric panel (c) shows; ignored under "
                        "--both_mode_panels. "
                        "'correct' is the fraction of trials whose mode labels were all right "
                        "-- the selection question; see --mode_scope. 'survival' is the "
                        "fraction that then "
                        "reached the goal, conditioned on having chosen right -- the execution "
                        "question. The two dissociate: on the two-box course the gain is "
                        "selection, on the staircase selection is near ceiling and the gain is "
                        "execution.")
    p.add_argument("--no_counts", action="store_true",
                   help="omit the cell/episode annotation.")
    return p.parse_args()


def _bars(ax, res: dict, terrains: list[str], field: str, ylabel: str,
          flip: bool = False) -> None:
    """One clustered bar panel with cluster-bootstrap error bars.

    Args:
        ax: The axes.
        res: The reduction, keyed by terrain name.
        terrains: Terrain order, left to right.
        field: The json field stem, e.g. ``"pen"``.
        ylabel: Axis label.
        flip: Plot ``1 - value``, so a lower-is-better quantity points the same way as the
            success panel. The interval flips with it -- ``[lo, hi]`` becomes
            ``[1 - hi, 1 - lo]`` -- because a percentile interval maps through a monotone
            transform exactly.

    Raises:
        SystemExit: If a flipped value or interval bound would fall below zero, i.e. the
            penalty exceeded 1 and ``1 - x`` is no longer a score on the same scale as (a).
    """
    width = 0.36
    for j, (arm, label, colour) in enumerate(ARMS):
        xs = [i + (j - 0.5) * width for i in range(len(terrains))]
        vs = [res[t][f"{field}_{arm}"] for t in terrains]
        ls = [res[t][f"{field}_{arm}_lo"] for t in terrains]
        hs = [res[t][f"{field}_{arm}_hi"] for t in terrains]
        if flip and min([1.0 - h for h in hs] + [1.0 - v for v in vs]) < 0.0:
            raise SystemExit(
                f"REFUSING to flip {field!r}: a value or interval bound exceeds 1, so "
                "`1 - x` is not a score on the same 0-1 scale as panel (a). Plot the "
                "quantity itself.")
        ys = [1.0 - v for v in vs] if flip else vs
        # Asymmetric by construction: a percentile bootstrap interval is not centred on the
        # estimate, and forcing it to be would misreport the direction of the uncertainty.
        # Under the flip the two arms swap roles: the upper bound becomes the lower one.
        lo = [h - v for v, h in zip(vs, hs)] if flip else [v - l for v, l in zip(vs, ls)]
        hi = [v - l for v, l in zip(vs, ls)] if flip else [h - v for v, h in zip(vs, hs)]
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


def _mode_panel(ax, files: list[str], metric: str, scope: str = "all") -> list[str]:
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
        scope: ``"all"`` or ``"first"`` -- what the operator graded; sets the axis label.

    Returns:
        Cluster labels, so the caller can annotate counts.
    """
    recs = []
    for f in files:
        path, _, given = f.partition("=")
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        r["_label"] = given.replace("\\n", "\n") if given else ""
        recs.append(r)
    # Course first, speed within it, so a cluster pair reads left-to-right as slow then fast
    # -- but a caller who NAMED every cluster gets the order they passed, because the point of
    # naming them is usually to line panel (c) up with panel (a), and (a)'s order is the
    # reduction's, not alphabetical.
    if not all(r["_label"] for r in recs):
        recs.sort(key=lambda r: (KLASS_LABELS.get(r["klass"], r["klass"]), r["speed"],
                                 int(r.get("terrain_id", -1))))
    # The tile id joins the label only where it has to. Several tiles of ONE course is the
    # normal shape of this probe now (the boxes course alone has four rolled), and without
    # this two of them render as the same cluster label with different bars under it.
    stem = [r["_label"] or KLASS_LABELS.get(r["klass"], r["klass"]) for r in recs]
    dup = {t for t in stem if stem.count(t) > 1}
    labels = []
    for r, t in zip(recs, stem):
        head = f"{t} (t{int(r['terrain_id'])})" if t in dup and not r["_label"] else t
        # No speed on the tick: every probe in this figure is at the same one, so repeating it
        # four times says nothing and costs a line of label height. The speed is still in the
        # count note under the figure and belongs in the caption.
        labels.append(head)

    width = 0.36
    for j, (arm, label, colour) in enumerate(ARMS):
        xs, ys, lo, hi = [], [], [], []
        for i, r in enumerate(recs):
            a = r["arms"].get({"a": "parent", "b": "finetuned"}[arm])
            if a is None:
                continue
            k, n = ((a["correct"], a["n"]) if metric == "correct"
                    else (a["surv_success"], a["surv_n"]))
            if metric == "survival" and n > 0 and k == 0 and a["correct"] > 0:
                raise SystemExit(
                    f"REFUSING --mode_metric survival on {r.get('course')!r} tile "
                    f"{r.get('terrain_id')}: {a['correct']} trials chose the right mode and "
                    "NONE is recorded as surviving. `mode_probe` rolls with "
                    "stop_on_goal=False, and `runner.run_episode` only ever sets "
                    "verdict='success' inside that branch -- so this is a structural zero, "
                    "not a measurement. Use --mode_metric correct, and take arrival from "
                    "`add_goal_arrival.py`'s goal.json instead.")
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
    ax.set_ylabel(("All modes correct" if scope == "all" else "Correct first mode")
                  if metric == "correct" else r"Reached goal $\mid$ correct mode")
    ax.set_xlim(-0.6, len(recs) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, color="0.85", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    # Head each dashed group with its COURSE, the same device panels (a) and (b) use for the
    # two regimes: the dashed rule alone says "these belong together" without saying what they
    # are, and the cluster labels below then have to carry both the course and the variant.
    head_of = [KLASS_LABELS.get(r["klass"], r["klass"]) for r in recs]
    bounds = [0] + [i for i in range(1, len(recs))
                    if head_of[i] != head_of[i - 1]] + [len(recs)]
    for i in bounds[1:-1]:
        ax.axvline(i - 0.5, color="0.55", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        # Black, matching the regime labels over (a) and (b): the three sit in one band across
        # the top of the figure, so one grey and one black reads as an oversight.
        ax.annotate(head_of[lo],
                    xy=((lo + hi - 1) / 2.0, 1.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=12,
                    color=plot_style.COLORS["black"], annotation_clip=False)
    # The note uses the CLUSTER's own label, not the class label: with several tiles per
    # course the class label repeats, and a note reading "Up-into-down ... Up-into-down" names
    # neither of them. Per-arm counts are spelled out whenever they differ, which happens
    # while a probe is part-graded -- a figure must not round 24 up to "25 trials/arm".
    out = []
    for r, t in zip(recs, stem):
        ns = {a: int(v["n"]) for a, v in r["arms"].items()}
        cnt = (f"{next(iter(ns.values()))} trials/arm" if len(set(ns.values())) == 1
               else ", ".join(f"{a} {n}" for a, n in sorted(ns.items())) + " trials")
        out.append(f"{t.replace(chr(10), ' ')} @ {r['speed']:.2f}: {cnt}")
    return out


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
    field, ylabel, _lower_better = PANEL_B[args.panel_b]
    flip_b = not _lower_better

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
        _bars(axes[1], res, terrains, field, ylabel, flip=flip_b)
        mode_note = _mode_panel(axes[2], list(args.mode_json), "correct", args.mode_scope)
        _mode_panel(axes[3], list(args.mode_json), "survival", args.mode_scope)
    elif n_mode:
        # One row of three. Width follows the CLUSTER count rather than the panel count: a
        # wrapped tick label is as wide as its own bar pair, so a panel cannot be squeezed
        # without its labels colliding.
        widths = [len(terrains), len(terrains), n_mode]
        fig, grid = plt.subplots(1, 3, figsize=(base_w * sum(widths) / 4.0, base_h),
                                 gridspec_kw={"width_ratios": widths})
        axes = list(np.atleast_1d(grid))
        _bars(axes[0], res, terrains, "succ", "Success rate")
        _bars(axes[1], res, terrains, field, ylabel, flip=flip_b)
        mode_note = _mode_panel(axes[2], list(args.mode_json), args.mode_metric,
                                args.mode_scope)
    else:
        fig_w = base_w * (1.0 + 0.20 * max(len(terrains) - 2, 0))
        fig, grid = plt.subplots(1, 2, figsize=(fig_w, base_h))
        axes = list(np.atleast_1d(grid))
        _bars(axes[0], res, terrains, "succ", "Success rate")
        _bars(axes[1], res, terrains, field, ylabel, flip=flip_b)
    axes[0].set_ylim(0.0, 1.0)
    # A flipped panel shares (a)'s axis on purpose: the point of the flip is that the two
    # panels read alike, and an auto-scaled 0.6-1.0 axis would exaggerate the gaps it keeps.
    axes[1].set_ylim(0.0, 1.0 if flip_b else None)
    for ax, tag in zip(axes, "abcd"):
        ax.set_title(f"({tag})", loc="left")
    # One ROW of two entries, not a two-row box: the cluster count changes between
    # campaigns, and a tall upper-left box clipped the second cluster's bar at four
    # clusters. A single row sits above every bar on a 0-1 success axis.
    axes[0].legend(loc="upper left", ncol=2, frameon=True)

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
                            ha="center", va="bottom", fontsize=12,
                            color=plot_style.COLORS["black"], annotation_clip=False)

    fig.tight_layout()
    # The OOD course count is COUNTED, not stated: this figure is redrawn per campaign and the
    # number of clusters changes with it, so a hardcoded "two" silently mislabels the next one.
    n_ood = sum(1 for t in terrains if t not in IN_DISTRIBUTION)
    n_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(n_ood, str(n_ood))
    caption = (
        f"AWR fine-tuning of the flow-matching generator on {n_word} out-of-distribution "
        f"course{'' if n_ood == 1 else 's'}. "
        f"(a) success rate; (b) {ylabel.lower()}. Bars are cell-averaged. "
        "Error bars are 95% cluster-t intervals over (tile, speed) cells -- one value "
        "per cell, mean +- t(C-1) x SD/sqrt(C) -- because episodes sharing a tile share "
        "its geometry, so the cell is the unit of independence and the cell count is "
        "the sample size. A single-stage cluster bootstrap agrees to within 0.0008. "
        "Success is scored over each environment's first episode, the design whose "
        "membership cannot depend on episode "
        "duration. The penalty is exposure-standardised over downrange position, so a "
        "longer-surviving arm is not charged for reaching deeper course."
        + (f" Penalty weights: contact {weights['contact_weight']:g}, penetration "
           f"{weights['penetration_weight']:g}."
           if weights.get("contact_weight") is not None else "")
        # Panel (c) is a different measurement from (a) and (b) -- a human judgement, one
        # tile, independent noise draws -- so it needs its own sentence rather than inheriting
        # theirs. In particular its interval is BINOMIAL and that is correct here, which is
        # exactly the opposite of the claim the sentence above makes about (a) and (b).
        + ((" (c) the fraction of trials in which EVERY mode decision of the traverse was the "
            "correct one for the course"
            if args.mode_scope == "all" else
            " (c) the fraction of trials whose FIRST mode decision was the correct one for "
            "the course")
           + ", labelled by a human who was shown the trials blind and interleaved "
           "between arms, with the arm neither displayed nor recoverable. Each cluster is one "
           "tile at one commanded speed, and the arms differ only in the flow-matching noise "
           "draw; error bars are 95% Wilson intervals, which unlike (a) and (b) is the right "
           "interval here because the trials share no tile to cluster on."
           if mode_note else ""))
    paths = plot_style.save_figure(fig, args.out, caption=caption)
    for p in paths:
        print(f"wrote {p}")
    print(f"\npanel (b) = {args.panel_b}: " + ", ".join(
        f"{t} {res[t][field + '_a']:.4f} -> {res[t][field + '_b']:.4f} "
        f"({res[t][field + '_cells_down']}/{res[t]['n_cells']} cells down)" for t in terrains))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
