# Copyright 2026 Zachary Olkin. All rights reserved.

"""The fine-tune ALGORITHM ablation figure: success and contact penalty, one bar per arm.

    python scripts/rsl_rl/plot_algo_ladder.py ladder_numbers.json --out logs/paper/algo_ladder

``plot_finetune_bars.py`` draws the same two quantities and is the wrong shape for this
comparison. It clusters by TERRAIN and pairs pre-trained against fine-tuned, which is right when
the question is "what did the fine-tune do on each course". Here every arm shares ONE parent and
one course, so that layout repeats an identical grey bar four times -- half the ink, and it reads
as though each algorithm had its own baseline. This draws the parent once, as a reference line
with its interval as a band, and gives each algorithm a single bar.

**The two panels do not agree on which direction is good, and the figure says so** rather than
leaving a reader to infer it from the axis label: success is higher-better, contact penalty is
lower-better. That is the whole reason both panels are needed here -- the arm that ties on (a)
is clearly worse on (b), and a success-only figure would report the ablation as a three-way tie
at the top.

Intervals are the reduction's own: a 95% cluster bootstrap over ``(tile, speed)`` cells,
resampling the cell and then the episodes within it. A pooled binomial has 14-70% actual coverage
on this data because episodes sharing a tile share its geometry, so the CELL count -- printed on
the figure -- is the real limit on precision.

Reads ``fig_reduce.py``'s json and restates none of its numbers.
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

#: Panels, as ``(json key stem, axis label, higher_is_better)``.
PANELS: tuple[tuple[str, str, bool], ...] = (
    ("succ", "Success rate  ($\\uparrow$ better)", True),
    ("con", "Mean contact penalty  ($\\downarrow$ better)", False),
)


def _draw(ax, arms: list[str], data: dict, stem: str, ylabel: str, higher: bool,
          shown: list[str]) -> None:
    """Draw one panel: the parent as a reference band, then one bar per arm.

    Args:
        ax: Target axes.
        arms: Arm labels, in plotting order.
        data: The reduction's dict.
        stem: ``succ`` or ``con``.
        ylabel: Y axis label.
        higher: True when larger is better; decides only which bars are coloured as regressions.
        shown: Display names for the ticks, aligned with ``arms``.
    """
    # The parent is identical in every arm's block -- it is one eval directory, reduced against
    # each arm in turn. Asserted rather than assumed: if a future json pairs different parents
    # into one figure, a single reference line would be a quiet lie.
    p = [data[a][f"{stem}_a"] for a in arms]
    assert max(p) - min(p) < 1e-9 or stem != "succ", (
        f"the arms do not share a parent on {stem}: {dict(zip(arms, p))}. This figure draws one "
        "reference line and that is only honest for a shared baseline.")
    par, par_lo, par_hi = p[0], data[arms[0]][f"{stem}_a_lo"], data[arms[0]][f"{stem}_a_hi"]

    xs = np.arange(len(arms), dtype=float)
    ys = np.array([data[a][f"{stem}_b"] for a in arms])
    lo = ys - np.array([data[a][f"{stem}_b_lo"] for a in arms])
    hi = np.array([data[a][f"{stem}_b_hi"] for a in arms]) - ys

    ax.axhspan(par_lo, par_hi, color=plot_style.COLORS["grey"], alpha=0.45, zorder=0, lw=0)
    ax.axhline(par, color="0.35", lw=1.4, ls=(0, (5, 3)), zorder=1,
               label="Pre-trained generator")
    # Colour carries the SIGN of the change, not the arm's identity: the one arm that moves the
    # wrong way should not need the caption to be spotted.
    better = (ys > par) if higher else (ys < par)
    cols = [plot_style.COLORS["blue"] if b else plot_style.COLORS["red"] for b in better]
    ax.bar(xs, ys, 0.62, color=cols, edgecolor="black", linewidth=1.2, zorder=2)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=1.6,
                capsize=4, capthick=1.6, zorder=3)

    ax.set_xticks(xs)
    # ROTATED when they will not fit flat. A double-column figure gives each panel about 3.5
    # inches, which four multi-word ticks overrun -- and matplotlib overlaps them silently
    # rather than shrinking, so "AWR Arrived Filtered Plain" renders as one run-together
    # string. Measured against the panel width rather than guessed at a label length, because
    # the same names fit fine on a single-panel figure.
    width_in = ax.get_figure().get_size_inches()[0] / len(ax.get_figure().axes)
    per_tick_chars = (width_in / len(shown)) / 0.085          # ~0.085 in per character at 10pt
    if max(len(t) for t in shown) > per_tick_chars:
        ax.set_xticklabels(shown, rotation=22, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(shown)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(ys.max() + hi.max(), par_hi) * 1.22)
    ax.yaxis.grid(True, color="0.85", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    """Render the two-panel ablation figure."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("numbers", help="fig_numbers.json from fig_reduce.py")
    ap.add_argument("--out", default="logs/paper/algo_ladder", help="output path, no extension")
    ap.add_argument("--order", default="", help="comma-separated arm order; default json order")
    ap.add_argument("--labels", default="",
                    help="comma-separated display names for --order, '|' for a line break. "
                         "The json keys stay authoritative; only the ticks change.")
    a = ap.parse_args()

    plot_style.apply_paper_style()
    data = json.loads(pathlib.Path(a.numbers).read_text())
    arms = ([s.strip() for s in a.order.split(",") if s.strip()]
            or [k for k in data if not k.startswith("_")])
    missing = [x for x in arms if x not in data]
    if missing:
        raise SystemExit(f"{missing} not in {a.numbers}; it has {[k for k in data]}")

    shown = [s.strip().replace("|", "\n") for s in a.labels.split(",") if s.strip()]
    if shown and len(shown) != len(arms):
        raise SystemExit(f"--labels has {len(shown)} names for {len(arms)} arms")
    fig, axes = plt.subplots(1, 2, figsize=plot_style.figsize(2, 1))
    for ax, (stem, ylabel, higher) in zip(axes, PANELS, strict=True):
        _draw(ax, arms, data, stem, ylabel, higher, shown or arms)
    axes[0].legend(loc="upper right", frameon=True, fontsize="small")

    d0 = data[arms[0]]
    fig.text(0.5, 0.005,
             f"{d0['n_cells']} (tile, speed) cells / {d0['n_tiles']} tiles, "
             f"{d0['n_episodes_b']} episodes per arm",
             ha="center", va="bottom", fontsize="small", color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    caption = (
        "Fine-tune algorithm ablation on the held-out box course, all arms from one "
        "pre-trained generator. (a) success rate, higher is better; (b) mean contact penalty, "
        "lower is better. Bars are the fine-tuned arm; the dashed line and grey band are the "
        "shared pre-trained generator and its interval. Error bars are 95% cluster bootstrap "
        "intervals over (tile, speed) cells, resampling the cell and then the episodes within "
        "it -- a pooled binomial interval has 14-70% actual coverage here, because episodes "
        "sharing a tile share its geometry. Success is scored over each environment's first "
        "episode, the design whose membership cannot depend on episode duration. The contact "
        "penalty is exposure-standardised over downrange position, so a longer-surviving arm "
        "is not charged for reaching deeper course. Red marks an arm that moved the wrong way.")
    for f in plot_style.save_figure(fig, a.out, caption=caption):
        print(f"wrote {f}")
    for arm in arms:
        d = data[arm]
        print(f"  {arm:12s} success {d['succ_a']:.4f} -> {d['succ_b']:.4f}   "
              f"contact {d['con_a']:.4f} -> {d['con_b']:.4f} "
              f"({d['con_cells_down']}/{d['n_cells']} cells down)")


if __name__ == "__main__":
    main()
