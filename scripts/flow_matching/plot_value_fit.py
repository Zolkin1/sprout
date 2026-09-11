# Copyright 2026 Zachary Olkin. All rights reserved.

"""Plot a critic fit's learning curves: train BCE and held-out AUC, per variant and fold.

Reads the fit's own stdout, because that is where the per-epoch history actually is --
``fit_value.py`` logs it to W&B and prints it, but writes no history file.  A run piped
through ``tee`` block-buffers, so the on-disk ``fit.log`` can be hours stale while the run is
healthy; W&B's captured ``files/output.log`` is unbuffered and is the reliable source.

What the plot is FOR, and it is not the training loss.  Train BCE falls whether the critic is
learning the task or memorising which episodes failed -- every window of an episode carries
the same label, so the effective sample size is the episode count and the capacity to
memorise is enormous.  The two AUC curves are what separate those:

* **seen tiles** -- held-out EPISODES from tiles the fold trained on.
* **unseen tiles** -- terrain no fold has ever seen.

Their GAP is the measurement.  A high, flat seen-tile curve with a collapsing unseen-tile one
is a critic reading the tile rather than the state.  The marked point is the epoch actually
kept, selected on the seen-tile curve -- never on the unseen-tile one, which is the
generalisation measure and must not also be the selection criterion.

Usage::

    python scripts/flow_matching/plot_value_fit.py \\
        wandb/run-*/files/output.log --out logs/value/v2/fit_curves
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "rsl_rl"))

import plot_style  # noqa: E402

_EPOCH = re.compile(r"\[(?P<tag>[\w/]+)\]\s+epoch\s+(?P<ep>\d+)/(?P<tot>\d+)\s+train BCE\s+"
                    r"(?P<bce>[\d.]+)")
_AUC = re.compile(r"(?P<scope>val_seen_tiles|val_UNSEEN_tiles)\s+AUC\s+(?P<auc>[\d.]+)")
_BEST = re.compile(r"\[(?P<tag>[\w/]+)\]\s+restoring epoch\s+(?P<ep>\d+)\s+\(val AUC\s+"
                   r"(?P<auc>[\d.]+)")


def parse(path: str) -> tuple[dict, dict]:
    """Parse per-epoch curves and the kept epoch out of a fit's stdout.

    Args:
        path: A ``fit_value.py`` stdout capture.

    Returns:
        ``(curves, best)``.  ``curves[tag]`` holds parallel ``epoch`` / ``bce`` /
        ``val_seen_tiles`` / ``val_UNSEEN_tiles`` lists; ``best[tag]`` is ``(epoch, auc)``.
    """
    curves: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: {"epoch": [], "bce": [], "val_seen_tiles": [], "val_UNSEEN_tiles": []})
    best: dict[str, tuple[int, float]] = {}
    tag = None
    with open(path, errors="replace") as f:
        for line in f:
            m = _EPOCH.search(line)
            if m:
                tag = m.group("tag")
                curves[tag]["epoch"].append(int(m.group("ep")))
                curves[tag]["bce"].append(float(m.group("bce")))
                continue
            m = _AUC.search(line)
            if m and tag is not None:
                curves[tag][m.group("scope")].append(float(m.group("auc")))
                continue
            m = _BEST.search(line)
            if m:
                best[m.group("tag")] = (int(m.group("ep")), float(m.group("auc")))
    return dict(curves), best


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="fit_value.py stdout (wandb files/output.log is unbuffered)")
    p.add_argument("--out", required=True, help="output path without extension")
    args = p.parse_args()

    curves, best = parse(args.log)
    if not curves:
        raise SystemExit(f"no epoch lines found in {args.log}")
    plot_style.apply_paper_style()
    import matplotlib.pyplot as plt

    tags = sorted(curves)
    fig, axes = plt.subplots(1, 2, figsize=plot_style.figsize(2, 1, aspect=0.52))
    C = plot_style.COLORS
    colors = [C["blue"], C["cyan"], C["orange"], C["red"], C["teal"], C["magenta"]]

    for i, tag in enumerate(tags):
        c, col = curves[tag], colors[i % len(colors)]
        axes[0].plot(c["epoch"], c["bce"], color=col, label=plot_style.tex_escape(tag))
        n = min(len(c["epoch"]), len(c["val_seen_tiles"]))
        if n:
            axes[1].plot(c["epoch"][:n], c["val_seen_tiles"][:n], color=col)
        m = min(len(c["epoch"]), len(c["val_UNSEEN_tiles"]))
        if m:
            axes[1].plot(c["epoch"][:m], c["val_UNSEEN_tiles"][:m], color=col, ls="--")
        if tag in best:
            ep, auc = best[tag]
            axes[1].plot([ep], [auc], marker="o", color=col, markersize=12,
                         markerfacecolor="none", markeredgewidth=3, zorder=5)

    axes[0].set_xlabel(r"Epoch")
    axes[0].set_ylabel(r"Train BCE")
    axes[1].set_xlabel(r"Epoch")
    axes[1].set_ylabel(r"AUC")
    axes[0].legend(loc="best")
    # Colour identifies the (variant, fold) and is already spelled out on the left panel;
    # repeating it here as eight entries buries the curves it is meant to explain. The right
    # panel only has to say what the two LINE STYLES mean.
    from matplotlib.lines import Line2D  # noqa: PLC0415
    axes[1].legend(handles=[Line2D([], [], color="0.3", ls="-", label=r"seen tiles"),
                            Line2D([], [], color="0.3", ls="--", label=r"unseen tiles")],
                   loc="lower right")
    fig.tight_layout()
    plot_style.save_figure(fig, args.out)
    print(f"[plot] wrote {args.out}.png / .svg")
    for tag in tags:
        c = curves[tag]
        s, u = c["val_seen_tiles"], c["val_UNSEEN_tiles"]
        if s and u:
            n = min(len(s), len(u))
            print(f"  {tag:16s} last seen {s[n-1]:.3f} / unseen {u[n-1]:.3f}  "
                  f"gap {s[n-1]-u[n-1]:+.3f}"
                  + (f"   kept epoch {best[tag][0]}" if tag in best else "  (still running)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
