# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pull training curves from Weights & Biases and render publication-quality plots.

Fetches one or more metrics across one or more runs (by display name or run id)
from the W&B API and plots them with LaTeX fonts / thick lines / a
colorblind-friendly palette, saving PNG + SVG. Multiple ``--metrics`` become a
subplot grid (one panel per metric). Optional wandb-style EMA smoothing.

Two modes:
  * per-run (``--runs`` [+ ``--labels``]): one line per run.
  * grouped/multi-seed (``--group LABEL run1 run2 ...``, repeatable): each group
    is a condition whose runs are seeds; plots the across-seed MEAN line with a
    +-1 std (or SEM) shaded band. This is the multi-seed paper form.

Standalone -- needs only ``wandb``, ``pandas``, ``matplotlib`` (no IsaacLab / sim).

Examples
--------
Multi-seed 2x2 panel, two conditions x N seeds each::

    python scripts/rsl_rl/plot_wandb_rewards.py \
        --group Kinematic run_kin_s0 run_kin_s1 run_kin_s2 run_kin_s3 \
        --group Dynamic   run_dyn_s0 run_dyn_s1 run_dyn_s2 run_dyn_s3 \
        --metrics Train/mean_reward Train/mean_episode_length Metrics/traj_ref/v Loss/symmetry \
        --ylabels "Mean reward" "Episode length" 'CLF value $V$' "Symmetry loss" \
        --smooth 0.9 --out training_panels_seeds
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_paper_style() -> None:
    mpl.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "lines.linewidth": 4, "lines.markersize": 10, "lines.markeredgewidth": 2,
        "axes.linewidth": 1.5, "axes.labelsize": 20, "axes.titlesize": 22,
        "axes.titlepad": 12, "axes.labelpad": 8,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "xtick.major.width": 1.5, "ytick.major.width": 1.5,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.fontsize": 15, "legend.framealpha": 0.9, "legend.edgecolor": "0.4",
        "legend.fancybox": False, "legend.handlelength": 2.0,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })
    try:
        fig_t, ax_t = plt.subplots()
        ax_t.set_title(r"$\alpha$"); fig_t.canvas.draw(); plt.close(fig_t)
    except Exception:
        print("[warn] LaTeX not available -- falling back to mathtext (fonts won't match Computer Modern).")
        mpl.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"})


COLOR_CYCLE = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377", "#BBBBBB", "#000000"]
SINGLE_COL = (3.5, 2.8)
DOUBLE_COL = (7.0, 4.5)
PANEL_SIZE = (4.8, 3.0)  # per-metric panel (w, h) in inches for the multi-panel grid


def _ema(values: np.ndarray, weight: float) -> np.ndarray:
    """wandb-style exponential moving average (weight in [0, 1); 0 = no smoothing)."""
    if weight <= 0.0 or len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    last = float(values[0])
    for i, v in enumerate(values):
        last = last * weight + (1.0 - weight) * float(v)
        out[i] = last
    return out


def _resolve_run(api, entity: str, project: str, ref: str):
    matches = list(api.runs(f"{entity}/{project}", filters={"display_name": ref}))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[warn] {len(matches)} runs named '{ref}'; using the most recent.")
        return sorted(matches, key=lambda r: r.created_at)[-1]
    try:
        return api.run(f"{entity}/{project}/{ref}")
    except Exception as exc:
        raise SystemExit(f"[error] could not resolve run '{ref}' by name or id: {exc}")


def _seed_xy(df: pd.DataFrame, x_key: str, metric: str, smooth: float):
    """Return (x, y) for one run's metric, dropna+sorted, EMA-smoothed. None if absent."""
    if metric not in df.columns:
        return None
    sub = df.dropna(subset=[metric]).sort_values(x_key)
    if sub.empty:
        return None
    return sub[x_key].to_numpy(dtype=float), _ema(sub[metric].to_numpy(dtype=float), smooth)


def _aggregate_seeds(seed_curves, band: str, n_grid: int = 400):
    """Mean (+ band) across seed curves on a common x grid via interpolation.

    Returns (x_grid, mean, lo, hi). For a single seed, lo=hi=mean (no band).
    """
    if len(seed_curves) == 1:
        x, y = seed_curves[0]
        return x, y, y, y
    x_lo = max(c[0].min() for c in seed_curves)
    x_hi = min(c[0].max() for c in seed_curves)
    grid = np.linspace(x_lo, x_hi, n_grid)
    ys = np.vstack([np.interp(grid, x, y) for x, y in seed_curves])  # [S, n_grid]
    mean = ys.mean(axis=0)
    if band == "none":
        return grid, mean, mean, mean
    spread = ys.std(axis=0)
    if band == "sem":
        spread = spread / math.sqrt(ys.shape[0])
    return grid, mean, mean - spread, mean + spread


def main() -> None:
    p = argparse.ArgumentParser(description="Plot W&B training curves (per-run or multi-seed), paper-styled.")
    p.add_argument("--runs", nargs="+", default=None, help="Run names/ids (per-run mode).")
    p.add_argument("--labels", nargs="+", default=None, help="Legend labels for --runs (default = run name).")
    p.add_argument("--group", nargs="+", action="append", default=None, metavar="LABEL RUN...",
                   help="A condition: LABEL followed by its seed runs. Repeat for multiple conditions.")
    p.add_argument("--metrics", nargs="+", default=None, help="Metric keys; each becomes a subplot.")
    p.add_argument("--metric", default=None, help="Single-metric alias for --metrics.")
    p.add_argument("--ylabels", nargs="+", default=None, help="Y-axis label per metric.")
    p.add_argument("--ylabel", default=None, help="Single-metric y-axis label.")
    p.add_argument("--log_metrics", nargs="+", default=None,
                   help="Metric key(s) to render with a log y-axis (e.g. Loss/symmetry); others stay linear.")
    p.add_argument("--x_key", default="_step", help="X-axis metric key (default _step).")
    p.add_argument("--xlabel", default="Iteration")
    p.add_argument("--smooth", type=float, default=0.0, help="EMA weight in [0,1); 0 = off.")
    p.add_argument("--band", choices=["std", "sem", "none"], default="std",
                   help="Multi-seed shaded band: +-1 std (default), SEM, or none.")
    p.add_argument("--ncols", type=int, default=2)
    p.add_argument("--figsize", choices=["single", "double"], default="double", help="Only for a single metric.")
    p.add_argument("--panel_size", nargs=2, type=float, default=list(PANEL_SIZE), metavar=("W", "H"),
                   help="Per-metric panel size (inches) for the multi-panel grid (default 4.8 3.0). Grid figure "
                        "= (W*ncols, H*nrows); a bar chart of matching width + one-panel height is (W*ncols, H).")
    p.add_argument("--exact_size", action="store_true",
                   help="Save at exactly the figure size (bbox_inches=None) instead of tight-cropping, so widths "
                        "match a companion figure pixel-for-pixel when compositing (e.g. bar chart below).")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--entity", default="zolkin-california-institute-of-technology-caltech")
    p.add_argument("--project", default="robot_rl")
    p.add_argument("--out", default="wandb_curve", help="Output basename (writes .png and .svg).")
    p.add_argument("--save_to_runs", action="store_true",
                   help="Save into each involved run's <logs_root>/<experiment_name>/<run>/figures/.")
    p.add_argument("--logs_root", default="logs/rsl_rl")
    p.add_argument("--experiment_name", default="g1")
    args = p.parse_args()

    metrics = list(args.metrics) if args.metrics else ([args.metric] if args.metric else None)
    if not metrics:
        raise SystemExit("[error] provide --metrics or --metric.")
    ylabels = list(args.ylabels) if args.ylabels else ([args.ylabel] if args.ylabel else None)
    if ylabels and len(ylabels) != len(metrics):
        raise SystemExit("[error] --ylabels must have one entry per metric.")

    # Build groups: [(label, [run_refs])]. --group wins; else each --run is a singleton.
    if args.group:
        groups = [(g[0], g[1:]) for g in args.group]
        if any(len(runs) == 0 for _, runs in groups):
            raise SystemExit("[error] each --group needs LABEL followed by >=1 run.")
    elif args.runs:
        labels = args.labels if args.labels else args.runs
        if len(labels) != len(args.runs):
            raise SystemExit("[error] --labels must have one entry per --run.")
        groups = [(lab, [r]) for lab, r in zip(labels, args.runs)]
    else:
        raise SystemExit("[error] provide --group (multi-seed) or --runs.")

    import wandb
    api = wandb.Api(timeout=60)
    _apply_paper_style()

    # Fetch every unique run once (all metrics + x in one history scan).
    all_refs = list(dict.fromkeys(r for _, runs in groups for r in runs))
    dfs, run_names = {}, {}
    for ref in all_refs:
        run = _resolve_run(api, args.entity, args.project, ref)
        rows = list(run.scan_history(keys=[args.x_key, *metrics]))
        dfs[ref] = pd.DataFrame(rows)
        run_names[ref] = run.name

    n = len(metrics)
    panel_w, panel_h = args.panel_size
    if n == 1:
        ncols, nrows = 1, 1
        figsize = SINGLE_COL if args.figsize == "single" else DOUBLE_COL
    else:
        ncols = min(args.ncols, n)
        nrows = math.ceil(n / ncols)
        figsize = (panel_w * ncols, panel_h * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.flat

    for mi, metric in enumerate(metrics):
        ax = flat[mi]
        for gi, (label, runs) in enumerate(groups):
            color = COLOR_CYCLE[gi % len(COLOR_CYCLE)]
            curves = [c for c in (_seed_xy(dfs[r], args.x_key, metric, args.smooth) for r in runs) if c is not None]
            if not curves:
                print(f"[warn] group '{label}' has no data for '{metric}'.")
                continue
            gx, mean, lo, hi = _aggregate_seeds(curves, args.band)
            ax.plot(gx, mean, color=color, label=label)
            if len(curves) > 1 and args.band != "none":
                ax.fill_between(gx, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(ylabels[mi] if ylabels else metric.replace("_", r"\_"))
        if args.log_metrics and metric in set(args.log_metrics):
            ax.set_yscale("log")
        if args.grid:
            ax.grid(True, alpha=0.3, linewidth=0.8)
    for j in range(n, nrows * ncols):
        flat[j].set_visible(False)

    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        if n == 1:
            flat[0].legend(loc="best"); fig.tight_layout()
        else:
            fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                       bbox_to_anchor=(0.5, 1.0), frameon=True)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    base = os.path.basename(args.out)
    dests = []
    if args.save_to_runs:
        for ref in all_refs:
            run_dir = os.path.join(args.logs_root, args.experiment_name, run_names[ref])
            if not os.path.isdir(run_dir):
                print(f"[warn] run dir not found (skipping save-to-run): {run_dir}")
                continue
            fig_dir = os.path.join(run_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            dests.append(os.path.join(fig_dir, base))
        if not dests:
            dests.append(args.out)
    else:
        dests.append(args.out)
    # --exact_size: save the full canvas at exactly `figsize` (override the paper
    # style's tight-crop) so a companion bar chart of the same width lines up.
    save_ctx = mpl.rc_context({"savefig.bbox": "standard"}) if args.exact_size else contextlib.nullcontext()
    with save_ctx:
        for d in dests:
            for ext in ("png", "svg"):
                fig.savefig(f"{d}.{ext}")
    print("[done] wrote:\n  " + "\n  ".join(f"{d}.png / {d}.svg" for d in dests))
    if n > 1:
        print(f"[size] grid = {figsize[0]:.2f} x {figsize[1]:.2f} in ({ncols}x{nrows} panels of "
              f"{panel_w:g} x {panel_h:g}). For a matching bar chart (same width, one panel tall):\n"
              f"       plot_tracking_bars.py --match_grid {ncols} --panel_size {panel_w:g} {panel_h:g}"
              f"{' --exact_size' if args.exact_size else ''}  -> {panel_w*ncols:.2f} x {panel_h:g} in")


if __name__ == "__main__":
    main()
