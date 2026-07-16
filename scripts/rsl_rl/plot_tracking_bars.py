# Copyright 2026 Zachary Olkin. All rights reserved.

"""Publication-quality grouped bar chart of a per-frame comparison metric.

Reads the durable ``comparison_stats.csv`` written by ``eval_compare_policies.py``
(in each run's ``policy_comparison/``) and renders a paper-styled grouped bar
chart -- one cluster per body frame, one bar per policy -- with LaTeX fonts, the
colorblind palette, and clean labels. Standalone (``pandas`` + ``matplotlib``).

The CSV holds long-format rows ``policy,velocity,metric,value``. By default this
plots ``frame_track_err:<frame>`` (mean position tracking error, metres) at one
velocity; use ``--metric_prefix`` for another family (e.g. ``pos_mse_group``).

Examples
--------
Pelvis/ankle tracking error, two runs, clean labels, saved into the run folder::

    python scripts/rsl_rl/plot_tracking_bars.py \
        --run 2026-07-14_12-02-37_running_dynamic \
        --labels "Kinematic" "Dynamic" \
        --ylabel "Position tracking error (m)" \
        --save_to_runs --out tracking_error_bars
"""

from __future__ import annotations

import argparse
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
        "lines.linewidth": 4, "axes.linewidth": 1.5,
        "axes.labelsize": 20, "axes.titlesize": 22, "axes.labelpad": 8,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "xtick.major.width": 1.5, "ytick.major.width": 1.5,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.fontsize": 15, "legend.framealpha": 0.9, "legend.edgecolor": "0.4",
        "legend.fancybox": False, "legend.handlelength": 1.6,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })
    try:
        fig_t, ax_t = plt.subplots()
        ax_t.set_title(r"$\alpha$"); fig_t.canvas.draw(); plt.close(fig_t)
    except Exception:
        print("[warn] LaTeX not available -- falling back to mathtext.")
        mpl.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"})


COLOR_CYCLE = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377", "#BBBBBB", "#000000"]


def _pretty(name: str) -> str:
    """Clean a frame/category key for a label: 'left_ankle_roll_link' -> 'Left ankle'."""
    s = name.replace("_link", "").replace("_roll", "").replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-styled grouped bar of a per-frame comparison metric.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Run name; reads its policy_comparison/comparison_stats.csv.")
    src.add_argument("--stats_csv", help="Explicit path to a comparison_stats.csv.")
    p.add_argument("--metric_prefix", default="frame_track_err",
                   help="Plot all '<prefix>:<category>' rows (default frame_track_err).")
    p.add_argument("--velocity", type=float, default=None, help="Which vx (default: first in the CSV).")
    p.add_argument("--labels", nargs="+", default=None, help="Clean policy labels, in CSV policy order.")
    p.add_argument("--group", nargs="+", action="append", default=None, metavar="LABEL RUN...",
                   help="Multi-seed: LABEL followed by its seed run names (must match CSV policies). "
                        "Repeat per condition. Bars show the across-seed mean; error bars = +-1 std "
                        "ACROSS SEEDS (overrides the within-run std). Requires one eval over all seeds.")
    p.add_argument("--ylabel", default="Position tracking error (m)")
    p.add_argument("--title", default=None, help="Optional title (off by default; papers use captions).")
    p.add_argument("--no_error_bars", action="store_true",
                   help="Disable +-1 std error bars (drawn by default when '<prefix>_std:*' rows exist).")
    p.add_argument("--figsize", nargs=2, type=float, default=[7.0, 4.5])
    p.add_argument("--out", default="tracking_bars", help="Output basename (writes .png and .svg).")
    p.add_argument("--save_to_runs", action="store_true",
                   help="Save into each involved run's figures/ folder instead of the CWD.")
    p.add_argument("--logs_root", default="logs/rsl_rl")
    p.add_argument("--experiment_name", default="g1")
    args = p.parse_args()

    csv_path = args.stats_csv or os.path.join(
        args.logs_root, args.experiment_name, args.run, "policy_comparison", "comparison_stats.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"[error] comparison_stats.csv not found: {csv_path}")
    df_all = pd.read_csv(csv_path)
    vel = args.velocity if args.velocity is not None else float(df_all["velocity"].iloc[0])
    df_all = df_all[np.isclose(df_all["velocity"], vel)].copy()

    mean_df = df_all[df_all["metric"].str.startswith(f"{args.metric_prefix}:")].copy()
    if mean_df.empty:
        raise SystemExit(f"[error] no '{args.metric_prefix}:*' rows at vx={vel} in {csv_path}")
    mean_df["category"] = mean_df["metric"].str.split(":", n=1).str[1]

    policies = list(dict.fromkeys(mean_df["policy"]))     # preserve CSV order
    categories = list(dict.fromkeys(mean_df["category"]))
    if args.labels and len(args.labels) != len(policies):
        raise SystemExit(f"[error] --labels needs {len(policies)} entries (policies: {policies}).")
    disp = args.labels if args.labels else [_pretty(p_) for p_ in policies]

    mpivot = mean_df.pivot_table(index="policy", columns="category", values="value")

    if args.group:
        # Multi-seed: one series per condition; value = across-seed mean, error =
        # across-seed std (the paper-standard N-seed error bar).
        groups = [(g[0], g[1:]) for g in args.group]
        series_labels = [lab for lab, _ in groups]
        missing = [r for _, runs in groups for r in runs if r not in mpivot.index]
        if missing:
            raise SystemExit(f"[error] group run(s) not in the CSV policies {list(mpivot.index)}: {missing}")

        def _reduce(fn):
            return np.array([[fn([mpivot.loc[r, cat] for r in runs]) for cat in categories]
                             for _, runs in groups])
        values = _reduce(np.nanmean)
        errors = None if args.no_error_bars else _reduce(np.nanstd)
    else:
        series_labels = disp
        values = np.array([[mpivot.loc[pol, cat] for cat in categories] for pol in policies])
        # +-1 std error bars from the '<prefix>_std:*' rows (within-run std), when present.
        errors = None
        if not args.no_error_bars:
            std_df = df_all[df_all["metric"].str.startswith(f"{args.metric_prefix}_std:")].copy()
            if not std_df.empty:
                std_df["category"] = std_df["metric"].str.split(":", n=1).str[1]
                spivot = std_df.pivot_table(index="policy", columns="category", values="value")
                errors = np.array([[spivot.loc[pol, cat] if (pol in spivot.index and cat in spivot.columns)
                                    else np.nan for cat in categories] for pol in policies])
            else:
                print(f"[warn] no '{args.metric_prefix}_std:*' rows found -- no error bars "
                      "(re-run eval_compare_policies to regenerate the CSV with stds).")

    _apply_paper_style()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    x = np.arange(len(categories))
    width = 0.8 / max(len(series_labels), 1)
    for i, lab in enumerate(series_labels):
        offset = (i - (len(series_labels) - 1) / 2.0) * width
        yerr = errors[i] if errors is not None else None
        ax.bar(x + offset, values[i], width, yerr=yerr, capsize=5,
               error_kw={"elinewidth": 1.5, "capthick": 1.5},
               color=COLOR_CYCLE[i % len(COLOR_CYCLE)], edgecolor="black", linewidth=1.0, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty(c) for c in categories])
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title)
    ax.legend(loc="best")
    fig.tight_layout()

    base = os.path.basename(args.out)
    dests = []
    if args.save_to_runs:
        for pol in policies:
            d = os.path.join(args.logs_root, args.experiment_name, pol, "figures")
            if not os.path.isdir(os.path.dirname(d)):
                print(f"[warn] run dir not found, skipping: {os.path.dirname(d)}")
                continue
            os.makedirs(d, exist_ok=True)
            dests.append(os.path.join(d, base))
        if not dests:
            dests.append(args.out)
    else:
        dests.append(args.out)
    for d in dests:
        for ext in ("png", "svg"):
            fig.savefig(f"{d}.{ext}")
    print("[done] wrote:\n  " + "\n  ".join(f"{d}.png / {d}.svg" for d in dests))


if __name__ == "__main__":
    main()
