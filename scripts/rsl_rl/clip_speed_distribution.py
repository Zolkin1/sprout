#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Commanded-speed distribution per terrain family in a clip library.

Answers "how fast does each family actually ask the robot to go", which is not derivable
from the clip count: the families differ in both clip length and standing fraction, so
30% of the clips can be 52% of the time.

Two things this has to get right, both of which silently skew the numbers otherwise:

* **Padding.** Every clip is stored at ``num_steps`` rows but only ``clip_len`` are real,
  and the tail REPEATS the final command. A 201-step clip therefore carries 799 padded
  rows, so an unmasked histogram counts its last velocity ~4x more than the rest of the
  clip put together. Everything here is masked to ``clip_len``.
* **Which label is the family.** ``skill_id`` is degenerate in the motionbricks libraries --
  every clip is 0 under a single mislabelled ``skill_list = ["stairs_up"]`` -- so the real
  family axis is ``terrain_type_code``, indexing the ``terrain_type_names`` attribute.

Companion to ``command_distribution_stats.py``, which reports skill *occupancy* and
steady/ramping structure but not the speed distribution. numpy + h5py only; no IsaacLab.

Usage:
    python scripts/rsl_rl/clip_speed_distribution.py --clip clips/g1_mb_v5.h5
    python scripts/rsl_rl/clip_speed_distribution.py --clip clips/g1_mb_v5.h5 --channel wz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- plot style
# The SHARED style, applied by the caller -- never at import. This module used to write rcParams at
# module scope, which is exactly what ``plot_style``'s own docstring says poisons every other plot in
# the process: importing this file for its ``collect()`` alone silently restyled anything else the
# process later drew. It also carried a private PALETTE, the only copy in the repo.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import COLOR_CYCLE as PALETTE  # noqa: E402
from plot_style import apply_paper_style, tex_escape  # noqa: E402

CHANNELS: dict[str, int] = {"vx": 0, "vy": 1, "wz": 2}
# Coarse buckets for the grouped bar chart. Standing is split out rather than folded into
# the first moving bucket: it is ~28% of the flat family and would otherwise dominate it.
BUCKETS: list[tuple[str, float, float]] = [
    (r"backward", -np.inf, -0.1), (r"stand", -0.1, 0.1), (r"0.1--0.5", 0.1, 0.5),
    (r"0.5--1", 0.5, 1.0), (r"1--1.5", 1.0, 1.5), (r"1.5--2", 1.5, 2.0),
    (r"2--2.5", 2.0, 2.5), (r"2.5--3", 2.5, 3.0), (r"$\geq$3", 3.0, np.inf),
]


def collect(path: str, channel: int, edges: np.ndarray, max_levels: int = 256) -> dict:
    """Accumulate per-family histograms, bucket shares and percentiles from a clip library.

    Reads in blocks and histograms as it goes, so a multi-GB ``target_vel`` never has to be
    resident. Padded rows are dropped per clip via ``clip_len``.

    Args:
        path: Clip library ``.h5``.
        channel: Column of ``target_vel`` to profile (see :data:`CHANNELS`).
        edges: Histogram bin edges, shared across families so they can be overlaid.

    Returns:
        Mapping with ``names``, per-family ``hist`` / ``bucket`` counts, ``steps``,
        ``clips``, thinned ``pooled`` samples for percentiles, and the library ``dt``.
    """
    with h5py.File(path, "r") as f:
        names: list[str] = json.loads(f.attrs["terrain_type_names"])
        code = np.asarray(f["terrain_type_code"])
        clen = np.asarray(f["clip_len"]).astype(int)
        dt = float(f.attrs["dt"])
        tv = f["target_vel"]
        n_clips, n_steps = tv.shape[0], tv.shape[1]
        nf = len(names)
        hist = np.zeros((nf, len(edges) - 1))
        bucket = np.zeros((nf, len(BUCKETS)))
        steps = np.zeros(nf)
        clips = np.zeros(nf, dtype=int)
        pooled: list[list[np.ndarray]] = [[] for _ in range(nf)]
        levels: list[dict[float, int] | None] = [{} for _ in range(nf)]
        step_idx = np.arange(n_steps)
        for start in range(0, n_clips, 250):
            blk = np.asarray(tv[start:start + 250])[:, :, channel]      # [b, T]
            cc, ll = code[start:start + 250], clen[start:start + 250]
            valid = step_idx[None, :] < ll[:, None]
            for i in range(blk.shape[0]):
                fam = int(cc[i])
                v = blk[i][valid[i]]
                clips[fam] += 1
                steps[fam] += v.size
                hist[fam] += np.histogram(v, bins=edges)[0]
                for b, (_lab, lo, hi) in enumerate(BUCKETS):
                    bucket[fam, b] += np.count_nonzero((v >= lo) & (v < hi))
                pooled[fam].append(v[::17])                              # thinned percentiles
                # Track the discrete command levels while a family still looks quantized.
                # Some families are a sweep over a handful of constant speeds (terrain and
                # transition are 50 levels, 0.6..2.6, one per clip), and binning those into a
                # fixed grid ALIASES: at 0.0408 spacing under a 0.05 bin each bin swallows 1
                # or 2 levels, and the alternation shows up as a comb that looks like
                # structure in the data but is an artefact of the plot. Detected here so the
                # renderer can draw them as levels instead of guessing a bin width.
                if levels[fam] is not None:
                    for val, cnt in zip(*np.unique(v, return_counts=True)):
                        levels[fam][round(float(val), 6)] = \
                            levels[fam].get(round(float(val), 6), 0) + int(cnt)
                    if len(levels[fam]) > max_levels:
                        levels[fam] = None                               # continuous
    return {
        "names": names, "hist": hist, "bucket": bucket, "steps": steps, "clips": clips,
        "dt": dt, "levels": levels,
        "pooled": [np.concatenate(p) if p else np.zeros(0) for p in pooled],
    }


def report(d: dict, channel_name: str) -> list[int]:
    """Print the per-family summary table and return the indices of populated families."""
    live = [i for i in range(len(d["names"])) if d["clips"][i] > 0]
    empty = [d["names"][i] for i in range(len(d["names"])) if d["clips"][i] == 0]
    print(f"padding EXCLUDED via clip_len; channel = {channel_name}\n")
    print(f"{'family':<12s}{'clips':>7s}{'steps':>10s}{'time(s)':>9s}{'share':>7s}"
          f"{'p05':>7s}{'p50':>7s}{'p95':>7s}{'max':>7s}")
    print("-" * 73)
    total_t = d["steps"].sum() * d["dt"]
    for i in live:
        v = d["pooled"][i]
        t = d["steps"][i] * d["dt"]
        print(f"{d['names'][i]:<12s}{d['clips'][i]:>7d}{int(d['steps'][i]):>10d}{t:>9.0f}"
              f"{100 * t / total_t:>6.1f}%{np.percentile(v, 5):>7.2f}"
              f"{np.percentile(v, 50):>7.2f}{np.percentile(v, 95):>7.2f}{v.max():>7.2f}")
    if empty:
        print(f"\nfamilies with no clips: {', '.join(empty)}")
    return live


def plot(d: dict, live: list[int], channel_name: str, out: str) -> None:
    """Write the grouped bucket bar chart plus one filled histogram per family.

    The families are drawn as SMALL MULTIPLES rather than overlaid: ``terrain`` and
    ``transition`` have near-identical distributions and one hides the other completely
    when superimposed.
    """
    names, bucket, hist = d["names"], d["bucket"], d["hist"]
    edges = d["edges"]
    ctr = 0.5 * (edges[1:] + edges[:-1])
    nf = len(live)

    fig = plt.figure(figsize=(13.0, 4.6 + 2.5 * nf))
    gs = fig.add_gridspec(1 + nf, 1, height_ratios=[2.6] + [1.0] * nf, hspace=0.35)

    ax = fig.add_subplot(gs[0])                                   # ---- grouped bar chart
    width = 0.8 / nf
    x = np.arange(len(BUCKETS))
    for n, fam in enumerate(live):
        share = 100 * bucket[fam] / max(bucket[fam].sum(), 1)
        ax.bar(x + (n - (nf - 1) / 2) * width, share, width * 0.92,
               color=PALETTE[n % len(PALETTE)], edgecolor="0.25", linewidth=0.8,
               label=rf"\texttt{{{names[fam]}}}")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in BUCKETS])
    ax.set_ylabel(r"\% of that family's time")
    ax.set_xlabel(rf"Commanded ${channel_name}$ bucket (m/s)")
    # Above the axes, not inside it: the tallest bar reaches ~30% and an in-axes legend
    # sits right on top of it.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=nf, frameon=False)
    ax.set_ylim(0, 1.08 * max(1e-9, (100 * bucket[live] /
                                     np.maximum(bucket[live].sum(1, keepdims=True), 1)).max()))
    ax.grid(True, alpha=0.3, linewidth=0.8, axis="y")

    bw = float(edges[1] - edges[0])
    for n, fam in enumerate(live):                                 # ---- small multiples
        axh = fig.add_subplot(gs[1 + n])
        # Percent of THAT family's steps per bin -- probability mass, not a density (not
        # divided by bin width), so each panel sums to 100%. Percent rather than fraction
        # so the units match the bar chart above.
        lv = d["levels"][fam]
        if lv is not None and len(lv) > 0:
            # Quantized family: one stem per ACTUAL command level. Binning these would alias
            # the level grid against the bin grid and manufacture a comb (see collect()).
            xs = np.array(sorted(lv)); ws = np.array([lv[x] for x in xs], dtype=float)
            ws = 100.0 * ws / ws.sum()
            axh.vlines(xs, 0, ws, color=PALETTE[n % len(PALETTE)], lw=2.0)
            axh.plot(xs, ws, "o", color=PALETTE[n % len(PALETTE)], ms=4)
            axh.set_ylabel(r"\% per level")
            axh.set_ylim(0, max(1.35 * ws.max(), 1e-9))
            tag = rf"\texttt{{{names[fam]}}} ({len(xs)} discrete levels)"
        else:
            h = 100.0 * hist[fam] / max(hist[fam].sum(), 1)
            axh.fill_between(ctr, h, step="mid", color=PALETTE[n % len(PALETTE)], alpha=0.75)
            axh.step(ctr, h, where="mid", color=PALETTE[n % len(PALETTE)], lw=2.5)
            axh.set_ylabel(r"\% per bin")
            tag = rf"\texttt{{{names[fam]}}}"
        v = d["pooled"][fam]
        axh.axvline(np.percentile(v, 50), color="k", lw=2, ls="--")
        axh.annotate(tag, xy=(0.015, 0.86), xycoords="axes fraction",
                     fontsize=17, ha="left", va="top")
        axh.grid(True, alpha=0.3, linewidth=0.8)
        axh.set_xlim(edges[0], edges[-1])
        if n == nf - 1:
            axh.set_xlabel(rf"Commanded ${channel_name}$ (m/s) in {bw:.2f} m/s bins; "
                           r"dashed $=$ median")
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}")
    print(f"\nwrote {out}.png / .svg")


def main() -> int:
    """Profile a clip library's commanded speeds per terrain family."""
    apply_paper_style()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Clip library .h5 (no default on purpose).")
    p.add_argument("--out", default="logs/clip_speed_distribution",
                   help="Output path stem; .png and .svg are written.")
    p.add_argument("--channel", choices=sorted(CHANNELS), default="vx",
                   help="Which target_vel channel to profile.")
    p.add_argument("--lo", type=float, default=-1.5, help="Histogram lower edge.")
    p.add_argument("--hi", type=float, default=4.0, help="Histogram upper edge.")
    p.add_argument("--bins", type=int, default=110, help="Histogram bin count.")
    args = p.parse_args()

    edges = np.linspace(args.lo, args.hi, args.bins + 1)
    d = collect(args.clip, CHANNELS[args.channel], edges)
    d["edges"] = edges
    print(f"{args.clip}")
    live = report(d, args.channel)
    if not live:
        print("no populated families")
        return 1
    plot(d, live, args.channel, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
