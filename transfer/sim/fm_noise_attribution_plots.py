# Copyright 2026 Zachary Olkin. All rights reserved.

"""The four plots and the per-pair table for :mod:`fm_noise_attribution`.

Split out for the same reason the eval harness splits its reducer: this half imports no
MuJoCo and no onnxruntime, so a result can be re-plotted anywhere the json landed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "rsl_rl"))

import plot_style  # noqa: E402


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct at 0 and 1, where the normal interval is not.

    The single-substitution rates live at the extremes by design (most steps should be
    harmless), and a normal interval there returns a width of zero, which would read as
    certainty rather than as no data.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def stage_plot(args) -> int:
    """Write the four figures and the per-pair table."""
    from fm_noise_attribution import flip_points  # noqa: PLC0415

    with open(os.path.join(args.out, "attribution.json")) as f:
        state = json.load(f)
    if "prefix" not in state:
        raise SystemExit("no prefix results in attribution.json -- run --stage prefix first.")
    plot_style.apply_paper_style()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    C = plot_style.COLORS
    pre = state["prefix"]
    flips = flip_points(pre)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Plot 1: outcome vs k, one line per pair, both directions ------------------------
    # Wider than the shared DOUBLE_COL: the x label is long and at the default width the two
    # panels' labels collided and ran off the page.
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))
    for ax, direction, title in (
            (axes[0], "prefix_B_then_A", r"B's first $k$ draws, then A's"),
            (axes[1], "prefix_A_then_B", r"A's first $k$ draws, then B's")):
        for j, pid in enumerate(sorted({r["pair_id"] for r in pre})):
            sel = sorted([r for r in pre if r["pair_id"] == pid
                          and r["direction"] == direction], key=lambda r: r["k"])
            if not sel:
                continue
            ks = [r["k"] for r in sel]
            fell = [0 if r["verdict"] == "success" else 1 for r in sel]
            # Jittered so overlapping step functions stay countable. Kept small enough that a
            # jittered "success" never reaches the "fail" level.
            ax.plot(ks, np.array(fell) * 0.8 + 0.1 + 0.012 * j, drawstyle="steps-post",
                    color=plot_style.COLOR_CYCLE[j % len(plot_style.COLOR_CYCLE)],
                    linewidth=2, alpha=0.85)
            fp = flips.get(f"{pid}:{direction}", {})
            if fp.get("flip_k") is not None:
                ax.axvline(fp["flip_k"], color="0.8", linewidth=1, zorder=0)
        # Stated per panel because the two ENDPOINTS differ, and that is the whole reading of
        # the figure: on the left k=0 is A (success) and k=len(B) is B; on the right k=0 is B
        # (failure) and k=len(A) is A. The ranges differ too -- B's episodes are 9-30 chunks
        # where A's are 45-49 -- so the panels are deliberately not on a shared x scale.
        ax.set_xlabel(r"$k$ (chunks taken from the first sequence)")
        ax.set_title(title, fontsize=16)
        ax.set_yticks([0.1, 0.9])
        ax.set_yticklabels([r"success", r"fail"])
        ax.set_ylim(-0.05, 1.15)
    axes[0].set_ylabel(r"outcome")
    axes[0].text(0.02, -0.20, r"$k=0$: pure A $\rightarrow$ must succeed", fontsize=13,
                 transform=axes[0].transAxes, color="0.35")
    axes[1].text(0.02, -0.20, r"$k=0$: pure B $\rightarrow$ must fail", fontsize=13,
                 transform=axes[1].transAxes, color="0.35")
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(fig_dir, "plot1_outcome_vs_k"))
    plt.close(fig)

    # ---- Plot 2: commitment point distribution -------------------------------------------
    fwd = [v for k, v in flips.items() if v["direction"] == "prefix_B_then_A"
           and v["flip_k"] is not None]
    # ONLY the non-oscillating pairs have a commitment point. Histogramming an oscillating
    # pair's first-sustained-flip would report a commitment for a rollout whose outcome
    # depends on a CONJUNCTION of draws, where by 8.4 no such point exists.
    clean = [v for v in fwd if not v["oscillates"]]
    # Wide, for the same reason as plot 1: at DOUBLE_COL these two x labels overprint.
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    if clean:
        ks = [v["flip_k"] for v in clean]
        gaps = [v["termination_chunk_B"] - v["flip_k"] for v in clean]
        # 8.1's headline on the right: how long BEFORE the visible failure the outcome was
        # settled. BOTH terms in CHUNKS -- dividing a control-step count by the sweep length
        # is what made the first version of this read as A's duration minus a chunk index.
        for ax, data, colour, xlabel in (
                (axes[0], ks, C["blue"], r"commitment chunk index"),
                (axes[1], gaps, C["orange"], r"chunks: commitment $\rightarrow$ termination")):
            lo, hi = min(data), max(data)
            ax.hist(data, bins=np.arange(lo - 0.5, hi + 1.5, 1.0),
                    color=colour, edgecolor="black")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"pairs")
            # Counts are integers and there are only a handful, so a decimal axis invents a
            # precision the sample does not have.
            ax.set_yticks(range(0, max(3, len(clean)) + 1))
            ax.set_xticks(range(int(lo), int(hi) + 1, max(1, (int(hi) - int(lo)) // 8 + 1)))
    else:
        for a in axes:
            a.text(0.5, 0.5, "no pair has a\ncommitment point", ha="center", va="center",
                   transform=a.transAxes, fontsize=16)
    fig.suptitle(rf"only {len(clean)} of {len({v['pair_id'] for v in fwd})} pairs have a "
                 r"commitment point at all; the rest oscillate ($n$ is small, read with care)",
                 fontsize=15)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(fig_dir, "plot2_commitment_distribution"))
    plt.close(fig)

    # ---- Plot 3: single-substitution failure rate vs i ------------------------------------
    if state.get("single"):
        sing = state["single"]
        fig, ax = plt.subplots(figsize=plot_style.figsize(2, 1, aspect=0.5))
        idxs = sorted({r["i"] for r in sing})
        rate, lo, hi = [], [], []
        for i in idxs:
            sel = [r for r in sing if r["i"] == i]
            k = sum(1 for r in sel if r["verdict"] != "success")
            rate.append(k / len(sel))
            a, b = _wilson(k, len(sel))
            lo.append(a)
            hi.append(b)
        ax.plot(idxs, rate, marker="o", color=C["red"])
        ax.fill_between(idxs, lo, hi, color=C["red"], alpha=0.2)
        ax.set_xlabel(r"substituted chunk index $i$")
        ax.set_ylabel(r"failure rate (fresh draws)")
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        plot_style.save_figure(fig, os.path.join(fig_dir, "plot3_single_sub_rate"))
        plt.close(fig)

    # ---- Plot 4: divergence curves, grouped by eventual outcome ---------------------------
    fig, ax = plt.subplots(figsize=plot_style.figsize(2, 1, aspect=0.5))
    stride = int(state.get("div_stride", 5)) or 5
    for r in pre:
        c = r.get("div_curve") or []
        if not c:
            continue
        x = np.arange(len(c)) * stride
        ax.plot(x, c, linewidth=1, alpha=0.35,
                color=(C["red"] if r["verdict"] != "success" else C["teal"]))
    from matplotlib.lines import Line2D  # noqa: PLC0415
    ax.legend(handles=[Line2D([], [], color=C["teal"], label=r"ended in success"),
                       Line2D([], [], color=C["red"], label=r"ended in failure")],
              loc="upper left")
    ax.set_xlabel(r"control step")
    ax.set_ylabel(r"$\|$base $-$ A's base$\|$ (m)")
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(fig_dir, "plot4_divergence"))
    plt.close(fig)

    # ---- Table ---------------------------------------------------------------------------
    print(f"\n{'pair':>5} {'A':>4} {'B':>4} {'flip(B->A)':>11} {'flip(A->B)':>11} "
          f"{'termB':>6} {'gap':>6}  osc")
    n_osc = 0
    for pid in sorted({r["pair_id"] for r in pre}):
        f1 = flips.get(f"{pid}:prefix_B_then_A", {})
        f2 = flips.get(f"{pid}:prefix_A_then_B", {})
        osc = bool(f1.get("oscillates") or f2.get("oscillates"))
        n_osc += int(osc)
        k1, k2 = f1.get("flip_k"), f2.get("flip_k")
        term = f1.get("termination_chunk_B", 0)
        # Both in CHUNKS: the commitment index and B's own termination. Mixing the control-step
        # count with a chunk index here is what made the first printing of this column wrong.
        gap = "" if k1 is None else f"{term - k1:6.1f}"
        seed_a = next((r["seed_A"] for r in pre if r["pair_id"] == pid), -1)
        seed_b = next((r["seed_B"] for r in pre if r["pair_id"] == pid), -1)
        print(f"{pid:>5} {seed_a:>4} {seed_b:>4} {str(k1):>11} {str(k2):>11} "
              f"{term:>6} {gap:>6}  {'YES' if osc else ''}")
    n_pairs = len({r["pair_id"] for r in pre})
    print(f"\noscillating pairs: {n_osc}/{n_pairs}")
    if n_osc:
        print("  §8.4: for those, failure depends on a CONJUNCTION of draws rather than on a "
              "prefix -- no single commitment point exists and chunk-level advantage is "
              "ill-defined for them.")
    # ONLY the non-oscillating pairs have a commitment point to summarise. Averaging over an
    # oscillating pair would report a "commitment chunk" for a rollout whose outcome depends on
    # a conjunction of draws, where by §8.4 no such point exists.
    clean = [v for v in fwd if not v["oscillates"]]
    if clean:
        ks = [v["flip_k"] for v in clean]
        gaps = [v["termination_chunk_B"] - v["flip_k"] for v in clean]
        print(f"commitment chunk (NON-oscillating pairs only): median {np.median(ks):.1f}  "
              f"range {min(ks)}-{max(ks)}  over {len(clean)}/{n_pairs} pairs")
        print(f"commitment-to-manifestation gap: median {np.median(gaps):.1f} chunks "
              f"({np.median(gaps) * 0.24:.2f} s)")
    else:
        print(f"NO pair has a commitment point: all {n_pairs} oscillate. Chunk-level advantage "
              "is ill-defined on this tile -- see §8.4.")
    print(f"\n[attr] figures -> {fig_dir}")
    return 0
