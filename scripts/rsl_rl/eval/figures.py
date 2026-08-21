# Copyright 2026 Zachary Olkin. All rights reserved.

"""Figures for the eval report.

Each ``fig_*`` builds and RETURNS ``(fig, caption)``; the caller saves, sizes and closes.
That split (borrowed from ``transfer/obelisk/plot_utils/plot_from_bag.py``'s registry, rather
than ``play_plots``' side-effecting ``plot_*(...) -> None``) buys two things: the png/svg/dpi
policy lives in exactly one place, and a figure is testable without touching the filesystem.

Captions are not decoration.  The paper style forbids titles, so the caption sidecar is where
the definitions in force for that figure live -- which success notion, which stratum, which
interval, and how many clusters it came from.  Without it a reader has to guess.
"""

from __future__ import annotations

import numpy as np

import plot_style as ps

#: The two terrain regimes, which is a principled facet rather than an empirical oddity: the
#: five families in ``motion_gen.motionbricks.terrain_families.FAMILIES`` are each DEFINED
#: with an exit segment ("walk in, climb, walk off the top platform"), and those are exactly
#: the ones whose completion metric can say anything. The rest end on the terrain.
RUNOUT_FAMILIES = ("stairs_up", "stairs_down", "jump_on", "jump_off", "jump_off_front")


def _cells(reduced: dict) -> list[dict]:
    return list(reduced["cells"])


def _families(reduced: dict) -> list[str]:
    """Families present, ordered by the baseline cell's success (hardest first)."""
    fams: set[str] = set()
    for c in _cells(reduced):
        fams |= set(c["by_family"])
    base = _cells(reduced)[0]

    def key(f: str) -> tuple:
        e = base["by_family"].get(f, {}).get("success") or {}
        v = e.get("value")
        return (v if v is not None and np.isfinite(v) else 2.0, f)

    return sorted(fams, key=key)


def _val(entry: dict | None, col: str) -> tuple[float, float, float]:
    """``(value, lo_err, hi_err)`` for an error bar, NaN where not reportable."""
    e = (entry or {}).get(col)
    if not isinstance(e, dict) or e.get("value") is None:
        return (np.nan, 0.0, 0.0)
    v = float(e["value"])
    lo, hi = e.get("ci_lo"), e.get("ci_hi")
    if lo is None or not np.isfinite(lo):
        return (v, 0.0, 0.0)
    return (v, max(0.0, v - float(lo)), max(0.0, float(hi) - v))


def _cell_colour(key: str, i: int) -> str:
    """Colour by arm, shade by DR preset, so a cell looks the same in every figure."""
    fam = ps.COLORS["black"] if key.startswith("clip") else ps.COLORS["orange"]
    if "__none__" in key:
        return ps.COLORS["grey"] if key.startswith("clip") else ps.COLORS["cyan"]
    if "__ood__" in key:
        return ps.COLORS["blue"] if key.startswith("clip") else ps.COLORS["red"]
    return fam


def _label(key: str) -> str:
    return ps.tex_escape(key.replace("__", " / ").replace("/ -", ""))


def fig_success_by_family(reduced: dict):
    """Grouped bars: success per family per cell, with across-tile error bars."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cells, fams = _cells(reduced), _families(reduced)
    n = len(cells)
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    width = 0.8 / max(1, n)
    x = np.arange(len(fams))
    for i, c in enumerate(cells):
        vals, lo, hi = zip(*[_val(c["by_family"].get(f), "success") for f in fams])
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
               yerr=[lo, hi], capsize=3, color=_cell_colour(c["key"], i),
               edgecolor="black", linewidth=0.8, label=_label(c["key"]),
               error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([ps.tex_escape(f) for f in fams], rotation=35, ha="right", fontsize=12)
    ax.set_ylabel(r"Success rate")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", ncol=max(1, n // 3))
    caption = (
        "Success rate per terrain family. Bars are the CLIP-AVERAGED rate; error bars are a "
        f"{reduced['ci']:.0%} percentile bootstrap resampling TILES "
        f"(B={reduced['boot']}, seed={reduced['seed']}). Success = no failure termination "
        "fired and the episode was not censored. Families are ordered hardest-first by the "
        "first cell. Tile counts per family are in report.json; a bootstrap over ~10 clusters "
        "is anticonservative."
    )
    return fig, caption


def fig_survival_curves(reduced: dict):
    """Kaplan-Meier survival against episode progress, one line per cell."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    for i, c in enumerate(_cells(reduced)):
        s = c["survival"]
        t, v = np.asarray(s["times"], float), np.asarray(s["survival"], float)
        ax.step(t, v, where="post", color=_cell_colour(c["key"], i), label=_label(c["key"]))
    ax.set_xlabel(r"Episode progress (steps / achievable steps)")
    ax.set_ylabel(r"Surviving fraction")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    caption = (
        "Kaplan-Meier survival. Successful episodes are right-CENSORED, not events -- "
        "averaging a failure-time column over failures only conditions on failing, and "
        "flatters a policy that fails late. The x axis is the fraction of the ACHIEVABLE "
        "episode length (clip length less the reserved observation window), so families whose "
        "clips differ 3.3x in length are comparable. RMST and median survival are in "
        "report.json; a median of NaN means the curve never reached 0.5."
    )
    return fig, caption


def fig_terrain_progress(reduced: dict):
    """ECDF of terrain progress per cell, over families where a pinned span exists."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    drew = False
    for i, c in enumerate(_cells(reduced)):
        vals = []
        for f, entry in c["by_family"].items():
            e = entry.get("terrain_progress")
            if isinstance(e, dict) and e.get("value") is not None:
                vals.append(float(e["value"]))
        if not vals:
            continue
        v = np.sort(np.asarray(vals, float))
        ax.step(np.concatenate([[0.0], v]),
                np.concatenate([[0.0], np.arange(1, v.size + 1) / v.size]),
                where="post", color=_cell_colour(c["key"], i), label=_label(c["key"]))
        drew = True
    ax.set_xlabel(r"Terrain progress (fraction of the reachable pinned span)")
    ax.set_ylabel(r"Cumulative fraction of families")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    if drew:
        ax.legend(loc="best")
    caption = (
        "Distribution of per-family mean terrain progress: how far along the terrain-pinned "
        "span the robot got before stopping. Denominated by the REACHABLE span "
        "min(terrain_end, last usable step), so a successful episode scores exactly 1.0 on "
        "every family -- with the full span as the denominator a success on a family whose "
        "clip ends ON the terrain would score 0.83-0.91 and the success point-mass would "
        "smear. This is the terrain metric that is informative on the families where the "
        "binary completion test degenerates to the success rate."
    )
    return fig, caption


def fig_terrain_completion(reduced: dict):
    """Completion beside plain success, so the gap between them is the message."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cells = _cells(reduced)
    fams = [f for f in _families(reduced)
            if any((c["by_family"].get(f) or {}).get("completion_informative")
                   for c in cells)]
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    if not fams:
        ax.text(0.5, 0.5, "no family has an informative\nterrain-completion metric",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ("Terrain completion is degenerate for every family in this run: the "
                     "pinned span reaches the end of the episode, so completion IS success.")
    n = len(cells)
    width = 0.8 / max(1, n)
    x = np.arange(len(fams))
    for i, c in enumerate(cells):
        comp = [_val(c["by_family"].get(f), "terrain_completed")[0] for f in fams]
        succ = [_val(c["by_family"].get(f), "success")[0] for f in fams]
        pos = x + i * width - 0.4 + width / 2
        ax.bar(pos, succ, width * 0.92, color=_cell_colour(c["key"], i), alpha=0.35,
               edgecolor="none", label=f"{_label(c['key'])} (survived)" if i == 0 else None)
        ax.bar(pos, comp, width * 0.92, color=_cell_colour(c["key"], i),
               edgecolor="black", linewidth=0.8, label=_label(c["key"]))
    ax.set_xticks(x)
    ax.set_xticklabels([ps.tex_escape(f) for f in fams], rotation=35, ha="right", fontsize=12)
    ax.set_ylabel(r"Rate")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=11)
    caption = (
        "Terrain completion (solid) against plain survival (pale), for the families where the "
        "two differ. Completion = survived to min(terrain end + 2 s, last usable step). "
        "Families whose pinned span reaches the end of the episode are omitted, because there "
        "completion equals survival by construction and a bar would read as an independent "
        "result."
    )
    return fig, caption


def fig_velocity_error(reduced: dict):
    """Per-clip commanded-velocity RMSE, linear and yaw in separate panels (units differ)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cells = _cells(reduced)
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    for ax, col, unit in ((axes[0], "vel_rmse_lin", r"m/s"), (axes[1], "vel_rmse_yaw", r"rad/s")):
        pts, labels = [], []
        for c in cells:
            vals = [_val(c["by_family"].get(f), col)[0] for f in c["by_family"]]
            vals = [v for v in vals if np.isfinite(v)]
            pts.append(vals or [np.nan])
            labels.append(_label(c["key"]))
        ax.boxplot(pts, showfliers=False, widths=0.55)
        for i, vals in enumerate(pts, start=1):
            ax.plot(np.full(len(vals), i) + np.random.default_rng(0).normal(0, 0.05, len(vals)),
                    vals, "o", markersize=5, color=ps.COLORS["blue"], alpha=0.7,
                    markeredgewidth=0)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(rf"RMSE ({unit})")
    caption = (
        "Commanded-velocity tracking error. One point per FAMILY mean (the unit of analysis "
        "is the clip, aggregated to families here); linear and yaw are separate panels "
        "because their units differ and a mean across them would be dimensionally "
        "meaningless. Errors are the sqrt of the stored per-episode MSE: MSE is what averages "
        "across episodes, so square-rooting happens once, at the end. Unconditional -- every "
        "completed episode, including the ones that fell."
    )
    return fig, caption


def fig_termination_breakdown(reduced: dict):
    """Stacked termination causes per cell."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cells = _cells(reduced)
    names: list[str] = []
    for c in cells:
        for k in c["terminations"]:
            if not k.startswith("_") and k not in names:
                names.append(k)
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    y = np.arange(len(cells))
    left = np.zeros(len(cells))
    palette = [ps.COLORS[k] for k in ("teal", "red", "blue", "orange", "magenta", "cyan")]
    for j, name in enumerate(names):
        w = np.array([c["terminations"].get(name, 0.0) for c in cells], float)
        ax.barh(y, w, left=left, color=palette[j % len(palette)], edgecolor="black",
                linewidth=0.6, label=ps.tex_escape(name))
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels([_label(c["key"]) for c in cells], fontsize=11)
    ax.set_xlabel(r"Fraction of episodes")
    ax.legend(loc="best", fontsize=11)
    caption = (
        "Which termination term ended each episode. This is a termination-cause breakdown, "
        "NOT a fall-cause breakdown: several terms can fire on the same step (the "
        "multiple-term fraction is in report.json), and the derived fall flag is not a "
        "termination term at all. Bars can therefore sum past 1."
    )
    return fig, caption


def fig_dr_degradation(reduced: dict):
    """Success across the DR presets, one line per arm."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    order = ["none", "train", "ood", "ood_push"]
    series: dict[str, dict[str, float]] = {}
    for c in _cells(reduced):
        cell = c["cell"]
        arm = f"{cell['reference_mode']}/{cell['fm_label']}"
        series.setdefault(arm, {})[cell["dr_preset"]] = _val(c["overall"], "success")[0]
    presets = [p for p in order if any(p in v for v in series.values())]
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    for i, (arm, vals) in enumerate(sorted(series.items())):
        y = [vals.get(p, np.nan) for p in presets]
        ax.plot(range(len(presets)), y, "-o", label=ps.tex_escape(arm),
                color=ps.COLOR_CYCLE[i % len(ps.COLOR_CYCLE)])
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels([ps.tex_escape(p) for p in presets])
    ax.set_xlabel(r"Domain randomization preset")
    ax.set_ylabel(r"Success rate")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    caption = (
        "Success across the randomization presets. ``none`` is the NOISE FLOOR rather than a "
        "determinism check: measured, two envs on the same clip and tile diverge even with "
        "every randomization term nulled, so this point is the resolution below which no "
        "difference between the other cells means anything. ``train`` reproduces the training "
        "distribution term-for-term from the canonical config; ``ood`` widens each range "
        "about its midpoint with the resolved values recorded in the cell manifest."
    )
    return fig, caption


#: ``name -> builder``.  Each returns ``(fig, caption)``.
FIGURES = {
    "success_by_family": fig_success_by_family,
    "survival_curves": fig_survival_curves,
    "terrain_progress": fig_terrain_progress,
    "terrain_completion": fig_terrain_completion,
    "velocity_error": fig_velocity_error,
    "termination_breakdown": fig_termination_breakdown,
    "dr_degradation": fig_dr_degradation,
}

DEFAULT_FIGURES = tuple(FIGURES)


def draw_all(reduced: dict, out_dir, *, which: str = "default", svg: bool = True) -> list[str]:
    """Draw the requested figures. Each is wrapped, so one failure does not lose the rest."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ps.apply_paper_style()
    names = (list(DEFAULT_FIGURES) if which in ("default", "all")
             else [n.strip() for n in which.split(",") if n.strip()])
    written: list[str] = []
    for name in names:
        fn = FIGURES.get(name)
        if fn is None:
            print(f"[figures] unknown figure {name!r}; have {sorted(FIGURES)}")
            continue
        try:
            fig, caption = fn(reduced)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not lose the others
            print(f"[figures] {name} failed: {type(exc).__name__}: {exc}")
            continue
        written += ps.save_figure(fig, f"{out_dir}/{name}", svg=svg, caption=caption)
        plt.close(fig)
    return written
