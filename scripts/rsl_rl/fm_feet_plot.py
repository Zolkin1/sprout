#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""x-z figures of what the generator proposed at one re-plan: feet, base, terrain, depth.

The question these answer is "would any of these draws have put a foot on the tread", which is
two-dimensional -- downrange against height, read against the step profile -- and therefore
invisible both in a 3-D scene (the terrain occludes the very geometry being judged) and in a
scalar score (it averages 62 nodes and every channel).

One panel per CHANNEL -- each foot, then the base -- sharing the terrain profile, carrying:

* every sampled draw over the horizon (thin), the SERVED one highlighted;
* the previous re-plan's draws, faint: how the proposal changed at this solve;
* the pre-made clip over the same nodes, i.e. the target;
* the history the generator conditioned on, marked apart from the older served tail, because
  that handful of nodes is the whole difference between teacher forcing and the closed loop;
* the current step, as a star.

:func:`panels_figure` stacks one ROW per bundle, which is how a teacher-forced run and a
closed-loop run are compared at the same step: same terrain, same axes, different history.

Pure drawing. The caller assembles the world-frame arrays (``fm_feet_data.FeetPlotSource``), so
this module carries no FK, frame or bundle-layout reasoning and can be exercised on synthetic
input.
"""

from __future__ import annotations

import matplotlib

# Only claim the backend if nobody has chosen an interactive one: the keyboard browser selects
# TkAgg before importing this, and forcing Agg here would hand it a window that never draws.
if not matplotlib.get_backend().lower().startswith(("tk", "qt", "gtk", "macosx", "wx", "nbagg")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_C = {
    "draw": "#7A7F8A",        # grey   -- a rejected sample
    "served": "#12E05A",      # green  -- the draw that was served
    "prev": "#3D4450",        # dark   -- the previous solve's samples
    "clip": "#F28C1E",        # orange -- the pre-made clip (the target)
    "hist": "#00D5FF",        # cyan   -- the nodes fed back as history
    "tail": "#5A6472",        # slate  -- older served output, for context
    "terrain": "#4A4A4A",
}


def _draw_terrain(ax, segs, xlim) -> None:
    """Fill the step profile under ``segs`` = ``[(x0, x1, top)]`` in world x/z."""
    floor = ax.get_ylim()[0]
    for x0, x1, top in segs or []:
        if x1 < xlim[0] or x0 > xlim[1]:
            continue
        # Fill only to the axis floor: a fixed depth would force the y range open and squash
        # the few centimetres of clearance this figure exists to show.
        ax.fill_between([x0, x1], [top, top], y2=floor, color=_C["terrain"], alpha=0.35,
                        linewidth=0.0, zorder=0)
        ax.plot([x0, x1], [top, top], color=_C["terrain"], lw=1.4, zorder=1)
        ax.plot([x0, x0], [top - 0.14, top], color=_C["terrain"], lw=1.0, ls=":", zorder=1)


def xz_panel(ax, ch: dict, terrain_segs, *, scores=None, chosen=None, history_source="?",
             legend: bool = False) -> None:
    """Draw one channel's x-z panel.

    Args:
        ax: Target axes.
        ch: One channel from ``FeetPlotSource.inputs`` -- ``name`` plus ``draws [n, F, 3]`` and
            the optional ``prev`` / ``clip`` / ``hist`` / ``tail`` ``[*, 3]`` curves.
        terrain_segs: World ``[(x0, x1, top)]`` step profile.
        scores: ``[n]`` per-draw score, for the served draw's label.
        chosen: Index of the served draw.
        history_source: Named in the history curve's label.
        legend: Collect handles on this panel (the caller makes one shared legend).
    """
    draws = np.asarray(ch["draws"])
    xlim = (float(np.nanmin(draws[:, :, 0])) - 0.25, float(np.nanmax(draws[:, :, 0])) + 0.25)
    zs = [draws[:, :, 2].ravel()]
    for key in ("clip", "hist", "tail"):
        v = ch.get(key)
        if v is not None and np.size(v):
            zs.append(np.asarray(v)[:, 2].ravel())
    tops = [t for x0, x1, t in (terrain_segs or []) if x1 >= xlim[0] and x0 <= xlim[1]]
    allz = np.concatenate([np.asarray(z, float) for z in zs] + [np.asarray(tops or [0.0])])
    ax.set_ylim(float(np.nanmin(allz)) - 0.12, float(np.nanmax(allz)) + 0.10)
    # Equal aspect: this is read as geometry ("does the swing clear that nose"), and an unequal
    # aspect makes a 2 cm miss look like a comfortable margin.
    ax.set_aspect("equal", adjustable="box")
    _draw_terrain(ax, terrain_segs, xlim)

    prev = ch.get("prev")
    if prev is not None:
        for d in range(np.shape(prev)[0]):
            ax.plot(prev[d][:, 0], prev[d][:, 2], color=_C["prev"], lw=0.7, alpha=0.55,
                    zorder=2, label="previous solve's draws" if (legend and d == 0) else None)
    for d in range(draws.shape[0]):
        if chosen is not None and d == int(chosen):
            continue
        ax.plot(draws[d][:, 0], draws[d][:, 2], color=_C["draw"], lw=0.9, alpha=0.85, zorder=3,
                label=f"draws (n={draws.shape[0]})" if (legend and d == 0) else None)
    if chosen is not None and int(chosen) < draws.shape[0]:
        c = int(chosen)
        lbl = f"served draw #{c}" + (f" (score {scores[c]:.3f})" if scores is not None else "")
        ax.plot(draws[c][:, 0], draws[c][:, 2], color=_C["served"], lw=2.6, zorder=6,
                label=lbl if legend else None)
    for key, color, lw, z, name in (
            ("tail", _C["tail"], 1.4, 4, "served so far"),
            ("clip", _C["clip"], 2.4, 5, "pre-made clip")):
        v = ch.get(key)
        if v is not None and np.shape(v)[0] > 1:
            v = np.asarray(v)
            ax.plot(v[:, 0], v[:, 2], color=color, lw=lw, alpha=0.85, zorder=z,
                    label=name if legend else None)
    hist = ch.get("hist")
    if hist is not None and np.shape(hist)[0]:
        hist = np.asarray(hist)
        # The label stays SOURCE-FREE: one shared legend serves every row, and rows can differ
        # in where their history came from (that is the whole point of stacking them). Each
        # row's own y label states its source.
        ax.plot(hist[:, 0], hist[:, 2], color=_C["hist"], lw=2.2, marker="o", ms=4.0, zorder=7,
                label=f"history ({hist.shape[0]} nodes)" if legend else None)
    here = draws[chosen if chosen is not None else 0, 0]
    ax.plot([here[0]], [here[2]], marker="*", ms=14, color="#FFFFFF",
            markeredgecolor="#111111", zorder=9, label="current step" if legend else None)

    ax.set_xlim(*xlim)
    # The channel name goes INSIDE the axes: the band above is spent on the shared legend.
    ax.text(0.015, 0.965, ch["name"], transform=ax.transAxes, va="top", ha="left", fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0})
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def panels_figure(rows: list[dict], *, suptitle: str = "", fig: plt.Figure | None = None
                  ) -> plt.Figure:
    """One row of x-z panels per bundle, with that row's depth images on the right.

    Args:
        rows: One ``FeetPlotSource.inputs`` dict per bundle, plus an optional ``row_label``.
        suptitle: Figure title.
        fig: Reuse this figure (cleared first) instead of making one -- what the interactive
            browser does, so stepping does not leak a figure per frame.

    Returns:
        The figure.
    """
    n_rows = max(len(rows), 1)
    n_ch = max(len(r["channels"]) for r in rows)
    n_depth = max(len(r.get("depth_images") or []) for r in rows)
    if fig is None:
        fig = plt.figure(figsize=(4.6 * n_ch + 2.6 * bool(n_depth), 3.9 * n_rows))
    else:
        fig.clear()
    widths = [1.0] * n_ch + [0.42] * n_depth
    gs = fig.add_gridspec(n_rows, n_ch + n_depth, width_ratios=widths,
                          hspace=0.34, wspace=0.22)
    for i, row in enumerate(rows):
        for j, ch in enumerate(row["channels"]):
            ax = fig.add_subplot(gs[i, j])
            xz_panel(ax, ch, row.get("terrain_segs"), scores=row.get("scores"),
                     chosen=row.get("chosen"), history_source=row.get("history_source", "?"),
                     legend=(i == 0 and j == 0))
            ax.set_xlabel("world x [m]")
            if j == 0:
                # Every row states its own clip AND its own position within that clip: with
                # several bundles stacked they can be on different clip steps (one may have
                # reset), and a label that only named the rollout step would hide that.
                label = row.get("row_label") or f"{row.get('mode', '?')} / " \
                                                f"history={row.get('history_source', '?')}"
                clip = row.get("clip_label")
                ax.set_ylabel(f"{label}\n{clip}\nworld z [m]" if clip
                              else f"{label}\nworld z [m]", fontsize=8)
        for k, img in enumerate(row.get("depth_images") or []):
            ax = fig.add_subplot(gs[i, n_ch + k])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title((row.get("depth_labels") or [f"depth {k}"])[k], fontsize=7)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=min(len(labels), 7), loc="upper center",
               bbox_to_anchor=(0.5, 0.995), framealpha=0.9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.045)
    return fig


def feet_xz_figure(**kw) -> plt.Figure:
    """Single-bundle figure -- the viser button's entry point.

    Accepts one ``FeetPlotSource.inputs`` dict (as keyword arguments) plus ``title_extra``.
    """
    extra = kw.pop("title_extra", "")
    step, dt = kw.get("step", 0), kw.get("dt", 0.02)
    title = (f"{kw.get('clip_label', '')}  |  rollout step {step} (t={step * dt:.2f} s)  |  "
             f"history from {str(kw.get('history_source', '?')).upper()}{extra}")
    return panels_figure([kw], suptitle=title)


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render a figure to an ``(H, W, 3)`` uint8 array for a viser image panel."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
