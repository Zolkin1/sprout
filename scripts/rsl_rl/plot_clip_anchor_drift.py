#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Diagnose lateral / heading drift of terrain-aware clips (straight-walk check).

Transition (and terrain) clips are meant to walk STRAIGHT up the corridor
centerline (commanded ``vy = wz = 0``), but the stitched reference drifts because
each stride's anchor is the previous stride's reference stance-foot landing pose,
whose small residual lateral / yaw components accumulate.  This script reads a
clip library ``.h5`` (no IsaacLab) and, per terrain type, plots:

* a **top-down** (x, y) view of the pelvis + both ankle paths with the terrain's
  stair treads (y-extent) and corridor edges overlaid -- so you can see the path
  wander off the treads;
* **lateral deviation** ``dy(t) = pelvis_y - centerline`` overlaid across sample
  clips, with the tread half-width drawn as the "off the stairs" threshold;
* **heading** ``yaw(t)`` (pelvis body heading + the right-ankle *anchor* yaw)
  overlaid across sample clips.

It also PRINTS a per-clip decomposition of the total lateral drift into the part
explained by heading integration (``int vx * sin(yaw) dt``) vs the residual
per-stride lateral part, so you can tell whether the drift is mostly yaw, mostly
lateral, or both.

Example::

    python scripts/rsl_rl/plot_clip_anchor_drift.py --lib clips/g1_terrain_v1.h5 \
        --out-dir plots/drift --types transition terrain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np

# --- Paper-plot styling (with graceful LaTeX fallback) ---
mpl.rcParams.update({
    "font.family": "serif",
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.5,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt  # noqa: E402

try:
    mpl.rcParams["text.usetex"] = True
    _f, _a = plt.subplots()
    _a.set_title(r"$\alpha$")
    _f.canvas.draw()
    plt.close(_f)
except Exception:
    print("[plot_clip_anchor_drift] LaTeX unavailable -- falling back to mathtext.")
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"

COLORS = {"blue": "#0077BB", "orange": "#EE7733", "teal": "#009988",
          "red": "#CC3311", "magenta": "#EE3377", "grey": "#BBBBBB", "black": "#000000"}


def _yaw_from_quat_cols(y: np.ndarray, names: list[str], frame: str) -> np.ndarray | None:
    """Heading yaw (rad) over time for ``frame`` from its ``ori_{x,y,z,w}`` columns."""
    try:
        qx = y[:, names.index(f"{frame}:ori_x")]
        qy = y[:, names.index(f"{frame}:ori_y")]
        qz = y[:, names.index(f"{frame}:ori_z")]
        qw = y[:, names.index(f"{frame}:ori_w")]
    except ValueError:
        return None
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.unwrap(np.arctan2(siny_cosp, cosy_cosp))


def _pos_cols(y: np.ndarray, names: list[str], frame: str) -> np.ndarray:
    """``[T, 3]`` world (clip-frame) position of ``frame`` over time."""
    return np.stack([y[:, names.index(f"{frame}:pos_{a}")] for a in ("x", "y", "z")], axis=1)


def _stair_bands(m: dict) -> tuple[list[tuple[float, float, float, float]], float, float]:
    """Stair tread rectangles ``(x0, x1, y0, y1)`` + (centerline_y, corridor_width)."""
    size_y = float(m["size_y"])
    w = float(m["stair_width"])
    rects = []
    cy_vals = []
    for b in m["bands"]:
        if b["block_type"] != "StairBlock":
            continue
        x0, x1 = float(b["aabb"][0]), float(b["aabb"][1])
        cy = float(np.mean(np.array(b["stair_top_centers"], dtype=float)[:, 1]))
        cy_vals.append(cy)
        rects.append((x0, x1, cy - w / 2.0, cy + w / 2.0))
    centerline = float(np.mean(cy_vals)) if cy_vals else size_y / 2.0
    return rects, centerline, w


def _stride_mean(sig: np.ndarray, stride_idx: np.ndarray) -> np.ndarray:
    """Piecewise-constant per-stride mean of ``sig`` (removes intra-stride sway).

    The pelvis / ankle heading swings tens of degrees WITHIN a stride from normal
    gait sway; that oscillation swamps a ``int vx*sin(psi)`` decomposition even
    though it nets to ~0.  Averaging over each stride segment (contiguous run of a
    constant ``stride_idx``) leaves only the slowly-accumulating heading trend --
    which is what actually integrates into lateral drift.
    """
    out = np.empty_like(sig)
    change = np.nonzero(np.diff(stride_idx))[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [sig.shape[0]]))
    for s, e in zip(starts, ends):
        out[s:e] = sig[s:e].mean()
    return out


def _decompose(dy: np.ndarray, yaw: np.ndarray, vx: np.ndarray, dt: float,
               stride_idx: np.ndarray) -> tuple[dict, np.ndarray]:
    """Split total lateral drift into heading-driven vs residual (per-stride lateral).

    Uses the per-stride-averaged heading ``psi`` (sway removed) so the split is
    robust.  Heading-driven lateral is the world-frame sideways travel a body
    walking at forward speed ``vx`` along heading ``psi`` accrues:
    ``int vx * sin(psi) dt``.  The residual is whatever lateral drift is left --
    the per-stride sideways landing offset that survives even at ``psi = 0``.
    Returns the stats dict and the smoothed heading trend (for plotting).
    """
    psi = _stride_mean(yaw, stride_idx)
    total = float(dy[-1] - dy[0])
    yaw_lat = float(np.sum(vx * np.sin(psi) * dt))
    stats = {
        "total": total,
        "yaw_driven": yaw_lat,
        "residual": total - yaw_lat,
        "final_yaw_deg": float(np.degrees(psi[-1] - psi[0])),
        "max_abs_dy": float(np.max(np.abs(dy - dy[0]))),
    }
    return stats, psi


def _plot_type(f: h5py.File, names: list[str], man_by_id: dict, dt: float,
               idxs: list[int], tname: str, out_dir: Path) -> None:
    """Write the 3-panel drift figure for up to ``len(idxs)`` clips of one type."""
    ref_frame = json.loads(f.attrs["ref_frame"]) if "ref_frame" in f.attrs else "right_ankle_roll_link"
    body = "pelvis_link" if any(n.startswith("pelvis_link:") for n in names) else ref_frame

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    ax_top, ax_lat, ax_yaw = axes
    n_show = len(idxs)

    print(f"\n=== {tname} ({n_show} clip(s)) ===")
    print(f"{'clip':>5} {'terr':>5} {'total dy':>9} {'yaw-driv':>9} {'residual':>9} "
          f"{'final yaw':>10} {'max|dy|':>8}")

    for j, ci in enumerate(idxs):
        y = f["y_des"][ci]
        T = y.shape[0]
        t = np.arange(T) * dt
        tid = int(f["terrain_id"][ci])
        m = man_by_id.get(tid)

        pel = _pos_cols(y, names, body)
        rank = _pos_cols(y, names, ref_frame)
        lank = _pos_cols(y, names, "left_ankle_roll_link") if any(
            n.startswith("left_ankle_roll_link:") for n in names) else None

        yaw_body = _yaw_from_quat_cols(y, names, body)

        # Centerline: prefer the terrain's stair center; else the spawn-y.
        if m is not None:
            rects, centerline, w = _stair_bands(m)
        else:
            rects, centerline, w = [], float(pel[0, 1]), np.nan

        vx = f["target_vel"][ci][:, 0] if "target_vel" in f else np.gradient(pel[:, 0], dt)
        stride = f["stride_idx"][ci]
        dec, psi_trend = _decompose(
            pel[:, 1], yaw_body if yaw_body is not None else np.zeros(T), vx, dt, stride)
        print(f"{ci:>5} {tid:>5} {dec['total']:>+9.3f} {dec['yaw_driven']:>+9.3f} "
              f"{dec['residual']:>+9.3f} {dec['final_yaw_deg']:>+9.2f}d {dec['max_abs_dy']:>8.3f}")

        first = (j == 0)
        a_main = 1.0 if first else 0.35
        lw_main = 2.4 if first else 1.2

        # --- Panel 1: top-down (x, y). First clip in full detail + terrain. ---
        if first:
            for (x0, x1, y0, y1) in rects:
                ax_top.fill_between([x0, x1], [y0, y0], [y1, y1],
                                    color=COLORS["grey"], alpha=0.35, zorder=0,
                                    label="stair tread" if (x0, x1) == rects[0][:2] else None)
            if m is not None:
                ax_top.axhline(centerline, color=COLORS["black"], ls=":", lw=1.5, label="centerline")
            ax_top.plot(rank[:, 0], rank[:, 1], color=COLORS["orange"], lw=1.6, label="R ankle (anchor)")
            if lank is not None:
                ax_top.plot(lank[:, 0], lank[:, 1], color=COLORS["teal"], lw=1.6, label="L ankle")
        ax_top.plot(pel[:, 0], pel[:, 1], color=COLORS["blue"], lw=lw_main, alpha=a_main,
                    label="pelvis" if first else None)

        # --- Panel 2: lateral deviation from centerline vs time ---
        ax_lat.plot(t, pel[:, 1] - centerline, color=COLORS["blue"], lw=lw_main, alpha=a_main,
                    label="pelvis" if first else None)
        if first and np.isfinite(w):
            ax_lat.axhline(+w / 2.0, color=COLORS["red"], ls="--", lw=1.5, label="tread edge")
            ax_lat.axhline(-w / 2.0, color=COLORS["red"], ls="--", lw=1.5)

        # --- Panel 3: heading yaw vs time ---
        # Faint raw pelvis heading (with gait sway) for the first clip only;
        # bold the per-stride-averaged trend (what actually accumulates) for all.
        if first and yaw_body is not None:
            ax_yaw.plot(t, np.degrees(yaw_body - yaw_body[0]), color=COLORS["grey"],
                        lw=1.0, alpha=0.7, label="pelvis heading (raw, w/ sway)")
        ax_yaw.plot(t, np.degrees(psi_trend - psi_trend[0]), color=COLORS["blue"],
                    lw=lw_main, alpha=a_main, label="heading trend (per-stride mean)" if first else None)

    ax_top.set_xlabel(r"forward $x$ (m)")
    ax_top.set_ylabel(r"lateral $y$ (m)")
    ax_top.set_title("top-down path (y exaggerated)")
    # NOT equal-aspect: the lateral drift (~cm-dm) is tiny next to the ~16 m
    # corridor, so equal scaling would compress it into an invisible sliver.
    ax_top.legend(loc="best")

    ax_lat.axhline(0.0, color=COLORS["grey"], lw=1.0)
    ax_lat.set_xlabel(r"time (s)")
    ax_lat.set_ylabel(r"lateral deviation $\Delta y$ (m)")
    ax_lat.set_title("lateral drift from centerline")
    ax_lat.legend(loc="best")

    ax_yaw.axhline(0.0, color=COLORS["grey"], lw=1.0)
    ax_yaw.set_xlabel(r"time (s)")
    ax_yaw.set_ylabel(r"heading $\Delta\psi$ (deg)")
    ax_yaw.set_title("heading drift")
    ax_yaw.legend(loc="best")

    fig.suptitle(f"{tname} clips -- anchor / heading drift", fontsize=17)
    fig.tight_layout()
    stem = out_dir / f"anchor_drift_{tname}"
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)
    print(f"[plot_clip_anchor_drift] wrote {stem}.png / .svg")


def main() -> int:
    """Parse args, load the library, and write one drift figure per terrain type."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5 path.")
    p.add_argument("--out-dir", default="plots/drift", help="Directory for the output figures.")
    p.add_argument("--types", nargs="+", default=["transition", "terrain"],
                   help="Terrain types to plot (subset of flat/terrain/transition).")
    p.add_argument("--num", type=int, default=4, help="Sample clips per type to overlay.")
    p.add_argument("--clips", type=int, nargs="+", default=None,
                   help="Explicit clip indices (overrides --types / --num).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.lib, "r") as f:
        names = json.loads(f.attrs["pos_output_names"])
        dt = float(json.loads(f.attrs["dt"]))
        tnames = json.loads(f.attrs["terrain_type_names"])
        tcode = f["terrain_type_code"][...]
        man = json.loads(f.attrs["terrain_manifest"]) if "terrain_manifest" in f.attrs else []
        man_by_id = {int(mm["terrain_id"]): mm for mm in man}

        if args.clips is not None:
            _plot_type(f, names, man_by_id, dt, list(args.clips), "custom", out_dir)
            return 0

        for tname in args.types:
            if tname not in tnames:
                print(f"[plot_clip_anchor_drift] unknown type '{tname}' -- skipping")
                continue
            code = tnames.index(tname)
            idxs = np.where(tcode == code)[0].tolist()[: args.num]
            if not idxs:
                print(f"[plot_clip_anchor_drift] no '{tname}' clips -- skipping")
                continue
            _plot_type(f, names, man_by_id, dt, idxs, tname, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
