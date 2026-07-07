#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Sanity plots for a terrain-aware clip library: one sample per terrain type.

For a ``flat`` / ``terrain`` / ``transition`` sample clip each, plots a side view
(terrain surface profile + pelvis and foot height vs forward position), the
commanded forward-velocity profile, and the per-step skill timeline — enough to
eyeball that flat clips stay flat, terrain clips climb their staircase, and
transition clips walk in then climb.  Reads only the ``.h5`` (no IsaacLab).

Example::

    python scripts/rsl_rl/plot_terrain_clips.py --lib clips/g1_terrain_v1.h5 --out-dir plots/terrain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

# --- Paper-plot styling (with graceful LaTeX fallback) ---
mpl.rcParams.update({
    "font.family": "serif",
    "lines.linewidth": 3,
    "lines.markersize": 8,
    "axes.linewidth": 1.5,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 12,
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
    print("[plot_terrain_clips] LaTeX unavailable -- falling back to mathtext.")
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"

COLORS = {"blue": "#0077BB", "orange": "#EE7733", "teal": "#009988",
          "red": "#CC3311", "grey": "#BBBBBB", "black": "#000000"}


def _surface_profile(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the terrain surface polyline (x, z) across all bands (clip-world)."""
    d = float(m["dims"]["d"])
    bands = m.get("bands")
    if bands is None:  # legacy single-stair manifest
        bands = [{"block_type": "StairBlock", "aabb": [m["span_xmin"], m["span_xmax"], 0, 0],
                  "entry_z": m["entry_z"], "exit_z": m["exit_z"], "stair_top_centers": m["stair_top_centers"]}]
    xs, zs = [], []
    for b in bands:
        x0, x1 = float(b["aabb"][0]), float(b["aabb"][1])
        if b["block_type"] == "StairBlock":
            for cx, _cy, cz in np.array(b["stair_top_centers"], dtype=float):
                left, right = cx - d / 2.0, cx + d / 2.0
                xs += [left, right]             # implicit vertical riser at ``left`` then the tread top
                zs += [float(cz), float(cz)]
        else:  # flat band at its level
            xs += [x0, x1]
            zs += [float(b["exit_z"]), float(b["exit_z"])]
    return np.array(xs), np.array(zs)


def _plot_clip(data, ci: int, ttype: str, out_dir: Path) -> None:
    """Write the 3-panel sanity figure for clip ``ci``."""
    names = data.meta["pos_output_names"]
    skill_list = data.meta["skill_list"]
    dt = float(data.meta["dt"])
    px = names.index("pelvis_link:pos_x"); pz = names.index("pelvis_link:pos_z")
    rax = names.index("right_ankle_roll_link:pos_x"); raz = names.index("right_ankle_roll_link:pos_z")
    lax = names.index("left_ankle_roll_link:pos_x"); laz = names.index("left_ankle_roll_link:pos_z")

    y = data.y_des[ci].cpu().numpy()
    t = np.arange(y.shape[0]) * dt
    vx = data.target_vel[ci, :, 0].cpu().numpy() if data.target_vel.numel() else np.zeros_like(t)
    skill = data.skill_id[ci].cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.0))

    # --- Panel 1: side view (forward x vs height z) + terrain surface ---
    ax = axes[0]
    tid = int(data.terrain_id[ci])
    if tid >= 0:
        m = next(mm for mm in data.attrs["terrain_manifest"] if mm["terrain_id"] == tid)
        sx, sz = _surface_profile(m)
        ax.plot(sx, sz, color=COLORS["grey"], lw=4, label="terrain", zorder=1)
        ax.fill_between(sx, sz, sz.min() - 0.1, color=COLORS["grey"], alpha=0.25, zorder=0)
    else:
        ax.axhline(0.0, color=COLORS["grey"], lw=4, label="terrain")
    # Plot each frame's height against its OWN forward x (the swing foot leads
    # the pelvis, so using pelvis x for the feet would shift them spuriously).
    ax.plot(y[:, px], y[:, pz], color=COLORS["blue"], label="pelvis")
    ax.plot(y[:, rax], y[:, raz], color=COLORS["orange"], lw=2, label="R ankle")
    ax.plot(y[:, lax], y[:, laz], color=COLORS["teal"], lw=2, label="L ankle")
    ax.set_xlabel(r"forward $x$ (m)")
    ax.set_ylabel(r"height $z$ (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left")

    # --- Panel 2: commanded forward velocity ---
    ax = axes[1]
    ax.plot(t, vx, color=COLORS["red"])
    ax.set_xlabel(r"time (s)")
    ax.set_ylabel(r"cmd $v_x$ (m/s)")

    # --- Panel 3: skill timeline ---
    ax = axes[2]
    ax.plot(t, skill, color=COLORS["black"], drawstyle="steps-post")
    ax.set_yticks(range(len(skill_list)))
    ax.set_yticklabels([s.replace("_", r"\_") if mpl.rcParams["text.usetex"] else s for s in skill_list])
    ax.set_xlabel(r"time (s)")
    ax.set_ylabel("skill")

    fig.suptitle(f"{ttype} clip (id {ci}" + (f", terrain {tid})" if tid >= 0 else ")"), fontsize=18)
    fig.tight_layout()
    stem = out_dir / f"terrain_clip_{ttype}"
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)
    print(f"[plot_terrain_clips] wrote {stem}.png / .svg")


def main() -> int:
    """Parse args, load the library, and write one sanity figure per terrain type."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5 path.")
    p.add_argument("--out-dir", default="plots/terrain", help="Directory for the output figures.")
    args = p.parse_args()

    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_io import (
        load_clip_library,
    )

    data = load_clip_library(args.lib)
    names = data.attrs.get("terrain_type_names", ["flat", "terrain", "transition"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = data.terrain_type_code.cpu().numpy() if data.terrain_type_code.numel() else np.zeros(data.num_clips, int)
    for code, tname in enumerate(names):
        matches = np.where(codes == code)[0]
        if matches.size == 0:
            print(f"[plot_terrain_clips] no '{tname}' clips -- skipping")
            continue
        _plot_clip(data, int(matches[0]), tname, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
