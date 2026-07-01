#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Time-series plots of a generated motion clip.

Reconstructs each reference frame's world pose from a dense clip and plots
position (x/y/z) and orientation (roll/pitch/yaw, degrees) vs. time, plus the
sampled target velocity, with foot stance phases shaded.  Use ``--start`` /
``--end`` to zoom into a time window (the y-axes rescale to that window).

Example::

    python scripts/rsl_rl/plot_clip.py --lib clips/g1_flat_v1.h5 --clip 0 --out /tmp/clip0_ts.png
    python scripts/rsl_rl/plot_clip.py --lib clips/g1_flat_v1.h5 --clip 0 --start 6 --end 10
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_io import (
    load_clip_library,
)

# Default frames to plot (base + feet); pass --wrists to add the hands.
_DEFAULT_FRAMES = ["pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link"]
_WRIST_FRAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
_COLORS = {
    "pelvis_link": "tab:orange",
    "left_ankle_roll_link": "tab:blue",
    "right_ankle_roll_link": "tab:red",
    "left_wrist_yaw_link": "tab:cyan",
    "right_wrist_yaw_link": "tab:purple",
}


def _frame_pos_idx(pos_names: list[str], frame: str) -> list[int]:
    return [pos_names.index(f"{frame}:pos_{a}") for a in ("x", "y", "z")]


def _frame_ori_idx(pos_names: list[str], frame: str) -> list[int]:
    # xyzw (scalar last) to match Isaac Lab's quat math; YAML stores ori_w first.
    return [pos_names.index(f"{frame}:ori_{a}") for a in ("x", "y", "z", "w")]


def _rpy_deg(quat_track) -> np.ndarray:
    """World roll/pitch/yaw (degrees) [T, 3] for a clip-world quat track."""
    r, p, y = euler_xyz_from_quat(quat_track)
    rpy = torch.stack([wrap_to_pi(r), wrap_to_pi(p), wrap_to_pi(y)], dim=-1)
    return torch.rad2deg(rpy).cpu().numpy()


def main() -> int:
    """Parse args, reconstruct world tracks for one clip window, and plot vs time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5 path.")
    p.add_argument("--clip", type=int, default=0, help="Clip index.")
    p.add_argument("--out", default=None, help="Output image path (default clip_<i>_ts.png).")
    p.add_argument("--wrists", action="store_true", help="Also plot the wrist frames.")
    p.add_argument("--start", type=float, default=0.0, help="Window start time (s).")
    p.add_argument("--end", type=float, default=None, help="Window end time (s; default clip end).")
    p.add_argument("--no-contact-shading", action="store_true", help="Disable foot stance shading.")
    args = p.parse_args()

    lib = load_clip_library(args.lib, "cpu")
    ci = args.clip
    pos_names = lib.meta["pos_output_names"]
    dt = float(lib.meta["dt"])
    contact_frames = lib.meta["contact_frames"]
    n_steps = lib.num_steps

    # Resolve the [start, end) window to a step range.
    end = args.end if args.end is not None else n_steps * dt
    i0 = int(np.clip(round(args.start / dt), 0, n_steps - 1))
    i1 = int(np.clip(round(end / dt), i0 + 1, n_steps))
    sl = slice(i0, i1)
    t = np.arange(i0, i1) * dt

    y = lib.y_des[ci][sl]  # clip-world

    frames = list(_DEFAULT_FRAMES) + (_WRIST_FRAMES if args.wrists else [])
    frames = [f for f in frames if f"{f}:pos_x" in pos_names]
    tracks = {f: y[:, _frame_pos_idx(pos_names, f)].cpu().numpy() for f in frames}
    # Orientation is shown for the base (pelvis) only -- the feet just wrap
    # through +-180 deg and aren't a useful read.
    base_frame = "pelvis_link" if "pelvis_link" in frames else frames[0]
    base_rpy = _rpy_deg(y[:, _frame_ori_idx(pos_names, base_frame)])

    has_vel = lib.target_vel.numel() > 0
    n_rows = 7 if has_vel else 6
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows), sharex=True)

    def _shade(ax):
        if args.no_contact_shading:
            return
        for cidx, cf in enumerate(contact_frames):
            c = lib.contact[ci, sl, cidx].cpu().numpy().astype(bool)
            col = _COLORS.get(cf, "gray")
            edges = np.diff(c.astype(int))
            starts = list(np.where(edges == 1)[0] + 1)
            ends = list(np.where(edges == -1)[0] + 1)
            if c[0]:
                starts = [0] + starts
            if c[-1]:
                ends = ends + [len(c)]
            for s, e in zip(starts, ends):
                ax.axvspan(t[s], t[min(e, len(c) - 1)], color=col, alpha=0.06)

    def _short(f):
        return f.replace("_roll_link", "").replace("_yaw_link", "").replace("_link", "")

    # Rows 0-2: position x/y/z.  Rows 3-5: orientation roll/pitch/yaw (deg).
    pos_labels = ["x position (m)", "y position (m)", "z height (m)"]
    rpy_labels = ["pelvis roll (deg)", "pelvis pitch (deg)", "pelvis yaw (deg)"]
    for axis_i in range(3):
        ax = axes[axis_i]
        _shade(ax)
        for f in frames:
            ax.plot(t, tracks[f][:, axis_i], color=_COLORS.get(f), lw=1.6, label=_short(f))
        ax.set_ylabel(pos_labels[axis_i])
        ax.grid(True, alpha=0.3)
    for axis_i in range(3):
        ax = axes[3 + axis_i]
        _shade(ax)
        ax.plot(t, base_rpy[:, axis_i], color=_COLORS.get(base_frame), lw=1.6)
        ax.set_ylabel(rpy_labels[axis_i])
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", ncol=len(frames), fontsize=8)

    if has_vel:
        tv = lib.target_vel[ci][sl].cpu().numpy()
        axv = axes[6]
        for k, name, col in [(0, "vx", "tab:green"), (1, "vy", "tab:olive"), (2, "wz", "tab:brown")]:
            axv.plot(t, tv[:, k], color=col, lw=1.6, label=name)
        axv.set_ylabel("target vel\n(m/s, rad/s)")
        axv.legend(loc="upper left", ncol=3, fontsize=8)
        axv.grid(True, alpha=0.3)

    sk = lib.skill_id[ci, sl].cpu().numpy()
    skills = [lib.meta["skill_list"][s] for s in sorted(set(sk.tolist()))]
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(
        f"clip {ci}  [{t[0]:.2f}-{t[-1]:.2f}s of {lib.meta['clip_length_s']}s, dt={dt}s]  "
        f"skills: {', '.join(skills)}"
        + ("   [shaded = foot stance]" if not args.no_contact_shading else "")
    )

    out = args.out or f"clip_{ci}_ts.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot_clip] wrote {out}  (clip {ci}: steps {i0}-{i1}, frames={frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
