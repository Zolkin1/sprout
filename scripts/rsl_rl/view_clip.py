#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Visualise an entire stitched motion clip at once (headless matplotlib).

Reconstructs the world-frame pelvis and foot tracks for one clip from a
dense clip library and plots them in 3D, coloured by the active skill, with
a marker at each foot touchdown.  This is the fast validation loop for the
offline generator: a good clip shows a smooth pelvis path, feet planted
during stance (flat track segments), and clean stride-to-stride seams.

Example::

    python scripts/rsl_rl/view_clip.py --lib clips/g1_flat_v1.h5 --clip 0 \\
        --out /tmp/clip0.png
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_io import (
    load_clip_library,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.world_lift import (
    lift_position,
)


def _world_track(y_des, anchor_pos, anchor_quat, idx) -> np.ndarray:
    """World xyz track [T, 3] for the frame whose pos outputs are at ``idx``."""
    pos_rel = y_des[:, idx]                       # [T, 3]
    return lift_position(anchor_pos, anchor_quat, pos_rel).cpu().numpy()


def main() -> int:
    """Parse args, reconstruct world tracks for one clip, and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5 path.")
    p.add_argument("--clip", type=int, default=0, help="Clip index to view.")
    p.add_argument("--out", default=None, help="Output image path (defaults to clip_<i>.png).")
    args = p.parse_args()

    lib = load_clip_library(args.lib, "cpu")
    ci = args.clip
    names = lib.meta["pos_output_names"]
    skill_list = lib.meta["skill_list"]

    y = lib.y_des[ci]                 # [T, P]
    ap = lib.anchor_pos[ci]           # [T, 3]
    aq = lib.anchor_quat[ci]          # [T, 4]
    skill = lib.skill_id[ci].cpu().numpy()
    contact = lib.contact[ci].cpu().numpy()
    cframes = lib.meta["contact_frames"]

    def idx(frame, axes=("pos_x", "pos_y", "pos_z")):
        return [names.index(f"{frame}:{a}") for a in axes]

    pelvis = _world_track(y, ap, aq, idx("pelvis_link"))
    feet = {f: _world_track(y, ap, aq, idx(f)) for f in cframes}

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Pelvis path coloured by active skill.
    cmap = plt.get_cmap("tab10")
    for s in range(len(skill_list)):
        m = skill == s
        if m.any():
            ax.scatter(pelvis[m, 0], pelvis[m, 1], pelvis[m, 2], s=4,
                       color=cmap(s), label=f"pelvis: {skill_list[s]}")

    # Foot tracks + touchdown markers (contact rising edges).
    foot_colors = {cframes[0]: "tab:blue", cframes[-1]: "tab:red"}
    for fi, (fname, track) in enumerate(feet.items()):
        ax.plot(track[:, 0], track[:, 1], track[:, 2], lw=0.8, alpha=0.5,
                color=foot_colors.get(fname, "gray"), label=f"foot: {fname}")
        c = contact[:, fi].astype(bool)
        td = np.where(c[1:] & ~c[:-1])[0] + 1  # rising edges = touchdowns
        ax.scatter(track[td, 0], track[td, 1], track[td, 2], marker="o", s=40,
                   edgecolor="k", facecolor=foot_colors.get(fname, "gray"))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(
        f"clip {ci}  ({lib.meta['clip_length_s']}s, dt={lib.meta['dt']}s)  "
        f"pelvis net xy = {np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2]):.2f} m"
    )
    ax.legend(loc="upper left", fontsize=7)
    # Equal aspect on xy for an undistorted ground track.
    xr = float(np.ptp(pelvis[:, 0])) or 1.0
    yr = float(np.ptp(pelvis[:, 1])) or 1.0
    ax.set_box_aspect((max(xr, yr), max(xr, yr), max(0.5, float(np.ptp(pelvis[:, 2])))))

    out = args.out or f"clip_{ci}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[view_clip] wrote {out}  (clip {ci}: {lib.num_steps} steps, "
          f"skills present={sorted(set(skill.tolist()))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
