# Copyright 2026 Zachary Olkin. All rights reserved.

"""Draw the generator's prediction against the clip it was supposed to reproduce.

The error tables say the difference is a constant plus a ramp, mostly in the knees. This is
the picture that has to agree with that, on individual motions rather than medians -- a
perturbation designed off a table and never checked against a trajectory is a perturbation
designed off a summary statistic.

Six panels, chosen for what decides a fall on stairs:

* **root height** and **root downrange** -- the reference the noise table already covers;
* **foot height** -- swing clearance, which is what catches a riser;
* **foot downrange** -- where the foot lands along the tread;
* **knee** and **hip pitch** -- the two joints carrying most of the measured error.

Solid is the CLIP (what should happen), dashed is the GENERATOR, one colour per trajectory,
so a vertical gap between a matched pair is the error at that look-ahead.

    python scripts/flow_matching/plot_fm_foot_error.py \\
        --traj <run>/fm_foot_error_traj.npz --out fm_foot_error.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "rsl_rl"))
import plot_style  # noqa: E402


def main() -> int:
    """Draw the comparison figure."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", required=True, help="fm_foot_error_traj.npz")
    p.add_argument("--out", default="fm_foot_error.png")
    p.add_argument("--foot", default="left_ankle_roll_link")
    p.add_argument("--max_traj", type=int, default=4)
    p.add_argument("--error", action="store_true",
                   help="Plot the DIFFERENCE (generator minus clip) instead of both curves. "
                        "Two overlaid trajectories hide whether the gap is a ramp, a bump or "
                        "a wobble -- which is the question a perturbation has to answer.")
    a = p.parse_args()

    import matplotlib.pyplot as plt
    plot_style.apply_paper_style()

    z = np.load(a.traj, allow_pickle=True)
    jn = json.loads(str(z["joint_names"]))
    blocks = json.loads(str(z["blocks"]))
    bodies = json.loads(str(z["bodies"]))
    dt = float(z["dt"])
    bi = bodies.index(a.foot) if a.foot in bodies else 0
    bp0 = blocks["base_pose"][0]
    jp0 = blocks["joint_pos"][0]
    n = min(a.max_traj, z["pred"].shape[0])
    t = np.arange(z["pred"].shape[1]) * dt

    side = a.foot.split("_")[0]                       # left / right
    ji_knee = jn.index(f"{side}_knee_joint")
    ji_hip = jn.index(f"{side}_hip_pitch_joint")

    panels = [
        ("root height (m)", lambda w, b: w[:, bp0 + 2]),
        ("root downrange (m)", lambda w, b: w[:, bp0 + 0]),
        (f"{side} foot height (m)", lambda w, b: b[:, bi, 2]),
        (f"{side} foot downrange (m)", lambda w, b: b[:, bi, 0]),
        (f"{side} knee (rad)", lambda w, b: w[:, jp0 + ji_knee]),
        (f"{side} hip pitch (rad)", lambda w, b: w[:, jp0 + ji_hip]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (label, fn) in zip(axes.ravel(), panels):
        for i in range(n):
            c = colors[i % len(colors)]
            tv = fn(z["targ"][i], z["targ_bodies"][i])
            pv = fn(z["pred"][i], z["pred_bodies"][i])
            if a.error:
                ax.plot(t, pv - tv, color=c, lw=1.5)
            else:
                ax.plot(t, tv, color=c, lw=1.8, alpha=0.9)
                ax.plot(t, pv, color=c, lw=1.4, ls="--")
        if a.error:
            ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
        ax.set_ylabel(("error: " if a.error else "") + label)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("look-ahead (s)")
    if not a.error:
        axes[0][0].plot([], [], color="0.3", lw=1.8, label="clip (target)")
        axes[0][0].plot([], [], color="0.3", lw=1.4, ls="--", label="generator")
        axes[0][0].legend(loc="best", fontsize=9)
    fig.suptitle((f"{n} stairs_up windows: generator MINUS clip over one 1.24 s plan"
                  if a.error else
                  f"{n} stairs_up windows: generator vs clip over one 1.24 s plan")
                 + "  (teacher-forced, so both start from the same state)", fontsize=11)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.splitext(a.out)[0], svg=False)
    print(f"[plot] wrote {os.path.splitext(a.out)[0]}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
