#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Diagnose a motion-matching ``.h5`` clip: velocity tracking, root continuity, foot ground.

Reads only the clip (method-agnostic), so it compares vanilla vs beam output directly:

* **tracking** -- body-frame root velocity (vx fwd, vy left, wz yaw) vs the stored
  ``target_vel`` command (mean abs error per component + overall);
* **root continuity** -- per-step world-root xy jump (a cut should stay ~ a normal step);
* **foot ground** -- min ankle-roll height + how many frames a foot penetrates below 0.

    python scripts/rsl_rl/diagnose_mm_clip.py clipA.h5 clipB.h5 ...
"""

from __future__ import annotations

import argparse

import numpy as np

from motion_gen.clips.clip_io import load_clip_library


def _yaw_wxyz(q):    # [T,4] -> [T]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _cdiff(x, dt):
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def diagnose(path: str) -> dict:
    lib = load_clip_library(path)
    meta = lib.meta
    pn = list(meta["pos_output_names"])
    dt = float(meta["dt"])
    y = np.asarray(lib.y_des)[0]                         # [T, C]
    tvel = np.asarray(lib.target_vel)[0]                 # [T, 3]
    T = y.shape[0]

    px = y[:, pn.index("pelvis_link:pos_x")]
    py = y[:, pn.index("pelvis_link:pos_y")]
    pz = y[:, pn.index("pelvis_link:pos_z")]
    ori = np.stack([y[:, pn.index(f"pelvis_link:ori_{a}")] for a in "wxyz"], axis=-1)
    yaw = np.unwrap(_yaw_wxyz(ori))

    vwx, vwy = _cdiff(px, dt), _cdiff(py, dt)            # world root lin vel
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx = cy * vwx + sy * vwy                             # body fwd
    vy = -sy * vwx + cy * vwy                            # body left
    wz = _cdiff(yaw, dt)
    act = np.stack([vx, vy, wz], axis=-1)
    err = np.abs(act - tvel)

    step_xy = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)     # per-step world xy displacement
    med = float(np.median(step_xy)) if len(step_xy) else 0.0
    # A true cut discontinuity is a single unphysical world-root jump.  Playback re-anchors
    # SE(2) at every cut, so this should be ~0; flag steps above ~6 m/s (0.12 m at dt=0.02).
    disc_thresh = 6.0 * dt
    big = int((step_xy > disc_thresh).sum())

    lz = y[:, pn.index("left_ankle_roll_link:pos_z")]
    rz = y[:, pn.index("right_ankle_roll_link:pos_z")]
    foot_min = float(min(lz.min(), rz.min()))
    pen = int(((lz < -0.02).sum() + (rz < -0.02).sum()))

    return {
        "T": T,
        "track_mae": float(err.mean()),
        "mae_vx": float(err[:, 0].mean()), "mae_vy": float(err[:, 1].mean()),
        "mae_wz": float(err[:, 2].mean()),
        "peak_step_xy": float(step_xy.max()) if len(step_xy) else 0.0,
        "median_step_xy": med, "big_jumps": big,
        "pelvis_z_range": (float(pz.min()), float(pz.max())),
        "foot_min_z": foot_min, "foot_penetrations": pen,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+")
    args = p.parse_args()
    for c in args.clips:
        d = diagnose(c)
        print(f"\n=== {c}  (T={d['T']}) ===")
        print(f"  tracking MAE  overall={d['track_mae']:.3f}  "
              f"vx={d['mae_vx']:.3f}  vy={d['mae_vy']:.3f}  wz={d['mae_wz']:.3f}")
        print(f"  root xy step  peak={d['peak_step_xy']:.3f}  median={d['median_step_xy']:.3f}  "
              f"discontinuities(>6 m/s)={d['big_jumps']}")
        print(f"  pelvis z      [{d['pelvis_z_range'][0]:.3f}, {d['pelvis_z_range'][1]:.3f}]")
        print(f"  foot ground   min_z={d['foot_min_z']:.3f}  penetrations(<-0.02)={d['foot_penetrations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
