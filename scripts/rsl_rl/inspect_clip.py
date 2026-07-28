#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline matplotlib inspection of a clip library ``.h5``.

Both modes read only the ``.h5`` (no IsaacLab, no simulator), so this is the fast
validation loop after generating a library.  For the interactive 3-D viewer with a URDF
robot, use ``view_clip_viser.py`` instead; for seam/segment boundaries specifically, see
``plot_clip_segments.py``.

Modes
-----
``3d``
    Whole-clip 3-D pelvis + foot tracks, coloured by active skill, with a marker at each
    foot touchdown.  A good clip shows a smooth pelvis path, feet planted during stance
    (flat track segments), and clean stride-to-stride seams.
``timeseries``
    Per-frame position (x/y/z) and pelvis orientation (roll/pitch/yaw, deg) vs time, plus
    the commanded target velocity, with foot stance phases shaded.  ``--start`` / ``--end``
    zoom into a window (the y-axes rescale to it).

Examples::

    python scripts/rsl_rl/inspect_clip.py --mode 3d         --lib clips/g1_mb_v4.h5 --clip 0
    python scripts/rsl_rl/inspect_clip.py --mode timeseries --lib clips/g1_mb_v4.h5 --clip 0 --start 6 --end 10

Two further modes were dropped when this tool was assembled: an anchor/heading DRIFT check
and a per-terrain-type side-view sanity plot.  Both parsed the legacy *bands* terrain
manifest (``dims`` / ``bands`` / ``stair_width``); every library the current pipeline emits
uses the *overlay* schema (``overlays`` / ``size_x`` / ``family``), so both had been raising
KeyError on real input since well before this merge.  The diagnostics are worth having --
porting them to the overlay schema is feature work, not cleanup.  Until then, use
``view_terrain_viser.py`` / ``view_terrain_isaac.py`` for terrain geometry and ``mb_qc.py``
for the objective per-library quality report (foot slide, seam spikes, tracking).
"""

from __future__ import annotations

import argparse

import numpy as np

# Per-frame colours for the timeseries / 3d modes.
_FRAME_COLORS = {
    "pelvis_link": "tab:orange",
    "left_ankle_roll_link": "tab:blue",
    "right_ankle_roll_link": "tab:red",
    "left_wrist_yaw_link": "tab:cyan",
    "right_wrist_yaw_link": "tab:purple",
}
_DEFAULT_FRAMES = ["pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link"]
_WRIST_FRAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


##
# Shared column helpers (numpy; work on either an h5 slab or a .cpu().numpy() tensor)
##
def _pos_cols(y: np.ndarray, names: list[str], frame: str) -> np.ndarray:
    """``[T, 3]`` clip-world position of ``frame`` over time."""
    return np.stack([y[:, names.index(f"{frame}:pos_{a}")] for a in ("x", "y", "z")], axis=1)


def _quat_cols(y: np.ndarray, names: list[str], frame: str) -> np.ndarray | None:
    """``[T, 4]`` quaternion of ``frame`` as ``(x, y, z, w)``, or None if absent."""
    try:
        return np.stack([y[:, names.index(f"{frame}:ori_{a}")] for a in ("x", "y", "z", "w")], axis=1)
    except ValueError:
        return None


def _rpy_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw in degrees ``[T, 3]`` from an ``(x, y, z, w)`` quat track.

    XYZ extrinsic convention, matching ``isaaclab.utils.math.euler_xyz_from_quat`` (which
    also takes scalar-last) -- reimplemented in numpy so this script needs no IsaacLab.
    """
    qx, qy, qz, qw = (quat_xyzw[:, i] for i in range(4))
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.degrees(np.stack([roll, pitch, yaw], axis=-1))


def _short(frame: str) -> str:
    return frame.replace("_roll_link", "").replace("_yaw_link", "").replace("_link", "")


##
# Mode: 3d
##
def _mode_3d(args) -> int:
    """Whole-clip 3-D pelvis + foot tracks, coloured by skill, touchdowns marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    from motion_gen.clips.clip_io import load_clip_library

    lib = load_clip_library(args.lib, "cpu")
    ci = args.clip
    names = lib.meta["pos_output_names"]
    skill_list = lib.meta["skill_list"]

    y = lib.y_des[ci].cpu().numpy()          # [T, P] clip-world (no lift needed)
    skill = lib.skill_id[ci].cpu().numpy()
    contact = lib.contact[ci].cpu().numpy()
    cframes = lib.meta["contact_frames"]

    pelvis = _pos_cols(y, names, "pelvis_link")
    feet = {f: _pos_cols(y, names, f) for f in cframes}

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

    out = args.out or f"clip_{ci}_3d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[inspect_clip:3d] wrote {out}  (clip {ci}: {lib.num_steps} steps, "
          f"skills present={sorted(set(skill.tolist()))})")
    return 0


##
# Mode: timeseries
##
def _mode_timeseries(args) -> int:
    """Position / pelvis orientation / commanded velocity vs time, stance shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from motion_gen.clips.clip_io import load_clip_library

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

    y = lib.y_des[ci][sl].cpu().numpy()  # clip-world

    frames = list(_DEFAULT_FRAMES) + (_WRIST_FRAMES if args.wrists else [])
    frames = [f for f in frames if f"{f}:pos_x" in pos_names]
    tracks = {f: _pos_cols(y, pos_names, f) for f in frames}
    # Orientation is shown for the base (pelvis) only -- the feet just wrap through
    # +-180 deg and aren't a useful read.
    base_frame = "pelvis_link" if "pelvis_link" in frames else frames[0]
    base_rpy = _rpy_deg(_quat_cols(y, pos_names, base_frame))

    has_vel = lib.target_vel.numel() > 0
    n_rows = 7 if has_vel else 6
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows), sharex=True)

    def _shade(ax):
        if args.no_contact_shading:
            return
        for cidx, cf in enumerate(contact_frames):
            c = lib.contact[ci, sl, cidx].cpu().numpy().astype(bool)
            col = _FRAME_COLORS.get(cf, "gray")
            edges = np.diff(c.astype(int))
            starts = list(np.where(edges == 1)[0] + 1)
            ends = list(np.where(edges == -1)[0] + 1)
            if c[0]:
                starts = [0] + starts
            if c[-1]:
                ends = ends + [len(c)]
            for s, e in zip(starts, ends):
                ax.axvspan(t[s], t[min(e, len(c) - 1)], color=col, alpha=0.06)

    # Rows 0-2: position x/y/z.  Rows 3-5: pelvis orientation roll/pitch/yaw (deg).
    pos_labels = ["x position (m)", "y position (m)", "z height (m)"]
    rpy_labels = ["pelvis roll (deg)", "pelvis pitch (deg)", "pelvis yaw (deg)"]
    for axis_i in range(3):
        ax = axes[axis_i]
        _shade(ax)
        for f in frames:
            ax.plot(t, tracks[f][:, axis_i], color=_FRAME_COLORS.get(f), lw=1.6, label=_short(f))
        ax.set_ylabel(pos_labels[axis_i])
        ax.grid(True, alpha=0.3)
    for axis_i in range(3):
        ax = axes[3 + axis_i]
        _shade(ax)
        ax.plot(t, base_rpy[:, axis_i], color=_FRAME_COLORS.get(base_frame), lw=1.6)
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
    print(f"[inspect_clip:timeseries] wrote {out}  (clip {ci}: steps {i0}-{i1}, frames={frames})")
    return 0


_MODES = {"3d": _mode_3d, "timeseries": _mode_timeseries}


def main() -> int:
    """Parse args and dispatch to the requested inspection mode."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=sorted(_MODES),
                   help="Which inspection to run (see the module docstring).")
    p.add_argument("--lib", required=True, help="Clip library .h5 path.")
    p.add_argument("--clip", type=int, default=0, help="Clip index.")
    p.add_argument("--out", default=None, help="Output image path.")
    p.add_argument("--wrists", action="store_true", help="[timeseries only] Also plot the wrist frames.")
    p.add_argument("--start", type=float, default=0.0, help="[timeseries only] Window start time (s).")
    p.add_argument("--end", type=float, default=None,
                   help="[timeseries only] Window end time (s; default clip end).")
    p.add_argument("--no-contact-shading", action="store_true",
                   help="[timeseries only] Disable foot stance shading.")
    args = p.parse_args()

    return _MODES[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
