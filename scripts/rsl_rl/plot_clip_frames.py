#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-frame inspection plots for a single clip in a clip library ``.h5``.

Pick a clip (and optionally a frame window) and plot, over those frames:

* x, y, z of the ROOT link (pelvis) and each FOOT (both ankles);
* heading yaw of the root + feet;
* the commanded velocity (vx, vy, wz) that drove stride selection;
* the skill-transition INTERPOLATION state -- whether each frame is mid
  cross-fade and the blend weight ``alpha`` (0 = pure old stride, 1 = pure new).

Over each interpolation region the two PRE-blend nominals are overlaid on the
position / yaw panels -- the old-stride continuation (dashed) and the pure new
segment (dotted) -- so the blended (solid) value can be read against them.

Stride seams (``stride_idx`` changes) are marked as vertical lines on every
panel so a position/yaw jump can be read against the seam that caused it.

The interpolation panel needs the ``blend_active`` / ``blend_alpha`` datasets,
which are only written when the library was generated with ``--debug-interp``
(``scripts/rsl_rl/generate_terrain_clips.py --debug-interp ...``); without them
the panel shows a note and the rest still plots.

Reads the ``.h5`` directly (h5py + numpy, no IsaacLab).

Example::

    python scripts/rsl_rl/plot_clip_frames.py --clip_path clips/g1_terrain_v1.h5 \
        --clip 16 --frames 60 90 --out plots/clip16_60-90.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "lines.linewidth": 1.8,
    "axes.linewidth": 1.2,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt  # noqa: E402

_COL = {"root": "#000000", "left": "#0077BB", "right": "#EE7733",
        "alpha": "#009988", "seam": "#BBBBBB"}


def _attr(f: h5py.File, key: str, default=None):
    """Decode a (possibly JSON-encoded) root attribute."""
    if key not in f.attrs:
        return default
    v = f.attrs[key]
    if isinstance(v, (str, bytes)):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def _pos(y: np.ndarray, names: list[str], frame: str) -> np.ndarray:
    """``[T, 3]`` clip-world position of ``frame`` (raises if the frame is absent)."""
    return np.stack([y[:, names.index(f"{frame}:pos_{a}")] for a in ("x", "y", "z")], axis=1)


def _yaw(y: np.ndarray, names: list[str], frame: str, unwrap: bool = True) -> np.ndarray | None:
    """Heading yaw (deg) of ``frame`` from its ``ori_{x,y,z,w}`` columns.

    ``unwrap`` is disabled for the NaN-gapped nominals (unwrap propagates NaN
    across the gaps); a short blend region needs no unwrapping anyway.
    """
    try:
        qx, qy, qz, qw = (y[:, names.index(f"{frame}:ori_{a}")] for a in ("x", "y", "z", "w"))
    except ValueError:
        return None
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.degrees(np.unwrap(yaw) if unwrap else yaw)


def _mark_seams(ax, frames: np.ndarray, stride: np.ndarray) -> None:
    """Vertical lines where stride_idx changes within the window (a stride swap)."""
    chg = np.nonzero(np.diff(stride) != 0)[0] + 1  # index (in window) of the new stride
    for i in chg:
        ax.axvline(frames[i], color=_COL["seam"], ls="--", lw=1.0, zorder=0)


def _shade_blend(ax, frames: np.ndarray, active: np.ndarray | None) -> None:
    """Light background over frames that are mid cross-fade (interpolation regime)."""
    if active is None:
        return
    ax.fill_between(frames, 0, 1, where=active.astype(bool), transform=ax.get_xaxis_transform(),
                    color=_COL["alpha"], alpha=0.10, step="mid", zorder=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip_path", required=True, help="Clip library .h5 path.")
    p.add_argument("--clip", type=int, default=0, help="Clip index to plot.")
    p.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"), default=None,
                   help="Inclusive frame window [START, END]; default the whole clip.")
    p.add_argument("--root-frame", default="pelvis_link", help="Root/base output frame name.")
    p.add_argument("--feet", nargs="+", default=["left_ankle_roll_link", "right_ankle_roll_link"],
                   help="Foot output frame names (first=left, second=right for coloring).")
    p.add_argument("--out", default=None, help="Output PNG (default: show interactively).")
    args = p.parse_args()

    with h5py.File(args.clip_path, "r") as f:
        n_clips = int(f["y_des"].shape[0])
        if not 0 <= args.clip < n_clips:
            raise SystemExit(f"--clip {args.clip} out of range [0, {n_clips})")
        y = f["y_des"][args.clip]                       # [T, P]
        names = _attr(f, "pos_output_names")
        stride = f["stride_idx"][args.clip] if "stride_idx" in f else np.zeros(y.shape[0], int)
        skills = _attr(f, "skill_list", [])
        skill_id = f["skill_id"][args.clip] if "skill_id" in f else None
        tid = int(f["terrain_id"][args.clip]) if "terrain_id" in f else -1
        tvel = f["target_vel"][args.clip] if "target_vel" in f and f["target_vel"].ndim == 3 else None
        has_blend = "blend_active" in f and "blend_alpha" in f
        blend_active = f["blend_active"][args.clip] if has_blend else None
        blend_alpha = f["blend_alpha"][args.clip] if has_blend else None
        has_nominals = "blend_old" in f and "blend_new" in f
        blend_old = f["blend_old"][args.clip] if has_nominals else None   # [T, P] (NaN off-blend)
        blend_new = f["blend_new"][args.clip] if has_nominals else None
    if names is None:
        raise SystemExit("clip library has no 'pos_output_names' attr -- cannot resolve frames.")

    T = y.shape[0]
    lo, hi = (0, T - 1) if args.frames is None else (max(0, args.frames[0]), min(T - 1, args.frames[1]))
    sl = slice(lo, hi + 1)
    frames = np.arange(lo, hi + 1)

    left, right = args.feet[0], (args.feet[1] if len(args.feet) > 1 else args.feet[0])
    tracks = [("root", args.root_frame), ("left", left), ("right", right)]
    pos = {k: _pos(y, names, fr)[sl] for k, fr in tracks}
    yaw = {k: (_yaw(y, names, fr)[sl] if _yaw(y, names, fr) is not None else None) for k, fr in tracks}
    stride_w = stride[sl]
    act = blend_active[sl] if has_blend else None
    # Pre-blend nominals (NaN off-blend, so they only draw over interp regions).
    if has_nominals:
        pos_old = {k: _pos(blend_old, names, fr)[sl] for k, fr in tracks}
        pos_new = {k: _pos(blend_new, names, fr)[sl] for k, fr in tracks}
        yaw_old = {k: _yaw(blend_old, names, fr, unwrap=False)[sl] for k, fr in tracks}
        yaw_new = {k: _yaw(blend_new, names, fr, unwrap=False)[sl] for k, fr in tracks}

    nrows = 6
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 14), sharex=True)
    ttype = {-1: "flat"}.get(tid, f"terrain {tid}")
    fig.suptitle(
        f"{Path(args.clip_path).name}  |  clip {args.clip} ({ttype})  |  frames {lo}-{hi}",
        fontsize=15,
    )

    for ax, comp, ci in ((axes[0], "x", 0), (axes[1], "y", 1), (axes[2], "z", 2)):
        for k, _fr in tracks:
            ax.plot(frames, pos[k][:, ci], color=_COL[k], label=k)
            if has_nominals:  # old = dashed, new = dotted (only over blend regions)
                ax.plot(frames, pos_old[k][:, ci], color=_COL[k], ls="--", lw=1.0, alpha=0.7)
                ax.plot(frames, pos_new[k][:, ci], color=_COL[k], ls=":", lw=1.4, alpha=0.7)
        ax.set_ylabel(f"{comp} [m]")
        _mark_seams(ax, frames, stride_w)
        _shade_blend(ax, frames, act)
    axes[0].legend(ncol=3, loc="upper left")
    if has_nominals:  # style legend on the y panel (solid=blended, --=old, :=new)
        from matplotlib.lines import Line2D
        axes[1].legend(handles=[
            Line2D([0], [0], color=_COL["seam"], ls="-", label="blended"),
            Line2D([0], [0], color=_COL["seam"], ls="--", label="old nominal"),
            Line2D([0], [0], color=_COL["seam"], ls=":", label="new nominal"),
        ], ncol=3, loc="upper left")

    for k, _fr in tracks:
        if yaw[k] is not None:
            axes[3].plot(frames, yaw[k], color=_COL[k], label=k)
        if has_nominals and yaw_old[k] is not None:
            axes[3].plot(frames, yaw_old[k], color=_COL[k], ls="--", lw=1.0, alpha=0.7)
            axes[3].plot(frames, yaw_new[k], color=_COL[k], ls=":", lw=1.4, alpha=0.7)
    axes[3].set_ylabel("yaw [deg]")
    _mark_seams(axes[3], frames, stride_w)
    _shade_blend(axes[3], frames, act)

    # Commanded velocity (vx, vy, wz) that drove the stride selection.
    axv = axes[4]
    if tvel is not None:
        tv = tvel[sl]
        for j, (lab, col) in enumerate((("vx [m/s]", _COL["root"]), ("vy [m/s]", _COL["left"]),
                                        ("wz [rad/s]", _COL["right"]))):
            axv.plot(frames, tv[:, j], color=col, label=lab)
        axv.axhline(0.0, color=_COL["seam"], lw=0.8, zorder=0)
        axv.legend(ncol=3, loc="upper left")
    else:
        axv.text(0.5, 0.5, "no target_vel in library", ha="center", va="center",
                 transform=axv.transAxes, color=_COL["right"])
    axv.set_ylabel("cmd vel")
    _mark_seams(axv, frames, stride_w)
    _shade_blend(axv, frames, act)

    ax = axes[5]
    if has_blend:
        alp = blend_alpha[sl]
        ax.plot(frames, alp, color=_COL["alpha"], label=r"blend $\alpha$")
        ax.fill_between(frames, 0, 1, where=act.astype(bool), color=_COL["alpha"], alpha=0.15,
                        step="mid", label="interpolating")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"interp $\alpha$")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "no blend data\n(regenerate with --debug-interp)",
                ha="center", va="center", transform=ax.transAxes, color=_COL["right"])
        ax.set_ylabel(r"interp $\alpha$")
    _mark_seams(ax, frames, stride_w)
    ax.set_xlabel("frame")

    # Annotate the skill at each seam (top panel), if available.
    if skill_id is not None and skills:
        for i in np.nonzero(np.diff(stride_w) != 0)[0] + 1:
            nm = skills[int(skill_id[sl][i])] if int(skill_id[sl][i]) < len(skills) else "?"
            axes[0].annotate(nm, (frames[i], axes[0].get_ylim()[1]), fontsize=8,
                             rotation=90, va="top", ha="right", color=_COL["seam"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out)
        print(f"[plot_clip_frames] wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
