# Copyright 2026 Zachary Olkin. All rights reserved.

"""Show the same depth frames at several sampling resolutions, side by side.

Choosing a camera resolution from a number is hard to argue about; choosing it by looking at
whether the treads are visible is not. This renders nothing itself -- it reads the outputs of
``augment_registration.py --visual_only --render_resolution WxH``, which re-render ONE
recording's windows at several grids, so every panel in a row is the same instant of the same
motion and the only difference is how finely it was sampled.

Windows are picked where the reference is ABOUT TO CLIMB -- the predicted root rises by more
than a riser over the horizon -- because that is the instant the generator has to decide what
kind of terrain it is looking at, and the instant every failure in this investigation happens.
It needs no tile transform: the climb is read from ``ref_future`` itself.

Depth is drawn in METRES, undoing the contract's ``(d - offset) * scale`` with each file's own
spec rather than a shared constant, so a file rendered under a different contract cannot be
silently plotted on the wrong scale.

    python scripts/rsl_rl/plot_depth_resolution.py \\
        --renders res_30x26.h5 res_40x34.h5 res_60x52.h5 --out depth_resolution.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style  # noqa: E402


def climb_windows(f, n_want: int, riser_m: float = 0.10, flat_frac: float = 0.4) -> np.ndarray:
    """Rows where the reference is ABOUT to climb -- still flat now, rising later.

    "Climbs at some point in the horizon" is the wrong selector: it mostly returns windows
    already on the treads, where the decision has been made and the stairs fill the frame.
    The instant that matters is the approach, so a window qualifies only if the first
    ``flat_frac`` of its horizon is level and the rise comes after it.

    Args:
        f: An open recording.
        n_want: How many rows to return, spread over those that qualify.
        riser_m: Minimum total rise to count as a climb.
        flat_frac: Fraction of the horizon that must still be level.

    Returns:
        Row indices, ascending.
    """
    z = np.asarray(f["ref_future"][:, :, 2])          # root_pos_w z, per node
    k = max(int(flat_frac * z.shape[1]), 1)
    rise = z.max(axis=1) - z[:, 0]
    flat_now = (np.abs(z[:, :k] - z[:, :1]).max(axis=1) < 0.03)
    ok = np.flatnonzero((rise > riser_m) & flat_now)
    if ok.size == 0:
        raise SystemExit(f"no window is level for its first {k} nodes and then climbs "
                         f"{riser_m} m; loosen --riser or the flat fraction.")
    return ok[np.linspace(0, ok.size - 1, min(n_want, ok.size)).astype(int)]


def to_metres(v: np.ndarray, cam: dict) -> np.ndarray:
    """Undo the contract's normalization for one camera.

    Args:
        v: Normalized depth values.
        cam: That camera's entry from ``depth_camera_spec``.

    Returns:
        Depth in metres.
    """
    return v / float(cam["scale"]) + float(cam["offset"])


def centre_profile(img: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """The image's centre column as a terrain cross-section in the CAMERA's frame.

    An image grid shows that something is there; it does not show whether the treads are
    separable. Unprojecting one column does: each pixel becomes a (forward, up) point, so a
    staircase draws as a staircase and the question "can this sampling resolve a 0.209 m
    tread" is answered by looking at whether the steps are steps or a ramp.

    Frame is the camera's own (boresight = forward), so the profile is tilted by whatever the
    torso was doing. That is deliberate -- correcting it needs the camera pose, and the tilt
    does not affect whether the steps are separable.

    Args:
        img: ``(H, W)`` depth in metres.
        cam: That camera's ``depth_camera_spec`` entry.

    Returns:
        ``(forward, up)`` in metres, one point per pixel row.
    """
    h, w = img.shape
    col = img[:, w // 2]
    rows = np.arange(h)
    el = np.arctan((float(cam["cy"]) - rows) / float(cam["fy"]))   # + is up from boresight
    return col * np.cos(el), col * np.sin(el)


def main() -> int:
    """Draw the comparison grid."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renders", nargs="+", required=True,
                   help="Recordings of the SAME windows at different resolutions.")
    p.add_argument("--rows", type=int, default=4, help="Windows to show.")
    p.add_argument("--camera", default="far", choices=("far", "near", "both"))
    p.add_argument("--out", default="depth_resolution.png")
    a = p.parse_args()

    import h5py
    import matplotlib.pyplot as plt
    plot_style.apply_paper_style()

    files = [h5py.File(r, "r") for r in a.renders]
    specs = [json.loads(f.attrs["depth_camera_spec"]) for f in files]
    # The row set comes from the FIRST file and is used for all of them: the renders are the
    # same windows in the same order, so a per-file choice would compare different instants.
    rows = climb_windows(files[0], a.rows)
    cams = [0, 1] if a.camera == "both" else [1 if a.camera == "far" else 0]

    ncol = len(files) * len(cams)
    # One colour range per row, taken over every panel in it, so the resolutions are
    # comparable to each other but each instant is stretched to its own content.
    lo, hi = [], []
    for row in rows:
        vals = [to_metres(np.asarray(f["depth"][row, ci]), sp[ci])
                for f, sp in zip(files, specs) for ci in cams]
        allv = np.concatenate(vals)
        allv = allv[np.isfinite(allv)]
        lo.append(float(np.percentile(allv, 2))); hi.append(float(np.percentile(allv, 98)))

    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(2.1 * ncol, 2.0 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c_i, ci in enumerate(cams):
            for fi, (f, sp) in enumerate(zip(files, specs)):
                cam = sp[ci]
                h, w = int(cam["height"]), int(cam["width"])
                img = to_metres(np.asarray(f["depth"][row, ci]).reshape(h, w), cam)
                ax = axes[r][c_i * len(files) + fi]
                # Scale to THIS ROW's own range, shared across the resolutions in it. A fixed
                # 0..max_dist span puts every ground pixel in one shade -- the structure being
                # judged is centimetres of step against metres of range.
                ax.imshow(img, cmap="viridis", interpolation="nearest",
                          vmin=lo[r], vmax=hi[r])
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{cam['obs_term'].replace('depth_', '')} {w}x{h}",
                                 fontsize=9)
                if c_i * len(files) + fi == 0:
                    ax.set_ylabel(f"row {row}\n{lo[r]:.1f}-{hi[r]:.1f} m", fontsize=8)
    fig.suptitle("Same instants, same lenses, different sampling — depth in metres, "
                 "nearest-neighbour (no smoothing)", fontsize=10)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.splitext(a.out)[0], svg=False)

    # Companion: the centre column as a cross-section, which is where a tread either is or is
    # not separable.
    fig2, ax2 = plt.subplots(len(rows), 1, figsize=(7.0, 2.4 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for f, sp in zip(files, specs):
            ci = cams[-1]
            cam = sp[ci]
            h, w = int(cam["height"]), int(cam["width"])
            img = to_metres(np.asarray(f["depth"][row, ci]).reshape(h, w), cam)
            fwd, up = centre_profile(img, cam)
            ax2[r][0].plot(fwd, up, marker="o", ms=2.5, lw=1.0,
                           label=f"{w}x{h}")
        ax2[r][0].set_title(f"row {row}: centre-column cross-section, "
                            f"{cam['obs_term'].replace('depth_', '')} camera", fontsize=9)
        ax2[r][0].set_xlabel("forward from camera (m)")
        ax2[r][0].set_ylabel("up (m)")
        ax2[r][0].grid(alpha=0.3)
        if r == 0:
            ax2[r][0].legend(title="sampling", fontsize=8)
    fig2.tight_layout()
    plot_style.save_figure(fig2, os.path.splitext(a.out)[0] + "_profile", svg=False)
    print(f"[depth-res] wrote {os.path.splitext(a.out)[0]}_profile.png")

    for f in files:
        f.close()
    print(f"[depth-res] wrote {os.path.splitext(a.out)[0]}.png over {len(rows)} window(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
