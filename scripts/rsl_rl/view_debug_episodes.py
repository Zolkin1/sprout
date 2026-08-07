#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Viser debugger for recorded episodes: 3D replay + diagnostic plots, no Isaac needed.

Stage 2 of the pair (``debug_episodes.py`` records the bundle). Serves:

**3D scene** -- terrain mesh rebuilt from the clip library's manifest, the robot posed from
its recorded joints, reference ribbons for the pelvis + feet (and the FM's predicted horizon
when the bundle came from an FM-in-the-loop run), plus the robot's ACTUAL frame paths so
reference-vs-actual divergence is visible in space.

**Plots** (also written as PNGs next to the bundle, one set per episode):

1. ``vel``   -- commanded vs achieved velocity (vx, vy, wz)
2. ``scan``  -- height scan AS OBSERVED vs RAW: per-step min/mean/max envelopes plus the
                fraction of rays sitting at the +-1 clip bound. This is the plot that shows
                clipping: a flat-topped observed envelope under a raw one that escapes it.
3. ``ref``   -- reference vs actual height for pelvis + feet, and the 3D tracking error
4. ``scanimg`` -- the raw scan as a heat map over time (rays x steps), which makes missed
                rays and saturated bands obvious at a glance

GUI: play/pause, a step scrubber, playback speed, an episode selector, and per-frame
visibility toggles.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
      python scripts/rsl_rl/view_debug_episodes.py --bundle debug_bundle.npz [--png-only]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from robot_rl.flow_matching.viz.robot import load_urdf, recording_to_urdf_perm, xyzw_to_wxyz
from view_clip_viser import (
    _DEFAULT_URDF,
    _terrain_mesh,
    _tube_mesh,
)

_REF_COLOR = {"pelvis_link": (0.95, 0.55, 0.10),
              "left_ankle_roll_link": (0.20, 0.45, 1.00),
              "right_ankle_roll_link": (0.90, 0.20, 0.25)}
_ACT_COLOR = (0.65, 0.65, 0.65)
_FM_COLOR = (0.15, 0.85, 0.45)
_CLIP_BOUND = 1.0                      # matches the height_scan obs term clip

# Figure sizes, in inches. PNGs are WIDE (good on a monitor); the viser copies are PORTRAIT
# because ``gui.add_image`` has no size control -- it always renders at the fixed GUI panel
# width, so displayed height = panel_width * (h / w). A tall figure is the only way to make
# an in-GUI plot bigger.
_PNG_SIZES = {"vel": (11.0, 7.0), "scan": (11.0, 6.5), "scanimg": (11.0, 4.2),
              "ref": (11.0, 6.5)}
_GUI_ASPECT = {"vel": 1.45, "scan": 1.35, "scanimg": 0.85, "ref": 1.35}


def _gui_sizes(width: float, stretch: float) -> dict[str, tuple[float, float]]:
    """Portrait figure sizes for the viser panel (taller = larger on screen)."""
    return {k: (width, width * a * stretch) for k, a in _GUI_ASPECT.items()}


# --------------------------------------------------------------------------- plots

def _fig_to_rgb(fig) -> np.ndarray:
    """Rasterise a matplotlib figure to an RGB array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return buf


def _plots_for_episode(b: dict, meta: dict, ep: int, png_dir: str | None,
                       sizes: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """Build every diagnostic plot for one episode; optionally save PNGs. Returns RGB arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = float(meta["dt"])
    frames = list(meta["frames"])
    T = b["cmd_vel"].shape[0]
    t = np.arange(T) * dt
    imgs: dict[str, np.ndarray] = {}

    def _save(fig, name):
        imgs[name] = _fig_to_rgb(fig)
        if png_dir:
            p = os.path.join(png_dir, f"ep{ep:02d}_{name}.png")
            fig.savefig(p, dpi=110, bbox_inches="tight")
            print(f"  wrote {p}")
        plt.close(fig)

    # 1. commanded vs achieved velocity
    cmd, ach = b["cmd_vel"][:, ep], b["ach_vel"][:, ep]
    fig, axes = plt.subplots(3, 1, figsize=sizes["vel"], sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ("vx [m/s]", "vy [m/s]", "wz [rad/s]"))):
        a_i = k if k < 2 else 3                      # ach_vel = (vx, vy, vz, wz)
        if cmd.shape[1] > k:
            ax.plot(t, cmd[:, k], lw=1.6, label="commanded (policy obs)")
        ax.plot(t, ach[:, a_i], lw=1.2, alpha=0.85, label="achieved (robot)")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"episode {ep}: commanded vs achieved velocity")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    _save(fig, "vel")

    # 2. height scan: observed vs raw, with clip-saturation fraction
    if "scan_raw" in b and b["scan_raw"].shape[-1] > 1:
        raw, ob = b["scan_raw"][:, ep], b["scan_obs"][:, ep]
        fin = np.isfinite(raw)
        rmin = np.where(fin.any(1), np.nanmin(np.where(fin, raw, np.nan), axis=1), np.nan)
        rmax = np.where(fin.any(1), np.nanmax(np.where(fin, raw, np.nan), axis=1), np.nan)
        rmean = np.where(fin.any(1), np.nanmean(np.where(fin, raw, np.nan), axis=1), np.nan)
        fig, axes = plt.subplots(2, 1, figsize=sizes["scan"], sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        axes[0].fill_between(t, rmin, rmax, alpha=0.25, color="tab:red", label="raw min-max")
        axes[0].plot(t, rmean, lw=1.2, color="tab:red", label="raw mean")
        # On a DEPTH task the policy has no height-scan term at all, so ``scan_obs`` is a
        # width-1 placeholder and there is no "what the policy saw" band to draw -- only the
        # privileged raw sensor below. Plotting it anyway would label a flat line "observed".
        if ob.shape[-1] == raw.shape[-1]:
            axes[0].fill_between(t, ob.min(1), ob.max(1), alpha=0.3, color="tab:blue",
                                label="observed min-max (clipped)")
        for s in (-_CLIP_BOUND, _CLIP_BOUND):
            axes[0].axhline(s, ls="--", lw=1.0, color="k", alpha=0.6)
        axes[0].set_ylabel("ground clearance [m]")
        axes[0].set_title(f"episode {ep}: height scan -- observed (clipped) vs raw sensor")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(alpha=0.3)
        sat = ((~fin) | (np.abs(raw) >= _CLIP_BOUND)).mean(1)
        axes[1].plot(t, 100 * sat, lw=1.3, color="tab:purple", label="at ±1 bound (incl. missed)")
        axes[1].plot(t, 100 * (~fin).mean(1), lw=1.0, color="tab:orange", label="missed rays")
        axes[1].set_ylabel("% rays at bound")
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylim(0, 100)
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.3)
        _save(fig, "scan")

        # 4. raw scan heat map (rays x time)
        fig, ax = plt.subplots(figsize=sizes["scanimg"])
        show = np.where(np.isfinite(raw), raw, np.nan).T
        im = ax.imshow(show, aspect="auto", origin="lower", cmap="viridis",
                       extent=(0.0, float(t[-1]) if T > 1 else dt, 0, show.shape[0]),
                       vmin=np.nanpercentile(show, 1), vmax=np.nanpercentile(show, 99))
        ax.set_title(f"episode {ep}: raw height scan (white = missed ray)")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("ray index")
        fig.colorbar(im, ax=ax, label="clearance [m]")
        _save(fig, "scanimg")

    # 3. reference vs actual, per frame
    ref, act = b["ref_pos"][:, ep], b["body_pos"][:, ep]
    n = min(ref.shape[0 + 1], act.shape[1], len(frames))
    fig, axes = plt.subplots(2, 1, figsize=sizes["ref"], sharex=True)
    for i in range(n):
        c = _REF_COLOR.get(frames[i], (0.5, 0.5, 0.5))
        axes[0].plot(t, ref[:, i, 2], lw=1.5, color=c, label=f"{frames[i]} ref z")
        axes[0].plot(t, act[:, i, 2], lw=1.0, ls="--", color=c, alpha=0.8,
                     label=f"{frames[i]} actual z")
        axes[1].plot(t, np.linalg.norm(ref[:, i] - act[:, i], axis=-1), lw=1.2, color=c,
                     label=f"{frames[i]}")
    axes[0].set_ylabel("world z [m]")
    axes[0].set_title(f"episode {ep}: reference vs actual (height, and 3D error)")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("|ref - actual| [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)
    _save(fig, "ref")
    return imgs


def _ribbon(path: np.ndarray, radius: float):
    """``_tube_mesh`` over the FINITE points of ``path``, or ``None`` if too few remain.

    Every reference/prediction array carries NaN rows for steps where the quantity did not
    exist yet -- step 0 above all, which the recorder samples before the command term has
    ever run. Feeding those to ``_tube_mesh`` would either crash it or, worse, sweep a tube
    through whatever coordinate the NaN degrades to. Filtering keeps the ribbon continuous
    across the gap rather than inventing a segment to it.

    Args:
        path: ``[T, 3]`` centerline, possibly containing non-finite rows.
        radius: Tube radius (m).

    Returns:
        ``(vertices, faces)``, or ``None`` when fewer than 2 finite points survive.
    """
    p = np.asarray(path, np.float64)
    p = p[np.isfinite(p).all(axis=-1)]
    return _tube_mesh(p, radius) if p.shape[0] >= 2 else None


_DEPTH_CAM_COLORS = ((255, 90, 40), (60, 170, 255))     # near = orange, far = blue


def _depth_points(hits: np.ndarray, counts: list[int]):
    """World-frame depth-camera hit points + a per-CAMERA colour, missed rays dropped.

    Coloured by which camera saw the point rather than by range: the two cameras overlap by
    design (near covers the feet, far picks up where near stops), and the question the dots
    answer is which one is carrying a given piece of terrain -- and whether the robot's own
    limbs are occluding it, since these are self-scanning.

    Args:
        hits: ``[sum(counts), 3]`` world-frame hits, cameras concatenated in order.
        counts: ray count per camera, summing to ``hits.shape[0]``.

    Returns:
        ``(points [M, 3] float32, colours [M, 3] uint8)`` with non-finite rays removed.
    """
    cam = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(counts)]) \
        if counts else np.zeros(hits.shape[0], np.int32)
    ok = np.isfinite(hits).all(axis=-1)
    if not ok.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    pts = hits[ok].astype(np.float32)
    col = np.asarray([_DEPTH_CAM_COLORS[c % len(_DEPTH_CAM_COLORS)] for c in cam[ok]], np.uint8)
    return pts, col


def _scan_points(hits: np.ndarray, vals: np.ndarray, vmin: float, vmax: float):
    """World-frame scan hit points + per-point colours, missed rays dropped.

    Mirrors the height-scanner debug markers Isaac Lab / MuJoCo draw, but coloured by ground
    clearance so the dots double as a 3D read of the field the depth image shows.
    """
    from matplotlib import cm
    ok = np.isfinite(hits).all(axis=-1) & np.isfinite(vals)
    if not ok.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    pts = hits[ok].astype(np.float32)
    norm = np.clip((vals[ok] - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    col = (255.0 * cm.get_cmap("viridis")(norm)[:, :3]).astype(np.uint8)
    return pts, col


# --------------------------------------------------------------------------- 3D scene

# The tile-mesh builders now live in ``view_clip_viser`` next to the box / stair geometry
# they call, so this viewer and ``fm_debug_scene`` share one implementation.


# ------------------------------------------------------- live height-scan "depth image"

def _scan_grid(vals: np.ndarray, n_fwd: int, n_lat: int) -> np.ndarray:
    """Reshape a flat scan into a ``[n_fwd, n_lat]`` grid (rows = FORWARD, cols = lateral).

    The ray index varies FASTEST along forward-x -- verified from the recorded world hit
    points, whose consecutive entries step +0.1 m in x with y constant. So the flat layout is
    ``[lateral, forward]`` and a transpose puts forward on the image's vertical axis.
    """
    return np.asarray(vals, dtype=np.float64).reshape(n_lat, n_fwd).T


def _depth_image(img: np.ndarray, vmin: float, vmax: float, upscale: int = 6,
                 missed=(1.0, 0.0, 1.0)) -> np.ndarray:
    """Colour-map one camera's METRIC depth image for the GUI panel.

    Unlike the height scan this needs no reshape -- the sensor already hands back an ``H x W``
    plane in metres. Near is dark, far is bright; non-finite pixels (a ray that hit nothing
    within range) are painted ``missed`` rather than clamped, so an empty pixel is never read
    as a real surface at the far plane.

    Args:
        img: ``[H, W]`` distance-to-image-plane, in metres.
        vmin/vmax: fixed colour limits (per episode) so frames are comparable over time.
        upscale: nearest-neighbour pixel size per cell.
        missed: RGB 0-1 for non-finite pixels.

    Returns:
        ``[H * upscale, W * upscale, 3]`` uint8 image.
    """
    from matplotlib import cm
    fin = np.isfinite(img)
    norm = np.clip((np.where(fin, img, vmin) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = cm.get_cmap("magma")(norm)[..., :3]
    rgb[~fin] = missed
    out = (255.0 * rgb).astype(np.uint8)
    return np.repeat(np.repeat(out, upscale, axis=0), upscale, axis=1)


def _scan_image(vals: np.ndarray, vmin: float, vmax: float, rows: int, cols: int,
                upscale: int = 14, missed=(1.0, 0.0, 1.0)) -> np.ndarray:
    """Colour-map one step's height scan into a depth image.

    The scan is a ``rows x cols`` ground-clearance grid ahead of the robot, so it reads like
    a small depth camera: bright = far below the pelvis, dark = close/above. Missed rays
    (non-finite) are painted ``missed`` (magenta) rather than clamped, so a ray that hit
    nothing is never confused with a real reading.

    Args:
        vals: ``[rows * cols]`` clearance values for this step.
        vmin/vmax: fixed colour limits (per episode) so frames are comparable over time.
        rows/cols: scan grid shape (forward x lateral).
        upscale: nearest-neighbour pixel size per cell.
        missed: RGB 0-1 for non-finite cells.

    Returns:
        ``[rows * upscale, cols * upscale, 3]`` uint8 image, forward pointing UP.
    """
    from matplotlib import cm
    g = _scan_grid(vals, rows, cols)
    fin = np.isfinite(g)
    norm = np.clip((g - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = cm.get_cmap("viridis")(np.where(fin, norm, 0.0))[..., :3]
    rgb[~fin] = np.asarray(missed)
    rgb = np.flipud(rgb)                                    # row 0 = nearest -> draw at bottom
    img = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    return (255.0 * img).astype(np.uint8)


def _clip_mask_image(raw: np.ndarray, bound: float, rows: int, cols: int,
                     upscale: int = 14) -> np.ndarray:
    """Image marking what the +-``bound`` observation clip removes for this step.

    grey = passes through untouched, orange = clamped (real reading the policy never sees),
    magenta = missed ray (mapped to the band edge by ``sanitize_height_scan``).
    """
    g = _scan_grid(raw, rows, cols)
    fin = np.isfinite(g)
    rgb = np.full(g.shape + (3,), 0.25)
    rgb[fin & (np.abs(g) >= bound)] = (1.0, 0.55, 0.0)
    rgb[~fin] = (1.0, 0.0, 1.0)
    rgb = np.flipud(rgb)
    img = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    return (255.0 * img).astype(np.uint8)


def main() -> int:
    """Serve the debug bundle in viser (and write the PNG set)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, help="Bundle .npz from debug_episodes.py.")
    p.add_argument("--urdf", default=_DEFAULT_URDF, help="G1 URDF path.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--png-dir", default=None, help="PNG output dir (default: <bundle>_plots/).")
    p.add_argument("--png-only", action="store_true", help="Write PNGs and exit; no viser server.")
    p.add_argument("--ribbon-radius", type=float, default=0.012)
    p.add_argument("--center", choices=("robot", "cell", "world"), default="robot",
                   help="Per-episode scene origin. Envs sit on a terrain GRID, so in world "
                        "coordinates a robot is hundreds of metres out and the camera opens "
                        "on empty space. 'robot' (default) puts the spawn pose at the origin; "
                        "'cell' puts the env's terrain-cell corner there, which still leaves "
                        "the robot ~40 m out because flat_buffer widens the tile; 'world' "
                        "keeps raw world coordinates. x/y only -- z is never shifted, so "
                        "ground stays at z=0 and heights stay physical.")
    p.add_argument("--scan-grid", type=int, nargs=2, default=(16, 11),
                   metavar=("N_FORWARD", "N_LATERAL"),
                   help="Height-scan grid as (forward, lateral) cell counts. Default 16x11 = "
                        "GridPatternCfg(resolution=0.1, size=[1.5, 1.0]). The ray index varies "
                        "fastest along FORWARD, so the flat scan reshapes as [lateral, forward].")
    p.add_argument("--scan-dot-size", type=float, default=0.018,
                   help="Radius of the 3D height-scan dots.")
    p.add_argument("--scan-cell-px", type=int, default=14, help="Pixels per scan cell.")
    p.add_argument("--gui-plot-width", type=float, default=5.0,
                   help="Width (inches) of the in-GUI plot figures.")
    p.add_argument("--gui-plot-stretch", type=float, default=1.0,
                   help="Extra height multiplier for the in-GUI plots. gui.add_image renders "
                        "at the fixed panel WIDTH, so a taller figure is the only way to make "
                        "the plot appear larger; raise this to enlarge, lower to shrink.")
    args = p.parse_args()

    d = np.load(args.bundle, allow_pickle=True)
    b = {k: d[k] for k in d.files if k != "meta_json"}
    meta = json.loads(str(d["meta_json"]))
    n_ep = int(meta["num_captured"])
    T = int(b["cmd_vel"].shape[0])
    frames = list(meta["frames"])
    print(f"[INFO] {args.bundle}: {n_ep} episodes x {T} steps, dt={meta['dt']}, "
          f"task={meta['task']}, fm={'yes' if meta['is_fm'] else 'no'}")

    png_dir = args.png_dir or (os.path.splitext(args.bundle)[0] + "_plots")
    os.makedirs(png_dir, exist_ok=True)
    gui_sizes = _gui_sizes(args.gui_plot_width, args.gui_plot_stretch)
    all_imgs = []
    for ep in range(n_ep):
        print(f"[INFO] plots for episode {ep}:")
        _plots_for_episode(b, meta, ep, png_dir, _PNG_SIZES)          # wide -> PNG on disk
        if not args.png_only:
            all_imgs.append(_plots_for_episode(b, meta, ep, None, gui_sizes))   # portrait -> GUI
    if args.png_only:
        print(f"[INFO] PNGs in {png_dir}; skipping the viewer (--png-only).")
        return 0

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    urdf = load_urdf(args.urdf)
    perm = recording_to_urdf_perm(list(meta["joint_names"]), urdf)
    robot = ViserUrdf(server, urdf, root_node_name="/robot")

    gui = server.gui
    ep_slider = gui.add_slider("episode", min=0, max=max(n_ep - 1, 0), step=1, initial_value=0)
    step_slider = gui.add_slider("step", min=0, max=T - 1, step=1, initial_value=0)
    playing = gui.add_checkbox("play", initial_value=False)
    speed = gui.add_slider("speed x", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    show_ref = gui.add_checkbox("reference ribbons", initial_value=True)
    show_act = gui.add_checkbox("actual paths", initial_value=True)
    show_fm = gui.add_checkbox("FM horizon", initial_value=bool(meta["is_fm"]))
    # Depth is what the POLICY sees on these tasks; the height scan survives only as a critic
    # input, so its dots default OFF and the camera points default ON.
    has_depth = "depth_hits" in b
    depth_counts = [int(n) for n in meta.get("depth_ray_counts", [])]
    if has_depth and sum(depth_counts) != b["depth_hits"].shape[2]:
        # One colour per camera is only meaningful if the counts partition the concatenation.
        print(f"[WARN] depth_ray_counts {depth_counts} != {b['depth_hits'].shape[2]} rays; "
              "colouring all points as one camera.")
        depth_counts = [b["depth_hits"].shape[2]]
    show_depth3d = gui.add_checkbox("depth-camera points", initial_value=has_depth)
    show_scan3d = gui.add_checkbox("height-scan dots (critic only)", initial_value=False)
    readout = gui.add_markdown("")
    # One plot per TAB: stacked images each get a slice of the panel, tabs give one plot the
    # whole height. Combined with the portrait aspect above, that is as large as viser allows.
    plot_handles: dict[str, object] = {}
    tabs = gui.add_tab_group()
    rows, cols = int(args.scan_grid[0]), int(args.scan_grid[1])
    n_rays = b["scan_raw"].shape[-1] if "scan_raw" in b else 0
    live_ok = n_rays == rows * cols
    # Separate gate: the RAW sensor is always recorded (it is privileged and the critic keeps
    # it), but the policy's own scan term is absent on the depth tasks, where ``scan_obs``
    # collapses to width 1. Only the OBSERVED panel depends on that term existing.
    obs_ok = live_ok and "scan_obs" in b and b["scan_obs"].shape[-1] == rows * cols
    if not live_ok and n_rays:
        print(f"[WARN] {n_rays} rays != {rows}x{cols}={rows * cols}; pass --scan-grid "
              f"to match the sensor pattern. Live depth images disabled.")
    live = {}
    # Both live panels are OFF by default. The height scan is critic-only on these tasks, and
    # the camera images are a lot of panel for something the 3-D points already show -- so the
    # sidebar starts clean and each is a tick away.
    has_dimg = "depth_img" in b
    dcam_names = [str(c) for c in meta.get("depth_cameras", [])]
    show_scan_img = gui.add_checkbox("show height-scan images", initial_value=False)
    show_depth_img = gui.add_checkbox("show depth-camera images", initial_value=False)
    if live_ok or has_dimg:
        with tabs.add_tab("sensors (live)"):
            scan_range = gui.add_markdown("")
            if has_dimg:
                for ci in range(b["depth_img"].shape[2]):
                    nm = dcam_names[ci] if ci < len(dcam_names) else f"camera {ci}"
                    live[f"depth{ci}"] = gui.add_image(
                        np.zeros((8, 8, 3), np.uint8),
                        label=f"{nm} -- metric depth (magenta = no hit)")
            if live_ok:
                live["raw"] = gui.add_image(
                    np.zeros((rows * args.scan_cell_px, cols * args.scan_cell_px, 3), np.uint8),
                    label="RAW scan (magenta = missed ray)")
                if obs_ok:
                    live["obs"] = gui.add_image(
                        np.zeros((rows * args.scan_cell_px, cols * args.scan_cell_px, 3), np.uint8),
                        label="OBSERVED scan (what the policy sees, clipped)")
                live["mask"] = gui.add_image(
                    np.zeros((rows * args.scan_cell_px, cols * args.scan_cell_px, 3), np.uint8),
                    label="clip mask: orange = clamped, magenta = missed")
    for h in live.values():                 # both panels start hidden, matching the checkboxes
        h.visible = False
    for name in all_imgs[0]:
        with tabs.add_tab(name):
            plot_handles[name] = gui.add_image(all_imgs[0][name])

    state = {"ep": -1, "ribbons": [], "terrain": None, "vlim": (0.0, 1.0), "scan3d": None,
             "org": np.zeros(3, np.float64), "n1": 0, "fm3d": None, "depth3d": None,
             "dlim": (0.0, 1.0), "end_kind": ""}

    def _origin(ep: int) -> np.ndarray:
        """The x/y shift subtracted from every world position drawn for episode ``ep``.

        Returned with ``z = 0`` deliberately: the vertical axis is gravity-defined and the
        terrain meshes sit at their true heights, so shifting z would put the ground somewhere
        other than where the height scan and the root-height plots say it is.
        """
        if args.center == "cell":
            o = np.asarray(b["cell_corner"][ep], np.float64).copy()
        elif args.center == "robot":
            o = np.asarray(b["root_pos"][0, ep], np.float64).copy()
        else:
            return np.zeros(3, np.float64)
        o[2] = 0.0
        return o
    has_hits = "scan_hits" in b

    def rebuild(ep: int) -> None:
        """Swap all per-episode scene nodes + plot images."""
        for h in state["ribbons"]:
            h.remove()
        state["ribbons"] = []
        if state["terrain"] is not None:
            state["terrain"].remove()
            state["terrain"] = None
        org = _origin(ep)
        state["org"] = org
        dn = b["done"][:, ep]
        state["n1"] = int(np.argmax(dn)) + 1 if dn.any() else int(b["done"].shape[0])
        # Which of the two ended it. ``done`` alone cannot say: it is terminated OR truncated,
        # so a survivor hitting the episode cap looks identical to a fall. Absent the
        # ``time_out`` array (an older bundle) say so rather than guessing from the step.
        end = state["n1"] - 1
        if not dn.any():
            state["end_kind"] = "no reset within the recording"
        elif "time_out" not in b:
            state["end_kind"] = "ended (bundle has no time_out array -- cause unknown)"
        elif bool(b["time_out"][end, ep]):
            state["end_kind"] = "TIME-OUT -- survived to the episode cap"
        else:
            state["end_kind"] = "TERMINATED -- fell / hit a termination"
        tm = _terrain_mesh(meta["terrain_manifest"], int(b["terrain_id"][ep])) \
            if len(meta["terrain_manifest"]) else None
        if tm is not None:
            v, f = tm
            v = v + b["cell_corner"][ep][None, :] - org     # manifest is tile-local
            state["terrain"] = server.scene.add_mesh_simple(
                "/terrain", vertices=v, faces=f, color=(0.55, 0.55, 0.58), opacity=1.0)
        # Ribbons cover the FIRST episode only. The bundle keeps recording through a
        # termination, and on reset the env teleports the robot to a fresh spawn -- so a
        # ribbon over all T steps welds the pre-fall path to a straight multi-metre jump and
        # then to an entirely different episode. This also matches the survival metric, which
        # is first-episode.
        n1 = state["n1"]
        # reference ribbons + actual paths
        for i, fr in enumerate(frames):
            if i >= b["ref_pos"].shape[1]:
                break
            c = _REF_COLOR.get(fr, (0.5, 0.5, 0.5))
            m = _ribbon(b["ref_pos"][:n1, ep, i] - org, args.ribbon_radius)
            if m is not None:
                state["ribbons"].append(server.scene.add_mesh_simple(
                    f"/ref/{fr}", vertices=m[0], faces=m[1], color=c, visible=show_ref.value))
            m = _ribbon(b["body_pos"][:n1, ep, i] - org, args.ribbon_radius * 0.7)
            if m is not None:
                state["ribbons"].append(server.scene.add_mesh_simple(
                    f"/act/{fr}", vertices=m[0], faces=m[1], color=_ACT_COLOR,
                    visible=show_act.value))
        # NB: the FM horizon is NOT built here -- it is per-STEP, drawn in ``pose``. Built
        # here it showed only the horizon predicted at step 0 and then sat frozen while the
        # step slider moved, which reads as "the FM output is missing" (what you see is the
        # reference ribbon, which for a clip-driven run is indeed just the clip).
        for name, h in plot_handles.items():
            if name in all_imgs[ep]:
                h.image = all_imgs[ep][name]
        msg = []
        if live_ok:
            r = b["scan_raw"][:, ep]
            f = np.isfinite(r)
            lo = float(np.percentile(r[f], 1)) if f.any() else 0.0
            hi = float(np.percentile(r[f], 99)) if f.any() else 1.0
            state["vlim"] = (lo, hi)
            msg.append(f"scan colour scale **{lo:+.2f} .. {hi:+.2f} m**")
        if has_dimg:
            di = b["depth_img"][:n1, ep]
            f = np.isfinite(di)
            dlo = float(np.percentile(di[f], 1)) if f.any() else 0.0
            dhi = float(np.percentile(di[f], 99)) if f.any() else 1.0
            state["dlim"] = (dlo, dhi)
            msg.append(f"depth colour scale **{dlo:.2f} .. {dhi:.2f} m**")
        if msg:
            scan_range.content = "  \n".join(msg) + "  \n(fixed per episode)"
        state["ep"] = ep

    def pose(ep: int, k: int) -> None:
        """Pose the robot at step ``k`` of episode ``ep``."""
        org = state["org"]
        # The generator's predicted window AS OF STEP k -- the thing the FM actually emitted,
        # redrawn every step. It is refreshed only every ``fm_update_every`` steps, so it
        # holds still for K steps and then jumps: that cadence is the point, and a static
        # step-0 ribbon hid it entirely.
        if (meta["is_fm"] and "fm_pred" in b and b["fm_pred"].ndim == 4
                and b["fm_pred"].shape[2] >= 2):
            if state["fm3d"] is not None:
                state["fm3d"].remove()
                state["fm3d"] = None
            m = _ribbon(b["fm_pred"][k, ep] - org, args.ribbon_radius * 1.3)
            if m is not None:
                state["fm3d"] = server.scene.add_mesh_simple(
                    "/fm/horizon", vertices=m[0], faces=m[1], color=_FM_COLOR,
                    visible=show_fm.value)
        robot.update_cfg(b["joint_pos"][k, ep][perm])
        server.scene.add_frame("/robot", show_axes=False,
                               position=tuple(b["root_pos"][k, ep] - org),
                               wxyz=tuple(xyzw_to_wxyz(b["root_quat"][k, ep][None, :])[0]))
        raw = b["scan_raw"][k, ep] if "scan_raw" in b else np.zeros(1)
        if has_hits:
            lo, hi = state["vlim"]
            pts, col = _scan_points(b["scan_hits"][k, ep] - org, raw, lo, hi)
            state["scan3d"] = server.scene.add_point_cloud(
                "/scan_dots", points=pts, colors=col, point_size=args.scan_dot_size,
                point_shape="circle", visible=show_scan3d.value)
        if has_depth:
            if state["depth3d"] is not None:
                state["depth3d"].remove()
            pts, col = _depth_points(b["depth_hits"][k, ep] - org, depth_counts)
            state["depth3d"] = server.scene.add_point_cloud(
                "/depth_dots", points=pts, colors=col, point_size=args.scan_dot_size,
                point_shape="circle", visible=show_depth3d.value)
        fin = np.isfinite(raw)
        sat = 100.0 * float(((~fin) | (np.abs(raw) >= _CLIP_BOUND)).mean())
        # Only recolour a panel that is actually on screen -- these run every step, and the
        # scan images in particular are pure waste on a depth task.
        if live_ok and show_scan_img.value:
            lo, hi = state["vlim"]
            live["raw"].image = _scan_image(raw, lo, hi, rows, cols, args.scan_cell_px)
            if obs_ok:
                live["obs"].image = _scan_image(b["scan_obs"][k, ep], lo, hi, rows, cols,
                                                args.scan_cell_px)
            live["mask"].image = _clip_mask_image(raw, _CLIP_BOUND, rows, cols,
                                                  args.scan_cell_px)
        if has_dimg and show_depth_img.value:
            dlo, dhi = state["dlim"]
            for ci in range(b["depth_img"].shape[2]):
                live[f"depth{ci}"].image = _depth_image(b["depth_img"][k, ep, ci], dlo, dhi)
        readout.content = (
            f"**episode {ep}** step {k}/{T - 1}  t={k * float(meta['dt']):.2f}s  \n"
            f"first episode ends at step **{state['n1'] - 1}**: {state['end_kind']}"
            f"{'  ** past here is a NEW episode after reset **' if k >= state['n1'] else ''}  \n"
            f"terrain_id **{int(b['terrain_id'][ep])}**  \n"
            f"origin (`--center {args.center}`) = "
            f"{np.round(org[:2], 2).tolist()} -- add to read WORLD x/y  \n"
            f"cmd (vx,vy,wz) = {np.round(b['cmd_vel'][k, ep][:3], 3).tolist()}  \n"
            f"achieved (vx,vy,wz) = "
            f"{np.round(b['ach_vel'][k, ep][[0, 1, 3]], 3).tolist()}  \n"
            f"raw scan min/max = {np.nanmin(np.where(fin, raw, np.nan)):+.2f} / "
            f"{np.nanmax(np.where(fin, raw, np.nan)):+.2f} m  \n"
            f"rays at bound = **{sat:.1f}%**")

    @ep_slider.on_update
    def _(_evt) -> None:
        rebuild(int(ep_slider.value))
        pose(int(ep_slider.value), int(step_slider.value))

    @step_slider.on_update
    def _(_evt) -> None:
        pose(int(ep_slider.value), int(step_slider.value))

    @show_scan3d.on_update
    def _(_evt) -> None:
        if state["scan3d"] is not None:
            state["scan3d"].visible = show_scan3d.value

    @show_depth3d.on_update
    def _(_evt) -> None:
        if state["depth3d"] is not None:
            state["depth3d"].visible = show_depth3d.value

    # The image panels are built once and hidden; toggling repaints the current step so a
    # freshly-shown panel is never a stale frame from whenever it was last visible.
    for chk, keys in ((show_scan_img, ("raw", "obs", "mask")),
                      (show_depth_img, tuple(k for k in live if k.startswith("depth")))):
        @chk.on_update
        def _(_evt, chk=chk, keys=keys) -> None:
            for kk in keys:
                if kk in live:
                    live[kk].visible = chk.value
            pose(int(ep_slider.value), int(step_slider.value))

    # ``show_fm`` is NOT in this loop: the FM horizon is redrawn per step and so does not live
    # in ``state["ribbons"]``, which only holds the per-episode meshes.
    @show_fm.on_update
    def _(_evt) -> None:
        if state["fm3d"] is not None:
            state["fm3d"].visible = show_fm.value

    for chk, prefix in ((show_ref, "/ref/"), (show_act, "/act/")):
        @chk.on_update
        def _(_evt, chk=chk, prefix=prefix) -> None:
            for h in state["ribbons"]:
                if h.name.startswith(prefix):
                    h.visible = chk.value

    rebuild(0)
    pose(0, 0)
    print(f"[INFO] viser on http://localhost:{args.port}   (PNGs in {png_dir})")
    dt = float(meta["dt"])
    while True:
        if playing.value:
            nxt = int(step_slider.value) + 1
            step_slider.value = 0 if nxt > T - 1 else nxt
        time.sleep(dt / max(speed.value, 1e-3))
    return 0


if __name__ == "__main__":
    main()
