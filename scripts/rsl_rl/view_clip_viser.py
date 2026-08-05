#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Interactive viser view of a generated motion clip (no Isaac Sim needed).

Serves a 3D scene in the browser showing, for one clip:

  * every reference frame's world trajectory (pelvis, both feet, both
    wrists, ...) as a coloured polyline;
  * a row of translucent "ghost" robot meshes posed in the clip's joint
    configuration at evenly-spaced points along the trajectory; and
  * one solid robot you can scrub/play through the whole clip with a frame
    slider.

GUI controls let you switch clips, change the ghost count, toggle the frame
trajectories, and play/pause/scrub.  The robot mesh is the G1 URDF posed via
``viser.extras.ViserUrdf``.

Example::

    python scripts/rsl_rl/view_clip_viser.py --lib clips/g1_flat_v1.h5 --clip 0
    # then open the printed http://localhost:8080 URL
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from motion_gen.joint_names import is_root_frame
from motion_gen.clips.clip_io import (
    load_clip_library,
)
from robot_rl.flow_matching.viz.robot import load_urdf, recording_to_urdf_perm, xyzw_to_wxyz
from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.overlay import overlay_sub_boxes

# Distinct colours (RGB 0-1) cycled across reference frames.
_FRAME_COLORS = [
    (0.95, 0.55, 0.10),  # orange  (pelvis, by convention first)
    (0.20, 0.45, 1.00),  # blue
    (0.90, 0.20, 0.25),  # red
    (0.20, 0.80, 0.40),  # green
    (0.80, 0.30, 0.90),  # magenta
    (0.20, 0.80, 0.85),  # cyan
    (0.85, 0.80, 0.20),  # yellow
]
_SCRUBBER_COLOR = (1.0, 1.0, 1.0)       # solid white-ish robot
_GHOST_COLOR = (0.45, 0.55, 0.70)       # translucent steel-blue ghosts
_GHOST_ALPHA_NEAR, _GHOST_ALPHA_FAR = 0.7, 0.18
_KEYFRAME_COLOR = (0.95, 0.80, 0.15)    # gold -- pinned target keyframes (Phase-2 mb_encode overlay)
_KEYFRAME_ALPHA = 0.55
# A TERRAIN keyframe is drawn red because it is only half a pin: the replay's pose is honoured, but its
# root xy is thrown away and replaced by the command's exact-velocity rollout (see
# mb_env.target_pose_override / full_agent.py's target_root_realignment branch).  So a red ghost's POSE is
# what Motion Bricks was asked to reach, while its PLACE is wherever the commanded velocity happened to
# put it -- not where the terrain is.  Gold ghosts (our steady stride) are pose-only by design.
_KEYFRAME_TERRAIN_COLOR = (0.90, 0.25, 0.20)
# The 29-DOF asset, matching what is trained (see CLAUDE.md: g1_21j_box_merged_29j_hands).  The old
# 21-DOF default actuates neither waist roll/pitch nor any of the 6 wrists, so posing a 29-joint clip
# with it silently FROZE those 8 joints at zero -- the motion looked subtly wrong (stiff torso) in a way
# that was easy to blame on the clip rather than the viewer.
_DEFAULT_URDF = "robot_assets/g1/g1_21j/g1_21j_box_merged_29j_hands.urdf"


def _load_keyframe_sidecar(path: str, urdf) -> dict[int, dict]:
    """Load an ``mb_encode.py`` ``_keyframes.npz`` sidecar -> per-clip posed-keyframe arrays.

    Returns ``{clip_idx: {"pos": [K,3], "wxyz": [K,4], "cfg": [K,nj], "t": [K]}}`` where ``cfg`` is
    already permuted into the URDF joint order. Keyframe qpos is MuJoCo layout
    ``[root_pos(3), root_quat_wxyz(4), joints(29)]`` with joints in the sidecar's ``joint_names`` order.
    """
    import json as _json

    d = np.load(path, allow_pickle=False)
    clip_idx, qpos = d["clip_idx"], d["qpos"]
    tks = d["t"] if "t" in d else np.zeros(len(qpos), np.float32)
    kinds = np.array(_json.loads(str(d["kind"]))) if "kind" in d else np.array(["stride"] * len(qpos))
    joint_names = _json.loads(str(d["joint_names"]))
    perm = recording_to_urdf_perm(joint_names, urdf)
    out: dict[int, dict] = {}
    for ci in np.unique(clip_idx):
        m = clip_idx == ci
        out[int(ci)] = {
            "pos": qpos[m, 0:3].astype(np.float32),
            "wxyz": qpos[m, 3:7].astype(np.float32),
            "cfg": qpos[m, 7:][:, perm].astype(np.float32),
            "t": tks[m].astype(np.float32),
            "kind": kinds[m],
        }
    return out


def _frame_pos_indices(pos_names: list[str]) -> dict[str, list[int]]:
    """Map each WORLD frame name -> its [pos_x, pos_y, pos_z] indices in ``pos_names``.

    Root-frame (``@root``) channels are skipped: they are pelvis-relative, so plotting them
    in the world scene would draw a cluster of meaningless points around the origin.
    """
    frames: dict[str, list[int]] = {}
    for axis_i, name in enumerate(pos_names):
        if ":pos_" in name and not is_root_frame(name):
            frame, axis = name.split(":pos_")
            frames.setdefault(frame, [None, None, None])
            frames[frame]["xyz".index(axis)] = axis_i
    return {f: idx for f, idx in frames.items() if None not in idx}


def _frame_ori_indices(pos_names: list[str], frame: str) -> list[int] | None:
    """Indices of ``frame:ori`` in ``pos_names`` as xyzw (Isaac Lab convention),
    or None.  The YAML stores ori scalar-first (ori_w), so gather [x, y, z, w]."""
    try:
        return [pos_names.index(f"{frame}:ori_{a}") for a in ("x", "y", "z", "w")]
    except ValueError:
        return None


def _joint_info(pos_names: list[str]) -> tuple[list[int], list[str]]:
    """Return (indices into pos outputs, joint names) for the ``joint:*`` outputs."""
    idx, names = [], []
    for i, n in enumerate(pos_names):
        if n.startswith("joint:"):
            idx.append(i)
            names.append(n.split(":", 1)[1])
    return idx, names


def _tube_mesh(path: np.ndarray, radius: float, n_seg: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Build a tube mesh of given radius swept along a 3D polyline.

    Used to visualise the base-pose-error termination corridor: the actual
    base must stay within ``radius`` of the reference base path, so the swept
    radius-``radius`` tube is the allowed region before termination.

    Args:
        path: ``[T, 3]`` centerline points.
        radius: Tube radius (m).
        n_seg: Cross-section ring resolution.

    Returns:
        ``(vertices [T*n_seg, 3], faces [M, 3])``.
    """
    p = np.asarray(path, dtype=np.float64)
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    t /= np.clip(np.linalg.norm(t, axis=1, keepdims=True), 1e-9, None)

    # Stable per-point frame: right = tangent x up (fall back to x if vertical).
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(t, up)
    bad = np.linalg.norm(right, axis=1) < 1e-6
    right[bad] = np.cross(t[bad], np.array([1.0, 0.0, 0.0]))
    right /= np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-9, None)
    newup = np.cross(right, t)
    newup /= np.clip(np.linalg.norm(newup, axis=1, keepdims=True), 1e-9, None)

    ang = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    verts = (
        p[:, None, :]
        + radius * (cos[None, :, None] * right[:, None, :] + sin[None, :, None] * newup[:, None, :])
    ).reshape(-1, 3)

    nt = len(p)
    ti = np.arange(nt - 1)[:, None]
    ki = np.arange(n_seg)[None, :]
    a = ti * n_seg + ki
    b = ti * n_seg + (ki + 1) % n_seg
    c = (ti + 1) * n_seg + ki
    d = (ti + 1) * n_seg + (ki + 1) % n_seg
    f1 = np.stack([a, b, d], axis=-1).reshape(-1, 3)
    f2 = np.stack([a, d, c], axis=-1).reshape(-1, 3)
    return verts.astype(np.float32), np.concatenate([f1, f2], axis=0).astype(np.uint32)


def _box(xmin, xmax, ymin, ymax, zmin, zmax) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box as (8 vertices, 12 triangle faces)."""
    v = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # -y
        [2, 3, 7], [2, 7, 6],  # +y
        [1, 2, 6], [1, 6, 5],  # +x
        [3, 0, 4], [3, 4, 7],  # -x
    ], dtype=np.uint32)
    return v, f


def _ramp(xmin, xmax, ymin, ymax, base_z, z_at_xmin, z_at_xmax) -> tuple[np.ndarray, np.ndarray]:
    """Prism with a flat bottom (``base_z``) and a top tilted in x (a uniform slope)."""
    v = np.array([
        [xmin, ymin, base_z], [xmax, ymin, base_z], [xmax, ymax, base_z], [xmin, ymax, base_z],
        [xmin, ymin, z_at_xmin], [xmax, ymin, z_at_xmax], [xmax, ymax, z_at_xmax], [xmin, ymax, z_at_xmin],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
    ], dtype=np.uint32)
    return v, f


def _stair_decor_boxes(b: dict, dims: dict, symin: float, symax: float) -> list[tuple]:
    """Axis-aligned boxes for one stair band's decoration (walls / posts).

    Mirrors the shared builder ``trimesh/stair.py`` (``_add_wall_one_side`` /
    ``_add_posts_one_side``) so the viser view matches the sim collider: walls / posts
    span ``[min_top - h, max_top + side.height]`` in z, sit just outside the tread
    edge (``symin`` / ``symax``) in y, and posts march uniformly along x.
    """
    decor = b.get("decor") or {}
    centers = np.array(b["stair_top_centers"], dtype=float)
    if centers.size == 0:
        return []
    x0, x1 = float(b["aabb"][0]), float(b["aabb"][1])
    h = float(dims["h"])
    z_bot = float(centers[:, 2].min()) - h
    boxes: list[tuple] = []
    for side_key, is_left in (("left", True), ("right", False)):
        side = decor.get(side_key) or {}
        kind = side.get("kind", "none")
        if kind == "none":
            continue
        th = float(side.get("thickness", 0.0))
        z_top = float(centers[:, 2].max()) + float(side.get("height", 0.0))
        y_lo, y_hi = (symin - th, symin) if is_left else (symax, symax + th)
        if kind == "wall":
            boxes.append((x0, x1, y_lo, y_hi, z_bot, z_top))
        elif kind == "posts" and th > 1e-5:
            sp = float(side.get("spacing", 0.0))
            cur = 0.0
            while cur + th <= (x1 - x0) + 1e-5:
                boxes.append((x0 + cur, x0 + cur + th, y_lo, y_hi, z_bot, z_top))
                cur += th + sp
    return boxes


def _stair_terrain_mesh(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build a solid multi-band terrain mesh (verts, faces) from a manifest entry.

    Flat bands render as a full-corridor-width slab at their level; stair bands
    render as solid tread pillars (or thin floating slabs when ``decor.float_thickness
    > 0``) spanning the (narrower, centered) stair width, plus any per-side wall / post
    decoration; slope bands render as a tilted ramp (0 -> ``rise`` over the band span).
    All in the clip-world (local-corner) frame the clip is stored in.  Falls back to the
    legacy single-stair keys for pre-multi-band manifests.
    """
    size_y = float(m.get("size_y", 1.2))
    w = float(m["dims"]["w"])
    cy = size_y / 2.0
    symin, symax = cy - w / 2.0, cy + w / 2.0
    base_z = -0.08
    boxes, ramps, noses = [], [], []
    bands = m.get("bands")
    if bands is None:  # legacy single-stair manifest
        bands = [{"block_type": "StairBlock", "aabb": [m["span_xmin"], m["span_xmax"], 0.0, size_y],
                  "exit_z": m["exit_z"], "stair_top_centers": m["stair_top_centers"]}]
    for b in bands:
        x0, x1 = float(b["aabb"][0]), float(b["aabb"][1])
        if b["block_type"] == "StairBlock":
            d, h = float(m["dims"]["d"]), float(m["dims"]["h"])
            decor = b.get("decor") or {}
            ft = float(decor.get("float_thickness", 0.0))
            nr = min(float(decor.get("nosing_radius", 0.0)), 0.5 * min(h, d))
            nseg = int(decor.get("nosing_segments", 6))
            for cx, _cy, cz in np.array(b["stair_top_centers"], dtype=float):  # treads
                ztop = float(cz)
                zbot = (ztop - ft) if ft > 0.0 else base_z   # thin floating slab vs solid pillar
                boxes.append((cx - d / 2.0, cx + d / 2.0, symin, symax, zbot, ztop))
                if nr > 1e-5:  # rounded lip over the (up-stair) leading -x edge
                    noses.append((cx - d / 2.0, symin, symax, ztop - nr, nr, nseg))
            boxes.extend(_stair_decor_boxes(b, m["dims"], symin, symax))
        elif b["block_type"] == "SlopeBlock":  # tilted ramp, full width, 0 -> rise over the span
            rise = float(b.get("rise", m["dims"].get("rise", 0.0)))
            ramps.append((x0, x1, symin, symax, base_z, 0.0, rise))
        else:  # flat slab, full corridor width, at the band's level
            boxes.append((x0, x1, 0.0, size_y, base_z, float(b["exit_z"])))

    # Accumulate every piece generically (boxes/ramps are 8-vert, nosing tubes vary).
    pieces = [_box(*b) for b in boxes] + [_ramp(*r) for r in ramps]
    for front_x, y_lo, y_hi, cz, rad, seg in noses:
        path = np.array([[front_x, y_lo, cz], [front_x, y_hi, cz]], dtype=float)
        pieces.append(_tube_mesh(path, rad, max(6, 2 * seg)))
    verts, faces, off = [], [], 0
    for v, f in pieces:
        verts.append(v); faces.append(f + off); off += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def _box_terrain_mesh(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flat ground slab + overlay volume(s) for a ``box`` manifest entry -> (verts, faces).

    The clip is baked in its local-corner frame, so the ground spans ``[0,size_x] x [0,size_y]``
    and each overlay sits at its ``pos_xy`` (local) -- the same frame the clip's world base path
    is drawn in, so the box lines up under the jump.  Support-structure variants (floating top,
    center / corner legs) are expanded via the shared :func:`overlay_sub_boxes` so the viser view
    matches the sim collider.
    """
    size_x, size_y = float(m["size_x"]), float(m["size_y"])
    ovs = m.get("overlays", [])
    # A stair/descent tile has treads going BELOW z=0; the treads are the ground, so skip the flat
    # z=0 slab (it would slice through the descent). Flat-based tiles (box jump) keep their slab.
    below = any(float(ov.get("base_z", 0.0)) < -0.05 for ov in ovs)
    boxes = [] if below else [(0.0, size_x, 0.0, size_y, -0.08, 0.0)]    # ground slab (flat-based only)
    for ov in ovs:
        for cx, cy2, cz, sx, sy, sz in overlay_sub_boxes(
            str(ov.get("variant", "solid")),
            tuple(float(v) for v in ov["size_xyz"]),
            (float(ov["pos_xy"][0]), float(ov["pos_xy"][1])),
            float(ov.get("base_z", 0.0)),
            float(ov.get("slab_thickness", 0.1)),
            tuple(ov.get("leg_size", (0.1, 0.1))),
            float(ov.get("leg_inset", 0.05)),
        ):
            boxes.append((cx - sx / 2.0, cx + sx / 2.0, cy2 - sy / 2.0, cy2 + sy / 2.0,
                          cz - sz / 2.0, cz + sz / 2.0))
    verts, faces, off = [], [], 0
    for b in boxes:
        v, f = _box(*b)
        verts.append(v); faces.append(f + off); off += 8
    return np.concatenate(verts), np.concatenate(faces)


def _overlay_mesh(m: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a tile mesh from the OVERLAY schema (the one MB / current libraries write).

    Each overlay is an axis-aligned box: ``pos_xy`` is its CENTRE in tile-local coords,
    ``size_xyz`` its full extent, ``base_z`` the bottom face (so the top surface sits at
    ``base_z + size_xyz[2]`` -- that identity is what matched the raycaster's mesh z-max
    exactly, which is how the schema was confirmed).

    :func:`_stair_terrain_mesh` handles the LEGACY ``dims`` schema and raises
    ``KeyError('dims')`` on these, so this is the fallback path.
    """
    verts, faces, n = [], [], 0
    for ov in m.get("overlays", []):
        (px, py), (sx, sy, sz) = ov["pos_xy"], ov["size_xyz"]
        bz = float(ov.get("base_z", 0.0))
        v, f = _box(px - sx / 2.0, px + sx / 2.0, py - sy / 2.0, py + sy / 2.0, bz, bz + sz)
        verts.append(np.asarray(v))
        faces.append(np.asarray(f) + n)
        n += len(v)
    if not verts:
        return None
    return np.concatenate(verts, 0), np.concatenate(faces, 0)


def _terrain_mesh(manifest: list[dict], tid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild ONE tile's mesh from a clip library's terrain manifest.

    The entry point every offline viewer uses to put a recorded rollout on its terrain:
    tries the overlay schema first, then the legacy ``dims`` one. Returns ``None`` (with a
    warning) rather than raising, so a viewer still shows the motion when a tile's geometry
    cannot be rebuilt.

    Args:
        manifest: The library's ``terrain_manifest`` attribute, already parsed.
        tid: The tile's ``terrain_id``; ``-1`` (flat) has no entry and yields ``None``.

    Returns:
        ``(vertices, faces)`` in TILE-LOCAL coordinates -- add the tile corner -- or ``None``.
    """
    for m in manifest:
        if int(m.get("terrain_id", -2)) != int(tid):
            continue
        if m.get("overlays"):
            try:
                return _overlay_mesh(m)
            except Exception as e:                                   # noqa: BLE001
                print(f"[WARN] overlay mesh for tid={tid}: {e}")
        try:                                                          # legacy 'dims' schema
            if m.get("type") in ("terrain", "transition"):
                return _stair_terrain_mesh(m)
            return _box_terrain_mesh(m)
        except Exception as e:                                       # noqa: BLE001
            print(f"[WARN] terrain mesh for tid={tid}: {e}")
            return None
    return None


def _central_diff(a: np.ndarray, dt: float) -> np.ndarray:
    d = np.zeros_like(a)
    if a.shape[0] > 2:
        d[1:-1] = (a[2:] - a[:-2]) / (2.0 * dt)
        d[0] = (a[1] - a[0]) / dt
        d[-1] = (a[-1] - a[-2]) / dt
    return d


def _contact_image(contact: np.ndarray, foot_z: np.ndarray, foot_speed: np.ndarray,
                   names: list, dt: float):
    """Render the per-foot contact SCHEDULE as a timeline, over foot height and foot speed.

    Three rows that have to be read together to judge a stride: the contact raster (when the clip says
    a foot is planted), the foot's height, and its horizontal speed.  A planted foot should be flat on
    both of the latter for the whole shaded span -- any ramp there IS the skate, and any shading while
    the foot is high is the schedule disagreeing with the motion.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    n = contact.shape[0]
    t = np.arange(n) * dt
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.6, 1, 1]})
    cols = [(0.85, 0.25, 0.2), (0.2, 0.45, 0.85)]
    for k in range(contact.shape[1]):
        c = contact[:, k].astype(bool)
        axes[0].fill_between(t, k, k + 0.8, where=c, color=cols[k % 2], step="mid", lw=0)
        axes[1].plot(t, foot_z[:, k], color=cols[k % 2], lw=1.2, label=names[k])
        axes[2].plot(t, foot_speed[:, k], color=cols[k % 2], lw=1.2)
        for ax in axes[1:]:
            ax.fill_between(t, *ax.get_ylim(), where=c, color=cols[k % 2], alpha=0.10, step="mid", lw=0)
    axes[0].set_yticks([0.4, 1.4][:contact.shape[1]])
    axes[0].set_yticklabels([nm.replace("_ankle_roll_link", "") for nm in names], fontsize=7)
    axes[0].set_ylim(0, contact.shape[1]); axes[0].set_title("contact schedule", fontsize=8)
    axes[1].set_ylabel("foot z (m)", fontsize=8)
    axes[2].set_ylabel("foot speed (m/s)", fontsize=8); axes[2].set_xlabel("t (s)", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    axes[1].legend(fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def _tracking_image(base_pos: np.ndarray, base_quat_wxyz: np.ndarray,
                    target_vel: np.ndarray, dt: float):
    """Render a body-frame root-velocity-vs-target tracking plot to an RGB array.

    Actual: world pelvis velocity rotated into the yaw (body) frame -> (vx fwd, vy left),
    plus yaw rate.  Target: the stored command ``target_vel`` (vx, vy, wz).  Returns
    ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    w, x, y, z = base_quat_wxyz[:, 0], base_quat_wxyz[:, 1], base_quat_wxyz[:, 2], base_quat_wxyz[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    vw = _central_diff(base_pos[:, :2], dt)                      # world xy velocity
    c, s = np.cos(yaw), np.sin(yaw)
    vx_b = c * vw[:, 0] + s * vw[:, 1]                           # forward
    vy_b = -s * vw[:, 0] + c * vw[:, 1]                          # left
    wz = _central_diff(np.unwrap(yaw)[:, None], dt)[:, 0]        # yaw rate
    T = base_pos.shape[0]
    t = np.arange(T) * dt
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 4.4), dpi=100, sharex=True)
    rows = [(axs[0], vx_b, target_vel[:, 0], "vx fwd"),
            (axs[1], vy_b, target_vel[:, 1], "vy left"),
            (axs[2], wz, target_vel[:, 2], "wz yaw")]
    for ax, act, tgt, lab in rows:
        ax.plot(t, act, "k", lw=1.3, label="actual")
        ax.plot(t, tgt, "--", color="#d62728", lw=1.2, label="target")
        ax.set_ylabel(lab, fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    axs[0].set_title("body-frame root velocity vs target", fontsize=9)
    axs[0].legend(fontsize=7, loc="upper right", ncol=2)
    axs[2].set_xlabel("t (s)", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    h, wd = fig.canvas.get_width_height()[::-1]
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, wd, 4)[..., :3].copy()
    plt.close(fig)
    return img


class ClipScene:
    """Builds and updates all viser nodes for one clip at a time."""

    def __init__(self, server: viser.ViserServer, lib, urdf, max_ghosts: int, term_dist: float = 0.25):
        self.server = server
        self.lib = lib
        self.urdf = urdf
        self.max_ghosts = max_ghosts
        self.term_dist = term_dist
        self.num_steps = lib.num_steps

        # Per-clip terrain manifest (empty for flat-only / pre-terrain libraries).
        self.terrain_manifest = {
            int(m["terrain_id"]): m for m in lib.attrs.get("terrain_manifest", [])
        }
        self.clip_terrain_id = lib.terrain_id
        self._terrain_handle: viser.SceneNodeHandle | None = None
        self._grid_handle: viser.SceneNodeHandle | None = None
        self._spawngoal_handle: viser.SceneNodeHandle | None = None

        self.pos_names = list(lib.meta["pos_output_names"])
        self.contact_frames = list(lib.meta["contact_frames"])
        self.base_frame = lib.meta.get("ref_frame_base", "pelvis_link")
        if f"{self.base_frame}:pos_x" not in self.pos_names:
            self.base_frame = "pelvis_link"

        self.frame_pos = _frame_pos_indices(self.pos_names)
        self.base_pos_idx = self.frame_pos[self.base_frame]
        self.base_ori_idx = _frame_ori_indices(self.pos_names, self.base_frame)
        self.joint_idx, self.joint_names = _joint_info(self.pos_names)
        self.joint_perm = recording_to_urdf_perm(self.joint_names, urdf)

        # Static frame-trajectory polylines (one handle per frame, reused).
        self._track_handles: dict[str, viser.SceneNodeHandle] = {}
        # Robot mesh pool: solid scrubber + translucent ghosts.
        self._scrub_frame = server.scene.add_frame("/scrubber", show_axes=False)
        self._scrub = ViserUrdf(
            server, urdf, root_node_name="/scrubber",
            mesh_color_override=(*_SCRUBBER_COLOR, 1.0),
        )
        self._ghosts: list[tuple[viser.FrameHandle, ViserUrdf]] = []
        for s in range(max_ghosts):
            frac = s / max(max_ghosts - 1, 1)
            alpha = _GHOST_ALPHA_NEAR + (_GHOST_ALPHA_FAR - _GHOST_ALPHA_NEAR) * frac
            fr = server.scene.add_frame(f"/ghosts/{s}", show_axes=False, visible=False)
            gh = ViserUrdf(
                server, urdf, root_node_name=f"/ghosts/{s}",
                mesh_color_override=(*_GHOST_COLOR, alpha),
            )
            self._ghosts.append((fr, gh))

        # Pinned-keyframe ghost pool (gold) -- the Phase-2 target poses fed to Motion Bricks' predict().
        # Populated only when a --keyframes sidecar is supplied; sized on first set_keyframes.
        self._keyframes: dict[int, dict] | None = None
        self._kf_max = 12                 # pool cap; see _ensure_kf_pool
        # One entry per keyframe slot: {"stride": (frame, urdf), "terrain": (frame, urdf)}.
        self._kf_ghosts: list[dict[str, tuple[viser.FrameHandle, ViserUrdf]]] = []
        self._kf_markers: viser.SceneNodeHandle | None = None
        self.show_keyframes = False   # default off -- many robot meshes can hang the browser; toggle via GUI

        # Termination-distance corridor tube around the base path (hidden by default).
        self._tube_handle: viser.SceneNodeHandle | None = None

        # Cached per-clip arrays (filled by load_clip).
        self._world_base_pos = None   # [T, 3]
        self._world_base_quat = None  # [T, 4] wxyz
        self._joint_cfg = None        # [T, nj] (urdf order)
        self._target_vel = None       # [T, 3] sampled (vx, vy, wz), or None if absent
        self.show_tracks = True
        self.show_tube = False
        self.show_command = True
        self._cmd_handle = None       # command-reference path polyline
        self.show_plot = True
        self._plot_handle = None      # body-frame velocity-tracking plot (GUI image)
        self._contact_handle = None   # per-foot contact-schedule timeline (GUI image)
        self.num_ghosts = min(max_ghosts, max_ghosts)
        self.clip_idx = -1

    # ------------------------------------------------------------------

    def _world_track(self, y_des, idx) -> np.ndarray:
        """Clip-world [T,3] track for the frame whose pos outputs are at ``idx``.

        Clips are stored clip-world, so the track is read straight from ``y_des``.
        """
        return y_des[:, idx].cpu().numpy()

    def load_clip(self, clip_idx: int) -> None:
        """Reconstruct world tracks + robot poses for ``clip_idx`` and redraw."""
        self.clip_idx = clip_idx
        # Respect the clip's OWN length: clips are stored padded to the library's max T (a rectangular
        # h5 array), the tail being held-static frames.  The RL uses clip_len (reset samples within it,
        # termination cuts at clip_len-1), so the pad is inert there; the viewer must likewise stop at
        # clip_len or it shows a long frozen tail after the real motion.
        cl = self.lib.clip_len
        y = self.lib.y_des[clip_idx]                 # [T, P] clip-world (padded)
        n = int(cl[clip_idx]) if (cl is not None and cl.numel()) else y.shape[0]
        self.num_steps = max(1, min(n, y.shape[0]))  # per-clip real length (drives slider / play / ghosts)
        y = y[: self.num_steps]
        skill = self.lib.skill_id[clip_idx].cpu().numpy()[: self.num_steps]
        contact = self.lib.contact[clip_idx].cpu().numpy()[: self.num_steps]

        # Robot base world pose + joint config for every step.  The stored
        # quaternion is xyzw (Isaac Lab); viser frames want wxyz.
        self._world_base_pos = y[:, self.base_pos_idx].cpu().numpy()
        self._world_base_quat = xyzw_to_wxyz(y[:, self.base_ori_idx].cpu().numpy())
        self._joint_cfg = y[:, self.joint_idx].cpu().numpy()[:, self.joint_perm]

        # Sampled (target) velocity per step, if the library stored it.
        tv = self.lib.target_vel
        self._target_vel = tv[clip_idx].cpu().numpy()[: self.num_steps] if tv.numel() > 0 else None

        # Terrain mesh for this clip (flat clips keep the bare grid).
        for h in (self._terrain_handle, self._grid_handle, self._spawngoal_handle):
            if h is not None:
                h.remove()
        self._terrain_handle = self._grid_handle = self._spawngoal_handle = None
        tid = int(self.clip_terrain_id[clip_idx]) if self.clip_terrain_id.numel() else -1
        if tid in self.terrain_manifest:
            mm = self.terrain_manifest[tid]
            tv_v, tv_f = (_box_terrain_mesh(mm) if (mm.get("type") == "box" or "overlays" in mm)
                          else _stair_terrain_mesh(mm))
            self._terrain_handle = self.server.scene.add_mesh_simple(
                "/terrain", vertices=tv_v, faces=tv_f,
                color=(0.62, 0.62, 0.66), opacity=1.0, side="double", flat_shading=True,
            )
            # Ground grid sized to the terrain chunk. For a stair/descent tile (treads below z=0),
            # drop the grid to the geometry floor instead of drawing a misleading z=0 plane.
            sx, sy = float(mm["size_x"]), float(mm["size_y"])
            _bzs = [float(ov.get("base_z", 0.0)) for ov in mm.get("overlays", [])]
            gz = min(_bzs) if (_bzs and min(_bzs) < -0.05) else 0.0
            self._grid_handle = self.server.scene.add_grid(
                "/grid", width=sx, height=sy, cell_size=1.0, position=(sx / 2.0, sy / 2.0, gz),
            )
            # Spawn (green) + nav goal (red) markers.
            pts, cols = [], []
            sp = mm.get("spawn_anchor")
            if sp is not None:
                pts.append([sp[0], sp[1], 0.05]); cols.append((0.1, 0.9, 0.1))
            goal = mm.get("nav_goal")
            if goal is not None:
                pts.append([goal[0], goal[1], 0.05]); cols.append((0.9, 0.1, 0.1))
            if pts:
                self._spawngoal_handle = self.server.scene.add_point_cloud(
                    "/spawn_goal", points=np.array(pts, np.float32),
                    colors=np.array(cols, np.float32), point_size=0.15, point_shape="circle",
                )
        else:
            # Flat clip: grid sized to the base-path bounding box + margin.
            bp = self._world_base_pos
            lo = bp[:, :2].min(axis=0) - 3.0
            hi = bp[:, :2].max(axis=0) + 3.0
            wx, wy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
            self._grid_handle = self.server.scene.add_grid(
                "/grid", width=wx, height=wy, cell_size=1.0,
                position=(float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0), 0.0),
            )

        # Termination-distance corridor: tube of radius max_pos_dist around the base path.
        if self._tube_handle is not None:
            self._tube_handle.remove()
        tube_v, tube_f = _tube_mesh(self._world_base_pos, self.term_dist)
        self._tube_handle = self.server.scene.add_mesh_simple(
            "/term_tube", vertices=tube_v, faces=tube_f,
            color=(0.95, 0.25, 0.25), opacity=0.16, side="double",
            flat_shading=False, visible=self.show_tube,
        )

        # Frame trajectory polylines.
        for fi, (frame, idx) in enumerate(self.frame_pos.items()):
            track = self._world_track(y, idx)                       # [T, 3]
            segs = np.stack([track[:-1], track[1:]], axis=1)        # [T-1, 2, 3]
            color = _FRAME_COLORS[fi % len(_FRAME_COLORS)]
            name = f"/tracks/{frame}"
            if frame in self._track_handles:
                self._track_handles[frame].points = segs.astype(np.float32)
            else:
                self._track_handles[frame] = self.server.scene.add_line_segments(
                    name, points=segs.astype(np.float32), colors=color,
                    line_width=4.0 if frame == self.base_frame else 2.5,
                    visible=self.show_tracks,
                )

        # Command reference path: integrate the stored target_vel (vx, vy, wz) in the
        # world frame from the clip's start pose -- where the base "should" go if it
        # followed the velocity command open-loop.  Drawn at pelvis height so the gap to
        # the base track (orange) IS the motion-matching tracking error.
        if self._cmd_handle is not None:
            self._cmd_handle.remove()
            self._cmd_handle = None
        if self._target_vel is not None:
            dt = float(self.lib.meta["dt"])
            bp = self._world_base_pos
            q0 = self._world_base_quat[0]                       # wxyz
            yaw = float(np.arctan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]),
                                   1 - 2 * (q0[2] ** 2 + q0[3] ** 2)))
            T = bp.shape[0]
            ref = np.zeros((T, 3), np.float32)
            x, yy = float(bp[0, 0]), float(bp[0, 1])
            for t in range(T):
                ref[t] = (x, yy, float(bp[t, 2]))
                vx, vy, wz = (float(v) for v in self._target_vel[t])
                x += (np.cos(yaw) * vx - np.sin(yaw) * vy) * dt
                yy += (np.sin(yaw) * vx + np.cos(yaw) * vy) * dt
                yaw += wz * dt
            csegs = np.stack([ref[:-1], ref[1:]], axis=1).astype(np.float32)
            self._cmd_handle = self.server.scene.add_line_segments(
                "/command_path", points=csegs, colors=(0.1, 0.95, 0.2),
                line_width=6.0, visible=self.show_command,
            )

        # Body-frame root-velocity-vs-target tracking plot, embedded in the GUI panel.
        if self._plot_handle is not None:
            self._plot_handle.remove()
            self._plot_handle = None
        if self._target_vel is not None:
            img = _tracking_image(self._world_base_pos, self._world_base_quat,
                                  self._target_vel, float(self.lib.meta["dt"]))
            if img is not None:
                self._plot_handle = self.server.gui.add_image(
                    img, label="tracking: body vel vs target", visible=self.show_plot)

        # Contact schedule as a timeline, beside the foot height and speed it should agree with.
        if self._contact_handle is not None:
            self._contact_handle.remove()
            self._contact_handle = None
        if contact is not None and contact.shape[0] >= 2:
            fz, fs = [], []
            for cf in self.contact_frames:
                if cf not in self.frame_pos:
                    fz, fs = [], []
                    break
                tr = self._world_track(y, self.frame_pos[cf])
                fz.append(tr[:, 2])
                fs.append(np.r_[0.0, np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=-1)
                                / float(self.lib.meta["dt"])])
            if fz:
                img = _contact_image(contact, np.stack(fz, 1), np.stack(fs, 1),
                                     list(self.contact_frames), float(self.lib.meta["dt"]))
                if img is not None:
                    self._contact_handle = self.server.gui.add_image(
                        img, label="contact schedule / foot z / foot speed", visible=self.show_plot)

        # NOTE: foot-touchdown markers used to be drawn here from the ``contact`` dataset.  Removed --
        # nothing in the loops-and-in-betweening generator computes contact from the spliced motion, so
        # the flag the clip carries is whatever ClipOutputBuilder derived geometrically and the dots did
        # not line up with what the robot was visibly doing.  Better absent than quietly wrong.

        self.refresh_ghosts()
        self._refresh_keyframes()
        self.set_frame(0)

    def set_keyframes(self, kf_by_clip: dict[int, dict], max_ghosts: int = 12) -> None:
        """Attach pinned-keyframe data. Builds NO geometry -- see :meth:`_ensure_kf_pool`."""
        self._keyframes = kf_by_clip
        self._kf_max = int(max_ghosts)

    def _ensure_kf_pool(self) -> None:
        """Build the keyframe ghost pool, ONCE, the first time they are actually shown.

        Deliberately lazy and capped. Two ghosts per slot -- gold and red -- because ``ViserUrdf``
        fixes its colour at construction and slots are reused as you switch clips, so a slot cannot
        be recoloured in place. That doubling is what makes an eager pool expensive: sizing it to the
        busiest clip built 37 x 2 = 74 full G1 URDFs (~2000 meshes) at STARTUP even though
        ``show_keyframes`` defaults off, which hung the browser before a single keyframe was
        requested. Now nothing is built until the toggle goes on, and never more than ``--max-
        keyframes`` slots; a clip with more keyframes than that is SUBSAMPLED evenly (reported, not
        silently truncated -- see :meth:`_refresh_keyframes`).
        """
        if self._kf_ghosts or not self._keyframes:
            return
        pool = min(self._kf_max, max((len(v["pos"]) for v in self._keyframes.values()), default=0))
        for s in range(pool):
            pair = {}
            for tag, col in (("stride", _KEYFRAME_COLOR), ("terrain", _KEYFRAME_TERRAIN_COLOR)):
                fr = self.server.scene.add_frame(f"/keyframes/{tag}/{s}", show_axes=False, visible=False)
                gh = ViserUrdf(self.server, self.urdf, root_node_name=f"/keyframes/{tag}/{s}",
                               mesh_color_override=(*col, _KEYFRAME_ALPHA))
                pair[tag] = (fr, gh)
            self._kf_ghosts.append(pair)
        print(f"[view_clip_viser] built {pool} keyframe ghost slots on first show "
              f"({2 * pool} robots)")

    def _kf_slots(self, kf: dict) -> np.ndarray:
        """Which of this clip's keyframes get a ghost: all of them, or an even subsample."""
        n = len(kf["pos"])
        if n <= len(self._kf_ghosts):
            return np.arange(n)
        return np.unique(np.round(np.linspace(0, n - 1, len(self._kf_ghosts))).astype(int))

    def _refresh_keyframes(self) -> None:
        """Pose the keyframe ghosts (+ root markers) for the current clip; hide the rest.

        Gold = pinned against our steady stride, red = pinned against the terrain replay.  A red ghost's
        POSE is what Motion Bricks was told to reach; its PLACE is not -- the terrain window's root xy is
        discarded and replaced by the command rollout, so where it stands is a consequence of the
        commanded velocity, not of where the terrain actually is.
        """
        kf = (self._keyframes or {}).get(self.clip_idx)
        if self.show_keyframes and kf is not None:
            self._ensure_kf_pool()
        if self._kf_markers is not None:
            self._kf_markers.remove()
            self._kf_markers = None
        # ROOT markers below show EVERY keyframe; only the posed robots are subsampled, so a capped
        # pool still tells you where all of them are.
        sel = self._kf_slots(kf) if (kf is not None and self._kf_ghosts) else np.zeros(0, int)
        for s, pair in enumerate(self._kf_ghosts):
            live = (kf is not None and self.show_keyframes and s < len(sel))
            src = int(sel[s]) if live else 0
            tag = str(kf["kind"][src]) if live else ""
            for name, (fr, gh) in pair.items():
                if live and name == (tag if tag in pair else "stride"):
                    fr.position = kf["pos"][src]
                    fr.wxyz = kf["wxyz"][src]
                    gh.update_cfg(kf["cfg"][src])
                    fr.visible = True
                else:
                    fr.visible = False
        if kf is not None and self.show_keyframes and len(kf["pos"]):
            cols = np.array([_KEYFRAME_TERRAIN_COLOR if str(k) == "terrain" else _KEYFRAME_COLOR
                             for k in kf["kind"]], np.float32)
            self._kf_markers = self.server.scene.add_point_cloud(
                "/keyframe_roots", points=kf["pos"].astype(np.float32), colors=cols,
                point_size=0.05, point_shape="circle",
            )

    def set_show_keyframes(self, show: bool) -> None:
        self.show_keyframes = show
        self._refresh_keyframes()

    def refresh_ghosts(self) -> None:
        """Pose the active ghost robots at evenly-spaced steps along the clip."""
        if self._world_base_pos is None:
            return
        active = (
            np.unique(np.round(np.linspace(0, self.num_steps - 1, self.num_ghosts)).astype(int))
            if self.num_ghosts > 0 else np.array([], dtype=int)
        )
        for s, (fr, gh) in enumerate(self._ghosts):
            if s < len(active):
                k = int(active[s])
                fr.position = self._world_base_pos[k]
                fr.wxyz = self._world_base_quat[k]
                gh.update_cfg(self._joint_cfg[k])
                fr.visible = True
            else:
                fr.visible = False

    def set_frame(self, step: int) -> None:
        """Pose the solid scrubber robot at ``step``."""
        if self._world_base_pos is None:
            return
        step = int(np.clip(step, 0, self.num_steps - 1))
        self._scrub_frame.position = self._world_base_pos[step]
        self._scrub_frame.wxyz = self._world_base_quat[step]
        self._scrub.update_cfg(self._joint_cfg[step])

    def set_show_tracks(self, show: bool) -> None:
        self.show_tracks = show
        for h in self._track_handles.values():
            h.visible = show

    def set_num_ghosts(self, n: int) -> None:
        self.num_ghosts = int(np.clip(n, 0, self.max_ghosts))
        self.refresh_ghosts()

    def set_show_tube(self, show: bool) -> None:
        self.show_tube = show
        if self._tube_handle is not None:
            self._tube_handle.visible = show

    def set_show_command(self, show: bool) -> None:
        self.show_command = show
        if self._cmd_handle is not None:
            self._cmd_handle.visible = show

    def set_show_plot(self, show: bool) -> None:
        self.show_plot = show
        if self._plot_handle is not None:
            self._plot_handle.visible = show
        if self._contact_handle is not None:
            self._contact_handle.visible = show

    def target_vel_at(self, step: int) -> np.ndarray | None:
        """Sampled (vx, vy, wz) command at ``step``, or None if not stored."""
        if self._target_vel is None:
            return None
        return self._target_vel[int(np.clip(step, 0, self.num_steps - 1))]


def main() -> int:
    """Parse args, build the viser scene, and serve until interrupted."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5 path.")
    p.add_argument("--clip", type=int, default=0, help="Initial clip index.")
    p.add_argument("--urdf", default=_DEFAULT_URDF, help="G1 URDF path.")
    p.add_argument("--port", type=int, default=8080, help="Viser server port.")
    p.add_argument("--max-ghosts", type=int, default=10, help="Ghost robot pool size.")
    p.add_argument("--ghosts", type=int, default=0,
                   help="Initial trajectory-ghost count (default 0 -- many robot meshes can hang the browser; "
                        "raise via the GUI slider as needed).")
    p.add_argument("--term-dist", type=float, default=0.25,
                   help="Base-pose-error termination distance (m); radius of the corridor tube.")
    p.add_argument("--keyframes", default=None,
                   help="Optional mb_encode.py _keyframes.npz sidecar: overlay the pinned target poses (gold).")
    p.add_argument("--max-keyframes", type=int, default=12,
                   help="Posed keyframe robots to build (each costs TWO URDFs, gold + red). Built "
                        "lazily on first show; a clip with more keyframes is subsampled evenly, "
                        "while the root markers still show every one.")
    args = p.parse_args()

    lib = load_clip_library(args.lib, "cpu")
    urdf = load_urdf(args.urdf)
    print(f"[view_clip_viser] {lib.num_clips} clips x {lib.num_steps} steps, dt={lib.meta['dt']}s")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")  # grid is (re)created per clip in load_clip, sized to the chunk

    scene = ClipScene(server, lib, urdf, max_ghosts=args.max_ghosts, term_dist=args.term_dist)
    scene.num_ghosts = min(args.ghosts, args.max_ghosts)
    keyframes = _load_keyframe_sidecar(args.keyframes, urdf) if args.keyframes else None
    if keyframes is not None:
        scene.set_keyframes(keyframes, max_ghosts=args.max_keyframes)
        nkf = sum(len(v["pos"]) for v in keyframes.values())
        print(f"[view_clip_viser] loaded {nkf} pinned keyframes over {len(keyframes)} clips from {args.keyframes}")

    # ---- GUI ----
    gui_clip = server.gui.add_slider("clip", 0, lib.num_clips - 1, 1, args.clip)
    gui_frame = server.gui.add_slider("frame", 0, lib.num_steps - 1, 1, 0)
    gui_play = server.gui.add_checkbox("play", True)
    gui_ghosts = server.gui.add_slider("ghosts", 0, args.max_ghosts, 1, scene.num_ghosts)
    gui_tracks = server.gui.add_checkbox("show trajectories", True)
    gui_command = server.gui.add_checkbox("command path (green)", True)
    gui_plot = server.gui.add_checkbox("tracking plot", True)
    gui_tube = server.gui.add_checkbox(f"termination tube ({args.term_dist:.2f} m)", False)
    if keyframes is not None:
        gui_keyframes = server.gui.add_checkbox("pinned keyframes (gold)", scene.show_keyframes)
    gui_vel = server.gui.add_text("target vel (vx,vy,wz)", "", disabled=True)
    gui_info = server.gui.add_text("info", "", disabled=True)

    def _update_info():
        ci = scene.clip_idx
        step = int(gui_frame.value)
        sk = lib.skill_id[ci].cpu().numpy()
        cur_skill = lib.meta["skill_list"][int(sk[min(step, len(sk) - 1)])]
        present = sorted(set(sk.tolist()))
        skills = [lib.meta["skill_list"][s] for s in present]
        # Terrain type / id of this clip (flat clips have terrain_id -1).
        type_names = lib.attrs.get("terrain_type_names", ["flat"])
        ttype = type_names[int(lib.terrain_type_code[ci])] if lib.terrain_type_code.numel() else "?"
        tid = int(lib.terrain_id[ci]) if lib.terrain_id.numel() else -1
        tinfo = f"{ttype}" + (f" (terrain {tid})" if tid >= 0 else "")
        # The clip control is a bare numeric slider, so a library that names its clips (``clip_labels``,
        # e.g. one assembled to compare generator settings side by side) is unreadable without this.
        labels = lib.attrs.get("clip_labels") or lib.attrs.get("merge_labels") or []
        lab = f" \"{labels[ci]}\"" if ci < len(labels) else ""
        gui_info.value = (
            f"clip {ci}{lab} [{tinfo}] | step {step}/{scene.num_steps - 1} | now: {cur_skill} | "
            f"clip skills: {', '.join(skills)}"
        )
        tv = scene.target_vel_at(step)
        gui_vel.value = (
            f"vx={tv[0]:+.2f}  vy={tv[1]:+.2f}  wz={tv[2]:+.2f}"
            if tv is not None else "(not stored in this clip — regenerate)"
        )

    @gui_clip.on_update
    def _(_=None):
        scene.load_clip(int(gui_clip.value))
        gui_frame.max = scene.num_steps - 1          # clip's real length (no padding tail)
        gui_frame.value = 0
        _update_info()

    @gui_frame.on_update
    def _(_=None):
        scene.set_frame(int(gui_frame.value))
        _update_info()

    @gui_ghosts.on_update
    def _(_=None):
        scene.set_num_ghosts(int(gui_ghosts.value))

    @gui_tracks.on_update
    def _(_=None):
        scene.set_show_tracks(bool(gui_tracks.value))

    @gui_command.on_update
    def _(_=None):
        scene.set_show_command(bool(gui_command.value))

    @gui_plot.on_update
    def _(_=None):
        scene.set_show_plot(bool(gui_plot.value))

    @gui_tube.on_update
    def _(_=None):
        scene.set_show_tube(bool(gui_tube.value))

    if keyframes is not None:
        @gui_keyframes.on_update
        def _(_=None):
            scene.set_show_keyframes(bool(gui_keyframes.value))

    scene.load_clip(args.clip)
    gui_frame.max = scene.num_steps - 1              # clip's real length (no padding tail)
    _update_info()
    print(f"[view_clip_viser] serving on http://localhost:{args.port}  (Ctrl-C to stop)")

    dt = float(lib.meta["dt"])
    try:
        while True:
            if gui_play.value:
                nxt = (int(gui_frame.value) + 1) % scene.num_steps
                scene.set_frame(nxt)       # pose the scrubber directly
                gui_frame.value = nxt       # reflect in the slider / info
                _update_info()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[view_clip_viser] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
