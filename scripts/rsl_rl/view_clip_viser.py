#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Interactive viser view of a generated motion clip (no Isaac Sim needed).

Serves a 3D scene in the browser showing, for one clip:

  * every reference frame's world trajectory (pelvis, both feet, both
    wrists, ...) as a coloured polyline, with a marker at each foot
    touchdown;
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

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_io import (
    load_clip_library,
)
from robot_rl.flow_matching.viz.robot import load_urdf, recording_to_urdf_perm, xyzw_to_wxyz

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
_DEFAULT_URDF = "robot_assets/g1/g1_21j/g1_21j.urdf"


def _frame_pos_indices(pos_names: list[str]) -> dict[str, list[int]]:
    """Map each frame name -> its [pos_x, pos_y, pos_z] indices in ``pos_names``."""
    frames: dict[str, list[int]] = {}
    for axis_i, name in enumerate(pos_names):
        if ":pos_" in name:
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


class ClipScene:
    """Builds and updates all viser nodes for one clip at a time."""

    def __init__(self, server: viser.ViserServer, lib, urdf, max_ghosts: int, term_dist: float = 0.25):
        self.server = server
        self.lib = lib
        self.urdf = urdf
        self.max_ghosts = max_ghosts
        self.term_dist = term_dist
        self.num_steps = lib.num_steps

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
        self._touchdown_handles: list[viser.SceneNodeHandle] = []
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

        # Termination-distance corridor tube around the base path (hidden by default).
        self._tube_handle: viser.SceneNodeHandle | None = None

        # Cached per-clip arrays (filled by load_clip).
        self._world_base_pos = None   # [T, 3]
        self._world_base_quat = None  # [T, 4] wxyz
        self._joint_cfg = None        # [T, nj] (urdf order)
        self._target_vel = None       # [T, 3] sampled (vx, vy, wz), or None if absent
        self.show_tracks = True
        self.show_tube = False
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
        y = self.lib.y_des[clip_idx]                 # [T, P] clip-world
        skill = self.lib.skill_id[clip_idx].cpu().numpy()
        contact = self.lib.contact[clip_idx].cpu().numpy()

        # Robot base world pose + joint config for every step.  The stored
        # quaternion is xyzw (Isaac Lab); viser frames want wxyz.
        self._world_base_pos = y[:, self.base_pos_idx].cpu().numpy()
        self._world_base_quat = xyzw_to_wxyz(y[:, self.base_ori_idx].cpu().numpy())
        self._joint_cfg = y[:, self.joint_idx].cpu().numpy()[:, self.joint_perm]

        # Sampled (target) velocity per step, if the library stored it.
        tv = self.lib.target_vel
        self._target_vel = tv[clip_idx].cpu().numpy() if tv.numel() > 0 else None

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

        # Foot touchdown markers (contact rising edges per contact frame).
        for h in self._touchdown_handles:
            h.remove()
        self._touchdown_handles = []
        for ci, cf in enumerate(self.contact_frames):
            if cf not in self.frame_pos:
                continue
            track = self._world_track(y, self.frame_pos[cf])
            c = contact[:, ci].astype(bool)
            td = np.where(c[1:] & ~c[:-1])[0] + 1
            if td.size == 0:
                continue
            fi = list(self.frame_pos).index(cf)
            h = self.server.scene.add_point_cloud(
                f"/touchdown/{cf}", points=track[td].astype(np.float32),
                colors=_FRAME_COLORS[fi % len(_FRAME_COLORS)], point_size=0.04,
                point_shape="circle", visible=self.show_tracks,
            )
            self._touchdown_handles.append(h)

        self.refresh_ghosts()
        self.set_frame(0)

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
        for h in self._touchdown_handles:
            h.visible = show

    def set_num_ghosts(self, n: int) -> None:
        self.num_ghosts = int(np.clip(n, 0, self.max_ghosts))
        self.refresh_ghosts()

    def set_show_tube(self, show: bool) -> None:
        self.show_tube = show
        if self._tube_handle is not None:
            self._tube_handle.visible = show

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
    p.add_argument("--ghosts", type=int, default=8, help="Initial ghost count.")
    p.add_argument("--term-dist", type=float, default=0.25,
                   help="Base-pose-error termination distance (m); radius of the corridor tube.")
    args = p.parse_args()

    lib = load_clip_library(args.lib, "cpu")
    urdf = load_urdf(args.urdf)
    print(f"[view_clip_viser] {lib.num_clips} clips x {lib.num_steps} steps, dt={lib.meta['dt']}s")

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=20.0, height=20.0, cell_size=1.0)
    server.scene.set_up_direction("+z")

    scene = ClipScene(server, lib, urdf, max_ghosts=args.max_ghosts, term_dist=args.term_dist)
    scene.num_ghosts = min(args.ghosts, args.max_ghosts)

    # ---- GUI ----
    gui_clip = server.gui.add_slider("clip", 0, lib.num_clips - 1, 1, args.clip)
    gui_frame = server.gui.add_slider("frame", 0, lib.num_steps - 1, 1, 0)
    gui_play = server.gui.add_checkbox("play", True)
    gui_ghosts = server.gui.add_slider("ghosts", 0, args.max_ghosts, 1, scene.num_ghosts)
    gui_tracks = server.gui.add_checkbox("show trajectories", True)
    gui_tube = server.gui.add_checkbox(f"termination tube ({args.term_dist:.2f} m)", False)
    gui_vel = server.gui.add_text("target vel (vx,vy,wz)", "", disabled=True)
    gui_info = server.gui.add_text("info", "", disabled=True)

    def _update_info():
        ci = scene.clip_idx
        step = int(gui_frame.value)
        sk = lib.skill_id[ci].cpu().numpy()
        cur_skill = lib.meta["skill_list"][int(sk[min(step, len(sk) - 1)])]
        present = sorted(set(sk.tolist()))
        skills = [lib.meta["skill_list"][s] for s in present]
        gui_info.value = (
            f"clip {ci} | step {step}/{lib.num_steps - 1} | now: {cur_skill} | clip skills: {', '.join(skills)}"
        )
        tv = scene.target_vel_at(step)
        gui_vel.value = (
            f"vx={tv[0]:+.2f}  vy={tv[1]:+.2f}  wz={tv[2]:+.2f}"
            if tv is not None else "(not stored in this clip — regenerate)"
        )

    @gui_clip.on_update
    def _(_=None):
        scene.load_clip(int(gui_clip.value))
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

    @gui_tube.on_update
    def _(_=None):
        scene.set_show_tube(bool(gui_tube.value))

    scene.load_clip(args.clip)
    _update_info()
    print(f"[view_clip_viser] serving on http://localhost:{args.port}  (Ctrl-C to stop)")

    dt = float(lib.meta["dt"])
    try:
        while True:
            if gui_play.value:
                nxt = (int(gui_frame.value) + 1) % lib.num_steps
                scene.set_frame(nxt)       # pose the scrubber directly
                gui_frame.value = nxt       # reflect in the slider / info
                _update_info()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[view_clip_viser] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
