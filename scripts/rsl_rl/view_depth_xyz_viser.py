# Copyright 2026 Zachary Olkin. All rights reserved.

"""The depth cameras' xyz channels, in the frame the generator actually reads them in.

Companion viewer for ``scripts/rsl_rl/check_depth_xyz_agreement.py --dump``. Simulator-free:
it needs only the ``.npz``.

Everything here is drawn in the ROOT'S YAW-ALIGNED FRAME -- the origin is the pelvis, +z is
gravity, and the frame carries no roll or pitch. That is not a display convention, it is the
frame the ``extero_channels >= 4`` rows are expressed in, so what you see is what the model
is handed.

**What to look for.** Turn on several poses at once. The robot leans, rolls, turns its waist
and bends its knees; the TERRAIN cloud should stay put, because it did not move and the frame
did not move with the robot. That invariance is the entire reason the channels exist, and it
is the thing a wrong frame breaks while every shape and magnitude stays plausible.

Three honest caveats on that overlay, none of them a bug:

* the clouds coincide only where the cameras SAW the same ground. Tilting swings a steeply
  downward-looking pair, so each pose also sees some terrain the others do not;
* points on the ROBOT ITSELF (the cameras self-scan) genuinely do move, since the robot moved
  within a level frame. ``self-hit radius`` hides points near the origin to take them out of
  the picture -- it is a crude range filter, not a mesh-id lookup;
* a yaw turns the frame with the robot, so the ``yaw`` pose's terrain cloud is expected to
  come back ROTATED. It is the one pose in the dump that should not overlay, and it is there
  to prove the rest of the overlay is not a rendering accident.

Usage (repo root, venv active -- ``env_transfer`` is enough, no Isaac needed)::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
        python scripts/rsl_rl/view_depth_xyz_viser.py --npz depth_xyz.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from robot_rl.flow_matching.viz.robot import load_urdf, recording_to_urdf_perm, xyzw_to_wxyz

#: The hand links are REAL fixed-joint links in this asset, which is what the depth cameras
#: see (see CLAUDE.md: as bare <visual> elements the ray caster misses them entirely). Using
#: a hand-less URDF here would draw a robot that does not match its own self-scan.
_DEFAULT_URDF = "robot_assets/g1/g1_21j/g1_21j_box_merged_29j_hands.urdf"

#: One colour per pose, ordered as ``_POSES`` in the check script. Chosen to stay distinct
#: when several clouds are overlaid on top of each other.
_COLORS: tuple[tuple[int, int, int], ...] = (
    (235, 235, 235),      # rest        -- white, the datum
    (250, 150, 40),       # pitch_fwd   -- orange
    (250, 90, 150),       # pitch_back  -- pink
    (120, 220, 120),      # roll        -- green
    (230, 70, 70),        # yaw         -- red, the one that SHOULD move
    (100, 180, 250),      # waist       -- blue
    (200, 140, 250),      # knees       -- violet
)


def _cloud(points: np.ndarray, valid: np.ndarray, min_radius: float) -> np.ndarray:
    """Flatten one pose's per-camera points into a single filtered ``(N, 3)`` cloud.

    Args:
        points: ``(n_cam, 3, K)`` xyz in the yaw frame.
        valid: ``(n_cam, K)`` true where the pixel carries a real reading.
        min_radius: Drop points within this distance of the origin (the crude self-hit
            filter). ``0`` keeps everything.

    Returns:
        ``(N, 3)`` float32 points.
    """
    pts = points.transpose(0, 2, 1).reshape(-1, 3)          # (n_cam * K, 3)
    keep = valid.reshape(-1)
    if min_radius > 0.0:
        keep = keep & (np.linalg.norm(pts, axis=-1) >= min_radius)
    return pts[keep].astype(np.float32)


def main() -> None:
    """Serve the yaw-frame point clouds and posed robots for every captured pose."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", required=True, help="Dump from check_depth_xyz_agreement.py.")
    ap.add_argument("--urdf", default=_DEFAULT_URDF, help="29-DOF G1 URDF.")
    ap.add_argument("--port", type=int, default=8080, help="Viser port.")
    args = ap.parse_args()

    data = np.load(args.npz)
    pose_names = [str(p) for p in data["pose_names"]]
    cam_names = [str(c) for c in data["cam_names"]]
    joint_names = [str(j) for j in data["joint_names"]]

    urdf = load_urdf(args.urdf)
    perm = recording_to_urdf_perm(joint_names, urdf)

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    robots, clouds, cams = {}, {}, {}
    for i, name in enumerate(pose_names):
        q_rp = data[f"{name}/root_quat_rp"]                 # xyzw, yaw removed
        # The robot's root sits at the frame ORIGIN by construction -- the frame is anchored
        # to it -- carrying only the roll/pitch that survives yaw_quat.
        frame = server.scene.add_frame(
            f"/{name}", show_axes=False, position=(0.0, 0.0, 0.0),
            wxyz=tuple(xyzw_to_wxyz(q_rp).astype(float)),
        )
        vurdf = ViserUrdf(server, urdf, root_node_name=f"/{name}")
        vurdf.update_cfg(np.asarray(data[f"{name}/joint_pos"])[perm])
        robots[name] = frame

        # Camera triads go at the SCENE root, not under the robot node: cam_pos/cam_rot are
        # already expressed in the yaw frame, so nesting them would apply the tilt twice.
        cams[name] = [
            server.scene.add_frame(
                f"/cams_{name}/{cam_names[c]}", show_axes=True, axes_length=0.25,
                axes_radius=0.006, position=tuple(float(v) for v in
                                                  data[f"{name}/cam_pos"][c]),
                wxyz=tuple(_mat_to_wxyz(data[f"{name}/cam_rot"][c]).astype(float)),
            )
            for c in range(len(cam_names))
        ]
        pts = _cloud(data[f"{name}/points"], data[f"{name}/valid"], 0.0)
        clouds[name] = server.scene.add_point_cloud(
            f"/cloud_{name}", points=pts,
            colors=np.tile(np.array(_COLORS[i % len(_COLORS)], dtype=np.uint8),
                           (pts.shape[0], 1)),
            point_size=0.012,
        )

    with server.gui.add_folder("Poses"):
        show = {
            name: server.gui.add_checkbox(
                f"{name}  ({_COLORS[i % len(_COLORS)]})", initial_value=(name in
                                                                        ("rest", "pitch_fwd"))
            )
            for i, name in enumerate(pose_names)
        }
    with server.gui.add_folder("Draw"):
        show_robot = server.gui.add_checkbox("robot", initial_value=True)
        show_cams = server.gui.add_checkbox("camera frames", initial_value=True)
        radius = server.gui.add_slider("self-hit radius (m)", min=0.0, max=1.5, step=0.05,
                                       initial_value=0.0)
        size = server.gui.add_slider("point size", min=0.002, max=0.04, step=0.002,
                                     initial_value=0.012)

    def redraw(_=None) -> None:
        """Apply the GUI state to every scene node."""
        for name in pose_names:
            on = show[name].value
            robots[name].visible = on and show_robot.value
            for f in cams[name]:
                f.visible = on and show_cams.value
            pts = _cloud(data[f"{name}/points"], data[f"{name}/valid"], radius.value)
            clouds[name].points = pts
            clouds[name].colors = np.tile(
                np.array(_COLORS[pose_names.index(name) % len(_COLORS)], dtype=np.uint8),
                (pts.shape[0], 1),
            )
            clouds[name].point_size = size.value
            clouds[name].visible = on

    for handle in list(show.values()) + [show_robot, show_cams, radius, size]:
        handle.on_update(redraw)
    redraw()

    print(f"[viser] serving on http://localhost:{args.port}  ({len(pose_names)} poses)")
    print("[viser] tip: enable several poses -- the robot moves, the terrain should not "
          "(except 'yaw').")
    while True:
        time.sleep(1.0)


def _mat_to_wxyz(rot: np.ndarray) -> np.ndarray:
    """Quaternion ``(w, x, y, z)`` from a 3x3 rotation matrix.

    Shepperd's method: pick the branch with the largest denominator, which is what keeps it
    stable near each of the four degenerate cases.

    Args:
        rot: ``(3, 3)`` rotation matrix.

    Returns:
        ``(4,)`` quaternion, scalar-FIRST (viser's convention).
    """
    m, t = np.asarray(rot, dtype=np.float64), float(np.trace(rot))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i], q[1 + j], q[1 + k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return q


if __name__ == "__main__":
    main()
