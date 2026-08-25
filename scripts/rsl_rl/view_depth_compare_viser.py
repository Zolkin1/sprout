# Copyright 2026 Zachary Olkin. All rights reserved.

"""Both simulators' depth scans, side by side in 3D.

Companion viewer for ``scripts/rsl_rl/compare_depth_sim2sim.py --dump``.  Simulator-free:
it only needs the ``.npz``, so it runs anywhere the venv does.

The 2-D difference images say THAT the two sims disagree; they cannot say WHERE in space,
which is what you need to tell a displaced mesh from a displaced camera.  This draws, per
captured posture:

* the robot at the pose both sims were placed at (they are written the same joint angles
  and the same settled root pose, so the two skeletons should coincide exactly -- use the
  "separate sims" slider to pull MuJoCo aside and confirm);
* each camera's mount frame as an axis triad, per sim.  Red is +x, the optical axis.  If
  Isaac's and MuJoCo's triads do not sit on top of each other, the contract is being
  applied differently and nothing downstream is meaningful;
* the depth hit points, coloured per sim, so a displaced mesh shows up as two offset
  sheets of dots rather than as a smear in an image;
* optional red segments joining corresponding rays whose depths differ by more than a
  threshold -- the 3-D version of the difference panel.

Hit points are RECONSTRUCTED from each sim's own depth image using the shared ray pattern
(``dirs_cam``) and that sim's own camera pose::

    ray_length = image_plane_depth / cos_theta        (cos_theta = dirs_cam[:, 0])
    hit_world  = camera_pos + ray_length * (dirs_cam @ rot.T)

so a dot that lands somewhere different is a genuine depth or pose difference, not an
artefact of two different reconstruction paths.

Usage (from the repo root, with the venv active)::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
        python scripts/rsl_rl/compare_depth_sim2sim.py --poses 3 --dump depth_cmp.npz
    PYTHONPATH=source/robot_rl:source/motion_gen \\
        python scripts/rsl_rl/view_depth_compare_viser.py --npz depth_cmp.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import viser
from viser.extras import ViserUrdf

from robot_rl.flow_matching.viz.robot import DEFAULT_URDF, load_urdf, recording_to_urdf_perm, xyzw_to_wxyz

_DEFAULT_URDF = DEFAULT_URDF

# One hue per simulator, one shade per camera, so sim identity survives a screenshot.
_ISAAC_NEAR = (70, 220, 110)
_ISAAC_FAR = (30, 140, 220)
_MUJOCO_NEAR = (255, 165, 40)
_MUJOCO_FAR = (225, 80, 180)
_MISMATCH = (240, 40, 40)

#: Scene-node name -> the prefix that simulator's arrays carry in the dump.
_KEY = {"isaac": "isaac", "mujoco": "mj"}


def _hit_points(depth_m: np.ndarray, dirs_cam: np.ndarray, cos_theta: np.ndarray,
                cam_pos: np.ndarray, cam_rot: np.ndarray,
                max_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct world hit points from one camera's image-plane depth.

    Args:
        depth_m: ``[R]`` per-ray depth to the image plane, metres.
        dirs_cam: ``[R, 3]`` unit ray directions in the camera frame.
        cos_theta: ``[R]`` the rays' optical-axis component (``dirs_cam[:, 0]``).
        cam_pos: ``[3]`` camera position in world.
        cam_rot: ``[3, 3]`` camera rotation, columns = camera axes in world.
        max_dist: Range at or beyond which a reading means "no hit".

    Returns:
        ``([R, 3]`` world points, ``[R]`` valid mask``)``.  Invalid rows are finite but
        meaningless; filter with the mask before drawing.
    """
    valid = depth_m < max_dist - 1e-6
    ray_len = depth_m / np.maximum(cos_theta, 1e-9)
    return cam_pos[None, :] + ray_len[:, None] * (dirs_cam @ cam_rot.T), valid


def main() -> None:
    """Serve both sims' robots, camera frames and depth clouds."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="depth_cmp.npz", help="From compare_depth_sim2sim --dump.")
    ap.add_argument("--urdf", default=_DEFAULT_URDF, help="29-DOF G1 URDF.")
    ap.add_argument("--point_size", type=float, default=0.012, help="Depth dot radius (m).")
    ap.add_argument("--axes_length", type=float, default=0.18, help="Camera triad length (m).")
    ap.add_argument("--port", type=int, default=8080, help="Viser port.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    tags = [str(t) for t in d["tag"]]
    sensors = [str(s) for s in d["sensors"]]
    joint_names = [str(j) for j in d["joint_names"]]
    n_poses, n_cams = len(tags), len(sensors)
    dirs_cam, cos_theta, max_dist = d["dirs_cam"], d["cos_theta"], d["max_dist"]

    urdf = load_urdf(args.urdf)
    perm = recording_to_urdf_perm(joint_names, urdf)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=12.0, height=12.0, cell_size=0.5)

    # Two robots: identical by construction, so they overlay until pulled apart.
    robots = {}
    for sim in ("isaac", "mujoco"):
        frame = server.scene.add_frame(f"/{sim}", show_axes=False)
        robots[sim] = (frame, ViserUrdf(server, urdf, root_node_name=f"/{sim}"))

    # Deliberately NOT under "/{sim}": that node is the robot's root frame, sitting at the
    # pelvis, and viser composes a child's transform with its parent's -- so a world camera
    # position written there lands ~0.8 m too high.  These carry world poses, so they hang
    # off a parentless node.
    cam_frames = {(sim, i): server.scene.add_frame(
        f"/cams/{sim}/{sensors[i]}", show_axes=True,
        axes_length=args.axes_length, axes_radius=args.axes_length * 0.035)
        for sim in ("isaac", "mujoco") for i in range(n_cams)}

    colors = {("isaac", 0): _ISAAC_NEAR, ("isaac", 1): _ISAAC_FAR,
              ("mujoco", 0): _MUJOCO_NEAR, ("mujoco", 1): _MUJOCO_FAR}
    clouds = {(sim, i): server.scene.add_point_cloud(
        f"/cloud/{sim}/{sensors[i]}", points=np.zeros((1, 3)),
        colors=colors.get((sim, i), (200, 200, 200)), point_size=args.point_size)
        for sim in ("isaac", "mujoco") for i in range(n_cams)}

    segs = server.scene.add_line_segments(
        "/mismatch", points=np.zeros((1, 2, 3)), colors=_MISMATCH, line_width=2.0)

    with server.gui.add_folder("Posture"):
        slider = server.gui.add_slider("index", 0, max(n_poses - 1, 0), 1, 0)
        info = server.gui.add_markdown("")
    with server.gui.add_folder("Show"):
        g_isaac = server.gui.add_checkbox("IsaacLab robot", True)
        g_mujoco = server.gui.add_checkbox("MuJoCo robot", True)
        g_cams = [server.gui.add_checkbox(f"{s} dots", True) for s in sensors]
        g_axes = server.gui.add_checkbox("camera axes", True)
        g_seg = server.gui.add_checkbox("mismatch segments", True)
        g_tol = server.gui.add_slider("mismatch > (m)", 0.005, 1.0, 0.005, 0.02)
        g_sep = server.gui.add_slider("separate sims (m)", 0.0, 3.0, 0.05, 0.0)

    def redraw(_=None) -> None:
        """Push posture ``slider.value`` into the scene."""
        p = int(slider.value)
        cfg = np.asarray(d["joint_pos"][p])[perm]
        root_pos = np.asarray(d["root_pos"][p], dtype=float)
        root_wxyz = xyzw_to_wxyz(np.asarray(d["root_quat_xyzw"][p], dtype=float))
        # MuJoCo is drawn shifted along +y so the two can be told apart on demand; every
        # world quantity on that side gets the same shift.
        shift = {"isaac": np.zeros(3), "mujoco": np.array([0.0, float(g_sep.value), 0.0])}

        for sim, (frame, vurdf) in robots.items():
            vurdf.update_cfg(cfg)
            frame.position = root_pos + shift[sim]
            frame.wxyz = root_wxyz
            frame.visible = g_isaac.value if sim == "isaac" else g_mujoco.value

        lines, worst, counts = [], 0.0, {}
        for sim in ("isaac", "mujoco"):
            k = _KEY[sim]
            pos_key, rot_key = f"{k}_cam_pos", f"{k}_cam_rot"
            depth_key = f"{k}_depth_m"
            for i in range(n_cams):
                cam_pos = np.asarray(d[pos_key][p][i], dtype=float)
                cam_rot = np.asarray(d[rot_key][p][i], dtype=float)
                pts, valid = _hit_points(
                    np.asarray(d[depth_key][p][i], dtype=float), dirs_cam[i], cos_theta[i],
                    cam_pos, cam_rot, float(max_dist[i]))
                pts = pts + shift[sim]
                clouds[(sim, i)].points = pts[valid] if valid.any() else np.zeros((1, 3))
                clouds[(sim, i)].visible = g_cams[i].value
                cam_frames[(sim, i)].position = cam_pos + shift[sim]
                cam_frames[(sim, i)].wxyz = _mat_to_wxyz(cam_rot)
                cam_frames[(sim, i)].visible = g_axes.value
                counts[(sim, i)] = int(valid.sum())

        # Mismatch segments join the SAME ray's two reconstructions, so a segment is a
        # per-pixel disagreement rather than a nearest-neighbour guess.
        for i in range(n_cams):
            if not g_cams[i].value:
                continue
            a, va = _hit_points(np.asarray(d["isaac_depth_m"][p][i], dtype=float),
                                dirs_cam[i], cos_theta[i],
                                np.asarray(d["isaac_cam_pos"][p][i], dtype=float),
                                np.asarray(d["isaac_cam_rot"][p][i], dtype=float),
                                float(max_dist[i]))
            b, vb = _hit_points(np.asarray(d["mj_depth_m"][p][i], dtype=float),
                                dirs_cam[i], cos_theta[i],
                                np.asarray(d["mj_cam_pos"][p][i], dtype=float),
                                np.asarray(d["mj_cam_rot"][p][i], dtype=float),
                                float(max_dist[i]))
            gap = np.abs(np.asarray(d["isaac_depth_m"][p][i], dtype=float)
                         - np.asarray(d["mj_depth_m"][p][i], dtype=float))
            worst = max(worst, float(gap.max()))
            sel = (gap > float(g_tol.value)) & va & vb
            if sel.any():
                lines.append(np.stack([a[sel], b[sel] + shift["mujoco"]], axis=1))
        segs.points = np.concatenate(lines) if lines else np.zeros((1, 2, 3))
        segs.visible = g_seg.value and bool(lines)

        rows = "\n".join(
            f"| {sensors[i]} | {counts[('isaac', i)]} | {counts[('mujoco', i)]} |"
            for i in range(n_cams))
        info.content = (
            f"**{tags[p]}**  ({p + 1} / {n_poses})\n\n"
            f"worst per-ray depth gap **{worst:.4f} m**; "
            f"{sum(int(x.shape[0]) for x in lines)} rays over {g_tol.value:.3f} m\n\n"
            f"| camera | isaac hits | mujoco hits |\n|---|---|---|\n{rows}"
        )

    for widget in (slider, g_isaac, g_mujoco, g_axes, g_seg, g_tol, g_sep, *g_cams):
        widget.on_update(redraw)
    redraw()
    print(f"[viser] serving on http://localhost:{args.port}")
    while True:
        server.sleep_forever()


def _mat_to_wxyz(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> ``(w, x, y, z)`` quaternion, via the numerically stable branch.

    Args:
        rot: ``[3, 3]`` rotation matrix.

    Returns:
        ``[4]`` quaternion.
    """
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array([0.25 * s, (rot[2, 1] - rot[1, 2]) / s,
                         (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s])
    i = int(np.argmax(np.diagonal(rot)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (rot[k, j] - rot[j, k]) / s
    q[1 + i], q[1 + j], q[1 + k] = 0.25 * s, (rot[j, i] + rot[i, j]) / s, (rot[k, i] + rot[i, k]) / s
    return q


if __name__ == "__main__":
    main()
