# Copyright 2026 Zachary Olkin. All rights reserved.

"""Inspect the spawn states that trip ``ee_pose_error``, in viser.

Companion viewer for ``scripts/rsl_rl/capture_reset_states.py``.  Simulator-free: it only
needs the ``.npz``, so it runs anywhere the venv does.

Per case it draws the robot at the pose it was actually spawned in, plus two sets of spheres:

* GREEN  -- ``y_act``, where the robot's tracked bodies really are.
* RED    -- ``y_des``, where the reward and the ``ee_pose_error`` termination want them.
* BLUE (off by default) -- ``body_pos_w`` read live at capture time.  Measured identical to
  green, so it is only here to prove measurement lag is not the story.

Red-to-green line segments make the separation legible at a glance.  What the captured data
shows is a PURE PLANAR RIGID TRANSLATION -- every tracked body off by the same ~0.57 m with
dz identically zero -- so the red skeleton should appear as a correctly-shaped copy of the
robot displaced downrange, not as a distorted or mistracked pose.

Usage (from the repo root, with the venv active)::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
    python scripts/rsl_rl/view_reset_states_viser.py --npz reset_states.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import viser
from viser.extras import ViserUrdf

from robot_rl.flow_matching.viz.robot import DEFAULT_URDF, load_urdf, recording_to_urdf_perm, xyzw_to_wxyz

_DEFAULT_URDF = DEFAULT_URDF

_GREEN = (60, 200, 90)
_RED = (225, 70, 70)
_BLUE = (70, 130, 230)


def main() -> None:
    """Serve the captured spawn states with actual-vs-target body spheres."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="reset_states.npz", help="From capture_reset_states.py.")
    ap.add_argument("--urdf", default=_DEFAULT_URDF, help="29-DOF G1 URDF.")
    ap.add_argument("--radius", type=float, default=0.045, help="Sphere radius (m).")
    ap.add_argument("--port", type=int, default=8080, help="Viser port.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    frames = [str(x) for x in d["frame_names"]]
    joint_names = [str(x) for x in d["joint_names"]]
    n_cases = int(d["act_p"].shape[0])
    limit = float(d["ee_limit"])

    urdf = load_urdf(args.urdf)
    perm = recording_to_urdf_perm(joint_names, urdf)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=12.0, height=12.0, cell_size=0.5)

    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")

    act_s = [server.scene.add_icosphere(f"/act/{n}", radius=args.radius, color=_GREEN)
             for n in frames]
    des_s = [server.scene.add_icosphere(f"/des/{n}", radius=args.radius, color=_RED,
                                        opacity=0.65) for n in frames]
    live_s = [server.scene.add_icosphere(f"/live/{n}", radius=args.radius * 0.55, color=_BLUE,
                                         visible=False) for n in frames]
    links = server.scene.add_line_segments(
        "/links", points=np.zeros((len(frames), 2, 3)), colors=(255, 210, 0), line_width=3.0)

    with server.gui.add_folder("Case"):
        slider = server.gui.add_slider("index", 0, max(n_cases - 1, 0), 1, 0)
        info = server.gui.add_markdown("")
    with server.gui.add_folder("Show"):
        g_act = server.gui.add_checkbox("actual (green)", True)
        g_des = server.gui.add_checkbox("target (red)", True)
        g_live = server.gui.add_checkbox("live read (blue)", False)
        g_link = server.gui.add_checkbox("separation lines", True)

    def redraw(_=None) -> None:
        """Push case ``slider.value`` into the scene."""
        c = int(slider.value)
        viser_urdf.update_cfg(np.asarray(d["joint_pos"][c])[perm])
        robot_frame.position = np.asarray(d["root_pos"][c])
        robot_frame.wxyz = xyzw_to_wxyz(np.asarray(d["root_quat"][c]))

        a, t, lv = d["act_p"][c], d["des_p"][c], d["live_p"][c]
        for i in range(len(frames)):
            act_s[i].position, des_s[i].position, live_s[i].position = a[i], t[i], lv[i]
            act_s[i].visible, des_s[i].visible = g_act.value, g_des.value
            live_s[i].visible = g_live.value
        links.points = np.stack([a, t], axis=1)
        links.visible = g_link.value

        sep = np.linalg.norm(a - t, axis=-1)
        off = (t - a).mean(axis=0)
        rows = "\n".join(f"| {n} | {s:.3f} |" for n, s in zip(frames, sep))
        info.content = (
            f"**case {c} / {n_cases - 1}**  env {int(d['env_id'][c])}, "
            f"clip {int(d['clip_id'][c])} @ step {int(d['start_step'][c])}\n\n"
            f"worst **{sep.max():.3f} m** vs limit {limit} m &nbsp;|&nbsp; "
            f"mean {sep.mean():.3f} m\n\n"
            f"mean offset (target - actual): "
            f"dx **{off[0]:+.3f}**, dy **{off[1]:+.3f}**, dz **{off[2]:+.3f}** m\n\n"
            f"| frame | sep (m) |\n|---|---|\n{rows}"
        )

    slider.on_update(redraw)
    for box in (g_act, g_des, g_live, g_link):
        box.on_update(redraw)
    redraw()

    print(f"[view_reset_states_viser] {n_cases} cases, {len(frames)} frames "
          f"-> http://localhost:{args.port}")
    import time

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
