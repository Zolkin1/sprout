#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Step through the terrain probe: for each distraction, the scene, both depth images, both plans.

The probe reports how far a piece of scenery moved the generator's plan. This is the picture
behind that number -- what the cameras were actually given, and what the generator drew with
it -- because a displacement in metres does not say whether the plan bent sensibly or fell
apart, and a depth delta does not say whether the wall looked like a wall.

**Everything is drawn in the ROBOT's frame at the probe instant.** Every condition shares one
clip, one step and one pose, so the robot sits at the origin facing +x in all of them and the
view does not move as you step through. That is the whole reason the comparison is legible:
the control's plan and the condition's plan are the same object under one changed variable,
so they can be laid on top of each other rather than beside each other.

Nothing here re-derives geometry. The boxes are the ones the builder authored, carried through
in the probe's dump, so this cannot disagree with what was rendered -- which matters, because
a viewer that quietly re-computes placement is a viewer that can agree with itself and with
nothing else.

Depth is shown as the NORMALISED observation the model reads, on a fixed colour scale so the
two cameras and the two conditions are comparable; magenta is "no reading". The third strip is
the absolute difference from the control -- the stimulus, made visible.

IsaacLab-free.

    python scripts/rsl_rl/fm_terrain_probe.py ... --dump_viz probe_viz.npz
    python scripts/rsl_rl/view_terrain_probe_viser.py --viz probe_viz.npz
    # then open the printed http://localhost:8080 URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "source", "robot_rl"))
from robot_rl.flow_matching.viz.robot import DEFAULT_URDF, load_urdf, recording_to_urdf_perm  # noqa: E402

#: The robot that was trained and deployed. Not the 21-DoF one other viewers still default
#: to -- the probe's joints are the perceptive robot's 29 and would be mis-mapped.
_DEFAULT_URDF = DEFAULT_URDF

#: Feet, drawn from forward kinematics because the generator does not predict them: the
#: served window is base pose + joints only (9 + 6 + 29), and ``target_frame_names`` lists
#: frames that are TRACKED, not predicted.
_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")

#: Control plan, condition plan, and the decor.
_CTRL_COLOR = (150, 150, 150)
_COND_COLOR = (240, 130, 40)
_BOX_COLOR = (0.55, 0.50, 0.62)


def depth_rgb(img: np.ndarray, band: float, upscale: int = 8) -> np.ndarray:
    """Colour one normalised depth frame; magenta where the sensor returned no reading.

    The scale is FIXED to the observation's own clip band rather than to each frame's
    content, so two cameras and two conditions can be compared by eye. Auto-scaling would
    make an empty frame and a wall-filled one look equally busy.

    Args:
        img: ``(H, W)`` normalised depth.
        band: The clip band's upper edge, which is also the no-reading sentinel.
        upscale: Nearest-neighbour magnification (a 30x26 frame is unreadable at native size).

    Returns:
        ``(H*upscale, W*upscale, 3)`` uint8 RGB.
    """
    from matplotlib import colormaps

    miss = img >= band - 1e-3
    lo, hi = -band, band
    norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = colormaps["viridis"](norm)[..., :3]
    rgb[miss] = (1.0, 0.0, 1.0)
    rgb = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    return (255.0 * rgb).astype(np.uint8)


def diff_rgb(cond: np.ndarray, ctrl: np.ndarray, upscale: int = 8) -> np.ndarray:
    """Absolute depth difference from the control, on a fixed 0..2 scale, as ``inferno``."""
    from matplotlib import colormaps

    d = np.abs(cond - ctrl)
    rgb = colormaps["inferno"](np.clip(d / 2.0, 0.0, 1.0))[..., :3]
    rgb = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    return (255.0 * rgb).astype(np.uint8)


def box_mesh(b: dict) -> trimesh.Trimesh:
    """One authored decor box as a trimesh, in the robot frame."""
    sx, sy, sz = (float(v) for v in b["size_xyz"])
    px, py = (float(v) for v in b["pos_xy"])
    m = trimesh.creation.box(extents=(sx, sy, sz))
    m.apply_translation((px, py, float(b.get("base_z", 0.0)) + sz / 2.0))
    return m


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` -> ``(..., 3, 3)``; the 6 numbers are the first two COLUMNS.

    Mirrors ``flow_matching.transforms.rot6d_to_matrix`` (Gram-Schmidt, columns) rather than
    guessing a convention -- reading them as rows would silently transpose every foot.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def foot_paths(urdf, perm: np.ndarray, joints: np.ndarray, ori6: np.ndarray,
               base: np.ndarray) -> dict[str, np.ndarray]:
    """Forward-kinematic foot paths for one plan, in the probe frame.

    Args:
        urdf: The loaded ``yourdfpy`` model.
        perm: Window-joint-order -> URDF-actuated-order permutation. The window order is
            Isaac's INTERLEAVED one (left_hip_pitch, right_hip_pitch, waist_yaw, ...), which
            differs from the URDF's per-limb order in 27 of 29 positions.
        joints: ``(F, nj)`` predicted joint angles.
        ori6: ``(F, 6)`` predicted base rotation.
        base: ``(F, 3)`` predicted base position.

    Returns:
        ``{link: (F, 3)}`` in the same frame as ``base``.
    """
    rot = rot6d_to_matrix(ori6)
    out = {k: np.zeros((len(joints), 3)) for k in _FEET}
    for t in range(len(joints)):
        urdf.update_cfg(joints[t][perm])
        for link in _FEET:
            local = urdf.get_transform(link, urdf.base_link)[:3, 3]
            out[link][t] = base[t] + rot[t] @ local
    return out


def polyline(pts: np.ndarray) -> np.ndarray:
    """``(F, 3)`` points -> the ``(F-1, 2, 3)`` segment array viser wants."""
    return np.stack([pts[:-1], pts[1:]], axis=1)


def main() -> int:
    """Serve the probe viewer."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viz", required=True, help=".npz from fm_terrain_probe.py --dump_viz")
    p.add_argument("--urdf", default=_DEFAULT_URDF,
                   help="Robot to draw and to forward-kinematic the feet with.")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()

    z = np.load(a.viz, allow_pickle=True)
    conds = [str(c) for c in z["conditions"]]
    cams = [str(c) for c in z["cameras"]]
    depth, plans = z["depth"], z["plans"]               # (C, cam, H, W), (C, F, 3)
    band = float(z["clip_band"])
    local = json.loads(str(z["conditions_local"]))
    probe = json.loads(str(z["probe"]))

    # The plan's base position is relative to the PELVIS at the probe instant, while the
    # decor and the ground grid are referenced to the ground. Lift the plans by the pelvis
    # height so everything shares one frame -- without it the whole trajectory floats 0.78 m
    # low and the feet would appear to be underground. Checked: after the lift the FK feet
    # land 0.03-0.19 m above the ground, i.e. a stance foot (the ankle frame sits 0.034 above
    # the sole) and a swing foot.
    plans = plans.copy()
    plans[..., 2] += float(probe["pose"]["root_z"])

    urdf = load_urdf(a.urdf)
    pose = probe["pose"]
    # TWO permutations, because there are two joint orderings in play and they differ in 27
    # of 29 positions. The CLIP's ``pos_output_names`` is per-limb (left_hip_pitch,
    # left_hip_roll, left_hip_yaw, left_knee, ...); the FM's ``ctx.joint_names`` is Isaac's
    # interleaved one (left_hip_pitch, right_hip_pitch, waist_yaw, ...). Using one order's
    # names to permute the other's values is silent -- it produces a pose, just not this one.
    # Measured when it happened here: left_hip_roll came out at +1.161 rad, a 66 degree
    # splay, with the left ankle 0.66 m out sideways.
    perm_pose = recording_to_urdf_perm(list(pose["joint_names"]), urdf)
    perm_plan = recording_to_urdf_perm([str(j) for j in z["joint_names"]], urdf)

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=16.0, height=16.0, cell_size=0.5)
    # The robot AS IT STANDS at the probe instant. Its roll and pitch are kept (the frame is
    # only yaw-aligned), which matters here more than anywhere: the torso lean is what
    # decides where the cameras point, so drawing it upright would show a scene the cameras
    # never had.
    robot = ViserUrdf(server, urdf, root_node_name="/robot")
    server.scene.add_frame("/robot", show_axes=False,
                           position=(0.0, 0.0, float(pose["root_z"])),
                           wxyz=tuple(float(v) for v in pose["root_quat_wxyz"]))
    robot.update_cfg(np.asarray(pose["joint_pos"], dtype=np.float64)[perm_pose])
    server.scene.add_frame("/origin", axes_length=0.4, axes_radius=0.012,
                           position=(0.0, 0.0, 0.0))
    feet_ctrl = foot_paths(urdf, perm_plan, z["plan_joints"][0], z["plan_ori6"][0], plans[0])
    server.scene.add_line_segments(
        "/plan_control", points=polyline(plans[0]),
        colors=np.full((len(plans[0]) - 1, 2, 3), _CTRL_COLOR, np.uint8), line_width=9.0)
    for link, pts in feet_ctrl.items():
        server.scene.add_line_segments(
            f"/foot_control_{link}", points=polyline(pts),
            colors=np.full((len(pts) - 1, 2, 3), _CTRL_COLOR, np.uint8), line_width=5.0)

    gui = server.gui
    gui.add_markdown(
        f"### terrain probe\n"
        f"clip **{probe['clip_id']}** step **{probe['start_step']}**, "
        f"{probe['speed']:.2f} m/s, torso **{probe['torso_pitch_deg']:+.1f}deg** nose-down "
        f"(far camera meets the ground near **{probe['far_cam_ground_m']:.1f} m**).\n\n"
        f"Grey = no decor, orange = this condition. Thick = base path, thin = feet (forward kinematics of the predicted joints -- the generator outputs base pose and joints only, never foot positions).")
    sel = gui.add_slider("condition", min=1, max=len(conds) - 1, step=1, initial_value=1)
    name = gui.add_text("name", initial_value=conds[1], disabled=True)
    readout = gui.add_markdown("")
    show_ctrl = gui.add_checkbox("show the no-decor plan", initial_value=True)
    show_box = gui.add_checkbox("show the decor", initial_value=True)
    show_feet = gui.add_checkbox("show the FK foot paths", initial_value=True)

    views = {}
    with gui.add_folder("depth: no decor"):
        for c in cams:
            views[f"ctrl_{c}"] = gui.add_image(np.zeros((8, 8, 3), np.uint8), label=c)
    with gui.add_folder("depth: this condition"):
        for c in cams:
            views[f"cond_{c}"] = gui.add_image(np.zeros((8, 8, 3), np.uint8), label=c)
    with gui.add_folder("depth: difference (inferno, 0..2)"):
        for c in cams:
            views[f"diff_{c}"] = gui.add_image(np.zeros((8, 8, 3), np.uint8), label=c)

    handles: list = []

    def redraw(_=None) -> None:
        """Rebuild every condition-dependent node for the selected condition."""
        i = int(sel.value)
        name.value = conds[i]
        for h in handles:
            h.remove()
        handles.clear()

        if show_box.value:
            for j, b in enumerate(local[i]):
                handles.append(server.scene.add_mesh_simple(
                    f"/decor_{j}", **_mesh_kwargs(box_mesh(b))))
        handles.append(server.scene.add_line_segments(
            "/plan_condition", points=polyline(plans[i]),
            colors=np.full((plans.shape[1] - 1, 2, 3), _COND_COLOR, np.uint8), line_width=9.0))
        if show_feet.value:
            for link, pts in foot_paths(urdf, perm_plan, z["plan_joints"][i], z["plan_ori6"][i],
                                        plans[i]).items():
                handles.append(server.scene.add_line_segments(
                    f"/foot_condition_{link}", points=polyline(pts),
                    colors=np.full((len(pts) - 1, 2, 3), _COND_COLOR, np.uint8),
                    line_width=5.0))

        for k, c in enumerate(cams):
            views[f"ctrl_{c}"].image = depth_rgb(depth[0, k], band)
            views[f"cond_{c}"].image = depth_rgb(depth[i, k], band)
            views[f"diff_{c}"].image = diff_rgb(depth[i, k], depth[0, k])

        d = np.linalg.norm(plans[i] - plans[0], axis=-1)
        seen = [f"{c} {np.abs(depth[i, k] - depth[0, k]).mean():.4f}" for k, c in enumerate(cams)]
        readout.content = (
            f"**{conds[i]}**\n\n"
            f"plan moved: mean **{d.mean():.3f} m**, peak **{d.max():.3f} m**, "
            f"at the far node **{d[-1]:.3f} m**\n\n"
            f"depth changed: {', '.join(seen)}\n\n"
            f"{'*the cameras saw nothing here -- the plan difference is not about geometry*' if max(np.abs(depth[i, k] - depth[0, k]).mean() for k in range(len(cams))) < 1e-3 else ''}")

    def toggle_ctrl(_=None) -> None:
        """The control plan is a fixed node, so it only needs showing or hiding."""
        server.scene.add_line_segments(
            "/plan_control", points=polyline(plans[0]),
            colors=np.full((plans.shape[1] - 1, 2, 3), _CTRL_COLOR, np.uint8),
            line_width=9.0, visible=bool(show_ctrl.value))

    redraw()
    toggle_ctrl()
    sel.on_update(redraw)
    show_box.on_update(redraw)
    show_feet.on_update(redraw)
    show_ctrl.on_update(toggle_ctrl)

    print(f"[view_terrain_probe] {len(conds)} conditions; open http://localhost:{a.port}")
    while True:
        time.sleep(1.0)


def _mesh_kwargs(m: trimesh.Trimesh) -> dict:
    """viser mesh arguments for one trimesh, translucent so a plan behind it stays visible."""
    return {"vertices": np.asarray(m.vertices), "faces": np.asarray(m.faces),
            "color": _BOX_COLOR, "opacity": 0.55, "side": "double", "flat_shading": True}


if __name__ == "__main__":
    raise SystemExit(main())
