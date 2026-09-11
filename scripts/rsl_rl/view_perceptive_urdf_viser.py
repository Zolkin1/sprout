#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Interactive viser view of the perceptive G1: the robot, its camera mounts, and the
coordinate frame each camera actually casts rays from.

The point is a SANITY CHECK on the mount CAD, and specifically on its DATUM. The URDF pins
each mount mesh's own origin to the declared raycast point -- the design intent is that the
STL origin sits on the left imager, which is the frame a stereo depth camera reports depth
in -- so what this viewer shows is the mesh exactly where the numbers say, with nothing
derived and nothing fitted. If the export's datum is right, the drawn triad sits inside the
left lens. If it is not, the mount hangs off the robot by the datum error, in metres, which
is the whole point: the fix is a re-export, not a fudge factor here.

What is drawn:

* the robot from ``robot_assets/g1_perceptive/g1_perceptive.urdf`` (mounts included);
* a **big triad** at each ``*_optical_link`` -- the ray origin, in the SIM's convention:
  red = boresight (+x), green = +y (left), blue = +z (up);
* a **small triad** at each ``*_mount_link`` -- the STL's OWN origin and axes, at the same
  position, rotated into the CAD's z-forward convention. The two coincide by construction,
  so this is how to read the mesh's coordinates off the screen;
* **magenta dots** at the two imager-sized solids found inside each assembly, with the
  gap between the nearer one and the ray origin printed. Zero means the datum is on the
  camera. This is a REPORT, not an input -- placement does not use it (see
  ``make_perceptive_urdf``'s module docstring for why);
* **yellow spheres** at each welded payload's centre of mass, labelled with its mass --
  the payload masses are otherwise invisible in exactly the view meant to
  sanity-check them (the backpack SHELL does now have a mesh, ASSY_backpack.STL, but
  its mass sits on separate thor/zed links that have none);
* a **boresight ray** from each camera to where it meets the ground plane, with the range
  printed in the GUI, so "the far camera looks 4.2 m ahead" is checkable rather than
  asserted;
* the **frustum** edges at the trained field of view, so the coverage overlap between the
  two cameras is visible.

IsaacLab-free: yourdfpy + viser only. Run under the full-stack env (it has viser)::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
        env_isaaclab/bin/python scripts/rsl_rl/view_perceptive_urdf_viser.py
    # then open the printed http://localhost:8080 URL
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from functools import partial

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from robot_rl.flow_matching.viz.robot import DEFAULT_URDF  # noqa: E402

_DEFAULT_URDF = os.path.join(_REPO, DEFAULT_URDF)

#: Trained field of view per camera, AFTER the edge crop (``g1_depth_env_cfg`` fov_h/fov_v
#: minus twice fov_crop_deg). Quoted here rather than imported because that module pulls in
#: IsaacLab; ``--fov`` overrides when they are retuned.
_FOV_DEG = {"camera_lower": (100.0, 70.0), "camera_upper": (70.0, 42.0)}

#: Ground plane height in the pelvis frame -- the sole at the nominal stance
#: (``extero_nominal_ground_z``). Only used to draw where a boresight lands.
_GROUND_Z = -0.7869

_COLOURS = {"camera_lower": (255, 140, 40), "camera_upper": (60, 170, 255)}


def _frustum_rays(fov_h_deg: float, fov_v_deg: float) -> np.ndarray:
    """Unit vectors along the four frustum corners, in the optical frame (+x forward)."""
    th, tv = (np.tan(np.radians(f) / 2.0) for f in (fov_h_deg, fov_v_deg))
    corners = np.array([[1.0, +th, +tv], [1.0, -th, +tv],
                        [1.0, -th, -tv], [1.0, +th, -tv]])
    return corners / np.linalg.norm(corners, axis=1, keepdims=True)


#: The sim's nominal standing pose (``G1_CFG.init_state.joint_pos``), by joint-name suffix.
#: Quoted rather than imported for the same reason as the FOVs -- that module needs IsaacLab.
_STANCE_RAD = {
    "_hip_pitch_joint": -0.25, "_knee_joint": 0.46, "_ankle_pitch_joint": -0.25,
    "_shoulder_pitch_joint": 0.31, "_elbow_joint": 0.8,
    "left_shoulder_roll_joint": 0.24, "right_shoulder_roll_joint": -0.24,
}


def _configuration(urdf: yourdfpy.URDF, stance: bool) -> np.ndarray:
    """Joint vector in ``urdf.actuated_joint_names`` order.

    Args:
        urdf: The loaded robot.
        stance: ``True`` for the sim's nominal standing pose, ``False`` for all-zeros.

    Returns:
        The configuration to render.

    Note:
        Zeros is the DEFAULT deliberately: the depth contract's nominal ground plane is
        derived at the zero configuration (``extero_nominal_ground_z``), so a camera drawn
        against a zero-pose robot is being compared against the same datum the FM uses.
        ``--stance`` shows the crouch the robot actually trains in, which sits ~6 cm lower.
    """
    names = list(urdf.actuated_joint_names)
    if not stance:
        return np.zeros(len(names))
    out = np.zeros(len(names))
    for i, n in enumerate(names):
        for key, val in _STANCE_RAD.items():
            if n == key or n.endswith(key):
                out[i] = val
                break
    return out


def _ground_hit(origin: np.ndarray, direction: np.ndarray, ground_z: float) -> float | None:
    """Distance along ``direction`` until ``ground_z``, or ``None`` if it never gets there."""
    if direction[2] >= -1e-9:
        return None
    return float((ground_z - origin[2]) / direction[2])


def _load_builder(urdf_path: str):
    """Import ``make_perceptive_urdf`` from the URDF's own directory, or ``None``.

    By path rather than by name: the builder lives beside the asset it writes and is not on
    any package path, and deriving it from the URDF keeps the two in step if the asset moves.
    ``None`` (not an exception) when it is unavailable -- the imager markers are a diagnostic
    overlay, and the robot is worth drawing without them.

    Args:
        urdf_path: The URDF being viewed.

    Returns:
        The loaded module, or ``None`` if it or its dependencies are missing.
    """
    path = os.path.join(os.path.dirname(urdf_path), "make_perceptive_urdf.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("make_perceptive_urdf", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE executing: @dataclass resolves its own module via
    # sys.modules[cls.__module__], so a module that is not there yet dies on the first
    # decorated class with a bare "'NoneType' object has no attribute '__dict__'".
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # noqa: BLE001 - overlay is optional
        sys.modules.pop(spec.name, None)
        print(f"[view] imager markers unavailable: {exc}")
        return None
    return mod


def _mesh_path(urdf: yourdfpy.URDF, link: str, urdf_path: str) -> str | None:
    """Absolute path of ``link``'s first visual mesh, or ``None`` if it has none."""
    visuals = getattr(urdf.link_map[link], "visuals", None) or []
    for v in visuals:
        mesh = getattr(getattr(v, "geometry", None), "mesh", None)
        if mesh is not None and mesh.filename:
            return os.path.join(os.path.dirname(urdf_path), mesh.filename)
    return None


def _draw_imagers(server: viser.ViserServer, builder, urdf: yourdfpy.URDF, cam: str,
                  mount_T: np.ndarray, ray_origin: np.ndarray, urdf_path: str) -> float | None:
    """Mark the imager-sized solids inside ``cam``'s mesh; return the nearest one's distance.

    Purely diagnostic. The mount is placed from the declared raycast point, so these markers
    cannot move anything -- they answer "did the CAD datum land on the camera?", which is
    the one question the pinned placement can no longer answer for itself.

    Args:
        server: The viser server to draw into.
        builder: The loaded ``make_perceptive_urdf`` module, or ``None`` to skip.
        urdf: The loaded robot, for the mesh path.
        cam: Camera name prefix, e.g. ``camera_lower``.
        mount_T: ``4x4`` pose of ``<cam>_mount_link`` in the frame being drawn.
        ray_origin: Where rays actually start, in the same frame.
        urdf_path: The URDF file, for resolving the mesh's relative filename.

    Returns:
        Distance (m) from ``ray_origin`` to the nearest imager centre, or ``None`` if the
        search could not run -- which is a fact about the export, not an error.
    """
    if builder is None:
        return None
    mesh_path = _mesh_path(urdf, f"{cam}_mount_link", urdf_path)
    if mesh_path is None:
        return None
    try:
        centres, baseline, _ = builder.find_camera_pair(mesh_path)
    except Exception as exc:                       # noqa: BLE001 - reported, never fatal
        print(f"[view] {cam}: no stereo pair found in the mesh ({exc})")
        return None
    world = np.array([mount_T[:3, :3] @ c + mount_T[:3, 3] for c in centres])
    server.scene.add_point_cloud(
        f"/imagers/{cam}", points=world.astype(np.float32),
        colors=np.full((len(world), 3), (255, 0, 200), np.uint8), point_size=0.012,
    )
    gap = float(np.min(np.linalg.norm(world - ray_origin, axis=1)))
    print(f"[view] {cam}: stereo pair baseline {baseline * 1000:.1f} mm; nearest imager "
          f"{gap * 1000:.1f} mm from the ray origin")
    return gap


def _draw_payloads(server: viser.ViserServer, builder, urdf: yourdfpy.URDF,
                   base: str) -> list[str]:
    """Mark each welded payload's centre of mass, and report the mass it adds.

    A payload with no visual mesh is INVISIBLE in the robot render while still being 2 kg
    of the robot -- which is the whole reason to draw it. The marker is the link origin,
    which is the centre of mass by construction (see :class:`PayloadSpec`).

    Args:
        server: The viser server to draw into.
        builder: The loaded ``make_perceptive_urdf`` module, or ``None`` to skip.
        urdf: The loaded robot.
        base: Frame to draw in.

    Returns:
        One report line per payload, for the GUI panel.
    """
    if builder is None:
        return []
    lines = []
    for pl in getattr(builder, "PAYLOADS", ()):
        link = f"{pl.name}_link"
        if link not in urdf.link_map:
            continue
        pos = urdf.get_transform(link, base)[:3, 3]
        server.scene.add_icosphere(f"/payload/{pl.name}", radius=0.022,
                                   color=(230, 200, 40), opacity=0.85,
                                   position=tuple(float(v) for v in pos))
        server.scene.add_label(f"/payload_label/{pl.name}",
                               text=f"{pl.name} {pl.mass_kg * 1000:.0f} g",
                               position=tuple(float(v) for v in pos + np.array([0, 0, 0.05])))
        est = " (inertia box ESTIMATED)" if pl.size_is_estimate else ""
        lines.append(f"{pl.name}: {pl.mass_kg * 1000:.0f} g at "
                     f"({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m in {base}{est}")
        print(f"[view] {lines[-1]}")
    return lines


def main() -> None:
    """Parse args, build the scene, and serve it."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--urdf", default=_DEFAULT_URDF)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--base", default="pelvis_link",
                   help="Frame everything is drawn in (the URDF root link).")
    p.add_argument("--ground_z", type=float, default=_GROUND_Z,
                   help="Ground height in the base frame, for the boresight ray.")
    p.add_argument("--ray_len", type=float, default=12.0,
                   help="Length of a boresight that never meets the ground (m).")
    p.add_argument("--stance", action="store_true",
                   help="Render the sim's nominal standing pose instead of all-zero joints "
                        "(the zero pose is the datum the depth contract is derived at).")
    args = p.parse_args()

    print(f"[view] loading {args.urdf}")
    # Same mesh resolution the other viewers use (robot_rl.flow_matching.viz.robot.load_urdf):
    # yourdfpy calls the handler with a ``fname`` KEYWORD, so a bare lambda(fn) raises.
    urdf = yourdfpy.URDF.load(
        args.urdf, build_scene_graph=True, load_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic,
                                 dir=os.path.dirname(args.urdf)),
    )
    cams = sorted(n[: -len("_optical_link")] for n in urdf.link_map
                  if n.endswith("_optical_link"))
    if not cams:
        raise SystemExit(f"{args.urdf} has no *_optical_link; is this the perceptive URDF?")
    print(f"[view] cameras: {', '.join(cams)}")

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True
    vurdf = ViserUrdf(server, urdf, root_node_name="/robot")
    # REQUIRED. ViserUrdf creates the per-link nodes but leaves them at identity until a
    # configuration is pushed, so without this every limb -- and every mount mesh -- renders
    # piled at the root: a floating torso with the camera brackets buried in it or flung out
    # by their own mesh-internal offset. The URDF is fine; the scene graph just has no pose.
    cfg = _configuration(urdf, args.stance)
    vurdf.update_cfg(cfg)

    # A ground quad, purely for scale and to make the boresight landing legible.
    g = 6.0
    server.scene.add_mesh_simple(
        "/ground",
        vertices=np.array([[-g, -g, args.ground_z], [g, -g, args.ground_z],
                           [g, g, args.ground_z], [-g, g, args.ground_z]], np.float32),
        faces=np.array([[0, 1, 2], [0, 2, 3]], np.int64),
        color=(0.55, 0.57, 0.60), opacity=0.35, side="double",
    )

    builder = _load_builder(args.urdf)
    lines: list[str] = []
    for cam in cams:
        T = urdf.get_transform(f"{cam}_optical_link", args.base)
        R, o = T[:3, :3], T[:3, 3]
        colour = _COLOURS.get(cam, (200, 200, 200))

        server.scene.add_frame(f"/optical/{cam}", show_axes=True, axes_length=0.18,
                               axes_radius=0.005, position=tuple(float(v) for v in o),
                               wxyz=tuple(float(v) for v in _mat_to_wxyz(R)))
        # The STL's OWN frame: same origin, rotated into the CAD's z-forward convention.
        # Coincident by construction -- the mesh is PINNED to the raycast point -- so this
        # triad is how to read a mesh coordinate off the screen, not a second datum.
        Tm = urdf.get_transform(f"{cam}_mount_link", args.base)
        server.scene.add_frame(f"/stl_origin/{cam}", show_axes=True, axes_length=0.07,
                               axes_radius=0.003,
                               position=tuple(float(v) for v in Tm[:3, 3]),
                               wxyz=tuple(float(v) for v in _mat_to_wxyz(Tm[:3, :3])))
        gap = _draw_imagers(server, builder, urdf, cam, Tm, o, args.urdf)

        bore = R[:, 0]
        hit = _ground_hit(o, bore, args.ground_z)
        reach = hit if hit is not None else args.ray_len
        server.scene.add_line_segments(
            f"/boresight/{cam}",
            points=np.array([[o, o + bore * reach]], np.float32),
            colors=np.array([[colour, colour]], np.uint8), line_width=4.0,
        )

        fov = _FOV_DEG.get(cam)
        if fov:
            rays = _frustum_rays(*fov) @ R.T
            ends, segs = [], []
            for r in rays:
                d = _ground_hit(o, r, args.ground_z)
                ends.append(o + r * (d if d is not None else args.ray_len))
            for i in range(4):
                segs.append([o, ends[i]])
                segs.append([ends[i], ends[(i + 1) % 4]])
            server.scene.add_line_segments(
                f"/frustum/{cam}", points=np.array(segs, np.float32),
                colors=np.full((len(segs), 2, 3), colour, np.uint8), line_width=1.5,
            )
        pitch = np.degrees(np.arctan2(-bore[2], bore[0]))
        where = f"ground at {hit:.2f} m" if hit is not None else "never meets the ground"
        # Spelled out in the scene, because a triad is ambiguous: this frame is IsaacLab's
        # ray-caster convention (+x = optical axis, +y = left, +z = up), NOT the OpenCV /
        # OpenGL one where the camera looks down z. Read the BLUE axis as "forward" here and
        # a correctly-aimed camera looks like it is pointing up and at the wrong angle.
        server.scene.add_label(f"/label/{cam}",
                               text=f"{cam}: RED=+x=boresight, {pitch:.0f} deg DOWN",
                               position=tuple(float(v) for v in o + np.array([0, 0, 0.09])))
        # Same thresholds the builder prints against, read from it rather than restated --
        # two tolerances for one fact is how they drift apart.
        ok = getattr(builder, "_DATUM_OK_MM", 0.5) * 1e-3
        close = getattr(builder, "_DATUM_CLOSE_MM", 3.0) * 1e-3
        datum = ("datum NOT CHECKED (no stereo pair found in the mesh)" if gap is None else
                 "STL datum is on the nearest imager" if gap <= ok else
                 f"STL datum is {gap * 1000:.1f} mm from the nearest imager, within tolerance"
                 if gap <= close else
                 f"**STL datum is {gap * 1000:.1f} mm OFF the nearest imager**")
        lines.append(f"{cam}: origin ({o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f}) m in "
                     f"{args.base}, pitch {pitch:.2f} deg, boresight {where}; {datum}")
        print(f"[view] {lines[-1]}")

    lines += _draw_payloads(server, builder, urdf, args.base)

    server.gui.add_markdown("### perceptive G1\n" + "\n\n".join(f"- {ln}" for ln in lines)
                            + f"\n\nGround drawn at z = {args.ground_z:+.4f} m."
                            + "\n\nBig triad = optical frame, where rays start "
                              "(RED +x = boresight, GREEN +y = left, BLUE +z = up)."
                            + "\n\nSmall triad = the STL's own axes, same origin, in the "
                              "CAD's z-forward convention."
                            + "\n\nMagenta dots = the imager-sized solids found inside each "
                              "assembly. They are a REPORT on the export's datum; nothing is "
                              "placed from them."
                            + "\n\nYellow spheres = welded payload centres of mass. A payload "
                              "with no mesh yet is otherwise invisible while still being part "
                              "of the robot's mass.")
    # get_port(), not args.port: viser silently falls back to the next free port when the
    # requested one is taken, and printing the requested one sends you to the wrong tab.
    print(f"[view] serving on http://localhost:{server.get_port()} -- Ctrl-C to stop")
    while True:
        time.sleep(1.0)


def _mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), viser's convention."""
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    return np.array(q, dtype=np.float64)


if __name__ == "__main__":
    main()
