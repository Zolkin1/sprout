# Copyright 2026 Zachary Olkin. All rights reserved.

"""Verify the MuJoCo depth cameras: coverage, self-occlusion, articulation, geometry.

The MuJoCo counterpart of ``scripts/rsl_rl/verify_depth_self_scan.py``, which checks the
same properties on the IsaacLab side. Four checks, in order of how much they pin down:

1. **Geometry** -- on flat ground, every floor pixel is compared against the closed-form
   ray/``z=0`` intersection. Catches a wrong mount pose, a transposed rotation, or
   ray-length-vs-image-plane-depth confusion.
2. **Coverage** -- per-camera value range, how many pixels read "no reading", and the
   ground footprint each camera actually sees.
3. **Self-occlusion** -- a per-body census of which links appear in each image. The robot
   MUST see itself; that is what the self-scan targets are for.
4. **Articulation** -- bending the knees must change the census, proving the images track
   live kinematics rather than a frozen pose.

The camera contract comes from an exported ``tracker.onnx`` when one is given (the real
deployment path), otherwise from the training env cfg via ``--from_env_cfg`` (which needs
isaaclab importable, but no simulator).

``--from_env_cfg`` reads the LIVE ``G1-teacher-clf-terrain-depth`` cfgs
(``G1TeacherTerrainDepthObservationsCfg`` policy group + ``G1TeacherTerrainDepthSceneCfg``,
i.e. ``g1_depth_env_cfg``'s ``_NEAR`` / ``_FAR``), so it always reflects what the task
currently declares.  The robot geometry is separate and comes from the MuJoCo model
(``--robot_name`` / ``--scene``, default ``robots/g1/g1_perceptive_basic_scene.xml``).

``--viewer`` opens the MuJoCo viewer with the live hit cloud (yellow = near, cyan-green =
far) plus each camera's mount frame as RGB axes -- red is +x, the optical axis, which is
the quickest way to eyeball a mount pitch.  No simulator, no policy and no trained
checkpoint are involved in either mode.

Run from ``transfer/sim``:

    python verify_depth_cameras.py --from_env_cfg
    python verify_depth_cameras.py --load_run <run> --scene terrain_scene --save_dir /tmp/depth
    python verify_depth_cameras.py --from_env_cfg --viewer
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import mujoco
import numpy as np

_SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SIM)
sys.path.insert(0, os.path.dirname(_SIM))  # transfer/

from deploy.depth_obs import denormalize_depth_np, invalid_depth_value  # noqa: E402
from depth_camera import MujocoDepthCameras  # noqa: E402

ROOT = os.path.abspath(os.path.join(_SIM, "..", ".."))


def spec_from_tracker(run_dir: str) -> list[dict]:
    """Read the depth contract out of a run's exported ``tracker.onnx``.

    Args:
        run_dir: Run directory (its ``exported/tracker.onnx`` is read).

    Returns:
        The per-camera contract.

    Raises:
        FileNotFoundError: If the run has no exported tracker.
        ValueError: If the tracker carries no depth contract.
    """
    from deploy.tracker_session import TrackerSession

    onnx_path = os.path.join(run_dir, "exported", "tracker.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"{onnx_path} not found; export it with scripts/rsl_rl/export_tracker_onnx.py."
        )
    spec = TrackerSession(onnx_path).get_depth_camera_spec()
    if not spec:
        raise ValueError(
            f"{onnx_path} has no depth_camera_spec -- either the policy has no cameras, or "
            "it was exported before the contract existed (re-export it)."
        )
    return spec


def spec_from_env_cfg() -> list[dict]:
    """Build the depth contract from the training env cfg, without a simulator.

    Goes through ``g1_depth_env_cfg.depth_camera_specs_from_cfg``, which instantiates the
    REAL observation and scene cfgs, so this reflects whatever the task currently declares
    -- no camera number, term name or clip is restated here.

    Returns:
        The per-camera contract for ``G1-teacher-clf-terrain-depth``.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        depth_camera_specs_from_cfg,
    )

    return depth_camera_specs_from_cfg()


def denormalise(image: np.ndarray, cam: dict) -> np.ndarray:
    """Recover metres from a normalised image, via the deploy runtime's own inverse."""
    return denormalize_depth_np(image, float(cam["offset"]), float(cam["scale"]))


def draw_frame_axes(scn, pos: np.ndarray, rot: np.ndarray,
                    length: float = 0.15, radius: float = 0.006) -> None:
    """Draw a coordinate frame as three RGB arrows in a viewer scene.

    Red / green / blue are the frame's +x / +y / +z.  For a depth camera +x is the
    optical axis, so the red arrow is where the camera actually looks -- the quickest way
    to sanity-check a mount pitch by eye.

    An ``mjGEOM_ARROW`` points along its own local +z, so each axis is drawn with the
    frame's columns cyclically permuted to put the wanted one third.  All three
    permutations are cyclic, hence still right-handed (det +1) and safe as rotations.

    Args:
        scn: The viewer's ``user_scn`` to append geoms to.
        pos: ``(3,)`` frame origin in world coordinates.
        rot: ``(3, 3)`` frame orientation, columns = the frame's axes in world.
        length: Arrow length, metres.
        radius: Arrow shaft radius, metres.
    """
    size = np.array([radius, radius, length])
    for axis, rgba in enumerate(((1.0, 0.2, 0.2, 1.0),      # +x, optical axis
                                 (0.2, 1.0, 0.2, 1.0),      # +y
                                 (0.3, 0.45, 1.0, 1.0))):   # +z
        if scn.ngeom >= scn.maxgeom:
            return
        mat = rot[:, [(axis + 1) % 3, (axis + 2) % 3, axis]]
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW,
            size, pos, np.ascontiguousarray(mat).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1


def check_geometry(model, data, cams: MujocoDepthCameras, spec: list[dict]) -> bool:
    """Compare floor pixels against the analytic ray/ground intersection.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (forward kinematics current).
        cams: The camera caster.
        spec: The per-camera contract.

    Returns:
        True if every camera agrees with the closed form (or had no floor pixels).
    """
    print("\n== geometry: cast vs the analytic ground plane (flat ground only) ==")
    images, hits = cams.compute(data, want_hits=True)
    ok = True
    for i, (cam, camera) in enumerate(zip(spec, cams.cameras)):
        pos, rot = camera.pose(data)
        dirs_w = camera.dirs_cam @ rot.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -pos[2] / dirs_w[:, 2]
        on_floor = (dirs_w[:, 2] < 0) & (t > 0) & np.isfinite(hits[i, :, 2]) \
            & (np.abs(hits[i, :, 2]) < 1e-6)
        if not on_floor.any():
            print(f"  {camera.sensor:20s} no floor pixels (not a flat scene?) -- skipped")
            continue
        want = np.minimum(t[on_floor] * camera.cos_theta[on_floor], camera.max_distance)
        err = float(np.abs(denormalise(images[i][on_floor], cam) - want).max())
        status = "ok  " if err < 1e-6 else "FAIL"
        ok &= err < 1e-6
        print(f"  {status} {camera.sensor:20s} {int(on_floor.sum()):4d} floor px, "
              f"max abs error {err:.2e} m")
    return ok


def report_coverage(data, cams: MujocoDepthCameras, spec: list[dict]) -> None:
    """Print each camera's value range, invalid-pixel count and ground footprint."""
    print("\n== coverage ==")
    images, hits = cams.compute(data, want_hits=True)
    for i, (cam, camera) in enumerate(zip(spec, cams.cameras)):
        image = images[i]
        n_invalid = int(np.isclose(image, invalid_depth_value(cam)).sum())
        metres = denormalise(image, cam)
        valid = ~np.isclose(image, invalid_depth_value(cam))
        pos, _ = camera.pose(data)
        floor = hits[i][np.isfinite(hits[i, :, 0])]
        print(f"  {camera.sensor:20s} at [{pos[0]:6.3f} {pos[1]:6.3f} {pos[2]:6.3f}] "
              f"({camera.width}x{camera.height})")
        if valid.any():
            print(f"    depth   {metres[valid].min():5.2f} .. {metres[valid].max():5.2f} m "
                  f"(normalised {image.min():6.3f} .. {image.max():6.3f})")
        print(f"    invalid {n_invalid:4d}/{camera.num_rays} px "
              f"({100.0 * n_invalid / camera.num_rays:.1f}%)  "
              f"saturated at the obs clip: {int((np.abs(image) >= 2.0).sum())} px")
        if len(floor):
            print(f"    sees x {floor[:, 0].min():6.3f} .. {floor[:, 0].max():6.3f} m, "
                  f"y {floor[:, 1].min():6.3f} .. {floor[:, 1].max():6.3f} m, "
                  f"z {floor[:, 2].min():6.3f} .. {floor[:, 2].max():6.3f} m")


def census(model, data, cams: MujocoDepthCameras) -> list[collections.Counter]:
    """Count, per camera, how many pixels landed on each body (or terrain / nothing).

    Reads the hit provenance of the caster's OWN cast (``camera.hit`` / ``camera.geom_ids``)
    rather than issuing a second one, so the census can only ever describe the rays the
    policy actually sees.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        cams: The camera caster.

    Returns:
        One counter per camera, keyed by body name plus ``TERRAIN`` and ``NO READING``.
    """
    cams.compute(data)
    out = []
    for camera in cams.cameras:
        counter = collections.Counter()
        for ray in range(camera.num_rays):
            if not camera.hit[ray]:
                counter["NO READING"] += 1
                continue
            geom = int(camera.geom_ids[ray])
            if int(model.geom_group[geom]) == 2:
                counter["TERRAIN"] += 1
            else:
                counter[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(model.geom_bodyid[geom]))] += 1
        out.append(counter)
    return out


def report_census(counters, cams, label: str) -> None:
    """Print a per-body pixel census for every camera."""
    print(f"\n== self-occlusion census ({label}) ==")
    for camera, counter in zip(cams.cameras, counters):
        bodies = {k: v for k, v in counter.items() if k not in ("TERRAIN", "NO READING")}
        print(f"  {camera.sensor:20s} terrain={counter['TERRAIN']:4d} "
              f"no-reading={counter['NO READING']:4d} self={sum(bodies.values()):4d}")
        for body, count in sorted(bodies.items(), key=lambda kv: -kv[1]):
            print(f"      {count:4d}  {body}")


def save_pngs(data, cams, spec, out_dir: str) -> None:
    """Write one grayscale PNG per camera: brighter is nearer, invalid pixels black.

    Stretched over each frame's OWN valid range rather than the sensor's full range --
    a camera looking at ground a metre away uses a few percent of an 8 m range, and
    scaling to the range would render a flat white square.

    Args:
        data: MuJoCo data.
        cams: The camera caster.
        spec: The per-camera contract.
        out_dir: Directory to write the PNGs into.
    """
    try:
        from PIL import Image
    except ImportError:
        print(f"[WARN] Pillow not installed; skipping --save_dir {out_dir}.")
        return
    os.makedirs(out_dir, exist_ok=True)
    images = cams.compute(data)
    for i, (cam, camera) in enumerate(zip(spec, cams.cameras)):
        invalid = np.isclose(images[i], invalid_depth_value(cam))
        metres = denormalise(images[i], cam)
        lo, hi = (metres[~invalid].min(), metres[~invalid].max()) if (~invalid).any() else (0.0, 1.0)
        norm = np.clip((metres - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        png = (255.0 * (1.0 - norm)).astype(np.uint8)   # brighter = nearer
        png[invalid] = 0                                 # "no reading" -> black
        path = os.path.join(out_dir, f"{camera.sensor}.png")
        Image.fromarray(png.reshape(camera.height, camera.width), mode="L").resize(
            (camera.width * 12, camera.height * 12), Image.NEAREST).save(path)
        print(f"  wrote {path}  ({lo:.2f}..{hi:.2f} m stretched to white..black, "
              f"invalid black)")


def main() -> None:
    """Parse arguments, run the checks, and exit non-zero on failure."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--load_run", default=None,
                        help="Run name under logs/rsl_rl/<experiment>/ whose tracker.onnx "
                             "carries the camera contract (the deployment path).")
    source.add_argument("--from_env_cfg", action="store_true",
                        help="Take the contract straight from the training env cfg "
                             "(needs isaaclab importable; no simulator, no export).")
    parser.add_argument("--experiment", default="g1")
    parser.add_argument("--robot_name", default="g1_perceptive")
    parser.add_argument("--scene", default="basic_scene",
                        help="MuJoCo scene; the geometry check needs a flat one.")
    parser.add_argument("--knee_angle", type=float, default=1.2,
                        help="Knee angle (rad) for the articulation check.")
    parser.add_argument("--save_dir", default=None, help="Write per-camera depth PNGs here.")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer with the depth hit cloud and each "
                             "camera's mount frame (RGB axes) drawn.")
    parser.add_argument("--depth_noise", action="store_true", default=False,
                        help="Apply the training sensor noise (for eyeballing its effect).")
    args = parser.parse_args()

    spec = (spec_from_env_cfg() if args.from_env_cfg
            else spec_from_tracker(os.path.join(ROOT, "logs", "rsl_rl", args.experiment,
                                                args.load_run)))
    print("== camera contract ==")
    for cam in spec:
        print(f"  {cam.get('sensor', cam['obs_term']):20s} {cam['width']}x{cam['height']} "
              f"on {cam['mount_body']} at {[round(v, 4) for v in cam['offset_pos']]}, "
              f"fx={cam['fx']:.4f} fy={cam['fy']:.4f}, "
              f"range {cam['min_dist']}..{cam['max_dist']} m (sentinel {cam['max_distance']})")

    scene = os.path.join(_SIM, "robots", "g1", f"{args.robot_name}_{args.scene}.xml")
    if not os.path.exists(scene):
        sys.exit(f"[ERROR] Scene not found: {scene} (generate it with generate_stairs_scene.py)")
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"\nscene: {os.path.basename(scene)} "
          f"({model.ngeom} geoms, {int((model.geom_group == 2).sum())} terrain)")

    cams = MujocoDepthCameras(model, spec)
    if args.depth_noise:
        cams.enable_noise(np.random.default_rng(0))
        print("[INFO] training sensor noise applied")

    ok = check_geometry(model, data, cams, spec)
    report_coverage(data, cams, spec)

    base = census(model, data, cams)
    report_census(base, cams, "nominal pose")
    self_pixels = [sum(v for k, v in c.items() if k not in ("TERRAIN", "NO READING")) for c in base]
    if not any(self_pixels):
        print("\n[FAIL] No camera sees any part of the robot -- self-occlusion is broken "
              "(check the visual geom group).")
        ok = False
    elif not self_pixels[0]:
        print("\n[NOTE] The first camera sees no self pixels; only later ones do.")

    for joint in ("left_knee_joint", "right_knee_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = args.knee_angle
    mujoco.mj_forward(model, data)
    bent = census(model, data, cams)
    report_census(bent, cams, f"knees at {args.knee_angle} rad")
    if bent == base:
        print("\n[FAIL] Bending the knees changed nothing -- the images are not tracking "
              "live kinematics.")
        ok = False
    else:
        print(f"\n[ok] The census changed with the knees ({sum(map(len, base))} -> "
              f"{sum(map(len, bent))} distinct hit targets), so limbs articulate in the image.")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    if args.save_dir:
        print("\n== depth images ==")
        save_pngs(data, cams, spec, args.save_dir)

    if args.viewer:
        # Bound to a distinct name: `import mujoco.viewer` here would shadow the
        # module-level `mujoco` for the whole function.
        from mujoco import viewer as mj_viewer

        print("\n[INFO] Viewer: yellow = near camera hits, cyan-green = far camera hits.")
        print("[INFO] Each camera's mount frame is drawn as RGB axes "
              "(red = +x / optical axis, green = +y, blue = +z).")
        for cam, camera in zip(spec, cams.cameras):
            pos, _ = camera.pose(data)
            print(f"         {cam['sensor']:20s} origin {np.array2string(pos, precision=3)}")
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                _, hits = cams.compute(data, want_hits=True)
                scn = viewer.user_scn
                scn.ngeom = 0
                # Axes first, so they are never the geoms dropped by the maxgeom guard.
                for camera in cams.cameras:
                    draw_frame_axes(scn, *camera.pose(data))
                for cam_i, cam_hits in enumerate(hits):
                    rgba = np.array([1.0, 0.85, 0.1, 0.8] if cam_i == 0
                                    else [0.1, 0.9, 0.6, 0.8], dtype=np.float32)
                    for point in cam_hits[::4]:
                        if np.isfinite(point).all() and scn.ngeom < scn.maxgeom:
                            mujoco.mjv_initGeom(
                                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.012, 0.0, 0.0]), point, np.eye(3).flatten(), rgba)
                            scn.ngeom += 1
                viewer.sync()
        os._exit(0)

    print("\n" + ("ALL DEPTH CAMERA CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
