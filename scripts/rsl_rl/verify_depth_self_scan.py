# Copyright 2026 Zachary Olkin. All rights reserved.

"""Verify the depth cameras actually see the robot's own body, and that it MOVES.

The self-scan is easy to get silently wrong.  If the ray-cast targets point at the
outer (kinematic) link Xforms, IsaacLab merges each target's whole subtree into one
rigid mesh frozen at its spawn pose: the depth image still shows a leg-shaped blob,
it just never articulates.  A depth image alone therefore proves nothing.  This
checks the two things that do:

1. **Presence** -- with ``update_mesh_ids`` on, the camera reports per pixel WHICH
   mesh the ray hit.  Index 0 is the terrain; higher indices are robot links, named
   here from the sensor's own target order.  No self-hits means the target
   expressions never resolved.
2. **Articulation** -- the knees are then driven to a large flexion and the per-link
   pixel counts must change.  This is the check a frozen merged blob fails.

It builds a MINIMAL scene -- flat mesh ground, the G1, and the two cameras straight
from :mod:`g1_depth_env_cfg`'s own factory -- rather than a full teacher env, so it
exercises the real sensor configuration without depending on the rest of the task
(commands, clips, rewards, contact sensors).  No trained policy is involved.

Run under the full Isaac env (env_isaaclab).  Note ``python -u``: Kit can hard-exit
and swallow buffered stdout.

Headless pass/fail check::

    PYTHONPATH=$PWD/source/robot_rl:$PWD/source/motion_gen \\
        <env_isaaclab py> -u scripts/rsl_rl/verify_depth_self_scan.py --headless

VISUAL check in the Kit GUI.  ``--viz kit`` is REQUIRED: in IsaacLab 3.0 ``--headless``
is deprecated and omitting ``--viz`` *is* the headless default, so without it the run
is windowless and no markers are drawn::

    PYTHONPATH=$PWD/source/robot_rl:$PWD/source/motion_gen \\
        <env_isaaclab py> -u scripts/rsl_rl/verify_depth_self_scan.py \\
            --viz kit --animate --height_scan --num_envs 1

Ray hits are drawn as spheres, one colour per sensor: **green** = near depth camera,
**blue** = far depth camera, **red** = pelvis height scanner.  Each also gets an
explicit marker prim path; ``VisualizationMarkers`` would otherwise auto-suffix the
shared ``/Visuals/RayCaster`` default, which works but makes the stage harder to read.

``--animate`` turns gravity off, writes the joint targets directly and pins the root,
making the robot a kinematic mannequin: it holds its stance and its place, so the only
motion in the scene is the sweep being visualised.  ``--height_scan`` adds the
teacher's pelvis height scanner alongside, to compare coverage.

Add ``--save_dir plots/depth`` to write per-camera depth and mesh-id PNGs.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--num_envs", type=int, default=2, help="Environments to build.")
parser.add_argument(
    "--settle_steps", type=int, default=10, help="Sim steps before sampling the baseline frame."
)
parser.add_argument(
    "--knee_angle", type=float, default=1.2,
    help="Knee flexion (rad) commanded for the articulation check.",
)
parser.add_argument("--save_dir", default=None, help="If set, write depth / mesh-id PNGs here.")
parser.add_argument(
    "--min_self_pixels", type=int, default=1,
    help="Note a camera seeing fewer than this many robot pixels.",
)
parser.add_argument(
    "--animate", action="store_true",
    help="After the checks, sweep the legs on a loop (gravity off, joints written "
         "directly) so the scan can be watched in the GUI. Runs until the window closes.",
)
parser.add_argument(
    "--animate_steps", type=int, default=0,
    help="Stop the sweep after this many steps (0 = run until the window closes). "
         "Mainly to smoke-test the sweep headlessly.",
)
parser.add_argument("--sweep_hz", type=float, default=0.25, help="Leg sweep frequency.")
parser.add_argument(
    "--sweep_amp", type=float, default=1.2, help="Peak knee flexion (rad) during the sweep."
)
parser.add_argument(
    "--height_scan", action="store_true",
    help="Also add the teacher's pelvis height scanner, to compare its coverage.",
)
parser.add_argument(
    "--capture", default=None,
    help="With --viz kit, save a viewport screenshot here (PNG) before animating. "
         "Use it to confirm the scene is actually rendering.",
)
parser.add_argument(
    "--camera_frames", action=argparse.BooleanOptionalAction, default=True,
    help="Draw an axis triad at each depth camera's pose (GUI only). "
         "Disable with --no-camera_frames.",
)
parser.add_argument(
    "--frame_scale", type=float, default=0.25,
    help="Size of the camera axis triads.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs after the Kit app (and pxr) are up ----
import copy
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from robot_rl.assets.robots.g1_21j import G1_CFG
from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
    _FAR_PITCH_DEG,
    _FAR_POS,
    _NEAR_PITCH_DEG,
    _NEAR_POS,
    _make_depth_camera,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg import (
    FLAT_MESH_TERRAIN_CFG,
    G1TeacherSceneCfg,
)

CAMERAS = ("depth_camera_near", "depth_camera_far")

# In IsaacLab 3.0 `--headless` is deprecated and omitting `--viz` IS headless, so the
# presence of the Kit backend -- not `args.headless` -- decides whether there is a
# window to draw markers into.
GUI = "kit" in (args.visualizer or [])


def _marker_cfg(name: str, color: tuple[float, float, float], radius: float = 0.02):
    """Ray-hit marker config with its own prim path and colour.

    Every ray caster defaults to ``RAY_CASTER_MARKER_CFG`` at ``/Visuals/RayCaster``.
    Sharing it is not actually broken -- ``VisualizationMarkers`` resolves collisions
    via ``get_next_free_prim_path`` -- but naming each sensor's markers keeps the stage
    readable, and the distinct colours are what let the three scans be told apart in
    the viewport.

    Args:
        name: Suffix for the marker prim path.
        color: RGB diffuse colour in 0-1.
        radius: Sphere radius in metres.

    Returns:
        The marker configuration.
    """
    cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
    cfg.prim_path = f"/Visuals/RayCaster/{name}"
    cfg.markers["hit"].radius = radius
    cfg.markers["hit"].visual_material.diffuse_color = color
    return cfg


def _probe_camera(pos, pitch_deg, name: str, color):
    """Build a depth camera configured for provenance readout.

    Args:
        pos: Mount offset passed through to the real factory.
        pitch_deg: Downward pitch passed through to the real factory.
        name: Scene name, used for the marker prim path.
        color: Marker colour for this camera's ray hits.

    Returns:
        The camera cfg, with per-pixel mesh ids on and update decimation off.
    """
    cam = _make_depth_camera(pos, pitch_deg)
    cam.update_mesh_ids = True   # per-pixel mesh provenance -- the point of this script
    cam.update_period = 0.0      # refresh every step so a pose change shows immediately
    # The markers landing on the legs ARE the self-scan, visually.
    cam.debug_vis = GUI
    cam.visualizer_cfg = _marker_cfg(name, color, radius=0.015)
    return cam


@configclass
class DepthProbeSceneCfg(InteractiveSceneCfg):
    """Flat mesh ground + the G1 + both depth cameras, and nothing else.

    The ground must be a triangle MESH: a ray caster cannot hit the infinite PhysX
    ``plane`` primitive (the same reason the teacher's recording cfg swaps in a flat
    mesh).
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_MESH_TERRAIN_CFG,
        collision_group=-1,
        debug_vis=False,
    )
    # Without a light the viewport renders pure black -- geometry needs illumination,
    # while the axis gizmos are overlay-drawn and appear regardless, which makes an
    # unlit scene look like "the camera is fine but nothing is there".  Texture-free so
    # this works with no nucleus access, unlike the training scenes' HDR sky.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.95)),
    )
    sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=0.53),
    )
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    depth_camera_near = _probe_camera(
        _NEAR_POS, _NEAR_PITCH_DEG, "depth_near", (0.1, 0.9, 0.2)   # green
    )
    depth_camera_far = _probe_camera(
        _FAR_POS, _FAR_PITCH_DEG, "depth_far", (0.2, 0.4, 1.0)      # blue
    )

    def __post_init__(self):
        # The teacher's own height scanner, for a side-by-side coverage comparison.
        # Taken from an INSTANCE: @configclass turns the field into a dataclass field,
        # so it is not readable off the class itself.
        if args.height_scan:
            teacher_scene = G1TeacherSceneCfg(num_envs=1, env_spacing=1.0)
            self.height_scanner = teacher_scene.height_scanner.replace(
                debug_vis=GUI,
                visualizer_cfg=_marker_cfg("height_scan", (1.0, 0.1, 0.1)),  # red
            )


def _target_labels(sensor_cfg) -> list[str]:
    """Human-readable label per mesh index, in the sensor's own target order.

    The ray caster flattens targets in ``mesh_prim_paths`` order, one mesh per target
    per env, so the index in ``image_mesh_ids`` lines up with this list.

    Args:
        sensor_cfg: The camera configuration.

    Returns:
        Labels indexed by mesh id.
    """
    labels = []
    for entry in sensor_cfg.mesh_prim_paths:
        path = entry if isinstance(entry, str) else entry.prim_expr
        labels.append(path.rstrip("/").rsplit("/", 1)[-1])
    return labels


def _as_torch(data):
    """Return ``data`` as a torch tensor, converting from a warp array if needed.

    IsaacLab 3.0's articulation buffers are warp arrays under the GPU pipeline, which
    do not have torch's tensor methods.

    Args:
        data: A torch tensor or warp array.

    Returns:
        The equivalent torch tensor.
    """
    if isinstance(data, torch.Tensor):
        return data
    import warp as wp

    return wp.to_torch(data)


def _self_hit_counts(sensor, labels: list[str]) -> dict[str, int]:
    """Count pixels per robot link across all envs.

    Args:
        sensor: The depth camera sensor.
        labels: Mesh-index labels from :func:`_target_labels`.

    Returns:
        Link name -> pixel count, for robot meshes only (mesh index > 0).
    """
    ids = sensor.data.image_mesh_ids.squeeze(-1).reshape(-1)
    counts: dict[str, int] = {}
    for idx in ids.unique().tolist():
        if idx <= 0:  # 0 == terrain, -1 == missed
            continue
        label = labels[idx] if idx < len(labels) else f"<mesh {idx}>"
        counts[label] = int((ids == idx).sum().item())
    return counts


def _save_pngs(scene, out_dir: str, tag: str) -> None:
    """Write per-camera depth and mesh-id images for env 0.

    Args:
        scene: The interactive scene.
        out_dir: Directory to write into.
        tag: Filename suffix distinguishing the frame.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    for cam in CAMERAS:
        sensor = scene.sensors[cam]
        depth = sensor.data.output["distance_to_image_plane"][0].squeeze(-1).cpu().numpy()
        ids = sensor.data.image_mesh_ids[0].squeeze(-1).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        im0 = axes[0].imshow(depth)
        axes[0].set_title(f"{cam} depth [m]")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(ids, interpolation="nearest")
        axes[1].set_title("mesh id (0 = terrain)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        path = os.path.join(out_dir, f"{cam}_{tag}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"[depth]   wrote {path}")


def _make_camera_frame_markers():
    """Axis-triad markers for the depth-camera poses, or ``None`` if disabled.

    Returns:
        A :class:`VisualizationMarkers` with one triad instance per camera, or ``None``.
    """
    if not (GUI and args.camera_frames):
        return None
    cfg = copy.deepcopy(FRAME_MARKER_CFG)
    # Keep only the triad; the stock cfg also carries a "connecting_line" marker we
    # have no use for, and marker index 0 must be the frame.
    cfg.markers = {"frame": cfg.markers["frame"]}
    cfg.markers["frame"].scale = (args.frame_scale,) * 3
    cfg.prim_path = "/Visuals/CameraFrames"
    return VisualizationMarkers(cfg)


def _draw_camera_frames(markers, scene) -> None:
    """Move the axis triads onto the cameras' current world poses.

    Orientation is the sensor's ``quat_w_world``, i.e. IsaacLab's "world" camera
    convention (view down +X, +Z up) rather than the ROS or OpenGL conventions.

    Args:
        markers: The triad markers, or ``None`` to skip.
        scene: The interactive scene.
    """
    if markers is None:
        return
    pos = torch.cat([scene.sensors[c].data.pos_w for c in CAMERAS], dim=0)
    quat = torch.cat([scene.sensors[c].data.quat_w_world for c in CAMERAS], dim=0)
    markers.visualize(translations=pos, orientations=quat)


def _capture_viewport(path: str, settle_frames: int = 150) -> bool:
    """Save the active viewport to ``path``, pumping the app so RTX converges.

    Args:
        path: Destination PNG.
        settle_frames: App updates to run before and after the capture request.

    Returns:
        True if the file was written.
    """
    import os

    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        print("[depth] no active viewport to capture.")
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    for _ in range(settle_frames):
        simulation_app.update()
    capture_viewport_to_file(viewport, path)
    for _ in range(settle_frames):
        simulation_app.update()
    ok = os.path.exists(path)
    print(f"[depth] viewport capture {'written to ' + path if ok else 'FAILED'}")
    return ok


def _animate(sim, scene, robot, frame_markers=None) -> None:
    """Sweep the legs on a loop so the scan can be watched, until the window closes.

    Gravity is off and BOTH the joint positions and the root pose are written every
    step, making the robot a kinematic mannequin: it holds its stance and its place,
    so the only motion in the scene is the sweep.

    Pinning the root matters. Writing joint positions into a floating-base articulation
    imparts momentum to the base, and with gravity off nothing arrests it -- the robot
    slowly flies away, eventually taking the pelvis height scanner off the terrain.
    Its rays then all miss, and IsaacLab's ``RayCaster._debug_vis_callback`` filters the
    resulting ``inf`` hits down to an empty set and raises "Number of markers cannot be
    zero" on every render.

    Args:
        sim: The simulation context.
        scene: The interactive scene.
        robot: The G1 articulation.
    """
    import math

    knee_ids, _ = robot.find_joints(".*_knee_joint")
    hip_ids, _ = robot.find_joints(".*_hip_pitch_joint")
    ankle_ids, _ = robot.find_joints(".*_ankle_pitch_joint")
    base = _as_torch(robot.data.joint_pos).clone()
    dt = sim.get_physics_dt()
    zeros = torch.zeros_like(base)

    # Captured once and rewritten every step -- round-tripped verbatim so no quaternion
    # convention is assumed.
    root_pose = _as_torch(robot.data.root_pose_w).clone()
    root_vel = torch.zeros((root_pose.shape[0], 6), device=root_pose.device)
    ray_casters = [n for n in (*CAMERAS, "height_scanner") if n in scene.sensors]

    print(
        f"\n[depth] animating: knee sweep 0 -> {args.sweep_amp} rad at {args.sweep_hz} Hz.\n"
        "        The ray-hit markers landing on the legs are the self-scan.\n"
        "        Close the Kit window to exit."
    )
    t, step = 0.0, 0
    while simulation_app.is_running():
        if args.animate_steps and step >= args.animate_steps:
            print(f"[depth] sweep stopped after {step} steps (--animate_steps).")
            break
        # Cosine ramp 0 -> amp -> 0, with hip and ankle counter-rotating so the foot
        # roughly keeps its orientation and the leg folds visibly under the camera.
        phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * args.sweep_hz * t)
        angle = args.sweep_amp * phase
        target = base.clone()
        target[:, knee_ids] = angle
        target[:, hip_ids] = -0.5 * angle
        target[:, ankle_ids] = -0.5 * angle
        robot.write_joint_state_to_sim(target, zeros)
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        # Ray casters are lazy: their buffers only refresh on a `.data` access or a
        # forced recompute. Without this the markers would freeze at the last frame
        # computed before the sweep started.
        for name in ray_casters:
            scene.sensors[name].update(dt, force_recompute=True)
        _draw_camera_frames(frame_markers, scene)
        t += dt
        step += 1

    if ray_casters:
        drift = (_as_torch(robot.data.root_pose_w)[:, :3] - root_pose[:, :3]).norm(dim=-1)
        print(f"[depth] root drift over the sweep: {drift.max().item():.2e} m")
        if "height_scanner" in scene.sensors:
            hits = scene.sensors["height_scanner"].data.ray_hits_w.reshape(-1, 3)
            n_finite = int(torch.isfinite(hits).all(dim=1).sum())
            print(f"[depth] height_scanner still has {n_finite}/{hits.shape[0]} finite hits")


def main() -> int:
    """Build the probe scene, census self-hits, flex the knees, re-census.

    Returns:
        0 on success, 1 if a check failed.
    """
    # Animating writes joint state every step, so gravity would only fight it and
    # collapse the robot; turn it off and treat the robot as a kinematic mannequin.
    gravity = (0.0, 0.0, 0.0) if args.animate else (0.0, 0.0, -9.81)
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args.device, gravity=gravity)
    )
    scene_cfg = DepthProbeSceneCfg(num_envs=args.num_envs, env_spacing=2.5)
    print(f"[depth] building {args.num_envs} env(s) ...")
    scene = InteractiveScene(scene_cfg)
    if GUI:
        # Frame the robot from the front-left so the legs and the ray hits are visible.
        sim.set_camera_view(eye=(1.6, -1.6, 1.2), target=(0.0, 0.0, 0.6))
    else:
        print(
            "[depth] NOTE: no Kit viewer (pass '--viz kit' for one); markers are disabled."
        )
    sim.reset()

    labels = {cam: _target_labels(getattr(scene_cfg, cam)) for cam in CAMERAS}
    n_targets = len(labels[CAMERAS[0]])
    print(f"[depth] {n_targets} ray-cast targets per camera (1 terrain + {n_targets - 1} robot)")

    frame_markers = _make_camera_frame_markers()
    if frame_markers is not None:
        print(f"[depth] camera axis triads at {frame_markers.prim_path} (--no-camera_frames to hide)")

    for _ in range(args.settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    _draw_camera_frames(frame_markers, scene)

    ok = True

    # ---------------- 1. presence ----------------
    print("\n[depth] === self-hit census (baseline) ===")
    baseline: dict[str, dict[str, int]] = {}
    for cam in CAMERAS:
        sensor = scene.sensors[cam]
        depth = sensor.data.output["distance_to_image_plane"].squeeze(-1)
        finite = depth[torch.isfinite(depth)]
        counts = _self_hit_counts(sensor, labels[cam])
        baseline[cam] = counts
        total_self = sum(counts.values())
        n_pixels = depth.numel()
        print(
            f"  {cam}: depth min={finite.min().item():.3f} max={finite.max().item():.3f} "
            f"mean={finite.mean().item():.3f} | non-finite="
            f"{int((~torch.isfinite(depth)).sum())}/{n_pixels}"
        )
        print(f"    robot pixels: {total_self}/{n_pixels} ({100.0 * total_self / n_pixels:.1f} %)")
        for link, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"      {link:32s} {count}")
        if total_self < args.min_self_pixels:
            # Not necessarily a bug: a far-field camera pitched shallowly may
            # legitimately frame only ground.  Only a total absence across ALL cameras
            # means the targets failed to resolve.
            print(f"    NOTE: {cam} frames no robot geometry (check its extrinsics).")

    if not any(sum(c.values()) for c in baseline.values()):
        print("\n  FAIL: no camera sees any robot geometry -- ray-cast targets did not resolve.")
        ok = False

    if GUI:
        # Confirm the marker prims really exist, so "I see nothing" can be pinned on
        # the viewport/camera rather than on the sensors never having created markers.
        stage = sim_utils.get_current_stage()
        for name in CAMERAS + (("height_scanner",) if args.height_scan else ()):
            sensor = scene.sensors[name]
            path = getattr(getattr(sensor, "ray_visualizer", None), "prim_path", None)
            exists = path is not None and stage.GetPrimAtPath(path).IsValid()
            print(f"[depth] markers for {name}: {path} valid={exists}")

    if args.height_scan:
        hits = scene.sensors["height_scanner"].data.ray_hits_w.reshape(-1, 3)
        finite = torch.isfinite(hits).all(dim=1)
        print(
            f"\n[depth] height_scanner: {int(finite.sum())}/{hits.shape[0]} finite hits"
            + (f", z range [{hits[finite, 2].min():.3f}, {hits[finite, 2].max():.3f}]"
               if bool(finite.any()) else " -- ALL MISSED")
        )

    if args.save_dir:
        _save_pngs(scene, args.save_dir, "baseline")

    # ---------------- 2. articulation ----------------
    robot = scene["robot"]
    knee_ids, knee_names = robot.find_joints(".*_knee_joint")
    print(f"\n[depth] flexing {knee_names} to {args.knee_angle} rad ...")
    joint_pos_nominal = _as_torch(robot.data.joint_pos).clone()
    joint_pos = joint_pos_nominal.clone()
    joint_pos[:, knee_ids] = args.knee_angle
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    print("\n[depth] === self-hit census (knees flexed) ===")
    for cam in CAMERAS:
        counts = _self_hit_counts(scene.sensors[cam], labels[cam])
        before, after = baseline[cam], counts
        changed = {k for k in set(before) | set(after) if before.get(k, 0) != after.get(k, 0)}
        print(f"  {cam}: robot pixels {sum(before.values())} -> {sum(after.values())}")
        for link in sorted(changed):
            print(f"      {link:32s} {before.get(link, 0)} -> {after.get(link, 0)}")
        # Only meaningful for a camera that framed the body in the first place.
        if sum(before.values()) == 0 and sum(after.values()) == 0:
            print(f"    (skipped -- {cam} frames no robot geometry)")
        elif not changed:
            print(
                f"    FAIL: {cam} did not change when the knees moved -- the limb meshes are "
                "not tracking their link transforms (frozen merged blob)."
            )
            ok = False

    if args.save_dir:
        _save_pngs(scene, args.save_dir, "knees_flexed")

    print("\n[depth] RESULT:", "PASS" if ok else "FAIL")

    if args.capture and GUI:
        # Undo the articulation check's knee flex first, or the screenshot shows the
        # robot with its legs folded up rather than in its nominal stance.
        robot.write_joint_state_to_sim(joint_pos_nominal, torch.zeros_like(joint_pos_nominal))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        _draw_camera_frames(frame_markers, scene)
        _capture_viewport(args.capture)

    if args.animate:
        _animate(sim, scene, robot, frame_markers)
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
