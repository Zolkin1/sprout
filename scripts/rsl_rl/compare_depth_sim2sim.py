# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compare the IsaacLab depth cameras against the MuJoCo re-implementation, pixel by pixel.

The MuJoCo port (``transfer/sim/depth_camera.py``) reproduces the training sensor from the
contract embedded in the ONNX artifacts. Its ray pattern, normalisation and ground geometry
are each unit-tested in isolation; this is the end-to-end check that the assembled thing
agrees with the sensor it is imitating, on the same robot in the same pose.

Both sides are put on FLAT ground with identical joint angles and root pose, all depth
randomisation and noise off, and the two 30x26 images are differenced. Also checked:

* **Mount frame** -- the world offset of ``torso_link`` from the pelvis in each simulator.
  The contract's camera offsets are expressed in that body's frame, so a USD/MJCF skew here
  moves both cameras and is the one failure mode the contract itself cannot catch.
* **Known delta** -- MuJoCo casts every visual mesh in its self-scan geom group, which
  includes the HAND meshes; IsaacLab's ``SELF_SCAN_LINKS`` excludes them (see the TODO in
  ``g1_self_scan.py``). Hand pixels are reported separately so the residual is readable.

Run under the full Isaac env (``env_isaaclab``), from the repo root. ``python -u`` matters:
Kit can hard-exit and swallow buffered stdout.

    PYTHONPATH=$PWD/source/robot_rl:$PWD/source/motion_gen \\
        <env_isaaclab py> -u scripts/rsl_rl/compare_depth_sim2sim.py

    # a few random postures instead of just the nominal stance
    ... scripts/rsl_rl/compare_depth_sim2sim.py --poses 5 --tol 0.02
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--poses", type=int, default=3,
                    help="Postures to compare: the nominal stance plus this many-1 random ones.")
parser.add_argument("--joint_noise", type=float, default=0.25,
                    help="Random joint perturbation (rad) for the non-nominal postures.")
parser.add_argument("--tol", type=float, default=0.01,
                    help="Pass threshold on the 99th-percentile |depth difference|, in metres.")
parser.add_argument("--robot_name", default="g1_29j", help="MuJoCo robot model to compare against.")
parser.add_argument("--scene", default="basic_scene", help="MuJoCo (flat) scene name.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", default=None, help="Write side-by-side difference PNGs here.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs after the Kit app (and pxr) are up ----
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robot_rl.assets.robots.g1_21j import G1_CFG  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (  # noqa: E402
    _FAR,
    _NEAR,
    _make_depth_camera,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg import (  # noqa: E402
    FLAT_MESH_TERRAIN_CFG,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (  # noqa: E402
    normalize_depth,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "transfer"))
sys.path.insert(0, os.path.join(REPO, "transfer", "sim"))

from deploy.depth_obs import denormalize_depth_np  # noqa: E402

CAMERAS = ("depth_camera_near", "depth_camera_far")


@configclass
class _CompareSceneCfg(InteractiveSceneCfg):
    """Flat mesh ground + the G1 + both depth cameras, with all randomisation off.

    The ground must be a triangle MESH: a ray caster cannot hit the infinite PhysX
    ``plane`` primitive. Matches the MuJoCo ``basic_scene``'s single ground plane.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_MESH_TERRAIN_CFG,
        collision_group=-1,
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.95)),
    )
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    depth_camera_near = _make_depth_camera(_NEAR)
    depth_camera_far = _make_depth_camera(_FAR)

    def __post_init__(self):
        # Refresh every step so a written pose shows up immediately (the trained cameras
        # update once per 0.02 s control step).
        for name in CAMERAS:
            getattr(self, name).update_period = 0.0


def _as_torch(data):
    """Return ``data`` as a torch tensor, converting from a warp array if needed."""
    if isinstance(data, torch.Tensor):
        return data
    import warp as wp

    return wp.to_torch(data)


def _isaac_spec() -> list[dict]:
    """The camera contract for this task, via the production simulator-free entry point.

    Returns:
        The per-camera contract, ordered near then far.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        depth_camera_specs_from_cfg,
    )

    return depth_camera_specs_from_cfg()


def _isaac_depth(scene, spec: list[dict]) -> np.ndarray:
    """Normalised depth for env 0, one row per camera, as the observation term produces it.

    No noiser is registered in this scene, so this is the clean (``-play``) signal.

    Args:
        scene: The interactive scene.
        spec: The per-camera contract (supplies each camera's normalisation).

    Returns:
        ``(num_cameras, H*W)`` normalised depth.
    """
    rows = []
    for name, cam in zip(CAMERAS, spec):
        raw = scene.sensors[name].data.output["distance_to_image_plane"].squeeze(-1)
        norm = normalize_depth(
            raw.double(),
            offset=cam["offset"], scale=cam["scale"],
            min_dist=cam["min_dist"], max_dist=cam["max_dist"],
            min_clip_val=cam["min_clip_val"], max_clip_val=cam["max_clip_val"],
        )
        rows.append(norm[0].reshape(-1).cpu().numpy())
    return np.stack(rows)


def _mujoco_setup(spec: list[dict]):
    """Load the MuJoCo model + caster for the comparison.

    Args:
        spec: The camera contract read off the IsaacLab side.

    Returns:
        ``(model, data, cams, hinge_qpos_addresses)``.
    """
    import mujoco

    from depth_camera import MujocoDepthCameras

    path = os.path.join(REPO, "transfer", "sim", "robots", "g1",
                        f"{args.robot_name}_{args.scene}.xml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; generate it with transfer/sim/generate_stairs_scene.py "
            f"--robot {args.robot_name} --scene {args.scene} --flat"
        )
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    hinges = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    }
    return model, data, MujocoDepthCameras(model, spec), hinges


def _mujoco_depth(model, data, cams, hinges, joint_pos: dict, root_pos, root_quat_xyzw):
    """Place the MuJoCo robot at the given pose and cast.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        cams: The camera caster.
        hinges: ``{hinge joint name: qpos address}``.
        joint_pos: ``{joint_name: angle}`` from the IsaacLab side.
        root_pos: World pelvis position ``(3,)``.
        root_quat_xyzw: World pelvis quaternion ``(x, y, z, w)``.

    Returns:
        The normalised images ``(T, H*W)``.
    """
    import mujoco

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = root_pos
    data.qpos[3:7] = [root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]]
    for name, addr in hinges.items():
        data.qpos[addr] = joint_pos.get(name, 0.0)
    mujoco.mj_forward(model, data)
    return cams.compute(data)  # geom ids stay on each _Camera for _hand_mask


def _mount_frame_check(scene, model, data) -> bool:
    """Compare the mount body's offset from the pelvis in the two simulators.

    Args:
        scene: The IsaacLab interactive scene.
        model: MuJoCo model.
        data: MuJoCo data (kinematics current, robot at its default pose).

    Returns:
        True if the two frames agree to under a millimetre.
    """
    import mujoco

    robot = scene.articulations["robot"]
    body_names = list(robot.data.body_names)
    torso_i = body_names.index("torso_link")
    pelvis_i = body_names.index("pelvis_link" if "pelvis_link" in body_names else "pelvis")
    pos_w = _as_torch(robot.data.body_pos_w)[0]
    isaac_offset = (pos_w[torso_i] - pos_w[pelvis_i]).cpu().numpy()

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    mj_offset = np.asarray(data.xpos[torso] - data.xpos[pelvis])

    err = float(np.abs(isaac_offset - mj_offset).max())
    ok = err < 1e-3
    print("\n== mount frame (torso_link relative to the pelvis, default pose) ==")
    print(f"  isaac  {np.round(isaac_offset, 6)}")
    print(f"  mujoco {np.round(mj_offset, 6)}")
    print(f"  {'ok  ' if ok else 'FAIL'} max abs difference {err * 1000:.3f} mm "
          "(the contract's camera offsets are in this frame)")
    return ok


def _hand_mask(model, cams, index: int) -> np.ndarray:
    """Pixels whose ray hit a HAND body -- the known IsaacLab/MuJoCo self-scan delta.

    Reads the provenance of the caster's last cast, so this describes exactly the rays the
    comparison just differenced.

    Args:
        model: MuJoCo model.
        cams: The camera caster (already ``compute``d for the current pose).
        index: Which camera to mask.

    Returns:
        Boolean mask over that camera's pixels.
    """
    import mujoco

    camera = cams.cameras[index]
    mask = np.zeros(camera.num_rays, dtype=bool)
    for ray in np.flatnonzero(camera.hit):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[camera.geom_ids[ray]]))
        mask[ray] = bool(body) and ("hand" in body or "palm" in body)
    return mask


def _save_comparison(out_dir: str, tag: str, cam: dict, isaac_row, mj_row, camera) -> None:
    """Write an isaac | mujoco | difference strip for one camera."""
    try:
        from PIL import Image
    except ImportError:
        print(f"[WARN] Pillow not installed; skipping --save_dir {out_dir}.")
        return
    os.makedirs(out_dir, exist_ok=True)
    height, width = camera.height, camera.width
    to_m = lambda row: denormalize_depth_np(row, cam["offset"], cam["scale"])
    both = np.concatenate([to_m(isaac_row), to_m(mj_row)])
    lo, hi = float(both.min()), float(both.max())
    gray = lambda row: (255.0 * (1.0 - np.clip((to_m(row) - lo) / max(hi - lo, 1e-9), 0, 1))
                        ).astype(np.uint8).reshape(height, width)
    diff = np.abs(to_m(isaac_row) - to_m(mj_row))
    diff_img = (255.0 * np.clip(diff / max(diff.max(), 1e-9), 0, 1)).astype(np.uint8).reshape(
        height, width)
    strip = np.concatenate([gray(isaac_row), np.full((height, 2), 128, np.uint8),
                            gray(mj_row), np.full((height, 2), 128, np.uint8), diff_img], axis=1)
    path = os.path.join(out_dir, f"{tag}_{camera.sensor}.png")
    Image.fromarray(strip, mode="L").resize(
        (strip.shape[1] * 8, height * 8), Image.NEAREST).save(path)
    print(f"    wrote {path}  (isaac | mujoco | |difference|, max {diff.max():.3f} m)")


def main() -> int:
    """Build both sims, compare depth over several postures, and report.

    Returns:
        0 if every posture agreed within tolerance, 1 otherwise.
    """
    torch.manual_seed(args.seed)
    # Gravity off: the robot is a kinematic mannequin here, so a written posture stays put
    # instead of collapsing between the write and the sensor read.
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args.device, gravity=(0.0, 0.0, 0.0))
    )
    scene = InteractiveScene(_CompareSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    for _ in range(5):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    spec = _isaac_spec()
    print("== camera contract (from the IsaacLab cfg) ==")
    for cam in spec:
        print(f"  {cam['sensor']:20s} {cam['width']}x{cam['height']} on {cam['mount_body']} "
              f"at {[round(v, 4) for v in cam['offset_pos']]}, fx={cam['fx']:.4f} "
              f"fy={cam['fy']:.4f}, range {cam['min_dist']}..{cam['max_dist']} m")

    model, data, cams, hinges = _mujoco_setup(spec)
    robot = scene.articulations["robot"]
    isaac_joint_names = list(robot.data.joint_names)
    missing = [n for n in isaac_joint_names if n not in hinges]
    if missing:
        print(f"\n[FAIL] The MuJoCo model lacks joint(s) {missing}.")
        return 1

    nominal = _as_torch(robot.data.joint_pos)[0].clone()
    root_pose = _as_torch(robot.data.root_pose_w)[0].clone()
    ok = True

    for pose_i in range(max(1, args.poses)):
        if pose_i == 0:
            joint_pos, tag = nominal.clone(), "nominal"
        else:
            joint_pos = nominal + args.joint_noise * (2.0 * torch.rand_like(nominal) - 1.0)
            tag = f"random{pose_i}"
        target = joint_pos.unsqueeze(0)
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        robot.write_root_pose_to_sim_index(root_pose=root_pose.unsqueeze(0))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        isaac_rows = _isaac_depth(scene, spec)
        # Read the pose BACK rather than reusing what was written: whatever the articulation
        # actually settled at is the pose the depth image was rendered from, and that is what
        # MuJoCo has to be placed at for the comparison to mean anything.
        actual_joints = _as_torch(robot.data.joint_pos)[0].cpu().numpy()
        actual_root = _as_torch(robot.data.root_pose_w)[0].cpu().numpy()
        drift = float(np.abs(actual_joints - joint_pos.cpu().numpy()).max())
        angles = {n: float(actual_joints[i]) for i, n in enumerate(isaac_joint_names)}
        # IsaacLab root_pose_w is (pos, quat) with the quat in xyzw on this branch.
        mj_rows = _mujoco_depth(model, data, cams, hinges,
                                angles, actual_root[:3], actual_root[3:7])

        if pose_i == 0 and not _mount_frame_check(scene, model, data):
            ok = False

        print(f"\n== depth difference [{tag}] ==")
        print(f"  posture: root {np.round(actual_root[:3], 4)} quat(xyzw) "
              f"{np.round(actual_root[3:7], 4)}; settled {drift * 1000:.2f} mrad from the "
              "written joint angles (both sims placed at the settled pose)")
        for i, (cam, camera) in enumerate(zip(spec, cams.cameras)):
            to_m = lambda row: denormalize_depth_np(row, cam["offset"], cam["scale"])
            diff = np.abs(to_m(isaac_rows[i]) - to_m(mj_rows[i]))
            hands = _hand_mask(model, cams, i)
            body = diff[~hands]
            p99 = float(np.percentile(body, 99)) if body.size else 0.0
            passed = p99 < args.tol
            ok &= passed
            print(f"  {'ok  ' if passed else 'FAIL'} {camera.sensor:20s} "
                  f"excluding hands ({int(hands.sum())} px): "
                  f"mean {body.mean():.4f} m  p99 {p99:.4f} m  max {body.max():.4f} m")
            if hands.any():
                print(f"       hand pixels (known delta): mean {diff[hands].mean():.4f} m  "
                      f"max {diff[hands].max():.4f} m")
            if args.save_dir:
                _save_comparison(args.save_dir, tag, cam, isaac_rows[i], mj_rows[i], camera)

    print("\n" + ("DEPTH MATCHES ACROSS SIMULATORS" if ok else "DEPTH MISMATCH -- see FAIL rows"))
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
