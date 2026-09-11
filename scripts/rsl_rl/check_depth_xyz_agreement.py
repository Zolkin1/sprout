# Copyright 2026 Zachary Olkin. All rights reserved.

"""Do the OFFLINE and LIVE depth-xyz paths produce the same cloud from the same image?

The generator's ``extero_channels >= 4`` rows exist in two implementations, and a training
run uses one while play and deploy use the other:

* **offline** -- ``FutureStateDataset._camera_pose_yaw_frame`` rebuilds the camera pose from
  the recording's root quaternion, its joint angles and the contract's NOMINAL mount, with
  ``motion_gen.FKChain`` walking an MJCF. This is what every training window is built from,
  and it is the set a robot actually has (IMU + encoders + calibration).
* **live** -- ``mdp.depth_camera_pose_yaw_frame`` reads the mount body straight out of the
  Isaac articulation and composes the same nominal offset.

Two different kinematic descriptions of one robot (MJCF vs the USD articulation), so they
cannot be assumed equal, and a disagreement is a TRAIN/SERVE SKEW: the model would be fitted
on clouds that are systematically offset from the ones it is later handed. No amount of
training corrects that, and nothing else in the repo compares the two.

The depth IMAGE is taken from the live path and fed to both, deliberately: the dataset does
the same thing -- it unprojects the depth row the recording already holds -- so this is the
honest analogue of what training does, and a comparison that re-read the sensor would risk
measuring ``DepthNoiser`` instead of the geometry. (``DepthNoiser`` now shares one draw per
camera per control step, and this script drives ``sim.step`` itself so the step counter never
advances -- one frame for the whole run. That is what makes the two paths comparable, and it
is also why no noise STATISTIC may be read off this script.)

The cameras are the GENERATOR's, which is every camera the robot carries. The policy's group
holds the near one alone, and it is the FAR camera -- the one that reads terrain ahead -- whose
registration the xyz channels exist to fix, so scoping this to the policy would drop precisely
the interesting half while ``n_cam`` adapted and the check still printed a pass.

What is compared, per pose:

* the CAMERA POSE in the yaw frame (the root cause if the clouds disagree);
* the xyz channels, over the pixels carrying a real reading;
* the depth and validity planes, which must be bit-identical since channel 0 is the input.

``--dump`` writes the clouds, poses and joint configurations for
``scripts/rsl_rl/view_depth_xyz_viser.py``, which draws them in the yaw frame so the
invariance can be eyeballed: the robot tilts, the terrain does not.

Run under ``env_isaaclab``. ``python -u`` because Kit can hard-exit and swallow buffered
stdout::

    PYTHONPATH=$PWD/source/robot_rl:$PWD/source/motion_gen \\
        <env_isaaclab py> -u scripts/rsl_rl/check_depth_xyz_agreement.py \\
            --clip clips/g1_mb_v5_small_body.h5 --clip_device cpu \\
            --dump depth_xyz.npz
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False,      # so --clip can never bind to --clip_device (see play.py)
)
parser.add_argument("--task", default="G1-teacher-clf-terrain-depth-play",
                    help="Depth task to build. Must carry the camera pair.")
parser.add_argument("--clip", required=True,
                    help="Clip library (.h5). Required -- it defines the TERRAIN as well as "
                         "the references, so there is deliberately no default.")
parser.add_argument("--clip_device", default="cpu",
                    help="Hold the clip library here. 'cpu' keeps a multi-GB library off a "
                         "small GPU.")
parser.add_argument("--clip_id", type=int, default=-1,
                    help="Pin every env to this clip, so the terrain under the robot is "
                         "reproducible. <0 leaves the reset sampling.")
parser.add_argument("--num_envs", type=int, default=2, help="Environments to build.")
parser.add_argument("--settle_steps", type=int, default=8,
                    help="Sim steps after reset, before the first pose is written.")
parser.add_argument("--tol_mm", type=float, default=5.0,
                    help="Pass threshold on the median xyz disagreement, in millimetres.")
parser.add_argument("--dump", default=None,
                    help="Write an .npz for view_depth_xyz_viser.py.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True                   # the cameras do not exist without it

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from motion_gen.features import _REPO_ROOT  # noqa: E402

# Recursive contact-sensor traversal. MUST come after the SimulationApp exists (same note as
# play.py); without it the G1's per-link contact sensors find no bodies.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
import robot_rl.tasks  # noqa: F401,E402
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import FutureStateDataset as DS  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (  # noqa: E402
    install_fm_depth_group,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (  # noqa: E402
    depth_camera_pose_yaw_frame,
)

#: Poses to compare at, as ``(name, (roll, pitch, yaw) delta on the root, {joint: angle})``.
#: The root DELTA is applied on top of the settled orientation and the root POSITION is
#: pinned, which is what makes the tilt cases a controlled experiment: a physical lean
#: pivots about the feet and translates the root, and translation moves the cloud legitimately
#: (the frame is anchored at the root), so it would mask the thing being tested.
#:
#: The set is chosen to exercise both halves of the frame independently -- root roll/pitch is
#: the ``yaw_quat`` half, the waist is the FK half -- plus a knee flexion, which moves only
#: what the cameras see OF THE ROBOT and so should leave the terrain cloud alone.
_POSES: tuple[tuple[str, tuple[float, float, float], dict[str, float]], ...] = (
    ("rest",        (0.00,  0.00, 0.0), {}),
    ("pitch_fwd",   (0.00,  0.25, 0.0), {}),
    ("pitch_back",  (0.00, -0.25, 0.0), {}),
    ("roll",        (0.20,  0.00, 0.0), {}),
    ("yaw",         (0.00,  0.00, 0.6), {}),
    ("waist",       (0.00,  0.00, 0.0), {"waist_yaw_joint": 0.5, "waist_pitch_joint": 0.3}),
    ("knees",       (0.00,  0.00, 0.0), {"left_knee_joint": 1.0, "right_knee_joint": 1.0}),
)


def _torch(array) -> torch.Tensor:
    """Whatever the sim hands back (torch or warp) as a torch tensor.

    IsaacLab 3.0's articulation buffers are warp-backed, and the rest of this script is
    torch. Mirrors the helper the depth observation module uses.

    Args:
        array: A warp array or an existing torch tensor.

    Returns:
        The same data as a torch tensor.
    """
    if isinstance(array, torch.Tensor):
        return array
    import warp as wp  # noqa: PLC0415
    return wp.to_torch(array)


def _quat_from_rpy(roll: float, pitch: float, yaw: float, device, dtype) -> torch.Tensor:
    """Scalar-LAST quaternion ``(x, y, z, w)`` for an intrinsic Z-Y-X rotation.

    Scalar-last because every quaternion on the Isaac side of this codebase is -- the
    articulation buffers and ``isaaclab.utils.math`` alike. Building this one scalar-first
    would be a silent axis permutation that still normalises to 1 and still round-trips.

    Args:
        roll: Rotation about x, radians.
        pitch: Rotation about y, radians.
        yaw: Rotation about z, radians.
        device: Torch device for the result.
        dtype: Torch dtype for the result.

    Returns:
        The quaternion, shape ``(4,)``.
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return torch.tensor(
        [sr * cp * cy - cr * sp * sy,        # x
         cr * sp * cy + sr * cp * sy,        # y
         cr * cp * sy - sr * sp * cy,        # z
         cr * cp * cy + sr * sp * sy],       # w
        device=device, dtype=dtype,
    )


def _settle_and_read(env, robot, root: torch.Tensor, jpos: torch.Tensor) -> None:
    """Hold a written pose long enough for the ray-casters to re-cast against it.

    Three separate staleness traps, all of them measured in ``augment_registration.py`` and
    all of them silent -- a stale image looks like a placement bug rather than a stale image:

    1. the casters run at ``update_period`` against a smaller ``physics_dt``, so one
       ``sim.step`` need not tick them at all;
    2. they lag one READ CYCLE, so the first read after a move returns the PREVIOUS pose's
       image (89% of mismatched frames were bit-equal to the previous placement);
    3. physics drifts the written pose unless it is re-asserted before every step.

    Args:
        env: The unwrapped environment.
        robot: The articulation.
        root: Full root state to hold, ``(N, 13)``.
        jpos: Joint positions to hold, ``(N, nj)``.
    """
    period = float(getattr(env.scene.sensors["depth_camera_near"].cfg, "update_period", 0.0)
                   or 0.0)
    n_steps = 2 * max(2, int(round(period / float(env.physics_dt))) + 1)
    zero_v = torch.zeros_like(jpos)
    for _ in range(n_steps):
        robot.write_root_state_to_sim(root)
        robot.write_joint_position_to_sim(jpos)
        robot.write_joint_velocity_to_sim(zero_v)
        robot.set_joint_position_target(jpos)
        env.sim.step(render=True)
    for _ in range(2):                                   # keep the SECOND read (trap 2)
        env.scene.update(env.physics_dt)
        for name in ("depth_camera_near", "depth_camera_far"):
            if name in env.scene.sensors:
                env.scene.sensors[name].update(env.physics_dt, force_recompute=True)


def main() -> int:
    """Compare the offline and live depth-xyz paths over a set of held poses."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.traj_ref.path = args_cli.clip
    env_cfg.commands.traj_ref.clip_device = args_cli.clip_device
    if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain,
                                                     "clip_library_path"):
        # Must stay identical to the command's path: tile terrain ids line up with clip
        # terrain ids, and nothing asserts it.
        env_cfg.scene.terrain.clip_library_path = args_cli.clip
    if args_cli.clip_id >= 0:
        env_cfg.events.reset_on_clip.params["pin_clip_ids"] = " ".join(
            [str(args_cli.clip_id)] * args_cli.num_envs
        )

    # --task defaults to a TEACHER task, which installs no generator group of its own; put it
    # on, at the scene's own resolution, so the check covers both cameras.
    install_fm_depth_group(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    u.reset()
    for _ in range(args_cli.settle_steps):
        u.sim.step(render=True)
    u.scene.update(u.physics_dt)

    groups = mdp.fm_depth_groups(u)
    spec = mdp.depth_camera_specs(u, groups)
    n_cam = len(spec)
    n_px = int(spec[0]["width"]) * int(spec[0]["height"])
    joint_names = list(robot.data.joint_names)
    mjcf = str(_REPO_ROOT / FlowMatchingConfig().fk_model_path)
    print(f"[check] {n_cam} cameras x {n_px} px, groups={groups}")
    print(f"[check] offline FK from {mjcf}")

    # The layout of root_state_w's orientation block is ASSUMED below (it is composed with
    # quat_mul, whose convention must be the same one root_quat_w reports in). Assert it
    # rather than trust it: a mismatch here is a silent ~180 deg error downstream.
    assert torch.allclose(_torch(robot.data.root_state_w)[:, 3:7],
                          _torch(robot.data.root_quat_w), atol=1e-6), \
        "root_state_w[:, 3:7] is not root_quat_w -- the quaternion layout changed."

    base_root = _torch(robot.data.root_state_w).clone()
    base_jpos = _torch(robot.data.joint_pos).clone()
    dev, dtype = base_root.device, base_root.dtype

    rows, dump = [], {}
    for name, rpy, joints in _POSES:
        root = base_root.clone()
        root[:, 7:] = 0.0                                   # no velocity: hold the pose
        dq = _quat_from_rpy(*rpy, dev, dtype).expand(root.shape[0], 4)
        root[:, 3:7] = quat_mul(dq, base_root[:, 3:7])
        jpos = base_jpos.clone()
        for jname, angle in joints.items():
            jpos[:, joint_names.index(jname)] = angle

        _settle_and_read(u, robot, root, jpos)

        # LIVE. One call: it applies the depth noise, and channel 0 of its output is the
        # array both paths must unproject for this to be a geometry comparison.
        live = mdp.depth_xyz_stack(u, groups, channels=5)            # (B, n_cam, 5*K)
        b = live.shape[0]
        live_img = live.reshape(b, n_cam, 5, n_px)
        depth_rows = live_img[:, :, 0, :].reshape(b, n_cam * n_px)

        # OFFLINE, from exactly what a recording stores -- and ON CPU, exactly as the dataset
        # runs it. Not a convenience: ``FKChain`` builds its per-body constants on the DEFAULT
        # device, so handing it CUDA joint angles dies in the first matmul. Training never
        # notices (the dataset is host-side), and moving the comparison here rather than
        # forcing the chain onto the GPU keeps this faithful to what actually builds a window.
        q_w = _torch(robot.data.root_quat_w)
        jp_w = _torch(robot.data.joint_pos)
        off_poses = DS._camera_pose_yaw_frame(
            spec, q_w.cpu(), jp_w.cpu(), joint_names, mjcf
        )
        off_poses = [(p.to(dev), r.to(dev)) for p, r in off_poses]
        offline = DS._depth_xyz_rows(depth_rows, spec, off_poses, n_cam, n_px, channels=5)
        off_img = offline.reshape(b, n_cam, 5, n_px)

        # Camera pose, the root cause if the clouds disagree.
        pos_err = rot_err = 0.0
        for i, cam in enumerate(spec):
            live_p, live_r = depth_camera_pose_yaw_frame(u, cam)
            off_p, off_r = off_poses[i]
            pos_err = max(pos_err, float((live_p - off_p).norm(dim=-1).max()))
            # Angle between the two rotations, from the ANTISYMMETRIC part of R_live^T R_off.
            #
            # NOT the textbook arccos((trace - 1) / 2). Near identity that formula amplifies
            # error by a square root -- an absolute error e in the trace becomes sqrt(e) in
            # the angle -- so float32 inputs (e ~ 1e-7) put a FLOOR of ~4e-4 rad on anything
            # it can report, whatever precision the arithmetic is done in. It duly reported a
            # steady ~0.4 mrad here, which the 11 um of point agreement at metre range already
            # falsifies. For small angles R^T R' = I + [w]x, so reading w off the skew part is
            # linear in the data error instead, and stays honest at 1e-7.
            delta = live_r.transpose(1, 2) @ off_r
            skew = 0.5 * (delta - delta.transpose(1, 2))
            omega = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)
            rot_err = max(rot_err, float(omega.norm(dim=-1).max()))

        valid = off_img[:, :, 4, :] > 0.5
        d_xyz = (live_img[:, :, 1:4, :] - off_img[:, :, 1:4, :]).norm(dim=2)[valid]
        rows.append((name, float(d_xyz.median()), float(d_xyz.max()),
                     pos_err, rot_err, int(valid.sum()),
                     float((live_img[:, :, 0] - off_img[:, :, 0]).abs().max()),
                     float((live_img[:, :, 4] - off_img[:, :, 4]).abs().max())))

        if args_cli.dump:
            q_rp = quat_mul(quat_inv(yaw_quat(q_w)), q_w)         # root tilt, yaw removed
            dump[f"{name}/points"] = off_img[0, :, 1:4, :].cpu().numpy()
            dump[f"{name}/valid"] = valid[0].cpu().numpy()
            dump[f"{name}/depth"] = live_img[0, :, 0, :].cpu().numpy()
            dump[f"{name}/joint_pos"] = jp_w[0].cpu().numpy()
            dump[f"{name}/root_quat_rp"] = q_rp[0].cpu().numpy()
            dump[f"{name}/cam_pos"] = np.stack([p[0].cpu().numpy() for p, _ in off_poses])
            dump[f"{name}/cam_rot"] = np.stack([r[0].cpu().numpy() for _, r in off_poses])

    print(f"\n{'pose':<12} {'xyz med':>9} {'xyz max':>9} {'campos':>9} {'camrot':>9} "
          f"{'valid px':>9} {'d==':>6} {'v==':>6}")
    print("-" * 78)
    worst = 0.0
    for name, med, mx, pe, re_, nv, dd, dv in rows:
        worst = max(worst, med)
        print(f"{name:<12} {med * 1e3:8.3f}m {mx * 1e3:8.3f}m {pe * 1e3:8.3f}m "
              f"{re_ * 1e3:8.3f}r {nv:9d} {dd:6.0e} {dv:6.0e}")
    print("\n(xyz/campos in mm, camrot in mrad; 'd==' and 'v==' are the depth and validity "
          "planes,\n which must be 0 -- channel 0 is the shared input and validity derives "
          "from it.)")

    ok = worst <= args_cli.tol_mm * 1e-3
    print(f"\n[check] worst median xyz disagreement {worst * 1e3:.3f} mm "
          f"vs tolerance {args_cli.tol_mm} mm -> {'PASS' if ok else 'FAIL'}")

    if args_cli.dump:
        dump["joint_names"] = np.array(joint_names)
        dump["pose_names"] = np.array([p[0] for p in _POSES])
        dump["cam_names"] = np.array([str(c["sensor"]) for c in spec])
        np.savez(args_cli.dump, **dump)
        print(f"[check] wrote {args_cli.dump}")

    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
