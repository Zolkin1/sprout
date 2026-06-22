# Copyright 2026 Zachary Olkin. All rights reserved.

"""Sim2sim entry point for the flow-matching generator + RL tracker pair.

Drives the ``GeneratorTrackerPolicy`` in MuJoCo with two command sources:

* ``--mode joystick``: drive the velocity command ``(vx, vy, wz)`` by hand with a
  gamepad (left stick = vx/vy, right stick x = wz), scaled by ``--max_*``.
* ``--mode eval``: a scripted velocity profile from ``experiments.velocity_commands``
  (``--command``), logging tracking metrics to CSV.

Run from ``transfer/sim/`` (the MuJoCo assets are resolved relative to cwd):

    python gen_tracker_runner.py --mode joystick --load_run <tracker_run>
    python gen_tracker_runner.py --mode eval --command smooth_ramp_running --log

The flow-matching checkpoint defaults to the one the tracker was trained against
(read from the run's ``params/env.yaml``); override with ``--fm_checkpoint``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from gen_tracker_policy import GeneratorTrackerPolicy, _wxyz_to_xyzw  # also puts transfer/ on sys.path
from robot import Robot
from fm.quat_math import quat_apply, quat_box_minus, quat_inv
import experiments.velocity_commands as vc


def find_latest_run(log_root_path: str) -> str | None:
    """Return the most recently modified run directory name under ``log_root_path``."""
    run_dirs = glob.glob(os.path.join(log_root_path, "*"))
    return os.path.basename(max(run_dirs, key=os.path.getmtime)) if run_dirs else None


def fm_checkpoint_from_env_yaml(run_dir: str) -> str | None:
    """Read ``commands.traj_ref.fm_checkpoint`` from a run's ``params/env.yaml``."""
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.exists(env_yaml):
        return None
    with open(env_yaml) as f:
        cfg = yaml.unsafe_load(f)
    try:
        return cfg["commands"]["traj_ref"]["fm_checkpoint"]
    except (KeyError, TypeError):
        return None


class JoystickCommand:
    """Read ``(vx, vy, wz)`` from a gamepad, scaled by per-axis maxima."""

    def __init__(self, max_vx: float, max_vy: float, max_wz: float):
        """Initialise pygame and open joystick 0 (falls back to a constant command)."""
        import pygame

        self._pygame = pygame
        self.scale = np.array([max_vx, max_vy, max_wz])
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            print("[WARN] No joystick detected; using a constant [0.5, 0, 0] command.")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Using controller: {self.joystick.get_name()}")

    def __call__(self, sim_time: float) -> np.ndarray:
        """Return the current command ``(vx, vy, wz)``."""
        if self.joystick is None:
            return np.array([0.5, 0.0, 0.0])
        for _ in self._pygame.event.get():
            pass
        vx = -self.joystick.get_axis(1) * self.scale[0]
        vy = -self.joystick.get_axis(0) * self.scale[1]
        wz = -self.joystick.get_axis(3) * self.scale[2]
        return np.array([vx, vy, wz])


def build_state(robot: Robot, policy: GeneratorTrackerPolicy, command: np.ndarray) -> dict:
    """Assemble the world-state dict ``GeneratorTrackerPolicy.create_obs`` expects.

    Reads only base pose + base world-frame velocity + joints + command (and, when
    the loaded generator's history needs them, the extra-frame world states).

    Args:
        robot: The MuJoCo robot wrapper.
        policy: The loaded generator + tracker policy (for joint order + ctx).
        command: Velocity command ``(vx, vy, wz)``.

    Returns:
        State dict (numpy, FM/isaac joint order, world frame, xyzw quats).
    """
    qpos, qvel = robot.mj_data.qpos, robot.mj_data.qvel
    lin_w, ang_w = robot.get_base_world_velocity(policy.base_frame_name)
    isaac = policy.get_joint_names()
    jp = policy._rl.convert_joint_order(np.asarray(qpos[7:]), robot.joint_names, isaac)
    jv = policy._rl.convert_joint_order(np.asarray(qvel[6:]), robot.joint_names, isaac)
    state = {
        "root_pos": np.asarray(qpos[:3], dtype=np.float64),
        "root_quat": _wxyz_to_xyzw(qpos[3:7]),
        "root_lin_vel_w": lin_w,
        "root_ang_vel_w": ang_w,
        "base_ang_vel": np.asarray(qvel[3:6], dtype=np.float64),  # body-local = Isaac base_ang_vel
        "joint_pos": jp,
        "joint_vel": jv,
        "command": np.asarray(command, dtype=np.float64),
    }
    if policy.ctx.history_num_extra_frames > 0:
        fr = robot.get_body_world_states(policy.ctx.history_frame_names)  # [K, 13]
        state.update(frame_pos=fr[:, 0:3], frame_quat=fr[:, 3:7], frame_lin=fr[:, 7:10], frame_ang=fr[:, 10:13])
    return state


def run(policy: GeneratorTrackerPolicy, robot: Robot, command_fn, total_time: float,
        headless: bool, log_path: str | None) -> bool:
    """Run the control loop; return True if the robot did not fall.

    Args:
        policy: Loaded generator + tracker policy.
        robot: MuJoCo robot wrapper (gains already set).
        command_fn: ``sim_time -> (vx, vy, wz)`` command source.
        total_time: Stop after this many seconds (``<= 0`` runs until the viewer closes).
        headless: Run without the viewer.
        log_path: CSV path for per-step metrics, or ``None`` to skip logging.
    """
    decim = int(round(policy.get_dt() / robot.mj_model.opt.timestep))
    sync_every = max(1, math.ceil((1 / 50) / robot.mj_model.opt.timestep))

    # Warm-start the generator history with the (standing) initial state, cmd = 0.
    policy.warm_start(build_state(robot, policy, np.zeros(3)))

    # Per-joint torque readout + effort limits in the tracker (isaac) joint order.
    isaac = policy.get_joint_names()
    act_ids = [mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_pos") for n in isaac]
    effort_limits = []
    for n in isaac:
        jid = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
        fr = robot.mj_model.jnt_actfrcrange[jid]
        effort_limits.append(float(max(abs(fr[0]), abs(fr[1]))))

    log_file = writer = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["time", "cmd_vx", "cmd_vy", "cmd_wz", "vx_b", "vy_b", "wz_b",
                         "pos_err", "ori_err", "joint_pos_err", "base_z", "fell",
                         *[f"tau_{n}" for n in isaac]])
        # Sidecar metadata for the metrics module (joint order + effort limits).
        meta = {"joint_names_isaac": isaac, "effort_limits": effort_limits,
                "dt": policy.get_dt(), "fm_update_every": policy.fm_update_every}
        with open(os.path.join(os.path.dirname(log_path), "sim_meta.yaml"), "w") as mf:
            yaml.dump(meta, mf, default_flow_style=False, sort_keys=False)

    def control_step() -> tuple[np.ndarray, dict]:
        cmd = np.asarray(command_fn(robot.mj_data.time), dtype=np.float64)
        state = build_state(robot, policy, cmd)
        obs = policy.create_obs(state)
        action = policy.get_action(obs, robot.joint_names)
        robot.apply_action(action)
        return action, state

    def log_step(state: dict) -> None:
        if not writer:
            return
        ref = policy._reference_from_fm()
        rq = torch.tensor(state["root_quat"])
        ref_pos = ref["current"]["root_pos"].double()
        ref_quat = ref["current"]["root_quat"].double()
        pos_err = float((ref_pos - torch.tensor(state["root_pos"])).norm())
        ori_err = float(quat_box_minus(rq, ref_quat).norm())
        jp_err = float((ref["current"]["jp"].double() - torch.tensor(state["joint_pos"])).norm())
        lin_b = quat_apply(quat_inv(rq), torch.tensor(state["root_lin_vel_w"]))
        torques = [float(robot.mj_data.actuator_force[a]) for a in act_ids]
        writer.writerow([robot.mj_data.time, *state["command"].tolist(),
                         float(lin_b[0]), float(lin_b[1]), float(state["base_ang_vel"][2]),
                         pos_err, ori_err, jp_err, float(state["root_pos"][2]), int(robot.failed()),
                         *torques])

    robot.set_pd_gains_from_policy(policy)
    success = True

    if headless:
        if total_time <= 0:
            raise ValueError("Headless mode requires --total_time > 0.")
        while robot.mj_data.time < total_time:
            _, state = control_step()
            if robot.failed():
                success = False
                break
            for i in range(decim):
                robot.step()
                if i % sync_every == 0:
                    log_step(state)
    else:
        with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
            viewer.cam.trackbodyid = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            while viewer.is_running():
                if total_time > 0 and robot.mj_data.time > total_time:
                    break
                _, state = control_step()
                if robot.failed():
                    success = False
                    break
                for i in range(decim):
                    robot.step()
                    if i % sync_every == 0:
                        log_step(state)
                        viewer.sync()

    if log_file:
        log_file.close()
        print(f"[INFO] Wrote metrics log to {log_path}")
    return success


def main() -> None:
    """Parse arguments, build the policy + robot, and run the control loop."""
    parser = argparse.ArgumentParser(description="Run the G1 generator+tracker pair in MuJoCo.")
    parser.add_argument("--experiment", type=str, default="g1")
    parser.add_argument("--load_run", type=str, default=None, help="Tracker run dir (default: latest).")
    parser.add_argument("--fm_checkpoint", type=str, default=None,
                        help="Generator checkpoint (default: read from the run's params/env.yaml).")
    parser.add_argument("--mode", choices=["joystick", "eval"], default="joystick")
    parser.add_argument("--command", type=str, default="smooth_ramp_running",
                        help="Scripted velocity profile in experiments.velocity_commands (eval mode).")
    parser.add_argument("--scene", type=str, default="basic_scene")
    parser.add_argument("--robot_name", type=str, default="g1_21j")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fm_device", type=str, default=None)
    parser.add_argument("--fm_update_every", type=int, default=2)
    parser.add_argument("--total_time", type=float, default=-1.0)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--use_ic", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--max_vx", type=float, default=1.0)
    parser.add_argument("--max_vy", type=float, default=0.5)
    parser.add_argument("--max_wz", type=float, default=1.0)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    log_root = os.path.join(root_dir, "logs", "rsl_rl", args.experiment)

    run_name = args.load_run or find_latest_run(log_root)
    if not run_name:
        sys.exit(f"[ERROR] No runs found in {log_root}")
    run_dir = os.path.join(log_root, run_name)
    exported = os.path.join(run_dir, "exported")
    policy_path = os.path.join(exported, "policy.pt")
    param_path = os.path.join(exported, "policy_parameters.yaml")
    for p in (policy_path, param_path):
        if not os.path.exists(p):
            sys.exit(f"[ERROR] Missing {p}. Run play.py to export the tracker first.")

    fm_ckpt = args.fm_checkpoint or fm_checkpoint_from_env_yaml(run_dir)
    if not fm_ckpt:
        sys.exit("[ERROR] No FM checkpoint: pass --fm_checkpoint (none found in params/env.yaml).")
    if not os.path.isabs(fm_ckpt):
        fm_ckpt = os.path.join(root_dir, fm_ckpt)
    print(f"[INFO] Tracker run: {run_name}\n[INFO] FM checkpoint: {fm_ckpt}")

    policy = GeneratorTrackerPolicy(
        param_path, policy_path, fm_ckpt,
        device=args.device, fm_device=args.fm_device, fm_update_every=args.fm_update_every,
    )
    policy.load()

    robot = Robot(robot_name=args.robot_name, scene_name=args.scene,
                  joystick_scaling=np.ones(3), input_function=lambda t: np.zeros(3))
    if args.use_ic:
        robot.set_initial_condition(policy._rl)

    if args.mode == "joystick":
        command_fn = JoystickCommand(args.max_vx, args.max_vy, args.max_wz)
    else:
        if not hasattr(vc, args.command):
            sys.exit(f"[ERROR] No velocity profile '{args.command}' in experiments.velocity_commands.")
        command_fn = getattr(vc, args.command)

    log_path = None
    if args.log:
        log_dir = os.path.join(run_dir, "mujoco_gen_tracker_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "sim_log.csv")

    success = run(policy, robot, command_fn, args.total_time, args.headless, log_path)
    print(f"[INFO] Run finished. Survived: {success}")

    if log_path:
        import gen_tracker_metrics
        gen_tracker_metrics.create_metrics(os.path.dirname(log_path))


if __name__ == "__main__":
    main()
