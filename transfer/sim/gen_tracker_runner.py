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
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

# Torch-free deployment runtime (numpy + onnxruntime); transfer/ on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.gen_tracker import GenTrackerPolicy, wxyz_to_xyzw  # noqa: E402
from deploy.quat_math_np import quat_apply, quat_box_minus, quat_inv  # noqa: E402
from deploy.tracker_session import convert_joint_order  # noqa: E402

from robot import Robot  # noqa: E402
from clip_reference import ClipReference  # noqa: E402
import experiments.velocity_commands as vc  # noqa: E402


def find_latest_run(log_root_path: str) -> str | None:
    """Return the most recently modified run directory name under ``log_root_path``."""
    run_dirs = glob.glob(os.path.join(log_root_path, "*"))
    return os.path.basename(max(run_dirs, key=os.path.getmtime)) if run_dirs else None


class JoystickCommand:
    """Read ``(vx, vy, wz)`` from a gamepad, scaled by per-axis maxima."""

    def __init__(self, max_vx: float, max_vy: float, max_wz: float, start_button: int = 0):
        """Initialise pygame and open joystick 0 (falls back to a constant command)."""
        import pygame

        self._pygame = pygame
        self.scale = np.array([max_vx, max_vy, max_wz])
        self.start_button = start_button
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            print("[WARN] No joystick detected; using a constant [0.5, 0, 0] command.")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Using controller: {self.joystick.get_name()}")

    def button_pressed(self, idx: int | None = None) -> bool:
        """Return True if a gamepad button (default the start/A button) is currently pressed."""
        if self.joystick is None:
            return False
        for _ in self._pygame.event.get():
            pass
        return bool(self.joystick.get_button(self.start_button if idx is None else idx))

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


class RealtimePacer:
    """Throttle the loop so wall-clock keeps pace with sim time.

    Sleeps until the wall-clock elapsed since :meth:`reset` reaches the (scaled)
    sim-time elapsed. Never speeds up a compute-bound sim: if the loop is already
    behind real time the slack is negative and no sleep happens.
    """

    def __init__(self, rt_factor: float):
        """Store the speed factor (``<= 0`` disables pacing, e.g. headless/eval)."""
        self.rt_factor = rt_factor
        self._wall0: float | None = None
        self._sim0: float | None = None

    def reset(self, sim_time: float) -> None:
        """Anchor the pacer at the current wall clock and sim time."""
        self._wall0 = time.perf_counter()
        self._sim0 = sim_time

    def sync(self, sim_time: float) -> None:
        """Sleep until wall-clock elapsed reaches the scaled sim elapsed (no-op if behind)."""
        if self.rt_factor <= 0 or self._wall0 is None:
            return
        target = self._wall0 + (sim_time - self._sim0) / self.rt_factor
        slack = target - time.perf_counter()
        if slack > 0:
            time.sleep(slack)


def build_state(robot: Robot, policy: GenTrackerPolicy, command: np.ndarray) -> dict:
    """Assemble the world-state dict ``GenTrackerPolicy.create_obs`` expects.

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
    jp = convert_joint_order(np.asarray(qpos[7:]), robot.joint_names, isaac)
    jv = convert_joint_order(np.asarray(qvel[6:]), robot.joint_names, isaac)
    state = {
        "root_pos": np.asarray(qpos[:3], dtype=np.float64),
        "root_quat": wxyz_to_xyzw(qpos[3:7]),
        "root_lin_vel_w": lin_w,
        "root_ang_vel_w": ang_w,
        "base_ang_vel": np.asarray(qvel[3:6], dtype=np.float64),  # body-local = Isaac base_ang_vel
        "joint_pos": jp,
        "joint_vel": jv,
        "command": np.asarray(command, dtype=np.float64),
    }
    if policy.ctx.num_extra_frames > 0:
        fr = robot.get_body_world_states(policy.ctx.frame_names)  # [K, 13]
        state.update(frame_pos=fr[:, 0:3], frame_quat=fr[:, 3:7], frame_lin=fr[:, 7:10], frame_ang=fr[:, 10:13])
    return state


def load_standing_pose(path: str, isaac_joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ``*_config_solution.yaml`` standing config into a robot spawn state.

    Reads the solved standing pose from ``bezier_coeffs`` (joint angles keyed by name
    and the ``pelvis_link`` frame pose), using the first control point of each spline.

    Args:
        path: Path to the standing ``config_solution.yaml``.
        isaac_joint_names: Joint order to return joint positions in (the FM/tracker order).

    Returns:
        ``(joint_pos_isaac, base_pos, base_quat_wxyz)`` -- joints in ``isaac_joint_names``
        order, pelvis world position (x, y re-zeroed), and normalized ``(w, x, y, z)`` quat.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    domain = cfg["domain_sequence"][0]  # e.g. "double_support"
    bz = cfg[domain]["bezier_coeffs"]
    joints = bz["joints"]
    jp = np.array([float(joints[n][0]) for n in isaac_joint_names], dtype=np.float64)
    pelvis = bz["frames"]["pelvis_link"]
    base_pos = np.array([0.0, 0.0, float(pelvis["pos_z"][0])], dtype=np.float64)  # re-zero x, y
    quat = np.array([float(pelvis["ori_w"][0]), float(pelvis["ori_x"][0]),
                     float(pelvis["ori_y"][0]), float(pelvis["ori_z"][0])], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return jp, base_pos, quat


def _yaw_from_xyzw(q: np.ndarray) -> float:
    """Yaw angle (rad) of an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rpy_from_xyzw(q: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw (rad) of an ``(x, y, z, w)`` quaternion (ZYX convention)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _add_sphere(scn, pos: np.ndarray, rgba: tuple, radius: float) -> None:
    """Append a sphere geom to a viewer ``user_scn`` (no-op if at capacity)."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_arrow(scn, base: np.ndarray, vec: np.ndarray, rgba: tuple, width: float = 0.02) -> None:
    """Append an arrow geom from ``base`` along ``vec`` (skipped if ~zero length)."""
    vec = np.asarray(vec, dtype=np.float64)
    if float(np.linalg.norm(vec)) < 1e-3 or scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(base, dtype=np.float64), np.asarray(base + vec, dtype=np.float64))
    scn.ngeom += 1


def run(policy: GenTrackerPolicy, robot: Robot, command_fn, total_time: float,
        headless: bool, log_path: str | None, warmup_time: float, rt_factor: float,
        standing_config: str | None, viz: bool, vel_arrow_scale: float, vel_arrow_height: float,
        hold_kp_scale: float, hold_kv_scale: float, spawn_z_offset: float,
        cmd_max_acc: float, fm_traj_points: int,
        clip: ClipReference | None = None, clip_handoff_time: float = -1.0) -> bool:
    """Run the warmup + control loop; return True if the robot did not fall.

    The robot first PD-holds a stable stand for ``warmup_time`` seconds (at least the
    FM history length) while the generator history fills with real snapshots, then
    hands control to the FM + tracker. Playback is paced to real time (scaled by
    ``rt_factor``) in viewer mode; headless/eval runs are never throttled.

    Args:
        policy: Loaded generator + tracker policy.
        robot: MuJoCo robot wrapper (gains already set).
        command_fn: ``sim_time -> (vx, vy, wz)`` command source.
        total_time: Stop after this many seconds (``<= 0`` runs until the viewer closes).
        headless: Run without the viewer.
        log_path: CSV path for per-step metrics, or ``None`` to skip logging.
        warmup_time: Seconds to PD-hold the stand before live control (``<= 0`` skips
            the hold and falls back to a static warm-start fill).
        rt_factor: Real-time speed multiple for viewer playback (``1.0`` = real time,
            ``0.5`` = half speed; ``<= 0`` disables pacing).
        standing_config: Path to a standing ``config_solution.yaml`` to spawn + hold; if
            missing/None the robot holds the tracker's ``default_joint_angles`` instead.
        viz: Draw FM root/feet spheres + velocity arrows in the viewer (viewer mode only).
        vel_arrow_scale: Arrow length in metres per (m/s) of planar speed.
        vel_arrow_height: Height (m) above the pelvis to draw the velocity arrows.
        hold_kp_scale: Leg (hip/knee/ankle) kp multiplier during the hold. Fixed-joint PD
            at the XML gains cannot balance the robot; a stiff, well-damped hold keeps it
            standing while the buffer fills. Wiped by the RL gains at handoff.
        hold_kv_scale: Leg kv (damping) multiplier during the hold.
        spawn_z_offset: Metres to raise the spawn pelvis height above the standing-config
            value (the optimizer's pelvis frame sits ~3 cm below the MuJoCo foot-contact
            height, so without this the feet penetrate the floor and pop up).
        cmd_max_acc: Velocity-command rate limit (m/s^2, rad/s^2) applied to the live
            command to match training. Overridden by ``max_acc`` in the tracker metadata
            when present; ``<= 0`` disables ramping.
        fm_traj_points: Number of evenly-spaced spheres drawn along the FM predicted root
            (and feet) trajectory in the viewer; ``<= 0`` draws every predicted node (a
            dense ribbon, matching the IsaacLab debug viz).
        clip: Optional baked-clip reference source. When given, the robot spawns on the
            clip's step-0 pose and the tracker is driven by the clip reference (no FM)
            instead of the standing warmup + FM path.
        clip_handoff_time: In clip mode, sim time (s) at which to hand the reference off
            from the clip to the FM generator (the FM history is kept warm during the
            clip phase). ``< 0`` never hands off (clip-only playback).
    """
    dt = policy.get_dt()
    decim = int(round(dt / robot.mj_model.opt.timestep))
    sync_every = max(1, math.ceil((1 / 50) / robot.mj_model.opt.timestep))

    # Per-joint torque readout + effort limits in the tracker (isaac) joint order.
    isaac = policy.get_joint_names()
    act_ids = [mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_pos") for n in isaac]
    effort_limits = []
    for n in isaac:
        jid = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
        fr = robot.mj_model.jnt_actfrcrange[jid]
        effort_limits.append(float(max(abs(fr[0]), abs(fr[1]))))

    # Stable-stand warmup: spawn + PD-hold the solved standing pose (so the generator
    # history fills with in-distribution standing snapshots) for at least the FM window.
    if clip is not None:
        # Clip mode: spawn directly on the clip's start-step pose and hand control to the
        # tracker immediately (no standing warmup), RL gains from the start -- matching
        # the IsaacLab reset_on_motion_sequence spawn. The FM history is filled live by
        # the clip loop's per-step observe() so a later clip->FM handoff is seamless.
        c_pos, c_quat_wxyz, c_jp = clip.spawn_state()
        c_pos = c_pos.copy()
        c_pos[2] += spawn_z_offset  # lift feet clear of the floor (optimizer frame offset)
        hold_target = convert_joint_order(c_jp, isaac, robot.joint_names)
        robot.set_base_and_joints(c_pos, c_quat_wxyz, hold_target)
        robot.set_pd_gains_from_policy(policy)
        warmup_steps = 0
        print(f"[INFO] Clip mode: spawned on clip step {clip.start_step} pose "
              f"(t={clip.start_step * clip.dt:.2f}s, z+{spawn_z_offset}); RL gains set.")
    elif standing_config and os.path.exists(standing_config):
        jp_isaac, base_pos, base_quat = load_standing_pose(standing_config, isaac)
        base_pos[2] += spawn_z_offset  # lift feet clear of the floor (optimizer frame offset)
        hold_target = convert_joint_order(jp_isaac, isaac, robot.joint_names)
        robot.set_base_and_joints(base_pos, base_quat, hold_target)
        print(f"[INFO] Standing pose spawned + held from {standing_config}")
    else:
        if standing_config:
            print(f"[WARN] standing_config not found ({standing_config}); holding default_joint_angles.")
        hold_target = convert_joint_order(policy._trk.default_joint_angles, isaac, robot.joint_names)
    # Stiffen the legs for the hold: fixed-joint PD at the XML gains tips over (the ankles
    # alone can't balance the inverted pendulum), so a stiff, well-damped leg holds a stand.
    if clip is None and (hold_kp_scale != 1.0 or hold_kv_scale != 1.0):
        robot.scale_gains(hold_kp_scale, hold_kv_scale, name_filter=["hip", "knee", "ankle"])
        print(f"[INFO] Hold gains: leg kp x{hold_kp_scale}, kv x{hold_kv_scale} "
              "(reverts to RL gains at handoff).")
    if clip is None:
        warmup_steps = max(policy.ctx.N, int(math.ceil(warmup_time / dt))) if warmup_time > 0 else 0
        if warmup_steps == 0:
            # No hold: fall back to a static warm-start fill so the FM is valid at step 0.
            policy.warm_start(build_state(robot, policy, np.zeros(3)))

    # Predicted extra-frame indices that are feet (for the FM viz spheres).
    foot_idx = [i for i, n in enumerate(policy.ctx.target_frame_names)
                if "ankle" in n.lower() or "foot" in n.lower()]
    if viz and not headless:
        print(f"[INFO] FM target frames: {policy.ctx.target_frame_names}; feet -> {foot_idx}")

    # Manual handoff gate: in interactive (joystick) viewer runs, hold the stand until the
    # operator presses the start button, so standing can be verified before going live.
    gate = (not headless) and isinstance(command_fn, JoystickCommand) and command_fn.joystick is not None
    min_hold = max(warmup_steps, policy.ctx.N) if gate else warmup_steps

    # Velocity-command rate limit, matching training. Prefer the value embedded in the
    # tracker metadata; otherwise fall back to --cmd_max_acc (the clip-library ramp rate).
    meta_max_acc = policy._trk.params.get("max_acc")
    eff_max_acc = meta_max_acc if meta_max_acc is not None else cmd_max_acc
    print(f"[INFO] Command rate limit: {eff_max_acc} "
          f"({'from tracker metadata' if meta_max_acc is not None else 'from --cmd_max_acc default'})"
          + ("" if eff_max_acc and eff_max_acc > 0 else " (disabled)"))

    ee_names = list(policy.ctx.target_frame_names)
    jp_cols = [c for jn in isaac for c in (f"jpref_{jn}", f"jpact_{jn}")]
    ee_cols = [f"ee{kind}_{n}_{ax}" for n in ee_names for kind in ("ref", "act") for ax in ("x", "y", "z")]

    log_file = writer = None
    log_meta = log_meta_path = None
    # Sim time (s) at which the FM generator takes over the reference (clip->FM handoff,
    # or the warmup->FM handoff in non-clip runs). Stays None if the FM never drives
    # (clip-only playback). Captured live below and written to sim_meta.yaml at the end
    # so the metrics plots can mark the transition.
    handoff_time = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["time", "cmd_vx", "cmd_vy", "cmd_wz", "vx_b", "vy_b", "wz_b",
                         "pos_err", "ori_err", "joint_pos_err",
                         "ref_pos_x", "ref_pos_y", "ref_pos_z", "act_pos_x", "act_pos_y", "act_pos_z",
                         "ref_roll", "ref_pitch", "ref_yaw", "act_roll", "act_pitch", "act_yaw",
                         "base_z", "fell",
                         *[f"tau_{n}" for n in isaac], *jp_cols, *ee_cols])
        # Sidecar metadata for the metrics module (joint order + effort limits + EE frames).
        log_meta = {"joint_names_isaac": isaac, "effort_limits": effort_limits,
                    "ee_frame_names": ee_names, "dt": policy.get_dt(),
                    "fm_update_every": policy.fm_update_every}
        log_meta_path = os.path.join(os.path.dirname(log_path), "sim_meta.yaml")
        with open(log_meta_path, "w") as mf:
            yaml.dump(log_meta, mf, default_flow_style=False, sort_keys=False)

    vel_cmd = np.zeros(3)  # rate-limited command state (ramps from 0 at handoff)

    def control_step() -> tuple[np.ndarray, dict]:
        nonlocal vel_cmd
        desired = np.asarray(command_fn(robot.mj_data.time), dtype=np.float64)
        if eff_max_acc and eff_max_acc > 0:
            lim = eff_max_acc * dt
            vel_cmd = np.clip(desired, vel_cmd - lim, vel_cmd + lim)
        else:
            vel_cmd = desired
        state = build_state(robot, policy, vel_cmd)
        obs = policy.create_obs(state)
        action = policy.get_action(obs, robot.joint_names)
        robot.apply_action(action)
        return action, state

    def log_step(state: dict, ref: dict | None = None) -> None:
        if not writer:
            return
        ref = ref if ref is not None else policy._reference_from_fm()
        cur = ref["current"]
        rq = np.asarray(state["root_quat"], dtype=np.float64)
        root_pos = np.asarray(state["root_pos"], dtype=np.float64)
        ref_pos = np.asarray(cur["root_pos"], dtype=np.float64)
        ref_quat = np.asarray(cur["root_quat"], dtype=np.float64)
        ref_jp = np.asarray(cur["jp"], dtype=np.float64)
        act_jp = np.asarray(state["joint_pos"], dtype=np.float64)
        pos_err = float(np.linalg.norm(ref_pos - root_pos))
        ori_err = float(np.linalg.norm(quat_box_minus(rq, ref_quat)))
        jp_err = float(np.linalg.norm(ref_jp - act_jp))
        ref_rpy = _rpy_from_xyzw(ref_quat)  # desired base orientation (roll, pitch, yaw)
        act_rpy = _rpy_from_xyzw(rq)         # achieved base orientation
        inv_root = quat_inv(rq)
        lin_b = quat_apply(inv_root, np.asarray(state["root_lin_vel_w"], dtype=np.float64))
        torques = [float(robot.mj_data.actuator_force[a]) for a in act_ids]
        # Per-joint FM reference vs actual (isaac order).
        jp_vals = [v for i in range(len(isaac)) for v in (ref_jp[i], act_jp[i])]
        # Per-EE FM reference vs actual position, expressed in the current base frame.
        ee_vals: list[float] = []
        if ee_names:
            ref_fp = np.asarray(cur["frame_pos"], dtype=np.float64)         # (k, 3) world
            act_fp = robot.get_body_world_states(ee_names)[:, 0:3]          # (k, 3) world
            for k in range(len(ee_names)):
                rb = quat_apply(inv_root, ref_fp[k] - root_pos)
                ab = quat_apply(inv_root, act_fp[k] - root_pos)
                ee_vals += [rb[0], rb[1], rb[2], ab[0], ab[1], ab[2]]
        writer.writerow([robot.mj_data.time, *state["command"].tolist(),
                         float(lin_b[0]), float(lin_b[1]), float(state["base_ang_vel"][2]),
                         pos_err, ori_err, jp_err,
                         float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2]),
                         float(root_pos[0]), float(root_pos[1]), float(root_pos[2]),
                         float(ref_rpy[0]), float(ref_rpy[1]), float(ref_rpy[2]),
                         float(act_rpy[0]), float(act_rpy[1]), float(act_rpy[2]),
                         float(root_pos[2]), int(robot.failed()),
                         *torques, *jp_vals, *ee_vals])

    def draw_viz(viewer, state: dict) -> None:
        """Populate ``user_scn`` with FM root/feet spheres + desired/actual velocity arrows."""
        if viewer is None or not viz:
            return
        scn = viewer.user_scn
        scn.ngeom = 0
        base_pos = np.asarray(state["root_pos"], dtype=np.float64)
        # FM prediction spheres (only once the generator has a valid prediction).
        if policy._pred_valid:
            if fm_traj_points <= 0:
                idxs = np.arange(policy.ctx.F)  # every predicted node -- dense ribbon (like IsaacLab)
            else:
                n_pts = max(1, min(fm_traj_points, policy.ctx.F))
                idxs = np.unique(np.linspace(0, policy.ctx.F - 1, n_pts).astype(int))
            rp = policy._pred["root_pos"][0]  # (F, 3) predicted root path
            for i in idxs:
                _add_sphere(scn, rp[int(i)], (1.0, 0.55, 0.0, 0.85), 0.02)
            if foot_idx:
                fp = policy._pred["frame_pos"][0]  # (F, k, 3) predicted feet paths
                for i in idxs:
                    for fi in foot_idx:
                        _add_sphere(scn, fp[int(i), fi], (0.0, 0.9, 1.0, 0.95), 0.02)
        # Velocity arrows above the pelvis: green = desired (joystick), blue = actual.
        head = base_pos + np.array([0.0, 0.0, vel_arrow_height])
        yaw = _yaw_from_xyzw(np.asarray(state["root_quat"], dtype=np.float64))
        cmd = np.asarray(state["command"], dtype=np.float64)
        cmd_w = np.array([cmd[0] * math.cos(yaw) - cmd[1] * math.sin(yaw),
                          cmd[0] * math.sin(yaw) + cmd[1] * math.cos(yaw), 0.0])
        act_w = np.asarray(state["root_lin_vel_w"], dtype=np.float64) * np.array([1.0, 1.0, 0.0])
        _add_arrow(scn, head, cmd_w * vel_arrow_scale, (0.0, 1.0, 0.0, 1.0))   # desired
        _add_arrow(scn, head, act_w * vel_arrow_scale, (0.0, 0.0, 1.0, 1.0))   # actual

    def advance_physics(state: dict, viewer, ref: dict | None = None) -> None:
        """Step ``decim`` physics sub-steps (logging + viewer sync), then pace to real time."""
        for i in range(decim):
            robot.step()
            if i % sync_every == 0:
                log_step(state, ref)
                if viewer is not None:
                    viewer.sync()
        pacer.sync(robot.mj_data.time)

    def run_loop(viewer) -> bool:
        """Warmup hold then live FM + tracker control; return False if the robot fell."""
        nonlocal handoff_time
        # Warmup: PD-hold the stand (cmd = 0, XML default gains) while the history fills.
        # When gated, hold until the start button is pressed (after the buffer is full).
        if gate:
            print(f"[INFO] Holding standing pose -- press gamepad button {command_fn.start_button} "
                  "(A) to hand control to the RL tracker.")
        held = 0
        while True:
            if viewer is not None and not viewer.is_running():
                return True
            if total_time > 0 and robot.mj_data.time > total_time:
                return True
            state = build_state(robot, policy, np.zeros(3))
            policy.observe(state)
            robot.apply_action(hold_target)
            draw_viz(viewer, state)
            advance_physics(state, viewer)
            if robot.failed():
                print("[WARN] Robot fell during the warmup hold (check PD gains / spawn pose).")
                return False
            held += 1
            if held >= min_hold and (not gate or command_fn.button_pressed()):
                break
        # Switch from the XML default gains to the RL tracker gains for live control.
        robot.set_pd_gains_from_policy(policy)
        handoff_time = robot.mj_data.time
        print(f"[INFO] Handoff: switched to RL PD gains, handing control to the FM + tracker "
              f"(t={handoff_time:.2f}s).")
        # Live: hand control to the FM + tracker.
        while True:
            if viewer is not None and not viewer.is_running():
                return True
            if total_time > 0 and robot.mj_data.time > total_time:
                return True
            _, state = control_step()
            if robot.failed():
                return False
            draw_viz(viewer, state)
            advance_physics(state, viewer)

    def run_clip_loop(viewer) -> bool:
        """Play the baked clip through the tracker; optionally hand off to the FM.

        Each step the command is the clip's ``target_vel`` (already ramped at
        generation, so no extra rate limit) and the tracker reference is the clip
        (``observe()`` keeps the FM history warm). Once ``clip_handoff_time`` is
        reached the reference source switches to the live FM generator; the command
        keeps following the authored clip velocity profile. ``clip_handoff_time < 0``
        never hands off (clip-only). Returns False if the robot fell.
        """
        nonlocal handoff_time
        clip_only = clip_handoff_time < 0
        handed_off = False
        clip_step = clip.start_step
        while True:
            if viewer is not None and not viewer.is_running():
                return True
            if total_time > 0 and robot.mj_data.time > total_time:
                return True
            s = min(clip_step, clip.num_steps - 1)
            state = build_state(robot, policy, clip.command_at(s))
            if (not clip_only) and robot.mj_data.time >= clip_handoff_time:
                if not handed_off:
                    handoff_time = robot.mj_data.time
                    print(f"[INFO] Clip->FM handoff at t={handoff_time:.2f}s "
                          f"(clip step {s}/{clip.num_steps - 1}); FM now drives the reference.")
                    handed_off = True
                # Feed the FM the clip's ACTUAL upcoming command horizon (matches IsaacLab's
                # get_command_velocity_horizon), not the held current command.
                policy.set_future_command(clip.future_command_horizon(s, policy.ctx.F))
                obs = policy.create_obs(state)  # FM reference (history warm from observe())
                ref = None                       # log against the FM reference
            else:
                # Ego-centric (receding-horizon) reference: periodically re-place the
                # clip onto the robot's current root (planar pos + heading) so the
                # reference recedes ahead of the robot, mirroring the teacher's
                # training-time ego re-anchoring. World-anchored mode (default) leaves
                # the fixed placement untouched.
                if clip.ego_anchor and (clip_step - clip.start_step) % max(clip.ego_anchor_every, 1) == 0:
                    clip.reanchor_ego(s, state["root_pos"], state["root_quat"])
                policy.observe(state)            # keep the FM history warm for handoff
                ref = clip.reference_at(s, policy.future_offsets)
                obs = policy._build_obs(state, reference=ref)
            action = policy.get_action(obs, robot.joint_names)
            robot.apply_action(action)
            if robot.failed():
                return False
            draw_viz(viewer, state)
            advance_physics(state, viewer, ref)
            clip_step += 1

    loop_fn = run_clip_loop if clip is not None else run_loop

    pacer = RealtimePacer(rt_factor if not headless else 0.0)
    if headless and total_time <= 0:
        raise ValueError("Headless mode requires --total_time > 0.")

    pacer.reset(robot.mj_data.time)
    if headless:
        success = loop_fn(None)
    else:
        with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
            viewer.cam.trackbodyid = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            success = loop_fn(viewer)

    if log_file:
        log_file.close()
        print(f"[INFO] Wrote metrics log to {log_path}")
    if log_meta is not None:
        # Re-dump now that the actual FM-handoff time is known (None if the FM never
        # took over, e.g. clip-only playback) so the metrics plots can mark the transition.
        log_meta["fm_handoff_time"] = handoff_time
        with open(log_meta_path, "w") as mf:
            yaml.dump(log_meta, mf, default_flow_style=False, sort_keys=False)
    return success


def main() -> None:
    """Parse arguments, build the policy + robot, and run the control loop."""
    parser = argparse.ArgumentParser(description="Run the G1 generator+tracker pair in MuJoCo.")
    parser.add_argument("--experiment", type=str, default="g1")
    parser.add_argument("--load_run", type=str, default=None, help="Tracker run dir (default: latest).")
    parser.add_argument("--fm_onnx", type=str, default=None,
                        help="Generator ONNX (default: <run>/exported/fm.onnx).")
    parser.add_argument("--mode", choices=["joystick", "eval"], default="joystick")
    parser.add_argument("--command", type=str, default="smooth_ramp_running",
                        help="Scripted velocity profile in experiments.velocity_commands (eval mode).")
    parser.add_argument("--scene", type=str, default="basic_scene")
    parser.add_argument("--robot_name", type=str, default="g1_21j")
    parser.add_argument("--num_threads", type=int, default=0, help="onnxruntime intra-op threads (0=auto).")
    parser.add_argument("--fm_update_every", type=int, default=2,
                        help="Generator re-inference cadence (control steps). Fallback only: "
                             "overridden by 'fm_update_every' in the tracker metadata if present.")
    parser.add_argument("--total_time", type=float, default=-1.0)
    parser.add_argument("--warmup_time", type=float, default=1.0,
                        help="Seconds to PD-hold a stand before live control (0 disables; "
                             "always at least the FM history window).")
    parser.add_argument("--rt_factor", type=float, default=1.0,
                        help="Viewer playback speed vs real time (1.0=real, 0.5=half; <=0 disables pacing).")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--use_ic", action="store_true", default=False)
    parser.add_argument("--standing_config", type=str, default=None,
                        help="Standing config_solution.yaml to spawn + hold during warmup "
                             "(default: merged library 'standing' clip; pass '' to disable).")
    parser.add_argument("--no_viz", dest="viz", action="store_false", default=True,
                        help="Disable the FM root/feet spheres + velocity arrows in the viewer.")
    parser.add_argument("--vel_arrow_scale", type=float, default=0.5,
                        help="Velocity-arrow length in metres per (m/s).")
    parser.add_argument("--vel_arrow_height", type=float, default=0.6,
                        help="Height (m) above the pelvis to draw the velocity arrows.")
    parser.add_argument("--hold_kp_scale", type=float, default=6.0,
                        help="Leg kp multiplier during the standing hold (XML gains can't "
                             "balance the robot); reverts to RL gains at handoff.")
    parser.add_argument("--hold_kv_scale", type=float, default=8.0,
                        help="Leg kv (damping) multiplier during the standing hold.")
    parser.add_argument("--spawn_z_offset", type=float, default=0.03,
                        help="Metres to raise the spawn pelvis above the standing-config height "
                             "(optimizer frame sits ~3 cm below the MuJoCo foot-contact height).")
    parser.add_argument("--cmd_max_acc", type=float, default=2.0,
                        help="Velocity-command rate limit (m/s^2, rad/s^2) to match training; "
                             "the clip library ramps at 2.0. Overridden by tracker metadata "
                             "'max_acc' if present; <=0 disables.")
    parser.add_argument("--fm_traj_points", type=int, default=0,
                        help="Number of spheres drawn along the FM predicted root/feet trajectory; "
                             "<=0 draws every predicted node (dense ribbon, like IsaacLab).")
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--max_vx", type=float, default=1.0)
    parser.add_argument("--max_vy", type=float, default=0.5)
    parser.add_argument("--max_wz", type=float, default=1.0)
    parser.add_argument("--start_button", type=int, default=0,
                        help="Gamepad button index to end the standing hold and start the RL "
                             "tracker (default 0 = A on an Xbox pad).")
    parser.add_argument("--clip", type=str, default=None,
                        help="Baked clip-library .h5 (e.g. clips/g1_parity_v1.h5). Spawns on the "
                             "clip's step-0 pose and drives the tracker from the clip reference "
                             "(no FM) -- the deterministic IsaacLab<->MuJoCo parity playback.")
    parser.add_argument("--clip_handoff_time", type=float, default=None,
                        help="In --clip mode, sim time (s) to hand the reference off from the clip "
                             "to the FM generator. Omit (default) for clip-only playback; pass a "
                             "value inside the clip (e.g. 8.0) for a clip->FM handoff run.")
    parser.add_argument("--clip_start_time", type=float, default=0.0,
                        help="In --clip mode, seconds into the clip to begin playback (default 0). "
                             "The robot spawns on the clip pose at this time and the reference is "
                             "served from here -- e.g. to skip a standing prefix. Sim time and the "
                             "--clip_handoff_time gate are still measured from this start.")
    parser.add_argument("--clip_ego_anchor", action="store_true", default=False,
                        help="In --clip mode, use the ego-centric (receding-horizon) reference: "
                             "periodically re-place the clip onto the robot's current root (planar "
                             "pos + heading), matching the teacher's ego-anchored training. Default "
                             "(off) is the world-anchored fixed-placement clip.")
    parser.add_argument("--clip_ego_anchor_every", type=int, default=10,
                        help="Re-anchor cadence (control steps) for --clip_ego_anchor; should match "
                             "the teacher's ego_anchor_every (default 10, ~0.2s at dt=0.02).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    log_root = os.path.join(root_dir, "logs", "rsl_rl", args.experiment)

    run_name = args.load_run or find_latest_run(log_root)
    if not run_name:
        sys.exit(f"[ERROR] No runs found in {log_root}")
    run_dir = os.path.join(log_root, run_name)
    exported = os.path.join(run_dir, "exported")
    tracker_onnx = os.path.join(exported, "tracker.onnx")
    fm_onnx = args.fm_onnx or os.path.join(exported, "fm.onnx")
    # Clip-only playback (a clip with no FM handoff) drives the tracker straight from the
    # clip reference, so the generator is optional -- this is how a pure clip-tracking
    # teacher (which has no fm.onnx) is played in MuJoCo. Every other mode (joystick/eval,
    # or a clip->FM handoff) needs the generator.
    clip_only = bool(args.clip) and args.clip_handoff_time is None
    if not os.path.exists(tracker_onnx):
        sys.exit(f"[ERROR] Missing {tracker_onnx}. Export with export_tracker_onnx.py first.")
    if clip_only:
        fm_onnx = None
    elif not os.path.exists(fm_onnx):
        sys.exit(f"[ERROR] Missing {fm_onnx}. Export with export_fm_onnx.py, or play clip-only "
                 f"(--clip <h5> without --clip_handoff_time) to run the tracker with no FM.")
    print(f"[INFO] Tracker run: {run_name}\n[INFO] tracker.onnx: {tracker_onnx}\n"
          f"[INFO] fm.onnx: {fm_onnx if fm_onnx else '(none -- tracker-only clip playback)'}")

    policy = GenTrackerPolicy(
        tracker_onnx, fm_onnx, fm_update_every=args.fm_update_every, num_threads=args.num_threads,
    )
    policy.load()

    robot = Robot(robot_name=args.robot_name, scene_name=args.scene,
                  joystick_scaling=np.ones(3), input_function=lambda t: np.zeros(3))
    if args.use_ic:
        robot.set_initial_condition(policy._trk)

    if args.clip:
        # Clip playback drives the velocity command from the clip's target_vel, so the
        # joystick / scripted command source is unused (run_clip_loop ignores command_fn).
        # Skip building it -- in particular the pygame/SDL joystick init, which otherwise
        # conflicts with the MuJoCo viewer's GLFW context at interpreter teardown and
        # produces a benign-but-ugly shutdown segfault.
        command_fn = lambda _t: np.zeros(3)
    elif args.mode == "joystick":
        command_fn = JoystickCommand(args.max_vx, args.max_vy, args.max_wz, args.start_button)
    else:
        if not hasattr(vc, args.command):
            sys.exit(f"[ERROR] No velocity profile '{args.command}' in experiments.velocity_commands.")
        command_fn = getattr(vc, args.command)

    log_path = None
    if args.log:
        log_dir = os.path.join(run_dir, "mujoco_gen_tracker_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "sim_log.csv")

    # Default the standing pose to the merged library's 'standing' clip ('' disables).
    standing_config = args.standing_config
    if standing_config is None:
        standing_config = os.path.join(root_dir, "hf", "trajectories", "retargeted",
                                       "2026-06-16_16-30-26_merged", "standing",
                                       "standing_config_solution.yaml")

    # Optional baked-clip playback (clip-only or clip->FM handoff).
    clip = None
    clip_handoff_time = -1.0
    total_time = args.total_time
    if args.clip:
        clip_path = args.clip if os.path.isabs(args.clip) else os.path.join(root_dir, args.clip)
        clip = ClipReference(clip_path, policy.get_joint_names(),
                             list(policy.ctx.target_frame_names), policy.base_frame_name,
                             start_time=args.clip_start_time,
                             ego_anchor=args.clip_ego_anchor,
                             ego_anchor_every=args.clip_ego_anchor_every)
        if abs(clip.dt - policy.get_dt()) > 1e-6:
            print(f"[WARN] Clip dt {clip.dt} != policy dt {policy.get_dt()}; "
                  "regenerate the clip at the control dt.")
        clip_handoff_time = args.clip_handoff_time if args.clip_handoff_time is not None else -1.0
        # Playable duration from the chosen start to the clip end (sim time begins at 0 here).
        remaining_s = (clip.num_steps - clip.start_step) * clip.dt
        if total_time <= 0:  # clip-only -> remaining length; clip->FM -> a few extra seconds of FM.
            total_time = remaining_s if clip_handoff_time < 0 else remaining_s + 5.0
        print(f"[INFO] Clip: {clip_path}\n"
              f"[INFO] {clip.num_steps} steps, {clip.clip_length_s:.2f}s @ dt={clip.dt}; "
              f"start={clip.start_step * clip.dt:.2f}s (step {clip.start_step}); "
              f"handoff={'clip-only' if clip_handoff_time < 0 else f'{clip_handoff_time:.2f}s'}; "
              f"anchor={f'ego/every {args.clip_ego_anchor_every}' if args.clip_ego_anchor else 'world'}; "
              f"total_time={total_time:.2f}s")

    success = run(policy, robot, command_fn, total_time, args.headless, log_path,
                  args.warmup_time, args.rt_factor, standing_config, args.viz,
                  args.vel_arrow_scale, args.vel_arrow_height, args.hold_kp_scale,
                  args.hold_kv_scale, args.spawn_z_offset, args.cmd_max_acc, args.fm_traj_points,
                  clip=clip, clip_handoff_time=clip_handoff_time)
    print(f"[INFO] Run finished. Survived: {success}")

    if log_path:
        import gen_tracker_metrics
        gen_tracker_metrics.create_metrics(os.path.dirname(log_path))


if __name__ == "__main__":
    main()
