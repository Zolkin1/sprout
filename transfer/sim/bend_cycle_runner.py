# Copyright 2026 Zachary Olkin. All rights reserved.

"""Sim2sim entry point for the bend-cycle policy, driven by joystick buttons.

The bend-cycle policy performs the full bend-over + bend-up motion as ONE periodic
trajectory parameterised by a phasing variable ``phi in [0, 1]``.  Here ``phi`` is
driven by two gamepad buttons instead of a clock:

* press **A** -> sweep ``phi`` 0 -> 0.5 (bend over),
* press **B** -> sweep ``phi`` 0.5 -> 1.0 (bend back up; wraps to standing at ``phi`` 0),
* press **A** -> sweep 0 -> 0.5 again, and so on.

Each sweep advances at the clip's natural rate (``dt / total_time`` per step, read
from the tracker metadata) so the motion plays at training speed, then HOLDS at the
target phase until the next press -- matching the fixed-phase holds the policy was
trained with.  Reuses the torch-free ONNX policy (:class:`BendCyclePolicy`) and the
MuJoCo :class:`Robot` wrapper.

Run from ``transfer/sim/`` (MuJoCo assets resolve relative to cwd):

    python bend_cycle_runner.py --load_run <bend_cycle_run>
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

# Torch-free deployment runtime (numpy + onnxruntime); put transfer/ on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.bend_cycle_policy import BendCyclePolicy  # noqa: E402
from deploy.gen_tracker import wxyz_to_xyzw  # noqa: E402

from robot import Robot  # noqa: E402


def find_latest_run(log_root: str) -> str | None:
    """Return the most recently modified run directory name under ``log_root``."""
    runs = glob.glob(os.path.join(log_root, "*"))
    return os.path.basename(max(runs, key=os.path.getmtime)) if runs else None


class PhaseButtonController:
    """Monotonic-forward phase driver: A -> bent (0.5), B -> standing (1.0 == 0).

    Holds ``phi`` at the last target until a button sets a new one, then advances
    ``phi`` forward (wrapping at 1.0) at ``dt / total_time`` per step until it
    reaches the target -- so each press plays one half of the cycle at the clip's
    natural speed and then holds.  Advancing forward-only keeps the reference on the
    trajectory (the clip is only defined in the forward direction); a target of 1.0
    lands on ``phi = 0`` (standing), from which A bends over again.
    """

    def __init__(self, bent_phi: float = 0.5, phi: float = 0.0):
        """Start held at ``phi`` (default 0.0 = standing) with no active target.

        Args:
            bent_phi: The phase 'bend over' (A) sweeps to and holds -- the phase the
                policy trained its bent hold at (the bend_down->bend_up domain
                boundary, ~0.474; read from the policy metadata by the runner).
            phi: Initial phase.
        """
        self.bent_phi = float(bent_phi)
        self.phi = float(phi)
        self.target: float | None = None

    def press_a(self) -> None:
        """Request the bent-over pose (sweep forward to the bent phase)."""
        self.target = self.bent_phi

    def press_b(self) -> None:
        """Request the standing pose (sweep forward to ``phi`` 1.0, i.e. back to 0)."""
        self.target = 1.0

    def step(self, dt: float, total_time: float) -> float:
        """Advance ``phi`` one control step toward the current target; return ``phi``.

        Args:
            dt: Control timestep (s).
            total_time: Clip period (s); the phase advances at ``dt / total_time``.

        Returns:
            The updated phasing variable in ``[0, 1)``.
        """
        if self.target is None:
            return self.phi
        step_delta = dt / total_time
        dist = (self.target - self.phi) % 1.0  # forward distance to target, in [0, 1)
        if dist == 0.0 or step_delta >= dist:
            self.phi = self.target % 1.0        # target 1.0 -> phi 0.0 (standing)
            self.target = None                  # reached -> hold
        else:
            self.phi = (self.phi + step_delta) % 1.0
        return self.phi


class Buttons:
    """Poll gamepad A/B buttons with rising-edge detection (pygame)."""

    def __init__(self, a_idx: int = 0, b_idx: int = 1):
        """Open joystick 0 and read buttons ``a_idx`` (A) / ``b_idx`` (B)."""
        import pygame

        self._pg = pygame
        self.a_idx, self.b_idx = a_idx, b_idx
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            print("[WARN] No joystick detected; A/B phase control disabled "
                  "(the policy just holds standing).")
            self.js = None
        else:
            self.js = pygame.joystick.Joystick(0)
            self.js.init()
            print(f"Using controller: {self.js.get_name()} (A=button {a_idx}, B=button {b_idx})")
        self._prev_a = self._prev_b = False

    def poll(self) -> tuple[bool, bool]:
        """Return ``(a_rising, b_rising)`` -- True only on the press (rising) edge."""
        if self.js is None:
            return False, False
        for _ in self._pg.event.get():
            pass
        a = bool(self.js.get_button(self.a_idx))
        b = bool(self.js.get_button(self.b_idx))
        a_rise, b_rise = (a and not self._prev_a), (b and not self._prev_b)
        self._prev_a, self._prev_b = a, b
        return a_rise, b_rise


class RealtimePacer:
    """Sleep so wall-clock keeps pace with (scaled) sim time; never speeds up."""

    def __init__(self, rt_factor: float):
        """Store the speed factor (``<= 0`` disables pacing, e.g. headless)."""
        self.rt_factor = rt_factor
        self._wall0: float | None = None
        self._sim0: float | None = None

    def reset(self, sim_time: float) -> None:
        """Anchor the pacer at the current wall clock and sim time."""
        self._wall0, self._sim0 = time.perf_counter(), sim_time

    def sync(self, sim_time: float) -> None:
        """Sleep until wall-clock elapsed reaches the scaled sim elapsed (no-op if behind)."""
        if self.rt_factor <= 0 or self._wall0 is None:
            return
        slack = self._wall0 + (sim_time - self._sim0) / self.rt_factor - time.perf_counter()
        if slack > 0:
            time.sleep(slack)


def plot_bend_cycle_logs(csv_path: str) -> None:
    """Read a bend-cycle sim log CSV and write diagnostic + joint-tracking PNGs alongside it.

    Diagnostics figure: base height + pitch, whole-body CoM_x vs the foot support
    (left/right ankle x), foot heights, and ankle-pitch torques -- the quantities
    that reveal a bent-pose tip. Joint figure: per-joint commanded (policy target)
    vs achieved position, mirroring the IsaacLab reference-vs-actual grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = d["time"]
    joints = [c[4:] for c in d.dtype.names if c.startswith("cmd_")]
    out_dir = os.path.dirname(csv_path)

    def overlay_phi(ax, label: bool = True):
        """Add the phase variable phi as a light gray twin-axis trace (0..1) for correlation."""
        ap = ax.twinx()
        ap.plot(t, d["phi"], color="0.55", lw=1.0, alpha=0.75, label="phi")
        ap.set_ylim(-0.05, 1.05)
        if label:
            ap.set_ylabel("phi (phase)", color="0.45", fontsize=8)
        else:
            ap.set_yticks([])
        return ap

    # --- diagnostics ---
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]; a.plot(t, d["base_z"], "b-", label="base z (m)")
    a.set_ylabel("base z (m)", color="b"); a2 = a.twinx()
    a2.plot(t, np.degrees(d["pitch"]), "r-", label="pitch"); a2.set_ylabel("base pitch (deg)", color="r")
    a.set_title("base height + pitch"); a.set_xlabel("t (s)"); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(t, d["com_x"], "k-", lw=2, label="CoM x")
    a.plot(t, d["lfoot_x"], "g--", label="left foot x"); a.plot(t, d["rfoot_x"], "m--", label="right foot x")
    a.fill_between(t, np.minimum(d["lfoot_x"], d["rfoot_x"]), np.maximum(d["lfoot_x"], d["rfoot_x"]),
                   color="gray", alpha=0.2, label="foot support (x)")
    a.set_title("whole-body CoM_x vs foot support (tips if CoM leaves the band)")
    a.set_xlabel("t (s)"); a.set_ylabel("world x (m)"); a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3)
    overlay_phi(a)

    a = ax[1, 0]
    a.plot(t, d["lfoot_z"], "g-", label="left foot z"); a.plot(t, d["rfoot_z"], "m-", label="right foot z")
    a.set_title("foot heights (lift-off)"); a.set_xlabel("t (s)"); a.set_ylabel("z (m)")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    a = ax[1, 1]
    for j, c in [("left_ankle_pitch_joint", "g"), ("right_ankle_pitch_joint", "m")]:
        if f"tau_{j}" in d.dtype.names:
            a.plot(t, d[f"tau_{j}"], c + "-", label=f"{j} tau")
    a.set_title("ankle-pitch torque (saturation?)"); a.set_xlabel("t (s)"); a.set_ylabel("torque (Nm)")
    a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); p1 = os.path.join(out_dir, "diagnostics.png"); fig.savefig(p1, dpi=110); plt.close(fig)

    # --- per-joint commanded vs achieved ---
    ncol = 4; nrow = int(np.ceil(len(joints) / ncol))
    fig, axg = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow), squeeze=False)
    for k, j in enumerate(joints):
        a = axg[k // ncol][k % ncol]
        a.plot(t, d[f"cmd_{j}"], "b-", label="commanded")
        a.plot(t, d[f"pos_{j}"], "r--", label="achieved")
        a.set_title(j, fontsize=8); a.tick_params(labelsize=6); a.grid(alpha=0.3)
    axg[0][0].legend(fontsize=7)
    for k in range(len(joints), nrow * ncol):
        axg[k // ncol][k % ncol].axis("off")
    fig.suptitle("Commanded (policy target) vs achieved joint position", fontsize=12)
    fig.tight_layout(); p2 = os.path.join(out_dir, "joint_tracking.png"); fig.savefig(p2, dpi=110); plt.close(fig)

    # --- all leg-joint torques (left vs right overlaid per joint type) ---
    leg_types = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
    fig, axl = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    for k, jt in enumerate(leg_types):
        a = axl[k // 3][k % 3]
        for side, c in (("left", "g"), ("right", "m")):
            col = f"tau_{side}_{jt}_joint"
            if col in d.dtype.names:
                a.plot(t, d[col], c + "-", label=side)
        a.set_title(f"{jt} torque", fontsize=10); a.set_xlabel("t (s)")
        a.set_ylabel("torque (Nm)"); a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)
        overlay_phi(a, label=(k % 3 == 2))  # phi label only on the rightmost column
    fig.suptitle("Leg-joint torques (left vs right, phi overlaid)", fontsize=12)
    fig.tight_layout(); p3 = os.path.join(out_dir, "leg_torques.png"); fig.savefig(p3, dpi=110); plt.close(fig)

    # --- sin/cos phase observations: prefer the RECORDED values (exactly what the
    # actor consumed); fall back to reconstruction for logs written before they existed.
    if "sin_phase_obs" in d.dtype.names and "cos_phase_obs" in d.dtype.names:
        sin_p, cos_p = d["sin_phase_obs"], d["cos_phase_obs"]
        src = "recorded from obs"
    else:
        two_pi_phi = 2.0 * np.pi * d["phi"]
        sin_p, cos_p = np.sin(two_pi_phi), np.cos(two_pi_phi)
        src = "reconstructed from phi"
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, sin_p, "b-", label="sin_phase"); ax[0].plot(t, cos_p, "r-", label="cos_phase")
    ax[0].set_ylabel("phase obs"); ax[0].set_title(f"sin / cos phase observations vs time ({src})")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3); ax[0].set_ylim(-1.1, 1.1)
    ax[1].plot(t, d["phi"], "k-", label="phi"); ax[1].set_ylabel("phi (phase)")
    ax[1].set_xlabel("t (s)"); ax[1].set_ylim(-0.05, 1.05); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)
    fig.tight_layout(); p4 = os.path.join(out_dir, "phase_obs.png"); fig.savefig(p4, dpi=110); plt.close(fig)
    print(f"[INFO] Wrote plots:\n  {p1}\n  {p2}\n  {p3}\n  {p4}")


def run(policy: BendCyclePolicy, robot: Robot, buttons: Buttons, phase: PhaseButtonController,
        sim_duration: float, headless: bool, rt_factor: float, warmup_time: float,
        log_path: str | None = None) -> bool:
    """Run the (optional) settle hold + live button-driven control loop.

    Args:
        policy: Loaded bend-cycle ONNX policy.
        robot: MuJoCo robot wrapper (spawned on the XML standing keyframe).
        buttons: Gamepad A/B reader (rising-edge).
        phase: The button-driven phase state machine.
        sim_duration: Stop after this many sim seconds (``<= 0`` runs until the
            viewer closes; required ``> 0`` when headless).
        headless: Run without the viewer.
        rt_factor: Viewer playback speed vs real time (``<= 0`` disables pacing).
        warmup_time: Seconds to PD-hold the default pose before going live (the
            policy balances once live; a short settle avoids the keyframe transient).

    Returns:
        True if the robot did not fall.
    """
    dt = policy.get_dt()
    decim = int(round(dt / robot.mj_model.opt.timestep))
    sync_every = max(1, math.ceil((1 / 50) / robot.mj_model.opt.timestep))
    pacer = RealtimePacer(rt_factor if not headless else 0.0)

    # RL PD gains on the actuators; robot is already on the XML standing keyframe.
    robot.set_pd_gains_from_policy(policy)
    hold_target = policy.convert_joint_order(
        np.asarray(policy._trk.default_joint_angles), policy.get_joint_names(), robot.joint_names)

    if headless and sim_duration <= 0:
        raise ValueError("Headless mode requires --total_time > 0.")

    # --- optional per-step logging ---
    joints_mj = list(robot.joint_names)
    log_rows: list[list[float]] = []
    if log_path:
        mm = robot.mj_model
        pelvis_id = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        lfoot_id = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        rfoot_id = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        act_ids = [mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_pos") for j in joints_mj]
        # Slices of the sin/cos phase terms in the assembled obs, so we log the EXACT
        # (scaled) values fed to the actor -- not a reconstruction from phi.
        sin_i, _ = policy.obs_term_slice("sin_phase")
        cos_i, _ = policy.obs_term_slice("cos_phase")

    def record(phi: float, action: np.ndarray, obs: np.ndarray) -> None:
        """Append one row of state + commanded targets + physical diagnostics.

        ``sin_phase_obs`` / ``cos_phase_obs`` are read straight out of the assembled
        observation ``obs`` (the exact values the actor consumed this step).
        """
        dta = robot.mj_data
        q, qd = dta.qpos, dta.qvel
        w, x, y, z = q[3], q[4], q[5], q[6]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        com = dta.subtree_com[pelvis_id]      # whole-robot CoM (world)
        lf, rf = dta.xpos[lfoot_id], dta.xpos[rfoot_id]
        tau = [float(dta.actuator_force[a]) for a in act_ids]
        obs_flat = np.asarray(obs).reshape(-1)
        sin_obs, cos_obs = float(obs_flat[sin_i]), float(obs_flat[cos_i])
        log_rows.append([float(dta.time), float(phi), sin_obs, cos_obs,
                         float(q[0]), float(q[1]), float(q[2]),
                         roll, pitch, yaw, float(qd[3]), float(qd[4]), float(qd[5]),
                         float(com[0]), float(com[1]), float(com[2]),
                         float(lf[0]), float(lf[2]), float(rf[0]), float(rf[2]),
                         *[float(v) for v in action], *[float(v) for v in q[7:]],
                         *[float(v) for v in qd[6:]], *tau])

    def advance(viewer) -> None:
        """Step ``decim`` physics sub-steps (viewer sync), then pace to real time."""
        for i in range(decim):
            robot.step()
            if viewer is not None and i % sync_every == 0:
                viewer.sync()
        pacer.sync(robot.mj_data.time)

    def control_step(viewer) -> None:
        """Read buttons, advance the phase, run the policy, apply the action."""
        a_rise, b_rise = buttons.poll()
        if a_rise:
            phase.press_a()
            print(f"[t={robot.mj_data.time:5.2f}s] A -> bend over (phi -> 0.5)")
        if b_rise:
            phase.press_b()
            print(f"[t={robot.mj_data.time:5.2f}s] B -> bend up   (phi -> 1.0/0)")
        phi = phase.step(dt, policy.total_time)
        qpos, qvel = robot.mj_data.qpos, robot.mj_data.qvel
        obs = policy.create_obs(
            wxyz_to_xyzw(qpos[3:7]), np.asarray(qvel[3:6]),
            np.asarray(qpos[7:]), np.asarray(qvel[6:]), robot.joint_names, phi)
        action = policy.get_action(obs, robot.joint_names)
        robot.apply_action(action)
        if log_path:
            record(phi, action, obs)

    def loop(viewer) -> bool:
        """Settle hold then live control; return False if the robot fell."""
        warm_steps = int(math.ceil(max(0.0, warmup_time) / dt))
        for _ in range(warm_steps):
            if viewer is not None and not viewer.is_running():
                return True
            robot.apply_action(hold_target)
            advance(viewer)
            if robot.failed():
                print("[WARN] Robot fell during the warmup hold.")
                return False
        pacer.reset(robot.mj_data.time)
        print("[INFO] Live: press A to bend over, B to bend back up.")
        while True:
            if viewer is not None and not viewer.is_running():
                return True
            if sim_duration > 0 and robot.mj_data.time > sim_duration:
                return True
            control_step(viewer)
            if robot.failed():
                print("[WARN] Robot fell.")
                return False
            advance(viewer)

    pacer.reset(robot.mj_data.time)
    if headless:
        success = loop(None)
    else:
        with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
            tid = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            viewer.cam.trackbodyid = tid
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            success = loop(viewer)

    if log_path and log_rows:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        header = (["time", "phi", "sin_phase_obs", "cos_phase_obs",
                   "base_x", "base_y", "base_z", "roll", "pitch", "yaw",
                   "avx", "avy", "avz", "com_x", "com_y", "com_z",
                   "lfoot_x", "lfoot_z", "rfoot_x", "rfoot_z"]
                  + [f"cmd_{j}" for j in joints_mj] + [f"pos_{j}" for j in joints_mj]
                  + [f"vel_{j}" for j in joints_mj] + [f"tau_{j}" for j in joints_mj])
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(log_rows)
        print(f"[INFO] Wrote {len(log_rows)} log rows to {log_path}")
        plot_bend_cycle_logs(log_path)
    return success


def main() -> None:
    """Parse arguments, build the policy + robot, and run the button-driven loop."""
    parser = argparse.ArgumentParser(description="Run the G1 bend-cycle policy in MuJoCo (A/B phase).")
    parser.add_argument("--experiment", type=str, default="g1")
    parser.add_argument("--load_run", type=str, default=None, help="Bend-cycle run dir (default: latest).")
    parser.add_argument("--tracker_onnx", type=str, default=None,
                        help="Explicit path to tracker.onnx (overrides --load_run resolution; use when "
                             "the run's logs live outside this checkout, e.g. running worktree code).")
    parser.add_argument("--scene", type=str, default="basic_scene")
    parser.add_argument("--robot_name", type=str, default="g1_21j")
    parser.add_argument("--num_threads", type=int, default=0, help="onnxruntime intra-op threads (0=auto).")
    parser.add_argument("--total_time", type=float, default=-1.0,
                        help="Stop after this many sim seconds (<=0 = until viewer close; required headless).")
    parser.add_argument("--warmup_time", type=float, default=0.25,
                        help="Seconds to PD-hold the default pose before going live.")
    parser.add_argument("--rt_factor", type=float, default=1.0,
                        help="Viewer playback speed vs real time (1.0=real; <=0 disables pacing).")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--a_button", type=int, default=0, help="Gamepad button index for 'bend over' (default 0=A).")
    parser.add_argument("--b_button", type=int, default=1, help="Gamepad button index for 'bend up' (default 1=B).")
    parser.add_argument("--bent_phi", type=float, default=None,
                        help="Phase 'bend over' (A) holds at. Default: read from the policy metadata "
                             "(the bend_down->bend_up domain boundary, ~0.474), else 0.5.")
    parser.add_argument("--log", action="store_true", default=False,
                        help="Log per-step state + commanded targets to CSV and write diagnostic plots.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="CSV path for --log (default: <run_dir>/mujoco_bend_cycle_logs/sim_log.csv).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    log_root = os.path.join(root_dir, "logs", "rsl_rl", args.experiment)

    if args.tracker_onnx:
        tracker_onnx = args.tracker_onnx
        if not os.path.exists(tracker_onnx):
            sys.exit(f"[ERROR] --tracker_onnx path does not exist: {tracker_onnx}")
        print(f"[INFO] tracker.onnx (explicit): {tracker_onnx}")
    else:
        run_name = args.load_run or find_latest_run(log_root)
        if not run_name:
            sys.exit(f"[ERROR] No runs found in {log_root}")
        tracker_onnx = os.path.join(log_root, run_name, "exported", "tracker.onnx")
        if not os.path.exists(tracker_onnx):
            sys.exit(f"[ERROR] Missing {tracker_onnx}. Export it with:\n"
                     f"  python scripts/rsl_rl/play.py --task G1-bend_cycle-clf-play --load_run {run_name}\n"
                     f"  python scripts/rsl_rl/export_tracker_onnx.py --run_dir {os.path.join(log_root, run_name)}\n"
                     f"  (or pass --tracker_onnx <path> directly)")
        print(f"[INFO] Bend-cycle run: {run_name}\n[INFO] tracker.onnx: {tracker_onnx}")

    policy = BendCyclePolicy(tracker_onnx, num_threads=args.num_threads)
    bent_phi = args.bent_phi if args.bent_phi is not None else policy.bent_phi
    print(f"[INFO] Loaded policy: num_obs={policy._num_obs}, total_time={policy.total_time:.3f}s, "
          f"bend-over holds at phi={bent_phi:.4f}"
          + ("" if "bent_hold_phi" in policy._trk.params else " (metadata had none -> fallback)"))

    # ``input_function`` set so the Robot skips its own (velocity) joystick init --
    # the A/B buttons are read by our own ``Buttons`` handler instead.
    robot = Robot(robot_name=args.robot_name, scene_name=args.scene,
                  joystick_scaling=np.ones(3), input_function=lambda t: np.zeros(3))

    buttons = Buttons(args.a_button, args.b_button)
    phase = PhaseButtonController(bent_phi=bent_phi)

    log_path = None
    if args.log or args.log_path:
        if args.log_path:
            log_path = args.log_path
        else:
            run_dir = os.path.dirname(os.path.dirname(tracker_onnx))  # <run>/exported/tracker.onnx -> <run>
            log_path = os.path.join(run_dir, "mujoco_bend_cycle_logs", "sim_log.csv")
        print(f"[INFO] Logging to {log_path} (+ diagnostics.png, joint_tracking.png alongside).")

    success = run(policy, robot, buttons, phase, args.total_time, args.headless,
                  args.rt_factor, args.warmup_time, log_path=log_path)
    print(f"[INFO] Run finished. Survived: {success}")


if __name__ == "__main__":
    main()
