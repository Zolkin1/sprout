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


def run(policy: BendCyclePolicy, robot: Robot, buttons: Buttons, phase: PhaseButtonController,
        sim_duration: float, headless: bool, rt_factor: float, warmup_time: float) -> bool:
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
        return loop(None)
    with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
        tid = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        viewer.cam.trackbodyid = tid
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        return loop(viewer)


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

    success = run(policy, robot, buttons, phase, args.total_time, args.headless,
                  args.rt_factor, args.warmup_time)
    print(f"[INFO] Run finished. Survived: {success}")


if __name__ == "__main__":
    main()
