# Copyright 2026 Zachary Olkin. All rights reserved.

"""Sim2sim entry point for the flow-matching generator + RL tracker pair.

Drives the torch-free ``GenTrackerPolicy`` (``transfer/deploy``) in MuJoCo with two
command sources:

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
from deploy.gen_tracker import GenTrackerPolicy, wxyz_to_xyzw, xyzw_to_wxyz  # noqa: E402
from deploy.quat_math_np import quat_apply, quat_box_minus, quat_inv  # noqa: E402
from deploy.tracker_session import convert_joint_order  # noqa: E402
from deploy.window_np import window_block_slices  # noqa: E402

from robot import Robot  # noqa: E402
from clip_reference import ClipReference  # noqa: E402
from depth_camera import MujocoDepthCameras  # noqa: E402
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


def build_state(
    robot: Robot, policy: GenTrackerPolicy, command: np.ndarray,
    depth_cams: MujocoDepthCameras | None = None, want_depth_hits: bool = False,
) -> dict:
    """Assemble the world-state dict ``GenTrackerPolicy.create_obs`` expects.

    Reads only base pose + base world-frame velocity + joints + command (and, when
    the loaded generator's history needs them, the extra-frame world states), plus
    whichever exteroception the loaded artifacts actually use.

    Args:
        robot: The MuJoCo robot wrapper.
        policy: The loaded generator + tracker policy (for joint order + ctx).
        command: Velocity command ``(vx, vy, wz)``.
        depth_cams: The depth cameras to cast, or ``None`` when the policy has none.
        want_depth_hits: Also collect the depth rays' world hit points (viewer only).

    Returns:
        State dict (numpy, FM/isaac joint order, world frame, xyzw quats).
    """
    qpos, qvel = robot.mj_data.qpos, robot.mj_data.qvel
    lin_w, ang_w = robot.get_base_world_velocity(policy.base_frame_name)
    isaac = policy.get_joint_names()
    jp = convert_joint_order(np.asarray(qpos[7:]), robot.joint_names, isaac)
    jv = convert_joint_order(np.asarray(qvel[6:]), robot.joint_names, isaac)
    state = {
        # Copies, not views: ``np.asarray`` on a float64 slice of mj_data aliases live physics,
        # so a consumer that keeps the array (the FM history, the CSV log) would silently see the
        # state as of whenever it reads, not as of now.
        "root_pos": np.array(qpos[:3], dtype=np.float64, copy=True),
        "root_quat": wxyz_to_xyzw(qpos[3:7]),
        "root_lin_vel_w": lin_w,
        "root_ang_vel_w": ang_w,
        "base_ang_vel": np.array(qvel[3:6], dtype=np.float64, copy=True),  # body-local = Isaac base_ang_vel
        "joint_pos": jp,
        "joint_vel": jv,
        "command": np.asarray(command, dtype=np.float64),
    }
    if policy.ctx.num_extra_frames > 0:
        fr = robot.get_body_world_states(policy.ctx.frame_names)  # [K, 13]
        state.update(frame_pos=fr[:, 0:3], frame_quat=fr[:, 3:7], frame_lin=fr[:, 7:10], frame_ang=fr[:, 10:13])
    # Height scan (per-ray ground clearance): raycast a yaw-aligned grid under the pelvis
    # against the MuJoCo terrain, replicating IsaacLab mdp.height_scan. Computed when the
    # tracker observes a scan (spec present) or the FM conditions on one -- NOT merely
    # because the FM has wide exteroception tokens, which a depth FM also has.
    spec = policy._trk.get_height_scan_spec() if hasattr(policy._trk, "get_height_scan_spec") else None
    fm_uses_scan = (
        getattr(policy.ctx, "extero_source", "height_scan") == "height_scan"
        and getattr(policy.ctx, "height_dim", 1) > 1
    )
    if spec is not None or fm_uses_scan:
        s = spec or {}
        off = s.get("offset_pos", [0.75, 0.0, 20.0])
        clr, hits = robot.compute_height_scan(
            state["root_pos"], state["root_quat"],
            resolution=float(s.get("resolution", 0.1)), size=tuple(s.get("size", (1.5, 1.0))),
            offset_fwd=float(off[0]), ray_start_h=float(off[2]),
            sensor_offset=float(s.get("sensor_offset", 0.5)),
        )
        # Missed rays must match IsaacLab's encoding, NOT 0.0. Isaac's miss is ``pos_z - inf -
        # offset`` = -inf, which sanitize_height_scan maps to -HEIGHT_SCAN_CLIP = -1.0; that is
        # what both the teacher obs and the FM height token were trained on. Mapping a miss to
        # 0.0 instead reads as "ground exactly at the nominal sensor plane" -- plausible flat
        # ground where Isaac reports the band edge. Flat terrain never misses, so this was
        # invisible until isolated (has_floor=False) stair/box tiles, where rays overshoot the
        # tile edge and ~2-3% of them miss right at the stair transition.
        state["height_scan"] = np.nan_to_num(clr, nan=-1.0)
        state["height_scan_hits"] = hits
    # Depth cameras: cast the policy's own pinhole grids against the terrain AND the
    # robot's visual meshes, normalised exactly as the observation term does. One cast
    # feeds BOTH consumers -- the tracker's depth obs terms (keyed by term name) and the
    # FM's exteroception tokens (stacked, matching mdp.depth_obs_stack) -- because there
    # is one physical camera pair -- the policy keys the rows by obs term itself, from the
    # contract it loaded. Both cameras run at the control rate, so a single cast per step
    # is exact: no staleness to model.
    if depth_cams is not None:
        if want_depth_hits:
            state["depth"], state["depth_hits"] = depth_cams.compute(robot.mj_data, want_hits=True)
        else:
            state["depth"] = depth_cams.compute(robot.mj_data)
    return state


def load_standing_pose(path: str, isaac_joint_names: list[str],
                       fallback: dict[str, float] | None = None,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ``*_config_solution.yaml`` standing config into a robot spawn state.

    Reads the solved standing pose from ``bezier_coeffs`` (joint angles keyed by name
    and the ``pelvis_link`` frame pose), using the first control point of each spline.

    The solved configs predate the 29-DOF unlock, so they cover only the 21 original joints.
    Any joint the file lacks is filled from ``fallback`` (the policy's own
    ``default_joint_angles``, which IS the rest value it expects for those DOFs) rather than
    raising -- otherwise a 29-DOF policy cannot spawn from a 21-joint standing solution.

    Args:
        path: Path to the standing ``config_solution.yaml``.
        isaac_joint_names: Joint order to return joint positions in (the FM/tracker order).
        fallback: Per-joint angle for names absent from the file; missing -> 0.0.

    Returns:
        ``(joint_pos_isaac, base_pos, base_quat_wxyz)`` -- joints in ``isaac_joint_names``
        order, pelvis world position (x, y re-zeroed), and normalized ``(w, x, y, z)`` quat.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    domain = cfg["domain_sequence"][0]  # e.g. "double_support"
    bz = cfg[domain]["bezier_coeffs"]
    joints = bz["joints"]
    fallback = fallback or {}
    filled = [n for n in isaac_joint_names if n not in joints]
    if filled:
        print(f"[INFO] standing config has no {len(filled)} joint(s) {filled}; "
              f"using the policy's default angles for them.")
    jp = np.array([float(joints[n][0]) if n in joints else float(fallback.get(n, 0.0))
                   for n in isaac_joint_names], dtype=np.float64)
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


def _print_collider_inventory(model) -> None:
    """List which bodies actually own a COLLIDING geom, and which only look solid.

    Worth printing rather than eyeballing: MuJoCo renders geom groups 0, 1 and 2 by
    default, so a collision geom and the visual mesh it duplicates are drawn on top of each
    other. A geom whose contact is disabled therefore looks exactly like one that collides,
    and a limb with no collider at all looks exactly like one that has a full mesh. Neither
    is visible in the viewer without turning groups off (``--viz_geomgroup``).
    """
    name = lambda b: mujoco.mj_name2id and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    colliding, inert = {}, {}
    for g in range(model.ngeom):
        b = name(model.geom_bodyid[g])
        if b == "world":
            continue
        tgt = colliding if (model.geom_contype[g] or model.geom_conaffinity[g]) else inert
        tgt[b] = tgt.get(b, 0) + 1
    solid_only = sorted(set(inert) - set(colliding))
    print(f"[INFO] MuJoCo colliders: {len(colliding)} bodies collide, "
          f"{len(solid_only)} are drawn but CANNOT collide.")
    print(f"       collide : {', '.join(sorted(colliding))}")
    print(f"       drawn-only: {', '.join(solid_only)}")


def _apply_viz_options(viewer, model, geomgroup: list[int] | None, contacts: bool) -> None:
    """Set the viewer's geom-group + contact visualisation options.

    Args:
        viewer: The passive viewer handle.
        model: The MuJoCo model (for the collider inventory print).
        geomgroup: Groups to RENDER (0 = collision, 1 = visual meshes, 2 = terrain). ``None``
            keeps MuJoCo's default, which renders 0, 1 and 2 together.
        contacts: Show contact points + forces, and make geoms transparent so the points are
            not hidden inside the meshes.
    """
    if geomgroup is not None:
        for i in range(len(viewer.opt.geomgroup)):
            viewer.opt.geomgroup[i] = 1 if i in geomgroup else 0
        print(f"[INFO] rendering geom groups {sorted(geomgroup)} only "
              "(0 = collision, 1 = visual, 2 = terrain).")
    if contacts:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        print("[INFO] contact points + forces on, geoms transparent. NOTE: nothing will be "
              "drawn where nothing touches -- an empty view means no contact, which for a "
              "self-collision question is the answer rather than a failure.")
    _print_collider_inventory(model)


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


def box_top(robot: Robot, spec: str) -> tuple[float, float, float] | None:
    """Centre + top height of a named box geom in the loaded scene.

    Reads the geometry from the MJCF rather than re-deriving it, so it stays correct for any
    scene ``generate_stairs_scene.py`` emits (or a hand-edited one).

    Args:
        robot: The MuJoCo robot wrapper (its model holds the scene).
        spec: Box geom name (``box2``) or a bare index (``2`` -> ``box2``).

    Returns:
        ``(x, y, top_z)`` in world metres, or None when no such geom exists.
    """
    name = f"box{spec}" if spec.strip().isdigit() else spec.strip()
    gid = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        avail = [mujoco.mj_id2name(robot.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                 for i in range(robot.mj_model.ngeom)]
        print(f"[WARN] no geom named {name!r}; boxes in this scene: "
              f"{[a for a in avail if a.startswith('box')]}")
        return None
    pos, half = robot.mj_model.geom_pos[gid], robot.mj_model.geom_size[gid]
    return float(pos[0]), float(pos[1]), float(pos[2] + half[2])


def run(policy: GenTrackerPolicy, robot: Robot, command_fn, total_time: float,
        headless: bool, log_path: str | None, warmup_time: float, rt_factor: float,
        standing_config: str | None, viz: bool, vel_arrow_scale: float, vel_arrow_height: float,
        hold_kp_scale: float, hold_kv_scale: float, spawn_z_offset: float,
        clip_spawn_z_offset: float,
        cmd_max_acc: float, fm_traj_points: int, spawn_on_box: str | None = None,
        clip: ClipReference | None = None, clip_handoff_time: float = -1.0,
        depth_cams: MujocoDepthCameras | None = None, depth_viz_stride: int = 8,
        viz_geomgroup: list[int] | None = None, viz_contacts: bool = False) -> bool:
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
        warmup_time: Seconds to PD-hold the stand before live control. DEFAULTS TO 0 (skip the
            hold; fall back to a static warm-start fill), because ``null_history_warmup``
            makes the FM valid at step 0 without needing real snapshots in the window, so the
            hold only delays the thing you launched the runner to see. A positive value also
            arms the joystick start-button gate in viewer runs.
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
        clip_spawn_z_offset: Metres to raise the pelvis in CLIP mode. Default 0: a clip frame is
            authored with the feet already on the terrain, and IsaacLab's reset spawns it
            verbatim, so any lift here is a pure sim2sim mismatch -- and an expensive one. It
            shifts the root AND every height-scan ray uniformly, and the FM's height token is
            sensitive enough that 3 cm of lift moved its predicted window by 0.17 m. The
            standing-spawn lift below is a different thing (it clears the solved standing pose
            off the floor) and keeps its own default.
        spawn_z_offset: Metres to raise the spawn pelvis height above the standing-config
            value (the optimizer's pelvis frame sits ~3 cm below the MuJoCo foot-contact
            height, so without this the feet penetrate the floor and pop up).
        cmd_max_acc: Velocity-command rate limit (m/s^2, rad/s^2) applied to the live
            command; ``<= 0`` disables ramping (the runner's default -- see ``--cmd_max_acc``).
            Overridden by ``max_acc`` in the tracker metadata when present.
        fm_traj_points: Number of evenly-spaced spheres drawn along the FM predicted root
            (and feet) trajectory in the viewer; ``<= 0`` draws every predicted node (a
            dense ribbon, matching the IsaacLab debug viz).
        spawn_on_box: Name or index of a scene box to spawn standing on TOP of, instead of
            the floor (``None`` -> floor). Only the support height changes; the standing
            joint pose is kept. Warns and falls back to the floor if the box is not found.
        clip: Optional baked-clip reference source. When given, the robot spawns on the
            clip's step-0 pose and the tracker is driven by the clip reference (no FM)
            instead of the standing warmup + FM path.
        clip_handoff_time: In clip mode, sim time (s) at which to hand the reference off
            from the clip to the FM generator (the FM history is kept warm during the
            clip phase). ``< 0`` never hands off (clip-only playback).
        depth_cams: Depth cameras to cast each control step, or ``None`` when the loaded
            policy has none. Feeds the tracker's depth obs terms and the FM's
            exteroception tokens from one cast.
        depth_viz_stride: Draw every Nth depth pixel's hit point in the viewer. The two
            cameras produce 1560 hits per step, well past ``user_scn.maxgeom``, so the
            cloud is subsampled.
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
        c_pos, c_quat_wxyz, c_jp, c_vel_w, c_jv = clip.spawn_state()
        c_pos = c_pos.copy()
        c_pos[2] += clip_spawn_z_offset   # 0 by default: spawn EXACTLY where IsaacLab does
        hold_target = convert_joint_order(c_jp, isaac, robot.joint_names)
        # Velocities too, not just the pose: clips start mid-stride, and Isaac's reset writes
        # root + joint velocity from ``dy_des``. Spawning at rest starts the robot a whole
        # stride-speed behind its reference and stalls the FM history (see spawn_state).
        robot.set_base_and_joints(c_pos, c_quat_wxyz, hold_target,
                                  base_lin_vel_w=c_vel_w[:3], base_ang_vel_w=c_vel_w[3:6],
                                  joint_vel_mujoco=convert_joint_order(c_jv, isaac,
                                                                       robot.joint_names))
        robot.set_pd_gains_from_policy(policy)
        warmup_steps = 0
        print(f"[INFO] Clip mode: spawned on clip step {clip.start_step} state "
              f"(t={clip.start_step * clip.dt:.2f}s, z+{clip_spawn_z_offset}, "
              f"|v|={np.linalg.norm(c_vel_w[:3]):.2f} m/s); RL gains set.")
    elif standing_config and os.path.exists(standing_config):
        jp_isaac, base_pos, base_quat = load_standing_pose(
            standing_config, isaac,
            fallback=dict(zip(isaac, np.asarray(policy._trk.default_joint_angles).ravel())),
        )
        base_pos[2] += spawn_z_offset  # lift feet clear of the floor (optimizer frame offset)
        # Optionally stand ON a box instead of the floor: keep the solved standing JOINTS and
        # just translate the base to the box top, so the pose is unchanged and only the
        # support height moves (what a jump-off / dismount test needs as its start state).
        if spawn_on_box is not None:
            top = box_top(robot, spawn_on_box)
            if top is not None:
                bx, by, bz = top
                base_pos = np.array([bx, by, base_pos[2] + bz], dtype=np.float64)
                print(f"[INFO] Spawning ON box {spawn_on_box!r}: top z={bz:.3f} m at "
                      f"(x={bx:.2f}, y={by:.2f}) -> pelvis z={base_pos[2]:.3f} m")
            else:
                print("[WARN] --spawn_on_box ignored (box not found); spawning on the floor.")
        hold_target = convert_joint_order(jp_isaac, isaac, robot.joint_names)
        robot.set_base_and_joints(base_pos, base_quat, hold_target)
        print(f"[INFO] Standing pose spawned + held from {standing_config}")
    else:
        if standing_config:
            print(f"[WARN] standing_config not found ({standing_config}); holding default_joint_angles.")
        hold_target = convert_joint_order(policy._trk.default_joint_angles, isaac, robot.joint_names)
    # Stiffen the legs for the hold: fixed-joint PD at the XML gains tips over (the ankles
    # alone can't balance the inverted pendulum), so a stiff, well-damped leg holds a stand.
    if clip is None and warmup_time > 0 and (hold_kp_scale != 1.0 or hold_kv_scale != 1.0):
        robot.scale_gains(hold_kp_scale, hold_kv_scale, name_filter=["hip", "knee", "ankle"])
        print(f"[INFO] Hold gains: leg kp x{hold_kp_scale}, kv x{hold_kv_scale} "
              "(reverts to RL gains at handoff).")
    elif clip is None and warmup_time <= 0:
        print("[INFO] warmup_time<=0: spawning straight into FM+tracker control "
              "(no standing hold, no stiffened hold gains, no start-button gate).")
    if clip is None:
        warmup_steps = max(policy.ctx.N, int(math.ceil(warmup_time / dt))) if warmup_time > 0 else 0
        if warmup_steps == 0:
            # No hold: fall back to a static warm-start fill so the FM is valid at step 0.
            policy.warm_start(build_state(robot, policy, np.zeros(3), depth_cams))

    # Predicted extra-frame indices that are feet (for the FM viz spheres).
    foot_idx = [i for i, n in enumerate(policy.ctx.target_frame_names)
                if "ankle" in n.lower() or "foot" in n.lower()]
    if viz and not headless:
        print(f"[INFO] FM target frames: {policy.ctx.target_frame_names}; feet -> {foot_idx}")

    # Manual handoff gate: in interactive (joystick) viewer runs, hold the stand until the
    # operator presses the start button, so standing can be verified before going live.
    # ``warmup_time <= 0`` is an explicit request to spawn straight into FM control, so it
    # disables the gate too -- otherwise a joystick run would still wait for a button press
    # and "no warmup" would silently mean "hold forever until pressed".
    gate = ((not headless) and warmup_time > 0
            and isinstance(command_fn, JoystickCommand) and command_fn.joystick is not None)
    min_hold = max(warmup_steps, policy.ctx.N) if gate else warmup_steps

    # Velocity-command rate limit. Prefer the value embedded in the tracker metadata (the rate
    # the policy trained with, so sim2sim can't silently drift from training); otherwise use
    # --cmd_max_acc, which defaults to 0 = disabled.
    meta_max_acc = policy._trk.params.get("max_acc")
    eff_max_acc = meta_max_acc if meta_max_acc is not None else cmd_max_acc
    print(f"[INFO] Command rate limit: {eff_max_acc} "
          f"({'from tracker metadata' if meta_max_acc is not None else 'from --cmd_max_acc'})"
          + ("" if eff_max_acc and eff_max_acc > 0 else " -- DISABLED, raw command passes through"))

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
        # Height scan is a 176-vector per step: too wide for the CSV, so it streams to a
        # sibling .npy (<log>.scan.npy) that the sim2sim comparison loads alongside.
        scan_rows: list[np.ndarray] = []
        # Per-step DEPTH images, the exteroception a depth policy actually consumes. Streamed
        # to a sibling .npy for the same reason the scan is: far too wide for the CSV. Without
        # it a sim2sim divergence cannot be attributed -- the depth cameras are the most
        # likely place for the two simulators to disagree and were the one signal not
        # recorded anywhere.
        depth_rows: list[np.ndarray] = []
        # Per-step FM predicted WINDOW (root + extra frames), so a viewer can replay what the
        # generator actually emitted at each step rather than only the one frame that was
        # applied. Padded with NaN before the generator has a valid prediction.
        pred_rows: list[np.ndarray] = []
        predf_rows: list[np.ndarray] = []
        writer.writerow(["time", "cmd_vx", "cmd_vy", "cmd_wz", "vx_b", "vy_b", "wz_b",
                         "pos_err", "ori_err", "joint_pos_err",
                         "ref_pos_x", "ref_pos_y", "ref_pos_z", "act_pos_x", "act_pos_y", "act_pos_z",
                         "ref_roll", "ref_pitch", "ref_yaw", "act_roll", "act_pitch", "act_yaw",
                         "base_z", "fell",
                         *[f"tau_{n}" for n in isaac],
                         # The COMMANDED joint target the servo tracks, i.e. the policy's
                         # action after scaling and the default-pose offset. Logged because
                         # jpref/jpact only show the reference and the outcome -- neither
                         # says what the policy actually asked for, which is the first thing
                         # to look at when a joint does not follow.
                         *[f"ctrl_{n}" for n in isaac],
                         *jp_cols, *ee_cols])
        # Sidecar metadata for the metrics module (joint order + effort limits + EE frames).
        log_meta = {"joint_names_isaac": isaac, "effort_limits": effort_limits,
                    "ee_frame_names": ee_names, "dt": policy.get_dt(),
                    "fm_update_every": policy.fm_update_every,
                    "zero_noise": bool(getattr(policy, "zero_noise", False)),
                    "cmd_max_acc": float(eff_max_acc or 0.0)}
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
        state = build_state(robot, policy, vel_cmd, depth_cams, want_depth_hits=viz)
        obs = policy.create_obs(state)
        action = policy.get_action(obs, robot.joint_names)
        robot.apply_action(action)
        return action, state

    # ------------------------------------------------------------------
    # Forward kinematics of the PREDICTED window (viewer ribbon + log columns).
    # ------------------------------------------------------------------
    # A scratch MjData on the sim's own model, so FK of a hypothetical pose never disturbs
    # the live one. `mj_kinematics` is the whole cost -- no dynamics, no collision.
    _fk_data = mujoco.MjData(robot.mj_model)
    _fk_body_ids = [mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in policy.ctx.target_frame_names]
    _fk_missing = [n for n, i in zip(policy.ctx.target_frame_names, _fk_body_ids) if i < 0]
    if _fk_missing:
        print(f"[WARN] FK frames absent from the MJCF, so no predicted-frame ribbon: {_fk_missing}")

    def _fm_frame_paths_w(node_idxs) -> np.ndarray | None:
        """World positions of the FM target frames implied by the predicted window.

        The slim window predicts ``joint_pos`` but no ``ee_pos``, so the frames are not
        served -- they are IMPLIED, and forward kinematics recovers them: set the reference
        base pose + reference joints on a scratch ``MjData`` and read the body positions
        back. MuJoCo is the right engine for it here because the sim is already holding the
        very model the FM's FK losses are defined against
        (``FlowMatchingConfig.fk_model_path`` is this MJCF), so nothing can drift between
        the two.

        What this DOES mean: the frames are exactly where the predicted joints put them,
        given the predicted base. That is a stronger statement than the retired ``ee_pos``
        block made -- that one was predicted INDEPENDENTLY and could disagree with FK of the
        same node's joints. Here disagreement is impossible by construction, so the ribbon
        shows the pose the tracker is being asked to achieve, not a second opinion about it.
        Both inputs carry the generator's error, and the joint error reaches the frames
        amplified by the link lengths.

        Takes no pose: the lift goes through the prediction's own stored re-plan anchor, so
        the result is correct at any ``fm_update_every`` and there is no stale-frame trap for
        the caller to fall into.

        Args:
            node_idxs: ABSOLUTE window node indices to evaluate (0..F-1, not offsets from
                the cursor), so the caller pays only for the nodes it draws.

        Returns:
            ``[len(node_idxs), n_frames, 3]`` world positions, or ``None`` when the window
            cannot supply a base pose or joints (or no frame resolved in the MJCF).
        """
        lifted = policy.predicted_base_pose_w()
        jp_isaac = policy.predicted_joint_pos()
        if lifted is None or jp_isaac is None or not any(i >= 0 for i in _fk_body_ids):
            return None
        ref_p, ref_q = lifted
        idxs = np.asarray(node_idxs, dtype=int)
        out = np.full((idxs.size, len(_fk_body_ids), 3), np.nan)
        for row, n in enumerate(idxs):
            # qpos = [pos(3), quat(4, WXYZ), joints(nj in MUJOCO order)].
            _fk_data.qpos[0:3] = ref_p[n]
            _fk_data.qpos[3:7] = xyzw_to_wxyz(ref_q[n])
            _fk_data.qpos[7:] = convert_joint_order(jp_isaac[n], isaac, robot.joint_names)
            mujoco.mj_kinematics(robot.mj_model, _fk_data)
            for f, bid in enumerate(_fk_body_ids):
                if bid >= 0:
                    out[row, f] = _fk_data.xpos[bid]
        return out

    # One-shot latch for the "--log FM branch has NaN columns" notice (a list so the
    # closure can append without a nonlocal).
    _warned_fm_log_nan: list[bool] = []

    def _fm_log_reference() -> dict:
        """A ``log_step`` reference for the FM branch, rebuilt from the predicted window.

        The clip branch hands ``log_step`` a real world-frame reference; the FM branch has
        none to hand it, because the prediction IS the observation -- that is the point of
        the representation. So every column is reconstructed from the predicted window, and
        the shape stays identical so the column contract ``gen_tracker_metrics`` /
        ``compare_sim2sim`` read does not move:

        * ``root_pos`` / ``root_quat`` -- lifted back through the prediction's own re-plan
          anchor, so correct at any ``fm_update_every``;
        * ``jp`` -- the ``joint_pos`` block verbatim (already in isaac order, so it lines
          up elementwise with ``state["joint_pos"]``);
        * ``frame_pos`` -- FK of those joints on that base (see :func:`_fm_frame_paths_w`),
          not a served ``ee_pos`` block, which this window does not carry.

        Reads the node at the CURSOR, not node 0: at ``K > 1`` node 0 is the last re-plan,
        which is up to ``K-1`` steps stale, while the tracker is following
        ``served_node_index``.

        A block the window does drop leaves its columns NaN rather than zero, and says so
        once -- a zero would be silently averaged into the metrics.

        Returns:
            ``{"window": [<served reference>]}``, matching the clip branch's shape.
        """
        nj, k_fr = len(policy.get_joint_names()), len(ee_names)
        n = policy.served_node_index
        lifted = policy.predicted_base_pose_w()
        jp = policy.predicted_joint_pos()
        fp = _fm_frame_paths_w([n])
        cur = {
            "root_pos": lifted[0][n] if lifted is not None else np.full(3, np.nan),
            "root_quat": lifted[1][n] if lifted is not None else np.full(4, np.nan),
            "jp": jp[n] if jp is not None else np.full(nj, np.nan),
            "frame_pos": fp[0] if fp is not None else np.full((k_fr, 3), np.nan),
        }
        # Announce once, but only once a prediction EXISTS: before that every field is
        # legitimately NaN (warmup), and reporting it then would name the whole reference as
        # unavailable when nothing is actually missing.
        if policy._pred_valid and not _warned_fm_log_nan:
            _warned_fm_log_nan.append(True)
            missing = [k for k, v in cur.items() if not np.isfinite(v).all()]
            print(f"[INFO] --log on the FM branch: reference columns rebuilt from the "
                  f"predicted window (blocks {policy.reference_blocks}); the ee*_ columns "
                  f"are FK of the predicted joints, not a served ee_pos block."
                  + (f" NaN (no block supplies them): {missing}." if missing
                     else " Every reference column is populated."))
        return {"window": [cur]}

    def log_step(state: dict, ref: dict | None = None) -> None:
        if not writer:
            return
        ref = ref if ref is not None else _fm_log_reference()
        cur = ref["window"][0]          # offset 0 = the current step
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
        ctrls = [float(robot.mj_data.ctrl[a]) for a in act_ids]
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
        if "height_scan" in state:
            scan_rows.append(np.asarray(state["height_scan"], dtype=np.float32).ravel())
        if "depth" in state:
            # (num_cameras, H*W) NORMALISED exactly as the policy sees it, so a diff against
            # the Isaac side compares the observation rather than raw metres.
            depth_rows.append(np.asarray(state["depth"], dtype=np.float32))
        f_len, k_fr = int(policy.ctx.F), len(ee_names)
        # Per-step predicted WINDOW sidecars: the same lift + FK draw_viz uses, over EVERY
        # node (the viewer may stride, a replay should not).
        pred_path = policy.predicted_root_path_w()
        pred_rows.append(np.asarray(pred_path, dtype=np.float32) if pred_path is not None
                         else np.full((f_len, 3), np.nan, dtype=np.float32))
        pred_frames = _fm_frame_paths_w(np.arange(f_len))
        predf_rows.append(np.asarray(pred_frames, dtype=np.float32) if pred_frames is not None
                          else np.full((f_len, k_fr, 3), np.nan, dtype=np.float32))
        writer.writerow([robot.mj_data.time, *state["command"].tolist(),
                         float(lin_b[0]), float(lin_b[1]), float(state["base_ang_vel"][2]),
                         pos_err, ori_err, jp_err,
                         float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2]),
                         float(root_pos[0]), float(root_pos[1]), float(root_pos[2]),
                         float(ref_rpy[0]), float(ref_rpy[1]), float(ref_rpy[2]),
                         float(act_rpy[0]), float(act_rpy[1]), float(act_rpy[2]),
                         float(root_pos[2]), int(robot.failed()),
                         *torques, *ctrls, *jp_vals, *ee_vals])

    def draw_viz(viewer, state: dict) -> None:
        """Populate ``user_scn`` with the sensor hits + FM root/feet spheres + velocity arrows."""
        if viewer is None or not viz:
            return
        scn = viewer.user_scn
        scn.ngeom = 0
        base_pos = np.asarray(state["root_pos"], dtype=np.float64)
        # Height-scan ray hits (magenta dots on the terrain under/ahead of the robot).
        hits = state.get("height_scan_hits")
        if hits is not None:
            for p in np.asarray(hits, dtype=np.float64):
                if np.isfinite(p).all():
                    _add_sphere(scn, p, (1.0, 0.0, 1.0, 0.9), 0.015)
        # Depth-camera hits, one colour per camera (near yellow, far cyan-green). Drawn
        # every Nth pixel: 1560 hits per step would blow past user_scn.maxgeom and stall
        # the viewer, and a subsampled cloud still shows where each camera is looking.
        depth_hits = state.get("depth_hits")
        if depth_hits is not None:
            colours = ((1.0, 0.85, 0.1, 0.8), (0.1, 0.9, 0.6, 0.8))
            for cam_i, cam_hits in enumerate(np.asarray(depth_hits, dtype=np.float64)):
                rgba = colours[cam_i % len(colours)]
                sub = cam_hits[::max(1, depth_viz_stride)]
                for p in sub[np.isfinite(sub).all(axis=1)]:
                    _add_sphere(scn, p, rgba, 0.012)
        # FM prediction spheres (only once the generator has a valid prediction).
        if policy._pred_valid:
            if fm_traj_points <= 0:
                idxs = np.arange(policy.ctx.F)  # every predicted node -- dense ribbon (like IsaacLab)
            else:
                n_pts = max(1, min(fm_traj_points, policy.ctx.F))
                idxs = np.unique(np.linspace(0, policy.ctx.F - 1, n_pts).astype(int))
            # The prediction is a reference WINDOW in the anchor's frame, not a world path:
            # lift it back through the anchor to draw it. Exact at fm_update_every=1 (the
            # only cadence this runtime serves), where the anchor is the robot in `state`.
            rp = policy.predicted_root_path_w()
            if rp is not None:                                   # (F, 3) predicted root path
                for i in idxs:
                    _add_sphere(scn, rp[int(i)], (1.0, 0.55, 0.0, 0.85), 0.02)
            # Feet: the slim window predicts joints but no ee_pos, so the feet are IMPLIED
            # rather than served -- FK of the predicted joints on the predicted base
            # recovers them (see _fm_frame_paths_w).
            if foot_idx:
                fp = _fm_frame_paths_w(idxs)
                if fp is not None:
                    for row in range(fp.shape[0]):
                        for fi in foot_idx:
                            p = fp[row, fi]
                            if np.isfinite(p).all():
                                _add_sphere(scn, p, (0.0, 0.9, 1.0, 0.95), 0.02)
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

    report_every_s = 5.0
    _next_report_t = [report_every_s]  # next SIM time (s) at which to report (mutable cell)

    def maybe_report_timing() -> None:
        """Print the mean FM + tracker inference time every ``report_every_s`` sim seconds.

        Cadence is on sim time (``robot.mj_data.time``) so it fires in both the paced
        viewer (~every 5 real s) and unpaced headless/eval runs. The reported values are
        real per-call wall-clock compute times (ms), averaged over the window.
        """
        if not hasattr(policy, "pop_avg_infer_ms"):
            return
        t = robot.mj_data.time
        if t < _next_report_t[0]:
            return
        s = policy.pop_avg_infer_ms()
        msg = f"[timing t={t:.1f}s] tracker {s['trk_ms']:.2f} ms/step (n={s['trk_n']})"
        if s["fm_n"]:
            msg += (f" | FM {s['fm_ms']:.2f} ms/infer (n={s['fm_n']}, "
                    f"every {policy.fm_update_every} steps)")
        print(msg)
        _next_report_t[0] += report_every_s

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
            state = build_state(robot, policy, np.zeros(3), depth_cams)
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
            maybe_report_timing()

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
            state = build_state(robot, policy, clip.command_at(s), depth_cams,
                                want_depth_hits=viz)
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
                ref = clip.reference_at(s, policy.window_offsets)
                obs = policy._build_obs(state, reference=ref)
            action = policy.get_action(obs, robot.joint_names)
            robot.apply_action(action)
            if robot.failed():
                return False
            draw_viz(viewer, state)
            advance_physics(state, viewer, ref)
            maybe_report_timing()
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
            _apply_viz_options(viewer, robot.mj_model, viz_geomgroup, viz_contacts)
            success = loop_fn(viewer)

    if log_file:
        log_file.close()
        print(f"[INFO] Wrote metrics log to {log_path}")
        if scan_rows:
            scan_path = f"{log_path}.scan.npy"
            np.save(scan_path, np.stack(scan_rows))
            print(f"[INFO] Wrote {len(scan_rows)} height-scan rows to {scan_path}")
        if depth_rows:
            depth_path = f"{log_path}.depth.npy"
            np.save(depth_path, np.stack(depth_rows))
            print(f"[INFO] Wrote {len(depth_rows)} depth frames "
                  f"{np.stack(depth_rows).shape} to {depth_path}")
        if pred_rows:
            np.save(f"{log_path}.pred.npy", np.stack(pred_rows))
            np.save(f"{log_path}.predf.npy", np.stack(predf_rows))
            print(f"[INFO] Wrote {len(pred_rows)} FM predicted windows (root + "
                  f"{len(ee_names)} frames) to {log_path}.pred.npy / .predf.npy")
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
    parser.add_argument("--fm_update_every", type=int, default=1,
                        help="Generator re-inference cadence K (control steps); 1 = every step, "
                             "which is the only cadence that pays the full FM cost every tick. "
                             "Fallback only: overridden by 'fm_update_every' in the tracker "
                             "metadata if present, so pin K at EXPORT to choose it. K>1 serves "
                             "each plan for K steps, re-anchoring the served nodes onto the "
                             "measured base -- exact in MuJoCo, but on hardware it needs the "
                             "base pose to be good over K-1 steps of egomotion. Needs "
                             "F >= max_window_offset + K (F=62, offset 50 => K <= 12).")
    # FM sampler noise. Zero noise is the sim2sim DEFAULT because it visibly tracks better in
    # MuJoCo than drawing x0 ~ N(0, I); the deploy library default stays Gaussian, so hardware
    # keeps the training-matched stochastic sampling unless it opts in too.
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--zero_noise", dest="zero_noise", action="store_true", default=True,
                             help="Integrate the flow from x0 = 0 instead of x0 ~ N(0, I): one "
                                  "fixed, fully deterministic FM trajectory per conditioning. "
                                  "DEFAULT in this runner (chosen empirically -- it looks better "
                                  "in MuJoCo); pass --gaussian_noise for the trained behaviour.")
    noise_group.add_argument("--gaussian_noise", dest="zero_noise", action="store_false",
                             help="Draw the sampler's x0 ~ N(0, I) -- the stochastic sampling the "
                                  "FM was trained for and what transfer/deploy uses on hardware. "
                                  "Turns off the zero-noise default.")
    parser.add_argument("--total_time", type=float, default=-1.0)
    parser.add_argument("--spawn_on_box", type=str, default=None, metavar="NAME_OR_INDEX",
                        help="Spawn standing on top of a scene box instead of the floor, e.g. "
                             "'--spawn_on_box 2' or '--spawn_on_box box2' (the generated "
                             "terrain scene has box0..box3 at 0.25/0.40/0.55/0.70 m). The "
                             "standing joint pose is kept; only the support height changes. "
                             "Pair with --warmup_time 0 to hand straight to the FM on the box.")
    parser.add_argument("--warmup_time", type=float, default=0.0,
                        help="Seconds to PD-hold a stand before live control. DEFAULT 0 = spawn "
                             "DIRECTLY into FM+tracker control: the FM history is warm-started "
                             "from the spawn state, the leg hold-gain stiffening is skipped, and "
                             "the joystick start-button gate is disabled. The FM is valid at step "
                             "0 regardless, because --null_history_warmup (also default on) "
                             "samples it through the learned null-history embedding. Pass a "
                             "POSITIVE value (e.g. 1.0, the old default) to get the stand hold "
                             "back -- and, in a joystick viewer run, the start-button gate that "
                             "holds until you press start. Ignored in --clip mode, which always "
                             "spawns on the clip pose with RL gains.")
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
    parser.add_argument("--clip_spawn_z_offset", type=float, default=0.0,
                        help="Pelvis lift in clip mode (default 0 = spawn exactly on the clip "
                             "frame, matching IsaacLab). Non-zero desynchronises the two sims' "
                             "root height AND their whole height scan.")
    parser.add_argument("--spawn_z_offset", type=float, default=0.03,
                        help="Metres to raise the spawn pelvis above the standing-config height "
                             "(optimizer frame sits ~3 cm below the MuJoCo foot-contact height).")
    parser.add_argument("--cmd_max_acc", type=float, default=0.0,
                        help="Velocity-command rate limit (m/s^2, rad/s^2) on the live command. "
                             "DEFAULT 0 = DISABLED: the raw joystick/profile command passes "
                             "straight through (chosen empirically -- it looks better in MuJoCo). "
                             "Pass 2.0 to match the clip library's training ramp rate. Overridden "
                             "by tracker metadata 'max_acc' when that key is present.")
    nh = parser.add_mutually_exclusive_group()
    nh.add_argument("--null_history_warmup", dest="null_history_warmup", action="store_true",
                    default=True,
                    help="Sample the FM from step 0 using its learned null-history embedding "
                         "while the N-step window fills (default; the RL side does the "
                         "same unconditionally). Needs an fm.onnx exported with the "
                         "history_null input.")
    nh.add_argument("--no_null_history_warmup", dest="null_history_warmup",
                    action="store_false",
                    help="Legacy: withhold the FM until the history window is full, fabricating "
                         "a repeated static history at spawn.")
    parser.add_argument("--fm_traj_points", type=int, default=0,
                        help="Number of spheres drawn along the FM predicted root/feet trajectory; "
                             "<=0 draws every predicted node (dense ribbon, like IsaacLab).")
    parser.add_argument("--depth_noise", action="store_true", default=False,
                        help="Corrupt the depth images with the TRAINING noise (plane bias, "
                             "per-pixel noise, blur, dropout) and jitter the camera mounts. Off by "
                             "default, matching the -play task the policy was exported from.")
    parser.add_argument("--depth_noise_seed", type=int, default=0,
                        help="Seed for --depth_noise.")
    parser.add_argument("--viz_geomgroup", type=str, default=None, metavar="LIST",
                        help="Comma-separated geom GROUPS to render: 0 = collision geoms, "
                             "1 = visual meshes, 2 = terrain. MuJoCo's default renders 0, 1 "
                             "and 2 TOGETHER, which is why a collider and the visual mesh it "
                             "duplicates are indistinguishable, and why disabling a geom's "
                             "contact changes nothing on screen. Use '0,2' to see only what "
                             "can actually collide, or '1,2' for only what the depth cameras "
                             "ray-cast.")
    parser.add_argument("--viz_contacts", action="store_true", default=False,
                        help="Show contact points + forces and make geoms transparent. An "
                             "empty view means nothing is touching -- which for a "
                             "self-collision question is the answer, not a failure.")
    parser.add_argument("--depth_viz_stride", type=int, default=8,
                        help="Draw every Nth depth pixel's hit point in the viewer (1560 hits per "
                             "step would exceed the scene's geom capacity).")
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Where --log writes sim_log.csv + metrics (default: "
                             "<run_dir>/mujoco_gen_tracker_logs). That default is ONE fixed "
                             "path per run dir, so concurrent runs clobber each other; give "
                             "each a separate --log_dir to sweep clips in parallel.")
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
    parser.add_argument("--clip_id", type=int, default=0,
                        help="In --clip mode, which clip of the library to play (default 0). Must "
                             "match the Isaac side for a like-for-like sim2sim comparison.")
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
    parser.add_argument("--keep_xml_gains", action="store_true", default=False,
                        help="Keep the MJCF's hand-written actuator kp/kv instead of the gains "
                             "the policy exported. Default OFF: the trained gains win, so a "
                             "retune in training reaches sim2sim without editing the XML.")
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
    # A NEGATIVE handoff time also means clip-only: run_clip_loop documents "<0 never hands
    # off", so requiring an fm.onnx for it demanded a generator that is provably never used.
    clip_only = bool(args.clip) and (args.clip_handoff_time is None or args.clip_handoff_time < 0)
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
        zero_noise=args.zero_noise,
        null_history_warmup=args.null_history_warmup,
    )
    policy.load()

    robot = Robot(robot_name=args.robot_name, scene_name=args.scene,
                  joystick_scaling=np.ones(3), input_function=lambda t: np.zeros(3))
    # Drive the MuJoCo servos with the gains the policy actually TRAINED with (exported in
    # policy_parameters.yaml) rather than the MJCF's hand-written ones, so retuning gains in
    # training propagates to sim2sim with no XML edit. Skip with --keep_xml_gains.
    if not args.keep_xml_gains:
        robot.apply_policy_gains(policy.get_kp(), policy.get_kd(), policy.get_joint_names())
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
        # Default is one FIXED path per run dir, so two concurrent runs would overwrite each
        # other's CSV; --log_dir is what lets a sweep fan out over clips in parallel.
        log_dir = args.log_dir or os.path.join(run_dir, "mujoco_gen_tracker_logs")
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
                             clip_id=args.clip_id,
                             start_time=args.clip_start_time,
                             ego_anchor=args.clip_ego_anchor,
                             ego_anchor_every=args.clip_ego_anchor_every,
                             root_frame_bodies=policy._trk.get_root_frame_bodies())
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

    # Depth cameras, if the loaded artifacts use them. The contract comes from the ONNX
    # metadata (checked for tracker/generator agreement inside policy.load()), so the
    # cameras cast here are exactly the ones the networks were trained on.
    depth_cams = None
    depth_spec = policy.get_depth_camera_spec()
    if depth_spec:
        depth_cams = MujocoDepthCameras(robot.mj_model, depth_spec)
        if args.depth_noise:
            rng = np.random.default_rng(args.depth_noise_seed)
            depth_cams.enable_noise(rng)
            depth_cams.jitter_extrinsics(rng)
            print("[INFO] Depth sensor noise ON (plane bias, per-pixel noise, blur, dropout) "
                  "plus mount jitter -- the training distribution, not the play config.")

    success = run(policy, robot, command_fn, total_time, args.headless, log_path,
                  args.warmup_time, args.rt_factor, standing_config, args.viz,
                  args.vel_arrow_scale, args.vel_arrow_height, args.hold_kp_scale,
                  args.hold_kv_scale, args.spawn_z_offset, args.clip_spawn_z_offset,
                  args.cmd_max_acc, args.fm_traj_points,
                  viz_geomgroup=([int(g) for g in args.viz_geomgroup.split(",") if g.strip() != ""]
                                 if args.viz_geomgroup else None),
                  viz_contacts=args.viz_contacts,
                  spawn_on_box=args.spawn_on_box,
                  clip=clip, clip_handoff_time=clip_handoff_time,
                  depth_cams=depth_cams, depth_viz_stride=args.depth_viz_stride)
    print(f"[INFO] Run finished. Survived: {success}")

    if log_path:
        import gen_tracker_metrics
        gen_tracker_metrics.create_metrics(os.path.dirname(log_path))

    # Hard-exit to skip the native GL teardown. At interpreter exit the MuJoCo viewer's
    # GLFW context and pygame/SDL (joystick) race destroying their GLX contexts on the X
    # server -- this prints "GLXBadContext" and, worse, can DEADLOCK the main thread in
    # that C teardown, leaving the window "not responding" and unresponsive to Ctrl-C
    # (the SIGINT handler never runs because we're stuck in native code). Everything the
    # run produces (metrics CSV, plots, stdout) is already flushed/written by here, so
    # terminating at the OS level is safe and avoids the hang. Only for viewer runs --
    # headless/eval has no GL context to tear down and returns normally.
    if not args.headless:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
