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
import signal
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

# Torch-free deployment runtime (numpy + onnxruntime); transfer/ on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.artifacts import resolve_fm_onnx  # noqa: E402
from deploy.fm_session import default_depth_lag_steps  # noqa: E402
from deploy.clip_reference import ClipReference, resolve_case  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy, wxyz_to_xyzw, xyzw_to_wxyz  # noqa: E402
from deploy.quat_math_np import (  # noqa: E402
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_box_minus,
    quat_inv,
    subtract_frame_transforms,
    yaw_quat,
)
from deploy.fk import DEFAULT_FK_MODEL, WindowFK  # noqa: E402
from deploy.diagnostics import obs_checksum  # noqa: E402
from deploy.replay_stream import split_checksum  # noqa: E402
from deploy.tracker_session import convert_joint_order  # noqa: E402
from deploy.window_np import window_block_slices  # noqa: E402

from robot import Robot  # noqa: E402
from depth_camera import MujocoDepthCameras  # noqa: E402
import experiments.velocity_commands as vc  # noqa: E402


def find_latest_run(log_root_path: str) -> str | None:
    """Return the most recently modified run directory name under ``log_root_path``."""
    run_dirs = glob.glob(os.path.join(log_root_path, "*"))
    return os.path.basename(max(run_dirs, key=os.path.getmtime)) if run_dirs else None


class JoystickCommand:
    """Read ``(vx, vy, wz)`` from a gamepad, scaled by per-axis maxima."""

    def __init__(self, max_vx: float, max_vy: float, max_wz: float):
        """Initialise pygame and open joystick 0 (falls back to a constant command)."""
        import pygame

        self._pygame = pygame            # __call__ pumps the event queue through it
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


class FmDepthLag:
    """Hands the GENERATOR a depth frame `steps` control steps old, with its own pose.

    The generator trains on frames delayed 20-60 ms (the depth pipeline's compute plus the
    ROS hop), so feeding it a fresh one in simulation is off-distribution in exactly the
    variable that model was fitted to. Hardware needs none of this -- its lag is physical and
    already in the frames -- which is why this lives in the sim runner and not in
    ``transfer/deploy``.

    The POSE travels with the image. The 5-channel row's xyz planes are unprojected from the
    camera pose, so a lagged image unprojected with the CURRENT pose is displaced by the
    egomotion -- precisely the registration error those channels exist to remove. Buffering
    the pair keeps them consistent, matching what ``dataset._select_depth_lag`` does offline.

    Only the generator is delayed. The tracker's own ``state["depth"]`` observation stays
    live: it was trained on fresh frames, and delaying it is a separate question.
    """

    def __init__(self, steps: int, why: str = "") -> None:
        """Set up the ring buffer.

        Args:
            steps: Lag in control steps (20 ms each). ``<= 0`` disables it entirely, leaving
                the state dict byte-identical to what it was before this existed.
            why: Where the value came from, for the one-time announcement.
        """
        self.steps = max(int(steps), 0)
        self._buf: list[tuple] = []
        if self.steps:
            print(f"[INFO] FM camera lag: {self.steps} step(s) = {20 * self.steps} ms "
                  f"({why}). The generator sees a frame that old, with the camera pose it "
                  "was captured at; the tracker's own depth obs stays live.")
        else:
            print(f"[INFO] FM camera lag: OFF ({why}). The generator sees the live frame, "
                  "which is off-distribution if it trained with latency.")

    def apply(self, state: dict) -> dict:
        """Replace the generator's depth + pose in ``state`` with the lagged pair.

        Args:
            state: The state dict just built for this step; mutated in place.

        Returns:
            The same dict, for chaining.
        """
        if not self.steps:
            return state
        # `fm_depth` when the generator samples its own grid, else a COPY of the tracker's --
        # `_fm_depth_reading` treats `fm_depth` as authoritative whenever it is present, so
        # this reaches the generator without touching what the tracker observes.
        img = state.get("fm_depth")
        if img is None:
            img = state.get("depth")
        if img is None:
            return state
        self._buf.append((np.array(img, copy=True), state.get("fm_camera_pose")))
        del self._buf[:-(self.steps + 1)]
        # Early in an episode the buffer is short: repeat the OLDEST frame held rather than
        # zero-filling. A zero depth row is a valid reading -- a surface at the near plane --
        # so a zero-filled warm-up would condition the generator on a wall that is not there.
        img_lag, pose_lag = self._buf[max(len(self._buf) - 1 - self.steps, 0)]
        state["fm_depth"] = img_lag
        if pose_lag is not None:
            state["fm_camera_pose"] = pose_lag
        return state


def build_state(
    robot: Robot, policy: GenTrackerPolicy, command: np.ndarray,
    depth_cams: MujocoDepthCameras | None = None, want_depth_hits: bool = False,
    want_raw_depth: bool = False,
    fm_depth_cams: MujocoDepthCameras | None = None,
    fm_lag: "FmDepthLag | None" = None,
) -> dict:
    """Assemble the world-state dict ``GenTrackerPolicy.create_obs`` expects.

    Reads only base pose + base world-frame velocity + joints + command (and, when
    the loaded generator's history needs them, the extra-frame world states), plus
    whichever exteroception the loaded artifacts actually use.

    Args:
        robot: The MuJoCo robot wrapper.
        policy: The loaded generator + tracker policy (for joint order + ctx).
        command: Velocity command ``(vx, vy, wz)``.
        depth_cams: The TRACKER's depth cameras to cast, or ``None`` when the policy has
            none.
        want_depth_hits: Also collect the depth rays' world hit points (viewer only).
        want_raw_depth: Also collect pre-normalisation depth in metres (replay dump only).
        fm_depth_cams: The GENERATOR's cameras, when it samples the same lenses on its own
            grid. ``None`` when the two samplings match, in which case the generator reads
            ``state["depth"]`` like it always did.

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
    # Read the context DIRECTLY, not through getattr-with-a-default. Both context types
    # (FmContext and _TrackerOnlyCtx) always define these, so the default was never doing
    # any work -- but it would have silently absorbed the extero rename: a default of 1
    # makes this False, and the runner would just stop computing state["height_scan"] for a
    # height-scan FM. An AttributeError is the outcome to want here.
    fm_uses_scan = (
        policy.ctx.extero_source == "height_scan" and policy.ctx.extero_sensor_dim > 1
    )
    if spec is not None or fm_uses_scan:
        # PREFER the generator's own spec, then the tracker's, and REFUSE rather than default.
        # The old `s.get(..., 0.1 / (1.5, 1.0))` fallback is the POLICY's historical grid and
        # cannot be assumed for a generator: a 14641-ray FM would be handed 176 values and die
        # inside a Normalizer broadcast -- or, at a coincidental width, be fed a completely
        # different patch of ground at exactly the right shape.
        fm_spec = next(iter(getattr(policy.ctx, "extero_spec", None) or []), None)
        s = fm_spec if (fm_spec and "resolution" in fm_spec) else spec
        if s is None:
            raise ValueError(
                "this generator conditions on a height scan but neither it nor the tracker "
                "states the grid geometry (resolution/size/offset). There is no safe default: "
                "the historical 0.1 m / 1.5x1.0 m grid is the POLICY's, and casting it for a "
                "generator trained on another one silently substitutes the wrong signal. "
                "Re-export the artifacts with a height_scan_spec."
            )
        if bool(s.get("privileged")) and not policy.allow_privileged_extero:
            raise ValueError(
                f"this generator conditions on the PRIVILEGED dense scan "
                f"({s.get('size')} m @ {s.get('resolution')} m = {s.get('num_rays')} rays), "
                "which is exact noise-free ground truth and not reproducible on a ROBOT. "
                "MuJoCo can reproduce it -- it casts the same grid against the same terrain "
                "-- so pass --allow_privileged_extero to run the pair here. Nothing measured "
                "under it says anything about hardware."
            )
        off = s.get("offset_pos", [0.75, 0.0, 20.0])
        clr, hits = robot.compute_height_scan(
            state["root_pos"], state["root_quat"],
            resolution=float(s["resolution"]), size=tuple(s["size"]),
            offset_fwd=float(off[0]), offset_lat=float(off[1]), ray_start_h=float(off[2]),
            sensor_offset=float(s.get("sensor_offset", 0.5)),
        )
        if fm_uses_scan and clr.shape[-1] != int(policy.ctx.extero_sensor_dim):
            raise ValueError(
                f"cast {clr.shape[-1]} rays but the generator conditions on "
                f"{int(policy.ctx.extero_sensor_dim)}. The grid geometry and the checkpoint "
                "disagree; a Normalizer broadcast would be the next thing to fail, further "
                "from the cause."
            )
        # Missed rays must match IsaacLab's encoding, NOT 0.0. Isaac's miss is ``pos_z - inf -
        # offset`` = -inf, and ``sanitize_height_scan`` maps a non-finite reading to the band
        # edge; that is what both the teacher obs and the FM token were trained on. Mapping a
        # miss to 0.0 instead reads as "ground exactly at the nominal sensor plane" -- plausible
        # flat ground where Isaac reports the band edge. Flat terrain never misses, so this was
        # invisible until isolated (has_floor=False) stair/box tiles, where rays overshoot the
        # tile edge and ~2-3% of them miss right at the stair transition.
        #
        # Leave the miss as -inf rather than substituting here: each consumer then applies its
        # OWN sanitize with its own band (the FM's comes from the checkpoint), exactly as in
        # Isaac. Pre-substituting a number hard-codes one consumer's band for all of them --
        # which was silently wrong the moment the FM's band became configurable.
        state["height_scan"] = np.where(np.isnan(clr), -np.inf, clr)
        state["height_scan_hits"] = hits
    # The generator's camera POSES, when it conditions on more than depth. Same construction
    # as the Isaac term: the MOUNT BODY's live pose composed with the contract's NOMINAL
    # offset (never the jittered one -- a robot knows its calibration and nothing more),
    # expressed in the root's yaw-aligned frame. The heading cancels out of that frame, which
    # is what lets this run with no odometry.
    if int(getattr(policy.ctx, "extero_channels", 1)) > 1:
        fm_spec = policy.get_fm_depth_camera_spec() or policy.get_depth_camera_spec() or []
        poses = []
        for cam in fm_spec:
            mount = robot.get_body_world_states([str(cam["mount_body"])])[0]
            cam_p_w, cam_q_w = combine_frame_transforms(
                mount[0:3], mount[3:7],
                np.asarray(cam["offset_pos"], dtype=np.float64),
                np.asarray(cam["offset_quat_xyzw"], dtype=np.float64),
            )
            p_rel, q_rel = subtract_frame_transforms(
                state["root_pos"], yaw_quat(state["root_quat"]), cam_p_w, cam_q_w
            )
            poses.append((p_rel, matrix_from_quat(q_rel)))
        state["fm_camera_pose"] = poses

    # Depth cameras: cast the policy's own pinhole grids against the terrain AND the
    # robot's visual meshes, normalised exactly as the observation term does. One cast
    # feeds BOTH consumers -- the tracker's depth obs terms (keyed by term name) and the
    # FM's exteroception tokens (stacked, matching mdp.depth_obs_stack) -- because there
    # is one physical camera pair -- the policy keys the rows by obs term itself, from the
    # contract it loaded. Both cameras run at the control rate, so a single cast per step
    # is exact: no staleness to model.
    if depth_cams is not None:
        # `depth_raw` is METRES along the optical axis, recorded only for --dump_replay: the
        # C++ runtime is fed raw and normalises internally, and the normalisation clips and
        # maps misses to a sentinel, so it cannot be inverted after the fact.
        if want_depth_hits and want_raw_depth:
            state["depth"], state["depth_hits"], state["depth_raw"] = depth_cams.compute(
                robot.mj_data, want_hits=True, want_raw=True)
        elif want_depth_hits:
            state["depth"], state["depth_hits"] = depth_cams.compute(robot.mj_data, want_hits=True)
        elif want_raw_depth:
            state["depth"], state["depth_raw"] = depth_cams.compute(robot.mj_data, want_raw=True)
        else:
            state["depth"] = depth_cams.compute(robot.mj_data)
    # A SECOND cast, only when the generator samples the same lenses on a different grid.
    # There is nothing to share: the arrays have different widths, and resizing the
    # tracker's would either invent detail (up) or throw away exactly the detail the
    # generator was trained to use (down). The viewer's hit cloud stays the tracker's.
    if fm_depth_cams is not None:
        state["fm_depth"] = fm_depth_cams.compute(robot.mj_data)
    # LAST, so it sees the finished pair and can substitute both. A no-op at lag 0.
    if fm_lag is not None:
        fm_lag.apply(state)
    return state


def _record_replay_tick(writer, policy, state: dict, obs: np.ndarray,
                        prev_action: np.ndarray) -> None:
    """Append one tick's inputs and this runtime's answers to the replay stream.

    Called AFTER get_action, so `obs`, the action and the served window are this tick's, while
    `prev_action` was captured before the call -- the ordering matters and is the reason this
    is a function rather than an inline block.

    The camera pose is flattened per camera in CONTRACT order, matching the C++
    HardwareReadings layout, so the replay tool's mapping is one line per field.

    Args:
        writer: The :class:`deploy.replay_stream.ReplayWriter`.
        policy: The live policy, for the served window and the raw action.
        state: The state dict this tick was built from.
        obs: The assembled observation.
        prev_action: The previous tick's raw network output.
    """
    poses = state.get("fm_camera_pose") or []
    cam_pos = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p, _ in poses]) \
        if poses else np.zeros(0)
    cam_rot = np.concatenate([np.asarray(r, dtype=np.float64).ravel() for _, r in poses]) \
        if poses else np.zeros(0)
    raw = state.get("depth_raw")
    writer.append(
        inputs={
            "root_pos": state["root_pos"],
            "base_quat_xyzw": state["root_quat"],
            "base_ang_vel_b": state["base_ang_vel"],
            "base_lin_vel_w": state["root_lin_vel_w"],
            "joint_pos": state["joint_pos"],
            "joint_vel": state["joint_vel"],
            "command": state["command"],
            "depth_raw": None if raw is None else np.asarray(raw, dtype=np.float64).ravel(),
            "fm_camera_pos": cam_pos,
            "fm_camera_rot": cam_rot,
            "height_scan": state.get("height_scan"),
            "prev_action": prev_action,
        },
        outputs={
            "obs": np.asarray(obs, dtype=np.float64).ravel(),
            "action": np.asarray(policy._trk.action_isaac, dtype=np.float64).ravel(),
            "window": policy.last_served_window(),
            # Split, because a 64-bit digest does not fit a float64 exactly -- see
            # replay_stream.split_checksum.
            "obs_checksum_hi": split_checksum(
                obs_checksum(np.asarray(obs, dtype=np.float32)))[0],
            "obs_checksum_lo": split_checksum(
                obs_checksum(np.asarray(obs, dtype=np.float32)))[1],
            "ctrl_step": float(policy._ctrl_step),
            # 1.0 on a tick the generator re-inferred. The comparator groups by this: a
            # divergence confined to re-plan ticks points at the generator, one spread over
            # every tick points at the observation assembly.
            "replanned": float(policy._ctrl_step % max(policy.fm_update_every, 1) == 0),
            "fm_history_null": (float(bool(plan["history_null"]))
                                if (plan := policy.last_plan()) is not None else -1.0),
            # The conditioning the LIVE plan came from. Held between re-plans, exactly as the
            # C++'s last_conditioning() is, so the two line up tick for tick.
            "fm_extero": (np.zeros(0) if plan is None
                          else np.asarray(plan["extero_tokens"], dtype=np.float64).ravel()),
            "fm_future_cmd": (np.zeros(0) if plan is None
                              else np.asarray(plan["future_cmd"], dtype=np.float64).ravel()),
            "fm_history_nodes": (np.zeros(0) if plan is None
                                 else np.asarray(plan["history_nodes"],
                                                 dtype=np.float64).ravel()),
            "fm_history_cmd": (np.zeros(0) if plan is None
                               else np.asarray(plan["history_cmd"], dtype=np.float64).ravel()),
        },
    )


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


class InterruptFlag:
    """Make Ctrl-C end the run at the next control step instead of unwinding through it.

    A bare ``KeyboardInterrupt`` fires wherever the interpreter happens to be -- typically
    deep inside the observation build -- and propagates out through the viewer's context
    manager. Two things go wrong there:

    * the GL drawable is destroyed while the viewer thread still has a buffer swap in
      flight, which the X server reports as ``GLXBadDrawable`` and Xlib treats as **fatal**,
      so an interrupted run ends in what looks like a crash;
    * **every artifact the run was going to write is lost** -- the metrics CSV, the scan
      rows, the FM prediction windows and the plots are all written *after* the loop, and
      the exception skips the lot.

    Setting a flag instead lets the loop return normally, so an interrupted run finishes the
    way a ``--total_time`` run does: the viewer is closed by its own context manager, the
    artifacts are written, and the exit goes through the same ``os._exit(0)`` that already
    exists to dodge the native GL teardown race. A SECOND Ctrl-C restores the default
    handler, so a genuinely wedged run can still be killed.
    """

    def __init__(self) -> None:
        self.hit = False
        self._previous = None

    def __enter__(self) -> "InterruptFlag":
        """Install the handler, remembering whatever was there before."""
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *_exc) -> bool:
        """Restore the previous handler."""
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
        return False

    def _handle(self, _signum, _frame) -> None:
        """Flag the interrupt; a second one goes back to the default (immediate) behaviour."""
        if self.hit:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n[WARN] second interrupt: the next one exits immediately, with no "
                  "artifacts written.")
            return
        self.hit = True
        print("\n[INFO] interrupt: stopping at the next control step and writing artifacts "
              "(Ctrl-C again to skip that).")


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
        headless: bool, log_path: str | None, rt_factor: float,
        standing_config: str | None, viz: bool, vel_arrow_scale: float, vel_arrow_height: float,
        spawn_z_offset: float,
        clip_spawn_z_offset: float,
        cmd_max_acc: float, fm_traj_points: int, spawn_on_box: str | None = None,
        clip: ClipReference | None = None, clip_handoff_time: float = -1.0,
        depth_cams: MujocoDepthCameras | None = None, depth_viz_stride: int = 8,
        viz_geomgroup: list[int] | None = None, viz_contacts: bool = False,
        fm_depth_cams: MujocoDepthCameras | None = None,
        replay=None, fm_lag: FmDepthLag | None = None) -> bool | None:
    """Run the control loop; return True if the robot did not fall.

    Returns ``None`` when the user interrupts with Ctrl-C, which is neither survival nor a
    fall. The interrupt is taken at a control-step boundary so the artifacts below still get
    written -- see :class:`InterruptFlag` for why that matters and what it replaced.

    The FM + tracker have control from the first step -- there is no warmup hold, and no
    way to fabricate a generator history. Playback is paced to real time (scaled by
    ``rt_factor``) in viewer mode; headless/eval runs are never throttled.

    Args:
        policy: Loaded generator + tracker policy.
        robot: MuJoCo robot wrapper (gains already set).
        command_fn: ``sim_time -> (vx, vy, wz)`` command source.
        total_time: Stop after this many seconds (``<= 0`` runs until the viewer closes).
        headless: Run without the viewer.
        log_path: CSV path for per-step metrics, or ``None`` to skip logging.
        rt_factor: Real-time speed multiple for viewer playback (``1.0`` = real time,
            ``0.5`` = half speed; ``<= 0`` disables pacing).
        standing_config: Path to a standing ``config_solution.yaml`` to spawn + hold; if
            missing/None the robot holds the tracker's ``default_joint_angles`` instead.
        viz: Draw FM root/feet spheres + velocity arrows in the viewer (viewer mode only).
        vel_arrow_scale: Arrow length in metres per (m/s) of planar speed.
        vel_arrow_height: Height (m) above the pelvis to draw the velocity arrows.
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
        depth_cams: The TRACKER's depth cameras to cast each control step, or ``None`` when
            the loaded policy has none. Feeds the tracker's depth obs terms, and the FM's
            exteroception tokens too when both sample the same grid.
        fm_depth_cams: The GENERATOR's cameras, cast separately when it samples the same
            lenses on its own grid (``policy.depth_sampling_is_split()``).
        depth_viz_stride: Draw every Nth depth pixel's hit point in the viewer. The two
            cameras produce 1560 hits per step, well past ``user_scn.maxgeom``, so the
            cloud is subsampled.
    """
    dt = policy.get_dt()
    # Checked at every control-step boundary by all three loops (see InterruptFlag).
    interrupt = InterruptFlag()
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
    # RL tracker gains from the first step. The clip branch above has already done this; this
    # is the non-clip path, which used to spawn on the XML gains and switch at a handoff that
    # no longer exists.
    if clip is None:
        robot.set_pd_gains_from_policy(policy)
        print("[INFO] Spawning straight into FM+tracker control on the RL gains (no standing "
              "hold: the generator runs from step 0 on its learned null-history embedding).")

    # Predicted extra-frame indices that are feet (for the FM viz spheres).
    foot_idx = [i for i, n in enumerate(policy.ctx.target_frame_names)
                if "ankle" in n.lower() or "foot" in n.lower()]
    if viz and not headless:
        print(f"[INFO] FM target frames: {policy.ctx.target_frame_names}; feet -> {foot_idx}")

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
                         *[f"tau_{n}" for n in isaac], *jp_cols, *ee_cols])
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
        # The PREVIOUS tick's raw network output, read before this tick overwrites it. This is
        # what the `actions` observation term is, and recording it is what lets the C++ side be
        # driven fully teacher-forced -- otherwise a replay is autoregressive through that one
        # term and the ticks after the first divergence stop being independent samples.
        prev_action = (np.asarray(policy._trk.action_isaac, dtype=np.float64).copy()
                       if replay is not None else None)
        state = build_state(robot, policy, vel_cmd, depth_cams, want_depth_hits=viz,
                            want_raw_depth=replay is not None,
                            fm_depth_cams=fm_depth_cams, fm_lag=fm_lag)
        obs = policy.create_obs(state)
        action = policy.get_action(obs, robot.joint_names)
        robot.apply_action(action)
        if replay is not None:
            _record_replay_tick(replay, policy, state, obs, prev_action)
        return action, state

    # ------------------------------------------------------------------
    # Forward kinematics of the PREDICTED window (viewer ribbon + log columns).
    # ------------------------------------------------------------------
    # deploy.fk, shared with the ROS marker node (g1_control fm_plan_viz), so the RViz ribbon
    # and this viewer's ribbon are the same object by construction and a disagreement between
    # the two views is a real one. It builds its own pinocchio model rather than borrowing
    # robot.mj_model, so FK of a hypothetical pose cannot touch the live sim -- and it is a
    # DIFFERENT model from the one this file simulates, which deploy/tests/test_fk_parity.py
    # measures at 4.6e-7 m over these frames.
    _fk = WindowFK(DEFAULT_FK_MODEL, policy.ctx.target_frame_names, isaac)
    if _fk.missing:
        print(f"[WARN] FK frames absent from the URDF, so no predicted-frame ribbon: {_fk.missing}")

    def _fm_frame_paths_w(node_idxs) -> np.ndarray | None:
        """World positions of the FM target frames implied by the predicted window.

        The slim window predicts ``joint_pos`` but no ``ee_pos``, so the frames are not
        served -- they are IMPLIED, and forward kinematics recovers them: set the reference
        base pose + reference joints on a scratch model and read the frame positions back.
        The engine is ``deploy.fk.WindowFK`` on the robot's URDF -- which agrees with
        ``FlowMatchingConfig.fk_model_path``, the model the FM's FK losses are defined
        against, to 4.6e-7 m on these frames (``deploy/tests/test_fk_parity.py``) -- and the
        SAME object the ROS marker node draws with, so this ribbon and the RViz one cannot
        disagree.

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
        if lifted is None or jp_isaac is None or not _fk.resolved:
            return None
        ref_p, ref_q = lifted
        idxs = np.asarray(node_idxs, dtype=int)
        return _fk.frame_positions(ref_p[idxs], ref_q[idxs], jp_isaac[idxs])

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
                         *torques, *jp_vals, *ee_vals])

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

    def run_loop(viewer) -> bool | None:
        """Live FM + tracker control from the first step; return False if the robot fell.

        Returns ``None`` if the user interrupted the run -- neither survival nor a fall.

        There is no warmup hold. The generator drives from step 0 with its learned
        null-history embedding covering the not-yet-full window, which is what the RL side
        does and therefore what the pair was trained for. Every alternative that existed here
        fabricated a history -- a stiffened PD hold that filled the window with standing
        snapshots, or a static fill that repeated one snapshot into all N slots -- and a
        fabricated history is off-distribution: real history always has motion between its
        steps. The hold additionally put the robot in a state no training episode started
        from.
        """
        nonlocal handoff_time
        handoff_time = robot.mj_data.time
        # Live: the FM + tracker have control from here.
        while True:
            if interrupt.hit:
                return None
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

    def run_clip_loop(viewer) -> bool | None:
        """Play the baked clip through the tracker; optionally hand off to the FM.

        Each step the command is the clip's ``target_vel`` (already ramped at
        generation, so no extra rate limit) and the tracker reference is the clip
        (``observe()`` keeps the FM history warm). Once ``clip_handoff_time`` is
        reached the reference source switches to the live FM generator; the command
        keeps following the authored clip velocity profile. ``clip_handoff_time < 0``
        never hands off (clip-only). Returns False if the robot fell, ``None`` if the
        user interrupted the run.
        """
        nonlocal handoff_time
        clip_only = clip_handoff_time < 0
        handed_off = False
        clip_step = clip.start_step
        while True:
            if interrupt.hit:
                return None
            if viewer is not None and not viewer.is_running():
                return True
            if total_time > 0 and robot.mj_data.time > total_time:
                return True
            s = min(clip_step, clip.real_steps - 1)
            # Same teacher-forcing hook the FM loop uses: the `actions` observation term is
            # the runtime's own previous output, so a replay that did not record it would be
            # autoregressive through that one term and its ticks would stop being independent.
            prev_action = (np.asarray(policy._trk.action_isaac, dtype=np.float64).copy()
                           if replay is not None else None)
            state = build_state(robot, policy, clip.command_at(s), depth_cams,
                                fm_depth_cams=fm_depth_cams,
                                want_depth_hits=viz,
                                want_raw_depth=replay is not None)
            if (not clip_only) and robot.mj_data.time >= clip_handoff_time:
                if not handed_off:
                    handoff_time = robot.mj_data.time
                    print(f"[INFO] Clip->FM handoff at t={handoff_time:.2f}s "
                          f"(clip step {s}/{clip.real_steps - 1}); FM now drives the reference.")
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
                # ONE number for both decisions: the clip's own elapsed tick. The policy
                # latches its window anchor on it too (create_obs_from_reference below), and
                # under window_anchor='plan' the two must fire on the same step.
                clip_tick = clip_step - clip.start_step
                if clip.ego_anchor and clip_tick % max(clip.ego_anchor_every, 1) == 0:
                    clip.reanchor_ego(s, state["root_pos"], state["root_quat"])
                # Keep the generator's cadence counter moving while the clip drives, so the
                # first re-plan after the handoff lands on the phase it would have.
                policy.advance_dormant_step()
                ref = clip.reference_at(s, policy.window_offsets)
                obs = policy.create_obs_from_reference(state, ref, clip_tick=clip_tick)
            action = policy.get_action(obs, robot.joint_names)
            robot.apply_action(action)
            # Recorded on the CLIP path too, so --dump_replay means the same thing in both
            # modes. Without this the flag silently produced an empty stream here, and the
            # comparison built on it died at write() rather than at the missing call.
            if replay is not None:
                _record_replay_tick(replay, policy, state, obs, prev_action)
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
    # The handler is installed only around the loop, so a Ctrl-C during setup or during the
    # artifact writing below still behaves normally.
    with interrupt:
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


def _scene_tile_local_origin(mj_model):
    """The tile-local point this scene put at its world origin, or ``None``.

    ``generate_clip_tile_scene.py`` emits it as a ``<custom><numeric>`` precisely so this does
    not have to be passed by hand: a wrong value silently mis-places the robot instead of
    raising, which is the failure it exists to prevent. A scene without the field (the flat
    ``basic_scene``, or any tile scene generated before it existed) returns ``None`` and the
    clip is used in its own frame, unchanged.

    Args:
        mj_model: The compiled MuJoCo model.

    Returns:
        ``[3]`` float array, or ``None``.
    """
    import mujoco
    import numpy as np

    for i in range(int(mj_model.nnumeric)):
        if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_NUMERIC, i) != "tile_local_origin":
            continue
        adr = int(mj_model.numeric_adr[i])
        return np.asarray(mj_model.numeric_data[adr:adr + 3], dtype=np.float64)
    return None


def main() -> None:
    """Parse arguments, build the policy + robot, and run the control loop."""
    parser = argparse.ArgumentParser(description="Run the G1 generator+tracker pair in MuJoCo.")
    parser.add_argument("--experiment", type=str, default="g1")
    parser.add_argument("--load_run", type=str, default=None, help="Tracker run dir (default: latest).")
    parser.add_argument("--fm_depth_lag_steps", type=int, default=None,
                        help="Synthetic camera lag for the GENERATOR, in control steps of "
                             "20 ms (camera and control both run at 50 Hz). Omit to derive "
                             "it from the loaded fm.onnx: the rounded mean of the "
                             "depth_latency_probs it trained under, so the shipped "
                             "(0, .10, .40, .50) gives 2 steps = 40 ms against a 48 ms mean. "
                             "0 disables it. A graph exported before the latency model "
                             "carries no distribution and derives 0. Only the generator is "
                             "delayed -- the tracker's own depth observation stays live, and "
                             "hardware adds nothing, its lag being physical already.")
    parser.add_argument("--fm_onnx", type=str, default=None,
                        help="Which generator to load: a path, or a bare filename resolved "
                             "inside <run>/exported/. Exports are named for the FM checkpoint "
                             "(e.g. 2026-08-12_17-07-47_scan_pin1_fm.onnx), so one run dir "
                             "holds every generator trained against that teacher. Optional "
                             "when the dir holds exactly one; REQUIRED when it holds several "
                             "-- defaulting to the newest is the ambiguity the naming exists "
                             "to remove.")
    parser.add_argument("--mode", choices=["joystick", "eval"], default="joystick")
    parser.add_argument("--command", type=str, default="smooth_ramp_running",
                        help="Scripted velocity profile in experiments.velocity_commands (eval mode).")
    parser.add_argument("--scene", type=str, default="basic_scene")
    parser.add_argument("--robot_name", type=str, default="g1_perceptive")
    parser.add_argument("--num_threads", type=int, default=0, help="onnxruntime intra-op threads (0=auto).")
    parser.add_argument("--tracker_on_cpu", action="store_true", default=False,
                        help="Run the tracker on the CPU while the generator keeps the GPU. "
                             "Costs the tracker nothing and keeps it out of the generator's "
                             "captured submission; this is what the hardware controller does "
                             "by default.")
    parser.add_argument("--fm_cuda_graph", action="store_true", default=False,
                        help="Replay a captured CUDA graph for the generator instead of "
                             "launching its ~3000 kernels individually (9.8 -> 8.3 ms per "
                             "re-plan on an RTX 4070). Needs a CUDA provider and a "
                             "fixed-batch fm.onnx; refused loudly if either is missing.")
    parser.add_argument("--providers", default="auto",
                        help="Comma-separated onnxruntime execution providers for BOTH nets, "
                             "e.g. 'cuda' or 'cuda,cpu' (aliases: cpu/cuda/trt). Default "
                             "'auto' = the GPU when the installed wheel has one, else the "
                             "CPU. This is a SIM default, not the deployment one: the "
                             "generator is ~99%% of the step cost and runs it in ~100 ms on a "
                             "CPU, so sim2sim defaults to the GPU while transfer/deploy still "
                             "requires hardware to state its provider. 'cpu' forces the "
                             "deployment-representative timing. CPU is always appended as a "
                             "fallback for ops the accelerator lacks. A NAMED provider the "
                             "wheel lacks raises; 'auto' never does. What a session actually "
                             "registered is printed at load -- onnxruntime falls back to the "
                             "CPU with a warning, not an error.")
    parser.add_argument("--allow_privileged_extero", action="store_true", default=False,
                        help="Run a generator conditioned on the PRIVILEGED dense height scan "
                             "(exact, noise-free, 121x121 @ 0.025 m). Refused by default "
                             "because no robot can produce that reading; MuJoCo can, by "
                             "casting the same grid against the same terrain, so this makes "
                             "the Isaac-only diagnostic testable in a second simulator. It "
                             "costs ~20 ms/step of ray casting (measured, 14641 rays), and "
                             "any survival number it produces is a statement about the model "
                             "given perfect terrain -- never about deployment.")
    parser.add_argument("--allow_robot_asset_mismatch", action="store_true", default=False,
                        help="run a tracker/generator pair whose robot fingerprints disagree. "
                             "SIM ONLY. The fingerprint hashes whole files, so an edit to a "
                             "part of the MJCF neither model reads -- the actuator PD gains, "
                             "which sim2sim overwrites from the policy anyway -- trips it "
                             "without any model being stale. Read the reported fields first: "
                             "joint_names, body_names or total_mass differing is never benign.")
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
                             "The FM takes control immediately.")
    parser.add_argument("--rt_factor", type=float, default=1.0,
                        help="Viewer playback speed vs real time (1.0=real, 0.5=half; <=0 disables pacing).")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="No viewer window. NOT just a display choice: a headed run is a "
                             "DIFFERENT TRAJECTORY from a headless one, measured. Two headless "
                             "runs of the same command are bit-identical (max |dx| = 0), and so "
                             "are runs differing in --rt_factor or in the depth hit-point "
                             "gather -- but opening the viewer changes the outcome, "
                             "deterministically and reproducibly (tile20 at constant 0.8: "
                             "survives 16 s headless, falls at 6.04 s headed, the same 6.04 s "
                             "every time). viewer.sync() is the only viewer-exclusive call in "
                             "the loop; it perturbs mjData enough to redirect a contact-rich "
                             "walk. NEVER COMPARE ACROSS MODES -- pick one for an A/B. Watching "
                             "is still reproducible, just a different trajectory than the "
                             "numbers a headless sweep produces.")
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
    parser.add_argument("--fm_traj_points", type=int, default=0,
                        help="Number of spheres drawn along the FM predicted root/feet trajectory; "
                             "<=0 draws every predicted node (dense ribbon, like IsaacLab).")
    parser.add_argument("--depth_noise", action="store_true", default=False,
                        help="Corrupt the depth images with the TRAINING noise (plane bias, "
                             "per-pixel noise, blur, dropout) and jitter the camera mounts. Off by "
                             "default, matching the -play task the policy was exported from.")
    parser.add_argument("--depth_noise_seed", type=int, default=0,
                        help="Seed for --depth_noise.")
    parser.add_argument("--viz_geomgroup", type=str, default="1,2,3", metavar="LIST",
                        help="Comma-separated geom GROUPS to render. 0 = collision geoms, "
                             "1 = visual meshes, 2 = terrain, 3 = the camera mounts. "
                             "DEFAULT '1,2,3' = the robot as it looks, on its terrain. "
                             "MuJoCo's own default is 0,1,2, which draws every collider on "
                             "top of the visual mesh it duplicates -- so the two are "
                             "indistinguishable and disabling a geom's contact changes "
                             "nothing on screen. Pass '' for MuJoCo's default, '0,2' to see "
                             "only what can actually collide, or '1,2' for exactly what the "
                             "depth cameras ray-cast. Group 3 is separate precisely BECAUSE "
                             "the cameras must not cast it: the mounts are solid CAD "
                             "assemblies whose origin sits inside them, so casting one "
                             "returns 780/780 pixels at 7-24 mm -- under min_distance, hence "
                             "all invalid, i.e. that camera goes silently blind.")
    parser.add_argument("--viz_contacts", action="store_true", default=False,
                        help="Show contact points + forces and make geoms transparent. An "
                             "empty view means nothing is touching -- which for a "
                             "self-collision question is the answer, not a failure.")
    parser.add_argument("--depth_viz_stride", type=int, default=8,
                        help="Draw every Nth depth pixel's hit point in the viewer (1560 hits per "
                             "step would exceed the scene's geom capacity).")
    # TODO(mcap-sim2sim): this writes sim_log.csv plus .npy sidecars, while the obelisk side
    # now records MCAP with schemas (g1_msgs/ControllerDebug + FmPlan) that already carry
    # most of these columns. Having this runner write MCAP too would let ONE set of tooling
    # read both sims -- so "walks in MuJoCo, falls in obelisk" becomes a diff rather than two
    # sets of plots eyeballed side by side. Needs the `mcap` python lib plus schemas written
    # by hand (there is no ROS here to generate them), and gen_tracker_metrics.py currently
    # consumes the CSV, so it is a real change rather than a swap.
    # ---- what counts as a fall (MuJoCo sim2sim only; nothing in transfer/deploy sees these)
    parser.add_argument("--fall_height", type=float, default=0.2,
                        help="Pelvis height above the scene's LOWEST terrain surface, below "
                             "which the run is scored a fall (metres). Terrain-relative "
                             "because the old absolute 0.2 m marked every descending tile as "
                             "fallen on arrival: tile20's bottom slab is at -0.777 m, so a "
                             "healthy robot standing there has a pelvis at ~0.0.")
    parser.add_argument("--fall_tilt_deg", type=float, default=70.0,
                        help="Maximum lean from upright, in ROLL and PITCH separately "
                             "(degrees). Not yaw -- yaw is heading, and a yaw bound would "
                             "fail a robot that merely turned around. This is what catches a "
                             "collapse on an UPPER terrain level, which stays far above the "
                             "scene-minimum height bound.")
    parser.add_argument("--dump_replay", type=str, default=None, metavar="DIR",
                        help="Record every tick's INPUTS and this runtime's answers to DIR, as "
                             "the manifest+blob fixture format the C++ side already reads "
                             "(deploy/replay_stream.py). Feeding that stream to "
                             "deploy_cpp's replay tool -- synchronous, single-threaded, zero "
                             "noise -- is what separates 'the C++ computes something "
                             "different' from 'the C++ is fed something different'. Costs a "
                             "second depth return (raw metres) and a few tens of MB per run; "
                             "the control path is otherwise untouched.")
    parser.add_argument("--fm_lead_ticks", type=int, default=0,
                        help="Emulate the C++ runtime's generator LEAD: condition the "
                             "generator on the exteroception and velocity command from this "
                             "many ticks ago, which is what the async stack's early request "
                             "actually sees (its history is NOT stale -- it projects the rows "
                             "it will serve, exactly). 0 = this runner's synchronous "
                             "behaviour. The C++ default fm_lead_sec=0.02 at dt=0.02 is 1 "
                             "tick, so --fm_lead_ticks 1 is the matched comparison; 3 "
                             "reproduces the old 60 ms default.")
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Where --log writes sim_log.csv + metrics (default: "
                             "<run_dir>/mujoco_gen_tracker_logs). That default is ONE fixed "
                             "path per run dir, so concurrent runs clobber each other; give "
                             "each a separate --log_dir to sweep clips in parallel.")
    parser.add_argument("--max_vx", type=float, default=1.0)
    parser.add_argument("--max_vy", type=float, default=0.5)
    parser.add_argument("--max_wz", type=float, default=1.0)
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
    parser.add_argument("--clip_case", type=str, default=None,
                        help="In --clip mode, name a case from the library's JSON sidecar (e.g. "
                             "'stand', 'stand_walk_stand', 'terrain' in the debug library under "
                             "transfer/obelisk/g1_control/resource/debug_clips/) "
                             "instead of giving --clip_id. Names survive a library regeneration; "
                             "indices do not. Only libraries built by make_debug_clips.py have a "
                             "sidecar. Mutually exclusive with --clip_id.")
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
    parser.add_argument("--clip_ego_anchor_every", type=int, default=0,
                        help="Re-anchor cadence (control steps) for --clip_ego_anchor. 0 (the "
                             "default) DERIVES it from the tracker's own fm_update_every, which "
                             "under window_anchor='plan' is not a preference but a requirement: "
                             "the clip placement and the window anchor have to be solved from the "
                             "robot at the same instant or the egomotion between them leaks into "
                             "base_pose. Set it explicitly only to reproduce an older run.")
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
    # Generators are named for the checkpoint they came from, so a run's exported/ can hold
    # several; --fm_onnx names one (a bare filename resolves inside exported/). Ambiguity is
    # refused rather than defaulted -- reported as a CLI error, since it is a configuration
    # question for the caller and not a bug to hand back a traceback for.
    try:
        fm_onnx = resolve_fm_onnx(exported, args.fm_onnx, required=False)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"[ERROR] {exc}")
    # Clip-only playback (a clip with no FM handoff) drives the tracker straight from the
    # clip reference, so the generator is optional -- this is how a pure clip-tracking
    # teacher (which has no fm.onnx) is played in MuJoCo. Every other mode (joystick/eval,
    # or a clip->FM handoff) needs the generator.
    clip_only = bool(args.clip) and args.clip_handoff_time is None
    if not os.path.exists(tracker_onnx):
        sys.exit(f"[ERROR] Missing {tracker_onnx}. Export with export_tracker_onnx.py first.")
    if clip_only:
        fm_onnx = None
    elif fm_onnx is None:
        sys.exit(f"[ERROR] No generator in {exported}. Export one with play.py "
                 "--fm_checkpoint or scripts/flow_matching/export_fm_onnx.py, or play "
                 "clip-only (--clip <h5> without --clip_handoff_time) to run the tracker "
                 "with no FM.")
    print(f"[INFO] Tracker run: {run_name}\n[INFO] tracker.onnx: {tracker_onnx}\n"
          f"[INFO] generator: "
          f"{fm_onnx if fm_onnx else '(none -- tracker-only clip playback)'}")

    policy = GenTrackerPolicy(
        tracker_onnx, fm_onnx, fm_update_every=args.fm_update_every, num_threads=args.num_threads,
        zero_noise=args.zero_noise,
        fm_cuda_graph=args.fm_cuda_graph,
        tracker_on_cpu=args.tracker_on_cpu,
        providers=([p.strip() for p in args.providers.split(",") if p.strip()] or None),
        allow_privileged_extero=args.allow_privileged_extero,
        allow_robot_asset_mismatch=args.allow_robot_asset_mismatch,
        fm_lead_ticks=args.fm_lead_ticks,
    )
    policy.load()
    replay = None
    if args.dump_replay:
        from deploy.replay_stream import ReplayWriter
        replay = ReplayWriter(args.dump_replay, policy.get_num_actions(),
                              {"tracker_onnx": tracker_onnx, "fm_onnx": fm_onnx or ""})
        print(f"[INFO] recording a replay stream to {args.dump_replay}")
    if args.fm_lead_ticks:
        print(f"[INFO] emulating the async runtime's generator lead: the generator sees the "
              f"exteroception and command from {args.fm_lead_ticks} tick(s) ago "
              f"({args.fm_lead_ticks * policy.get_dt() * 1e3:.0f} ms). The tracker's own "
              f"observation is NOT delayed.")
    # What each graph REGISTERED, not what was asked for. onnxruntime reports a failed CUDA
    # session as a warning and carries on running on the CPU, so with the GPU now the
    # default this line is the only thing standing between "the generator is on the GPU" and
    # a 5x-slower run that never said so.
    print(f"[INFO] onnxruntime providers: {policy.active_providers()}")

    robot = Robot(robot_name=args.robot_name, scene_name=args.scene,
                  joystick_scaling=np.ones(3), input_function=lambda t: np.zeros(3),
                  fall_height=args.fall_height,
                  fall_tilt_rad=float(np.deg2rad(args.fall_tilt_deg)))
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
        command_fn = JoystickCommand(args.max_vx, args.max_vy, args.max_wz)
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
        clip_id = args.clip_id
        if args.clip_case is not None:
            if args.clip_id != 0:
                sys.exit("[ERROR] Pass either --clip_case or --clip_id, not both.")
            clip_id = resolve_case(clip_path, args.clip_case)
            print(f"[INFO] Clip case '{args.clip_case}' -> clip {clip_id}.")
        # A TILE scene re-centres its tile so a chosen tile-local point sits at the world
        # origin, while a clip's poses stay in the library's tile-local frame. Nothing
        # reconciled the two, so a world-anchored clip spawned the robot by the whole offset
        # -- 8.6 m downrange and 0.13 m high on tile 0, onto the far platform instead of the
        # approach, which reads as the tracker instantly falling.
        #
        # ClipReference already has the mechanism: _lift_pos is `anchor_pos + p_rel`, so an
        # anchor of MINUS the origin expresses every clip pose in scene coordinates. Nothing
        # in ClipReference changes.
        #
        # Ego anchoring (--clip_ego_anchor) overrides the placement from the robot's measured
        # pose, which is already in scene coordinates, so it self-corrects and this is a no-op
        # for it -- it is the world-anchored default that needs the shift.
        tile_origin = _scene_tile_local_origin(robot.mj_model)
        clip = ClipReference(clip_path, policy.get_joint_names(),
                             list(policy.ctx.target_frame_names), policy.base_frame_name,
                             clip_id=clip_id,
                             start_time=args.clip_start_time,
                             ego_anchor=args.clip_ego_anchor,
                             ego_anchor_every=(args.clip_ego_anchor_every
                                               or policy.fm_update_every),
                             root_frame_bodies=policy._trk.get_root_frame_bodies())
        if tile_origin is not None:
            clip.set_placement(-tile_origin, np.array([0.0, 0.0, 0.0, 1.0]))
            print("[INFO] Clip placed into the tile scene: subtracting its "
                  f"tile-local origin {tile_origin.tolist()} m from every pose.")
        policy.assert_clip_reference_compatible(clip)
        if abs(clip.dt - policy.get_dt()) > 1e-6:
            print(f"[WARN] Clip dt {clip.dt} != policy dt {policy.get_dt()}; "
                  "regenerate the clip at the control dt.")
        clip_handoff_time = args.clip_handoff_time if args.clip_handoff_time is not None else -1.0
        # Playable duration from the chosen start to the clip end (sim time begins at 0 here).
        # Measured against the REAL length: the padded tail is a frozen pose that still
        # carries the last command, so running into it looks like a tracking failure.
        remaining_s = (clip.real_steps - clip.start_step) * clip.dt
        if total_time <= 0:  # clip-only -> remaining length; clip->FM -> a few extra seconds of FM.
            total_time = remaining_s if clip_handoff_time < 0 else remaining_s + 5.0
        print(f"[INFO] Clip: {clip_path}\n"
              f"[INFO] {clip.real_steps} real steps of {clip.num_steps} padded, "
              f"{clip.real_length_s:.2f}s @ dt={clip.dt}; "
              f"start={clip.start_step * clip.dt:.2f}s (step {clip.start_step}); "
              f"handoff={'clip-only' if clip_handoff_time < 0 else f'{clip_handoff_time:.2f}s'}; "
              f"anchor={f'ego/every {clip.ego_anchor_every}' if args.clip_ego_anchor else 'world'}"
              f" ({policy.window_anchor}); "
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

    # A SECOND caster when the generator samples those same lenses on its own grid. Two
    # instances rather than one: MujocoDepthCameras returns its cameras as ONE array, which
    # cannot hold two widths -- and that refusal is still correct, since each artifact's own
    # cameras do share a resolution.
    fm_depth_cams = None
    if policy.depth_sampling_is_split():
        fm_spec = policy.get_fm_depth_camera_spec()
        fm_depth_cams = MujocoDepthCameras(robot.mj_model, fm_spec)
        print(f"[INFO] Generator samples the cameras separately at "
              f"{fm_spec[0]['width']}x{fm_spec[0]['height']} "
              f"(tracker: {depth_spec[0]['width']}x{depth_spec[0]['height']}); casting both.")
        if args.depth_noise:
            fm_depth_cams.enable_noise(np.random.default_rng(args.depth_noise_seed))
            # The mount jitter is SHARED: one physical camera cannot be in two places, and
            # the two casters are two samplings of it. (The per-frame noise draws stay
            # independent -- they cannot be shared across differing pixel counts.)
            fm_depth_cams.mirror_extrinsics_from(depth_cams)

    # From the CLI when given, else from the graph's own trained distribution. Derived
    # rather than hard-coded so a checkpoint trained at a different latency brings its own
    # number, and a pre-latency graph correctly gets none.
    _probs = getattr(policy.ctx, "depth_latency_probs", None)
    # --fm_lead_ticks and --dump_replay both exist to line this runner up against the C++
    # runtime, and the C++ applies NO synthetic camera lag. Left alone the two STACK: the
    # generator would see `fm_lead_ticks + fm_depth_lag_steps` ticks while the C++ sees
    # `fm_lead_ticks`, so the documented matched comparison (--fm_lead_ticks 1) would silently
    # become 3 ticks / 60 ms against 1. The lag default is DERIVED and non-zero, so that
    # happens without anyone asking for it, and every C++-parity conclusion drawn from the run
    # would be wrong with nothing on screen to say so.
    #
    # --dump_replay is worse than a mismatch. `_record_replay_tick` records `depth_raw` from
    # the tracker's LIVE cast beside the `fm_camera_pose` the lag has already replaced with a
    # stale one, so the stream is internally inconsistent: the C++ rebuilds the generator's
    # extero row from live pixels unprojected with an old pose -- exactly the egomotion
    # displacement the xyz channels exist to remove. compare_replay.py then reports an
    # `fm_extero` divergence that is the harness's fault, which is the one thing that harness
    # promises to rule out.
    _cpp_parity = bool(args.fm_lead_ticks) or bool(args.dump_replay)
    if args.fm_depth_lag_steps is not None:
        if _cpp_parity and args.fm_depth_lag_steps > 0:
            sys.exit(
                f"[FATAL] --fm_depth_lag_steps {args.fm_depth_lag_steps} cannot be combined "
                "with --fm_lead_ticks / --dump_replay. Those emulate the C++ runtime, which "
                "applies no synthetic camera lag, so the two delays would stack and the "
                "comparison would not be the one you asked for. Drop one, or pass "
                "--fm_depth_lag_steps 0 to state that you want the lag off.")
        fm_lag = FmDepthLag(args.fm_depth_lag_steps, "--fm_depth_lag_steps")
    elif _cpp_parity:
        fm_lag = FmDepthLag(0, "forced off: --fm_lead_ticks/--dump_replay emulate the C++ "
                               "runtime, which applies none. Pass --fm_depth_lag_steps to "
                               "override, and read the note above first")
    else:
        fm_lag = FmDepthLag(default_depth_lag_steps(_probs),
                            f"derived from fm.onnx depth_latency_probs={_probs}")
    success = run(policy, robot, command_fn, total_time, args.headless, log_path,
                  args.rt_factor, standing_config, args.viz,
                  args.vel_arrow_scale, args.vel_arrow_height,
                  args.spawn_z_offset, args.clip_spawn_z_offset,
                  args.cmd_max_acc, args.fm_traj_points,
                  viz_geomgroup=([int(g) for g in args.viz_geomgroup.split(",") if g.strip() != ""]
                                 if args.viz_geomgroup else None),
                  viz_contacts=args.viz_contacts,
                  spawn_on_box=args.spawn_on_box,
                  clip=clip, clip_handoff_time=clip_handoff_time,
                  depth_cams=depth_cams, depth_viz_stride=args.depth_viz_stride,
                  fm_depth_cams=fm_depth_cams, replay=replay, fm_lag=fm_lag)
    # None = interrupted, which is neither survival nor a fall; saying "Survived: None"
    # would put a non-result into the same line every sweep greps for.
    if success is None:
        print("[INFO] Run INTERRUPTED by the user; artifacts below are partial.")
    else:
        print(f"[INFO] Run finished. Survived: {success}")
    if replay is not None:
        # Written after the loop, so a run that fell still yields every tick it managed --
        # those ticks are exactly the interesting ones.
        manifest = replay.write(extra_attrs={
            "scene": args.scene, "robot": args.robot_name, "command": args.command,
            "fm_lead_ticks": int(args.fm_lead_ticks), "survived": success,
            "dt": float(policy.get_dt()),
            # The POLICY's joint order. Every consumer that compares joints needs it: the
            # robot's order differs on 27 of the 29, and subtracting one from the other gives
            # a large, smooth, fictitious error.
            "joint_names_isaac": list(policy.get_joint_names()),
            "window_offsets": [int(o) for o in policy.window_offsets],
            "reference_blocks": [str(b) for b in policy.reference_blocks],
            "node_dim": int(policy.ctx.D),
        })
        print(f"[INFO] wrote the replay stream to {manifest}")

    if log_path:
        # Stamp the loop's OWN verdict into sim_meta.yaml before the metrics run.
        #
        # The `fell` CSV column structurally cannot carry it: the row is written inside
        # control_step() with robot.failed() evaluated BEFORE that step's physics, and when the
        # fall is detected the loop returns immediately without writing another row. So the
        # column stays 0 for a run that fell, and gen_tracker_metrics -- which inferred
        # survival from that column -- reported `"survived": true, "time_to_fall_s": null`
        # while the console printed `Survived: False`. A sweep reads the JSON, not the console,
        # so falls were being counted as successes.
        meta_path = os.path.join(os.path.dirname(log_path), "sim_meta.yaml")
        try:
            with open(meta_path) as mf:
                meta = yaml.safe_load(mf) or {}
            meta["survived"] = success
            # Detection time, which is the last simulated instant either way. `None` when the
            # user interrupted: that is neither survival nor a fall.
            meta["time_to_fall_s"] = (None if success is not False
                                      else float(robot.mj_data.time))
            meta["sim_end_time_s"] = float(robot.mj_data.time)
            # WHICH trigger fired, which the CSV cannot say either: "it fell" and "it pitched
            # face-first off a step while its pelvis was still high" are different findings.
            # "unknown" only if the predicate has gone quiet since the loop returned.
            meta["failure_reason"] = ("" if success is not False
                                      else robot.failure_reason() or "unknown")
            with open(meta_path, "w") as mf:
                yaml.dump(meta, mf, default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print(f"[WARN] no sim_meta.yaml at {meta_path}; the metrics will infer survival "
                  "from the `fell` column, which cannot see a fall on the final step.")

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
