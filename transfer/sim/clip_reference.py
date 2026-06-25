# Copyright 2026 Zachary Olkin. All rights reserved.

"""Torch-free baked-clip reference source for the MuJoCo sim2sim runner.

Loads a dense motion-clip library (the HDF5 produced by
``scripts/rsl_rl/generate_parity_clip.py`` / ``generate_clips.py``) with numpy
and serves, per control step, the same world-frame reference dict the online
flow-matching generator emits -- ``{"current": {...}, "future": [...]}`` -- so it
drops straight into ``GenTrackerPolicy._build_obs(state, reference=...)``.  This
lets the MuJoCo runner play a fixed clip through the tracker with **no FM**
(deterministic), and is the counterpart of IsaacLab's ``MotionSequenceCommand``
playback for sim2sim parity.

The clip stores its outputs in a single *clip-world* frame (the per-step stance
anchor baked in at generation time).  The runtime applies a single per-env
*placement* (yaw + xy of the clip in the sim world); the lift mirrors
``traj_tracking/world_lift.py`` exactly.  For parity the placement is the
identity (anchor at the origin, identity yaw), so the lift is a pass-through and
the robot spawns on the clip's step-0 pose.

No torch / isaaclab imports -- numpy + h5py + ``deploy.quat_math_np`` only.
"""

from __future__ import annotations

import json
import math

import h5py
import numpy as np

from deploy.quat_math_np import quat_apply, quat_inv, quat_mul


def _yaw_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only ``(x, y, z, w)`` quaternion of an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    h = 0.5 * yaw
    return np.array([0.0, 0.0, math.sin(h), math.cos(h)], dtype=np.float64)


def _anchor_inv_yaw(anchor_quat: np.ndarray) -> np.ndarray:
    """Lift rotation taking anchor-relative vectors into the world (mirrors world_lift)."""
    return quat_inv(_yaw_quat(quat_inv(anchor_quat)))


class ClipReference:
    """Serve a baked motion clip as world-frame tracker references (numpy).

    Args:
        path: Clip-library ``.h5`` path.
        joint_names_isaac: Joint order the reference ``jp``/``jv`` are returned in
            (the FM/tracker isaac order, ``policy.get_joint_names()``).
        ee_frame_names: End-effector output frames, in the tracker's target order
            (``policy.ctx.target_frame_names``); their world positions feed
            ``ee_ref``.
        base_frame_name: Trajectory output frame used as the robot root
            (``policy.base_frame_name``, e.g. ``"pelvis_link"``).
        clip_id: Which clip in the library to play (parity libraries hold one).
        start_time: Seconds into the clip at which playback begins.  The robot
            spawns on the clip pose at this time and the reference is served from
            here (e.g. to skip a standing prefix).  Clamped to ``[0, clip_end)``.
    """

    def __init__(
        self,
        path: str,
        joint_names_isaac: list[str],
        ee_frame_names: list[str],
        base_frame_name: str = "pelvis_link",
        clip_id: int = 0,
        start_time: float = 0.0,
    ) -> None:
        with h5py.File(str(path), "r") as f:
            self.y = np.asarray(f["y_des"][clip_id], dtype=np.float64)    # [T, P] clip-world
            self.dy = np.asarray(f["dy_des"][clip_id], dtype=np.float64)  # [T, V]
            self.tv = (
                np.asarray(f["target_vel"][clip_id], dtype=np.float64)    # [T, 3] (vx, vy, wz)
                if "target_vel" in f else None
            )
            pos_names = json.loads(f.attrs["pos_output_names"])
            vel_names = json.loads(f.attrs["vel_output_names"])
            self.dt = float(json.loads(f.attrs["dt"]))
        self.num_steps = int(self.y.shape[0])
        self.clip_length_s = self.num_steps * self.dt
        # Step at which playback begins (spawn pose + first served reference).
        self.start_step = int(min(max(round(float(start_time) / self.dt), 0), self.num_steps - 1))

        def pi(name: str) -> int:
            return pos_names.index(name)

        def vi(name: str) -> int:
            return vel_names.index(name)

        b = base_frame_name
        self._base_pos = [pi(f"{b}:pos_{a}") for a in ("x", "y", "z")]
        self._base_ori = [pi(f"{b}:ori_{a}") for a in ("x", "y", "z", "w")]  # xyzw
        self._base_lin = [vi(f"{b}:pos_{a}") for a in ("x", "y", "z")]
        self._base_ang = [vi(f"{b}:ori_{a}") for a in ("x", "y", "z")]
        self._jp = [pi(f"joint:{n}") for n in joint_names_isaac]
        self._jv = [vi(f"joint:{n}") for n in joint_names_isaac]
        self._ee_pos = [[pi(f"{fr}:pos_{a}") for a in ("x", "y", "z")] for fr in ee_frame_names]
        self._ee_ori = [[pi(f"{fr}:ori_{a}") for a in ("x", "y", "z", "w")] for fr in ee_frame_names]
        self._ee_lin = [[vi(f"{fr}:pos_{a}") for a in ("x", "y", "z")] for fr in ee_frame_names]
        self._ee_ang = [[vi(f"{fr}:ori_{a}") for a in ("x", "y", "z")] for fr in ee_frame_names]
        self._num_ee = len(ee_frame_names)

        # Identity placement (clip-world == sim-world) -> lift is a pass-through.
        self.set_placement(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    # ------------------------------------------------------------------
    # Placement + world lift (mirrors traj_tracking/world_lift.py)
    # ------------------------------------------------------------------

    def set_placement(self, anchor_pos: np.ndarray, anchor_quat_xyzw: np.ndarray) -> None:
        """Set the per-clip world placement (yaw-only ground frame)."""
        self._anchor_pos = np.asarray(anchor_pos, dtype=np.float64)
        self._inv_yaw = _anchor_inv_yaw(np.asarray(anchor_quat_xyzw, dtype=np.float64))

    def _lift_pos(self, p_rel: np.ndarray) -> np.ndarray:
        return self._anchor_pos + quat_apply(self._inv_yaw, p_rel)

    def _lift_quat(self, q_rel: np.ndarray) -> np.ndarray:
        return quat_mul(self._inv_yaw, q_rel)

    def _lift_vec(self, v_rel: np.ndarray) -> np.ndarray:
        return quat_apply(self._inv_yaw, v_rel)

    # ------------------------------------------------------------------
    # Reference assembly
    # ------------------------------------------------------------------

    def _frame_state(self, step: int) -> dict:
        """World-frame reference at ``step`` (root + joints + ee), placement-lifted."""
        s = int(min(max(step, 0), self.num_steps - 1))
        y, dy = self.y[s], self.dy[s]
        state = {
            "root_pos": self._lift_pos(y[self._base_pos]),
            "root_quat": self._lift_quat(y[self._base_ori]),     # xyzw
            "root_lin": self._lift_vec(dy[self._base_lin]),
            "root_ang": self._lift_vec(dy[self._base_ang]),
            "jp": y[self._jp].copy(),                            # isaac order
            "jv": dy[self._jv].copy(),
        }
        if self._num_ee > 0:
            state["frame_pos"] = np.stack([self._lift_pos(y[i]) for i in self._ee_pos])
            state["frame_quat"] = np.stack([self._lift_quat(y[i]) for i in self._ee_ori])
            state["frame_lin"] = np.stack([self._lift_vec(dy[i]) for i in self._ee_lin])
            state["frame_ang"] = np.stack([self._lift_vec(dy[i]) for i in self._ee_ang])
        else:
            state["frame_pos"] = np.zeros((0, 3))
            state["frame_quat"] = np.zeros((0, 4))
            state["frame_lin"] = np.zeros((0, 3))
            state["frame_ang"] = np.zeros((0, 3))
        return state

    def state_at(self, step: int) -> dict:
        """Public world-frame reference state at ``step`` (root + joints + ee).

        Returns the placement-lifted reference dict (keys ``root_pos``,
        ``root_quat`` xyzw, ``root_lin``, ``root_ang``, ``jp``/``jv`` in the
        configured joint order, and ``frame_*`` for the ee frames). Used by the
        sim2sim FM probe to drive both sims to identical clip-derived states.
        """
        return self._frame_state(step)

    def command_at(self, step: int) -> np.ndarray:
        """Commanded base velocity ``(vx, vy, wz)`` at ``step`` (clip ``target_vel``)."""
        if self.tv is None:
            return np.zeros(3)
        s = int(min(max(step, 0), self.num_steps - 1))
        return self.tv[s].copy()

    def future_command_horizon(self, step: int, num_future: int) -> np.ndarray:
        """Clip ``target_vel`` over steps ``step+1 .. step+num_future`` (clamped), ``[F, 3]``.

        The decoder's future-command conditioning, matching IsaacLab
        ``GeneratorTrackingCommand.get_command_velocity_horizon`` (steps s+1..s+F).
        Lets the clip→FM handoff feed the FM the *actual* upcoming command rather
        than the held current one, so the MuJoCo FM is conditioned exactly as in
        IsaacLab.
        """
        if self.tv is None:
            return np.zeros((num_future, 3))
        idx = np.minimum(step + 1 + np.arange(num_future), self.num_steps - 1)
        return self.tv[idx].copy()

    def reference_at(self, step: int, future_offsets: list[int]) -> dict:
        """Assemble the ``{"current", "future"}`` reference dict for ``_build_obs``.

        Each future step carries its actual ``target_vel`` under ``"command"`` so
        ``_build_obs`` reproduces IsaacLab's lookahead ``future_command_velocity``.
        Future steps clamp to the clip end (matching the IsaacLab convention).
        """
        cur = self._frame_state(step)
        fut = []
        for o in future_offsets:
            fs = self._frame_state(step + o)
            fs["command"] = self.command_at(step + o)
            fut.append(fs)
        return {"current": cur, "future": fut}

    def spawn_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Spawn pose at ``start_step``: ``(base_pos, base_quat_wxyz, joint_pos_isaac)``.

        ``base_quat`` is converted to MuJoCo ``(w, x, y, z)`` order for
        ``Robot.set_base_and_joints``; joints stay in the isaac order.
        """
        s0 = self._frame_state(self.start_step)
        q = s0["root_quat"]  # xyzw
        base_quat_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        return s0["root_pos"].copy(), base_quat_wxyz, s0["jp"].copy()
