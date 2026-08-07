# Copyright 2026 Zachary Olkin. All rights reserved.

"""One recorded run, decoded into aligned arrays.

The bag holds streams at wildly different rates -- the estimator at 1 kHz, the control tick
at 50 Hz, the cameras at 50 Hz on their own clock. Anything that wants to look at "the robot
at tick k" has to pair them, and doing that ad hoc in each plot is how two views end up
disagreeing about what a moment contained. So the pairing happens once, here: the CONTROL
TICK is the timeline, and every other stream is sampled at the newest sample at or before
each tick -- which is what the controller itself saw.
"""

from __future__ import annotations

import numpy as np

TICK = "/obelisk/g1/debug/tick"
FM_PLAN = "/obelisk/g1/debug/fm_plan"
EST = "/obelisk/g1/est_state"
CTRL = "/obelisk/g1/ctrl"
VEL_CMD = "/obelisk/g1/vel_cmd"
DEPTH = ("/depth/near/image_processed", "/depth/far/image_processed")

TOPICS = {TICK, FM_PLAN, EST, CTRL, VEL_CMD, *DEPTH}

#: Widths of the fixed reference-window blocks. ``joint_pos``/``joint_vel`` are the joint
#: count, and ``ee_pos`` is whatever remains -- see :func:`block_slices`.
_FIXED_BLOCK_WIDTH = {"base_pose": 9, "base_vel": 6}


def block_slices(blocks: list[str], node_dim: int, num_joints: int) -> dict[str, slice]:
    """Where each block sits within one reference-window node.

    The tick message names its own layout (``reference_blocks`` / ``reference_node_dim``),
    so a reader never has to restate the tracker's export. That matters because the block
    set is a training choice: this teacher drops ``joint_vel`` and ``ee_pos``, and a
    hardcoded layout would silently mis-slice a checkpoint that keeps them.

    Args:
        blocks: Block names in order, from ``ControllerDebug.reference_blocks``.
        node_dim: Width of one node, from ``reference_node_dim``.
        num_joints: Joint count (``len(action_isaac)``).

    Returns:
        Block name -> slice within a node.
    """
    widths = dict(_FIXED_BLOCK_WIDTH, joint_pos=num_joints, joint_vel=num_joints)
    out, off = {}, 0
    for i, name in enumerate(blocks):
        if name in widths:
            width = widths[name]
        else:  # ee_pos, whose width depends on the frame count: take what is left over.
            width = node_dim - off - sum(widths.get(b, 0) for b in blocks[i + 1:])
        out[name] = slice(off, off + width)
        off += width
    return out


def quat_rotate_inv(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` from the world frame into the body frame of ``q``.

    Args:
        q_xyzw: ``[..., 4]`` quaternion, xyzw.
        v: ``[..., 3]`` world-frame vector.

    Returns:
        ``[..., 3]`` body-frame vector.
    """
    q = np.asarray(q_xyzw, dtype=np.float64)
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(xyz, np.asarray(v, dtype=np.float64))
    return v - w * t + np.cross(xyz, t)


def rot6d_to_rpy(d6: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw of a 6D-encoded rotation.

    ``base_pose``'s orientation half is the first two COLUMNS of ``R_yawT R_ref`` -- a
    continuous parameterisation the network can regress, but unreadable on a plot. Recover
    the matrix by Gram-Schmidt and report intrinsic ZYX angles, which are the reference's
    attitude RELATIVE TO the robot's current yaw frame: i.e. the orientation error.

    Args:
        d6: ``[..., 6]`` rot6d.

    Returns:
        ``[..., 3]`` roll, pitch, yaw in radians.
    """
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    r = np.stack([b1, b2, b3], axis=-1)          # columns
    pitch = np.arcsin(np.clip(-r[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(r[..., 2, 1], r[..., 2, 2])
    yaw = np.arctan2(r[..., 1, 0], r[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def _sample_at(src_t: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Index of the newest ``src_t`` sample at or before each ``at`` (0 if none)."""
    if src_t.size == 0:
        return np.zeros(at.shape, dtype=int)
    return np.clip(np.searchsorted(src_t, at, side="right") - 1, 0, src_t.size - 1)


class Run:
    """A decoded run: the control-tick timeline plus everything sampled onto it."""

    def __init__(self, data: dict[str, list], start: float | None = None,
                 end: float | None = None) -> None:
        """Decode and align.

        Args:
            data: ``{topic: [(t, msg)]}`` from ``bag_reader.read_bag``.
            start: Trim to at or after this second (relative to the first message).
            end: Trim to at or before this second.

        Raises:
            SystemExit: If the bag carries no control ticks.
        """
        import sys
        if not data.get(TICK):
            sys.exit(
                f"[ERROR] no {TICK} in this bag. Either the controller ran with "
                f"debug_topics:=false, or it never came up. Topics present: {sorted(data)}"
            )
        self.t0 = min(v[0][0] for v in data.values() if v)

        def window(topic):
            out = [(t - self.t0, m) for t, m in data.get(topic, [])]
            return [(t, m) for t, m in out
                    if (start is None or t >= start) and (end is None or t <= end)]

        ticks = window(TICK)
        self.t = np.array([t for t, _ in ticks])
        self.ticks = [m for _, m in ticks]
        self.plans = window(FM_PLAN)

        self.clip_mode = bool(self.ticks[0].clip_mode)
        self.armed = np.array([m.armed for m in self.ticks], dtype=bool)
        self.num_joints = len(self.ticks[0].action_isaac) or None

        # --- joint ordering ---
        # The policy's order is NOT the robot's (27 of 29 positions differ on the G1), so
        # every hardware-order array is ALSO exposed permuted into policy order. Comparing
        # a reference against a measurement in the wrong order yields a large, smooth,
        # entirely fictitious error -- it looks like a tracking problem, not a bug.
        self.joint_names_isaac = list(getattr(self.ticks[0], "joint_names_isaac", []) or [])
        hw_src = (data.get(CTRL) or data.get(EST) or [(0, None)])[0][1]
        self.joint_names_hw = list(getattr(hw_src, "joint_names", []) or [])
        self._to_isaac = None
        if self.joint_names_isaac and self.joint_names_hw:
            try:
                self._to_isaac = np.array([self.joint_names_hw.index(n)
                                           for n in self.joint_names_isaac])
            except ValueError:
                self._to_isaac = None

        # --- estimator, sampled onto the tick timeline ---
        est = window(EST)
        if est:
            et = np.array([t for t, _ in est])
            idx = _sample_at(et, self.t)
            q_base = np.array([list(m.q_base) for _, m in est])[idx]
            v_base = np.array([list(m.v_base) for _, m in est])[idx]
            self.joint_pos = np.array([list(m.q_joints) for _, m in est])[idx]
            self.base_quat = q_base[:, 3:7]                       # xyzw
            self.base_lin_b = quat_rotate_inv(self.base_quat, v_base[:, 0:3])
            self.base_ang_b = v_base[:, 3:6]
        else:
            n = len(self.t)
            self.joint_pos = np.zeros((n, 0))
            self.base_quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
            self.base_lin_b = np.zeros((n, 3))
            self.base_ang_b = np.zeros((n, 3))

        # --- commanded targets, sampled onto the tick timeline ---
        ctrl = window(CTRL)
        if ctrl:
            ct = np.array([t for t, _ in ctrl])
            self.joint_target = np.array([list(m.pos_target) for _, m in ctrl])[
                _sample_at(ct, self.t)]
        else:
            self.joint_target = np.zeros((len(self.t), 0))

        # --- depth, kept as raw frames plus a per-tick index (images are big) ---
        self.depth_frames, self.depth_index = [], []
        for topic in DEPTH:
            frames = window(topic)
            if not frames:
                self.depth_frames.append([])
                self.depth_index.append(np.zeros(len(self.t), dtype=int))
                continue
            ft = np.array([t for t, _ in frames])
            self.depth_frames.append([
                np.frombuffer(bytes(m.data), dtype=np.float32).reshape(m.height, m.width)
                for _, m in frames])
            self.depth_index.append(_sample_at(ft, self.t))

        # --- per-tick scalars pulled off the debug message ---
        self.command = np.array([list(m.command_effective) for m in self.ticks])
        self.latency = np.array([m.latency_ms for m in self.ticks])
        self.clip_step = np.array([m.clip_step for m in self.ticks])
        self.fm_cursor = np.array([m.fm_cursor for m in self.ticks])
        self.action = np.array([list(m.action_isaac) or [np.nan] for m in self.ticks])

        self.joint_pos_isaac = self._as_isaac(self.joint_pos)
        self.joint_target_isaac = self._as_isaac(self.joint_target)
        self._decode_reference()

    def _as_isaac(self, arr: np.ndarray) -> np.ndarray | None:
        """Permute a hardware-order joint array into the policy's order.

        Args:
            arr: ``[T, nj]`` in the robot's joint order.

        Returns:
            The same data in policy order, or None if the two orders could not be resolved
            (an older bag with no ``joint_names_isaac``) -- in which case the caller must
            NOT fall back to comparing raw, which is the bug this exists to stop.
        """
        if self._to_isaac is None or arr.size == 0 or arr.shape[1] != self._to_isaac.size:
            return None
        return arr[:, self._to_isaac]

    def _decode_reference(self) -> None:
        """Unpack the flat reference window into per-block arrays.

        Every block is expressed relative to the robot AT THAT TICK, so no world frame and
        no re-anchoring is needed to read it -- ``base_pose``'s position half IS the
        tracking error, in the robot's yaw frame.
        """
        first = self.ticks[0]
        self.offsets = np.array(list(first.reference_offsets))
        self.blocks = list(first.reference_blocks)
        self.node_dim = int(first.reference_node_dim)
        self.slices = block_slices(self.blocks, self.node_dim, self.num_joints or 0)

        k, d = len(self.offsets), self.node_dim
        win = np.full((len(self.t), k, d), np.nan)
        for i, m in enumerate(self.ticks):
            flat = np.asarray(m.reference_window, dtype=np.float64)
            if flat.size == k * d:
                win[i] = flat.reshape(k, d)
        self.window = win
        # base_pose = [R_yawT (p_ref - p_act) (3), rot6d(R_yawT R_ref) (6)]
        self.ref_pos_err = (win[:, :, self.slices["base_pose"]][:, :, :3]
                            if "base_pose" in self.slices else None)
        self.ref_rpy_err = (rot6d_to_rpy(win[:, :, self.slices["base_pose"]][:, :, 3:9])
                            if "base_pose" in self.slices else None)
        self.ref_joint = (win[:, :, self.slices["joint_pos"]]
                          if "joint_pos" in self.slices else None)

    def nearest_tick(self, t: float) -> int:
        """Index of the tick closest to time ``t``."""
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def depth_at(self, tick: int, cam: int) -> np.ndarray | None:
        """The depth frame camera ``cam`` had delivered by ``tick``, or None."""
        if not self.depth_frames[cam]:
            return None
        return self.depth_frames[cam][self.depth_index[cam][tick]]

    def summary(self) -> str:
        """One line describing what this run is."""
        n_armed = int(self.armed.sum())
        mode = "DEBUG CLIP (no generator)" if self.clip_mode else "generator"
        return (f"{len(self.t)} ticks ({n_armed} armed, {self.t[-1] - self.t[0]:.1f} s), "
                f"{len(self.plans)} generator plans, {mode}, {self.num_joints} joints")
