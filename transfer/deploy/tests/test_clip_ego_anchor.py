# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the ego-centric (receding-horizon) clip re-anchoring in the deploy clip player.

Numpy-only (no torch / isaaclab / simulator): builds a tiny synthetic clip-library
``.h5`` and drives :class:`deploy.clip_reference.ClipReference`.  Verifies the
``reanchor_ego`` math mirrors IsaacLab's ``world_lift.ego_placement`` /
``MotionSequenceCommand._reanchor_ego_envs`` -- after re-anchoring, the served
reference's current-step base reproduces the robot's planar position + heading
AND height (z), keeps only the clip's tilt, and the future window recedes from
the re-anchored placement.  Also pins the world-anchored default (identity
placement = pass-through) so both modes are covered.

Run: ``python transfer/deploy/tests/test_clip_ego_anchor.py``
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # transfer/
from deploy.clip_reference import ClipReference, _yaw_angle  # noqa: E402

BASE = "pelvis_link"
JOINTS = ["j0", "j1"]


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """``(x, y, z, w)`` quaternion from intrinsic roll/pitch/yaw (rad)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64)


def _make_clip(path: str, T: int = 6, dt: float = 0.02) -> None:
    """Write a minimal single-clip library: base (pos+ori+vel) + joints, no ee."""
    pos_names = (
        [f"{BASE}:pos_{a}" for a in ("x", "y", "z")]
        + [f"{BASE}:ori_{a}" for a in ("x", "y", "z", "w")]
        + [f"joint:{n}" for n in JOINTS]
    )
    vel_names = (
        [f"{BASE}:pos_{a}" for a in ("x", "y", "z")]
        + [f"{BASE}:ori_{a}" for a in ("x", "y", "z")]
        + [f"joint:{n}" for n in JOINTS]
    )
    P, V = len(pos_names), len(vel_names)
    y = np.zeros((1, T, P), dtype=np.float64)
    dy = np.zeros((1, T, V), dtype=np.float64)
    for s in range(T):
        # Clip-world base: marching downrange in x, drifting in y, with a turning
        # heading and a fixed nonzero tilt (so yaw extraction must ignore roll/pitch).
        y[0, s, 0:3] = [0.3 * s + 1.0, 0.1 * s, 0.78 + 0.01 * s]  # x, y, z (natural height)
        q = _quat_from_euler(0.15, -0.2, 0.25 * s)                # roll, pitch, yaw(s)
        y[0, s, 3:7] = q
        y[0, s, 7:9] = [0.05 * s, -0.03 * s]                      # joints
        dy[0, s, 0:3] = [1.0, 0.0, 0.0]
    with h5py.File(path, "w") as f:
        f.create_dataset("y_des", data=y)
        f.create_dataset("dy_des", data=dy)
        f.attrs["pos_output_names"] = json.dumps(pos_names)
        f.attrs["vel_output_names"] = json.dumps(vel_names)
        f.attrs["dt"] = json.dumps(dt)


def _clip(path: str, ego: bool) -> ClipReference:
    return ClipReference(path, JOINTS, [], BASE, ego_anchor=ego, ego_anchor_every=3)


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.h5")
        _make_clip(path)

        # --- World-anchored default: identity placement is a pass-through. ---
        cw = _clip(path, ego=False)
        assert cw.ego_anchor is False
        st = cw.state_at(2)
        assert np.allclose(st["root_pos"], [0.3 * 2 + 1.0, 0.1 * 2, 0.78 + 0.01 * 2], atol=1e-9), \
            "world-anchored state_at should equal the clip-world base (identity placement)"

        # --- Ego re-anchor: current base snaps onto the robot (planar pos + heading). ---
        ce = _clip(path, ego=True)
        assert ce.ego_anchor is True and ce.ego_anchor_every == 3
        step = 2
        base_pos_cf = ce.y[step][ce._base_pos]
        robot_pos = np.array([5.0, -2.0, 0.93], dtype=np.float64)        # arbitrary world pos
        robot_quat = _quat_from_euler(-0.1, 0.2, 1.1)                    # tilted + yawed
        ce.reanchor_ego(step, robot_pos, robot_quat)
        cur = ce.state_at(step)

        assert np.allclose(cur["root_pos"][:2], robot_pos[:2], atol=1e-9), \
            "ego current base xy must match the robot root xy"
        assert abs(_wrap(_yaw_angle(cur["root_quat"]) - _yaw_angle(robot_quat))) < 1e-9, \
            "ego current base heading must match the robot yaw"
        assert abs(cur["root_pos"][2] - robot_pos[2]) < 1e-9, \
            "ego must snap the current base height (z) onto the robot root z (matches the FM anchor)"
        # The clip's natural base height (~0.80) must NOT survive -- it is overwritten by robot z.
        assert abs(cur["root_pos"][2] - base_pos_cf[2]) > 1e-3

        # --- Future window recedes from the re-anchored placement (same transform). ---
        # Offset 0 is the current step and heads the window; [1, 2] are the look-ahead.
        ref = ce.reference_at(step, [0, 1, 2])
        assert np.allclose(ref["window"][0]["root_pos"][:2], robot_pos[:2], atol=1e-9)
        # Future step's planar offset from current == placement-rotated clip delta.
        yaw = _yaw_angle(robot_quat) - _yaw_angle(ce.y[step][ce._base_ori])
        c, sn = math.cos(yaw), math.sin(yaw)
        d_cf = ce.y[step + 1][ce._base_pos][:2] - base_pos_cf[:2]
        exp = np.array([c * d_cf[0] - sn * d_cf[1], sn * d_cf[0] + c * d_cf[1]])
        got = ref["window"][1]["root_pos"][:2] - ref["window"][0]["root_pos"][:2]
        assert np.allclose(got, exp, atol=1e-9), "future must recede via the same placement"

        # Ego differs from world-anchored at the same step (sanity).
        assert not np.allclose(cur["root_pos"][:2], st["root_pos"][:2])

    print("OK: clip ego re-anchor matches ego_placement (planar pos + heading + robot z); "
          "world-anchored pass-through preserved")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    main()
