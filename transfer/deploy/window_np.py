# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-numpy port of the reference-window representation (no torch, no isaaclab).

Mirror of :mod:`robot_rl.flow_matching.window` for the torch-free deployment runtime,
parity-tested against it directly in ``transfer/deploy/tests/test_deploy_parity.py``.
See that module for what the blocks mean and why ``base_pose``'s two halves currently do
NOT point the same way; this file restates only the arithmetic.

Per node::

    [base_pose(6), base_vel(6), joint_pos(nj), joint_vel(nj), ee_pos(3 * K)]

Only the FORWARD map lives here. The deploy runtime needs it for the CLIP reference path
(tracker-only playback, where a world-frame reference is handed in); the generator predicts
window nodes directly, so nothing on the FM path transforms anything. The inverse
(``window_to_world``) is a simulation-side diagnostic and is deliberately not ported --
add it here only if something off-sim genuinely needs to draw a world-frame reference.

All quaternions are ``(x, y, z, w)``. Unlike the torch original, every helper here
broadcasts, so the anchor needs no explicit expansion.
"""

from __future__ import annotations

import numpy as np

from .quat_math_np import quat_apply_inverse, quat_box_minus


def window_blocks(num_joints: int, num_ee_frames: int) -> list[tuple[str, int]]:
    """Per-node block layout ``[(name, width), ...]`` in packing order.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.

    Returns:
        The same table :func:`robot_rl.flow_matching.window.window_blocks` returns.
    """
    return [
        ("base_pose", 6),
        ("base_vel", 6),
        ("joint_pos", num_joints),
        ("joint_vel", num_joints),
        ("ee_pos", 3 * num_ee_frames),
    ]


def window_dim(num_joints: int, num_ee_frames: int) -> int:
    """Width ``D`` of one window node: ``12 + 2 * nj + 3 * K``."""
    return sum(width for _, width in window_blocks(num_joints, num_ee_frames))


def window_block_slices(num_joints: int, num_ee_frames: int) -> dict[str, slice]:
    """Column slice of each block within one ``[..., D]`` window node."""
    out: dict[str, slice] = {}
    offset = 0
    for name, width in window_blocks(num_joints, num_ee_frames):
        out[name] = slice(offset, offset + width)
        offset += width
    return out


def world_to_window(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    ref_pos_w: np.ndarray,
    ref_quat_w: np.ndarray,
    ref_lin_w: np.ndarray,
    ref_ang_w: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    ee_pos_w: np.ndarray,
) -> np.ndarray:
    """Build window nodes from world-frame reference state and a measured anchor.

    Args:
        root_pos_w: ``[..., 3]`` anchor position (the robot's measured base), world frame.
            Broadcast against the node axis.
        root_quat_w: ``[..., 4]`` anchor orientation (xyzw), world frame.
        ref_pos_w: ``[..., K, 3]`` reference base position per node, world frame.
        ref_quat_w: ``[..., K, 4]`` reference base orientation per node (xyzw).
        ref_lin_w: ``[..., K, 3]`` reference base linear velocity per node, world frame.
        ref_ang_w: ``[..., K, 3]`` reference base angular velocity per node, world frame.
        joint_pos: ``[..., K, nj]`` reference joint positions, **robot joint order**.
        joint_vel: ``[..., K, nj]`` reference joint velocities, robot joint order.
        ee_pos_w: ``[..., K, E, 3]`` reference end-effector positions, world frame.

    Returns:
        ``[..., K, D]`` window nodes.
    """
    act_p = np.asarray(root_pos_w, dtype=np.float64)
    act_q = np.asarray(root_quat_w, dtype=np.float64)
    ref_p = np.asarray(ref_pos_w, dtype=np.float64)
    ref_q = np.asarray(ref_quat_w, dtype=np.float64)

    # base_pose, in the LEGACY mixed convention -- position ref-minus-actual in the robot
    # frame, orientation actual-minus-ref in the reference frame. See
    # ``robot_rl.flow_matching.window`` for why that is still the case and what flipping
    # it costs; this line and that one move together.
    pos_err_b = quat_apply_inverse(act_q, ref_p - act_p)
    ori_err_b = quat_apply_inverse(act_q, quat_box_minus(np.broadcast_to(act_q, ref_q.shape), ref_q))

    lin_b = quat_apply_inverse(act_q, np.asarray(ref_lin_w, dtype=np.float64))
    ang_b = quat_apply_inverse(act_q, np.asarray(ref_ang_w, dtype=np.float64))

    # ee positions relative to THAT node's own reference base.
    ee = np.asarray(ee_pos_w, dtype=np.float64)
    ee_rel = quat_apply_inverse(ref_q[..., None, :], ee - ref_p[..., None, :])
    ee_flat = ee_rel.reshape(*ee_rel.shape[:-2], -1)

    return np.concatenate(
        [pos_err_b, ori_err_b, lin_b, ang_b,
         np.asarray(joint_pos, dtype=np.float64), np.asarray(joint_vel, dtype=np.float64),
         ee_flat],
        axis=-1,
    )
