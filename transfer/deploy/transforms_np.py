# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-numpy flow-matching state transforms (no torch, no isaaclab).

Numpy port of ``robot_rl.flow_matching.transforms`` for the torch-free deployment
runtime, parity-tested against it directly in
``transfer/deploy/tests/test_deploy_parity.py``.
Single source of truth for the relative-frame state layout (per timestep)::

    [joint_pos(nj), joint_vel(nj),
     p_rel(3), v_rel(3), ori6d(6), w_rel(3),                    # base (root) block
     <per extra frame: pos_rel(3), linvel_rel(3), ori6d(6), angvel_rel(3)> ...]

and the 6D continuous-rotation representation (first two rotation-matrix columns).
All quaternions are ``(x, y, z, w)``. Parity-tested against the torch version.
"""

from __future__ import annotations

import numpy as np

from .quat_math_np import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    subtract_frame_transforms,
    yaw_quat,
)

FRAME_BLOCK_DIM = 3 + 3 + 6 + 3  # [pos(3), lin_vel(3), ori6d(6), ang_vel(3)] = 15


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize vectors to unit length along the last axis."""
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), a_min=eps, a_max=None)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """6D rep ``(..., 6)`` = concat(column 0, column 1) of a rotation matrix."""
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Recover a rotation matrix ``(..., 3, 3)`` from the 6D rep via Gram-Schmidt."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def quat_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to the 6D rotation representation."""
    return matrix_to_rot6d(matrix_from_quat(quat_xyzw))


def rot6d_to_quat(d6: np.ndarray) -> np.ndarray:
    """Convert the 6D rotation representation to xyzw quaternions."""
    return quat_from_matrix(rot6d_to_matrix(d6))


def target_state_dim(num_joints: int, num_extra_frames: int = 0) -> int:
    """Width of the relative-frame state vector ``2*nj + 15*(1 + K)``."""
    return 2 * num_joints + FRAME_BLOCK_DIM * (1 + num_extra_frames)


def frame_block_start(num_joints: int, frame_index: int) -> int:
    """Start offset of a frame block within the state vector (0 = root/base)."""
    return 2 * num_joints + FRAME_BLOCK_DIM * frame_index


def height_above_ground(p_ref: np.ndarray, ground_z: float = 0.0) -> np.ndarray:
    """Root height above the ground at the anchor, the FM's height token.

    The xyz+yaw relative frame removes the anchor's absolute z, so this scalar
    (``p_ref.z - ground_z``) restores ground clearance -- a stand-in for a future
    heightmap. On flat ground ``ground_z = 0`` and it is just the root world z.
    """
    return p_ref[..., 2:3] - ground_z


def _frame_relative_block(p_ref, q_ref, p_w, q_w, v_w, w_w) -> np.ndarray:
    """Build one ``[pos(3), lin_vel(3), ori6d(6), ang_vel(3)]`` block.

    The reference frame is an **xyz + yaw** ground frame: the anchor's position
    ``p_ref`` (incl. z) and yaw are removed, roll/pitch are kept (only
    ``yaw_quat(q_ref)`` is factored out), so ``ori6d`` retains the body's full tilt.
    Absolute ground clearance is supplied separately by the height token.
    """
    q_yaw = yaw_quat(q_ref)
    p_rel, q_rel = subtract_frame_transforms(p_ref, q_yaw, p_w, q_w)
    ori6d = matrix_to_rot6d(matrix_from_quat(q_rel))
    v_rel = quat_apply_inverse(q_yaw, v_w)
    w_rel = quat_apply_inverse(q_yaw, w_w)
    return np.concatenate([p_rel, v_rel, ori6d, w_rel], axis=-1)


def relative_state(
    p_w, q_w, v_w, w_w, joint_pos, joint_vel, p_ref, q_ref,
    frame_pos_w=None, frame_quat_w=None, frame_lin_vel_w=None, frame_ang_vel_w=None,
) -> np.ndarray:
    """Build the relative-frame state vector for a batch of steps ``(M, ·)``.

    The root pose (and each extra frame) is expressed relative to the per-step
    reference ``(p_ref, q_ref)``; velocities are pure-rotated into that frame.
    Returns ``(M, 2*nj + 15*(1 + K))``.
    """
    root_block = _frame_relative_block(p_ref, q_ref, p_w, q_w, v_w, w_w)
    parts = [joint_pos, joint_vel, root_block]
    if frame_pos_w is not None:
        m, k = frame_pos_w.shape[0], frame_pos_w.shape[1]
        pr = np.broadcast_to(p_ref[:, None, :], (m, k, 3)).reshape(m * k, 3)
        qr = np.broadcast_to(q_ref[:, None, :], (m, k, 4)).reshape(m * k, 4)
        block = _frame_relative_block(
            pr, qr,
            frame_pos_w.reshape(m * k, 3), frame_quat_w.reshape(m * k, 4),
            frame_lin_vel_w.reshape(m * k, 3), frame_ang_vel_w.reshape(m * k, 3),
        ).reshape(m, k * FRAME_BLOCK_DIM)
        parts.append(block)
    return np.concatenate(parts, axis=-1)


def _block_to_absolute(block, p_ref, q_ref):
    """Invert one ``[pos(3), lin_vel(3), ori6d(6), ang_vel(3)]`` block to world.

    Uses the same **xyz + yaw** anchor as :func:`_frame_relative_block`
    (``yaw_quat(q_ref)``); the round-trip is exact (``yaw_quat`` is idempotent).
    """
    q_yaw = yaw_quat(q_ref)
    p_rel = block[..., 0:3]
    v_rel = block[..., 3:6]
    ori6d = block[..., 6:12]
    w_rel = block[..., 12:15]
    q_rel = rot6d_to_quat(ori6d)
    p_w, q_w = combine_frame_transforms(p_ref, q_yaw, p_rel, q_rel)
    v_w = quat_apply(q_yaw, v_rel)
    w_w = quat_apply(q_yaw, w_rel)
    return p_w, q_w, v_w, w_w


def state_to_absolute(rel_state, p_ref, q_ref, num_joints, num_extra_frames=0) -> dict:
    """Invert :func:`relative_state` back to world-frame quantities.

    Returns a dict with ``joint_pos``, ``joint_vel``, ``root_pos_w``,
    ``root_quat_w``, ``root_link_lin_vel_w``, ``root_link_ang_vel_w`` and, when
    ``num_extra_frames > 0``, ``frame_pos_w``/``frame_quat_w``/``frame_lin_vel_w``/
    ``frame_ang_vel_w`` each ``(..., K, ·)``.
    """
    nj = num_joints
    joint_pos = rel_state[..., 0:nj]
    joint_vel = rel_state[..., nj : 2 * nj]
    base = frame_block_start(nj, 0)
    p_w, q_w, v_w, w_w = _block_to_absolute(rel_state[..., base : base + FRAME_BLOCK_DIM], p_ref, q_ref)
    out = {
        "joint_pos": joint_pos, "joint_vel": joint_vel,
        "root_pos_w": p_w, "root_quat_w": q_w,
        "root_link_lin_vel_w": v_w, "root_link_ang_vel_w": w_w,
    }
    if num_extra_frames > 0:
        fp, fq, fv, fw = [], [], [], []
        for k in range(num_extra_frames):
            b = frame_block_start(nj, 1 + k)
            kp, kq, kv, kw = _block_to_absolute(rel_state[..., b : b + FRAME_BLOCK_DIM], p_ref, q_ref)
            fp.append(kp); fq.append(kq); fv.append(kv); fw.append(kw)
        out["frame_pos_w"] = np.stack(fp, axis=-2)
        out["frame_quat_w"] = np.stack(fq, axis=-2)
        out["frame_lin_vel_w"] = np.stack(fv, axis=-2)
        out["frame_ang_vel_w"] = np.stack(fw, axis=-2)
    return out
