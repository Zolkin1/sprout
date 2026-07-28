# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-numpy flow-matching state transforms (no torch, no isaaclab).

Numpy port of ``transfer/fm/transforms.py`` for the torch-free deployment runtime.
Single source of truth for the relative-frame state layout (per timestep)::

    [joint_pos(nj),
     p_rel(3), ori6d(6),                                        # base (root) block
     <per extra frame: pos_rel(3), ori6d(6)> ...]

and the 6D continuous-rotation representation (first two rotation-matrix columns).
The state is **position-only**: velocities are reconstructed by finite-differencing
the predicted position/orientation sequence (see ``fm_session.py``), never stored.
All quaternions are ``(x, y, z, w)``. Parity-tested against the torch version.
"""

from __future__ import annotations

import numpy as np

from .quat_math_np import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
    subtract_frame_transforms,
)

FRAME_BLOCK_DIM = 3 + 6  # [pos(3), ori6d(6)] = 9


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


def so3_log(r: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of rotation matrices ``(..., 3, 3)`` via the SO(3) log.

    Numpy mirror of the torch ``transforms.so3_log`` (atan2-based, finite small-angle
    limit), used to finite-difference an angular velocity from a quaternion sequence.
    """
    v_raw = np.stack(
        [r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]],
        axis=-1,
    )  # = 2 * sin(theta) * axis
    s = 0.5 * np.linalg.norm(v_raw, axis=-1)
    c = 0.5 * (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1.0)
    theta = np.arctan2(s, c)
    factor = np.where(s < 1e-6, 0.5, theta / (2.0 * np.clip(s, 1e-12, None)))
    return v_raw * factor[..., None]


def target_state_dim(num_joints: int, num_extra_frames: int = 0) -> int:
    """Width of the relative-frame state vector ``nj + 9*(1 + K)``."""
    return num_joints + FRAME_BLOCK_DIM * (1 + num_extra_frames)


def frame_block_start(num_joints: int, frame_index: int) -> int:
    """Start offset of a frame block within the state vector (0 = root/base)."""
    return num_joints + FRAME_BLOCK_DIM * frame_index


def _frame_relative_block(p_ref, q_ref, p_w, q_w) -> np.ndarray:
    """Build one ``[pos(3), ori6d(6)]`` block."""
    p_rel, q_rel = subtract_frame_transforms(p_ref, q_ref, p_w, q_w)
    ori6d = matrix_to_rot6d(matrix_from_quat(q_rel))
    return np.concatenate([p_rel, ori6d], axis=-1)


def relative_state(
    p_w, q_w, joint_pos, p_ref, q_ref,
    frame_pos_w=None, frame_quat_w=None,
) -> np.ndarray:
    """Build the position-only relative-frame state vector for a batch of steps ``(M, ·)``.

    The root pose (and each extra frame) is expressed relative to the per-step
    reference ``(p_ref, q_ref)``. Velocities are not represented (derived downstream
    by finite-differencing the position/orientation sequence). Returns
    ``(M, nj + 9*(1 + K))``.
    """
    root_block = _frame_relative_block(p_ref, q_ref, p_w, q_w)
    parts = [joint_pos, root_block]
    if frame_pos_w is not None:
        m, k = frame_pos_w.shape[0], frame_pos_w.shape[1]
        pr = np.broadcast_to(p_ref[:, None, :], (m, k, 3)).reshape(m * k, 3)
        qr = np.broadcast_to(q_ref[:, None, :], (m, k, 4)).reshape(m * k, 4)
        block = _frame_relative_block(
            pr, qr,
            frame_pos_w.reshape(m * k, 3), frame_quat_w.reshape(m * k, 4),
        ).reshape(m, k * FRAME_BLOCK_DIM)
        parts.append(block)
    return np.concatenate(parts, axis=-1)


def _block_to_absolute(block, p_ref, q_ref):
    """Invert one ``[pos(3), ori6d(6)]`` block to a world pose (position + orientation)."""
    p_rel = block[..., 0:3]
    ori6d = block[..., 3:9]
    q_rel = rot6d_to_quat(ori6d)
    p_w, q_w = combine_frame_transforms(p_ref, q_ref, p_rel, q_rel)
    return p_w, q_w


def state_to_absolute(rel_state, p_ref, q_ref, num_joints, num_extra_frames=0) -> dict:
    """Invert :func:`relative_state` back to world-frame positions + orientations.

    Velocities are NOT decoded here (position-only state); callers finite-difference
    the predicted position/orientation sequence. Returns a dict with ``joint_pos``,
    ``root_pos_w``, ``root_quat_w`` and, when ``num_extra_frames > 0``,
    ``frame_pos_w``/``frame_quat_w`` each ``(..., K, ·)``.
    """
    nj = num_joints
    joint_pos = rel_state[..., 0:nj]
    base = frame_block_start(nj, 0)
    p_w, q_w = _block_to_absolute(rel_state[..., base : base + FRAME_BLOCK_DIM], p_ref, q_ref)
    out = {
        "joint_pos": joint_pos,
        "root_pos_w": p_w, "root_quat_w": q_w,
    }
    if num_extra_frames > 0:
        fp, fq = [], []
        for k in range(num_extra_frames):
            b = frame_block_start(nj, 1 + k)
            kp, kq = _block_to_absolute(rel_state[..., b : b + FRAME_BLOCK_DIM], p_ref, q_ref)
            fp.append(kp); fq.append(kq)
        out["frame_pos_w"] = np.stack(fp, axis=-2)
        out["frame_quat_w"] = np.stack(fq, axis=-2)
    return out
