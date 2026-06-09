# Copyright 2026 Zachary Olkin. All rights reserved.

"""Geometry transforms for the flow-matching state representation.

This module is the single source of truth for two conventions:

1. The **relative-frame state vector** layout (per timestep), order::

       [joint_pos(nj), joint_vel(nj), p_rel(3), v_rel(3), ori6d(6), w_rel(3)]

   where root quantities are expressed relative to a reference root frame.

2. The **6D continuous rotation** representation (Zhou et al., 2019): the first
   two *columns* of the rotation matrix. The exact column ordering is reused on
   both the forward and inverse paths so the round-trip is exact.

All quaternions are ``(x, y, z, w)``, matching ``robot.data`` and
``isaaclab.utils.math`` in this repository.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    subtract_frame_transforms,
)


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors to unit length along the last dimension."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the 6D representation (first two columns).

    Args:
        matrix: Rotation matrices, shape ``(..., 3, 3)``.

    Returns:
        6D rotation representation, shape ``(..., 6)`` = concat(column 0, column 1).
    """
    col0 = matrix[..., :, 0]
    col1 = matrix[..., :, 1]
    return torch.cat([col0, col1], dim=-1)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Recover a rotation matrix from the 6D representation via Gram-Schmidt.

    Robust to non-orthonormal inputs (e.g. raw network outputs): the result is
    always a valid rotation matrix.

    Args:
        d6: 6D rotation representation, shape ``(..., 6)``.

    Returns:
        Orthonormal rotation matrices, shape ``(..., 3, 3)`` with the recovered
        vectors as columns.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def quat_to_rot6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``(x, y, z, w)`` to the 6D rotation representation."""
    return matrix_to_rot6d(matrix_from_quat(quat_xyzw))


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    """Convert the 6D rotation representation to quaternions ``(x, y, z, w)``."""
    return quat_from_matrix(rot6d_to_matrix(d6))


def target_state_dim(num_joints: int) -> int:
    """Width of the relative-frame state vector for a robot with ``num_joints``."""
    return 2 * num_joints + 3 + 3 + 6 + 3


def orientation_slice(num_joints: int) -> tuple[int, int]:
    """Return ``(start, end)`` of the 6D orientation block within the state vector."""
    start = 2 * num_joints + 3 + 3
    return start, start + 6


def relative_state(
    p_w: torch.Tensor,
    q_w: torch.Tensor,
    v_w: torch.Tensor,
    w_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
) -> torch.Tensor:
    """Build the relative-frame state vector for a batch of steps.

    Every input is ``(M, ·)`` where ``M`` is the number of steps. Root pose is
    expressed relative to the (per-step) reference frame ``(p_ref, q_ref)``;
    root velocities are rotated into that frame by a **pure rotation** (the
    reference frame is a stationary world snapshot, not a moving body frame, so
    no ``w x r`` term is added).

    Args:
        p_w: Root positions (world), shape ``(M, 3)``.
        q_w: Root quaternions (world, xyzw), shape ``(M, 4)``.
        v_w: Root linear velocities (world), shape ``(M, 3)``.
        w_w: Root angular velocities (world), shape ``(M, 3)``.
        joint_pos: Raw joint positions, shape ``(M, nj)``.
        joint_vel: Raw joint velocities, shape ``(M, nj)``.
        p_ref: Reference position (world), shape ``(M, 3)``.
        q_ref: Reference quaternion (world, xyzw), shape ``(M, 4)``.

    Returns:
        Relative-state tensor, shape ``(M, 2*nj + 15)``.
    """
    p_rel, q_rel = subtract_frame_transforms(p_ref, q_ref, p_w, q_w)
    ori6d = matrix_to_rot6d(matrix_from_quat(q_rel))
    v_rel = quat_apply_inverse(q_ref, v_w)
    w_rel = quat_apply_inverse(q_ref, w_w)
    return torch.cat([joint_pos, joint_vel, p_rel, v_rel, ori6d, w_rel], dim=-1)


def relative_state_from_window(
    p_w: torch.Tensor,
    q_w: torch.Tensor,
    v_w: torch.Tensor,
    w_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ref_index: int,
) -> torch.Tensor:
    """Single-window convenience wrapper around :func:`relative_state`.

    Args:
        p_w, q_w, v_w, w_w, joint_pos, joint_vel: Per-step quantities, shape ``(W, ·)``.
        ref_index: Index of the reference step within the window.

    Returns:
        Relative-state tensor, shape ``(W, 2*nj + 15)``.
    """
    w = p_w.shape[0]
    p_ref = p_w[ref_index : ref_index + 1].expand(w, -1)
    q_ref = q_w[ref_index : ref_index + 1].expand(w, -1)
    return relative_state(p_w, q_w, v_w, w_w, joint_pos, joint_vel, p_ref, q_ref)


def state_to_absolute(
    rel_state: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    num_joints: int,
) -> dict[str, torch.Tensor]:
    """Invert :func:`relative_state` back to world-frame quantities.

    Args:
        rel_state: Relative-state tensor, shape ``(M, 2*nj + 15)``.
        p_ref: Reference position (world), shape ``(M, 3)``.
        q_ref: Reference quaternion (world, xyzw), shape ``(M, 4)``.
        num_joints: Number of joints ``nj``.

    Returns:
        Dict with ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w``,
        ``root_lin_vel_w``, ``root_ang_vel_w``.
    """
    nj = num_joints
    joint_pos = rel_state[..., 0:nj]
    joint_vel = rel_state[..., nj : 2 * nj]
    p_rel = rel_state[..., 2 * nj : 2 * nj + 3]
    v_rel = rel_state[..., 2 * nj + 3 : 2 * nj + 6]
    ori6d = rel_state[..., 2 * nj + 6 : 2 * nj + 12]
    w_rel = rel_state[..., 2 * nj + 12 : 2 * nj + 15]
    q_rel = rot6d_to_quat(ori6d)
    p_w, q_w = combine_frame_transforms(p_ref, q_ref, p_rel, q_rel)
    v_w = quat_apply(q_ref, v_rel)
    w_w = quat_apply(q_ref, w_rel)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos_w": p_w,
        "root_quat_w": q_w,
        "root_lin_vel_w": v_w,
        "root_ang_vel_w": w_w,
    }


def ori6d_orthonormality_penalty(d6: torch.Tensor) -> torch.Tensor:
    """Penalty encouraging the 6D rep to encode two orthonormal vectors.

    Args:
        d6: 6D rotation representation, shape ``(..., 6)``.

    Returns:
        Scalar penalty: squared dot product plus squared unit-norm deviations.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    dot = (a1 * a2).sum(dim=-1)
    n1 = a1.norm(dim=-1)
    n2 = a2.norm(dim=-1)
    return dot.pow(2).mean() + (n1 - 1.0).pow(2).mean() + (n2 - 1.0).pow(2).mean()
