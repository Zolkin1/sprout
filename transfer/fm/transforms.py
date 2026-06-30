# Copyright 2026 Zachary Olkin. All rights reserved.

"""Geometry transforms for the flow-matching state representation.

This module is the single source of truth for two conventions:

1. The **relative-frame state vector** layout (per timestep), order::

       [joint_pos(nj), joint_vel(nj),
        p_rel(3), v_rel(3), ori6d(6), w_rel(3),                 # base (root) block
        <per extra frame: pos_rel(3), linvel_rel(3), ori6d(6), angvel_rel(3)> ...]

   where root *and* every extra body frame are expressed relative to a single
   reference root frame. The base block and each extra-frame block share the
   identical ``[pos(3), lin_vel(3), ori6d(6), ang_vel(3)]`` layout, so an extra
   frame is just "another base". ``num_extra_frames=0`` reproduces the original
   base-and-joints-only vector (``2*nj + 15``).

2. The **6D continuous rotation** representation (Zhou et al., 2019): the first
   two *columns* of the rotation matrix. The exact column ordering is reused on
   both the forward and inverse paths so the round-trip is exact.

All quaternions are ``(x, y, z, w)``, matching ``robot.data`` and the repo's
``isaaclab.utils.math``.  This is the vendored, isaaclab-free copy used by the
sim2sim / hardware deployment code: the frame-transform helpers come from the
local :mod:`quat_math` (parity-tested against ``isaaclab.utils.math``).
"""

from __future__ import annotations

import torch

from .quat_math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    subtract_frame_transforms,
    yaw_quat,
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


def so3_log(r: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle) of rotation matrices ``(..., 3, 3)``.

    The SO(3) log map. Numerically stable near identity -- uses ``atan2`` (not
    ``arccos``, whose gradient is singular at ``cos = 1``) and a finite small-angle
    limit -- so a finite-difference angular velocity between nearly-equal consecutive
    rotations (``so3_log(R[t+1] @ R[t]^T)``) has well-behaved values and gradients.

    Args:
        r: Rotation matrices, shape ``(..., 3, 3)``.

    Returns:
        Rotation vectors ``(..., 3)``.
    """
    v_raw = torch.stack(
        [r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]],
        dim=-1,
    )  # = 2 * sin(theta) * axis
    s = 0.5 * torch.linalg.norm(v_raw, dim=-1)                    # sin(theta) >= 0
    c = 0.5 * (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1.0)  # cos(theta)
    theta = torch.atan2(s, c)                                     # [0, pi]
    # rotation vector = theta * axis = v_raw * theta / (2 sin theta); -> v_raw/2 as theta->0.
    factor = torch.where(s < 1e-6, torch.full_like(s, 0.5), theta / (2.0 * s.clamp_min(1e-12)))
    return v_raw * factor.unsqueeze(-1)


# Per-frame block width: [pos(3), lin_vel(3), ori6d(6), ang_vel(3)]. The base
# (root) block and every extra body frame share this layout.
FRAME_BLOCK_DIM = 3 + 3 + 6 + 3  # = 15


def target_state_dim(num_joints: int, num_extra_frames: int = 0) -> int:
    """Width of the relative-frame state vector.

    Args:
        num_joints: Number of joints ``nj``.
        num_extra_frames: Number of extra body frames beyond the root (each adds
            a :data:`FRAME_BLOCK_DIM`-wide block).

    Returns:
        ``2*nj + FRAME_BLOCK_DIM * (1 + num_extra_frames)``.
    """
    return 2 * num_joints + FRAME_BLOCK_DIM * (1 + num_extra_frames)


def frame_block_start(num_joints: int, frame_index: int) -> int:
    """Start offset of a frame block within the state vector.

    Args:
        num_joints: Number of joints ``nj``.
        frame_index: ``0`` for the root/base block, ``1..K`` for extra frames.

    Returns:
        The start column of that frame's ``[pos, lin_vel, ori6d, ang_vel]`` block.
    """
    return 2 * num_joints + FRAME_BLOCK_DIM * frame_index


def height_above_ground(p_ref: torch.Tensor, ground_z: float = 0.0) -> torch.Tensor:
    """Root height above the ground at the anchor, the FM's height token.

    The xyz+yaw relative frame removes the anchor's absolute z, so this scalar
    (``p_ref.z - ground_z``) restores ground clearance -- a stand-in for a future
    heightmap. On flat ground ``ground_z = 0`` and it is just the root world z.

    Args:
        p_ref: Anchor (reference) positions (world), shape ``(..., 3)``.
        ground_z: World floor height under the anchor (flat ground -> 0.0).

    Returns:
        Height above ground, shape ``(..., 1)``.
    """
    return p_ref[..., 2:3] - ground_z


def orientation_slice(num_joints: int, frame_index: int = 0) -> tuple[int, int]:
    """Return ``(start, end)`` of a frame's 6D orientation block.

    Args:
        num_joints: Number of joints ``nj``.
        frame_index: ``0`` for the root/base block, ``1..K`` for extra frames.
    """
    start = frame_block_start(num_joints, frame_index) + 3 + 3
    return start, start + 6


def orientation_slices(num_joints: int, num_extra_frames: int = 0) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` 6D-orientation slice of every frame (base first)."""
    return [orientation_slice(num_joints, i) for i in range(1 + num_extra_frames)]


def _frame_relative_block(
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    p_w: torch.Tensor,
    q_w: torch.Tensor,
    v_w: torch.Tensor,
    w_w: torch.Tensor,
) -> torch.Tensor:
    """Build one ``[pos(3), lin_vel(3), ori6d(6), ang_vel(3)]`` block (any leading dims).

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
    return torch.cat([p_rel, v_rel, ori6d, w_rel], dim=-1)


def relative_state(
    p_w: torch.Tensor,
    q_w: torch.Tensor,
    v_w: torch.Tensor,
    w_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    frame_pos_w: torch.Tensor | None = None,
    frame_quat_w: torch.Tensor | None = None,
    frame_lin_vel_w: torch.Tensor | None = None,
    frame_ang_vel_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the relative-frame state vector for a batch of steps.

    Every input is ``(M, ·)`` where ``M`` is the number of steps. The root pose
    (and each extra frame's pose) is expressed relative to the (per-step)
    reference frame ``(p_ref, q_ref)``; velocities are rotated into that frame by
    a **pure rotation** (the reference frame is a stationary world snapshot, not a
    moving body frame, so no ``w x r`` term is added). Each extra frame appends a
    block identical in layout to the root block.

    Args:
        p_w: Root positions (world), shape ``(M, 3)``.
        q_w: Root quaternions (world, xyzw), shape ``(M, 4)``.
        v_w: Root linear velocities (world), shape ``(M, 3)``.
        w_w: Root angular velocities (world), shape ``(M, 3)``.
        joint_pos: Raw joint positions, shape ``(M, nj)``.
        joint_vel: Raw joint velocities, shape ``(M, nj)``.
        p_ref: Reference position (world), shape ``(M, 3)``.
        q_ref: Reference quaternion (world, xyzw), shape ``(M, 4)``.
        frame_pos_w: Extra-frame positions (world), shape ``(M, K, 3)`` or ``None``.
        frame_quat_w: Extra-frame quaternions (world, xyzw), shape ``(M, K, 4)``.
        frame_lin_vel_w: Extra-frame linear velocities (world), shape ``(M, K, 3)``.
        frame_ang_vel_w: Extra-frame angular velocities (world), shape ``(M, K, 3)``.

    Returns:
        Relative-state tensor, shape ``(M, 2*nj + 15*(1 + K))``.
    """
    root_block = _frame_relative_block(p_ref, q_ref, p_w, q_w, v_w, w_w)
    parts = [joint_pos, joint_vel, root_block]
    if frame_pos_w is not None:
        m, k = frame_pos_w.shape[0], frame_pos_w.shape[1]
        pr = p_ref.unsqueeze(1).expand(m, k, 3).reshape(m * k, 3)
        qr = q_ref.unsqueeze(1).expand(m, k, 4).reshape(m * k, 4)
        block = _frame_relative_block(
            pr, qr,
            frame_pos_w.reshape(m * k, 3), frame_quat_w.reshape(m * k, 4),
            frame_lin_vel_w.reshape(m * k, 3), frame_ang_vel_w.reshape(m * k, 3),
        ).reshape(m, k * FRAME_BLOCK_DIM)
        parts.append(block)
    return torch.cat(parts, dim=-1)


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


def _block_to_absolute(
    block: torch.Tensor, p_ref: torch.Tensor, q_ref: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def state_to_absolute(
    rel_state: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    num_joints: int,
    num_extra_frames: int = 0,
) -> dict[str, torch.Tensor]:
    """Invert :func:`relative_state` back to world-frame quantities.

    Args:
        rel_state: Relative-state tensor, shape ``(M, 2*nj + 15*(1 + K))``.
        p_ref: Reference position (world), shape ``(M, 3)``.
        q_ref: Reference quaternion (world, xyzw), shape ``(M, 4)``.
        num_joints: Number of joints ``nj``.
        num_extra_frames: Number of extra body frames ``K`` to decode.

    Returns:
        Dict with ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w``,
        ``root_link_lin_vel_w``, ``root_link_ang_vel_w``. When ``K > 0`` also includes
        ``frame_pos_w``/``frame_quat_w``/``frame_lin_vel_w``/``frame_ang_vel_w``,
        each shape ``(..., K, ·)`` in the frame order used by :func:`relative_state`.
    """
    nj = num_joints
    joint_pos = rel_state[..., 0:nj]
    joint_vel = rel_state[..., nj : 2 * nj]
    base = frame_block_start(nj, 0)
    p_w, q_w, v_w, w_w = _block_to_absolute(
        rel_state[..., base : base + FRAME_BLOCK_DIM], p_ref, q_ref
    )
    out = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos_w": p_w,
        "root_quat_w": q_w,
        "root_link_lin_vel_w": v_w,
        "root_link_ang_vel_w": w_w,
    }
    if num_extra_frames > 0:
        fp, fq, fv, fw = [], [], [], []
        for k in range(num_extra_frames):
            b = frame_block_start(nj, 1 + k)
            kp, kq, kv, kw = _block_to_absolute(
                rel_state[..., b : b + FRAME_BLOCK_DIM], p_ref, q_ref
            )
            fp.append(kp); fq.append(kq); fv.append(kv); fw.append(kw)
        out["frame_pos_w"] = torch.stack(fp, dim=-2)
        out["frame_quat_w"] = torch.stack(fq, dim=-2)
        out["frame_lin_vel_w"] = torch.stack(fv, dim=-2)
        out["frame_ang_vel_w"] = torch.stack(fw, dim=-2)
    return out


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
