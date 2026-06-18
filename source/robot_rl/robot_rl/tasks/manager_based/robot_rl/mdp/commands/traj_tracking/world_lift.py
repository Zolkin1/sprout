# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared world-frame lift for anchor-relative trajectory outputs.

Trajectory outputs (positions, orientations, velocities) are stored
yaw-aligned relative to a stance/contact anchor frame (see
:meth:`BaseTrajectoryCommand.compute_measured_output`).  This module holds
the single inverse transform that lifts those anchor-relative quantities
back into the world frame, given the anchor's world pose.

It is the exact inverse of the forward yaw-alignment ``_align_yaw`` /
``_align_yaw_batched`` in :mod:`base_trajectory_cmd`::

    forward:  v_rel = R_yaw(q_a)^{-1} @ v_w     (world  -> anchor-relative)
    inverse:  v_w   = R_yaw(q_a)      @ v_rel   (anchor -> world)

Both the offline clip generator and the runtime command import these
helpers so trajectory generation and runtime tracking use a provably
identical lift.  Quaternions are xyzw to match Isaac Lab's math utils.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat


def anchor_inv_yaw(anchor_quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation that lifts anchor-relative vectors into the world.

    This is the inverse of the ``yaw_quat(quat_inv(anchor_quat))`` rotation
    that ``_align_yaw`` applies in the forward (world -> anchor) direction.

    Args:
        anchor_quat: ``[..., 4]`` anchor-frame world quaternions (xyzw).

    Returns:
        ``[..., 4]`` lift quaternions (xyzw).
    """
    return quat_inv(yaw_quat(quat_inv(anchor_quat)))


# ----------------------------------------------------------------------
# Single-frame lift: anchor [N, .] and quantities [N, .] share leading dim.
# ----------------------------------------------------------------------

def lift_position(anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_rel: torch.Tensor) -> torch.Tensor:
    """Lift an anchor-relative position to the world frame.

    Args:
        anchor_pos: ``[N, 3]`` anchor world positions.
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        pos_rel: ``[N, 3]`` anchor-relative positions.

    Returns:
        ``[N, 3]`` world positions.
    """
    return anchor_pos + quat_apply(anchor_inv_yaw(anchor_quat), pos_rel)


def lift_orientation(anchor_quat: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Lift an anchor-relative orientation to the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        quat_rel: ``[N, 4]`` anchor-relative quaternions (xyzw).

    Returns:
        ``[N, 4]`` world quaternions (xyzw).
    """
    return quat_mul(anchor_inv_yaw(anchor_quat), quat_rel)


def lift_vector(anchor_quat: torch.Tensor, vec_rel: torch.Tensor) -> torch.Tensor:
    """Rotate an anchor-relative vector (lin/ang velocity) into the world.

    No translation is applied -- velocities only rotate.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        vec_rel: ``[N, 3]`` anchor-relative vectors.

    Returns:
        ``[N, 3]`` world vectors.
    """
    return quat_apply(anchor_inv_yaw(anchor_quat), vec_rel)


# ----------------------------------------------------------------------
# Batched lift: one anchor [N, .] applied to B frames per env [N, B, .].
# Mirrors the expand/reshape pattern of ``_align_yaw_batched``.
# ----------------------------------------------------------------------

def lift_position_batched(anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_rel: torch.Tensor) -> torch.Tensor:
    """Lift ``B`` anchor-relative positions per env to the world frame.

    Args:
        anchor_pos: ``[N, 3]`` anchor world positions.
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        pos_rel: ``[N, B, 3]`` anchor-relative positions.

    Returns:
        ``[N, B, 3]`` world positions.
    """
    n, b, _ = pos_rel.shape
    inv_yaw = anchor_inv_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    rotated = quat_apply(inv_yaw, pos_rel.reshape(n * b, 3)).reshape(n, b, 3)
    return anchor_pos.unsqueeze(1) + rotated


def lift_orientation_batched(anchor_quat: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Lift ``B`` anchor-relative orientations per env to the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        quat_rel: ``[N, B, 4]`` anchor-relative quaternions (xyzw).

    Returns:
        ``[N, B, 4]`` world quaternions (xyzw).
    """
    n, b, _ = quat_rel.shape
    inv_yaw = anchor_inv_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_mul(inv_yaw, quat_rel.reshape(n * b, 4)).reshape(n, b, 4)


def lift_vector_batched(anchor_quat: torch.Tensor, vec_rel: torch.Tensor) -> torch.Tensor:
    """Rotate ``B`` anchor-relative vectors per env into the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        vec_rel: ``[N, B, 3]`` anchor-relative vectors.

    Returns:
        ``[N, B, 3]`` world vectors.
    """
    n, b, _ = vec_rel.shape
    inv_yaw = anchor_inv_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_apply(inv_yaw, vec_rel.reshape(n * b, 3)).reshape(n, b, 3)


# ----------------------------------------------------------------------
# Inverse (world -> anchor-relative), batched.  These are the forward
# yaw-alignment (the inverse of the lift), used to re-express one frame's
# world quantities in a different anchor frame (e.g. seam cross-fade).
# ----------------------------------------------------------------------

def _fwd_yaw(anchor_quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation taking world vectors into the anchor frame."""
    return yaw_quat(quat_inv(anchor_quat))


def unlift_position_batched(anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
    """World positions ``[N, B, 3]`` -> anchor-relative ``[N, B, 3]``."""
    n, b, _ = pos_w.shape
    fwd = _fwd_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    rel = (pos_w - anchor_pos.unsqueeze(1)).reshape(n * b, 3)
    return quat_apply(fwd, rel).reshape(n, b, 3)


def unlift_orientation_batched(anchor_quat: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """World orientations ``[N, B, 4]`` (xyzw) -> anchor-relative ``[N, B, 4]``."""
    n, b, _ = quat_w.shape
    fwd = _fwd_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_mul(fwd, quat_w.reshape(n * b, 4)).reshape(n, b, 4)


def unlift_vector_batched(anchor_quat: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    """World vectors ``[N, B, 3]`` -> anchor-relative ``[N, B, 3]`` (rotation only)."""
    n, b, _ = vec_w.shape
    fwd = _fwd_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_apply(fwd, vec_w.reshape(n * b, 3)).reshape(n, b, 3)


# ----------------------------------------------------------------------
# Convenience: full base-frame world state from anchor-relative outputs.
# ----------------------------------------------------------------------

def lift_base_world_state(
    y_des: torch.Tensor,
    dy_des: torch.Tensor,
    anchor_pose: torch.Tensor,
    idx: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Lift the base-frame channels of desired outputs to a world state vector.

    Replicates the per-base-frame lift in
    :meth:`BaseTrajectoryCommand.get_reference_world_state`.  The returned
    layout matches the raw robot ``state`` recorded by
    ``scripts/rsl_rl/record_data.py``::

        [root_pos_w(3), root_quat_w(4, xyzw), root_lin_vel_w(3),
         root_ang_vel_w(3), joint_pos(nj), joint_vel(nj)]

    Args:
        y_des: ``[N, P]`` desired position outputs (anchor-relative).
        dy_des: ``[N, V]`` desired velocity outputs (anchor-relative).
        anchor_pose: ``[N, 7]`` anchor world pose ``[pos(3), quat(4) xyzw]``.
        idx: index dict (from ``_reference_world_indices``) with long-tensor
            keys ``pos, ori, lin_vel, ang_vel, joint_pos, joint_vel``.

    Returns:
        ``[N, 3 + 4 + 3 + 3 + 2*nj]`` world-frame reference state.
    """
    anchor_pos = anchor_pose[:, :3]
    inv_yaw = anchor_inv_yaw(anchor_pose[:, 3:7])

    base_pos_w = anchor_pos + quat_apply(inv_yaw, y_des[:, idx["pos"]])
    base_quat_w = quat_mul(inv_yaw, y_des[:, idx["ori"]])
    base_lin_w = quat_apply(inv_yaw, dy_des[:, idx["lin_vel"]])
    base_ang_w = quat_apply(inv_yaw, dy_des[:, idx["ang_vel"]])

    joint_pos = y_des[:, idx["joint_pos"]]
    joint_vel = dy_des[:, idx["joint_vel"]]

    return torch.cat(
        [base_pos_w, base_quat_w, base_lin_w, base_ang_w, joint_pos, joint_vel], dim=-1
    )
