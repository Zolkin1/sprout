# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared yaw-only lift between a (yaw-aligned) anchor frame and the world.

Given an anchor's world pose, these helpers lift quantities expressed
yaw-aligned-relative to that anchor back into the world frame.  They are
the exact inverse of the forward yaw-alignment ``_align_yaw`` /
``_align_yaw_batched`` in :mod:`base_trajectory_cmd`::

    forward:  v_rel = R_yaw(q_a)^{-1} @ v_w     (world  -> anchor-relative)
    inverse:  v_w   = R_yaw(q_a)      @ v_rel   (anchor -> world)

Two callers use this lift:

  * the **offline clip generator** bakes its per-step stance anchor into the
    stored outputs via :func:`bake_outputs_to_world`, so a generated clip is a
    single clip-world frame (no runtime stance re-anchoring);
  * the **runtime command** applies a single per-env *placement* (yaw + xy of
    the clip in the sim world) to those clip-world outputs.

Quaternions are xyzw to match Isaac Lab's math utils.
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


# ----------------------------------------------------------------------
# Generation: bake a per-step anchor schedule into the stored outputs.
# ----------------------------------------------------------------------

def bake_outputs_to_world(
    y_des: torch.Tensor,
    dy_des: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    pos_output_names: list[str],
    vel_output_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift dense anchor-relative outputs into the clip-world frame, in place.

    Converts a clip's anchor-relative reference outputs into a single
    clip-world frame by lifting every output frame through its own per-step
    anchor.  After baking, the stored clip needs no runtime stance
    re-anchoring -- only a per-env placement.  Uses the exact same lift the
    runtime/placement applies (:func:`lift_position` / :func:`lift_orientation`
    / :func:`lift_vector`).

    Per output frame ``f``: position channels (``{f}:pos_{x,y,z}``) are
    translated+rotated, orientation channels (``{f}:ori_{x,y,z,w}``, xyzw) are
    quat-composed, and velocity channels (linear ``{f}:pos_*`` and angular
    ``{f}:ori_*`` in ``vel_output_names``) are rotated only.  ``joint:*``
    channels are frame-invariant and pass through untouched.

    Args:
        y_des: ``[n, T, P]`` anchor-relative position outputs (mutated in place).
        dy_des: ``[n, T, V]`` anchor-relative velocity outputs (mutated in place).
        anchor_pos: ``[n, T, 3]`` per-step world anchor positions (clip frame).
        anchor_quat: ``[n, T, 4]`` per-step world anchor quaternions (xyzw).
        pos_output_names: Ordered position-output names (includes ``ori_w``).
        vel_output_names: Ordered velocity-output names.

    Returns:
        ``(y_des, dy_des)`` lifted into the clip-world frame.
    """
    n, t = anchor_pos.shape[0], anchor_pos.shape[1]
    ap = anchor_pos.reshape(n * t, 3)
    aq = anchor_quat.reshape(n * t, 4)
    yf = y_des.reshape(n * t, y_des.shape[-1])
    dyf = dy_des.reshape(n * t, dy_des.shape[-1])
    dev = yf.device

    def _idx(names: list[str], frame: str, comps: tuple[str, ...], kind: str) -> torch.Tensor:
        return torch.tensor(
            [names.index(f"{frame}:{kind}_{a}") for a in comps], dtype=torch.long, device=dev
        )

    # Position blocks (translate + rotate), plus orientation where present.
    for f in sorted({nm.split(":", 1)[0] for nm in pos_output_names if ":pos_" in nm}):
        pos_i = _idx(pos_output_names, f, ("x", "y", "z"), "pos")
        yf[:, pos_i] = lift_position(ap, aq, yf[:, pos_i])
        if f"{f}:ori_w" in pos_output_names:
            ori_i = _idx(pos_output_names, f, ("x", "y", "z", "w"), "ori")
            yf[:, ori_i] = lift_orientation(aq, yf[:, ori_i])

    # Velocity blocks (rotate only): linear (pos_*) and angular (ori_*).
    for f in sorted({nm.split(":", 1)[0] for nm in vel_output_names if ":pos_" in nm}):
        lin_i = _idx(vel_output_names, f, ("x", "y", "z"), "pos")
        dyf[:, lin_i] = lift_vector(aq, dyf[:, lin_i])
    for f in sorted({nm.split(":", 1)[0] for nm in vel_output_names if ":ori_" in nm}):
        ang_i = _idx(vel_output_names, f, ("x", "y", "z"), "ori")
        dyf[:, ang_i] = lift_vector(aq, dyf[:, ang_i])

    return y_des, dy_des
