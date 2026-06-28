# Copyright 2026 Zachary Olkin. All rights reserved.

"""Closed-loop velocity command for FM-tracking envs (pure torch, IsaacLab-free).

A closed-loop FM env feeds the flow-matching generator a velocity command that
tracks the clip's *moving* base position reference, rather than the clip's
open-loop ``target_vel``.  The command is a feedforward (the clip ``target_vel``)
plus a proportional position controller, expressed in the robot heading frame
(the same ``(vx, vy, wz)`` convention as ``target_vel``):

    v_xy = v_ff[:2] + Kp * R(-yaw) @ (clip_pos - robot_pos)[:2] - Kd * v_robot_b[:2]
    wz   = v_ff[2]  + Kyaw * wrap_to_pi(clip_yaw - robot_yaw)   - Kd_yaw * wz_robot_b

Kept dependency-free (only ``torch``) so the math can be unit-tested without a
running IsaacLab / simulator.
"""

from __future__ import annotations

import math

import torch


def roll_buckets(
    rand_fm: torch.Tensor,
    rand_cl: torch.Tensor,
    clip_fraction: float,
    closed_loop_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env bucket assignment from two uniform ``[0, 1)`` draws.

    Args:
        rand_fm: ``[N]`` uniform draws deciding FM vs clip.
        rand_cl: ``[N]`` uniform draws deciding the open/closed split within FM.
        clip_fraction: Fraction of envs that track the clip (the rest track FM).
        closed_loop_fraction: Fraction of the FM envs that are closed-loop.

    Returns:
        ``(fm_mask, closed_loop_mask)`` boolean ``[N]`` tensors.  ``fm_mask`` is
        True where the env tracks the FM; ``closed_loop_mask`` (a subset of
        ``fm_mask``) is True for closed-loop FM envs.
    """
    fm_mask = rand_fm >= clip_fraction
    closed_loop_mask = fm_mask & (rand_cl < closed_loop_fraction)
    return fm_mask, closed_loop_mask


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi)``."""
    return torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi


def _yaw_from_quat(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw angle (rad) about world z from an ``(x, y, z, w)`` quaternion ``[..., 4]``."""
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def compute_closed_loop_command(
    v_ff: torch.Tensor,
    clip_pos_w: torch.Tensor,
    clip_quat_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    robot_lin_vel_b: torch.Tensor,
    robot_ang_vel_b: torch.Tensor,
    kp_pos: float,
    kd_pos: float,
    kp_yaw: float,
    kd_yaw: float,
    max_lin: float,
    max_ang: float,
) -> torch.Tensor:
    """Feedforward + proportional closed-loop velocity command ``[N, 3]``.

    All quaternions are ``(x, y, z, w)`` (scalar-last).  The position error is
    rotated into the robot heading frame so the returned command shares the
    ``target_vel`` ``(vx, vy, wz)`` heading-frame convention.

    Args:
        v_ff: ``[N, 3]`` feedforward command (clip ``target_vel``), heading frame.
        clip_pos_w: ``[N, 3]`` clip base world position reference.
        clip_quat_w: ``[N, 4]`` clip base world orientation (xyzw).
        robot_pos_w: ``[N, 3]`` robot base world position.
        robot_quat_w: ``[N, 4]`` robot base world orientation (xyzw).
        robot_lin_vel_b: ``[N, 3]`` robot base-frame linear velocity (for Kd).
        robot_ang_vel_b: ``[N, 3]`` robot base-frame angular velocity (for Kd_yaw).
        kp_pos: Proportional gain on the planar position error.
        kd_pos: Derivative gain (damping) on the robot planar velocity.
        kp_yaw: Proportional gain on the wrapped yaw error.
        kd_yaw: Derivative gain (damping) on the robot yaw rate.
        max_lin: Symmetric clamp (m/s) on each planar command component.
        max_ang: Symmetric clamp (rad/s) on the yaw-rate command.

    Returns:
        ``[N, 3]`` closed-loop command ``(vx, vy, wz)`` in the heading frame,
        clamped to the configured limits.
    """
    robot_yaw = _yaw_from_quat(robot_quat_w)
    clip_yaw = _yaw_from_quat(clip_quat_w)

    err_w = clip_pos_w[:, :2] - robot_pos_w[:, :2]
    cos_y, sin_y = torch.cos(robot_yaw), torch.sin(robot_yaw)
    # R(-yaw) @ err_w : rotate the world planar error into the heading frame.
    err_bx = cos_y * err_w[:, 0] + sin_y * err_w[:, 1]
    err_by = -sin_y * err_w[:, 0] + cos_y * err_w[:, 1]

    v_x = v_ff[:, 0] + kp_pos * err_bx - kd_pos * robot_lin_vel_b[:, 0]
    v_y = v_ff[:, 1] + kp_pos * err_by - kd_pos * robot_lin_vel_b[:, 1]

    yaw_err = _wrap_to_pi(clip_yaw - robot_yaw)
    wz = v_ff[:, 2] + kp_yaw * yaw_err - kd_yaw * robot_ang_vel_b[:, 2]

    v_x = v_x.clamp(-max_lin, max_lin)
    v_y = v_y.clamp(-max_lin, max_lin)
    wz = wz.clamp(-max_ang, max_ang)
    return torch.stack([v_x, v_y, wz], dim=-1)
