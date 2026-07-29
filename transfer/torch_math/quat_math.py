# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-torch quaternion / frame-transform math (no isaaclab dependency).

Vendored reimplementation of the handful of ``isaaclab.utils.math`` functions the
flow-matching transforms and the tracker observation terms rely on, so the
sim2sim / hardware deployment code carries **no** isaaclab (or robot_rl) import.

Convention: every quaternion is ``(x, y, z, w)`` (scalar-last), matching this
repository's ``isaaclab.utils.math`` and ``robot.data`` (verified empirically:
``matrix_from_quat([0.707,0,0,0.707])`` is a 90 deg rotation about +x). MuJoCo
quaternions are ``(w, x, y, z)`` and must be rolled to xyzw before use here.

``tests/test_quat_math_parity.py`` asserts these match ``isaaclab.utils.math``
numerically; keep them in lock-step.
"""

from __future__ import annotations

import torch

# Eps for small-angle / near-degenerate guards (matches isaaclab's tolerances).
_EPS = 1.0e-7


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a (x) b`` of two xyzw quaternions (broadcasts)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([x, y, z, w], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate ``[-x, -y, -z, w]`` of an xyzw quaternion."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([-x, -y, -z, w], dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of an xyzw quaternion (conjugate / squared-norm; exact for unit quats)."""
    return quat_conjugate(q) / (q * q).sum(dim=-1, keepdim=True)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` (..., 3) by xyzw quaternion ``q`` (...,4)."""
    qvec = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.linalg.cross(qvec, v, dim=-1)
    return v + w * t + torch.linalg.cross(qvec, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` by the inverse of xyzw quaternion ``q``."""
    return quat_apply(quat_conjugate(q), v)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Heading-only xyzw quaternion: drop roll/pitch, keep the yaw (z-rotation).

    Torch/xyzw port of ``isaaclab.utils.math.yaw_quat``: extracts the yaw angle and
    rebuilds the pure-z rotation ``(0, 0, sin(yaw/2), cos(yaw/2))`` (already unit).
    """
    qx, qy, qz, qw = q.unbind(-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    half = 0.5 * yaw
    out = torch.zeros_like(q)
    out[..., 2] = torch.sin(half)
    out[..., 3] = torch.cos(half)
    return out


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix ``(..., 3, 3)`` from an xyzw quaternion ``(..., 4)``."""
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    m = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    return m.reshape(*q.shape[:-1], 3, 3)


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion ``(..., 4)`` from a rotation matrix ``(..., 3, 3)``.

    Branchless four-candidate formulation (numerically stable; the candidate with
    the largest pivot is selected per element). Returns a unit quaternion; the
    overall sign (``q`` vs ``-q``) is canonicalised to a non-negative ``w``.
    """
    m00 = matrix[..., 0, 0]; m01 = matrix[..., 0, 1]; m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]; m11 = matrix[..., 1, 1]; m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]; m21 = matrix[..., 2, 1]; m22 = matrix[..., 2, 2]

    # q_abs[..., k] = 2 * |component k| for k in (w, x, y, z), via stabilised sqrt.
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )

    # Candidate quaternions for each pivot, laid out xyzw.
    quat_by_w = torch.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 0] ** 2], dim=-1)
    quat_by_x = torch.stack([q_abs[..., 1] ** 2, m10 + m01, m02 + m20, m21 - m12], dim=-1)
    quat_by_y = torch.stack([m10 + m01, q_abs[..., 2] ** 2, m12 + m21, m02 - m20], dim=-1)
    quat_by_z = torch.stack([m02 + m20, m12 + m21, q_abs[..., 3] ** 2, m10 - m01], dim=-1)

    candidates = torch.stack([quat_by_w, quat_by_x, quat_by_y, quat_by_z], dim=-2)
    denom = torch.clamp(2.0 * q_abs, min=4.0 * _EPS).unsqueeze(-1)
    candidates = candidates / denom

    pivot = torch.argmax(q_abs, dim=-1)
    quat = torch.gather(
        candidates, -2, pivot[..., None, None].expand(*pivot.shape, 1, 4)
    ).squeeze(-2)

    # Canonicalise sign (w >= 0) so the round-trip is deterministic.
    quat = torch.where((quat[..., 3:4] < 0.0), -quat, quat)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def combine_frame_transforms(
    p01: torch.Tensor,
    q01: torch.Tensor,
    p12: torch.Tensor,
    q12: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``T01 (x) T12`` -> ``(p02, q02)`` (xyzw quaternions)."""
    q02 = quat_mul(q01, q12)
    p02 = p01 + quat_apply(q01, p12)
    return p02, q02


def subtract_frame_transforms(
    p01: torch.Tensor,
    q01: torch.Tensor,
    p02: torch.Tensor,
    q02: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`combine_frame_transforms`: ``T01^-1 (x) T02`` -> ``(p12, q12)``."""
    q01_inv = quat_conjugate(q01)
    q12 = quat_mul(q01_inv, q02)
    p12 = quat_apply(q01_inv, p02 - p01)
    return p12, q12


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Axis-angle (so(3)) rotation vector ``(..., 3)`` from an xyzw quaternion.

    Faithful port of ``isaaclab.utils.math.axis_angle_from_quat``: the quaternion
    sign is canonicalised (``w >= 0``) so the short-way rotation is taken, and a
    Taylor expansion is used near zero angle.
    """
    quat = quat * (1.0 - 2.0 * (quat[..., 3:4] < 0.0))
    mag = torch.linalg.norm(quat[..., :3], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 3])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48.0
    )
    return quat[..., :3] / sin_half_angles_over_angles.unsqueeze(-1)


def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Tangent-space difference ``q1 (-) q2`` -> rotation vector ``(..., 3)``.

    The axis-angle (so(3)) representation of the relative rotation
    ``q1 (x) q2^-1``, matching ``isaaclab.utils.math.quat_box_minus``.
    """
    return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2)))
