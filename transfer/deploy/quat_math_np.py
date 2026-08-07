# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-numpy quaternion / frame-transform math (no torch, no isaaclab).

Numpy port of ``transfer/torch_math/quat_math.py`` for the torch-free deployment runtime
(numpy + onnxruntime). Convention: every quaternion is ``(x, y, z, w)``
(scalar-last), matching the repo's ``isaaclab.utils.math`` and ``robot.data``.
Last-axis component layout; leading dims broadcast.

``transfer/deploy/tests/test_deploy_parity.py`` asserts these match the torch
``transfer/torch_math/quat_math`` numerically; keep them in lock-step.
"""

from __future__ import annotations

import numpy as np

_EPS = 1.0e-7


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a (x) b`` of two xyzw quaternions (broadcasts)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate ``[-x, -y, -z, w]`` of an xyzw quaternion."""
    out = -q
    out[..., 3] = q[..., 3]
    return out


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of an xyzw quaternion (conjugate / squared-norm; exact for unit quats)."""
    return quat_conjugate(q) / (q * q).sum(axis=-1, keepdims=True)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` (..., 3) by xyzw quaternion ``q`` (...,4)."""
    qvec = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the inverse of xyzw quaternion ``q``."""
    return quat_apply(quat_conjugate(q), v)


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Heading-only xyzw quaternion: drop roll/pitch, keep the yaw (z-rotation).

    Numpy/xyzw port of ``isaaclab.utils.math.yaw_quat``: extracts the yaw angle and
    rebuilds the pure-z rotation ``(0, 0, sin(yaw/2), cos(yaw/2))`` (already unit).
    """
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    half = 0.5 * yaw
    out = np.zeros_like(q)
    out[..., 2] = np.sin(half)
    out[..., 3] = np.cos(half)
    return out


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``(..., 3, 3)`` from an xyzw quaternion ``(..., 4)``."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
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
    m = np.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1)
    return m.reshape(*q.shape[:-1], 3, 3)


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """xyzw quaternion ``(..., 4)`` from a rotation matrix ``(..., 3, 3)``.

    Branchless four-candidate formulation; selects the candidate with the largest
    pivot per element and canonicalises ``w >= 0``.
    """
    m00 = matrix[..., 0, 0]; m01 = matrix[..., 0, 1]; m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]; m11 = matrix[..., 1, 1]; m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]; m21 = matrix[..., 2, 1]; m22 = matrix[..., 2, 2]

    q_abs = np.sqrt(
        np.clip(
            np.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            ),
            a_min=0.0,
            a_max=None,
        )
    )

    quat_by_w = np.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 0] ** 2], axis=-1)
    quat_by_x = np.stack([q_abs[..., 1] ** 2, m10 + m01, m02 + m20, m21 - m12], axis=-1)
    quat_by_y = np.stack([m10 + m01, q_abs[..., 2] ** 2, m12 + m21, m02 - m20], axis=-1)
    quat_by_z = np.stack([m02 + m20, m12 + m21, q_abs[..., 3] ** 2, m10 - m01], axis=-1)

    candidates = np.stack([quat_by_w, quat_by_x, quat_by_y, quat_by_z], axis=-2)
    denom = np.clip(2.0 * q_abs, a_min=4.0 * _EPS, a_max=None)[..., None]
    candidates = candidates / denom

    pivot = np.argmax(q_abs, axis=-1)
    quat = np.take_along_axis(candidates, pivot[..., None, None], axis=-2)[..., 0, :]
    quat = np.where(quat[..., 3:4] < 0.0, -quat, quat)
    return quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), a_min=_EPS, a_max=None)


def combine_frame_transforms(p01, q01, p12, q12):
    """Compose ``T01 (x) T12`` -> ``(p02, q02)`` (xyzw quaternions)."""
    q02 = quat_mul(q01, q12)
    p02 = p01 + quat_apply(q01, p12)
    return p02, q02


def subtract_frame_transforms(p01, q01, p02, q02):
    """Inverse of :func:`combine_frame_transforms`: ``T01^-1 (x) T02`` -> ``(p12, q12)``."""
    q01_inv = quat_conjugate(q01)
    q12 = quat_mul(q01_inv, q02)
    p12 = quat_apply(q01_inv, p02 - p01)
    return p12, q12


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Axis-angle (so(3)) rotation vector ``(..., 3)`` from an xyzw quaternion."""
    quat = quat * (1.0 - 2.0 * (quat[..., 3:4] < 0.0))
    mag = np.linalg.norm(quat[..., :3], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 3])
    angle = 2.0 * half_angle
    sin_half_over_angle = np.where(
        np.abs(angle) > eps, np.sin(half_angle) / np.where(np.abs(angle) > eps, angle, 1.0),
        0.5 - angle * angle / 48.0,
    )
    return quat[..., :3] / sin_half_over_angle[..., None]


def quat_box_minus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Tangent-space difference ``q1 (-) q2`` -> rotation vector ``(..., 3)``."""
    return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2)))


def quat_unique(q: np.ndarray) -> np.ndarray:
    """Put an xyzw quaternion in the standard form with a non-negative real part.

    ``q`` and ``-q`` are the same rotation; picking one makes results comparable
    componentwise. Mirrors ``isaaclab.utils.math.quat_unique``.
    """
    return np.where(q[..., 3:4] < 0.0, -q, q)


def quat_from_axis_angle(delta: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Exponential map: rotation vector ``(..., 3)`` -> xyzw quaternion ``(..., 4)``.

    The inverse of :func:`axis_angle_from_quat`. ``eps`` floors the norm so a zero
    perturbation gives identity instead of dividing by zero -- the same guard, in the same
    place, as ``isaaclab.utils.math.quat_box_plus``.
    """
    delta = np.asarray(delta, dtype=np.float64)
    norm = np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), eps)
    half = 0.5 * norm
    xyz = (delta / norm) * np.sin(half)
    q = np.concatenate([xyz, np.cos(half)], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_box_plus(q: np.ndarray, delta: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Apply an axis-angle increment: ``Exp(delta) * q``, sign-normalised.

    The inverse of :func:`quat_box_minus` and the port of
    ``isaaclab.utils.math.quat_box_plus``, which is what ``flow_matching.window`` uses to
    lift a window's ``base_pose`` orientation half back to the world. Parity-tested against
    it, because a convention slip here rotates a reference rather than failing.

    Args:
        q: ``(..., 4)`` base orientation, xyzw.
        delta: ``(..., 3)`` axis-angle increment.
        eps: Norm floor for ``delta``.

    Returns:
        ``(..., 4)`` xyzw quaternion with a non-negative real part -- possibly the negation
        of what a caller expects, which is the SAME rotation. Compare rotations, not
        components.
    """
    return quat_unique(quat_mul(quat_from_axis_angle(delta, eps), q))
