# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vendored quaternion math (IsaacLab-free).

Faithful copies of the four ``isaaclab.utils.math`` helpers the offline generation
path uses, so ``motion_gen`` needs no IsaacLab import.  Kept byte-for-byte equivalent
to the IsaacLab definitions (same **wxyz** quaternion convention, same formulas) and
parity-tested against them; do not "improve" the math here without re-checking parity.
"""

from __future__ import annotations

import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize ``x`` to unit length along the last dim (matches IsaacLab)."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a (w, x, y, z) quaternion. Shape ``(..., 4)``."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a (w, x, y, z) quaternion (conjugate of the normalized quat)."""
    return normalize(quat_conjugate(q))


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (w, x, y, z) quaternions. Shape ``(..., 4)``."""
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw-only (w, x, y, z) quaternion. Shape ``(..., 4)``."""
    shape = quat.shape
    quat_yaw = quat.clone().view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (x, y, z) by the (w, x, y, z) quaternion ``quat``."""
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix ``[..., 3, 3]`` -> unit quaternion ``[..., 4]`` (wxyz).

    Numerically stable branchless form (Shepperd's method by the largest diagonal term).
    """
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    # Four candidate quaternions (one per Shepperd case), pick the numerically best.
    q_w = torch.stack([1.0 + m00 + m11 + m22, m[:, 2, 1] - m[:, 1, 2],
                       m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], dim=-1)
    q_x = torch.stack([m[:, 2, 1] - m[:, 1, 2], 1.0 + m00 - m11 - m22,
                       m[:, 1, 0] + m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0]], dim=-1)
    q_y = torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] + m[:, 0, 1],
                       1.0 - m00 + m11 - m22, m[:, 2, 1] + m[:, 1, 2]], dim=-1)
    q_z = torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0],
                       m[:, 2, 1] + m[:, 1, 2], 1.0 - m00 - m11 + m22], dim=-1)
    cand = torch.stack([q_w, q_x, q_y, q_z], dim=1)                 # [N, 4, 4]
    diag = torch.stack([m00 + m11 + m22, m00, m11, m22], dim=-1)    # trace + diagonals
    pick = diag.argmax(dim=-1)                                      # [N]
    q = cand[torch.arange(cand.shape[0], device=m.device), pick]   # [N, 4]
    q = normalize(q)
    # canonical sign (w >= 0)
    q = torch.where((q[:, :1] < 0), -q, q)
    return q.reshape(*shape, 4)
