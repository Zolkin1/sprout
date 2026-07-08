# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared ingest helpers: one finite-difference velocity method for every source.

Both traj-opt strides and bones-seed clips get their velocities from central finite
differences of the (resampled) positions/quaternions, so a bones frame and a traj-opt
frame with the same motion produce the same feature -- the matcher never sees a spurious
difference from mismatched velocity conventions.
"""

from __future__ import annotations

import torch

from ..math_utils import quat_conjugate, quat_mul


def central_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Time derivative of ``x[F, ...]`` (central interior, one-sided ends)."""
    d = torch.zeros_like(x)
    if x.shape[0] == 1:
        return d
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def angular_velocity_from_quat(quat: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity ``[F, 3]`` from a wxyz quaternion series ``[F, 4]``.

    ``omega = 2 * (dq/dt) * conj(q)`` (vector part), the standard quaternion-rate identity.
    """
    qdot = central_diff(quat, dt)
    w = 2.0 * quat_mul(qdot, quat_conjugate(quat))
    return w[:, 1:]


def joint_channel_index(pos_output_names: list[str], joint_names: list[str]) -> list[int]:
    """Indices of ``joint:<name>`` channels in ``pos_output_names``, ordered by ``joint_names``."""
    return [pos_output_names.index(f"joint:{jn}") for jn in joint_names]


def frame_pose_index(pos_output_names: list[str], frame: str) -> tuple[list[int], list[int]]:
    """``(pos_idx[3], ori_idx[4])`` channel indices for ``<frame>:pos_*`` / ``:ori_*`` (wxyz)."""
    pos = [pos_output_names.index(f"{frame}:pos_{a}") for a in ("x", "y", "z")]
    ori = [pos_output_names.index(f"{frame}:ori_{a}") for a in ("w", "x", "y", "z")]
    return pos, ori
