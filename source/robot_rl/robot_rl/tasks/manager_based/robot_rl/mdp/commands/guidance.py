# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-function helpers for terrain velocity guidance.

Terrain blocks can declare a guidance target (see
``terrains.velocity_guidance_cfg``); the multiskill velocity command then
PD-steers guided envs toward it instead of using randomly-sampled lateral /
yaw commands.  The math lives here, tensor-in / tensor-out, so unit tests
can hit it without instantiating any IsaacLab command term.
"""

from __future__ import annotations

import math

import torch


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi)``."""
    return torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi


def guidance_pd_commands(
    target_y: torch.Tensor,
    yaw_target: torch.Tensor,
    root_y: torch.Tensor,
    lat_vel_w: torch.Tensor,
    heading_w: torch.Tensor,
    gains: torch.Tensor,
    vy_range: torch.Tensor,
    wz_range: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lateral-position + heading PD toward a block guidance target.

    The lateral loop tracks the block-frame (world) y position, so its
    damping term is the world-frame lateral velocity — the derivative of
    the tracked signal, which also sees the drift a misaligned heading
    induces (``vx * sin(heading)``).  The outputs are written into the
    body-frame command channels under the small-heading approximation the
    heading loop itself enforces (same convention as the closed-loop mode
    in ``velocity_commands.py``).

    Args:
        target_y: ``[K]`` world-frame target y (block exit-edge midpoint).
        yaw_target: ``[K]`` world-frame heading target (block travel
            direction).
        root_y: ``[K]`` world-frame robot y.
        lat_vel_w: ``[K]`` world-frame robot lateral (y) velocity.
        heading_w: ``[K]`` world-frame robot heading.
        gains: ``[K, 3]`` PD gains ``(y_kp, y_kd, heading_kp)``.
        vy_range: ``[K, 2]`` (min, max) clamp on the lateral output.
        wz_range: ``[K, 2]`` (min, max) clamp on the yaw output.

    Returns:
        Tuple ``(vy, wz)`` of ``[K]`` clamped command channels.
    """
    vy = gains[:, 0] * (target_y - root_y) - gains[:, 1] * lat_vel_w
    wz = gains[:, 2] * _wrap_to_pi(yaw_target - heading_w)
    vy = torch.clamp(vy, min=vy_range[:, 0], max=vy_range[:, 1])
    wz = torch.clamp(wz, min=wz_range[:, 0], max=wz_range[:, 1])
    return vy, wz
