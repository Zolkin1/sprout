# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain representation + foot contact geometry for the terrain-penetration loss.

For now the terrain is a flat plane (:class:`FlatTerrain`), but the loss only ever calls
``terrain.penetration(world_xyz)`` -> per-point depth below the surface (>= 0), so a future
signed-distance-field / heightfield terrain drops in behind the same interface (return
``(-sdf(world_xyz)).clamp_min(0)``).

The contact geometry is the G1 foot box (IsaacLab uses a box foot contact). The default
box matches ``robot_assets/g1/g1_21j/g1_21j_box_merged.urdf`` -- sole center
``(0.03592, 0, -0.02517)`` and full size ``(0.20311, 0.06547, 0.01851)`` in the
``ankle_roll_link`` frame -- so :func:`box_sole_points` samples a grid on the box's bottom
(sole) face, expressed in that foot frame.
"""

from __future__ import annotations

import torch

# G1 foot box (ankle_roll_link frame), from g1_21j_box_merged.urdf.
G1_FOOT_BOX_CENTER: tuple[float, float, float] = (0.03592, 0.0, -0.02517)
G1_FOOT_BOX_HALF: tuple[float, float, float] = (0.101555, 0.032735, 0.009255)


def box_sole_points(
    center: tuple[float, float, float] = G1_FOOT_BOX_CENTER,
    half_extents: tuple[float, float, float] = G1_FOOT_BOX_HALF,
    grid: tuple[int, int] = (3, 2),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform grid of points on the box **sole** (bottom face), in the foot frame.

    Args:
        center: Box center ``(x, y, z)`` in the ankle/foot frame.
        half_extents: Box half-sizes ``(hx, hy, hz)``.
        grid: ``(nx, ny)`` grid resolution over the sole face.
        dtype: Output dtype.

    Returns:
        Contact points, shape ``(nx*ny, 3)`` -- all at the sole z (``center_z - hz``).
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents
    nx, ny = grid
    xs = torch.linspace(cx - hx, cx + hx, nx, dtype=dtype)
    ys = torch.linspace(cy - hy, cy + hy, ny, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    z = torch.full((nx * ny,), cz - hz, dtype=dtype)
    return torch.stack([gx.reshape(-1), gy.reshape(-1), z], dim=-1)  # (nx*ny, 3)


class FlatTerrain:
    """Flat ground at world height ``ground_z``; penetration is depth below it.

    The seam a future SDF/heightfield terrain implements: same ``penetration`` signature.
    """

    def __init__(self, ground_z: float = 0.0):
        """Store the world-frame ground height (m)."""
        self.ground_z = float(ground_z)

    def penetration(self, world_xyz: torch.Tensor) -> torch.Tensor:
        """Depth below the ground for each point, shape ``(...,)`` (0 if at/above ground).

        Args:
            world_xyz: World-frame points, shape ``(..., 3)``.
        """
        return (self.ground_z - world_xyz[..., 2]).clamp_min(0.0)

    def to(self, device: torch.device | str) -> "FlatTerrain":
        """No tensors to move; present for parity with a future tensor-backed terrain."""
        return self
