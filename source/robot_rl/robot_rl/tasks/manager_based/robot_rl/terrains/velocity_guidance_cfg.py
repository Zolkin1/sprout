# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-block velocity guidance configuration.

A terrain block can declare that envs standing on it should not receive
randomly-sampled lateral / yaw velocity commands, but instead be PD-steered
toward a block-derived target (e.g. the midpoint of the block's exit edge).
The importer turns this declaration into per-block tensors and exposes them
via ``MetaCompositeTerrainImporter.velocity_guidance_at``; the multiskill
velocity command consumes the query each step and overwrites the vy / wz
channels for guided envs.

``mode`` selects how the importer derives the target from the block's
geometry — a registry key so future terrain types can add their own target
construction without touching the velocity-command code path.
"""

from __future__ import annotations

from isaaclab.utils import configclass

GUIDANCE_MODE_END_OF_BLOCK = "end_of_block"
"""Steer toward the midpoint of the block's exit edge (walkable-y center at
``aabb_xmax``), facing the block's travel direction."""

KNOWN_GUIDANCE_MODES = (GUIDANCE_MODE_END_OF_BLOCK,)


@configclass
class BlockVelocityGuidanceCfg:
    """Velocity guidance declaration for one terrain block.

    The PD acts on the block-frame lateral position (world y, since blocks
    are axis-aligned) and on the heading error to the block's travel
    direction.  The damping term uses the world-frame lateral velocity —
    the derivative of the tracked signal — which also captures the drift a
    misaligned heading induces (``vx * sin(heading)``); see the multiskill
    velocity command for the body-frame caveat.
    """

    mode: str = GUIDANCE_MODE_END_OF_BLOCK
    """Target-construction mode (registry key, see ``KNOWN_GUIDANCE_MODES``)."""

    lin_vel_y: tuple[float, float] = (-0.4, 0.4)
    """Clamp on the lateral PD output (m/s).  Default matches the stair
    library's lateral conditioning grid."""

    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    """Clamp on the yaw PD output (rad/s).  Keep modest when the active
    skill's library has no yawing trajectories."""

    y_kp: float = 1.5
    """Lateral position gain (matches the midpoint of the random-target
    ``y_kp`` sampling range)."""

    y_kd: float = 0.3
    """Lateral velocity damping gain (world-frame lateral velocity)."""

    heading_kp: float = 0.5
    """Heading error gain toward the block's travel direction."""
