# Copyright 2026 Zachary Olkin. All rights reserved.

"""Overlay collision volumes placed on top of a composite sub-terrain.

Unlike :class:`TerrainBlock`, which tiles a walkable *surface* left-to-right along
x, an overlay is a finite collision *volume* placed at an ``(x, y)`` somewhere
inside the sub-terrain -- a standalone box to jump onto / off / over, a low
obstacle to hurdle, or (later) an overhead bar to crouch under.  Overlays overlap
the ground blocks in x rather than partitioning it, so they live in a separate
``overlays`` list on :class:`CompositeSubTerrainCfg` instead of the ``blocks``
sequence.

A box's **height** is the hard design constraint (it drives which jump motion is
generated); its footprint (``size_xyz`` x/y) and position are authored to *fit*
the generated clip's foot placement (see the box-clip generator), the same way a
staircase's tread width is sized to the clip rather than the clip forced onto a
fixed width.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Literal

import trimesh

from isaaclab.utils import configclass


@configclass
class OverlayVolumeCfg:
    """One finite collision volume placed on a composite sub-terrain.

    Coordinates are sub-terrain-local (the same frame the blocks build in); the
    composer / importer translate the assembled tile into the world.
    """

    kind: Literal["box", "obstacle", "overhead"] = "box"
    """``"box"`` -- a solid volume with a walkable top (jump onto / stand on /
    jump off, or jump over -- the behavior lives in the clip, not the geometry);
    ``"obstacle"`` -- a low volume to hurdle; ``"overhead"`` -- a raised bar to
    crouch under (``base_z > 0``)."""

    size_xyz: tuple[float, float, float] = MISSING
    """Full extents ``(sx, sy, sz)`` of the volume (m)."""

    pos_xy: tuple[float, float] = MISSING
    """Center of the volume in the sub-terrain-local xy plane (m)."""

    base_z: float = 0.0
    """Sub-terrain-local z of the volume's *bottom* face (m). ``0`` sits the
    volume on the tile ground; ``> 0`` floats it (an overhead bar)."""

    walkable_top: bool = True
    """Whether the volume's top face is a landing surface (a box the feet land
    on). ``False`` marks a pure collision obstacle. Advisory metadata -- the mesh
    is identical either way; consumers use it to decide projection / sensing."""


@dataclass
class OverlayOutput:
    """Result of :func:`build_overlay` (sub-terrain-local coordinates)."""

    mesh: trimesh.Trimesh
    """The volume's trimesh box geometry."""

    aabb: tuple[float, float, float, float, float, float]
    """3-D AABB ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the volume."""

    kind: str
    """The overlay :attr:`OverlayVolumeCfg.kind`."""

    walkable_top: bool
    """Whether the top face is a landing surface (mirrors the cfg)."""

    top_z: float
    """z of the volume's top face (``base_z + sz``); a box's walkable height."""


def build_overlay(cfg: OverlayVolumeCfg) -> OverlayOutput:
    """Build one overlay volume's mesh + AABB in sub-terrain-local coordinates.

    Args:
        cfg: The overlay configuration.

    Returns:
        An :class:`OverlayOutput` with a solid box mesh centered at
        ``(pos_xy, base_z + sz/2)``.

    Raises:
        ValueError: If any extent in ``size_xyz`` is non-positive.
    """
    sx, sy, sz = (float(v) for v in cfg.size_xyz)
    if not (sx > 0.0 and sy > 0.0 and sz > 0.0):
        raise ValueError(f"OverlayVolumeCfg.size_xyz must be all-positive; got {cfg.size_xyz}.")
    px, py = float(cfg.pos_xy[0]), float(cfg.pos_xy[1])
    base_z = float(cfg.base_z)

    mesh = trimesh.creation.box(extents=(sx, sy, sz))
    mesh.apply_translation((px, py, base_z + sz / 2.0))

    return OverlayOutput(
        mesh=mesh,
        aabb=(px - sx / 2.0, px + sx / 2.0, py - sy / 2.0, py + sy / 2.0, base_z, base_z + sz),
        kind=str(cfg.kind),
        walkable_top=bool(cfg.walkable_top),
        top_z=base_z + sz,
    )
