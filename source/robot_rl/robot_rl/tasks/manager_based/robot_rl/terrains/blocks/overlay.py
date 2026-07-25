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

    variant: Literal[
        "solid", "floating", "floating_center_leg", "floating_corner_legs"
    ] = "solid"
    """Support structure under the (invariant) walkable top:
    ``"solid"`` -- filled box (default); ``"floating"`` -- a thin top slab only
    (gap below); ``"floating_center_leg"`` -- slab + one central support leg;
    ``"floating_corner_legs"`` -- slab + four corner legs (table-like).  The top
    face is at ``base_z + sz`` for **every** variant, so the walkable/landing
    height (and any baked jump motion) is unchanged -- only the geometry *below*
    the top differs."""

    slab_thickness: float = 0.1
    """Thickness (m) of the top slab for the ``floating*`` variants. Ignored for
    ``"solid"``."""

    leg_size: tuple[float, float] = (0.1, 0.1)
    """Support-leg cross-section ``(lx, ly)`` (m) for the ``*_leg(s)`` variants."""

    leg_inset: float = 0.05
    """Inset (m) of the four corner legs from the box footprint edge
    (``floating_corner_legs`` only)."""


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


# Legs poke this far up into the top slab so the concatenated mesh has no hairline
# gap at the slab/leg seam (which would create a degenerate collider manifold).
_LEG_SLAB_OVERLAP = 0.005


def overlay_sub_boxes(
    variant: str,
    size_xyz: tuple[float, float, float],
    pos_xy: tuple[float, float],
    base_z: float,
    slab_thickness: float,
    leg_size: tuple[float, float],
    leg_inset: float,
) -> list[tuple[float, float, float, float, float, float]]:
    """Decompose an overlay volume into its axis-aligned sub-boxes.

    Returns a list of ``(cx, cy, cz, sx, sy, sz)`` (center + full extents), the same
    numbers every renderer needs.  This is the **single source of truth** for the
    box-variant geometry -- :func:`build_overlay` (IsaacLab collider / grey viser
    mesh) and the hand-rolled ``view_clip_viser`` mesh both call it, so they cannot
    drift apart.

    The walkable top face is at ``base_z + size_xyz[2]`` for every variant; only the
    support *below* the top changes.

    Args:
        variant: One of ``"solid"``, ``"floating"``, ``"floating_center_leg"``,
            ``"floating_corner_legs"``.
        size_xyz: Full box extents ``(sx, sy, sz)`` (m).
        pos_xy: Box footprint center ``(px, py)`` (m).
        base_z: z of the box's bottom face (m).
        slab_thickness: Top-slab thickness for the ``floating*`` variants (m).
        leg_size: Support-leg cross-section ``(lx, ly)`` (m).
        leg_inset: Corner-leg inset from the footprint edge (m).

    Returns:
        List of sub-box ``(cx, cy, cz, sx, sy, sz)`` tuples.
    """
    sx, sy, sz = (float(v) for v in size_xyz)
    px, py = float(pos_xy[0]), float(pos_xy[1])
    base_z = float(base_z)
    top_z = base_z + sz

    if variant == "solid":
        return [(px, py, base_z + sz / 2.0, sx, sy, sz)]

    # Floating variants: a thin top slab (+ optional support legs beneath it).
    slab_t = min(float(slab_thickness), sz)
    boxes = [(px, py, top_z - slab_t / 2.0, sx, sy, slab_t)]
    if variant == "floating":
        return boxes

    lx, ly = float(leg_size[0]), float(leg_size[1])
    leg_h = sz - slab_t + _LEG_SLAB_OVERLAP
    if leg_h <= 1e-6:  # slab fills the whole height; no room for legs
        return boxes
    leg_cz = base_z + leg_h / 2.0
    if variant == "floating_center_leg":
        boxes.append((px, py, leg_cz, lx, ly, leg_h))
    elif variant == "floating_corner_legs":
        ox = max(0.0, sx / 2.0 - float(leg_inset) - lx / 2.0)
        oy = max(0.0, sy / 2.0 - float(leg_inset) - ly / 2.0)
        for sxn in (-1.0, 1.0):
            for syn in (-1.0, 1.0):
                boxes.append((px + sxn * ox, py + syn * oy, leg_cz, lx, ly, leg_h))
    else:
        raise ValueError(f"unknown overlay variant {variant!r}")
    return boxes


def build_overlay(cfg: OverlayVolumeCfg) -> OverlayOutput:
    """Build one overlay volume's mesh + AABB in sub-terrain-local coordinates.

    Args:
        cfg: The overlay configuration.

    Returns:
        An :class:`OverlayOutput` whose ``mesh`` is the (possibly multi-part) box
        variant, with the walkable top at ``base_z + sz``.  ``aabb`` is the full
        footprint x height and is identical across variants, so every downstream
        landing-height / placement consumer is unchanged.

    Raises:
        ValueError: If any extent in ``size_xyz`` is non-positive.
    """
    sx, sy, sz = (float(v) for v in cfg.size_xyz)
    if not (sx > 0.0 and sy > 0.0 and sz > 0.0):
        raise ValueError(f"OverlayVolumeCfg.size_xyz must be all-positive; got {cfg.size_xyz}.")
    px, py = float(cfg.pos_xy[0]), float(cfg.pos_xy[1])
    base_z = float(cfg.base_z)

    subs = overlay_sub_boxes(
        str(cfg.variant), (sx, sy, sz), (px, py), base_z,
        float(cfg.slab_thickness), tuple(cfg.leg_size), float(cfg.leg_inset),
    )
    parts = []
    for cx, cy, cz, bx, by, bz in subs:
        m = trimesh.creation.box(extents=(bx, by, bz))
        m.apply_translation((cx, cy, cz))
        parts.append(m)
    mesh = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)

    return OverlayOutput(
        mesh=mesh,
        aabb=(px - sx / 2.0, px + sx / 2.0, py - sy / 2.0, py + sy / 2.0, base_z, base_z + sz),
        kind=str(cfg.kind),
        walkable_top=bool(cfg.walkable_top),
        top_z=base_z + sz,
    )
