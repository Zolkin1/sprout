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

from motion_gen.terrain.overlay_boxes import overlay_sub_boxes


#: The values :attr:`OverlayVolumeCfg.kind` accepts, as data rather than only as a type
#: annotation.  ``@configclass`` does not validate a ``Literal`` at runtime, so
#: ``kind="wall"`` is accepted and silently drawn as a plain box -- which is exactly how a
#: new shape would look correct in Isaac and be missing in MuJoCo.  Manifest readers check
#: against this, mirroring the refusal :func:`overlay_sub_boxes` makes for ``variant``.
OVERLAY_KINDS: frozenset[str] = frozenset(("box", "obstacle", "overhead"))


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

    decor: bool = False
    """VISUAL ONLY: the depth cameras see it, the height scan does not, and it does
    not collide.

    Unlike :attr:`walkable_top` this is NOT advisory -- it changes which mesh the
    volume is built into. Decoration exists so the far camera has vertical structure
    to look at (at 15 deg pitch, 26.9% of its pixels return the range sentinel on
    flat ground), and the whole point is that it costs nothing else: the robot cannot
    walk into a wall it was never trained to avoid, and the height scan -- which
    clips at +/-1 m and would read a 1.5 m and a 5 m wall identically -- never sees
    one.

    Each simulator needs a different mechanism, which is why this is a schema field
    rather than something a renderer can infer. IsaacLab needs a SEPARATE PRIM: the
    height scanner is a ``RayCaster`` and takes exactly one mesh prim, while the
    depth cameras are ``MultiMeshRayCasterCamera`` and take a list. MuJoCo needs geom
    group 1 with ``contype=0 conaffinity=0``, because depth casts groups {1, 2} and
    the scan casts {2}."""

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


#: :func:`overlay_sub_boxes` lives in ``motion_gen.terrain.overlay_boxes`` and is re-exported
#: here so every call site that imports it from this module is unchanged.
#:
#: It MOVED because the decomposition of an overlay dict into solid boxes belongs to the
#: package that WRITES the manifest, not to the one that reads it -- and because the
#: terrain-consistency reward needs it in processes that have no IsaacLab (an offline
#: inspector, a unit test), which importing this module cannot give them.  The vendored
#: MuJoCo twin in ``transfer/sim/overlay_geom.py`` is unaffected and still pinned against
#: this by ``test_clip_tile_scene.py``.


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
