# Copyright 2026 Zachary Olkin. All rights reserved.

"""Manifest overlay volumes -> the axis-aligned COLLIDER boxes every consumer needs.

This is the single source of truth for what an overlay's ``variant`` *means* geometrically.
It lives here, in the package that WRITES the terrain manifest, rather than on the RL side
that reads it: :mod:`motion_gen.terrain.freeform` and the box / stair generators author the
overlay dicts, so the decomposition of one of those dicts into solid boxes is theirs to
define.  ``robot_rl...terrains.blocks.overlay`` re-exports :func:`overlay_sub_boxes` so every
existing call site is unchanged, and the IsaacLab-free MuJoCo copy in
``transfer/sim/overlay_geom.py`` stays where it is -- it is the deploy-env vendored twin and
is already pinned against this by ``test_clip_tile_scene.py``.

The reason this module exists at all is the terrain-consistency reward: it needs the exact
collider set of a tile, in torch, in a process that may have no IsaacLab (an offline
inspector, a unit test) and may have no MuJoCo either.  Everything here is stdlib + numpy.
"""

from __future__ import annotations

import numpy as np

#: The variants :func:`overlay_sub_boxes` understands, in declaration order.
OVERLAY_VARIANTS: tuple[str, ...] = (
    "solid", "floating", "floating_center_leg", "floating_corner_legs",
)

# Legs poke this far up into the top slab so the concatenated mesh has no hairline
# gap at the slab/leg seam (which would create a degenerate collider manifold).
_LEG_SLAB_OVERLAP = 0.005

#: Thickness (m) of the synthetic slab that stands in for a tile's ground PLANE.
#:
#: A ``has_floor=True`` tile gets its ground from a ``FlatBlockCfg``, which is a surface and
#: not an overlay, so the manifest carries no box for it.  The SDF needs a solid, and an
#: infinite half-space would make every point under the tile "inside the terrain" -- which is
#: true of a plane and false of the tile, whose edges a robot can walk off.  1 m is far
#: thicker than anything a plan reaches through and thin enough to stay a box.
GROUND_SLAB_THICKNESS_M: float = 1.0


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
    box-variant geometry -- ``build_overlay`` (IsaacLab collider / grey viser mesh), the
    hand-rolled ``view_clip_viser`` mesh and :func:`collider_boxes_from_manifest_tile` all
    call it, so they cannot drift apart.

    The walkable top face is at ``base_z + size_xyz[2]`` for every variant; only the
    support *below* the top changes.

    Args:
        variant: One of :data:`OVERLAY_VARIANTS`.
        size_xyz: Full box extents ``(sx, sy, sz)`` (m).
        pos_xy: Box footprint center ``(px, py)`` (m).
        base_z: z of the box's bottom face (m).
        slab_thickness: Top-slab thickness for the ``floating*`` variants (m).
        leg_size: Support-leg cross-section ``(lx, ly)`` (m).
        leg_inset: Corner-leg inset from the footprint edge (m).

    Returns:
        List of sub-box ``(cx, cy, cz, sx, sy, sz)`` tuples.

    Raises:
        ValueError: If ``variant`` is not one of :data:`OVERLAY_VARIANTS`.
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


def overlay_sub_boxes_from_manifest(
    ov: dict,
) -> list[tuple[float, float, float, float, float, float]]:
    """:func:`overlay_sub_boxes` applied to one raw manifest overlay dict.

    Reads the same defaults ``OverlayVolumeCfg`` declares, so an overlay written before a
    field existed decomposes exactly as it did then.  ``decor`` is NOT consulted -- this
    returns the geometry, and whether that geometry collides is the caller's question.

    Args:
        ov: One entry of a manifest tile's ``overlays`` list.

    Returns:
        List of sub-box ``(cx, cy, cz, sx, sy, sz)`` tuples.
    """
    return overlay_sub_boxes(
        str(ov.get("variant", "solid")),
        tuple(float(v) for v in ov["size_xyz"]),          # type: ignore[arg-type]
        tuple(float(v) for v in ov["pos_xy"]),            # type: ignore[arg-type]
        float(ov.get("base_z", 0.0)),
        float(ov.get("slab_thickness", 0.1)),
        tuple(float(v) for v in ov.get("leg_size", (0.1, 0.1))),   # type: ignore[arg-type]
        float(ov.get("leg_inset", 0.05)),
    )


def is_decor(ov: dict) -> bool:
    """Whether an overlay is the VISUAL-ONLY decoration channel.

    Decor has no collider in either simulator (IsaacLab puts it on its own prim with no
    collision API; MuJoCo emits it with ``contype=0 conaffinity=0``), so it must be excluded
    from anything that answers "where is the ground".  A decorated wall the robot walks
    through is the whole point of the channel.

    Args:
        ov: One manifest overlay dict.

    Returns:
        ``True`` when the overlay is decoration.
    """
    return bool(ov.get("decor", False))


def collider_boxes_from_manifest_tile(entry: dict) -> np.ndarray:
    """Every COLLIDING box of one manifest tile, in the tile's local-corner frame.

    The tile's collision geometry is exactly (a) its non-``decor`` overlays, expanded through
    :func:`overlay_sub_boxes` so a ``floating`` tread contributes its slab and not the gap
    beneath it, plus (b) a ground slab when ``has_floor`` is set.  Nothing else: a tile is a
    flat ground block with boxes on it.

    Args:
        entry: One element of a library's ``terrain_manifest`` attribute.

    Returns:
        ``(B, 6)`` float64 array of ``(cx, cy, cz, sx, sy, sz)`` -- centers and FULL extents.

    Raises:
        ValueError: If the tile uses the LEGACY ``bands`` / ``dims`` schema, or carries a
            non-zero overlay ``yaw``.  Both are refusals rather than best-efforts.  A legacy
            tile's stairs are rebuilt arithmetically by ``StairBlock`` and never appear as
            boxes, so silently returning only its ground would hand back a flat floor for a
            staircase -- which is precisely the flat-``ground_z`` mistake that made the FM's
            terrain-penetration loss inert (commit ``27723572``).  A yawed overlay is already
            refused by both the Isaac loader and the MuJoCo emitter.
    """
    tid = entry.get("terrain_id", "?")
    if "overlays" not in entry:
        raise ValueError(
            f"terrain tile {tid} uses the legacy 'bands'/'dims' manifest schema, which "
            "carries no explicit box list -- its stairs are recomputed by StairBlock at "
            "load. There is no correct collider set to return here, and returning only the "
            "ground would be a FLAT floor under a staircase (the exact defect that made the "
            "FM terrain-penetration loss inert). Regenerate the library with a current "
            "generator, which emits the overlay schema."
        )

    boxes: list[tuple[float, float, float, float, float, float]] = []
    for i, ov in enumerate(entry["overlays"]):
        if float(ov.get("yaw", 0.0)) != 0.0:
            raise ValueError(
                f"terrain tile {tid} overlay {i} has yaw={ov['yaw']}, which this "
                "axis-aligned decomposition cannot express (and which both the Isaac loader "
                "and the MuJoCo emitter already refuse)."
            )
        if is_decor(ov):
            continue
        boxes.extend(overlay_sub_boxes_from_manifest(ov))

    if bool(entry.get("has_floor", True)):
        sx, sy = float(entry["size_x"]), float(entry["size_y"])
        t = GROUND_SLAB_THICKNESS_M
        boxes.append((sx / 2.0, sy / 2.0, -t / 2.0, sx, sy, t))

    if not boxes:
        raise ValueError(
            f"terrain tile {tid} has has_floor=False and no colliding overlays, so it has "
            "no geometry at all. That is a generation bug, not a terrain a robot can cross."
        )
    return np.asarray(boxes, dtype=np.float64)


def flat_tile_collider_boxes(size_x: float, size_y: float) -> np.ndarray:
    """The collider set of a FLAT tile, which has no manifest entry at all.

    ``terrain_id == -1`` tiles are generated rather than authored (the flat clips need only
    somewhere level to stand), so they are absent from ``terrain_manifest`` and their ground
    has to be stated here.  Their decoration lives in the library's separate
    ``flat_tile_decor`` attribute and is visual-only, so it contributes nothing.

    Args:
        size_x / size_y: The flat tile's footprint (m).

    Returns:
        ``(1, 6)`` float64 array holding the ground slab.
    """
    t = GROUND_SLAB_THICKNESS_M
    return np.asarray([(size_x / 2.0, size_y / 2.0, -t / 2.0, size_x, size_y, t)],
                      dtype=np.float64)
