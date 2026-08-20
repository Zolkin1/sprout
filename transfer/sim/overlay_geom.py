# Copyright 2026 Zachary Olkin. All rights reserved.

"""Dependency-free port of the terrain overlay sub-box decomposition.

The IsaacLab-side original is
``robot_rl.tasks.manager_based.robot_rl.terrains.blocks.overlay.overlay_sub_boxes``, which
is the single source of truth for what a manifest overlay's ``variant`` actually MEANS
geometrically.  This file exists because that module imports ``isaaclab.utils.configclass``
and everything under ``transfer/`` runs in ``env_transfer``, which has no IsaacLab -- the
same reason ``transfer/deploy/quat_math_np.py`` is a port of ``transfer/torch_math``.

Like that pair, the two copies are held together by a parity test rather than by an import:
``source/robot_rl/tests/test_clip_tile_scene.py`` runs both over a grid of variants and
parameters and requires them to agree exactly.  Keep this file stdlib-only so it stays
importable from the MuJoCo emitters.

Why this matters, concretely: before it existed, the MuJoCo tile emitters treated every
overlay as one solid box.  348 of the 1260 overlays in ``clips/g1_mb_v5_body.h5`` are
``variant="floating"`` -- a thin top slab with an OPEN GAP beneath -- so MuJoCo built solid
pillars where Isaac built floating slats.  A top-down height field cannot see the
difference (the top face is invariant across variants, by design), which is why the
existing agreement test could not catch it; the depth cameras see it immediately.
"""

from __future__ import annotations

#: Legs poke this far up into the top slab so the concatenated geometry has no hairline gap
#: at the slab/leg seam.  Mirrors ``blocks/overlay.py``'s constant of the same name.
_LEG_SLAB_OVERLAP = 0.005

#: The variants ``overlay_sub_boxes`` understands, in the order ``blocks/overlay.py`` declares.
OVERLAY_VARIANTS = ("solid", "floating", "floating_center_leg", "floating_corner_legs")


def overlay_sub_boxes(
    variant: str,
    size_xyz: tuple[float, float, float],
    pos_xy: tuple[float, float],
    base_z: float,
    slab_thickness: float = 0.1,
    leg_size: tuple[float, float] = (0.1, 0.1),
    leg_inset: float = 0.05,
) -> list[tuple[float, float, float, float, float, float]]:
    """Decompose an overlay volume into its axis-aligned sub-boxes.

    The walkable top face is at ``base_z + size_xyz[2]`` for every variant; only the support
    *below* the top changes.

    Args:
        variant: One of :data:`OVERLAY_VARIANTS`.
        size_xyz: Full box extents ``(sx, sy, sz)`` (m).
        pos_xy: Box footprint center ``(px, py)`` (m).
        base_z: z of the box's bottom face (m).
        slab_thickness: Top-slab thickness for the ``floating*`` variants (m).
        leg_size: Support-leg cross-section ``(lx, ly)`` (m).
        leg_inset: Corner-leg inset from the footprint edge (m).

    Returns:
        List of sub-box ``(cx, cy, cz, sx, sy, sz)`` tuples (center + full extents).

    Raises:
        ValueError: On an unknown ``variant``.
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


#: Geom group for VISUAL-ONLY decoration, and the contact flags that go with it.
#:
#: ``transfer/sim/depth_camera.py`` casts groups {1 = per-link visual meshes, 2 = terrain}
#: and ``Robot.compute_height_scan`` casts {2}. So decoration belongs in group 1: the depth
#: cameras see it, the height scan does not. ``contype=0 conaffinity=0`` makes it
#: non-colliding, which is the third property -- a wall the robot cannot walk into having
#: never been trained to avoid one.
#:
#: The IsaacLab side achieves the same three properties differently, with a separate prim
#: (``ClipTerrainImporter.DECOR_PRIM_PATH``), because there the height scanner is a
#: ``RayCaster`` bound to exactly one mesh prim. Two mechanisms, one meaning; the pair is
#: checked by ``source/robot_rl/tests/test_self_scan_matches_mujoco.py``'s sibling for
#: terrain, and by the emitter tests here.
DECOR_GEOM_GROUP = 1
DECOR_CONTACT_ATTRS = 'contype="0" conaffinity="0"'


def is_decor(ov: dict) -> bool:
    """Whether an overlay is visual-only decoration."""
    return bool(ov.get("decor", False))


def overlay_sub_boxes_from_manifest(
    ov: dict,
) -> list[tuple[float, float, float, float, float, float]]:
    """:func:`overlay_sub_boxes` applied to a raw manifest overlay dict.

    Reads the same keys and defaults the IsaacLab loader's ``_overlays_from_manifest`` does,
    so a manifest entry decomposes identically on both sides.

    Args:
        ov: One ``overlays`` entry from a clip library's ``terrain_manifest``.

    Returns:
        List of sub-box ``(cx, cy, cz, sx, sy, sz)`` tuples.
    """
    leg_size = ov.get("leg_size", (0.1, 0.1))
    return overlay_sub_boxes(
        str(ov.get("variant", "solid")),
        tuple(float(v) for v in ov["size_xyz"]),
        (float(ov["pos_xy"][0]), float(ov["pos_xy"][1])),
        float(ov.get("base_z", 0.0)),
        float(ov.get("slab_thickness", 0.1)),
        (float(leg_size[0]), float(leg_size[1])),
        float(ov.get("leg_inset", 0.05)),
    )
