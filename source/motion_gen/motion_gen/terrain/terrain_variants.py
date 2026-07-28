# Copyright 2026 Zachary Olkin. All rights reserved.

"""Resolve terrain-geometry variants from generation-time fraction knobs.

The offline clip generator must decide, per terrain, *which* geometry variant to
bake (a stair's decoration, a box's support structure) and record the resolved,
explicit parameters in the clip manifest -- the RL importer and the viewers rebuild
the mesh from that manifest and do **not** share the generator's RNG state.

This module owns that fraction -> variant *policy* for both families.  It is pure
numpy (no IsaacLab, no torch): the terrain block/mesh construction lives in
``robot_rl`` (it is built on IsaacLab ``@configclass``), but the *selection* logic
belongs here so it is centralized and unit-testable without IsaacLab.  The returned
dicts match exactly the manifest schema the ``robot_rl`` loader reads back.
"""

from __future__ import annotations

import numpy as np

# Canonical box variants, in a fixed order (also the ``np.random.choice`` order).
BOX_VARIANTS = ("solid", "floating", "floating_center_leg", "floating_corner_legs")

# Accept short aliases in the CLI fraction spec for the two verbose leg variants.
_BOX_VARIANT_ALIASES = {
    "center_leg": "floating_center_leg",
    "corner_legs": "floating_corner_legs",
}


def resolve_stair_decor(
    rng: np.random.Generator,
    *,
    float_frac: float = 0.0,
    side_wall_frac: float = 0.0,
    side_posts_frac: float = 0.0,
    nosing_frac: float = 0.0,
    float_thick_range: tuple[float, float] = (0.03, 0.06),
    wall_thickness: float = 0.1,
    wall_height: float = 1.0,
    post_thickness: float = 0.06,
    post_spacing: float = 0.4,
    post_height: float = 1.0,
    nosing_radius_range: tuple[float, float] = (0.02, 0.05),
    nosing_segments: int = 6,
) -> dict:
    """Resolve one staircase span's decoration into an explicit manifest dict.

    Each side is drawn **independently** so any combination is possible (e.g. a wall
    on one side and railing posts on the other).  Per side the draw is: ``wall`` with
    probability ``side_wall_frac``, else ``posts`` with probability ``side_posts_frac``,
    else ``none`` (so the two fracs should sum to ``<= 1``).  Floating treads and a
    rounded nosing are independent Bernoulli draws.

    Args:
        rng: A seeded ``numpy`` generator (draws are consumed in a fixed order:
            floating, nosing, left side, right side -- so results are reproducible).
        float_frac: Probability the treads are floating (riser-less) thin slabs.
        side_wall_frac: Per-side probability of a solid sidewall.
        side_posts_frac: Per-side probability of railing posts.
        nosing_frac: Probability of a rounded leading nosing on each tread.
        float_thick_range: Floating-tread thickness range (m).
        wall_thickness: Sidewall thickness (m).
        wall_height: Sidewall height above the top tread (m).
        post_thickness: Railing-post cross-section side length (m).
        post_spacing: Gap between consecutive posts (m).
        post_height: Railing-post height above the top tread (m).
        nosing_radius_range: Rounded-nosing radius range (m).
        nosing_segments: Rounded-nosing arc tessellation.

    Returns:
        A ``decor`` dict: ``{float_thickness, nosing_radius, nosing_segments,
        left, right}`` where each side is
        ``{kind: "none"|"wall"|"posts", thickness, height, spacing}``.
    """

    def _side() -> dict:
        u = float(rng.random())
        if u < side_wall_frac:
            return {"kind": "wall", "thickness": float(wall_thickness),
                    "height": float(wall_height), "spacing": 0.0}
        if u < side_wall_frac + side_posts_frac:
            return {"kind": "posts", "thickness": float(post_thickness),
                    "height": float(post_height), "spacing": float(post_spacing)}
        return {"kind": "none", "thickness": 0.0, "height": 0.0, "spacing": 0.0}

    float_thickness = float(rng.uniform(*float_thick_range)) if float(rng.random()) < float_frac else 0.0
    nosing_radius = float(rng.uniform(*nosing_radius_range)) if float(rng.random()) < nosing_frac else 0.0
    left = _side()
    right = _side()
    return {
        "float_thickness": float_thickness,
        "nosing_radius": nosing_radius,
        "nosing_segments": int(nosing_segments),
        "left": left,
        "right": right,
    }


def normalize_box_variant_fracs(fracs: dict | None) -> dict:
    """Normalize a (possibly aliased / partial) box-variant weight dict.

    Args:
        fracs: Mapping of variant name (canonical or the ``center_leg`` /
            ``corner_legs`` aliases) to a non-negative weight.  ``None`` / empty /
            all-zero => ``{"solid": 1.0}``.

    Returns:
        A dict over all of :data:`BOX_VARIANTS` whose values sum to 1.0.

    Raises:
        ValueError: On an unknown variant name or a negative weight.
    """
    weights = {v: 0.0 for v in BOX_VARIANTS}
    for k, v in (fracs or {}).items():
        canon = _BOX_VARIANT_ALIASES.get(k, k)
        if canon not in weights:
            raise ValueError(f"unknown box variant {k!r}; valid: {BOX_VARIANTS} (+ aliases {_BOX_VARIANT_ALIASES})")
        if float(v) < 0.0:
            raise ValueError(f"box variant weight for {k!r} must be non-negative; got {v}")
        weights[canon] += float(v)
    total = sum(weights.values())
    if total <= 0.0:
        return {v: (1.0 if v == "solid" else 0.0) for v in BOX_VARIANTS}
    return {v: w / total for v, w in weights.items()}


def resolve_box_variant(
    rng: np.random.Generator,
    box_size: tuple[float, float, float],
    *,
    fracs: dict | None = None,
    slab_thickness: float = 0.1,
    leg_size: tuple[float, float] = (0.1, 0.1),
    leg_inset: float = 0.05,
) -> dict:
    """Resolve one box's support-structure variant into overlay manifest fields.

    The walkable top height is unchanged by the variant; only the geometry below it
    differs (a floating top, a floating top on a central leg, or on four corner
    legs).  Geometry params are clamped to the box footprint / height so degenerate
    slabs and oversized legs cannot be produced.

    Args:
        rng: A seeded ``numpy`` generator.
        box_size: The box's full extents ``(sx, sy, sz)`` (m).
        fracs: Variant weight dict (see :func:`normalize_box_variant_fracs`).
        slab_thickness: Desired top-slab thickness for floating variants (m).
        leg_size: Desired support-leg cross-section ``(lx, ly)`` (m).
        leg_inset: Corner-leg inset from the footprint edge (m).

    Returns:
        A dict of overlay fields ``{variant, slab_thickness, leg_size, leg_inset}``
        to merge into the box's overlay manifest entry.
    """
    sx, sy, sz = (float(v) for v in box_size)
    weights = normalize_box_variant_fracs(fracs)
    variant = str(rng.choice(BOX_VARIANTS, p=[weights[v] for v in BOX_VARIANTS]))

    # Clamp so the slab never eats the whole box and legs fit inside the footprint.
    slab = min(float(slab_thickness), 0.6 * sz)
    slab = max(slab, min(0.02, sz))
    lx = max(0.02, min(float(leg_size[0]), 0.45 * sx))
    ly = max(0.02, min(float(leg_size[1]), 0.45 * sy))
    inset = max(0.0, min(float(leg_inset), 0.45 * min(sx, sy)))
    return {"variant": variant, "slab_thickness": float(slab),
            "leg_size": [float(lx), float(ly)], "leg_inset": float(inset)}
