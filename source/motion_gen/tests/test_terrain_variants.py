# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the terrain-variant resolution policy (stair decor + box support-legs).

This module is deliberately IsaacLab-free (pure numpy), so it also asserts the
resolvers can be imported and run without IsaacLab installed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from motion_gen.motion_matching.terrain_variants import (
    BOX_VARIANTS,
    normalize_box_variant_fracs,
    resolve_box_variant,
    resolve_stair_decor,
)


def test_stair_decor_disabled_is_bare():
    """All-zero fracs => no floating, no nosing, both sides bare."""
    d = resolve_stair_decor(np.random.default_rng(0))
    assert d["float_thickness"] == 0.0
    assert d["nosing_radius"] == 0.0
    assert d["left"]["kind"] == "none" and d["right"]["kind"] == "none"


def test_stair_decor_sides_independent_proportions():
    """Each side is drawn independently at the requested per-side fractions."""
    rng = np.random.default_rng(1)
    left = Counter()
    right = Counter()
    n = 8000
    for _ in range(n):
        d = resolve_stair_decor(rng, side_wall_frac=0.5, side_posts_frac=0.3)
        left[d["left"]["kind"]] += 1
        right[d["right"]["kind"]] += 1
    for side in (left, right):
        assert abs(side["wall"] / n - 0.5) < 0.03
        assert abs(side["posts"] / n - 0.3) < 0.03
        assert abs(side["none"] / n - 0.2) < 0.03


def test_stair_decor_deterministic_and_geometry():
    """Same seed => identical dict; geometry fields come from the params."""
    kw = dict(side_wall_frac=1.0, wall_thickness=0.12, wall_height=1.3)
    a = resolve_stair_decor(np.random.default_rng(7), **kw)
    b = resolve_stair_decor(np.random.default_rng(7), **kw)
    assert a == b
    assert a["left"]["kind"] == "wall" and a["left"]["thickness"] == 0.12 and a["left"]["height"] == 1.3


def test_box_fracs_normalize_and_aliases():
    w = normalize_box_variant_fracs({"solid": 1, "center_leg": 1, "corner_legs": 2})
    assert set(w) == set(BOX_VARIANTS)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["floating_center_leg"] == 0.25 and w["floating_corner_legs"] == 0.5
    # Empty / all-zero => solid.
    assert normalize_box_variant_fracs(None)["solid"] == 1.0
    assert normalize_box_variant_fracs({})["solid"] == 1.0


def test_box_variant_proportions_and_determinism():
    rng = np.random.default_rng(2)
    fracs = {"solid": 0.25, "floating": 0.25, "center_leg": 0.25, "corner_legs": 0.25}
    counts = Counter(resolve_box_variant(rng, (0.6, 0.6, 0.4), fracs=fracs)["variant"] for _ in range(8000))
    assert set(counts) == set(BOX_VARIANTS)
    for v in BOX_VARIANTS:
        assert abs(counts[v] / 8000 - 0.25) < 0.03
    a = resolve_box_variant(np.random.default_rng(9), (0.6, 0.6, 0.4), fracs={"floating": 1})
    b = resolve_box_variant(np.random.default_rng(9), (0.6, 0.6, 0.4), fracs={"floating": 1})
    assert a == b


def test_box_variant_geometry_clamped_to_box():
    """Slab / legs are clamped so they never exceed a small box's height / footprint."""
    vp = resolve_box_variant(
        np.random.default_rng(0), (0.2, 0.2, 0.1), fracs={"corner_legs": 1},
        slab_thickness=0.5, leg_size=(0.3, 0.3), leg_inset=0.5,
    )
    assert vp["variant"] == "floating_corner_legs"
    assert vp["slab_thickness"] <= 0.6 * 0.1 + 1e-9         # <= 60% of height
    assert vp["leg_size"][0] <= 0.45 * 0.2 + 1e-9           # <= 45% of footprint
    assert vp["leg_inset"] <= 0.45 * 0.2 + 1e-9


def test_unknown_box_variant_raises():
    import pytest
    with pytest.raises(ValueError):
        normalize_box_variant_fracs({"triangle": 1.0})
