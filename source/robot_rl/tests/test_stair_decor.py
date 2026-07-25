# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for resolved stair decoration (floating treads + per-side walls / posts).

Exercises the deterministic ``decor`` path of ``trimesh/stair.py::_build_stair_steps``
(no RNG when ``cfg.decor`` is set) and the manifest round-trip helper
``clip_terrain_loader::_stair_decor_from_manifest``.  Isaac sim is not required.
"""

from __future__ import annotations

import numpy as np

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.stair import StairBlockCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
    _stair_decor_from_manifest,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair import _build_stair_steps
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_decor_cfg import (
    StairDecorCfg,
    StairSideDecorCfg,
)

_N_TREADS = 10  # 3.0 m span / 0.30 m depth


def _cfg(decor):
    return StairBlockCfg(size_x=3.0, step_dim_options=[(0.15, 0.30)],
                         stair_width_range=(1.0, 1.0), decor=decor)


def _build(decor):
    return _build_stair_steps(_cfg(decor), size_x=3.0, size_y=1.4, going_up=True)


def _verts(meshes):
    return np.concatenate([m.vertices for m in meshes])


def test_bare_decor_is_treads_only():
    meshes, _c, *_ = _build(StairDecorCfg())
    assert len(meshes) == _N_TREADS


def test_decor_is_deterministic():
    """Explicit decor => byte-identical mesh across calls (no RNG on the resolved path)."""
    d = StairDecorCfg(
        left=StairSideDecorCfg(kind="wall", thickness=0.1, height=1.0),
        right=StairSideDecorCfg(kind="posts", thickness=0.06, height=1.0, spacing=0.4),
    )
    m1, c1, *_ = _build(d)
    m2, c2, *_ = _build(d)
    assert _verts(m1).shape == _verts(m2).shape
    assert np.allclose(_verts(m1), _verts(m2))
    assert np.allclose(c1, c2)


def test_per_side_independent():
    """A wall on the -y side only: one extra mesh, on the low-y edge, none on +y."""
    meshes, _c, *_ = _build(StairDecorCfg(left=StairSideDecorCfg(kind="wall"),
                                          right=StairSideDecorCfg(kind="none")))
    assert len(meshes) == _N_TREADS + 1
    wall = meshes[-1]
    assert float(wall.centroid[1]) < 0.7          # y_center; wall is on the -y (left) side


def test_wall_and_posts_both_sides_counts():
    d = StairDecorCfg(
        left=StairSideDecorCfg(kind="wall"),
        right=StairSideDecorCfg(kind="posts", thickness=0.06, spacing=0.4),
    )
    meshes, _c, *_ = _build(d)
    # 10 treads + 1 wall + posts marched over 3.0 m at (thickness 0.06 + spacing 0.4).
    n_posts = 0
    x = 0.0
    while x + 0.06 <= 3.0 + 1e-5:
        n_posts += 1
        x += 0.06 + 0.4
    assert len(meshes) == _N_TREADS + 1 + n_posts


def test_floating_treads_thin_tops_unchanged():
    """Floating treads are thin slabs but leave the tread-top centers unchanged."""
    _mb, cb, *_ = _build(StairDecorCfg())
    mf, cf, *_ = _build(StairDecorCfg(float_thickness=0.05))
    assert np.allclose(cb, cf)                     # motion-relevant tread tops identical
    for tread in mf:
        assert abs(float(tread.bounding_box.extents[2]) - 0.05) < 1e-6


def test_rounded_nosing_adds_lip_without_moving_tops():
    """A rounded nosing adds one lip mesh per tread; tread tops stay put; radius clamps."""
    _mb, cb, *_ = _build(StairDecorCfg())
    mn, cn, *_ = _build(StairDecorCfg(nosing_radius=0.04, nosing_segments=6))
    assert len(mn) == 2 * _N_TREADS               # one lip per tread
    assert np.allclose(cb, cn)                    # tread-top centers unchanged
    # Radius is clamped to <= min(step_height, step_depth)/2 = min(0.15, 0.30)/2.
    from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair import _rounded_nosing
    lip = _rounded_nosing(0.0, 0.7, 0.15, 1.0, radius=5.0, segments=6, step_height=0.15, step_depth=0.30)
    r_eff = 0.5 * float(lip.bounding_box.extents[2])
    assert r_eff <= 0.075 + 1e-6
    assert _rounded_nosing(0.0, 0.7, 0.15, 1.0, radius=0.0, segments=6,
                           step_height=0.15, step_depth=0.30) is None


def test_manifest_round_trip():
    decor_dict = {
        "float_thickness": 0.05, "nosing_radius": 0.0, "nosing_segments": 6,
        "left": {"kind": "wall", "thickness": 0.1, "height": 1.0, "spacing": 0.0},
        "right": {"kind": "posts", "thickness": 0.06, "height": 1.0, "spacing": 0.4},
    }
    d = _stair_decor_from_manifest(decor_dict)
    assert d.float_thickness == 0.05
    assert d.left.kind == "wall" and d.left.thickness == 0.1
    assert d.right.kind == "posts" and d.right.spacing == 0.4
    # No decor key (bare / pre-decor manifests) => None => legacy solid stairs.
    assert _stair_decor_from_manifest(None) is None
    assert _stair_decor_from_manifest({}) is None
