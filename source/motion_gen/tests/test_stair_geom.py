# Copyright 2026 Zachary Olkin. All rights reserved.

"""Unit tests for the shared isaaclab-free staircase tread layout (``stair_geom``).

This is the single source of truth for tread geometry -- the RL terrain block builder and the
offline clip generator both consume it, so these lock the conventions (depth tiling, z anchoring,
up/down ordering, offsets, partial last tread).
"""

from __future__ import annotations

import numpy as np

from motion_gen.motion_matching.stair_geom import stair_tread_layout


def test_depth_tiles_span_and_count():
    lay = stair_tread_layout(size_x=1.90, size_y=2.0, step_height=0.15, step_depth_nominal=0.19,
                             going_up=True)
    assert lay.num_steps == 10
    assert abs(lay.step_depth - 0.19) < 1e-6           # 1.90 / 10
    assert lay.partial_step_length is None
    # tread x-centers at (i+0.5)*d
    xs = lay.tread_top_centers[:, 0]
    assert np.allclose(xs, (np.arange(10) + 0.5) * lay.step_depth, atol=1e-5)


def test_up_vs_down_ordering():
    up = stair_tread_layout(size_x=1.0, size_y=1.0, step_height=0.1, step_depth_nominal=0.2,
                            going_up=True)      # 5 steps
    dn = stair_tread_layout(size_x=1.0, size_y=1.0, step_height=0.1, step_depth_nominal=0.2,
                            going_up=False)
    # up: stair 0 lowest, ascending; down: stair 0 highest, descending.
    assert up.tread_top_centers[0, 2] < up.tread_top_centers[-1, 2]
    assert dn.tread_top_centers[0, 2] > dn.tread_top_centers[-1, 2]
    # same set of tread heights either way (just reversed order)
    assert np.allclose(np.sort(up.tread_top_centers[:, 2]), np.sort(dn.tread_top_centers[:, 2]))


def test_centered_vs_start_z_zero():
    c = stair_tread_layout(size_x=1.0, size_y=1.0, step_height=0.1, step_depth_nominal=0.2,
                           going_up=True)                       # centered
    z = stair_tread_layout(size_x=1.0, size_y=1.0, step_height=0.1, step_depth_nominal=0.2,
                           going_up=True, start_z_zero=True)
    assert abs(c.z_lowest_top + c.z_highest_top) < 1e-6         # symmetric about 0
    assert abs(z.z_lowest_top) < 1e-6                            # lowest tread top at 0
    assert abs(z.z_highest_top - (z.num_steps - 1) * 0.1) < 1e-6


def test_offsets_shift_centers():
    lay = stair_tread_layout(size_x=1.0, size_y=2.0, step_height=0.1, step_depth_nominal=0.2,
                             going_up=True, x_offset=3.0, y_offset=1.0)
    assert np.allclose(lay.tread_top_centers[:, 1], 1.0 + 2.0 / 2.0)   # y = y_offset + size_y/2
    assert abs(float(lay.tread_top_centers[0, 0]) - (3.0 + 0.5 * lay.step_depth)) < 1e-5


def test_partial_last_step():
    lay = stair_tread_layout(size_x=1.05, size_y=1.0, step_height=0.1, step_depth_nominal=0.2,
                             going_up=True, allow_partial_last_step=True)
    assert lay.num_steps == 6                        # 5 full (0.2) + partial (0.05)
    assert abs(lay.step_depths[:-1].max() - 0.2) < 1e-6
    assert lay.partial_step_length is not None and abs(lay.partial_step_length - 0.05) < 1e-6
    assert abs(float(lay.step_depths[-1]) - 0.05) < 1e-6


def test_matches_build_stair_steps_if_available():
    """When IsaacLab is importable, the layout must reproduce ``_build_stair_steps`` centers."""
    try:
        from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair import _build_stair_steps
        from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg import (  # noqa: F401
            MeshXStairsUpTerrainCfg,
        )
    except Exception:
        import pytest
        pytest.skip("IsaacLab / robot_rl terrain configs not importable in this env")

    class _Cfg:
        step_dim_options = [(0.15, 0.19)]
        stair_width_range = (0.777, 0.777)
        start_z_zero = False
        allow_partial_last_step = False
        decor = None
        float_prob = wall_prob = pole_prob = 0.0

    _, centers, sh, sd, ns, _, _ = _build_stair_steps(
        _Cfg(), size_x=1.90, size_y=2.0, going_up=False, x_offset=0.5, y_offset=0.25,
        step_dim_override=(0.15, 0.19))
    lay = stair_tread_layout(size_x=1.90, size_y=2.0, step_height=0.15, step_depth_nominal=0.19,
                             going_up=False, x_offset=0.5, y_offset=0.25)
    assert ns == lay.num_steps and abs(sd - lay.step_depth) < 1e-6
    assert np.allclose(centers, lay.tread_top_centers, atol=1e-5)
