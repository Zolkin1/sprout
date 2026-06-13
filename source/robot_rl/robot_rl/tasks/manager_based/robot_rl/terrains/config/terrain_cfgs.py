# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import (
    BlockChoice,
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    RandomizedCompositeSubTerrainCfg,
    SlopeBlockCfg,
    StairBlockCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_terrain_generator_cfg import (
    MetaTerrainGeneratorCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.flat_cfg import (
    MeshFlatTerrainCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg import (
    MeshXStairsDownTerrainCfg,
    MeshXStairsUpTerrainCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.velocity_guidance_cfg import (
    BlockVelocityGuidanceCfg,
)

# Randomized multiskill terrain: each sub-terrain cell is a fresh random walk
# along x mixing flat, stair-up (with partial trailing tread), stair-down, and
# slopes. Sub-terrain cells are placed directly adjacent in y; if a gap is
# desired, set the per-block width range smaller than the cell's y extent so
# empty strips appear inside each cell instead of as separate columns.
_RANDOM_SIZE_X = 20.0 #12.0
_RANDOM_SIZE_Y = 10.0
_RANDOM_STAIR_STEP_DIM_OPTIONS = [(0.10, 0.30), (0.14, 0.28), (0.16, 0.32)]
_RANDOM_NUM_COLS = 4

# Fallback stair (step_height, step_depth) for ``FLAT_STAIR_MULTISKILL_CFG``.
# The terrain-multiskill env cfg overrides ``step_dim_options`` with the grid
# scanned from the trajectory library when stairs are active, so these only
# matter when that cfg is used without the scan.
_LONG_STAIRS_STEP_DEPTH = 0.233
_LONG_STAIRS_STEP_HEIGHT = 0.135

RANDOMIZED_MULTISKILL_TERRAIN_CFG = MetaTerrainGeneratorCfg(
    curriculum=False,
    size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
    border_width=0.0,
    num_rows=4,
    num_cols=_RANDOM_NUM_COLS,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    inter_column_borders=[],
    sub_terrains={
        "randomized": RandomizedCompositeSubTerrainCfg(
            proportion=1.0,
            size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
            origin_block_index=0,
            force_flat_origin=True,
            length_range=(1.0, 3.0),
            trailing_flat_length_range=(1.5, 3.0),
            choices=[
                BlockChoice(
                    cfg=FlatBlockCfg(
                        skill_probs={"walk_forward": 0.5, "running": 0.4, "standing": 0.1},
                        flat_width_range=(1.5, _RANDOM_SIZE_Y),
                    ),
                    weight=2.0,
                ),
                BlockChoice(
                    cfg=StairBlockCfg(
                        direction="up",
                        skill_probs={"stair_up": 1.0},
                        step_dim_options=_RANDOM_STAIR_STEP_DIM_OPTIONS,
                        stair_width_range=(1.5, _RANDOM_SIZE_Y),
                        float_prob=0.0,
                        wall_prob=0.0,
                        pole_prob=0.0,
                    ),
                    weight=1.0,
                    length_range=(1.5, 4.0),
                ),
                BlockChoice(
                    cfg=StairBlockCfg(
                        direction="down",
                        skill_probs={"stair_down": 1.0},
                        step_dim_options=_RANDOM_STAIR_STEP_DIM_OPTIONS,
                        stair_width_range=(1.5, _RANDOM_SIZE_Y),
                        float_prob=0.0,
                        wall_prob=0.0,
                        pole_prob=0.0,
                    ),
                    weight=1.0,
                    length_range=(1.5, 4.0),
                ),
                BlockChoice(
                    cfg=SlopeBlockCfg(
                        direction="up",
                        skill_probs={"walk_forward": 1.0},
                        rise_range=(0.2, 0.8),
                        slope_width_range=(1.5, _RANDOM_SIZE_Y),
                    ),
                    weight=0.5,
                    length_range=(1.5, 3.5),
                ),
                BlockChoice(
                    cfg=SlopeBlockCfg(
                        direction="down",
                        skill_probs={"walk_forward": 1.0},
                        rise_range=(0.2, 0.8),
                        slope_width_range=(1.5, _RANDOM_SIZE_Y),
                    ),
                    weight=0.5,
                    length_range=(1.5, 3.5),
                ),
            ],
        ),
    },
)
"""Randomized multiskill terrain with lateral void gaps."""


# Three-sub-terrain multiskill mix:
#   * ``pure_flat``     — legacy ``MeshFlatTerrainCfg`` (entire cell is one
#     flat plane).
#   * ``pure_stair_up`` — legacy ``MeshXStairsUpTerrainCfg`` (entire cell is
#     a single ascending staircase, full y width).
#   * ``flat_stair_up`` — composite cells that interleave flat and stair-up
#     blocks only (no stair-down, no slopes); the first block is forced to
#     flat so the spawn lands on level ground.
# The pure sub-terrains use the same legacy types as ``STAIR_WALK_CFG``;
# only the mixed sub-terrain needs the randomized composite (the only way to
# interleave block types within one cell).
# Flat skill distribution shared by the pure-flat cells and the buffer
# columns (the buffer must look like flat ground to the skill sampler).
_FLAT_SKILL_PROBS = {
    "walk_forward": 0.35,
    "running": 0.35,
    "walk_backwards": 0.2,
    "standing": 0.1,
}

FLAT_STAIR_MULTISKILL_CFG = MetaTerrainGeneratorCfg(
    curriculum=True,
    size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
    border_width=75.0,
    num_rows=10,
    # 22 content columns + 7 buffer columns from ``inter_column_borders``.
    num_cols=29,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    # 70 m flat buffer between the pure-flat and pure-stair column sections
    # so robots with lateral / yaw commands don't run off the stair-section
    # edge.  Buffer cells are unspawnable but carry flat skill metadata.
    inter_column_borders=[("pure_flat", 70.0)],
    border_skill_probs=dict(_FLAT_SKILL_PROBS),
    sub_terrains={
        "pure_flat": MeshFlatTerrainCfg(
            proportion=0.2,
            skill_probs=dict(_FLAT_SKILL_PROBS),
        ),
        "pure_stair_up": MeshXStairsUpTerrainCfg(
            proportion=0.2,
            # Fallback single size — the terrain-multiskill env cfg overrides
            # ``step_dim_options`` (on a deepcopy) with the stair grid scanned
            # from the trajectory library (see ``stair_dim_scan``).
            step_dim_options=[(_LONG_STAIRS_STEP_HEIGHT, _LONG_STAIRS_STEP_DEPTH)],
            # Lowest stair top sits at z=0 (default ``start_z_zero=True``);
            # spawn the env on stair 0 so the robot starts at ground level.
            origin_distance_from_back=0.5 * _LONG_STAIRS_STEP_DEPTH,
            stair_width_range=(1.5, _RANDOM_SIZE_Y),
            float_prob=0.5,
            wall_prob=0.0,
            pole_prob=0.0,
            # PD-steer toward the far edge of the cell (centered) instead of
            # sampling random lateral / yaw on the stairs.
            velocity_guidance=BlockVelocityGuidanceCfg(),
        ),
        "flat_stair_up": RandomizedCompositeSubTerrainCfg(
            proportion=0.6,
            size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
            origin_block_index=0,
            force_flat_origin=True,
            length_range=(1.0, 2.0),
            trailing_flat_length_range=(1.5, 3.0),
            choices=[
                BlockChoice(
                    cfg=FlatBlockCfg(
                        skill_probs={"walk_forward": 0.98, "running": 0.0, "standing": 0.02},
                        flat_width_range=(1.5, _RANDOM_SIZE_Y),
                        # Guided flat blocks keep robots centered + aligned
                        # (steering to their own exit-edge midpoint) so they
                        # enter the following stair block cleanly.
                        velocity_guidance=BlockVelocityGuidanceCfg(),
                    ),
                    weight=1.0,
                ),
                BlockChoice(
                    cfg=StairBlockCfg(
                        direction="up",
                        skill_probs={"stair_up": 1.0},
                        step_dim_options=[(_LONG_STAIRS_STEP_HEIGHT, _LONG_STAIRS_STEP_DEPTH)],
                        stair_width_range=(1.5, _RANDOM_SIZE_Y),
                        float_prob=0.5,
                        wall_prob=0.0,
                        pole_prob=0.0,
                        velocity_guidance=BlockVelocityGuidanceCfg(),
                    ),
                    weight=1.0,
                    length_range=(0.5, 2.0),
                ),
            ],
        ),
    },
)

##
# Flat Only
##
FLAT_MULTISKILL_CFG = MetaTerrainGeneratorCfg(
    curriculum=True,
    size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
    border_width=75.0,
    num_rows=10,
    # 22 content columns + 7 buffer columns from ``inter_column_borders``.
    num_cols=29,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    # 70 m flat buffer between the pure-flat and pure-stair column sections
    # so robots with lateral / yaw commands don't run off the stair-section
    # edge.  Buffer cells are unspawnable but carry flat skill metadata.
    inter_column_borders=[("pure_flat", 70.0)],
    border_skill_probs=dict(_FLAT_SKILL_PROBS),
    sub_terrains={
        "pure_flat": MeshFlatTerrainCfg(
            proportion=1.0,
            skill_probs=dict(_FLAT_SKILL_PROBS),
        ),
    },
)