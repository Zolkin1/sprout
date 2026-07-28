# Copyright 2026 Zachary Olkin. All rights reserved.

"""Unit tests for the composable terrain-blocks system.

These tests exercise the block geometry, the composite assembler, and the
the composite assembler's per-block AABB / stair-span geometry and
``skill_probs_at`` / ``_project_world`` queries. Isaac sim is not required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import (
    BlockChoice,
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    RandomizedCompositeSubTerrainCfg,
    SlopeBlockCfg,
    StairBlockCfg,
    composite_terrain,
    randomized_composite_terrain,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.velocity_guidance_cfg import (
    BlockVelocityGuidanceCfg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_cfg(size_xy: tuple[float, float]):
    """Build a minimal stand-in for the importer's ``self.cfg`` attribute."""
    class _FakeGen:
        size = size_xy

    class _FakeImporterCfg:
        terrain_generator = _FakeGen()

    return _FakeImporterCfg()


def _stair_up_cfg(size_x: float = 4.0) -> StairBlockCfg:
    """Return a deterministic stair-up cfg for tests.

    Uses ``step_depth = 0.5`` so the default ``size_x = 4.0`` divides
    exactly into 8 treads (the new composite ``StairBlock`` requires
    ``size_x`` to be an integer multiple of ``step_depth``).
    """
    return StairBlockCfg(
        size_x=size_x,
        direction="up",
        skill_probs={"stair_up": 1.0},
        step_dim_options=[(0.2, 0.5)],
        stair_width_range=(1.5, 1.5),
    )


def _flat_cfg(size_x: float = 3.0, skill_probs: dict | None = None) -> FlatBlockCfg:
    """Return a flat-block cfg."""
    sp = skill_probs or {"walk_forward": 1.0}
    return FlatBlockCfg(size_x=size_x, skill_probs=sp)


# ---------------------------------------------------------------------------
# Block geometry
# ---------------------------------------------------------------------------


def test_flat_block_aabb_and_origin():
    cfg = _flat_cfg(size_x=2.5)
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(1.0, 0.0), subterrain_size_y=3.0)
    assert out.aabb == (1.0, 3.5, 0.0, 3.0)
    assert out.needs_projection is False
    assert out.needs_directional_cmd is False
    # Default flat block (no flat_width_range) spans full subterrain y.
    assert out.extras == {"flat_width": 3.0}
    np.testing.assert_allclose(out.origin, [2.25, 1.5, 0.0])
    assert len(out.meshes) == 1


def test_stair_block_extras_and_aabb():
    np.random.seed(0)
    cfg = _stair_up_cfg(size_x=4.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(2.0, 0.0), subterrain_size_y=2.0)

    assert out.aabb == (2.0, 6.0, 0.0, 2.0)
    assert out.needs_projection is True
    assert out.needs_directional_cmd is True

    extras = out.extras
    assert extras["is_stair"] is True
    assert extras["direction"] == "up"
    centers = extras["stair_top_centers"]
    assert centers.shape[1] == 3
    # All centers shifted by x_offset=2.0.
    assert torch.all(centers[:, 0] >= 2.0)
    # Step depth divides the block exactly.
    step_depth = float(extras["stair_dimension"][1])
    assert pytest.approx(step_depth * extras["num_steps"], rel=1e-6) == 4.0


# ---------------------------------------------------------------------------
# Composite assembler
# ---------------------------------------------------------------------------


def test_composite_rejects_bad_size_sum():
    cfg = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_flat_cfg(4.0), _flat_cfg(4.0)],
    )
    with pytest.raises(ValueError, match="does not match"):
        composite_terrain(0.0, cfg)


def test_composite_rejects_empty_blocks():
    cfg = CompositeSubTerrainCfg(size=(10.0, 2.0), proportion=1.0, blocks=[])
    with pytest.raises(ValueError, match="at least one block"):
        composite_terrain(0.0, cfg)


def test_composite_rejects_bad_skill_probs():
    cfg = CompositeSubTerrainCfg(
        size=(2.0, 2.0),
        proportion=1.0,
        blocks=[_flat_cfg(2.0, skill_probs={"walk_forward": 0.5})],  # sums to 0.5
    )
    with pytest.raises(ValueError, match="sums to"):
        composite_terrain(0.0, cfg)


def test_composite_three_blocks():
    np.random.seed(1)
    cfg = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0), _stair_up_cfg(4.0)],
    )
    meshes, origin, md = composite_terrain(0.5, cfg)
    assert len(md["blocks"]) == 3
    assert md["blocks"][0]["aabb"] == (0.0, 4.0, 0.0, 2.0)
    assert md["blocks"][1]["aabb"] == (4.0, 6.0, 0.0, 2.0)
    assert md["blocks"][2]["aabb"] == (6.0, 10.0, 0.0, 2.0)
    assert md["blocks"][0]["needs_projection"] is True
    assert md["blocks"][1]["needs_projection"] is False
    assert md["blocks"][2]["needs_projection"] is True
    assert origin is not None
    assert len(meshes) > 1


# ---------------------------------------------------------------------------
# Z chaining across blocks
# ---------------------------------------------------------------------------


def _stair_down_cfg(size_x: float = 4.0) -> StairBlockCfg:
    """Stair-down variant for chaining tests."""
    cfg = _stair_up_cfg(size_x)
    cfg.direction = "down"
    cfg.skill_probs = {"stair_down": 1.0}
    return cfg


def test_flat_block_honors_base_z():
    """A flat block's mesh and metadata sit at the requested base_z."""
    np.random.seed(0)
    cfg = _flat_cfg(size_x=2.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=1.5,
    )
    assert out.entry_z == pytest.approx(1.5)
    assert out.exit_z == pytest.approx(1.5)
    assert out.origin[2] == pytest.approx(1.5)
    z_min, z_max = out.meshes[0].bounds[:, 2]
    assert z_min == pytest.approx(1.5, abs=1e-5)
    assert z_max == pytest.approx(1.5, abs=1e-5)


def test_stair_up_honors_base_z():
    """Stair-up: leftmost tread sits one step above ``base_z``; ``entry_z =
    base_z`` (the connecting-ground level); ``exit_z = base_z + N * step_h``.
    """
    np.random.seed(0)
    cfg = _stair_up_cfg(size_x=4.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.5,
    )
    centers = out.extras["stair_top_centers"]
    step_h, _, _ = out.extras["stair_dimension"]
    num_steps = out.extras["num_steps"]
    assert float(centers[0, 2]) == pytest.approx(0.5 + step_h)
    assert float(centers[-1, 2]) == pytest.approx(0.5 + num_steps * step_h)
    assert out.entry_z == pytest.approx(0.5)
    assert out.exit_z == pytest.approx(0.5 + num_steps * step_h)


def test_stair_down_honors_base_z():
    """Stair-down: leftmost (highest) tread sits one step below ``base_z``;
    ``entry_z = base_z``; ``exit_z = base_z - N * step_h``.
    """
    np.random.seed(0)
    cfg = _stair_down_cfg(size_x=4.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=2.0,
    )
    centers = out.extras["stair_top_centers"]
    step_h, _, _ = out.extras["stair_dimension"]
    num_steps = out.extras["num_steps"]
    assert float(centers[0, 2]) == pytest.approx(2.0 - step_h)
    assert float(centers[-1, 2]) == pytest.approx(2.0 - num_steps * step_h)
    assert out.entry_z == pytest.approx(2.0)
    assert out.exit_z == pytest.approx(2.0 - num_steps * step_h)


def test_composite_threads_z_across_blocks():
    """stair-up → flat → stair-down forms a continuous elevation profile."""
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0), _stair_down_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, cfg)
    blocks = md["blocks"]
    assert blocks[0]["entry_z"] == pytest.approx(0.0)
    assert blocks[0]["exit_z"] > 0.0
    # Flat platform inherits the stair-up's exit height (chain in action).
    assert blocks[1]["entry_z"] == pytest.approx(blocks[0]["exit_z"])
    assert blocks[1]["exit_z"] == pytest.approx(blocks[0]["exit_z"])
    # Stair-down enters at the flat's height and descends back to 0.
    assert blocks[2]["entry_z"] == pytest.approx(blocks[1]["exit_z"])
    assert blocks[2]["exit_z"] == pytest.approx(0.0, abs=1e-5)


def test_composite_start_z_offsets_first_block():
    """``start_z`` shifts the entire chain so the first block starts elevated."""
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(4.0, 2.0),
        proportion=1.0,
        start_z=3.0,
        blocks=[_flat_cfg(4.0)],
    )
    _, origin, md = composite_terrain(0.5, cfg)
    assert md["blocks"][0]["entry_z"] == pytest.approx(3.0)
    assert md["blocks"][0]["exit_z"] == pytest.approx(3.0)
    assert origin[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Stair-block platforms
# ---------------------------------------------------------------------------


def _stair_up_with_platforms_cfg(
    size_x: float, start_plat: float, end_plat: float
) -> StairBlockCfg:
    cfg = _stair_up_cfg(size_x)
    cfg.start_platform_length = start_plat
    cfg.end_platform_length = end_plat
    return cfg


def test_stair_block_rejects_oversize_platforms():
    cfg = _stair_up_with_platforms_cfg(size_x=2.0, start_plat=1.0, end_plat=1.5)
    with pytest.raises(ValueError, match="no positive x extent"):
        cfg.class_type(cfg, difficulty=0.0).build(
            local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.0,
        )


def test_stair_block_platforms_shrink_stair_region():
    """Stairs occupy size_x - start_platform - end_platform; extras report boundaries."""
    np.random.seed(0)
    cfg = _stair_up_with_platforms_cfg(size_x=6.0, start_plat=1.0, end_plat=0.5)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.0,
    )
    # AABB still covers the full block.
    assert out.aabb == (0.0, 6.0, 0.0, 2.0)
    # Stair tread region boundaries.
    assert out.extras["start_platform_xmax"] == pytest.approx(1.0)
    assert out.extras["end_platform_xmin"] == pytest.approx(5.5)
    # Stairs filled the inner 4.5m, so the first tread should sit just past
    # the start platform.
    centers = out.extras["stair_top_centers"]
    step_d = out.extras["stair_dimension"][1]
    assert float(centers[0, 0]) >= 1.0
    assert float(centers[0, 0]) <= 1.0 + step_d + 1e-5


def test_stair_block_platforms_sit_at_entry_and_exit_z():
    """Start platform at entry_z (= base_z); end platform at exit_z."""
    np.random.seed(0)
    cfg = _stair_up_with_platforms_cfg(size_x=6.0, start_plat=1.0, end_plat=0.5)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.3,
    )
    # First inserted mesh is the start platform; last is the end platform.
    start_plat_mesh = out.meshes[0]
    end_plat_mesh = out.meshes[-1]
    # Each platform plate is essentially zero-thickness in z.
    sp_z = float(start_plat_mesh.bounds[0, 2])
    ep_z = float(end_plat_mesh.bounds[0, 2])
    assert sp_z == pytest.approx(out.entry_z, abs=1e-5)
    assert ep_z == pytest.approx(out.exit_z, abs=1e-5)


# ---------------------------------------------------------------------------
# Randomized composite sub-terrain
# ---------------------------------------------------------------------------


def _flat_choice_cfg(weight: float = 1.0) -> BlockChoice:
    return BlockChoice(
        cfg=FlatBlockCfg(skill_probs={"walk_forward": 1.0}),
        weight=weight,
    )


def _stair_choice_cfg(weight: float = 1.0) -> BlockChoice:
    return BlockChoice(
        cfg=StairBlockCfg(
            direction="up",
            skill_probs={"stair_up": 1.0},
            step_dim_options=[(0.135, 0.233)],
            stair_width_range=(1.5, 1.5),
        ),
        weight=weight,
    )


def _slope_choice_cfg(weight: float = 1.0) -> BlockChoice:
    return BlockChoice(
        cfg=SlopeBlockCfg(
            direction="up",
            skill_probs={"walk_forward": 1.0},
            rise_range=(0.3, 0.5),
            slope_width_range=(1.5, 1.5),
        ),
        weight=weight,
    )


def _randomized_cfg(size_x: float = 12.0, **overrides) -> RandomizedCompositeSubTerrainCfg:
    base = dict(
        proportion=1.0,
        size=(size_x, 2.0),
        origin_block_index=0,
        force_flat_origin=True,
        length_range=(1.0, 2.5),
        choices=[_flat_choice_cfg(2.0), _stair_choice_cfg(1.0), _slope_choice_cfg(1.0)],
    )
    base.update(overrides)
    return RandomizedCompositeSubTerrainCfg(**base)


def test_randomized_composite_length_sum():
    np.random.seed(0)
    cfg = _randomized_cfg(size_x=12.0)
    meshes, origin, meta = randomized_composite_terrain(0.0, cfg)
    total = sum(b.size_x for b in cfg.blocks)
    assert pytest.approx(total, abs=1e-5) == 12.0
    assert len(meshes) > 0
    assert origin is not None
    assert len(meta["blocks"]) == len(cfg.blocks)


def test_randomized_composite_deterministic():
    np.random.seed(42)
    cfg_a = _randomized_cfg(size_x=10.0)
    randomized_composite_terrain(0.0, cfg_a)
    types_a = [type(b).__name__ for b in cfg_a.blocks]
    lengths_a = [b.size_x for b in cfg_a.blocks]

    np.random.seed(42)
    cfg_b = _randomized_cfg(size_x=10.0)
    randomized_composite_terrain(0.0, cfg_b)
    types_b = [type(b).__name__ for b in cfg_b.blocks]
    lengths_b = [b.size_x for b in cfg_b.blocks]

    assert types_a == types_b
    np.testing.assert_allclose(lengths_a, lengths_b, atol=1e-6)


def test_randomized_composite_zero_weights_raises():
    np.random.seed(0)
    cfg = _randomized_cfg(
        force_flat_origin=False,
        choices=[_flat_choice_cfg(0.0), _stair_choice_cfg(0.0)],
    )
    with pytest.raises(ValueError, match="weights are zero"):
        randomized_composite_terrain(0.0, cfg)


def test_randomized_composite_empty_choices_raises():
    np.random.seed(0)
    cfg = _randomized_cfg(choices=[])
    with pytest.raises(ValueError, match="non-empty"):
        randomized_composite_terrain(0.0, cfg)


def test_randomized_composite_per_choice_length_override():
    np.random.seed(7)
    # Flat block with a tight length-range override — exercises the
    # per-choice length_range path independently of stair rounding (which
    # always rounds DOWN to N*step_depth and so can't be used to assert a
    # tight upper bound on sampled length).
    tight_flat = BlockChoice(
        cfg=FlatBlockCfg(skill_probs={"walk_forward": 1.0}),
        weight=1.0,
        length_range=(0.8, 0.81),
    )
    cfg = _randomized_cfg(
        size_x=10.0,
        force_flat_origin=False,
        choices=[tight_flat],
        length_range=(5.0, 5.0),  # global range, ignored by tight_flat
    )
    randomized_composite_terrain(0.0, cfg)
    for b in cfg.blocks[:-1]:
        # Every non-tail block must respect the tight override (tail may
        # absorb the remainder via the stretch policy).
        assert 0.8 - 1e-6 <= b.size_x <= 0.81 + 1e-6


def test_randomized_composite_force_flat_origin():
    # The forced spawn block must inherit the configured FlatBlockCfg from
    # ``choices`` (skill_probs, flat_width_range, etc.) — not the legacy
    # ``FlatBlockCfg()`` default, which would silently re-introduce
    # running-eligible flat skill_probs on the spawn block.
    custom_skill_probs = {"walk_forward": 0.9, "standing": 0.1}
    for seed in (1, 2, 3, 4, 5):
        np.random.seed(seed)
        cfg = _randomized_cfg(
            size_x=8.0,
            force_flat_origin=True,
            choices=[
                BlockChoice(
                    cfg=FlatBlockCfg(skill_probs=dict(custom_skill_probs)),
                    weight=1.0,
                ),
                _stair_choice_cfg(1.0),
            ],
            length_range=(1.0, 2.0),
        )
        randomized_composite_terrain(0.0, cfg)
        assert isinstance(cfg.blocks[0], FlatBlockCfg)
        assert cfg.blocks[0].skill_probs == custom_skill_probs


def test_randomized_composite_force_flat_origin_without_flat_choice_raises():
    # With ``force_flat_origin=True`` and no FlatBlockCfg in ``choices`` the
    # builder used to silently fall back to a default FlatBlockCfg (which
    # advertises running). Now it surfaces the inconsistency.
    np.random.seed(0)
    cfg = _randomized_cfg(
        size_x=8.0,
        force_flat_origin=True,
        choices=[_stair_choice_cfg(1.0)],
        length_range=(1.0, 2.0),
    )
    with pytest.raises(ValueError, match="FlatBlockCfg"):
        randomized_composite_terrain(0.0, cfg)


def test_randomized_composite_distribution_sanity():
    np.random.seed(123)
    type_counts = {"FlatBlockCfg": 0, "StairBlockCfg": 0}
    # Use disjoint per-choice length so we tile evenly.
    choices = [
        BlockChoice(
            cfg=FlatBlockCfg(skill_probs={"walk_forward": 1.0}),
            weight=3.0,
        ),
        BlockChoice(
            cfg=StairBlockCfg(
                direction="up",
                skill_probs={"stair_up": 1.0},
                step_dim_options=[(0.135, 0.233)],
                stair_width_range=(1.5, 1.5),
            ),
            weight=1.0,
        ),
    ]
    for _ in range(200):
        cfg = RandomizedCompositeSubTerrainCfg(
            proportion=1.0,
            size=(20.0, 2.0),
            origin_block_index=0,
            force_flat_origin=False,
            length_range=(1.0, 1.0001),
            choices=choices,
        )
        randomized_composite_terrain(0.0, cfg)
        for b in cfg.blocks:
            type_counts[type(b).__name__] += 1
    total = sum(type_counts.values())
    flat_frac = type_counts["FlatBlockCfg"] / total
    # Expected 0.75, allow ±0.05 with 200 * ~20 = 4000 samples.
    assert abs(flat_frac - 0.75) < 0.05


# ---------------------------------------------------------------------------
# Stair partial cut-off
# ---------------------------------------------------------------------------


def test_stair_block_exact_tiling_no_partial():
    """Composite ``StairBlock`` requires ``size_x = N * step_depth`` exactly
    and never emits a partial trailing tread. Each tread is full depth.
    """
    np.random.seed(0)
    cfg = StairBlockCfg(
        size_x=3.0,  # exactly 3 treads of step_depth 1.0
        direction="up",
        skill_probs={"stair_up": 1.0},
        step_dim_options=[(0.15, 1.0)],
        stair_width_range=(1.5, 1.5),
    )
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0)
    extras = out.extras
    assert extras["num_steps"] == 3
    step_depth = float(extras["stair_dimension"][1])
    assert pytest.approx(step_depth, abs=1e-6) == 1.0
    # No partial tread ever.
    assert extras["partial_step_length"] is None
    # entry_z is the connecting ground; exit_z is N * step_h above it.
    assert pytest.approx(out.entry_z, abs=1e-6) == 0.0
    assert pytest.approx(out.exit_z, abs=1e-6) == 3 * 0.15


def test_stair_block_size_x_not_multiple_raises():
    """If ``size_x`` is not an integer multiple of ``step_depth``, the
    composite StairBlock raises rather than silently rescaling.
    """
    np.random.seed(0)
    cfg = StairBlockCfg(
        size_x=3.1,
        direction="up",
        skill_probs={"stair_up": 1.0},
        step_dim_options=[(0.15, 1.0)],
        stair_width_range=(1.5, 1.5),
    )
    with pytest.raises(ValueError, match="integer multiple of step_depth"):
        cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0)


def test_stair_block_chain_continuous_staircase():
    """Two back-to-back stair blocks chain into a single continuous staircase:
    block B's first tread sits one step above block A's last tread.
    """
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(6.0, 2.0),
        proportion=1.0,
        blocks=[
            StairBlockCfg(
                size_x=3.0,  # 3 treads
                direction="up",
                skill_probs={"stair_up": 1.0},
                step_dim_options=[(0.15, 1.0)],
                stair_width_range=(1.5, 1.5),
            ),
            StairBlockCfg(
                size_x=3.0,  # 3 more treads
                direction="up",
                skill_probs={"stair_up": 1.0},
                step_dim_options=[(0.15, 1.0)],
                stair_width_range=(1.5, 1.5),
            ),
        ],
    )
    _, _, md = composite_terrain(0.0, cfg)
    a = md["blocks"][0]
    b = md["blocks"][1]
    # Block A's exit_z = base_z + N_A * step_h; block B's first tread top =
    # A.exit_z + step_h (the chain "continues" without any flat plateau).
    a_centers = a["extras"]["stair_top_centers"]
    b_centers = b["extras"]["stair_top_centers"]
    step_h = a["extras"]["stair_dimension"][0]
    assert pytest.approx(b["entry_z"], abs=1e-6) == a["exit_z"]
    assert pytest.approx(float(a_centers[-1, 2]), abs=1e-6) == a["exit_z"]
    assert pytest.approx(float(b_centers[0, 2]), abs=1e-6) == a["exit_z"] + step_h


# ---------------------------------------------------------------------------
# Slope block
# ---------------------------------------------------------------------------


def test_slope_block_aabb_and_z_thread():
    np.random.seed(0)
    cfg = SlopeBlockCfg(
        size_x=2.0,
        direction="up",
        skill_probs={"walk_forward": 1.0},
        rise_range=(0.5, 0.5),
        slope_width_range=(1.5, 1.5),
    )
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(1.0, 0.0), subterrain_size_y=2.0)
    assert out.aabb == (1.0, 3.0, 0.0, 2.0)
    assert out.needs_projection is False
    assert out.needs_directional_cmd is True
    assert pytest.approx(out.exit_z - out.entry_z, abs=1e-6) == 0.5
    assert out.extras["is_slope"] is True
    assert out.extras["direction"] == "up"


def test_slope_block_with_platforms():
    np.random.seed(0)
    cfg = SlopeBlockCfg(
        size_x=4.0,
        direction="down",
        skill_probs={"walk_forward": 1.0},
        rise_range=(0.4, 0.4),
        slope_width_range=(1.5, 1.5),
        start_platform_length=0.5,
        end_platform_length=0.5,
    )
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0)
    # 1 platform + 1 slope + 1 platform = 3 meshes.
    assert len(out.meshes) == 3
    # entry_z stays at base_z; exit_z descends.
    assert pytest.approx(out.entry_z, abs=1e-6) == 0.0
    assert pytest.approx(out.exit_z, abs=1e-6) == -0.4
    # Platforms shrink slope region.
    assert pytest.approx(out.extras["start_platform_xmax"], abs=1e-6) == 0.5
    assert pytest.approx(out.extras["end_platform_xmin"], abs=1e-6) == 3.5


def test_slope_block_inside_randomized_composite():
    np.random.seed(0)
    cfg = RandomizedCompositeSubTerrainCfg(
        proportion=1.0,
        size=(15.0, 2.0),
        origin_block_index=0,
        force_flat_origin=True,
        length_range=(2.0, 3.0),
        choices=[
            _flat_choice_cfg(1.0),
            _stair_choice_cfg(1.0),
            _slope_choice_cfg(1.0),
        ],
    )
    meshes, origin, meta = randomized_composite_terrain(0.0, cfg)
    assert any(b["block_type"] == "SlopeBlock" for b in meta["blocks"])
    # z-threading invariant: each block's entry_z equals previous exit_z.
    for prev, curr in zip(meta["blocks"], meta["blocks"][1:]):
        assert pytest.approx(curr["entry_z"], abs=1e-6) == prev["exit_z"]




# ---------------------------------------------------------------------------
# Velocity guidance
# ---------------------------------------------------------------------------

def _guided_flat_cfg(size_x: float = 3.0) -> FlatBlockCfg:
    """Flat block declaring end-of-block velocity guidance."""
    return FlatBlockCfg(
        size_x=size_x,
        skill_probs={"walk_forward": 1.0},
        velocity_guidance=BlockVelocityGuidanceCfg(
            lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.5, 0.5),
            y_kp=1.5, y_kd=0.3, heading_kp=0.5,
        ),
    )


def test_velocity_guidance_packed_into_block_metadata():
    """``composite_terrain`` carries the block cfg's guidance through to the
    per-block metadata dicts; unguided blocks carry ``None``."""
    sub = CompositeSubTerrainCfg(
        size=(6.0, 2.0),
        proportion=1.0,
        blocks=[_guided_flat_cfg(3.0), _flat_cfg(3.0)],
    )
    _, _, md = composite_terrain(0.0, sub)
    assert md["blocks"][0]["velocity_guidance"] is not None
    assert md["blocks"][1]["velocity_guidance"] is None


