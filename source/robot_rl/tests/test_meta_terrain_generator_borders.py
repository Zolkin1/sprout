# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for inter-column border placement and metadata.

``_compute_border_column_info`` and ``_generate_flat_border_terrain`` only
read ``self.cfg``, so they're exercised on an uninitialised instance with a
stub cfg — no sim context needed.
"""

from __future__ import annotations

from types import SimpleNamespace

from robot_rl.tasks.manager_based.robot_rl.terrains.meta_terrain_generator import (
    MetaTerrainGenerator,
)


def _stub_cfg(
    num_cols: int,
    proportions: dict[str, float],
    borders: list[tuple[str, float]],
    border_skill_probs: dict[str, float] | None = None,
) -> SimpleNamespace:
    sub_terrains = {
        name: SimpleNamespace(proportion=p) for name, p in proportions.items()
    }
    return SimpleNamespace(
        num_cols=num_cols,
        size=(20.0, 10.0),
        sub_terrains=sub_terrains,
        inter_column_borders=borders,
        border_skill_probs=border_skill_probs or {"walk_forward": 0.5, "standing": 0.5},
    )


def _gen_with_cfg(cfg: SimpleNamespace) -> MetaTerrainGenerator:
    gen = object.__new__(MetaTerrainGenerator)
    gen.cfg = cfg
    return gen


class TestBorderColumnPlacement:
    def test_no_borders(self):
        cfg = _stub_cfg(22, {"flat": 0.5, "stairs": 0.5}, borders=[])
        gen = _gen_with_cfg(cfg)
        border_cols, content_n, kinds = gen._compute_border_column_info(cfg)
        assert border_cols == set()
        assert content_n == 22
        assert all(k == "content" for k in kinds.values())

    def test_boundary_matches_column_assignment_rule(self):
        """Regression: with a 0.6/0.4 split over 22 content columns, the
        column-assignment rule gives the first sub-terrain 14 columns
        (index/22 + 0.001 < 0.6 holds through index 13).  The buffer must sit
        after ALL of them — the old ``int(0.6 * 22) = 13`` boundary stranded
        one flat column on the stair side."""
        cfg = _stub_cfg(
            29, {"pure_flat": 0.6, "pure_stair_up": 0.4},
            borders=[("pure_flat", 70.0)],
        )
        gen = _gen_with_cfg(cfg)
        border_cols, content_n, kinds = gen._compute_border_column_info(cfg)

        assert content_n == 22
        assert len(border_cols) == 7  # ceil(70 / 10)
        # Columns 0-13 content (flat), 14-20 border, 21-28 content (stairs).
        assert border_cols == set(range(14, 21))
        assert all(kinds[c] == "content" for c in range(0, 14))
        assert all(kinds[c] == "fixed_border" for c in range(14, 21))
        assert all(kinds[c] == "content" for c in range(21, 29))

    def test_zero_proportion_subterrain(self):
        """A trailing zero-proportion sub-terrain (e.g. the disabled
        composite cells) must not shift the buffer placement."""
        cfg = _stub_cfg(
            29,
            {"pure_flat": 0.6, "pure_stair_up": 0.4, "flat_stair_up": 0.0},
            borders=[("pure_flat", 70.0)],
        )
        gen = _gen_with_cfg(cfg)
        border_cols, content_n, _ = gen._compute_border_column_info(cfg)
        assert content_n == 22
        assert border_cols == set(range(14, 21))


class TestBorderMetadata:
    def test_border_skill_probs_propagate(self):
        probs = {"walk_forward": 0.35, "running": 0.35, "walk_backwards": 0.2, "standing": 0.1}
        cfg = _stub_cfg(
            29, {"pure_flat": 0.6, "pure_stair_up": 0.4},
            borders=[("pure_flat", 70.0)],
            border_skill_probs=probs,
        )
        gen = _gen_with_cfg(cfg)
        _, _, meta_data = gen._generate_flat_border_terrain()
        assert meta_data["is_border"] is True
        assert meta_data["skill_probs"] == probs
        # The generator must hand out a copy, not the cfg's dict.
        meta_data["skill_probs"]["standing"] = 0.0
        assert cfg.border_skill_probs["standing"] == 0.1
