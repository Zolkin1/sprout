# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

from dataclasses import field

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.flat import flat_terrain


def _default_flat_skill_probs() -> dict[str, float]:
    """Legacy flat-cell skill distribution (no walk_backwards)."""
    return {"walk_forward": 0.45, "running": 0.45, "standing": 0.10}


@configclass
class MeshFlatTerrainCfg(SubTerrainBaseCfg):
    """Flat plane filling the entire sub-terrain cell, centered at z=0.

    Mirrors the ``MeshXStairsUp/Down`` pair: returns the meta-system 3-tuple
    so it plugs directly into :class:`MetaTerrainGenerator` and exposes
    explicit ``meta_data`` (``is_border=False`` so envs can spawn here).
    """

    function = flat_terrain

    skill_probs: dict[str, float] = field(default_factory=_default_flat_skill_probs)
    """Skill distribution advertised by the cell's metadata. Must sum to 1.0
    and every key must be in the importer's ``skill_list``."""
