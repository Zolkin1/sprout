# Copyright 2026 Zachary Olkin. All rights reserved.

"""Composite sub-terrain assembled from an ordered list of terrain blocks."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .base import TerrainBlockCfg, validate_skill_probs
from .overlay import OverlayVolumeCfg, build_overlay

if TYPE_CHECKING:
    pass

_LAYOUT_TOL = 1e-5


def composite_terrain(
    difficulty: float, cfg: "CompositeSubTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Assemble a sub-terrain from a list of terrain blocks.

    Blocks are placed left-to-right along x. Each block emits geometry in
    sub-terrain-local coordinates. The composer accumulates an x-offset,
    invokes each block's ``build``, concatenates the meshes, and packs
    per-block metadata into ``meta_data["blocks"]`` for the importer.

    Args:
        difficulty: Sub-terrain difficulty scalar in ``[0, 1]``.
        cfg: The composite sub-terrain configuration.

    Returns:
        Tuple ``(meshes, origin, meta_data)`` in sub-terrain-local
        coordinates. ``meta_data["blocks"]`` is a list of per-block dicts:
        ``{"aabb", "skill_probs", "needs_projection", "needs_directional_cmd",
        "extras"}``. ``meta_data["overlays"]`` is a list of per-overlay dicts
        ``{"kind", "aabb", "pos_xy", "size_xyz", "base_z", "walkable_top",
        "top_z"}`` (empty when the sub-terrain has no overlays).

    Raises:
        ValueError: If ``cfg.blocks`` is empty, the block ``size_x`` values
            don't sum to ``cfg.size[0]``, or any block emits invalid
            ``skill_probs``.
    """
    if not cfg.blocks:
        raise ValueError("CompositeSubTerrainCfg.blocks must contain at least one block.")

    size_x = float(cfg.size[0])
    size_y = float(cfg.size[1])

    total_size_x = float(sum(b.size_x for b in cfg.blocks))
    if abs(total_size_x - size_x) > _LAYOUT_TOL:
        raise ValueError(
            f"CompositeSubTerrainCfg: sum of block size_x ({total_size_x:.6f}) "
            f"does not match sub-terrain size_x ({size_x:.6f}) "
            f"(tol={_LAYOUT_TOL})."
        )

    if not 0 <= cfg.origin_block_index < len(cfg.blocks):
        raise ValueError(
            f"origin_block_index={cfg.origin_block_index} out of range "
            f"[0, {len(cfg.blocks)})."
        )

    all_meshes: list[trimesh.Trimesh] = []
    block_metadata: list[dict] = []
    origin: np.ndarray | None = None

    x_offset = 0.0
    current_z = float(cfg.start_z)
    for i, block_cfg in enumerate(cfg.blocks):
        validate_skill_probs(
            block_cfg.skill_probs, context=f"{type(block_cfg).__name__}[{i}]"
        )
        block = block_cfg.class_type(block_cfg, difficulty)
        out = block.build((x_offset, 0.0), size_y, base_z=current_z)

        # Sanity-check the block reported the AABB the composer placed.
        expected_aabb = (x_offset, x_offset + float(block_cfg.size_x), 0.0, size_y)
        if not np.allclose(out.aabb, expected_aabb, atol=_LAYOUT_TOL):
            raise ValueError(
                f"Block {i} ({type(block_cfg).__name__}) returned aabb {out.aabb} "
                f"but composer expected {expected_aabb}."
            )

        all_meshes.extend(out.meshes)
        # Default walkable_aabb to the full block footprint when the block
        # doesn't restrict its walkable y-extent.  The importer uses this
        # for the debug-viz outline rectangle so it hugs the actual block
        # geometry instead of the full sub-terrain.
        walkable = out.walkable_aabb if out.walkable_aabb is not None else out.aabb
        block_metadata.append({
            "aabb": tuple(float(v) for v in out.aabb),
            "walkable_aabb": tuple(float(v) for v in walkable),
            "skill_probs": dict(out.skill_probs),
            "needs_projection": bool(out.needs_projection),
            "needs_directional_cmd": bool(out.needs_directional_cmd),
            "entry_z": float(out.entry_z),
            "exit_z": float(out.exit_z),
            "extras": dict(out.extras),
            "block_type": type(block).__name__,
            "velocity_guidance": block_cfg.velocity_guidance,
        })

        if i == cfg.origin_block_index:
            if out.origin is not None:
                origin = np.array(out.origin, dtype=np.float64)
            else:
                origin = np.array(
                    [x_offset + block_cfg.size_x / 2.0, size_y / 2.0, current_z],
                    dtype=np.float64,
                )

        # Thread elevation forward so the next block's left edge sits flush
        # with this block's right edge.
        current_z = float(out.exit_z)
        x_offset += float(block_cfg.size_x)

    assert origin is not None  # set inside the loop on origin_block_index iteration.

    # Overlay collision volumes (boxes / obstacles / overhead bars) sit ON TOP of
    # the ground blocks at their own ``pos_xy`` -- they overlap the blocks in x
    # rather than partitioning it, so they're built after the left-to-right block
    # loop and merged into the same mesh list.
    overlay_metadata: list[dict] = []
    for ov_cfg in cfg.overlays:
        ov = build_overlay(ov_cfg)
        all_meshes.append(ov.mesh)
        overlay_metadata.append({
            "kind": ov.kind,
            "aabb": tuple(float(v) for v in ov.aabb),
            "pos_xy": (float(ov_cfg.pos_xy[0]), float(ov_cfg.pos_xy[1])),
            "size_xyz": tuple(float(v) for v in ov_cfg.size_xyz),
            "base_z": float(ov_cfg.base_z),
            "walkable_top": bool(ov.walkable_top),
            "top_z": float(ov.top_z),
        })

    meta_data = {
        "is_border": False,
        "blocks": block_metadata,
        "overlays": overlay_metadata,
    }
    # Pass through an optional per-cell terrain id (used by the clip-terrain
    # importer to map each grid cell -> the offline clip terrain it rebuilds, so
    # envs can be assigned a clip from their own terrain).  ``-1`` marks the flat
    # (no-terrain) cell, matching the flat clips' ``terrain_id``.
    if cfg.terrain_id is not None:
        meta_data["terrain_id"] = int(cfg.terrain_id)
    return all_meshes, origin, meta_data


@configclass
class CompositeSubTerrainCfg(SubTerrainBaseCfg):
    """Sub-terrain assembled from an ordered list of :class:`TerrainBlockCfg`.

    Blocks are laid out left-to-right along the sub-terrain's x-axis. Each
    block spans the full sub-terrain y extent. The sum of block ``size_x``
    values must equal ``size[0]`` of the sub-terrain (within ``1e-5``).
    """

    function = composite_terrain

    blocks: list[TerrainBlockCfg] = field(default_factory=list)
    """Ordered list of block configurations, placed left-to-right along x."""

    overlays: list[OverlayVolumeCfg] = field(default_factory=list)
    """Finite collision volumes (boxes / obstacles / overhead bars) placed on top
    of the ground blocks at their own ``pos_xy``. Unlike :attr:`blocks` these do
    not partition the x-axis -- they overlap it. Empty (default) = plain terrain."""

    origin_block_index: int = 0
    """Index of the block whose spawn origin is used for the sub-terrain.
    Defaults to the first block. The block must expose a non-``None`` origin;
    otherwise the composite falls back to that block's footprint center."""

    start_z: float = 0.0
    """Walkable-surface z at the first block's left edge. Successive blocks
    chain off the previous block's ``exit_z``, so this is the elevation of
    the whole sub-terrain's entry point."""

    terrain_id: int | None = None
    """Optional id tying this cell to an offline clip-library terrain. Passed
    through into ``composite_terrain``'s ``meta_data`` so the clip-terrain
    importer can expose a per-env terrain id and sample clips matching the cell.
    ``None`` (default) omits it; the clip loader sets it per terrain (``-1`` for
    the flat cell)."""
