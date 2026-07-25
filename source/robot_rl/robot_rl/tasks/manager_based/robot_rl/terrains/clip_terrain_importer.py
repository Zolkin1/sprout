# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain importer that assembles a clip-library terrain directly (no grid).

Instead of IsaacLab's uniform :class:`TerrainGeneratorCfg` grid (which forces one
footprint on every cell -> heavy flat padding), this importer builds the terrain
mesh from a clip library's manifest with per-class tile footprints
(:func:`...terrains.clip_terrain_loader.assemble_clip_terrain`) and hands it to the
base :class:`TerrainImporter` via the same two public calls the grid path uses:
:meth:`import_mesh` + :meth:`configure_env_origins`.

It exposes, per env (following the ``terrain_levels`` / ``terrain_types`` mapping
the base curriculum sets up):

* :attr:`env_terrain_id` -- the terrain id of the env's tile (``-1`` = flat),
  matching the clips' per-clip ``terrain_id`` for assignment;
* :attr:`env_cell_corner` / :attr:`env_cell_center` -- the tile's local-corner /
  center in world, for placing a clip on the tile
  (``clip_terrain_loader.clip_frame_origin``).
"""

from __future__ import annotations

import torch
from typing_extensions import override

import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter

from motion_gen.clips.clip_io import load_clip_library
from .clip_terrain_loader import assemble_clip_terrain


class ClipTerrainImporter(TerrainImporter):
    """Custom-mesh importer for clip-library terrains with per-env terrain id."""

    cfg: "object"

    env_terrain_id: torch.Tensor
    """[num_envs] int64 terrain id of each env's tile (``-1`` = flat)."""

    env_cell_corner: torch.Tensor
    """[num_envs, 3] world position of each env's tile local (0,0) corner (z=0)."""

    env_cell_center: torch.Tensor
    """[num_envs, 3] world position of each env's tile center (z=0)."""

    env_clip_x_offset: torch.Tensor
    """[num_envs] x-shift of the clip within its tile (transition back-buffer)."""

    def __init__(self, cfg) -> None:
        # Mirror TerrainImporter.__init__'s buffer setup, but import a custom
        # assembled mesh instead of dispatching on cfg.terrain_type.
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore
        self.terrain_prim_paths = []
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = {}

        lib = load_clip_library(cfg.clip_library_path, device="cpu")
        mesh, origins, layout = assemble_clip_terrain(
            lib, num_rows=cfg.num_rows, num_flat_tiles=cfg.num_flat_tiles,
            col_gap=cfg.col_gap, row_gap=cfg.row_gap, size_margin=cfg.size_margin,
            flat_buffer=cfg.flat_buffer, transition_back_buffer=cfg.transition_back_buffer,
        )
        self.import_mesh("terrain", mesh)
        # Base curriculum origin distribution: terrain_levels (rows) + terrain_types
        # (columns == tiles); env_origins = origins[level, type] = tile center.
        self.configure_env_origins(origins)

        self._layout = layout
        self._tile_terrain_id = layout.tile_terrain_id.to(self.device)
        self._tile_corner = layout.tile_corner.to(self.device)
        self._tile_footprint = layout.tile_footprint.to(self.device)
        self._tile_clip_x_offset = layout.tile_clip_x_offset.to(self.device)
        self.clip_size_y = layout.clip_size_y
        self.terrain_type_code = layout.terrain_type_code
        self._refresh_env_arrays()

        self.set_debug_vis(cfg.debug_vis)

    @override
    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        super().update_env_origins(env_ids, move_up, move_down)
        if self.terrain_origins is not None:
            self._refresh_env_arrays()

    def _refresh_env_arrays(self) -> None:
        """Per-env terrain id + tile corner/center from the env->tile mapping."""
        lv, ty = self.terrain_levels, self.terrain_types
        self.env_terrain_id = self._tile_terrain_id[lv, ty]
        self.env_cell_corner = self._tile_corner[lv, ty]
        self.env_clip_x_offset = self._tile_clip_x_offset[lv, ty]
        fp = self._tile_footprint[lv, ty]
        center = self.env_cell_corner.clone()
        center[:, 0] += fp[:, 0] / 2.0
        center[:, 1] += fp[:, 1] / 2.0
        self.env_cell_center = center
