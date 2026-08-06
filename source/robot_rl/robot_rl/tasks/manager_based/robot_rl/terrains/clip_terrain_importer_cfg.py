# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-cfg companion to ``clip_terrain_importer.py``.

Kept separate so importing this cfg does NOT pull in the runtime importer (and its
``isaaclab.markers`` chain that pre-loads ``pxr`` before ``AppLauncher``). The
runtime class is resolved via the string ``class_type`` at ``cfg.class_type(cfg)``.
"""

from dataclasses import MISSING

from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class ClipTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for :class:`ClipTerrainImporter`.

    Builds a variable-size terrain directly from a clip library's manifest (one
    tile per terrain, per-class footprints, no uniform-grid padding) instead of a
    ``TerrainGeneratorCfg`` grid.  ``terrain_type`` / ``terrain_generator`` from the
    base cfg are ignored -- the importer assembles its own mesh from
    ``clip_library_path``.
    """

    class_type: type | str = "{DIR}.clip_terrain_importer:ClipTerrainImporter"

    clip_library_path: str = MISSING
    """Path to the clip-library ``.h5`` (terrain manifest + clips) to build from."""

    num_rows: int = 1
    """Replicas of the tile set (each a strip in y) -> more distinct env spawns."""

    num_flat_tiles: int | None = None
    """Number of flat patches. ``None`` derives it from the library's generation
    metadata (``generator_config.counts.flat``) so flat/terrain/transition tile counts
    match the generation split -- NOT one tile per flat clip. See
    ``clip_terrain_loader.default_num_flat_tiles``."""

    col_gap: float = 1.0
    """Void gap (m) between tiles along x."""

    row_gap: float = 2.0
    """Void gap (m) between replica strips along y."""

    size_margin: float = 1.0
    """Extra flat (m) added to the terrain / transition footprints."""

    flat_buffer: float = 20.0
    """Flat-tile margin (m) on each side beyond the flat clips' bbox.

    20 m, not the original 2 m, because 2 m SILENTLY BOUNDED the survival metric: the tracker
    task terminates only on ``time_out`` and ``base_fallen``, so a robot that drifts off the
    platform registers as a fall and is indistinguishable from falling over. Measured on the
    yaw_flat3k pair, closed loop, 200 envs, seed 42 -- 2 m: 176/200, 10 m: 188, 20 m: 194,
    40 m: 189. So ~14-18 of the 24 "failures" at 2 m were the platform ending, not the policy.

    TODO(flat-buffer): the series is NOT monotonic (194 -> 189 from 20 m to 40 m), which cannot
    be an edge effect -- on flat ground a bigger platform is strictly more permissive. Most
    likely this resizes the TILES, which moves env origins and clip placement, so each value is
    a different set of initial conditions rather than one variable swept. Decisive test: re-run
    one margin at a second seed and see whether the 188/194/189 spread is just that. Until then
    20 m is a pragmatic choice, not a measured optimum, and survival numbers are only comparable
    at a FIXED value. The real fix is probably a divergence termination on the tracker (the
    teacher stops at 0.3 m of base error; the tracker has no such term), so survival measures
    tracking rather than how long until the robot leaves the mesh."""

    transition_back_buffer: float = 1.0
    """Flat (m) prepended to each transition tile behind the robot's start."""
