# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-cfg companion to ``clip_terrain_importer.py``.

Kept separate so importing this cfg does NOT pull in the runtime importer (and its
``isaaclab.markers`` chain that pre-loads ``pxr`` before ``AppLauncher``). The
runtime class is resolved via the string ``class_type`` at ``cfg.class_type(cfg)``.
"""

from dataclasses import MISSING

from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass

from motion_gen.terrain.decorate import DEFAULT_FLAT_BUFFER


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

    decor_overlay_path: str = ""
    """Optional clutter sidecar (``scripts/rsl_rl/author_clutter.py``) merged into the
    manifest at load.

    Visual-only geometry authored against a closed-loop recording, kept out of the library
    because the point is to have SEVERAL variants of it and only the decoration differs
    between them. Empty means the library as it stands, which is itself a variant -- the one
    every existing recording was made under.

    It cannot move the ground: everything it adds is ``decor``, which ``_tile_mesh`` splits
    out before the walkable mesh is built, so the tile layout and every recorded world pose
    stay valid. That is what lets a recording be re-rendered against it."""

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

    flat_buffer: float = DEFAULT_FLAT_BUFFER
    """Flat-tile margin (m) on each side beyond the flat clips' bbox.

    20 m purely to be safe. The concern is that a small margin silently bounds the survival
    metric: the tracker task terminates only on ``time_out`` and ``base_fallen``, so a robot
    that walks off the platform registers as a fall and is indistinguishable from falling
    over. A wide margin makes that impossible; it is not a measured optimum.

    Sweeping 2 / 10 / 20 / 40 m produced no effect outside the binomial noise floor, in both
    closed loop and clip control, so nothing here is tuned -- 20 m just buys margin cheaply.

    The real fix is probably a divergence termination on the tracker (the teacher stops at
    0.3 m of base error; the tracker has no such term), so survival measures tracking rather
    than how long until the robot leaves the mesh."""

    transition_back_buffer: float = 1.0
    """Flat (m) prepended to each transition tile behind the robot's start."""
