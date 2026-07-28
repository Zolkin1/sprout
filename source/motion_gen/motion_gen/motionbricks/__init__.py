# Copyright 2026 Zachary Olkin. All rights reserved.

"""Motion Bricks generator backend for offline clip generation.

This subpackage bridges NVIDIA's Motion Bricks (a learned latent motion generator) into the
existing offline-clip pipeline as a *selectable backend alongside* the DP motion matcher. It is
split across two stages that never share a Python process:

* **Stage 1** (Motion Bricks conda env, GPU) drives the model and writes an intermediate raw-motion
  file. That code lives in ``scripts/rsl_rl/mb_generate.py`` / ``mb_encode.py`` and imports the
  external ``motionbricks`` package -- it is *not* part of this subpackage.
* **Stage 2** (this subpackage, isaaclab-free) reads the raw-motion file and converts it into the
  clip-library ``.h5`` the RL side already consumes, reusing
  :class:`motion_gen.motion_matching.clip_output.ClipOutputBuilder` and
  :class:`motion_gen.clips.clip_io.ClipLibraryWriter`.

The intermediate file (:mod:`.raw_schema`) is the only contract between the two stages, so Stage 2
carries no dependency on Motion Bricks or on the external clone path.

The terrain path adds two more numpy-only modules that the Motion Bricks side also loads by path
(:mod:`.terrain_plan`, the placed-terrain hand-off contract, and :mod:`.terrain_families`, the per-skill
table + boundary-frame gates), plus :mod:`.terrain_splice` (torch, Stage 2 only) which joins the MB
segments around the replay.
"""

from .raw_schema import RawMotion, load_raw_motion, save_raw_motion
from .joint_map import MB_QPOS_ROOT_POS, MB_QPOS_ROOT_QUAT, MB_QPOS_JOINTS, split_qpos, reindex_to
from .mb_ingest import raw_to_world, resample_times, resample_rows, fit_to_length
from .terrain_families import (
    FAMILIES,
    SOLE_Z,
    BoundaryGate,
    TerrainFamily,
    boundary_candidates,
    box_platform_sign,
    episodic_boundaries,
    family_for,
    pick_boundary,
    sole_heights,
)
from .terrain_plan import TerrainPlan, load_terrain_plan, sample_qpos, save_terrain_plan
from .terrain_splice import (
    align_boundary,
    footprint_recenter,
    match_overlap,
    overlays_for,
    splice_terrain,
)

__all__ = [
    "RawMotion",
    "load_raw_motion",
    "save_raw_motion",
    "MB_QPOS_ROOT_POS",
    "MB_QPOS_ROOT_QUAT",
    "MB_QPOS_JOINTS",
    "split_qpos",
    "reindex_to",
    "raw_to_world",
    "resample_times",
    "resample_rows",
    "fit_to_length",
    "FAMILIES",
    "SOLE_Z",
    "BoundaryGate",
    "TerrainFamily",
    "boundary_candidates",
    "box_platform_sign",
    "episodic_boundaries",
    "family_for",
    "pick_boundary",
    "sole_heights",
    "TerrainPlan",
    "load_terrain_plan",
    "sample_qpos",
    "save_terrain_plan",
    "align_boundary",
    "footprint_recenter",
    "match_overlap",
    "overlays_for",
    "splice_terrain",
]
