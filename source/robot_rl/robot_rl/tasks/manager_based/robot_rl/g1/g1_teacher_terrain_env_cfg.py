# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 teacher on terrain: terrain-aware open-loop motion-clip tracking (stairs).

Terrain counterpart to :mod:`g1_teacher_env_cfg`.  The scene ground is a
:class:`ClipTerrainImporter` assembled from the SAME clip library the command
tracks (one tile per staircase / transition terrain + flat patches), so every env
is pinned to a tile with a known ``terrain_id``.  At reset,
:func:`mdp.reset_on_terrain_motion_sequence` gives each env a clip whose per-clip
``terrain_id`` matches its tile and places it so the reference feet ride the treads.

Everything else is inherited from the flat teacher unchanged: the open-loop
clip-tracking command, the CLF rewards, the base-relative / heading-invariant policy
observations, and the pelvis height scan -- which now rides the staircase mesh and
gives the policy its terrain shape.

Envs are spread uniformly across all terrains from step 0 (no difficulty
curriculum): the base importer's ``terrain_types`` assignment already gives every
terrain column an even share of envs.  Replica rows (``num_rows``) exist only for env
capacity, so all rows are used (``max_init_terrain_level=None``).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer_cfg import (
    ClipTerrainImporterCfg,
)

from .g1_teacher_env_cfg import G1TeacherEnvCfg, G1TeacherSceneCfg

# Default terrain clip library (staircases + transitions + flat patches), generated
# with scripts/rsl_rl/generate_terrain_clips.py at the env control dt (0.02).  The
# terrain mesh AND the tracked clips both come from this one file.
DEFAULT_TERRAIN_CLIP_PATH = "clips/g1_terrain_v2.h5"


##
# Scene
##
@configclass
class G1TeacherTerrainSceneCfg(G1TeacherSceneCfg):
    """Teacher scene whose ground is a clip-library terrain (stairs + flat tiles).

    Replaces the flat-mesh :class:`TerrainImporterCfg` with a
    :class:`ClipTerrainImporterCfg` that assembles one tile per terrain from the clip
    library's manifest.  The inherited ``height_scanner`` already targets
    ``/World/ground`` -- where the importer imports its mesh -- so it raycasts the
    staircases with no change.  ``clip_library_path`` is re-pinned to the command's
    library in :meth:`G1TeacherTerrainEnvCfg.__post_init__` (single source of truth).
    """

    terrain = ClipTerrainImporterCfg(
        prim_path="/World/ground",
        clip_library_path=DEFAULT_TERRAIN_CLIP_PATH,
        # Replica rows for env capacity (rows are pure replicas of the tile set).
        num_rows=10,
        # No difficulty ordering -> use ALL replica rows (not just [0, max_init]).
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )


##
# Environment
##
@configclass
class G1TeacherTerrainEnvCfg(G1TeacherEnvCfg):
    """Terrain-aware open-loop motion-sequence tracking env (stairs + flat)."""

    scene: G1TeacherTerrainSceneCfg = G1TeacherTerrainSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        # The ClipTerrainImporter builds its mesh from clip_library_path and IGNORES
        # terrain_type / terrain_generator; the parent set the flat mesh on them, so
        # clear the flat generator to avoid confusion.
        self.scene.terrain.terrain_generator = None
        # Terrain-aware clip sampling: each env gets a clip matching its tile terrain.
        self.events.reset_on_clip.func = mdp.reset_on_terrain_motion_sequence
        # Single clip-library source of truth: the importer (terrain mesh + per-env
        # terrain_id) and the command (clip tracking) MUST read the same file, or the
        # tile terrain ids won't line up with the clips' terrain ids.
        self.commands.traj_ref.path = DEFAULT_TERRAIN_CLIP_PATH
        self.scene.terrain.clip_library_path = DEFAULT_TERRAIN_CLIP_PATH
        assert self.commands.traj_ref.path == self.scene.terrain.clip_library_path, (
            "terrain clip library (scene.terrain.clip_library_path) and command clip "
            "library (commands.traj_ref.path) must be the same file."
        )
        # Clips are baked axis-aligned with their staircase, so placement is a pure
        # translation onto the tile -- no reset heading randomization.
        self.commands.traj_ref.randomize_heading = False


@configclass
class G1TeacherTerrainEnvCfgPlay(G1TeacherTerrainEnvCfg):
    """Smaller, noise-free variant for play / debugging (one env per terrain)."""

    def __post_init__(self):
        super().__post_init__()
        # One replica row + one env per tile -> every terrain is shown exactly once.
        self.scene.terrain.num_rows = 1
        self.scene.num_envs = 24
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the reference-trajectory markers + the height-scan ray hits.
        self.commands.traj_ref.debug_vis = True
        self.scene.height_scanner.debug_vis = True
