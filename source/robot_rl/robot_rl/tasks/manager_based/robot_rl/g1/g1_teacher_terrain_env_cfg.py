# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 teacher on terrain: the concrete open-loop motion-clip tracking env.

Supplies the ground for the abstract :mod:`g1_teacher_env_cfg`.  The scene ground is a
:class:`ClipTerrainImporter` assembled from the SAME clip library the command tracks (one
tile per staircase / box / transition terrain + flat patches), so every env is pinned to a
tile with a known ``terrain_id``.  At reset,
:func:`mdp.reset_on_terrain_motion_sequence` gives each env a clip whose per-clip
``terrain_id`` matches its tile and places it so the reference feet ride the treads.

A flat run is simply a library whose tiles are flat -- there is no separate flat env.

Everything else is inherited unchanged: the open-loop clip-tracking command, the CLF
rewards, the base-relative / heading-invariant policy observations, and the pelvis height
scan -- which rides the staircase mesh and gives the policy its terrain shape.

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

# There is deliberately NO default terrain clip library: the terrain mesh AND the tracked
# clips both come from one ``.h5``, and a stale default silently trains on the wrong terrain
# (the old ``clips/g1_terrain_v2.h5`` also predates the per-clip anchor mask, so every clip
# resolved to the LEGACY x/y/z/yaw-free mask and the reference detached from its staircase).
# Pass the library explicitly with ``--clip <lib.h5>``; it sets both fields at once.


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
    staircases with no change.  ``clip_library_path`` is left MISSING: ``--clip`` sets it
    (and the command's ``path``) to the one library both must read.
    """

    terrain = ClipTerrainImporterCfg(
        prim_path="/World/ground",
        # num_flat_tiles left as None -> derived from the library's generation metadata
        # (generator_config.counts.flat), so flat/terrain/transition tile counts match
        # the split you generated with (total tiles == num_terrains) and envs are spread
        # evenly across terrain types. (Without this the default would be one flat tile
        # PER flat clip -- thousands -- diluting stair envs to ~0; see
        # terrains.clip_terrain_loader.default_num_flat_tiles.)
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
        # Terrain-aware clip sampling: each env gets a clip matching its tile terrain.
        # The library is REQUIRED from the CLI (``--clip``), which sets the command's
        # ``path`` and ``scene.terrain.clip_library_path`` together -- the importer
        # (terrain mesh + per-env terrain_id) and the command (clip tracking) MUST read
        # the same file, or the tile terrain ids won't line up with the clips'. Both are
        # verified non-empty and identical at env build (ClipTerrainImporter /
        # MotionSequenceCommand / reset_on_terrain_motion_sequence); this cfg hook runs
        # BEFORE the CLI override, so it cannot check them itself.
        self.events.reset_on_clip.func = mdp.reset_on_terrain_motion_sequence
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
        # Local play is usually on a small GPU; a large terrain clip library would OOM
        # if loaded onto the GPU, so keep the clip library on CPU (device-safe gather).
        # Override with --clip_device on the command line if you have GPU headroom.
        self.commands.traj_ref.clip_device = "cpu"
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the reference-trajectory markers + the height-scan ray hits.
        self.commands.traj_ref.debug_vis = True
        self.scene.height_scanner.debug_vis = True
