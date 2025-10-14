from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import trimesh
if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones_cfg import LongStonesTerrainCfg

def long_stones_terrain(
    difficulty: float, cfg: LongStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict[str,np.ndarray]]:
    meshes = []
    
    # resample difficulty-dependent parameters
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else None)
    rel_x = rng.uniform(*cfg.rel_stone_x_range, cfg.num_stones)
    rel_z = rng.uniform(*cfg.rel_stone_z_range, cfg.num_stones)

    # terrain bounds and center
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]

    # --- Start platform
    start_dims = cfg.start_platform_size
    start_pos = (
        0.5 * start_dims[0],
        terrain_center[1],
        -start_dims[2] / 2,
    )
    start_box = trimesh.creation.box(start_dims, trimesh.transformations.translation_matrix(start_pos))
    meshes.append(start_box)

    # --- Stepping stones
    curr_x = cfg.start_platform_size[0]  - cfg.stone_size[0] / 2
    curr_y, curr_z = terrain_center[1], start_pos[2] 

    for i in range(cfg.num_stones):
        dx, dz = rel_x[i], rel_z[i]
        curr_x += dx
        curr_z += dz
        box_dims = cfg.stone_size
        box_pos = (curr_x, curr_y, curr_z)
        stone = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes.append(stone)

    # --- End platform
    end_platform_size = (cfg.size[0] - rel_x.sum() - cfg.start_platform_size[0], cfg.stone_size[1], cfg.stone_size[2])
    end_dims = end_platform_size
    end_pos = (
        curr_x + cfg.stone_size[0] / 2 + 0.5 * end_dims[0],
        terrain_center[1],
        curr_z,
    )
    end_box = trimesh.creation.box(end_dims, trimesh.transformations.translation_matrix(end_pos))
    meshes.append(end_box)

    # origin (where robot spawns)
    origin = np.array([start_pos[0], terrain_center[1], 0.0])
    
    terrain_info: dict[str, np.ndarray] = {
    "rel_x": rel_x,
    "rel_z": rel_z,
    "start_platform_pos": np.array([cfg.start_platform_size[0] - cfg.stone_size[0] / 2, terrain_center[1], 0.0]),
    "origin": origin
    }


    return meshes, origin, terrain_info