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
    cfg.resample(difficulty)
    # resample difficulty-dependent parameters
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else None)
    rel_x = rng.uniform(*cfg.rel_stone_x_range, cfg.num_stones)
    rel_z = rng.uniform(*cfg.rel_stone_z_range, cfg.num_stones)

    # --- Start platform
    start_platform_size = cfg.start_platform_size
    start_platform_center_pos = (
        0.5 * start_platform_size[0],
        0.5 * start_platform_size[0],
        -cfg.start_platform_size[2] / 2
    )
    start_box = trimesh.creation.box(start_platform_size, trimesh.transformations.translation_matrix(start_platform_center_pos))
    meshes.append(start_box)

    # --- Stepping stones
    curr_x = cfg.start_platform_size[0]  - cfg.stone_target_x / 2
    curr_y, curr_z = start_platform_center_pos[1], start_platform_center_pos[2] 

    box_dims = cfg.stone_size
    for i in range(cfg.num_stones):
        dx, dz = rel_x[i], rel_z[i]
        curr_x += dx
        curr_z += dz
        box_pos = (curr_x, curr_y, curr_z)
        stone = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes.append(stone)

    # --- End platform
    end_platform_size = (cfg.size[0] - rel_x.sum() - cfg.start_platform_size[0], cfg.stone_size[1], cfg.stone_size[2])
    end_dims = end_platform_size
    end_pos = (
        curr_x + cfg.stone_size[0] / 2 + 0.5 * end_dims[0],
        start_platform_center_pos[1],
        curr_z,
    )
    end_box = trimesh.creation.box(end_dims, trimesh.transformations.translation_matrix(end_pos))
    meshes.append(end_box)

    # origin (where robot spawns)
    origin = np.array([start_platform_center_pos[0], start_platform_center_pos[1], 0.0])
    
    terrain_info: dict[str, np.ndarray] = {
    "rel_x": rel_x,
    "rel_z": rel_z,
    # center position of the start stone (the virtual stone at the end of the start platform), from specified origin
    "start_stone_pos": np.array([start_platform_size[0]/2 - cfg.stone_target_x / 2, 0., 0.]),
    "stone_x": cfg.stone_size[0]
    }
    #todo consider different yaw angle?


    return meshes, origin, terrain_info




def long_stones_terrain_with_platform_underneath(
    difficulty: float, cfg: LongStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict[str,np.ndarray]]:
    meshes = []
    cfg.resample(difficulty)
    # resample difficulty-dependent parameters
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else None)
    rel_x = rng.uniform(*cfg.rel_stone_x_range, cfg.num_stones)
    rel_z = rng.uniform(*cfg.rel_stone_z_range, cfg.num_stones)

    # --- Start platform
    start_platform_size = cfg.start_platform_size
    start_platform_center_pos = (
        0.5 * start_platform_size[0],
        0.5 * start_platform_size[0],
        -cfg.start_platform_size[2] / 2
    )
    start_box = trimesh.creation.box(start_platform_size, trimesh.transformations.translation_matrix(start_platform_center_pos))
    meshes.append(start_box)

    # --- Stepping stones
    curr_x = cfg.start_platform_size[0]  - cfg.stone_target_x / 2
    curr_y, curr_z = start_platform_center_pos[1], start_platform_center_pos[2] 
    min_z = curr_z
    box_dims = cfg.stone_size
    for i in range(cfg.num_stones):
        dx, dz = rel_x[i], rel_z[i]
        curr_x += dx
        curr_z += dz
        min_z = min(min_z, curr_z)  # Track lowest stone
        box_pos = (curr_x, curr_y, curr_z)
        stone = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes.append(stone)

    # --- End platform
    end_platform_size = (cfg.size[0] - rel_x.sum() - cfg.start_platform_size[0], cfg.stone_size[1], cfg.stone_size[2])
    end_dims = end_platform_size
    end_pos = (
        curr_x + cfg.stone_size[0] / 2 + 0.5 * end_dims[0],
        start_platform_center_pos[1],
        curr_z,
    )
    end_box = trimesh.creation.box(end_dims, trimesh.transformations.translation_matrix(end_pos))
    meshes.append(end_box)

    # origin (where robot spawns)
    origin = np.array([start_platform_center_pos[0], start_platform_center_pos[1], 0.0])
    
    terrain_info: dict[str, np.ndarray] = {
    "rel_x": rel_x,
    "rel_z": rel_z,
    # center position of the start stone (the virtual stone at the end of the start platform), from specified origin
    "start_stone_pos": np.array([start_platform_size[0]/2 - cfg.stone_target_x / 2, 0., 0.]),
    "stone_x": cfg.stone_size[0]
    }
    
    # Add a large flat platform underneath the stones
    total_length = cfg.size[0]
    platform_thickness = 0.05  # thickness of underlying platform
    platform_size = (total_length, cfg.size[1], platform_thickness)
    platform_center = (
        total_length / 2,
        start_platform_center_pos[1],
        min_z - platform_thickness / 2 + cfg.underneath_platform_z,
    )
    base_platform = trimesh.creation.box(
        platform_size,
        trimesh.transformations.translation_matrix(platform_center),
    )
    meshes.append(base_platform)
    
    


    return meshes, origin, terrain_info


def upstairs_with_platform_underneath(
    difficulty: float, cfg: LongStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict[str,np.ndarray]]:
    meshes = []
    cfg.resample(difficulty)
    # resample difficulty-dependent parameters
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else None)
    rel_x = rng.uniform(*cfg.rel_stone_x_range, cfg.num_stones)
    rel_z = rng.uniform(*cfg.rel_stone_z_range, cfg.num_stones)

    # --- Start platform
    start_platform_size = cfg.start_platform_size
    start_platform_center_pos = (
        0.5 * start_platform_size[0],
        0.5 * start_platform_size[0],
        -cfg.start_platform_size[2] / 2
    )
    start_box = trimesh.creation.box(start_platform_size, trimesh.transformations.translation_matrix(start_platform_center_pos))
    meshes.append(start_box)

    # --- Stepping stones
    curr_x = cfg.start_platform_size[0]  - cfg.stone_target_x / 2
    curr_y, curr_z = start_platform_center_pos[1], start_platform_center_pos[2] 

    box_dims = cfg.stone_size
    for i in range(cfg.num_stones):
        dx, dz = rel_x[i], rel_z[i]
        curr_x += dx
        curr_z += dz
        box_pos = (curr_x, curr_y, curr_z)
        stone = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes.append(stone)

    # --- End platform
    end_platform_size = (cfg.size[0] - rel_x.sum() - cfg.start_platform_size[0], cfg.stone_size[1], cfg.stone_size[2])
    end_dims = end_platform_size
    end_pos = (
        curr_x + cfg.stone_size[0] / 2 + 0.5 * end_dims[0],
        start_platform_center_pos[1],
        curr_z,
    )
    end_box = trimesh.creation.box(end_dims, trimesh.transformations.translation_matrix(end_pos))
    meshes.append(end_box)

    # origin (where robot spawns)
    origin = np.array([start_platform_center_pos[0], start_platform_center_pos[1], 0.0])
    
    terrain_info: dict[str, np.ndarray] = {
    "rel_x": rel_x,
    "rel_z": rel_z,
    # center position of the start stone (the virtual stone at the end of the start platform), from specified origin
    "start_stone_pos": np.array([start_platform_size[0]/2 - cfg.stone_target_x / 2, 0., 0.]),
    "stone_x": cfg.stone_size[0]
    }
    
    # Add a large flat platform underneath the stones, width = 3 * stone_y
    total_length = cfg.size[0]
    platform_thickness = 0.05  # thickness of underlying platform
    platform_size = (total_length, cfg.stone_size[1] * 3, platform_thickness)
    platform_center = (
        total_length / 2,
        start_platform_center_pos[1],
        -0.5 - platform_thickness / 2,
    )
    base_platform = trimesh.creation.box(
        platform_size,
        trimesh.transformations.translation_matrix(platform_center),
    )
    meshes.append(base_platform)
    
    


    return meshes, origin, terrain_info