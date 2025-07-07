from typing import TYPE_CHECKING

import trimesh
import numpy as np
from isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.terrains.stair_cfg import MeshProgressiveYStairsTerrainCfg


def progressive_x_stairs_terrain(
    difficulty: float, cfg: "MeshProgressiveYStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a staircase terrain along +x direction (robot forward), where each step has a level top surface
    and the step height increases progressively."""

    # Unpack usable area
    terrain_width = cfg.size[1]
    terrain_length = cfg.size[0] - 2 * cfg.border_width  # now x-direction is stair length
    step_depth = cfg.step_width
    min_h, max_h = cfg.step_height_range

    # Number of steps
    num_steps = int(terrain_length // step_depth)

    # Linearly increasing step heights
    step_heights = np.linspace(min_h, max_h, num_steps)

    # Generate steps
    meshes_list = []
    cum_z = 0.0
    for i in range(num_steps):
        h = step_heights[i]
        # Position: extend in +x
        pos_x = cfg.border_width + i * step_depth + step_depth / 2
        pos_y = cfg.size[1] / 2  # centered in y
        pos_z = cum_z + h / 2

        box_dims = (step_depth, terrain_width, h)  # [x, y, z]
        box_pos = (pos_x, pos_y, pos_z)

        mesh = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes_list.append(mesh)

        cum_z += h

    # Optional border
    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_heights[0] / 2]
        inner = (terrain_length, terrain_width)
        meshes_list += make_border(cfg.size, inner, step_heights[0], border_center)

    # Origin is at the base of stairs: [start x, center y, base z]
    origin = np.array([cfg.border_width, cfg.size[1] / 2, 0.0])
    return meshes_list, origin


