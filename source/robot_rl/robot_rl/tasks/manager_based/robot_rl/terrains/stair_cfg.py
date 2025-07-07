from dataclasses import MISSING
from typing import Literal

import trimesh
import numpy as np
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from robot_rl.tasks.manager_based.robot_rl.terrains.stair import progressive_x_stairs_terrain




@configclass
class MeshProgressiveYStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a long staircase with progressively taller steps along +y."""

    function = progressive_x_stairs_terrain

    step_height_range: tuple[float, float] = (0.03, 0.15)
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = 0.25
    """The depth of each step in y-direction (in m)."""

    border_width: float = 0.0
    """The width of the border around the terrain (in m)."""


import numpy as np
from typing import Union

def get_step_height_at_x(
    x_vals: Union[np.ndarray, list], cfg: "MeshProgressiveYStairsTerrainCfg"
) -> np.ndarray:
    """
    Given a batch of x-coordinates, return the cumulative step height at each x
    for a staircase that increases in +x direction with growing step heights.

    Args:
        x_vals: Array-like of x positions (shape: [N]).
        cfg: Stair terrain config.

    Returns:
        np.ndarray of terrain heights (shape: [N]).
    """
    x_vals = np.asarray(x_vals)
    usable_x = x_vals - cfg.border_width

    step_depth = cfg.step_width
    terrain_length = cfg.size[0] - 2 * cfg.border_width
    num_steps = int(terrain_length // step_depth)

    # Generate increasing step heights
    step_heights = np.linspace(cfg.step_height_range[0], cfg.step_height_range[1], num_steps)
    cum_heights = np.cumsum(step_heights)  # [num_steps]

    # Initialize output
    heights = np.zeros_like(x_vals)

    # Mask for positions before the stairs
    mask_below = usable_x < 0
    heights[mask_below] = 0.0

    # Mask for positions beyond the stairs
    mask_above = usable_x >= terrain_length
    heights[mask_above] = cum_heights[-1]

    # Mask for positions within stair range
    mask_inside = ~(mask_below | mask_above)
    step_indices = (usable_x[mask_inside] // step_depth).astype(int)
    heights[mask_inside] = cum_heights[step_indices]

    return heights


