from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

from dataclasses import MISSING
from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones import long_stones_terrain, long_stones_terrain_with_platform_underneath, upstairs_with_platform_underneath
from robot_rl.tasks.manager_based.robot_rl.constants import STONES
import numpy as np


@configclass
class LongStonesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a long stones pattern."""

    function = long_stones_terrain_with_platform_underneath #long_stones_terrain

    num_stones: int = STONES.num_stones
    num_init_steps: int = STONES.num_init_steps
    stone_size: tuple[float, float, float] = (STONES.stone_x, STONES.stone_y, STONES.stone_z)  # (x,y,z) size of each stone
    rel_stone_x_range: tuple[float, float] = (0.0, 0.0)
    rel_stone_z_range: tuple[float, float] = (0.0, 0.0)
    start_platform_size: tuple[float, float, float] = (STONES.start_platform_x, STONES.stone_y, STONES.stone_z)
    
    stone_target_x: float = STONES.stone_x  # target size x between stones, used for curriculum

    size: tuple[float, float] = (STONES.terrain_size_x, STONES.terrain_size_y)  # (x,y) overall terrain size

    def resample(self, difficulty):
        """Resample relative distances for each stone given difficulty."""
        # interpolate rel_stone_z_range with difficulty
        self.rel_stone_z_range = (-STONES.rel_stone_z_max*difficulty, STONES.rel_stone_z_max*difficulty)
        
        self.rel_stone_x_range = (STONES.rel_stone_x_min, STONES.rel_stone_x_min + (STONES.rel_stone_x_max - STONES.rel_stone_x_min)*difficulty)
        
        # # interpolate stone_x size with difficulty
        # max_val = STONES.rel_stone_x[1]                  # fill all gap
        # min_val = self.stone_target_x               # base size at hardest
        # stone_x_sampled = (1 - difficulty) * max_val + difficulty * min_val

        # # assign updated stone size
        # self.stone_size = (self.stone_size[0], self.stone_size[1], self.stone_size[2])

class UpStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with up stairs pattern."""

    function = MISSING  # to be defined

    num_steps: int = 10        # number of steps
    
    rel_stair_x_range: tuple[float, float] = (0.2, 0.5)  # (min,max) relative x distance range between stairs
    rel_stair_z_range: tuple[float, float] = (0.0, 0.2)  # (min,max) relative z distance range between stairs
    
    rel_x: float   # fixed relative x distance between stairs
    rel_z: float   # fixed relative z distance between stairs


    size: tuple[float, float] = (STONES.terrain_size_x, STONES.terrain_size_y)   # (x,y) overall terrain size
    def resample(self, difficulty):
        """Resample parameters based on difficulty if needed."""
        self.rel_x = (1 - difficulty) * STONES.rel_stair_x_range[1] + difficulty * STONES.rel_stair_x_range[0]
        self.rel_z = (1 - difficulty) * STONES.rel_stair_z_range[1] + difficulty * STONES.rel_stair_z_range[0]
