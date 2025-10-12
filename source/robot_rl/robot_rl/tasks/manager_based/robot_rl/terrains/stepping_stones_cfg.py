from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

from dataclasses import MISSING
from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones import long_stones_terrain
from robot_rl.tasks.manager_based.robot_rl.constants import STONES
import numpy as np


@configclass
class LongStonesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a long stones pattern."""

    function = long_stones_terrain

    num_stones: int = STONES.num_stones
    num_init_steps: int = STONES.num_init_steps
    stone_size: tuple[float, float, float] = (STONES.stone_x, STONES.stone_y, STONES.stone_z)  # (x,y,z) size of each stone
    rel_stone_x_range: tuple[float, float] = (STONES.rel_stone_x[0], STONES.rel_stone_x[1])
    rel_stone_z_range: tuple[float, float] = (STONES.rel_stone_z[0], STONES.rel_stone_z[1])
    start_platform_size: tuple[float, float, float] = (STONES.start_platform_x, STONES.stone_y, 0.1)
    end_platform_size: tuple[float, float, float] = (1.0, STONES.stone_y, 0.1)

    size: tuple[float, float] = (STONES.terrain_size_x, STONES.terrain_size_y)  # (x,y) overall terrain size
   #  rel_x: np.ndarray = np.zeros(num_stones)  # relative x distances between stones
   #  rel_z: np.ndarray = np.zeros(num_stones)  # relative z distances between stones
   #  abs_x: np.ndarray = np.zeros(num_stones+STONES.num_init_steps)  # absolute x positions of stones
   #  abs_z: np.ndarray = np.zeros(num_stones+STONES.num_init_steps)  # absolute z positions of stones
    def resample(self, difficulty):
        """Resample relative distances for each stone."""
        # interpolate rel_stone_z_range with difficulty
        self.rel_stone_z_range = (self.rel_stone_z_range[0]*difficulty, self.rel_stone_z_range[1]*difficulty)
        # interpolate stone_x size with difficulty
        min_val = self.rel_stone_x_range[1]     # fill all gap
        max_val = self.stone_size[0]                # base size at hardest
        stone_x = (1 - difficulty) * min_val + difficulty * max_val

        # assign updated stone size
        self.stone_size = (stone_x, self.stone_size[1], self.stone_size[2])
        
      #   self.rel_x = rng.uniform(*self.rel_stone_x_range, self.num_stones)
      #   self.rel_z = rng.uniform(*self.rel_stone_z_range, self.num_stones)

      #   x0 = self.start_platform_size[0] - self.stone_size[0] / 2
      #   self.abs_x = np.concatenate([x0*np.ones(self.num_init_steps), x0 + np.cumsum(self.rel_x)])
      #   self.abs_z = np.concatenate([np.zeros(self.num_init_steps), np.cumsum(self.rel_z)])
      #   self.end_platform_size = (self.size[0] -self.rel_x.sum()-self.start_platform_size[0], self.end_platform_size[1], self.end_platform_size[2])

