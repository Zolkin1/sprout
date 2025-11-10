from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

from dataclasses import MISSING
from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones import long_stones_terrain, long_stones_terrain_with_platform_underneath, upstairs_with_platform_underneath
from robot_rl.tasks.manager_based.robot_rl.constants import STONES
import numpy as np

@configclass
class BasicStonesTerrainCfg(SubTerrainBaseCfg):
    """Basic configuration for a terrain with stones pattern."""
    function = long_stones_terrain_with_platform_underneath  # to be defined
    size: tuple[float, float] = (STONES.terrain_size_x, STONES.terrain_size_y)  # (x,y) overall terrain size
    num_stones: int = STONES.num_stones
    num_init_steps: int = STONES.num_init_steps
    
    stone_size: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x,y,z) size of each stone
    rel_stone_x_range: tuple[float, float] = (0.0, 0.0)
    rel_stone_z_range: tuple[float, float] = (0.0, 0.0)
    start_platform_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    underneath_platform_z: float = 0.0  # z pos of the underneath platform relative to stone top

    
    stone_target_x: float = 0.2  # target size x between stones, used for curriculum
    
    stone_length_min: float = 0.2
    stone_width_min: float = 0.5
    stone_height_min: float = 0.1
    stone_height_max: float = 1.0
    underneath_platform_z_min: float = -1.0  # z pos of the underneath platform relative to stone top
    underneath_platform_z_max: float = 0.0  # z pos of the underneath platform relative to stone top
    def resample_basic(self):
        """Resample parameters based on difficulty if needed."""
        stone_y = np.random.uniform(self.stone_width_min, STONES.terrain_size_y)
        stone_z = np.random.uniform(self.stone_height_min, self.stone_height_max)
        self.underneath_platform_z = np.random.uniform(self.underneath_platform_z_min, self.underneath_platform_z_max)
        self.start_platform_size = (STONES.start_platform_x, stone_y, stone_z)
        return stone_y, stone_z

@configclass
class LongStonesFlatTerrainCfg(BasicStonesTerrainCfg):
    def resample(self, difficulty):
        """Resample relative distances for each stone given difficulty."""
        # interpolate rel_stone_z_range with difficulty
        self.rel_stone_x_range = (STONES.rel_stone_x_min, STONES.rel_stone_x_min + (STONES.rel_stone_x_max - STONES.rel_stone_x_min)*difficulty)
        stone_x = np.random.uniform(self.stone_length_min, STONES.rel_stone_x_min)
        stone_y, stone_z = self.resample_basic()
        self.stone_size = (stone_x, stone_y, stone_z)
        return

@configclass
class LongStonesTerrainCfg(BasicStonesTerrainCfg):
    def resample(self, difficulty):
        """Resample relative distances for each stone given difficulty."""
        self.rel_stone_x_range = (STONES.rel_stone_x_min, STONES.rel_stone_x_max)
        self.rel_stone_z_range = (-STONES.rel_stone_z_max*difficulty, STONES.rel_stone_z_max*difficulty)
        
        # interpolate stone_x size with difficulty
        max_val = STONES.rel_stone_x_max  # fill all gap
        min_val = self.stone_target_x  # base size at hardest
        stone_x = (1 - difficulty) * max_val + difficulty * min_val
        
        stone_y, stone_z = self.resample_basic()
        self.stone_size = (stone_x, stone_y, stone_z)
        return
        
@configclass
class StairsTerrainCfg(BasicStonesTerrainCfg):
    stair_x_min: float = 0.25
    stair_x_max: float = 0.5
    stair_z_max: float = 0.3
    is_upstairs: bool = True
    def resample(self, difficulty):
        stone_y, stone_z = self.resample_basic()
        rel_x = STONES.rel_stone_x_min + (STONES.rel_stone_x_max - STONES.rel_stone_x_min)*difficulty
        self.rel_stone_x_range = (rel_x, rel_x)
        rel_z = difficulty * self.stair_z_max * (2 * self.is_upstairs -1)
        self.rel_stone_z_range = (rel_z, rel_z)
        self.stone_size = (rel_x, stone_y, stone_z)
        return