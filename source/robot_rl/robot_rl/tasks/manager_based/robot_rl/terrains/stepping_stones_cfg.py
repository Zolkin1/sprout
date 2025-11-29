from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

from dataclasses import MISSING
from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones import long_stones_terrain_with_platform_underneath, tilted_long_stones_terrain_with_platform_underneath
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
    
    stone_length_min: float = 0.13
    stone_width_min: float = 0.75
    stone_width_max: float = 1.25 #real stones 1.22
    stone_height_min: float = 0.05 #thickness of the stones, should not matter too much
    stone_height_max: float = 0.15
    underneath_platform_z_min: float = -1.0  # z pos of the underneath platform relative to stone top
    underneath_platform_z_max: float = 0.0  # z pos of the underneath platform relative to stone top
    def resample_basic(self):
        """Resample parameters based on difficulty if needed."""
        stone_y = np.random.uniform(self.stone_width_min, self.stone_width_max)
        stone_z = np.random.uniform(self.stone_height_min, self.stone_height_max)
        self.underneath_platform_z = np.random.uniform(self.underneath_platform_z_min, self.underneath_platform_z_max)
        self.start_platform_size = (STONES.start_platform_x, stone_y, stone_z)
        return stone_y, stone_z

@configclass
class LongStonesFlatTerrainCfg(BasicStonesTerrainCfg): #flat stones terrain
    def resample(self, difficulty):
        """Resample relative distances for each stone given difficulty."""
        # interpolate rel_stone_z_range with difficulty
        self.rel_stone_x_range = (STONES.rel_stone_x_min, STONES.rel_stone_x_min + (STONES.rel_stone_x_max - STONES.rel_stone_x_min)*difficulty)
        stone_x = np.random.uniform(self.stone_length_min, STONES.rel_stone_x_min)
        stone_y, stone_z = self.resample_basic()
        self.stone_size = (stone_x, stone_y, stone_z)
        return

@configclass
class LongStonesTerrainCfg(BasicStonesTerrainCfg): #stepping stones terrain with vertical variation
    def resample(self, difficulty):
        """Resample relative distances for each stone given difficulty."""
        self.rel_stone_x_range = (STONES.rel_stone_x_min, STONES.rel_stone_x_min + (STONES.rel_stone_x_max - STONES.rel_stone_x_min)*difficulty)
        self.rel_stone_z_range = (-STONES.rel_stone_z_max*difficulty, STONES.rel_stone_z_max*difficulty)
        
        # # interpolate stone_x size with difficulty
        # max_val = STONES.rel_stone_x_min  # fill all gap
        # min_val = self.stone_target_x  # base size at hardest
        stone_x = np.random.uniform(self.stone_length_min, STONES.rel_stone_x_min)
        
        stone_y, stone_z = self.resample_basic()
        self.stone_size = (stone_x, stone_y, stone_z)
        return
@configclass
class TiltedStonesTerrainCfg(BasicStonesTerrainCfg): #stepping stones terrain with vertical and pitch variation 
    function = tilted_long_stones_terrain_with_platform_underneath  # to be defined
    abs_stone_pitch_range: tuple[float, float] = (0.0, 0.0)  # pitch angle range in radians
    abs_pitch_max: float = 0.4
    def resample(self, difficulty):
        """Resample relative distances for each stone given difficulty."""
        self.rel_stone_x_range = (STONES.rel_stone_x_min, STONES.rel_stone_x_min + (STONES.rel_stone_x_max - STONES.rel_stone_x_min)*difficulty)
        self.rel_stone_z_range = (-STONES.rel_stone_z_max*difficulty, STONES.rel_stone_z_max*difficulty)

        self.abs_stone_pitch_range = (-self.abs_pitch_max*difficulty, self.abs_pitch_max*difficulty)

        # # interpolate stone_x size with difficulty
        # max_val = STONES.rel_stone_x_min  # fill all gap
        # min_val = self.stone_target_x  # base size at hardest
        stone_x = np.random.uniform(self.stone_length_min, STONES.rel_stone_x_min)
        
        stone_y, stone_z = self.resample_basic()
        self.stone_size = (stone_x, stone_y, stone_z)
        return        
@configclass
class StairsTerrainCfg(BasicStonesTerrainCfg): #stairs
    stair_x_min: float = 0.2 #typical 0.254 - 0.28
    stair_x_max: float = 0.3
    stair_z_max: float = 0.2 #typical stairs 0.177 - 0.203
    is_upstairs: bool = True
    def resample(self, difficulty):
        stone_y, stone_z = self.resample_basic()
        rel_x = np.random.uniform(self.stair_x_min, self.stair_x_max)
        self.rel_stone_x_range = (rel_x, rel_x)
        # old: difficult terrain has only hard stairs
        # rel_z = difficulty * self.stair_z_max * (2 * self.is_upstairs -1)
        
        # Sample height uniformly from [min, difficulty * max]
        # Easy terrain: small range [0, 0.02] (if difficulty=0.1)
        # Hard terrain: full range [0, 0.2] (if difficulty=1.0)
        max_height = difficulty * self.stair_z_max
        rel_z = np.random.uniform(0.0, max_height)
        # Apply upstairs/downstairs direction
        rel_z = rel_z * (2 * self.is_upstairs - 1)
        self.rel_stone_z_range = (rel_z, rel_z)
        self.stone_size = (rel_x, stone_y, stone_z)
        return
    
class FlatGroundTestingCfg(BasicStonesTerrainCfg): #flat ground for testing
    def resample(self, difficulty):
        stone_y, stone_z = self.resample_basic()
        stone_x = 0.2
        self.rel_stone_x_range = (0.2, 0.2)
        self.rel_stone_z_range = (0.0, 0.0)
        self.stone_size = (stone_x, stone_y, stone_z)
        return    