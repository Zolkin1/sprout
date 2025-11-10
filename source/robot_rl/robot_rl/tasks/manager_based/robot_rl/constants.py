from dataclasses import dataclass

IS_DEBUG = False
TEST_FLAT = False
ZERO_EPS = 1e-8

@dataclass
class StonesConfig:
   num_stones: int = 20
   stone_x: float = 0.2  # stone x (length in meters)
   stone_y: float = 1.0  # stone y (width in meters)
   stone_z: float = 0.1  # stone z (height in meters)
   # rel_stone_x = [0.3, 0.7]  # relative x distance range between stones
   
   rel_stone_x_min = 0.3
   rel_stone_x_max = 0.7
   rel_stone_z_max = 0.1  # relative z distance range between stones
   start_platform_x = 1.0
   num_init_steps = 2
   terrain_size_x = num_stones * rel_stone_x_max +  start_platform_x #terrain size in x direction
   terrain_size_y = stone_y * 3 #terrain size in y direction
   
   robot_launch_x_lb = (start_platform_x-stone_x/2.0)/2.0 - (start_platform_x-stone_x/2.0)/4.0
   robot_launch_x_ub = (start_platform_x-stone_x/2.0)/2.0 + (start_platform_x-stone_x/2.0)/4.0
   

STONES = StonesConfig()