import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.constants import STONES
def finished_long_stones(env, output_command_name: str) -> torch.Tensor:
    """
    Terminates the episode early if the robot has reached (or is close to)
    the final stepping stone.
      Args:
         env: The environment instance.
         stones_command_name: The name of the stones command term.
         output_command_name: The name of the output command term (e.g., hlip_ref).
      Returns:
         A boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    # --- Retrieve command terms ---
    output_command = env.command_manager.get_term(output_command_name)
    # --- Current and target positions (x–z) ---
    current_st_foot_pos = output_command.stance_foot_pos_0 # (num_envs, 3)
    terrain = env.scene.terrain
    distance = current_st_foot_pos[:, 0] - env.scene.env_origins[:, 0]

    termination_flag = distance > terrain.cfg.terrain_generator.size[0] * 0.95
   #  if torch.any(termination_flag):
   #       print(f"Finished stepping stones for {termination_flag.sum().item()} environments.")
    return termination_flag

def long_stones_deviation(env,output_command_name: str) -> torch.Tensor:
    """
    Computes the deviation of the robot's current position from the target stepping stone.
      Args:
         env: The environment instance.
         stones_command_name: The name of the stones command term.
         output_command_name: The name of the output command term (e.g., hlip_ref).
      Returns:
         A tensor of shape (num_envs,) representing the deviation distance.
    """
    # --- Retrieve command terms ---
    output_command = env.command_manager.get_term(output_command_name)
    # --- Current and target positions (x–z) ---
    current_st_foot_pos = output_command.stance_foot_pos_0 # (num_envs, 3)
    current_st_foot_pos_y = current_st_foot_pos[:, 1] # (num_envs,)
    end_stone_pos_y = output_command.abs_y[:, -1]  # (num_envs,)

    distance = torch.abs(current_st_foot_pos_y - end_stone_pos_y)
    
    termination_flag = distance > STONES.stone_y / 2.0  # deviated too far in y direction (> half stone width)
    
   #  if torch.any(termination_flag):
   #     print(f"Deviation termination triggered for {termination_flag.sum().item()} environments.")
    
    return termination_flag

def com_z_too_low(env, output_command_name: str) -> torch.Tensor:
    """
    Terminates the episode early if the robot's center of mass (CoM) z-position
    is below a certain threshold.
      Args:
         env: The environment instance.
         output_command_name: The name of the output command term (e.g., hlip_ref).
      Returns:
         A boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    # --- Retrieve command terms ---
    output_command = env.command_manager.get_term(output_command_name)
    # --- Current and target positions (x–z) ---
    current_st_foot_pos_z = output_command.stance_foot_pos_0[:, 2] # (num_envs, )
    current_sw_foot_pos_z = output_command.swing_foot_pos_0[:, 2] # (num_envs, )
    current_base_com_z = output_command.robot.data.root_com_pos_w[:, 2]  # (num_envs, )

    termination_flag = (current_base_com_z < current_st_foot_pos_z ) | (current_base_com_z < current_sw_foot_pos_z)
   #  if torch.any(termination_flag):
   #     print(f"Zcom too low termination triggered for {termination_flag.sum().item()} environments.")
    return termination_flag
 
 
def stationary_termination(env, velocity_threshold: float, duration_threshold: float) -> torch.Tensor:
    """Terminate if robot is stationary for too long.
    
    Args:
        env: The environment.
        velocity_threshold: Minimum velocity magnitude (m/s) to be considered moving.
        duration_threshold: Maximum time (s) allowed to be stationary.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    # Initialize stationary time tracker if needed
    if not hasattr(env, '_stationary_time'):
        env._stationary_time = torch.zeros(env.num_envs, device=env.device)
    
    # Get current base velocity
    base_vel = env.scene.articulations["robot"].data.root_lin_vel_w[:, :2]  # XY only
    speed = torch.norm(base_vel, dim=-1)
    
    # Check if stationary
    is_stationary = speed < velocity_threshold
    
    # Update stationary time
    env._stationary_time = torch.where(
        is_stationary,
        env._stationary_time + env.step_dt,  # increment
        torch.zeros_like(env._stationary_time)  # reset
    )
    
    # Terminate if stationary for too long
    return env._stationary_time > duration_threshold