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
    current_st_foot_pos_xz = current_st_foot_pos[:, [0, 2]]
    end_stone_pos_xz = torch.stack([output_command.abs_x[:, -1], 
                                 output_command.abs_z[:, -1]], dim=1) # (num_envs, 2)
    distance = torch.norm(current_st_foot_pos_xz - end_stone_pos_xz, dim=1)

    termination_flag = distance < 0.01  # close enough to last stone (x<10cm)
   #  termination_flag = (current_st_foot_pos[:, 0] - output_command.abs_x[:, -1]) > 0.0
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