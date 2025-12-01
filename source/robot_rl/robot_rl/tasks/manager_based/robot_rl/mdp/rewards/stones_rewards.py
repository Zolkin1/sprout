
import torch


def standing_penalty(env, velocity_threshold: float = 0.2) -> torch.Tensor:
    """Penalize when robot is stationary."""
    base_vel_x = env.scene.articulations["robot"].data.root_lin_vel_w[:, 0]  # only x direction
    speed = torch.abs(base_vel_x)

    # Penalty when robot is standing (low speed)
    is_standing = speed < velocity_threshold

    return is_standing.float()  # 1.0 when standing, 0.0 when moving


def swing_foot_position_error_at_contact_reward(env, command_name: str, kd: float = 0.1) -> torch.Tensor:
    """Penalize the error between the swing foot position and the target position."""
    cmd = env.command_manager.get_term(command_name)
    #pick x and z components only
    swing_foot_error = cmd.swing_foot_error_at_contact[:, [0, 2]]  # shape: (num_envs, 2)
    reward = torch.exp(-swing_foot_error.norm(dim=-1) / kd)
    # Zero out reward when NOT at contact
    reward = reward * cmd.mask_at_contact.float()
    return reward  # shape: (num_envs,)