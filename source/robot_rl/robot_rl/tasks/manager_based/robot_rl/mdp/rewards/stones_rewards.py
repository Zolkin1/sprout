
import torch


def standing_penalty(env, velocity_threshold: float = 0.2) -> torch.Tensor:
    """Penalize when robot is stationary."""
    base_vel_x = env.scene.articulations["robot"].data.root_lin_vel_w[:, 0]  # only x direction
    speed = torch.abs(base_vel_x)

    # Penalty when robot is standing (low speed)
    is_standing = speed < velocity_threshold

    return is_standing.float()  # 1.0 when standing, 0.0 when moving