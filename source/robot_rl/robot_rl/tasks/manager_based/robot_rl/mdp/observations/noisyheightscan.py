
from __future__ import annotations

from typing import TYPE_CHECKING

# file: my_project/mdp_extensions.py
import torch
import torch.nn.functional as F
from isaaclab.utils.math import quat_apply

def noisy_height_scan(env, sensor_cfg):
    """Same as mdp.height_scan, but with LiDAR-like noise and rotation errors."""
    # --- get clean height map from RayCaster
    height_map = env.scene[sensor_cfg.name].data  # shape (num_envs, H, W)

    # --- (1) Vertical noise
    if not hasattr(env, "vertical_bias"):
        env.vertical_bias = torch.empty(env.num_envs, device=height_map.device).uniform_(-0.02, 0.02)
    vertical_step_noise = torch.empty_like(height_map).uniform_(-0.005, 0.005)
    height_map = height_map + env.vertical_bias.view(-1, 1, 1) + vertical_step_noise

    # --- (2) Map rotation noise (simulate odometry error)
    B, H, W = height_map.shape
    yaw_noise = torch.empty(B, device=height_map.device).uniform_(-3, 3) * torch.pi / 180
    rot_mats = torch.stack([
        torch.cos(yaw_noise), -torch.sin(yaw_noise),
        torch.sin(yaw_noise),  torch.cos(yaw_noise)
    ], dim=-1).view(B, 2, 2)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=height_map.device),
        torch.linspace(-1, 1, W, device=height_map.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    grid = grid @ rot_mats.transpose(1, 2)
    height_map = F.grid_sample(height_map.unsqueeze(1), grid, align_corners=True).squeeze(1)

    # --- (3) Foothold extension / smoothing
    kernel = torch.ones((1, 1, 3, 3), device=height_map.device) / 9.0
    height_map = F.conv2d(height_map.unsqueeze(1), kernel, padding=1).squeeze(1)

    return height_map
