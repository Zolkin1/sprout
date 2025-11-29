
from __future__ import annotations

from typing import TYPE_CHECKING



from dataclasses import dataclass
import torch
import torch.nn.functional as F



def height_scan_isaaclab(env, sensor_cfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

@dataclass
class HeightMapDomainRandCfg:
    """Domain randomization config for elevation maps (geometric transforms only).
    
    Note: Vertical offset and noise are handled by NoiseModelWithAdditiveBias
    """
    
    
    # Roll/pitch rotation noise (height bias from tilt)
    roll_pitch_noise_range: tuple[float, float] = (-0.05, 0.05) 
    
    # Yaw rotation noise (odometry error)
    yaw_rotation_range: tuple[float, float] = (-0.05, 0.05) 
    yaw_resample_interval: int = 1
    
    
    # Map repeat (delay simulation)
    map_repeat_prob: float = 0.2  # 20% chance to use stale map

def height_scan_full_domain_rand(
    env, 
    sensor_cfg,
    offset: float = 0.5,
    cfg: HeightMapDomainRandCfg = None
) -> torch.Tensor:
    """Height scan with full domain randomization.
    
    Implements geometric elevation map domain randomizations:
    1. Yaw rotation (odometry error)
    2. Roll/pitch bias (tilt error) 
    3. Map repeat/delay (stale observations)
    
    Note: Vertical offset + noise are handled by NoiseModelWithAdditiveBias
    
    Args:
        env: The RL environment
        sensor_cfg: Height scanner sensor configuration
        offset: Offset to subtract from height values (default: 0.5)
        cfg: Domain randomization parameters (uses defaults if None)
    
    Returns:
        Flattened height map (num_envs, num_rays) with domain randomization applied
    """
    
    if cfg is None:
        cfg = HeightMapDomainRandCfg()
    
    # ========== 1. Get base height map (CORRECTED) ==========
    sensor = env.scene.sensors[sensor_cfg.name]
    # height = sensor_height - hit_point_height - offset
    height_data = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    height_data = torch.clamp(height_data, min=-2.0, max=2.0)
    
    num_envs = height_data.shape[0]
    num_rays = height_data.shape[1]
    grid_size = int(num_rays ** 0.5)
    
    # Reshape to 2D grid for spatial operations
    height_map = height_data.view(num_envs, grid_size, grid_size)
    
    # ========== 2. Yaw rotation ==========
    if not hasattr(env, '_height_yaw_noise'):
        env._height_yaw_noise = torch.zeros(num_envs, device=env.device)
        env._height_yaw_counter = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    
    env._height_yaw_counter += 1
    resample_mask = env._height_yaw_counter >= cfg.yaw_resample_interval
    
    if resample_mask.any():
        yaw_min, yaw_max = cfg.yaw_rotation_range
        new_yaw = torch.rand(num_envs, device=env.device) * (yaw_max - yaw_min) + yaw_min
        env._height_yaw_noise = torch.where(resample_mask, new_yaw, env._height_yaw_noise)
        env._height_yaw_counter = torch.where(resample_mask, 
                                              torch.zeros_like(env._height_yaw_counter), 
                                              env._height_yaw_counter)
    
    cos_yaw = torch.cos(env._height_yaw_noise)
    sin_yaw = torch.sin(env._height_yaw_noise)
    
    theta = torch.zeros(num_envs, 2, 3, device=env.device)
    theta[:, 0, 0] = cos_yaw
    theta[:, 0, 1] = -sin_yaw
    theta[:, 1, 0] = sin_yaw
    theta[:, 1, 1] = cos_yaw
    
    grid = F.affine_grid(theta, height_map.unsqueeze(1).size(), align_corners=False)
    height_map = F.grid_sample(height_map.unsqueeze(1), grid, 
                               mode='bilinear', align_corners=False).squeeze(1)
    
    # ========== 3. Roll/pitch bias ==========
    bias_min, bias_max = cfg.roll_pitch_noise_range
    x_bias = torch.rand(num_envs, device=env.device) * (bias_max - bias_min) + bias_min
    y_bias = torch.rand(num_envs, device=env.device) * (bias_max - bias_min) + bias_min
    
    x_gradient = torch.linspace(-1, 1, grid_size, device=env.device)
    y_gradient = torch.linspace(-1, 1, grid_size, device=env.device)
    x_grid, y_grid = torch.meshgrid(x_gradient, y_gradient, indexing='ij')
    x_grid = x_grid.unsqueeze(0).expand(num_envs, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(num_envs, -1, -1)
    
    roll_pitch_noise = x_bias.view(-1, 1, 1) * x_grid + y_bias.view(-1, 1, 1) * y_grid
    height_map = height_map + roll_pitch_noise
    
    
    # ========== 4. Map repeat ==========
    if not hasattr(env, '_height_map_buffer'):
        env._height_map_buffer = height_map.clone()
    
    update_mask = torch.rand(num_envs, device=env.device) > cfg.map_repeat_prob
    update_mask_expanded = update_mask.view(-1, 1, 1).expand_as(height_map)
    
    env._height_map_buffer = torch.where(update_mask_expanded, height_map, env._height_map_buffer)
    
    #check if nan
    if torch.isnan(env._height_map_buffer).any():
        raise ValueError("NaN detected in height map buffer after domain randomization.")
    
    # ========== Return: Flatten back to (num_envs, num_rays) ==========
    return env._height_map_buffer.view(num_envs, -1)   
