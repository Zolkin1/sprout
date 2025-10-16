import math
from typing import TYPE_CHECKING

from warp import quat

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    wrap_to_pi,
    yaw_quat,
)



if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.stones_cmd_cfg import StonesCommandCfg

from robot_rl.tasks.manager_based.robot_rl.constants import IS_DEBUG, STONES



class StonesCommandTerm(CommandTerm):
    def __init__(self, cfg: "StonesCommandCfg", env):
        super().__init__(cfg, env)

        self.debug_vis = cfg.debug_vis
        
        self.robot = env.scene[cfg.asset_name]
        
        self.next_stone_pos = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)
        self.nextnext_stone_pos = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)
        
        
        self.stone_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=self.device)
        self.stone_quat[:, 0] = 1.0 # identity quat
        
        # todo: for debug!
        # generate random ith step, from 0 to num_stones-1 as interger tensor
        self.ith_step = torch.randint(0, STONES.num_stones, (env.num_envs,), dtype=torch.long, device=self.device)

        

        self.abs_x = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.abs_z = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device) 
        self.abs_y = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
    @property
    def command(self):
        return self.next_stone_pos
    
    def _resample_command(self, env_ids):
        self._update_command()
        return
    
    def _update_command(self):
        # === Check if episode buffer exists ===
        if not hasattr(self._env, "episode_length_buf"):
            return
        
        # Extract terrain info
        terrain = self._env.scene.terrain
        rel_x = terrain.env_terrain_infos["rel_x"] #(num_envs, num_stones)
        rel_z = terrain.env_terrain_infos["rel_z"] #(num_envs, num_stones)
        start_stone_pos_w = terrain.env_terrain_infos["start_stone_pos"] + terrain.env_origins #(num_envs, 3)
        
        
        # === Reset handling ===
        reset_mask = self._env.episode_length_buf == 0
        if reset_mask.any():
            # Get positions for reset environments
            robot_pos_w_init = self.robot.data.root_pos_w[reset_mask]  # (num_reset, 3)
            start_pos = start_stone_pos_w[reset_mask]  # (num_reset, 3)
        
            # evenly interpolate x positions from robot to platform
            t = torch.linspace( 1, STONES.num_init_steps, STONES.num_init_steps, device=self.device) / STONES.num_init_steps  # (num_init_steps,)
        
            # Compute initial stepping stones (interpolated from robot to platform)
            abs_x_init = robot_pos_w_init[:, 0:1] + (start_pos[:, 0:1] - robot_pos_w_init[:, 0:1]) * t #(num_reset, num_init_steps)
            abs_z_init = start_pos[:, 2:3].expand_as(abs_x_init) #(num_reset, num_init_steps)

            # Concatenate with terrain stone sequence (cumulative offsets from start position)
            stone_x_offsets = torch.cumsum(rel_x[reset_mask], dim=1) # (num_reset, num_stones)
            stone_z_offsets = torch.cumsum(rel_z[reset_mask], dim=1) # (num_reset, num_stones)

            self.abs_x[reset_mask] = torch.cat([abs_x_init, start_pos[:, 0:1] + stone_x_offsets], dim=1)
            self.abs_z[reset_mask] = torch.cat([abs_z_init, start_pos[:, 2:3] + stone_z_offsets], dim=1)
            self.abs_y[reset_mask] = start_pos[:, 1:2]
        
        # --- Current stepping stone ---
        self.next_stone_pos[:, 0] = torch.gather(self.abs_x, 1, self.ith_step.unsqueeze(1)).squeeze(1)
        self.next_stone_pos[:, 1] = start_stone_pos_w[:, 1]
        self.next_stone_pos[:, 2] = torch.gather(self.abs_z, 1, self.ith_step.unsqueeze(1)).squeeze(1)
        
        # --- Next stepping stone ---
        idx_next = torch.clamp(
            self.ith_step + 1, 
            max=STONES.num_stones + STONES.num_init_steps - 1
        )
        
        self.nextnext_stone_pos[:, 0] = torch.gather(self.abs_x, 1, idx_next.unsqueeze(1)).squeeze(1)
        self.nextnext_stone_pos[:, 1] = start_stone_pos_w[:, 1]
        self.nextnext_stone_pos[:, 2] = torch.gather(self.abs_z, 1, idx_next.unsqueeze(1)).squeeze(1)

        from robot_rl.tasks.manager_based.robot_rl.constants import IS_DEBUG
        if IS_DEBUG:
            # Debug prints - show full tensors
            with torch.no_grad():
                torch.set_printoptions(profile="full", linewidth=1500, precision=4, sci_mode=False)
                print("=" * 80)
                print("MLIP DEBUG: y_out (positions/orientations):")
                
        return

    def _update_metrics(self):
        return



    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            self.nextstone_visualizer = VisualizationMarkers(self.cfg.nextstone_cfg)
            self.nextnextstone_visualizer = VisualizationMarkers(self.cfg.nextnextstone_cfg)
            self.nextstone_visualizer.set_visibility(True)
            self.nextnextstone_visualizer.set_visibility(True)
            
            self.origin_visualizer = VisualizationMarkers(self.cfg.originframe_cfg)
            self.origin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "nextstone_visualizer"):
                self.nextstone_visualizer.set_visibility(False)
            if hasattr(self, "nextnextstone_visualizer"):
                self.nextnextstone_visualizer.set_visibility(False)
        return
    
    def _debug_vis_callback(self, event):
        if self.debug_vis:
            self.nextstone_visualizer.visualize(self.next_stone_pos,self.stone_quat)
            self.nextnextstone_visualizer.visualize(self.nextnext_stone_pos,self.stone_quat)
            self.origin_visualizer.visualize(self._env.scene.terrain.env_origins, self.stone_quat)