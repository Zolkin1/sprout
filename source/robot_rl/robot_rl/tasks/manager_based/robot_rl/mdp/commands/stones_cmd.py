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
        
        # for debug!
        #genertate random ith step, from 0 to num_stones-1 as interger tensor
        self.ith_step = torch.randint(0, STONES.num_stones, (env.num_envs,), dtype=torch.long, device=self.device)

        self.abs_x = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.abs_z = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device) 
    @property
    def command(self):
        return self.next_stone_pos
    
    def _resample_command(self, env_ids):
        self._update_command()
        return
    
    def _update_command(self):
        rel_x = self._env.scene.terrain.env_terrain_infos["rel_x"]
        rel_z = self._env.scene.terrain.env_terrain_infos["rel_z"]
        platform_pos_w = (
          self._env.scene.terrain.env_terrain_infos["start_platform_pos"]
          + self._env.scene.terrain.env_origins
        )  # shape (num_envs, 3)
        
        
        # === Check if episode buffer exists ===
        if not hasattr(self._env, "episode_length_buf"):
            return
        
        # === Reset handling ===
        reset_mask = self._env.episode_length_buf == 0
        if reset_mask.any():
            # record robot base positions for environments being reset
            robot_pos_w_init = self.robot.data.root_pos_w[reset_mask]  # (num_reset, 3)
        
            # evenly interpolate x positions from robot to platform
            t = torch.linspace(
                1, STONES.num_init_steps, STONES.num_init_steps, device=self.device
            ) / STONES.num_init_steps  # (num_init_steps,)
        
            abs_x_init = (
                robot_pos_w_init[:, 0:1]
                + (platform_pos_w[reset_mask, 0:1] - robot_pos_w_init[:, 0:1]) * t[None, :]
            )  # (num_reset, num_init_steps)
        
            # concatenate with terrain stone sequence
            abs_x_reset = torch.cat(
                [
                    abs_x_init,
                    platform_pos_w[reset_mask, 0:1]
                    + torch.cumsum(rel_x[reset_mask], dim=1),
                ],
                dim=1,
            )
        
            abs_z_reset = torch.cat(
                [
                    torch.zeros_like(abs_x_init),
                    platform_pos_w[reset_mask, 2:3]
                    + torch.cumsum(rel_z[reset_mask], dim=1),
                ],
                dim=1,
            )
        
            # assign back
            self.abs_x[reset_mask] = abs_x_reset
            self.abs_z[reset_mask] = abs_z_reset
        
        # --- Current stepping stone ---
        self.next_stone_pos[:, 0] = torch.gather(self.abs_x, 1, self.ith_step.unsqueeze(1)).squeeze(1)
        self.next_stone_pos[:, 1] = platform_pos_w[:, 1]
        self.next_stone_pos[:, 2] = torch.gather(self.abs_z, 1, self.ith_step.unsqueeze(1)).squeeze(1)
        
        # --- Next stepping stone ---
        idx_next = torch.clamp(
            self.ith_step + 1, 
            max=STONES.num_stones + STONES.num_init_steps - 1
        )
        
        self.nextnext_stone_pos[:, 0] = torch.gather(self.abs_x, 1, idx_next.unsqueeze(1)).squeeze(1)
        self.nextnext_stone_pos[:, 1] = platform_pos_w[:, 1]
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