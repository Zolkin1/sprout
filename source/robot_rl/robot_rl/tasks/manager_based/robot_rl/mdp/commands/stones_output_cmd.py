import math
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    matrix_from_quat
)

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.clf import CLF

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.beizer import bezier_deg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.hlip import HLIP_3D, HLIP_P2, solve_time2reach_pdes_batched, solve_orbital_energy_batched, solve_velocity_or_momentum_positive_from_E_batched
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.cubic import cubic_spline_coeff_batched, cubic_spline_eval
from enum import IntEnum

class YIdx(IntEnum):
    comx = 0
    comy = 1
    comz = 2
    pelvis_roll = 3
    pelvis_pitch = 4
    pelvis_yaw = 5
    swing_foot_x = 6
    swing_foot_y = 7
    swing_foot_z = 8
    swing_foot_roll = 9
    swing_foot_pitch = 10
    swing_foot_yaw = 11
    


if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.stones_output_cmd_cfg import StonesOutputCommandCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.mlip_cmd import euler_rates_to_omega_b, get_euler_from_quat, _transfer_to_local_frame
from robot_rl.tasks.manager_based.robot_rl.constants import IS_DEBUG, STONES, TEST_FLAT, ZERO_EPS

from .phase_var import PhaseVar

def tensor_t_now(env):
    return env.episode_length_buf * env.step_dt

def convert_momentum_to_hlip_convention(h_st_w: torch.Tensor) -> torch.Tensor:
    h_st_w_converted = h_st_w.clone()
    h_st_w_converted[:,0] = -h_st_w_converted[:,0]
    return h_st_w_converted

def compute_z_tilde_batched(zc: torch.Tensor, hdes: torch.Tensor, ldes: torch.Tensor, xcom_st: torch.Tensor) -> torch.Tensor:
    """compute z_tilde for HLIP given desired foot placement and current com state in stance foot frame
    z0, hdes, ldes, xcom_st are (N, ) tensors
    return:
    z_tilde (N, ) tensor
    """
    z_tilde = zc.clone()
    nonzero_ldes_mask = ldes.abs() > ZERO_EPS
    z_tilde[nonzero_ldes_mask] = zc[nonzero_ldes_mask] - hdes[nonzero_ldes_mask] / ldes[nonzero_ldes_mask] * xcom_st[nonzero_ldes_mask]
    return z_tilde

def convert_foot_pos_to_midsole(foot_pos: torch.Tensor, 
                                mid_foot_offset: torch.Tensor, 
                                foot_ori_quat: torch.Tensor) -> torch.Tensor:
    """Convert foot position and velocity from foot origin frame to mid-sole frame.
    foot_pos: (N, 3) tensor
    mid_foot_offset: (3, ) tensor
    foot_ori_quat: (N, 4) tensor
    return:
    midsole_pos: (N, 3) tensor
    """
    mid_foot_offset_world = quat_apply(foot_ori_quat, mid_foot_offset.unsqueeze(0).expand(foot_ori_quat.shape[0], -1))
    midsole_pos = foot_pos + mid_foot_offset_world  # (N, 3)
    return midsole_pos

def convert_foot_pos_vel_to_midsole(foot_pos: torch.Tensor, 
                                foot_vel: torch.Tensor, 
                                mid_foot_offset: torch.Tensor, 
                                foot_ori_quat: torch.Tensor,
                                foot_omega_w: torch.Tensor) -> torch.Tensor:
    """Convert foot position and velocity from foot origin frame to mid-sole frame.
    foot_pos: (N, 3) tensor
    foot_vel: (N, 3) tensor
    mid_foot_offset: (3, ) tensor
    foot_ori_quat: (N, 4) tensor
    foot_omega_w: (N, 3) tensor
    return:
    midsole_pos: (N, 3) tensor
    midsole_vel: (N, 3) tensor
    """
    mid_foot_offset_world = quat_apply(foot_ori_quat, mid_foot_offset.unsqueeze(0).expand(foot_ori_quat.shape[0], -1))
    midsole_pos = foot_pos + mid_foot_offset_world  # (N, 3)
    
    midsole_vel = foot_vel + torch.cross(foot_omega_w, mid_foot_offset_world, dim=1)  # (N, 3)

    return midsole_pos, midsole_vel


class StonesOutputCommandTerm(CommandTerm):
    z0: torch.Tensor  # (num_envs,) nominal CoM height
    y_nom: float  # nominal lateral foot placement
    use_momentum: bool #where the second coordinate of HLIP is momentum or velocity
    tSSplus: torch.Tensor  # (num_envs,) time of last SS+ event 
    TSS: torch.Tensor  # (num_envs,) single support duration
    TDS: torch.Tensor  # (num_envs,) double support duration

    phase_var: PhaseVar  # (num_envs,) phase variable for footstep timing
    stance_idx: torch.Tensor  # (num_envs,) current stance foot index (0: left, 1: right)
    
    ith_step: torch.Tensor  # (num_envs,) current step index
    
    stance_foot_pos_0: torch.Tensor  # (num_envs, 3) stance foot position at the beginning of the step
    swing_foot_pos_0: torch.Tensor  # (num_envs, 3) swing foot position at the beginning of the step
    stance_foot_ori_quat_0: torch.Tensor  # (num_envs, 4) stance foot orientation at the beginning of the step
    stance_foot_ori_0: torch.Tensor  # (num_envs, 3) stance foot orientation (roll, pitch, yaw) at the beginning of the step

    hdes: torch.Tensor  # (num_envs,) swing foot target z position from current stance foot
    ldes: torch.Tensor  # (num_envs,) swing foot target x position from current stance foot
    hdes_next: torch.Tensor  # (num_envs,) swing foot target z position from next stance foot
    ldes_next: torch.Tensor  # (num_envs,) swing foot target x position from next stance foot

    xcomf_des: torch.Tensor  # (num_envs,) desired xCOM position from current stance foot
    zcomf_des: torch.Tensor  # (num_envs,) desired zCOM position from current stance foot
    dzcomf_des: torch.Tensor  # (num_envs,) desired zCOM velocity from current stance foot
    
    current_stone_pos: torch.Tensor  # (num_envs, 3) current stone position in world frame
    next_stone_pos:  torch.Tensor  # (num_envs, 3) next stone position in world frame
    nextnext_stone_pos:  torch.Tensor  # (num_envs, 3) next next stone position in world frame
    
    stone_quat: torch.Tensor  # (num_envs, 4) stone orientation quaternion in world frame
    
    abs_x: torch.Tensor  # (num_envs, STONES.num_stones + STONES.num_init_steps) absolute x positions of stones
    abs_y: torch.Tensor  # (num_envs, STONES.num_stones + STONES.num_init_steps) absolute y positions of stones
    abs_z: torch.Tensor  # (num_envs, STONES.num_stones + STONES.num_init_steps) absolute z positions of stones
    last_stone_x: torch.Tensor  # (num_envs,) absolute x position of the last stone
    last_stone_z: torch.Tensor  # (num_envs,) absolute z position of the last stone
    max_stone_idx: int  # maximum stone index
    stone_width: torch.Tensor  # (num_envs,) stone width in y direction
    
    def __init__(self, cfg: "StonesOutputCommandCfg", env):
        super().__init__(cfg, env)

        #mid foot offset from roll_link frame to foot mid sole frame
        self.mid_foot_offset = torch.tensor([0.035, 0.0, -0.047558], device=self.device)

        self.z0 = torch.full((self.num_envs,), cfg.z0, device=self.device)
        self.y_nom = cfg.y_nom
        self.use_momentum = cfg.use_momentum    
        
        self.tSSplus = tensor_t_now(env)
        self.TSS = torch.full((self.num_envs,), 0.4, device=self.device)
        self.TDS = torch.full((self.num_envs,), 0.0, device=self.device)

        self.phase_var = PhaseVar(self.tSSplus , self.tSSplus + self.TSS)
        self.stance_idx = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device) #start with left foot stance


        self.debug_vis = cfg.debug_vis

        self.robot = env.scene[cfg.asset_name]
        self.terrain = env.scene.terrain
        
        #list of int, left foot idx 0, right foot idx 1
        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        self.upper_body_joint_idx = self.robot.find_joints(cfg.upper_body_joint_name)[0]

        self.metrics = {}

        n_output = 12 + len(self.upper_body_joint_idx)
        self.y_out = torch.zeros((self.num_envs, n_output), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, n_output), device=self.device)
        self.y_act = torch.zeros((self.num_envs, n_output), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, n_output), device=self.device)

        grav = torch.abs(torch.tensor(self._env.cfg.sim.gravity[2], device=self.device))
        
        if TEST_FLAT:
            self.hlip = HLIP_3D(
                num_envs=self.num_envs,
                grav=grav,
                z0=self.z0,
                TSS=self.TSS,
                TDS=self.TDS,
                use_momentum=self.use_momentum,
                use_feedback=False
            )
        else:
            self.hlip = HLIP_P2(
                num_envs=self.num_envs,
                grav=grav,
                z0=self.z0,
                TSS=self.TSS,
                TDS=self.TDS,
                use_momentum=self.use_momentum,
                use_feedback=False
            )

        self.mass = sum(self.robot.data.default_mass.T)[0].to(self.device)

        self.delta_yaw = 0.0

        self.clf = CLF(
            n_output,
            self._env.cfg.sim.dt * self._env.cfg.sim.render_interval,
            batch_size=self.num_envs,
            Q_weights=np.array(cfg.Q_weights),
            R_weights=np.array(cfg.R_weights),
            device=self.device,
        )

        self.v = torch.zeros((self.num_envs), device=self.device)
        self.vdot = torch.zeros((self.num_envs), device=self.device)
        self.v_buffer = torch.zeros((self.num_envs, 100), device=self.device)
        self.vdot_buffer = torch.zeros((self.num_envs, 100), device=self.device)


        self.stance_foot_pos_0 = torch.zeros((self.num_envs, 3), device=self.device) 
        self.swing_foot_pos_0 = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_ori_quat_0 = torch.zeros((self.num_envs, 4), device=self.device) 
        self.stance_foot_ori_quat_0_zerorollpitch = torch.zeros((self.num_envs, 4), device=self.device)
        self.stance_foot_ori_0 = torch.zeros((self.num_envs, 3), device=self.device) 

        self.is_initialized = False
        
 
        """stones related
        for ith step:
        prev stone is the stone where swing foot is initially placed
        current stone is the stone where stance foot is placed
        next stone is the stone where swing foot is stepping to
        next next stone is the stone after next stone, for previewing
        """
        self.ith_step = torch.full((self.num_envs,), 0, dtype=torch.long, device=self.device)
        self.prev_stone_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.current_stone_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.next_stone_pos = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)
        self.nextnext_stone_pos = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)

        #output related
        self.hdes = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.ldes = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        # self.pitchdes = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.hdes_next = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.ldes_next = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.xcomf_des = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.zcom0_des = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.zcomf_des = self.z0.clone()
        self.dzcomf_des = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
                                
        self.stone_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=self.device)
        self.stone_quat[:, 0] = 1.0 # identity quat

        self.abs_x = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.abs_y = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.abs_pitch = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.abs_z = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.last_stone_x = torch.zeros((env.num_envs,), dtype=torch.float32, device=self.device)
        self.last_stone_z = torch.zeros((env.num_envs,), dtype=torch.float32, device=self.device)
        self.max_stone_idx = STONES.num_stones + STONES.num_init_steps -1
        self.stone_width = torch.zeros((env.num_envs,), dtype=torch.float32, device=self.device)
        
        self.is_walking_env = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.is_standing_cmd = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        self.swing_foot_pos_actual = torch.zeros((self.num_envs, 3), device=self.device)
        self.swing_foot_error_at_contact = torch.zeros((self.num_envs, 3), device=self.device)
        self.mask_at_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # for debug visualization
        self.foottarget_vis_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.foottarget_vis_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.swingfoot_vis_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.swingfoot_vis_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.com_frame_vis_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.com_frame_vis_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # for stepping stones, always targeting zero yaw
        self.target_yaw_stone = torch.zeros_like(self.stance_foot_ori_0[:, 2])
        

    @property
    def command(self):
        return self.v
    
    def _resample_command(self, env_ids):
        self._update_command()
        return
    
    def _update_command(self):
         #check nan
        if torch.isnan(self.robot.data.body_com_pos_w).any():
            raise ValueError("NaN detected in body_com_pos_w")

        self.timeBasedDomainContactStatusSwitchwithStanding()

        self.update_walking_target(self.is_walking_env)
        
        self.compute_actual()

        if TEST_FLAT == False:
            self.compute_desired_stones(self.is_walking_env)
        else:
            self.compute_desired_flat()

        self.compute_desired_standing(~self.is_walking_env)

        # how to handle for the first step?
        # i.e. v is not defined
        self.vdot, self.v = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, self.cfg.yaw_idx)
        # if torch.sum(self.v_buffer) == 0:
        #     # (E,) -> (E,1) -> broadcast to (E,100) on assignment
        #     self.v_buffer[:] = self.v.unsqueeze(1)
        #     self.vdot_buffer[:] = self.vdot.unsqueeze(1)

        # else:
        #     self.v_buffer = torch.cat([self.v_buffer[:, 1:], self.v.unsqueeze(-1)], dim=-1)
        #     self.vdot_buffer = torch.cat([self.vdot_buffer[:, 1:], self.vdot.unsqueeze(-1)], dim=-1)
        with torch.no_grad():
            if torch.sum(self.v_buffer) == 0:
                # First step: fill everything
                self.v_buffer[:] = self.v.unsqueeze(1)
                self.vdot_buffer[:] = self.vdot.unsqueeze(1)
            else:
                # Shift left and append in place
                self.v_buffer[:, :-1] = self.v_buffer[:, 1:]
                self.v_buffer[:, -1] = self.v
                self.vdot_buffer[:, :-1] = self.vdot_buffer[:, 1:]
                self.vdot_buffer[:, -1] = self.vdot
        from robot_rl.tasks.manager_based.robot_rl.constants import IS_DEBUG
        if IS_DEBUG:
            # Debug prints - show full tensors
            with torch.no_grad():
                torch.set_printoptions(profile="full", linewidth=1500, precision=4, sci_mode=False)
                print("=" * 80)
                print("MLIP DEBUG: y_out (positions/orientations):")
                print(self.y_out)
                print("MLIP DEBUG: dy_out (velocities/angular velocities):")
                print(self.dy_out)
                print("MLIP DEBUG: y_act (actual positions/orientations):")
                print(self.y_act)
                print("MLIP DEBUG: dy_act (actual velocities/angular velocities):")
                print(self.dy_act)
                print("MLIP DEBUG: v (CLF value):")
                print(self.v)
                print("MLIP DEBUG: vdot (CLF derivative):")
                print(self.vdot)
                print("=" * 80)
        return

    def _update_metrics(self):
        self.metrics["error_sw_x"] = torch.abs(self.y_out[:, YIdx.swing_foot_x] - self.y_act[:, YIdx.swing_foot_x])
        self.metrics["error_sw_y"] = torch.abs(self.y_out[:, YIdx.swing_foot_y] - self.y_act[:, YIdx.swing_foot_y])
        self.metrics["error_sw_z"] = torch.abs(self.y_out[:, YIdx.swing_foot_z] - self.y_act[:, YIdx.swing_foot_z])
        self.metrics["error_sw_roll"] = torch.abs(self.y_out[:, YIdx.swing_foot_roll] - self.y_act[:, YIdx.swing_foot_roll])
        self.metrics["error_sw_pitch"] = torch.abs(self.y_out[:, YIdx.swing_foot_pitch] - self.y_act[:, YIdx.swing_foot_pitch])
        self.metrics["error_sw_yaw"] = torch.abs(self.y_out[:, YIdx.swing_foot_yaw] - self.y_act[:, YIdx.swing_foot_yaw])

        self.metrics["error_com_x"] = torch.abs(self.y_out[:, YIdx.comx] - self.y_act[:, YIdx.comx])
        self.metrics["error_com_y"] = torch.abs(self.y_out[:, YIdx.comy] - self.y_act[:, YIdx.comy])
        self.metrics["error_com_z"] = torch.abs(self.y_out[:, YIdx.comz] - self.y_act[:, YIdx.comz])
        self.metrics["error_pelvis_roll"] = torch.abs(self.y_out[:, YIdx.pelvis_roll] - self.y_act[:, YIdx.pelvis_roll])
        self.metrics["error_pelvis_pitch"] = torch.abs(self.y_out[:, YIdx.pelvis_pitch] - self.y_act[:, YIdx.pelvis_pitch])
        self.metrics["error_pelvis_yaw"] = torch.abs(self.y_out[:, YIdx.pelvis_yaw] - self.y_act[:, YIdx.pelvis_yaw])

        self.metrics["error_ub_joints"] = torch.abs(self.y_out[:, 12:] - self.y_act[:, 12:]).mean(dim=1)
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot
        self.metrics["avg_clf"] = torch.mean(self.v_buffer, dim=-1)
        return

    def reset_stone_global(self, reset_mask):
        if reset_mask.any():
            # Extract terrain info
            rel_x = self.terrain.env_terrain_infos["rel_x"] #(num_envs, num_stones)
            rel_z = self.terrain.env_terrain_infos["rel_z"] #(num_envs, num_stones)
            start_stone_pos_w = self.terrain.env_terrain_infos["start_stone_pos"] + self.terrain.env_origins #(num_envs, 3)
            self.stone_width = self.terrain.env_terrain_infos["stone_y"].squeeze(-1)  #(num_envs,)
            abs_pitch = self.terrain.env_terrain_infos["abs_pitch"] #(num_envs, num_stones)
            # Get positions for reset environments
            foot_pos_w_init = self.stance_foot_pos_0[reset_mask]  # (num_reset, 3)
            start_pos = start_stone_pos_w[reset_mask]  # (num_reset, 3)
        
            # evenly interpolate x positions from robot to platform
            t = torch.linspace( 1, STONES.num_init_steps, STONES.num_init_steps, device=self.device) / STONES.num_init_steps  # (num_init_steps,)
        
            # Compute initial stepping stones (interpolated from robot to platform)
            abs_x_init = foot_pos_w_init[:, 0:1] + (start_pos[:, 0:1] - foot_pos_w_init[:, 0:1]) * t #(num_reset, num_init_steps)
            abs_z_init = start_pos[:, 2:3].expand_as(abs_x_init) #(num_reset, num_init_steps)

            # Concatenate with terrain stone sequence (cumulative offsets from start position)
            stone_x_offsets = torch.cumsum(rel_x[reset_mask], dim=1) # (num_reset, num_stones)
            stone_z_offsets = torch.cumsum(rel_z[reset_mask], dim=1) # (num_reset, num_stones)

            self.abs_x[reset_mask] = torch.cat([abs_x_init, start_pos[:, 0:1] + stone_x_offsets], dim=1)
            self.abs_z[reset_mask] = torch.cat([abs_z_init, start_pos[:, 2:3] + stone_z_offsets], dim=1)
            self.abs_y[reset_mask] = start_pos[:, 1:2] #use stone y as constant y position
            self.abs_pitch[reset_mask] = torch.cat([torch.zeros((reset_mask.sum(), STONES.num_init_steps), device=self.device), abs_pitch[reset_mask]], dim=1)
            self.last_stone_x[reset_mask] = self.abs_x[reset_mask, -1]
            self.last_stone_z[reset_mask] = self.abs_z[reset_mask, -1]
            self.max_stone_idx = self.abs_x.shape[1] - 1

        return
    
    def update_ithstep_stones_info(self, ithstep, mask, current_stone_pos):
        """Update next and next-next stepping stone positions only for masked envs."""
        if not torch.any(mask):
            return
        
        # Determine which envs are within vs beyond terrain
        beyond_terrain = ithstep[mask] > self.max_stone_idx
        at_last_stone = ithstep[mask] == self.max_stone_idx
  
        # --- Upcoming stepping stone, the target stone for swing foot ---
        idx_next = torch.clamp(ithstep[mask], min=0, max=self.max_stone_idx)
        self.next_stone_pos[mask, 0] = torch.gather(self.abs_x[mask], 1, idx_next.unsqueeze(1)).squeeze(1)
        self.next_stone_pos[mask, 1] = torch.gather(self.abs_y[mask], 1, idx_next.unsqueeze(1)).squeeze(1)
        self.next_stone_pos[mask, 2] = torch.gather(self.abs_z[mask], 1, idx_next.unsqueeze(1)).squeeze(1)
        
        # self.pitchdes[mask] = torch.gather(self.abs_pitch[mask], 1, idx_next.unsqueeze(1)).squeeze(1)

        # --- Next Upcoming stepping stone, preview ---
        idx_next_next = torch.clamp(ithstep[mask] + 1, min=0, max=self.max_stone_idx)
        self.nextnext_stone_pos[mask, 0] = torch.gather(self.abs_x[mask], 1, idx_next_next.unsqueeze(1)).squeeze(1)
        self.nextnext_stone_pos[mask, 1] = torch.gather(self.abs_y[mask], 1, idx_next_next.unsqueeze(1)).squeeze(1)
        self.nextnext_stone_pos[mask, 2] = torch.gather(self.abs_z[mask], 1, idx_next_next.unsqueeze(1)).squeeze(1)
        
        # === Case 1: At last stone - next is real, next-next is pseudo ===
        if torch.any(at_last_stone):
            at_last_mask_global = torch.zeros_like(mask, dtype=torch.bool)
            at_last_mask_global[mask] = at_last_stone
            # next_stone_pos already correct from gather (last real stone)
            # Create pseudo next-next stone
            self.nextnext_stone_pos[at_last_mask_global, 0] = self.next_stone_pos[at_last_mask_global, 0] + 0.3
            self.nextnext_stone_pos[at_last_mask_global, 1] = self.next_stone_pos[at_last_mask_global, 1]
            self.nextnext_stone_pos[at_last_mask_global, 2] = self.next_stone_pos[at_last_mask_global, 2]
            # check pitchdes for pseudo stone

        # === Case 2: Beyond terrain - both next and next-next are pseudo ===
        if torch.any(beyond_terrain):
            beyond_mask_global = torch.zeros_like(mask, dtype=torch.bool)
            beyond_mask_global[mask] = beyond_terrain

            # Calculate steps beyond
            steps_beyond = ithstep[beyond_mask_global] - self.max_stone_idx

            # Create pseudo stones
            self.next_stone_pos[beyond_mask_global, 0] = self.last_stone_x[beyond_mask_global] + 0.3 * steps_beyond
            self.next_stone_pos[beyond_mask_global, 1] = self.abs_y[beyond_mask_global, 0]
            self.next_stone_pos[beyond_mask_global, 2] = self.last_stone_z[beyond_mask_global]

            self.nextnext_stone_pos[beyond_mask_global, 0] = self.last_stone_x[beyond_mask_global] + 0.3 * (steps_beyond + 1)
            self.nextnext_stone_pos[beyond_mask_global, 1] = self.abs_y[beyond_mask_global, 0]
            self.nextnext_stone_pos[beyond_mask_global, 2] = self.last_stone_z[beyond_mask_global]
            
            # For pseudo stones, set pitch to zero
            # self.pitchdes[beyond_mask_global] = 0.0
            
        
        self.hdes[mask] = self.next_stone_pos[mask, 2] - current_stone_pos[mask, 2]
        self.ldes[mask] = self.next_stone_pos[mask, 0] - current_stone_pos[mask, 0]
        if self.cfg.use_stance_foot_pos_as_ref == True:
            self.ldes[mask] = self.next_stone_pos[mask, 0] - self.stance_foot_pos_0[mask, 0]
            self.hdes[mask] = self.next_stone_pos[mask, 2] - self.stance_foot_pos_0[mask, 2]
            #TODO: check hdes with stance foot pos as ref
            
            
        self.hdes_next[mask] = self.nextnext_stone_pos[mask, 2] - self.next_stone_pos[mask, 2]
        self.ldes_next[mask] = self.nextnext_stone_pos[mask, 0] - self.next_stone_pos[mask, 0]
        
        # #debug, hardcode 0.3
        # self.ldes[mask] = 0.3
        # self.ldes_next[mask] = 0.3
        # self.hdes[mask] = 0.0
        # self.hdes_next[mask] = 0.0
        
        if torch.any(self.ldes[mask] <= ZERO_EPS):
            #raise error
            print("Warning: ldes is non-positive!")
        if torch.any(self.ldes_next < ZERO_EPS):
            #raise error
            print("Warning: ldes_next is non-positive!")
        return

    def timeBasedDomainContactStatusSwitch(self):
        tnow = tensor_t_now(self._env)

        #mask for time-based step switch
        mask_next_step = tnow >= (self.tSSplus + self.TSS - ZERO_EPS) 
        #mask for env reset, return is not guaranteed to be boolean tensor
        mask_reset_buf = getattr(self._env, "reset_buf", None) 

        if not self.is_initialized:
            mask_next_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
            mask_reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
            self.is_initialized = True
        #only mask next step for non-resetting envs, to avoid double update    
        mask_next_step = mask_next_step & torch.logical_not(mask_reset_buf.bool()) 
        
        if torch.any(mask_next_step):
            self.reset_stance_swing_foot(mask_next_step)
            self.ith_step[mask_next_step] += 1 #next step, increment ith step
            
            if TEST_FLAT == False:
                self.prev_stone_pos[mask_next_step] = self.current_stone_pos[mask_next_step].clone()

                self.current_stone_pos[mask_next_step, 0] = self.next_stone_pos[mask_next_step, 0]
                self.current_stone_pos[mask_next_step, 1] = self.stance_foot_pos_0[mask_next_step, 1]
                self.current_stone_pos[mask_next_step, 2] = self.next_stone_pos[mask_next_step, 2]
                # self.current_stone_pos[mask_next_step] = self.stance_foot_pos_0[mask_next_step]
                
                # mask_no_progress_this_step_local = (self.next_stone_pos[mask_next_step,0] - self.current_stone_pos[mask_next_step,0] ) > 0.2
                # mask_no_progress_this_step_global = torch.zeros_like(mask_next_step, dtype=torch.bool)
                # mask_no_progress_this_step_global[mask_next_step] = mask_no_progress_this_step_local
                # self.ith_step[mask_no_progress_this_step_global] -=1 #if no progress this step, do not increment ith step
                
                # mask_update_z = torch.logical_not(mask_no_progress_this_step_global) & mask_next_step
                # #update current stone z pos based on next stone z pos, to avoid z offset bewteen foot contact point and stance foot origin frame
                # self.current_stone_pos[mask_update_z, 2] = self.next_stone_pos[mask_update_z,2] 
                
                # Update next stone info
                self.update_ithstep_stones_info(self.ith_step, mask_next_step, self.current_stone_pos) #update next and next-next stone pos
                # # Handle skip logic
                # too_far_mask_local = self.ldes[mask_next_step] <= 0.0
                # while torch.any(too_far_mask_local):
                #     too_far_mask_global = torch.zeros_like(mask_next_step, dtype=torch.bool)
                #     too_far_mask_global[mask_next_step] = too_far_mask_local
                #     self.ith_step[too_far_mask_global] +=1 #if ldes is negative or zero, skip this stone by incrementing ith_step again
                #     self.update_ithstep_stones_info(self.ith_step, too_far_mask_global, self.current_stone_pos)
                #     too_far_mask_local = self.ldes[mask_next_step] <= 0.0  
            self.reset_impact(mask_next_step)
        
        if torch.any(mask_reset_buf):
            self.ith_step[mask_reset_buf] = 0 #env reset, reset ith step
            self.stance_idx[mask_reset_buf] = 0 #reset to left foot stance
            self.reset_stance_swing_foot(mask_reset_buf)
            if TEST_FLAT == False:
                #update current stone pos, assume first step z is always zero
                self.current_stone_pos[mask_reset_buf, 0:2] = self.stance_foot_pos_0[mask_reset_buf, 0:2].clone() #initialize current stone x-y pos based on stance foot
                self.current_stone_pos[mask_reset_buf, 2] = 0 #initialize current stone z pos to zero, asumming always launch on the platform
                #update prev stone pos, 
                self.prev_stone_pos[mask_reset_buf, 0] = self.current_stone_pos[mask_reset_buf,0].clone()
                self.prev_stone_pos[mask_reset_buf, 1] = self.current_stone_pos[mask_reset_buf, 1].clone() + self.cfg.y_nom
                self.prev_stone_pos[mask_reset_buf, 2] = self.current_stone_pos[mask_reset_buf, 2].clone()
                self.reset_stone_global(mask_reset_buf)
                self.update_ithstep_stones_info(self.ith_step, mask_reset_buf, self.current_stone_pos) #update next and next-next stone pos
            self.reset_impact(mask_reset_buf)

        #update phase var
        self.phase_var.update(tnow)

        return

    def timeBasedDomainContactStatusSwitchwithStanding(self):
        tnow = tensor_t_now(self._env)

        self.is_standing_cmd = self._env.command_manager.get_term("base_velocity").is_standing_env  # (N,)
    
        #mask for time-based step switch
        mask_timer_expired = ((tnow >= (self.tSSplus + self.TSS - ZERO_EPS) ) & self.is_walking_env)
        #mask for env reset, return is not guaranteed to be boolean tensor
        mask_reset_buf = getattr(self._env, "reset_buf", None) 
        #mask for stand to walk 
        mask_stand2walk = (~self.is_walking_env) & (~self.is_standing_cmd)  # Standing but command says walk


        if not self.is_initialized:
            mask_timer_expired = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
            mask_stand2walk = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
            mask_reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
            self.is_initialized = True
        #Only process timer events for non-resetting and walking envs to avoid double update
        mask_timer_expired = mask_timer_expired & torch.logical_not(mask_reset_buf.bool()) 
        #similarly, only process stand2walk for non-resetting and standing envs
        mask_stand2walk = mask_stand2walk & torch.logical_not(mask_reset_buf.bool()) 
        
        
        self.swing_foot_error_at_contact = torch.zeros((self.num_envs, 3), device=self.device)
        self.mask_at_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # TYPE 1: TIMER EXPIRED
        if torch.any(mask_timer_expired):
            # Walk -> Walk: walking & timer expired, not standing command
            mask_walk2walk = mask_timer_expired & (~self.is_standing_cmd) 
            
            self.swing_foot_error_at_contact[mask_walk2walk] = self.swing_foot_pos_actual[mask_walk2walk] - self.next_stone_pos[mask_walk2walk]
            self.mask_at_contact = mask_walk2walk.clone()
            
            
            # Walk -> Stand: walking & timer expired, standing command
            mask_walk2stand = mask_timer_expired & self.is_standing_cmd 
        
            self.reset_stance_swing_foot(mask_timer_expired)
            self.ith_step[mask_timer_expired] += 1 #next step, increment ith step
            self.prev_stone_pos[mask_timer_expired] = self.current_stone_pos[mask_timer_expired].clone()
            self.current_stone_pos[mask_timer_expired, 0] = self.next_stone_pos[mask_timer_expired, 0]
            self.current_stone_pos[mask_timer_expired, 1] = self.stance_foot_pos_0[mask_timer_expired, 1]
            self.current_stone_pos[mask_timer_expired, 2] = self.next_stone_pos[mask_timer_expired, 2]

            # Update next stone info
            self.update_ithstep_stones_info(self.ith_step, mask_timer_expired, self.current_stone_pos) #update next and next-next stone pos
            #only update tplus, TSS, hlip params for walk2walk envs
            self.reset_impact(mask_walk2walk)
            
            if mask_walk2stand.any():
                self.is_walking_env[mask_walk2stand] = False
                self.enter_standing_mode(mask_walk2stand)

        
        # TYPE 2: STAND -> WALK (immediate, no timer needed)
        if torch.any(mask_stand2walk):     
            self.reset_impact(mask_stand2walk)   
            self.is_walking_env[mask_stand2walk] = True

        # TYPE 3: ENVIRONMENT RESET
        if torch.any(mask_reset_buf):
            # Initialize is_walking_env based on command
            reset_as_walking = mask_reset_buf & (~self.is_standing_cmd)
            reset_as_standing = mask_reset_buf & self.is_standing_cmd
            # Common reset
            self.ith_step[mask_reset_buf] = 0 #env reset, reset ith step
            self.stance_idx[mask_reset_buf] = 0 #reset to left foot stance
            self.reset_stance_swing_foot(mask_reset_buf) #reset stance and swing foot, so always start from right
            #TODO: on hardware, robot always start with left foot stance
            
            #update current stone pos, assume first step z is always zero
            self.current_stone_pos[mask_reset_buf, 0:2] = self.stance_foot_pos_0[mask_reset_buf, 0:2].clone() #initialize current stone x-y pos based on stance foot
            self.current_stone_pos[mask_reset_buf, 2] = 0 #initialize current stone z pos to zero, asumming always launch on the platform
            #update prev stone pos, 
            self.prev_stone_pos[mask_reset_buf, 0] = self.current_stone_pos[mask_reset_buf,0].clone()
            self.prev_stone_pos[mask_reset_buf, 1] = self.current_stone_pos[mask_reset_buf, 1].clone() + self.cfg.y_nom
            self.prev_stone_pos[mask_reset_buf, 2] = self.current_stone_pos[mask_reset_buf, 2].clone()
            self.reset_stone_global(mask_reset_buf)
            self.update_ithstep_stones_info(self.ith_step, mask_reset_buf, self.current_stone_pos) #update next and next-next stone pos
            
            #only do reset impact for walking envs
            self.reset_impact(reset_as_walking)
            #initialize is_walking_env
            self.is_walking_env[reset_as_walking] = True
            self.is_walking_env[reset_as_standing] = False
            self.enter_standing_mode(reset_as_standing)

            

        #update phase var
        self.phase_var.update(tnow)
        # Freeze phase for standing environments
        standing_mask = ~self.is_walking_env
        if standing_mask.any():
            self.phase_var.tau[standing_mask] = 0.0
            self.phase_var.time_in_step[standing_mask] = 0.0
            self.phase_var.dtau[standing_mask] = 0.0


        return

    def reset_stance_swing_foot(self, mask): 
        """Reset the stance and swing foot positions and orientations.

        Args:
            mask (torch.Tensor): A boolean mask indicating which environments to reset.
            
        Updates:
            self.stance_idx: Switches the stance foot index for the masked environments.
            self.stance_foot_pos_0: Updates the stance foot position at the beginning of the step.
            self.stance_foot_ori_quat_0: Updates the stance foot orientation quaternion at the beginning of the step.
            self.stance_foot_ori_0: Updates the stance foot orientation (roll, pitch, yaw) at the beginning of the step.
            self.swing_foot_pos_0: Updates the swing foot position at the beginning of the step.    
        """
        #update stance swing foot info
        self.stance_idx[mask] = 1 - self.stance_idx[mask]  # switch stance foot, for short stones in the future, consider assign based on stone position
        foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_quat_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
        self.stance_foot_pos_0[mask, :] = foot_pos_w[mask, self.stance_idx[mask], :]
        self.stance_foot_ori_quat_0[mask, :] = foot_quat_w[mask, self.stance_idx[mask], :]
        self.stance_foot_ori_0[mask, :] = get_euler_from_quat(self.stance_foot_ori_quat_0[mask])  
        self.stance_foot_ori_quat_0_zerorollpitch[mask, :] = quat_from_euler_xyz(
            torch.zeros((torch.sum(mask),), device=self.device),
            torch.zeros((torch.sum(mask),), device=self.device),
            self.stance_foot_ori_0[mask, 2]
        ) 
        self.swing_foot_pos_0[mask, :] = foot_pos_w[mask, 1 - self.stance_idx[mask], :]
        self.stance_foot_pos_0[mask, :] = convert_foot_pos_to_midsole(foot_pos = self.stance_foot_pos_0[mask, :],
                                    mid_foot_offset = self.mid_foot_offset,
                                    foot_ori_quat = self.stance_foot_ori_quat_0[mask, :])
        self.swing_foot_pos_0[mask, :] = convert_foot_pos_to_midsole(foot_pos = self.swing_foot_pos_0[mask, :],
                                    mid_foot_offset = self.mid_foot_offset,
                                    foot_ori_quat = foot_quat_w[mask, 1 - self.stance_idx[mask], :])
        return 
    

    def enter_standing_mode(self, mask):
        """
        Initialize standing state for masked environments.
        """
        if not mask.any():
            return
        mask_ldesnext_nonzero = self.ldes_next.abs() > ZERO_EPS
        middle_term = torch.where(
            mask_ldesnext_nonzero,
            (self.hdes_next / self.ldes_next) * (1.0 - self.cfg.eps) * self.ldes,
            torch.zeros_like(self.ldes_next)
        )
        
        tmp = - middle_term + self.hdes #equivalent to eps=0 case
        self.zcomf_des[mask] = self.z0[mask] + tmp[mask] / 2.0
        
        # self.zcomf_des[mask] = torch.clamp(
        #     self.zcomf_des[mask],
        #     min=self.z0[mask] - 0.2,  # Don't go too low
        #     max=self.z0[mask] + 0.2   # Don't go too high
        # )
        return
    
    def reset_impact(self, mask):
        #to be called when impact is detected
        self.tSSplus[mask] = tensor_t_now(self._env)[mask]
        #recompute TSS
        if TEST_FLAT:

            self.TSS[mask] = self._env.cfg.commands.step_period.period_range[0]/2.0
        else: 
            target_yaw = torch.zeros((self.num_envs,), device=self.device)
            self.com_pos_target_yaw0, self.com_state2_target_yaw0 = self.compute_com_rel_to_stance_state_target_yaw_frame(self.use_momentum, target_yaw)
            x0 = self.get_sagittal_com_states_stacked(self.com_pos_target_yaw0, self.com_state2_target_yaw0, self.use_momentum)

            self.xcomf_des[mask] = self.cfg.eps * self.ldes[mask]
            #initial com z pos is same as old final com z pos
            self.zcom0_des[mask] = self.zcomf_des[mask] # self.com_pos_target_yaw0[mask, 2] - self.stance_foot_pos_0[mask, 2]
            #final desired com z pos based on desired 2-step preview
            mask_ldesnext_nonzero = self.ldes_next.abs() > ZERO_EPS
            middle_term = torch.where(
                mask_ldesnext_nonzero,
                (self.hdes_next / self.ldes_next) * (1.0 - self.cfg.eps) * self.ldes,
                torch.zeros_like(self.ldes_next)
            )
            
            tmp = self.cfg.eps * self.hdes - middle_term + self.hdes
            self.zcomf_des[mask] = self.z0[mask] + tmp[mask] / 2.0

            #todo, used z_tilde now, but may need to change to z0
            #todo, also may need to change to real com z pos instead of z0
            z_tilde_mask = compute_z_tilde_batched(self.com_pos_target_yaw0[mask,2], self.hdes[mask], self.ldes[mask], self.com_pos_target_yaw0[mask,0])
            TSS_raw_mask = solve_time2reach_pdes_batched(x0=x0[mask,:],
                                                           pdes=self.xcomf_des[mask],
                                                           z_tilde=z_tilde_mask,
                                                           use_momentum=self.use_momentum)

            #for sagittal place, nan occurs only if 1) when E0 < 0, clamp to min TSS 2) p0 > pdes, clamp to min TSS
            TSS_raw_mask_nan_removed = torch.nan_to_num(TSS_raw_mask, nan=self.cfg.TSS_min)
            self.TSS[mask] = torch.clamp(TSS_raw_mask_nan_removed, min=self.cfg.TSS_min, max=self.cfg.TSS_max)
            if torch.isnan(self.TSS).any() or torch.isinf(self.TSS).any():
                print("Warning: NaN detected in TSS computation!")
        self.hlip.update_hlip_partial_noDS(self.TSS[mask], mask)
        # self.hlip.update_hlip(self.z0, self.TSS, self.TDS)
        
        self.phase_var.reconfigure(self.tSSplus , self.tSSplus + self.TSS)

        return

    def compute_com_rel_to_stance_state_world_frame(self, use_momentum: bool):
        """Compute CoM position and either mass-normalized angular momentum or CoM velocity in world frame.
        Returns:
            com_pos_w: (num_envs, 3) CoM position in world frame
            h_st_w or com_vel_w: (num_envs, 3) mass-normalized angular momentum about stance foot in world frame or CoM velocity in world frame
            note that Lx of h_st_w is oppoisite sign as in HLIP convention
        """
        
        data = self.robot.data
        
        # compute real COM
        link_pos_w = data.body_com_pos_w # (num_envs, num_bodies, 3)
        link_mass = data.default_mass.to(self.device) # (num_envs, num_bodies)
        com_pos_w = (link_mass.unsqueeze(-1) * link_pos_w).sum(dim=1) / self.mass # (num_envs,  3)
        com_rel_st_w = com_pos_w - self.stance_foot_pos_0  # (num_envs, 3)
        
        lin_vel_w = data.body_com_lin_vel_w # (num_envs, num_bodies, 3)
        com_vel_w = (link_mass.unsqueeze(-1) * lin_vel_w).sum(dim=1) / self.mass # (num_envs,  3)
        if use_momentum:
            # compute mass-normalized angular momentum
            link_inertia_b = data.default_inertia.to(self.device).view(*data.default_inertia.shape[:-1], 3, 3) # (num_envs, num_bodies, 3, 3)
            link_vel_w = data.body_com_lin_vel_w # (num_envs, num_bodies, 3)
            link_omega_w = data.body_com_ang_vel_w # (num_envs, num_bodies, 3)
            link_quat_w = data.body_quat_w # (num_envs, num_bodies, 4)
            R_wb = matrix_from_quat(link_quat_w)  # (N,B,3,3)
            link_inertia_w = R_wb @ link_inertia_b @ R_wb.transpose(-1, -2) # (num_envs, num_bodies, 3, 3)
            term_rot = (link_inertia_w @ link_omega_w.unsqueeze(-1)).squeeze(-1) # (num_envs, num_bodies, 3)
            term_trans = torch.cross(link_pos_w - com_pos_w.unsqueeze(1), link_mass.unsqueeze(-1) * link_vel_w, dim=-1) # (num_envs, num_bodies, 3)
            h_com_w = (term_rot.sum(dim=1) + term_trans.sum(dim=1)) / self.mass # (num_envs, 3) world frame angular momentum about CoM, mass-normalized
            #h_st_foot = h_com + r_com2st_foot x v_com
            h_st_w = h_com_w + torch.cross(com_pos_w - self.stance_foot_pos_0, com_vel_w, dim=-1) # (num_envs, 3) world frame angular momentum about stance foot, mass-normalized
            return com_rel_st_w, h_st_w

        return com_rel_st_w, com_vel_w

    def compute_com_rel_to_stance_state_target_yaw_frame(self, use_momentum: bool, target_yaw: torch.Tensor):
        """Compute CoM position and either mass-normalized angular momentum or CoM velocity in target yaw frame."""
        com_rel_st_w, com_state2_w = self.compute_com_rel_to_stance_state_world_frame(use_momentum)
        quat_target_yaw = quat_from_euler_xyz(
            torch.zeros_like(target_yaw), torch.zeros_like(target_yaw), target_yaw
        )  # Shape: (N,4)
        com_pos_target_yaw = quat_apply(quat_inv(quat_target_yaw), com_rel_st_w)
        com_state2_target_yaw = quat_apply(quat_inv(quat_target_yaw), com_state2_w)
        return com_pos_target_yaw, com_state2_target_yaw

    def get_sagittal_com_states_stacked(self, com3d_state1: torch.Tensor, com3d_state2: torch.Tensor, use_momentum: bool):
        if use_momentum: 
            # return com_x and Ly
            return torch.stack([com3d_state1[:,0], com3d_state2[:,1]], dim=1)  # Shape: (num_envs, 2)
        else:
            # return com_x and com_vx
            return torch.stack([com3d_state1[:,0], com3d_state2[:,0]], dim=1)  # Shape: (num_envs, 2)
    
    def get_coronal_com_states_stacked(self, com3d_state1: torch.Tensor, com3d_state2: torch.Tensor, use_momentum: bool):
        if use_momentum: 
            # return com_y and LxLIP (LxLIP = -Lx)
            return torch.stack([com3d_state1[:,1], -com3d_state2[:,0]], dim=1)  # Shape: (num_envs, 2)
        else:
            # return com_y and com_vy
            return torch.stack([com3d_state1[:,1], com3d_state2[:,1]], dim=1)  # Shape: (num_envs, 2)
    
    def update_walking_target(self, mask):
        #given velocity command, update MLIP
        base_vdes = self._env.command_manager.get_command("base_velocity")  # (N,3)
        if TEST_FLAT:
            # 3D HLIP: uses vx, vy, y_nom
            self.hlip.update_desired_walking(base_vdes[:,0],base_vdes[:,1], self.cfg.y_nom)
            self.delta_yaw = base_vdes[:, 2] * self.phase_var.time_in_step
            self.yaw_dot = base_vdes[:, 2].clone()
            self.target_yaw = self.stance_foot_ori_0[:, 2] + self.delta_yaw
        else:
            # P2 HLIP: uses only lateral vy, y_nom
            delta_y = (self.prev_stone_pos[mask, 1] + self.current_stone_pos[mask, 1])/2.0 - self.terrain.env_origins[mask, 1]
            self.hlip.update_desired_walking_partial(base_vdes[mask, 1]-delta_y/self.TSS[mask], self.cfg.y_nom, mask)
            # delta_y = (self.prev_stone_pos[:, 1] + self.current_stone_pos[:, 1])/2.0 - self.terrain.env_origins[:,1]
            # self.hlip.update_desired_walking(base_vdes[:,1]-delta_y/self.TSS, self.cfg.y_nom)
            
            # for stepping stones, always targeting zero yaw
            self.yaw_dot = (self.target_yaw_stone - self.stance_foot_ori_0[:, 2]) / self.TSS
            self.yaw_dot = torch.clamp(self.yaw_dot, 
                                       min=-torch.ones_like(self.target_yaw_stone) * 0.5, 
                                       max=torch.ones_like(self.target_yaw_stone) * 0.5)

            self.delta_yaw = self.yaw_dot * self.phase_var.time_in_step
            self.target_yaw = self.stance_foot_ori_0[:, 2] + self.delta_yaw
        return
        

    def compute_actual(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame,
        i.e. local = self.stance_foot_ori_quat_0 frame."""

        # Convenience
        data = self.robot.data


        # Foot frame pos, quat, lin vel, ang vel (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_quat_w = data.body_quat_w[:, self.feet_bodies_idx, :]
        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        batch_idx = torch.arange(self.num_envs, device=self.device)
        
        
        
        #for holonomic constraint, current stance foot position and orientation
        # self.stance_foot_pos = foot_pos_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ori = get_euler_from_quat(foot_quat_w[batch_idx, self.stance_idx, :])
        #for holonomic_constraint_vel
        # self.stance_foot_vel = foot_lin_vel_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[batch_idx, self.stance_idx, :]
        


        self.stance_foot_pos, self.stance_foot_vel = convert_foot_pos_vel_to_midsole(foot_pos=foot_pos_w[batch_idx, self.stance_idx, :],
                                        foot_vel=foot_lin_vel_w[batch_idx, self.stance_idx, :],
                                        mid_foot_offset=self.mid_foot_offset,
                                        foot_ori_quat=foot_quat_w[batch_idx, self.stance_idx, :],
                                        foot_omega_w=foot_ang_vel_w[batch_idx, self.stance_idx, :])
        
        self.swing_foot_pos_actual, swing_foot_lin_vel = convert_foot_pos_vel_to_midsole(foot_pos=foot_pos_w[batch_idx, 1-self.stance_idx, :],
                                                                foot_vel=foot_lin_vel_w[batch_idx, 1-self.stance_idx, :],
                                                                mid_foot_offset=self.mid_foot_offset,
                                                                foot_ori_quat=foot_quat_w[batch_idx, 1-self.stance_idx, :],
                                                                foot_omega_w=foot_ang_vel_w[batch_idx, 1-self.stance_idx, :])
        
        # Convert foot positions to the robot's yaw-aligned local frame
        swing2stance0_local = _transfer_to_local_frame(
            self.swing_foot_pos_actual - self.stance_foot_pos_0, self.stance_foot_ori_quat_0_zerorollpitch
        )
        swing_foot_lin_vel_local = _transfer_to_local_frame(
            swing_foot_lin_vel, self.stance_foot_ori_quat_0_zerorollpitch
        )
        


        # Center of mass to stance foot vector in local frame
        #TODO: used generic root com pos for now
        com_w = data.root_com_pos_w
        com2stance_local = _transfer_to_local_frame(com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0_zerorollpitch)
        # Velocities (world frame)
        com_vel_w = data.root_com_vel_w[:, 0:3]
        # Convert velocities to local frame
        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0_zerorollpitch)
        
        
        
        # Pelvis orientation (Euler XYZ)
        pelvis_eulxyz = get_euler_from_quat(data.root_quat_w)
        pelvis_omega_body = data.root_ang_vel_b
        
        # Swing Foot orientations (Euler XYZ)
        swing_foot_eulxyz = get_euler_from_quat(foot_quat_w[batch_idx, 1-self.stance_idx, :])
        swing_foot_omega_body = quat_apply(
            quat_inv(foot_quat_w[batch_idx, 1-self.stance_idx, :]), foot_ang_vel_w[batch_idx, 1-self.stance_idx, :]
        )

        # Upper body joint positions and velocities
        upper_body_joint_pos = self.robot.data.joint_pos[:, self.upper_body_joint_idx]
        upper_body_joint_vel = self.robot.data.joint_vel[:, self.upper_body_joint_idx]

        
        self.y_act = torch.cat(
            [com2stance_local, pelvis_eulxyz, swing2stance0_local, swing_foot_eulxyz,  upper_body_joint_pos], dim=-1
        )
  
        self.dy_act = torch.cat(
            [com_vel_local, pelvis_omega_body, swing_foot_lin_vel_local, swing_foot_omega_body, upper_body_joint_vel],
            dim=-1,
        )
        return
    
    
    def compute_dzcomf_des(self, mask: torch.Tensor):
        if not mask.any():
            return
        N_masked = mask.sum().item()
        #predict preimpact sagittal com x and vx/Ly
        com_now_full, com_d_now_full = self.compute_com_rel_to_stance_state_target_yaw_frame(
            self.use_momentum, 
            torch.zeros((self.num_envs,), device=self.device)
        ) #Shape: (N,3)
        time2impact = self.TSS - self.phase_var.time_in_step #Shape: (N,)
        
        xnow_sagittal = self.get_sagittal_com_states_stacked(com_now_full, com_d_now_full, self.use_momentum) #Shape: (N,2)

        x_sagittal_minus, state2_sagittal_minus = self.hlip.get_com_state_from_x0_sagittal(xnow_sagittal, time2impact)#Shape: (N,)
        
        x_sagittal_minus_masked = x_sagittal_minus[mask] #Shape: (N_masked,)
        state2_sagittal_minus_masked = state2_sagittal_minus[mask] #Shape: (N_masked,)
        
        #compute post-impact x com pos: p_plus
        x_plus = x_sagittal_minus_masked - self.ldes[mask] #Shape: (N_masked,2)

        # if torch.any(self.ldes[mask] <= 0):
        #     print("Warning: ldes is negative in dzcomf_des computation!")
        
        # Safe division for ldes_next term
        term_ldes_next = torch.where(
            self.ldes_next[mask] > ZERO_EPS,
            x_plus * self.hdes_next[mask] / self.ldes_next[mask],
            torch.zeros_like(x_plus)
        )
        #interpolate z0 between real and desired
        z0LIPplus_real = com_now_full[mask,2] - self.hdes[mask] - term_ldes_next
        z0LIPplus_des = self.zcomf_des[mask] - self.hdes[mask] - term_ldes_next
        z0_interp = self.phase_var.tau[mask] * z0LIPplus_real + (1.0 - self.phase_var.tau[mask]) * z0LIPplus_des #Shape: (N_masked,)

        E_envs = torch.ones((self.num_envs,), device=self.device) * self.cfg.E_star
        E_envs[self.is_standing_cmd] = 0.0  #if commanded standing, set E = 0
        E_envs_masked = E_envs[mask]  #Shape: (N_masked,)

        #solve for Lyplus which corresponds to desired oribital energy Estar
        state2_sagittal_plus = solve_velocity_or_momentum_positive_from_E_batched(E=E_envs_masked,
                                                                    p=x_plus,
                                                                    z_tilde=z0_interp,
                                                                    use_momentum=self.use_momentum) #Shape: (N_masked,)
        vx_com_now = self.robot.data.root_com_vel_w[mask, 0]  #Shape: (N_masked,)
        if self.use_momentum:
            self.dzcomf_des[mask] = (state2_sagittal_plus - state2_sagittal_minus_masked + self.hdes[mask] * vx_com_now) / self.ldes[mask]
        else:
            self.dzcomf_des[mask] = ((state2_sagittal_plus - state2_sagittal_minus_masked)*z0_interp + self.hdes[mask] * vx_com_now) / self.ldes[mask]

        if torch.any(self.dzcomf_des.isnan()):
            print("Warning: dzcomf_des is NaN in dzcomf_des computation!")
            #nan to zero
            self.dzcomf_des[self.dzcomf_des.isnan()] = 0.0

        self.dzcomf_des[self.ldes < 0] = 0.0
        self.dzcomf_des = torch.clamp(self.dzcomf_des,
                                       min=-torch.ones_like(self.hdes) * 1.0,
                                       max=torch.ones_like(self.hdes) * 1.0)
        return

    def compute_desired_stones_nomask(self):

        N = self.num_envs
        #half delta yaw for pelvis and full on swing foot
        pelvis_eulxyz = torch.zeros((N, 3), device=self.device)
        pelvis_eulxyz[:, 2] = self.stance_foot_ori_0[:, 2] + self.delta_yaw * 0.5
        pelvis_eulxyz_dot = torch.zeros((N, 3), device=self.device)
        pelvis_eulxyz_dot[:, 2] = self.yaw_dot * 0.5
        swingfoot_eulxyz = torch.zeros((N, 3), device=self.device)
        swingfoot_eulxyz[:, 2] = self.stance_foot_ori_0[:, 2] + self.delta_yaw
        swingfoot_eulxyz_dot = torch.zeros((N, 3), device=self.device)
        swingfoot_eulxyz_dot[:, 2] = self.yaw_dot

        upper_body_joint_pos, upper_body_joint_vel = self.generate_upper_body_ref() #Shape: (N, num_upper_joints)
        
        zeros_N = torch.zeros((N,), device=self.device)
        quat_target_frame = quat_from_euler_xyz(zeros_N, zeros_N, self.target_yaw)  # Shape: (N,4)
        quat_targetyaw_to_stanceyaw = quat_from_euler_xyz(zeros_N, zeros_N, self.target_yaw-self.stance_foot_ori_0[:, 2])  # Shape: (N,4)
        
        
        # Swingfoot rel to stancefoot x based on stepping stone position, in world frame
        Ux = self.ldes
        # Ux = torch.clamp(Ux, min=-1.0, max=1.0)
        # Get desired foot placements from HLIP in target yaw frame, no feedback used here
        #TODO: later consider add feedback
        Uy = self.hlip.get_desired_foot_placement(self.stance_idx) #Shape: (N,)
        #clamp step y to avoid too large step and foot crossing
        left_stance = (self.stance_idx == 0)
        right_stance = (self.stance_idx == 1)
        Uy[left_stance] = torch.clamp(Uy[left_stance], min=-self.cfg.y_sw_max, max=-self.cfg.y_sw_min)
        Uy[right_stance] = torch.clamp(Uy[right_stance], min=self.cfg.y_sw_min, max=self.cfg.y_sw_max)
        
        
        footplacement_target_frame = torch.stack([Ux, Uy, self.hdes], dim=-1) # Shape: (N,3)
        # Create horizontal control points with batch dimension
        horizontal_control_points = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], device=self.device)
        bht = bezier_deg(0, self.phase_var.tau, self.TSS, horizontal_control_points, 4) #Shape: (N,)
        dbht = bezier_deg(1, self.phase_var.tau, self.TSS, horizontal_control_points, 4) #Shape: (N,)
        # Horizontal X and Y (linear interpolation)
        #TODO: use stancepos0 or not
        x_init_target_frame = self.prev_stone_pos[:, 0] - self.current_stone_pos[:, 0]
        p_swing_x = ((1 - bht) * x_init_target_frame + bht * Ux) #Shape: (N,)
        v_swing_x = ((-dbht) * x_init_target_frame + dbht * Ux)  #Shape: (N,)
        
        #clamp swing foot y to avoid too large step
        y0 = self.prev_stone_pos[:, 1] - self.current_stone_pos[:, 1] #Shape: (N,)
        p_swing_y = ((1 - bht) * (y0) + bht * Uy) #Shape: (N,)
        v_swing_y = ((-dbht) * y0 + dbht * Uy)  #Shape: (N,)
        
        
        # #TODO: check this added swing pitch angle and velocity
        # swingfoot_eulxyz[:, 1] = (1 - bht) * self.stance_foot_ori_0[:,1] + bht * self.pitchdes
        # swingfoot_eulxyz_dot[:, 1] = dbht * (self.pitchdes - self.stance_foot_ori_0[:,1])

        # swing foot z
        z_sw_max = torch.full((N,), self.cfg.z_sw_max, device=self.device)

        # z_init = self.swing_foot_pos_0[:, 2] - self.stance_foot_pos_0[:, 2]  # Initial swing foot height relative to stance foot
        z_init = self.prev_stone_pos[:, 2] - self.current_stone_pos[:, 2]
        z_sw_neg = self.hdes + self.cfg.z_sw_min #adjusted based on stepping stone height
        z_candidates = torch.stack([
            self.prev_stone_pos[:, 2],
            self.current_stone_pos[:, 2],
            self.next_stone_pos[:, 2]
        ], dim=1)
        z_max_stones = z_candidates.max(dim=1).values  # shape (num_envs,)
        z_sw_max = z_max_stones - self.current_stone_pos[:, 2] + self.cfg.z_sw_max 

        control_v = torch.stack(
        [   z_init,  # Start
            z_init + 0.2 * (z_sw_max - z_init),
            z_init + 0.6 * (z_sw_max - z_init),
            z_sw_max,  # Peak at mid-swing
            z_sw_neg + 0.5 * (z_sw_max - z_sw_neg),
            z_sw_neg + 0.05 * (z_sw_max - z_sw_neg),
            z_sw_neg,  # End
        ], dim=1)  # Shape: (N,7)
        degree_v = control_v.shape[1] - 1
        if isinstance(self.phase_var.tau, float):
            phase_tensor = torch.full((N,), self.phase_var.tau, device=self.device)
            T_tensor = torch.full((N,), self.TSS, device=self.device)
        else:
            phase_tensor = self.phase_var.tau
            T_tensor = self.TSS
        p_swing_z = bezier_deg(0, phase_tensor, T_tensor, control_v, degree_v)
        v_swing_z = bezier_deg(1, phase_tensor, T_tensor, control_v, degree_v)

        # Combine to get full swing foot position and velocity
        swing_foot_pos_local = quat_apply(quat_targetyaw_to_stanceyaw, torch.stack([p_swing_x, p_swing_y, p_swing_z], dim=-1))  # Shape: (N,3)
        swing_foot_vel_local = quat_apply(quat_targetyaw_to_stanceyaw, torch.stack([v_swing_x, v_swing_y, v_swing_z], dim=-1))  # Shape: (N,3)
            
        
        
        
        
        
        
        
        
        # com y based on HLIP
        #TODO: how should I guide comy 
        #TODO: lower comy tracking weight?
        com_y_target_frame, com_dy_target_frame = self.hlip.get_desired_com_state(self.stance_idx, self.phase_var.time_in_step) #Shape: (N,)
        if self.use_momentum: #convert angular momentum to velocity
            com_dy_target_frame = com_dy_target_frame / self.z0 #vy^d = LxLIP^d / z0, as LxLIP = -Lx
            
        #TODO: did not use z_tilde here
        # com x based on target orbital energy and comf position at the end of SS
        v_or_L_target_frame = solve_velocity_or_momentum_positive_from_E_batched(E=self.cfg.E_star, 
                                                                    p=self.xcomf_des, #target frame
                                                                    z_tilde=self.z0, #z_tilde=compute_z_tilde_batched(self.z0, self.hdes, self.ldes, self.xcomf_des), 
                                                                    use_momentum=self.use_momentum)
        # xT = torch.stack([self.xcomf_des, v_or_L], dim=1)  # Shape: (N,2)
        # com_x, com_dx = self.hlip.get_desired_com_state_from_end_of_SS_sagittal(xT, self.TSS - self.phase_var.time_in_step) #Shape: (N,)
        if self.use_momentum:
            vT = v_or_L_target_frame / self.z0 #Ly/z0
            v0 = self.com_state2_target_yaw0[:,1] / self.z0 #Ly/z0
        else:
            vT = v_or_L_target_frame
            v0 = self.com_state2_target_yaw0[:,0] #vx
            
        cubic_comx_coeff = cubic_spline_coeff_batched(y0=self.com_pos_target_yaw0[:,0],
                                                dy0=v0,
                                                y1=self.xcomf_des,
                                                dy1=vT,
                                                dtaudt=self.phase_var.dtau) #Shape: (N,4)
        
        com_x_target_frame, com_dx_target_frame = cubic_spline_eval(cubic_comx_coeff, self.phase_var.tau, self.phase_var.dtau)    
            
        #com z 
        self.compute_dzcomf_des()
        #TODO: MUST check if it is better to use true zcom0
        cubic_comz_coeff = cubic_spline_coeff_batched(self.zcom0_des , 
                                                torch.zeros_like(self.zcom0_des),
                                                self.zcomf_des ,
                                                self.dzcomf_des,
                                                self.phase_var.dtau) #Shape: (N,4)
        
        com_z_w, com_dz_w = cubic_spline_eval(cubic_comz_coeff, self.phase_var.tau, self.phase_var.dtau)
        # com_z_w = self.z0
        # com_dz_w = torch.zeros_like(com_z_w)
        # Concatenate x y z components
        com_pos_des = torch.stack(
            [com_x_target_frame, com_y_target_frame, com_z_w], dim=-1
        )  # Shape: (N,3)
        com_vel_des = torch.stack([com_dx_target_frame, com_dy_target_frame, com_dz_w], dim=-1)  # Shape: (N,3)

        com_pos_des_local = quat_apply(quat_targetyaw_to_stanceyaw, com_pos_des)  # [N,3]
        com_vel_des_local = quat_apply(quat_targetyaw_to_stanceyaw, com_vel_des)  # [N,3]



        #convert euler rates to omega body
        omega_pelvis_ref = euler_rates_to_omega_b(pelvis_eulxyz, pelvis_eulxyz_dot)
        omega_foot_ref = euler_rates_to_omega_b(swingfoot_eulxyz, swingfoot_eulxyz_dot)  # (N,3)
        
        self.y_out = torch.cat(
            [com_pos_des_local, pelvis_eulxyz, swing_foot_pos_local, swingfoot_eulxyz, upper_body_joint_pos], dim=-1
        )

        self.dy_out = torch.cat(
            [com_vel_des_local, omega_pelvis_ref, swing_foot_vel_local, omega_foot_ref, upper_body_joint_vel], dim=-1
        )
        
        if self.debug_vis:
            self.foottarget_vis_quat = quat_target_frame #Shape: (N,4)
            self.foottarget_vis_pos = quat_apply(quat_target_frame, footplacement_target_frame) + self.stance_foot_pos_0  #Shape: (N,3)
            self.swingfoot_vis_quat = quat_target_frame #Shape: (N,4)
            self.swingfoot_vis_pos = quat_apply(self.stance_foot_ori_quat_0_zerorollpitch, swing_foot_pos_local) + self.stance_foot_pos_0
            self.com_frame_vis_quat = quat_target_frame
            self.com_frame_vis_pos = quat_apply(quat_target_frame, com_pos_des) + self.stance_foot_pos_0  # Shape: (N,3)
        return    
    
    def compute_desired_stones(self, walking_mask: torch.Tensor):
        if not walking_mask.any():
            return
        N_walking = torch.sum(walking_mask).item()
        
        N = self.num_envs
        #half delta yaw for pelvis and full on swing foot
        pelvis_eulxyz = torch.zeros((N_walking, 3), device=self.device)
        pelvis_eulxyz[:, 2] = self.stance_foot_ori_0[walking_mask, 2] + self.delta_yaw[walking_mask] * 0.5
        pelvis_eulxyz_dot = torch.zeros((N_walking, 3), device=self.device)
        pelvis_eulxyz_dot[:, 2] = self.yaw_dot[walking_mask] * 0.5
        swingfoot_eulxyz = torch.zeros((N_walking, 3), device=self.device)
        swingfoot_eulxyz[:, 2] = self.stance_foot_ori_0[walking_mask, 2] + self.delta_yaw[walking_mask]
        swingfoot_eulxyz_dot = torch.zeros((N_walking, 3), device=self.device)
        swingfoot_eulxyz_dot[:, 2] = self.yaw_dot[walking_mask]

        # Generate upper body ref for ALL envs, then index (like standing)
        full_upper_pos, full_upper_vel = self.generate_upper_body_ref()
        upper_body_joint_pos = full_upper_pos[walking_mask]
        upper_body_joint_vel = full_upper_vel[walking_mask]

        zeros_N = torch.zeros((N_walking,), device=self.device)
        quat_target_frame = quat_from_euler_xyz(zeros_N, zeros_N, self.target_yaw[walking_mask])  # Shape: (N,4)
        quat_targetyaw_to_stanceyaw = quat_from_euler_xyz(zeros_N, zeros_N, self.target_yaw[walking_mask]-self.stance_foot_ori_0[walking_mask, 2])  # Shape: (N,4)

        
        # Swingfoot rel to stancefoot x based on stepping stone position, in world frame
        Ux = self.ldes[walking_mask]
        # Ux = torch.clamp(Ux, min=-1.0, max=1.0)
        # Get desired foot placements from HLIP in target yaw frame, no feedback used here
        #TODO: later consider add feedback
        Uy_full = self.hlip.get_desired_foot_placement(self.stance_idx)
        Uy = Uy_full[walking_mask]
        # Index stance_idx for the walking envs
        stance_idx_walking = self.stance_idx[walking_mask]
        left_stance = (stance_idx_walking == 0)
        right_stance = (stance_idx_walking == 1)
        #clamp step y to avoid too large step and foot crossing
        Uy[left_stance] = torch.clamp(Uy[left_stance], min=-self.cfg.y_sw_max, max=-self.cfg.y_sw_min)
        Uy[right_stance] = torch.clamp(Uy[right_stance], min=self.cfg.y_sw_min, max=self.cfg.y_sw_max)


        footplacement_target_frame = torch.stack([Ux, Uy, self.hdes[walking_mask]], dim=-1) # Shape: (N_walking,3)
        # Create horizontal control points with batch dimension
        horizontal_control_points = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], device=self.device)
        bht = bezier_deg(0, self.phase_var.tau[walking_mask], self.TSS[walking_mask], horizontal_control_points, 4) #Shape: (N_walking,)
        dbht = bezier_deg(1, self.phase_var.tau[walking_mask], self.TSS[walking_mask], horizontal_control_points, 4) #Shape: (N_walking,)
        # Horizontal X and Y (linear interpolation)
        #TODO: use stancepos0 or not
        x_init_target_frame = self.prev_stone_pos[walking_mask, 0] - self.current_stone_pos[walking_mask, 0]
        p_swing_x = ((1 - bht) * x_init_target_frame + bht * Ux) #Shape: (N_walking,)
        v_swing_x = ((-dbht) * x_init_target_frame + dbht * Ux)  #Shape: (N_walking,)

        #clamp swing foot y to avoid too large step
        y0 = self.prev_stone_pos[walking_mask, 1] - self.current_stone_pos[walking_mask, 1] #Shape: (N_walking,)
        p_swing_y = ((1 - bht) * (y0) + bht * Uy) #Shape: (N_walking,)
        v_swing_y = ((-dbht) * y0 + dbht * Uy)  #Shape: (N_walking,)


        #  check this added swing pitch angle and velocity
        # swingfoot_eulxyz[:, 1] = (1 - bht) * self.stance_foot_ori_0[:,1] + bht * self.pitchdes
        # swingfoot_eulxyz_dot[:, 1] = dbht * (self.pitchdes - self.stance_foot_ori_0[:,1])

        # swing foot z
        z_sw_max = torch.full((N_walking,), self.cfg.z_sw_max, device=self.device)

        # z_init = self.swing_foot_pos_0[:, 2] - self.stance_foot_pos_0[:, 2]  # Initial swing foot height relative to stance foot
        z_init = self.prev_stone_pos[walking_mask, 2] - self.current_stone_pos[walking_mask, 2]
        z_sw_neg = self.hdes[walking_mask] + self.cfg.z_sw_min #adjusted based on stepping stone height
        z_candidates = torch.stack([
            self.prev_stone_pos[walking_mask, 2],
            self.current_stone_pos[walking_mask, 2],
            self.next_stone_pos[walking_mask, 2]
        ], dim=1)
        z_max_stones = z_candidates.max(dim=1).values  # shape (N_walking,)
        z_sw_max = z_max_stones - self.current_stone_pos[walking_mask, 2] + self.cfg.z_sw_max # shape (N_walking,)

        control_v = torch.stack(
        [   z_init,  # Start
            z_init + 0.2 * (z_sw_max - z_init),
            z_init + 0.6 * (z_sw_max - z_init),
            z_sw_max,  # Peak at mid-swing
            z_sw_neg + 0.5 * (z_sw_max - z_sw_neg),
            z_sw_neg + 0.05 * (z_sw_max - z_sw_neg),
            z_sw_neg,  # End
        ], dim=1)  # Shape: (N_walking,7)
        degree_v = control_v.shape[1] - 1
        if isinstance(self.phase_var.tau, float):
            phase_tensor = torch.full((N_walking,), self.phase_var.tau, device=self.device)
            T_tensor = torch.full((N_walking,), self.TSS, device=self.device)
        else:
            phase_tensor = self.phase_var.tau[walking_mask]
            T_tensor = self.TSS[walking_mask]
        p_swing_z = bezier_deg(0, phase_tensor, T_tensor, control_v, degree_v)
        v_swing_z = bezier_deg(1, phase_tensor, T_tensor, control_v, degree_v)

        # Combine to get full swing foot position and velocity
        swing_foot_pos_local = quat_apply(quat_targetyaw_to_stanceyaw, torch.stack([p_swing_x, p_swing_y, p_swing_z], dim=-1))  # Shape: (N_walking,3)
        swing_foot_vel_local = quat_apply(quat_targetyaw_to_stanceyaw, torch.stack([v_swing_x, v_swing_y, v_swing_z], dim=-1))  # Shape: (N_walking,3)

        
        # com y based on HLIP
        #TODO: how should I guide comy 
        #TODO: lower comy tracking weight?
        com_y_target_frame, com_dy_target_frame = self.hlip.get_desired_com_state_partial(
            self.stance_idx[walking_mask], self.phase_var.time_in_step[walking_mask], walking_mask
        )
        if self.use_momentum: #convert angular momentum to velocity
            com_dy_target_frame = com_dy_target_frame / self.z0[walking_mask] #vy^d = LxLIP^d / z0, as LxLIP = -Lx

        #TODO: did not use z_tilde here
        # com x based on target orbital energy and comf position at the end of SS
        v_or_L_target_frame = solve_velocity_or_momentum_positive_from_E_batched(E=self.cfg.E_star, 
                                                                    p=self.xcomf_des[walking_mask], #target frame
                                                                    z_tilde=self.z0[walking_mask], #z_tilde=compute_z_tilde_batched(self.z0, self.hdes, self.ldes, self.xcomf_des), 
                                                                    use_momentum=self.use_momentum)
        if self.use_momentum:
            vT = v_or_L_target_frame / self.z0[walking_mask] #Ly/z0
            v0 = self.com_state2_target_yaw0[walking_mask,1] / self.z0[walking_mask] #Ly/z0
        else:
            vT = v_or_L_target_frame[walking_mask]
            v0 = self.com_state2_target_yaw0[walking_mask,0] #vx

        cubic_comx_coeff = cubic_spline_coeff_batched(y0=self.com_pos_target_yaw0[walking_mask,0],
                                                dy0=v0,
                                                y1=self.xcomf_des[walking_mask],
                                                dy1=vT,
                                                dtaudt=self.phase_var.dtau[walking_mask]) #Shape: (N,4)

        com_x_target_frame, com_dx_target_frame = cubic_spline_eval(cubic_comx_coeff, self.phase_var.tau[walking_mask], self.phase_var.dtau[walking_mask])

        #com z
        self.compute_dzcomf_des(walking_mask) #TODO
        #TODO: MUST check if it is better to use true zcom0
        cubic_comz_coeff = cubic_spline_coeff_batched(self.zcom0_des[walking_mask] , 
                                                torch.zeros_like(self.zcom0_des[walking_mask]),
                                                self.zcomf_des[walking_mask] ,
                                                self.dzcomf_des[walking_mask],
                                                self.phase_var.dtau[walking_mask]) #Shape: (N,4)

        com_z_w, com_dz_w = cubic_spline_eval(cubic_comz_coeff, self.phase_var.tau[walking_mask], self.phase_var.dtau[walking_mask])
        # com_z_w = self.z0
        # com_dz_w = torch.zeros_like(com_z_w)
        # Concatenate x y z components
        com_pos_des = torch.stack(
            [com_x_target_frame, com_y_target_frame, com_z_w], dim=-1
        )  # Shape: (N,3)
        com_vel_des = torch.stack([com_dx_target_frame, com_dy_target_frame, com_dz_w], dim=-1)  # Shape: (N,3)

        com_pos_des_local = quat_apply(quat_targetyaw_to_stanceyaw, com_pos_des)  # [N,3]
        com_vel_des_local = quat_apply(quat_targetyaw_to_stanceyaw, com_vel_des)  # [N,3]



        #convert euler rates to omega body
        omega_pelvis_ref = euler_rates_to_omega_b(pelvis_eulxyz, pelvis_eulxyz_dot)
        omega_foot_ref = euler_rates_to_omega_b(swingfoot_eulxyz, swingfoot_eulxyz_dot)  # (N,3)
        
        self.y_out[walking_mask] = torch.cat(
            [com_pos_des_local, pelvis_eulxyz, swing_foot_pos_local, swingfoot_eulxyz, upper_body_joint_pos], dim=-1
        )

        self.dy_out[walking_mask] = torch.cat(
            [com_vel_des_local, omega_pelvis_ref, swing_foot_vel_local, omega_foot_ref, upper_body_joint_vel], dim=-1
        )
        
        if self.debug_vis:
            self.foottarget_vis_quat[walking_mask] = quat_target_frame #Shape: (N_walking,4)
            self.foottarget_vis_pos[walking_mask] = footplacement_target_frame + self.stance_foot_pos_0[walking_mask]  #Shape: (N_walking,3)
            self.swingfoot_vis_quat[walking_mask] = quat_target_frame #Shape: (N_walking,4)
            self.swingfoot_vis_pos[walking_mask] = quat_apply(self.stance_foot_ori_quat_0_zerorollpitch[walking_mask, :], swing_foot_pos_local) + self.stance_foot_pos_0[walking_mask]
            self.com_frame_vis_quat[walking_mask] = quat_target_frame
            self.com_frame_vis_pos[walking_mask] = com_pos_des + self.stance_foot_pos_0[walking_mask]  # Shape: (N,3)
        return

    def compute_desired_standing(self, standing_mask):
        if not standing_mask.any():
            return

        N_standing = standing_mask.sum().item()

        pelvis_eulxyz = torch.zeros((N_standing, 3), device=self.device) #Shape: (N,3)
        pelvis_eulxyz[:, 2] =  self.stance_foot_ori_0[standing_mask, 2]
        #TODO: pitch can be non-zero for standing on stepping stone, may update eul y later
        swingfoot_eulxyz = pelvis_eulxyz #Shape: (N,3)
        # swingfoot_eulxyz[:, 1] = self.stance_foot_ori_0[standing_mask, 1]

        #no need to convert, just standing
        omega_pelvis_ref = torch.zeros((N_standing, 3), device=self.device)#Shape: (N,3)
        omega_foot_ref = torch.zeros((N_standing, 3), device=self.device)#Shape: (N,3)
        
        # Get upper body refs for ALL envs, then index
        full_upper_pos, full_upper_vel = self.generate_upper_body_ref_static()
        upper_body_joint_pos = full_upper_pos[standing_mask]
        upper_body_joint_vel = full_upper_vel[standing_mask]
        
        
        zeros_N = torch.zeros((N_standing,), device=self.device)
        quat_world_to_stanceyaw = quat_from_euler_xyz(zeros_N, zeros_N,
                                                       - self.stance_foot_ori_0[standing_mask, 2])  # Shape: (N,4)

        # swing foot stays at swing foot pos0 in world, which is also target yaw frame
        p_swing_init = self.swing_foot_pos_0[standing_mask] - self.stance_foot_pos_0[standing_mask]  #Shape: (N,)

        # Combine to get full swing foot position and velocity
        swing_foot_pos_local = quat_apply(quat_world_to_stanceyaw, p_swing_init)  # Shape: (N,3)
        swing_foot_vel_local = torch.zeros((N_standing,3), device=self.device)  # Shape: (N,3)

        #TODO: for now, used stance foot pos for x, so relative to stance foot, com x=0
        com_y = p_swing_init[:, 1] / 2.0
        #TODO: used 0 for com x position, option p_swing_init[:, 0] / 2.0 
        com_pos_des = torch.stack(
            [zeros_N, com_y, self.zcomf_des[standing_mask]], dim=-1
        )
        com_pos_des_local = quat_apply(quat_world_to_stanceyaw, com_pos_des)  # [N,3]
        com_vel_des_local = torch.zeros((N_standing,3), device=self.device)  # Shape: (N,3)

        self.y_out[standing_mask] = torch.cat(
            [com_pos_des_local, pelvis_eulxyz, swing_foot_pos_local, swingfoot_eulxyz, upper_body_joint_pos], dim=-1
        )

        self.dy_out[standing_mask] = torch.cat(
            [com_vel_des_local, omega_pelvis_ref, swing_foot_vel_local, omega_foot_ref, upper_body_joint_vel], dim=-1
        )
        if self.debug_vis:
            quat_target_frame = quat_from_euler_xyz(zeros_N, zeros_N, self.target_yaw[standing_mask])  # Shape: (N,4)
            self.foottarget_vis_quat[standing_mask] = quat_target_frame #Shape: (N,4)
            self.foottarget_vis_pos[standing_mask] = self.swing_foot_pos_0[standing_mask]  #Shape: (N,3)
            self.swingfoot_vis_quat[standing_mask] = quat_target_frame #Shape: (N,4)
            self.swingfoot_vis_pos[standing_mask] =  self.swing_foot_pos_0[standing_mask]
            self.com_frame_vis_quat[standing_mask] = quat_target_frame
            self.com_frame_vis_pos[standing_mask] = com_pos_des + self.stance_foot_pos_0[standing_mask]  # Shape: (N,3)
        
        return  
        
    def generate_upper_body_ref(self):
        
        forward_vel = self._env.command_manager.get_command("base_velocity")[:, 0] # (num_envs,)



        phase = torch.pi * (self.phase_var.tau + self.stance_idx) # (num_envs,)


        # unpack your cfg scalars
        sh_pitch0, sh_roll0, sh_yaw0 = self.cfg.shoulder_ref
        elb0 = self.cfg.elbow_ref
        waist_yaw0 = self.cfg.waist_yaw_ref

        # build every amp as a (num_envs,) tensor
        sh_pitch_amp = sh_pitch0 * forward_vel  # (num_envs,)
        sh_roll_amp = sh_roll0 * torch.ones_like(forward_vel) # (num_envs,)
        sh_yaw_amp = sh_yaw0 * torch.ones_like(forward_vel) # (num_envs,)
        elb_amp = elb0 * forward_vel  # (num_envs,)
        waist_amp = waist_yaw0 * torch.ones_like(forward_vel) # (num_envs,)

        # stack into (num_envs,num_upper_joints)
        amp = torch.stack(
            [
                waist_amp,
                sh_pitch_amp,
                sh_pitch_amp,
                sh_roll_amp,
                sh_roll_amp,
                sh_yaw_amp,
                sh_yaw_amp,
                elb_amp,
                elb_amp,
            ],
            dim=1,
        ).to(self.device)

        # your sign & offset stay (num_upper_joints, ) each
        sign = torch.tensor(
            [
                1,  # waist_yaw
                1,
                -1,  # L/R shoulder_pitch
                1,
                -1,  # L/R shoulder_roll
                1,
                -1,  # L/R shoulder_yaw
                1,
                -1,  # L/R elbow
            ],
            device=self.device,
        )

        offset = torch.tensor(
            [
                torch.pi,  # waist_yaw
                torch.pi / 2,  # L_sh_pitch
                torch.pi / 2,  # R_sh_pitch
                torch.pi / 2,  # L_sh_roll
                torch.pi / 2,  # R_sh_roll
                0,  # L_sh_yaw
                0,  # R_sh_yaw
                torch.pi / 2,  # L_elbow
                torch.pi / 2,  # R_elbow
            ],
            device=self.device,
        )

        # joint offsets: (num_envs, num_upper_joints)
        joint_offset = self.robot.data.default_joint_pos[:, self.upper_body_joint_idx]

        # refs: everything now broadcast to (num_envs, num_upper_joints)
        offset = offset.unsqueeze(0).expand(self.num_envs, -1)
        sign = sign.unsqueeze(0).expand(self.num_envs, -1)
        phase = phase.unsqueeze(1).expand(-1, len(self.upper_body_joint_idx))
        ref = amp * sign * torch.sin(phase + offset) + joint_offset

        # velocity
        dphase_dt = self.phase_var.dtau.unsqueeze(1).expand(-1, len(self.upper_body_joint_idx))  # (num_envs, num_upper_joints)
        ref_dot = amp * sign * torch.cos(phase + offset) * dphase_dt

        return ref, ref_dot

    def generate_upper_body_ref_static(self):
        # return static upper body reference positions and zero velocities
        joint_offset = self.robot.data.default_joint_pos[:, self.upper_body_joint_idx]
        ref = joint_offset
        ref_dot = torch.zeros_like(ref)
        return ref, ref_dot

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            #foot target and swing foot intermediate visualizer
            self.foottarget_visualizer = VisualizationMarkers(self.cfg.foottarget_cfg)
            self.foottarget_visualizer.set_visibility(True)
            self.swingfoot_visualizer = VisualizationMarkers(self.cfg.swingfoot_cfg)
            self.swingfoot_visualizer.set_visibility(True)
            
            #stone related visualizer
            self.currentstone_visualizer = VisualizationMarkers(self.cfg.currentstone_cfg)
            self.currentstone_visualizer.set_visibility(True)
            self.nextstone_visualizer = VisualizationMarkers(self.cfg.nextstone_cfg)
            self.nextstone_visualizer.set_visibility(True)
            self.nextnextstone_visualizer = VisualizationMarkers(self.cfg.nextnextstone_cfg)    
            self.nextnextstone_visualizer.set_visibility(True)
            self.terrain_origin_visualizer = VisualizationMarkers(self.cfg.originframe_cfg)
            self.terrain_origin_visualizer.set_visibility(True)
            
            #com frame visualizer
            self.comref_visualizer = VisualizationMarkers(self.cfg.comrefframe_cfg)
            self.comref_visualizer.set_visibility(True)
        else:
            if hasattr(self, "foottarget_visualizer"):
                self.foottarget_visualizer.set_visibility(False)
            if hasattr(self, "swingfoot_visualizer"):
                self.swingfoot_visualizer.set_visibility(False)    
            if hasattr(self, "currentstone_visualizer"):
                self.currentstone_visualizer.set_visibility(False)
            if hasattr(self, "nextstone_visualizer"):
                self.nextstone_visualizer.set_visibility(False)
            if hasattr(self, "nextnextstone_visualizer"):
                self.nextnextstone_visualizer.set_visibility(False)
            if hasattr(self, "terrain_origin_visualizer"):
                self.terrain_origin_visualizer.set_visibility(False)    
            if hasattr(self, "comref_visualizer"):
                self.comref_visualizer.set_visibility(False)    
        return
    
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        if self.debug_vis:
            self.foottarget_visualizer.visualize(self.foottarget_vis_pos,self.foottarget_vis_quat)
            self.swingfoot_visualizer.visualize(self.swingfoot_vis_pos, self.swingfoot_vis_quat)
            #so that center of visualized block is at stone center
            stone_center_offset = torch.tensor([0.0, 0.0, -STONES.stone_z/2.0], device=self.device)
            self.currentstone_visualizer.visualize(self.current_stone_pos + stone_center_offset ,self.stone_quat)
            next_stone_pos = self.current_stone_pos + stone_center_offset
            next_stone_pos[:,0] += self.ldes
            next_stone_pos[:,2] += self.hdes
            self.nextstone_visualizer.visualize(self.next_stone_pos+ stone_center_offset ,self.stone_quat)
            # nextnext_stone_pos = self.current_stone_pos + stone_center_offset
            # nextnext_stone_pos[:,0] += self.ldes + self.ldes_next
            # nextnext_stone_pos[:,2] += self.hdes + self.hdes_next
            self.nextnextstone_visualizer.visualize(self.nextnext_stone_pos+ stone_center_offset ,self.stone_quat)
            # self.terrain_origin_visualizer.visualize(self.terrain.env_origins, self.stone_quat)
            self.comref_visualizer.visualize(self.com_frame_vis_pos, self.com_frame_vis_quat)
            