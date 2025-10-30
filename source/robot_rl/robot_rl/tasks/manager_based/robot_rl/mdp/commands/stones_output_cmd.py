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
    stance_foot_pitch = 12


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
    
    def __init__(self, cfg: "StonesOutputCommandCfg", env):
        super().__init__(cfg, env)

        self.z0 = torch.full((self.num_envs,), cfg.z0, device=self.device)
        self.y_nom = cfg.y_nom
        self.use_momentum = cfg.use_momentum    
        
        self.tSSplus = tensor_t_now(env)
        self.TSS = torch.full((self.num_envs,), env.cfg.commands.step_period.period_range[0]/2.0, device=self.device)
        self.TDS = torch.full((self.num_envs,), 0.0, device=self.device)

        self.phase_var = PhaseVar(self.tSSplus , self.tSSplus + self.TSS)
        self.stance_idx = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device) #start with left foot stance


        self.debug_vis = cfg.debug_vis

        self.robot = env.scene[cfg.asset_name]
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
        self.hdes_next = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.ldes_next = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.xcomf_des = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.zcom0_des = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.zcomf_des = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.dzcomf_des = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
                                
        #todo: for long stones now, always identity quat
        self.stone_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=self.device)
        self.stone_quat[:, 0] = 1.0 # identity quat

        self.abs_x = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.abs_y = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)
        self.abs_z = torch.zeros((env.num_envs, STONES.num_stones + STONES.num_init_steps), dtype=torch.float32, device=self.device)

    @property
    def command(self):
        return self.v
    
    def _resample_command(self, env_ids):
        self._update_command()
        return
    
    def _update_command(self):
        self.timeBasedDomainContactStatusSwitch()

        self.update_walking_target()
        
        self.compute_actual()

        if TEST_FLAT == False:
            self.compute_desired_stones()
        else:
            self.compute_desired_flat()    
        

        # how to handle for the first step?
        # i.e. v is not defined
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, self.cfg.yaw_idx)
        self.vdot = vdot
        self.v = vcur
        if torch.sum(self.v_buffer) == 0:
            # (E,) -> (E,1) -> broadcast to (E,100) on assignment
            self.v_buffer[:] = self.v.unsqueeze(1)
            self.vdot_buffer[:] = self.vdot.unsqueeze(1)

        else:
            self.v_buffer = torch.cat([self.v_buffer[:, 1:], self.v.unsqueeze(-1)], dim=-1)
            self.vdot_buffer = torch.cat([self.vdot_buffer[:, 1:], self.vdot.unsqueeze(-1)], dim=-1)

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
        self.metrics["error_st_pitch"] = torch.abs(self.y_out[:, YIdx.stance_foot_pitch] - self.y_act[:, YIdx.stance_foot_pitch])

        self.metrics["error_com_x"] = torch.abs(self.y_out[:, YIdx.comx] - self.y_act[:, YIdx.comx])
        self.metrics["error_com_y"] = torch.abs(self.y_out[:, YIdx.comy] - self.y_act[:, YIdx.comy])
        self.metrics["error_com_z"] = torch.abs(self.y_out[:, YIdx.comz] - self.y_act[:, YIdx.comz])
        self.metrics["error_pelvis_roll"] = torch.abs(self.y_out[:, YIdx.pelvis_roll] - self.y_act[:, YIdx.pelvis_roll])
        self.metrics["error_pelvis_pitch"] = torch.abs(self.y_out[:, YIdx.pelvis_pitch] - self.y_act[:, YIdx.pelvis_pitch])
        self.metrics["error_pelvis_yaw"] = torch.abs(self.y_out[:, YIdx.pelvis_yaw] - self.y_act[:, YIdx.pelvis_yaw])

        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot
        self.metrics["avg_clf"] = torch.mean(self.v_buffer, dim=-1)
        return



    def reset_stone_global(self, reset_mask):
        if reset_mask.any():
            # Extract terrain info
            terrain = self._env.scene.terrain
            rel_x = terrain.env_terrain_infos["rel_x"] #(num_envs, num_stones)
            rel_z = terrain.env_terrain_infos["rel_z"] #(num_envs, num_stones)
            start_stone_pos_w = terrain.env_terrain_infos["start_stone_pos"] + terrain.env_origins #(num_envs, 3)
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
            self.abs_y[reset_mask] = start_pos[:, 1:2] #use stone y as constant y position
        return
    
    def update_ithstep_stones_info(self, ithstep, mask, current_stone_pos):
        """Update next and next-next stepping stone positions only for masked envs."""
        if not torch.any(mask):
            return
        # --- Upcoming stepping stone, the target stone for swing foot ---
        idx_next = torch.clamp(
            ithstep[mask], 
            max=STONES.num_stones + STONES.num_init_steps - 1
        )
        self.next_stone_pos[mask, 0] = torch.gather(self.abs_x[mask], 1, idx_next.unsqueeze(1)).squeeze(1)
        self.next_stone_pos[mask, 1] = torch.gather(self.abs_y[mask], 1, idx_next.unsqueeze(1)).squeeze(1)
        self.next_stone_pos[mask, 2] = torch.gather(self.abs_z[mask], 1, idx_next.unsqueeze(1)).squeeze(1)

        # --- Next Upcoming stepping stone, preview ---
        idx_next_next = torch.clamp(
            ithstep[mask] + 1, 
            max=STONES.num_stones + STONES.num_init_steps - 1
        )

        self.nextnext_stone_pos[mask, 0] = torch.gather(self.abs_x[mask], 1, idx_next_next.unsqueeze(1)).squeeze(1)
        self.nextnext_stone_pos[mask, 1] = torch.gather(self.abs_y[mask], 1, idx_next_next.unsqueeze(1)).squeeze(1)
        self.nextnext_stone_pos[mask, 2] = torch.gather(self.abs_z[mask], 1, idx_next_next.unsqueeze(1)).squeeze(1)
        
        self.hdes[mask] = self.next_stone_pos[mask, 2] - current_stone_pos[mask, 2]
        self.ldes[mask] = self.next_stone_pos[mask, 0] - current_stone_pos[mask, 0]
        self.hdes_next[mask] = self.nextnext_stone_pos[mask, 2] - self.next_stone_pos[mask, 2]
        self.ldes_next[mask] = self.nextnext_stone_pos[mask, 0] - self.next_stone_pos[mask, 0]

        done_mask_local = ithstep[mask] >= (STONES.num_stones + STONES.num_init_steps - 2)
        if torch.any(done_mask_local):
            # Set fallback values for those envs
            done_mask_global = torch.zeros_like(mask, dtype=torch.bool)
            done_mask_global[mask] = done_mask_local
            self.ldes[done_mask_global] = 0.3
            self.ldes_next[done_mask_global] = 0.3
        
        
        if torch.any(self.ldes_next < ZERO_EPS):
            #raise error
            print("Warning: ldes_next is non-positive!")
        return

    def timeBasedDomainContactStatusSwitch(self):
        tnow = tensor_t_now(self._env)
        
        #todo: for now, no DS phase
        # mask_SS2DS = tnow >= (self.tSSplus + self.TSS)
        # mask_DS2SS = tnow >= (self.tDSplus + self.TDS)

        mask_next_step = tnow >= (self.tSSplus + self.TSS - ZERO_EPS) #mask for time-based step switch
        mask_reset_buf = getattr(self._env, "reset_buf", None) #mask for env reset

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
                self.prev_stone_pos[mask_next_step] = self.current_stone_pos[mask_next_step]
                self.current_stone_pos[mask_next_step, 0:2] = self.stance_foot_pos_0[mask_next_step, 0:2] #update current stone x-y pos based on stance foot 
                mask_no_progress_this_step_local = (self.next_stone_pos[mask_next_step,0] - self.current_stone_pos[mask_next_step,0] ) > STONES.stone_x
                mask_no_progress_this_step_global = torch.zeros_like(mask_next_step, dtype=torch.bool)
                mask_no_progress_this_step_global[mask_next_step] = mask_no_progress_this_step_local
                self.ith_step[mask_no_progress_this_step_global] -=1 #if no progress this step, do not increment ith step
                
                mask_update_z = torch.logical_not(mask_no_progress_this_step_global) & mask_next_step
                #update current stone z pos based on next stone z pos, to avoid z offset bewteen foot contact point and stance foot origin frame
                self.current_stone_pos[mask_update_z, 2] = self.next_stone_pos[mask_update_z,2] 
                
                self.update_ithstep_stones_info(self.ith_step, mask_next_step, self.current_stone_pos) #update next and next-next stone pos
            self.reset_impact(mask_next_step)
        
        if torch.any(mask_reset_buf):
            self.ith_step[mask_reset_buf] = 0 #env reset, reset ith step
            self.reset_stance_swing_foot(mask_reset_buf)
            if TEST_FLAT == False:
                #update current stone pos, assume first step z is always zero
                self.current_stone_pos[mask_reset_buf, 0:2] = self.stance_foot_pos_0[mask_reset_buf, 0:2] #initialize current stone x-y pos based on stance foot
                self.current_stone_pos[mask_reset_buf, 2] = 0 #initialize current stone z pos to zero, asumming always launch on the platform
                self.prev_stone_pos[mask_reset_buf] = self.current_stone_pos[mask_reset_buf]
                self.reset_stone_global(mask_reset_buf)
                self.update_ithstep_stones_info(self.ith_step, mask_reset_buf, self.current_stone_pos) #update next and next-next stone pos
            self.reset_impact(mask_reset_buf)

        #update phase var
        self.phase_var.update(tnow)
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
        foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
        self.stance_foot_pos_0[mask, :] = foot_pos_w[mask, self.stance_idx[mask], :]
        self.stance_foot_ori_quat_0[mask, :] = foot_ori_w[mask, self.stance_idx[mask], :]
        self.stance_foot_ori_0[mask, :] = get_euler_from_quat(self.stance_foot_ori_quat_0[mask])  
        self.stance_foot_ori_quat_0_zerorollpitch[mask, :] = quat_from_euler_xyz(
            torch.zeros((torch.sum(mask),), device=self.device),
            torch.zeros((torch.sum(mask),), device=self.device),
            self.stance_foot_ori_0[mask, 2]
        )
        self.swing_foot_pos_0[mask, :] = foot_pos_w[mask, 1 - self.stance_idx[mask], :]
        return 
    
    def reset_impact(self, mask):
        #to be called when impact is detected
        self.tSSplus[mask] = tensor_t_now(self._env)[mask]
        #recompute TSS
        if TEST_FLAT:

            self.TSS[mask] = self._env.cfg.commands.step_period.period_range[0]/2.0
        else: 
            #todo: assume zero target yaw for stones now
            target_yaw = torch.zeros((self.num_envs,), device=self.device)
            self.com_pos_target_yaw0, self.com_state2_target_yaw0 = self.compute_com_rel_to_stance_state_target_yaw_frame(self.use_momentum, target_yaw)
            x0 = self.get_sagittal_com_states_stacked(self.com_pos_target_yaw0, self.com_state2_target_yaw0, self.use_momentum)

            self.xcomf_des[mask] = self.cfg.eps * self.ldes[mask]
            mask_ldesnext_nonzero = self.ldes_next.abs() > ZERO_EPS
            middle_term = torch.where(
                mask_ldesnext_nonzero,
                (self.hdes_next / self.ldes_next) * (1.0 - self.cfg.eps) * self.ldes,
                torch.zeros_like(self.ldes_next)
            )
            tmp = self.cfg.eps * self.hdes - middle_term + self.hdes
            self.zcomf_des[mask] = self.z0[mask] + tmp[mask] / 2.0
            #todo, for now, set com z initial reference to z0, in practice, relate to hdes
            self.zcom0_des[mask] = self.z0[mask]  # self.com_pos_target_yaw0[mask, 2] - self.stance_foot_pos_0[mask, 2]
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
        self.hlip.update_hlip(self.z0, self.TSS, self.TDS)
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
            return torch.cat([com3d_state1[:,1], -com3d_state2[:,0]], dim=1)  # Shape: (num_envs, 2)
        else:
            # return com_y and com_vy
            return torch.cat([com3d_state1[:,1], com3d_state2[:,1]], dim=1)  # Shape: (num_envs, 2)
    
    def update_walking_target(self):
        #given velocity command, update MLIP
        base_vdes = self._env.command_manager.get_command("base_velocity")  # (N,3)
        if TEST_FLAT:
            # 3D HLIP: uses vx, vy, y_nom
            self.hlip.update_desired_walking(base_vdes[:,0],base_vdes[:,1], self.cfg.y_nom)
            self.delta_yaw = base_vdes[:, 2] * self.phase_var.time_in_step
            self.yaw_dot = base_vdes[:, 2]
            self.target_yaw = self.stance_foot_ori_0[:, 2] + self.delta_yaw
        else:
            # P2 HLIP: uses only lateral vy, y_nom; but in fact never update
            self.hlip.update_desired_walking(base_vdes[:,1], self.cfg.y_nom)
            # for stepping stones, always targeting zero yaw
            # todo: consider adding different terrain yaw and replace zeros_like below
            self.target_yaw = torch.zeros_like(self.stance_foot_ori_0[:, 2])
            
            self.yaw_dot = (self.target_yaw - self.stance_foot_ori_0[:, 2]) / self.TSS
            self.delta_yaw = self.yaw_dot * self.phase_var.time_in_step
        return
        

    def compute_actual(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame,
        i.e. local = self.stance_foot_ori_quat_0 frame."""
        
        
        
        
        # Convenience
        data = self.robot.data
        root_quat = data.root_quat_w
        
        #use data.heading_w for yaw

        # 1. Foot base frame positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]
        batch_idx = torch.arange(self.num_envs, device=self.device)
        
        #for holonomic constraint, current stance foot position and orientation
        self.stance_foot_pos = foot_pos_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ori = get_euler_from_quat(foot_ori_w[batch_idx, self.stance_idx, :])

        # Convert foot positions to the robot's yaw-aligned local frame
        
        swing_idx = 1-self.stance_idx
        swing2stance0_local = _transfer_to_local_frame(
            foot_pos_w[batch_idx, swing_idx, :] - self.stance_foot_pos_0, self.stance_foot_ori_quat_0_zerorollpitch
        )

        # Center of mass to stance foot vector in local frame
        #todo: used generic root com pos for now
        com_w = data.root_com_pos_w
        com2stance_local = _transfer_to_local_frame(com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0_zerorollpitch)

        # Pelvis orientation (Euler XYZ)
        pelvis_eulxyz = get_euler_from_quat(root_quat)

        # Foot orientations (Euler XYZ)
        swing_foot_eulxyz = get_euler_from_quat(foot_ori_w[batch_idx, swing_idx, :])


        # 2. Velocities (world frame)
        com_vel_w = data.root_com_vel_w[:, 0:3]

        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[batch_idx, self.stance_idx, :]
        # Convert velocities to local frame

        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)

        pelvis_omega_body = data.root_ang_vel_b

        swing_foot_lin_vel_local = _transfer_to_local_frame(
            foot_lin_vel_w[batch_idx, swing_idx, :], self.stance_foot_ori_quat_0
        )

        swing_foot_omega_body = quat_apply(
            quat_inv(foot_ori_w[batch_idx, swing_idx, :]), foot_ang_vel_w[batch_idx, swing_idx, :]
        )

        # 3. Upper body joint positions and velocities
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
    
    
    def compute_dzcomf_des(self):
        #predict preimpact sagittal com x and vx/Ly
        com_now, com_d_now = self.compute_com_rel_to_stance_state_target_yaw_frame(self.use_momentum, torch.zeros((self.num_envs,), device=self.device))
        time2impact = self.TSS - self.phase_var.time_in_step
        if self.use_momentum:
            xnow_sagittal = torch.stack([com_now[:,0], com_d_now[:,1]], dim=1) #Shape: (N,2)
        else:
            xnow_sagittal = torch.stack([com_now[:,0], com_d_now[:,0]], dim=1) #Shape: (N,2)
        x_sagittal_minus, state2_sagittal_minus = self.hlip.get_com_state_from_x0_sagittal(xnow_sagittal, time2impact)#Shape: (N,2)
        #compute post-impact x com pos: p_plus
        x_plus = x_sagittal_minus - self.ldes
        
        z0LIPplus_real = com_now[:,2] - self.hdes - x_plus * self.hdes_next / self.ldes_next
        z0LIPplus_des = self.zcomf_des - self.hdes - x_plus * self.hdes_next / self.ldes_next
        z0_interp = self.phase_var.tau * z0LIPplus_real + (1.0 - self.phase_var.tau) * z0LIPplus_des
        
        state2_sagittal_plus = solve_velocity_or_momentum_positive_from_E_batched(E=self.cfg.E_star,
                                                                    p=x_plus,
                                                                    z_tilde=z0_interp,
                                                                    use_momentum=self.use_momentum)
        vx_com_now = com_d_now[:,0]
        if self.use_momentum:
            self.dzcomf_des = (state2_sagittal_plus - state2_sagittal_minus + self.hdes * vx_com_now) / self.ldes
        else:
            self.dzcomf_des = ((state2_sagittal_plus - state2_sagittal_minus)*z0_interp + self.hdes * vx_com_now) / self.ldes 
        
        return 

    def compute_desired_stones(self):
        N = self.num_envs

        pelvis_eulxyz = torch.zeros((N, 3), device=self.device) #Shape: (N,3)
        pelvis_eulxyz[:, 2] =  self.stance_foot_ori_0[:, 2] + self.delta_yaw
        pelvis_eulxyz_dot = torch.zeros((N, 3), device=self.device)#Shape: (N,3)
        pelvis_eulxyz_dot[:, 2] = self.yaw_dot
        swingfoot_eulxyz = pelvis_eulxyz #Shape: (N,3)
        swingfoot_eulxyz_dot = pelvis_eulxyz_dot #Shape: (N,3)
        #convert euler rates to omega body
        omega_pelvis_ref = euler_rates_to_omega_b(pelvis_eulxyz, pelvis_eulxyz_dot)
        omega_foot_ref = euler_rates_to_omega_b(swingfoot_eulxyz, swingfoot_eulxyz_dot)  # (N,3)
        
        upper_body_joint_pos, upper_body_joint_vel = self.generate_upper_body_ref() #Shape: (N, num_upper_joints)
        
        zeros_N = torch.zeros((N,), device=self.device)
        quat_target_frame = quat_from_euler_xyz(zeros_N, zeros_N, self.target_yaw)  # Shape: (N,4)
        quat_targetyaw_to_stanceyaw = quat_from_euler_xyz(zeros_N, zeros_N, self.target_yaw-self.stance_foot_ori_0[:, 2])  # Shape: (N,4)
        
        
        # Swingfoot rel to stancefoot x based on stepping stone position, in world frame
        Ux = self.ldes
        # Get desired foot placements from HLIP in target yaw frame, no feedback used here
        #TODO: later consider add feedback
        Uy = self.hlip.get_desired_foot_placement(self.stance_idx) #Shape: (N,)
        #TODO: for now target yaw frame = world frame, later consider adding different terrain yaw
        footplacement_target_frame = torch.stack([Ux, Uy, torch.zeros((N), device=self.device)], dim=-1) # Shape: (N,3)
        # Create horizontal control points with batch dimension
        horizontal_control_points = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], device=self.device)
        bht = bezier_deg(0, self.phase_var.tau, self.TSS, horizontal_control_points, 4) #Shape: (N,)
        dbht = bezier_deg(1, self.phase_var.tau, self.TSS, horizontal_control_points, 4) #Shape: (N,)
        # Horizontal X and Y (linear interpolation)
        x_init_target_frame = self.prev_stone_pos[:, 0] - self.current_stone_pos[:, 0]
        
        
        p_swing_x = ((1 - bht) * x_init_target_frame + bht * Ux) #Shape: (N,)
        v_swing_x = ((-dbht) * x_init_target_frame + dbht * Ux)  #Shape: (N,)
        
        #clamp swing foot y to avoid too large step
        sign_swy = torch.sign(Uy)
        #todo: for now, since not going sideway, just assume swy_0 = swy_T
        y0 = sign_swy * self.cfg.y_nom #Shape: (N,)
        p_swing_y = ((1 - bht) * (y0) + bht * Uy) #Shape: (N,)
        v_swing_y = ((-dbht) * y0 + dbht * Uy)  #Shape: (N,)


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
        com_y_target_frame, com_dy_target_frame = self.hlip.get_desired_com_state(self.stance_idx, self.phase_var.time_in_step) #Shape: (N,)
        if self.use_momentum: #convert angular momentum to velocity
            com_dy_target_frame = com_dy_target_frame / self.z0 #vy^d = LxLIP^d / z0
            
        
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
        # self.compute_dzcomf_des()
        # #TODO: check if it is better to use true zcom0
        # cubic_comz_coeff = cubic_spline_coeff_batched(self.zcom0_des , 
        #                                         torch.zeros_like(self.zcom0_des),
        #                                         self.zcomf_des ,
        #                                         self.dzcomf_des,
        #                                         self.phase_var.dtau) #Shape: (N,4)
        
        # com_z, com_dz = cubic_spline_eval(cubic_comz_coeff, self.phase_var.tau, self.phase_var.dtau)
        com_z_w = self.z0
        com_dz_w = torch.zeros_like(com_z_w)
        # Concatenate x y z components
        com_pos_des = torch.stack(
            [com_x_target_frame, com_y_target_frame, com_z_w], dim=-1
        )  # Shape: (N,3)
        com_vel_des = torch.stack([com_dx_target_frame, com_dy_target_frame, com_dz_w], dim=-1)  # Shape: (N,3)

        com_pos_des_local = quat_apply(quat_targetyaw_to_stanceyaw, com_pos_des)  # [N,3]
        com_vel_des_local = quat_apply(quat_targetyaw_to_stanceyaw, com_vel_des)  # [N,3]

        #todo: for DS, swing foot should remain constant

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
    
    def compute_desired_flat(self):
        N = self.num_envs

        pelvis_eulxyz = torch.zeros((N, 3), device=self.device) #Shape: (N,3)
        pelvis_eulxyz[:, 2] =  self.stance_foot_ori_0[:, 2] + self.delta_yaw
        pelvis_eulxyz_dot = torch.zeros((N, 3), device=self.device)#Shape: (N,3)
        pelvis_eulxyz_dot[:, 2] = self.yaw_dot
        swingfoot_eulxyz = pelvis_eulxyz #Shape: (N,3)
        swingfoot_eulxyz_dot = pelvis_eulxyz_dot #Shape: (N,3)
        
        upper_body_joint_pos, upper_body_joint_vel = self.generate_upper_body_ref() #Shape: (N, num_upper_joints)
        
        # Get desired foot placements and CoM trajectory from HLIP in target yaw frame, no feedback used here
        Ux,  Uy = self.hlip.get_desired_foot_placement(self.stance_idx) #Shape: (N,)
        com_x, com_dx, com_y, com_dy = self.hlip.get_desired_com_state(self.stance_idx, self.phase_var.time_in_step) #Shape: (N,)
        
        if self.use_momentum:
            com_dx = com_dx / self.z0
            com_dy = com_dy / self.z0

        # Concatenate x y z components
        com_pos_des = torch.stack(
            [com_x, com_y, torch.ones((N,), device=self.device) * self.z0], dim=-1
        )  # Shape: (N,3)
        com_vel_des = torch.stack([com_dx, com_dy, torch.zeros((N), device=self.device)], dim=-1)  # Shape: (N,3)
        foot_target = torch.stack([Ux, Uy, torch.zeros((N), device=self.device)], dim=-1) # Shape: (N,3)
        

        # based on yaw velocity, update com_pos_des, com_vel_des, foot_target,
        
        pelvis_eulxyz_actual = get_euler_from_quat(self.robot.data.root_quat_w) #Shape: (N,3)
        # quat_delta_yaw = quat_from_euler_xyz(
        #     torch.zeros_like(self.delta_yaw), torch.zeros_like(self.delta_yaw), self.target_yaw - pelvis_eulxyz_actual[:, 2]  # roll=0  # pitch=0  # yaw=Δyaw
        # ) # Shape: (N,4)
        #todo: check which one is better
        quat_delta_yaw = quat_from_euler_xyz(
            torch.zeros_like(self.delta_yaw), torch.zeros_like(self.delta_yaw), self.delta_yaw  # roll=0  # pitch=0  # yaw=Δyaw
        ) # Shape: (N,4)
        swing_foot_target_yaw_adjusted = quat_apply(quat_delta_yaw, foot_target)  # [N,3]
        com_pos_des_yaw_adjusted = quat_apply(quat_delta_yaw, com_pos_des)  # [N,3]
        com_vel_des_yaw_adjusted = quat_apply(quat_delta_yaw, com_vel_des)  # [N,3]
        # self.foot_target = swing_foot_target_yaw_adjusted # Shape: (N,3)
        #clamp swing foot y to avoid too large step
        sign_swy = torch.sign(swing_foot_target_yaw_adjusted[:, 1])
        # self.foot_target[:, 1] = torch.clamp(
        #     swing_foot_target_yaw_adjusted[:, 1],
        #     min=self.cfg.foot_target_range_y[0],
        #     max=self.cfg.foot_target_range_y[1],
        # ) * sign_swy

        

        # Create horizontal control points with batch dimension
        horizontal_control_points = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], device=self.device)

        bht = bezier_deg(0, self.phase_var.tau, self.TSS, horizontal_control_points, 4) #Shape: (N,)
        dbht = bezier_deg(1, self.phase_var.tau, self.TSS, horizontal_control_points, 4) #Shape: (N,)
        # Horizontal X and Y (linear interpolation)
        p_swing_x = ((1 - bht) * (-swing_foot_target_yaw_adjusted[:, 0]) + bht * swing_foot_target_yaw_adjusted[:, 0]) #Shape: (N,)
        v_swing_x = ((-dbht) * (-swing_foot_target_yaw_adjusted[:, 0]) + dbht * swing_foot_target_yaw_adjusted[:, 0])  #Shape: (N,)
        #todo: for now, since not going sideway, just assume swy_0 = swy_T
        y0 = sign_swy * self.cfg.y_nom #Shape: (N,)
        p_swing_y = ((1 - bht) * (y0) + bht * swing_foot_target_yaw_adjusted[:, 1]) #Shape: (N,)
        v_swing_y = ((-dbht) * (swing_foot_target_yaw_adjusted[:, 1]) + dbht * swing_foot_target_yaw_adjusted[:, 1])  #Shape: (N,)


        # swing foot z
        z_sw_max = torch.full((N,), self.cfg.z_sw_max, device=self.device)
        z_sw_neg = torch.full((N,), self.cfg.z_sw_min, device=self.device)
        degree_v = 6
        zsw0 = 0
        z_init = torch.full((N,), zsw0, device=self.device)
        control_v = torch.stack(
        [   z_init,  # Start
            z_init + 0.2 * (z_sw_max - z_init),
            z_init + 0.6 * (z_sw_max - z_init),
            z_sw_max,  # Peak at mid-swing
            z_sw_neg + 0.5 * (z_sw_max - z_sw_neg),
            z_sw_neg + 0.05 * (z_sw_max - z_sw_neg),
            z_sw_neg,  # End
        ], dim=1)  # Shape: (N,7)
        if isinstance(self.phase_var.tau, float):
            phase_tensor = torch.full((N,), self.phase_var.tau, device=self.device)
            T_tensor = torch.full((N,), self.TSS, device=self.device)
        else:
            phase_tensor = self.phase_var.tau
            T_tensor = self.TSS
        p_swing_z = bezier_deg(0, phase_tensor, T_tensor, control_v, degree_v)
        v_swing_z = bezier_deg(1, phase_tensor, T_tensor, control_v, degree_v)

        # Combine to get full swing foot position and velocity
        swing_foot_pos = torch.stack([p_swing_x, p_swing_y, p_swing_z], dim=-1)  # Shape: (N,3)
        swing_foot_vel = torch.stack([v_swing_x, v_swing_y, v_swing_z], dim=-1)  # Shape: (N,3)

        

        #todo: for DS, swing foot should remain constant

        omega_pelvis_ref = euler_rates_to_omega_b(pelvis_eulxyz, pelvis_eulxyz_dot)
        omega_foot_ref = euler_rates_to_omega_b(swingfoot_eulxyz, swingfoot_eulxyz_dot)  # (N,3)

        self.y_out = torch.cat(
            [com_pos_des_yaw_adjusted, pelvis_eulxyz, swing_foot_pos, swingfoot_eulxyz, upper_body_joint_pos], dim=-1
        )

        self.dy_out = torch.cat(
            [com_vel_des_yaw_adjusted, omega_pelvis_ref, swing_foot_vel, omega_foot_ref, upper_body_joint_vel], dim=-1
        )
        
        if self.debug_vis:
            self.foottarget_vis_quat = quat_from_euler_xyz(swingfoot_eulxyz[:, 0], swingfoot_eulxyz[:, 1], swingfoot_eulxyz[:, 2]) #Shape: (N,4)
            self.quat_target_frame = quat_from_euler_xyz(torch.zeros_like(self.target_yaw), torch.zeros_like(self.target_yaw), self.target_yaw) #Shape: (N,4)
            self.foottarget_vis_pos = self.stance_foot_pos_0 + quat_apply(self.quat_target_frame, foot_target) #Shape: (N,3)
            self.swingfoot_vis_pos = swing_foot_pos + self.stance_foot_pos_0
            self.swingfoot_vis_quat = quat_from_euler_xyz(swingfoot_eulxyz[:, 0], swingfoot_eulxyz[:, 1], swingfoot_eulxyz[:, 2]) #Shape: (N,4)
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



    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            #foot target and swing foot intermediate visualizer
            self.foottarget_visualizer = VisualizationMarkers(self.cfg.foottarget_cfg)
            self.foottarget_visualizer.set_visibility(True)
            self.swingfoot_visualizer = VisualizationMarkers(self.cfg.swingfoot_cfg)
            self.swingfoot_visualizer.set_visibility(True)
            
            #stone related visualizer
            self.nextstone_visualizer = VisualizationMarkers(self.cfg.nextstone_cfg)
            self.nextstone_visualizer.set_visibility(True)
            self.nextnextstone_visualizer = VisualizationMarkers(self.cfg.nextnextstone_cfg)    
            self.nextnextstone_visualizer.set_visibility(True)
            # self.terrain_origin_visualizer = VisualizationMarkers(self.cfg.originframe_cfg)
            # self.terrain_origin_visualizer.set_visibility(True)
            
            #com frame visualizer
            self.comref_visualizer = VisualizationMarkers(self.cfg.comrefframe_cfg)
            self.comref_visualizer.set_visibility(True)
        else:
            if hasattr(self, "foottarget_visualizer"):
                self.foottarget_visualizer.set_visibility(False)
            if hasattr(self, "swingfoot_visualizer"):
                self.swingfoot_visualizer.set_visibility(False)    
            if hasattr(self, "nextstone_visualizer"):
                self.nextstone_visualizer.set_visibility(False)
            if hasattr(self, "nextnextstone_visualizer"):
                self.nextnextstone_visualizer.set_visibility(False)
            # if hasattr(self, "terrain_origin_visualizer"):
            #     self.terrain_origin_visualizer.set_visibility(False)    
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
            stone_center_offset = torch.tensor([0.0, 0.0, -STONES.stone_z/2.0], device=self.device)
            self.nextstone_visualizer.visualize(self.next_stone_pos + stone_center_offset ,self.stone_quat)
            self.nextnextstone_visualizer.visualize(self.nextnext_stone_pos + stone_center_offset,self.stone_quat)
            # self.terrain_origin_visualizer.visualize(self._env.scene.terrain.env_origins, self.stone_quat)
            self.comref_visualizer.visualize(self.com_frame_vis_pos, self.com_frame_vis_quat)