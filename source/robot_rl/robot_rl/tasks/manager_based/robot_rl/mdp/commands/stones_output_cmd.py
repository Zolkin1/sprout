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
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.hlip import HLIP_3D

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

# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz


if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.stones_output_cmd_cfg import StonesOutputCommandCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.mlip_cmd import euler_rates_to_omega_b, get_euler_from_quat, _transfer_to_local_frame


from .phase_var import PhaseVar

def tensor_t_now(env):
    return env.episode_length_buf * env.step_dt

class StonesOutputCommandTerm(CommandTerm):
    z0: torch.Tensor  # (num_envs,) nominal CoM height
    y_nom: float  # nominal lateral foot placement
    TSS: torch.Tensor  # (num_envs,) single support duration
    TDS: torch.Tensor  # (num_envs,) double support duration
    
    tSSplus: torch.Tensor  # (num_envs,) time of last SS+ event 

    
    stance_idx: torch.Tensor  # (num_envs,) current stance foot index (0: left, 1: right)
    
    ith_step: torch.Tensor  # (num_envs,) current step index
    def __init__(self, cfg: "StonesOutputCommandCfg", env):
        super().__init__(cfg, env)

        self.z0 = torch.full((self.num_envs,), cfg.z0, device=self.device)
        self.y_nom = cfg.y_nom
        
        if env.cfg.commands.step_period.period_range[0] == env.cfg.commands.step_period.period_range[1]:
            self.tSSplus = tensor_t_now(env)
            self.TSS = torch.full((self.num_envs,), env.cfg.commands.step_period.period_range[0]/2.0, device=self.device)
            self.TDS = torch.full((self.num_envs,), 0.0, device=self.device)

            self.phase_var = PhaseVar(self.tSSplus , self.tSSplus + self.TSS)
            self.stance_idx = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device) #start with left foot stance
            
        else:
            raise ValueError("StonesOutputCommandTerm requires fixed step period.")

        self.debug_vis = cfg.debug_vis

        self.robot = env.scene[cfg.asset_name]
        #list of int, left foot idx 0, right foot idx 1
        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        self.upper_body_joint_idx = self.robot.find_joints(cfg.upper_body_joint_name)[0]

        self.foot_target = torch.zeros((self.num_envs, 3), device=self.device)

        self.metrics = {}

        n_output = 12 + len(self.upper_body_joint_idx)
        self.y_out = torch.zeros((self.num_envs, n_output), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, n_output), device=self.device)
        self.y_act = torch.zeros((self.num_envs, n_output), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, n_output), device=self.device)

        grav = torch.abs(torch.tensor(self._env.cfg.sim.gravity[2], device=self.device))
        
        self.hlip = HLIP_3D(
            num_envs=self.num_envs,
            grav=grav,
            z0=self.z0,
            TSS=self.TSS,
            TDS=self.TDS,
            use_momentum=False,
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

        self.ith_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        
        self.stance_foot_pos_0 = torch.zeros((self.num_envs, 3), device=self.device) 
        self.stance_foot_ori_quat_0 = torch.zeros((self.num_envs, 4), device=self.device) 
        self.stance_foot_ori_0 = torch.zeros((self.num_envs, 3), device=self.device) 

    @property
    def command(self):
        return self.foot_target
    
    def _resample_command(self, env_ids):
        self._update_command()
        return
    
    def _update_command(self):
        self.timeBasedDomainContactStatusSwitch()
        self.update_walking_target()
        self.compute_desired()
        self.compute_actual()

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
        # max_clf = self._env.reward_manager.get_term_cfg("clf_reward").params["max_clf"]
        # self.metrics["max_clf"] = torch.ones((self.num_envs), device=self.device) * max_clf
        # return self.foot_target  # Return the foot target tensor for observation

    def timeBasedDomainContactStatusSwitch(self):
        tnow = tensor_t_now(self._env)
        
        #todo: for now, no DS phase
        # mask_SS2DS = tnow >= (self.tSSplus + self.TSS)
        # mask_DS2SS = tnow >= (self.tDSplus + self.TDS)
        eps = 1e-6
        
        mask_next_step = tnow >= (self.tSSplus + self.TSS - eps) 
        reset_buf = getattr(self._env, "reset_buf", None)
        if reset_buf is None:
            #initialization, make true tensor of size (num_envs, )
            mask_next_step = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        else:
            mask_next_step = mask_next_step | reset_buf.bool()
        
        
        
        
        if torch.any(mask_next_step):
            #update for envs that need to switch
            self.tSSplus[mask_next_step] = tnow[mask_next_step]
            #recompute TSS
            #todo: fixed for now
            self.TSS[mask_next_step] = self._env.cfg.commands.step_period.period_range[0]/2.0
            self.hlip.update_hlip(self.z0, self.TSS, self.TDS)
            self.phase_var.reconfigure(self.tSSplus , self.tSSplus + self.TSS)
            self.stance_idx[mask_next_step] = 1 - self.stance_idx[mask_next_step]  # switch stance foot
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            self.stance_foot_pos_0[mask_next_step, :] = foot_pos_w[mask_next_step, self.stance_idx[mask_next_step], :]
            self.stance_foot_ori_quat_0[mask_next_step, :] = foot_ori_w[mask_next_step, self.stance_idx[mask_next_step], :]
            self.stance_foot_ori_0[mask_next_step, :] = get_euler_from_quat(self.stance_foot_ori_quat_0[mask_next_step])
            #update ith_step
            self.ith_step[mask_next_step] += 1


        #update phase var
        self.phase_var.update(tnow)

            


        return

            
    def update_walking_target(self):
        #given velocity command, update MLIP
        base_vdes = self._env.command_manager.get_command("base_velocity")  # (N,3)

        self.hlip.update_desired_walking(base_vdes[:,0],base_vdes[:,1], self.cfg.y_nom)

        self.delta_yaw = base_vdes[:, 2] * self.phase_var.time_in_step
        self.yaw_dot = base_vdes[:, 2]
        self.target_yaw = self.stance_foot_ori_0[:, 2] + self.delta_yaw

    def compute_actual(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        data = self.robot.data
        root_quat = data.root_quat_w
        
        #use data.heading_w for yaw

        # 1. Foot base frame positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]
        batch_idx = torch.arange(self.num_envs, device=self.device)
        
        #for holonomic constraint
        self.stance_foot_pos = foot_pos_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ori = get_euler_from_quat(foot_ori_w[batch_idx, self.stance_idx, :])

        # Convert foot positions to the robot's yaw-aligned local frame
        
        swing_idx = 1-self.stance_idx
        swing2stance_local = _transfer_to_local_frame(
            foot_pos_w[batch_idx, swing_idx, :] - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )

        # Center of mass to stance foot vector in local frame
        com_w = data.root_com_pos_w
        com2stance_local = _transfer_to_local_frame(com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0)
        
        
        # compute real COM
        link_pos_w = data.body_com_pos_w # (num_envs, num_bodies, 3)
        link_mass = data.default_mass.to(self.device) # (num_envs, num_bodies)
        com_pos_w = (link_mass.unsqueeze(-1) * link_pos_w).sum(dim=1) / self.mass # (num_envs,  3)

        # compute mass-normalized angular momentum
        link_inertia_b = data.default_inertia.to(self.device).view(*data.default_inertia.shape[:-1], 3, 3) # (num_envs, num_bodies, 3, 3)
        link_vel_w = data.body_com_lin_vel_w # (num_envs, num_bodies, 3)
        link_omega_w = data.body_com_ang_vel_w # (num_envs, num_bodies, 3)
        link_quat_w = data.body_quat_w # (num_envs, num_bodies, 4)
        R_wb = matrix_from_quat(link_quat_w)  # (N,B,3,3)
        link_inertia_w = R_wb @ link_inertia_b @ R_wb.transpose(-1, -2) # (num_envs, num_bodies, 3, 3)
        term_rot = (link_inertia_w @ link_omega_w.unsqueeze(-1)).squeeze(-1) # (num_envs, num_bodies, 3)
        term_trans = torch.cross(link_pos_w - com_pos_w.unsqueeze(1), link_mass.unsqueeze(-1) * link_vel_w, dim=-1) # (num_envs, num_bodies, 3)
        h_ang_w = (term_rot.sum(dim=1) + term_trans.sum(dim=1)) / self.mass # (num_envs, 3)

        # Pelvis orientation (Euler XYZ)
        pelvis_ori = get_euler_from_quat(root_quat)

        # Foot orientations (Euler XYZ)
        swing_foot_ori = get_euler_from_quat(foot_ori_w[batch_idx, swing_idx, :])


        # 2. Velocities (world frame)
        com_vel_w = data.root_com_vel_w[:, 0:3]

        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[batch_idx, self.stance_idx, :]
        # Convert velocities to local frame

        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)

        pelvis_omega_local = data.root_ang_vel_b

        foot_lin_vel_local_swing = _transfer_to_local_frame(
            foot_lin_vel_w[batch_idx, swing_idx, :], self.stance_foot_ori_quat_0
        )

        foot_ang_vel_local_swing = quat_apply(
            quat_inv(foot_ori_w[batch_idx, swing_idx, :]), foot_ang_vel_w[batch_idx, swing_idx, :]
        )

        swing2stance_vel = foot_lin_vel_local_swing

        upper_body_joint_pos = self.robot.data.joint_pos[:, self.upper_body_joint_idx]
        upper_body_joint_vel = self.robot.data.joint_vel[:, self.upper_body_joint_idx]

        
        self.y_act = torch.cat(
            [com2stance_local, pelvis_ori, swing2stance_local, swing_foot_ori,  upper_body_joint_pos], dim=-1
        )

        self.dy_act = torch.cat(
            [com_vel_local, pelvis_omega_local, swing2stance_vel, foot_ang_vel_local_swing, upper_body_joint_vel],
            dim=-1,
        )
        
        
    
    def compute_desired(self):
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
        # self.y_out = torch.cat(
        #     [com_pos_des_yaw_adjusted, pelvis_eulxyz, swing_foot_pos, swingfoot_eulxyz,  stance_foot_pitch_angle, upper_body_joint_pos], dim=-1
        # )

        # self.dy_out = torch.cat(
        #     [com_vel_des_yaw_adjusted, omega_pelvis_ref, swing_foot_vel, omega_foot_ref, stance_foot_pitch_vel, upper_body_joint_vel], dim=-1
        # )
        self.y_out = torch.cat(
            [com_pos_des_yaw_adjusted, pelvis_eulxyz, swing_foot_pos, swingfoot_eulxyz, upper_body_joint_pos], dim=-1
        )

        self.dy_out = torch.cat(
            [com_vel_des_yaw_adjusted, omega_pelvis_ref, swing_foot_vel, omega_foot_ref, upper_body_joint_vel], dim=-1
        )
        
        if self.debug_vis:
            self.swingfoot_quat = quat_from_euler_xyz(swingfoot_eulxyz[:, 0], swingfoot_eulxyz[:, 1], swingfoot_eulxyz[:, 2]) #Shape: (N,4)
            self.quat_target_frame = quat_from_euler_xyz(torch.zeros_like(self.target_yaw), torch.zeros_like(self.target_yaw), self.target_yaw) #Shape: (N,4)
            self.swingfoot_world_pos = self.stance_foot_pos_0 + quat_apply(self.quat_target_frame, foot_target) #Shape: (N,3)
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
            self.footprint_visualizer = VisualizationMarkers(self.cfg.footprint_cfg)
            self.footprint_visualizer.set_visibility(True)
        else:
            if hasattr(self, "footprint_visualizer"):
                self.footprint_visualizer.set_visibility(False)
        return
    
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        if self.debug_vis:
            self.footprint_visualizer.visualize(self.swingfoot_world_pos,self.swingfoot_quat)