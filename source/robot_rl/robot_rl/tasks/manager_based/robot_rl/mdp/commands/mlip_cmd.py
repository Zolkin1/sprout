import math
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    wrap_to_pi,
    yaw_quat,
)

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.clf import CLF

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.beizer import bezier_deg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.mlip import MLIP_3D

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
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.mlip_cmd_cfg import MLIPCommandCfg


def euler_rates_to_omega_b(eul: torch.Tensor, eul_rates: torch.Tensor) -> torch.Tensor:
    """
    Convert XYZ extrinsic Euler angle rates into body‐frame angular velocity.

    Args:
        eul:        Tensor of shape (..., 3), Euler angles [φ, θ, ψ]
        eul_rates:  Tensor of shape (..., 3), Euler‐angle rates [φ̇, θ̇, ψ̇]
    Returns:
        omega:      Tensor of shape (..., 3), angular velocity [ωₓ, ωᵧ, ω_z]
    """
    # unpack
    roll, pitch, yaw = eul.unbind(-1)
    
    # Precompute sines/cosines
    s_pitch = torch.sin(pitch)
    c_pitch = torch.cos(pitch)
    s_roll = torch.sin(roll)
    c_roll = torch.cos(roll)
    
    zeros = torch.zeros_like(pitch)
    ones = torch.ones_like(pitch)
    
    RateMatrix = torch.stack(
        [
            torch.stack([ones,   zeros,  -s_pitch], dim=-1),
            torch.stack([zeros, c_roll,  c_pitch*s_roll], dim=-1),
            torch.stack([zeros, -s_roll, c_pitch*c_roll], dim=-1),
        ],
        dim=-2,
    )

    # apply to rates: ω = M @ eul_rates
    omega = torch.einsum("...ij,...j->...i", RateMatrix, eul_rates)
    return omega


def get_euler_from_quat(quat):

    euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
    euler_x = wrap_to_pi(euler_x)
    euler_y = wrap_to_pi(euler_y)
    euler_z = wrap_to_pi(euler_z)
    return torch.stack([euler_x, euler_y, euler_z], dim=-1)

 


def _transfer_to_local_frame(vec, root_quat):
    # apply -local_yaw rotation to vec
    return quat_apply(yaw_quat(quat_inv(root_quat)), vec)


from .phase_var import MLIPPhaseVarGlobal

class MLIPCommandTerm(CommandTerm):
    def __init__(self, cfg: "MLIPCommandCfg", env):
        super().__init__(cfg, env)
        self.z0 = cfg.z0
        self.y_nom = cfg.y_nom
        
        if env.cfg.commands.step_period.period_range[0] == env.cfg.commands.step_period.period_range[1]:
            self._phase_var = MLIPPhaseVarGlobal(env.cfg.commands.step_period.period_range[0])
        else:
            raise ValueError("MLIPCommandTerm requires fixed step period.")

        #todo: check this
        self.debug_vis = cfg.debug_vis
        if self.debug_vis:
            self.footprint_visualizer = VisualizationMarkers(cfg.footprint_cfg)

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
        self.mlip = MLIP_3D(
            num_envs=self.num_envs,
            grav=grav,
            z0=self.z0,
            TOA=self._phase_var.T_oa,
            TFA=self._phase_var.T_fa,
            TUA=self._phase_var.T_ua,
            footlength=self.cfg.foot_length, #approximate foot length for G1
            use_momentum=False
        )

        self.mass = sum(self.robot.data.default_mass.T)[0]
        
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
        self.old_stance_idx = None

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
        #update phase var
        self._phase_var.update(self._env.sim.current_time)
        # for per env update
        # t = _env.episode_length_buf.unsqueeze(1) * _env.step_dt
        #use old_stance_idx to detect stance switch
        if self.old_stance_idx is None or self.old_stance_idx != self._phase_var.stance_idx:
            if self.old_stance_idx is None:
                self.old_stance_idx = self._phase_var.stance_idx
            # update stance foot pos, ori
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            self.stance_foot_pos_0 = foot_pos_w[:, self._phase_var.stance_idx, :]
            self.stance_foot_ori_quat_0 = foot_ori_w[:, self._phase_var.stance_idx, :]
            self.stance_foot_ori_0 = get_euler_from_quat(foot_ori_w[:, self._phase_var.stance_idx, :])
            # self.swing2stance_foot_pos_0 = _transfer_to_local_frame(
            #     foot_pos_w[:, self._phase_var.swing_idx, :] - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
            # )
            

        self.old_stance_idx = self._phase_var.stance_idx
        return

            
    def update_walking_target(self):
        #given velocity command, update MLIP
        base_vdes = self._env.command_manager.get_command("base_velocity")  # (N,3)

        
        if self.cfg.use_flat_foot:
            self.mask_forward = torch.full((self.num_envs,), False, device=self.device, dtype=torch.bool)
            self.mask_backward = torch.full((self.num_envs,), False, device=self.device, dtype=torch.bool)
            self.mask_flat = torch.full((self.num_envs,), True, device=self.device, dtype=torch.bool)
        else:
            vel_thresh = self.cfg.foot_length/self._phase_var.Tstep
            self.mask_forward = base_vdes[:, 0] > vel_thresh
            self.mask_backward = base_vdes[:, 0] < -vel_thresh
            self.mask_flat = torch.abs(base_vdes[:, 0]) <= vel_thresh
            
        self.mlip.update_desired_walking(base_vdes, self.cfg.y_nom, 
                                         self.mask_forward, self.mask_backward, self.mask_flat)

        self.delta_yaw = base_vdes[:, 2] * self._phase_var.time_in_step
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

        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[:, self._phase_var.stance_idx, :]
        self.stance_foot_ori = get_euler_from_quat(foot_ori_w[:, self._phase_var.stance_idx, :])

        # Convert foot positions to the robot's yaw-aligned local frame

        swing2stance_local = _transfer_to_local_frame(
            foot_pos_w[:, self._phase_var.swing_idx, :] - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )

        # Center of mass to stance foot vector in local frame
        com_w = data.root_com_pos_w
        com2stance_local = _transfer_to_local_frame(com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0)

        # Pelvis orientation (Euler XYZ)
        pelvis_ori = get_euler_from_quat(root_quat)

        # Foot orientations (Euler XYZ)
        swing_foot_ori = get_euler_from_quat(foot_ori_w[:, self._phase_var.swing_idx, :])

        # stance foot orientation
        stance_foot_ori = get_euler_from_quat(foot_ori_w[:, self._phase_var.stance_idx, :])

        # 2. Velocities (world frame)
        com_vel_w = data.root_com_vel_w[:, 0:3]

        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[:, self._phase_var.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[:, self._phase_var.stance_idx, :]
        # Convert velocities to local frame

        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)

        pelvis_omega_local = data.root_ang_vel_b

        foot_lin_vel_local_swing = _transfer_to_local_frame(
            foot_lin_vel_w[:, self._phase_var.swing_idx, :], self.stance_foot_ori_quat_0
        )

        foot_ang_vel_local_swing = quat_apply(
            quat_inv(foot_ori_w[:, self._phase_var.swing_idx, :]), foot_ang_vel_w[:, self._phase_var.swing_idx, :]
        )

        stance_foot_ang_vel_local_swing = quat_apply(
            quat_inv(foot_ori_w[:, self._phase_var.stance_idx, :]), foot_ang_vel_w[:, self._phase_var.stance_idx, :]
        )

        swing2stance_vel = foot_lin_vel_local_swing

        upper_body_joint_pos = self.robot.data.joint_pos[:, self.upper_body_joint_idx]
        upper_body_joint_vel = self.robot.data.joint_vel[:, self.upper_body_joint_idx]

        # self.y_act = torch.cat(
        #     [com2stance_local, pelvis_ori, swing2stance_local, swing_foot_ori, stance_foot_ori[:, 1].unsqueeze(-1), upper_body_joint_pos], dim=-1
        # )

        # self.dy_act = torch.cat(
        #     [com_vel_local, pelvis_omega_local, swing2stance_vel, foot_ang_vel_local_swing, stance_foot_ang_vel_local_swing[:, 1].unsqueeze(-1), upper_body_joint_vel],
        #     dim=-1,
        # )
        
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
        
        
        # Get desired foot placements and CoM trajectory from MLIP in target yaw frame
        Ux,  Uy = self.mlip.get_desired_foot_placement(self._phase_var.stance_idx) #Shape: (N,)
        com_x, com_dx, com_y, com_dy = self.mlip.get_desired_com_state(self._phase_var.stance_idx, self._phase_var.time_in_step) #Shape: (N,)

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
        self.foot_target = swing_foot_target_yaw_adjusted # Shape: (N,3)
        #clamp swing foot y to avoid too large step
        sign_swy = torch.sign(swing_foot_target_yaw_adjusted[:, 1])
        self.foot_target[:, 1] = torch.clamp(
            torch.abs(swing_foot_target_yaw_adjusted[:, 1]),
            min=self.cfg.foot_target_range_y[0],
            max=self.cfg.foot_target_range_y[1],
        ) * sign_swy

        

        # Create horizontal control points with batch dimension
        horizontal_control_points = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], device=self.device)

        bht = bezier_deg(0, self._phase_var.phase, self._phase_var.T_ss, horizontal_control_points, 4)
        dbht = bezier_deg(1, self._phase_var.phase, self._phase_var.T_ss, horizontal_control_points, 4)
        # Horizontal X and Y (linear interpolation)
        p_swing_x = ((1 - bht) * (-swing_foot_target_yaw_adjusted[:, 0]) + bht * swing_foot_target_yaw_adjusted[:, 0]) #Shape: (N,)
        v_swing_x = ((-dbht) * (-swing_foot_target_yaw_adjusted[:, 0]) + dbht * swing_foot_target_yaw_adjusted[:, 0])  #Shape: (N,)
        #todo: for now, since not going sideway, just assume swy_0 = swy_T
        y0 = sign_swy * self.cfg.y_nom
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
        if isinstance(self._phase_var.phase, float):
            phase_tensor = torch.full((N,), self._phase_var.phase, device=self.device)
            T_tensor = torch.full((N,), self._phase_var.T_ss, device=self.device)
        else:
            phase_tensor = self._phase_var.phase
            T_tensor = self._phase_var.T_ss
        p_swing_z = bezier_deg(0, phase_tensor, T_tensor, control_v, degree_v)
        v_swing_z = bezier_deg(1, phase_tensor, T_tensor, control_v, degree_v)

        # Combine to get full swing foot position and velocity
        swing_foot_pos = torch.stack([p_swing_x, p_swing_y, p_swing_z], dim=-1)  # Shape: (N,3)
        swing_foot_vel = torch.stack([v_swing_x, v_swing_y, v_swing_z], dim=-1)  # Shape: (N,3)

        
        
        
        stance_foot_pitch_angle = torch.full((N, 1), 0.0, device=self.device)
        stance_foot_pitch_vel = torch.full((N, 1), 0.0, device=self.device)
        if self._phase_var.domain == "FA":
            pass
        elif self._phase_var.domain == "UA" and not self.cfg.use_flat_foot:
            bht_ua = bezier_deg(0, self._phase_var.phase_ua, self._phase_var.T_ua, horizontal_control_points, 4)
            dbht_ua = bezier_deg(1, self._phase_var.phase_ua, self._phase_var.T_ua, horizontal_control_points, 4)
            #heel to toe
            stance_foot_pitch_angle[self.mask_forward] = bht_ua * self.cfg.foot_pitch_ref
            stance_foot_pitch_vel[self.mask_forward] = dbht_ua * self.cfg.foot_pitch_ref
            swingfoot_eulxyz[self.mask_forward, 1] = -bht_ua * self.cfg.foot_pitch_ref
            swingfoot_eulxyz_dot[self.mask_forward, 1] = -dbht_ua * self.cfg.foot_pitch_ref
            # toe to heel
            stance_foot_pitch_angle[self.mask_backward] = -bht_ua * self.cfg.foot_pitch_ref
            stance_foot_pitch_vel[self.mask_backward] = -dbht_ua * self.cfg.foot_pitch_ref
            swingfoot_eulxyz[self.mask_backward, 1] = bht_ua * self.cfg.foot_pitch_ref
            swingfoot_eulxyz_dot[self.mask_backward, 1] = dbht_ua * self.cfg.foot_pitch_ref
        elif self._phase_var.domain == "OA" and not self.cfg.use_flat_foot:
            bht_oa = bezier_deg(0, self._phase_var.phase_oa, self._phase_var.T_oa, horizontal_control_points, 4)
            dbht_oa = bezier_deg(1, self._phase_var.phase_oa, self._phase_var.T_oa, horizontal_control_points, 4)
            stance_foot_pitch_angle[self.mask_forward] = self.cfg.foot_pitch_ref #todo: or smooth from y0
            swingfoot_eulxyz[self.mask_forward, 1] = (torch.tensor(1.0, device=self.device) - bht_oa) * (-self.cfg.foot_pitch_ref)
            swingfoot_eulxyz_dot[self.mask_forward, 1] = dbht_oa * self.cfg.foot_pitch_ref
            stance_foot_pitch_angle[self.mask_backward] = -self.cfg.foot_pitch_ref
            swingfoot_eulxyz[self.mask_backward, 1] = (torch.tensor(1.0, device=self.device) - bht_oa) * self.cfg.foot_pitch_ref
            swingfoot_eulxyz_dot[self.mask_backward, 1] = -dbht_oa * self.cfg.foot_pitch_ref
        else:
            pass
            
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
        # # Debug prints - show full tensors
        # with torch.no_grad():
        #     torch.set_printoptions(profile="full", linewidth=1500, precision=4, sci_mode=False)
        #     print("=" * 80)
        #     print("MLIP DEBUG: y_out (positions/orientations):")
        #     print(self.y_out)
        #     print("MLIP DEBUG: dy_out (velocities/angular velocities):")
        #     print(self.dy_out)
        #     print("=" * 80)
        # return

        
    def generate_upper_body_ref(self):
        # phase: [B]
        forward_vel = self._env.command_manager.get_command("base_velocity")[:, 0]

        Tswing = self._phase_var.Tstep
        #0.0125
        tp = (self._env.sim.current_time % (2 * Tswing)) / (2 * Tswing)
        #0.0785
        phase = 2 * torch.pi * tp

        # unpack your cfg scalars
        sh_pitch0, sh_roll0, sh_yaw0 = self.cfg.shoulder_ref
        elb0 = self.cfg.elbow_ref
        waist_yaw0 = self.cfg.waist_yaw_ref

        # build every amp as a [B] tensor
        sh_pitch_amp = sh_pitch0 * forward_vel  # [B]
        sh_roll_amp = sh_roll0 * torch.ones_like(forward_vel)
        sh_yaw_amp = sh_yaw0 * torch.ones_like(forward_vel)
        elb_amp = elb0 * forward_vel
        waist_amp = waist_yaw0 * torch.ones_like(forward_vel)

        # stack into [B,9]
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

        # your sign & offset stay [9] each
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

        # joint offsets: [B,9]
        joint_offset = self.robot.data.default_joint_pos[:, self.upper_body_joint_idx]

        # refs: everything now broadcast to [B,9]
        offset = offset.unsqueeze(0).expand(self.num_envs, -1)
        ref = amp * sign * torch.sin(phase + offset) + joint_offset

        # velocity
        dphase_dt = 2 * torch.pi / (2 * Tswing)  # scalar
        ref_dot = amp * sign * torch.cos(phase + offset) * dphase_dt

        return ref, ref_dot


