import torch
import math
import yaml
from isaaclab.utils import configclass
import numpy as np

from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
from robot_rl.assets.robots.g1_21j import build_relabel_matrix
# from robot_rl.assets.robots.exo_cfg import JointTrajectoryConfig

from .hzd_cmd import bezier_deg, JointTrajectoryConfig
from .clf import CLF

# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz

from typing import TYPE_CHECKING




if TYPE_CHECKING:
    from ..cmd_cfg import HZDCommandCfg


class HZDCommandTerm(CommandTerm):
    def __init__(self, cfg: "HZDCommandCfg", env):
        super().__init__(cfg, env)
       
        self.env = env
        self.robot = env.scene[cfg.asset_name]

        self.debug_vis = cfg.debug_vis


        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        self.hip_yaw_idx,_ = self.robot.find_joints(".*_hip_yaw_.*")
        self.metrics = {}
     
        # self.com_z = torch.ones((self.num_envs), device=self.device)*self.z0

        # load joint trajectory config from YAML
        yaml_path = "source/robot_rl/robot_rl/assets/robots/single_support_config_solution.yaml"
        self.jt_config = JointTrajectoryConfig()
        self.jt_config.load_from_yaml(yaml_path, self.robot)
        self.T = self.env.cfg.commands.step_period.period_range[0]/2
        
        right_jt_coeffs = self.jt_config.joint_trajectories
    
        left_jt_coeffs = self.jt_config.remap_jt_symmetric()
   

        left_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)
        right_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)

        
        for key in self.jt_config.joint_trajectories.keys():
            joint_idx = self.robot.find_joints(key)[0]
            right_coeffs[joint_idx] = torch.tensor(right_jt_coeffs[key], device=self.device)
            left_coeffs[joint_idx] = torch.tensor(left_jt_coeffs[key], device=self.device)

        self.right_coeffs = right_coeffs
        self.left_coeffs = left_coeffs


        self.mass = sum(self.robot.data.default_mass.T)[0]
        
        self.clf = CLF(
           cfg.num_outputs, self.env.cfg.sim.dt,
           batch_size=self.num_envs,
            Q_weights=np.array(cfg.Q_weights),
            R_weights=np.array(cfg.R_weights),
            device=self.device
        )
        
        self.v = torch.zeros((self.num_envs), device=self.device)
        self.stance_idx = None

        self.y_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.y_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)


    @property
    def command(self):
        return self.y_out
    

    def _resample_command(self, env_ids):
        self._update_command()
        # Do nothing here
        # device = self.env.command_manager.get_command("base_velocity").device
        
        return
    
    def _update_metrics(self):
        # Update metrics using actual joint names from the YAML file
        for i, joint_name in enumerate(self.robot.joint_names):
            error_key = f"error_{joint_name}"
            self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot



    def update_Stance_Swing_idx(self):
        Tswing = self.T 
        tp = (self.env.sim.current_time % (2 * Tswing)) / (2 * Tswing)  
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)

        new_stance_idx = int(0.5 - 0.5 * torch.sign(phi_c))
        self.swing_idx = 1 - new_stance_idx
        
        if self.stance_idx is None or new_stance_idx != self.stance_idx:
            if self.stance_idx is None:
                self.stance_idx = new_stance_idx

            # update stance foot pos, ori
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            self.stance_foot_pos_0 = foot_pos_w[:, new_stance_idx, :]
            self.stance_foot_ori_quat_0 = foot_ori_w[:, new_stance_idx, :]
            self.stance_foot_ori_0 = self.get_euler_from_quat(foot_ori_w[:, new_stance_idx, :])
       
        self.stance_idx = new_stance_idx

        if tp < 0.5:
            self.phase_var = 2 * tp
        else:
            self.phase_var = 2 * tp - 1
        self.cur_swing_time = self.phase_var * Tswing
        

    def generate_reference_trajectory(self):
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)
        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)

        if self.stance_idx == 1:
            ctrl_points = self.right_coeffs
        else:
            ctrl_points = self.left_coeffs
     
        

        phase_var_tensor = torch.full((N,), self.phase_var, device=self.device)
        des_jt_pos = bezier_deg(
            0, phase_var_tensor, T, ctrl_points, torch.tensor(self.cfg.bez_deg, device=self.device)
        )
        
        des_jt_vel = bezier_deg(1, phase_var_tensor, T, ctrl_points, self.cfg.bez_deg)

        yaw_offset = base_velocity[:, 2] 
        des_jt_pos[:, self.hip_yaw_idx[self.stance_idx]] += yaw_offset


        self.y_out = des_jt_pos
        self.dy_out = des_jt_vel

        

    def nom_bezier_curve(self, control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates multiple Bezier curves at given time points using the control points.

        Args:
            control_points (torch.Tensor): Tensor of shape [num_curves, num_control_points, dim]
                                        representing the control points for each curve.
            t (torch.Tensor): Tensor of shape [num_curves] representing the time point at which to evaluate each Bezier curve.
        
        Returns:
            torch.Tensor: Tensor of shape [num_curves, dim] representing the evaluated points on the Bezier curves.
        """
        dim, num_control_points = control_points.shape
        n = num_control_points - 1

        num_curves = t.shape[0]
        # Initialize the result tensor to zeros: shape [num_curves, dim]
        curve_points = torch.zeros((num_curves, dim), dtype=control_points.dtype, device=control_points.device)
        
        for i in range(n + 1):
            binomial_coeff = math.comb(n, i)  # Binomial coefficient
            bernstein_poly = binomial_coeff * (t ** i) * ((1 - t) ** (n - i))  # Bernstein basis polynomial

            # Add the contribution of each control point to each curve
            curve_points += bernstein_poly.unsqueeze(-1) * control_points[:, i]

        return curve_points

    def get_euler_from_quat(self, quat):

        euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
        euler_x = wrap_to_pi(euler_x)
        euler_y = wrap_to_pi(euler_y)
        euler_z = wrap_to_pi(euler_z)
        return torch.stack([euler_x, euler_y, euler_z], dim=-1)

    def get_actual_state(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        data = self.robot.data


        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]

        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[:, self.stance_idx, :]
        self.stance_foot_ori = self.get_euler_from_quat(foot_ori_w[:, self.stance_idx, :])

        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[:, self.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[:, self.stance_idx, :]

        jt_pos = data.joint_pos
        jt_vel = data.joint_vel
        # 4. Assemble state vectors
        self.y_act = jt_pos

        self.dy_act = jt_vel

    def _update_command(self):
        
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        # how to handle for the first step?
        # i.e. v is not defined
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, [])
        self.vdot = vdot
        self.v = vcur
       
          
