import torch
import math
import numpy as np
from abc import ABC, abstractmethod

from isaaclab.managers import CommandTerm

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.clf import CLF

from typing import TYPE_CHECKING
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import JointTrajectoryConfig, get_euler_from_quat
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.ee_traj import EndEffectorTrajectoryConfig #, EndEffectorTracker
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi,quat_from_euler_xyz

if TYPE_CHECKING:
    from ..cmd_cfg import HZDCommandCfg


class HZDCommandTerm(CommandTerm, ABC):
    """Abstract base class for HZD (Hybrid Zero Dynamics) command terms."""
    
    def __init__(self, cfg: "HZDCommandCfg", env):
        super().__init__(cfg, env)
        
        self.env = env
        self.robot = env.scene[cfg.asset_name]
        self.debug_vis = cfg.debug_vis
        
        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        self.hip_yaw_idx, _ = self.robot.find_joints(".*_hip_yaw_.*")
        self.metrics = {}
        
        self.mass = sum(self.robot.data.default_mass.T)[0]
        
        
        
        self.v = torch.zeros((self.num_envs), device=self.device)
        self.stance_idx = None
        
        self.y_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.y_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.yaw_output_idx = []


    @property
    def command(self):
        return self.y_out


    def initiate_clf(self):
        self.clf = CLF(
            self.cfg.num_outputs, self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            Q_weights=np.array(self.cfg.Q_weights),
            R_weights=np.array(self.cfg.R_weights),
            device=self.device
        )

    def _resample_command(self, env_ids):
        self._update_command()
        return

    def _update_metrics(self):
        # Base metrics that are common to all HZD commands
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

    def update_stance_swing_idx(self):
        """Update stance and swing indices based on phase."""
        Tswing = self._get_swing_period()

        tp = (self.env.sim.current_time % (2 * Tswing)) / (2 * Tswing)
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + Tswing), device=self.env.device)

        new_stance_idx = int(0.5 + 0.5 * torch.sign(phi_c))
        self.swing_idx = 1 - new_stance_idx
        
        if self.stance_idx is None or new_stance_idx != self.stance_idx:
            if self.stance_idx is None:
                self.stance_idx = new_stance_idx

            # Update stance foot pos, ori
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            self.stance_foot_pos_0 = foot_pos_w[:, new_stance_idx, :]
            self.stance_foot_ori_quat_0 = foot_ori_w[:, new_stance_idx, :]
            self.stance_foot_ori_0 = get_euler_from_quat(foot_ori_w[:, new_stance_idx, :])
       
        self.stance_idx = new_stance_idx

        if tp < 0.5:
            self.phase_var = 2 * tp
        else:
            self.phase_var = 2 * tp - 1
        self.cur_swing_time = self.phase_var * Tswing

    def _update_command(self):
        """Update the command by generating reference and computing CLF."""
        self.update_stance_swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, self.yaw_output_idx)
        self.vdot = vdot
        self.v = vcur

    @abstractmethod
    def _get_leg_period(self) -> float:
        """Get the swing period for phase calculation."""
        pass

    @abstractmethod
    def generate_reference_trajectory(self):
        """Generate reference trajectory. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_actual_state(self):
        """Get actual state. Must be implemented by subclasses."""
        pass

    def get_stance_foot_pose(self):
        """Get stance foot pose data similar to JointTrajectoryConfig.get_stance_foot_pose."""
        data = self.robot.data
        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]

        # Store raw foot positions
        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]
        self.stance_foot_pos = foot_pos_w[:, self.stance_idx, :]
        self.stance_foot_ori = get_euler_from_quat(foot_ori_w[:, self.stance_idx, :])
        self.stance_foot_vel = foot_lin_vel_w[:, self.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[:, self.stance_idx, :]

