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
        self.yaw_output_idx = []

    @property
    def command(self):
        return self.y_out

    def _resample_command(self, env_ids):
        self._update_command()
        return

    def _update_metrics(self):
        # Base metrics that are common to all HZD commands
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

    def update_Stance_Swing_idx(self):
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
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, self.yaw_output_idx)
        self.vdot = vdot
        self.v = vcur

    @abstractmethod
    def _get_swing_period(self) -> float:
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



class JointTrajectoryHZDCommandTerm(HZDCommandTerm):
    """HZD command term that uses joint trajectory references."""
    
    def __init__(self, cfg: "HZDCommandCfg", env):
        super().__init__(cfg, env)
        
        # Load joint trajectory config from YAML
        self.ref_config = JointTrajectoryConfig()
        self.ref_config.reorder_and_remap_jt(cfg, self.robot, self.device)

    def _get_swing_period(self) -> float:
        """Get the swing period from the reference configuration."""
        return self.ref_config.T

    def generate_reference_trajectory(self):
        """Generate reference trajectory using joint trajectory config."""
        ref_pos, ref_vel = self.ref_config.get_ref_traj(self)
        self.y_out = ref_pos
        self.dy_out = ref_vel

    def get_actual_state(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        self.ref_config.get_stance_foot_pose(self)
        jt_pos, jt_vel = self.ref_config.get_actul_traj(self)
        self.y_act = jt_pos
        self.dy_act = jt_vel

    def _update_metrics(self):
        """Update metrics specific to joint trajectory tracking."""
        # Call parent method for base metrics
        super()._update_metrics()
        
        # Update metrics using actual joint names from the robot
        for i, joint_name in enumerate(self.robot.joint_names):
            error_key = f"error_{joint_name}"
            self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])


class EndEffectorTrajectoryHZDCommandTerm(HZDCommandTerm):
    """HZD command term that uses end effector trajectory references."""
    
    def __init__(self, cfg: "HZDCommandCfg", env):
        super().__init__(cfg, env)
        
        # Load end effector trajectory config from YAML
        self.ee_config = EndEffectorTrajectoryConfig(yaml_path=cfg.yaml_path)
        

        # also need to remap root state
        root_quat = torch.tensor(self.ee_config.init_root_state[3:], dtype=torch.float32,device=self.device)
        init_root_state_eul = get_euler_from_quat(root_quat.unsqueeze(0)).squeeze(0)
        init_root_state_eul[0] = -init_root_state_eul[0]
        init_root_state_eul[2] = -init_root_state_eul[2]
        init_root_state_quat = quat_from_euler_xyz(init_root_state_eul[0].unsqueeze(0),init_root_state_eul[1].unsqueeze(0),init_root_state_eul[2].unsqueeze(0))
        
        self.init_root_state = torch.tensor(self.ee_config.init_root_state, dtype=torch.float32,device=self.device)

        self.init_root_state[3:] = init_root_state_quat
        self.init_root_vel = torch.tensor(self.ee_config.init_root_vel, dtype=torch.float32,device=self.device)
        

        #nered to reorder the joint based on joitn order

        # Initialize end effector tracker
        self.ee_tracker = EndEffectorTracker(
            self.ee_config.constraint_specs, 
            env.scene
        )

        
        self.waist_joint_idx, _ = self.robot.find_joints(".*waist_yaw.*")
        self.foot_yaw_output_idx = 11
        self.ori_idx_list = [
            [3,4,5],
            [9,10,11],
        ]

        num_jt = len(self.ee_config.joint_order)
        init_joint_pos = torch.zeros((num_jt), dtype=torch.float32,device=self.device)
        init_joint_vel = torch.zeros((num_jt), dtype=torch.float32,device=self.device)

        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import build_relabel_matrix
        
        R = build_relabel_matrix()
        init_joint_pos_relabel = R @ self.ee_config.init_joint_pos
        init_joint_vel_relabel = R @ self.ee_config.init_joint_vel

        for i in range(num_jt):
            joint_name = self.ee_config.joint_order[i]
            new_joint_idx = self.robot.find_joints(joint_name)[0]
            init_joint_pos[new_joint_idx] = init_joint_pos_relabel[i]
            init_joint_vel[new_joint_idx] = init_joint_vel_relabel[i]
        
        
        self.init_joint_pos = init_joint_pos
        self.init_joint_vel = init_joint_vel

        self.tp = torch.zeros((self.num_envs), device=self.device)

        self.T = torch.full((self.num_envs,), self.ee_config.T, device=self.device)
        
        # Reorder and remap end effector coefficients
        self.ee_config.reorder_and_remap_ee(cfg, self.ee_tracker, self.device)
        self.yaw_output_idx = [5,11,16,20]

    def _get_swing_period(self) -> float:
        """Get the swing period from the end effector configuration."""
        return self.ee_config.T

    def generate_reference_trajectory(self):
        """Generate reference trajectory using end effector trajectories."""
        ref_pos, ref_vel = self.ee_config.get_ref_traj(self)
        self.y_out = ref_pos
        self.dy_out = ref_vel

    def get_actual_state(self):
        """Get actual state for end effector trajectories."""
        # Get stance foot pose data
        self.get_stance_foot_pose()
        
        # Get actual trajectory from end effector tracker
        act_pos, act_vel = self.ee_config.get_actual_traj(self)
        self.y_act = act_pos
        self.dy_act = act_vel
        # import pdb; pdb.set_trace()

    def _update_metrics(self):
        """Update metrics specific to end effector trajectory tracking."""
        # Call parent method for base metrics
        super()._update_metrics()
        
        # Update metrics using pre-generated axis names
        for axis_info in self.ee_config.axis_names:
            error_key = axis_info['name']
            index = axis_info['index']
            self.metrics[error_key] = torch.abs(
                self.y_out[:, index] - 
                self.y_act[:, index]
            )

    def get_stance_foot_pose(self):
            """Get stance foot pose data similar to JointTrajectoryConfig.get_stance_foot_pose."""
            stance_foot_frame = "left_foot_middle" if self.stance_idx == 0 else "right_foot_middle"
            stance_foot_pos, stance_foot_ori,stance_foot_quat = self.ee_tracker.get_pose(stance_foot_frame) 
            self.stance_foot_pos = stance_foot_pos
            self.stance_foot_ori = stance_foot_ori
            stance_foot_vel, stance_foot_ang_vel = self.ee_tracker.get_velocity(stance_foot_frame, self.robot.data)
            self.stance_foot_vel = stance_foot_vel
            self.stance_foot_ang_vel = stance_foot_ang_vel

    def update_Stance_Swing_idx(self):
            """Update stance and swing indices based on phase."""
            Tswing = self._get_swing_period()

            tp = (self.env.sim.current_time % (2 * Tswing)) / (2 * Tswing)
            phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + Tswing), device=self.env.device)

            new_stance_idx = int(0.5 - 0.5 * torch.sign(phi_c))
            self.swing_idx = 1 - new_stance_idx
            
            if self.stance_idx is None or new_stance_idx != self.stance_idx:
                if self.stance_idx is None:
                    self.stance_idx = new_stance_idx


                stance_foot_frame = "left_foot_middle" if new_stance_idx == 0 else "right_foot_middle"
                stance_foot_pos, stance_foot_ori,stance_foot_quat = self.ee_tracker.get_pose(stance_foot_frame) 
                
             
                self.stance_foot_pos_0 = stance_foot_pos
                self.stance_foot_ori_quat_0 = stance_foot_quat
                self.stance_foot_ori_0 = stance_foot_ori
        
            self.stance_idx = new_stance_idx

            if tp < 0.5:
                self.phase_var = 2 * tp
            else:
                self.phase_var = 2 * tp - 1
            self.cur_swing_time = self.phase_var * Tswing
            self.tp = torch.full((self.num_envs,), tp, device=self.device)

def create_hzd_command_term(cfg, env):
    """
    Factory function to create the appropriate HZD command term based on configuration.
    
    Args:
        cfg: Configuration object (JointTrajectoryHZDCommandCfg, BaseTrajectoryHZDCommandCfg, etc.)
        env: Environment object
        
    Returns:
        Appropriate HZD command term instance
    """
    # The configuration's class_type will determine which command term to create
    return cfg.class_type(cfg, env)
       
          
