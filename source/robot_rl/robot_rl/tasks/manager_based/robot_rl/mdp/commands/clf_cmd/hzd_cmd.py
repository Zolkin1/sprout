import torch
import math
import numpy as np
from abc import ABC, abstractmethod

from isaaclab.managers import CommandTerm

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.clf import CLF

from typing import TYPE_CHECKING
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import JointTrajectoryConfig, get_euler_from_quat
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.ee_traj import EndEffectorTrajectoryConfig, EndEffectorTracker


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

    @property
    def command(self):
        return self.y_out

    def _resample_command(self, env_ids):
        self._update_command()
        return

    def _update_metrics(self):
        # Update metrics using actual joint names from the robot
        for i, joint_name in enumerate(self.robot.joint_names):
            error_key = f"error_{joint_name}"
            self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

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
        
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, [])
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


class EndEffectorTrajectoryHZDCommandTerm(HZDCommandTerm):
    """HZD command term that uses end effector trajectory references."""
    
    def __init__(self, cfg: "HZDCommandCfg", env):
        super().__init__(cfg, env)
        
        # Load end effector trajectory config from YAML
        self.ee_config = EndEffectorTrajectoryConfig()
        
        # Initialize end effector tracker
        self.ee_tracker = EndEffectorTracker(
            self.ee_config.constraint_specs, 
            env.scene.env_regex_ns
        )
        
        # Store constraint specifications for easy access
        self.constraint_specs = self.ee_config.constraint_specs
        
        # Initialize output tensors for end effector references
        self.ee_ref_pos = torch.zeros((self.num_envs, 0), device=self.device)
        self.ee_ref_ori = torch.zeros((self.num_envs, 0), device=self.device)
        self.ee_act_pos = torch.zeros((self.num_envs, 0), device=self.device)
        self.ee_act_ori = torch.zeros((self.num_envs, 0), device=self.device)

    def _get_swing_period(self) -> float:
        """Get the swing period from the end effector configuration."""
        return self.ee_config.T

    def generate_reference_trajectory(self):
        """Generate reference trajectory using end effector trajectories."""
        # Get base velocity for yaw offset
        base_velocity = self.env.command_manager.get_command("base_velocity")
        N = base_velocity.shape[0]
        T = torch.full((N,), self.ee_config.T, dtype=torch.float32, device=self.device)
        
        # Initialize reference tensors
        ref_positions = []
        ref_orientations = []
        
        # Generate references for each constraint specification
        for spec in self.constraint_specs:
            if "frame" not in spec:
                continue
                
            frame_name = spec["frame"]
            constraint_type = spec["type"]
            
            if constraint_type in ["ee_pos", "com_pos"]:
                # Position constraint
                ref_pos = self.ee_config.evaluate_bezier_trajectory(
                    frame_name, constraint_type, 
                    torch.full((N,), self.phase_var, device=self.device), 
                    T, self.cfg.bez_deg
                )
                ref_positions.append(ref_pos)
                
            elif constraint_type in ["ee_ori"]:
                # Orientation constraint
                ref_ori = self.ee_config.evaluate_bezier_trajectory(
                    frame_name, constraint_type,
                    torch.full((N,), self.phase_var, device=self.device),
                    T, self.cfg.bez_deg
                )
                ref_orientations.append(ref_ori)
        
        # Concatenate all references
        if ref_positions:
            self.ee_ref_pos = torch.cat(ref_positions, dim=1)
        if ref_orientations:
            self.ee_ref_ori = torch.cat(ref_orientations, dim=1)
        
        # For now, we'll use zero joint references as placeholders
        # In a full implementation, you would convert EE references to joint references
        self.y_out = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        self.dy_out = torch.zeros((N, self.cfg.num_outputs), device=self.device)

    def get_actual_state(self):
        """Get actual state for end effector trajectories."""
        # Get actual joint positions and velocities
        jt_pos = self.robot.data.joint_pos
        jt_vel = self.robot.data.joint_vel
        self.y_act = jt_pos
        self.dy_act = jt_vel
        
        # Get stance foot pose data
        self.get_stance_foot_pose()
        
        # Get actual end effector poses
        act_positions = []
        act_orientations = []
        
        for spec in self.constraint_specs:
            if "frame" not in spec:
                continue
                
            frame_name = spec["frame"]
            constraint_type = spec["type"]
            
            try:
                pos, ori = self.ee_tracker.get_pose(frame_name)
                
                if constraint_type in ["ee_pos", "com_pos"]:
                    act_positions.append(pos.unsqueeze(0).expand(self.num_envs, -1))
                elif constraint_type in ["ee_ori"]:
                    act_orientations.append(ori.unsqueeze(0).expand(self.num_envs, -1))
            except Exception as e:
                print(f"Warning: Could not get pose for frame {frame_name}: {e}")
                # Add zero tensors as fallback
                if constraint_type in ["ee_pos", "com_pos"]:
                    act_positions.append(torch.zeros((self.num_envs, 3), device=self.device))
                elif constraint_type in ["ee_ori"]:
                    act_orientations.append(torch.zeros((self.num_envs, 3), device=self.device))
        
        # Concatenate all actual poses
        if act_positions:
            self.ee_act_pos = torch.cat(act_positions, dim=1)
        if act_orientations:
            self.ee_act_ori = torch.cat(act_orientations, dim=1)


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
       
          
