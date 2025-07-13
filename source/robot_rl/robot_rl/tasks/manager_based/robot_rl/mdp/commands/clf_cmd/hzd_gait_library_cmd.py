import torch
import math
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cmd import HZDCommandTerm
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.gait_library_traj import (
    GaitLibraryEndEffectorConfig, GaitLibraryJointConfig, StairGaitLibraryTrajectoryConfig
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import get_euler_from_quat
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.ee_traj import EndEffectorTracker

class GaitLibraryHZDCommandTerm(HZDCommandTerm):
    """HZD command term that uses a gait library with velocity-based selection."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        # Initialize gait library based on trajectory type
        if hasattr(cfg, 'gait_library_path'):
            config_name = getattr(cfg, 'config_name', 'single_support')
            
            # Check if it's a stair gait library (height-based) or regular gait library (velocity-based)
            if hasattr(cfg, 'gait_height_ranges'):
                # Stair gait library
                self.gait_config = StairGaitLibraryTrajectoryConfig(
                    cfg.gait_library_path, 
                    cfg.gait_height_ranges,
                    cfg.trajectory_type,
                    config_name
                )
            elif hasattr(cfg, 'gait_velocity_ranges'):
                # Regular gait library
                if cfg.trajectory_type == "end_effector":
                    self.gait_config = GaitLibraryEndEffectorConfig(
                        cfg.gait_library_path, 
                        cfg.gait_velocity_ranges,
                        config_name
                    )
                else:
                    self.gait_config = GaitLibraryJointConfig(
                        cfg.gait_library_path, 
                        cfg.gait_velocity_ranges,
                        config_name
                    )
            else:
                raise ValueError("Gait library configuration missing: either gait_height_ranges or gait_velocity_ranges required")
            
            # Initialize end effector tracker if needed
            if cfg.trajectory_type == "end_effector":
                self.ee_tracker = EndEffectorTracker(
                    self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]].constraint_specs,
                    env.scene
                )
        else:
            raise ValueError("Gait library configuration missing: gait_library_path required")
        
        # Set up trajectory-specific attributes
        if cfg.trajectory_type == "end_effector":
            self.waist_joint_idx, _ = self.robot.find_joints(".*waist_yaw.*")
            self.joint_idx_list = self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]].get_joint_idx_list(self)
            self.foot_yaw_output_idx = 11
            self.ori_idx_list = [[3, 4, 5], [9, 10, 11]]
            self.yaw_output_idx = [5, 11, 16, 20]
        
        # Reorder and remap coefficients
        self.gait_config.reorder_and_remap(cfg, self.device)
        self.tp = torch.zeros((self.env.num_envs,), device=self.device)

    def _get_swing_period(self) -> float:
        """Get the swing period from the gait configuration."""
        return self.gait_config.T

    def generate_reference_trajectory(self):
        """Generate reference trajectory using gait library."""
        ref_pos, ref_vel = self.gait_config.get_ref_traj(self)
        self.y_out = ref_pos
        self.dy_out = ref_vel

    def get_actual_state(self):
        """Get actual state for gait library trajectories."""
        # Get stance foot pose data
        self.get_stance_foot_pose()
        
        # Get actual trajectory from gait library
        act_pos, act_vel = self.gait_config.get_actual_traj(self)
        self.y_act = act_pos
        self.dy_act = act_vel

    def _update_metrics(self):
        """Update metrics specific to gait library tracking."""
        # Call parent method for base metrics
        super()._update_metrics()
        
        # Update metrics based on trajectory type
        if hasattr(self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]], 'axis_names'):
            # End-effector trajectories
            for axis_info in self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]].axis_names:
                error_key = axis_info['name']
                index = axis_info['index']
                self.metrics[error_key] = torch.abs(
                    self.y_out[:, index] - 
                    self.y_act[:, index]
                )
        else:
            # Joint trajectories
            for i, joint_name in enumerate(self.robot.joint_names):
                error_key = f"error_{joint_name}"
                self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

    def get_stance_foot_pose(self):
        """Get stance foot pose data similar to JointTrajectoryConfig.get_stance_foot_pose."""
        # TODO: Remove
        # stance_foot_frame = "left_foot_middle" if self.stance_idx == 0 else "right_foot_middle"
        # stance_foot_pos, stance_foot_ori,stance_foot_quat = self.ee_tracker.get_pose(stance_foot_frame)
        # self.stance_foot_pos = stance_foot_pos
        # self.stance_foot_ori = stance_foot_ori
        # stance_foot_vel, stance_foot_ang_vel = self.ee_tracker.get_velocity(stance_foot_frame, self.robot.data)
        # self.stance_foot_vel = stance_foot_vel
        # self.stance_foot_ang_vel = stance_foot_ang_vel

        stance_foot_idx = self.feet_bodies_idx[1] if self.stance_idx == 0 else self.feet_bodies_idx[0]
        # TODO: Verify
        self.stance_foot_pos = self.robot.data.body_pos_w[:, stance_foot_idx, :]
        stance_foot_quat = self.robot.data.body_quat_w[:, stance_foot_idx, :]
        self.stance_foot_ori = get_euler_from_quat(stance_foot_quat)

        self.stance_foot_vel = self.robot.data.body_lin_vel_w[:, stance_foot_idx, :]
        self.stance_foot_ang_vel = self.robot.data.body_ang_vel_w[:, stance_foot_idx, :]


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

            # Update stance foot pose based on trajectory type
            if hasattr(self, 'ee_tracker'):
                # End-effector trajectories
                stance_foot_frame = "left_foot_middle" if new_stance_idx == 0 else "right_foot_middle"
                stance_foot_pos, stance_foot_ori, stance_foot_quat = self.ee_tracker.get_pose(stance_foot_frame) 
                
                self.stance_foot_pos_0 = stance_foot_pos
                self.stance_foot_ori_quat_0 = stance_foot_quat
                self.stance_foot_ori_0 = stance_foot_ori
            else:
                # Joint trajectories
                foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
                foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
                self.stance_foot_pos_0 = foot_pos_w[:, new_stance_idx, :]
                self.stance_foot_ori_quat_0 = foot_ori_w[:, new_stance_idx, :]
                self.stance_foot_ori_0 = get_euler_from_quat(foot_ori_w[:, new_stance_idx, :])
       
        just_reset = (self.env.episode_length_buf == 0)
        # Handle episode reset: force re-alignment of stance foot
        if just_reset.any():
          if hasattr(self, 'ee_tracker'):
               # End-effector mode: assumes single-env, so just do once
               stance_foot_frame = "left_foot_middle" if new_stance_idx == 0 else "right_foot_middle"
               stance_foot_pos, stance_foot_ori, stance_foot_quat = self.ee_tracker.get_pose(stance_foot_frame)

               self.stance_foot_pos_0[just_reset] = stance_foot_pos[just_reset]
               self.stance_foot_ori_quat_0[just_reset] = stance_foot_quat[just_reset]
               self.stance_foot_ori_0[just_reset] = stance_foot_ori[just_reset]
          else:
               # Joint trajectory version: selective batched update
               foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]  # [N, 2, 3]
               foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]  # [N, 2, 4]
               idx = new_stance_idx.view(-1, 1, 1)  # [N,1,1]

               updated_pos = foot_pos_w.gather(1, idx.expand(-1, 1, 3)).squeeze(1)
               updated_quat = foot_ori_w.gather(1, idx.expand(-1, 1, 4)).squeeze(1)
               updated_ori = get_euler_from_quat(updated_quat)

               self.stance_foot_pos_0[just_reset] = updated_pos[just_reset]
               self.stance_foot_ori_quat_0[just_reset] = updated_quat[just_reset]
               self.stance_foot_ori_0[just_reset] = updated_ori[just_reset]
          

        self.stance_idx = new_stance_idx

        #expand to N envs
        self.tp = torch.full((self.env.num_envs,), tp, device=self.device)
        if tp < 0.5:
            self.phase_var = 2 * tp
        else:
            self.phase_var = 2 * tp - 1
        self.cur_swing_time = self.phase_var * Tswing 