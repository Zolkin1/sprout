import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_stair_base import HZDStairBaseCommandTerm
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.ee_traj import EndEffectorTrajectoryConfig, EndEffectorTracker, bezier_deg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.hlip_cmd import euler_rates_to_omega

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cmd_cfg import HZDStairEECommandCfg


class HZDStairEECommandTerm(HZDStairBaseCommandTerm):
    """HZD stair command term that uses end effector trajectory references for different terrain types."""
    
    def __init__(self, cfg: "HZDStairEECommandCfg", env):
        super().__init__(cfg, env)
        
        # Initialize end effector tracker
        self.ee_tracker = EndEffectorTracker([], env.scene)
        
        # Load three separate end effector trajectory configs from YAML files
        # Flat terrain reference trajectory
        flat_yaml_path = "source/robot_rl/robot_rl/assets/robots/single_support_config_solution_ee.yaml"
        self.ee_config_flat = EndEffectorTrajectoryConfig(flat_yaml_path)
        self.ee_config_flat.reorder_and_remap_ee(cfg, self.ee_tracker, self.device)
        
        # Stair up reference trajectory
        stair_up_yaml_path = "source/robot_rl/robot_rl/assets/robots/stair_config_solution_ee.yaml"
        self.ee_config_stair_up = EndEffectorTrajectoryConfig(stair_up_yaml_path)
        self.ee_config_stair_up.reorder_and_remap_ee(cfg, self.ee_tracker, self.device)
        
        # Stair down reference trajectory
        stair_down_yaml_path = "source/robot_rl/robot_rl/assets/robots/downstair_config_solution_ee.yaml"
        self.ee_config_stair_down = EndEffectorTrajectoryConfig(stair_down_yaml_path)
        self.ee_config_stair_down.reorder_and_remap_ee(cfg, self.ee_tracker, self.device)
        
        # Initialize end effector specific variables
        self.waist_joint_idx, _ = self.robot.find_joints(".*waist_yaw.*")
        self.foot_yaw_output_idx = 11
        self.ori_idx_list = [
            [3, 4, 5],
            [9, 10, 11],
        ]
        self.yaw_output_idx = [5, 11, 16, 20]

    def _get_flat_swing_period(self) -> float:
        """Get the swing period for flat terrain."""
        return self.ee_config_flat.T

    def _get_stair_up_swing_period(self) -> float:
        """Get the swing period for stair up terrain."""
        return self.ee_config_stair_up.T

    def _get_stair_down_swing_period(self) -> float:
        """Get the swing period for stair down terrain."""
        return self.ee_config_stair_down.T

    def generate_reference_trajectory(self):
        """Generate reference trajectory based on terrain type and stance."""
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)
        N = base_velocity.shape[0]
        T = self.T  # Use the per-env T that was set in update_Stance_Swing_idx

        # Define height thresholds for trajectory selection
        height_threshold = 0.01  # 1cm threshold to determine if it's a step up/down
        
        # Create trajectory selection masks
        flat_mask = torch.abs(self.z_height) < height_threshold
        stair_up_mask = self.z_height >= height_threshold
        stair_down_mask = self.z_height <= -height_threshold
        
        # Create stance selection masks
        right_stance_mask = (self.stance_idx == 1)  # Right foot stance
        left_stance_mask = (self.stance_idx == 0)   # Left foot stance
        
        # Initialize output tensors
        des_ee_pos = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        des_ee_vel = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        
        # Evaluate trajectories for each terrain type and stance
        phase_var_tensor = self.phase_var
        
        # Right stance trajectories
        if torch.any(right_stance_mask):
            # Flat terrain trajectory - right stance
            flat_right_mask = flat_mask & right_stance_mask
            if torch.any(flat_right_mask):
                flat_pos = bezier_deg(0, phase_var_tensor[flat_right_mask], T[flat_right_mask], 
                                    self.ee_config_flat.right_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                flat_vel = bezier_deg(1, phase_var_tensor[flat_right_mask], T[flat_right_mask], 
                                    self.ee_config_flat.right_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                des_ee_pos[flat_right_mask] = flat_pos
                des_ee_vel[flat_right_mask] = flat_vel
            
            # Stair up trajectory - right stance
            stair_up_right_mask = stair_up_mask & right_stance_mask
            if torch.any(stair_up_right_mask):
                stair_up_pos = bezier_deg(0, phase_var_tensor[stair_up_right_mask], T[stair_up_right_mask], 
                                        self.ee_config_stair_up.right_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_up_vel = bezier_deg(1, phase_var_tensor[stair_up_right_mask], T[stair_up_right_mask], 
                                        self.ee_config_stair_up.right_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                des_ee_pos[stair_up_right_mask] = stair_up_pos
                des_ee_vel[stair_up_right_mask] = stair_up_vel
            
            # Stair down trajectory - right stance
            stair_down_right_mask = stair_down_mask & right_stance_mask
            if torch.any(stair_down_right_mask):
                stair_down_pos = bezier_deg(0, phase_var_tensor[stair_down_right_mask], T[stair_down_right_mask], 
                                          self.ee_config_stair_down.right_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_down_vel = bezier_deg(1, phase_var_tensor[stair_down_right_mask], T[stair_down_right_mask], 
                                          self.ee_config_stair_down.right_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                des_ee_pos[stair_down_right_mask] = stair_down_pos
                des_ee_vel[stair_down_right_mask] = stair_down_vel
        
        # Left stance trajectories
        if torch.any(left_stance_mask):
            # Flat terrain trajectory - left stance
            flat_left_mask = flat_mask & left_stance_mask
            if torch.any(flat_left_mask):
                flat_pos = bezier_deg(0, phase_var_tensor[flat_left_mask], T[flat_left_mask], 
                                    self.ee_config_flat.left_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                flat_vel = bezier_deg(1, phase_var_tensor[flat_left_mask], T[flat_left_mask], 
                                    self.ee_config_flat.left_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                des_ee_pos[flat_left_mask] = flat_pos
                des_ee_vel[flat_left_mask] = flat_vel
            
            # Stair up trajectory - left stance
            stair_up_left_mask = stair_up_mask & left_stance_mask
            if torch.any(stair_up_left_mask):
                stair_up_pos = bezier_deg(0, phase_var_tensor[stair_up_left_mask], T[stair_up_left_mask], 
                                        self.ee_config_stair_up.left_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_up_vel = bezier_deg(1, phase_var_tensor[stair_up_left_mask], T[stair_up_left_mask], 
                                        self.ee_config_stair_up.left_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                des_ee_pos[stair_up_left_mask] = stair_up_pos
                des_ee_vel[stair_up_left_mask] = stair_up_vel
            
            # Stair down trajectory - left stance
            stair_down_left_mask = stair_down_mask & left_stance_mask
            if torch.any(stair_down_left_mask):
                stair_down_pos = bezier_deg(0, phase_var_tensor[stair_down_left_mask], T[stair_down_left_mask], 
                                          self.ee_config_stair_down.left_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_down_vel = bezier_deg(1, phase_var_tensor[stair_down_left_mask], T[stair_down_left_mask], 
                                          self.ee_config_stair_down.left_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                des_ee_pos[stair_down_left_mask] = stair_down_pos
                des_ee_vel[stair_down_left_mask] = stair_down_vel

        # Apply yaw offset based on base velocity and swing time
        delta_psi = base_velocity[:, 2] * self.cur_swing_time
        des_ee_pos[:, self.foot_yaw_output_idx] += delta_psi
        des_ee_vel[:, self.foot_yaw_output_idx] += base_velocity[:, 2]

        # Convert euler rates to angular velocities for orientation indices
        for i in self.ori_idx_list:
            des_ee_vel[:, i] = euler_rates_to_omega(des_ee_pos[:, i], des_ee_vel[:, i])

        self.y_out = des_ee_pos
        self.dy_out = des_ee_vel

    def get_actual_state(self):
        """Get actual state for end effector trajectories."""
        # Get stance foot pose data
        self.get_stance_foot_pose()
        
        # Get actual trajectory from end effector tracker
        act_pos, act_vel = self.ee_config_flat.get_actual_traj(self)
        self.y_act = act_pos
        self.dy_act = act_vel

    def get_stance_foot_pose(self):
        """Get stance foot pose data using end effector tracker."""
        stance_foot_frame = "left_foot_middle" if self.stance_idx == 0 else "right_foot_middle"
        stance_foot_pos, stance_foot_ori, stance_foot_quat = self.ee_tracker.get_pose(stance_foot_frame) 
        self.stance_foot_pos = stance_foot_pos
        self.stance_foot_ori = stance_foot_ori
        stance_foot_vel, stance_foot_ang_vel = self.ee_tracker.get_velocity(stance_foot_frame, self.robot.data)
        self.stance_foot_vel = stance_foot_vel
        self.stance_foot_ang_vel = stance_foot_ang_vel

    def _update_metrics(self):
        """Update metrics specific to end effector trajectory tracking."""
        # Call parent method for base metrics
        super()._update_metrics()
        
        # Update metrics using pre-generated axis names from the flat config
        for axis_info in self.ee_config_flat.axis_names:
            error_key = axis_info['name']
            index = axis_info['index']
            self.metrics[error_key] = torch.abs(
                self.y_out[:, index] - 
                self.y_act[:, index]
            ) 