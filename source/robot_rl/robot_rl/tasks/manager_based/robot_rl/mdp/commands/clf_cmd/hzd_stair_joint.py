import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_stair_base import HZDStairBaseCommandTerm
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import JointTrajectoryConfig, bezier_deg

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cmd_cfg import HZDStairJointCommandCfg


class HZDStairJointCommandTerm(HZDStairBaseCommandTerm):
    """HZD stair command term that uses joint trajectory references for different terrain types."""
    
    def __init__(self, cfg: "HZDStairJointCommandCfg", env):
        super().__init__(cfg, env)
        
        # Load three separate joint trajectory configs from YAML files
        # Flat terrain reference trajectory
        flat_yaml_path = "source/robot_rl/robot_rl/assets/robots/single_support_config_solution.yaml"
        self.jt_config_flat = JointTrajectoryConfig(flat_yaml_path)
        self.jt_config_flat.reorder_and_remap_jt(cfg, self.robot, self.device)
        
        # Stair up reference trajectory
        stair_up_yaml_path = "source/robot_rl/robot_rl/assets/robots/stair_config_solution.yaml"
        self.jt_config_stair_up = JointTrajectoryConfig(stair_up_yaml_path)
        self.jt_config_stair_up.reorder_and_remap_jt(cfg, self.robot, self.device)
        
        # Stair down reference trajectory
        stair_down_yaml_path = "source/robot_rl/robot_rl/assets/robots/downstair_config_solution.yaml"
        self.jt_config_stair_down = JointTrajectoryConfig(stair_down_yaml_path)
        self.jt_config_stair_down.reorder_and_remap_jt(cfg, self.robot, self.device)

    def _get_flat_swing_period(self) -> float:
        """Get the swing period for flat terrain."""
        return self.jt_config_flat.T

    def _get_stair_up_swing_period(self) -> float:
        """Get the swing period for stair up terrain."""
        return self.jt_config_stair_up.T

    def _get_stair_down_swing_period(self) -> float:
        """Get the swing period for stair down terrain."""
        return self.jt_config_stair_down.T

    def _get_swing_period(self) -> float:
        """Get the swing period - required by base HZDCommandTerm."""
        # This method is required by the base class but not used in stair logic
        # Return the flat terrain period as default
        return self.jt_config_flat.T

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
        des_jt_pos = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        des_jt_vel = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        
        # Evaluate trajectories for each terrain type and stance
        phase_var_tensor = self.phase_var
        
        # Right stance trajectories
        if torch.any(right_stance_mask):
            # Flat terrain trajectory - right stance
            flat_right_mask = flat_mask & right_stance_mask
            if torch.any(flat_right_mask):
                flat_pos = bezier_deg(0, phase_var_tensor[flat_right_mask], T[flat_right_mask], 
                                    self.jt_config_flat.right_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                flat_vel = bezier_deg(1, phase_var_tensor[flat_right_mask], T[flat_right_mask], 
                                    self.jt_config_flat.right_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[flat_right_mask] = flat_pos
                des_jt_vel[flat_right_mask] = flat_vel
            
            # Stair up trajectory - right stance
            stair_up_right_mask = stair_up_mask & right_stance_mask
            if torch.any(stair_up_right_mask):
                stair_up_pos = bezier_deg(0, phase_var_tensor[stair_up_right_mask], T[stair_up_right_mask], 
                                        self.jt_config_stair_up.right_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_up_vel = bezier_deg(1, phase_var_tensor[stair_up_right_mask], T[stair_up_right_mask], 
                                        self.jt_config_stair_up.right_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_up_right_mask] = stair_up_pos
                des_jt_vel[stair_up_right_mask] = stair_up_vel
            
            # Stair down trajectory - right stance
            stair_down_right_mask = stair_down_mask & right_stance_mask
            if torch.any(stair_down_right_mask):
                stair_down_pos = bezier_deg(0, phase_var_tensor[stair_down_right_mask], T[stair_down_right_mask], 
                                          self.jt_config_stair_down.right_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_down_vel = bezier_deg(1, phase_var_tensor[stair_down_right_mask], T[stair_down_right_mask], 
                                          self.jt_config_stair_down.right_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_down_right_mask] = stair_down_pos
                des_jt_vel[stair_down_right_mask] = stair_down_vel
        
        # Left stance trajectories
        if torch.any(left_stance_mask):
            # Flat terrain trajectory - left stance
            flat_left_mask = flat_mask & left_stance_mask
            if torch.any(flat_left_mask):
                flat_pos = bezier_deg(0, phase_var_tensor[flat_left_mask], T[flat_left_mask], 
                                    self.jt_config_flat.left_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                flat_vel = bezier_deg(1, phase_var_tensor[flat_left_mask], T[flat_left_mask], 
                                    self.jt_config_flat.left_coeffs, 
                                    torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[flat_left_mask] = flat_pos
                des_jt_vel[flat_left_mask] = flat_vel
            
            # Stair up trajectory - left stance
            stair_up_left_mask = stair_up_mask & left_stance_mask
            if torch.any(stair_up_left_mask):
                stair_up_pos = bezier_deg(0, phase_var_tensor[stair_up_left_mask], T[stair_up_left_mask], 
                                        self.jt_config_stair_up.left_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_up_vel = bezier_deg(1, phase_var_tensor[stair_up_left_mask], T[stair_up_left_mask], 
                                        self.jt_config_stair_up.left_coeffs, 
                                        torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_up_left_mask] = stair_up_pos
                des_jt_vel[stair_up_left_mask] = stair_up_vel
            
            # Stair down trajectory - left stance
            stair_down_left_mask = stair_down_mask & left_stance_mask
            if torch.any(stair_down_left_mask):
                stair_down_pos = bezier_deg(0, phase_var_tensor[stair_down_left_mask], T[stair_down_left_mask], 
                                          self.jt_config_stair_down.left_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_down_vel = bezier_deg(1, phase_var_tensor[stair_down_left_mask], T[stair_down_left_mask], 
                                          self.jt_config_stair_down.left_coeffs, 
                                          torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_down_left_mask] = stair_down_pos
                des_jt_vel[stair_down_left_mask] = stair_down_vel

        # Apply yaw offset based on base velocity
        yaw_offset = base_velocity[:, 2] 
        batch_idx = torch.arange(N, device=self.device)
        des_jt_pos[batch_idx, self.hip_yaw_idx[batch_idx, self.stance_idx]] += yaw_offset

        self.y_out = des_jt_pos
        self.dy_out = des_jt_vel

    def _update_metrics(self):
        """Update metrics specific to joint trajectory tracking."""
        # Call parent method for base metrics
        super()._update_metrics()
        
        # Update metrics using actual joint names from the robot
        for i, joint_name in enumerate(self.robot.joint_names):
            error_key = f"error_{joint_name}"
            self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i]) 