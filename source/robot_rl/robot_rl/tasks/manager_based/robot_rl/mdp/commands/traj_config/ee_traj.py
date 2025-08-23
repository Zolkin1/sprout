import torch
import numpy as np
from typing import List, Dict
from isaaclab.utils.math import wrap_to_pi, quat_apply, quat_from_euler_xyz

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import get_euler_from_quat
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.hlip_cmd import _transfer_to_local_frame, euler_rates_to_omega
from .base_traj import BaseTrajectoryConfig

def relable_ee_hand_coeffs():
    """Build relabel matrix for end effector coefficients."""
    R = np.eye(21)

    # only need to swap left and right palm coeffs
    # com pos: [1,-1,1]
    R[1, 1] = -1
    # pelvis: [-1,1,-1]
    R[3, 3] = -1
    R[5, 5] = -1
    # swing_foot_pos:[1,-1,1]
    R[7, 7] = -1
    # swing_foot_or: [-1,1,-1]
    R[9, 9] = -1
    R[11, 11] = -1
    # waist yaw
    R[12, 12] = -1
    # swing_hand_pos: [1,-1,1]
    R[14, 14] = -1
    # swing_hand_ori: [-1,1,-1]
    R[16, 16] = -1
    # stance_hand_pos: [1,-1,1]
    R[18, 18] = -1
    # stance_hand_ori: [-1,1,-1]
    R[20, 20] = -1

    return R

def relable_ee_coeffs():
    """Build relabel matrix for end effector coefficients."""
    R = np.eye(21)

    # only need to swap left and right palm coeffs
    # com pos: [1,-1,1]
    R[1, 1] = -1
    # pelvis: [-1,1,-1]
    R[3, 3] = -1
    R[5, 5] = -1
    # swing_foot_pos:[1,-1,1]
    R[7, 7] = -1
    # swing_foot_or: [-1,1,-1]
    R[9, 9] = -1
    R[11, 11] = -1
    # waist yaw
    R[12, 12] = -1

    #swap arm coeffs
    arm_offset = 12 + 1
    left_arm = arm_offset + np.array([0, 1, 2, 3])
    right_arm = arm_offset + np.array([4, 5, 6, 7])

    tmp = R[left_arm, :].copy()
    R[left_arm, :] = R[right_arm, :]
    R[right_arm, :] = tmp

    # Sign flips: shoulder_roll, shoulder_yaw
    flip_arm = arm_offset + np.array([1, 2, 5, 6])  # left/right roll/yaw
    R[flip_arm, :] *= -1

    return R

def relable_ee_stance_coeffs():
    """Build a relabelling matrix for end effector coefficients including the stance foot."""
    R = np.eye(27)

    ##
    # COM
    ##
    # com pos: [1,-1,1]
    R[1, 1] = -1

    ##
    # Pelvis
    ##
    # pelvis: [-1,1,-1]
    R[3, 3] = -1
    R[5, 5] = -1

    ##
    # Swing foot
    ##
    # swing_foot_pos:[1,-1,1]
    R[7, 7] = -1
    # swing_foot_or: [-1,1,-1]
    R[9, 9] = -1
    R[11, 11] = -1

    ##
    # Stance Foot
    ##
    # stance_foot_pos: [1, -1, 1]
    R[13, 13] = -1
    # stance_foot_ori: [-1, 1, -1]
    R[15, 15] = -1
    R[17, 17] = -1

    ##
    # Joints
    ##
    # waist yaw
    R[18, 18] = -1

    #swap arm coeffs
    arm_offset = 18 + 1
    left_arm = arm_offset + np.array([0, 1, 2, 3])
    right_arm = arm_offset + np.array([4, 5, 6, 7])

    tmp = R[left_arm, :].copy()
    R[left_arm, :] = R[right_arm, :]
    R[right_arm, :] = tmp

    # Sign flips: shoulder_roll, shoulder_yaw
    flip_arm = arm_offset + np.array([1, 2, 5, 6])  # left/right roll/yaw
    R[flip_arm, :] *= -1

    return R

class EndEffectorTrajectoryConfig(BaseTrajectoryConfig):
    """Configuration class for end effector trajectories."""
    
    def __init__(self, yaml_path="source/robot_rl/robot_rl/assets/robots/single_support_config_solution_ee.yaml"):
        self.constraint_specs = []
        super().__init__(yaml_path)
    
    def _load_specific_data(self, data):
        """Load end effector specific data from YAML."""
        # Load constraint specifications

        # Read in the domain sequence
        for domain_name in self.domain_seq:
            if data[domain_name]['constraint_specs'] is None:
                raise ValueError("No constraint specs in the solution file!")

            # For now just assuming these are all the same TODO: may want to break this assumption
            self.constraint_specs = data[domain_name]['constraint_specs']

            # Reshape bezier coefficients to [num_virtual_const, num_control_points]

            ##
            # Compute the number of virtual constraints
            ##
            def count_constraint_entries(data):
                total_count = 0
                for spec in data:
                    if 'axes' in spec:
                        total_count += len(spec['axes'])
                    if 'joint_names' in spec:
                        total_count += len(spec['joint_names'])
                return total_count
            
            num_virtual_const = count_constraint_entries(self.constraint_specs)
            print("num_virtual_constraints: ", num_virtual_const)

            bezier_coeffs = data[domain_name]['bezier_coeffs']
            num_control_points = data[domain_name]['spline_order'] + 1
            bezier_coeffs_reshaped = np.array(bezier_coeffs).reshape(num_virtual_const, num_control_points)

            self.bezier_coeffs[domain_name] = bezier_coeffs_reshaped
            self.joint_order[domain_name] = data[domain_name]['joint_order']

    def get_constraint_frames(self) -> List[str]:
        """Extract frame names from constraint specs."""
        frames = []
        for spec in self.constraint_specs:
            if "frame" in spec:
                frames.append(spec["frame"])
        return frames

    def reorder_and_remap(self, cfg, device):
        """Reorder and remap end effector coefficients using hardcoded relabeling matrix."""
        # reorder for each domain
        for domain_name in self.domain_seq:
            # Load all bezier coefficients from YAML
            self.right_coeffs[domain_name] = torch.tensor(self.bezier_coeffs[domain_name], dtype=torch.float32, device=device)

            # Apply relabeling matrix to get left coefficients
            if self.bezier_coeffs[domain_name].shape[0] == 21:
                R = relable_ee_coeffs()
            elif self.bezier_coeffs[domain_name].shape[0] == 27:
                R = relable_ee_stance_coeffs()
            else:
                raise ValueError("No hard coded bezier relabelling matrix for these virtual constraints!")

            # Apply relabeling: left_coeffs = R @ right_coeffs
            left_coeffs = R @ self.bezier_coeffs[domain_name]

            self.left_coeffs[domain_name] = torch.tensor(left_coeffs, dtype=torch.float32, device=device)

            # Generate axis names for metrics
            self.generate_axis_names(domain_name)

    def get_joint_idx_list(self, hzd_cmd):
        """Get the joint index list for the given command."""
        joint_idx_list = []
        
        for joint_name in self.constraint_specs[-1]["joint_names"]:
            joint_idx, _ = hzd_cmd.robot.find_joints(joint_name)
            joint_idx_list.append(joint_idx)
        return joint_idx_list

    def generate_axis_names(self, domain_name):
        """Generate axis names for each constraint specification."""
        self.axis_names = []
        current_idx = 0
        
        for spec in self.constraint_specs:
            constraint_type = spec["type"]
            
            if constraint_type == "com_pos":
                axes = spec.get("axes", [0, 1, 2])
                axis_names = ["x", "y", "z"]
                # Generate metric names for COM position (only specified axes)
                for i, axis_idx in enumerate(axes):
                    metric_name = f"com_pos_{axis_names[axis_idx]}"
                    self.axis_names.append({
                        'name': metric_name,
                        'index': current_idx + i,
                        'domain': domain_name,
                    })
                current_idx += len(axes)
                
            elif constraint_type == "joint":
                output_dim = 1
                joint_names = spec["joint_names"]

                for joint_name in joint_names:
                    # Generate metric name for joint
                    metric_name = f"joint_{joint_name}"
                    self.axis_names.append({
                        'name': metric_name,
                        'index': current_idx,
                        'domain': domain_name,
                    })
                    current_idx += output_dim
                
            elif "frame" in spec:
                frame_name = spec["frame"]
                
                # Determine output dimension and axis names
                axes = spec.get("axes", [0, 1, 2])
                if constraint_type in ["ee_pos"]:
                    axis_names = ["x", "y", "z"]
                elif constraint_type in ["ee_ori"]:
                    axis_names = ["roll", "pitch", "yaw"]
                else:
                    axis_names = ["x", "y", "z"]
                
                # Generate metric names for each axis (only specified axes)
                for i, axis_idx in enumerate(axes):
                    metric_name = f"{frame_name}_{constraint_type}_{axis_names[axis_idx]}"
                    self.axis_names.append({
                        'name': metric_name,
                        'index': current_idx + i,
                        'domain': domain_name,
                    })
                
                current_idx += len(axes)

    def _apply_swing_modifications(self, hzd_cmd, des_pos, des_vel, base_velocity):
        """Apply end effector specific swing modifications."""
        # based on yaw velocity, update com_pos_des, com_vel_des, foot_target, foot_vel_des

        #if standing, don't modify yaw
        delta_psi = base_velocity[:, 2] * hzd_cmd.cur_swing_time

        if hzd_cmd.use_standing:
            #5,11
            stand_idx = torch.where(torch.norm(base_velocity, dim=1) < hzd_cmd.standing_threshold)[0]
            if stand_idx.numel() > 0:
                delta_psi[stand_idx] = 0
                base_velocity[stand_idx,2] = 0

        des_pos[:, hzd_cmd.yaw_output_idx] += delta_psi.unsqueeze(-1)
        des_vel[:, hzd_cmd.yaw_output_idx] += base_velocity[:, 2].unsqueeze(-1)

        q_delta_yaw = quat_from_euler_xyz(
            torch.zeros_like(delta_psi),               # roll=0
            torch.zeros_like(delta_psi),               # pitch=0
            delta_psi                                  # yaw=Δψ
        ) 

        #adjust foot target and com pos/vel to account for yaw change
        des_pos[:,[6,7,8]] = quat_apply(q_delta_yaw, des_pos[:,[6,7,8]])  # [B,3]
        des_vel[:,[6,7,8]] = quat_apply(q_delta_yaw, des_vel[:,[6,7,8]])  # [B,3]

        des_pos[:,[0,1,2]] = quat_apply(q_delta_yaw, des_pos[:,[0,1,2]])  # [B,3]
        des_vel[:,[0,1,2]] = quat_apply(q_delta_yaw, des_vel[:,[0,1,2]])  # [B,3]

        delta_y = base_velocity[:, 1] * hzd_cmd.cur_swing_time
        des_pos[:, hzd_cmd.foot_y_output_idx] += delta_y
        des_vel[:, hzd_cmd.foot_y_output_idx] += base_velocity[:, 1]

        for i in hzd_cmd.ori_idx_list:
            des_vel[:, i] = euler_rates_to_omega(des_pos[:, i], des_vel[:, i])

        return des_pos, des_vel

    def get_actual_traj(self, hzd_cmd):
        """Get actual trajectory from end effector tracker."""
        data = hzd_cmd.robot.data
        
        # Determine swing foot frame name based on stance
        # If stance_idx == 0 (left stance), then right foot is swing foot
        # If stance_idx == 1 (right stance), then left foot is swing foot
        swing_foot_idx = hzd_cmd.feet_bodies_idx[1] if hzd_cmd.stance_idx == 0 else hzd_cmd.feet_bodies_idx[0]
        stance_foot_idx = hzd_cmd.feet_bodies_idx[0] if hzd_cmd.stance_idx == 0 else hzd_cmd.feet_bodies_idx[1]

        # Get stance foot pos and velocity for relative positioning
        relative_foot_pos = hzd_cmd.stance_foot_pos_0
        
        # Get actual values for each constraint specification in order

        ##
        # COM virtual constraint
        ##
        com2stance_foot = hzd_cmd.robot.data.root_com_pos_w - relative_foot_pos
        com2stance_local = _transfer_to_local_frame(com2stance_foot, hzd_cmd.stance_foot_ori_quat_0)
       
        com_vel_w = hzd_cmd.robot.data.root_com_vel_w[:, 0:3]
        com_vel_local = _transfer_to_local_frame(com_vel_w, hzd_cmd.stance_foot_ori_quat_0)

        ##
        # Pelvis virtual constraint
        ##
        pelvis_ori = get_euler_from_quat(hzd_cmd.robot.data.root_quat_w)
        pelvis_ori[:, 2] = wrap_to_pi(pelvis_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])
        pelvis_omega = hzd_cmd.robot.data.root_ang_vel_b

        def _pos_ori_vel_virtual(idx, frame_rel_pos, frame_rel_quat, frame_rel_ori):
            """Compute the locations of a given frame relative to another frame."""
            frame_pos = data.body_pos_w[:, idx, :]
            frame_quat = data.body_quat_w[:, idx, :]
            frame_ori = get_euler_from_quat(frame_quat)

            frame_pos_rel = frame_pos - frame_rel_pos
            frame_pos_rel_local = _transfer_to_local_frame(frame_pos_rel, frame_rel_quat)

            frame_ori_rel = frame_ori
            frame_ori_rel[:, 2] = wrap_to_pi(frame_ori_rel[:, 2] - frame_rel_ori[:, 2])

            frame_lin_vel_w = data.body_lin_vel_w[:, idx, :]
            frame_ang_vel_w = data.body_ang_vel_w[:, idx, :]

            frame_vel_local = _transfer_to_local_frame(frame_lin_vel_w, frame_rel_quat)
            frame_ang_vel_local= _transfer_to_local_frame(frame_ang_vel_w, frame_rel_quat)

            return frame_pos_rel_local, frame_ori_rel, frame_vel_local, frame_ang_vel_local

        ##
        # Swing foot virtual constraints
        ##
        swing_pos_rel, swing_ori_rel, swing_vel_rel, swing_ang_vel_rel = _pos_ori_vel_virtual(
            swing_foot_idx, relative_foot_pos, hzd_cmd.stance_foot_ori_quat_0, hzd_cmd.stance_foot_ori_0)

        ##
        # Joint virtual constraints
        ##
        joint_pos  = hzd_cmd.robot.data.joint_pos[:, hzd_cmd.joint_idx_list]
        joint_vel = hzd_cmd.robot.data.joint_vel[:, hzd_cmd.joint_idx_list]

        ##
        # Stance foot virtual constraints
        ##
        if self.bezier_coeffs[self.domain_seq[0]].shape[0] == 27:
            stance_pos_rel, stance_ori_rel, stance_vel_rel, stance_ang_vel_rel = _pos_ori_vel_virtual(
                stance_foot_idx, relative_foot_pos, hzd_cmd.stance_foot_ori_quat_0, hzd_cmd.stance_foot_ori_0)

            # concatenate all the position values
            y_act = torch.cat([com2stance_local, pelvis_ori, swing_pos_rel, swing_ori_rel,
                               stance_pos_rel, stance_ori_rel, joint_pos.squeeze(-1)], dim=-1)

            # concatenate all the velocity values
            dy_act = torch.cat([com_vel_local, pelvis_omega, swing_vel_rel, swing_ang_vel_rel,
                                stance_vel_rel, stance_ang_vel_rel, joint_vel.squeeze(-1)], dim=-1)

            return y_act, dy_act

        # concatenate all the position values
        y_act = torch.cat([com2stance_local, pelvis_ori, swing_pos_rel, swing_ori_rel,
                          joint_pos.squeeze(-1)], dim=-1)

        # concatenate all the velocity values
        dy_act = torch.cat([com_vel_local, pelvis_omega, swing_vel_rel, swing_ang_vel_rel,
                           joint_vel.squeeze(-1)], dim=-1)
        
        return y_act, dy_act

    def get_ref_traj(self, ee_hzd_cmd):
        """Legacy method - now calls parent get_ref_traj."""
        return super().get_ref_traj(ee_hzd_cmd)


class StairEEtrajConfig(EndEffectorTrajectoryConfig):
    def __init__(self, yaml_path="source/robot_rl/robot_rl/assets/robots/single_support_config_solution_ee.yaml"):
        super().__init__(yaml_path)

    def get_actual_traj(self, hzd_cmd):
        """Get actual trajectory from end effector tracker."""
        raise ValueError("[StairEETrajConfig] Not used right now!")
        ee_tracker = hzd_cmd.ee_tracker

        # Determine swing foot frame name based on stance
        stance_foot_pos = hzd_cmd.stance_foot_pos_0 
        
        # Get actual values for each constraint specification in order
        com2stance_foot = hzd_cmd.robot.data.root_com_pos_w - stance_foot_pos
        com2stance_local = _transfer_to_local_frame(com2stance_foot, hzd_cmd.stance_foot_ori_quat_0)
       
        com_vel_w = hzd_cmd.robot.data.root_com_vel_w[:, 0:3]
        com_vel_local = _transfer_to_local_frame(com_vel_w, hzd_cmd.stance_foot_ori_quat_0)

        pelvis_ori = get_euler_from_quat(hzd_cmd.robot.data.root_quat_w)
        pelvis_ori[:, 2] = wrap_to_pi(pelvis_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])
        pelvis_omega = hzd_cmd.robot.data.root_ang_vel_b

        left_foot_pos, left_foot_ori, left_foot_quat = ee_tracker.get_pose("left_foot_middle")
        right_foot_pos, right_foot_ori, right_foot_quat = ee_tracker.get_pose("right_foot_middle")

        left2st_ft_pos = left_foot_pos - stance_foot_pos
        left2st_ft_ori = left_foot_ori - hzd_cmd.stance_foot_ori_0
        left2st_ft_ori[:, 2] = wrap_to_pi(left2st_ft_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])

        right2st_ft_pos = right_foot_pos - stance_foot_pos
        right2st_ft_ori = right_foot_ori - hzd_cmd.stance_foot_ori_0
        right2st_ft_ori[:, 2] = wrap_to_pi(right2st_ft_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])

        sw2st_foot_pos = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, left2st_ft_pos, right2st_ft_pos)
        sw2st_foot_ori = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, left2st_ft_ori, right2st_ft_ori)

        foot_lin_vel_w = hzd_cmd.robot.data.body_lin_vel_w[:, hzd_cmd.feet_bodies_idx, :]
        foot_ang_vel_w = hzd_cmd.robot.data.body_ang_vel_w[:, hzd_cmd.feet_bodies_idx, :]

        batched_idx = torch.arange(foot_lin_vel_w.shape[0])
        sw2st_foot_vel = foot_lin_vel_w[batched_idx, 1 - hzd_cmd.stance_idx]
        sw2st_foot_ang_vel = foot_ang_vel_w[batched_idx, 1 - hzd_cmd.stance_idx]

        sw2st_foot_vel_local = _transfer_to_local_frame(sw2st_foot_vel, hzd_cmd.stance_foot_ori_quat_0)
        sw2st_foot_ang_vel_local = _transfer_to_local_frame(sw2st_foot_ang_vel, hzd_cmd.stance_foot_ori_quat_0)

        waist_joint_pos = hzd_cmd.robot.data.joint_pos[:, hzd_cmd.waist_joint_idx]
        waist_joint_vel = hzd_cmd.robot.data.joint_vel[:, hzd_cmd.waist_joint_idx]

        # palm position
        left_hand_pos, left_hand_ori, left_hand_quat = ee_tracker.get_pose("left_hand_palm_joint")
        right_hand_pos, right_hand_ori, right_hand_quat = ee_tracker.get_pose("right_hand_palm_joint")

        left2st_ft_pos = left_hand_pos - stance_foot_pos
        left2st_ft_ori = left_hand_ori - hzd_cmd.stance_foot_ori_0
        left2st_ft_ori[:, 2] = wrap_to_pi(left2st_ft_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])

        right2st_ft_pos = right_hand_pos - stance_foot_pos
        right2st_ft_ori = right_hand_ori - hzd_cmd.stance_foot_ori_0
        right2st_ft_ori[:, 2] = wrap_to_pi(right2st_ft_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])
        
        left_hand_vel, left_hand_omega = ee_tracker.get_velocity("left_hand_palm_joint", hzd_cmd.robot.data)
        right_hand_vel, right_hand_omega = ee_tracker.get_velocity("right_hand_palm_joint", hzd_cmd.robot.data)

        stance_hand_pos = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, right2st_ft_pos, left2st_ft_pos)
        stance_hand_ori = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, right2st_ft_ori, left2st_ft_ori)
        swing_hand_pos = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, left2st_ft_pos, right2st_ft_pos)
        swing_hand_ori = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, left2st_ft_ori, right2st_ft_ori)

        swing_hand_vel = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, left_hand_vel, right_hand_vel)
        swing_hand_omega = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, left_hand_omega, right_hand_omega)
        stance_hand_vel = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, right_hand_vel, left_hand_vel)
        stance_hand_omega = torch.where(hzd_cmd.stance_idx.unsqueeze(-1) == 1, right_hand_omega, left_hand_omega)

        swing_hand_vel_local = _transfer_to_local_frame(swing_hand_vel, hzd_cmd.stance_foot_ori_quat_0)
        swing_hand_ang_vel_local = _transfer_to_local_frame(swing_hand_omega, hzd_cmd.stance_foot_ori_quat_0)
        stance_hand_vel_local = _transfer_to_local_frame(stance_hand_vel, hzd_cmd.stance_foot_ori_quat_0)
        stance_hand_ang_vel_local = _transfer_to_local_frame(stance_hand_omega, hzd_cmd.stance_foot_ori_quat_0)

        stance_hand_pos = stance_hand_pos - stance_foot_pos
        stance_hand_pos = _transfer_to_local_frame(stance_hand_pos, hzd_cmd.stance_foot_ori_quat_0)
        swing_hand_pos = swing_hand_pos - stance_foot_pos

        swing_hand_pos = _transfer_to_local_frame(swing_hand_pos, hzd_cmd.stance_foot_ori_quat_0)
        swing_hand_ori[:, 2] = wrap_to_pi(swing_hand_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])
        stance_hand_ori[:, 2] = wrap_to_pi(stance_hand_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])

        # concatenate all the position values
        y_act = torch.cat([com2stance_local, pelvis_ori, sw2st_foot_pos, sw2st_foot_ori,
                          waist_joint_pos,
                          swing_hand_pos, swing_hand_ori[:, 2].unsqueeze(-1), stance_hand_pos, stance_hand_ori[:, 2].unsqueeze(-1)], dim=-1)

        # concatenate all the velocity values
        dy_act = torch.cat([com_vel_local, pelvis_omega, sw2st_foot_vel_local, sw2st_foot_ang_vel_local,
                           waist_joint_vel,
                           swing_hand_vel_local, swing_hand_ang_vel_local[:, 2].unsqueeze(-1), stance_hand_vel_local, stance_hand_ang_vel_local[:, 2].unsqueeze(-1)], dim=-1)
        
        return y_act, dy_act
        

