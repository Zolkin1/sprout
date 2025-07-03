import torch
import numpy as np
from typing import List, Dict
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_from_euler_xyz,quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv, quat_apply

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import get_euler_from_quat, bezier_deg

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.hlip_cmd import _transfer_to_local_frame,euler_rates_to_omega



class EndEffectorTracker:
    def __init__(self, constraint_specs: List[Dict], scene=None):
        """
        Initialize EndEffectorTracker using frame transformer sensors for efficient tracking.
        
        Args:
            constraint_specs: List of constraint specifications containing frame names
            scene: Scene object containing the frame sensors
        """
        self.constraint_specs = constraint_specs
        self.scene = scene
        
        # Map frame names to their corresponding sensors
        self.frame_to_sensor_mapping = {
            "left_foot_middle": "left_foot_sensor",
            "right_foot_middle": "right_foot_sensor", 
            "left_hand_palm_joint": "left_hand_sensor",
            "right_hand_palm_joint": "right_hand_sensor",
            "pelvis_link": "pelvis_sensor",
        }
        
        # Map frame names to body names for velocity data
        self.frame_to_body_mapping = {
            "left_foot_middle": "left_ankle_roll_link",
            "right_foot_middle": "right_ankle_roll_link", 
            "left_hand_palm_joint": "left_elbow_link",
            "right_hand_palm_joint": "right_elbow_link",
            "pelvis_link": "pelvis_link",
        }
        
        # Will be populated when robot data is first available
        self.frame_to_body_idx_mapping = {}
        self.body_idx_mapping_initialized = False
        
        print("EndEffectorTracker initialized with frame transformer sensors")

    def _initialize_body_idx_mapping(self, robot_data):
        """Initialize the body index mapping once when robot data is available."""
        if self.body_idx_mapping_initialized:
            return
            
        for frame_name, body_name in self.frame_to_body_mapping.items():
            # Find body index by name
            body_idx = None
            for i, name in enumerate(robot_data.body_names):
                if body_name in name:
                    body_idx = i
                    break
                    
            if body_idx is None:
                print(f"Warning: Body '{body_name}' not found in robot data for frame '{frame_name}'")
                continue
                
            self.frame_to_body_idx_mapping[frame_name] = body_idx
        
        self.body_idx_mapping_initialized = True
        print(f"Body index mapping initialized: {self.frame_to_body_idx_mapping}")

    def get_pose(self, frame_name: str):
        """Returns (position, euler_orientation) for a given EE frame."""
        if self.scene is None:
            raise ValueError("Scene not available")
            
        if frame_name not in self.frame_to_sensor_mapping:
            raise ValueError(f"Frame '{frame_name}' not found in frame sensor mapping")
            
        sensor_name = self.frame_to_sensor_mapping[frame_name]

        frame_sensor = self.scene.sensors.get(sensor_name, None)
        
        if frame_sensor is None:
            raise ValueError(f"Frame sensor '{sensor_name}' not found in scene")
        
        # Get position and orientation from frame sensor
        # Each sensor has only one target frame at index 0
        pos = frame_sensor.data.target_pos_w[:, 0, :]  # Shape: (num_envs, 3)
        quat = frame_sensor.data.target_quat_w[:, 0, :]  # Shape: (num_envs, 4)
        
        # Convert quaternions to euler angles for all environments
        euler = get_euler_from_quat(quat)  # Shape: (num_envs, 3)
        return pos, euler,quat

    def get_velocity(self, frame_name: str, robot_data):
        """Returns (linear_velocity, angular_velocity) for a given EE frame from robot body data."""
        # Initialize body index mapping if not done yet
        self._initialize_body_idx_mapping(robot_data)
        
        if frame_name not in self.frame_to_body_idx_mapping:
            raise ValueError(f"Frame '{frame_name}' not found in body index mapping")
            
        body_idx = self.frame_to_body_idx_mapping[frame_name]
        
        # Get linear and angular velocity from robot body data
        lin_vel = robot_data.body_lin_vel_w[:, body_idx, :]  # Shape: (num_envs, 3)
        ang_vel = robot_data.body_ang_vel_w[:, body_idx, :]  # Shape: (num_envs, 3)
        
        return lin_vel, ang_vel

    def get_relabel_matrix(self, frame_name: str, is_orientation: bool) -> torch.Tensor:
        """Returns the relabel matrix for mirroring."""
        if is_orientation:
            mapping = [-1.0, 1.0, 1.0]  # roll flipped
        else:
            mapping = [1.0, -1.0, 1.0]  # y flipped
        return torch.diag(torch.tensor(mapping, dtype=torch.float32))

    def get_remapped_pose(self, frame_name: str, is_orientation: bool) -> torch.Tensor:
        pos, ori = self.get_pose(frame_name)
        raw = ori if is_orientation else pos
        remap = self.get_relabel_matrix(frame_name, is_orientation)
        return remap @ raw

    def get_all_poses(self) -> Dict[str, tuple]:
        """Get poses for all tracked frames."""
        poses = {}
        for frame_name in self.frame_to_sensor_mapping.keys():
            try:
                poses[frame_name] = self.get_pose(frame_name)
            except Exception as e:
                print(f"Warning: Could not get pose for frame {frame_name}: {e}")
        return poses


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
    # swing_hand_pos: [1,-1,1]
    R[14, 14] = -1
    # swing_hand_ori: [-1,1,-1]
    R[16, 16] = -1
    # stance_hand_pos: [1,-1,1]
    R[18, 18] = -1
    # stance_hand_ori: [-1,1,-1]
    R[20, 20] = -1

    # the traj is in right stance, so swing foot is left foot
    # need to swap left and right palm coeffs

    return R


class EndEffectorTrajectoryConfig:
    """Configuration class for end effector trajectories."""
    
    def __init__(self, yaml_path="source/robot_rl/robot_rl/assets/robots/single_support_config_solution_ee.yaml"):
        self.yaml_path = yaml_path
        self.constraint_specs = []
        self.bezier_coeffs = {}
        self.T = 0.0
        self.load_from_yaml()
    
    def load_from_yaml(self):
        """Load constraint specs and bezier coefficients from YAML file."""
        import yaml
        
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Load constraint specifications
        self.constraint_specs = data.get('constraint_specs', [])
        
        # Reshape bezier coefficients to [num_joints, num_control_points]
        # TODO: need to get num_output from constraint_specs
        num_output = 21
        bezier_coeffs = data['bezier_coeffs']
        num_control_points = data['spline_order'] + 1
        bezier_coeffs_reshaped = np.array(bezier_coeffs).reshape(num_output, num_control_points)
        
        self.bezier_coeffs = bezier_coeffs_reshaped
        
        # Load step period
        self.T = data['T'][0] if isinstance(data['T'], list) else data['T']
        self.joint_order = data['joint_order']
        
        return
    
    def get_constraint_frames(self) -> List[str]:
        """Extract frame names from constraint specs."""
        frames = []
        for spec in self.constraint_specs:
            if "frame" in spec:
                frames.append(spec["frame"])
        return frames

    def reorder_and_remap_ee(self, cfg, ee_tracker, device):
        """Reorder and remap end effector coefficients using hardcoded relabeling matrix."""
        # Load all bezier coefficients from YAML
        
        self.right_coeffs = torch.tensor(self.bezier_coeffs, dtype=torch.float32, device=device)

        # Apply relabeling matrix to get left coefficients
        R = relable_ee_coeffs()

        # Apply relabeling: left_coeffs = R @ right_coeffs
        left_coeffs = R @ self.bezier_coeffs

        self.left_coeffs = torch.tensor(left_coeffs, dtype=torch.float32, device=device)
        
        # Generate axis names for metrics
        self.generate_axis_names()
        
        return

    def generate_axis_names(self):
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
                        'index': current_idx + i
                    })
                current_idx += len(axes)
                
            elif constraint_type == "joint":
                output_dim = 1
                joint_name = spec["joint_name"]
                # Generate metric name for joint
                metric_name = f"joint_{joint_name}"
                self.axis_names.append({
                    'name': metric_name,
                    'index': current_idx
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
                        'index': current_idx + i
                    })
                
                current_idx += len(axes)

    def get_ref_traj(self, ee_hzd_cmd):
        """Get reference trajectory similar to joint trajectory approach."""
        base_velocity = ee_hzd_cmd.env.command_manager.get_command("base_velocity")
        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)


# based on yaw velocity, update com_pos_des, com_vel_des, foot_target,
        delta_psi = base_velocity[:,2] * ee_hzd_cmd.cur_swing_time

        # Choose coefficients based on stance
        if ee_hzd_cmd.stance_idx == 1:
            ctrl_points = self.right_coeffs
        else:
            ctrl_points = self.left_coeffs
     
        phase_var_tensor = torch.full((N,), ee_hzd_cmd.phase_var, device=ee_hzd_cmd.device)
        des_ee_pos = bezier_deg(
            0, phase_var_tensor, T, ctrl_points, torch.tensor(ee_hzd_cmd.cfg.bez_deg, device=ee_hzd_cmd.device)
        )
        
        des_ee_vel = bezier_deg(1, phase_var_tensor, T, ctrl_points, ee_hzd_cmd.cfg.bez_deg)

        des_ee_pos[:,ee_hzd_cmd.foot_yaw_output_idx] += delta_psi
        des_ee_vel[:,ee_hzd_cmd.foot_yaw_output_idx] += base_velocity[:,2]

        for i in ee_hzd_cmd.ori_idx_list:
               des_ee_vel[:,i] = euler_rates_to_omega(des_ee_pos[:,i], des_ee_vel[:,i])
        return des_ee_pos, des_ee_vel

    def get_actual_traj(self, ee_hzd_cmd):
        """Get actual trajectory from end effector tracker."""
        ee_tracker = ee_hzd_cmd.ee_tracker
        

        
        # Determine swing foot frame name based on stance
        # If stance_idx == 0 (left stance), then right foot is swing foot
        # If stance_idx == 1 (right stance), then left foot is swing foot
        swing_foot_frame = "right_foot_middle" if ee_hzd_cmd.stance_idx == 0 else "left_foot_middle"
        swing_hand_frame = "right_hand_palm_joint" if ee_hzd_cmd.stance_idx == 0 else "left_hand_palm_joint"
        stance_hand_frame = "left_hand_palm_joint" if ee_hzd_cmd.stance_idx == 0 else "right_hand_palm_joint"
        # Get stance foot pos and velocity for relative positioning
        stance_foot_pos = ee_hzd_cmd.stance_foot_pos_0 
        
        # Get actual values for each constraint specification in order
        com2stance_foot = ee_hzd_cmd.robot.data.root_com_pos_w - stance_foot_pos
        com2stance_local = _transfer_to_local_frame(com2stance_foot, ee_hzd_cmd.stance_foot_ori_quat_0)
       
        com_vel_w = ee_hzd_cmd.robot.data.root_com_vel_w[:,0:3]
        com_vel_local = _transfer_to_local_frame(com_vel_w, ee_hzd_cmd.stance_foot_ori_quat_0)

        pelvis_ori = get_euler_from_quat(ee_hzd_cmd.robot.data.root_quat_w)
        pelvis_ori[:,2] = wrap_to_pi(pelvis_ori[:,2] - ee_hzd_cmd.stance_foot_ori_0[:,2])
        pelvis_omega = ee_hzd_cmd.robot.data.root_ang_vel_b


        swing_foot_pos, swing_foot_ori,sw2st_foot_quat = ee_tracker.get_pose(swing_foot_frame) 
        sw2st_foot_pos = swing_foot_pos - stance_foot_pos
        
        sw2st_foot_ori = swing_foot_ori 
        sw2st_foot_ori[:,2] = wrap_to_pi(swing_foot_ori[:,2] - ee_hzd_cmd.stance_foot_ori_0[:,2])
   
        foot_lin_vel_w = ee_hzd_cmd.robot.data.body_lin_vel_w[:, ee_hzd_cmd.feet_bodies_idx, :]
        foot_ang_vel_w = ee_hzd_cmd.robot.data.body_ang_vel_w[:, ee_hzd_cmd.feet_bodies_idx, :]

        sw2st_foot_vel = foot_lin_vel_w[:,1-ee_hzd_cmd.stance_idx]
        sw2st_foot_ang_vel = foot_ang_vel_w[:,1-ee_hzd_cmd.stance_idx]

        sw2st_foot_vel_local = _transfer_to_local_frame(sw2st_foot_vel, ee_hzd_cmd.stance_foot_ori_quat_0)
        sw2st_foot_ang_vel_local = _transfer_to_local_frame(sw2st_foot_ang_vel, ee_hzd_cmd.stance_foot_ori_quat_0)

        waist_joint_pos = ee_hzd_cmd.robot.data.joint_pos[:, ee_hzd_cmd.waist_joint_idx]
        waist_joint_vel = ee_hzd_cmd.robot.data.joint_vel[:, ee_hzd_cmd.waist_joint_idx]

        #palm position
        swing_hand_pos, swing_hand_ori,swing_hand_quat = ee_tracker.get_pose(swing_hand_frame)
        swing_hand_vel, swing_hand_omega = ee_tracker.get_velocity(swing_hand_frame, ee_hzd_cmd.robot.data)

        #stance hand position
        stance_hand_pos, stance_hand_ori,stance_hand_quat = ee_tracker.get_pose(stance_hand_frame)
        stance_hand_vel, stance_hand_omega = ee_tracker.get_velocity(stance_hand_frame, ee_hzd_cmd.robot.data)

        swing_hand_vel_local = _transfer_to_local_frame(swing_hand_vel, ee_hzd_cmd.stance_foot_ori_quat_0)
        swing_hand_ang_vel_local = _transfer_to_local_frame(swing_hand_omega, ee_hzd_cmd.stance_foot_ori_quat_0)
        stance_hand_vel_local = _transfer_to_local_frame(stance_hand_vel, ee_hzd_cmd.stance_foot_ori_quat_0)
        stance_hand_ang_vel_local = _transfer_to_local_frame(stance_hand_omega, ee_hzd_cmd.stance_foot_ori_quat_0)

        stance_hand_pos = stance_hand_pos - stance_foot_pos
        stance_hand_pos = _transfer_to_local_frame(stance_hand_pos, ee_hzd_cmd.stance_foot_ori_quat_0)
        swing_hand_pos = swing_hand_pos - stance_foot_pos

        swing_hand_pos = _transfer_to_local_frame(swing_hand_pos, ee_hzd_cmd.stance_foot_ori_quat_0)
        swing_hand_ori[:,2] = wrap_to_pi(swing_hand_ori[:,2] - ee_hzd_cmd.stance_foot_ori_0[:,2])
        stance_hand_ori[:,2] = wrap_to_pi(stance_hand_ori[:,2] - ee_hzd_cmd.stance_foot_ori_0[:,2])

        #concatenate all the position values
        y_act = torch.cat([com2stance_local,pelvis_ori,sw2st_foot_pos,sw2st_foot_ori,
                      waist_joint_pos,
                      swing_hand_pos,swing_hand_ori[:,2].unsqueeze(-1),stance_hand_pos,stance_hand_ori[:,2].unsqueeze(-1)], dim=-1)


        #concatenate all the velocity values
        dy_act = torch.cat([com_vel_local,pelvis_omega,sw2st_foot_vel_local,sw2st_foot_ang_vel_local,
                          waist_joint_vel,
                          swing_hand_vel_local,swing_hand_ang_vel_local[:,2].unsqueeze(-1),stance_hand_vel_local,stance_hand_ang_vel_local[:,2].unsqueeze(-1)], dim=-1)

     
        
        return y_act, dy_act
