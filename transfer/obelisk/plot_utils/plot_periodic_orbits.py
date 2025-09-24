#!/usr/bin/env python3
"""
Script to plot periodic joint phase space orbits (position vs velocity) from gait library data.

Usage:
    python plot_periodic_orbits.py [--ctrl-logs=/path/to/ctrl/logs] [--gait-library=/path/to/gait/library]
    
If no gait library path is provided, uses the default full_library_v7 folder.
For now, only gait library plotting is implemented (ctrl-logs argument is reserved for future use).
Plots are automatically saved to the ctrl_logs directory.
"""

# TODO: This file needs a complete overhaul/re-write

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv
import ast
import glob
import pinocchio as pin
import scipy.spatial.transform as st


def find_default_gait_library() -> Optional[str]:
    """Find the default gait library folder."""
    # Get ROBOT_RL_ROOT from environment or use current working directory structure
    robot_rl_root = os.environ.get("ROBOT_RL_ROOT", "")
    if robot_rl_root:
        gait_lib_path = os.path.join(robot_rl_root, "robot_rl", "source", "robot_rl", "robot_rl", 
                                    "assets", "robots", "full_library_v7")
    else:
        # Try relative path from current working directory
        gait_lib_path = os.path.join("robot_rl", "source", "robot_rl", "robot_rl", 
                                    "assets", "robots", "full_library_v7")
    
    if os.path.exists(gait_lib_path):
        print(f"Using gait library: {gait_lib_path}")
        return gait_lib_path
    else:
        raise FileNotFoundError(f"Gait library not found at {gait_lib_path}")


def find_most_recent_ctrl_folder() -> Optional[str]:
    """Find the most recent control log folder in ctrl_logs directory."""
    import glob
    
    # Get ROBOT_RL_ROOT from environment
    robot_rl_root = os.environ.get("ROBOT_RL_ROOT", "")
    if not robot_rl_root:
        raise EnvironmentError("ROBOT_RL_ROOT environment variable not set")
    
    ctrl_logs_dir = os.path.join(robot_rl_root, "ctrl_logs")
    if not os.path.exists(ctrl_logs_dir):
        raise FileNotFoundError(f"ctrl_logs directory not found at {ctrl_logs_dir}")
    
    # Find all timestamped subdirectories
    folders = glob.glob(os.path.join(ctrl_logs_dir, "*"))
    folders = [f for f in folders if os.path.isdir(f)]
    
    if not folders:
        raise FileNotFoundError(f"No control log folders found in {ctrl_logs_dir}")
    
    # Sort by modification time and return most recent
    folders.sort(key=os.path.getmtime, reverse=True)
    most_recent = folders[0]
    print(f"Using most recent ctrl_logs folder: {most_recent}")
    return most_recent


def load_gait_data(gait_file_path: str) -> Dict[str, Any]:
    """Load gait data from YAML file."""
    if not os.path.exists(gait_file_path):
        raise FileNotFoundError(f"Gait file not found: {gait_file_path}")
    
    try:
        with open(gait_file_path, 'r') as f:
            gait_data = yaml.safe_load(f)
        return gait_data
    except Exception as e:
        raise RuntimeError(f"Error loading gait data from {gait_file_path}: {e}")


def get_joint_names() -> List[str]:
    """Return the joint names in Isaac order (matching plot_ctrl.py)."""
    return [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint", 
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ]


def get_symmetric_joint_name(joint_name: str) -> str:
    """Get the symmetric joint name (left->right, right->left)."""
    if joint_name.startswith("left_"):
        return joint_name.replace("left_", "right_")
    elif joint_name.startswith("right_"):
        return joint_name.replace("right_", "left_")
    else:
        # For joints like waist_yaw_joint that don't have left/right, return the same name
        return joint_name


def load_config(folder_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file in the log folder."""
    yaml_files = glob.glob(os.path.join(folder_path, "*.yaml"))
    if not yaml_files:
        print("Warning: No YAML config file found in log folder")
        return {}
    
    config_path = yaml_files[0]  # Use the first YAML file found
    print(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def parse_observation_data(obs_str: str) -> np.ndarray:
    """Parse observation string from CSV into numpy array."""
    try:
        # Convert string representation of list to actual list
        obs_list = ast.literal_eval(obs_str)
        return np.array(obs_list, dtype=np.float32)
    except Exception as e:
        print(f"Error parsing observation: {e}")
        return np.array([])


def decode_observations(observations: np.ndarray, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Decode observations based on the observation type and config."""
    if not config:
        print("Warning: No config available, using default decoding")
        return {"raw_observations": observations}
    
    # Get observation type and parameters
    control_params = config.get("onboard", {}).get("control", [{}])[0].get("params", {})
    obs_type = control_params.get("obs_type", "mlp")
    num_obs = control_params.get("num_obs", 74)
    qvel_scale = control_params.get("qvel_scale", 1.0)
    ang_vel_scale = control_params.get("ang_vel_scale", 1.0)
    cmd_scale = control_params.get("cmd_scale", [2.0, 2.0, 2.0])
    action_scale = control_params.get("action_scale", 0.25)
    default_angles_config = control_params.get("default_angles", [0.0] * 21)
    default_angles_names = control_params.get("default_angles_names", [])
    
    print(f"Decoding observations with type: {obs_type}, num_obs: {num_obs}")
    
    # Define joint names in Isaac order (hardware order)
    isaac_joint_names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint", 
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ]
    
    decoded = {}
    num_joints = 21
    
    # Convert default angles from config order to Isaac order
    config_name_to_value = {}
    for i, name in enumerate(default_angles_names):
        if i < len(default_angles_config):
            config_name_to_value[name] = default_angles_config[i]
    
    isaac_defaults = np.zeros(21)
    for isaac_idx, isaac_name in enumerate(isaac_joint_names):
        if isaac_name in config_name_to_value:
            isaac_defaults[isaac_idx] = config_name_to_value[isaac_name]
    
    # Decode based on observation structure (from controller.py)
    if obs_type in ["mlp", "gl", "cnn"]:
        # Common structure: [ang_vel(3), proj_g(3), cmd_vel(3), joint_pos(21), joint_vel(21), past_action(21), phase(2)]
        if obs_type == "mlp":
            # MLP: joint_pos, joint_vel, past_action, phase
            joint_positions = observations[:, 9:9+num_joints]
            joint_velocities = observations[:, 9+num_joints:9+2*num_joints] / qvel_scale
            sin_phase = observations[:, 9+3*num_joints]
            cos_phase = observations[:, 9+3*num_joints+1]
        elif obs_type == "gl":
            # GL: joint_vel, joint_pos, past_action, phase  
            joint_velocities = observations[:, 9:9+num_joints] / qvel_scale
            joint_positions = observations[:, 9+num_joints:9+2*num_joints]
            sin_phase = observations[:, 9+3*num_joints]
            cos_phase = observations[:, 9+3*num_joints+1]
        
        # Add default angles back to joint positions (they're stored as offsets from defaults)
        actual_joint_positions = joint_positions + isaac_defaults
        
        # Calculate phase from sin/cos
        phase = np.arctan2(sin_phase, cos_phase)
        # Normalize phase to 0-1 range
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        
        # Extract projected gravity (indices 3:6 in the observation)
        projected_gravity = observations[:, 3:6]
        
        decoded["joint_positions"] = actual_joint_positions
        decoded["joint_velocities"] = joint_velocities
        decoded["phase"] = phase_normalized
        decoded["projected_gravity"] = projected_gravity
        decoded["isaac_joint_names"] = isaac_joint_names
    
    return decoded


def load_ctrl_data(folder_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Load control data from folder with optional time filtering."""
    csv_path = os.path.join(folder_path, "g1_control.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"g1_control.csv not found in {folder_path}")
    
    # Load config
    config = load_config(folder_path)
    
    times = []
    observations = []
    
    print(f"Loading hardware data from: {csv_path}")
    if start_time is not None or end_time is not None:
        print(f"Filtering data: start_time={start_time}, end_time={end_time}")
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) >= 2:  # time, observation
                    time_val = float(row[0])
                    
                    # Apply time filtering
                    if start_time is not None and time_val < start_time:
                        continue
                    if end_time is not None and time_val > end_time:
                        break
                    
                    obs_data = parse_observation_data(row[1])
                    
                    if len(obs_data) > 0:
                        times.append(time_val)
                        observations.append(obs_data)
                        
                if row_idx % 1000 == 0:
                    print(f"Processed {row_idx} rows...")
                    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise
    
    times = np.array(times)
    observations = np.array(observations)
    
    if len(times) == 0:
        raise ValueError(f"No data found in specified time range: start_time={start_time}, end_time={end_time}")
    
    print(f"Loaded {len(times)} data points spanning {times[-1] - times[0]:.2f} seconds")
    print(f"Time range: {times[0]:.2f}s to {times[-1]:.2f}s")
    print(f"Observation shape: {observations.shape}")
    
    # Decode observations
    decoded_obs = decode_observations(observations, config)
    
    return times, decoded_obs, config


def map_hardware_to_gait_order(hardware_data: np.ndarray, isaac_joint_names: List[str], gait_joint_names: List[str]) -> np.ndarray:
    """Map hardware joint data from Isaac order to gait library order."""
    gait_ordered_data = np.zeros((hardware_data.shape[0], len(gait_joint_names)))
    
    for gait_idx, gait_joint_name in enumerate(gait_joint_names):
        if gait_joint_name in isaac_joint_names:
            isaac_idx = isaac_joint_names.index(gait_joint_name)
            gait_ordered_data[:, gait_idx] = hardware_data[:, isaac_idx]
        else:
            print(f"Warning: Joint '{gait_joint_name}' not found in hardware data")
    
    return gait_ordered_data


def extract_hardware_trajectories(decoded_obs: Dict[str, np.ndarray], gait_joint_order: List[str], joint_names_to_plot: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract hardware joint trajectories and map to gait library order."""
    isaac_joint_names = decoded_obs["isaac_joint_names"]
    hardware_positions = decoded_obs["joint_positions"]
    hardware_velocities = decoded_obs["joint_velocities"]
    phase = decoded_obs["phase"]
    
    # Map hardware data to gait library joint order
    gait_ordered_positions = map_hardware_to_gait_order(hardware_positions, isaac_joint_names, gait_joint_order)
    gait_ordered_velocities = map_hardware_to_gait_order(hardware_velocities, isaac_joint_names, gait_joint_order)
    
    # Extract data for joints we want to plot
    hardware_trajectories = {}
    for joint_name in joint_names_to_plot:
        if joint_name in gait_joint_order:
            joint_idx = gait_joint_order.index(joint_name)
            hardware_trajectories[joint_name] = {
                'positions': gait_ordered_positions[:, joint_idx],
                'velocities': gait_ordered_velocities[:, joint_idx],
                'phase': phase
            }
            print(f"Extracted hardware data for joint '{joint_name}': {len(phase)} data points")
        else:
            print(f"Warning: Joint '{joint_name}' not found in gait joint order")
    
    return hardware_trajectories


def _ncr(n, r):
    """Compute binomial coefficient (n choose r)."""
    import math
    return math.comb(n, r)


def bezier_deg_numpy(order: int, tau: np.ndarray, step_dur: float, control_points: np.ndarray, degree: int) -> np.ndarray:
    """
    Numpy version of Bezier curve evaluation.
    
    Args:
        order: 0 → position, 1 → time-derivative
        tau: array of shape [N], each in [0,1]
        step_dur: scalar, positive
        control_points: array of shape [n_dim, degree+1]
        degree: polynomial degree
        
    Returns:
        Array of shape [N, n_dim] with Bezier positions or derivatives
    """
    tau = np.clip(tau, 0.0, 1.0)
    
    if order == 1:
        # Derivative case
        cp_diff = control_points[:, 1:] - control_points[:, :-1]  # [n_dim, degree]
        coefs_diff = np.array([_ncr(degree - 1, i) for i in range(degree)])
        
        i_vec = np.arange(degree)
        tau_pow = tau[:, np.newaxis] ** i_vec[np.newaxis, :]  # [N, degree]
        one_minus_pow = (1 - tau)[:, np.newaxis] ** (degree - 1 - i_vec)[np.newaxis, :]  # [N, degree]
        
        weight_deriv = degree * coefs_diff[np.newaxis, :] * one_minus_pow * tau_pow  # [N, degree]
        Bdot = weight_deriv @ cp_diff.T  # [N, n_dim]
        
        return Bdot / step_dur
    else:
        # Position case
        coefs_pos = np.array([_ncr(degree, i) for i in range(degree + 1)])
        
        i_vec = np.arange(degree + 1)
        tau_pow = tau[:, np.newaxis] ** i_vec[np.newaxis, :]  # [N, degree+1]
        one_minus_pow = (1 - tau)[:, np.newaxis] ** (degree - i_vec)[np.newaxis, :]  # [N, degree+1]
        
        weight_pos = coefs_pos[np.newaxis, :] * one_minus_pow * tau_pow  # [N, degree+1]
        B = weight_pos @ control_points.T  # [N, n_dim]
        
        return B


def parse_ankle_bezier_coefficients(gait_data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Parse Bezier coefficients for ankle positions from gait library."""
    ankle_coeffs = {}
    
    domain_sequence = gait_data['domain_sequence']
    
    for domain in domain_sequence:
        domain_data = gait_data[domain]
        
        # Get bezier coefficients and reshape them
        bezier_coeffs = np.array(domain_data['bezier_coeffs'])
        spline_order = domain_data.get('spline_order', 5)  # Default to 5 if not specified
        num_control_points = spline_order + 1
        
        # Reshape coefficients - need to determine number of virtual constraints
        # For ankle positions, we're looking for swing_foot_pos and stance_foot_pos (if available)
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
        constraint_specs = domain_data['constraint_specs']
        num_virtual_const = count_constraint_entries(constraint_specs)
        bezier_coeffs_reshaped = bezier_coeffs.reshape(num_virtual_const, num_control_points)
        
        # Find swing foot position coefficients (indices 6, 7, 8 for x, y, z)
        # Based on the constraint ordering from ee_traj.py
        swing_foot_start_idx = 6  # swing_foot_pos starts at index 6
        swing_foot_coeffs = bezier_coeffs_reshaped[swing_foot_start_idx:swing_foot_start_idx+3, :]  # [3, num_control_points]
        
        # Check if stance foot coefficients are available (for newer gait libraries)
        stance_foot_coeffs = None
        if num_virtual_const >= 15:  # Has stance foot constraints
            stance_foot_start_idx = 12  # stance_foot_pos starts at index 12
            stance_foot_coeffs = bezier_coeffs_reshaped[stance_foot_start_idx:stance_foot_start_idx+3, :]  # [3, num_control_points]
        
        ankle_coeffs[domain] = {
            'swing_foot_coeffs': swing_foot_coeffs,
            'stance_foot_coeffs': stance_foot_coeffs,
            'period': domain_data['T'][0],
            'degree': spline_order
        }
        
        print(f"Domain '{domain}': Found swing foot coefficients shape {swing_foot_coeffs.shape}, "
              f"stance foot coefficients: {'Available' if stance_foot_coeffs is not None else 'Not available'}")
    
    return ankle_coeffs


def compute_gait_library_ankle_positions(ankle_coeffs: Dict[str, Dict[str, np.ndarray]], 
                                        phase: np.ndarray, compute_velocities: bool = False) -> Dict[str, np.ndarray]:
    """Compute ankle positions from gait library Bezier coefficients."""
    
    num_timesteps = len(phase)
    swing_ankle_positions = np.zeros((num_timesteps, 3))
    swing_ankle_velocities = np.zeros((num_timesteps, 3)) if compute_velocities else None
    
    domain_sequence = list(ankle_coeffs.keys())

    ssp_fraction = 0.154

    for t in range(num_timesteps):
        current_phase = phase[t]
        
        # Determine which domain we're in and the local tau within that domain
        if current_phase < ssp_fraction:   # SSP
            # First half - use first domain
            domain = domain_sequence[0]
            tau_in_domain = current_phase / ssp_fraction  # Map 0-0.5 to 0-1
        else:   # DSP
            # Second half - use second domain
            domain = domain_sequence[1] if len(domain_sequence) > 1 else domain_sequence[0]
            tau_in_domain = (current_phase -ssp_fraction) / (1 - ssp_fraction)  # Map 0.5-1.0 to 0-1

        # Get domain coefficients
        domain_coeffs = ankle_coeffs[domain]
        swing_coeffs = domain_coeffs['swing_foot_coeffs']
        period = domain_coeffs['period']
        degree = domain_coeffs['degree']
        
        # Evaluate Bezier curves
        tau_array = np.array([tau_in_domain])
        
        # Compute swing foot position
        swing_pos = bezier_deg_numpy(0, tau_array, period, swing_coeffs, degree)[0]  # [3]
        swing_ankle_positions[t] = swing_pos
        
        # Compute swing foot velocity if requested
        if compute_velocities:
            swing_vel = bezier_deg_numpy(1, tau_array, period, swing_coeffs, degree)[0]  # [3]
            swing_ankle_velocities[t] = swing_vel

        # Add the y PD signal
        # swing_ankle_positions[t] += (t/num_timesteps) * 0.355 * -0.05

        # # Compute stance foot position (if available)
        # if stance_coeffs is not None:
        #     stance_pos = bezier_deg_numpy(0, tau_array, period, stance_coeffs, degree)[0]  # [3]
        # else:
        #     # If no stance coefficients, assume stance foot stays at origin
        #     stance_pos = np.zeros(3)
    
    result = {
        'swing_ankle_positions': swing_ankle_positions,
        'phase': phase
    }
    
    if compute_velocities:
        result['swing_ankle_velocities'] = swing_ankle_velocities
    
    return result


def decode_projected_gravity_to_orientation(projected_gravity: np.ndarray) -> Tuple[float, float]:
    """
    Decode projected gravity vector to roll and pitch angles.
    
    Args:
        projected_gravity: [gx, gy, gz] projected gravity vector (normalized)
        
    Returns:
        Tuple of (roll, pitch) in radians
    """
    # Projected gravity is the gravity vector in the robot's body frame
    # For a normalized gravity vector [gx, gy, gz], we can compute:
    # roll = atan2(gy, gz)  (rotation around x-axis)
    # pitch = -asin(gx)     (rotation around y-axis, negative because forward is +x)
    
    gx, gy, gz = projected_gravity[0], -projected_gravity[1], -projected_gravity[2]
    
    # Compute roll (rotation around x-axis)
    roll = np.arctan2(gy, gz)
    
    # Compute pitch (rotation around y-axis)
    # Clamp gx to [-1, 1] to avoid numerical issues with asin
    gx_clamped = np.clip(gx, -1.0, 1.0)
    pitch = -np.arcsin(gx_clamped)
    
    return roll, pitch


def load_robot_model() -> Tuple[pin.Model, pin.Data]:
    """Load the robot model using Pinocchio."""
    # Use the specific URDF path provided
    urdf_path = os.path.join("/home/zolkin/AmberLab/Project-Isaac-RL/robot-rl/robot_rl/transfer/sim/robots/g1/", "g1_21j.urdf")
    
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"Robot URDF not found at {urdf_path}")
    
    print(f"Loading robot model from: {urdf_path}")
    
    # Load the model
    model = pin.buildModelFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    data = model.createData()
    
    print(f"Loaded robot model with {model.nq} DoF")
    return model, data


def compute_ankle_positions(model: pin.Model, data: pin.Data, joint_positions: np.ndarray, 
                           gait_joint_order: List[str], phase: np.ndarray, 
                           projected_gravity: Optional[np.ndarray] = None,
                           joint_velocities: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Compute ankle positions using forward kinematics."""
    
    # Map gait library joint order to Pinocchio joint order
    pin_joint_names = [model.names[i] for i in range(1, model.njoints)]  # Skip universe
    print(f"Pinocchio joint names: {pin_joint_names}")
    print(f"Gait joint order: {gait_joint_order}")
    
    # Create mapping from gait order to Pinocchio order
    joint_mapping = {}
    for gait_idx, gait_joint in enumerate(gait_joint_order):
        if gait_joint in pin_joint_names:
            pin_idx = model.getJointId(gait_joint)
            joint_mapping[gait_idx] = pin_idx
            print(f"Joint: {gait_joint}, idx: {pin_idx}")
        else:
            print(f"Warning: Joint {gait_joint} not found in Pinocchio model")
    
    # Get frame IDs for ankle links
    try:
        left_ankle_frame_id = model.getFrameId("left_ankle_roll_link")
        right_ankle_frame_id = model.getFrameId("right_ankle_roll_link")
    except:
        # Try alternative frame names
        try:
            left_ankle_frame_id = model.getFrameId("left_ankle_link")
            right_ankle_frame_id = model.getFrameId("right_ankle_link")
        except:
            raise ValueError("Could not find ankle frame IDs in robot model")
    
    print(f"Left ankle frame ID: {left_ankle_frame_id}, Right ankle frame ID: {right_ankle_frame_id}")
    
    num_timesteps = joint_positions.shape[0]
    left_ankle_positions = np.zeros((num_timesteps, 3))
    right_ankle_positions = np.zeros((num_timesteps, 3))
    
    # Initialize velocity arrays if joint velocities are provided
    compute_velocities = joint_velocities is not None
    if compute_velocities:
        left_ankle_velocities = np.zeros((num_timesteps, 3))
        right_ankle_velocities = np.zeros((num_timesteps, 3))
    
    for t in range(num_timesteps):
        # Create configuration vector in Pinocchio order
        q = pin.neutral(model)
        print(f"neutral q: {q}")

        # Map joint positions from gait order to Pinocchio order
        for gait_idx, pin_idx in joint_mapping.items():
            q[pin_idx + 5] = joint_positions[t, gait_idx]
        
        # Create velocity vector if velocities are provided
        if compute_velocities:
            v = np.zeros(model.nv)
            # Map joint velocities from gait order to Pinocchio order
            for gait_idx, pin_idx in joint_mapping.items():
                v[pin_idx + 4] = joint_velocities[t, gait_idx]
        
        # If we have projected gravity data, decode it to base orientation
        if projected_gravity is not None:
            roll, pitch = decode_projected_gravity_to_orientation(projected_gravity[t])
            print(f"Roll: {roll}, Pitch: {pitch}")

            # Set base orientation (assuming the first 7 DOF are: [x, y, z, qx, qy, qz, qw])
            # We'll set the base position to origin and orientation from roll/pitch
            # Convert roll/pitch to quaternion (yaw = 0)
            rotation = st.Rotation.from_euler('xyz', [roll, pitch, 0.0])
            quat = rotation.as_quat()  # Returns [qx, qy, qz, qw]

            print(f"Rotation quaternion: {quat}")
            # Set base pose in configuration vector
            q[0] = 0.0  # base x position
            q[1] = 0.0  # base y position
            q[2] = 0.0  # base z position
            q[3] = quat[0]  # base qx
            q[4] = quat[1]  # base qy
            q[5] = quat[2]  # base qy
            q[6] = quat[3]  # base qw
        
        # Compute forward kinematics
        if compute_velocities:
            # Compute both positions and velocities
            pin.forwardKinematics(model, data, q, v)
        else:
            # Compute only positions
            pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get ankle positions
        left_ankle_positions[t] = data.oMf[left_ankle_frame_id].translation
        right_ankle_positions[t] = data.oMf[right_ankle_frame_id].translation
        
        # Get ankle velocities if computing them
        if compute_velocities:
            # Compute frame velocities
            left_ankle_velocities[t] = pin.getFrameVelocity(model, data, left_ankle_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear  # TODO: Check the frame
            right_ankle_velocities[t] = pin.getFrameVelocity(model, data, right_ankle_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear  # TODO: Check the frame

    result = {
        'left_ankle_positions': left_ankle_positions,
        'right_ankle_positions': right_ankle_positions,
        'phase': phase
    }
    
    if compute_velocities:
        result['left_ankle_velocities'] = left_ankle_velocities
        result['right_ankle_velocities'] = right_ankle_velocities
    
    return result


def compute_swing_ankle_trajectories(ankle_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute swing ankle position relative to stance ankle.
    
    Uses current stance ankle position during single support phase,
    and the last single support stance position during flight phase.
    """
    if ankle_data is None:
        raise ValueError("ankle_data is None")
    
    required_keys = ['phase', 'left_ankle_positions', 'right_ankle_positions']
    for key in required_keys:
        if key not in ankle_data:
            raise KeyError(f"Required key '{key}' not found in ankle_data. Available keys: {list(ankle_data.keys())}")
    
    # TODO: I really should be using the odometry to compute these - the x and y satisfy the assumptions for the most part, z does not.
    phase = ankle_data['phase']
    left_positions = ankle_data['left_ankle_positions']
    right_positions = ankle_data['right_ankle_positions']
    
    # Check if velocities are available
    compute_velocities = 'left_ankle_velocities' in ankle_data and 'right_ankle_velocities' in ankle_data
    if compute_velocities:
        left_velocities = ankle_data['left_ankle_velocities']
        right_velocities = ankle_data['right_ankle_velocities']
    
    num_timesteps = len(phase)
    swing_relative_positions = np.zeros((num_timesteps, 3))
    if compute_velocities:
        swing_relative_velocities = np.zeros((num_timesteps, 3))
    
    # Track the last stance position from single support phase
    last_right_stance_pos = None
    last_left_stance_pos = None
    
    # We need to map phase to gait domains
    # Assuming: phase 0-0.5 is right stance (single support + flight), phase 0.5-1.0 is left stance
    # Single support is the first 15.4% of each half-phase
    single_support_fraction = 0.5 #0.154
    
    for t in range(num_timesteps):
        current_phase = phase[t]
        
        if current_phase < 0.5:
            # Right foot is stance, left foot is swing
            phase_in_half = current_phase / 0.5  # Normalize to 0-1 within this half
            
            if phase_in_half <= single_support_fraction:
                # Single support phase: use current stance ankle position
                current_stance_pos = right_positions[t]
                last_right_stance_pos = current_stance_pos  # Update last known stance position
            else:
                # Flight phase: use last stance position from single support
                current_stance_pos = last_right_stance_pos if last_right_stance_pos is not None else right_positions[t]
            
            # Swing ankle relative to stance
            swing_relative_positions[t][2] = left_positions[t][2] - current_stance_pos[2]   # TODO: Need to fix the z height by using the odometry
            swing_relative_positions[t] = left_positions[t] - right_positions[t]

            # Compute relative velocities if available
            if compute_velocities:
                # For relative motion: v_rel = v_swing - v_stance
                # During single support, stance ankle velocity is used directly
                # During flight, stance velocity is zero (using fixed reference)
                # if phase_in_half <= single_support_fraction:
                #     stance_vel = right_velocities[t]
                # else:
                #     stance_vel = np.zeros(3)  # Fixed reference during flight
                swing_relative_velocities[t] = left_velocities[t] -  right_velocities[t]
            
        else:
            # Left foot is stance, right foot is swing
            phase_in_half = (current_phase - 0.5) / 0.5  # Normalize to 0-1 within this half
            
            if phase_in_half <= single_support_fraction:
                # Single support phase: use current stance ankle position
                current_stance_pos = left_positions[t]
                last_left_stance_pos = current_stance_pos  # Update last known stance position
            else:
                # Flight phase: use last stance position from single support
                current_stance_pos = last_left_stance_pos if last_left_stance_pos is not None else left_positions[t]
            
            # Swing ankle relative to stance
            swing_relative_positions[t] = right_positions[t] - current_stance_pos
            swing_relative_positions[t][2] = right_positions[t][2] - current_stance_pos[2] # TODO: Need to fix the z height by using the odometry
            swing_relative_positions[t] = right_positions[t] - left_positions[t]

            # Compute relative velocities if available
            if compute_velocities:
                # For relative motion: v_rel = v_swing - v_stance
                # During single support, stance ankle velocity is used directly
                # During flight, stance velocity is zero (using fixed reference)
                # if phase_in_half <= single_support_fraction:
                #     stance_vel = left_velocities[t]
                # else:
                #     stance_vel = np.zeros(3)  # Fixed reference during flight
                swing_relative_velocities[t] = right_velocities[t] - left_velocities[t]
    
    result = {
        'swing_relative_x': swing_relative_positions[:, 0],
        'swing_relative_y': swing_relative_positions[:, 1], 
        'swing_relative_z': swing_relative_positions[:, 2],
        'phase': phase
    }
    
    if compute_velocities:
        result['swing_relative_x_vel'] = swing_relative_velocities[:, 0]
        result['swing_relative_y_vel'] = swing_relative_velocities[:, 1]
        result['swing_relative_z_vel'] = swing_relative_velocities[:, 2]
    
    return result


def extract_joint_trajectories(gait_data: Dict[str, Any], joint_names_to_plot: List[str]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Extract position and velocity trajectories for specified joints from both domains in gait data.
    
    For left leg joints, also extracts symmetric right leg data to complete the full gait period.
    
    Returns:
        Dict with structure: {joint_name: {domain: {'positions': np.array, 'velocities': np.array, 'times': np.array}, 'combined': {'positions': np.array, 'velocities': np.array, 'times': np.array}, 'full_cycle': {'positions': np.array, 'velocities': np.array, 'times': np.array}}}
    """
    # Check for domain sequence
    if 'domain_sequence' not in gait_data:
        raise KeyError("'domain_sequence' not found in gait data")
    
    domain_sequence = gait_data['domain_sequence']
    first_domain = domain_sequence[0]
    print(f"Found domains: {domain_sequence}")
    
    # Get joint order from the gait data (first domain)
    if 'joint_order' not in gait_data[first_domain]:
        raise KeyError(f"'joint_order' not found in domain '{first_domain}' of gait data")
    
    all_joint_names = gait_data[first_domain]['joint_order']
    print(f"Using joint order from YAML: {all_joint_names[:5]}...{all_joint_names[-3:]} ({len(all_joint_names)} total joints)")
    
    # Find indices of joints we want to plot
    joint_indices = {}
    for joint_name in joint_names_to_plot:
        if joint_name in all_joint_names:
            joint_indices[joint_name] = all_joint_names.index(joint_name)
        else:
            raise ValueError(f"Joint '{joint_name}' not found in joint names list")
    
    if not joint_indices:
        raise ValueError("No valid joint names found")
    
    trajectories = {}
    
    # Extract trajectories for each joint from each domain
    for joint_name, joint_idx in joint_indices.items():
        trajectories[joint_name] = {}
        
        for domain in domain_sequence:
            if domain not in gait_data:
                raise KeyError(f"Domain '{domain}' not found in gait data")
            
            domain_data = gait_data[domain]
            
            if 'q' not in domain_data:
                raise KeyError(f"No joint position data ('q') found for domain '{domain}'")
            if 'v' not in domain_data:
                raise KeyError(f"No joint velocity data ('v') found for domain '{domain}'")
            
            q_data = domain_data['q']
            v_data = domain_data['v']
            
            positions = []
            velocities = []
            times = []
            
            # Get domain period for time calculation
            if 'T' not in domain_data:
                raise KeyError(f"Period 'T' not found for domain '{domain}'")
            domain_period = domain_data['T'][0]
            
            time_steps = sorted(q_data.keys(), key=int)
            num_steps = len(time_steps)
            
            for i, time_step in enumerate(time_steps):
                # Calculate actual time within this domain
                time_in_domain = (i / (num_steps - 1)) * domain_period if num_steps > 1 else 0.0
                times.append(time_in_domain)
                
                # Extract positions
                q_values = q_data[time_step]
                if joint_idx >= len(q_values):
                    raise IndexError(f"Joint index {joint_idx} out of range for {domain} position time step {time_step} (only {len(q_values)} joints available)")
                positions.append(q_values[joint_idx + 7])
                
                # Extract velocities
                if time_step not in v_data:
                    raise KeyError(f"Velocity data missing for time step {time_step} in domain '{domain}'")
                v_values = v_data[time_step]
                if joint_idx >= len(v_values):
                    raise IndexError(f"Joint index {joint_idx} out of range for {domain} velocity time step {time_step} (only {len(v_values)} joints available)")
                velocities.append(v_values[joint_idx + 6])
            
            trajectories[joint_name][domain] = {
                'positions': np.array(positions),
                'velocities': np.array(velocities),
                'times': np.array(times)
            }
            print(f"Extracted {len(positions)} position/velocity pairs for joint '{joint_name}' in domain '{domain}'")
        
        # Combine trajectories from both domains
        all_positions = []
        all_velocities = []
        all_times = []
        cumulative_time = 0.0
        
        for domain in domain_sequence:
            domain_positions = trajectories[joint_name][domain]['positions']
            domain_velocities = trajectories[joint_name][domain]['velocities']
            domain_times = trajectories[joint_name][domain]['times'] + cumulative_time
            
            all_positions.extend(domain_positions)
            all_velocities.extend(domain_velocities)
            all_times.extend(domain_times)
            
            # Update cumulative time for next domain
            if 'T' in gait_data[domain]:
                cumulative_time += gait_data[domain]['T'][0]
        
        trajectories[joint_name]['combined'] = {
            'positions': np.array(all_positions),
            'velocities': np.array(all_velocities),
            'times': np.array(all_times)
        }
        print(f"Combined trajectory for '{joint_name}': {len(all_positions)} total position/velocity pairs")
    
    # Add symmetric data to complete full gait period for both left and right joints
    for joint_name, joint_idx in joint_indices.items():
        # Check if this joint has a symmetric counterpart
        symmetric_joint_name = get_symmetric_joint_name(joint_name)
        
        # Only add symmetric data if the joint has a left/right counterpart and it's different from itself
        if symmetric_joint_name != joint_name and symmetric_joint_name in all_joint_names:
            symmetric_joint_idx = all_joint_names.index(symmetric_joint_name)
            
            # Extract symmetric joint data from all domains
            symmetric_positions = []
            symmetric_velocities = []
            symmetric_times = []
            
            # Calculate symmetric time offset (starts after combined trajectory)
            half_cycle_duration = trajectories[joint_name]['combined']['times'][-1]
            symmetric_cumulative_time = half_cycle_duration
            
            for domain in domain_sequence:
                domain_data = gait_data[domain]
                q_data = domain_data['q']
                v_data = domain_data['v']
                domain_period = domain_data['T'][0]
                
                time_steps = sorted(q_data.keys(), key=int)
                num_steps = len(time_steps)
                
                for i, time_step in enumerate(time_steps):
                    # Calculate time for symmetric trajectory
                    time_in_domain = (i / (num_steps - 1)) * domain_period if num_steps > 1 else 0.0
                    symmetric_times.append(symmetric_cumulative_time + time_in_domain)
                    
                    # Extract symmetric joint positions and velocities
                    q_values = q_data[time_step]
                    v_values = v_data[time_step]
                    
                    if symmetric_joint_idx < len(q_values) and symmetric_joint_idx < len(v_values):
                        pos_val = q_values[symmetric_joint_idx + 7]
                        vel_val = v_values[symmetric_joint_idx + 6]
                        
                        # Flip sign for specific joint types that should be mirrored
                        if any(joint_type in joint_name for joint_type in ['hip_yaw', 'hip_roll', 'ankle_roll', "shoulder_roll"]):
                            pos_val = -pos_val
                            vel_val = -vel_val
                        
                        symmetric_positions.append(pos_val)
                        symmetric_velocities.append(vel_val)
                
                symmetric_cumulative_time += domain_period
            
            # Create full cycle trajectory (original + symmetric)
            full_positions = list(trajectories[joint_name]['combined']['positions'])
            full_velocities = list(trajectories[joint_name]['combined']['velocities'])
            full_times = list(trajectories[joint_name]['combined']['times'])
            full_positions.extend(symmetric_positions)
            full_velocities.extend(symmetric_velocities)
            full_times.extend(symmetric_times)
            
            trajectories[joint_name]['full_cycle'] = {
                'positions': np.array(full_positions),
                'velocities': np.array(full_velocities),
                'times': np.array(full_times)
            }
            print(f"Added symmetric data for '{joint_name}' using '{symmetric_joint_name}': {len(symmetric_positions)} symmetric pairs, {len(full_positions)} total full cycle pairs")
        else:
            # For joints without symmetry (like waist_yaw), use combined data as full cycle
            trajectories[joint_name]['full_cycle'] = trajectories[joint_name]['combined'].copy()
            if symmetric_joint_name == joint_name:
                print(f"Joint '{joint_name}' has no symmetric counterpart, using half cycle as full cycle")
            else:
                print(f"Warning: Symmetric joint '{symmetric_joint_name}' not found for '{joint_name}', using combined data as full cycle")
    
    return trajectories


def plot_phase_space_orbits(trajectories: Dict[str, Dict[str, Dict[str, np.ndarray]]], gait_data: Dict[str, Any], 
                           hardware_trajectories: Optional[Dict[str, Dict[str, np.ndarray]]], save_dir: str):
    """Plot phase space orbits (position vs velocity) for the joint trajectories."""
    # Set LaTeX configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
    })
    
    num_joints = len(trajectories)
    if num_joints == 0:
        raise ValueError("No trajectories to plot")
    
    # Create subplots - arrange in a single row if 3 or fewer joints
    if num_joints <= 3:
        fig, axes = plt.subplots(1, num_joints, figsize=(6 * num_joints, 5))
        if num_joints == 1:
            axes = [axes]  # Make it iterable
    else:
        # For more joints, use a 2xN grid
        rows = 2
        cols = (num_joints + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = axes.flatten()
    
    joint_names = list(trajectories.keys())
    
    # Get domain sequence for plotting individual domains
    if 'domain_sequence' not in gait_data:
        raise KeyError("'domain_sequence' not found in gait data")
    
    domain_sequence = gait_data['domain_sequence']
    print(f"Plotting phase space orbits for domains: {domain_sequence}")
    
    for i, joint_name in enumerate(joint_names):
        joint_data = trajectories[joint_name]
        
        # Get full cycle position and velocity data
        full_cycle_data = joint_data['full_cycle']
        full_positions = full_cycle_data['positions']
        full_velocities = full_cycle_data['velocities']
        
        # Get half cycle (combined) data for comparison
        combined_data = joint_data['combined']
        half_positions = combined_data['positions']
        half_velocities = combined_data['velocities']
        
        if len(full_positions) == 0 or len(full_velocities) == 0:
            raise ValueError(f"No valid trajectory data found for joint '{joint_name}'")
        
        # Plot full cycle phase space trajectory (main orbit)
        axes[i].plot(full_positions, full_velocities, 'b-', linewidth=2.5, label='Full Cycle Orbit')
        
        # Plot half cycle trajectory for comparison  
        axes[i].plot(half_positions, half_velocities, 'c--', linewidth=2, alpha=0.7, label='Half Cycle')
        
        # Plot individual domains with different colors
        colors = ['red', 'green', 'purple', 'orange']  # Colors for different domains
        
        for j, domain in enumerate(domain_sequence):
            if domain in joint_data:
                domain_pos = joint_data[domain]['positions']
                domain_vel = joint_data[domain]['velocities']
                
                if len(domain_pos) > 0 and len(domain_vel) > 0:
                    axes[i].plot(domain_pos, domain_vel, '--', linewidth=2, 
                               color=colors[j % len(colors)], alpha=0.7, 
                               label=f'{domain.replace("_", " ").title()}')
        
        # Plot hardware data if available
        if hardware_trajectories and joint_name in hardware_trajectories:
            hw_data = hardware_trajectories[joint_name]
            hw_phase = hw_data['phase']
            hw_positions = hw_data['positions']
            hw_velocities = hw_data['velocities']
            
            # Segment hardware data into individual gait cycles (0-1 phase)
            # Find phase transitions (when phase goes from high to low, indicating cycle restart)
            phase_diff = np.diff(hw_phase)
            cycle_boundaries = [0]  # Start of first cycle
            
            # Find where phase decreases significantly (cycle restart)
            for j in range(len(phase_diff)):
                if phase_diff[j] < -0.5:  # Phase wrapped from ~1 back to ~0
                    cycle_boundaries.append(j + 1)
            cycle_boundaries.append(len(hw_phase))  # End of last cycle
            
            # Plot each individual gait cycle as a separate line in phase space
            for cycle_idx in range(len(cycle_boundaries) - 1):
                start_idx = cycle_boundaries[cycle_idx]
                end_idx = cycle_boundaries[cycle_idx + 1]
                
                cycle_positions = hw_positions[start_idx:end_idx]
                cycle_velocities = hw_velocities[start_idx:end_idx]
                
                # Only plot cycles that have reasonable data
                if len(cycle_positions) > 10:  # Minimum points per cycle
                    # Plot individual cycle in phase space
                    label = 'Hardware Cycles' if cycle_idx == 0 else None  # Only label first cycle
                    axes[i].plot(cycle_positions, cycle_velocities, 'r-', 
                               linewidth=1, alpha=0.2, label=label)
        
        # Mark start and end points for full cycle
        axes[i].plot(full_positions[0], full_velocities[0], 'go', markersize=8, label='Start')
        axes[i].plot(full_positions[-1], full_velocities[-1], 'ro', markersize=8, label='End')
        
        # Mark the transition point (end of half cycle / start of symmetric part)
        half_cycle_length = len(half_positions)
        if half_cycle_length > 0 and half_cycle_length < len(full_positions):
            axes[i].plot(full_positions[half_cycle_length], full_velocities[half_cycle_length], 
                        'mo', markersize=6, label='Half Cycle End')
        
        # Format plot
        axes[i].set_title(f'{joint_name.replace("_", " ").title()}', fontsize=18)
        axes[i].set_xlabel('Position (rad)', fontsize=14)
        axes[i].set_ylabel('Velocity (rad/s)', fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10, loc='best')
        
    
    # Hide any unused subplots
    if num_joints > 3:
        for i in range(num_joints, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot to ctrl_logs directory
    save_path = os.path.join(save_dir, 'phase_space_orbits')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', facecolor='white') 
    plt.savefig(f"{save_path}.svg", bbox_inches='tight', facecolor='white')
    print(f"Saved phase space orbit plots to: {save_path}.[png/pdf/svg]")
    
    # plt.show()


def plot_joint_angles_over_time(trajectories: Dict[str, Dict[str, Dict[str, np.ndarray]]], gait_data: Dict[str, Any], 
                               hardware_trajectories: Optional[Dict[str, Dict[str, np.ndarray]]], save_dir: str):
    """Plot joint angles over phase (0-1) for the joint trajectories."""
    # Set LaTeX configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
    })
    
    num_joints = len(trajectories)
    if num_joints == 0:
        raise ValueError("No trajectories to plot")
    
    # Create subplots - arrange in a single row if 3 or fewer joints
    if num_joints <= 3:
        fig, axes = plt.subplots(1, num_joints, figsize=(6 * num_joints, 3))
        if num_joints == 1:
            axes = [axes]  # Make it iterable
    else:
        # For more joints, use a 2xN grid
        rows = 2
        cols = (num_joints + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
        axes = axes.flatten()
    
    joint_names = list(trajectories.keys())
    
    # Get domain sequence for vertical lines
    if 'domain_sequence' not in gait_data:
        raise KeyError("'domain_sequence' not found in gait data")
    
    domain_sequence = gait_data['domain_sequence']
    print(f"Plotting phase series for domains: {domain_sequence}")
    
    # Calculate domain phase boundaries for vertical lines
    total_period = 0
    for domain in domain_sequence:
        if 'T' in gait_data[domain]:
            total_period += gait_data[domain]['T'][0]
    
    domain_phase_boundaries = []
    cumulative_period = 0.0
    for domain in domain_sequence[:-1]:  # Skip last domain
        if 'T' in gait_data[domain]:
            cumulative_period += gait_data[domain]['T'][0]
            domain_phase_boundaries.append(cumulative_period / total_period)
    
    for i, joint_name in enumerate(joint_names):
        joint_data = trajectories[joint_name]
        
        # Get full cycle time and position data
        full_cycle_data = joint_data['full_cycle']
        full_times = full_cycle_data['times']
        full_positions = full_cycle_data['positions']
        
        # Convert time to phase (0-1 scale)
        if len(full_times) > 0:
            full_phase = full_times / full_times[-1]  # Normalize to 0-1
        else:
            full_phase = np.array([])
        
        # Get half cycle data for comparison
        combined_data = joint_data['combined']
        half_times = combined_data['times']
        half_positions = combined_data['positions']
        
        # Convert half cycle time to phase
        if len(half_times) > 0:
            half_phase = half_times / full_times[-1]  # Normalize using full cycle duration
        else:
            half_phase = np.array([])
        
        if len(full_phase) == 0 or len(full_positions) == 0:
            raise ValueError(f"No valid phase series data found for joint '{joint_name}'")
        
        # Plot full cycle gait library trajectory
        axes[i].plot(full_phase, full_positions, 'b-', linewidth=2.5, label='Gait Library', alpha=0.8)
        
        # Plot hardware data if available
        if hardware_trajectories and joint_name in hardware_trajectories:
            hw_data = hardware_trajectories[joint_name]
            hw_phase = hw_data['phase']
            hw_positions = hw_data['positions']
            
            # Segment hardware data into individual gait cycles (0-1 phase)
            # Find phase transitions (when phase goes from high to low, indicating cycle restart)
            phase_diff = np.diff(hw_phase)
            cycle_boundaries = [0]  # Start of first cycle
            
            # Find where phase decreases significantly (cycle restart)
            for j in range(len(phase_diff)):
                if phase_diff[j] < -0.5:  # Phase wrapped from ~1 back to ~0
                    cycle_boundaries.append(j + 1)
            cycle_boundaries.append(len(hw_phase))  # End of last cycle
            
            # Plot each individual gait cycle as a separate line
            labeled = False
            for cycle_idx in range(len(cycle_boundaries) - 1):
                start_idx = cycle_boundaries[cycle_idx]
                end_idx = cycle_boundaries[cycle_idx + 1]
                
                cycle_phase = hw_phase[start_idx:end_idx]
                cycle_positions = hw_positions[start_idx:end_idx]
                
                # Only plot cycles that have reasonable data
                if len(cycle_phase) > 10:  # Minimum points per cycle
                    # Sort this individual cycle by phase
                    sort_idx = np.argsort(cycle_phase)
                    cycle_phase_sorted = cycle_phase[sort_idx]
                    cycle_positions_sorted = cycle_positions[sort_idx]
                    
                    # Plot individual cycle
                    label = 'Measured' if not labeled else None
                    labeled = True if label == 'Measured' else labeled
                    axes[i].plot(cycle_phase_sorted, cycle_positions_sorted, 'r-', 
                               linewidth=1, alpha=0.2, label=label)
        
        # Add vertical lines to separate domains
        # for phase_boundary in domain_phase_boundaries:
        #     axes[i].axvline(x=phase_boundary, color='gray', linestyle=':', alpha=0.6)
        
        # Format plot
        axes[i].set_title(f'{joint_name.replace("_", " ").title()}', fontsize=22)
        axes[i].set_xlabel('Phase', fontsize=18)
        axes[i].set_ylabel('Joint Angle (rad)', fontsize=18)
        axes[i].tick_params(axis='both', which='major', labelsize=16)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 1)
        axes[i].legend(fontsize=12, loc='best')
        
    
    # Hide any unused subplots
    if num_joints > 3:
        for i in range(num_joints, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot to ctrl_logs directory
    save_path = os.path.join(save_dir, 'joint_angles_time_series')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.svg", bbox_inches='tight', facecolor='white')
    print(f"Saved joint angle time series plots to: {save_path}.[png/pdf/svg]")
    
    plt.show()


def plot_swing_ankle_trajectories(swing_data: Dict[str, np.ndarray], gait_data: Dict[str, Any], save_dir: str, hardware_trajectories: Optional[Dict[str, Dict[str, np.ndarray]]] = None):
    """Plot swing ankle position relative to stance ankle with gait library overlay."""
    # Set LaTeX configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
    })
    
    phase = swing_data['phase']
    swing_x = swing_data['swing_relative_x']
    swing_y = swing_data['swing_relative_y']
    swing_z = swing_data['swing_relative_z']
    
    # Parse and compute gait library swing ankle trajectories
    gait_lib_swing = None
    try:
        print("Computing gait library swing ankle trajectories...")
        ankle_coeffs = parse_ankle_bezier_coefficients(gait_data)
        
        # Create dense phase array for smooth gait library curves
        phase_dense = np.linspace(0, 1, 1000)
        gait_lib_ankles = compute_gait_library_ankle_positions(ankle_coeffs, phase_dense)
        
        # Compute swing relative to stance using the same logic as hardware
        gait_lib_swing_pos = np.zeros((len(phase_dense), 3))
        
        for t in range(len(phase_dense)):
            gait_lib_swing_pos[t] = gait_lib_ankles['swing_ankle_positions'][t]

        gait_lib_swing = {
            'phase': phase_dense/2,
            'swing_relative_x': gait_lib_swing_pos[:, 0],
            'swing_relative_y': gait_lib_swing_pos[:, 1],
            'swing_relative_z': gait_lib_swing_pos[:, 2]
        }
        print("Successfully computed gait library swing ankle trajectories")
        
    except Exception as e:
        print(f"Warning: Could not compute gait library swing trajectories: {e}")


    # Create 4 subplots for X, Y, left shoulder pitch, and right shoulder pitch
    fig, axes = plt.subplots(1, 4, figsize=(32, 2.8))
    
    # Segment data into individual gait cycles
    phase_diff = np.diff(phase)
    cycle_boundaries = [0]
    
    # Find where phase decreases significantly (cycle restart)
    for j in range(len(phase_diff)):
        if phase_diff[j] < -0.5:  # Phase wrapped from ~1 back to ~0
            cycle_boundaries.append(j + 1)
    cycle_boundaries.append(len(phase))
    
    # Plot each coordinate and both shoulder pitch joints
    coordinates = [('x', swing_x, 'm'), ('y', swing_y, 'm'), ('left_joint', None, 'rad'), ('right_joint', None, 'rad')] #('z', swing_z, 'm') # Removed z until the plotting is fixed (see odom issue)

    print(f"hardware traj: {hardware_trajectories.keys()}")

    for coord_idx, (coord_name, coord_data, unit) in enumerate(coordinates):
        print(f"Plotting {coord_name}...")
        # Handle joint data case
        if coord_name == 'left_joint':
            if hardware_trajectories and 'left_shoulder_pitch_joint' in hardware_trajectories:
                coord_data = hardware_trajectories['left_shoulder_pitch_joint']['positions']
                phase = hardware_trajectories['left_shoulder_pitch_joint']['phase']
        elif coord_name == 'right_joint':
            if hardware_trajectories and 'right_shoulder_pitch_joint' in hardware_trajectories:
                coord_data = hardware_trajectories['right_shoulder_pitch_joint']['positions']
                phase = hardware_trajectories['right_shoulder_pitch_joint']['phase']
        
        # Plot gait library trajectory if available
        if coord_name == 'left_joint':
            # For left joint data, we need to extract from gait_data trajectories
            try:
                # Extract joint trajectories from gait data
                joint_trajectories = extract_joint_trajectories(gait_data, ['left_shoulder_pitch_joint'])
                if 'left_shoulder_pitch_joint' in joint_trajectories:
                    joint_data = joint_trajectories['left_shoulder_pitch_joint']['full_cycle']
                    gait_times = joint_data['times']
                    gait_positions = joint_data['positions']
                    # Convert to phase (0-1)
                    gait_phase = gait_times / gait_times[-1] if len(gait_times) > 0 else np.array([])
                    
                    axes[coord_idx].plot(gait_phase, gait_positions, 'b-',
                                       linewidth=2, alpha=0.8, label='Gait Library')
            except Exception as e:
                print(f"Warning: Could not plot gait library data for left joint: {e}")
        elif coord_name == 'right_joint':
            # For right joint data, we need to extract from gait_data trajectories
            try:
                # Extract joint trajectories from gait data
                joint_trajectories = extract_joint_trajectories(gait_data, ['right_shoulder_pitch_joint'])
                if 'right_shoulder_pitch_joint' in joint_trajectories:
                    joint_data = joint_trajectories['right_shoulder_pitch_joint']['full_cycle']
                    gait_times = joint_data['times']
                    gait_positions = joint_data['positions']
                    # Convert to phase (0-1)
                    gait_phase = gait_times / gait_times[-1] if len(gait_times) > 0 else np.array([])
                    
                    axes[coord_idx].plot(gait_phase, gait_positions, 'b-',
                                       linewidth=2, alpha=0.8, label='Gait Library')
            except Exception as e:
                print(f"Warning: Could not plot gait library data for right joint: {e}")
        elif gait_lib_swing is not None:
            if coord_name == 'x':
                gait_lib_coord = gait_lib_swing['swing_relative_x']
            elif coord_name == 'y':
                gait_lib_coord = gait_lib_swing['swing_relative_y']
            else:  # Z
                gait_lib_coord = gait_lib_swing['swing_relative_z']

            axes[coord_idx].plot(gait_lib_swing['phase'], gait_lib_coord, 'b-',
                           linewidth=2, alpha=0.8, label='Gait Library')

            # Plot a second one for the other side of the robot
            if coord_name == 'x':
                gait_lib_coord = gait_lib_swing['swing_relative_x']
            elif coord_name == 'y':
                gait_lib_coord = -1 * gait_lib_swing['swing_relative_y']
            else:  # Z
                gait_lib_coord = gait_lib_swing['swing_relative_z']

            axes[coord_idx].plot(gait_lib_swing['phase'] + 0.5, gait_lib_coord, 'b-',
                           linewidth=2, alpha=0.8, label=None)
            if coord_name == 'y':
                axes[coord_idx].plot([0.5, 0.5], [gait_lib_coord[-1], -gait_lib_coord[0]], 'b-',
                               linewidth=2, alpha=0.8)

            axes[coord_idx].plot([0.5, 0.5], [gait_lib_coord[-1], gait_lib_coord[0]], 'b-',
                           linewidth=2, alpha=0.8)

        # Plot each individual hardware gait cycle
        labeled = False
        for cycle_idx in range(len(cycle_boundaries) - 1):
            start_idx = cycle_boundaries[cycle_idx]
            end_idx = cycle_boundaries[cycle_idx + 1]
            
            cycle_phase = phase[start_idx:end_idx]
            cycle_coord = coord_data[start_idx:end_idx]
            
            # Only plot cycles that have reasonable data
            if len(cycle_phase) > 10:
                # Sort this individual cycle by phase
                sort_idx = np.argsort(cycle_phase)
                cycle_phase_sorted = cycle_phase[sort_idx]
                cycle_coord_sorted = cycle_coord[sort_idx]
                
                # Plot individual cycle
                label = 'Measured' if not labeled else None
                labeled = True if label == 'Measured' else labeled
                axes[coord_idx].plot(cycle_phase_sorted, cycle_coord_sorted, 'r-', 
                                   linewidth=1, alpha=0.3, label=label)

        # Format plot
        if coord_name == 'left_joint':
            axes[coord_idx].set_title('Left Shoulder Pitch Joint', fontsize=24)
            axes[coord_idx].set_ylabel(f'Joint Angle ({unit})', fontsize=22)
        elif coord_name == 'right_joint':
            axes[coord_idx].set_title('Right Shoulder Pitch Joint', fontsize=24)
            axes[coord_idx].set_ylabel(f'Joint Angle ({unit})', fontsize=22)
        else:
            axes[coord_idx].set_title(f'Swing Ankle ${coord_name}$ Virtual Constraint'.format(coord_name), fontsize=24)
            axes[coord_idx].set_ylabel(f'${coord_name}$ Position ({unit})', fontsize=22)
        axes[coord_idx].set_xlabel('Phase', fontsize=22)
        axes[coord_idx].tick_params(axis='both', which='major', labelsize=22)
        axes[coord_idx].grid(True, alpha=0.3)
        axes[coord_idx].set_xlim(0, 1)
        axes[coord_idx].legend(fontsize=18, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot to ctrl_logs directory
    save_path = os.path.join(save_dir, 'swing_ankle_trajectories')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.svg", bbox_inches='tight', facecolor='white')
    print(f"Saved swing ankle trajectory plots to: {save_path}.[png/pdf/svg]")
    
    plt.show()


def plot_swing_ankle_phase_plots(swing_data: Dict[str, np.ndarray], gait_data: Dict[str, Any], save_dir: str, hardware_trajectories: Optional[Dict[str, Dict[str, np.ndarray]]] = None):
    """Plot phase plots (position vs velocity) for swing ankle trajectories and left shoulder pitch joint."""
    # Set LaTeX configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
    })
    
    phase = swing_data['phase']
    swing_x = swing_data['swing_relative_x']
    swing_y = swing_data['swing_relative_y']
    swing_z = swing_data['swing_relative_z']
    
    # Parse and compute gait library swing ankle trajectories
    gait_lib_swing = None
    try:
        print("Computing gait library swing ankle trajectories for phase plots...")
        ankle_coeffs = parse_ankle_bezier_coefficients(gait_data)
        
        # Create dense phase array for smooth gait library curves
        phase_dense = np.linspace(0, 1, 1000)
        gait_lib_ankles = compute_gait_library_ankle_positions(ankle_coeffs, phase_dense, compute_velocities=True)
        
        # Compute swing relative to stance using the same logic as hardware
        gait_lib_swing_pos = np.zeros((len(phase_dense), 3))
        gait_lib_swing_vel = np.zeros((len(phase_dense), 3))
        
        for t in range(len(phase_dense)):
            gait_lib_swing_pos[t] = gait_lib_ankles['swing_ankle_positions'][t]
            gait_lib_swing_vel[t] = gait_lib_ankles['swing_ankle_velocities'][t]

        gait_lib_swing = {
            'phase': phase_dense/2,
            'swing_relative_x': gait_lib_swing_pos[:, 0],
            'swing_relative_y': gait_lib_swing_pos[:, 1],
            'swing_relative_z': gait_lib_swing_pos[:, 2],
            'swing_relative_x_vel': gait_lib_swing_vel[:, 0],
            'swing_relative_y_vel': gait_lib_swing_vel[:, 1],
            'swing_relative_z_vel': gait_lib_swing_vel[:, 2]
        }
        print("Successfully computed gait library swing ankle trajectories for phase plots")
        
    except Exception as e:
        print(f"Warning: Could not compute gait library swing trajectories for phase plots: {e}")

    # Create 4 subplots for X, Y, Z phase plots, and left shoulder pitch
    fig, axes = plt.subplots(1, 4, figsize=(24, 2.5))
    
    # Segment data into individual gait cycles
    phase_diff = np.diff(phase)
    cycle_boundaries = [0]
    
    # Find where phase decreases significantly (cycle restart)
    for j in range(len(phase_diff)):
        if phase_diff[j] < -0.5:  # Phase wrapped from ~1 back to ~0
            cycle_boundaries.append(j + 1)
    cycle_boundaries.append(len(phase))
    
    # Plot each coordinate and left shoulder pitch joint
    coordinates = [('x', swing_x, 'm/s'), ('y', swing_y, 'm/s'), ('z', swing_z, 'm/s'), ('joint', None, 'rad/s')]
    
    for coord_idx, (coord_name, coord_data, vel_unit) in enumerate(coordinates):
        # Handle joint data case
        if coord_name == 'joint':
            if hardware_trajectories and 'left_shoulder_pitch_joint' in hardware_trajectories:
                coord_data = hardware_trajectories['left_shoulder_pitch_joint']['positions']
                coord_vel_data = hardware_trajectories['left_shoulder_pitch_joint']['velocities']
                phase = hardware_trajectories['left_shoulder_pitch_joint']['phase']
        else:
            # Use precomputed relative velocities from forward kinematics if available
            if coord_name == 'x' and 'swing_relative_x_vel' in swing_data:
                coord_vel_data = swing_data['swing_relative_x_vel']
            elif coord_name == 'y' and 'swing_relative_y_vel' in swing_data:
                coord_vel_data = swing_data['swing_relative_y_vel']
            elif coord_name == 'z' and 'swing_relative_z_vel' in swing_data:
                coord_vel_data = swing_data['swing_relative_z_vel']
            else:
                # Fall back to numerical differentiation for swing ankle coordinates
                coord_vel_data = np.gradient(coord_data)
        
        # Plot gait library trajectory if available
        if coord_name == 'joint':
            # For joint data, we need to extract from gait_data trajectories
            try:
                # Extract joint trajectories from gait data
                joint_trajectories = extract_joint_trajectories(gait_data, ['left_shoulder_pitch_joint'])
                if 'left_shoulder_pitch_joint' in joint_trajectories:
                    joint_data = joint_trajectories['left_shoulder_pitch_joint']['full_cycle']
                    gait_positions = joint_data['positions']
                    gait_velocities = joint_data['velocities']
                    
                    axes[coord_idx].plot(gait_positions, gait_velocities, 'b-',
                                       linewidth=2, alpha=0.8, label='Gait Library')
            except Exception as e:
                print(f"Warning: Could not plot gait library data for joint phase plot: {e}")
        elif gait_lib_swing is not None:
            if coord_name == 'x':
                gait_lib_coord = gait_lib_swing['swing_relative_x']
                gait_lib_vel = gait_lib_swing['swing_relative_x_vel']
            elif coord_name == 'y':
                gait_lib_coord = gait_lib_swing['swing_relative_y']
                gait_lib_vel = gait_lib_swing['swing_relative_y_vel']
            else:  # Z
                gait_lib_coord = gait_lib_swing['swing_relative_z']
                gait_lib_vel = gait_lib_swing['swing_relative_z_vel']
            
            axes[coord_idx].plot(gait_lib_coord, gait_lib_vel, 'b-',
                               linewidth=2, alpha=0.8, label='Gait Library')

            # Plot a second one for the other side of the robot
            if coord_name == 'x':
                gait_lib_coord2 = gait_lib_swing['swing_relative_x']
                gait_lib_vel2 = gait_lib_swing['swing_relative_x_vel']
            elif coord_name == 'y':
                gait_lib_coord2 = -1 * gait_lib_swing['swing_relative_y']
                gait_lib_vel2 = -1 * gait_lib_swing['swing_relative_y_vel']  # Sign flip velocity too
            else:  # Z
                gait_lib_coord2 = gait_lib_swing['swing_relative_z']
                gait_lib_vel2 = gait_lib_swing['swing_relative_z_vel']
                
            axes[coord_idx].plot(gait_lib_coord2, gait_lib_vel2, 'b-',
                               linewidth=2, alpha=0.8, label=None)
        
        # Plot each individual hardware gait cycle
        labeled = False
        for cycle_idx in range(len(cycle_boundaries) - 1):
            start_idx = cycle_boundaries[cycle_idx]
            end_idx = cycle_boundaries[cycle_idx + 1]
            
            cycle_phase = phase[start_idx:end_idx]
            cycle_coord = coord_data[start_idx:end_idx]
            cycle_vel = coord_vel_data[start_idx:end_idx]
            
            # Only plot cycles that have reasonable data
            if len(cycle_phase) > 10:
                # Plot individual cycle in phase space (position vs velocity)
                label = 'Measured' if not labeled else None
                labeled = True if label == 'Measured' else labeled
                axes[coord_idx].plot(cycle_coord, cycle_vel, 'r-', 
                                   linewidth=1, alpha=0.3, label=label)

        # Format plot
        if coord_name == 'joint':
            axes[coord_idx].set_title('Left Shoulder Pitch Joint', fontsize=18)
            axes[coord_idx].set_xlabel('Joint Angle (rad)', fontsize=14)
            axes[coord_idx].set_ylabel(f'Joint Velocity ({vel_unit})', fontsize=14)
        else:
            axes[coord_idx].set_title(f'Swing Ankle ${coord_name}$ Phase Plot', fontsize=18)
            axes[coord_idx].set_xlabel(f'${coord_name}$ Position (m)', fontsize=14)
            axes[coord_idx].set_ylabel(f'${coord_name}$ Velocity ({vel_unit})', fontsize=14)
        axes[coord_idx].tick_params(axis='both', which='major', labelsize=10)
        axes[coord_idx].grid(True, alpha=0.3)
        axes[coord_idx].legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save plot to ctrl_logs directory
    save_path = os.path.join(save_dir, 'swing_ankle_phase_plots')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.svg", bbox_inches='tight', facecolor='white')
    print(f"Saved swing ankle phase plots to: {save_path}.[png/pdf/svg]")
    
    plt.show()


def plot_ankle_positions_debug(ankle_data: Dict[str, np.ndarray], save_dir: str):
    """Debug plot showing absolute positions of both ankles over time."""
    # Set LaTeX configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
    })
    
    phase = ankle_data['phase']
    left_positions = ankle_data['left_ankle_positions']
    right_positions = ankle_data['right_ankle_positions']
    
    # Create 6 subplots: 2 rows (left/right ankles), 3 columns (X/Y/Z)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    # Segment data into individual gait cycles
    phase_diff = np.diff(phase)
    cycle_boundaries = [0]
    
    # Find where phase decreases significantly (cycle restart)
    for j in range(len(phase_diff)):
        if phase_diff[j] < -0.5:  # Phase wrapped from ~1 back to ~0
            cycle_boundaries.append(j + 1)
    cycle_boundaries.append(len(phase))
    
    # Plot data for each ankle and coordinate
    ankles = [('Left Ankle', left_positions, 0), ('Right Ankle', right_positions, 1)]
    coordinates = [('X', 0, 'm'), ('Y', 1, 'm'), ('Z', 2, 'm')]
    
    for ankle_name, ankle_positions, row in ankles:
        for coord_name, coord_idx, unit in coordinates:
            coord_data = ankle_positions[:, coord_idx]
            
            # Plot each individual gait cycle
            labeled = False
            for cycle_idx in range(len(cycle_boundaries) - 1):
                start_idx = cycle_boundaries[cycle_idx]
                end_idx = cycle_boundaries[cycle_idx + 1]
                
                cycle_phase = phase[start_idx:end_idx]
                cycle_coord = coord_data[start_idx:end_idx]
                
                # Only plot cycles that have reasonable data
                if len(cycle_phase) > 10:
                    # Sort this individual cycle by phase
                    sort_idx = np.argsort(cycle_phase)
                    cycle_phase_sorted = cycle_phase[sort_idx]
                    cycle_coord_sorted = cycle_coord[sort_idx]
                    
                    # Plot individual cycle
                    label = 'Measured' if not labeled else None
                    labeled = True if label == 'Measured' else labeled
                    axes[row, coord_idx].plot(cycle_phase_sorted, cycle_coord_sorted, 'b-', 
                                            linewidth=1, alpha=0.4, label=label)
            
            # Add vertical line at phase 0.5 (stance switch)
            axes[row, coord_idx].axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, label='Stance Switch')
            
            # Format plot
            axes[row, coord_idx].set_title(f'{ankle_name} {coord_name} Position', fontsize=12)
            axes[row, coord_idx].set_xlabel('Phase', fontsize=10)
            axes[row, coord_idx].set_ylabel(f'{coord_name} Position ({unit})', fontsize=10)
            axes[row, coord_idx].tick_params(axis='both', which='major', labelsize=9)
            axes[row, coord_idx].grid(True, alpha=0.3)
            axes[row, coord_idx].set_xlim(0, 1)
            if cycle_idx == 0 and coord_idx == 0:  # Only add legend to first subplot
                axes[row, coord_idx].legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    
    # Save plot to ctrl_logs directory
    save_path = os.path.join(save_dir, 'ankle_positions_debug')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.svg", bbox_inches='tight', facecolor='white')
    print(f"Saved ankle positions debug plots to: {save_path}.[png/pdf/svg]")
    
    plt.show()


def plot_ankle_velocities_comparison(ankle_data: Dict[str, np.ndarray], gait_data: Dict[str, Any], save_dir: str):
    """Plot comparison of gait library and hardware ankle velocities over phase."""
    # Set LaTeX configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
    })
    
    phase = ankle_data['phase']
    left_velocities = ankle_data['left_ankle_velocities']
    right_velocities = ankle_data['right_ankle_velocities']
    
    # Compute relative velocities for hardware data
    # During phase 0-0.5: left is swing, right is stance
    # During phase 0.5-1.0: right is swing, left is stance
    num_timesteps = len(phase)
    swing_relative_velocities = np.zeros((num_timesteps, 3))
    
    for t in range(num_timesteps):
        current_phase = phase[t]
        
        if current_phase < 0.5:
            # Left foot is swing, right foot is stance
            # Relative velocity = swing_velocity - stance_velocity
            swing_relative_velocities[t] = left_velocities[t] - right_velocities[t]
        else:
            # Right foot is swing, left foot is stance  
            # Relative velocity = swing_velocity - stance_velocity
            swing_relative_velocities[t] = right_velocities[t] - left_velocities[t]
    
    # Parse and compute gait library ankle velocities
    gait_lib_velocities = None
    try:
        print("Computing gait library ankle velocities for comparison...")
        ankle_coeffs = parse_ankle_bezier_coefficients(gait_data)
        
        # Create dense phase array for smooth gait library curves
        phase_dense = np.linspace(0, 1, 1000)
        gait_lib_ankles = compute_gait_library_ankle_positions(ankle_coeffs, phase_dense, compute_velocities=True)
        
        gait_lib_velocities = {
            'phase': phase_dense,
            'swing_ankle_velocities': gait_lib_ankles['swing_ankle_velocities']
        }
        print("Successfully computed gait library ankle velocities")
        
    except Exception as e:
        print(f"Warning: Could not compute gait library ankle velocities: {e}")
    
    # Create 3 subplots: 1 row (swing ankle relative velocities), 3 columns (X/Y/Z)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Segment hardware data into individual gait cycles
    phase_diff = np.diff(phase)
    cycle_boundaries = [0]
    
    # Find where phase decreases significantly (cycle restart)
    for j in range(len(phase_diff)):
        if phase_diff[j] < -0.5:  # Phase wrapped from ~1 back to ~0
            cycle_boundaries.append(j + 1)
    cycle_boundaries.append(len(phase))
    
    # Plot data for each coordinate
    coordinates = [('X', 0, 'm/s'), ('Y', 1, 'm/s'), ('Z', 2, 'm/s')]
    
    for coord_name, coord_idx, unit in coordinates:
        coord_data = swing_relative_velocities[:, coord_idx]
        
        # Plot gait library velocity if available
        if gait_lib_velocities is not None:
            gait_lib_coord = gait_lib_velocities['swing_ankle_velocities'][:, coord_idx]
            gait_lib_phase = gait_lib_velocities['phase']
            
            # Plot the swing trajectory twice to cover full gait cycle
            # First half (0-0.5): left foot swing phase
            axes[coord_idx].plot(gait_lib_phase/2, gait_lib_coord, 'b-',
                                linewidth=2, alpha=0.8, label='Gait Library')
            # Second half (0.5-1.0): right foot swing phase  
            axes[coord_idx].plot(gait_lib_phase/2 + 0.5, gait_lib_coord, 'b-',
                                linewidth=2, alpha=0.8, label=None)
        
        # Plot each individual hardware gait cycle
        labeled = False
        for cycle_idx in range(len(cycle_boundaries) - 1):
            start_idx = cycle_boundaries[cycle_idx]
            end_idx = cycle_boundaries[cycle_idx + 1]
            
            cycle_phase = phase[start_idx:end_idx]
            cycle_coord = coord_data[start_idx:end_idx]
            
            # Only plot cycles that have reasonable data
            if len(cycle_phase) > 10:
                # Sort this individual cycle by phase
                sort_idx = np.argsort(cycle_phase)
                cycle_phase_sorted = cycle_phase[sort_idx]
                cycle_coord_sorted = cycle_coord[sort_idx]
                
                # Plot individual cycle
                label = 'Hardware' if not labeled else None
                labeled = True if label == 'Hardware' else labeled
                axes[coord_idx].plot(cycle_phase_sorted, cycle_coord_sorted, 'r-', 
                                    linewidth=1, alpha=0.3, label=label)
        
        # Add vertical line at phase 0.5 (stance switch)
        axes[coord_idx].axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, label='Stance Switch')
        
        # Format plot
        axes[coord_idx].set_title(f'Swing Ankle Relative {coord_name} Velocity', fontsize=12)
        axes[coord_idx].set_xlabel('Phase', fontsize=10)
        axes[coord_idx].set_ylabel(f'{coord_name} Velocity ({unit})', fontsize=10)
        axes[coord_idx].tick_params(axis='both', which='major', labelsize=9)
        axes[coord_idx].grid(True, alpha=0.3)
        axes[coord_idx].set_xlim(0, 1)
        axes[coord_idx].legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    
    # Save plot to ctrl_logs directory
    save_path = os.path.join(save_dir, 'ankle_relative_velocities_comparison')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.svg", bbox_inches='tight', facecolor='white')
    print(f"Saved ankle relative velocities comparison plots to: {save_path}.[png/pdf/svg]")
    
    plt.show()


def plot_ankle_velocities_debug(ankle_data: Dict[str, np.ndarray], save_dir: str):
    """Debug plot showing absolute velocities of both ankles over time."""
    # Set LaTeX configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
    })
    
    phase = ankle_data['phase']
    left_velocities = ankle_data['left_ankle_velocities']
    right_velocities = ankle_data['right_ankle_velocities']
    
    # Create 6 subplots: 2 rows (left/right ankles), 3 columns (X/Y/Z)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    # Segment data into individual gait cycles
    phase_diff = np.diff(phase)
    cycle_boundaries = [0]
    
    # Find where phase decreases significantly (cycle restart)
    for j in range(len(phase_diff)):
        if phase_diff[j] < -0.5:  # Phase wrapped from ~1 back to ~0
            cycle_boundaries.append(j + 1)
    cycle_boundaries.append(len(phase))
    
    # Plot data for each ankle and coordinate
    ankles = [('Left Ankle', left_velocities, 0), ('Right Ankle', right_velocities, 1)]
    coordinates = [('X', 0, 'm/s'), ('Y', 1, 'm/s'), ('Z', 2, 'm/s')]
    
    for ankle_name, ankle_velocities, row in ankles:
        for coord_name, coord_idx, unit in coordinates:
            coord_data = ankle_velocities[:, coord_idx]
            
            # Plot each individual gait cycle
            labeled = False
            for cycle_idx in range(len(cycle_boundaries) - 1):
                start_idx = cycle_boundaries[cycle_idx]
                end_idx = cycle_boundaries[cycle_idx + 1]
                
                cycle_phase = phase[start_idx:end_idx]
                cycle_coord = coord_data[start_idx:end_idx]
                
                # Only plot cycles that have reasonable data
                if len(cycle_phase) > 10:
                    # Sort this individual cycle by phase
                    sort_idx = np.argsort(cycle_phase)
                    cycle_phase_sorted = cycle_phase[sort_idx]
                    cycle_coord_sorted = cycle_coord[sort_idx]
                    
                    # Plot individual cycle
                    label = 'Measured' if not labeled else None
                    labeled = True if label == 'Measured' else labeled
                    axes[row, coord_idx].plot(cycle_phase_sorted, cycle_coord_sorted, 'g-', 
                                            linewidth=1, alpha=0.4, label=label)
            
            # Add vertical line at phase 0.5 (stance switch)
            axes[row, coord_idx].axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, label='Stance Switch')
            
            # Format plot
            axes[row, coord_idx].set_title(f'{ankle_name} {coord_name} Velocity', fontsize=12)
            axes[row, coord_idx].set_xlabel('Phase', fontsize=10)
            axes[row, coord_idx].set_ylabel(f'{coord_name} Velocity ({unit})', fontsize=10)
            axes[row, coord_idx].tick_params(axis='both', which='major', labelsize=9)
            axes[row, coord_idx].grid(True, alpha=0.3)
            axes[row, coord_idx].set_xlim(0, 1)
            if cycle_idx == 0 and coord_idx == 0:  # Only add legend to first subplot
                axes[row, coord_idx].legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    
    # Save plot to ctrl_logs directory
    save_path = os.path.join(save_dir, 'ankle_velocities_debug')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{save_path}.svg", bbox_inches='tight', facecolor='white')
    print(f"Saved ankle velocities debug plots to: {save_path}.[png/pdf/svg]")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot phase space orbits (position vs velocity) from gait library')
    parser.add_argument('--ctrl-logs', help='Path to specific control logs folder (default: use most recent)')
    parser.add_argument('--gait-library', help='Path to gait library folder (default: use default full_library_v7)')
    parser.add_argument('--start-time', type=float, help='Start time in seconds for ctrl log data filtering')
    parser.add_argument('--end-time', type=float, help='End time in seconds for ctrl log data filtering')
    args = parser.parse_args()
    
    try:
        # Find most recent ctrl_logs folder for saving plots (or use specified one)
        if args.ctrl_logs:
            if not os.path.exists(args.ctrl_logs):
                raise FileNotFoundError(f"Specified ctrl_logs path not found: {args.ctrl_logs}")
            ctrl_logs_dir = args.ctrl_logs
            print(f"Using specified ctrl_logs folder: {ctrl_logs_dir}")
        else:
            ctrl_logs_dir = find_most_recent_ctrl_folder()
        
        # Determine gait library path
        if args.gait_library:
            if not os.path.exists(args.gait_library):
                raise FileNotFoundError(f"Gait library path not found: {args.gait_library}")
            gait_lib_path = args.gait_library
        else:
            gait_lib_path = find_default_gait_library()
        
        # Hardcode to use full_solution_230.yaml
        gait_filename = "full_solution_230.yaml"
        # gait_filename = "full_solution_210.yaml"
        gait_file_path = os.path.join(gait_lib_path, gait_filename)
        print(f"Loading gait data from: {gait_file_path}")
        
        # Load gait data
        gait_data = load_gait_data(gait_file_path)
        
        # Calculate total gait period from all domains
        domain_sequence = gait_data['domain_sequence']
        total_period = 0
        for domain in domain_sequence:
            if 'T' not in gait_data[domain]:
                raise KeyError(f"Period 'T' not found for domain '{domain}'")
            period = gait_data[domain]['T'][0]
            total_period += period
        print(f"Total gait period: {total_period} seconds across domains: {domain_sequence}")
        
        # Define joints to plot - temporarily showing all joints
        # Get all joints from gait library
        # first_domain = domain_sequence[0]
        # all_gait_joints = gait_data[first_domain]['joint_order']
        # joints_to_plot = all_gait_joints  # Plot all joints temporarily

        # Original hardcoded selection (commented out temporarily):
        joints_to_plot = [
            "left_knee_joint",
            "left_hip_pitch_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint"
        ]
        
        print(f"Plotting joints: {joints_to_plot}")
        
        # Extract gait library joint trajectories
        trajectories = extract_joint_trajectories(gait_data, joints_to_plot)
        
        # Load hardware data
        hardware_trajectories = None
        try:
            print("Loading hardware data for comparison...")
            times, decoded_obs, config = load_ctrl_data(ctrl_logs_dir, args.start_time, args.end_time)
            
            # Get gait joint order from the first domain
            first_domain = domain_sequence[0]
            gait_joint_order = gait_data[first_domain]['joint_order']
            
            # Extract hardware trajectories for comparison
            hardware_trajectories = extract_hardware_trajectories(decoded_obs, gait_joint_order, joints_to_plot)
            print(f"Successfully loaded hardware data with {len(decoded_obs['phase'])} data points")
            
        except Exception as e:
            print(f"Warning: Could not load hardware data: {e}")
            print("Continuing with gait library data only...")
        
        # Plot phase space orbits (with hardware overlay if available)
        # plot_phase_space_orbits(trajectories, gait_data, hardware_trajectories, ctrl_logs_dir)
        
        # Plot joint angles over phase (with hardware overlay if available)
        plot_joint_angles_over_time(trajectories, gait_data, hardware_trajectories, ctrl_logs_dir)
        
        # Compute and plot swing ankle trajectories if hardware data is available
        if hardware_trajectories:
            try:
                print("Computing swing ankle trajectories using forward kinematics...")
                
                # Load robot model
                model, data = load_robot_model()
                
                # Get all joint positions and velocities from hardware data (need all joints for FK)
                all_hardware_positions = decoded_obs['joint_positions']
                all_hardware_velocities = decoded_obs['joint_velocities']
                all_hardware_phase = decoded_obs['phase']
                
                # Map hardware data to gait joint order for FK computation
                isaac_joint_names = decoded_obs['isaac_joint_names']
                gait_joint_order = gait_data[first_domain]['joint_order']
                all_hardware_gait_ordered = map_hardware_to_gait_order(all_hardware_positions, isaac_joint_names, gait_joint_order)
                all_hardware_velocities_gait_ordered = map_hardware_to_gait_order(all_hardware_velocities, isaac_joint_names, gait_joint_order)
                
                # Get projected gravity from decoded observations
                all_hardware_projected_gravity = decoded_obs['projected_gravity']
                
                # Compute ankle positions and velocities using forward kinematics with projected gravity
                ankle_positions = compute_ankle_positions(model, data, all_hardware_gait_ordered, gait_joint_order, all_hardware_phase, all_hardware_projected_gravity, all_hardware_velocities_gait_ordered)
                
                if ankle_positions is None:
                    print("Error: ankle_positions is None - cannot compute swing ankle trajectories")
                    return
                
                # Plot debug ankle positions (absolute positions)
                plot_ankle_positions_debug(ankle_positions, ctrl_logs_dir)
                
                # # Plot debug ankle velocities (absolute velocities) if available
                # if 'left_ankle_velocities' in ankle_positions and 'right_ankle_velocities' in ankle_positions:
                #     plot_ankle_velocities_debug(ankle_positions, ctrl_logs_dir)
                #
                #     # Plot comparison of gait library and hardware ankle velocities
                #     plot_ankle_velocities_comparison(ankle_positions, gait_data, ctrl_logs_dir)
                
                # Compute swing ankle relative trajectories
                swing_trajectories = compute_swing_ankle_trajectories(ankle_positions)
                
                # Plot swing ankle trajectories with gait library overlay
                plot_swing_ankle_trajectories(swing_trajectories, gait_data, ctrl_logs_dir, hardware_trajectories)
                
                # # Plot swing ankle phase plots (position vs velocity)
                # plot_swing_ankle_phase_plots(swing_trajectories, gait_data, ctrl_logs_dir, hardware_trajectories)
                
                print("Swing ankle trajectory plotting complete!")
                
            except Exception as e:
                print(f"Warning: Could not compute swing ankle trajectories: {e}")
                print("Continuing without swing ankle plots...")
        else:
            print("No hardware data available for swing ankle trajectory computation")
        
        print("All plotting complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()