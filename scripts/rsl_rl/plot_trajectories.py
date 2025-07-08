import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import argparse


def find_most_recent_log_dir(base_path="logs/play"):
    """Find the most recent log directory"""
    if not os.path.exists(base_path):
        print(f"Error: Log directory {base_path} does not exist")
        return None
    # Get all timestamped directories
    dirs = glob.glob(os.path.join(base_path, "*"))
    if not dirs:
        print(f"No log directories found in {base_path}")
        return None
    # Sort by modification time (newest first)
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir


def load_data(log_dir):
    """Load all pickle files from the log directory"""
    data = {}
    for pkl_file in glob.glob(os.path.join(log_dir, "*.pkl")):
        var_name = os.path.basename(pkl_file).replace(".pkl", "")
        with open(pkl_file, "rb") as f:
            data[var_name] = pickle.load(f)
    return data


def format_joint_name(joint_name):
    """Format joint name for better readability in plots"""
    # Remove '_joint' suffix and replace underscores with spaces
    formatted = joint_name.replace('_joint', '').replace('_', ' ')
    
    # Capitalize first letter of each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    # Special formatting for common terms
    formatted = formatted.replace('Hip', 'Hip')
    formatted = formatted.replace('Knee', 'Knee')
    formatted = formatted.replace('Ankle', 'Ankle')
    formatted = formatted.replace('Shoulder', 'Shoulder')
    formatted = formatted.replace('Elbow', 'Elbow')
    formatted = formatted.replace('Waist', 'Waist')
    
    return formatted


def plot_trajectories(data, save_dir=None, trajectory_type=None):
    """Plot all trajectories with proper labels and units"""
    # Convert lists to numpy arrays and handle torch tensors
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    
    # Create time array
    time_steps = np.arange(len(processed_data[list(processed_data.keys())[0]]))
    
    # Detect trajectory type if not provided
    if trajectory_type is None:
        # Check if we have end effector style metrics (contains frame names)
        ee_metrics = [key for key in processed_data.keys()
                     if '_ee_pos_' in key or '_ee_ori_' in key]
        if ee_metrics:
            trajectory_type = 'end_effector'
        else:
            # Check if we have joint style metrics
            joint_metrics = [key for key in processed_data.keys()
                           if key.startswith('error_') and '_joint' in key]
            if joint_metrics:
                trajectory_type = 'joint'
            else:
                # Default to joint if we can't determine
                trajectory_type = 'joint'
    
    print(f"Detected trajectory type: {trajectory_type}")
    
    # Generate dynamic labels and units based on trajectory type
    if trajectory_type == 'joint':
        # Joint trajectory - use G1 joint names
        g1_joint_names = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
            'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint',
            'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint',
            'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'
        ]
        
        g1_formatted_labels = [format_joint_name(name) for name in g1_joint_names]
        
        state_labels = {
            'y_out': g1_formatted_labels,
            'dy_out': g1_formatted_labels,
            'base_velocity': ['Linear X', 'Linear Y', 'Angular Z'],
            "stance_foot_pos": ['X', 'Y', 'Z'],
            "stance_foot_ori": ['Roll', 'Pitch', 'Yaw'],
            'cur_swing_time': ['Time'],
            'y_act': g1_formatted_labels,
            'dy_act': g1_formatted_labels,
            'v': ['v'],
            'vdot': ['vdot'],
            'reward': ['Reward']
        }
        
        units = {
            'y_out': ['rad'] * 21,
            'dy_out': ['rad/s'] * 21,
            'base_velocity': ['m/s', 'm/s', 'rad/s'],
            'stance_foot_pos': ['m', 'm', 'm'],
            'stance_foot_ori': ['rad', 'rad', 'rad'],
            'cur_swing_time': ['s'],
            'y_act': ['rad'] * 21,
            'dy_act': ['rad/s'] * 21,
            'v': ['m/s'],
            'vdot': ['m/s²'],
            'reward': ['']
        }
        
        # Generate error labels dynamically from actual metric names
        error_labels = {}
        error_units = {}
        for key in processed_data.keys():
            if key.startswith('error_'):
                # Handle different error metric patterns
                if '_joint' in key:
                    # Joint error: error_joint_name
                    joint_name = key.replace('error_', '')
                    error_labels[key] = f"{format_joint_name(joint_name)} Error"
                    error_units[key] = 'rad'
                elif key in ['error_sw_x', 'error_sw_y', 'error_sw_z']:
                    # Swing foot position errors
                    axis = key.split('_')[-1].upper()
                    error_labels[key] = f"Swing Foot Position {axis}"
                    error_units[key] = 'm'
                elif key in ['error_sw_roll', 'error_sw_pitch', 'error_sw_yaw']:
                    # Swing foot orientation errors
                    axis = key.split('_')[-1].title()
                    error_labels[key] = f"Swing Foot Orientation {axis}"
                    error_units[key] = 'rad'
                elif key in ['error_com_x', 'error_com_y', 'error_com_z']:
                    # COM position errors
                    axis = key.split('_')[-1].upper()
                    error_labels[key] = f"COM Position {axis}"
                    error_units[key] = 'm'
                elif key in ['error_pelvis_roll', 'error_pelvis_pitch', 'error_pelvis_yaw']:
                    # Pelvis orientation errors
                    axis = key.split('_')[-1].title()
                    error_labels[key] = f"Pelvis Orientation {axis}"
                    error_units[key] = 'rad'
                else:
                    # Generic error handling
                    error_labels[key] = key.replace('error_', '').replace('_', ' ').title()
                    error_units[key] = 'mixed'
    
    elif trajectory_type == 'end_effector':
        # End effector trajectory - generate labels from metric names
        # Try to get axis names from the data if available
        axis_names = []
        if 'axis_names' in data and data['axis_names']:
            # Extract names from axis_names data (take first entry since it's the same for all timesteps)
            axis_names_data = data['axis_names'][0] if isinstance(data['axis_names'], list) else data['axis_names']
            if isinstance(axis_names_data, list):
                axis_names = [axis_info['name'] for axis_info in axis_names_data]
            else:
                axis_names = []
        else:
            # Fallback: try to infer from error metrics
            error_keys = [key for key in processed_data.keys() if key.startswith('error_')]
            if error_keys:
                # Extract dimension names from error keys
                axis_names = [key.replace('error_', '') for key in error_keys]
            else:
                # Final fallback: generic names
                n_dims = processed_data.get('y_out', [[]]).shape[2] if 'y_out' in processed_data else 0
                axis_names = [f'Dimension {i}' for i in range(n_dims)]
        
        state_labels = {
            'y_out': axis_names,
            'dy_out': [f"{name} Rate" for name in axis_names],
            'base_velocity': ['Linear X', 'Linear Y', 'Angular Z'],
            "stance_foot_pos": ['X', 'Y', 'Z'],
            "stance_foot_ori": ['Roll', 'Pitch', 'Yaw'],
            'cur_swing_time': ['Time'],
            'y_act': axis_names,
            'dy_act': [f"{name} Rate" for name in axis_names],
            'v': ['v'],
            'vdot': ['vdot'],
            'reward': ['Reward']
        }
        
        # Generate units based on axis names
        def get_unit_for_axis(axis_name):
            if 'pos' in axis_name or 'com_pos' in axis_name:
                return 'm'
            elif 'ori' in axis_name:
                return 'rad'
            elif 'joint' in axis_name:
                return 'rad'
            else:
                return 'mixed'
        
        axis_units = [get_unit_for_axis(name) for name in axis_names]
        rate_units = [f"{unit}/s" for unit in axis_units]
        
        units = {
            'y_out': axis_units,
            'dy_out': rate_units,
            'base_velocity': ['m/s', 'm/s', 'rad/s'],
            'stance_foot_pos': ['m', 'm', 'm'],
            'stance_foot_ori': ['rad', 'rad', 'rad'],
            'cur_swing_time': ['s'],
            'y_act': axis_units,
            'dy_act': rate_units,
            'v': ['m/s'],
            'vdot': ['m/s²'],
            'reward': ['']
        }
        
        # Generate error labels dynamically from end effector metrics
        error_labels = {}
        error_units = {}
        for key in processed_data.keys():
            if key.startswith('error_'):
                # Parse end effector error metrics
                parts = key.split('_')
                if len(parts) >= 3:
                    if '_ee_pos_' in key:
                        # Position constraint: error_frame_ee_pos_axis
                        frame_name = parts[1]  # e.g., 'left_palm'
                        axis = parts[-1].upper()
                        error_labels[key] = f"{frame_name.replace('_', ' ').title()} Position {axis}"
                        error_units[key] = 'm'
                    elif '_ee_ori_' in key:
                        # Orientation constraint: error_frame_ee_ori_axis
                        frame_name = parts[1]  # e.g., 'left_palm'
                        axis = parts[-1].title()
                        error_labels[key] = f"{frame_name.replace('_', ' ').title()} Orientation {axis}"
                        error_units[key] = 'rad'
                    elif '_com_pos_' in key:
                        # COM position constraint: error_com_pos_axis
                        axis = parts[-1].upper()
                        error_labels[key] = f"COM Position {axis}"
                        error_units[key] = 'm'
                    elif '_pelvis_ori_' in key:
                        # Pelvis orientation constraint: error_pelvis_ori_axis
                        axis = parts[-1].title()
                        error_labels[key] = f"Pelvis Orientation {axis}"
                        error_units[key] = 'rad'
                    else:
                        # Generic end effector error
                        error_labels[key] = key.replace('error_', '').replace('_', ' ').title()
                        error_units[key] = 'mixed'
                else:
                    # Fallback for simple error names
                    error_labels[key] = key.replace('error_', '').replace('_', ' ').title()
                    error_units[key] = 'mixed'
    
    else:
        # Fallback for unknown trajectory types
        state_labels = {}
        units = {}
        error_labels = {}
        error_units = {}

    # Helper for subplot indexing
    def get_ax(axs, idx, n_cols):
        if axs.ndim == 1:
            return axs[idx]
        return axs[idx // n_cols, idx % n_cols]

    # Hard code number of envs to plot
    N_ENVS_TO_PLOT = 2
    env_ids = list(range(N_ENVS_TO_PLOT))

    for env_id in env_ids:
        # --- Stance Foot Position and Orientation ---
        if "stance_foot_pos" and "stance_foot_ori" in processed_data:
            pos_data = processed_data["stance_foot_pos"]
            ori_data = processed_data["stance_foot_ori"]
            pos_data_0 = processed_data["stance_foot_pos_0"]
            ori_data_0 = processed_data["stance_foot_ori_0"]
            fig, axs = plt.subplots(2, 3, figsize=(15, 6))
            fig.suptitle(f"Stance Foot Position and Orientation (Env {env_id})", fontsize=16)
            pos_labels = ["X", "Y", "Z"]
            for i in range(3):
                ax = axs[0, i]
                ax.plot(time_steps, pos_data[:, env_id, i], label=f"Position {pos_labels[i]}", linewidth=2)
                ax.plot(time_steps, pos_data_0[:, env_id, i], label=f"Initial {pos_labels[i]}", linestyle='--', linewidth=2)
                ax.set_title(f"Position {pos_labels[i]}")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("m")
                ax.grid(True, alpha=0.3)
                ax.legend()
            ori_labels = ["Roll", "Pitch", "Yaw"]
            for i in range(3):
                ax = axs[1, i]
                ax.plot(time_steps, ori_data[:, env_id, i], label=f"Orientation {ori_labels[i]}", linewidth=2)
                ax.plot(time_steps, ori_data_0[:, env_id, i], label=f"Initial {ori_labels[i]}", linestyle='--', linewidth=2)
                ax.set_title(f"Orientation {ori_labels[i]}")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("rad")
                ax.grid(True, alpha=0.3)
                ax.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"stance_foot_pos_ori_env{env_id}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        # --- Positions (y_out vs y_act) ---
        if 'y_out' in processed_data and 'y_act' in processed_data:
            n_dims = processed_data['y_out'].shape[2]
            if trajectory_type == 'end_effector':
                title = f'Reference vs Actual End Effector Positions (Env {env_id})'
            else:
                title = f'Reference vs Actual Joint Positions (Env {env_id})'
            n_cols = 4
            n_rows = (n_dims + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(title, fontsize=16)
            axs = np.array(axs)
            for i in range(n_dims):
                ax = get_ax(axs, i, n_cols)
                ax.plot(time_steps, processed_data['y_out'][:, env_id, i], label='Reference', linewidth=2)
                ax.plot(time_steps, processed_data['y_act'][:, env_id, i], label='Actual', linestyle='--', linewidth=2)
                label = state_labels['y_out'][i] if i < len(state_labels['y_out']) else f'Dimension {i}'
                unit = units['y_out'][i] if i < len(units['y_out']) else ''
                ax.set_title(label, fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'Position ({unit})')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
            for i in range(n_dims, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'positions_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Velocities (dy_out vs dy_act) ---
        if 'dy_out' in processed_data and 'dy_act' in processed_data:
            n_dims = processed_data['dy_out'].shape[2]
            if trajectory_type == 'end_effector':
                title = f'Reference vs Actual End Effector Velocities (Env {env_id})'
            else:
                title = f'Reference vs Actual Joint Velocities (Env {env_id})'
            n_cols = 4
            n_rows = (n_dims + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(title, fontsize=16)
            axs = np.array(axs)
            for i in range(n_dims):
                ax = get_ax(axs, i, n_cols)
                ax.plot(time_steps, processed_data['dy_out'][:, env_id, i], label='Reference', linewidth=2)
                ax.plot(time_steps, processed_data['dy_act'][:, env_id, i], label='Actual', linestyle='--', linewidth=2)
                label = state_labels['dy_out'][i] if i < len(state_labels['dy_out']) else f'Dimension {i}'
                unit = units['dy_out'][i] if i < len(units['dy_out']) else ''
                ax.set_title(label, fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'Velocity ({unit})')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
            for i in range(n_dims, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'velocities_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Base Velocity ---
        if 'base_velocity' in processed_data:
            n_dims = processed_data['base_velocity'].shape[2]
            fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 3))
            fig.suptitle(f'Base Velocity (Env {env_id})', fontsize=16)
            for i in range(n_dims):
                ax = axs[i] if n_dims > 1 else axs
                ax.plot(time_steps, processed_data['base_velocity'][:, env_id, i], linewidth=2)
                label = state_labels['base_velocity'][i] if i < len(state_labels['base_velocity']) else f'Component {i}'
                unit = units['base_velocity'][i] if i < len(units['base_velocity']) else ''
                ax.set_title(label)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'Velocity ({unit})')
                ax.grid(True, alpha=0.3)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'base_velocity_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- v and vdot ---
        if 'v' in processed_data and 'vdot' in processed_data:
            v_data = processed_data['v']
            vdot_data = processed_data['vdot']
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(time_steps, v_data[:, env_id], label='CLF v', linewidth=2)
            axs[0].set_title('CLF (v)')
            axs[0].set_ylabel(units['v'][0] if 'v' in units else '')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()
            axs[0].set_ylim(0, 20.0)
            axs[1].plot(time_steps, vdot_data[:, env_id], label='CLF vdot', linewidth=2)
            axs[1].set_title('CLF (v̇)')
            axs[1].set_xlabel('Time Steps')
            axs[1].set_ylabel(units['vdot'][0] if 'vdot' in units else '')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()
            axs[1].set_ylim(-100.0, 100.0)
            alpha = 1.0
            decay = alpha * v_data[:, env_id] + vdot_data[:, env_id]
            axs[2].plot(time_steps, decay, label='CLF Decay', linewidth=2)
            axs[2].set_title('CLF Decay (v + αv̇)')
            axs[2].set_xlabel('Time Steps')
            axs[2].set_ylabel('Decay Rate')
            axs[2].grid(True, alpha=0.3)
            axs[2].legend()
            axs[2].set_ylim(-100.0, 100.0)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'v_and_vdot_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Error Metrics ---
        error_metrics = [key for key in processed_data.keys() if key.startswith('error_')]
        if error_metrics:
            n_metrics = len(error_metrics)
            n_cols = 4
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(f'Error Metrics (Env {env_id})', fontsize=16)
            axs = np.array(axs)
            for i, metric in enumerate(error_metrics):
                ax = get_ax(axs, i, n_cols)
                data = processed_data[metric]
                if data.ndim > 1:
                    plot_data = data[:, env_id]
                    ax.plot(time_steps, plot_data, label=error_labels.get(metric, metric), linewidth=2)
                else:
                    ax.plot(time_steps, data, label=error_labels.get(metric, metric), linewidth=2)
                ax.set_title(error_labels.get(metric, metric), fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(error_units.get(metric, ''))
                ax.grid(True, alpha=0.3)
                ax.legend()
            for i in range(n_metrics, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'error_metrics_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)


def plot_hzd_trajectories(data, save_dir=None):
    """Plot HZD trajectories with proper labels and units"""
    # Convert lists to numpy arrays and handle torch tensors
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    
    # Create time array
    time_steps = np.arange(len(processed_data[list(processed_data.keys())[0]]))
    
    # Define state labels and units for HZD
    state_labels = {
        'y_out': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'dy_out': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'base_velocity': ['Linear x', 'Linear y', 'Angular z'],
        "stance_foot_pos": ['x', 'y', 'z'],
        "stance_foot_ori": ['roll', 'pitch', 'yaw'],
        'cur_swing_time': ['Time'],
        'y_act': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'dy_act': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'v': ['Velocity'],
        'vdot': ['Acceleration'],
        'reward': ['Reward']
    }
    
    units = {
        'y_out': ['rad'] * 12,  # All joint angles are in radians
        'dy_out': ['rad/s'] * 12,  # All joint velocities are in rad/s
        'base_velocity': ['m/s', 'm/s', 'rad/s'],
        'stance_foot_pos': ['m', 'm', 'm'],
        'stance_foot_ori': ['rad', 'rad', 'rad'],
        'cur_swing_time': ['s'],
        'y_act': ['rad'] * 12,
        'dy_act': ['rad/s'] * 12,
        'v': ['m/s'],
        'vdot': ['m/s²'],
        'reward': ['']
    }

    # Add error metrics labels and units
    error_labels = {
        'error_LeftFrontalHipJoint': 'Left Frontal Hip Error',
        'error_RightFrontalHipJoint': 'Right Frontal Hip Error',
        'error_LeftTransverseHipJoint': 'Left Transverse Hip Error',
        'error_RightTransverseHipJoint': 'Right Transverse Hip Error',
        'error_LeftSagittalHipJoint': 'Left Sagittal Hip Error',
        'error_RightSagittalHipJoint': 'Right Sagittal Hip Error',
        'error_LeftSagittalKneeJoint': 'Left Knee Error',
        'error_RightSagittalKneeJoint': 'Right Knee Error',
        'error_LeftSagittalAnkleJoint': 'Left Ankle Error',
        'error_RightSagittalAnkleJoint': 'Right Ankle Error',
        'error_LeftHenkeAnkleJoint': 'Left Henke Ankle Error',
        'error_RightHenkeAnkleJoint': 'Right Henke Ankle Error'
    }

    error_units = {
        'error_LeftFrontalHipJoint': 'rad',
        'error_RightFrontalHipJoint': 'rad',
        'error_LeftTransverseHipJoint': 'rad',
        'error_RightTransverseHipJoint': 'rad',
        'error_LeftSagittalHipJoint': 'rad',
        'error_RightSagittalHipJoint': 'rad',
        'error_LeftSagittalKneeJoint': 'rad',
        'error_RightSagittalKneeJoint': 'rad',
        'error_LeftSagittalAnkleJoint': 'rad',
        'error_RightSagittalAnkleJoint': 'rad',
        'error_LeftHenkeAnkleJoint': 'rad',
        'error_RightHenkeAnkleJoint': 'rad'
    }
    
    # Helper for subplot indexing
    def get_ax(axs, idx, n_cols):
        if axs.ndim == 1:
            return axs[idx]
        return axs[idx // n_cols, idx % n_cols]

    env_ids = 0

    if "stance_foot_pos" and "stance_foot_ori" in processed_data:
        pos_data = processed_data["stance_foot_pos"]
        ori_data = processed_data["stance_foot_ori"]
        pos_data_0 = processed_data["stance_foot_pos_0"]
        ori_data_0 = processed_data["stance_foot_ori_0"]

        fig, axs = plt.subplots(2, 3, figsize=(15, 6))
        fig.suptitle("Stance Foot Position and Orientation", fontsize=16)

        # Position
        pos_labels = ["x", "y", "z"]
        for i in range(3):
            ax = axs[0, i]
            ax.plot(time_steps, pos_data[:, env_ids, i], label=f"pos {pos_labels[i]}")
            ax.plot(time_steps, pos_data_0[:, env_ids, i], label=f"pos_0 {pos_labels[i]}", linestyle='--')
            ax.set_title(f"Position {pos_labels[i]}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("m")
            ax.grid(True)
            ax.legend()

        # Orientation
        ori_labels = ["roll", "pitch", "yaw"]
        for i in range(3):
            ax = axs[1, i]
            ax.plot(time_steps, ori_data[:, env_ids, i], label=f"ori {ori_labels[i]}")
            ax.plot(time_steps, ori_data_0[:, env_ids, i], label=f"ori_0 {ori_labels[i]}", linestyle='--')
            ax.set_title(f"Orientation {ori_labels[i]}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("rad")
            ax.grid(True)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, "stance_foot_pos_ori.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # Plot positions (y_out vs y_act)
    if 'y_out' in processed_data and 'y_act' in processed_data:
        n_dims = processed_data['y_out'].shape[2]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Reference vs Actual Positions', fontsize=16)
        axs = np.array(axs)
        for i in range(n_dims):
            ax = get_ax(axs, i, n_cols)
            ax.plot(time_steps, processed_data['y_out'][:, env_ids, i], label='Reference', color='b')
            ax.plot(time_steps, processed_data['y_act'][:, env_ids, i], label='Actual', color='r', linestyle='--')
            label = state_labels['y_out'][i] if i < len(state_labels['y_out']) else f'Var {i}'
            unit = units['y_out'][i] if i < len(units['y_out']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
            if i == 0:
                ax.legend()
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'positions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot velocities (dy_out vs dy_act)
    if 'dy_out' in processed_data and 'dy_act' in processed_data:
        n_dims = processed_data['dy_out'].shape[2]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Reference vs Actual Velocities', fontsize=16)
        axs = np.array(axs)
        for i in range(n_dims):
            ax = get_ax(axs, i, n_cols)
            ax.plot(time_steps, processed_data['dy_out'][:, env_ids, i], label='Reference', color='b')
            ax.plot(time_steps, processed_data['dy_act'][:, env_ids, i], label='Actual', color='r', linestyle='--')
            label = state_labels['dy_out'][i] if i < len(state_labels['dy_out']) else f'Var {i}'
            unit = units['dy_out'][i] if i < len(units['dy_out']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
            if i == 0:
                ax.legend()
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'velocities.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot base velocity
    if 'base_velocity' in processed_data:
        n_dims = processed_data['base_velocity'].shape[2]
        fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 3))
        fig.suptitle('Base Velocity', fontsize=16)
        for i in range(n_dims):
            ax = axs[i] if n_dims > 1 else axs
            ax.plot(time_steps, processed_data['base_velocity'][:, env_ids, i])
            label = state_labels['base_velocity'][i] if i < len(state_labels['base_velocity']) else f'Var {i}'
            unit = units['base_velocity'][i] if i < len(units['base_velocity']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'base_velocity.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # Plot v and vdot as two subplots in one figure
    if 'v' in processed_data and 'vdot' in processed_data:
        v_data = processed_data['v']
        vdot_data = processed_data['vdot']
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        axs[0].plot(time_steps, v_data[:, env_ids], label='v', color='g')
        axs[0].set_title('clf v')
        axs[0].set_ylabel(units['v'][0] if 'v' in units else '')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(time_steps, vdot_data[:, env_ids], label='vdot', color='m')
        axs[1].set_title('clf vdot')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel(units['vdot'][0] if 'vdot' in units else '')
        axs[1].grid(True)
        axs[1].legend()
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'v_and_vdot.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # Plot error metrics
    error_metrics = [key for key in processed_data.keys() if key.startswith('error_')]
    if error_metrics:
        n_metrics = len(error_metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Error Metrics', fontsize=16)
        axs = np.array(axs)
        
        for i, metric in enumerate(error_metrics):
            ax = get_ax(axs, i, n_cols)
            data = processed_data[metric]
            # Handle both 1D and 2D arrays
            if data.ndim > 1:
                plot_data = data[:, env_ids]
            else:
                plot_data = data
            ax.plot(time_steps, plot_data, label=error_labels.get(metric, metric))
            ax.set_title(error_labels.get(metric, metric))
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(error_units.get(metric, ''))
            ax.grid(True)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'error_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot trajectory data from log files')
    parser.add_argument('--log_dir', type=str, help='Specific log directory to plot (optional)')
    parser.add_argument('--trajectory_type', type=str, 
                       choices=['joint', 'end_effector', 'auto'], 
                       default='auto', help='Type of trajectory to plot (default: auto-detect)')
    parser.add_argument('--base_path', type=str, default='logs/play', 
                       help='Base path to search for log directories (default: logs/play)')
    
    args = parser.parse_args()
    
    # Find the log directory
    if args.log_dir:
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            print(f"Error: Specified log directory {log_dir} does not exist")
            return
    else:
        log_dir = find_most_recent_log_dir(args.base_path)
        if not log_dir:
            return
    
    print(f"Loading data from {log_dir}")
    # Load the data
    data = load_data(log_dir)
    # Create a directory for plots
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Determine trajectory type
    trajectory_type = None if args.trajectory_type == 'auto' else args.trajectory_type
    
    # Plot the data with specified or auto-detected trajectory type
    plot_trajectories(data, save_dir=plot_dir, trajectory_type=trajectory_type)


if __name__ == "__main__":
    main() 