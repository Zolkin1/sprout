#!/usr/bin/env python3
"""
Compare tracking error between two policy runs.

This script loads data from two different run directories and creates comparison plots
showing the mean tracking error with standard deviation for both runs on the same figure.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import argparse


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

    return formatted


def detect_trajectory_type(data):
    """Detect whether this is a joint or end effector trajectory"""
    # Check if we have end effector style metrics (contains frame names)
    ee_metrics = [key for key in data.keys()
                 if '_ee_pos_' in key or '_ee_ori_' in key]
    if ee_metrics:
        return 'end_effector'
    else:
        # Check if we have joint style metrics
        joint_metrics = [key for key in data.keys()
                       if key.startswith('error_') and '_joint' in key]
        if joint_metrics:
            return 'joint'
        else:
            # Default to joint if we can't determine
            return 'joint'


def get_labels_and_units(trajectory_type, data):
    """Get appropriate labels and units based on trajectory type"""
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

        labels = [format_joint_name(name) for name in g1_joint_names]
        units = ['rad'] * 21

    elif trajectory_type == 'end_effector':
        # End effector trajectory - generate labels from metric names
        axis_names = []
        if 'axis_names' in data and data['axis_names']:
            # Extract names from axis_names data
            axis_names_data = data['axis_names'][0] if isinstance(data['axis_names'], list) else data['axis_names']
            if isinstance(axis_names_data, list):
                axis_names = [axis_info['name'] for axis_info in axis_names_data]
            else:
                axis_names = []

        if not axis_names:
            # Fallback: try to infer from error metrics
            error_keys = [key for key in data.keys() if key.startswith('error_')]
            if error_keys:
                axis_names = [key.replace('error_', '') for key in error_keys]
            else:
                # Final fallback: generic names
                n_dims = data.get('y_out', [[]]).shape[2] if 'y_out' in data else 0
                axis_names = [f'Dimension {i}' for i in range(n_dims)]

        labels = axis_names

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

        units = [get_unit_for_axis(name) for name in axis_names]

    else:
        labels = []
        units = []

    return labels, units


def process_data(data):
    """Convert lists to numpy arrays and handle torch tensors"""
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    return processed_data


def get_ax(axs, idx, n_cols):
    """Helper for subplot indexing"""
    if axs.ndim == 1:
        return axs[idx]
    return axs[idx // n_cols, idx % n_cols]


def compare_tracking_error(run1_dir, run2_dir, run1_name=None, run2_name=None,
                           save_dir=None, start_idx=50, end_idx=None):
    """
    Compare tracking error between two runs.

    Args:
        run1_dir: Path to first run directory
        run2_dir: Path to second run directory
        run1_name: Label for first run (default: directory name)
        run2_name: Label for second run (default: directory name)
        save_dir: Directory to save plots (default: current directory)
        start_idx: Time step to start plotting from (default: 50)
        end_idx: Time step to end plotting at (default: None, uses all data)
    """
    # Load data from both runs
    print(f"Loading data from run 1: {run1_dir}")
    data1 = load_data(run1_dir)
    print(f"Loading data from run 2: {run2_dir}")
    data2 = load_data(run2_dir)

    # Process data
    processed_data1 = process_data(data1)
    processed_data2 = process_data(data2)

    # Check if both have required data
    if 'y_out' not in processed_data1 or 'y_act' not in processed_data1:
        print("Error: Run 1 missing y_out or y_act data")
        return
    if 'y_out' not in processed_data2 or 'y_act' not in processed_data2:
        print("Error: Run 2 missing y_out or y_act data")
        return

    # Detect trajectory type (use run 1)
    trajectory_type = detect_trajectory_type(processed_data1)
    print(f"Detected trajectory type: {trajectory_type}")

    # Get labels and units
    labels, units = get_labels_and_units(trajectory_type, data1)

    # Calculate position errors (absolute value)
    position_error1 = (processed_data1['y_act'] - processed_data1['y_out'])   # time, envs, dimension
    position_error2 = (processed_data2['y_act'] - processed_data2['y_out'])

    # Get dimensions
    n_dims = position_error1.shape[2]

    # Create time arrays
    time_steps1 = np.arange(len(processed_data1['y_out']))
    time_steps2 = np.arange(len(processed_data2['y_out']))

    # Apply start and end indices
    time_steps1_subset = time_steps1[start_idx:end_idx]
    time_steps2_subset = time_steps2[start_idx:end_idx]
    position_error1_subset = position_error1[start_idx:end_idx, :, :]
    position_error2_subset = position_error2[start_idx:end_idx, :, :]

    # Calculate mean and std for both runs. This is mean over env
    mean_error1 = np.abs(np.mean(position_error1_subset, axis=1))  # Shape: [time_steps, n_dims]
    std_error1 = np.std(position_error1_subset, axis=1)
    mean_error2 = np.abs(np.mean(position_error2_subset, axis=1))
    std_error2 = np.std(position_error2_subset, axis=1)

    # Calculate overall mean tracking error across all dimensions and time steps
    overall_mean_error1 = np.mean(mean_error1)
    overall_mean_error2 = np.mean(mean_error2)

    # Set run names
    if run1_name is None:
        run1_name = os.path.basename(run1_dir.rstrip('/'))
    if run2_name is None:
        run2_name = os.path.basename(run2_dir.rstrip('/'))

    # Create comparison plot
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    fig.suptitle(f'Absolute Position Tracking Error Comparison: {run1_name} vs {run2_name}',
                 fontsize=16)
    axs = np.array(axs) if n_dims > 1 else np.array([axs])

    for i in range(n_dims):
        ax = get_ax(axs, i, n_cols)

        # Plot run 1
        line1 = ax.plot(time_steps1_subset, mean_error1[:, i], linewidth=2,
                       label=f'{run1_name} (mean)')
        color1 = line1[0].get_color()
        ax.fill_between(time_steps1_subset,
                       mean_error1[:, i] - std_error1[:, i],
                       mean_error1[:, i] + std_error1[:, i],
                       alpha=0.2, color=color1)

        # Plot run 2
        line2 = ax.plot(time_steps2_subset, mean_error2[:, i], linewidth=2,
                       label=f'{run2_name} (mean)')
        color2 = line2[0].get_color()
        ax.fill_between(time_steps2_subset,
                       mean_error2[:, i] - std_error2[:, i],
                       mean_error2[:, i] + std_error2[:, i],
                       alpha=0.2, color=color2)

        # Add mean tracking error lines for this dimension
        dim_mean_error1 = np.mean(mean_error1[:, i])
        dim_mean_error2 = np.mean(mean_error2[:, i])
        ax.axhline(y=dim_mean_error1, color=color1, linestyle=(0, (5, 5)), alpha=0.5,
                  linewidth=1.5, label=f'{run1_name} avg')
        ax.axhline(y=dim_mean_error2, color=color2, linestyle=(0, (5, 5)), alpha=0.5,
                  linewidth=1.5, label=f'{run2_name} avg')

        # Add zero reference line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

        label = labels[i] if i < len(labels) else f'Dimension {i}'
        unit = units[i] if i < len(units) else ''
        ax.set_title(f'{label} Error', fontsize=10)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'Error ({unit})')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='best', fontsize=7)

    # Hide unused subplots
    for i in range(n_dims, n_rows * n_cols):
        ax = get_ax(axs, i, n_cols)
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, 'tracking_error_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved mean error comparison plot to: {output_path}")

    plt.close(fig)

    # Create standard deviation comparison plot
    fig_std, axs_std = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    fig_std.suptitle(f'Position Tracking Error Std Dev Comparison: {run1_name} vs {run2_name}',
                     fontsize=16)
    axs_std = np.array(axs_std) if n_dims > 1 else np.array([axs_std])

    for i in range(n_dims):
        ax = get_ax(axs_std, i, n_cols)

        # Plot std dev for run 1
        line1 = ax.plot(time_steps1_subset, std_error1[:, i], linewidth=2,
                       label=f'{run1_name}')
        color1 = line1[0].get_color()

        # Plot std dev for run 2
        line2 = ax.plot(time_steps2_subset, std_error2[:, i], linewidth=2,
                       label=f'{run2_name}')
        color2 = line2[0].get_color()

        # Add mean std dev lines for this dimension
        dim_mean_std1 = np.mean(std_error1[:, i])
        dim_mean_std2 = np.mean(std_error2[:, i])
        ax.axhline(y=dim_mean_std1, color=color1, linestyle=(0, (5, 5)), alpha=0.5,
                  linewidth=1.5, label=f'{run1_name} avg')
        ax.axhline(y=dim_mean_std2, color=color2, linestyle=(0, (5, 5)), alpha=0.5,
                  linewidth=1.5, label=f'{run2_name} avg')

        label = labels[i] if i < len(labels) else f'Dimension {i}'
        unit = units[i] if i < len(units) else ''
        ax.set_title(f'{label} Std Dev', fontsize=10)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'Std Dev ({unit})')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='best', fontsize=7)

    # Hide unused subplots
    for i in range(n_dims, n_rows * n_cols):
        ax = get_ax(axs_std, i, n_cols)
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save std dev figure
    output_path_std = os.path.join(save_dir, 'tracking_error_std_comparison.png')
    plt.savefig(output_path_std, dpi=300, bbox_inches='tight')
    print(f"Saved std dev comparison plot to: {output_path_std}")

    plt.close(fig_std)

    # --- Velocity Tracking Error Comparison ---
    if 'dy_out' in processed_data1 and 'dy_act' in processed_data1 and \
       'dy_out' in processed_data2 and 'dy_act' in processed_data2:

        # Calculate velocity errors (absolute value)
        velocity_error1 = np.abs(processed_data1['dy_act'] - processed_data1['dy_out'])
        velocity_error2 = np.abs(processed_data2['dy_act'] - processed_data2['dy_out'])

        # Apply start and end indices
        velocity_error1_subset = velocity_error1[start_idx:end_idx, :, :]
        velocity_error2_subset = velocity_error2[start_idx:end_idx, :, :]

        # Calculate mean and std for both runs
        mean_vel_error1 = np.mean(velocity_error1_subset, axis=1)  # Shape: [time_steps, n_dims]
        std_vel_error1 = np.std(velocity_error1_subset, axis=1)
        mean_vel_error2 = np.mean(velocity_error2_subset, axis=1)
        std_vel_error2 = np.std(velocity_error2_subset, axis=1)

        # Create velocity error comparison plot
        fig_vel, axs_vel = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig_vel.suptitle(f'Absolute Velocity Tracking Error Comparison: {run1_name} vs {run2_name}',
                         fontsize=16)
        axs_vel = np.array(axs_vel) if n_dims > 1 else np.array([axs_vel])

        for i in range(n_dims):
            ax = get_ax(axs_vel, i, n_cols)

            # Plot run 1
            line1 = ax.plot(time_steps1_subset, mean_vel_error1[:, i], linewidth=2,
                           label=f'{run1_name} (mean)')
            color1 = line1[0].get_color()
            ax.fill_between(time_steps1_subset,
                           mean_vel_error1[:, i] - std_vel_error1[:, i],
                           mean_vel_error1[:, i] + std_vel_error1[:, i],
                           alpha=0.2, color=color1)

            # Plot run 2
            line2 = ax.plot(time_steps2_subset, mean_vel_error2[:, i], linewidth=2,
                           label=f'{run2_name} (mean)')
            color2 = line2[0].get_color()
            ax.fill_between(time_steps2_subset,
                           mean_vel_error2[:, i] - std_vel_error2[:, i],
                           mean_vel_error2[:, i] + std_vel_error2[:, i],
                           alpha=0.2, color=color2)

            # Add mean tracking error lines for this dimension
            dim_mean_vel_error1 = np.mean(mean_vel_error1[:, i])
            dim_mean_vel_error2 = np.mean(mean_vel_error2[:, i])
            ax.axhline(y=dim_mean_vel_error1, color=color1, linestyle=(0, (5, 5)), alpha=0.5,
                      linewidth=1.5, label=f'{run1_name} avg')
            ax.axhline(y=dim_mean_vel_error2, color=color2, linestyle=(0, (5, 5)), alpha=0.5,
                      linewidth=1.5, label=f'{run2_name} avg')

            # Add zero reference line
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

            # Use velocity labels
            vel_labels = [f"{lbl} Rate" for lbl in labels] if labels else [f'Dimension {i}' for i in range(n_dims)]
            vel_units = [f"{unit}/s" for unit in units] if units else [''] * n_dims

            label = vel_labels[i] if i < len(vel_labels) else f'Dimension {i}'
            unit = vel_units[i] if i < len(vel_units) else ''
            ax.set_title(f'{label} Error', fontsize=10)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(f'Error ({unit})')
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(loc='best', fontsize=7)

        # Hide unused subplots
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs_vel, i, n_cols)
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save velocity error figure
        output_path_vel = os.path.join(save_dir, 'velocity_tracking_error_comparison.png')
        plt.savefig(output_path_vel, dpi=300, bbox_inches='tight')
        print(f"Saved velocity error comparison plot to: {output_path_vel}")

        plt.close(fig_vel)

        # Create velocity std dev comparison plot
        fig_vel_std, axs_vel_std = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig_vel_std.suptitle(f'Velocity Tracking Error Std Dev Comparison: {run1_name} vs {run2_name}',
                             fontsize=16)
        axs_vel_std = np.array(axs_vel_std) if n_dims > 1 else np.array([axs_vel_std])

        for i in range(n_dims):
            ax = get_ax(axs_vel_std, i, n_cols)

            # Plot std dev for run 1
            line1 = ax.plot(time_steps1_subset, std_vel_error1[:, i], linewidth=2,
                           label=f'{run1_name}')
            color1 = line1[0].get_color()

            # Plot std dev for run 2
            line2 = ax.plot(time_steps2_subset, std_vel_error2[:, i], linewidth=2,
                           label=f'{run2_name}')
            color2 = line2[0].get_color()

            # Add mean std dev lines for this dimension
            dim_mean_vel_std1 = np.mean(std_vel_error1[:, i])
            dim_mean_vel_std2 = np.mean(std_vel_error2[:, i])
            ax.axhline(y=dim_mean_vel_std1, color=color1, linestyle=(0, (5, 5)), alpha=0.5,
                      linewidth=1.5, label=f'{run1_name} avg')
            ax.axhline(y=dim_mean_vel_std2, color=color2, linestyle=(0, (5, 5)), alpha=0.5,
                      linewidth=1.5, label=f'{run2_name} avg')

            label = vel_labels[i] if i < len(vel_labels) else f'Dimension {i}'
            unit = vel_units[i] if i < len(vel_units) else ''
            ax.set_title(f'{label} Std Dev', fontsize=10)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(f'Std Dev ({unit})')
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(loc='best', fontsize=7)

        # Hide unused subplots
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs_vel_std, i, n_cols)
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save velocity std dev figure
        output_path_vel_std = os.path.join(save_dir, 'velocity_tracking_error_std_comparison.png')
        plt.savefig(output_path_vel_std, dpi=300, bbox_inches='tight')
        print(f"Saved velocity std dev comparison plot to: {output_path_vel_std}")

        plt.close(fig_vel_std)

    # --- Detailed Tracking Analysis (Left Ankle + COM) ---
    # Find left ankle indices in the labels
    left_ankle_indices = []
    com_pos_x_idx = None
    com_pos_y_idx = None
    com_pos_z_idx = None
    for i, label in enumerate(labels):
        if 'left' in label.lower() and 'ankle' in label.lower():
            left_ankle_indices.append(i)
        if 'com_pos' in label.lower() and label.lower().endswith('x'):
            com_pos_x_idx = i
        if 'com_pos' in label.lower() and label.lower().endswith('y'):
            com_pos_y_idx = i
        if 'com_pos' in label.lower() and label.lower().endswith('z'):
            com_pos_z_idx = i

    time1 = time_steps1_subset*0.02
    time2 = time_steps2_subset*0.02

    if len(left_ankle_indices) >= 3:  # Need at least x, y, z components
        # Assume first 3 are x, y, z
        left_ankle_x_idx = left_ankle_indices[0]
        left_ankle_y_idx = left_ankle_indices[1] if len(left_ankle_indices) > 1 else left_ankle_indices[0]
        left_ankle_z_idx = left_ankle_indices[2] if len(left_ankle_indices) > 2 else left_ankle_indices[0]

        # Determine number of columns based on whether we found COM x
        n_detail_cols = 4 if com_pos_x_idx is not None else 3
        fig_ankle, axs_ankle = plt.subplots(4, n_detail_cols, figsize=(6 * n_detail_cols, 16))
        # fig_ankle.suptitle(f'Detailed Tracking Analysis: {run1_name} vs {run2_name}', fontsize=16)

        axes_info = [
            ('Left Ankle X', left_ankle_x_idx),
            ('Left Ankle Y', left_ankle_y_idx),
            ('Left Ankle Z', left_ankle_z_idx)
        ]

        if com_pos_x_idx is not None:
            axes_info.append(('COM X', com_pos_x_idx))

        for col, (axis_name, idx) in enumerate(axes_info):
            # Row 0: Position Error
            ax = axs_ankle[0, col]
            line1 = ax.plot(time1, mean_error1[:, idx], linewidth=3, alpha=0.5, label=f'{run1_name}')
            color1 = line1[0].get_color()
            # ax.fill_between(time_steps1_subset,
            #                np.maximum(mean_error1[:, idx] - std_error1[:, idx], 0),
            #                mean_error1[:, idx] + std_error1[:, idx],
            #                alpha=0.2, color=color1)
            line2 = ax.plot(time2, mean_error2[:, idx], linewidth=3, alpha=0.5, label=f'{run2_name}')
            color2 = line2[0].get_color()
            # ax.fill_between(time_steps2_subset,
            #                np.maximum(mean_error2[:, idx] - std_error2[:, idx], 0),
            #                mean_error2[:, idx] + std_error2[:, idx],
            #                alpha=0.2, color=color2)

            # Add mean tracking error lines
            dim_mean_pos_error1 = np.mean(mean_error1[:, idx])
            dim_mean_pos_error2 = np.mean(mean_error2[:, idx])
            ax.axhline(y=dim_mean_pos_error1, color=color1, linestyle=(0, (5, 5)), alpha=1,
                      linewidth=3.5, label=f'{run1_name} avg')
            ax.axhline(y=dim_mean_pos_error2, color=color2, linestyle=(0, (5, 5)), alpha=1,
                      linewidth=3.5, label=f'{run2_name} avg')

            # ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax.set_title(f'Position Error - {axis_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Error ({units[idx] if idx < len(units) else ""})')
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(loc='best', fontsize=7)

            # Row 1: Position Error Std Dev
            ax = axs_ankle[1, col]
            ax.plot(time1, std_error1[:, idx], linewidth=3, alpha=0.5, color=color1, label=f'{run1_name}')
            ax.plot(time2, std_error2[:, idx], linewidth=3, alpha=0.5, color=color2, label=f'{run2_name}')

            # Add mean std dev lines
            dim_mean_pos_std1 = np.mean(std_error1[:, idx])
            dim_mean_pos_std2 = np.mean(std_error2[:, idx])
            ax.axhline(y=dim_mean_pos_std1, color=color1, linestyle=(0, (5, 5)), alpha=1,
                      linewidth=3.5, label=f'{run1_name} avg')
            ax.axhline(y=dim_mean_pos_std2, color=color2, linestyle=(0, (5, 5)), alpha=1,
                      linewidth=3.5, label=f'{run2_name} avg')

            ax.set_title(f'Position Error Std Dev - {axis_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Std Dev ({units[idx] if idx < len(units) else ""})')
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(loc='best', fontsize=7)

            # Row 2: Velocity Error (if available)
            if 'dy_out' in processed_data1 and 'dy_act' in processed_data1 and \
               'dy_out' in processed_data2 and 'dy_act' in processed_data2:
                ax = axs_ankle[2, col]
                ax.plot(time1, mean_vel_error1[:, idx], linewidth=3, alpha=0.5, color=color1, label=f'{run1_name}')
                # ax.fill_between(time_steps1_subset,
                #                np.maximum(mean_vel_error1[:, idx] - std_vel_error1[:, idx], 0),
                #                mean_vel_error1[:, idx] + std_vel_error1[:, idx],
                #                alpha=0.2, color=color1)
                ax.plot(time2, mean_vel_error2[:, idx], linewidth=3, alpha=0.5, color=color2, label=f'{run2_name}')
                # ax.fill_between(time_steps2_subset,
                #                np.maximum(mean_vel_error2[:, idx] - std_vel_error2[:, idx], 0),
                #                mean_vel_error2[:, idx] + std_vel_error2[:, idx],
                #                alpha=0.2, color=color2)

                # Add mean velocity error lines
                dim_mean_vel_error1 = np.mean(mean_vel_error1[:, idx])
                dim_mean_vel_error2 = np.mean(mean_vel_error2[:, idx])
                ax.axhline(y=dim_mean_vel_error1, color=color1, linestyle=(0, (5, 5)), alpha=1,
                          linewidth=3.5, label=f'{run1_name} avg')
                ax.axhline(y=dim_mean_vel_error2, color=color2, linestyle=(0, (5, 5)), alpha=1,
                          linewidth=3.5, label=f'{run2_name} avg')

                # ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
                ax.set_title(f'Velocity Error - {axis_name}', fontsize=12, fontweight='bold')
                vel_unit = f"{units[idx]}/s" if idx < len(units) else ""
                ax.set_ylabel(f'Error ({vel_unit})')
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.legend(loc='best', fontsize=7)

                # Row 3: Velocity Error Std Dev
                ax = axs_ankle[3, col]
                ax.plot(time1, std_vel_error1[:, idx], linewidth=3, alpha=0.5, color=color1, label=f'{run1_name}')
                ax.plot(time2, std_vel_error2[:, idx], linewidth=3, alpha=0.5, color=color2, label=f'{run2_name}')

                # Add mean velocity std dev lines
                dim_mean_vel_std1 = np.mean(std_vel_error1[:, idx])
                dim_mean_vel_std2 = np.mean(std_vel_error2[:, idx])
                ax.axhline(y=dim_mean_vel_std1, color=color1, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=3.5, label=f'{run1_name} avg')
                ax.axhline(y=dim_mean_vel_std2, color=color2, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=3.5, label=f'{run2_name} avg')

                ax.set_title(f'Velocity Error Std Dev - {axis_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'Std Dev ({vel_unit})')
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.legend(loc='best', fontsize=7)
            else:
                # Hide velocity rows if data not available
                axs_ankle[2, col].set_visible(False)
                axs_ankle[3, col].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save as PNG
        output_path_ankle_png = os.path.join(save_dir, 'detailed_tracking_analysis.png')
        plt.savefig(output_path_ankle_png, dpi=300, bbox_inches='tight')
        print(f"Saved detailed tracking analysis (PNG) to: {output_path_ankle_png}")

        # Save as PDF
        output_path_ankle_pdf = os.path.join(save_dir, 'detailed_tracking_analysis.pdf')
        plt.savefig(output_path_ankle_pdf, bbox_inches='tight')
        print(f"Saved detailed tracking analysis (PDF) to: {output_path_ankle_pdf}")

        plt.close(fig_ankle)

        # --- Paper-Ready Figure (Transposed Layout) ---
        # Only create if we have COM x data
        if com_pos_x_idx is not None and 'dy_out' in processed_data1 and 'dy_act' in processed_data1:
            # 3 rows (Left Ankle X, Left Ankle Y, COM X) x 4 columns (Position, Position Std, Velocity, Velocity Std)
            fig_paper, axs_paper = plt.subplots(3, 4, figsize=(16, 5))

            # Define rows: Left Ankle X, Left Ankle Y, COM X
            row_info = [
                ('Swing Ankle X', left_ankle_x_idx),
                ('Swing Ankle Y', left_ankle_y_idx),
                ('COM X', com_pos_x_idx)
            ]

            for row, (row_name, idx) in enumerate(row_info):
                # Column 0: Position Error
                ax = axs_paper[row, 0]
                line1 = ax.plot(time1, mean_error1[:, idx], linewidth=2, alpha=0.5, label=f'{run1_name}')
                color1 = line1[0].get_color()
                line2 = ax.plot(time2, mean_error2[:, idx], linewidth=2, alpha=0.5, label=f'{run2_name}')
                color2 = line2[0].get_color()

                dim_mean_pos_error1 = np.mean(mean_error1[:, idx])
                dim_mean_pos_error2 = np.mean(mean_error2[:, idx])
                ax.axhline(y=dim_mean_pos_error1, color=color1, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run1_name} avg')
                ax.axhline(y=dim_mean_pos_error2, color=color2, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run2_name} avg')

                if row == 0:
                    ax.set_title('Position Error', fontsize=14, fontweight='bold')
                ax.set_ylabel(f'{row_name}\n({units[idx] if idx < len(units) else ""})', fontsize=11)
                ax.grid(True, alpha=0.3)
                if row == 2:
                    ax.set_xlabel('Time (s)', fontsize=11)
                if row == 0:
                    ax.legend(loc='best', fontsize=8)

                # Column 1: Position Error Std Dev
                ax = axs_paper[row, 1]
                ax.plot(time1, std_error1[:, idx], linewidth=2, alpha=0.5, color=color1, label=f'{run1_name}')
                ax.plot(time2, std_error2[:, idx], linewidth=2, alpha=0.5, color=color2, label=f'{run2_name}')

                dim_mean_pos_std1 = np.mean(std_error1[:, idx])
                dim_mean_pos_std2 = np.mean(std_error2[:, idx])
                ax.axhline(y=dim_mean_pos_std1, color=color1, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run1_name} avg')
                ax.axhline(y=dim_mean_pos_std2, color=color2, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run2_name} avg')

                if row == 0:
                    ax.set_title('Position Error Std Dev', fontsize=14, fontweight='bold')
                ax.set_ylabel(f'({units[idx] if idx < len(units) else ""})', fontsize=11)
                ax.grid(True, alpha=0.3)
                if row == 2:
                    ax.set_xlabel('Time (s)', fontsize=11)

                # Column 2: Velocity Error
                ax = axs_paper[row, 2]
                ax.plot(time1, mean_vel_error1[:, idx], linewidth=2, alpha=0.5, color=color1, label=f'{run1_name}')
                ax.plot(time2, mean_vel_error2[:, idx], linewidth=2, alpha=0.5, color=color2, label=f'{run2_name}')

                dim_mean_vel_error1 = np.mean(mean_vel_error1[:, idx])
                dim_mean_vel_error2 = np.mean(mean_vel_error2[:, idx])
                ax.axhline(y=dim_mean_vel_error1, color=color1, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run1_name} avg')
                ax.axhline(y=dim_mean_vel_error2, color=color2, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run2_name} avg')

                vel_unit = f"{units[idx]}/s" if idx < len(units) else ""
                if row == 0:
                    ax.set_title('Velocity Error', fontsize=14, fontweight='bold')
                ax.set_ylabel(f'({vel_unit})', fontsize=11)
                ax.grid(True, alpha=0.3)
                if row == 2:
                    ax.set_xlabel('Time (s)', fontsize=11)

                # Column 3: Velocity Error Std Dev
                ax = axs_paper[row, 3]
                ax.plot(time1, std_vel_error1[:, idx], linewidth=2, alpha=0.5, color=color1, label=f'{run1_name}')
                ax.plot(time2, std_vel_error2[:, idx], linewidth=2, alpha=0.5, color=color2, label=f'{run2_name}')

                dim_mean_vel_std1 = np.mean(std_vel_error1[:, idx])
                dim_mean_vel_std2 = np.mean(std_vel_error2[:, idx])
                ax.axhline(y=dim_mean_vel_std1, color=color1, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run1_name} avg')
                ax.axhline(y=dim_mean_vel_std2, color=color2, linestyle=(0, (5, 5)), alpha=1.0,
                          linewidth=2, label=f'{run2_name} avg')

                if row == 0:
                    ax.set_title('Velocity Error Std Dev', fontsize=14, fontweight='bold')
                ax.set_ylabel(f'({vel_unit})', fontsize=11)
                ax.grid(True, alpha=0.3)
                if row == 2:
                    ax.set_xlabel('Time (s)', fontsize=11)

            plt.tight_layout()

            # Save paper-ready figure
            output_path_paper_png = os.path.join(save_dir, 'paper_ready_tracking_analysis.png')
            plt.savefig(output_path_paper_png, dpi=300, bbox_inches='tight')
            print(f"Saved paper-ready tracking analysis (PNG) to: {output_path_paper_png}")

            output_path_paper_pdf = os.path.join(save_dir, 'paper_ready_tracking_analysis.pdf')
            plt.savefig(output_path_paper_pdf, bbox_inches='tight')
            print(f"Saved paper-ready tracking analysis (PDF) to: {output_path_paper_pdf}")

            # Print all mean values for the paper-ready figure (and swing ankle z, com y, com z)
            print("\n" + "="*80)
            print("Paper-Ready Figure Mean Values (+ Swing Ankle Z, COM Y, COM Z)")
            print("="*80)

            # Add swing ankle z and COM Y/Z to the print output
            print_info = row_info + [('Swing Ankle Z', left_ankle_z_idx)]
            if com_pos_y_idx is not None:
                print_info.append(('COM Y', com_pos_y_idx))
            if com_pos_z_idx is not None:
                print_info.append(('COM Z', com_pos_z_idx))

            for row_name, idx in print_info:
                print(f"\n{row_name}:")

                # Position Error
                dim_mean_pos_error1 = np.mean(mean_error1[:, idx])
                dim_mean_pos_error2 = np.mean(mean_error2[:, idx])
                unit = units[idx] if idx < len(units) else ""
                pct_diff_pos = ((dim_mean_pos_error1 - dim_mean_pos_error2) / dim_mean_pos_error2) * 100 if dim_mean_pos_error2 != 0 else 0
                print(f"  Position Error:         {run1_name}: {dim_mean_pos_error1:.6f} {unit}  |  {run2_name}: {dim_mean_pos_error2:.6f} {unit}  |  Diff: {pct_diff_pos:+.2f}%")

                # Position Error Std Dev
                dim_mean_pos_std1 = np.mean(std_error1[:, idx])
                dim_mean_pos_std2 = np.mean(std_error2[:, idx])
                pct_diff_pos_std = ((dim_mean_pos_std1 - dim_mean_pos_std2) / dim_mean_pos_std2) * 100 if dim_mean_pos_std2 != 0 else 0
                print(f"  Position Error Std Dev: {run1_name}: {dim_mean_pos_std1:.6f} {unit}  |  {run2_name}: {dim_mean_pos_std2:.6f} {unit}  |  Diff: {pct_diff_pos_std:+.2f}%")

                # Velocity Error
                dim_mean_vel_error1 = np.mean(mean_vel_error1[:, idx])
                dim_mean_vel_error2 = np.mean(mean_vel_error2[:, idx])
                vel_unit = f"{unit}/s" if unit else ""
                pct_diff_vel = ((dim_mean_vel_error1 - dim_mean_vel_error2) / dim_mean_vel_error2) * 100 if dim_mean_vel_error2 != 0 else 0
                print(f"  Velocity Error:         {run1_name}: {dim_mean_vel_error1:.6f} {vel_unit}  |  {run2_name}: {dim_mean_vel_error2:.6f} {vel_unit}  |  Diff: {pct_diff_vel:+.2f}%")

                # Velocity Error Std Dev
                dim_mean_vel_std1 = np.mean(std_vel_error1[:, idx])
                dim_mean_vel_std2 = np.mean(std_vel_error2[:, idx])
                pct_diff_vel_std = ((dim_mean_vel_std1 - dim_mean_vel_std2) / dim_mean_vel_std2) * 100 if dim_mean_vel_std2 != 0 else 0
                print(f"  Velocity Error Std Dev: {run1_name}: {dim_mean_vel_std1:.6f} {vel_unit}  |  {run2_name}: {dim_mean_vel_std2:.6f} {vel_unit}  |  Diff: {pct_diff_vel_std:+.2f}%")
            print("="*80 + "\n")

            plt.close(fig_paper)

    # Print comprehensive statistics for ALL dimensions
    print("\n" + "="*80)
    print("COMPREHENSIVE TRACKING STATISTICS - ALL DIMENSIONS")
    print("="*80)

    for i in range(n_dims):
        label = labels[i] if i < len(labels) else f'Dimension {i}'
        unit = units[i] if i < len(units) else ""

        print(f"\n{label}:")

        # Position Error
        dim_mean_pos_error1 = np.mean(mean_error1[:, i])
        dim_mean_pos_error2 = np.mean(mean_error2[:, i])
        pct_diff_pos = ((dim_mean_pos_error1 - dim_mean_pos_error2) / dim_mean_pos_error2) * 100 if dim_mean_pos_error2 != 0 else 0
        print(f"  Position Error:         {run1_name}: {dim_mean_pos_error1:.6f} {unit}  |  {run2_name}: {dim_mean_pos_error2:.6f} {unit}  |  Diff: {pct_diff_pos:+.2f}%")

        # Position Error Std Dev
        dim_mean_pos_std1 = np.mean(std_error1[:, i])
        dim_mean_pos_std2 = np.mean(std_error2[:, i])
        pct_diff_pos_std = ((dim_mean_pos_std1 - dim_mean_pos_std2) / dim_mean_pos_std2) * 100 if dim_mean_pos_std2 != 0 else 0
        print(f"  Position Error Std Dev: {run1_name}: {dim_mean_pos_std1:.6f} {unit}  |  {run2_name}: {dim_mean_pos_std2:.6f} {unit}  |  Diff: {pct_diff_pos_std:+.2f}%")

        # Velocity Error (if available)
        if 'dy_out' in processed_data1 and 'dy_act' in processed_data1 and \
           'dy_out' in processed_data2 and 'dy_act' in processed_data2:
            dim_mean_vel_error1 = np.mean(mean_vel_error1[:, i])
            dim_mean_vel_error2 = np.mean(mean_vel_error2[:, i])
            vel_unit = f"{unit}/s" if unit else ""
            pct_diff_vel = ((dim_mean_vel_error1 - dim_mean_vel_error2) / dim_mean_vel_error2) * 100 if dim_mean_vel_error2 != 0 else 0
            print(f"  Velocity Error:         {run1_name}: {dim_mean_vel_error1:.6f} {vel_unit}  |  {run2_name}: {dim_mean_vel_error2:.6f} {vel_unit}  |  Diff: {pct_diff_vel:+.2f}%")

            # Velocity Error Std Dev
            dim_mean_vel_std1 = np.mean(std_vel_error1[:, i])
            dim_mean_vel_std2 = np.mean(std_vel_error2[:, i])
            pct_diff_vel_std = ((dim_mean_vel_std1 - dim_mean_vel_std2) / dim_mean_vel_std2) * 100 if dim_mean_vel_std2 != 0 else 0
            print(f"  Velocity Error Std Dev: {run1_name}: {dim_mean_vel_std1:.6f} {vel_unit}  |  {run2_name}: {dim_mean_vel_std2:.6f} {vel_unit}  |  Diff: {pct_diff_vel_std:+.2f}%")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare position tracking error between two policy runs'
    )
    parser.add_argument(
        'run1_dir',
        type=str,
        help='Path to first run directory containing logged .pkl files'
    )
    parser.add_argument(
        'run2_dir',
        type=str,
        help='Path to second run directory containing logged .pkl files'
    )
    parser.add_argument(
        '--run1_name',
        type=str,
        default=None,
        help='Label for first run in plot legend (default: directory name)'
    )
    parser.add_argument(
        '--run2_name',
        type=str,
        default=None,
        help='Label for second run in plot legend (default: directory name)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save comparison plot (default: current directory)'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=50,
        help='Time step to start plotting from (default: 50)'
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        default=None,
        help='Time step to end plotting at (default: None, uses all data)'
    )

    args = parser.parse_args()

    # Verify directories exist
    if not os.path.exists(args.run1_dir):
        print(f"Error: Run 1 directory does not exist: {args.run1_dir}")
        return
    if not os.path.exists(args.run2_dir):
        print(f"Error: Run 2 directory does not exist: {args.run2_dir}")
        return

    # Compare tracking error
    compare_tracking_error(
        args.run1_dir,
        args.run2_dir,
        run1_name=args.run1_name,
        run2_name=args.run2_name,
        save_dir=args.save_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )


if __name__ == "__main__":
    main()
