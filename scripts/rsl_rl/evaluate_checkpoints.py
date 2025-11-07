#!/usr/bin/env python3
"""
Evaluate all checkpoints from a training run.

This script runs the play policy for every checkpoint in a training run directory,
logs the tracking data, and plots the average tracking error over training iterations.
"""

import argparse
import os
import sys
import glob
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def find_checkpoints(run_dir):
    """Find all model checkpoints in the run directory and sort by iteration"""
    checkpoint_files = glob.glob(os.path.join(run_dir, "model_*.pt"))

    # Sort by checkpoint number
    def get_checkpoint_num(path):
        basename = os.path.basename(path)
        # Extract number from model_XXX.pt
        num_str = basename.replace("model_", "").replace(".pt", "")
        return int(num_str)

    checkpoint_files.sort(key=get_checkpoint_num)

    checkpoints = []
    for ckpt_file in checkpoint_files:
        num = get_checkpoint_num(ckpt_file)
        checkpoints.append({
            'path': ckpt_file,
            'iteration': num,
            'basename': os.path.basename(ckpt_file)
        })

    return checkpoints


def load_data(log_dir):
    """Load all pickle files from the log directory"""
    data = {}
    for pkl_file in glob.glob(os.path.join(log_dir, "*.pkl")):
        var_name = os.path.basename(pkl_file).replace(".pkl", "")
        with open(pkl_file, "rb") as f:
            data[var_name] = pickle.load(f)
    return data


def process_data(data):
    """Convert lists to numpy arrays and handle torch tensors"""
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    return processed_data


def calculate_tracking_metrics(log_dir, start_idx=50):
    """Calculate average tracking error from logged data"""
    # Load and process data
    data = load_data(log_dir)

    if 'y_out' not in data or 'y_act' not in data:
        print(f"Warning: Missing y_out or y_act in {log_dir}")
        return None

    processed_data = process_data(data)

    # Calculate position error (absolute value)
    position_error = np.abs(processed_data['y_act'] - processed_data['y_out'])

    # Skip initial transient
    position_error = position_error[start_idx:, :, :]

    # Calculate mean error across environments and time for each dimension
    # Shape: position_error is [time_steps, n_envs, n_dims]
    mean_error_per_dim = np.mean(position_error, axis=(0, 1))  # Average over time and envs

    return mean_error_per_dim


def run_play_policy(checkpoint_path, run_name, env_type, exp_name, num_envs,
                   video_length, play_log_dir, checkpoint_num):
    """Run play_policy.py for a specific checkpoint"""

    # Create log directory for this checkpoint
    ckpt_log_dir = os.path.join(play_log_dir, f"checkpoint_{checkpoint_num}")
    os.makedirs(ckpt_log_dir, exist_ok=True)

    # Construct the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "scripts/rsl_rl/play_policy.py",
        "--env_type", env_type,
        "--exp_name", exp_name,
        "--load_run", run_name,
        f"--checkpoint=model_{checkpoint_num}",
        "--num_envs", str(num_envs),
        "--video_length", str(video_length),
        "--log_data",
        "--play_log_dir", ckpt_log_dir,
        "--headless",
    ]

    print(f"\nRunning checkpoint {checkpoint_num}...")
    print(f"Command: {' '.join(cmd)}")

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running checkpoint {checkpoint_num}:")
        print(result.stderr)
        print(result.stdout)
        return None

    print(f"Checkpoint {checkpoint_num} completed successfully")
    return ckpt_log_dir


def format_joint_name(joint_name):
    """Format joint name for better readability in plots"""
    formatted = joint_name.replace('_joint', '').replace('_', ' ')
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    return formatted


def get_constraint_labels(data):
    """Get labels for virtual constraints from logged data"""
    # Check if we have axis_names in the data (for end effector trajectories)
    if 'axis_names' in data and data['axis_names']:
        axis_names_data = data['axis_names'][0] if isinstance(data['axis_names'], list) else data['axis_names']
        if isinstance(axis_names_data, list):
            return [axis_info['name'] for axis_info in axis_names_data]

    # Otherwise, assume joint trajectory and use G1 joint names
    if 'y_out' in data:
        processed_data = {}
        for key, values in data.items():
            if isinstance(values[0], torch.Tensor):
                processed_data[key] = np.array([v.cpu().numpy() for v in values])
            else:
                processed_data[key] = np.array(values)

        n_dims = processed_data['y_out'].shape[2]
        g1_joint_names = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
            'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint',
            'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint',
            'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'
        ]
        return [format_joint_name(name) for name in g1_joint_names[:n_dims]]

    return []


def plot_tracking_error_vs_iteration(iterations, errors, labels, save_path, trajectory_type='joint'):
    """Plot average tracking error for each constraint vs training iteration"""

    n_dims = len(labels)
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    fig.suptitle('Average Tracking Error vs Training Iteration', fontsize=16)

    if n_dims == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten() if n_dims > 1 else np.array([axs])

    for i in range(n_dims):
        ax = axs[i]

        # Extract errors for this dimension across all iterations
        errors_for_dim = [err[i] for err in errors]

        # Plot
        ax.plot(iterations, errors_for_dim, linewidth=2, marker='o', markersize=4)
        ax.set_title(labels[i], fontsize=10)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Avg Absolute Error (rad)')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_dims, len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved tracking error plot to: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all checkpoints from a training run'
    )
    parser.add_argument(
        'run_dir',
        type=str,
        help='Path to the training run directory containing model_*.pt checkpoints'
    )
    parser.add_argument(
        '--env_type',
        type=str,
        required=True,
        help='Environment type (e.g., hzd_clf_custom, vanilla, etc.)'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name (e.g., hzd, baseline, lip). Auto-detected from run directory if not provided.'
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=64,
        help='Number of environments to simulate (default: 64)'
    )
    parser.add_argument(
        '--video_length',
        type=int,
        default=400,
        help='Length of simulation in steps (default: 400)'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=50,
        help='Time step to start analyzing from (default: 50)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save evaluation results (default: run_dir/checkpoint_evaluation)'
    )
    parser.add_argument(
        '--skip_simulation',
        action='store_true',
        default=False,
        help='Skip running simulations and only regenerate plots from existing logged data'
    )

    args = parser.parse_args()

    # Verify run directory exists
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory does not exist: {args.run_dir}")
        return

    # Find all checkpoints
    print(f"Searching for checkpoints in: {args.run_dir}")
    checkpoints = find_checkpoints(args.run_dir)

    if not checkpoints:
        print("Error: No checkpoints found in the run directory")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    for ckpt in checkpoints:
        print(f"  - Iteration {ckpt['iteration']}: {ckpt['basename']}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_dir, "checkpoint_evaluation")

    os.makedirs(args.output_dir, exist_ok=True)
    play_log_dir = os.path.join(args.output_dir, "logged_data")
    os.makedirs(play_log_dir, exist_ok=True)

    print(f"\nEvaluation results will be saved to: {args.output_dir}")

    # Get run name and experiment name from directory structure
    run_name = os.path.basename(args.run_dir)

    # Auto-detect exp_name if not provided
    # Expected structure: logs/g1_policies/<exp_name>/<env_type>/<run_name>
    if args.exp_name is None:
        # Get the parent of env_type directory
        run_dir_abs = os.path.abspath(args.run_dir)
        env_type_dir = os.path.dirname(run_dir_abs)  # .../hzd_clf_custom
        exp_name_dir = os.path.dirname(env_type_dir)  # .../hzd
        args.exp_name = os.path.basename(exp_name_dir)
        print(f"Auto-detected exp_name: {args.exp_name}")

    # Evaluate each checkpoint
    iterations = []
    errors = []

    for ckpt in checkpoints:
        ckpt_log_dir = os.path.join(play_log_dir, f"checkpoint_{ckpt['iteration']}")

        # Run simulation or skip if flag is set
        if not args.skip_simulation:
            # Run play policy
            ckpt_log_dir = run_play_policy(
                checkpoint_path=ckpt['path'],
                run_name=run_name,
                env_type=args.env_type,
                exp_name=args.exp_name,
                num_envs=args.num_envs,
                video_length=args.video_length,
                play_log_dir=play_log_dir,
                checkpoint_num=ckpt['iteration']
            )

            if ckpt_log_dir is None:
                raise RuntimeError(f"Failed to run play_policy for checkpoint {ckpt['iteration']}")
        else:
            # Check if logged data exists
            if not os.path.exists(ckpt_log_dir):
                raise RuntimeError(f"No logged data found for checkpoint {ckpt['iteration']} at {ckpt_log_dir}")
            print(f"Using existing logged data for checkpoint {ckpt['iteration']}")

        # Calculate tracking metrics
        mean_error = calculate_tracking_metrics(ckpt_log_dir, start_idx=args.start_idx)

        if mean_error is None:
            raise RuntimeError(f"Failed to calculate tracking metrics for checkpoint {ckpt['iteration']}")

        iterations.append(ckpt['iteration'])
        errors.append(mean_error)

        print(f"Iteration {ckpt['iteration']}: Mean error = {np.mean(mean_error):.6f}")

    if not iterations:
        print("Error: No successful evaluations")
        return

    # Get constraint labels from the first checkpoint's logged data
    first_checkpoint_log_dir = os.path.join(play_log_dir, f"checkpoint_{checkpoints[0]['iteration']}")
    first_checkpoint_data = load_data(first_checkpoint_log_dir)
    labels = get_constraint_labels(first_checkpoint_data)

    if not labels:
        print("Warning: Could not determine constraint labels, using generic names")
        n_dims = len(errors[0])
        labels = [f'Constraint {i}' for i in range(n_dims)]

    # Plot results
    plot_path = os.path.join(args.output_dir, "tracking_error_vs_iteration.png")
    plot_tracking_error_vs_iteration(iterations, errors, labels, plot_path)

    # Save raw data
    results = {
        'iterations': iterations,
        'errors': errors,
        'labels': labels,
        'run_dir': args.run_dir,
        'env_type': args.env_type
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved evaluation results to: {results_path}")

    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
