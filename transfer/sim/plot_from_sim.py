import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sim.log_utils import extract_data, find_most_recent_timestamped_folder


# Make plots
def plot_joints_and_actions(data, save_dir, joint_names=None):
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 16))

    FLOATING_BASE = 7

    for i in range(7):
        for j in range(3):
            joint_idx = i + 7 * j
            joint_label = joint_names[joint_idx] if joint_names and joint_idx < len(joint_names) else f"joint {joint_idx}"
            
            axes[i, j].plot(data["time"], data["qpos"][:, joint_idx + FLOATING_BASE], label="qpos")
            axes[i, j].plot(data["time"], data["action"][:, joint_idx], label="action")
            axes[i, j].set_xlabel("time", fontsize=10)
            axes[i, j].set_ylabel(f"{joint_label} (rad)", fontsize=10)
            axes[i, j].grid()
            axes[i, j].legend(fontsize=8)
            axes[i, j].tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "joints_and_actions.png"), dpi=150, bbox_inches='tight')


def plot_torques(data, save_dir, joint_names=None, torque_limits=None):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(16, 14))

    for i in range(6):
        for j in range(2):
            joint_idx = i + 6 * j
            joint_label = joint_names[joint_idx] if joint_names and joint_idx < len(joint_names) else f"joint {joint_idx}"
            
            # Plot actual torque
            axes[i, j].plot(data["time"], data["torque"][:, joint_idx], label="Actual")
            
            # Plot torque limits as dashed lines if available
            if torque_limits and joint_idx < len(torque_limits):
                torque_min, torque_max = torque_limits[joint_idx]
                axes[i, j].axhline(y=torque_max, color='r', linestyle='--', alpha=0.7, label=f'Max ({torque_max:.1f})')
                axes[i, j].axhline(y=torque_min, color='r', linestyle='--', alpha=0.7, label=f'Min ({torque_min:.1f})')
            
            axes[i, j].set_xlabel("time", fontsize=10)
            axes[i, j].set_ylabel(f"{joint_label} torque (Nm)", fontsize=10)
            axes[i, j].grid()
            axes[i, j].legend(fontsize=8)
            axes[i, j].tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "torques.png"), dpi=150, bbox_inches='tight')


def plot_vels(data, save_dir, joint_names=None):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))

    FLOATING_BASE = 6

    for i in range(6):
        for j in range(2):
            joint_idx = i + 6 * j
            joint_label = joint_names[joint_idx] if joint_names and joint_idx < len(joint_names) else f"joint {joint_idx}"
            
            axes[i, j].plot(data["time"], data["qvel"][:, joint_idx + FLOATING_BASE])
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"{joint_label} vel (rad/s)")
            axes[i, j].grid()

    plt.savefig(os.path.join(save_dir, "vels.png"))


def plot_base(data, save_dir):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

    for i in range(3):
        for j in range(2):
            if j == 0:
                axes[i, j].plot(data["time"], data["qpos"][:, i])
                axes[i, j].set_xlabel("time")
                axes[i, j].set_ylabel(f"qpos {i} (m)")
                axes[i, j].grid()
            else:
                axes[i, j].plot(data["time"], data["qvel"][:, i])
                axes[i, j].set_xlabel("time")
                axes[i, j].set_ylabel(f"qvel {i} (m/s)")
                axes[i, j].grid()

    plt.savefig(os.path.join(save_dir, "floating_base.png"))


def plot_ankles(data):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

    axes[0, 0].plot(data["time"], data["left_ankle_pos"][:, 0])
    axes[0, 1].plot(data["time"], data["left_ankle_pos"][:, 1])
    axes[0, 2].plot(data["time"], data["left_ankle_pos"][:, 2])
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("left_ankle_pos x (m)")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("left_ankle_pos y (m)")
    axes[0, 2].set_xlabel("time")
    axes[0, 2].set_ylabel("left_ankle_pos z (m)")

    axes[1, 0].plot(data["time"], data["right_ankle_pos"][:, 0])
    axes[1, 1].plot(data["time"], data["right_ankle_pos"][:, 1])
    axes[1, 2].plot(data["time"], data["right_ankle_pos"][:, 2])
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("right_ankle_pos x (m)")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("right_ankle_pos y (m)")
    axes[1, 2].set_xlabel("time")
    axes[1, 2].set_ylabel("right_ankle_pos z (m)")


def plot_velocity_comparison(data, save_dir):
    """Plot comparison between commanded and actual velocities."""
    time = data["time"]
    qvel = data["qvel"]
    commanded_vel = data["commanded_vel"]

    # Extract base velocities (first 3 elements of qvel)
    base_vel = qvel[:, :3]

    # Create figure with 3 subplots for x, y, and angular velocities
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Commanded vs Actual Velocities")

    # Plot x velocity
    axes[0].plot(time, commanded_vel[:, 0], "r--", label="Commanded")
    axes[0].plot(time, base_vel[:, 0], "b-", label="Actual")
    axes[0].set_ylabel("X Velocity (m/s)")
    axes[0].legend()
    axes[0].grid(True)

    # Plot y velocity
    axes[1].plot(time, commanded_vel[:, 1], "r--", label="Commanded")
    axes[1].plot(time, base_vel[:, 1], "b-", label="Actual")
    axes[1].set_ylabel("Y Velocity (m/s)")
    axes[1].legend()
    axes[1].grid(True)

    # Plot angular velocity
    axes[2].plot(time, commanded_vel[:, 2], "r--", label="Commanded")
    axes[2].plot(time, qvel[:, 5], "b-", label="Actual")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angular Velocity (rad/s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.savefig(os.path.join(save_dir, "velocity_comparison.png"))


def plot_position_comparison(data, save_dir):
    """Plot comparison between desired and actual positions."""
    time = data["time"]
    qpos = data["qpos"]
    commanded_vel = data["commanded_vel"]

    # Extract base position (first 3 elements of qpos)
    actual_pos = qpos[:, :3]
    # need to extract the initial yaw
    quat = qpos[:, 4:8]
    yaw = 2 * np.arctan2(quat[:, 2], quat[:, 3])
    actual_pos[:, 2] = yaw
    actual_yaw = yaw

    # Calculate desired position by integrating commanded velocity
    dt = time[1] - time[0]  # Assuming constant time step
    desired_pos = np.zeros_like(actual_pos)
    desired_pos[0] = actual_pos[0]  # Start from actual position

    for i in range(1, len(time)):
        desired_pos[i] = desired_pos[i - 1] + commanded_vel[i - 1] * dt

    # Create figure with 3 subplots for x, y, and angular positions
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Desired vs Actual Positions")

    # Plot x position
    axes[0].plot(time, desired_pos[:, 0], "r--", label="Desired")
    axes[0].plot(time, actual_pos[:, 0], "b-", label="Actual")
    axes[0].set_ylabel("X Position (m)")
    axes[0].legend()
    axes[0].grid(True)

    # Plot y position
    axes[1].plot(time, desired_pos[:, 1], "r--", label="Desired")
    axes[1].plot(time, actual_pos[:, 1], "b-", label="Actual")
    axes[1].set_ylabel("Y Position (m)")
    axes[1].legend()
    axes[1].grid(True)

    # Plot angular position
    axes[2].plot(time, desired_pos[:, 2], "r--", label="Desired")
    axes[2].plot(time, actual_yaw, "b-", label="Actual")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angular Position (rad)")
    axes[2].legend()
    axes[2].grid(True)

    plt.savefig(os.path.join(save_dir, "position_comparison.png"))


def create_plots_for_newest():
    # Load in the data from rerun
    log_dir = os.getcwd() + "/logs"
    print(f"Looking for logs in {log_dir}.")
    newest = find_most_recent_timestamped_folder(log_dir)

    print(f"Loading data from {newest}.")

    # TODO: Load in pkl or csv
    # Parse the config file
    with open(os.path.join(newest, "sim_config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        data = extract_data(os.path.join(newest, "sim_log.csv"), config)

    print("============== Data generated using " + config["simulator"] + " ===============")

    print(f"time shape: {data['time'].shape}")
    print(f"qpos shape: {data['qpos'].shape}")
    print(f"qvel shape: {data['qvel'].shape}")
    print(f"torque shape: {data['torque'].shape}")
    print(f"action shape: {data['action'].shape}")
    print(f"left_ankle_pos shape: {data['left_ankle_pos'].shape}")
    print(f"right_ankle_pos shape: {data['right_ankle_pos'].shape}")
    print(f"commanded_vel shape: {data['commanded_vel'].shape}")

    # Get joint names and torque limits from config if available
    joint_names = config.get("joint_names", None)
    torque_limits = config.get("torque_limits", None)
    if joint_names:
        print(f"Using joint names: {joint_names}")
    else:
        print("No joint names found in config, using default labels")
    
    if torque_limits:
        print(f"Using torque limits: {torque_limits}")
    else:
        print("No torque limits found in config")

    # Make a plot
    plot_joints_and_actions(data, newest, joint_names)
    plot_torques(data, newest, joint_names, torque_limits)
    # plot_vels(data, newest, joint_names)
    plot_base(data, newest)
    # import pdb; pdb.set_trace()
    # plot_ankles(data)

    # Plot velocity and position comparisons
    plot_velocity_comparison(data, newest)
    plot_position_comparison(data, newest)


if __name__ == "__main__":
    create_plots_for_newest()
    # plt.show()
