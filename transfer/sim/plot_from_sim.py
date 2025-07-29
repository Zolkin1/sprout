import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from log_utils import find_most_recent_timestamped_folder, extract_data


# Make plots
def plot_joints_and_actions(data, save_dir):
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(10, 10))

    FLOATING_BASE = 7

    for i in range(7):
        for j in range(3):
            axes[i, j].plot(data["time"], data["qpos"][:, i + 7 * j + FLOATING_BASE],label="qpos")
            axes[i, j].plot(data["time"], data["action"][:, i + 7 * j],label="action")
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"qpos {i + 7*j + FLOATING_BASE} (rad)")
            axes[i, j].grid()
            axes[i, j].legend()

    plt.savefig(os.path.join(save_dir, "joints_and_actions.png"))

def plot_torques(data, save_dir):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))

    for i in range(6):
        for j in range(2):
            axes[i, j].plot(data["time"], data["torque"][:, i + 6 * j])
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"torque {i + 6*j} (Nm)")
            axes[i, j].grid()

    plt.savefig(os.path.join(save_dir, "torques.png"))

def plot_vels(data, save_dir):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))

    FLOATING_BASE = 6

    for i in range(6):
        for j in range(2):
            axes[i, j].plot(data["time"], data["qvel"][:, i + 6 * j + FLOATING_BASE])
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"qvel {i + 6*j + FLOATING_BASE} (rad/s)")
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
    axes[0, 0].set_ylabel(f"left_ankle_pos x (m)")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(f"left_ankle_pos y (m)")
    axes[0, 2].set_xlabel("time")
    axes[0, 2].set_ylabel(f"left_ankle_pos z (m)")

    axes[1, 0].plot(data["time"], data["right_ankle_pos"][:, 0])
    axes[1, 1].plot(data["time"], data["right_ankle_pos"][:, 1])
    axes[1, 2].plot(data["time"], data["right_ankle_pos"][:, 2])
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(f"right_ankle_pos x (m)")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel(f"right_ankle_pos y (m)")
    axes[1, 2].set_xlabel("time")
    axes[1, 2].set_ylabel(f"right_ankle_pos z (m)")

def plot_velocity_comparison(data, save_dir):
    """Plot comparison between commanded and actual velocities."""
    time = data['time']
    qvel = data['qvel']
    commanded_vel = data['commanded_vel']
    
    # Extract base velocities (first 3 elements of qvel)
    base_vel = qvel[:, :3]
    
    # Create figure with 3 subplots for x, y, and angular velocities
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Commanded vs Actual Velocities')
    
    # Plot x velocity
    axes[0].plot(time, commanded_vel[:, 0], 'r--', label='Commanded')
    axes[0].plot(time, base_vel[:, 0], 'b-', label='Actual')
    axes[0].set_ylabel('X Velocity (m/s)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot y velocity
    axes[1].plot(time, commanded_vel[:, 1], 'r--', label='Commanded')
    axes[1].plot(time, base_vel[:, 1], 'b-', label='Actual')
    axes[1].set_ylabel('Y Velocity (m/s)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot angular velocity
    axes[2].plot(time, commanded_vel[:, 2], 'r--', label='Commanded')
    axes[2].plot(time, qvel[:, 5], 'b-', label='Actual')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angular Velocity (rad/s)')
    axes[2].legend()
    axes[2].grid(True)

    plt.savefig(os.path.join(save_dir, "velocity_comparison.png"))

def plot_position_comparison(data, save_dir):
    """Plot comparison between desired and actual positions."""
    time = data['time']
    qpos = data['qpos']
    commanded_vel = data['commanded_vel']
    
    # Extract base position (first 3 elements of qpos)
    actual_pos = qpos[:, :3]
    #need to extract the initial yaw 
    quat = qpos[:,4:8]
    yaw = 2 * np.arctan2(quat[:,2], quat[:,3])
    actual_pos[:,2] = yaw
    actual_yaw = yaw
    
    # Calculate desired position by integrating commanded velocity
    dt = time[1] - time[0]  # Assuming constant time step
    desired_pos = np.zeros_like(actual_pos)
    desired_pos[0] = actual_pos[0]  # Start from actual position
    
    for i in range(1, len(time)):
        desired_pos[i] = desired_pos[i-1] + commanded_vel[i-1] * dt
    

    # Create figure with 3 subplots for x, y, and angular positions
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Desired vs Actual Positions')
    
    # Plot x position
    axes[0].plot(time, desired_pos[:, 0], 'r--', label='Desired')
    axes[0].plot(time, actual_pos[:, 0], 'b-', label='Actual')
    axes[0].set_ylabel('X Position (m)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot y position
    axes[1].plot(time, desired_pos[:, 1], 'r--', label='Desired')
    axes[1].plot(time, actual_pos[:, 1], 'b-', label='Actual')
    axes[1].set_ylabel('Y Position (m)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot angular position
    axes[2].plot(time, desired_pos[:, 2], 'r--', label='Desired')
    axes[2].plot(time, actual_yaw, 'b-', label='Actual')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angular Position (rad)')
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

        robot = config["robot"]
        policy = config["policy"]
        policy_dt = config["policy_dt"]

        # print(data)

    print("============== Data generated using " + config["simulator"] + " ===============")

    print(f"time shape: {data['time'].shape}")
    print(f"qpos shape: {data['qpos'].shape}")
    print(f"qvel shape: {data['qvel'].shape}")
    print(f"torque shape: {data['torque'].shape}")
    print(f"action shape: {data['action'].shape}")
    print(f"left_ankle_pos shape: {data['left_ankle_pos'].shape}")
    print(f"right_ankle_pos shape: {data['right_ankle_pos'].shape}")
    print(f"commanded_vel shape: {data['commanded_vel'].shape}")

    # Make a plot
    plot_joints_and_actions(data, newest)
    # plot_torques(data)
    # plot_vels(data)
    plot_base(data, newest)
    # import pdb; pdb.set_trace()
    # plot_ankles(data)

    # Plot velocity and position comparisons
    plot_velocity_comparison(data, newest)
    plot_position_comparison(data, newest)

if __name__ == "__main__":
    create_plots_for_newest()
    plt.show()
