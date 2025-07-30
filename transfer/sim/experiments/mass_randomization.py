from typing import Literal
import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transfer.sim.simulation import Simulation
from transfer.sim.robot import Robot
from transfer.sim.rl_policy_wrapper import RLPolicy
from transfer.sim.plot_from_sim import create_plots_for_newest
from transfer.sim.log_utils import find_most_recent_timestamped_folder, extract_data

from performance_statistics import compute_stats
from velocity_commands import speed_steps, smooth_ramp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--simulator", type=str, help="Choice of simulator to run (isaac_sim or mujoco)")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    config_file_name = os.path.splitext(os.path.basename(args.config_file))[0]

    # Parse the config file with default values
    required_fields = {
        "checkpoint_path": str,
        "dt": float,
        "num_obs": int,
        "num_action": int,
        "period": int,
        "robot_name": str,
        "action_scale": float,
        "default_angles": list,
        "qvel_scale": float,
        "ang_vel_scale": float,
        "command_scale": float,
        "policy_type": Literal["mlp", "cnn"],
    }

    # Check for required fields
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in config file: {', '.join(missing_fields)}")

    # Make the RL policy
    policy = RLPolicy(
        dt=config["dt"],
        checkpoint_path=config["checkpoint_path"],
        num_obs=config["num_obs"],
        num_action=config["num_action"],
        period=config["period"],
        cmd_scale=config["command_scale"],
        action_scale=config["action_scale"],
        default_angles=config["default_angles"],
        qvel_scale=config["qvel_scale"],
        ang_vel_scale=config["ang_vel_scale"],
        height_map_scale=config.get("height_map_scale", None),
        policy_type=config["policy_type"]
    )

    # Create robot instance
    robot_instance = Robot(robot_name=config["robot_name"], scene_name=config.get("scene", "basic_scene"),
                           input_function=smooth_ramp)

    run_logs = []

    NUM_RUNS = 10

    for i in range(NUM_RUNS):
        # Adjust the torso mass position
        max_movement = np.array([0.02,0.02,0.02])
        pos_movement = robot_instance.randomize_torso_mass_pos(max_movement)

        # Create and run simulation
        sim = Simulation(policy, robot_instance, log=True,
                         log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
                         use_height_sensor=config.get("height_map_scale") is not None, tracking_body_name="torso_link")
        sim.run(total_time=12, force_disturbance=None)

        run_logs.append(sim.get_logging_folder())

        # Make plots and statistics
        # create_plots_for_newest()
        # compute_stats(0)

        # Log the robustness constants
        robustness_data = {
            'torso_mass_pos': pos_movement.tolist(),
        }

        with open(os.path.join(sim.get_logging_folder(), 'robustness_data.yaml'), 'w') as f:
            yaml.dump(robustness_data, f)

        robot_instance.reset_robot()
        robot_instance.reset_torso_mass_pos()

    # Load back in all the data
    actual_vel_list = []
    commanded_vel_list = []
    time_list = []
    for i in range(NUM_RUNS):
        # Parse the data
        log_dir = os.path.join(os.getcwd(), run_logs[i])

        # Load in the data
        with open(os.path.join(log_dir, "sim_config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            data = extract_data(os.path.join(log_dir, "sim_log.csv"), config)

            actual_vel_list.append(data["qvel"])
            commanded_vel_list.append(data["commanded_vel"])
            time_list.append(data["time"])

    time_np = np.stack(time_list)
    commanded_vel_np = np.stack(commanded_vel_list)
    actual_vel_np = np.stack(actual_vel_list)

    mean_actual = np.mean(actual_vel_np, axis=0)
    std_actual = np.std(actual_vel_np, axis=0)

    # Make some plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Commanded vs Actual Velocities')

    axes[0].plot(time_np[0, :], commanded_vel_np[0, :, 0], 'r--', label='Commanded')
    axes[1].plot(time_np[0, :], commanded_vel_np[0, :, 1], 'r--', label='Commanded')
    axes[2].plot(time_np[0, :], commanded_vel_np[0, :, 2], 'r--', label='Commanded')

    # Plot x velocity
    axes[0].plot(time_np[0], mean_actual[:, 0], 'b-', label=f'Actual_{i}')
    axes[0].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 0] - std_actual[:, 0], mean_actual[:, 0] + std_actual[:, 0], color='blue', alpha=0.3, label='±1 Std Dev')
    axes[0].set_ylabel('X Velocity (m/s)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot y velocity
    axes[1].plot(time_np[0, :], mean_actual[:, 1], 'b-', label=f'Actual_{i}')
    axes[1].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 1] - std_actual[:, 1], mean_actual[:, 1] + std_actual[:, 1], color='blue', alpha=0.3, label='±1 Std Dev')
    axes[1].set_ylabel('Y Velocity (m/s)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot angular velocity
    axes[2].plot(time_np[0, :], mean_actual[:, 2], 'b-', label=f'Actual_{i}')
    axes[2].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 2] - std_actual[:, 2], mean_actual[:, 2] + std_actual[:, 2], color='blue', alpha=0.3, label='±1 Std Dev')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angular Velocity (rad/s)')
    axes[2].legend()
    axes[2].grid(True)

    plt.savefig(f"experiments/plots/mass_randomization_{config_file_name}.png")

if __name__ == "__main__":
    main()
    plt.show()




