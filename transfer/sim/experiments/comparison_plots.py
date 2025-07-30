import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

from transfer.sim.log_utils import find_most_recent_timestamped_folder, extract_data

def main():
    """Take in multiple pre-run sims and create a single plot that compares it."""
    logs = ["2025-07-29-13-09-54", "2025-07-29-13-10-50", "2025-07-29-13-13-41"]
    run_names = ["Baseline (6kg mass)", "HZD RL (6kg mass)", "LIP RL (6kg mass)"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Commanded vs Actual Velocities')

    fig_p, axes_p = plt.subplots(2, 1, figsize=(10, 12))
    fig_p.suptitle('Commanded vs Actual Positions')

    for i in range(len(logs)):
        log = logs[i]
        log_dir = os.path.join(os.getcwd() + "/logs", log)
        # Load in the data
        with open(os.path.join(log_dir, "sim_config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            data = extract_data(os.path.join(log_dir, "sim_log.csv"), config)

            time = data['time']
            actual_vel = data['qvel']
            commanded_vel = data['commanded_vel']

            if i == 0:
                axes[0].plot(time, commanded_vel[:, 0], 'k--', label='Commanded', linewidth=3)
                axes[1].plot(time, commanded_vel[:, 1], 'k--', label='Commanded', linewidth=3)
                axes[2].plot(time, commanded_vel[:, 2], 'k--', label='Commanded', linewidth=3)

            if i == 0:
                color = 'blue'
            elif i == 1:
                color = 'orange'
            else:
                color = 'green'

            # Plot x velocity
            axes[0].plot(time, actual_vel[:, 0], color, label=run_names[i], linewidth=2)
            axes[0].set_ylabel('X Velocity (m/s)')
            axes[0].legend()
            axes[0].grid(True)

            # Plot y velocity
            axes[1].plot(time, actual_vel[:, 1], color, label=run_names[i], linewidth=2)
            axes[1].set_ylabel('Y Velocity (m/s)')
            axes[1].legend()
            axes[1].grid(True)

            # Plot angular velocity
            axes[2].plot(time, actual_vel[:, 5], color, label=run_names[i], linewidth=2)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Angular Velocity (rad/s)')
            axes[2].legend()
            axes[2].grid(True)


            ## Positions
            qpos = data['qpos']
            actual_pos = qpos[:, :3]

            # Calculate desired position by integrating commanded velocity
            dt = time[1] - time[0]  # Assuming constant time step
            desired_pos = np.zeros_like(actual_pos)
            desired_pos[0] = actual_pos[0]  # Start from actual position

            for j in range(1, len(time)):
                desired_pos[j] = desired_pos[j - 1] + commanded_vel[j - 1] * dt

            if i == 0:
                axes_p[0].plot(time, desired_pos[:, 0], 'k--', label='Commanded', linewidth=3)
                axes_p[1].plot(time, desired_pos[:, 1], 'k--', label='Commanded', linewidth=3)

            # Plot x position
            print(actual_pos)
            print(time)
            axes_p[0].plot(time, actual_pos[:, 0], color, label=run_names[i], linewidth=2)
            axes_p[0].set_ylabel('X Position (m)')
            axes_p[0].legend()
            axes_p[0].grid(True)

            # Plot y position
            axes_p[1].plot(time, actual_pos[:, 1], color, label=run_names[i], linewidth=2)
            axes_p[1].set_ylabel('Y Position (m)')
            axes_p[1].legend()
            axes_p[1].grid(True)



    # Save the plots
    plt.savefig("experiments/plots/velocity_comparison_plot.png")

if __name__ == "__main__":
    main()
    plt.show()