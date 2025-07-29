import os
import yaml
import matplotlib.pyplot as plt

from transfer.sim.log_utils import find_most_recent_timestamped_folder, extract_data

def main():
    """Take in multiple pre-run sims and create a single plot that compares it."""
    logs = ["2025-07-29-10-34-07", "2025-07-29-10-50-18", "2025-07-29-10-40-20"]
    run_names = ["Basline", "HZD RL", "LIP RL"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Commanded vs Actual Velocities')

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
                axes[0].plot(time, commanded_vel[:, 0], 'r--', label='Commanded')
                axes[1].plot(time, commanded_vel[:, 1], 'r--', label='Commanded')
                axes[2].plot(time, commanded_vel[:, 2], 'r--', label='Commanded')

            if i == 0:
                color = 'blue'
            elif i == 1:
                color = 'magenta'
            else:
                color = 'green'

            # Plot x velocity
            axes[0].plot(time, actual_vel[:, 0], color, label=run_names[i])
            axes[0].set_ylabel('X Velocity (m/s)')
            axes[0].legend()
            axes[0].grid(True)

            # Plot y velocity
            axes[1].plot(time, actual_vel[:, 1], color, label=run_names[i])
            axes[1].set_ylabel('Y Velocity (m/s)')
            axes[1].legend()
            axes[1].grid(True)

            # Plot angular velocity
            axes[2].plot(time, actual_vel[:, 2], color, label=run_names[i])
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Angular Velocity (rad/s)')
            axes[2].legend()
            axes[2].grid(True)


    # Save the plots
    plt.savefig("experiments/plots/velocity_comparison_plot.png")

if __name__ == "__main__":
    main()
    plt.show()