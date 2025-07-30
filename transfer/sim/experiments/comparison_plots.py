import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sim.log_utils import find_most_recent_timestamped_folder, extract_data

def main():
    """Take in multiple pre-run sims and create a single plot that compares it."""
    # Loop through every log in the "comparison_logs" folder and use those
    comp_logs_dir = Path("experiments/comparison_logs")

    policy_names = []
    if comp_logs_dir.exists() and comp_logs_dir.is_dir():
        for subfolder in comp_logs_dir.iterdir():
            if subfolder.is_dir():
                config_path = subfolder / "sim_config.yaml"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        try:
                            config = yaml.safe_load(f)
                            if "policy" in config:
                                policy_path = config.get("policy", None)
                                if policy_path is not None:
                                    policy_file = Path(policy_path).stem
                                    policy_names.append(policy_file)
                        except yaml.YAMLError as e:
                            print(f"YAML error in {config_path}: {e}")

        logs = [f for f in comp_logs_dir.iterdir() if f.is_dir()]
        print("Using comparison folders:", logs)
        print("Policy names:", policy_names)
    else:
        raise FileNotFoundError("Comparison logs folder not found!")

    run_names = policy_names
    # run_names = ["hzd_clf_minimum_reward", "hzd_dec_4_alpha_1", "hzd_dec_2_alpha_2", "hzd_dec_2_alpha_0.5", "hzd_dec_0"]
    run_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # For shared legend
    handles = []
    labels = []

    for i, log in enumerate(logs):
        # log_dir = os.path.join(os.getcwd(), "logs", log)
        log_dir = log

        # Load in the data
        with open(os.path.join(log_dir, "sim_config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            data = extract_data(os.path.join(log_dir, "sim_log.csv"), config)

        time = data['time']
        actual_vel = data['qvel']
        commanded_vel = data['commanded_vel']

        # Plot commanded velocity (only once)
        if i == 0:
            h_cmd_x, = axes[0].plot(time, commanded_vel[:, 0], 'k--', label='Commanded')
            h_cmd_y, = axes[1].plot(time, commanded_vel[:, 1], 'k--', label='Commanded')
            h_cmd_w, = axes[2].plot(time, commanded_vel[:, 2], 'k--', label='Commanded')
            handles.append(h_cmd_x)
            labels.append('Commanded')

        # Choose color
        color = run_colors[i]

        # Plot actual velocities
        h_x, = axes[0].plot(time, actual_vel[:, 0], color=color, label=run_names[i])
        h_y, = axes[1].plot(time, actual_vel[:, 1], color=color, label=run_names[i])
        h_w, = axes[2].plot(time, actual_vel[:, 2], color=color, label=run_names[i])

        if run_names[i] not in labels:
            handles.append(h_x)
            labels.append(run_names[i])

    # Set axis labels
    axes[0].set_ylabel(r'$v_x$ (m/s)')
    axes[1].set_ylabel(r'$v_y$ (m/s)')
    axes[2].set_ylabel(r'$\omega_z$ (rad/s)')

    axes[2].set_xlabel('Time (s)')

    for ax in axes:
        ax.grid(True)

    # Shared legend right above the plots, but still inside figure
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize='medium', bbox_to_anchor=(0.5, 0.995))

    # Pull plots up tight — no suptitle, no extra padding
    fig.subplots_adjust(top=0.94)




        # TODO: Fix
        # ## Positions
        # qpos = data['qpos']
        # actual_pos = qpos[:, :3]
        #
        # # Calculate desired position by integrating commanded velocity
        # dt = time[1] - time[0]  # Assuming constant time step
        # desired_pos = np.zeros_like(actual_pos)
        # desired_pos[0] = actual_pos[0]  # Start from actual position
        #
        # for j in range(1, len(time)):
        #     desired_pos[j] = desired_pos[j - 1] + commanded_vel[j - 1] * dt
        #
        # if i == 0:
        #     axes_p[0].plot(time, desired_pos[:, 0], 'k--', label='Commanded', linewidth=3)
        #     axes_p[1].plot(time, desired_pos[:, 1], 'k--', label='Commanded', linewidth=3)
        #
        # # Plot x position
        # print(actual_pos)
        # print(time)
        # axes_p[0].plot(time, actual_pos[:, 0], color, label=run_names[i], linewidth=2)
        # axes_p[0].set_ylabel('X Position (m)')
        # axes_p[0].legend()
        # axes_p[0].grid(True)
        #
        # # Plot y position
        # axes_p[1].plot(time, actual_pos[:, 1], color, label=run_names[i], linewidth=2)
        # axes_p[1].set_ylabel('Y Position (m)')
        # axes_p[1].legend()
        # axes_p[1].grid(True)



    # Save the plots
    os.makedirs("experiments/plots", exist_ok=True)
    plt.savefig("experiments/plots/velocity_comparison_plot.svg", bbox_inches='tight', transparent=True)



if __name__ == "__main__":
    main()
    plt.show()
