import os
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np

from sim.log_utils import extract_data

PLOT_MEANS = False

def get_index(time_vec, time: float):
    """Gets the index associated with a given time."""
    closest_idx = np.argmin(np.abs(time_vec - time))

    return closest_idx

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

    run_names = ["HZD tracking","HZD-CLF"]


    # run_names = ["hzd_clf_minimum_reward", "hzd_dec_4_alpha_1", "hzd_dec_2_alpha_2", "hzd_dec_2_alpha_0.5", "hzd_dec_0"]
    run_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan']

    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for all text
        "font.family": "serif",  # Use serif font (default LaTeX style)
        "font.serif": ["Computer Modern Roman"],  # LaTeX default font
    })
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

    # For shared legend
    handles = []
    labels = []

    num_runs = len(logs)

    for i, log in enumerate(logs):
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
            axes.plot(time, commanded_vel[:, 0], 'k', linewidth="2", label='$v_x^d$')
            # h_cmd_y, = axes[1].plot(time, commanded_vel[:, 1], 'k', linewidth="2", label='Commanded')
            # h_cmd_w, = axes[2].plot(time, commanded_vel[:, 2], 'k', linewidth="2", label='Commanded')
            # handles.append(h_cmd_x)
            # labels.append('Commanded')

        # Choose color
        color = run_colors[i]

        # Plot actual velocities
        alpha = 1
        linewidth = 2
        if PLOT_MEANS:
            alpha = 0.25
            linewidth = 3
        axes.plot(time, actual_vel[:, 0], linewidth=linewidth, color=color, label=run_names[i],zorder=num_runs-i,
                            alpha=alpha)
    
        # h_y, = axes[1].plot(time, actual_vel[:, 1], linewidth="3", color=color, label=run_names[i])
        # h_w, = axes[2].plot(time, actual_vel[:, 2], linewidth="3", color=color, label=run_names[i])

        if PLOT_MEANS:
            ## Mean plots
            # Get the steady state mean
            ss_idx_start = get_index(time, 13)
            ss_idx_end = get_index(time, 17) #time.size
            ss_x_mean = np.mean(actual_vel[ss_idx_start:ss_idx_end, 0])
            axes.plot(time[ss_idx_start:ss_idx_end], np.full(ss_idx_end - ss_idx_start, ss_x_mean), linewidth="3",
                         color=color, linestyle="--", label=f"{run_names[i]}_mean")
            
            ss_idx_start = get_index(time, 7)
            ss_idx_end = get_index(time, 11) #time.size
            ss_x_mean = np.mean(actual_vel[ss_idx_start:ss_idx_end, 0])
            axes.plot(time[ss_idx_start:ss_idx_end], np.full(ss_idx_end - ss_idx_start, ss_x_mean), linewidth="3",
                         color=color, linestyle="--", label=f"{run_names[i]}_mean")
            
            ss_idx_start = get_index(time, 20)
            ss_idx_end = get_index(time, 25) #time.size
            ss_x_mean = np.mean(actual_vel[ss_idx_start:ss_idx_end, 0])
            axes.plot(time[ss_idx_start:ss_idx_end], np.full(ss_idx_end - ss_idx_start, ss_x_mean), linewidth="3",
                         color=color, linestyle="--", label=f"{run_names[i]}_mean")


    # Set axis labels
    axes.set_ylabel(r'$v_x$ (m/s)',fontsize=20)
    # axes[1].set_ylabel(r'$v_y$ (m/s)')
    # axes[2].set_ylabel(r'$\omega_z$ (rad/s)')

    axes.set_xlabel('Time (s)',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # axes.legend()
    axes.grid()

    # Construct full unified legend with line styles + policy names
    run_handles = [
        Line2D([0], [0], color=run_colors[i], lw=2, linestyle='-', label=run_names[i])
        for i in range(len(logs))
    ]

    style_handles = []

    # Only add the "mean" style explanation if enabled
    if PLOT_MEANS:
        style_handles.append(Line2D([0], [0], color='black', linestyle='--', lw=2, label='moving average'))

    # Add reference line entry
    style_handles.append(Line2D([0], [0], color='black', linestyle='-', lw=2, label=r'$v_x^d$'))

    # Combine into one legend
    legend_handles = run_handles + style_handles

    axes.legend(handles=legend_handles,  loc="lower center",
     bbox_to_anchor=(0.5, 0.96),
     ncol=len(legend_handles),framealpha=0.0,
     fontsize=20,
     columnspacing=0.8)  # Puts all legend entries in one row fontsize = 20,
                # ncol=1, frameon=True)

    # if PLOT_MEANS:
    #     # Add manual style key as annotation or secondary legend
    #     style_handles = [
    #         Line2D([0], [0], color='black', linestyle='-', lw=2, label='actual'),
    #         Line2D([0], [0], color='black', linestyle='--', lw=2, label='mean'),
    #     ]
    #     style_legend = axes.legend(handles=style_handles, loc="lower right")
    #     axes.add_artist(style_legend)

    # # Legend for color/run names
    # run_legend = [Line2D([0], [0], color=run_colors[i], lw=2, label=run_names[i])
    #               for i in range(len(logs))]
    # run_legend.append(Line2D([0], [0], color='gray', linestyle='-', lw=2, label='$v_x^d$'))
    # axes.legend(handles=run_legend, loc="lower left")

    # Pull plots up tight — no suptitle, no extra padding
    # fig.subplots_adjust(top=0.94)




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
    plt.savefig("experiments/plots/velocity_comparison_plot.png", bbox_inches='tight', transparent=True)
    plt.savefig("experiments/plots/velocity_comparison_plot.pdf", bbox_inches='tight', transparent=True)



if __name__ == "__main__":
    main()
    plt.show()
