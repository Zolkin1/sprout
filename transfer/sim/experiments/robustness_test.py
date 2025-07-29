from typing import Literal
import argparse
import yaml
import os
import sys
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transfer.sim.simulation import Simulation
from transfer.sim.robot import Robot
from transfer.sim.rl_policy_wrapper import RLPolicy
from transfer.sim.plot_from_sim import create_plots_for_newest

from performance_statistics import compute_stats
from velocity_commands import speed_steps, smooth_ramp

FORCE_START = 3.0
FORCE_STOP = 3.125
FORCE_VEC = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ADDED_MASS = 0.

def force_robustness(sim_time):
    """Provide a force to be applied to the base based on the time."""

    if sim_time > 3 and sim_time < 3.125:   #1/8th second force
        return np.array(FORCE_VEC)
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--simulator", type=str, help="Choice of simulator to run (isaac_sim or mujoco)")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

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

    # Added mass robustness
    robot_instance.add_base_mass(ADDED_MASS)

    # Create and run simulation
    sim = Simulation(policy, robot_instance, log=config.get("log", False),
                     log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
                     use_height_sensor=config.get("height_map_scale") is not None, tracking_body_name="torso_link")
    sim.run(total_time=12, force_disturbance=force_robustness)

    # Make plots and statistics
    create_plots_for_newest()
    compute_stats(0)

    # Log the robustness constants
    robustness_data = {
        'force_start': FORCE_START,
        'force_stop': FORCE_STOP,
        'force_vec': FORCE_VEC,
        'added_mass': ADDED_MASS,
    }

    with open(os.path.join(sim.get_logging_folder(), 'robustness_data.yaml'), 'w') as f:
        yaml.dump(robustness_data, f)


if __name__ == "__main__":
    main()




