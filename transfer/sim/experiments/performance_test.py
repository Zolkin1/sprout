from typing import Literal
import argparse
import yaml
import os
import sys
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transfer.sim.mj_simulation import run_simulation
from transfer.sim.rl_policy_wrapper import RLPolicy
from transfer.sim.plot_from_sim import create_plots_for_newest

from performance_statistics import compute_stats

def step_to_max(sim_time):
    """Pass a given input as a function of time."""
    if sim_time > 1.:
        return np.array([0.75,0.0,0.0])
    else:
        return np.array([0.0,0.0,0.0])

def smooth_ramp(sim_time):
    """Compute a ramp input over a few seconds up to max speed."""
    RAMP_TIME = 2.0

    MAX_SPEED = 0.75

    slope = MAX_SPEED / RAMP_TIME

    return np.array([min(slope * sim_time, MAX_SPEED),0.0,0.0])

def speed_steps(sim_time):
    time_steps = np.array([3, 3, 3, 3])
    speeds = np.array([0.25, -0.25, 0.5, 0.75])

    # Compute start times of each interval
    start_times = np.cumsum(np.insert(time_steps[:-1], 0, 0))

    def _get_velocity_at_time(t):
        # Ensure t is within bounds
        if t < 0 or t >= np.sum(time_steps):
            raise ValueError("Time t is out of bounds.")

        # Find the index of the bin t belongs to
        idx = np.searchsorted(start_times, t, side='right') - 1
        return speeds[idx]

    return np.array([_get_velocity_at_time(sim_time), 0, 0])


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

    # Run the simulator with default values for optional fields
    run_simulation(
        policy=policy,
        robot=config["robot_name"],
        scene=config.get("scene", "basic_scene"),  # Default to basic_scene if not specified
        log=config.get("log", False),
        log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
        input_function=speed_steps, #step_to_max,
        total_time=12,
        use_height_sensor=config.get("height_map_scale") is not None,  # Enable height sensor if height_map_scale is present
        tracking_body_name="torso_link"
    )

    create_plots_for_newest()
    compute_stats(0)

if __name__ == "__main__":
    main()




