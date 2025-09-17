import argparse
import os
import sys
from typing import Literal

import yaml

# Add the project root to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_policy_wrapper import RLPolicy

from robot import Robot
from simulation import Simulation

from experiments.velocity_commands import speed_steps, smooth_ramp_running

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--simulator", type=str, help="Choice of simulator to run (isaac_sim or mujoco)")
    args = parser.parse_args()

    with open(args.config_file) as f:
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
        height_map_scale=config.get("height_map_scale"),
        policy_type=config["policy_type"],
    )

    # Create robot instance
    robot_instance = Robot(
        robot_name=config["robot_name"], scene_name=config.get("scene", "basic_scene"), input_function=smooth_ramp_running, use_pd=config["use_pd"]
    )

    # Create and run simulation
    sim = Simulation(
        policy,
        robot_instance,
        log=config.get("log", False),
        log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
        use_height_sensor=config.get("height_map_scale") is not None,
        tracking_body_name="torso_link",
    )
    sim.run(-1)  # Run forever


if __name__ == "__main__":
    main()
