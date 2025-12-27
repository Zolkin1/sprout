import argparse
import os
import sys
from typing import Literal

import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_policy_wrapper import RLPolicy

from transfer.sim.robot import Robot
# from transfer.sim.simulation import Simulation
from transfer.sim.simulation_recording import Simulation

import numpy as np
def my_force_disturbance(time: float) -> np.ndarray:
    """
    Apply a force disturbance based on simulation time.
    
    Args:
        time: Current simulation time in seconds
    
    Returns:
        6D wrench [fx, fy, fz, tx, ty, tz] to apply to the robot
    """
    # Example 1: Constant force in x-direction
    # return np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Example 2: Impulse force at specific time
    if 3.0 < time < 3.2:  # Apply force between 3.0 and 3.2 seconds
        return np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Example 3: Sinusoidal disturbance
    # force_magnitude = 50.0
    # frequency = 1.0  # Hz
    # fx = force_magnitude * np.sin(2 * np.pi * frequency * time)
    # return np.array([fx, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Example 4: Random disturbance at intervals
    # if int(time) % 3 == 0 and time % 1.0 < 0.01:  # Every 3 seconds
    #     return np.random.randn(6) * 20.0
    # else:
    #     return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])



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
        robot_name=config["robot_name"], scene_name=config.get("scene", "basic_scene"), input_function=None
    )
    print(config["log"])

    # # Create and run simulation
    # sim = Simulation(
    #     policy,
    #     robot_instance,
    #     log=config.get("log", False),
    #     log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
    #     use_height_sensor=config.get("height_map_scale") is not None,
    #     tracking_body_name="torso_link",
    # )
    
    
    sim = Simulation(
        policy,
        robot_instance,
        # log=config.get("log", False),
        # log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
        use_height_sensor=config.get("height_map_scale") is not None,
        tracking_body_name="torso_link",
        record_video=True,
        video_width=1080,
        video_height=1080,
        camera_config={
                'type': 'tracking',
                # 'lookat': [2.0, 0, 1],      # Point camera looks at [x, y, z]
                'body_name': 'torso_link', 
                'distance': 3.0,           # Distance from lookat point
                'azimuth': 270,             # Horizontal rotation (degrees)
                'elevation': 10          # Vertical angle (degrees)
            }
    )

    # sim.run(10)  # Run for 5 seconds
    
    
#     sim = Simulation(
#        policy,
#        robot_instance,
#        # log=config.get("log", False),
#        # log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
#        use_height_sensor=config.get("height_map_scale") is not None,
#        record_video=True,
#        video_width=5120,
#        video_height=1440,
#        camera_config={
#         'type': 'fixed',
#         'lookat': [10, 0, 1.0],     # Center of 20m terrain (x=10 if starts at 0)
#         'distance': 7.0,             # Very far
#         'azimuth': 90,
#         'elevation': 0              # Slightly looking down
#        },
#    )

#     sim = Simulation(
#        policy,
#        robot_instance,
#        # log=config.get("log", False),
#        # log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
#        use_height_sensor=config.get("height_map_scale") is not None,
#        record_video=True,
#        video_width=1920,
#        video_height=1080,
#        camera_config={
#                'type': 'fixed',
#                'lookat': [1.5, 0, 1],      # Point camera looks at [x, y, z]
#                'distance': 3.2,           # Distance from lookat point
#                'azimuth': 270,             # Horizontal rotation (degrees)
#                'elevation': -30           # Vertical angle (degrees)
#            }
#    )



    # sim.run(25)  # Run for 5 seconds
    
    sim.run(
        total_time=5.0,  # Run for 10 seconds
        force_disturbance=my_force_disturbance  # Pass the disturbance function
    )



if __name__ == "__main__":
    main()
