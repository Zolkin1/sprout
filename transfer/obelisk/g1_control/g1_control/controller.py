import os
from abc import ABC

import numpy as np
import torch
from datetime import datetime
import time
import csv
import shutil
from ament_index_python.packages import get_package_share_directory
from .policy import RLPolicy
from obelisk_control_msgs.msg import PDFeedForward, VelocityCommand
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound
from obelisk_py.core.utils.ros import spin_obelisk
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import Joy

# Changes TODO:
# - Take a list of hugging face policies
# - Also need to make sure we get the associated config files with the policies
# - Make the observations modular to be built based on config file
# - Handle holding multiple policies
# - Add a generic state machine for multiple skills that can be extended in the future
# - 


class VelocityTrackingController(ObeliskController, ABC):
    """Locomotion controller to track a given velocity."""

    def __init__(self, node_name: str = "velocity_tracking_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, PDFeedForward, EstimatedState)
        # Load policy
        self.declare_parameter("local_policy_name", "")
        # self.declare_parameter("obs_type", "")
        self.declare_parameter("hf_repo_id", "")
        self.declare_parameter("hf_policy_folder", "")
        
        # self.obs_type = self.get_parameter("obs_type").get_parameter_value().string_value
        # policy_name = self.get_parameter("local_policy_name").get_parameter_value().string_value
        hf_repo_id = self.get_parameter("hf_repo_id").get_parameter_value().string_value
        hf_policy_folder = self.get_parameter("hf_policy_folder").get_parameter_value().string_value
        
        self.policy_wrapper = RLPolicy(hf_repo_id, hf_policy_folder)

        self.get_logger().info(f"Loaded policy at {self.policy_wrapper.get_policy_path()}.")

        # Logging information
        self.declare_parameter("log", False)
        self.declare_parameter("log_decimation", 1)
        self.log = self.get_parameter("log").get_parameter_value().bool_value
        self.log_decimation = self.get_parameter("log_decimation").get_parameter_value().integer_value
        self.ctrl_count = 0

        self.start_time = self.get_clock().now().nanoseconds / 1e9

        if self.log:
            self.get_logger().info(f"Logging enabled. Decimation: {self.log_decimation}.")

            # Create log directory relative to $ROBOT_RL_ROOT
            root = os.environ.get("ROBOT_RL_ROOT", "")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(root, "ctrl_logs", timestamp)
            os.makedirs(log_dir, exist_ok=True)

            self.log_file = os.path.join(log_dir, "g1_control.csv")
            self.file = open(self.log_file, "w", buffering=8192)
            self.writer = csv.writer(self.file)
            self._lines_written = 0
            self._file_index = 0

            # Copy the config into the log directory
            self.declare_parameter("config_name", "")
            config_name = self.get_parameter("config_name").get_parameter_value().string_value
            if config_name != "":
                self.get_logger().info(f"Copying the config to the log directory ({config_name})...")
                config_path = os.path.join(root, "g1_control", "configs", config_name)
                shutil.copy2(config_path, log_dir)


            # Copy the policy into the log directory
            # self.get_logger().info("Copying the policy to the log directory...")
            # shutil.copy2(self.policy_wrapper.get_policy_path(), log_dir)

            # Make a file with the log ordering
            header_path = os.path.join(log_dir, "fields.csv")
            with open(header_path, "w") as f:
                f.write("time,observation,action\n")

        # Policy information
        self.declare_parameter("num_obs", 74)
        self.num_obs = self.get_parameter("num_obs").get_parameter_value().integer_value
        self.isaac_to_mujoco = {
            0: 0,  # left_hip_pitch
            1: 6,  # right_hip_pitch
            2: 12,  # waist_yaw
            3: 1,  # left_hip_roll
            4: 7,  # right_hip_roll
            5: 13,  # left_shoulder_pitch
            6: 17,  # right_shoulder_pitch
            7: 2,  # left_hip_yaw
            8: 8,  # right_hip_yaw
            9: 14,  # left_shoulder_roll
            10: 18,  # right_shoulder_roll
            11: 3,  # left_knee
            12: 9,  # right_knee
            13: 15,  # left_shoulder_yaw
            14: 19,  # right_shoulder_yaw
            15: 4,  # left_ankle_pitch
            16: 10,  # right_ankle_pitch
            17: 16,  # left_elbow
            18: 20,  # right_elbow
            19: 5,  # left_ankle_roll
            20: 11,  # right_ankle_roll
        }
        self.isaac_joints = {
            0: "left_hip_pitch_joint",
            1: "right_hip_pitch_joint",
            2: "waist_yaw_joint",
            3: "left_hip_roll_joint",
            4: "right_hip_roll_joint",
            5: "left_shoulder_pitch_joint",
            6: "right_shoulder_pitch_joint",
            7: "left_hip_yaw_joint",
            8: "right_hip_yaw_joint",
            9: "left_shoulder_roll_joint",
            10: "right_shoulder_roll_joint",
            11: "left_knee_joint",
            12: "right_knee_joint",
            13: "left_shoulder_yaw_joint",
            14: "right_shoulder_yaw_joint",
            15: "left_ankle_pitch_joint",
            16: "right_ankle_pitch_joint",
            17: "left_elbow_joint",
            18: "right_elbow_joint",
            19: "left_ankle_roll_joint",
            20: "right_ankle_roll_joint",
        }
        self.joint_names_mujoco = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]
        self.full_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self.num_motors = 21

        # # Set scales and defaults
        # self.declare_parameter("action_scale", 0.25)
        # self.action_scale = self.get_parameter("action_scale").get_parameter_value().double_value

        # self.declare_parameter("cmd_scale", [2.0, 2.0, 0.25])
        # self.cmd_scale = self.get_parameter("cmd_scale").get_parameter_value().double_array_value

        # self.declare_parameter("period", 0.8)
        # self.period = self.get_parameter("period").get_parameter_value().double_value

        # self.declare_parameter("num_actions", 0)
        # self.num_actions = self.get_parameter("num_actions").get_parameter_value().integer_value

        # self.declare_parameter("default_angles", [0.0])
        # self.declare_parameter("default_angles_names", [""])
        # self.default_angles = self.get_parameter("default_angles").get_parameter_value().double_array_value
        # self.default_angles_names = self.get_parameter("default_angles_names").get_parameter_value().string_array_value

        # self.default_angles_isaac = self._convert_to_isaac(self.default_angles, self.default_angles_names)

        # self.declare_parameter("qvel_scale", 1.0)
        # self.qvel_scale = self.get_parameter("qvel_scale").get_parameter_value().double_value

        # self.declare_parameter("ang_vel_scale", 1.0)
        # self.ang_vel_scale = self.get_parameter("ang_vel_scale").get_parameter_value().double_value

        # # Set PD gains
        # self.declare_parameter("kps", [25.0] * self.num_motors)
        # self.declare_parameter("kds", [0.5] * self.num_motors)
        # self.kps = self.get_parameter("kps").get_parameter_value().double_array_value
        # self.kds = self.get_parameter("kds").get_parameter_value().double_array_value

        self.kps, self.kds = self._add_wrist_kp_kd_mujoco(self.policy_wrapper.get_kp(self.joint_names_mujoco).tolist(),
                                                            self.policy_wrapper.get_kd(self.joint_names_mujoco).tolist())

        # Declare subscriber to velocity commands
        self.register_obk_subscription(
            "sub_vel_cmd_setting",
            self.vel_cmd_callback,  # type: ignore
            key="sub_vel_cmd_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand,
        )

        # Need the joystick for the E-stop
        self.register_obk_subscription(
            "sub_joy_setting",
            self.joy_callback,  # type: ignore
            key="sub_joy_key",  # key can be specified here or in the config file
            msg_type=Joy,
        )

        self.received_xhat = False


    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)

        self.cmd_vel = np.zeros((3,))
        # self.proj_g = np.zeros((3,))
        # self.proj_g[2] = -1
        # self.omega = np.zeros((3,))
        # self.phase = np.zeros((2,))
        self.zero_action = np.zeros((self.num_motors,))
        # self.action = self.zero_action.tolist()
        self.t_start = None

        self.get_logger().info("RL Velocity Tracking node configuration complete.")

        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: EstimatedState) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """
        self.joint_pos = np.array(x_hat_msg.q_joints)
        self.joint_names = x_hat_msg.joint_names

        self.joint_vel = np.array(x_hat_msg.v_joints)

        # self.proj_g = self.project_gravity(x_hat_msg.q_base[3:7])
        self.qfb = np.array(x_hat_msg.q_base[:7])
        self.vfb_ang = np.array(x_hat_msg.v_base[3:6])

        # self.omega = np.array(x_hat_msg.v_base[3:6])

        self.time = x_hat_msg.header.stamp.sec + x_hat_msg.header.stamp.nanosec * 1e-9

        self.received_xhat = True

    def vel_cmd_callback(self, cmd_msg: VelocityCommand):
        """Callback for velocity command messages."""
        self.cmd_vel[0] = cmd_msg.v_x
        self.cmd_vel[1] = cmd_msg.v_y
        self.cmd_vel[2] = cmd_msg.w_z
    
    def compute_control(self) -> PDFeedForward:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Generate input to RL model
        if self.received_xhat:
            # self.get_logger().info(f"Time: {(self.time - self.start_time):.4f}")
            obs = self.policy_wrapper.create_obs(
                self.qfb,
                self.vfb_ang,
                self.joint_pos,
                self.joint_vel,
                self.time - self.start_time,
                self.cmd_vel,
                self.joint_names,
            )

            # Call RL model
            self.action = self.policy_wrapper.get_action(obs, self.joint_names_mujoco)

            # setting the message
            pd_ff_msg = PDFeedForward()
            pd_ff_msg.header.stamp = self.get_clock().now().to_msg()
            pos_targ = self.action #self._convert_to_mujoco(self.action)lo
            pd_ff_msg.pos_target = self._add_wrist_mujoco(pos_targ.tolist())
            pd_ff_msg.vel_target = self._add_wrist_mujoco(self.zero_action.tolist())
            pd_ff_msg.feed_forward = self._add_wrist_mujoco(self.zero_action.tolist())

            # Add in the wrist joints

            pd_ff_msg.u_mujoco = np.concatenate([
                self._add_wrist_mujoco(pos_targ.tolist()),
                self._add_wrist_mujoco(self.zero_action.tolist()),
                self._add_wrist_mujoco(self.zero_action.tolist()),
            ]).tolist()
            pd_ff_msg.joint_names = self.full_joint_names

            pd_ff_msg.kp = self.kps
            pd_ff_msg.kd = self.kds

            self.obk_publishers["pub_ctrl"].publish(pd_ff_msg)

            # Log observation and action
            if self.log and self.ctrl_count % self.log_decimation == 0:
                self.log_data(obs, self.action)

            self.ctrl_count += 1

            assert is_in_bound(type(pd_ff_msg), ObeliskControlMsg)
            return pd_ff_msg
    
    # def _convert_to_mujoco(self, vec):
    #     mj_vec = np.zeros(21)
    #     for isaac_index, mujoco_index in self.isaac_to_mujoco.items():
    #         mj_vec[mujoco_index] = vec[isaac_index]

    #     return mj_vec

    # def _convert_to_isaac(self, joint_values, joint_names):
    #     ordered_values = np.zeros(len(self.isaac_joints), dtype=np.float64)
    #     name_to_index = {name: idx for idx, name in self.isaac_joints.items()}

    #     for name, value in zip(joint_names, joint_values):
    #         if name in name_to_index:
    #             index = name_to_index[name]
    #             ordered_values[index] = value
    #         else:
    #             self.get_logger().warning(f"Unknown joint name: {name}", once=True)
    #             # raise ValueError(f"Unknown joint name: {name}")

    #     return ordered_values

    def _add_wrist_mujoco(self, joint_values):
        """Add 0's to the wrist in mujoco ordering."""
        joint_values.insert(17, 0.0)
        joint_values.insert(18, 0.0)
        joint_values.insert(19, 0.0)

        joint_values.insert(24, 0.0)
        joint_values.insert(25, 0.0)
        joint_values.insert(26, 0.0)

        return joint_values

    def _add_wrist_kp_kd_mujoco(self, kp, kd):
        """Add 0's to the wrist in mujoco ordering."""
        kp.insert(17, 40.0)
        kp.insert(18, 40.0)
        kp.insert(19, 40.0)

        kp.insert(24, 40.0)
        kp.insert(25, 40.0)
        kp.insert(26, 40.0)


        kd.insert(17, 2.0)
        kd.insert(18, 2.0)
        kd.insert(19, 2.0)

        kd.insert(24, 2.0)
        kd.insert(25, 2.0)
        kd.insert(26, 2.0)

        return kp, kd

    def log_data(self, obs, action):
        log_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        obsaction = obs.tolist() + action.tolist()
        row = [log_time] + obsaction
        self.writer.writerow(row)

    def joy_callback(self, joy_msg: Joy):
        """Callback to watch for E-stop from joystick."""
        RIGHT_TRIGGER = 5
        if joy_msg.axes[RIGHT_TRIGGER] <= 0.1:
            raise RuntimeError("[Controller] Joystick emergency stop triggered!!")


def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, VelocityTrackingController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()
