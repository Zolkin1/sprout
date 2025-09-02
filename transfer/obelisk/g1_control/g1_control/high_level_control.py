import os
from abc import ABC

import numpy as np
import torch
from datetime import datetime
import time
import csv
import shutil
from ament_index_python.packages import get_package_share_directory
from obelisk_control_msgs.msg import PDFeedForward, VelocityCommand
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound
from obelisk_py.core.utils.ros import spin_obelisk
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import Joy

class HighLevelController(ObeliskController, ABC):
    """High level controller. This is responsible for converting input and some sensors into commanded velocities."""

    def __init__(self, node_name: str = "high_level_controller"):
        super().__init__(node_name, VelocityCommand, Joy)
        # Velocity limits
        self.declare_parameter("v_x_max", 1.0)
        self.declare_parameter("v_x_min", -1.0)
        self.declare_parameter("v_y_max", 0.5)
        self.declare_parameter("w_z_max", 0.5)

        self.v_x_max = self.get_parameter("v_x_max").get_parameter_value().double_value
        self.v_x_min = self.get_parameter("v_x_min").get_parameter_value().double_value
        self.v_y_max = self.get_parameter("v_y_max").get_parameter_value().double_value
        self.w_z_max = self.get_parameter("w_z_max").get_parameter_value().double_value

        # Joystick and state machine
        self.last_menu_press = self.get_clock().now().nanoseconds / 1e9
        self.last_A_press = self.get_clock().now().nanoseconds / 1e9
        self.last_B_press = self.get_clock().now().nanoseconds / 1e9
        self.last_RB_press = self.get_clock().now().nanoseconds / 1e9
        self.last_LB_press = self.get_clock().now().nanoseconds / 1e9
        self.last_X_press = self.get_clock().now().nanoseconds / 1e9
        self.last_Y_press = self.get_clock().now().nanoseconds / 1e9

        self.lin_vel_mode = "joystick"  # joystick, incremental, function
        self.ang_vel_mode = "joystick"  # joystick, odom

        # Incremental velocity parameters
        self.declare_parameter("vel_increment", 0.1)
        self.vel_increment = self.get_parameter("vel_increment").get_parameter_value().double_value

        # Straight line walking parameters
        self.declare_parameter("use_odom", False)
        self.use_odom = self.get_parameter("use_odom").get_parameter_value().bool_value
        if self.use_odom:
            self.declare_parameter("kp", 1.0)
            self.declare_parameter("kd", 0.5)
            self.kp = self.get_parameter("kp").get_parameter_value().double_value
            self.kd = self.get_parameter("kd").get_parameter_value().double_value

            self.yaw_target = 0.0
            self.yaw_rate_cmd = 0.0
            self.y_pos_target = 0.0
            self.y_vel_target = 0.0

            self.yaw_cur = 0.0
            self.y_pos_cur = 0.0

            # Declare subscriber to odometry
            self.register_obk_subscription(
                "sub_odom_setting",
                self.odom_callback,  # type: ignore
                key="sub_odom_key",  # key can be specified here or in the config file
                msg_type=Odometry,
            )

        # Declare subscriber to velocity commands from the Untiree joystick node
        self.register_obk_subscription(
            "sub_vel_cmd_setting",
            self.vel_cmd_callback,  # type: ignore
            key="sub_vel_cmd_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand,
        )

        self.cmd_vel = np.zeros((3,))

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry messages."""
        if self.ang_vel_mode == "odom":
            # P yaw controller:
            # # Get the yaw from the quaternion
            # q = msg.pose.pose.orientation
            # siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            # cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            # yaw = np.arctan2(siny_cosp, cosy_cosp)

            # # Update the yaw target using a PD controller
            # yaw_error = yaw - self.yaw_target
            # if yaw_error > np.pi:
            #     yaw_error -= 2 * np.pi
            # elif yaw_error < -np.pi:
            #     yaw_error += 2 * np.pi
            # yaw_rate_cmd = -self.kp * yaw_error

            # # Clamp the yaw rate command
            # self.yaw_rate_cmd = np.clip(yaw_rate_cmd, -self.w_z_max, self.w_z_max)

            # PD On y pos into yaw rate:
            y_pos = msg.pose.pose.position.y
            y_vel = msg.twist.twist.linear.y

            angular_vel = np.sign(self.cmd_vel[0]) * (-self.kp * (y_pos - self.y_pos_target) + -self.kd * (y_vel - self.y_vel_target))
            self.yaw_rate_cmd = np.clip(angular_vel, -self.w_z_max, self.w_z_max)

    def vel_cmd_callback(self, cmd_msg: VelocityCommand):
        """Callback for velocity command messages from the unitree joystick node."""
        if self.lin_vel_mode == "joystick":
            self.cmd_vel[0] = min(
                max(cmd_msg.v_x, self.v_x_min), self.v_x_max)
            self.cmd_vel[1] = min(max(cmd_msg.v_y, -self.v_y_max), self.v_y_max)
            self.cmd_vel[2] = min(max(cmd_msg.w_z, -self.w_z_max), self.w_z_max)
        elif self.lin_vel_mode == "function":
            RAMP_TIME = 2.0
            slope = self.v_x_max / RAMP_TIME

            now = self.get_clock().now().nanoseconds / 1e9
            self.cmd_vel[0] = min(slope * (now - self.joystick_exited), self.v_x_max)
            self.cmd_vel[1] = 0
            self.cmd_vel[2] = 0

            if now - self.joystick_exited > 8:
                self.lin_vel_mode = "joystick"
                self.get_logger().info("Joystick control re-enabled after timeout.")
        elif self.lin_vel_mode == "incremental":
            pass

    def update_x_hat(self, msg):
        """Receive the joystick message."""
        RIGHT_TRIGGER = 5
        if msg.axes[RIGHT_TRIGGER] <= 0.1:
            raise RuntimeError("[High Level] Joystick emergency stop triggered!!")

        MENU = 7
        now = self.get_clock().now().nanoseconds / 1e9
        if msg.buttons[MENU] >= 0.9 and now - self.last_menu_press > 0.5:
            self.last_menu_press = now
            self.get_logger().info("Button mappings: \n " \
            " E-STOP: Right Trigger. \n " \
            " Forward/Backward: Left Stick. \n " \
            " Turning: Right Stick. \n" \
            " Damping: Right D-Pad. \n" \
            " Low Level Ctrl: Bottom D-Pad. \n" \
            " User Pose: Squares. \n" \
            " Joystick Mode: A. \n" \
            " Incremental Velocity Mode: B. \n" \
            " Function Velocity Mode: X. \n" \
            " Increase Incremental Velocity: Right Bumper. \n" \
            " Decrease Incremental Velocity: Left Bumper. \n" \
            " Toggle Odom Correction: Y. \n" \
            " Zero odom targets: Right Bumper (while not in incremental mode).")

        A = 0
        if msg.buttons[A] >= 0.9 and now - self.last_A_press > 0.5:
            self.last_A_press = now
            self.lin_vel_mode = "joystick"
            self.joystick_exited = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info("Joystick control enabled!")

        B = 1
        if msg.buttons[B] >= 0.9 and now - self.last_B_press > 0.5:
            self.last_B_press = now
            self.lin_vel_mode = "incremental"
            self.cmd_vel = np.zeros((3,))
            self.cmd_vel[0] = 1.1
            self.get_logger().info("Joystick incremental velocity mode enabled!")

        X = 2
        if msg.buttons[X] >= 0.9 and now - self.last_X_press > 0.5:
            self.last_X_press = now
            self.lin_vel_mode = "function"
            self.get_logger().info("Function velocity mode enabled!")

        Y = 3
        if msg.buttons[Y] >= 0.9 and now - self.last_Y_press > 0.5:
            if self.use_odom:
                self.last_Y_press = now
                self.ang_vel_mode = "joystick" if self.ang_vel_mode == "odom" else "odom"
                if self.ang_vel_mode == "odom":
                    self.get_logger().info("Odom correction enabled!")
                else:
                    self.get_logger().info("Odom correction disabled!")

        RIGHT_BUMPER = 5
        if msg.buttons[RIGHT_BUMPER] >= 0.9 and now - self.last_RB_press > 0.2:
            if self.lin_vel_mode == "incremental":
                self.cmd_vel[0] += self.vel_increment
                vx_max = self.get_parameter("v_x_max").get_parameter_value().double_value
                if self.cmd_vel[0] > vx_max:
                    self.cmd_vel[0] = vx_max

                self.get_logger().info(f"----- INCREASING VELOCITY TO {self.cmd_vel[0]:.3f} m/s -----")
            elif self.use_odom:
                self.yaw_target = self.yaw_cur
                self.y_pos_target = self.y_pos_cur
                self.get_logger().info(f"Odom targets zeroed at: Y position: {self.y_pos_target}, yaw target: {self.yaw_target}!")
            self.last_RB_press = now


        LEFT_BUMPER = 4
        if self.lin_vel_mode == "incremental" and msg.buttons[LEFT_BUMPER] >= 0.9 and now - self.last_LB_press > 0.2:
            self.cmd_vel[0] -= self.vel_increment
            vx_min = self.get_parameter("v_x_min").get_parameter_value().double_value
            if self.cmd_vel[0] < vx_min:
                self.cmd_vel[0] = vx_min

            self.get_logger().info(f"----- DECREASING VELOCITY TO {self.cmd_vel[0]:.3f} m/s -----")
            self.last_LB_press = now

    def compute_control(self):
        """Return the commanded velocity."""
        msg = VelocityCommand()
        msg.v_x = float(self.cmd_vel[0])
        msg.v_y = float(self.cmd_vel[1])

        if self.ang_vel_mode == "joystick":
            msg.w_z = float(self.cmd_vel[2])
        elif self.use_odom and self.ang_vel_mode == "odom":
            msg.w_z = float(self.yaw_rate_cmd)

        self.obk_publishers["pub_ctrl"].publish(msg)

        return msg
    
def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, HighLevelController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()