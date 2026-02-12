import os
from abc import ABC
from typing import Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from datetime import datetime
import time
import csv
import shutil
from collections import deque
from ament_index_python.packages import get_package_share_directory
from obelisk_control_msgs.msg import PDFeedForward, VelocityCommand
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound
from obelisk_sensor_msgs.msg import ObkJointEncoders
from obelisk_py.core.utils.ros import spin_obelisk
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
import rclpy.duration
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TransformStamped

@dataclass
class OdomLog:
    pos_w: np.ndarray = np.zeros(3)
    quat_w: np.ndarray = np.zeros(4)
    lin_vel_w: np.ndarray = np.zeros(3)
    ang_vel_w: np.ndarray = np.zeros(3)
    yaw_w: float = 0.0
    world_yaw: float = 0.0
    world_origin: np.ndarray = np.zeros(3)
    y_error: float = 0.0
    yaw_error: float = 0.0
    y_target_w: float = 0.0
    yaw_target_w: float = 0.0
    target_x_vel: float = 0.0
    time: float = 0.0
    yaw_cmd: float = 0.0
    y_cmd: float = 0.0
    y_local_error: float = 0.0
    target_pos_w: np.ndarray = np.zeros(2)


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
        self.last_DPAD_UP_press = self.get_clock().now().nanoseconds / 1e9
        self.last_DPAD_DOWN_press = self.get_clock().now().nanoseconds / 1e9
        self.last_DPAD_RIGHT_press = self.get_clock().now().nanoseconds / 1e9
        self.last_DPAD_LEFT_press = self.get_clock().now().nanoseconds / 1e9

        self.control_mode = "joystick"  # joystick, feedback

        self.rec_joystick = False

        self.joy_cmd_vel = np.zeros(3)

        # Incremental velocity parameters
        self.declare_parameter("vel_increment", 0.1)
        self.vel_increment = self.get_parameter("vel_increment").get_parameter_value().double_value
        self.declare_parameter("vel_increment_start", 0.0)
        self.vel_increment_start = self.get_parameter("vel_increment_start").get_parameter_value().double_value

        # Straight line walking parameters
        self.declare_parameter("use_odom", False)
        self.use_odom = self.get_parameter("use_odom").get_parameter_value().bool_value
        if self.use_odom:
            self.declare_parameter("kp_yaw", 1.0)
            self.declare_parameter("kd_yaw", 0.5)
            self.kp_yaw = self.get_parameter("kp_yaw").get_parameter_value().double_value
            self.kd_yaw = self.get_parameter("kd_yaw").get_parameter_value().double_value

            self.declare_parameter("kp_x", 1.0)
            self.declare_parameter("kd_x", 0.5)
            self.kp_x = self.get_parameter("kp_x").get_parameter_value().double_value
            self.kd_x = self.get_parameter("kd_x").get_parameter_value().double_value

            self.declare_parameter("kp_y", 1.0)
            self.declare_parameter("kd_y", 0.5)
            self.kp_y = self.get_parameter("kp_y").get_parameter_value().double_value
            self.kd_y = self.get_parameter("kd_y").get_parameter_value().double_value

            # Feedback control parameters
            self.target_line_start = np.zeros(2, dtype=float)
            self.yaw_target_w = 0.0
            self.yaw_cur_w = 0.0

            self.yaw_world = 0.0
            self.world_origin = np.zeros(3)

            self.y_vel_target = 0.0

            self.incremental_vel = 0.0

            self.ang_z_window = deque(maxlen=20)
            self.pos_w_window = deque(maxlen=10)
            self.vel_w_window = deque(maxlen=10)

            self.lin_vel_w = np.zeros(3)
            self.pos_w = np.zeros(3)

            # self.y_pos_window = deque(maxlen=10)
            # self.y_vel_window = deque(maxlen=10)
            # self.x_vel_window = deque(maxlen=10)

            # self.yaw_cur = 0.0
            # self.y_pos_cur = 0.0

            self.odom_count = 0

            # Odometry logging setup
            self.declare_parameter("log_odom", False)
            self.log_odom_flag = self.get_parameter("log_odom").get_parameter_value().bool_value
            
            if self.log_odom_flag:
                self.get_logger().info("Odometry logging enabled.")

                self.declare_parameter("joy_rate", 0.05)
                self.joy_rate = self.get_parameter("joy_rate").get_parameter_value().double_value

                self.log_odom = OdomLog()

                # Create odom log directory relative to $ROBOT_RL_ROOT
                root = os.environ.get("ROBOT_RL_ROOT", "")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                odom_log_dir = os.path.join(root, "odom_logs", timestamp)
                os.makedirs(odom_log_dir, exist_ok=True)
                
                self.odom_log_file = os.path.join(odom_log_dir, "odom_data.csv")
                self.odom_file = open(self.odom_log_file, "w", buffering=8192)
                self.odom_writer = csv.writer(self.odom_file)
                
                # Write CSV header
                self.odom_writer.writerow([
                    "time",
                    "pos_w_x",
                    "pos_w_y",
                    "pos_w_z",
                    "quat_w_w",
                    "quat_w_x",
                    "quat_w_y",
                    "quat_w_z",
                    "lin_vel_w_x",
                    "lin_vel_w_y",
                    "lin_vel_w_z",
                    "ang_vel_w_x",
                    "ang_vel_w_y",
                    "ang_vel_w_z",
                    "yaw_w",
                    "yaw_target_w",
                    "yaw_error",
                    "yaw_cmd",
                    "x_vel_cmd",
                    "y_cmd",
                    "y_local_error",
                    "target_pos_w_x",
                    "target_pos_w_y",
                    "world_yaw",
                    "world_origin_x",
                    "world_origin_y",
                ])
                
                self.odom_start_time = self.get_clock().now().nanoseconds / 1e9

            else:
                self.log_odom = False

            # Need to get the waist joint
            self.waist_joint_angle = 0.0
            self.register_obk_subscription(
                "sub_joint_encoders",
                self.joint_encoders_callback,  # type: ignore
                ObkJointEncoders,
                key="sub_joint_encoders",  # key can be specified here or in the config file
            )

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

    def joint_encoders_callback(self, msg: ObkJointEncoders) -> None:
        """Callback for the joint encoders."""
        self.waist_joint_angle = msg.joint_pos[msg.joint_names.index("waist_yaw_joint")]

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry messages."""
        # All positions should be in the z-up world aligned frame
        # velocities are in the inverted body frame

        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position

        twist_w = msg.twist.twist

        # # negate the twist z vels
        # twist_w.angular.z = -msg.twist.twist.angular.z
        # twist_w.linear.z = -msg.twist.twist.linear.z

        ##
        # Get Yaw
        ##
        # Get the yaw from the quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp) - self.yaw_world
        self.yaw_cur_w = yaw - self.waist_joint_angle # TODO: Check sign/put back


        # Angular z moving avg:
        self.ang_z_vel = twist_w.angular.z
        self.ang_z_window.append(self.ang_z_vel)

        ##
        # Get Position Values
        ##
        # self.get_logger().info(f"World origin: {self.world_origin}")
        # self.get_logger().info(f"Position: {pos.x}, {pos.y}")

        self.pos_w[0], self.pos_w[1] = rotate_into_yaw(self.yaw_world,
                                                       pos.x - self.world_origin[0],
                                                       pos.y - self.world_origin[1])
        self.pos_w[2] = pos.z - self.world_origin[2]   # TODO: Add offset so its not at 0?

        self.pos_w_window.append(self.pos_w)


        ##
        # Get Velocities
        ##
        # Put them into the track frame
        self.lin_vel_w[0], self.lin_vel_w[1] = rotate_into_yaw(self.yaw_world, twist_w.linear.x, twist_w.linear.y)
        self.lin_vel_w[2] = twist_w.linear.z

        self.vel_w_window.append(self.lin_vel_w)

        if self.log_odom_flag:
            self.log_odom.pos_w = self.pos_w
            self.log_odom.lin_vel_w = self.lin_vel_w
            self.log_odom.quat_w = np.array([q.w, q.x, q.y, q.z])
            self.log_odom.ang_vel_w = np.array([twist_w.angular.x, twist_w.angular.y, twist_w.angular.z])
            self.log_odom.yaw_w = self.yaw_cur_w

        self.odom_count += 1

    def compute_odom_control(self, target_point) -> Tuple[float, float]:
        """
        Use the odometry measurements to compute PD feedback to the target.

        For now assuming that the target point is always the closest point to the desired line.
        In the yaw-target aligned frame this target point only has error in y. We only use feedforward for x for now.
        """

        ##
        # Yaw Control
        ##
        ang_z_filtered = sum(self.ang_z_window)/len(self.ang_z_window)

        yaw_error = self.yaw_cur_w - self.yaw_target_w
        if yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        elif yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        yaw_rate_cmd = -self.kp_yaw * yaw_error - self.kd_yaw * ang_z_filtered

        # Clamp the yaw rate command
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -self.w_z_max, self.w_z_max)

        ##
        # Y Control
        ##
        # Compute position error in global frame
        error_w = self.pos_w[:2] - target_point

        # Rotate position error into target yaw frame
        error_local = np.zeros(2)
        error_local[0], error_local[1] = rotate_into_yaw(self.yaw_target_w, error_w[0], error_w[1])

        # Rotate y velocity into target yaw frame
        vel_local = np.zeros(2)
        vel_local[0], vel_local[1] = rotate_into_yaw(self.yaw_target_w, self.lin_vel_w[0], self.lin_vel_w[1])

        # Compute control
        y_vel_local_cmd = -self.kp_y * error_local[1] - self.kd_y * (vel_local[1] - self.y_vel_target)
        y_vel_local_cmd = np.clip(y_vel_local_cmd, -self.v_y_max, self.v_y_max)

        # self.get_logger().info(f"y: {self.y_pos_cur}")

        if self.log_odom_flag:
            self.log_odom.yaw_error = yaw_error
            self.log_odom.y_error = error_local[1]

        return y_vel_local_cmd, yaw_rate_cmd


    def vel_cmd_callback(self, cmd_msg: VelocityCommand):
        """Callback for velocity command messages from the unitree joystick node."""
        self.joy_cmd_vel[0] = min(max(cmd_msg.v_x, self.v_x_min), self.v_x_max)
        self.joy_cmd_vel[1] = min(max(cmd_msg.v_y, -self.v_y_max), self.v_y_max)
        self.joy_cmd_vel[2] = min(max(cmd_msg.w_z, -self.w_z_max), self.w_z_max)
            
    def update_x_hat(self, msg):
        """Receive the joystick message."""
        self.rec_joystick = True
        RIGHT_BUMPER = 5
        Y = 3
        A = 0
        RIGHT_TRIGGER = 5
        MENU = 6
        DPAD_LEFT_RIGHT = 6
        DPAD_UP_DOWN = 7

        if msg.axes[RIGHT_TRIGGER] <= 0.1:
            raise RuntimeError("[High Level] Joystick emergency stop triggered!!")

        now = self.get_clock().now().nanoseconds / 1e9
        if msg.buttons[MENU] >= 0.9 and now - self.last_menu_press > 0.5:
            self.last_menu_press = now
            self.get_logger().info("Button mappings: \n " \
            " E-STOP: Right Trigger. \n " \
            " Forward/Backward: Left Stick. \n " \
            " Turning: Right Stick. \n" \
            " Joystick Mode: D-Pad Up. \n" \
            " Integrated Joystick + Feedback Mode: D-Pad Right. \n" \
            " Increase Incremental Velocity: Right Bumper + Y. \n" \
            " Decrease Incremental Velocity: Right Bumper + A. \n" \
            " Zero odom targets: Right Bumper (while not in incremental mode).")

        if msg.axes[DPAD_UP_DOWN] >= 0.9 and now - self.last_DPAD_UP_press > 0.5:
            self.last_DPAD_UP_press = now
            self.control_mode = "joystick"
            self.joystick_exited = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info("Joystick control enabled!")

        if msg.axes[DPAD_LEFT_RIGHT] <= -0.9 and now - self.last_DPAD_RIGHT_press > 0.5:
            self.last_DPAD_RIGHT_press = now
            self.control_mode = "integrated_joystick"
            self.get_logger().info("Integrated joystick control enabled!")
            # TODO: need to think more about how this mode will work

        if msg.buttons[RIGHT_BUMPER] >= 0.9 and self.control_mode == "integrated_joystick":
            if msg.buttons[Y] >= 0.9 and now - self.last_Y_press > 0.2:
                self.incremental_vel += self.vel_increment
                vx_max = self.get_parameter("v_x_max").get_parameter_value().double_value
                if self.incremental_vel > vx_max:
                    self.incremental_vel = vx_max
                self.get_logger().info(f"----- INCREASING VELOCITY TO {self.incremental_vel:.3f} m/s -----")
                self.last_Y_press = now
            if msg.buttons[A] >= 0.9 and now - self.last_A_press > 0.2:
                self.incremental_vel -= self.vel_increment
                vx_min = self.get_parameter("v_x_min").get_parameter_value().double_value
                if self.incremental_vel < vx_min:
                    self.incremental_vel = vx_min

                self.get_logger().info(f"----- DECREASING VELOCITY TO {self.incremental_vel:.3f} m/s -----")
                self.last_A_press = now

        elif msg.buttons[RIGHT_BUMPER] >= 0.9 and self.control_mode == "joystick" and now - self.last_RB_press > 0.5:
            self.yaw_world = self.yaw_cur_w
            self.yaw_target_w = 0.0
            self.world_origin = self.pos_w.copy()
            self.target_line_start = np.zeros(2)

            self.get_logger().info(f"World origin set to {self.world_origin} with yaw {self.yaw_world}.")

            self.last_RB_press = now

    def compute_control(self):
        """Return the commanded velocity."""
        if self.rec_joystick:
            msg = VelocityCommand()

            msg.v_x = float(self.joy_cmd_vel[0])
            msg.v_y = float(self.joy_cmd_vel[1])
            msg.w_z = float(self.joy_cmd_vel[2])

            if self.control_mode == "integrated_joystick":
                target = self.set_feedback_targets()
                y_cmd, yaw_cmd = self.compute_odom_control(target)

                msg.w_z = float(yaw_cmd)
                msg.v_y = float(y_cmd)
                msg.v_x = float(self.incremental_vel)

            elif self.control_mode != "joystick":
                raise ValueError("Unsupported control mode!")

            self.obk_publishers["pub_ctrl"].publish(msg)

            # TODO: Viz the trajectory and commands


            if self.log_odom_flag:
                self.log_odom.y_cmd = msg.v_y
                self.log_odom.yaw_cmd = msg.w_z
                self.log_odom.target_x_vel = msg.v_x

            # Log odometry data if enabled
            if self.log_odom_flag and self.odom_count % 1 == 0:
                self.write_log()

            return msg

    def set_feedback_targets(self):
        """
        Sets the global frame position targets for feedback.

        For now only doing yaw and y.
        """
        # Integrate the yaw target
        prev_yaw = self.yaw_target_w
        yaw_change = self.joy_rate*self.joy_cmd_vel[2]
        self.yaw_target_w = prev_yaw + yaw_change

        target = np.zeros(2, dtype=np.float64)

        second_point = self.target_line_start + np.array([np.cos(prev_yaw), np.sin(prev_yaw)])

        AB = second_point - self.target_line_start
        AC = self.pos_w[:2] - self.target_line_start

        AD = AB * np.dot(AB, AC)/np.dot(AB, AB)
        projection = self.target_line_start + AD

        if abs(self.joy_cmd_vel[2]) > 0.01:
            # projection = self.pos_w[:2]

            # # Compute the projected point
            # d = second_point - self.target_line_start
            # t = np.dot((self.pos_w[:2] - self.target_line_start), d)/np.dot(d,d)
            # projection = self.target_line_start + t*d

            # Need to compute the offset for the new target line
            self.target_line_start[0] = projection[0] + self.joy_rate * self.incremental_vel*(np.cos(yaw_change)) #(self.joy_cmd_vel[0]/self.joy_cmd_vel[2])*(np.sin(self.yaw_target_w) - np.sin(prev_yaw))

            self.target_line_start[1] = projection[1] + self.joy_rate * self.incremental_vel*(np.sin(yaw_change)) #(self.joy_cmd_vel[0]/self.joy_cmd_vel[2])*(np.cos(prev_yaw) - np.cos(self.yaw_target_w))

            target = self.target_line_start

            self.get_logger().info(f"y update: {self.incremental_vel*(np.sin(yaw_change))}")
            self.get_logger().info(f"Target line start: {self.target_line_start}")
            self.get_logger().info(f"Target position set to {target}")
            self.get_logger().info(f"Projected point is {projection}")
            self.get_logger().info(f"Yaw change is {yaw_change}")
            self.get_logger().info(f"joy yaw rate is {self.joy_cmd_vel[2]}")
            self.get_logger().info(f"Yaw target: {self.yaw_target_w}")
            self.get_logger().info(f"Yaw: {self.yaw_cur_w}\n")
        else:
            # # Just project - no yaw adjustments
            # # Compute the projected point
            # d = second_point - self.target_line_start
            # t = np.dot((self.pos_w[:2] - self.target_line_start), d) / np.dot(d, d)
            # target = self.target_line_start + t * d

            target = projection

        if self.log_odom_flag:
            self.log_odom.target_pos_w[:2] = target.copy()
            self.log_odom.yaw_target_w = self.yaw_target_w

        return target

    def write_log(self):
        """Write teh log."""
        current_time = self.get_clock().now().nanoseconds / 1e9 - self.odom_start_time

        self.log_odom.time = current_time

        # Write row to CSV
        self.odom_writer.writerow([
            self.log_odom.time,
            self.log_odom.pos_w[0],
            self.log_odom.pos_w[1],
            self.log_odom.pos_w[2],
            self.log_odom.quat_w[0],
            self.log_odom.quat_w[1],
            self.log_odom.quat_w[2],
            self.log_odom.quat_w[3],
            self.log_odom.lin_vel_w[0],
            self.log_odom.lin_vel_w[1],
            self.log_odom.lin_vel_w[2],
            self.log_odom.ang_vel_w[0],
            self.log_odom.ang_vel_w[1],
            self.log_odom.ang_vel_w[2],
            self.log_odom.yaw_w,
            self.log_odom.yaw_target_w,
            self.log_odom.yaw_error,
            self.log_odom.yaw_cmd,
            self.log_odom.target_x_vel,
            self.log_odom.y_cmd,
            self.log_odom.y_local_error,
            self.log_odom.target_pos_w[0],
            self.log_odom.target_pos_w[1],
            self.log_odom.world_yaw,
            self.log_odom.world_origin[0],
            self.log_odom.world_origin[1],
        ])

def rotate_into_yaw(yaw, x, y) -> tuple[float, float]:
    x_new = np.cos(yaw)*x + np.sin(yaw)*y
    y_new = -np.sin(yaw)*x + np.cos(yaw)*y

    return (x_new, y_new)

def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, HighLevelController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()