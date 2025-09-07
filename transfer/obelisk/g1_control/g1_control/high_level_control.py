import os
from abc import ABC

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

        self.rec_joystick = False

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

            self.yaw_target = 0.0
            self.yaw_rate_cmd = 0.0
            self.y_pos_target = 0.0
            self.y_vel_target = 0.0
            self.yaw_init = 0.0
            self.x_target = 0.0
            self.x_cmd = 0.0
            self.y_cmd = 0.0

            self.ang_z_window = deque(maxlen=20)
            self.y_pos_window = deque(maxlen=10)
            self.y_vel_window = deque(maxlen=10)

            self.yaw_cur = 0.0
            self.y_pos_cur = 0.0

            self.odom_count = 0

            # Odometry logging setup
            self.declare_parameter("log_odom", False)
            self.log_odom = self.get_parameter("log_odom").get_parameter_value().bool_value
            
            if self.log_odom:
                self.get_logger().info("Odometry logging enabled.")
                
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
                    "time", "pos_x", "pos_y", "pos_z", 
                    "quat_x", "quat_y", "quat_z", "quat_w",
                    "vel_x", "vel_y", "vel_z",
                    "ang_vel_x", "ang_vel_y", "ang_vel_z", "ang_z_filtered",
                    "yaw", "yaw_target", "yaw_error", "yaw_rate_cmd",
                    "x_cmd", "y_cmd", "y_vel_avg", "y_pos_target"
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
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        self.yaw_cur = yaw - self.waist_joint_angle # TODO: Check sign/put back


        # Angular z moving avg:
        self.ang_z_vel = twist_w.angular.z
        self.ang_z_window.append(self.ang_z_vel)

        ##
        # Get Position Values
        ##
        self.x_pos_cur, self.y_pos_cur = rotate_into_yaw(self.yaw_init, pos.x, pos.y)
        self.z_pos_cur = pos.z

        self.y_pos_window.append(self.y_pos_cur)


        ##
        # Get Velocities
        ##
        # Put them into the track frame
        self.x_vel_cur, self.y_vel_cur = rotate_into_yaw(self.yaw_init, twist_w.linear.x, twist_w.linear.y)

        # Flip z to the correct frame
        self.z_vel_cur = twist_w.linear.z

        self.y_vel_window.append(self.y_vel_cur)

        # Log odometry data if enabled
        if self.log_odom and self.odom_count % 1 == 0:
            current_time = self.get_clock().now().nanoseconds / 1e9 - self.odom_start_time
            
            # Extract orientation (quaternion)
            orient = msg.pose.pose.orientation
                        
            # Extract angular velocity
            ang_vel = msg.twist.twist.angular

            yaw_error = self.yaw_cur - self.yaw_target
            y_vel_avg = sum(self.y_vel_window)/len(self.y_vel_window)
            ang_z_filtered = sum(self.ang_z_window)/len(self.ang_z_window)

            # Write row to CSV
            # TODO: Make it easier to add stuff to the plots, add y vel to the plots
            self.odom_writer.writerow([
                current_time,
                self.x_pos_cur, self.y_pos_cur, self.z_pos_cur,
                orient.x, orient.y, orient.z, orient.w,
                self.x_vel_cur, self.y_vel_cur, self.z_vel_cur,
                ang_vel.x, ang_vel.y, self.ang_z_vel, ang_z_filtered,
                self.yaw_cur, self.yaw_target, yaw_error, self.yaw_rate_cmd,
                self.x_cmd, self.y_cmd, y_vel_avg, self.y_pos_target
            ])

        self.odom_count += 1

    def compute_odom_control(self):

        y_pos_avg = sum(self.y_pos_window)/len(self.y_pos_window)

        ##
        # Yaw Control
        ##
        target_dist = 3 # m
        # self.yaw_target = np.atan2(y_pos_avg - self.y_pos_target, target_dist) + self.yaw_init

        ang_z_filtered = sum(self.ang_z_window)/len(self.ang_z_window)

        yaw_error = self.yaw_cur - self.yaw_target
        if yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        elif yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        yaw_rate_cmd = -self.kp_yaw * yaw_error - self.kd_yaw * ang_z_filtered

        # Clamp the yaw rate command
        self.yaw_rate_cmd = np.clip(yaw_rate_cmd, -self.w_z_max, self.w_z_max)

        ##
        # Y Control
        ##
        y_vel_avg = sum(self.y_vel_window)/len(self.y_vel_window)

        # TODO: Remove/debug the gain sign
        # gain_sign = 1.0 if (self.yaw_cur - self.yaw_init) >= -np.pi/2 and (self.yaw_cur - self.yaw_init) <= np.pi/2 else -1.0
        self.y_cmd = -self.kp_y * (y_pos_avg - self.y_pos_target) - self.kd_y * (y_vel_avg - self.y_vel_target)
        # self.y_cmd *= gain_sign
        self.y_cmd = np.clip(self.y_cmd, -self.v_y_max, self.v_y_max)
        # TODO: How can y_cmd be positive when both errors are positive? Gain sign error? Logging error?
        # TODO: Try moving logging here

        # self.y_cmd = 0.0
        # self.get_logger().info(f"y: {self.y_pos_cur}")

        ##
        # X Control
        ##
        # self.x_vel_cmd = self.cmd_vel[0] - self.kp_x * (self.x_pos_cur - self.x_target) - self.kd_x * (self.x_vel_cur)
        self.x_cmd = self.cmd_vel[0]


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
            self.cmd_vel[1] = min(max(cmd_msg.v_y, -self.v_y_max), self.v_y_max)
            self.cmd_vel[2] = min(max(cmd_msg.w_z, -self.w_z_max), self.w_z_max)
            
    def update_x_hat(self, msg):
        """Receive the joystick message."""
        self.rec_joystick = True

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
            self.cmd_vel[0] = self.vel_increment_start     
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
                self.yaw_init = self.yaw_cur
                self.yaw_target = self.yaw_cur
                x, y = rotate_into_yaw(self.yaw_init, self.x_pos_cur, self.y_pos_cur)
                self.y_pos_target = y
                self.get_logger().info(f"Odom targets zeroed at: Y position: {self.y_pos_target}, yaw target: {self.yaw_target}!")

                self.x_target = self.x_pos_cur
                self.get_logger().info(f"X position target set to current X position: {self.x_target}.")
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
        if self.rec_joystick:
            msg = VelocityCommand()
            msg.v_x = float(self.cmd_vel[0])
            msg.v_y = float(self.cmd_vel[1])

            msg.w_z = float(self.cmd_vel[2])
            
            if self.use_odom and self.ang_vel_mode == "odom":
                self.compute_odom_control()

                msg.w_z = float(self.yaw_rate_cmd)

                msg.v_y = float(self.y_cmd)

                # TODO: Add option to set cmd vel with the x PD controller

            self.obk_publishers["pub_ctrl"].publish(msg)

            return msg
    
def rotate_into_yaw(yaw, x, y) -> tuple[float, float]:
    x_new = np.cos(yaw)*x + np.sin(yaw)*y
    y_new = -np.sin(yaw)*x + np.cos(yaw)*y

    return (x_new, y_new)

def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, HighLevelController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()