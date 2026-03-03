import os
from abc import ABC
from typing import Tuple
from dataclasses import dataclass, field, asdict
from scipy.spatial.transform import Rotation
import numpy as np
import torch
from datetime import datetime
import time
import csv
import shutil
from tf2_ros import StaticTransformBroadcaster
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
from sensor_msgs.msg import Joy, Imu, JoyFeedback
from geometry_msgs.msg import TransformStamped, PoseArray, Pose, Twist

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

        self.declare_parameter("temp_safety_threshold", 110.0)
        self.declare_parameter("temp_lower_start", 80.0)
        self.declare_parameter("safe_temp_speed", 1.1)
        self.temp_safety_threshold = self.get_parameter("temp_safety_threshold").get_parameter_value().double_value
        self.temp_lower_start = self.get_parameter("temp_lower_start").get_parameter_value().double_value
        self.safe_temp_speed = self.get_parameter("safe_temp_speed").get_parameter_value().double_value
        self.winding_temps = None
        self.temp_override = False
        self.temp_override_change_time = 0.0

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

            self.declare_parameter("traj_dt", 0.1)
            self.declare_parameter("traj_nodes", 10)
            self.traj_dt = self.get_parameter("traj_dt").get_parameter_value().double_value
            self.traj_nodes = self.get_parameter("traj_nodes").get_parameter_value().integer_value

            # Feedback control parameters
            self.target_line_start = np.zeros(2, dtype=float)
            self.yaw_target_w = 0.0
            self.yaw_cur_w = 0.0
            self.quat_w = np.zeros(4)
            self.quat_w[3] = 1.0

            self.yaw_world = 0.0
            self.world_origin = np.zeros(3)

            self.y_vel_target = 0.0

            self.incremental_vel = 0.0

            self.ang_z_window = deque(maxlen=20)
            self.pos_w_window = deque(maxlen=10)
            self.vel_w_window = deque(maxlen=10)

            # Timestamped history for finite difference velocity estimation
            self.pos_time_history = deque(maxlen=4)
            self.yaw_time_history = deque(maxlen=4)

            self.lin_vel_w = np.zeros(3)
            self.pos_w = np.zeros(3)
            self.quat_camera_init = np.zeros(4)
            self.R_odom_camera_init = None
            self.body_heading = np.array([1.0, 0.0, 0.0])
            self.tf_static_broadcaster = StaticTransformBroadcaster(self)
            self.tf_static_odom_world_broadcaster = StaticTransformBroadcaster(self)

            self.declare_parameter("odom_frame", "odom_frame")
            self.declare_parameter("camera_init_frame", "camera_init_frame")
            self.odom_frame_str = self.get_parameter("odom_frame").get_parameter_value().string_value
            self.camera_init_frame_str = self.get_parameter("camera_init_frame").get_parameter_value().string_value

            # self.y_pos_window = deque(maxlen=10)
            # self.y_vel_window = deque(maxlen=10)
            # self.x_vel_window = deque(maxlen=10)

            # MPC Commands
            self.mpc_cmd = np.zeros(3)  # x, y, yaw
            self.mpc_enabled = False

            self.lidar_imu_acc_hist = deque(maxlen=15)
            self.rec_lidar_imu = False

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


            # Declare subscriber to lidar IMU
            self.register_obk_subscription(
                "sub_lidar_imu",
                self.lidar_imu_callback,  # type: ignore
                key="sub_lidar_imu_key",  # key can be specified here or in the config file
                msg_type=Imu,
            )

        # Declare subscriber to velocity commands from the Untiree joystick node
        self.register_obk_subscription(
            "sub_vel_cmd_setting",
            self.vel_cmd_callback,  # type: ignore
            key="sub_vel_cmd_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand,
        )

        self.register_obk_publisher(
            "pub_joy_feedback",
            key="pub_joy_feedback_key",
            msg_type=JoyFeedback,
        )

        # Zeroed odometry
        self.register_obk_publisher(
            "pub_odom_data",
            key="pub_odom_data_key",
            msg_type=Odometry,
        )

        # Trajectory data
        self.register_obk_publisher(
            "pub_traj_data",
            key="pub_traj_data_key",
            msg_type=PoseArray,
        )

        # Safety velocity command
        self.register_obk_subscription(
            "sub_mpc_setting",
            key="sub_mpc_key",
            callback=self.mpc_command_callback,
            msg_type=Twist,
        )

        self.cmd_vel = np.zeros((3,))

    def joint_encoders_callback(self, msg: ObkJointEncoders) -> None:
        """Callback for the joint encoders."""
        self.waist_joint_angle = msg.joint_pos[msg.joint_names.index("waist_yaw_joint")]
        if len(msg.motor_winding_temps) > 0:
            self.winding_temps = msg.motor_winding_temps

    def lidar_imu_callback(self, msg) -> None:
        """Callback for the lidar imu."""
        # Store acceleration vectors
        lin_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.lidar_imu_acc_hist.append(lin_acc)

        self.rec_lidar_imu = True

    def mpc_command_callback(self, msg) -> None:
        self.mpc_cmd[0] = np.clip(msg.linear.x, self.v_x_min, self.v_x_max)
        self.mpc_cmd[1] = np.clip(msg.linear.y, -self.v_y_max, self.v_y_max)
        self.mpc_cmd[2] = np.clip(msg.angular.z, -self.w_z_max, self.w_z_max)  

    def pub_odom(self) -> None:
        """
        Callback for the timer to publish the odom data.

        Publishes odom data in the zeroed world frame.
        """
        # Publish the correct frame odom data
        msg = Odometry()
        msg.pose.pose.position.x = self.pos_w[0]
        msg.pose.pose.position.y = self.pos_w[1]
        msg.pose.pose.position.z = self.pos_w[2]

        msg.pose.pose.orientation.w = self.quat_w[3]
        msg.pose.pose.orientation.x = self.quat_w[0]
        msg.pose.pose.orientation.y = self.quat_w[1]
        msg.pose.pose.orientation.z = self.quat_w[2]

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame_str   # TODO: Check
        msg.child_frame_id = "pelvis"   # Is it the pelvis or the head?

        # No twist because I don't have it

        self.obk_publishers["pub_odom_data_key"].publish(msg)

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry messages."""
        # All positions should be in the z-up world aligned frame
        # velocities are in the inverted body frame

        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position

        # Note: FAST-LIO doesn't actually provide the velocities by default
        twist_w = msg.twist.twist

        # TODO: Can sim without velocity feedback

        # Orientation in camera init frame
        self.quat_camera_init[3] = q.w
        self.quat_camera_init[0] = q.x
        self.quat_camera_init[1] = q.y
        self.quat_camera_init[2] = q.z

        # Get the values in the camera init frame
        position_camera_init = np.array([pos.x, pos.y, pos.z])
        quat_camera_init = np.array([q.x, q.y, q.z, q.w])  # scipy format [x,y,z,w]
        R_camera_init = Rotation.from_quat(quat_camera_init)

        if self.R_odom_camera_init is not None:
            # Transform the data from the camera init frame into the world (odom) frame.

            # Transform position: p_odom = R @ p_camera_init + t
            self.pos_w = self.R_odom_camera_init.apply(position_camera_init) + self.t_odom_camera_init

            # Transform orientation: R_odom = R_transform @ R_camera_init
            R_odom = self.R_odom_camera_init * R_camera_init

            self.quat_w = R_odom.as_quat(scalar_first=False)    # TODO: Apply the roll to bring it back to z-up

            # Extract yaw from transformed orientation
            heading = R_odom.apply(self.body_heading)
            yaw = np.atan2(heading[1], heading[0])

            self.start = np.array([self.pos_w[0], self.pos_w[1], yaw])
            self.z0 = self.pos_w[2]
            self.frame_id = self.get_parameter("odom_frame").get_parameter_value().string_value
        else:
            # Get the yaw from the quaternion
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            # Use raw position values
            self.pos_w[0] = pos.x
            self.pos_w[1] = pos.y
            self.pos_w[2] = pos.z

        ##
        # Adjust Yaw
        ##
        self.yaw_cur_w = yaw - self.waist_joint_angle # TODO: Check sign/put back

        ##
        # Record position
        ##
        self.pos_w_window.append(self.pos_w)


        ##
        # Get Velocities via finite difference
        ##
        # Get timestamp from message header
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Store position and yaw with timestamp
        self.pos_time_history.append((current_time, self.pos_w.copy()))
        self.yaw_time_history.append((current_time, self.yaw_cur_w))

        # Compute velocities via finite difference
        self.lin_vel_w = self._compute_linear_velocity()
        ang_z_vel = self._compute_yaw_rate()
        self.ang_z_window.append(ang_z_vel)

        self.pub_odom()     # Publish the zeroed odom

        if self.log_odom_flag:
            self.log_odom.pos_w = self.pos_w
            self.log_odom.lin_vel_w = self.lin_vel_w
            self.log_odom.quat_w = np.array([q.w, q.x, q.y, q.z])
            self.log_odom.ang_vel_w = np.array([0.0, 0.0, ang_z_vel])
            self.log_odom.yaw_w = self.yaw_cur_w

        self.odom_count += 1

    def _wrap_angle_diff(self, angle_diff: float) -> float:
        """Wrap angle difference to [-pi, pi]."""
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        return angle_diff

    def _compute_linear_velocity(self) -> np.ndarray:
        """Compute linear velocity from position history using finite difference with moving average."""
        if len(self.pos_time_history) < 2:
            return np.zeros(3)

        # Compute velocity from oldest to newest for smoothing
        vel_local = np.zeros(3)
        velocities = []
        history = list(self.pos_time_history)
        for i in range(1, len(history)):
            dt = history[i][0] - history[i-1][0]
            if dt > 1e-6:
                vel = (history[i][1] - history[i-1][1]) / dt
                vel_local[:2] = rotate_into_yaw(self.yaw_cur_w, vel[0], vel[1])
                vel_local[2] = vel[2]
                velocities.append(vel_local)

        if not velocities:
            return np.zeros(3)

        return np.mean(velocities, axis=0)

    def _compute_yaw_rate(self) -> float:
        """Compute yaw rate from yaw history using finite difference with moving average."""
        if len(self.yaw_time_history) < 2:
            return 0.0

        rates = []
        history = list(self.yaw_time_history)
        for i in range(1, len(history)):
            dt = history[i][0] - history[i-1][0]
            if dt > 1e-6:
                dyaw = self._wrap_angle_diff(history[i][1] - history[i-1][1])
                rates.append(dyaw / dt)

        if not rates:
            return 0.0

        return float(np.mean(rates))

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
            self.log_odom.y_local_error = error_local[1]

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

        if msg.axes[DPAD_LEFT_RIGHT] >= 0.9 and now - self.last_DPAD_LEFT_press > 0.5:
            self.last_DPAD_LEFT_press = now
            self.mpc_enabled = not self.mpc_enabled
            self.get_logger().info("MPC status set to " + str(self.mpc_enabled))

        if msg.buttons[RIGHT_BUMPER] >= 0.9 and self.control_mode == "integrated_joystick":
            if msg.buttons[Y] >= 0.9 and now - self.last_Y_press > 0.1:
                self.incremental_vel += self.vel_increment
                vx_max = self.get_parameter("v_x_max").get_parameter_value().double_value
                if self.incremental_vel > vx_max:
                    self.incremental_vel = vx_max
                self.get_logger().info(f"----- INCREASING VELOCITY TO {self.incremental_vel:.3f} m/s -----")
                self.last_Y_press = now
            if msg.buttons[A] >= 0.9 and now - self.last_A_press > 0.1:
                self.incremental_vel -= self.vel_increment
                vx_min = self.get_parameter("v_x_min").get_parameter_value().double_value
                if self.incremental_vel < vx_min:
                    self.incremental_vel = vx_min

                self.get_logger().info(f"----- DECREASING VELOCITY TO {self.incremental_vel:.3f} m/s -----")
                self.last_A_press = now

        elif msg.buttons[RIGHT_BUMPER] >= 0.9 and self.control_mode == "joystick" and now - self.last_RB_press > 0.5:
            self.get_logger().info(f"Initializing Zeroing")
            self.zero_world()
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
    
                if self.mpc_enabled:  # Must be in integrated joystick to use the MPC commands for now
                    msg.v_x = float(self.mpc_cmd[0])
                    msg.v_y = float(self.mpc_cmd[1])
                    msg.w_z = float(self.mpc_cmd[2])

            elif self.control_mode != "joystick":
                raise ValueError("Unsupported control mode!")


            ##
            # Temperature Control
            ##
            if self.winding_temps is not None:
                if np.max(self.winding_temps) >= self.temp_safety_threshold:
                    self.get_logger().warn("Winding temperature exceeds safety threshold! Overriding target speed!")
                    self.temp_override_change_time = self.get_clock().now().nanoseconds / 1e9
                    self.temp_override = True

                if self.temp_override:
                    msg.v_x = min(msg.v_x, self.safe_temp_speed)
                    self.incremental_vel = msg.v_x  # Reset the incremental vel for an easier reset

                    if np.max(self.winding_temps) <= self.temp_lower_start:
                        self.get_logger().warn("Winding temperatures have cooled down. Allowing user control again.")
                        self.temp_override = False
                        self.temp_override_change_time = self.get_clock().now().nanoseconds / 1e9

                self.joy_feedback()

            self.obk_publishers["pub_ctrl"].publish(msg)

            # TODO: Viz the trajectory and commands

            # Publish the desired trajectory
            self.pub_traj()

            if self.log_odom_flag:
                self.log_odom.y_cmd = msg.v_y
                self.log_odom.yaw_cmd = msg.w_z
                self.log_odom.target_x_vel = msg.v_x

            # Log odometry data if enabled
            if self.log_odom_flag and self.odom_count % 1 == 0:
                self.write_log()

            return msg

    def pub_traj(self) -> None:
        """
        Publish the trajectory we want to track.
        """
        msg = PoseArray()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame_str

        for i in range(self.traj_nodes):
            point = Pose()

            # Compute time into traj
            t = i*self.traj_dt

            # Get the commanded velocity in the local x - no support for joy-sticking right now
            vel_world = self.incremental_vel * np.array([np.cos(self.yaw_target_w), np.sin(self.yaw_target_w)])

            # Get current point on the target line
            projection = self.project_position_onto_line(self.yaw_target_w)

            point.position.x = projection[0] + (t * vel_world[0])
            point.position.y = projection[1] + (t * vel_world[1])
            point.position.z = self.pos_w[2]

            # Convert yaw into a quaternion
            rot = Rotation.from_euler('z', self.yaw_target_w, degrees=False)
            quat = rot.as_quat(scalar_first=False)
            point.orientation.w = quat[3]
            point.orientation.x = quat[0]
            point.orientation.y = quat[1]
            point.orientation.z = quat[2]

            # # Velocities in the world frame
            # point.velocity.linear.x = vel_world[0]
            # point.velocity.linear.y = vel_world[1]
            # point.velocity.linear.z = 0.0
            #
            # point.velocity.angular.x = 0.0
            # point.velocity.angular.y = 0.0
            # point.velocity.angular.z = 0.0      # On the line there is no commanded yaw

            msg.poses.append(point)

        self.obk_publishers["pub_traj_data_key"].publish(msg)

    def set_feedback_targets(self):
        """
        Sets the global frame position targets for feedback.

        For now only doing yaw and y.
        """
        # Integrate the yaw target
        prev_yaw = self.yaw_target_w
        yaw_change = self.joy_rate*self.joy_cmd_vel[2]
        self.yaw_target_w = (prev_yaw + yaw_change) % (2*np.pi)

        target = np.zeros(2, dtype=np.float64)

        projection = self.project_position_onto_line(prev_yaw)

        if abs(self.joy_cmd_vel[2]) > 0.01:
            dp_w = self.joy_rate * self.incremental_vel * np.array([np.cos(self.yaw_target_w), np.sin(self.yaw_target_w)])

            # Need to compute the offset for the new target line
            self.target_line_start = projection + dp_w

            target = self.target_line_start

            self.get_logger().info(f"dp_w: {dp_w}")
            self.get_logger().info(f"Target line start: {self.target_line_start}")
            self.get_logger().info(f"Target position set to {target}")
            self.get_logger().info(f"Projected point is {projection}")
            self.get_logger().info(f"Yaw change is {yaw_change}")
            self.get_logger().info(f"joy yaw rate is {self.joy_cmd_vel[2]}")
            self.get_logger().info(f"Yaw target: {self.yaw_target_w}")
            self.get_logger().info(f"Yaw: {self.yaw_cur_w}\n")
        else:
            target = projection

        if self.log_odom_flag:
            self.log_odom.target_pos_w[:2] = target.copy()
            self.log_odom.yaw_target_w = self.yaw_target_w

        return target

    def project_position_onto_line(self, yaw):
        """Project the current position onto the ray starting at target_line_start in the direction of yaw."""
        second_point = self.target_line_start + np.array([np.cos(yaw), np.sin(yaw)])

        AB = second_point - self.target_line_start
        AC = self.pos_w[:2] - self.target_line_start

        t = np.dot(AB, AC) / np.dot(AB, AB)
        t = max(t, 0.0)  # Clamp to ray: don't project behind the start point
        projection = self.target_line_start + AB * t

        return projection

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

    def zero_world(self):
        self._zeroing_compute_transform()

        self.yaw_target_w = 0.0
        self.target_line_start = np.zeros(2)

        self.get_logger().info(f"World origin set to {self.world_origin} with yaw {self.yaw_world}.")

    def _zeroing_compute_transform(self) -> None:
        """Compute gravity-aligned, yaw-zeroed, position-zeroed transform."""
        if self.odom_count == 0:
            self.get_logger().warn("ZEROING: No odometry available!")
            raise ValueError("ZEROING: No odometry available!")

        if not self.rec_lidar_imu:
            self.get_logger().warn("ZEROING: No IMU available! Zeroing failed.")
            raise ValueError("ZEROING: No IMU available!")
        else:
            oR_b = self._compute_gravity_alignment()  # Compute the transform from aligns the gravity vector with the z axis

        # Get current pose
        cp_b = self.pos_w #self.odom_msg.pose.pose.position  # position of the body frame in the camera_init frame
        cquat_b = self.quat_camera_init #self.odom_msg.pose.pose.orientation  # orientation of the body frame in the camera_init frame

        cR_b = Rotation.from_quat(
            cquat_b)  # Rotation matrix describing orientation of body frame in the camera_init frame

        # Apply gravity alignment first to get gravity-aligned orientation
        oR_c = oR_b * cR_b.inv()

        # Compute rotation to zero yaw (in gravity-aligned frame)
        heading = oR_b.apply(self.body_heading)
        initial_yaw = np.atan2(heading[1], heading[0])
        R_zero_yaw = Rotation.from_euler('z', -(initial_yaw - self.waist_joint_angle))

        # Combined rotation: first apply gravity alignment, then yaw zeroing
        self.R_odom_camera_init = R_zero_yaw * oR_c

        # Translation: move initial position to origin in xy
        # p_odom = R @ p_camera_init + t
        # We want p_odom = [0, 0, z_offset] when p_camera_init = initial_position
        op_b = self.R_odom_camera_init.apply(cp_b)
        self.t_odom_camera_init = np.array([
            -op_b[0],
            -op_b[1],
            0.75 - op_b[2], #0.0  # Keep z offset from gravity-aligned frame
        ])

        self.get_logger().info(f"ZEROING: Transform computed. Initial yaw was {np.degrees(initial_yaw):.1f} deg")

        # Publish static transform
        self._publish_odom_camera_init_transform()

    def _compute_gravity_alignment(self) -> Rotation:
        """Compute rotation to align z-axis with up (opposite gravity).

        When robot is stationary, accelerometer reads reaction to gravity,
        which points UP (opposite to gravity). We compute rotation R such that
        R @ accel_reading = [0, 0, +1] (i.e., z-axis points up).

        Returns:
            Rotation that aligns sensor frame z-axis with world up.
        """

        # Compute mean of the measured accelerations
        mean_acc = np.zeros(3)
        for acc in self.lidar_imu_acc_hist:
            mean_acc += acc

        mean_acc /= len(self.lidar_imu_acc_hist)
        accel_mag = np.linalg.norm(mean_acc)

        if accel_mag < 0.1:
            self.get_logger().warn("ZEROING: Accelerometer reading too small!")
            raise ValueError("ZEROING: Accelerometer reading too small!")

        # Normalize accelerometer reading (points UP when stationary)
        g_b = mean_acc / accel_mag  # gravity in the body frame

        # Target: accelerometer (pointing up) should align with +z
        e_z = np.array([0.0, 0.0, 1.0])

        # Compute rotation from accel_normalized to target
        a = np.cross(g_b, e_z)
        s = np.linalg.norm(a)
        c = np.dot(g_b, e_z)

        if s < 1e-6:
            # Vectors are parallel
            if c > 0:
                return Rotation.identity()
            else:
                # 180 degree rotation around x-axis
                return Rotation.from_euler('x', np.pi)

        # Axis-angle approach
        axis = a / s
        angle = np.atan2(s, c)

        return Rotation.from_rotvec(axis * angle)

    def _publish_odom_camera_init_transform(self) -> None:
        """Publish odom -> camera_init static transform."""
        if self.tf_static_broadcaster is None:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame_str
        t.child_frame_id = self.camera_init_frame_str

        quat = self.R_odom_camera_init.as_quat()
        t.transform.translation.x = self.t_odom_camera_init[0]
        t.transform.translation.y = self.t_odom_camera_init[1]
        t.transform.translation.z = self.t_odom_camera_init[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_static_broadcaster.sendTransform(t)
        self.get_logger().info("ZEROING: Published static transform odom -> camera_init")

        ##
        # World -> odom frame tf
        ##
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = self.odom_frame_str

        quat = self.R_odom_camera_init.as_quat()
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_static_odom_world_broadcaster.sendTransform(t)
        self.get_logger().info("ZEROING: Published static transform odom -> world")


    def joy_feedback(self) -> None:
        """
        Rumble the controller whenever the temperature override changes state.
        """
        if (self.get_clock().now().nanoseconds / 1e9) - self.temp_override_change_time <= 1.0:
            msg = JoyFeedback()
            msg.intensity = 1.0
            msg.id = 0
            msg.type = JoyFeedback.TYPE_RUMBLE

            self.obk_publishers["pub_joy_feedback_key"].publish(msg)


def rotate_into_yaw(yaw, x, y) -> tuple[float, float]:
    x_new = np.cos(yaw)*x + np.sin(yaw)*y
    y_new = -np.sin(yaw)*x + np.cos(yaw)*y

    return (x_new, y_new)

def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, HighLevelController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()