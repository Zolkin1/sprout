import os
import csv
import shutil
import numpy as np
from abc import ABC
from typing import Optional
from datetime import datetime
from scipy.spatial.transform import Rotation

import tf2_ros
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from rclpy.executors import SingleThreadedExecutor
from obelisk_py.core.utils.ros import spin_obelisk
from obelisk_py.core.control import ObeliskController
from obelisk_control_msgs.msg import VelocityCommand
from visualization_msgs.msg import Marker, MarkerArray

from mpc.traj_opt import (
    solve_nominal_cbf, extract_solution, init_decision_var, init_params
)
from mpc.dynamics import CasadiUnicycle
from resource.problems import problem_dict

from enum import IntEnum
from sensor_msgs.msg import Joy, Imu


class Mode(IntEnum):
    IDLE = 0
    ZEROING = 1
    RUNNING = 2


class ZeroingState(IntEnum):
    IDLE = 0
    WAITING_STOP = 1
    COMPUTING = 2


class MPCController(ObeliskController, ABC):

    def __init__(self, node_name: str = "mpc_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, Twist, Odometry)

        self.frame_id = ""
        self.z0 = 0

        # Load policy
        self.declare_parameter("problem_name", "gap")
        self.declare_parameter("dt", 0.4)
        self.declare_parameter("z_max", [1e9, 1e9, 1e9])
        self.declare_parameter("z_min", [-1e9, -1e9, -1e9])
        self.declare_parameter("v_max", [0.8, 0.2, 0.8])
        self.declare_parameter("v_min", [-0.3, -0.2, -0.8])
        self.declare_parameter("body_heading", [1., 0., 0.])  # Vector in the lidar body frame which corrosponds to the heading of the robot.
        self.declare_parameter("warm_start", "start")
        self.declare_parameter("N", 25)
        self.declare_parameter("Q", [10., 10., 0.1])
        self.declare_parameter("R", [0.1, 0.5, 0.1])
        self.declare_parameter("cbf_alpha", 0.7)
        self.declare_parameter("cbf_delta", 0.2)
        self.declare_parameter("v_norm_max", -1.0)  # negative = disabled
        self.declare_parameter("dv_max", [-1.0, -1.0, -1.0])  # per-component input rate limits, negative = disabled
        self.declare_parameter("goal_deadzone", 0.1)
        self.declare_parameter("plan_topic", "/mpc_plan")
        self.declare_parameter("path_topic", "/robot_path")
        self.declare_parameter("debug_file", "mpc_debug.csv")

        problem_name = self.get_parameter("problem_name").get_parameter_value().string_value
        dt = self.get_parameter("dt").get_parameter_value().double_value
        z_max = np.array(self.get_parameter("z_max").get_parameter_value().double_array_value)
        z_min = np.array(self.get_parameter("z_min").get_parameter_value().double_array_value)
        v_max = np.array(self.get_parameter("v_max").get_parameter_value().double_array_value)
        v_min = np.array(self.get_parameter("v_min").get_parameter_value().double_array_value)
        warm_start = self.get_parameter("warm_start").get_parameter_value().string_value
        self.N = self.get_parameter("N").get_parameter_value().integer_value
        Q = np.diag(self.get_parameter("Q").get_parameter_value().double_array_value)
        R = np.diag(self.get_parameter("R").get_parameter_value().double_array_value)
        cbf_alpha = self.get_parameter("cbf_alpha").get_parameter_value().double_value
        cbf_delta = self.get_parameter("cbf_delta").get_parameter_value().double_value
        v_norm_max_val = self.get_parameter("v_norm_max").get_parameter_value().double_value
        self.v_norm_max = v_norm_max_val if v_norm_max_val > 0 else None
        dv_max_vals = np.array(self.get_parameter("dv_max").get_parameter_value().double_array_value)
        self.dv_max = dv_max_vals if np.all(dv_max_vals > 0) else None
        self.goal_deadzone = self.get_parameter("goal_deadzone").get_parameter_value().double_value
        debug_file = self.get_parameter("debug_file").get_parameter_value().string_value
        root = os.environ.get("ROBOT_RL_ROOT", "")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(root, "ctrl_logs", "mpc", timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.fn = os.path.join(self.log_dir, debug_file)
        self.body_heading = np.array(self.get_parameter("body_heading").get_parameter_value().double_array_value)
        self.body_heading /= np.linalg.norm(self.body_heading)

        self.start = problem_dict[problem_name]["start"]
        self.goal = problem_dict[problem_name]["goal"]
        self.obs = problem_dict[problem_name]["obs"]

        # Store world-frame obstacles/goal (fixed in world frame)
        self.goal_world = problem_dict[problem_name]["goal"].copy()
        self.obs_world = {
            'cx': problem_dict[problem_name]["obs"]['cx'].copy(),
            'cy': problem_dict[problem_name]["obs"]['cy'].copy(),
            'rx': problem_dict[problem_name]["obs"]['rx'].copy(),
            'ry': problem_dict[problem_name]["obs"]['ry'].copy(),
            'yaw': problem_dict[problem_name]["obs"]['yaw'].copy()
        }

        self.planning_model = CasadiUnicycle(dt, z_min, z_max, v_min, v_max)
        self.v_prev = np.zeros(self.planning_model.m)
        self.sol, self.solver = solve_nominal_cbf(
            self.start,
            self.goal,
            self.obs,
            self.planning_model,
            self.N,
            Q,
            R,
            alpha=cbf_alpha,
            delta=cbf_delta,
            warm_start=warm_start,
            debug_filename=self.fn,
            v_norm_max=self.v_norm_max,
            dv_max=self.dv_max
        )

        # MPC solution log file
        n, m = self.planning_model.n, self.planning_model.m
        mpc_log_header = ["time"]
        mpc_log_header += [f"z_ic_{j}" for j in range(n)]
        mpc_log_header += [f"z_g_{j}" for j in range(n)]
        mpc_log_header += [f"z_{k}_{j}" for k in range(self.N + 1) for j in range(n)]
        mpc_log_header += [f"v_{k}_{j}" for k in range(self.N) for j in range(m)]
        mpc_log_header += ["success"]
        self._mpc_log_file = open(os.path.join(self.log_dir, "mpc_log.csv"), "w")
        self._mpc_log_writer = csv.writer(self._mpc_log_file)
        self._mpc_log_writer.writerow(mpc_log_header)

        # Handle joystick for transitioning modes
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("btn_idle", 0)     # A
        self.declare_parameter("btn_zeroing", 1)  # B
        self.declare_parameter("btn_run", 3)      # Y (X=2, Y=3 on common layouts)

        # World->odom joystick control parameters
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("joy_translation_scale", 0.5)  # m/s per unit
        self.declare_parameter("joy_yaw_scale", 0.5)  # rad/s per unit

        # Zeroing parameters
        self.declare_parameter("imu_topic", "/livox/imu")
        self.declare_parameter("zeroing_position_threshold", 0.05)
        self.declare_parameter("zeroing_stable_duration", 2.0)
        self.declare_parameter("zeroing_check_period", 0.1)
        self.declare_parameter("zeroing_timeout", 30.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("camera_init_frame", "camera_init")
        self.declare_parameter("logging", False)
        self.log = self.get_parameter("logging").get_parameter_value().bool_value

        self.mode = Mode.IDLE
        self.requested_mode = Mode.IDLE
        self.is_zeroed = False

        # Zeroing state machine
        self.zeroing_state: ZeroingState = ZeroingState.IDLE
        self.zeroing_start_time: Optional[float] = None
        self.zeroing_last_position: Optional[np.ndarray] = None
        self.zeroing_timer = None
        self.zeroing_timeout_start: float = 0.0

        # Message storage
        self.odom_msg: Optional[Odometry] = None
        self.imu_msg: Optional[Imu] = None

        # Transform (camera_init -> odom)
        self.R_odom_camera_init: Optional[Rotation] = None
        self.t_odom_camera_init: Optional[np.ndarray] = None

        # World->odom SE(2) transform (joystick-controlled, identity at startup)
        self.world_odom_x: float = 0.0
        self.world_odom_y: float = 0.0
        self.world_odom_yaw: float = 0.0
        self._last_joy_time: Optional[float] = None

        # Logging rate limiting
        self._last_no_xhat_log_time: float = 0.0

        self._prev_buttons = None

        joy_topic = self.get_parameter("joy_topic").get_parameter_value().string_value
        self.create_subscription(Joy, joy_topic, self.on_joy, 10)

        # # IMU subscription for gravity alignment
        # imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        # self.create_subscription(Imu, imu_topic, self.imu_callback, 10)

        # TF broadcaster for odom->camera_init transform
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Static TF broadcaster for world->odom transform (can be updated by re-publishing)
        self.tf_world_odom_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Publish obstacles
        self.declare_parameter("obst_topic", "/mpc_obstacles")
        obs_topic = self.get_parameter("obst_topic").get_parameter_value().string_value
        self.obs_pub = self.create_publisher(MarkerArray, obs_topic, 10)
        self.declare_parameter("obst_period", 2.0)

        self.obs_timer = self.create_timer(
            self.get_parameter("obst_period").get_parameter_value().double_value,
            self.publish_obstacles
        )

        # Logging
        if self.log:
            self.get_logger().info(f"Logging enabled. Decimation: {self.log_decimation}.")

            # Create log directory relative to $LEGGED_LOCOMOTION_RL_ROOT
            root = os.environ.get("LEGGED_LOCOMOTION_RL_ROOT", "")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(root, "mpc", "ctrl_logs", timestamp)
            os.makedirs(log_dir, exist_ok=True)

            self.log_file = os.path.join(log_dir, "g1_mpc.csv")
            self.file = open(self.log_file, "w", buffering=8192)
            self.writer = csv.writer(self.file)
            # self.writer.writerow(["time"] + obs_header + act_header) # TODO: 
            self._lines_written = 0
            self._file_index = 0

            # Copy the config into the log directory
            self.declare_parameter("config_name", "")
            config_name = self.get_parameter("config_name").get_parameter_value().string_value
            if config_name != "":
                self.get_logger().info(f"Copying the config to the log directory ({config_name})...")
                config_path = os.path.join(root, "g1_control", "config", config_name)
                shutil.copy2(config_path, log_dir)

        self.received_xhat = False

        self.joystick_control = True
        self.joystick_exited = self.get_clock().now().nanoseconds / 1e9
        # TODO: exit joystick properly

        mpc_path_topic = self.get_parameter("plan_topic").get_parameter_value().string_value
        self.plan_pub = self.create_publisher(Path, mpc_path_topic, 10)
        path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.declare_parameter("path_length", 20)
        path_length = self.get_parameter("path_length").get_parameter_value().integer_value
        self.robot_path = np.zeros((path_length, 3))
        self.path_idx = 0

        # Publish initial identity world->odom transform
        self._publish_world_odom_transform()

    def on_joy(self, msg: Joy):
        if self._prev_buttons is None:
            self._prev_buttons = list(msg.buttons)
            return

        def rising(idx: int) -> bool:
            return msg.buttons[idx] == 1 and self._prev_buttons[idx] == 0

        btn_idle = self.get_parameter("btn_idle").get_parameter_value().integer_value
        btn_zero = self.get_parameter("btn_zeroing").get_parameter_value().integer_value
        btn_run = self.get_parameter("btn_run").get_parameter_value().integer_value

        if rising(btn_idle):
            self.requested_mode = Mode.IDLE
        elif rising(btn_zero):
            self.requested_mode = Mode.ZEROING
        elif rising(btn_run):
            self.requested_mode = Mode.RUNNING

        # Update world->odom transform from joystick axes
        # self.update_world_odom_from_joystick(msg)

        self._prev_buttons = list(msg.buttons)

    def update_world_odom_from_joystick(self, msg: Joy) -> None:
        """Update world->odom transform based on joystick axes."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self._last_joy_time is None:
            self._last_joy_time = current_time
            return

        dt = current_time - self._last_joy_time
        self._last_joy_time = current_time

        trans_scale = self.get_parameter("joy_translation_scale").get_parameter_value().double_value
        yaw_scale = self.get_parameter("joy_yaw_scale").get_parameter_value().double_value

        # axes[1] inverted: up=-1, we want up=+x
        # axes[0]: right=+1, we want right=-y
        delta_x = -msg.axes[1] * trans_scale * dt
        delta_y = -msg.axes[0] * trans_scale * dt
        delta_yaw = -msg.axes[3] * yaw_scale * dt  # negated for intuitive direction

        # Check if transform changed
        transform_changed = (delta_x != 0.0 or delta_y != 0.0 or delta_yaw != 0.0)

        self.world_odom_x += delta_x
        self.world_odom_y += delta_y

        # For yaw: rotate about world origin (not odom origin)
        # This requires rotating the translation vector when yaw changes
        if delta_yaw != 0.0:
            c = np.cos(delta_yaw)
            s = np.sin(delta_yaw)
            new_x = c * self.world_odom_x - s * self.world_odom_y
            new_y = s * self.world_odom_x + c * self.world_odom_y
            self.world_odom_x = new_x
            self.world_odom_y = new_y
            self.world_odom_yaw += delta_yaw

        # Normalize yaw to [-pi, pi]
        self.world_odom_yaw = np.arctan2(np.sin(self.world_odom_yaw), np.cos(self.world_odom_yaw))

        if transform_changed:
            self._publish_world_odom_transform()
            self.publish_obstacles()

    def _publish_world_odom_transform(self) -> None:
        """Publish world->odom static transform to TF (can be updated by re-publishing)."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter("world_frame").get_parameter_value().string_value
        t.child_frame_id = self.get_parameter("odom_frame").get_parameter_value().string_value

        t.transform.translation.x = self.world_odom_x
        t.transform.translation.y = self.world_odom_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = np.sin(self.world_odom_yaw / 2)
        t.transform.rotation.w = np.cos(self.world_odom_yaw / 2)

        self.tf_world_odom_broadcaster.sendTransform(t)

    def transform_obs_goal_to_odom(self) -> tuple[dict, np.ndarray]:
        """Transform obstacles and goal from world frame to odom frame.

        For ellipses, the yaw must be adjusted when rotating to the odom frame.
        If world->odom has rotation -world_odom_yaw, then ellipse orientations
        in odom frame are: ellipse_yaw_odom = ellipse_yaw_world - world_odom_yaw
        """
        cos_yaw = np.cos(-self.world_odom_yaw)
        sin_yaw = np.sin(-self.world_odom_yaw)

        # Transform obstacle centers
        cx_shifted = self.obs_world['cx'] - self.world_odom_x
        cy_shifted = self.obs_world['cy'] - self.world_odom_y
        obs_odom = {
            'cx': cos_yaw * cx_shifted - sin_yaw * cy_shifted,
            'cy': sin_yaw * cx_shifted + cos_yaw * cy_shifted,
            'rx': self.obs_world['rx'],  # semi-axes unchanged
            'ry': self.obs_world['ry'],
            'yaw': self.obs_world['yaw'] - self.world_odom_yaw  # rotate ellipse orientation
        }

        # Transform goal position and yaw
        g_shifted = self.goal_world[:2] - np.array([self.world_odom_x, self.world_odom_y])
        goal_odom = np.array([
            cos_yaw * g_shifted[0] - sin_yaw * g_shifted[1],
            sin_yaw * g_shifted[0] + cos_yaw * g_shifted[1],
            self.goal_world[2] - self.world_odom_yaw
        ])

        return obs_odom, goal_odom

    def start_zeroing(self) -> None:
        """Begin the non-blocking zeroing process."""
        self.get_logger().info("ZEROING: Starting calibration sequence...")

        # Reset zeroing state
        self.zeroing_state = ZeroingState.WAITING_STOP
        self.zeroing_start_time = None
        self.zeroing_last_position = None
        self.zeroing_timeout_start = self.get_clock().now().nanoseconds / 1e9
        self._gravity_samples: list[np.ndarray] = []

        # Create timer for periodic stability checks
        check_period = self.get_parameter("zeroing_check_period").get_parameter_value().double_value
        self.zeroing_timer = self.create_timer(check_period, self.zeroing_tick)

    def zeroing_tick(self) -> None:
        """Non-blocking zeroing state machine tick. Called by timer."""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check for timeout
        timeout = self.get_parameter("zeroing_timeout").get_parameter_value().double_value
        if current_time - self.zeroing_timeout_start > timeout:
            self.get_logger().error("ZEROING: Timeout! Aborting.")
            self.finish_zeroing(success=False)
            return

        # Publish zero velocity to keep robot stopped
        self.publish_zero_vel()

        if self.zeroing_state == ZeroingState.WAITING_STOP:
            self._zeroing_check_stability()
        elif self.zeroing_state == ZeroingState.COMPUTING:
            self._zeroing_compute_transform()

    def _zeroing_check_stability(self) -> None:
        """Check if robot has been stationary long enough."""
        if self.odom_msg is None:
            self.get_logger().info("ZEROING: Waiting for odometry...")
            return

        pos = self.odom_msg.pose.pose.position
        current_position = np.array([pos.x, pos.y, pos.z])

        threshold = self.get_parameter("zeroing_position_threshold").get_parameter_value().double_value
        stable_duration = self.get_parameter("zeroing_stable_duration").get_parameter_value().double_value
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.zeroing_last_position is None:
            # First position reading
            self.zeroing_last_position = current_position
            self.zeroing_start_time = current_time
            self.get_logger().info("ZEROING: Got first position, starting stability timer...")
            return

        # Check movement
        movement = np.linalg.norm(current_position - self.zeroing_last_position)

        if movement > threshold:
            # Robot moved - reset timer and gravity samples
            self.get_logger().info(f"ZEROING: Robot moved {movement:.3f}m, resetting stability timer...")
            self.zeroing_last_position = current_position
            self.zeroing_start_time = current_time
            self._gravity_samples = []
            return

        # Collect gravity samples during the second half of the stable period
        elapsed = current_time - self.zeroing_start_time
        if elapsed >= stable_duration / 2 and self.imu_msg is not None:
            accel = self.imu_msg.linear_acceleration
            self._gravity_samples.append(np.array([accel.x, accel.y, accel.z]))

        # Check if stable long enough
        if elapsed >= stable_duration:
            self.get_logger().info(f"ZEROING: Robot stable for {elapsed:.1f}s ({len(self._gravity_samples)} gravity samples). Computing transform...")
            self.zeroing_state = ZeroingState.COMPUTING

    def _zeroing_compute_transform(self) -> None:
        """Compute gravity-aligned, yaw-zeroed, position-zeroed transform."""
        if self.odom_msg is None:
            self.get_logger().warn("ZEROING: No odometry available!")
            return

        if self.imu_msg is None:
            self.get_logger().warn("ZEROING: No IMU available! Zeroing failed.")
            self.finish_zeroing(success=False)
        else:
            oR_b = self._compute_gravity_alignment()  # Compute the transform from aligns the gravity vector with the z axis

        # Get current pose
        cp_b = self.odom_msg.pose.pose.position       # position of the body frame in the camera_init frame
        cori_b = self.odom_msg.pose.pose.orientation  # orientation of the body frame in the camera_init frame

        cp_b = np.array([cp_b.x, cp_b.y, cp_b.z])
        cquat_b = np.array([cori_b.x, cori_b.y, cori_b.z, cori_b.w])  # scipy format [x,y,z,w]
        cR_b = Rotation.from_quat(cquat_b)            # Rotation matrix describing orientation of body frame in the camera_init frame

        # Apply gravity alignment first to get gravity-aligned orientation
        oR_c = oR_b * cR_b.inv()

        # Compute rotation to zero yaw (in gravity-aligned frame)
        heading = oR_b.apply(self.body_heading)
        initial_yaw = np.atan2(heading[1], heading[0])
        R_zero_yaw = Rotation.from_euler('z', -initial_yaw)

        # Combined rotation: first apply gravity alignment, then yaw zeroing
        self.R_odom_camera_init = R_zero_yaw * oR_c

        # Translation: move initial position to origin in xy
        # p_odom = R @ p_camera_init + t
        # We want p_odom = [0, 0, z_offset] when p_camera_init = initial_position
        op_b = self.R_odom_camera_init.apply(cp_b)
        self.t_odom_camera_init = np.array([
            -op_b[0],
            -op_b[1],
            0.0  # Keep z offset from gravity-aligned frame
        ])

        self.get_logger().info(f"ZEROING: Transform computed. Initial yaw was {np.degrees(initial_yaw):.1f} deg")

        # Publish static transform
        self._publish_odom_camera_init_transform()

        self.finish_zeroing(success=True)

    def _compute_gravity_alignment(self) -> Rotation:
        """Compute rotation to align z-axis with up (opposite gravity).

        When robot is stationary, accelerometer reads reaction to gravity,
        which points UP (opposite to gravity). We compute rotation R such that
        R @ accel_reading = [0, 0, +1] (i.e., z-axis points up).

        Uses averaged gravity samples collected during the second half of the
        zeroing stable period. Falls back to latest IMU if no samples available.

        Returns:
            Rotation that aligns sensor frame z-axis with world up.
        """
        if len(self._gravity_samples) > 0:
            accel_sensor = np.mean(self._gravity_samples, axis=0)
            self.get_logger().info(f"ZEROING: Using averaged gravity from {len(self._gravity_samples)} samples.")
        else:
            accel = self.imu_msg.linear_acceleration
            accel_sensor = np.array([accel.x, accel.y, accel.z])
            self.get_logger().warn("ZEROING: No gravity samples collected, using latest IMU reading.")
        accel_mag = np.linalg.norm(accel_sensor)

        if accel_mag < 0.1:
            self.get_logger().warn("ZEROING: Accelerometer reading too small, using identity.")
            return Rotation.identity()

        # Normalize accelerometer reading (points UP when stationary)
        g_b = accel_sensor / accel_mag  # gravity in the body frame

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

    # def _publish_odom_camera_init_transform(self) -> None:
    #     """Publish odom -> camera_init static transform."""
    #     if self.tf_static_broadcaster is None:
    #         return

    #     t = TransformStamped()
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = self.get_parameter("odom_frame").get_parameter_value().string_value
    #     t.child_frame_id = self.get_parameter("camera_init_frame").get_parameter_value().string_value

    #     quat = self.R_odom_camera_init.as_quat()
    #     t.transform.translation.x = self.t_odom_camera_init[0]
    #     t.transform.translation.y = self.t_odom_camera_init[1]
    #     t.transform.translation.z = self.t_odom_camera_init[2]
    #     t.transform.rotation.x = quat[0]
    #     t.transform.rotation.y = quat[1]
    #     t.transform.rotation.z = quat[2]
    #     t.transform.rotation.w = quat[3]

    #     self.tf_static_broadcaster.sendTransform(t)
    #     self.get_logger().info("ZEROING: Published static transform odom -> camera_init")

    def finish_zeroing(self, success: bool) -> None:
        """Complete the zeroing process and transition state."""
        # Cancel the timer
        if self.zeroing_timer is not None:
            self.zeroing_timer.cancel()
            self.zeroing_timer = None

        self.zeroing_state = ZeroingState.IDLE

        if success:
            self.is_zeroed = True
            # Reset world->odom transform to identity
            self.world_odom_x = 0.0
            self.world_odom_y = 0.0
            self.world_odom_yaw = 0.0
            self._publish_world_odom_transform()
            self.get_logger().info("ZEROING: Complete! Transforms reset.")
        else:
            self.get_logger().warn("ZEROING: Failed. Transform not updated.")

        # Transition back to IDLE
        self.mode = Mode.IDLE
        self.requested_mode = Mode.IDLE
        self.publish_zero_vel()
        self.get_logger().info("MPC mode: ZEROING -> IDLE (auto)")

    def update_x_hat(self, odom_msg: Odometry) -> None:
        """Update the state estimate from Odometry.

        Converts pose from camera_init frame to odom frame using stored transform.

        Parameters:
            odom_msg: The nav_msgs/Odometry message containing pose in camera_init frame.
        """
        # # Store message for zeroing state machine
        # self.odom_msg = odom_msg
        # self.received_xhat = True

        # # Extract position and orientation from Odometry
        # pos = odom_msg.pose.pose.position
        # ori = odom_msg.pose.pose.orientation

        # position_camera_init = np.array([pos.x, pos.y, pos.z])
        # quat_camera_init = np.array([ori.x, ori.y, ori.z, ori.w])  # scipy format [x,y,z,w]
        # R_camera_init = Rotation.from_quat(quat_camera_init)

        # # Apply transform if zeroing has been done
        # if self.R_odom_camera_init is not None:
        #     # Transform position: p_odom = R @ p_camera_init + t
        #     position_odom = self.R_odom_camera_init.apply(position_camera_init) + self.t_odom_camera_init

        #     # Transform orientation: R_odom = R_transform @ R_camera_init
        #     R_odom = self.R_odom_camera_init * R_camera_init

        #     # Extract yaw from transformed orientation
        #     heading = R_odom.apply(self.body_heading)
        #     yaw = np.atan2(heading[1], heading[0])

        #     self.start = np.array([position_odom[0], position_odom[1], yaw])
        #     self.z0 = position_odom[2]
        #     self.frame_id = self.get_parameter("odom_frame").get_parameter_value().string_value

        self.odom_msg = odom_msg
        self.received_xhat = True

        ori = odom_msg.pose.pose.orientation
        R_body = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w])
        heading = R_body.apply(self.body_heading)
        yaw = np.atan2(heading[1], heading[0])
        self.start = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])
        self.frame_id = odom_msg.header.frame_id
        self.z0 = odom_msg.pose.pose.position.z

    def compute_control(self) -> Twist:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Generate input to RL model
        if not self.received_xhat:
            self.get_logger().info("MPC has not received odometry.", throttle_duration_sec=5.0)
            return self.publish_zero_vel()

        self.update_mode_transitions()

        if self.mode == Mode.IDLE:
            return self.publish_zero_vel()

        if self.mode == Mode.ZEROING:
            return self.publish_zero_vel()

        if self.mode == Mode.RUNNING:
            self.robot_path[self.path_idx, :2] = self.start[:2]
            self.robot_path[self.path_idx, -1] = self.z0

            # Transform obstacles/goal from world to odom frame for MPC
            obs_odom, goal_odom = self.transform_obs_goal_to_odom()

            # Check if within goal deadzone
            goal_dist = np.linalg.norm(self.start[:2] - goal_odom[:2])
            if goal_dist < self.goal_deadzone:
                return self.publish_zero_vel()

            z_sol, v_sol = extract_solution(self.sol, self.N, self.planning_model.n, self.planning_model.m)
            v_prev = self.v_prev if self.dv_max is not None else None
            params = init_params(self.start, goal_odom, obs_odom, v_prev=v_prev)
            x_init = init_decision_var(z_sol, v_sol)

            self.sol = self.solver["solver"](
                x0=x_init,
                p=params,
                lbg=self.solver["lbg"],
                ubg=self.solver["ubg"],
                lbx=self.solver["lbx"],
                ubx=self.solver["ubx"]
            )

            # Check convergence
            stats = self.solver["solver"].stats()
            if not stats["success"]:
                self.get_logger().warn(f"MPC solve did not converge: {stats['return_status']}", throttle_duration_sec=1.0)

            z_sol, v_sol = extract_solution(self.sol, self.N, self.planning_model.n, self.planning_model.m)

            # Update v_prev for next solve's rate limit
            self.v_prev = v_sol[0, :]

            # self.get_logger().info(f"MPC Solve: \n\tIC: {self.start}\n\tTraj: {z_sol}")

            # setting the message
            vel_cmd = Twist()
            # vel_cmd.header.stamp = self.get_clock().now().to_msg()
            vel_cmd.linear.x = v_sol[0, 0]
            vel_cmd.linear.y = v_sol[0, 1]
            vel_cmd.angular.z = v_sol[0, 2]
            self.obk_publishers["pub_ctrl"].publish(vel_cmd)

            plan_msg = Path()
            plan_msg.header.frame_id = self.frame_id
            plan_msg.header.stamp = self.get_clock().now().to_msg()
            for i in range(self.N + 1):
                pose = PoseStamped()
                pose.header = plan_msg.header
                pose.pose.position.x = z_sol[i, 0]
                pose.pose.position.y = z_sol[i, 1]
                pose.pose.position.z = self.z0
                plan_msg.poses.append(pose)
            self.plan_pub.publish(plan_msg)

            path_msg = Path()
            path_msg.header.frame_id = self.frame_id
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for i in range(self.robot_path.shape[0]):
                pose = PoseStamped()
                pose.header = path_msg.header
                p = self.robot_path[(self.path_idx - i) % self.robot_path.shape[0], :]
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
            self.path_idx = (self.path_idx + 1) % self.robot_path.shape[0]

            # Publish robot pose arrow (blue) - shows current x, y, yaw in odom frame
            robot_marker = Marker()
            robot_marker.header.frame_id = self.frame_id
            robot_marker.header.stamp = self.get_clock().now().to_msg()
            robot_marker.ns = "mpc_robot"
            robot_marker.id = 0
            robot_marker.type = Marker.ARROW
            robot_marker.action = Marker.ADD
            robot_marker.pose.position.x = float(self.start[0])
            robot_marker.pose.position.y = float(self.start[1])
            robot_marker.pose.position.z = float(self.z0)
            robot_yaw = float(self.start[2])
            robot_marker.pose.orientation.x = 0.0
            robot_marker.pose.orientation.y = 0.0
            robot_marker.pose.orientation.z = float(np.sin(robot_yaw / 2))
            robot_marker.pose.orientation.w = float(np.cos(robot_yaw / 2))
            robot_marker.scale.x = 0.5
            robot_marker.scale.y = 0.1
            robot_marker.scale.z = 0.1
            robot_marker.color.r = 0.2
            robot_marker.color.g = 0.2
            robot_marker.color.b = 1.0
            robot_marker.color.a = 0.8
            robot_marker.lifetime.sec = 0
            robot_marker.lifetime.nanosec = 0
            self.obs_pub.publish(MarkerArray(markers=[robot_marker]))

            # Log MPC solution
            t = self.get_clock().now().nanoseconds / 1e9
            row = [t]
            row += self.start.tolist()
            row += goal_odom.tolist()
            row += z_sol.flatten().tolist()
            row += v_sol.flatten().tolist()
            row += [int(stats["success"])]
            self._mpc_log_writer.writerow(row)

            return vel_cmd

        # Fallback to publishing zero
        return self.publish_zero_vel()

    def publish_zero_vel(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.angular.z = 0.0
        self.obk_publishers["pub_ctrl"].publish(vel_cmd)
        return vel_cmd

    def publish_obstacles(self):
        world_frame = self.get_parameter("world_frame").get_parameter_value().string_value

        ma = MarkerArray()

        # Use world-frame obstacles for visualization
        cx_arr = self.obs_world["cx"]   # shape (K,)
        cy_arr = self.obs_world["cy"]   # shape (K,)
        rx_arr = self.obs_world["rx"]   # shape (K,)
        ry_arr = self.obs_world["ry"]   # shape (K,)
        yaw_arr = self.obs_world["yaw"]  # shape (K,)

        for i in range(len(rx_arr)):
            m = Marker()
            m.header.frame_id = world_frame
            m.header.stamp = self.get_clock().now().to_msg()

            m.ns = "mpc_obstacles"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            # position: center of elliptical cylinder
            m.pose.position.x = float(cx_arr[i])
            m.pose.position.y = float(cy_arr[i])
            m.pose.position.z = float(self.z0)

            # orientation: rotate around z-axis by ellipse yaw
            yaw = float(yaw_arr[i])
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = float(np.sin(yaw / 2))
            m.pose.orientation.w = float(np.cos(yaw / 2))

            # scale: x/y are DIAMETER (2 * semi-axis), z is HEIGHT
            m.scale.x = float(2.0 * rx_arr[i])
            m.scale.y = float(2.0 * ry_arr[i])
            m.scale.z = 1.0  # (z0-0.5) to (z0+0.5)

            # color (RGBA)
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
            m.color.a = 0.6

            # optional: keep alive briefly
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0

            ma.markers.append(m)

        # Add goal arrow (green) - shows position and yaw
        goal_marker = Marker()
        goal_marker.header.frame_id = world_frame
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "mpc_goal"
        goal_marker.id = 0
        goal_marker.type = Marker.ARROW
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(self.goal_world[0])
        goal_marker.pose.position.y = float(self.goal_world[1])
        goal_marker.pose.position.z = float(self.z0)
        # Convert yaw to quaternion (rotation around z-axis)
        goal_yaw = float(self.goal_world[2])
        goal_marker.pose.orientation.x = 0.0
        goal_marker.pose.orientation.y = 0.0
        goal_marker.pose.orientation.z = float(np.sin(goal_yaw / 2))
        goal_marker.pose.orientation.w = float(np.cos(goal_yaw / 2))
        goal_marker.scale.x = 0.5  # arrow length
        goal_marker.scale.y = 0.1  # arrow width
        goal_marker.scale.z = 0.1  # arrow height
        goal_marker.color.r = 0.2
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.2
        goal_marker.color.a = 0.8
        goal_marker.lifetime.sec = 0
        goal_marker.lifetime.nanosec = 0
        ma.markers.append(goal_marker)

        self.obs_pub.publish(ma)

    def update_mode_transitions(self):
        # joystick requested transitions
        if self.requested_mode == self.mode:
            return

        # if self.requested_mode == Mode.RUNNING and not self.is_zeroed:
        #     self.get_logger().warn("[MPC NODE]: RUN requested but not zeroed -> must ZERO first.")
        #     self.requested_mode = self.mode  # Clear invalid request
        #     return

        self.transition_to(self.requested_mode)

    def transition_to(self, new_mode: Mode):
        if new_mode == self.mode:
            return

        old = self.mode

        # zero on exit/enter of ZEROING
        if old == Mode.ZEROING or new_mode == Mode.ZEROING or new_mode == Mode.IDLE:
            self.publish_zero_vel()

        self.get_logger().info(f"MPC mode: {old.name} -> {new_mode.name}")
        self.mode = new_mode

        # Non-blocking: if we enter ZEROING, start the async process
        if self.mode == Mode.ZEROING:
            self.publish_zero_vel()
            self.start_zeroing()
            # Stay in ZEROING until finish_zeroing() is called by the state machine


def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, MPCController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()