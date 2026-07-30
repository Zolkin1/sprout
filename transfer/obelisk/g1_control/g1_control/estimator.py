# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 state estimator: joint encoders + pelvis IMU + FAST-LIO odometry -> EstimatedState.

Unlike a purely proprioceptive locomotion policy, the generator+tracker pair genuinely
needs base POSITION and WORLD-frame linear velocity: they are what the flow-matching
generator's history tokens are built from. Localisation is therefore load-bearing here,
not a convenience for a higher level, and losing it corrupts the reference the tracker is
chasing rather than merely degrading navigation.

The published contract, consumed by the controller:

    q_base = [x, y, z,            pos from odometry, WORLD frame
              qx, qy, qz, qw]     orientation from the IMU (xyzw)
    v_base = [vx, vy, vz,         linear velocity, WORLD frame (see below)
              wx, wy, wz]         angular velocity from the IMU, BODY frame

The frame of the linear term is the subtle part. ``nav_msgs/Odometry`` puts ``pose`` in
the header frame but ``twist`` in the CHILD frame, so FAST-LIO's twist is body-frame while
the policy wants world. It is rotated here by the IMU orientation
(``odom_twist_is_body_frame``, on by default). Getting this wrong is silent -- nothing
errors, every history token is just wrong -- which is why it is an explicit parameter
rather than an assumption buried in the controller.
"""

from __future__ import annotations

import numpy as np
from nav_msgs.msg import Odometry
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_py.core.estimation import ObeliskEstimator
from obelisk_py.core.utils.ros import spin_obelisk
from obelisk_sensor_msgs.msg import ObkJointEncoders
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Imu


def quat_apply(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion.

    A local copy rather than an import: this node must not depend on the torch-free deploy
    package just for one rotation, and the deploy package must not depend on ROS.

    Args:
        quat_xyzw: Unit quaternion ``(4,)`` in xyzw order.
        vec: Vector ``(3,)`` to rotate.

    Returns:
        The rotated vector ``(3,)``.
    """
    x, y, z, w = quat_xyzw
    u = np.array([x, y, z])
    return vec + 2.0 * np.cross(u, np.cross(u, vec) + w * vec)


class G1Estimator(ObeliskEstimator):
    """Fuses the encoders, the pelvis IMU and lidar odometry into one EstimatedState."""

    def __init__(self, node_name: str = "g1_estimator") -> None:
        """Register the sensor subscriptions and declare the frame-convention parameters."""
        super().__init__(node_name, EstimatedState)

        self.register_obk_subscription("sub_joint_encoders", self.joint_encoders_callback, ObkJointEncoders)
        self.register_obk_subscription("sub_pelvis_imu", self.pelvis_imu_callback, Imu)
        self.register_obk_subscription("sub_odom", self.odom_callback, Odometry)

        self.declare_parameter("odom_twist_is_body_frame", True)
        self.declare_parameter("odom_timeout", 0.5)
        self.twist_is_body = self.get_parameter("odom_twist_is_body_frame").value
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)

        self.joint_pos: list[float] = []
        self.joint_vel: list[float] = []
        self.joint_names: list[str] = []
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.base_ang_vel_b = np.zeros(3)
        self.base_pos = np.zeros(3)
        self.base_lin_vel = np.zeros(3)

        self.have_encoders = False
        self.have_imu = False
        self.last_odom_time: float | None = None
        self._warned_no_odom = False

    def joint_encoders_callback(self, msg: ObkJointEncoders) -> None:
        """Latch the joint state.

        Args:
            msg: Encoder readings, carrying their own joint names.
        """
        self.joint_pos = list(msg.joint_pos)
        self.joint_vel = list(msg.joint_vel)
        # Carried through verbatim: the controller maps these onto the policy's joint order
        # by NAME, so nothing downstream depends on the hardware's ordering.
        self.joint_names = list(msg.joint_names)
        self.have_encoders = True

    def pelvis_imu_callback(self, msg: Imu) -> None:
        """Latch base orientation and body-frame angular velocity.

        Args:
            msg: Pelvis IMU reading.
        """
        q = msg.orientation
        self.base_quat = np.array([q.x, q.y, q.z, q.w])
        w = msg.angular_velocity
        self.base_ang_vel_b = np.array([w.x, w.y, w.z])
        self.have_imu = True

    def odom_callback(self, msg: Odometry) -> None:
        """Latch base position and world-frame linear velocity from lidar odometry.

        Args:
            msg: Odometry from FAST-LIO. ``pose`` is in the header frame; ``twist`` is in
                the child frame, hence the rotation.
        """
        p = msg.pose.pose.position
        self.base_pos = np.array([p.x, p.y, p.z])
        v = msg.twist.twist.linear
        lin = np.array([v.x, v.y, v.z])
        self.base_lin_vel = quat_apply(self.base_quat, lin) if self.twist_is_body else lin
        self.last_odom_time = self.t

    def odom_is_stale(self) -> bool:
        """Whether odometry has gone quiet for longer than ``odom_timeout``.

        Returns:
            True when localisation cannot be trusted. Inert until the first message, so a
            stack brought up before FAST-LIO does not report staleness it cannot know about.
        """
        if self.odom_timeout <= 0.0 or self.last_odom_time is None:
            return False
        return (self.t - self.last_odom_time) > self.odom_timeout

    def compute_state_estimate(self) -> EstimatedState | None:
        """Publish the fused estimate, once every source has reported at least once.

        Returns:
            The published message, or ``None`` while still waiting on a sensor.
        """
        if not (self.have_imu and self.have_encoders):
            return None
        # TODO(logging-refactor): this warning is the ONLY consequence of a stale odometry.
        # The message below still publishes the last known base_pos / base_lin_vel as if
        # fresh, and EstimatedState carries no validity flag, so the controller cannot see
        # it (low_level.yml's comment claims it treats a flagged estimate as a dropout --
        # nothing implements that). Since the FM history is built from base position and
        # world velocity, a FAST-LIO dropout silently corrupts the reference the tracker
        # chases, which is the failure mode the comments call worse than stopping.
        if self.odom_is_stale() and not self._warned_no_odom:
            self.get_logger().warn(
                f"ODOMETRY DROPOUT: nothing for > {self.odom_timeout:.2f}s. The generator's "
                "history is built from base position and velocity, so its reference will "
                "drift until this recovers."
            )
            self._warned_no_odom = True
        elif not self.odom_is_stale() and self._warned_no_odom:
            self.get_logger().info("Odometry recovered.")
            self._warned_no_odom = False

        msg = EstimatedState()
        msg.q_joints = self.joint_pos
        msg.v_joints = self.joint_vel
        msg.joint_names = self.joint_names
        msg.base_link_name = "pelvis"
        msg.q_base = np.concatenate([self.base_pos, self.base_quat]).tolist()
        msg.v_base = np.concatenate([self.base_lin_vel, self.base_ang_vel_b]).tolist()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.obk_publishers["pub_est"].publish(msg)
        return msg


def main(args: list | None = None) -> None:
    """Spin the estimator."""
    spin_obelisk(args, G1Estimator, SingleThreadedExecutor)


if __name__ == "__main__":
    main()
