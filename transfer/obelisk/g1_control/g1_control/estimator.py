from typing import List, Optional

import numpy as np

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu

from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_py.core.estimation import ObeliskEstimator
from obelisk_py.core.utils.ros import spin_obelisk
from obelisk_sensor_msgs.msg import ObkJointEncoders
from rclpy.executors import SingleThreadedExecutor


class G1Estimator(ObeliskEstimator):
    """G1 Rl estimator."""

    def __init__(self, node_name: str = "g1_estimator") -> None:
        """Initialize the G1 estimator."""
        super().__init__(node_name, EstimatedState)

        self.register_obk_subscription(
            "sub_joint_encoders",
            self.joint_encoders_callback,  # type: ignore
            ObkJointEncoders,
            key="sub_joint_encoders",  # key can be specified here or in the config file
        )

        self.register_obk_subscription(
            "sub_pelvis_imu",
            self.pelvis_imu_callback,  # type: ignore
            Imu,
            key="sub_pelivs_imu",  # key can be specified here or in the config file
        )

        self.declare_parameter("opti_track_pos", False)
        self.use_opti_track_pos = self.get_parameter("opti_track_pos").get_parameter_value().bool_value
        if self.use_opti_track_pos:
            self.register_obk_subscription(
                "sub_opti_track_pose",
                self.optitrack_pose_callback,  # type: ignore
                PoseStamped,
                key="sub_opti_track_pose",  # key can be specified here or in the config file
            )

        self.base_pos = np.ones(3)  # np.zeros(3)
        self.base_quat = np.zeros(4)

        self.base_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)

        self.received_joint_encoders = False
        self.received_imu = False

    def joint_encoders_callback(self, msg: ObkJointEncoders) -> None:
        """Callback for the joint encoders."""
        self.joint_angles = msg.joint_pos
        self.joint_vels = msg.joint_vel
        self.joint_names = msg.joint_names

        self.received_joint_encoders = True

    def pelvis_imu_callback(self, msg: Imu) -> None:
        """Callback for the pelvis IMU."""
        self.base_quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

        self.base_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        self.received_imu = True

    def compute_state_estimate(self):
        estimated_state = EstimatedState()

        if self.received_imu and self.received_joint_encoders:
            estimated_state.q_joints = self.joint_angles
            estimated_state.v_joints = self.joint_vels
            estimated_state.joint_names = self.joint_names

            estimated_state.q_base = np.concatenate([self.base_pos, self.base_quat]).tolist()
            estimated_state.v_base = np.concatenate([self.base_vel, self.base_ang_vel]).tolist()

            estimated_state.header.stamp = self.get_clock().now().to_msg()

            self.obk_publishers["pub_est"].publish(estimated_state)

            return estimated_state

    def optitrack_pose_callback(self, msg: PoseStamped) -> None:
        """Callback for the optitrack pose."""
        self.base_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    

def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, G1Estimator, SingleThreadedExecutor)


if __name__ == "__main__":
    main()
