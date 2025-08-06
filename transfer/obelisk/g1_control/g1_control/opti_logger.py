from pathlib import Path
import csv
import os
from datetime import datetime

from geometry_msgs.msg import PoseStamped

from obelisk_py.core.sensing import ObeliskSensor
from obelisk_py.core.utils.ros import spin_obelisk
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

class OptiLogger(ObeliskSensor):
    """Logger for optitrack data."""

    def __init__(self, node_name: str = "optitrack_logger"):
        super().__init__(node_name)
        self._has_sensor_publisher = True   # Hack to get around the Obelisk check.

        # self.register_obk_subscription(
        #     "sub_optitrack_pose",
        #     self.optitrack_pose_callback,  # type: ignore
        #     PoseStamped,
        #     key="sub_optitrack_pose",  # key can be specified here or in the config file
        # )

        # Subscription to OptiTrack mocap data
        self.subscription = self.create_subscription(
            PoseStamped,
            '/G1_pelvis/pose',
            self.optitrack_pose_callback,
            qos_profile_sensor_data)

        self.start_time = None
        self.first_rec = False

        root = os.environ.get("ROBOT_RL_ROOT", "")

        # Make logging directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(root, "optitrack_logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, "g1_control.csv")
        self.file = open(self.log_file, "w", buffering=8192)
        self.writer = csv.writer(self.file)

    def optitrack_pose_callback(self, msg: PoseStamped) -> None:
        """Callback for the optitrack pose. Log the pose data."""
        if not self.first_rec:
            self.start_time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
            self.first_rec = not self.first_rec
            
        stamp_time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        time_offset = stamp_time - self.start_time
        time_float = time_offset.nanoseconds / 1e9
        pos = msg.pose.position
        ori = msg.pose.orientation
        
        self.writer.writerow([
            time_float,
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w
        ])

        # self.get_logger().info(f'Logged Pose at {time_float}.')

    # def compute_state_estimate(self):
    #     pass

def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, OptiLogger, SingleThreadedExecutor)


if __name__ == "__main__":
    main()