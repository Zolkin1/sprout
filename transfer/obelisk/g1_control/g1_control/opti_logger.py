from pathlib import Path
import csv
import os
from datetime import datetime

from geometry_msgs.msg import PoseStamped

from obelisk_py.core.sensing import ObeliskSensor
from obelisk_py.core.utils.ros import spin_obelisk
from rclpy.executors import SingleThreadedExecutor

class OptiLogger(ObeliskSensor):
    """Logger for optitrack data."""

    def __init__(self, node_name: str = "optitrack_logger"):
        super().__init__(node_name)
        self._has_sensor_publisher = True   # Hack to get around the Obelisk check.

        self.register_obk_subscription(
            "sub_optitrack_pose",
            self.optitrack_pose_callback,  # type: ignore
            PoseStamped,
            key="sub_optitrack_pose",  # key can be specified here or in the config file
        )

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
        timestamp_sec = msg.header.stamp.sec
        pos = msg.pose.position
        ori = msg.pose.orientation
        
        self.writer.writerow([
            timestamp_sec,
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w
        ])

        self.get_logger().info(f'Logged Pose at {timestamp_sec}.')

    # def compute_state_estimate(self):
    #     pass

def main(args: list | None = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, OptiLogger, SingleThreadedExecutor)


if __name__ == "__main__":
    main()