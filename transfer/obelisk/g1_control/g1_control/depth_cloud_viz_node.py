# Copyright 2026 Zachary Olkin. All rights reserved.

"""Unproject the depth images into a PointCloud2 so RViz can draw what the policy sees.

The plots answer "was depth healthy" with a rate and an invalid fraction. Those two numbers
cannot tell you the camera is pointed somewhere useless, or that the obstacle in front of
the robot never entered the frame -- and both of those are ordinary failures. This draws
the pixels in space, next to the robot they were measured from.

Sim and hardware alike, and identically: the topics are the same
(``/depth/<cam>/image_processed``), and the geometry comes from ``tracker.onnx``, which is
the single producer of the camera contract for every consumer off-sim. So nothing here
restates an intrinsic, a mount pose or a range, and a camera that is mis-drawn is a camera
the POLICY also has wrong.

WHY NOT the MuJoCo viewer's ray drawing. That is sim-only, it draws the rays obelisk cast
rather than the image the controller received (so it cannot show a decode, ordering or
staleness fault between them), and it costs 780 lines per camera per frame in the viewer
of the box also running the brain. This is the replacement, so the scan configs keep
``viz: false``.

FRAME CONVENTION, the thing most likely to be wrong here. The camera contract's frame is
``x = optical axis, y = left, z = up`` -- NOT the ROS optical convention (z forward, x
right, y down). It is the convention ``transfer/sim/depth_camera.pinhole_ray_directions``
builds rays in, and the ray for pixel ``(u, v)`` is ``[1, -(u+0.5-cx)/fx,
-(v+0.5-cy)/fy]``. Its x-component is exactly 1, so a z-depth ``d`` (which is what these
images carry -- ``distance_to_image_plane``, metres along the optical axis) unprojects by
plain multiplication, with no normalisation and no ray-length correction:

    X = d,  Y = -d * (u + 0.5 - cx) / fx,  Z = -d * (v + 0.5 - cy) / fy

The half-pixel is the pixel CENTRE and is part of the convention; dropping it shifts the
whole cloud by half a pixel of angle, which at 8 m is centimetres and looks like a
calibration error.

Points are published in the camera's MOUNT BODY frame (``torso_link``) rather than a
camera frame of their own, so this needs no static transform broadcaster and no new
dependency: ``robot_state_publisher`` already publishes ``torso_link`` from the URDF and
the estimator's joint states, which ``--rviz`` starts. The mount offset from the contract
is applied here instead.

Deliberately a PLAIN rclpy node launched with ``ros2 run``, following fm_plan_viz_node and
fsm_state_echo_node: as an obelisk entry it would share the lifecycle and node-name space
of the real controller, and a cross-config lifecycle transition would then kill a live
node.

Usage (``run_sim_low.sh --rviz`` starts it for you)::

    ros2 run g1_control depth_cloud_viz
    ros2 run g1_control depth_cloud_viz --ros-args -p hf_policy_folder:=<name>
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField

# realpath, not abspath: the package is symlink-installed, so abspath would resolve
# relative to install/ and land outside the repo. Same derivation and same assert as
# controller.py and fm_plan_viz_node.py -- the assert is what makes the hop count safe.
_TRANSFER = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
)
if not os.path.isdir(os.path.join(_TRANSFER, "deploy")):
    raise RuntimeError(
        f"the shared deploy runtime is not at {_TRANSFER}/deploy (derived from "
        f"{os.path.realpath(__file__)}); the repo layout moved under this node."
    )
if _TRANSFER not in sys.path:
    sys.path.insert(0, _TRANSFER)

from deploy.hardware import DEFAULT_HF_REPO_ID, fetch_artifacts  # noqa: E402
from deploy.quat_math_np import quat_apply  # noqa: E402
from deploy.tracker_session import TrackerSession  # noqa: E402

#: Must equal ``g1_control.controller.HF_CACHE_DIR``; both are built from the same anchor.
HF_CACHE_DIR = os.path.join(_TRANSFER, "obelisk", "g1_control", "resource", "hf_cache")

#: One float32 each: xyz, plus the depth again so RViz can colour by range without the
#: viewer having to know which axis is forward.
_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]
_POINT_STEP = 16

#: Must match controller.SENSOR_QOS. The depth publisher is BEST_EFFORT, and a subscriber
#: that asks for RELIABLE would never pair with it -- silently, which is the whole failure
#: mode net_check's endpoint counts exist to expose.
_SENSOR_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def unproject(depth_m: np.ndarray, spec: dict) -> np.ndarray:
    """Turn one z-depth image into points in the CAMERA frame (x forward, y left, z up).

    Args:
        depth_m: Flat row-major image, metres along the optical axis, length ``w * h``.
        spec: One camera's entry from the contract (``fx``/``fy``/``cx``/``cy``/
            ``width``/``height``/``min_dist``/``max_dist``).

    Returns:
        ``(n, 3)`` finite, in-range points. Pixels that are NaN, non-finite or outside
        ``[min_dist, max_dist]`` are DROPPED rather than clamped -- a saturated pixel drawn
        at max range is a wall that is not there, which is worse than a hole.
    """
    w, h = int(spec["width"]), int(spec["height"])
    u = np.tile(np.arange(w, dtype=np.float64), h)
    v = np.repeat(np.arange(h, dtype=np.float64), w)
    d = np.asarray(depth_m, dtype=np.float64)

    lo, hi = float(spec.get("min_dist", 0.0)), float(spec.get("max_dist", np.inf))
    keep = np.isfinite(d) & (d >= lo) & (d <= hi)
    d, u, v = d[keep], u[keep], v[keep]

    fx, fy = float(spec["fx"]), float(spec["fy"])
    cx, cy = float(spec["cx"]), float(spec["cy"])
    return np.stack([d, -d * (u + 0.5 - cx) / fx, -d * (v + 0.5 - cy) / fy], axis=-1)


class DepthCloudViz(Node):
    """Republish each depth image as a PointCloud2 in its mount body's frame."""

    def __init__(self) -> None:
        """Resolve the camera contract, then subscribe one camera at a time."""
        super().__init__("depth_cloud_viz")
        repo = self.declare_parameter("hf_repo_id", DEFAULT_HF_REPO_ID).value
        folder = self.declare_parameter("hf_policy_folder", "").value
        topics = list(self.declare_parameter(
            "depth_topics",
            ["/depth/near/image_processed", "/depth/far/image_processed"]).value)
        self.decimation = int(self.declare_parameter("decimation", 1).value)

        if not folder:
            raise RuntimeError(
                "hf_policy_folder is empty, so the camera contract cannot be resolved. "
                "Pass -p hf_policy_folder:=<name>; it must be the SAME policy the "
                "controller is running, or this draws a different camera than the one "
                "feeding the network.")

        # Metadata only: a CPU session on tracker.onnx, never run. ~20 MB and about a
        # second, paid once at startup. Read from the artifact rather than from
        # config/scan/*.yml on purpose -- those are generated, and sim-only, and this node
        # has to behave identically on hardware where they do not exist.
        tracker, _ = fetch_artifacts(repo, folder, HF_CACHE_DIR)
        self.spec = TrackerSession(tracker, 1, ["cpu"]).get_depth_camera_spec() or []
        if not self.spec:
            raise RuntimeError(f"{tracker} carries no depth_camera_spec; nothing to draw.")
        if len(self.spec) != len(topics):
            raise RuntimeError(
                f"the policy reads {len(self.spec)} camera(s) but depth_topics names "
                f"{len(topics)}. They pair BY ORDER, so a mismatch would draw one "
                f"camera's pixels with another's intrinsics.")

        self._count = [0] * len(self.spec)
        self.pubs, self.subs = [], []
        for i, (cam, topic) in enumerate(zip(self.spec, topics)):
            out = topic.rsplit("/", 1)[0] + "/points"
            self.pubs.append(self.create_publisher(PointCloud2, out, 1))
            self.subs.append(self.create_subscription(
                Image, topic, lambda m, k=i: self._on_image(m, k), _SENSOR_QOS))
            self.get_logger().info(
                f"depth camera {i} ('{cam.get('sensor')}') {int(cam['width'])}x"
                f"{int(cam['height'])} on {cam['mount_body']}: {topic} -> {out}")

    def _on_image(self, msg: Image, index: int) -> None:
        """Unproject one frame, move it into the mount body frame, publish.

        Args:
            msg: 32FC1 depth image, row-major, metres along the optical axis -- decoded
                exactly as ``controller.depth_callback`` does, so a decode fault shows up
                in both or neither.
            index: Which camera, in the contract's order.
        """
        self._count[index] += 1
        if self.decimation > 1 and self._count[index] % self.decimation:
            return
        cam = self.spec[index]
        flat = np.frombuffer(msg.data, dtype=np.float32)
        expected = int(cam["width"]) * int(cam["height"])
        if flat.size != expected:
            self.get_logger().warn(
                f"camera {index}: got {flat.size} pixels, contract says {expected}; "
                f"skipping. A wrong-sized frame is a different camera, not a resizable "
                f"image.", throttle_duration_sec=5.0)
            return

        pts_cam = unproject(flat, cam)
        # Camera -> mount body. The contract's offset is the camera's pose IN that body.
        q = np.asarray(cam["offset_quat_xyzw"], dtype=np.float64)
        pts = quat_apply(np.broadcast_to(q, (pts_cam.shape[0], 4)), pts_cam)
        pts += np.asarray(cam["offset_pos"], dtype=np.float64)

        self.pubs[index].publish(self._cloud(pts, pts_cam[:, 0], cam["mount_body"], msg))

    @staticmethod
    def _cloud(pts: np.ndarray, depth: np.ndarray, frame: str, src: Image) -> PointCloud2:
        """Pack points into an unordered PointCloud2.

        Args:
            pts: ``(n, 3)`` in the mount body frame.
            depth: ``(n,)`` range along the optical axis, carried as ``intensity``.
            frame: ``frame_id`` to publish in (the camera's mount body).
            src: The image this came from, for its header stamp.

        Returns:
            The cloud, ``n`` points wide and one row tall.
        """
        n = int(pts.shape[0])
        buf = np.empty((n, 4), dtype=np.float32)
        buf[:, :3] = pts
        buf[:, 3] = depth

        msg = PointCloud2()
        msg.header = src.header
        msg.header.frame_id = frame
        msg.height, msg.width = 1, n
        msg.fields = _FIELDS
        msg.is_bigendian = struct.pack("=i", 1) == struct.pack(">i", 1)
        msg.point_step = _POINT_STEP
        msg.row_step = _POINT_STEP * n
        msg.data = buf.tobytes()
        msg.is_dense = True
        return msg


def main() -> None:
    """Spin the depth cloud publisher."""
    rclpy.init()
    node = DepthCloudViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
