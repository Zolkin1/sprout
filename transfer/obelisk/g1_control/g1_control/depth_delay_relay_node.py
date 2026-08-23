# Copyright 2026 Zachary Olkin. All rights reserved.

"""Republish depth frames after a fixed delay, header stamp untouched. A TEST tool.

The generator unprojects each depth image with the camera's pose, and that is only correct
when pose and pixels describe the same instant. On hardware the ZED pipeline puts ~50 ms
between capture and the frame reaching the network, so the controller matches the pose to
the frame's capture stamp rather than to the current tick (see ``PoseSample`` in
``g1_controller.h``). MuJoCo has no such latency -- it casts depth synchronously from the
same state that produces the pose -- so in simulation the matching code is exercised but
never actually *tested*: every lookup resolves to ~now and any bug in it is invisible.

This node manufactures the missing latency. It subscribes to the world's depth topics and
republishes each frame ``delay_sec`` later **with its header stamp unchanged**, which is
exactly the shape of the real fault: the stamp says T, the message arrives at T + delay.
Point the controller's ``depth_topics`` at the delayed topics and the matching has something
to correct; turn the matching off (``depth_pose_match:=false``) and the terrain should visibly
swing when the torso pitches.

Run it SINGLE BOX first, where the world and the brain share a clock and the only variable is
the delay. Then run it across the two-box split, where the clocks differ by milliseconds, and
the difference between the two is what the clock skew is costing.

Deliberately a plain rclpy node launched with ``ros2 run``, for the same reason
``fsm_state_echo`` is: as an obelisk entry it would join a config's lifecycle and node-name
space, and it has no obelisk-managed pubs/subs.

Never launched by the hardware path, and it must not be: it exists to make the sim worse in
one specific, measurable way.
"""

from __future__ import annotations

import collections

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

#: How often the release queue is checked, seconds. An order below a 30 Hz frame interval, so
#: the delay this node adds is the requested one rather than the tick it is quantised to.
_POLL_SEC = 0.002


class DepthDelayRelay(Node):
    """One delayed republisher per depth topic."""

    def __init__(self) -> None:
        """Declare the topic mapping and the delay, and wire one relay per pair."""
        super().__init__("depth_delay_relay")
        self.declare_parameter("in_topics", [""])
        self.declare_parameter("out_topics", [""])
        self.declare_parameter("delay_sec", 0.05)

        in_topics = [t for t in self.get_parameter("in_topics").value if t]
        out_topics = [t for t in self.get_parameter("out_topics").value if t]
        self._delay = float(self.get_parameter("delay_sec").value)
        if not in_topics or len(in_topics) != len(out_topics):
            raise RuntimeError(
                f"in_topics {in_topics} and out_topics {out_topics} must be non-empty and the "
                "same length: this node relays a 1:1 mapping."
            )

        # (release_time, publisher_index, msg), in arrival order. Arrival order IS release
        # order because the delay is constant, so a deque and a peek at the front suffice.
        self._queue: collections.deque = collections.deque()
        # SENSOR-DATA QoS on both sides, which is best-effort. Not a detail: the world
        # publishes these best-effort and the controller subscribes with SensorDataQoS, and a
        # RELIABLE subscriber cannot be served by a best-effort publisher -- so the default QoS
        # here silently receives nothing at all. The symptom is not an error but a controller
        # that sits forever "waiting for the first frame", which looks like the world failing to
        # start rather than like a QoS mismatch in the relay.
        self._pubs = [self.create_publisher(Image, t, qos_profile_sensor_data)
                      for t in out_topics]
        self._subs = [
            self.create_subscription(
                Image, t, (lambda m, i=i: self._on_image(m, i)), qos_profile_sensor_data)
            for i, t in enumerate(in_topics)
        ]
        self.create_timer(_POLL_SEC, self._drain)
        self.get_logger().info(
            f"delaying {self._delay * 1e3:.0f} ms: " +
            ", ".join(f"{a} -> {b}" for a, b in zip(in_topics, out_topics)) +
            " (header stamps are NOT rewritten)"
        )

    def _on_image(self, msg: Image, index: int) -> None:
        """Queue one frame for release `delay_sec` from now.

        Args:
            msg: The frame, forwarded byte for byte. Its header -- and therefore its capture
                stamp -- is deliberately left alone; rewriting it would erase the very thing
                the controller is being asked to use.
            index: Which topic pair this frame arrived on.
        """
        self._queue.append((self.get_clock().now().nanoseconds + int(self._delay * 1e9),
                            index, msg))

    def _drain(self) -> None:
        """Publish every queued frame whose release time has passed."""
        now = self.get_clock().now().nanoseconds
        while self._queue and self._queue[0][0] <= now:
            _, index, msg = self._queue.popleft()
            self._pubs[index].publish(msg)


def main() -> None:
    """CLI entry point."""
    rclpy.init()
    node = DepthDelayRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
