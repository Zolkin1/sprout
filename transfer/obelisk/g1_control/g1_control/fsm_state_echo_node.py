# Copyright 2026 Zachary Olkin. All rights reserved.

"""Echo the commanded execution FSM back as the ACTUAL state, for simulation.

On hardware the G1 interface publishes its real execution-FSM state on
``/obelisk/g1/fsm_state``, and the controller arms off that and nothing else. MuJoCo has no
such state machine, so in sim the joystick's *command* on ``/obelisk/g1/exec_fsm`` is
echoed straight back as the actual state -- the robot is assumed to have complied. That
keeps the controller's arming logic identical in sim and on hardware instead of giving it
a simulation branch, which is exactly the branch you do not want in the code path that
decides whether to drive a real robot.

Deliberately a PLAIN rclpy node launched with ``ros2 run``, not an obelisk control entry.
As an obelisk node it would join the world config's ``control:`` section and share the
lifecycle and node-name space of the brain's real controller; a cross-config lifecycle
transition then kills a live node. It has no obelisk-managed pubs/subs anyway.
"""

from __future__ import annotations

import rclpy
from obelisk_control_msgs.msg import ExecutionFSM
from rclpy.node import Node


class FsmStateEcho(Node):
    """Republishes the commanded execution FSM as the achieved one."""

    def __init__(self) -> None:
        """Wire the echo and start the heartbeat."""
        super().__init__("fsm_state_echo")
        self.declare_parameter("sub_topic", "/obelisk/g1/exec_fsm")
        self.declare_parameter("pub_topic", "/obelisk/g1/fsm_state")
        # The hardware interface heartbeats its state, so a controller that starts after
        # the transition still learns about it. Match that or a late-arriving controller
        # would sit disarmed until the next joystick press.
        self.declare_parameter("heartbeat_period", 0.2)

        self._state = 0
        self._pub = self.create_publisher(
            ExecutionFSM, self.get_parameter("pub_topic").value, 10)
        self._sub = self.create_subscription(
            ExecutionFSM, self.get_parameter("sub_topic").value, self._on_cmd, 10)
        self.create_timer(float(self.get_parameter("heartbeat_period").value), self._heartbeat)
        self.get_logger().info("fsm_state_echo up (SIM ONLY: commanded FSM == actual FSM).")

    def _on_cmd(self, msg: ExecutionFSM) -> None:
        """Adopt and immediately republish a commanded state.

        Args:
            msg: The commanded execution FSM state.
        """
        if msg.cmd_exec_fsm_state != self._state:
            self.get_logger().info(
                f"FSM {self._state} -> {msg.cmd_exec_fsm_state}")
        self._state = msg.cmd_exec_fsm_state
        self._publish()

    def _heartbeat(self) -> None:
        """Republish the current state so late subscribers learn it."""
        self._publish()

    def _publish(self) -> None:
        """Publish the current state."""
        msg = ExecutionFSM()
        msg.cmd_exec_fsm_state = self._state
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)


def main(args: list | None = None) -> None:
    """Spin the echo node."""
    rclpy.init(args=args)
    node = FsmStateEcho()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
