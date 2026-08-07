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

``initial_state`` AUTO-ARMS the controller (see :data:`AUTO_ARM_STATE`). This is safe to
offer here and nowhere else: this node exists ONLY because MuJoCo has no execution FSM, and
is launched ONLY by ``run_sim_world.sh``. On hardware the G1 interface publishes the real
state and this file never runs -- so auto-arming is not a flag that could be left set on a
robot, it is a capability the hardware path structurally does not have. Putting the same
option on the controller would not have that property.
"""

from __future__ import annotations

import rclpy
from obelisk_control_msgs.msg import ExecutionFSM
from rclpy.node import Node

#: ``ExecFSMState::INIT``. The state a real robot powers up in, and the historical default
#: here -- the controller stays disarmed until a joystick command says otherwise.
INIT_STATE = 0

#: ``ExecFSMState::USER_CTRL``, which obelisk's TRANSITION_STRINGS calls LOW_LEVEL_CONTROL:
#: the one state in which the controller may drive. Publishing it from startup is what
#: "auto-arm" means.
AUTO_ARM_STATE = 3


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
        # State published before any joystick command arrives. AUTO_ARM_STATE means the
        # controller arms as soon as it has an estimate, with no button press -- which is
        # what you want in sim, where MuJoCo has nothing holding the robot up and every
        # second spent disarmed is a second of the robot collapsing.
        self.declare_parameter("initial_state", INIT_STATE)

        self._state = int(self.get_parameter("initial_state").value)
        self._pub = self.create_publisher(
            ExecutionFSM, self.get_parameter("pub_topic").value, 10)
        self._sub = self.create_subscription(
            ExecutionFSM, self.get_parameter("sub_topic").value, self._on_cmd, 10)
        self.create_timer(float(self.get_parameter("heartbeat_period").value), self._heartbeat)
        self.get_logger().info("fsm_state_echo up (SIM ONLY: commanded FSM == actual FSM).")
        if self._state == AUTO_ARM_STATE:
            # Loud, because "the robot started moving on its own" should never be a
            # surprise -- even in sim, and especially to someone reading a bag later.
            self.get_logger().warn(
                f"AUTO-ARM: publishing LOW_LEVEL_CONTROL ({AUTO_ARM_STATE}) from startup, so "
                "the controller arms with no joystick input. SIM ONLY -- on hardware this "
                "node does not run and the robot's own FSM decides. A joystick command "
                "still overrides this at any time (damping and e-stop included).")
        elif self._state != INIT_STATE:
            self.get_logger().info(f"Starting in FSM state {self._state}.")

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
