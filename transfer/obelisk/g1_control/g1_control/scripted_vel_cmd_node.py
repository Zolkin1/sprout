# Copyright 2026 Zachary Olkin. All rights reserved.

"""Drive the robot from a scripted velocity profile instead of the joystick.

So a run can be COMPARED. The MuJoCo sim2sim runner has scripted profiles
(``--mode eval --command <name>``) and this stack had only a human on a stick, which means
every obelisk-vs-sim2sim comparison differed by however the stick was pushed. This publishes
the SAME profile functions on ``/obelisk/g1/vel_cmd``, so the two stacks can be run on one
input and the difference between them is the stacks.

ONE SOURCE OF PROFILES, imported from ``transfer/sim/experiments/velocity_commands.py`` rather
than copied. A copy would let the two sims drift on the very thing being held constant, which
is the whole point of scripting it. That module is numpy-only -- no MuJoCo, no torch -- so
importing it here costs nothing.

THE CLOCK STARTS WHEN THE CONTROLLER ARMS, not when this node starts. Every profile is a
function of time from zero, and the controller takes ~10 s to load its ONNX pair; a clock
started at launch would be several seconds into the ramp before the robot could move, so the
two stacks would run different parts of the profile. Arming is read from
``/obelisk/g1/debug/tick``, which the controller publishes with an ``armed`` flag.

Before arming it publishes ZEROS rather than nothing: the controller's velocity-command
watchdog zeroes a stale command, and a silent topic would make the first armed tick see an
expired one.

A PLAIN rclpy node launched with ``ros2 run``, following ``fsm_state_echo_node``: as an
obelisk entry it would share the lifecycle and node-name space of the real controller, and a
cross-config transition would then kill a live node.

    ros2 run g1_control scripted_vel_cmd --ros-args -p profile:=constant_forward
"""

from __future__ import annotations

import os
import sys

import rclpy
from obelisk_control_msgs.msg import VelocityCommand
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile

from g1_msgs.msg import ControllerDebug

# The shared deploy tree, and its sim/ sibling for the profiles -- see the same block in
# controller.py for why this is realpath and not abspath (colcon --symlink-install imports
# through build/, and abspath does not resolve the symlink).
_TRANSFER = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
)
_SIM = os.path.join(_TRANSFER, "sim")
if not os.path.isdir(_SIM):
    raise RuntimeError(
        f"the MuJoCo sim tree is not at {_SIM} (derived from {os.path.realpath(__file__)}). "
        "The velocity profiles live there and are imported rather than copied, so that the "
        "two simulators cannot drift on the input being held constant."
    )
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

import experiments.velocity_commands as vc  # noqa: E402


class ScriptedVelCmd(Node):
    """Publishes a velocity profile, clocked from the moment the controller arms."""

    def __init__(self) -> None:
        """Resolve the profile and start publishing."""
        super().__init__("scripted_vel_cmd")
        #: Profile name in ``experiments.velocity_commands`` -- the same names the sim2sim
        #: runner's ``--command`` takes.
        self.declare_parameter("profile", "constant_forward")
        self.declare_parameter("topic", "/obelisk/g1/vel_cmd")
        self.declare_parameter("tick_topic", "/obelisk/g1/debug/tick")
        #: Publish rate. 50 Hz matches the control tick, so the command is never the stale
        #: input; the profiles are continuous functions so the rate is not load-bearing.
        self.declare_parameter("rate", 50.0)
        #: Stop publishing (and hold zeros) after this many seconds of profile time. <=0
        #: runs forever. A bounded run is what makes two recordings the same length.
        self.declare_parameter("duration", 0.0)
        #: Start the profile clock at ARM (default) or at node start. The latter exists only
        #: for driving a stack with no controller attached, e.g. a bring-up smoke test.
        self.declare_parameter("start_on_arm", True)

        name = str(self.get_parameter("profile").value)
        self._fn = getattr(vc, name, None)
        if self._fn is None or not callable(self._fn):
            # Named rather than defaulted: a typo silently falling back to some other profile
            # would produce a comparison of two different inputs that both looked scripted.
            available = sorted(n for n in dir(vc) if not n.startswith("_") and callable(
                getattr(vc, n)))
            raise RuntimeError(f"no velocity profile '{name}' in experiments.velocity_commands; "
                               f"available: {available}")
        self._duration = float(self.get_parameter("duration").value)
        self._start_on_arm = bool(self.get_parameter("start_on_arm").value)
        self._t0: float | None = None if self._start_on_arm else self._now()
        self._done = False

        self._pub = self.create_publisher(
            VelocityCommand, str(self.get_parameter("topic").value), QoSProfile(depth=10))
        self.create_subscription(ControllerDebug, str(self.get_parameter("tick_topic").value),
                                 self._on_tick, 10)
        period = 1.0 / max(float(self.get_parameter("rate").value), 1e-3)
        self.create_timer(period, self._publish)
        self.get_logger().info(
            f"scripted_vel_cmd up: profile '{name}', "
            + ("waiting for the controller to ARM before the clock starts"
               if self._start_on_arm else "clock started at node start")
            + (f", stopping after {self._duration:.1f} s" if self._duration > 0 else ""))

    def _now(self) -> float:
        """Node clock, in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_tick(self, msg: ControllerDebug) -> None:
        """Start the profile clock on the first ARMED tick.

        Args:
            msg: The controller's per-tick debug message.
        """
        if self._start_on_arm and self._t0 is None and msg.armed:
            self._t0 = self._now()
            self.get_logger().info("controller ARMED: starting the velocity profile now")

    def _publish(self) -> None:
        """Publish the profile's command, or zeros before the clock starts / after it ends."""
        msg = VelocityCommand()
        if self._t0 is None or self._done:
            # Zeros, not silence: the controller zeroes a STALE command, so a quiet topic
            # would have the first armed tick reading an expired one.
            msg.v_x, msg.v_y, msg.w_z = 0.0, 0.0, 0.0
        else:
            t = self._now() - self._t0
            if self._duration > 0 and t > self._duration:
                self._done = True
                self.get_logger().info(f"profile finished at {t:.2f} s; holding zeros")
                msg.v_x, msg.v_y, msg.w_z = 0.0, 0.0, 0.0
            else:
                v = self._fn(t)
                msg.v_x, msg.v_y, msg.w_z = float(v[0]), float(v[1]), float(v[2])
        self._pub.publish(msg)


def main(args=None) -> None:
    """Spin the node.

    Args:
        args: Passed through to rclpy.
    """
    rclpy.init(args=args)
    node = ScriptedVelCmd()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT gives KeyboardInterrupt; SIGTERM -- which is what tmux and stop.sh send --
        # invalidates the context under spin, which surfaces as ExternalShutdownException or,
        # depending on where the wait set was, an RCLError about an invalid context. None of
        # the three is a failure, and a traceback in the pane reads as one.
        pass
    except RCLError:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
