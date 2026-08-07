# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the property that makes sim auto-arm safe: hardware structurally cannot do it.

Auto-arming means the controller starts driving with no button press. In MuJoCo that is
correct -- nothing holds the robot up, so every second spent disarmed is a second of it
collapsing. On a real robot it is exactly the behaviour the arming logic exists to prevent.

What makes that safe here is not a default value but WHERE the option lives: it is a
parameter of ``fsm_state_echo``, a node that exists only because MuJoCo has no execution
FSM and is launched only by ``run_sim_world.sh``. On hardware the G1 interface publishes the
real FSM state and this node never runs. These tests fail if that separation is ever eroded
-- if the controller grows its own auto-arm, or if a hardware launcher starts the echo node.

Deliberately grep-based rather than behavioural: the property being protected is about which
file mentions what, and a behavioural test would need the container.

Run: ``pytest transfer/obelisk/tests/test_auto_arm_is_sim_only.py``
"""

from __future__ import annotations

import pathlib

OBELISK = pathlib.Path(__file__).resolve().parent.parent
ECHO = OBELISK / "g1_control" / "g1_control" / "fsm_state_echo_node.py"
CONTROLLER = OBELISK / "g1_control" / "g1_control" / "controller.py"
SIM_WORLD = OBELISK / "scripts" / "run_sim_world.sh"
HARDWARE = OBELISK / "scripts" / "run_hardware_low.sh"


def test_only_the_sim_echo_node_can_auto_arm():
    """The controller must have no auto-arm of its own -- arming follows the FSM, always."""
    text = CONTROLLER.read_text()
    for forbidden in ("auto_arm", "auto-arm", "AUTO_ARM"):
        assert forbidden not in text, (
            f"controller.py mentions {forbidden!r}. Arming must follow the robot's execution "
            "FSM and nothing else; an auto-arm option on the controller would exist on "
            "hardware too, where it is the one thing the arming logic is there to prevent."
        )


def test_hardware_launcher_never_starts_the_echo_node():
    """On a robot the FSM comes from the robot. Echoing a commanded state would fake it."""
    assert "fsm_state_echo" not in HARDWARE.read_text(), (
        "run_hardware_low.sh starts fsm_state_echo. That node reports the COMMANDED state as "
        "the achieved one, so on hardware it would tell the controller the robot had complied "
        "with a transition it may not have made."
    )


def test_sim_world_auto_arms_by_default_and_can_opt_out():
    """The sim default is auto-arm, with an explicit way back to the hardware-like flow."""
    text = SIM_WORLD.read_text()
    assert "initial_state:=3" in text, "run_sim_world.sh no longer auto-arms"
    assert "--no-auto-arm" in text, "no way to get the hardware-like disarmed start back"
    assert "AUTO_ARM=1" in text, "auto-arm is no longer the default"


def test_echo_node_defaults_to_disarmed():
    """The NODE's own default stays INIT; only the launcher opts in.

    So `ros2 run g1_control fsm_state_echo` by hand does not silently arm a controller.
    """
    text = ECHO.read_text()
    assert 'declare_parameter("initial_state", INIT_STATE)' in text, \
        "the echo node must default to INIT; auto-arm is opt-in from the launcher"
    assert "INIT_STATE = 0" in text and "AUTO_ARM_STATE = 3" in text


def test_auto_arm_is_announced_loudly():
    """A robot that starts moving on its own must never be a surprise in a log or a bag."""
    text = ECHO.read_text()
    assert "AUTO-ARM" in text and "get_logger().warn" in text, \
        "auto-arm must warn, not info -- it is the line someone looks for afterwards"
