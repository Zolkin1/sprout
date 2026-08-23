# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the ONE address the depth pane ssh's to, and the preflight that says when it is wrong.

The depth cameras do not come up locally: ``run_hardware_low.sh``'s depth pane ssh's to the
ZED box and starts the pair inside the container there. So the box's address is part of the
control path, and it is written down in two places -- ``docker/.env`` (authoritative, because
compose puts it in the container's environment) and a fallback in ``lib/obk_launch.sh`` for
when the file does not set it. When those two disagree the pane silently aims at whichever
box the container was CREATED with, produces no depth, and reads as a camera fault. That has
already happened once: the value named the onboard minipc (``unitree@192.168.123.164``) for a
while after the cameras moved onto their own box.

Nothing here can check what the real robot network looks like. What it can check is that the
repo states the target once, that the tracked example agrees with the fallback, and that the
launcher still refuses to open a depth pane it has not proved it can reach.

Run: ``pytest transfer/obelisk/tests/test_zed_link_target.py``
"""

from __future__ import annotations

import pathlib
import re

OBELISK = pathlib.Path(__file__).resolve().parent.parent
LIB = OBELISK / "scripts" / "lib" / "obk_launch.sh"
ENV_EXAMPLE = OBELISK / "docker" / ".env_example"
HARDWARE = OBELISK / "scripts" / "run_hardware_low.sh"
SETUP_LINK = OBELISK / "scripts" / "setup_zed_link.sh"

# The address is a fact about the robot's wiring, so it is spelled here too -- a test that
# read it out of the file under test would pass for any value, including a stale one.
ZED_HOST = "zolkin@192.168.123.221"
# The box the cameras USED to be reached through. Any surviving mention is a stale default.
OLD_HOST = "unitree@192.168.123.164"


def test_the_fallback_and_the_tracked_env_example_agree():
    """A .env copied from the example must not disagree with the built-in fallback."""
    lib = LIB.read_text()
    assert f'OBK_JETSON_HOST_FALLBACK:-{ZED_HOST}' in lib, (
        f"{LIB.name} no longer declares {ZED_HOST} as the fallback ssh target."
    )
    env = ENV_EXAMPLE.read_text()
    assert f"OBK_JETSON_HOST={ZED_HOST}" in env, (
        f"docker/.env_example does not set OBK_JETSON_HOST to {ZED_HOST}. It is the file "
        "every box's .env is copied from, and compose's value WINS inside the container, so "
        "an example that disagrees with the fallback hands the next box a broken depth pane."
    )


def test_no_stale_minipc_target_survives():
    """The cameras moved off the onboard minipc; a leftover default would send ssh back."""
    for path in (LIB, ENV_EXAMPLE, HARDWARE, SETUP_LINK):
        text = path.read_text()
        for line in text.splitlines():
            if OLD_HOST not in line:
                continue
            # Naming it as history is the point of the comment that does so.
            assert line.lstrip().startswith("#"), (
                f"{path.name} still uses {OLD_HOST} as a live value: {line.strip()!r}"
            )


def test_the_launcher_will_not_open_a_depth_pane_it_cannot_reach():
    """The preflight must gate the pane, and only --no-depth may skip it."""
    text = HARDWARE.read_text()
    assert "obk_jetson_preflight" in text, (
        "run_hardware_low.sh no longer preflights the ZED link. Without it a missing ssh key "
        "presents as a depth pane that never prints a frame -- indistinguishable, on a robot, "
        "from a broken camera."
    )
    gate = re.search(
        r'if \[ "\$SENSORS" = 1 \] && \[ "\$DEPTH" = 1 \]; then\s*\n\s*if ! obk_jetson_preflight',
        text,
    )
    assert gate, (
        "the preflight is no longer gated on SENSORS/DEPTH, or no longer fatal. --sim-test "
        "and --no-depth are the two runs that legitimately have no ZED box to reach."
    )


def test_a_hardware_run_marks_the_jetson_so_stop_can_reach_it():
    """``stop.sh --zed`` tests for the marker before spending an ssh; nothing else sets it."""
    text = HARDWARE.read_text()
    assert "obk_mark_jetson 1" in text, (
        "run_hardware_low.sh no longer marks the ZED box as carrying a node. stop.sh --zed "
        "checks that marker first, so without it the remote teardown is skipped SILENTLY and "
        "the pair keeps holding the cameras after the stack is down."
    )
