# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin that a depth dropout zeroes the velocity command, in BOTH controllers.

This exists because the C++ port silently dropped the behaviour. ``UpdateWatchdogs`` returned
"anything is stale", its one call site discarded the bool, and ``EffectiveCommand`` consulted
only ``vel_cmd_stale_`` -- so on the deployed controller a dead camera produced a warning, a
``depth_stale`` field in the debug tick, and no change to any motor command. The robot kept
walking at the last commanded velocity against an image that had stopped describing the world,
for as long as the camera stayed down (on the ZED pair, a 10 s respawn delay plus the 25-30 s
``sl::Camera::close()`` takes to release a ZED X). The python controller had always zeroed on
either watchdog, and ``low_level.yml`` documented that behaviour the whole time -- so the
config, the reference implementation and the deployed code disagreed with each other and
nothing failed.

Deliberately source-based rather than behavioural: ``G1Controller::EffectiveCommand`` is a
private method of a ROS node that links onnxruntime, so exercising it needs the deploy
container, and the property being protected is "this guard reads both latches", which is a
statement about the code. The python half IS behaviourally covered next door in
``test_controller_clip_mode.py``; what has no other guard is the port.

Run: ``pytest transfer/obelisk/tests/test_depth_dropout_zeroes_the_command.py``
"""

from __future__ import annotations

import pathlib
import re

OBELISK = pathlib.Path(__file__).resolve().parent.parent
CPP_SRC = OBELISK / "g1_control_cpp" / "src" / "g1_controller.cc"
CPP_HDR = OBELISK / "g1_control_cpp" / "include" / "g1_control" / "g1_controller.h"
PY_SRC = OBELISK / "g1_control" / "g1_control" / "controller.py"
LOW_LEVEL = OBELISK / "g1_control" / "config" / "common" / "low_level.yml"


def _cpp_body(text: str, signature: str) -> str:
    """Return the body of a top-level C++ definition, by its opening line.

    Args:
        text: the whole translation unit.
        signature: a substring identifying the definition's first line.

    Returns:
        Every line from the signature up to the closing brace in column zero.
    """
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if signature in ln and ln.endswith("{")]
    assert len(starts) == 1, f"expected one definition of {signature!r}, found {len(starts)}"
    start = starts[0]
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start:end + 1])
    raise AssertionError(f"unterminated definition for {signature!r}")


def _py_body(text: str, name: str) -> str:
    """Return the source of a python method, from its ``def`` to the next same-indent ``def``.

    Args:
        text: the whole module.
        name: the method name, without ``def``.

    Returns:
        The method's source lines.
    """
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.match(rf"\s*def {re.escape(name)}\b", ln)]
    assert len(starts) == 1, f"expected one def of {name!r}, found {len(starts)}"
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    for end in range(start + 1, len(lines)):
        stripped = lines[end].strip()
        if not stripped:
            continue
        cur = len(lines[end]) - len(lines[end].lstrip())
        if cur <= indent and stripped.startswith(("def ", "@")):
            return "\n".join(lines[start:end])
    return "\n".join(lines[start:])


def test_cpp_effective_command_zeroes_on_a_depth_dropout():
    """The deployed controller must stop asking for travel it cannot see."""
    body = _cpp_body(CPP_SRC.read_text(), "Vec3 G1Controller::EffectiveCommand()")
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("//"))
    assert "depth_stale_" in code, (
        "EffectiveCommand does not read depth_stale_. A depth dropout would then log a "
        "warning and leave the robot walking at the last commanded velocity on a frozen "
        f"image -- the exact regression this test exists for. Body:\n{body}"
    )
    guard = [ln for ln in code.splitlines() if "depth_stale_" in ln]
    assert any("return" in ln and "0.0" in ln for ln in guard), (
        "depth_stale_ is mentioned but does not gate a zero return. Seen:\n"
        + "\n".join(guard)
    )


def test_update_watchdogs_returns_nothing_so_no_call_site_can_drop_it():
    """The bool return is what made the missing response invisible; it must not come back."""
    hdr = CPP_HDR.read_text()
    assert re.search(r"\bvoid UpdateWatchdogs\(\)", hdr), (
        "UpdateWatchdogs must return void. A bool return here was read by nobody while "
        "EffectiveCommand consulted only half the latches, which is how a depth dropout "
        "came to change no motor command. The latches are the interface."
    )
    src = CPP_SRC.read_text()
    body = _cpp_body(src, "void G1Controller::UpdateWatchdogs()")
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        assert not re.match(r"return\s+\S", stripped), (
            f"UpdateWatchdogs returns a value again: {stripped!r}"
        )


def test_python_and_cpp_zero_on_the_same_watchdogs():
    """The deprecated python controller is the reference; the port must not diverge again."""
    py = PY_SRC.read_text()
    effective = _py_body(py, "_effective_command")
    assert "_update_watchdogs()" in effective, (
        "python _effective_command no longer routes through _update_watchdogs; the parity "
        f"this test checks is gone. Body:\n{effective}"
    )
    watchdogs = _py_body(py, "_update_watchdogs")
    returns = [ln.strip() for ln in watchdogs.splitlines() if ln.strip().startswith("return")]
    assert any("depth_stale" in ln for ln in returns), (
        "python _update_watchdogs no longer reports depth staleness to its caller, so this "
        f"test can no longer establish that both controllers agree. Returns: {returns}"
    )


def test_the_configured_promise_is_the_implemented_one():
    """``depth_timeout``'s comment claims the command is zeroed; that must stay true."""
    lines = LOW_LEVEL.read_text().splitlines()
    keys = [i for i, ln in enumerate(lines) if ln.strip().startswith("depth_timeout:")]
    assert len(keys) == 1, f"expected one depth_timeout key, found {len(keys)}"
    # Walk the contiguous comment block upwards rather than a fixed byte window: the
    # rationale above this key is long and grows, and a windowed search silently stops
    # covering the sentence it is meant to pin.
    start = keys[0]
    while start > 0 and lines[start - 1].strip().startswith("#"):
        start -= 1
    preamble = "\n".join(lines[start:keys[0]])
    assert preamble.strip(), "depth_timeout has no comment block above it at all"
    assert "Zero the velocity command" in preamble, (
        "the comment above depth_timeout no longer promises a zeroed command. If the "
        "behaviour changed, change this test; if only the comment changed, the config and "
        f"the code have come apart again. Block:\n{preamble}"
    )
