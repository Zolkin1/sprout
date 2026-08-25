# Copyright 2026 Zachary Olkin. All rights reserved.

"""Catch config knobs that look live and are not.

rclpy SILENTLY IGNORES an override for a parameter the node never declared. So a line in
the YAML reads exactly like a setting and does nothing at all, and the only way to find out
is to change it and notice the robot behaves identically. This stack has grown four such
knobs -- ``log``, ``log_decimation``, ``config_name`` and ``fm_update_every`` -- and each
one cost someone a debugging session before it was spotted.

Two directions, because they fail differently:

* a parameter the CONFIG sets but the NODE does not declare is silently ignored (the
  ``log: true`` class);
* a parameter the NODE declares but never reads is dead weight that still shows up in
  ``ros2 param list``, inviting someone to set it (the ``fm_update_every`` class -- it was
  declared, read, passed to the policy, and then overridden by artifact metadata every
  time, which this test would NOT have caught; see the note below).

The second class only checks that the node reads its own parameters at all. Whether the
value then survives is beyond grep, which is why ``fm_update_every`` needed a human. The
first class is the one that recurs and is fully checkable.

Run: ``pytest transfer/obelisk/tests/test_params_are_live.py``
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

OBELISK = pathlib.Path(__file__).resolve().parent.parent
CONTROLLER = OBELISK / "g1_control" / "g1_control" / "controller.py"
CONTROLLER_CPP = OBELISK / "g1_control_cpp" / "src" / "g1_controller.cc"
ESTIMATOR = OBELISK / "g1_control" / "g1_control" / "estimator.py"
CONFIG_DIR = OBELISK / "g1_control" / "config"

#: Node name in the configs -> the source(s) that may implement it. Only first-party nodes: an
#: obelisk or upstream node declares its parameters in another package.
#:
#: `low_level` has TWO implementations -- the python controller and the C++ one -- selected by
#: which config is launched (g1.yml vs g1_cpp.yml). A parameter is live if EITHER declares it,
#: and each config is checked against the pair rather than against one of them, because the
#: alternative is a test that fails whenever a knob exists on only one side.
FIRST_PARTY = {"low_level": (CONTROLLER, CONTROLLER_CPP), "g1_estimator": (ESTIMATOR,)}

#: Matches both `declare_parameter("x"` and the C++ `declare_parameter<double>("x"`. Without the
#: optional template argument this silently matches nothing in the C++ source, and the test then
#: passes VACUOUSLY against it -- which is worse than failing, because it looks like coverage.
_DECLARE = re.compile(r'declare_parameter(?:<[^>]*>)?\(\s*"([^"]+)"')
_GET = re.compile(r'get_parameter\(\s*"([^"]+)"')


def _declared(paths) -> set[str]:
    """Parameter names any of `paths` declares."""
    out: set[str] = set()
    for path in paths:
        if path.exists():
            out |= set(_DECLARE.findall(path.read_text()))
    return out


def _read(paths) -> set[str]:
    """Parameter names any of `paths` reads back."""
    out: set[str] = set()
    for path in paths:
        if path.exists():
            out |= set(_GET.findall(path.read_text()))
    return out


def _config_params(cfg: pathlib.Path) -> dict[str, dict]:
    """``{node name: params dict}`` for every node a config sets parameters on."""
    doc = yaml.safe_load(cfg.read_text()) or {}
    out = {}
    for section in ("control", "estimation"):
        for entry in doc.get(section) or []:
            if isinstance(entry, dict) and entry.get("params"):
                out[entry["name"]] = entry["params"]
    return out


@pytest.mark.parametrize("cfg", sorted(CONFIG_DIR.rglob("*.yml")), ids=lambda p: p.name)
def test_every_configured_parameter_is_declared(cfg: pathlib.Path):
    """A parameter the node never declares is silently dropped by rclpy."""
    for node, params in _config_params(cfg).items():
        if node not in FIRST_PARTY:
            continue
        declared = _declared(FIRST_PARTY[node])
        undeclared = sorted(set(params) - declared)
        assert not undeclared, (
            f"{cfg.name} sets {undeclared} on '{node}', which none of "
            f"{[p.name for p in FIRST_PARTY[node]]} declares. rclpy ignores overrides for "
            "undeclared parameters WITHOUT WARNING, so these read as live settings and do "
            "nothing. Either declare and use them, or delete them from the config."
        )


#: One entry per SOURCE file, not per node: an unread parameter is a property of the file that
#: declares it, and checking a node's files as a set would let one implementation cover for a
#: dead knob in the other.
_SOURCES = sorted({p for paths in FIRST_PARTY.values() for p in paths if p.exists()})


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_every_declared_parameter_is_read(path: pathlib.Path):
    """A declared-but-unread parameter appears in `ros2 param list` and does nothing."""
    unread = sorted(_declared([path]) - _read([path]))
    assert not unread, (
        f"{path.name} declares {unread} and never calls get_parameter on them. They show up "
        "in `ros2 param list` and in the config as though they were live. Delete them, or "
        "read them."
    )
