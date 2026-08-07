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
ESTIMATOR = OBELISK / "g1_control" / "g1_control" / "estimator.py"
CONFIG_DIR = OBELISK / "g1_control" / "config"

#: Node name in the configs -> the module that implements it. Only first-party nodes: an
#: obelisk or upstream node declares its parameters in C++ or in another package.
FIRST_PARTY = {"low_level": CONTROLLER, "g1_estimator": ESTIMATOR}


def _declared(path: pathlib.Path) -> set[str]:
    """Parameter names a node declares."""
    return set(re.findall(r'declare_parameter\(\s*"([^"]+)"', path.read_text()))


def _read(path: pathlib.Path) -> set[str]:
    """Parameter names a node reads back."""
    return set(re.findall(r'get_parameter\(\s*"([^"]+)"', path.read_text()))


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
            f"{cfg.name} sets {undeclared} on '{node}', which "
            f"{FIRST_PARTY[node].name} never declares. rclpy ignores overrides for "
            "undeclared parameters WITHOUT WARNING, so these read as live settings and do "
            "nothing. Either declare and use them, or delete them from the config."
        )


@pytest.mark.parametrize("node,path", sorted(FIRST_PARTY.items()), ids=lambda v: getattr(v, "name", v))
def test_every_declared_parameter_is_read(node: str, path: pathlib.Path):
    """A declared-but-unread parameter appears in `ros2 param list` and does nothing."""
    unread = sorted(_declared(path) - _read(path))
    assert not unread, (
        f"{path.name} declares {unread} and never calls get_parameter on them. They show up "
        "in `ros2 param list` and in the config as though they were live. Delete them, or "
        "read them."
    )
