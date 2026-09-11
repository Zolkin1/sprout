# Copyright 2026 Zachary Olkin. All rights reserved.

"""``prefetch_policy.py`` must load a pair under the CONTROLLER's rules, not its own.

The script's whole promise is "the controller will now start without touching the network",
which is a claim about the controller -- so any load-time rule it does not share is a way
for it to reach a different verdict than the node it is warming a cache for. It reached one
once: ``allow_robot_asset_mismatch`` was turned ON in ``common/low_level.yml`` for a known
actuator-gain-only ``mjcf_sha256`` delta, and the prefetch went on refusing the pair the
robot would have armed happily. A validator that is stricter than the thing it validates for
reads as a broken policy, and the fix is a shared switch rather than a second flag to
remember -- the config comment says to turn it back OFF after the generators are retrained,
and it can only be one place for that to work.

Deliberately does NOT assert the flag's VALUE: it is a temporary allowance and this test
must not have to be edited when it goes away. What is pinned is that the two read the same
key out of the same file, in both of its states.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import yaml

OBELISK = pathlib.Path(__file__).resolve().parent.parent
TRANSFER = OBELISK.parent
CONTROLLER_CC = OBELISK / "g1_control_cpp" / "src" / "g1_controller.cc"

sys.path.insert(0, str(TRANSFER))

_SPEC = importlib.util.spec_from_file_location(
    "prefetch_policy", OBELISK / "scripts" / "prefetch_policy.py")
prefetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prefetch)

#: The rules the prefetch has to share with the controller. One entry today; the list is the
#: point -- a second load-time param added to the node belongs here too.
SHARED_RULES = ("allow_robot_asset_mismatch",)


def test_the_default_config_is_the_fragment_every_launch_file_includes() -> None:
    """The default points at ``common/low_level.yml``, which is where the params live."""
    default = pathlib.Path(prefetch.DEFAULT_CONTROLLER_CONFIG)
    assert default.is_file(), f"{default} does not exist"
    assert default.parts[-2:] == ("common", "low_level.yml")


@pytest.mark.parametrize("rule", SHARED_RULES)
def test_prefetch_reads_the_same_rule_the_controller_declares(rule: str) -> None:
    """The reader finds the node's param, and the C++ controller declares it.

    Guards the two halves independently: a renamed key, a re-nested ``params:`` block or a
    renamed control entry breaks the python reader, and a param the node never declares is
    silently ignored by ROS (the failure ``test_params_are_live.py`` exists for).
    """
    params = prefetch.controller_params(prefetch.DEFAULT_CONTROLLER_CONFIG)
    assert rule in params, (
        f"{rule} is not reachable in the low_level params via controller_params(); the "
        "prefetch would silently fall back to its strict default")
    assert isinstance(params[rule], bool)
    assert f'"{rule}"' in CONTROLLER_CC.read_text(), (
        f"{rule} is in the config but the C++ controller never declares it, so rclpy "
        "ignores it and only the prefetch would obey it")


def test_the_value_is_taken_from_the_config_in_both_directions(tmp_path) -> None:
    """A true in the file reads true, a false reads false -- no baked-in default wins."""
    for value in (True, False):
        path = tmp_path / f"low_level_{value}.yml"
        path.write_text(yaml.safe_dump(
            {"control": [{"name": "low_level", "params": {"allow_robot_asset_mismatch": value}}]}))
        assert prefetch.controller_params(str(path))["allow_robot_asset_mismatch"] is value


@pytest.mark.parametrize("body", [
    "",                                                    # empty file -> None
    "control: []",                                         # no entries
    "control:\n  - name: something_else\n    params: {}",  # wrong node
    "control:\n  - name: low_level",                       # entry with no params
])
def test_an_unusable_config_falls_back_to_the_strict_defaults(tmp_path, body: str) -> None:
    """The safe direction for a validator to fail in is REFUSING, never allowing.

    Every one of these returns an empty mapping rather than raising, so ``.get(rule, False)``
    at the call site lands on the strict default and a mismatched pair is refused.
    """
    path = tmp_path / "low_level.yml"
    path.write_text(body)
    assert prefetch.controller_params(str(path)) == {}


def test_a_missing_config_falls_back_rather_than_raising(tmp_path) -> None:
    """An absent file is an empty mapping, not an OSError out of the middle of a fetch."""
    assert prefetch.controller_params(str(tmp_path / "nope.yml")) == {}
