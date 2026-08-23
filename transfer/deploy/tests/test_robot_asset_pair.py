# Copyright 2026 Zachary Olkin. All rights reserved.

"""A tracker and a generator running together must have been trained on one robot.

The deploy box cannot check an artifact against "the current robot" -- it has whatever
hardware it is bolted to. What it can check is the pair, exactly as it already does for
the depth-camera contract: there is one physical robot, so two halves trained on different
ones is a configuration error, not a tuning problem.

The interesting failures are all "same joints, different robot" -- a payload, a hand swap,
a gear ratio -- which is precisely what ``joint_names_isaac`` (the only robot-ish field the
metadata carried before) does NOT notice.

No ONNX files: the guard runs on two dicts.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import _assert_robot_assets_agree  # noqa: E402

JOINTS = [f"j{i}_joint" for i in range(29)]


def _asset(**over):
    base = {
        "hash": "a" * 64, "short": "aaaaaaaaaaaa", "version": 1,
        "covers": ["body_names", "joint_names", "total_mass_kg", "urdf_sha256"],
        "joint_names": list(JOINTS),
        "body_names": ["pelvis_link", "torso_link"],
        "total_mass_kg": 36.1683,
        "urdf_sha256": "b" * 64,
    }
    base.update(over)
    return base


def _agree(a, b):
    _assert_robot_assets_agree(a, b, "tracker.onnx", "fm.onnx")


def test_the_same_robot_passes():
    _agree(_asset(), _asset())


@pytest.mark.parametrize("absent", [None, {}])
def test_a_missing_fingerprint_is_not_an_error(absent):
    """Every artifact exported before the field existed has none; refusing those would
    break every pair on disk. The fingerprint guards against a NEW mismatch."""
    _agree(absent, _asset())
    _agree(_asset(), absent)
    _agree(absent, absent)


def test_a_payload_difference_is_refused():
    """Same joints, same bodies, 3 kg apart -- invisible to every other field."""
    with pytest.raises(ValueError, match="ROBOT ASSET MISMATCH") as e:
        _agree(_asset(), _asset(total_mass_kg=39.2))
    assert "total_mass_kg" in str(e.value)
    assert "joint_names" not in str(e.value), "the joints agree; do not report them"


def test_a_hand_swap_is_refused_and_named():
    with pytest.raises(ValueError) as e:
        _agree(_asset(body_names=["pelvis_link", "torso_link", "left_hand_index_0_link"]),
               _asset(body_names=["pelvis_link", "torso_link", "left_rubber_hand"]))
    msg = str(e.value)
    assert "left_hand_index_0_link" in msg and "left_rubber_hand" in msg


def test_a_regenerated_asset_is_refused():
    with pytest.raises(ValueError, match="urdf_sha256"):
        _agree(_asset(), _asset(urdf_sha256="c" * 64))


def test_a_field_only_one_side_carries_is_not_compared():
    """A generator whose recording could see less than the tracker still loads."""
    sparse = {"hash": "x", "short": "x", "version": 1, "joint_names": list(JOINTS)}
    _agree(_asset(), sparse)


def test_the_message_says_how_to_fix_it():
    with pytest.raises(ValueError) as e:
        _agree(_asset(), _asset(total_mass_kg=1.0))
    assert "Re-export both from one run" in str(e.value)
