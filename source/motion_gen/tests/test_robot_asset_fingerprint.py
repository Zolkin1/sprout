# Copyright 2026 Zachary Olkin. All rights reserved.

"""The robot fingerprint must notice the changes that nothing else in the repo notices.

The interesting cases are all "same joints, different robot": adding a backpack, swapping
the hands, changing a gear ratio. Those leave ``joint_names`` -- the only robot-ish field
any exported artifact carried before this -- untouched, which is exactly why a stale
checkpoint used to load cleanly against a new model.
"""

from __future__ import annotations

import os

import pytest

from motion_gen.features import _REPO_ROOT
from motion_gen.robot_asset import (
    ROBOT_ASSET_FINGERPRINT_VERSION,
    fingerprint_robot_asset,
    sha256_file,
    sha256_tree,
    verify_robot_asset,
)

JOINTS = ["a_joint", "b_joint", "c_joint"]
BODIES = ["pelvis", "torso", "hand"]


def _fp(**over):
    base = dict(joint_names=JOINTS, body_names=BODIES, total_mass_kg=36.1683)
    base.update(over)
    return fingerprint_robot_asset(**base)


##
# The digest itself.
##


def test_the_same_robot_gives_the_same_digest():
    assert _fp()["hash"] == _fp()["hash"]


def test_body_name_order_does_not_matter_but_joint_order_does():
    """Joint ORDER is identity -- it is the PhysX order every exported artifact indexes by.

    Body names are looked up by name, so their order carries no meaning and must not make
    two identical robots look different.
    """
    assert _fp(body_names=list(reversed(BODIES)))["hash"] == _fp()["hash"]
    assert _fp(joint_names=list(reversed(JOINTS)))["hash"] != _fp()["hash"]


def test_covers_lists_what_was_actually_hashed():
    fp = fingerprint_robot_asset(joint_names=JOINTS)
    assert fp["covers"] == ["joint_names"]
    assert "total_mass_kg" not in fp


def test_a_missing_file_hashes_empty_rather_than_raising():
    """A deploy box carries the ONNX pair and no model files; that must not be an error."""
    assert sha256_file("/nonexistent/robot.urdf") == ""
    assert sha256_tree("/nonexistent/dir") == ""
    assert fingerprint_robot_asset(urdf_path="/nonexistent/robot.urdf")["urdf_sha256"] == ""


##
# The changes this exists to catch.
##


def test_a_payload_changes_the_fingerprint_while_the_joints_do_not():
    """The backpack case: same joints, same bodies, 3 kg more robot."""
    before, after = _fp(), _fp(total_mass_kg=39.2)
    assert before["joint_names"] == after["joint_names"]
    assert before["hash"] != after["hash"]
    ok, msg = verify_robot_asset(before, after)
    assert not ok
    assert "total_mass_kg" in msg and "+3.0317 kg" in msg


def test_a_hand_swap_changes_the_fingerprint_while_the_joints_do_not():
    """The rubber-hand case: 16 fixed-joint links become 2, no actuated joint moves."""
    before = _fp(body_names=BODIES + [f"hand_{i}" for i in range(16)])
    after = _fp(body_names=BODIES + ["left_rubber_hand", "right_rubber_hand"])
    assert before["joint_names"] == after["joint_names"]
    ok, msg = verify_robot_asset(before, after)
    assert not ok
    assert "body_names" in msg and "left_rubber_hand" in msg


def test_a_regenerated_asset_changes_the_fingerprint(tmp_path):
    """A gear-ratio edit lives in the URDF's joint limits: no name or count moves."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="r"><joint name="a"><limit effort="88"/></joint></robot>')
    before = fingerprint_robot_asset(urdf_path=str(urdf), joint_names=JOINTS)
    urdf.write_text('<robot name="r"><joint name="a"><limit effort="139"/></joint></robot>')
    after = fingerprint_robot_asset(urdf_path=str(urdf), joint_names=JOINTS)
    assert before["hash"] != after["hash"]
    ok, msg = verify_robot_asset(before, after)
    assert not ok and "urdf_sha256" in msg


def test_the_usd_hash_covers_the_payload_layers(tmp_path):
    """A USD is a directory of layers; hashing only the root would miss a regenerated mesh."""
    root = tmp_path / "asset"
    (root / "payloads").mkdir(parents=True)
    (root / "asset.usda").write_text("#usda 1.0\n")
    (root / "payloads" / "geometries.usd").write_bytes(b"\x00" * 64)
    before = fingerprint_robot_asset(usd_path=str(root / "asset.usda"))
    (root / "payloads" / "geometries.usd").write_bytes(b"\x01" * 64)
    after = fingerprint_robot_asset(usd_path=str(root / "asset.usda"))
    assert before["usd_sha256"] != after["usd_sha256"]


##
# Verification policy.
##


def test_a_match_is_a_match():
    ok, msg = verify_robot_asset(_fp(), _fp())
    assert ok and "match" in msg


def test_only_shared_fields_are_compared():
    """A producer that saw only the MJCF still checks against one that saw everything."""
    sparse = fingerprint_robot_asset(joint_names=JOINTS)
    full = _fp()
    ok, _msg = verify_robot_asset(sparse, full)
    assert ok, "the shared field (joint_names) agrees, so this must pass"


def test_no_shared_field_is_a_mismatch_not_a_pass():
    """Agreeing by silence is the failure mode the module exists to remove."""
    a = fingerprint_robot_asset(joint_names=JOINTS)
    b = fingerprint_robot_asset(total_mass_kg=1.0)
    ok, msg = verify_robot_asset(a, b)
    assert not ok and "share no comparable field" in msg


@pytest.mark.parametrize("recorded,current", [(None, {"hash": "x"}), ({"hash": "x"}, None),
                                              ({}, {"hash": "x"})])
def test_an_absent_fingerprint_is_reported_not_assumed_fine(recorded, current):
    ok, msg = verify_robot_asset(recorded, current)
    assert not ok and msg


def test_a_version_bump_is_a_mismatch():
    old = dict(_fp(), version=ROBOT_ASSET_FINGERPRINT_VERSION - 1)
    ok, msg = verify_robot_asset(old, _fp())
    assert not ok and "version" in msg


##
# Against the real assets on disk.
##


def test_the_live_models_fingerprint_and_differ_from_each_other():
    urdf = _REPO_ROOT / "robot_assets" / "g1" / "g1_21j" / "g1_21j_box_merged_29j_hands.urdf"
    other = _REPO_ROOT / "robot_assets" / "g1" / "g1_21j" / "g1_21j_box_merged_29j.urdf"
    if not urdf.is_file() or not other.is_file():
        pytest.skip("robot_assets not present")
    a = fingerprint_robot_asset(urdf_path=str(urdf))
    b = fingerprint_robot_asset(urdf_path=str(other))
    assert a["urdf_sha256"] and len(a["urdf_sha256"]) == 64
    assert a["hash"] != b["hash"], "the pre-hands and hands URDFs must not fingerprint alike"


def test_the_usd_payload_tree_hashes(tmp_path):
    usd = (_REPO_ROOT / "robot_assets" / "g1" / "g1_21j"
           / "g1_21j_box_merged_29j_hands" / "g1_21j_box_merged_29j_hands.usda")
    if not usd.is_file():
        pytest.skip("robot_assets USD not present")
    fp = fingerprint_robot_asset(usd_path=str(usd))
    assert len(fp["usd_sha256"]) == 64
    assert fp["usd_name"] == os.path.basename(str(usd))
