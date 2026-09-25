# Copyright 2026 Zachary Olkin. All rights reserved.

"""Checks on SONIC's G1 as we spawn it for the baseline comparison.

Everything here is static (URDF text + config arithmetic), so it runs without a simulator. The
things that can only be checked in the simulator -- the resolved DOF order and the absence of a
world-anchoring root joint -- are covered by ``scripts/rsl_rl/inspect_sonic_asset.py`` and by
``SonicEnvBridge``'s startup assertion, which fails the rollout rather than mis-tracking quietly.

Tests needing the GR00T clone skip when it is absent.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from robot_rl.assets.robots.g1_21j import G1_ACTION_SCALE
from robot_rl.assets.robots.g1_sonic import (
    G1_SONIC_ACTION_SCALE,
    G1_SONIC_ISAAC_JOINT_ORDER,
    _SONIC_ACTUATORS,
    find_groot_root,
    sonic_patched_urdf_path,
)

OUR_URDF_TAIL = "robot_assets/g1/g1_21j/g1_21j_box_merged_29j.urdf"


def _clone_or_skip():
    """The GR00T clone, or skip the test."""
    try:
        return find_groot_root()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _revolute_limits(path) -> dict[str, tuple[float, float]]:
    """``{joint_name: (lower, upper)}`` for every revolute joint in a URDF."""
    out = {}
    for j in ET.parse(path).getroot().iter("joint"):
        if j.get("type") != "revolute":
            continue
        lim = j.find("limit")
        if lim is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def test_joint_order_is_29_unique_names() -> None:
    """The transcribed DOF order is complete and has no duplicates."""
    assert len(G1_SONIC_ISAAC_JOINT_ORDER) == 29
    assert len(set(G1_SONIC_ISAAC_JOINT_ORDER)) == 29


def test_joint_order_covers_exactly_the_urdf_revolute_joints() -> None:
    """The transcribed order names the same joints the URDF actuates -- no more, no fewer."""
    root = _clone_or_skip()
    urdf = root / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
    assert set(G1_SONIC_ISAAC_JOINT_ORDER) == set(_revolute_limits(urdf))


def test_joint_limits_match_our_robot() -> None:
    """SONIC's joint limits equal ours, so our clips are reachable poses on its robot.

    If they diverged, a clip pose could sit outside SONIC's limits and the reset would fight the
    solver -- worth knowing before blaming the policy for the resulting mess.
    """
    root = _clone_or_skip()
    import pathlib
    ours = pathlib.Path(__file__).resolve().parents[3] / OUR_URDF_TAIL
    if not ours.is_file():
        pytest.skip(f"{ours} not present")
    sonic = _revolute_limits(root / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf")
    mine = _revolute_limits(ours)
    assert set(sonic) == set(mine)
    for name in sonic:
        assert sonic[name] == pytest.approx(mine[name], abs=1e-6), name


def test_action_scale_is_a_quarter_of_effort_over_stiffness() -> None:
    """Every entry equals ``0.25 * effort_limit_sim / stiffness``, upstream's formula."""
    for actuator in _SONIC_ACTUATORS.values():
        eff, stiff = actuator.effort_limit_sim, actuator.stiffness
        names = actuator.joint_names_expr
        eff = eff if isinstance(eff, dict) else dict.fromkeys(names, eff)
        stiff = stiff if isinstance(stiff, dict) else dict.fromkeys(names, stiff)
        for n in names:
            assert G1_SONIC_ACTION_SCALE[n] == pytest.approx(0.25 * eff[n] / stiff[n])


def test_action_scale_matches_upstream_values() -> None:
    """Spot-check against the numbers upstream's ``G1_MODEL_12_ACTION_SCALE`` produces."""
    expect = {
        ".*_hip_pitch_joint": 0.350661,
        ".*_hip_roll_joint": 0.350661,
        ".*_knee_joint": 0.350661,
        ".*_hip_yaw_joint": 0.547546,
        "waist_yaw_joint": 0.547546,
        ".*_ankle_pitch_joint": 0.438577,
        ".*_elbow_joint": 0.438577,
        ".*_wrist_pitch_joint": 0.074501,
        ".*_wrist_yaw_joint": 0.074501,
    }
    for name, value in expect.items():
        assert G1_SONIC_ACTION_SCALE[name] == pytest.approx(value, abs=1e-6), name


def test_action_scale_is_double_ours_where_the_motor_class_agrees() -> None:
    """SONIC's ``0.25 *`` vs our ``0.125 *``: twice the joint range per unit of action.

    Only checked on joints where both robots use the same motor class and effort limit -- the hip
    pitch and the waist roll/pitch deliberately differ in class, so their ratio is not 2. This is
    why the robot and the action scale must always be swapped together.
    """
    same_class = [".*_hip_roll_joint", ".*_knee_joint", ".*_hip_yaw_joint", "waist_yaw_joint",
                  ".*_ankle_pitch_joint", ".*_ankle_roll_joint", ".*_shoulder_pitch_joint",
                  ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint",
                  ".*_wrist_roll_joint"]
    for n in same_class:
        assert G1_SONIC_ACTION_SCALE[n] == pytest.approx(2.0 * G1_ACTION_SCALE[n], rel=1e-9), n


def test_patched_urdf_renames_only_the_root_link() -> None:
    """``pelvis`` becomes ``pelvis_link``; ``pelvis_contour_link`` is untouched.

    The rename is what lets the SONIC env inherit our height-scanner prim path, contact-sensor
    prim paths and CLF base frame verbatim instead of re-deriving them for a second robot.
    """
    _clone_or_skip()
    text = open(sonic_patched_urdf_path()).read()
    assert '<link name="pelvis_link">' in text
    assert '"pelvis"' not in text                        # no bare root name left anywhere
    assert '<link name="pelvis_contour_link">' in text   # the near-namesake survived
    assert "package://" not in text                      # mesh paths absolutised
