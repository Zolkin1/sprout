# Copyright 2026 Zachary Olkin. All rights reserved.

"""The sim's depth-camera mounts and the robot description must state ONE geometry.

Two files carry the camera pose and neither reads the other:

* ``g1_depth_env_cfg._NEAR`` / ``_FAR`` -- what IsaacLab casts rays from, and therefore
  what everything downstream inherits: ``mdp.depth_camera_specs`` derives the deploy
  contract from it, which reaches ``tracker.onnx``, ``fm.onnx`` and MuJoCo;
* ``robot_assets/g1_perceptive/g1_perceptive.urdf`` -- the ``*_optical_link`` frames, which
  the viser check and (once regenerated) the USD and MJCF render.

A disagreement between them is invisible at runtime: the sim would cast rays from one place
while every picture of the robot showed the camera somewhere else, and the mismatch would
only surface as a registration error the generator cannot explain. Both are stated in the
``torso_link`` frame from the same CAD, so they should be equal to the micron.

Deliberately NOT a check that either is RIGHT -- that is what the STL-datum report in
``make_perceptive_urdf.py`` and a rendered frame are for. This only pins them together.
"""

from __future__ import annotations

import math
import pathlib
import xml.etree.ElementTree as ET

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
URDF_PATH = _REPO_ROOT / "robot_assets" / "g1_perceptive" / "g1_perceptive.urdf"

#: env-cfg spec -> the URDF joint that must agree with it.
PAIRS = (("_NEAR", "camera_lower_optical_joint"), ("_FAR", "camera_upper_optical_joint"))

#: The body both sides measure from.
MOUNT_BODY = "torso_link"

pytest.importorskip("isaaclab", reason="the env cfg imports IsaacLab")


@pytest.fixture(scope="module")
def urdf_joints() -> dict[str, ET.Element]:
    """Every joint in the perceptive URDF, by name."""
    if not URDF_PATH.is_file():
        pytest.fail(
            f"{URDF_PATH} is missing. It is generated -- run "
            "robot_assets/g1_perceptive/make_perceptive_urdf.py. Skipping instead of "
            "failing would let the sim and the description drift apart unnoticed, which is "
            "the whole thing this file exists to prevent."
        )
    return {j.get("name"): j for j in ET.parse(URDF_PATH).getroot().findall("joint")}


@pytest.fixture(scope="module")
def specs():
    """``_NEAR`` and ``_FAR`` from the env cfg, by attribute name."""
    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_depth_env_cfg as cfg
    return {name: getattr(cfg, name) for name, _ in PAIRS}


@pytest.mark.parametrize("spec_name,joint_name", PAIRS)
def test_mount_position_agrees(spec_name, joint_name, specs, urdf_joints):
    """The ray origin and the URDF's optical frame are the same point."""
    joint = urdf_joints[joint_name]
    assert joint.find("parent").get("link") == MOUNT_BODY, (
        f"{joint_name} hangs off {joint.find('parent').get('link')}, but the sensor offset "
        f"in the env cfg is measured from {MOUNT_BODY} -- the two are no longer comparable."
    )
    urdf_xyz = [float(v) for v in joint.find("origin").get("xyz").split()]
    cfg_xyz = list(specs[spec_name].pos)
    for axis, (a, b) in enumerate(zip(cfg_xyz, urdf_xyz)):
        assert a == pytest.approx(b, abs=1e-9), (
            f"{spec_name}.pos[{axis}] is {a} but {joint_name} says {b} "
            f"({abs(a - b) * 1000:.3f} mm apart)"
        )


@pytest.mark.parametrize("spec_name,joint_name", PAIRS)
def test_mount_pitch_agrees(spec_name, joint_name, specs, urdf_joints):
    """The downward pitch agrees, and only pitch is used.

    Checked as roll/pitch/yaw rather than as one angle: the env cfg builds its quaternion
    about y alone, so a URDF that grew a roll or yaw would still pass a pitch-only
    comparison while aiming the real camera somewhere else.

    The tolerance is 1e-6 deg rather than exact because the URDF stores the angle in
    RADIANS at nine significant figures, so 65 deg round-trips as 64.99999978. That is
    4 um of lateral error at the far camera's 12 m range -- far below anything the sensor
    model resolves, and not worth widening the file's precision for.
    """
    urdf_pitch_tol_deg = 1e-6
    roll, pitch, yaw = (float(v) for v in urdf_joints[joint_name].find("origin").get("rpy").split())
    assert roll == pytest.approx(0.0, abs=1e-12), f"{joint_name} has a roll of {roll} rad"
    assert yaw == pytest.approx(0.0, abs=1e-12), f"{joint_name} has a yaw of {yaw} rad"
    assert math.degrees(pitch) == pytest.approx(specs[spec_name].pitch_deg,
                                                abs=urdf_pitch_tol_deg)


def test_the_pair_straddles_the_midplane(specs):
    """Each camera's pose is its LEFT imager, so both sit off-centre on the SAME side.

    A stereo module reports depth in its left camera's frame, which is half a baseline
    off the mount centre. Zero here would mean somebody had "tidied up" the y back to the
    centreline and moved the ray origin by 25-60 mm.
    """
    for name, _joint in PAIRS:
        y = specs[name].pos[1]
        assert y > 0.0, (
            f"{name}.pos[1] is {y}: the pose should be the LEFT imager, which is +y on this "
            "robot, not the mount centre."
        )
