# Copyright 2026 Zachary Olkin. All rights reserved.

"""The ROS URDF must still be the TRAINED robot.

``g1_model/urdf/g1_29j.urdf`` was byte-identical on main, the deploy branch and the
training-refresh branch. Training moved to the perceptive G1 -- mode_11 waist offsets, a
2.64 kg backpack, rubber hands -- and the deploy stack went on FK'ing the old model. Nothing
errored: ``torso_link`` and all 29 joints exist in both files, so ``g1_deploy::CameraPoseFk``
resolved every name it looks up and placed both depth cameras exactly 10.0 mm high, on every
unprojected xyz channel. A wrong camera pose is a plausible cloud in the wrong place, so no
image inspection could have found it.

The fix was to stop keeping a copy: the ROS URDF is generated from the training URDF by
``g1_model/urdf/make_ros_urdf.py``. These tests are what make that stick -- that the committed
file is what the generator produces today, and that it agrees with the training URDF about
where the cameras are.

No ROS, no isaaclab. Pinocchio is required for the kinematic half and skipped if absent.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

#: The obelisk tree, then the repo root. Derived from this file's own position via the
#: `tests` -> `obelisk` -> `transfer` -> root chain by NAME rather than by a count of ".."
#: hops, so moving this file fails loudly instead of resolving somewhere plausible.
_TESTS = pathlib.Path(__file__).resolve().parent
_OBELISK = _TESTS.parent
_REPO = _OBELISK.parent.parent
assert _OBELISK.name == "obelisk" and _REPO.joinpath("robot_assets").is_dir(), (
    f"unexpected layout: {_OBELISK} / {_REPO}")

_SOURCE_URDF = _REPO / "robot_assets" / "g1_perceptive" / "g1_perceptive.urdf"
_ROS_URDF = _OBELISK / "g1_model" / "urdf" / "g1_perceptive.urdf"
_GENERATOR = _OBELISK / "g1_model" / "urdf" / "make_ros_urdf.py"

#: The invocation the committed file is expected to have come from -- repo-root-relative, as
#: the generator's docstring specifies. The banner echoes this string, so it is part of the
#: expected output.
_SOURCE_ARG = "robot_assets/g1_perceptive/g1_perceptive.urdf"


def _load_generator():
    """Import ``make_ros_urdf`` by path; it lives in a ROS package, not a python one."""
    spec = importlib.util.spec_from_file_location("make_ros_urdf", _GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_ros_urdf_is_what_the_generator_produces(tmp_path):
    """Regenerating from the training URDF reproduces the committed file exactly.

    Catches both halves of the drift: a stale ROS URDF whose source has since moved, and a
    hand edit to the generated file. Either is invisible at run time -- the deploy stack loads
    whatever is there and FKs it without complaint.
    """
    gen = _load_generator()
    out = tmp_path / "regenerated.urdf"
    # The REAL mesh directory, so nothing is copied: every mesh the source names is already in
    # the package, and this test is about the XML, not about moving 30 MB of STLs.
    gen.rewrite(_SOURCE_ARG, str(out), str(_OBELISK / "g1_model" / "meshes"))
    assert out.read_text(encoding="utf-8") == _ROS_URDF.read_text(encoding="utf-8"), (
        f"{_ROS_URDF.name} is not what make_ros_urdf.py produces from {_SOURCE_ARG}. Either "
        "the training URDF moved and this was not regenerated, or someone edited the "
        "generated file. Re-run:\n"
        f"    python transfer/obelisk/g1_model/urdf/make_ros_urdf.py --urdf {_SOURCE_ARG} "
        f"--out transfer/obelisk/g1_model/urdf/{_ROS_URDF.name}")


def test_every_mesh_the_ros_urdf_names_is_in_the_package():
    """A ``package://`` path RViz cannot resolve draws a robot with a piece missing, silently."""
    import re

    text = _ROS_URDF.read_text(encoding="utf-8")
    refs = set(re.findall(r'filename="package://g1_model/meshes/([^"]+)"', text))
    assert refs, "no package:// mesh references at all -- the rewrite did not happen"
    missing = sorted(r for r in refs
                     if not (_OBELISK / "g1_model" / "meshes" / r).is_file())
    assert not missing, f"referenced but absent from g1_model/meshes: {missing}"
    # And nothing escaped the rewrite.
    stray = re.findall(r'filename="(?!package://)([^"]+)"', text)
    assert not stray, f"mesh references that are not package:// : {sorted(set(stray))}"


def test_the_ros_urdf_places_the_cameras_where_the_training_urdf_does():
    """The kinematic half: same mount placement, to floating point.

    This is the property the 10 mm bug violated, asserted against the model training actually
    used rather than against a number written down here -- a literal would have agreed with
    the old robot just as happily.
    """
    pin = pytest.importorskip("pinocchio")

    #: The shipped pair's mount body and nominal offsets (g1_depth_env_cfg._NEAR / _FAR).
    #: Only the MOUNT's placement can differ between the two files; the offsets come from the
    #: artifact and are identical either way, so any difference here is the robot, not the
    #: contract.
    mount = "torso_link"
    offsets = [np.array([0.107, 0.025, 0.045]), np.array([0.085, 0.060, 0.285])]

    def cam_positions(urdf: str) -> list[np.ndarray]:
        model = pin.buildModelFromUrdf(urdf)
        data = model.createData()
        pin.forwardKinematics(model, data, np.zeros(model.nq))
        pin.updateFramePlacements(model, data)
        assert model.existFrame(mount), f"{urdf} has no {mount}"
        root_to_mount = (data.oMf[model.getFrameId(model.frames[1].name)].inverse()
                         * data.oMf[model.getFrameId(mount)])
        return [root_to_mount.act(o) for o in offsets]

    ros = cam_positions(str(_ROS_URDF))
    src = cam_positions(str(_SOURCE_URDF))
    for i, (a, b) in enumerate(zip(ros, src)):
        # Exact: the ROS copy is a mesh-path rewrite, so there is no arithmetic to lose
        # precision in. A non-zero difference here means it is a different robot.
        assert np.allclose(a, b, atol=0.0, rtol=0.0), (
            f"camera {i} sits {np.linalg.norm(a - b) * 1e3:.4f} mm apart between "
            f"{_ROS_URDF.name} and the training URDF")
