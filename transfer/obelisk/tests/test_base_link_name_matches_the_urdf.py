# Copyright 2026 Zachary Olkin. All rights reserved.

"""The estimator's ``base_link_name`` must be the root link of the URDF RViz loads.

Obelisk's viz node parents the base transform to exactly that string::

    base_tf_.child_frame_id = tf_prefix + msg.base_link_name     # obelisk_viz_robot_default.h

while ``robot_state_publisher`` builds every other link's TF from the URDF's own root. If the
two disagree the tree splits in half -- ``world -> <base_link_name>`` on one side, ``<urdf
root> -> every link`` on the other, with nothing joining them. RViz then reports "No transform
from [...]" for EVERY link and draws nothing, while reporting "URDF parsed OK" one line above.
The error therefore points at the model, which is fine, and not at the string, which is not.

That is exactly what the perceptive robot did: its root is ``pelvis_link`` where the previous
one was ``pelvis``, and the estimator still said ``pelvis``. Joint count, joint order and every
link name were otherwise unchanged, so nothing else noticed.

Reads the estimator as TEXT rather than importing it: the module pulls in rclpy and the obelisk
python package, neither of which is present outside the container, and this is a
string-vs-string question that needs neither.
"""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import pytest

_OBELISK = pathlib.Path(__file__).resolve().parent.parent
_ESTIMATOR = _OBELISK / "g1_control" / "g1_control" / "estimator.py"
_LOW_LEVEL = _OBELISK / "g1_control" / "config" / "common" / "low_level.yml"
_RVIZ = _OBELISK / "g1_control" / "rviz" / "g1_deploy.rviz"


def _configured_urdf() -> pathlib.Path:
    """The URDF robot_state_publisher is told to load, from the viz block."""
    text = _LOW_LEVEL.read_text(encoding="utf-8")
    pkg = re.search(r"^\s*robot_pkg:\s*(\S+)\s*$", text, re.M)
    urdf = re.search(r"^\s*urdf:\s*(\S+)\s*$", text, re.M)
    assert pkg and urdf, "low_level.yml no longer declares robot_pkg / urdf in the viz block"
    return _OBELISK / pkg.group(1) / "urdf" / urdf.group(1)


def _urdf_root(path: pathlib.Path) -> str:
    """The link that is no joint's child."""
    root = ET.parse(path).getroot()
    links = {l.get("name") for l in root.findall("link")}
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = sorted(links - children)
    assert len(roots) == 1, f"{path.name} has {len(roots)} root links: {roots}"
    return roots[0]


def _estimator_base_link_name() -> str:
    """The literal the estimator stamps into every EstimatedState."""
    m = re.search(r'^\s*msg\.base_link_name\s*=\s*"([^"]+)"',
                  _ESTIMATOR.read_text(encoding="utf-8"), re.M)
    assert m, "estimator.py no longer assigns msg.base_link_name to a literal"
    return m.group(1)


def test_the_estimator_names_the_urdfs_root_link():
    """Otherwise the TF tree splits and RViz draws nothing, with no error naming the cause."""
    urdf = _configured_urdf()
    assert urdf.is_file(), f"low_level.yml names {urdf}, which does not exist"
    root, stamped = _urdf_root(urdf), _estimator_base_link_name()
    assert stamped == root, (
        f"estimator.py stamps base_link_name={stamped!r} but {urdf.name}'s root link is "
        f"{root!r}. Obelisk parents the base transform to the FORMER and "
        f"robot_state_publisher builds every link from the LATTER, so the TF tree would be in "
        f"two disconnected halves and RViz would report 'No transform from [...]' for every "
        f"link while still saying 'URDF parsed OK'.")


def test_the_rviz_view_tracks_a_frame_that_exists():
    """A Target Frame that is not in the URDF leaves the camera unable to follow the robot."""
    urdf = _configured_urdf()
    links = {l.get("name") for l in ET.parse(urdf).getroot().findall("link")}
    m = re.search(r"^\s*Target Frame:\s*(\S+)\s*$", _RVIZ.read_text(encoding="utf-8"), re.M)
    if m is None:
        pytest.skip("g1_deploy.rviz declares no Target Frame")
    target = m.group(1)
    # `<Fixed Frame>` is rviz's own sentinel for "use the global fixed frame", which is always
    # legitimate and is not a link name.
    if target == "<Fixed":
        pytest.skip("Target Frame is rviz's <Fixed Frame> sentinel")
    assert target in links, (
        f"g1_deploy.rviz tracks Target Frame {target!r}, which is not a link in "
        f"{urdf.name}. Links include: {sorted(links)[:5]} ...")
