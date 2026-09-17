# Copyright 2026 Zachary Olkin. All rights reserved.

"""Every mesh the viewers draw must hang off its own link, not off that link's parent.

``ViserUrdf`` iterates ``scene.geometry.items()`` and uses each geometry KEY as if it were the
name of the graph NODE holding it. yourdfpy keys geometry by the ``<visual name=...>``
attribute, and trimesh keeps node names unique -- so a visual named after its own link gets the
bare name as its geometry key and a suffixed one as its node. viser then reads the LINK's
parent, bakes the fixed-joint transform into the vertices, and parents the result under that
same joint frame: the mount transform lands twice and the part floats off the robot by it.

``g1_perceptive.urdf`` names exactly three visuals that way -- both camera mounts and the
backpack shell. Nothing is malformed: all 39 meshes load, the joint tree is right, Isaac and
MuJoCo place all three correctly, and the only symptom is three parts hanging in space. So this
pins the loader's fix rather than the file, which is fingerprinted into every exported
``tracker.onnx`` / ``fm.onnx`` and must not be edited for a rendering bug.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("yourdfpy", reason="the viser viewers' URDF loader")

from robot_rl.flow_matching.viz.robot import (  # noqa: E402
    DEFAULT_URDF,
    _disambiguate_geometry_names,
    load_urdf,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
URDF_PATH = _REPO_ROOT / DEFAULT_URDF

#: The three parts bolted on by ``make_perceptive_urdf.py``, and the link each must ride.
BOLTED_ON = ("camera_lower_mount_link", "camera_upper_mount_link", "backpack_shell_link")


@pytest.fixture(scope="module")
def urdf():
    """The perceptive G1, loaded the way every viser viewer loads it."""
    if not URDF_PATH.exists():
        pytest.skip(f"{URDF_PATH} is absent")
    return load_urdf(str(URDF_PATH))


def test_every_geometry_key_is_its_own_graph_node(urdf):
    """The invariant ViserUrdf assumes and never checks."""
    assert set(urdf.scene.geometry) == set(urdf.scene.graph.nodes_geometry)


def test_bolted_on_parts_hang_off_their_own_link(urdf):
    """Each mount/backpack mesh must be parented to its link, not to the torso.

    This is the failure itself: with the parent one level too high, the mesh is posed by the
    mount's fixed joint twice.
    """
    parents = urdf.scene.graph.transforms.parents
    for link in BOLTED_ON:
        meshes = [g for g in urdf.scene.geometry if parents[g] == link]
        assert len(meshes) == 1, f"{link} should own exactly one mesh node, got {meshes}"


def test_the_parts_sit_on_the_robot(urdf):
    """A placement check with a number, so a future regression is caught by distance too.

    Each mesh's centroid must land within 20 cm of its link ORIGIN -- the parts are 7-25 cm
    across and their STL origins sit on the imager / mount face, so the true gaps are a few
    centimetres. The broken placement misses by a whole extra mount transform: 0.112 m on the
    lower camera, 0.293 m on the upper one and 0.153 m on the backpack.
    """
    urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
    parents = urdf.scene.graph.transforms.parents
    for link in BOLTED_ON:
        key = next(g for g in urdf.scene.geometry if parents[g] == link)
        mesh = urdf.scene.geometry[key].copy()
        mesh.apply_transform(urdf.scene.graph.get(key)[0])
        link_origin = urdf.scene.graph.get(link)[0][:3, 3]
        gap = float(np.linalg.norm(mesh.centroid - link_origin))
        assert gap < 0.20, f"{link}'s mesh sits {gap:.3f} m from its link origin"


def test_disambiguation_leaves_a_clean_urdf_alone():
    """A URDF whose visuals are unnamed needs no rename, and must not get one.

    The unnamed case is the norm (all 36 inherited G1 visuals), and its trimesh-derived keys
    -- ``pelvis.STL`` and friends -- are what the viser scene tree shows. Renaming those would
    be churn for nothing.
    """
    import yourdfpy

    robot = yourdfpy.Robot(
        name="two_links",
        links=[
            yourdfpy.Link(
                name="a",
                visuals=[yourdfpy.Visual(geometry=yourdfpy.Geometry(box=yourdfpy.Box((1, 1, 1))))],
            ),
            yourdfpy.Link(name="b"),
        ],
        joints=[yourdfpy.Joint(name="a_to_b", type="fixed", parent="a", child="b")],
    )
    assert _disambiguate_geometry_names(robot) == 0
    assert robot.links[0].visuals[0].name is None


def test_disambiguation_renames_a_visual_named_after_its_link():
    """The actual collision, on a two-link toy: the name must move off the link's."""
    import yourdfpy

    robot = yourdfpy.Robot(
        name="clash",
        links=[
            yourdfpy.Link(name="a"),
            yourdfpy.Link(
                name="b",
                visuals=[
                    yourdfpy.Visual(
                        name="b",
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box((1, 1, 1))),
                    )
                ],
            ),
        ],
        joints=[yourdfpy.Joint(name="a_to_b", type="fixed", parent="a", child="b")],
    )
    assert _disambiguate_geometry_names(robot) == 1
    assert robot.links[1].visuals[0].name not in {"a", "b"}
