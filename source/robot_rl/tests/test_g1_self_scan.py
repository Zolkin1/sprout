# Copyright 2026 Zachary Olkin. All rights reserved.

"""Guard the depth self-scan prim paths against G1 USD drift.

The ray-cast target expressions in
:mod:`robot_rl.tasks.manager_based.robot_rl.g1.g1_self_scan` hard-code the robot's
kinematic nesting, because ``find_matching_prims`` cannot express "any depth".  If
the USD is ever re-exported with a different hierarchy those paths silently stop
matching -- the sensor logs a warning and the robot just vanishes from its own depth
image.  That is exactly how the previous hand-written version rotted.

So these tests re-derive the topology from the asset and compare, rather than
restating it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from robot_rl.tasks.manager_based.robot_rl.g1 import g1_self_scan as ss


def _project_root() -> pathlib.Path:
    """Walk up until a directory containing ``robot_assets`` is found.

    Returns:
        The repository root.
    """
    path = pathlib.Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "robot_assets").is_dir():
            return path
        path = path.parent
    raise FileNotFoundError("Could not find project root containing 'robot_assets'")


# The USD the G1 articulation cfg actually spawns (see assets/robots/g1_21j.py).
_USD_DIR = (
    _project_root() / "robot_assets" / "g1" / "g1_21j" / "g1_21j_box_merged_29j" / "payloads"
)

pytestmark = pytest.mark.skipif(
    not _USD_DIR.is_dir(), reason=f"G1 USD payloads not present at {_USD_DIR}"
)


def _joint_graph() -> dict[str, str]:
    """Child-link -> parent-link, read from the USD's physics joints.

    Returns:
        The authoritative link topology.
    """
    text = (_USD_DIR / "Physics" / "physics.usda").read_text()
    tokens = re.findall(r"physics:(body0|body1) = </g1_21j/Geometry/([^>]+)>", text)
    graph, parent = {}, None
    for kind, path in tokens:
        leaf = path.split("/")[-1]
        if kind == "body0":
            parent = leaf
        elif parent is not None:
            graph[leaf] = parent
            parent = None
    return graph


def _prim_paths(layer: str) -> set[tuple[str, ...]]:
    """Every prim path in an ASCII USD layer, as tuples of prim names.

    Nesting is taken from indentation, which this generated asset uses consistently.

    Args:
        layer: File name of the layer inside the payloads directory.

    Returns:
        Set of prim paths.
    """
    paths: set[tuple[str, ...]] = set()
    stack: list[tuple[int, str]] = []
    pattern = re.compile(r'^(\s*)(?:def\s+\w+|over)\s+"([^"]+)"')
    for line in (_USD_DIR / layer).read_text().splitlines():
        m = pattern.match(line)
        if m is None:
            continue
        indent, name = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, name))
        paths.add(tuple(n for _, n in stack))
    return paths


def _has_tail(paths: set[tuple[str, ...]], tail: tuple[str, ...]) -> bool:
    """Whether any prim path ends with ``tail``.

    Args:
        paths: Prim paths to search.
        tail: The trailing chain to look for.

    Returns:
        True if some path ends with ``tail``.
    """
    return any(p[-len(tail):] == tail for p in paths if len(p) >= len(tail))


def test_parent_table_matches_usd_joint_graph():
    """The hard-coded link topology equals the USD's joint graph."""
    assert ss._PARENT == _joint_graph()


def test_root_link_is_the_only_parentless_link():
    """Every link except the root has a parent joint."""
    graph = _joint_graph()
    parents = set(graph.values())
    assert ss.ROOT_LINK in parents
    assert ss.ROOT_LINK not in graph


def test_self_scan_links_are_known_links():
    """Every self-scanned link exists in the USD topology."""
    graph = _joint_graph()
    for link in ss.SELF_SCAN_LINKS:
        assert link in graph, f"{link} is not a link in the USD"


def test_self_scan_covers_all_limbs():
    """The scan set is exactly the limbs -- no spine links, nothing missed.

    The spine (``waist_*``, ``torso_link``) carries the cameras themselves, so it is
    excluded; everything else that moves should be present.
    """
    graph = _joint_graph()
    spine = {"waist_yaw_link", "waist_roll_link", "torso_link"}
    assert set(ss.SELF_SCAN_LINKS) == set(graph) - spine
    assert len(ss.SELF_SCAN_LINKS) == 26


@pytest.mark.parametrize("link", ss.SELF_SCAN_LINKS)
def test_visual_prim_path_resolves_in_usd(link):
    """Each target path's chain exists in the USD, ending in the doubled visual Xform."""
    paths = _prim_paths("base.usda")
    expected = ss.link_chain(link) + (link,)
    assert _has_tail(paths, expected), f"{'/'.join(expected)} not found in base.usda"


@pytest.mark.parametrize("link", ss.SELF_SCAN_LINKS)
def test_visual_xform_references_its_geometry_instance(link):
    """The inner Xform is the instanceable geometry holder, not a bare transform."""
    text = (_USD_DIR / "base.usda").read_text()
    assert f"</Instances/{link}>" in text
    instances = (_USD_DIR / "instances.usda").read_text()
    assert f"</Geometries/{link}>" in instances


def test_root_visual_paths_resolve_in_usd():
    """The root link's geometry children exist -- it has no doubled inner Xform."""
    paths = _prim_paths("base.usda")
    for child in ss.PELVIS_GEOMETRY_CHILDREN:
        assert _has_tail(paths, (ss.ROOT_LINK, child))
    # The doubled-name convention really does not apply to the root.
    assert not _has_tail(paths, (ss.ROOT_LINK, ss.ROOT_LINK))
    with pytest.raises(ValueError):
        ss.visual_prim_path(ss.ROOT_LINK)


def test_outer_link_xform_holds_a_collider_the_inner_one_does_not():
    """Why we target the inner Xform: the outer one owns a collision ``Cube``.

    ``PRIMITIVE_MESH_TYPES`` includes ``Cube`` and the mesh predicate never checks
    ``purpose``, so a target on the outer Xform would ray-cast the guide collider
    too.
    """
    paths = _prim_paths("base.usda")
    cubes = [p for p in paths if p[-1] == "box"]
    assert cubes, "expected collision 'box' Cube prims in the asset"
    # Every collider hangs off an OUTER link Xform, never off a doubled inner one.
    for cube in cubes:
        parent, grandparent = cube[-2], cube[-3]
        assert parent != grandparent, f"collider {'/'.join(cube)} sits under an inner visual Xform"


def test_link_chain_shape():
    """Chains start at the root and end at the requested link."""
    chain = ss.link_chain("left_knee_link")
    assert chain[0] == ss.ROOT_LINK
    assert chain[-1] == "left_knee_link"
    assert chain == (
        "pelvis_link",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
    )
    assert ss.link_chain(ss.ROOT_LINK) == (ss.ROOT_LINK,)
    with pytest.raises(KeyError):
        ss.link_chain("not_a_link")


def test_prim_paths_are_unique_and_env_scoped():
    """No duplicate targets (the sensor warns and skips them) and all env-scoped."""
    paths = ss.self_scan_prim_paths()
    assert len(paths) == len(set(paths))
    assert len(paths) == len(ss.SELF_SCAN_LINKS) + len(ss.PELVIS_GEOMETRY_CHILDREN)
    for path in paths:
        assert path.startswith("{ENV_REGEX_NS}/")


def test_visual_path_is_the_link_path_plus_doubled_name():
    """The visual target is the mount path with the link name appended."""
    link = "right_ankle_roll_link"
    assert ss.visual_prim_path(link) == f"{ss.link_prim_path(link)}/{link}"
