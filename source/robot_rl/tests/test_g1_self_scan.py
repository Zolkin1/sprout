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

from robot_rl.assets.robots.g1_21j import G1_CFG
from robot_rl.tasks.manager_based.robot_rl.g1 import g1_self_scan as ss


# The USD the G1 articulation cfg actually spawns, DERIVED from the cfg rather than
# restated. It used to be a hand-copied path guarded by a module-level
# ``skipif(not _USD_DIR.is_dir())`` -- so renaming or moving the asset made the only
# USD-vs-code guard in the repo silently PASS instead of fail, which is precisely the
# rot these tests exist to catch. There is no skip now: a missing asset is a failure
# (:func:`test_the_spawned_usd_asset_is_present`).
#
# The old comment claimed the cfg could not be imported here because it pulls in
# IsaacLab and these tests run without a simulator. That is stale: importing it needs
# isaaclab on the path, not IsaacSim running, which is already true of every other test
# in this directory.
_USD_PATH = pathlib.Path(G1_CFG.spawn.usd_path)
_USD_DIR = _USD_PATH.parent / "payloads"


def test_the_spawned_usd_asset_is_present():
    """The asset the cfg names must exist, and its absence must FAIL rather than skip.

    Every other test here re-derives the link topology from these files. If they are
    missing the derivation is vacuous, so skipping would report success for a guard that
    never ran -- the failure mode that let the hand-written prim paths rot in the first
    place.
    """
    assert _USD_PATH.is_file(), (
        f"robot_rl.assets.robots.g1_21j.G1_CFG spawns {_USD_PATH}, which does not exist. "
        "Regenerate it with robot_assets/g1/g1_21j/urdf_to_usd_conversion.py, or point the "
        "cfg at the asset you meant."
    )
    for layer in ("base.usda", "instances.usda", pathlib.Path("Physics") / "physics.usda"):
        assert (_USD_DIR / layer).is_file(), f"{_USD_DIR / layer} is missing from the payload set"


def _joint_graph() -> dict[str, str]:
    """Child-link -> parent-link, read from the USD's physics joints.

    Returns:
        The authoritative link topology.
    """
    text = (_USD_DIR / "Physics" / "physics.usda").read_text()
    # The prim root is the URDF's ``<robot name=...>``, not the file stem, so match any
    # single root rather than hard-coding one -- a renamed asset must still be parsed
    # (and then compared), never silently yield an empty graph.
    tokens = re.findall(r"physics:(body0|body1) = </[^/>]+/Geometry/([^>]+)>", text)
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


def test_every_link_is_classified_as_scanned_or_not():
    """Each link is either a ray-cast target or an explicitly excused exclusion.

    The point is the PARTITION, not either list: a link added to the robot must be
    classified deliberately instead of defaulting to invisible. Defaulting to invisible is
    exactly what happened to the hands, and it cost a debugging cycle because the renderer
    looked perfect the whole time.

    Stated against ``ss.UNSCANNED_LINKS`` rather than a literal here so the REASON each link
    is excluded lives next to the exclusion, and so this test keeps working across a robot
    change instead of being rewritten as a new list of names.
    """
    graph = _joint_graph()
    scanned, excused = set(ss.SELF_SCAN_LINKS), set(ss.UNSCANNED_LINKS)
    assert not scanned & excused, (
        f"{sorted(scanned & excused)} are both cast and excused from casting"
    )
    unclassified = set(graph) - scanned - excused
    assert not unclassified, (
        f"{sorted(unclassified)} are links in the USD that nobody classified. Add them to "
        "SELF_SCAN_LINKS if a camera can see them, or to UNSCANNED_LINKS with the reason."
    )
    assert not (scanned | excused) - set(graph), (
        f"{sorted((scanned | excused) - set(graph))} are named in the code but are not links "
        "in the USD -- the asset was regenerated without them"
    )


def test_the_limbs_are_all_scanned():
    """Nothing that can occlude a camera is quietly on the excused list.

    The partition above is satisfiable by excusing everything, so this pins the part that
    actually matters: every leg, arm and hand link casts.
    """
    for link in ss.LEG_LINKS + ss.ARM_LINKS + ss.HAND_LINKS:
        assert link in ss.SELF_SCAN_LINKS, f"{link} is a limb and must be cast"


def test_hands_are_fixed_joint_links_on_the_wrist():
    """The hands must be real links, not visuals hung off the wrist.

    As bare visuals the importer gives them a NON-IDENTITY mesh->body offset, which
    ``MultiMeshRayCaster`` silently fails to intersect -- the depth cameras then see every
    limb except the hands. The rubber hand is exposed to the same trap, which is why
    ``merge_urdf_joints.py`` is told to ``--keep`` its palm joint instead of folding it into
    the wrist; if this fails, that flag was dropped.
    """
    graph = _joint_graph()
    for link in ss.HAND_LINKS:
        assert link in graph, f"{link} is not a link; check the --keep list on the merge"
        assert graph[link] == f"{link.split('_')[0]}_wrist_yaw_link"
    text = (_USD_DIR / "Physics" / "physics.usda").read_text()
    assert text.count('def PhysicsRevoluteJoint "') == 29, "the 29-DOF action space moved"
    # Derived, not pinned: every link that is not driven by one of the 29 revolute joints is
    # welded on by a fixed one. Pinning the count meant editing this line for each new
    # bolt-on, which is churn that teaches nobody anything.
    assert text.count('def PhysicsFixedJoint "') == len(graph) - 29


def test_leaf_links_are_not_doubled():
    """A leaf link gets ONE Xform, so its target must not repeat the name.

    This is the asymmetry with links that have children, and getting it wrong fails
    quietly (the ray caster warns and drops the target).
    """
    paths = _prim_paths("base.usda")
    for link in ss._LEAF_LINKS:
        wrist = ss._PARENT[link]
        assert _has_tail(paths, (wrist, link)), f"{wrist}/{link} not in base.usda"
        assert not _has_tail(paths, (link, link)), f"{link} is unexpectedly doubled"
        assert ss.visual_prim_path(link).endswith(f"/{wrist}/{link}")


@pytest.mark.parametrize("link", ss.SELF_SCAN_LINKS)
def test_visual_prim_path_resolves_in_usd(link):
    """Each target path's chain exists in the USD, ending in the visual Xform.

    Jointed links end in a doubled tail; hands do not (see ``visual_prim_path``).
    """
    paths = _prim_paths("base.usda")
    chain = ss.link_chain(link)
    expected = chain if link in ss._LEAF_LINKS else chain + (link,)
    assert _has_tail(paths, expected), f"{'/'.join(expected)} not found in base.usda"
    assert ss.visual_prim_path(link).endswith("/".join(("",) + expected[-2:]))


@pytest.mark.parametrize("link", ss.SELF_SCAN_LINKS)
def test_visual_xform_references_its_geometry_instance(link):
    """The targeted Xform is the instanceable geometry holder, not a bare transform.

    The index fingers reuse the middle fingers' instance (identical geometry, placed by
    a different transform), so the instance they name is not always their own -- what
    matters is that SOME geometry instance is referenced, not which.
    """
    text = (_USD_DIR / "base.usda").read_text()
    instances = (_USD_DIR / "instances.usda").read_text()
    referenced = re.findall(
        rf'def Xform "{re.escape(link)}" \(\s*\n\s*instanceable = true\s*\n'
        rf'\s*prepend references = @\./instances\.usda@</Instances/([^>]+)>',
        text,
    )
    assert referenced, f"{link} does not reference a geometry instance"
    assert f"</Geometries/{referenced[0]}>" in instances


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
