# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 USD prim-path construction for depth-camera self-scanning.

A ``MultiMeshRayCaster`` casts against a list of target prims.  For the robot to
appear in its own depth image, every moving link must be its OWN target with
``track_mesh_transforms=True``: the sensor then re-reads that link's pose each
update and moves its mesh to match.

Two properties of the G1 USD make the target expressions fiddly enough to be worth
generating instead of hand-writing:

1. Links are nested KINEMATICALLY, so a link's prim path is its whole chain down
   from the pelvis.  ``find_matching_prims`` matches each ``/``-separated token as
   a standalone regex against a single prim's name, so ``.*`` cannot span
   hierarchy levels -- the full chain has to be spelled out.
2. Each link Xform holds BOTH an inner, same-named Xform carrying that link's own
   visual mesh AND its child links (plus a ``purpose="guide"`` collision ``Cube``).
   The ray caster gathers meshes from a target's ENTIRE subtree
   (``get_all_matching_child_prims``, no depth limit) and ``merge_prim_meshes``
   concatenates them into ONE rigid mesh moved by that one target's transform.  A
   target pointed at the OUTER link Xform therefore welds the whole downstream limb
   -- and its colliders -- into a single body frozen at its spawn pose, so e.g. knee
   flexion would be invisible.  Pointing at the INNER Xform yields exactly that
   link's own visual mesh: one target, one rigid body, one transform.

Hence the doubled tail on every expression: ``.../left_knee_link/left_knee_link``.

The link topology below mirrors the joint graph of
``robot_assets/g1/g1_21j/g1_21j_box_merged_29j``.  ``tests/test_g1_self_scan.py``
re-derives it from that USD and fails if the two drift apart -- the guard against a
re-exported asset silently invalidating these paths.
"""

from __future__ import annotations

# Prim path of the articulation's link tree, relative to the env namespace.
DEFAULT_GEOMETRY_ROOT = "{ENV_REGEX_NS}/Robot/Geometry"

# The articulation root.  Unlike every other link it has NO inner same-named Xform;
# its geometry sits under differently-named children (see PELVIS_GEOMETRY_CHILDREN).
ROOT_LINK = "pelvis_link"

# Direct visual-geometry children of ``pelvis_link``.
PELVIS_GEOMETRY_CHILDREN: tuple[str, ...] = ("pelvis", "pelvis_contour_link")

# link -> parent link, from the USD's joint graph (physics:body0 -> physics:body1).
_PARENT: dict[str, str] = {
    # spine
    "waist_yaw_link": "pelvis_link",
    "waist_roll_link": "waist_yaw_link",
    "torso_link": "waist_roll_link",
    # left leg
    "left_hip_pitch_link": "pelvis_link",
    "left_hip_roll_link": "left_hip_pitch_link",
    "left_hip_yaw_link": "left_hip_roll_link",
    "left_knee_link": "left_hip_yaw_link",
    "left_ankle_pitch_link": "left_knee_link",
    "left_ankle_roll_link": "left_ankle_pitch_link",
    # right leg
    "right_hip_pitch_link": "pelvis_link",
    "right_hip_roll_link": "right_hip_pitch_link",
    "right_hip_yaw_link": "right_hip_roll_link",
    "right_knee_link": "right_hip_yaw_link",
    "right_ankle_pitch_link": "right_knee_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
    # left arm
    "left_shoulder_pitch_link": "torso_link",
    "left_shoulder_roll_link": "left_shoulder_pitch_link",
    "left_shoulder_yaw_link": "left_shoulder_roll_link",
    "left_elbow_link": "left_shoulder_yaw_link",
    "left_wrist_roll_link": "left_elbow_link",
    "left_wrist_pitch_link": "left_wrist_roll_link",
    "left_wrist_yaw_link": "left_wrist_pitch_link",
    # right arm
    "right_shoulder_pitch_link": "torso_link",
    "right_shoulder_roll_link": "right_shoulder_pitch_link",
    "right_shoulder_yaw_link": "right_shoulder_roll_link",
    "right_elbow_link": "right_shoulder_yaw_link",
    "right_wrist_roll_link": "right_elbow_link",
    "right_wrist_pitch_link": "right_wrist_roll_link",
    "right_wrist_yaw_link": "right_wrist_pitch_link",
}

LEG_LINKS: tuple[str, ...] = tuple(
    f"{side}_{joint}_link"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)

ARM_LINKS: tuple[str, ...] = tuple(
    f"{side}_{joint}_link"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
)

# Links a torso-mounted, downward-looking camera can actually see.  The arms are in
# frame as much as the legs are -- both cameras hang off the torso and look forward
# and down -- so they are included; the spine links (waist_*, torso) are the mount
# itself and are excluded as they cannot occlude their own camera usefully.
SELF_SCAN_LINKS: tuple[str, ...] = LEG_LINKS + ARM_LINKS


def link_chain(link: str) -> tuple[str, ...]:
    """Prim-name chain from the articulation root down to ``link``, inclusive.

    Args:
        link: Name of the link, e.g. ``"left_knee_link"``.

    Returns:
        The chain, root first, e.g.
        ``("pelvis_link", "left_hip_pitch_link", ..., "left_knee_link")``.

    Raises:
        KeyError: If ``link`` is not a known link.
    """
    if link == ROOT_LINK:
        return (ROOT_LINK,)
    if link not in _PARENT:
        raise KeyError(f"Unknown G1 link {link!r}. Known: {sorted(_PARENT) + [ROOT_LINK]}")
    chain: list[str] = [link]
    while chain[0] != ROOT_LINK:
        chain.insert(0, _PARENT[chain[0]])
    return tuple(chain)


def link_prim_path(link: str, root: str = DEFAULT_GEOMETRY_ROOT) -> str:
    """Path of a link's OUTER (kinematic) Xform.

    This is the correct prim for MOUNTING a sensor, and the wrong prim to ray-cast
    against -- see the module docstring.

    Args:
        link: Name of the link.
        root: Prim path of the link tree.

    Returns:
        The absolute prim path of the link's kinematic Xform.
    """
    return "/".join((root,) + link_chain(link))


def visual_prim_path(link: str, root: str = DEFAULT_GEOMETRY_ROOT) -> str:
    """Path of a link's INNER visual-mesh Xform -- the correct ray-cast target.

    Args:
        link: Name of the link.
        root: Prim path of the link tree.

    Returns:
        The absolute prim path holding exactly this link's own visual mesh.

    Raises:
        ValueError: If ``link`` is the root, which has no inner same-named Xform.
    """
    if link == ROOT_LINK:
        raise ValueError(
            f"{ROOT_LINK!r} has no inner same-named visual Xform; "
            "use root_visual_prim_paths() instead."
        )
    return f"{link_prim_path(link, root)}/{link}"


def root_visual_prim_paths(root: str = DEFAULT_GEOMETRY_ROOT) -> tuple[str, ...]:
    """Ray-cast target paths for the root link's geometry.

    Args:
        root: Prim path of the link tree.

    Returns:
        One path per direct visual child of the root link.
    """
    base = link_prim_path(ROOT_LINK, root)
    return tuple(f"{base}/{child}" for child in PELVIS_GEOMETRY_CHILDREN)


def self_scan_prim_paths(
    links: tuple[str, ...] = SELF_SCAN_LINKS,
    root: str = DEFAULT_GEOMETRY_ROOT,
    include_root_link: bool = True,
) -> tuple[str, ...]:
    """Every ray-cast target path needed to see the robot's own body.

    Args:
        links: Links to include, defaulting to :data:`SELF_SCAN_LINKS`.
        root: Prim path of the link tree.
        include_root_link: Whether to add the root link's geometry.  A mid-torso
            downward camera sees the pelvis, so this defaults to True.

    Returns:
        Absolute prim paths, one per rigid body whose mesh must move independently.
    """
    paths = tuple(visual_prim_path(link, root) for link in links)
    if include_root_link:
        paths += root_visual_prim_paths(root)
    return paths
