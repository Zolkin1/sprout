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

Hence the doubled tail on every link with children: ``.../left_knee_link/left_knee_link``.
LEAF links are the exception -- the converter gives them a single Xform, so they are NOT
doubled; see :func:`visual_prim_path`.

The link topology below mirrors the joint graph of
``robot_assets/g1/g1_21j/g1_21j_box_merged_29j_hands`` -- 29 revolute joints plus the 16
fixed joints that weld the hands on.  ``tests/test_g1_self_scan.py`` re-derives it from that
USD and fails if the two drift apart -- the guard against a re-exported asset silently
invalidating these paths.
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
    # hands -- FIXED joints onto the wrist (see HAND_LINKS)
    **{f"{side}_rubber_hand": f"{side}_wrist_yaw_link" for side in ("left", "right")},
    # bolted to the torso: the compute backpack's shell, and the two camera mounts.  All
    # FIXED joints, all leaves.  Only the shell is a ray-cast target -- see BACKPACK_LINKS
    # and the note on the cameras in SELF_SCAN_LINKS.
    "backpack_shell_link": "torso_link",
    "camera_lower_mount_link": "torso_link",
    "camera_upper_mount_link": "torso_link",
    # Bodies with no geometry of their own: the two payload masses, and the two optical
    # frames the rays are cast FROM.  They are in the joint graph because the USD has them,
    # and out of SELF_SCAN_LINKS because there is nothing to hit -- see UNSCANNED_LINKS.
    "backpack_thor_link": "torso_link",
    "backpack_zed_link": "torso_link",
    "camera_lower_optical_link": "torso_link",
    "camera_upper_optical_link": "torso_link",
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

# The hand: ONE passive rubber hand per side, welded to ``*_wrist_yaw_link`` by a fixed
# joint.  It is a real link in the joint graph (see ``_PARENT`` above) and therefore a real
# rigid body.
#
# It has to be.  The stock URDF hung the Dex3-1's eight meshes per side off the wrist as
# extra ``<visual>`` elements, which the importer turned into bare Xforms carrying a
# NON-IDENTITY mesh->body offset -- the only such meshes in the robot.
# ``MultiMeshRayCaster`` does not intersect meshes on that path, so the depth cameras saw
# every limb except the hands while the renderer showed them correctly.  The same trap
# applies to the rubber hand, which is why ``apply_mode11_deltas.py`` makes it a link and
# ``merge_urdf_joints.py`` is told to ``--keep`` its palm joint rather than folding it into
# the wrist.
HAND_LINKS: tuple[str, ...] = tuple(f"{side}_rubber_hand" for side in ("left", "right"))

# The compute backpack's shell.  A torso-mounted camera looking forward and down cannot see
# something bolted behind the torso, so this earns its place by SYMMETRY rather than by
# being in frame: MuJoCo casts geom group 1 unconditionally, which is every per-link visual
# mesh, so a body present in one caster and absent from the other is a pixel-level
# divergence waiting for a pose that does put it in view.  Cheap to include, and it removes
# a question rather than answering it.
BACKPACK_LINKS: tuple[str, ...] = ("backpack_shell_link",)

# Links whose Xform holds the visual mesh DIRECTLY, with no inner same-named Xform.
#
# The converter emits the inner wrapper when a link needs its geometry split out -- because
# it has child links, OR because it has a COLLISION as well as a visual, which is a separate
# prim.  A link with neither gets one Xform doing both jobs, and its ray-cast target is
# undoubled (see :func:`visual_prim_path`).
#
# So the hands are leaves and ``backpack_shell_link`` is NOT, despite both being childless
# bolt-ons: the shell carries a box collider, so its geometry sits at
# ``.../backpack_shell_link/backpack_shell_link`` beside a ``box`` Cube. Read off the
# generated stage rather than reasoned about -- giving the shell a collider silently moved
# it from one convention to the other.
_LEAF_LINKS: frozenset[str] = frozenset(HAND_LINKS)

# Links a torso-mounted, downward-looking camera can actually see.  The arms are in
# frame as much as the legs are -- both cameras hang off the torso and look forward
# and down -- so they are included; the spine links (waist_*, torso) are the mount
# itself and are excluded as they cannot occlude their own camera usefully.
#
# The hands are included to MATCH the MuJoCo sim2sim caster, whose geom group 1 contains
# them: with them absent here the two disagreed by ~60/780 near-camera pixels
# (``transfer/sim/depth_camera.py``).  They are real occluders in front of a
# downward-looking chest camera, so the agreeing answer is to cast them on both sides.
#
# The CAMERA MOUNTS are deliberately absent, and that is a decision rather than an omission:
# a camera has no use for a picture of its own housing, and the two mounts are the highest
# triangle-count meshes on the robot.  MuJoCo agrees rather than diverging:
# ``make_g1_mjcf.NO_RAYCAST_BODIES`` moves their visual geoms to group 3, out of the {1, 2}
# the depth cameras cast, and ``tests/test_self_scan_matches_mujoco.py`` fails if either
# side changes alone.
SELF_SCAN_LINKS: tuple[str, ...] = LEG_LINKS + ARM_LINKS + HAND_LINKS + BACKPACK_LINKS

# Every link that is deliberately NOT a ray-cast target, with the reason it is not.
#
# Stated so that ``SELF_SCAN_LINKS`` and this together must account for every link in the
# USD (``tests/test_g1_self_scan.py``). A link added to the robot then fails the build until
# somebody classifies it, rather than defaulting to invisible -- which is the failure this
# whole module exists to prevent, and the one the hands hit.
UNSCANNED_LINKS: dict[str, str] = {
    # The mount itself. A camera cannot usefully occlude its own view.
    "waist_yaw_link": "spine, carries the cameras",
    "waist_roll_link": "spine, carries the cameras",
    "torso_link": "spine, carries the cameras",
    # No geometry to intersect: payload masses and the ray origins themselves.
    "backpack_thor_link": "mass only, no visual mesh",
    "backpack_zed_link": "mass only, no visual mesh",
    "camera_lower_optical_link": "frame only, this IS the ray origin",
    "camera_upper_optical_link": "frame only, this IS the ray origin",
    # Has geometry, deliberately not cast -- see the note on SELF_SCAN_LINKS.
    "camera_lower_mount_link": "own housing, and the heaviest mesh on the robot",
    "camera_upper_mount_link": "own housing, and the heaviest mesh on the robot",
}


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
    """Path of a link's visual-mesh Xform -- the correct ray-cast target.

    Two conventions, because the hands sit one level shallower than jointed links:

    * a JOINTED link is an outer kinematic Xform wrapping an inner, same-named
      instanceable Xform, so its target is the doubled tail
      ``.../left_knee_link/left_knee_link``;
    * a LEAF link (every hand link) has no child links, so the converter gives it a single
      Xform that is both the kinematic node and the geometry holder.  Its target is
      therefore undoubled: ``.../left_wrist_yaw_link/left_hand_palm_link``.

    Doubling a hand path would name a prim that does not exist, and the failure is quiet:
    the ray caster logs "No mesh prims found" and that part of the hand is simply absent
    from the depth image.

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
    if link in _LEAF_LINKS:
        return link_prim_path(link, root)
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
