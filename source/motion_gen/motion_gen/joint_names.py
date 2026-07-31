# Copyright 2026 Zachary Olkin. All rights reserved.

"""The canonical G1 joint order, tracked body-link set, and clip channel-name grammar.

Lives on its own because it is the one piece of the (deprecated) motion-matching feature
code that the LIVE motionbricks ingest needs: ``motionbricks.joint_map`` maps a raw Motion
Bricks joint vector onto this order by NAME, and ``motionbricks.raw_schema`` documents the
contract against it.  Deliberately torch-free so both the generators and the RL side can
import it without pulling in FK.
"""

# Canonical G1 29-joint order (matches the MJCF joint names + the FM/clip joint set).  This is the
# FULL Unitree G1 body-DOF order -- identical to ``ingest.bones_ingest.CSV_JOINT_NAMES`` so the
# seed-CSV / clean / terrain ingests map straight through (their 29->this reindex is the identity).
# Was the 21-DOF subset (waist_yaw + shoulders/elbow only); widened to unlock waist roll/pitch +
# the 6 wrists.
G1_JOINT_NAMES: list[str] = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")]
)

# ---------------------------------------------------------------------------
# Root-frame body tracking
# ---------------------------------------------------------------------------

# Body links tracked in the ROOT (pelvis) frame -- the DeepMimic / Sonic style
# parameterization, where the reward grades Cartesian link error instead of joint angles.
#
# Every name here exists in BOTH the FK MJCF (``transfer/sim/robots/g1/g1_29j.xml``) and the
# Isaac articulation (``robot_assets/g1/g1_21j/g1_21j_box_merged_29j.urdf``, 30 links) -- the
# intersection is the hard ceiling, because the generator FKs from the MJCF while the RL
# measured side resolves through ``robot.body_names``.  That rules out ``head_link`` and the
# ``*_palm_link`` / finger bodies, which the MJCF has and the USD does not.
#
# The 4 clip end-effectors (``*_ankle_roll_link``, ``*_wrist_yaw_link``) ARE in this list, so
# they carry both a root-frame channel set and their existing WORLD-frame one.  That is
# deliberate: body mode swaps only ``joint:*`` out of the CLF, keeping the world-frame
# skeleton (pelvis + the 4 EE) so global foot placement stays penalised against the terrain.
#
# Pose + orientation of these 13 over-determines all 29 DoF (each distal link's 6-DoF pose
# pins its chain; elbow + shoulder_roll remove the arm's swivel redundancy), so the links
# left out here do not leave their joints unconstrained.
G1_BODY_TRACK_FRAMES: list[str] = (
    [f"{s}_{link}" for s in ("left", "right")
     for link in ("hip_roll_link", "knee_link", "ankle_roll_link")]
    + ["torso_link"]
    + [f"{s}_{link}" for s in ("left", "right")
       for link in ("shoulder_roll_link", "elbow_link", "wrist_yaw_link")]
)

# Marker appended to a body name to say "this frame's channels are expressed in the ROOT
# (pelvis) frame", e.g. ``left_knee_link@root:pos_x``.  Chosen so that:
#   * the ``frame:channel`` grammar is unchanged, so ``sagittal_reflector`` keeps mirroring by
#     the ``pos_y`` / ``ori_x`` / ``ori_z`` suffix and the ``left``/``right`` token; and
#   * the frame name is NOT a bare body name, so a consumer that has not been taught about
#     root-frame channels fails to resolve them rather than silently treating them as world.
ROOT_FRAME_MARKER = "@root"


def root_frame_name(body: str) -> str:
    """The root-frame output-frame name for ``body`` (e.g. ``"left_knee_link@root"``)."""
    return f"{body}{ROOT_FRAME_MARKER}"


def is_root_frame(name: str) -> bool:
    """Whether an output *frame* or *channel* name refers to a root-frame quantity."""
    return ROOT_FRAME_MARKER in name


def strip_root_marker(name: str) -> str:
    """The underlying body name for a root-frame frame name; a no-op for world frames."""
    return name.split(ROOT_FRAME_MARKER, 1)[0]


def output_frame_names(output_names: list[str]) -> list[str]:
    """Every frame appearing in ``output_names``, in first-seen order (no ``joint:``/``com:``).

    Args:
        output_names: Clip channel names, e.g. ``pos_output_names``.

    Returns:
        The distinct frame prefixes, root-frame markers included.
    """
    frames: list[str] = []
    for name in output_names:
        if ":" not in name or name.startswith(("joint:", "com:")):
            continue
        frame = name.split(":", 1)[0]
        if frame not in frames:
            frames.append(frame)
    return frames


def world_frame_names(output_names: list[str]) -> list[str]:
    """The WORLD-frame subset of :func:`output_frame_names`.

    The single gate every rigid clip transform must go through.  Root-frame channels are
    invariant under a rigid world transform of the clip, so rotating or translating them
    corrupts the data silently -- callers that re-datum, splice, or place a clip iterate this
    instead of every frame.

    Args:
        output_names: Clip channel names, e.g. ``pos_output_names``.

    Returns:
        The distinct frame prefixes that are expressed in the world / clip-world frame.
    """
    return [f for f in output_frame_names(output_names) if not is_root_frame(f)]


def world_position_channels(output_names: list[str], axis: str) -> list[int]:
    """Column indices of every WORLD-frame ``:pos_<axis>`` channel.

    What a clip translation must move.  Root-frame positions are pelvis-relative and so are
    unchanged by shifting the clip in the world -- adding an offset to them would corrupt the
    pose.  Use this instead of ``[i for i, n in enumerate(names) if n.endswith(":pos_x")]``.

    Args:
        output_names: Clip channel names, e.g. ``pos_output_names``.
        axis: ``"x"``, ``"y"``, or ``"z"``.

    Returns:
        Column indices, ascending.
    """
    suffix = f":pos_{axis}"
    return [i for i, n in enumerate(output_names)
            if n.endswith(suffix) and not is_root_frame(n)]
