# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-numpy port of the reference-window representation (no torch, no isaaclab).

Mirror of :mod:`robot_rl.flow_matching.window` for the torch-free deployment runtime,
parity-tested against it directly in ``transfer/deploy/tests/test_deploy_parity.py``.
See that module for what the blocks mean; this file restates only the arithmetic.

Per node::

    [base_pose(9), base_vel(6), joint_pos(nj), joint_vel(nj), ee_pos(3 * K)]

``base_pose`` is ``[R_yawT (p_ref - p_act), rot6d(R_yawT R_ref)]`` -- the reference pose in
the anchor's YAW (heading) frame, both halves the same way round. The anchor supplies heading
only, never roll or pitch, so a leaning robot does not rotate the frame its reference is
observed in; see the torch original for why.

Two maps live here:

* ``world_to_window`` -- the FORWARD map, needed for the CLIP reference path (tracker-only
  playback, where a world-frame reference is handed in).
* ``reanchor_window`` -- re-express nodes against a different anchor, needed to serve a
  prediction more than one control step after it was made (``fm_update_every > 1``).

The full inverse (``window_to_world``) stays a simulation-side diagnostic and is
deliberately NOT ported: it cannot even be completed for a block subset, since the return
leg needs every block. ``GenTrackerPolicy.predicted_base_pose_w`` ports just the
``base_pose`` case, which is all any off-sim viewer has ever needed.

All quaternions are ``(x, y, z, w)``. Unlike the torch original, every helper here
broadcasts, so the anchor needs no explicit expansion.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .quat_math_np import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
    yaw_quat,
)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """First two COLUMNS of a rotation matrix, concatenated.

    Args:
        matrix: ``[..., 3, 3]`` rotation matrices.

    Returns:
        ``[..., 6]``.
    """
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Rotation matrix from the 6D representation, via Gram-Schmidt.

    Robust to non-orthonormal input, which is the point: the generator emits a raw
    9-vector and only approximately satisfies the constraint.

    Args:
        d6: ``[..., 6]`` 6D rotation representation.

    Returns:
        ``[..., 3, 3]`` orthonormal matrices, the recovered vectors as COLUMNS.
    """
    def unit(v: np.ndarray) -> np.ndarray:
        """Normalize along the last axis, guarding a zero row."""
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

    b1 = unit(d6[..., 0:3])
    b2 = unit(d6[..., 3:6] - (b1 * d6[..., 3:6]).sum(axis=-1, keepdims=True) * b1)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """6D rotation representation of quaternions ``(x, y, z, w)``."""
    return matrix_to_rot6d(matrix_from_quat(q))


def rot6d_to_quat(d6: np.ndarray) -> np.ndarray:
    """Quaternion ``(x, y, z, w)`` from the 6D rotation representation."""
    return quat_from_matrix(rot6d_to_matrix(d6))

#: Blocks a window node packs, in order, when none are selected explicitly. Mirrors
#: ``robot_rl.flow_matching.window.REFERENCE_WINDOW_BLOCK_NAMES``.
REFERENCE_WINDOW_BLOCK_NAMES: tuple[str, ...] = (
    "base_pose", "base_vel", "joint_pos", "joint_vel", "ee_pos",
)


def resolve_reference_blocks(blocks: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a block selection and put it in canonical packing order.

    Args:
        blocks: Requested block names, or ``None`` for all of them.

    Returns:
        The selection, ordered as a window node packs it.

    Raises:
        ValueError: On an unknown name or an empty selection.
    """
    if blocks is None:
        return REFERENCE_WINDOW_BLOCK_NAMES
    wanted = set(blocks)
    unknown = wanted - set(REFERENCE_WINDOW_BLOCK_NAMES)
    if unknown:
        raise ValueError(
            f"unknown reference-window block(s) {sorted(unknown)}; "
            f"expected a subset of {list(REFERENCE_WINDOW_BLOCK_NAMES)}"
        )
    if not wanted:
        raise ValueError("a reference window needs at least one block")
    return tuple(b for b in REFERENCE_WINDOW_BLOCK_NAMES if b in wanted)


def window_blocks(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> list[tuple[str, int]]:
    """Per-node block layout ``[(name, width), ...]`` in packing order.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: Subset to include; defaults to all of them.

    Returns:
        The same table :func:`robot_rl.flow_matching.window.window_blocks` returns.
    """
    widths = {
        "base_pose": 9,   # position(3) + rot6d(6)
        "base_vel": 6,
        "joint_pos": num_joints,
        "joint_vel": num_joints,
        "ee_pos": 3 * num_ee_frames,
    }
    return [(b, widths[b]) for b in resolve_reference_blocks(blocks)]


def window_dim(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> int:
    """Width ``D`` of one window node for the given selection."""
    return sum(width for _, width in window_blocks(num_joints, num_ee_frames, blocks))


def window_block_slices(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> dict[str, slice]:
    """Column slice of each selected block within one ``[..., D]`` window node."""
    out: dict[str, slice] = {}
    offset = 0
    for name, width in window_blocks(num_joints, num_ee_frames, blocks):
        out[name] = slice(offset, offset + width)
        offset += width
    return out


#: Port of ``robot_rl.flow_matching.window.MEASURED_HISTORY_FALLBACK`` -- the node content a
#: MEASURED history row cannot fill on a robot (no odometry, no base linear-velocity
#: estimate) and takes from the last SERVED node instead.  Kept in step with that module by
#: ``deploy/tests/test_measured_history_np.py``, which compares the spans both sides produce
#: over two different block selections.
MEASURED_HISTORY_FALLBACK: tuple[str, ...] = ("base_pos", "base_lin_vel")

#: Where each :data:`MEASURED_HISTORY_FALLBACK` name sits inside its parent block.
_FALLBACK_HALVES: dict[str, tuple[str, int, int]] = {
    "base_pos": ("base_pose", 0, 3),
    "base_lin_vel": ("base_vel", 0, 3),
}


#: Port of ``robot_rl.flow_matching.window.HISTORY_ARM_SOURCE``. No ``"clip"`` entry is
#: needed on this side -- the deploy runtime has no teacher-forcing arm.
HISTORY_ARM_SOURCE: dict[str, str] = {
    "generator": "reference",
    "measured": "measured",
}


def assert_history_source_matches(history_from: str, checkpoint_source: str | None,
                                  where: str = "") -> None:
    """Refuse a serving arm the generator was not fitted for.

    Port of ``robot_rl.flow_matching.window.assert_history_source_matches``. The two history
    kinds produce rows of identical width, so a mismatch loads clean and is merely wrong --
    which is why this refuses at load rather than warning.

    Args:
        history_from: The runtime's serving arm.
        checkpoint_source: What the graph's ``fm_meta`` declares. ``None`` means a graph
            older than the flag, which is definitionally ``"reference"``.
        where: Optional context for the message (an artifact path).

    Raises:
        ValueError: On an unknown arm, or an arm the graph does not match.
    """
    want = HISTORY_ARM_SOURCE.get(history_from)
    if want is None:
        raise ValueError(
            f"history_from={history_from!r} is not one of {sorted(HISTORY_ARM_SOURCE)}.")
    got = str(checkpoint_source or "reference")
    if got != want:
        raise ValueError(
            f"history_from={history_from!r} needs a generator trained with "
            f"history_source={want!r}, but {where or 'this fm.onnx'} declares "
            f"history_source={got!r}.\n"
            "The two produce history rows of the SAME WIDTH, so this would load and run and "
            "simply feed the model a kind of history it was never fitted on. Either serve "
            f"the arm it was trained for, or point --fm_onnx at a {want!r}-trained graph.")


def measured_history_fallback_spans(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> tuple[tuple[int, int], ...]:
    """Column spans of :data:`MEASURED_HISTORY_FALLBACK` within one window node.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: Subset to include; defaults to all of them.

    Returns:
        ``((start, stop), ...)`` in packing order -- the same tuple
        :func:`robot_rl.flow_matching.window.measured_history_fallback_spans` returns.
    """
    slices = window_block_slices(num_joints, num_ee_frames, blocks)
    out = []
    for name in MEASURED_HISTORY_FALLBACK:
        parent, lo, hi = _FALLBACK_HALVES[name]
        if parent in slices:
            base = slices[parent].start
            out.append((base + lo, base + hi))
    return tuple(out)


def world_to_window(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    ref_pos_w: np.ndarray,
    ref_quat_w: np.ndarray,
    ref_lin_w: np.ndarray,
    ref_ang_w: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    ee_pos_w: np.ndarray,
    blocks: Sequence[str] | None = None,
) -> np.ndarray:
    """Build window nodes from world-frame reference state and a measured anchor.

    Args:
        root_pos_w: ``[..., 3]`` anchor position (the robot's measured base), world frame.
            Broadcast against the node axis.
        root_quat_w: ``[..., 4]`` anchor orientation (xyzw), world frame.
        ref_pos_w: ``[..., K, 3]`` reference base position per node, world frame.
        ref_quat_w: ``[..., K, 4]`` reference base orientation per node (xyzw).
        ref_lin_w: ``[..., K, 3]`` reference base linear velocity per node, world frame.
        ref_ang_w: ``[..., K, 3]`` reference base angular velocity per node, world frame.
        joint_pos: ``[..., K, nj]`` reference joint positions, **robot joint order**.
        joint_vel: ``[..., K, nj]`` reference joint velocities, robot joint order.
        ee_pos_w: ``[..., K, E, 3]`` reference end-effector positions, world frame.
        blocks: Layout blocks to emit; defaults to all of them. A dropped block is not
            computed at all.

    Returns:
        ``[..., K, D]`` window nodes.
    """
    selected = resolve_reference_blocks(blocks)
    act_p = np.asarray(root_pos_w, dtype=np.float64)
    # Yaw-only anchor, matching the torch original: heading, never roll or pitch.
    act_q = yaw_quat(np.asarray(root_quat_w, dtype=np.float64))
    ref_p = np.asarray(ref_pos_w, dtype=np.float64)
    ref_q = np.asarray(ref_quat_w, dtype=np.float64)

    def base_pose() -> np.ndarray:
        """The reference base pose in the anchor's yaw frame: ``[pos(3), rot6d(6)]``."""
        return np.concatenate([
            quat_apply_inverse(act_q, ref_p - act_p),
            quat_to_rot6d(quat_mul(np.broadcast_to(quat_conjugate(act_q), ref_q.shape),
                                   ref_q)),                        # R_yawT R_ref
        ], axis=-1)

    def base_vel() -> np.ndarray:
        """Reference base linear+angular velocity rotated into the anchor's yaw frame."""
        return np.concatenate([
            quat_apply_inverse(act_q, np.asarray(ref_lin_w, dtype=np.float64)),
            quat_apply_inverse(act_q, np.asarray(ref_ang_w, dtype=np.float64)),
        ], axis=-1)

    def ee_pos() -> np.ndarray:
        """EE positions relative to THAT node's own reference base."""
        ee = np.asarray(ee_pos_w, dtype=np.float64)
        ee_rel = quat_apply_inverse(ref_q[..., None, :], ee - ref_p[..., None, :])
        return ee_rel.reshape(*ee_rel.shape[:-2], -1)

    built = {
        "base_pose": base_pose,
        "base_vel": base_vel,
        "joint_pos": lambda: np.asarray(joint_pos, dtype=np.float64),
        "joint_vel": lambda: np.asarray(joint_vel, dtype=np.float64),
        "ee_pos": ee_pos,
    }
    return np.concatenate([built[b]() for b in selected], axis=-1)


def base_pose_block_to_world(
    block9: np.ndarray, anchor_pos: np.ndarray, anchor_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a window's ``base_pose`` block back to world coordinates.

    The inverse of :func:`world_to_window`'s ``base_pose`` case, and the only place off-sim
    that turns a window back into a world coordinate. Two callers -- the MuJoCo runner's
    viewer ribbon via :meth:`GenTrackerPolicy.predicted_base_pose_w`, and the ROS marker node
    reading ``/obelisk/g1/debug/fm_plan`` -- so a second copy of this algebra would be a
    second place for the yaw-only convention to be got wrong, in a drawing whose whole job is
    to make such a mistake visible.

    YAW-ONLY, matching how the block was built: the anchor contributes heading, so lifting
    through the full anchor rotation would tilt the whole ribbon by the robot's roll and pitch
    at the re-plan. Forward is ``quat_apply_inverse(q_yaw, p_ref - p_a)`` and
    ``rot6d(R_yawT R_ref)``, so the return leg is::

        ref_p = p_a + quat_apply(q_yaw, block[..., 0:3])
        ref_q = q_yaw * rot6d_to_quat(block[..., 3:9])

    The full inverse of a whole node is deliberately NOT ported (see the module docstring);
    this is the one block a viewer has ever needed.

    Args:
        block9: ``[..., 9]`` -- position(3) then rot6d(6).
        anchor_pos: ``[3]``, or anything broadcastable, the re-plan anchor position in world.
        anchor_quat: ``[4]`` xyzw re-plan anchor orientation. Only its yaw is used.

    Returns:
        ``([..., 3] world positions, [..., 4] xyzw world orientations)``. The quaternions come
        back sign-normalised (see :func:`quat_from_matrix`).
    """
    block9 = np.asarray(block9, dtype=np.float64)
    q = np.broadcast_to(yaw_quat(np.asarray(anchor_quat, dtype=np.float64)), block9.shape[:-1] + (4,))
    ref_p = np.asarray(anchor_pos, dtype=np.float64) + quat_apply(q, block9[..., 0:3])
    # R_ref = R_yaw (R_yawT R_ref); the 6D pair has to become a matrix here, so this is where
    # Gram-Schmidt runs (inside rot6d_to_quat).
    ref_q = quat_mul(q, rot6d_to_quat(block9[..., 3:9]))
    return ref_p, ref_q


def reanchor_window(
    window: np.ndarray,
    old_pos_w: np.ndarray,
    old_quat_w: np.ndarray,
    new_pos_w: np.ndarray,
    new_quat_w: np.ndarray,
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None = None,
) -> np.ndarray:
    """Re-express window nodes against a DIFFERENT anchor, block by block.

    Port of ``robot_rl.flow_matching.window.reanchor_window``; parity-tested against it.

    A window is relative to one measured robot pose. Serving the same nodes a few control
    steps later (``fm_update_every > 1``) means expressing them against where the robot is
    NOW, and only the blocks that reference the anchor need touching:

    * ``base_pose`` -- up to the world through the old anchor, straight back down through
      the new (the one block that genuinely round-trips);
    * ``base_vel`` -- a pure rotation ``R_newT R_old``, no position involved;
    * ``joint_pos`` / ``joint_vel`` -- frame-invariant, copied;
    * ``ee_pos`` -- relative to *that node's own* reference base, so anchor-invariant too,
      copied.

    Per block rather than through the world and back, which is what makes it work for a
    block SELECTION at all: the world round trip needs every block present on the return
    leg, and a window that drops ``joint_vel`` or ``ee_pos`` cannot supply them.

    Only the RELATIVE transform between the two anchors matters -- a common translation
    cancels and a common rotation is equivariant. Off-sim that is the useful property:
    serving at ``K > 1`` needs the base motion since the re-plan, not a global pose.

    Args:
        window: ``[..., K, D]`` window nodes anchored on the old pose.
        old_pos_w / old_quat_w: ``[..., 1, 3]`` / ``[..., 1, 4]`` anchor the nodes are in
            (or any shape that broadcasts against the node axis).
        new_pos_w / new_quat_w: The anchor to express them against.
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The selection ``window`` is packed with; defaults to all of them.

    Returns:
        ``[..., K, D]`` nodes in the new anchor's frame, same layout.
    """
    selected = resolve_reference_blocks(blocks)
    sl = window_block_slices(num_joints, num_ee_frames, selected)
    win = np.asarray(window, dtype=np.float64)
    old_p = np.asarray(old_pos_w, dtype=np.float64)
    old_q = yaw_quat(np.asarray(old_quat_w, dtype=np.float64))
    new_p = np.asarray(new_pos_w, dtype=np.float64)
    new_q = yaw_quat(np.asarray(new_quat_w, dtype=np.float64))

    out: list[np.ndarray] = []
    for name in selected:
        block = win[..., sl[name]]
        if name == "base_pose":
            # Both anchors are yaw-only, so every rotation here is a heading. Position
            # round-trips through the world; the orientation half does not need to.
            # R_yaw(new)T R_ref = (R_yaw(new)T R_yaw(old))(R_yaw(old)T R_ref), so each stored
            # column is just rotated by the anchor delta -- a PURE YAW rotation, exactly
            # linear, no Gram-Schmidt.
            ref_p = old_p + quat_apply(old_q, block[..., 0:3])
            q_delta = quat_mul(quat_conjugate(new_q), old_q)        # R_newT R_old
            out.append(np.concatenate([
                quat_apply_inverse(new_q, ref_p - new_p),
                quat_apply(q_delta, block[..., 3:6]),
                quat_apply(q_delta, block[..., 6:9]),
            ], axis=-1))
        elif name == "base_vel":
            out.append(np.concatenate([
                quat_apply_inverse(new_q, quat_apply(old_q, block[..., 0:3])),
                quat_apply_inverse(new_q, quat_apply(old_q, block[..., 3:6])),
            ], axis=-1))
        else:
            out.append(block)
    return np.concatenate(out, axis=-1)
