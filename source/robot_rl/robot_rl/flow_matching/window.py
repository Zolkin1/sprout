# Copyright 2026 Zachary Olkin. All rights reserved.

"""The reference-window state representation -- the one contract three subsystems share.

A *window node* is one timestep of reference, expressed relative to a measured robot pose
(the **anchor**). Per node, in packing order::

    [base_pose(6), base_vel(6), joint_pos(nj), joint_vel(nj), ee_pos(3 * K)]

with

* ``base_pose[0:3]`` -- ``R_actT (p_ref - p_act)``, the reference base position relative to
  the anchor, in the anchor's base frame;
* ``base_pose[3:6]`` -- ``R_actT Log(R_act R_refT) = Log(R_refT R_act)``, the reference base
  orientation error;
* ``base_vel`` -- the reference base linear+angular velocity rotated into the anchor frame;
* ``joint_pos`` / ``joint_vel`` -- frame-invariant, in the **robot's** joint order;
* ``ee_pos`` -- each end-effector position relative to **that node's own reference base**
  (posture, free of tracking error), 3 per frame.

.. admonition:: The two halves of ``base_pose`` disagree, deliberately, for now

    Position is ``ref - actual`` in the ROBOT's frame; orientation is ``actual - ref`` in the
    REFERENCE's frame. So the 6-vector is not one error vector -- it is two, pointing opposite
    ways in different frames. This is a known defect, kept **only** because flipping it
    silently invalidates every trained policy: the observation width does not change, so a
    stale teacher loads clean and steers wrong.

    Fixing it is a one-line change here (``quat_box_minus(ref_q, act_q)``) plus the matching
    line in :func:`window_to_world` (``quat_box_plus`` loses its ``-``), the numpy port in
    ``transfer/deploy/window_np.py``, ``mdp.base_pose_error`` /
    ``mdp.tracking_error_global``, the angular finite-difference ordering in
    ``flow.frame_consistency_terms``, and :data:`BASE_POSE_CONVENTION` below -- and then a
    TEACHER RETRAIN. Do it as its own change; ``BASE_POSE_CONVENTION`` travels into the
    exported artifacts so the two generations cannot be mixed by accident.

Everything here is relative, so no world coordinate survives into a window. The anchor is
divided out on the way in and only reappears if you deliberately call
:func:`window_to_world`.

Three producers share this module so the representation cannot drift between them:

1. ``mdp.observations.reference_window`` -- the policy observation, anchored on the robot now;
2. ``flow_matching.dataset`` -- the FM's training target *and* its autoregressive history,
   built offline from recorded world-frame arrays;
3. ``transfer.deploy.window_np`` -- the torch-free port, parity-tested against this one.

Because the FM predicts in exactly this representation, its output *is* the observation: no
world lift, in sim or at deploy.

.. warning::
    ``base_pose[3:6]`` is a ``quat_box_minus`` log-map value, so its magnitude saturates at
    ``pi``. A reference that rotates more than ``pi`` rad relative to the anchor over the
    horizon **wraps**, putting a jump discontinuity in both the FM's regression target and
    the policy's observation. The dataset build guards against it; see
    ``flow_matching.dataset``.

All quaternions are ``(x, y, z, w)``, matching ``robot.data`` and ``isaaclab.utils.math``.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_box_plus,
)

#: Direction and frame of ``base_pose``'s two difference terms, carried into the exported
#: tracker parameters and checked at deploy load. A change here is INVISIBLE in the
#: observation's shape, so this string is the only thing standing between a stale policy and
#: a clean-looking load. Bump it in the same commit as the maths. See the module docstring.
BASE_POSE_CONVENTION = "legacy_mixed_pos_ref_minus_actual_ori_actual_minus_ref"


#: Per-block uniform-noise HALF-WIDTHS over one window node, in RAW units.
#:
#: ONE table, used in two places that model the SAME physical quantity -- the generator's
#: prediction error:
#:
#: * ``mdp.reference_window``'s observation noise, which corrupts what the policy sees so
#:   it does not learn to trust a perfect reference;
#: * the FM's own autoregressive HISTORY noise (``flow_matching.dataset``), which corrupts
#:   the past nodes it conditions on.
#:
#: They were separate tables until the two sides came to speak the same representation, and
#: they had drifted by up to 10x (joint_pos 0.1 vs 0.01). Keeping them separate is not a
#: tuning knob, it is a bug waiting to happen: at deploy the history IS the model's own past
#: output, so the error corrupting it is by definition the error the observation noise
#: stands in for. Scale the FM side as a whole with ``FlowMatchingConfig.noise_scale`` if you
#: want to ablate the magnitude; do not fork the shape.
#:
#: One deliberate conservatism: this is a blanket over ALL window offsets, while the history
#: only ever carries node 0, whose error is the smallest of the horizon. So the history is
#: noised slightly harder than reality -- the safe direction.
#:
#: Position-like entries are per-axis ``(x, y, z)``; the vertical axis is tighter.
REFERENCE_WINDOW_NOISE: dict[str, float | list[float]] = {
    "base_pos": [0.05, 0.05, 0.01],   # reference base position error, metres
    "base_ori": 0.03,                 # reference base orientation error, radians
    "base_lin_vel": 0.25,             # m/s
    "base_ang_vel": 0.25,             # rad/s
    "joint_pos": 0.1,                 # rad
    "joint_vel": 0.5,                 # rad/s
    "ee_pos": [0.05, 0.05, 0.01],     # ee position rel. reference base, metres
}


def expand_window_blocks(
    num_joints: int,
    num_ee_frames: int,
    per_block: dict[str, float | list[float]],
) -> list[float]:
    """Expand per-block values to one value per channel of ONE window node.

    Shared by everything that needs a per-channel vector over the layout -- the
    observation's noise and scale, the FM's history noise -- so a layout change cannot land
    in one and not the others.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        per_block: Value per block name; position-like blocks (``base_pos``, ``base_ori``,
            ``base_lin_vel``, ``base_ang_vel``, ``ee_pos``) accept a scalar or an
            ``(x, y, z)`` triple.

    Returns:
        ``D`` values, in the channel order the node packs.

    Raises:
        ValueError: If the built width disagrees with the block layout.
    """
    def axes(key: str) -> list[float]:
        """The block's value as three per-axis entries."""
        value = per_block[key]
        return [float(v) for v in value] if isinstance(value, (tuple, list)) else [float(value)] * 3

    out = (
        axes("base_pos") + axes("base_ori")                        # base_pose  (6)
        + axes("base_lin_vel") + axes("base_ang_vel")              # base_vel   (6)
        + [float(per_block["joint_pos"])] * num_joints             # joint_pos  (nj)
        + [float(per_block["joint_vel"])] * num_joints             # joint_vel  (nj)
        + axes("ee_pos") * num_ee_frames                           # ee_pos     (3K)
    )
    expected = window_dim(num_joints, num_ee_frames)
    if len(out) != expected:
        raise ValueError(
            f"Built {len(out)} channels, expected {expected} (nj={num_joints}, "
            f"ee={num_ee_frames}). The per-block expansion has drifted from window_blocks."
        )
    return out


def window_blocks(num_joints: int, num_ee_frames: int) -> list[tuple[str, int]]:
    """Per-node block layout as ``[(name, width), ...]`` in packing order.

    The single source of truth for how one window node is packed. Everything that slices a
    window -- the observation's noise/scale expansion, the FM's normalizer, the ONNX layout
    guard, the diagnostics -- derives its offsets from here rather than restating them.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames the clip library exposes ``K``.

    Returns:
        ``[("base_pose", 6), ("base_vel", 6), ("joint_pos", nj), ("joint_vel", nj),
        ("ee_pos", 3 * K)]``.
    """
    return [
        ("base_pose", 6),
        ("base_vel", 6),
        ("joint_pos", num_joints),
        ("joint_vel", num_joints),
        ("ee_pos", 3 * num_ee_frames),
    ]


def window_dim(num_joints: int, num_ee_frames: int) -> int:
    """Width ``D`` of one window node.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.

    Returns:
        ``12 + 2 * nj + 3 * K``.
    """
    return sum(width for _, width in window_blocks(num_joints, num_ee_frames))


def window_block_slices(num_joints: int, num_ee_frames: int) -> dict[str, slice]:
    """Column slice of each block within one window node.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.

    Returns:
        Block name -> ``slice`` into the last axis of a ``[..., D]`` window.
    """
    out: dict[str, slice] = {}
    offset = 0
    for name, width in window_blocks(num_joints, num_ee_frames):
        out[name] = slice(offset, offset + width)
        offset += width
    return out


def split_window(
    window: torch.Tensor, num_joints: int, num_ee_frames: int
) -> dict[str, torch.Tensor]:
    """Split a ``[..., D]`` window into its named blocks (views, not copies).

    Args:
        window: Window nodes, last axis ``D``.
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.

    Returns:
        Block name -> ``[..., width]`` view.

    Raises:
        ValueError: If ``window``'s last axis disagrees with the layout.
    """
    expected = window_dim(num_joints, num_ee_frames)
    if window.shape[-1] != expected:
        raise ValueError(
            f"window is {window.shape[-1]} wide but the layout implies {expected} "
            f"(nj={num_joints}, ee={num_ee_frames})."
        )
    return {
        name: window[..., slc]
        for name, slc in window_block_slices(num_joints, num_ee_frames).items()
    }


def _broadcast_anchor(anchor: torch.Tensor, node_shape: torch.Size) -> torch.Tensor:
    """Expand an anchor to one entry per node and flatten to ``(-1, width)``.

    ``isaaclab.utils.math`` reshapes both of its arguments to 2-D and views the result back
    to the *vector's* shape, so it does not broadcast -- the anchor has to be materialised
    per node before it goes in. Accepts an anchor carrying the node axis already, or one
    with that axis of size 1 (or absent) to be repeated across nodes.

    Args:
        anchor: ``[..., width]``, broadcastable against ``node_shape + (width,)``.
        node_shape: Leading shape of the node-indexed inputs, e.g. ``(N, K)``.

    Returns:
        ``(prod(node_shape), width)``.
    """
    width = anchor.shape[-1]
    return torch.broadcast_to(anchor, tuple(node_shape) + (width,)).reshape(-1, width)


def world_to_window(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ref_pos_w: torch.Tensor,
    ref_quat_w: torch.Tensor,
    ref_lin_w: torch.Tensor,
    ref_ang_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ee_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Build window nodes from world-frame reference state and a measured anchor.

    The forward map, shared by the live observation, the offline dataset build and (in its
    numpy port) the deploy runtime. Node-indexed arguments share a leading shape
    ``B = (..., K)``; the anchor may carry that shape, carry it with the node axis of size
    ``1``, or omit the node axis entirely -- it is broadcast either way. That is what lets
    one call serve both the FM's target (one anchor over ``F`` future nodes) and its
    autoregressive history (a *per-row* anchor over ``N`` rows).

    Args:
        root_pos_w: ``[..., 3]`` anchor position (the measured robot base), world frame.
        root_quat_w: ``[..., 4]`` anchor orientation (xyzw), world frame.
        ref_pos_w: ``[..., K, 3]`` reference base position per node, world frame.
        ref_quat_w: ``[..., K, 4]`` reference base orientation per node (xyzw), world frame.
        ref_lin_w: ``[..., K, 3]`` reference base linear velocity per node, world frame.
        ref_ang_w: ``[..., K, 3]`` reference base angular velocity per node, world frame.
        joint_pos: ``[..., K, nj]`` reference joint positions, **robot joint order**.
        joint_vel: ``[..., K, nj]`` reference joint velocities, robot joint order.
        ee_pos_w: ``[..., K, E, 3]`` reference end-effector positions, world frame.

    Returns:
        ``[..., K, D]`` window nodes, ``D = window_dim(nj, E)``.
    """
    node_shape = ref_pos_w.shape[:-1]                        # (..., K)
    flat = math.prod(node_shape)

    act_p = _broadcast_anchor(root_pos_w, node_shape)         # (flat, 3)
    act_q = _broadcast_anchor(root_quat_w, node_shape)        # (flat, 4)
    ref_p = ref_pos_w.reshape(flat, 3)
    ref_q = ref_quat_w.reshape(flat, 4)

    # base_pose. NOTE the two halves disagree -- see the module docstring; this is the
    # legacy convention, kept until the teacher can be retrained.
    pos_err_b = quat_apply_inverse(act_q, ref_p - act_p)      # ref - actual, robot frame
    ori_err_w = quat_box_minus(act_q, ref_q)                  # Log(R_act R_refT)
    ori_err_b = quat_apply_inverse(act_q, ori_err_w)          # = Log(R_refT R_act)

    lin_b = quat_apply_inverse(act_q, ref_lin_w.reshape(flat, 3))
    ang_b = quat_apply_inverse(act_q, ref_ang_w.reshape(flat, 3))

    # ee positions relative to THAT node's own reference base: posture, free of the
    # tracking error that base_pose already carries.
    num_ee = ee_pos_w.shape[-2]
    ee_flat = ee_pos_w.reshape(flat, num_ee, 3)
    ee_q = ref_q[:, None, :].expand(flat, num_ee, 4).reshape(-1, 4)
    ee_rel = quat_apply_inverse(
        ee_q, (ee_flat - ref_p[:, None, :]).reshape(-1, 3)
    ).reshape(flat, num_ee * 3)

    node = torch.cat(
        [
            pos_err_b,
            ori_err_b,
            lin_b,
            ang_b,
            joint_pos.reshape(flat, -1),
            joint_vel.reshape(flat, -1),
            ee_rel,
        ],
        dim=-1,
    )
    return node.reshape(tuple(node_shape) + (node.shape[-1],))


def window_to_world(
    window: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    num_joints: int,
    num_ee_frames: int,
) -> dict[str, torch.Tensor]:
    """Exact inverse of :func:`world_to_window`: put the window back in the world.

    A window is entirely relative by design -- that is what keeps world coordinates away
    from the policy -- which also makes it impossible to eyeball. Given the anchor the
    forward direction divided out, this reconstructs the world-frame reference it encodes,
    so a viewer can draw *what the policy actually saw*, and so the FM tracker can still
    feed a world-frame CLF and debug markers in simulation. Nothing on the deploy path
    calls it.

    Feed a CLEAN window, not a corrupted observation: noise, ``clip`` and ``scale`` are
    applied by the observation manager AFTER this layout and are not inverted here.

    Args:
        window: ``[..., K, D]`` window nodes.
        root_pos_w: ``[..., 3]`` anchor position, broadcast over nodes as in
            :func:`world_to_window`.
        root_quat_w: ``[..., 4]`` anchor orientation (xyzw).
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.

    Returns:
        Mapping with ``base_pos_w`` / ``base_lin_vel_w`` / ``base_ang_vel_w`` ``[..., K, 3]``,
        ``base_quat_w`` ``[..., K, 4]`` (xyzw), ``joint_pos`` / ``joint_vel`` ``[..., K, nj]``
        (frame-invariant, passed through), and ``ee_pos_w`` ``[..., K, E, 3]``.

    Note:
        ``base_quat_w`` comes back through ``quat_box_plus``, which normalises the sign
        (``quat_unique``), so it may be the negation of the quaternion that went in. That is
        the same rotation; compare rotations, not raw components.
    """
    blocks = split_window(window, num_joints, num_ee_frames)
    node_shape = window.shape[:-1]                            # (..., K)
    flat = math.prod(node_shape)

    act_p = _broadcast_anchor(root_pos_w, node_shape)
    act_q = _broadcast_anchor(root_quat_w, node_shape)

    pos_err_b = blocks["base_pose"][..., 0:3].reshape(flat, 3)
    ori_err_b = blocks["base_pose"][..., 3:6].reshape(flat, 3)
    pos_w = act_p + quat_apply(act_q, pos_err_b)
    # Forward: ori_err_w = Log(R_act R_refT), i.e. Exp(ori_err_w) * q_ref = q_act, so
    # q_ref = Exp(-ori_err_w) * q_act. The minus is the legacy convention's, not a bug.
    quat_w = quat_box_plus(act_q, -quat_apply(act_q, ori_err_b))

    lin_w = quat_apply(act_q, blocks["base_vel"][..., 0:3].reshape(flat, 3))
    ang_w = quat_apply(act_q, blocks["base_vel"][..., 3:6].reshape(flat, 3))

    # ee positions are relative to THAT node's reference base, so they lift through
    # (pos_w, quat_w) -- not through the anchor.
    ee_rel = blocks["ee_pos"].reshape(flat, num_ee_frames, 3)
    ee_q = quat_w[:, None, :].expand(flat, num_ee_frames, 4).reshape(-1, 4)
    ee_pos_w = pos_w[:, None, :] + quat_apply(ee_q, ee_rel.reshape(-1, 3)).reshape(
        flat, num_ee_frames, 3
    )

    shape3 = tuple(node_shape) + (3,)
    return {
        "base_pos_w": pos_w.reshape(shape3),
        "base_quat_w": quat_w.reshape(tuple(node_shape) + (4,)),
        "base_lin_vel_w": lin_w.reshape(shape3),
        "base_ang_vel_w": ang_w.reshape(shape3),
        "joint_pos": blocks["joint_pos"].clone(),
        "joint_vel": blocks["joint_vel"].clone(),
        "ee_pos_w": ee_pos_w.reshape(tuple(node_shape) + (num_ee_frames, 3)),
    }
