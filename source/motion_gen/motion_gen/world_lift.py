# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared yaw-only lift between a (yaw-aligned) anchor frame and the world.

Given an anchor's world pose, these helpers lift quantities expressed
yaw-aligned-relative to that anchor back into the world frame.  They are
the exact inverse of the forward yaw-alignment ``_align_yaw`` /
``_align_yaw_batched`` in :mod:`base_trajectory_cmd`::

    forward:  v_rel = R_yaw(q_a)^{-1} @ v_w     (world  -> anchor-relative)
    inverse:  v_w   = R_yaw(q_a)      @ v_rel   (anchor -> world)

Two callers use this lift:

  * the **offline clip generator** bakes its per-step stance anchor into the
    stored outputs via :func:`bake_outputs_to_world`, so a generated clip is a
    single clip-world frame (no runtime stance re-anchoring);
  * the **runtime command** applies a single per-env *placement* (yaw + xy of
    the clip in the sim world) to those clip-world outputs.

Quaternions are xyzw to match Isaac Lab's math utils.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from motion_gen.math_utils import quat_apply, quat_inv, quat_mul, yaw_quat


# ----------------------------------------------------------------------
# Per-clip root-DOF anchor mask.
#
# A clip declares, per root DOF, whether that DOF is FREE (re-anchored onto
# the robot at runtime) or FIXED (taken from the clip on its terrain cell,
# not the robot).  The mask is a uint8 bitmask over the 6 DOFs below; a SET
# bit means FREE.  The legacy default reproduces today's ``ego_placement``
# (x, y, z, yaw FREE; roll, pitch kept from the clip).
# ----------------------------------------------------------------------

_ANCHOR_DOF_NAMES: list[str] = ["x", "y", "z", "roll", "pitch", "yaw"]  # bit index = list index
_ANCHOR_DOF_INDEX: dict[str, int] = {n: i for i, n in enumerate(_ANCHOR_DOF_NAMES)}


def anchor_mask_from_free(free: Iterable[str]) -> int:
    """Build a uint8 anchor bitmask (bit SET = FREE) from FREE DOF names.

    Args:
        free: DOF names to mark FREE (a subset of :data:`_ANCHOR_DOF_NAMES`).

    Returns:
        The integer bitmask.
    """
    mask = 0
    for name in free:
        if name not in _ANCHOR_DOF_INDEX:
            raise ValueError(f"unknown anchor DOF '{name}'; expected one of {_ANCHOR_DOF_NAMES}")
        mask |= 1 << _ANCHOR_DOF_INDEX[name]
    return mask


def anchor_mask_free_names(mask: int) -> list[str]:
    """The FREE DOF names encoded in ``mask`` (inverse of :func:`anchor_mask_from_free`)."""
    return [n for n, i in _ANCHOR_DOF_INDEX.items() if (int(mask) >> i) & 1]


# Named masks (computed, not hand-typed, to stay in lock-step with the bit order).
_ANCHOR_MASK_LEGACY: int = anchor_mask_from_free({"x", "y", "z", "yaw"})  # 0b100111 = 39; today's ego_placement
_ANCHOR_MASK_FLAT: int = anchor_mask_from_free({"x", "y", "yaw"})          # 0b100011 = 35; flat, z clip-fixed
_ANCHOR_MASK_STAIRS: int = anchor_mask_from_free({"y"})                    # 0b000010 = 2
_ANCHOR_MASK_SLOPE: int = anchor_mask_from_free({"x", "y"})                # 0b000011 = 3; yaw FIXED (uniform plane)
_ANCHOR_MASK_BOX: int = anchor_mask_from_free({"y"})                       # 0b000010 = 2; box MOUNT (jump-on)
# Box clips use a PER-FRAME mask.  This constant is the jump-ON (mount) phase: y FREE so the
# lateral landing follows the robot's approach (approach a little off-center -> the reference
# slides over, same as stairs), while x/z/yaw stay FIXED (world-pinned to the box) so the
# takeoff distance + height + heading make the ballistic arc land on the box.  The on-top
# walk / dismount / recovery phases use ``_ANCHOR_MASK_FLAT`` (ego).  Requires the box sized
# wide enough in y to absorb the free-y lateral spread (like stairs spanning the corridor).
# Frame note: the mask is applied in the WORLD frame and terrain clips place with identity
# heading (yaw=0), so world-y == lateral-across-the-box (holds only while axis-aligned).


def anchor_inv_yaw(anchor_quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation that lifts anchor-relative vectors into the world.

    This is the inverse of the ``yaw_quat(quat_inv(anchor_quat))`` rotation
    that ``_align_yaw`` applies in the forward (world -> anchor) direction.

    Args:
        anchor_quat: ``[..., 4]`` anchor-frame world quaternions (xyzw).

    Returns:
        ``[..., 4]`` lift quaternions (xyzw).
    """
    return quat_inv(yaw_quat(quat_inv(anchor_quat)))


# ----------------------------------------------------------------------
# Single-frame lift: anchor [N, .] and quantities [N, .] share leading dim.
# ----------------------------------------------------------------------

def lift_position(anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_rel: torch.Tensor) -> torch.Tensor:
    """Lift an anchor-relative position to the world frame.

    Args:
        anchor_pos: ``[N, 3]`` anchor world positions.
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        pos_rel: ``[N, 3]`` anchor-relative positions.

    Returns:
        ``[N, 3]`` world positions.
    """
    return anchor_pos + quat_apply(anchor_inv_yaw(anchor_quat), pos_rel)


def lift_orientation(anchor_quat: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Lift an anchor-relative orientation to the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        quat_rel: ``[N, 4]`` anchor-relative quaternions (xyzw).

    Returns:
        ``[N, 4]`` world quaternions (xyzw).
    """
    return quat_mul(anchor_inv_yaw(anchor_quat), quat_rel)


def lift_vector(anchor_quat: torch.Tensor, vec_rel: torch.Tensor) -> torch.Tensor:
    """Rotate an anchor-relative vector (lin/ang velocity) into the world.

    No translation is applied -- velocities only rotate.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        vec_rel: ``[N, 3]`` anchor-relative vectors.

    Returns:
        ``[N, 3]`` world vectors.
    """
    return quat_apply(anchor_inv_yaw(anchor_quat), vec_rel)


# ----------------------------------------------------------------------
# Placement: solve the per-env yaw+xy ground frame onto a target root.
# ----------------------------------------------------------------------

def _yaw_from_quat(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw angle (rad) about world z from an ``(x, y, z, w)`` quaternion ``[..., 4]``."""
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def ego_placement_masked(
    base_pos_cf: torch.Tensor,
    base_ori_cf: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    anchor_mask: torch.Tensor | int,
    cell_anchor_pos: torch.Tensor,
    cell_anchor_yaw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env placement honoring a per-clip root-DOF anchor mask.

    Solves for the runtime *placement* (a yaw-only ground frame: ``world_pos``
    plus a pure-yaw ``world_quat``) such that lifting the clip-world base pose
    ``(base_pos_cf, base_ori_cf)`` through it puts each root DOF where the clip's
    :func:`anchor_mask_from_free` mask says:

      * **FREE** DOF -> re-anchored onto the robot ``target`` (as today);
      * **FIXED** DOF -> pinned to the clip on its terrain cell, i.e. the
        component of ``cell_anchor_pos`` / ``cell_anchor_yaw`` (the reset
        placement), so the clip's baked motion plays out from there unchanged.

    The returned ``world_quat`` is always pure-yaw (roll/pitch are never
    re-anchored -- they ride through the ``base:ori`` lift as the clip's natural
    tilt), which every downstream lift relies on.  ``roll`` / ``pitch`` FREE bits
    are reserved and rejected (a full-quat placement would break the pure-yaw
    invariant).

    With ``anchor_mask == _ANCHOR_MASK_LEGACY`` and zero cell anchor this
    reproduces :func:`ego_placement` exactly.  All quaternions are ``(x, y, z, w)``.

    Args:
        base_pos_cf: ``[N, 3]`` clip-world base position at the current step.
        base_ori_cf: ``[N, 4]`` clip-world base orientation (xyzw) at the current step.
        target_pos_w: ``[N, 3]`` target root (robot) world position.
        target_quat_w: ``[N, 4]`` target root (robot) world orientation (xyzw).
        anchor_mask: ``[N]`` uint8 bitmask (bit SET = FREE), or a scalar int
            broadcast to all envs.
        cell_anchor_pos: ``[N, 3]`` terrain-cell world anchor for FIXED DOFs
            (the reset placement position); zeros for flat.
        cell_anchor_yaw: ``[N]`` terrain-cell heading (rad) for a FIXED yaw;
            ``None`` treats it as 0 (identity heading, all current terrain cases).

    Returns:
        ``(world_pos[N, 3], world_quat[N, 4] xyzw)`` -- the per-env placement.
    """
    n = base_pos_cf.shape[0]
    device, dtype = base_pos_cf.device, base_pos_cf.dtype
    if not torch.is_tensor(anchor_mask):
        anchor_mask = torch.full((n,), int(anchor_mask), dtype=torch.long, device=device)
    m = anchor_mask.to(device=device, dtype=torch.long)
    free = {name: ((m >> i) & 1).bool() for name, i in _ANCHOR_DOF_INDEX.items()}  # [N] bool each
    if bool(free["roll"].any()) or bool(free["pitch"].any()):
        raise ValueError("free roll/pitch is reserved and unsupported; the mask must FIX roll & pitch")

    if cell_anchor_yaw is None:
        cell_anchor_yaw = torch.zeros(n, device=device, dtype=dtype)
    else:
        cell_anchor_yaw = cell_anchor_yaw.to(device=device, dtype=dtype)

    # Placement yaw: robot-relative when yaw is FREE, else the terrain-cell heading.
    yaw_free = _yaw_from_quat(target_quat_w) - _yaw_from_quat(base_ori_cf)  # [N]
    yaw = torch.where(free["yaw"], yaw_free, cell_anchor_yaw)  # [N]
    half = 0.5 * yaw
    zeros = torch.zeros_like(yaw)
    world_quat = torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)

    # Planar rotation R(yaw) applied to the clip base xy (z is yaw-invariant); this
    # equals quat_apply(world_quat, base_pos_cf)[:, :2] for the pure-yaw world_quat.
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    rot_x = cos_y * base_pos_cf[:, 0] - sin_y * base_pos_cf[:, 1]
    rot_y = sin_y * base_pos_cf[:, 0] + cos_y * base_pos_cf[:, 1]

    # lifted base = world_pos + R(yaw) . base_pos_cf.  FREE d: solve so lifted == target[d];
    # FIXED d: set the placement origin to the cell anchor so lifted == cell + clip motion.
    world_pos = torch.empty_like(target_pos_w)
    world_pos[:, 0] = torch.where(free["x"], target_pos_w[:, 0] - rot_x, cell_anchor_pos[:, 0])
    world_pos[:, 1] = torch.where(free["y"], target_pos_w[:, 1] - rot_y, cell_anchor_pos[:, 1])
    world_pos[:, 2] = torch.where(free["z"], target_pos_w[:, 2] - base_pos_cf[:, 2], cell_anchor_pos[:, 2])
    return world_pos, world_quat


def ego_placement(
    base_pos_cf: torch.Tensor,
    base_ori_cf: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy per-env yaw+xy+z placement putting a clip's current-step base onto a target root.

    Thin wrapper over :func:`ego_placement_masked` with the legacy mask
    (``x, y, z, yaw`` FREE; roll/pitch kept from the clip) and a zero terrain-cell
    anchor -- i.e. the reference base xy+z+yaw are all snapped onto the robot root.
    This mirrors the FM deploy-time anchor (:meth:`FmTrackingCommand._run_fm`), which
    pins the reference base xy+z+yaw onto the robot so the receding-horizon reference
    rides at the robot's actual height; the future window then recedes with the clip's
    natural z variation.  Anchoring z (not the clip's natural height) means a sinking
    robot drags the reference height down with it -- so the teacher must learn to stay up.

    Prefer :func:`ego_placement_masked` for terrain-aware clips (per-DOF FREE/FIXED).

    Args:
        base_pos_cf: ``[N, 3]`` clip-world base position at the current step.
        base_ori_cf: ``[N, 4]`` clip-world base orientation (xyzw) at the current step.
        target_pos_w: ``[N, 3]`` target root world position (its z is snapped onto too).
        target_quat_w: ``[N, 4]`` target root world orientation (xyzw).

    Returns:
        ``(world_pos[N, 3], world_quat[N, 4] xyzw)`` -- the per-env placement.
    """
    return ego_placement_masked(
        base_pos_cf, base_ori_cf, target_pos_w, target_quat_w,
        anchor_mask=_ANCHOR_MASK_LEGACY,
        cell_anchor_pos=torch.zeros_like(target_pos_w),
    )


# ----------------------------------------------------------------------
# Batched lift: one anchor [N, .] applied to B frames per env [N, B, .].
# Mirrors the expand/reshape pattern of ``_align_yaw_batched``.
# ----------------------------------------------------------------------

def lift_position_batched(anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_rel: torch.Tensor) -> torch.Tensor:
    """Lift ``B`` anchor-relative positions per env to the world frame.

    Args:
        anchor_pos: ``[N, 3]`` anchor world positions.
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        pos_rel: ``[N, B, 3]`` anchor-relative positions.

    Returns:
        ``[N, B, 3]`` world positions.
    """
    n, b, _ = pos_rel.shape
    inv_yaw = anchor_inv_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    rotated = quat_apply(inv_yaw, pos_rel.reshape(n * b, 3)).reshape(n, b, 3)
    return anchor_pos.unsqueeze(1) + rotated


def lift_orientation_batched(anchor_quat: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Lift ``B`` anchor-relative orientations per env to the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        quat_rel: ``[N, B, 4]`` anchor-relative quaternions (xyzw).

    Returns:
        ``[N, B, 4]`` world quaternions (xyzw).
    """
    n, b, _ = quat_rel.shape
    inv_yaw = anchor_inv_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_mul(inv_yaw, quat_rel.reshape(n * b, 4)).reshape(n, b, 4)


def lift_vector_batched(anchor_quat: torch.Tensor, vec_rel: torch.Tensor) -> torch.Tensor:
    """Rotate ``B`` anchor-relative vectors per env into the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        vec_rel: ``[N, B, 3]`` anchor-relative vectors.

    Returns:
        ``[N, B, 3]`` world vectors.
    """
    n, b, _ = vec_rel.shape
    inv_yaw = anchor_inv_yaw(anchor_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_apply(inv_yaw, vec_rel.reshape(n * b, 3)).reshape(n, b, 3)


# ----------------------------------------------------------------------
# Convenience: full base-frame world state from anchor-relative outputs.
# ----------------------------------------------------------------------

def lift_base_world_state(
    y_des: torch.Tensor,
    dy_des: torch.Tensor,
    anchor_pose: torch.Tensor,
    idx: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Lift the base-frame channels of desired outputs to a world state vector.

    Replicates the per-base-frame lift in
    :meth:`BaseTrajectoryCommand.get_reference_world_state`.  The returned
    layout matches the raw robot ``state`` recorded by
    ``scripts/rsl_rl/record_data.py``::

        [root_pos_w(3), root_quat_w(4, xyzw), root_lin_vel_w(3),
         root_ang_vel_w(3), joint_pos(nj), joint_vel(nj)]

    Args:
        y_des: ``[N, P]`` desired position outputs (anchor-relative).
        dy_des: ``[N, V]`` desired velocity outputs (anchor-relative).
        anchor_pose: ``[N, 7]`` anchor world pose ``[pos(3), quat(4) xyzw]``.
        idx: index dict (from ``_reference_world_indices``) with long-tensor
            keys ``pos, ori, lin_vel, ang_vel, joint_pos, joint_vel``.

    Returns:
        ``[N, 3 + 4 + 3 + 3 + 2*nj]`` world-frame reference state.
    """
    anchor_pos = anchor_pose[:, :3]
    inv_yaw = anchor_inv_yaw(anchor_pose[:, 3:7])

    base_pos_w = anchor_pos + quat_apply(inv_yaw, y_des[:, idx["pos"]])
    base_quat_w = quat_mul(inv_yaw, y_des[:, idx["ori"]])
    base_lin_w = quat_apply(inv_yaw, dy_des[:, idx["lin_vel"]])
    base_ang_w = quat_apply(inv_yaw, dy_des[:, idx["ang_vel"]])

    joint_pos = y_des[:, idx["joint_pos"]]
    joint_vel = dy_des[:, idx["joint_vel"]]

    return torch.cat(
        [base_pos_w, base_quat_w, base_lin_w, base_ang_w, joint_pos, joint_vel], dim=-1
    )


# ----------------------------------------------------------------------
# Generation: bake a per-step anchor schedule into the stored outputs.
# ----------------------------------------------------------------------

def bake_outputs_to_world(
    y_des: torch.Tensor,
    dy_des: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    pos_output_names: list[str],
    vel_output_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift dense anchor-relative outputs into the clip-world frame, in place.

    Converts a clip's anchor-relative reference outputs into a single
    clip-world frame by lifting every output frame through its own per-step
    anchor.  After baking, the stored clip needs no runtime stance
    re-anchoring -- only a per-env placement.  Uses the exact same lift the
    runtime/placement applies (:func:`lift_position` / :func:`lift_orientation`
    / :func:`lift_vector`).

    Per output frame ``f``: position channels (``{f}:pos_{x,y,z}``) are
    translated+rotated, orientation channels (``{f}:ori_{x,y,z,w}``, xyzw) are
    quat-composed, and velocity channels (linear ``{f}:pos_*`` and angular
    ``{f}:ori_*`` in ``vel_output_names``) are rotated only.  ``joint:*``
    channels are frame-invariant and pass through untouched.

    Args:
        y_des: ``[n, T, P]`` anchor-relative position outputs (mutated in place).
        dy_des: ``[n, T, V]`` anchor-relative velocity outputs (mutated in place).
        anchor_pos: ``[n, T, 3]`` per-step world anchor positions (clip frame).
        anchor_quat: ``[n, T, 4]`` per-step world anchor quaternions (xyzw).
        pos_output_names: Ordered position-output names (includes ``ori_w``).
        vel_output_names: Ordered velocity-output names.

    Returns:
        ``(y_des, dy_des)`` lifted into the clip-world frame.
    """
    n, t = anchor_pos.shape[0], anchor_pos.shape[1]
    ap = anchor_pos.reshape(n * t, 3)
    aq = anchor_quat.reshape(n * t, 4)
    yf = y_des.reshape(n * t, y_des.shape[-1])
    dyf = dy_des.reshape(n * t, dy_des.shape[-1])
    dev = yf.device

    def _idx(names: list[str], frame: str, comps: tuple[str, ...], kind: str) -> torch.Tensor:
        return torch.tensor(
            [names.index(f"{frame}:{kind}_{a}") for a in comps], dtype=torch.long, device=dev
        )

    # Position blocks (translate + rotate), plus orientation where present.
    for f in sorted({nm.split(":", 1)[0] for nm in pos_output_names if ":pos_" in nm}):
        pos_i = _idx(pos_output_names, f, ("x", "y", "z"), "pos")
        yf[:, pos_i] = lift_position(ap, aq, yf[:, pos_i])
        if f"{f}:ori_w" in pos_output_names:
            ori_i = _idx(pos_output_names, f, ("x", "y", "z", "w"), "ori")
            yf[:, ori_i] = lift_orientation(aq, yf[:, ori_i])

    # Velocity blocks (rotate only): linear (pos_*) and angular (ori_*).
    for f in sorted({nm.split(":", 1)[0] for nm in vel_output_names if ":pos_" in nm}):
        lin_i = _idx(vel_output_names, f, ("x", "y", "z"), "pos")
        dyf[:, lin_i] = lift_vector(aq, dyf[:, lin_i])
    for f in sorted({nm.split(":", 1)[0] for nm in vel_output_names if ":ori_" in nm}):
        ang_i = _idx(vel_output_names, f, ("x", "y", "z"), "ori")
        dyf[:, ang_i] = lift_vector(aq, dyf[:, ang_i])

    return y_des, dy_des
