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
from motion_gen.joint_names import world_frame_names
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

#
# The translational bits are named in the TERRAIN's frame, not the world's: ``downrange`` is the
# axis the traversal crosses the terrain along, ``lateral`` is across it, ``height`` is up (always
# gravity-aligned).  Which world axes those are is carried per terrain by ``mask_frame_yaw`` (see
# :func:`ego_placement_masked`), NOT assumed.  The world-axis spelling this file used to carry is
# exactly how a 90-degree-rotated terrain family once slipped through: its clips travelled along
# world y, so "free y" handed the mask the traversal axis instead of the lateral offset.
_ANCHOR_DOF_NAMES: list[str] = ["downrange", "lateral", "height", "roll", "pitch", "yaw"]
_ANCHOR_DOF_INDEX: dict[str, int] = {n: i for i, n in enumerate(_ANCHOR_DOF_NAMES)}

# Accepted legacy spellings.  The BIT ORDER is unchanged, so every stored uint8 mask keeps its
# meaning; only the names moved.  Kept so generator CLIs (``--flat_anchor x,y,yaw``) and older call
# sites keep working -- but new code should use the terrain-frame names.
_ANCHOR_DOF_ALIASES: dict[str, str] = {"x": "downrange", "y": "lateral", "z": "height"}


def anchor_mask_from_free(free: Iterable[str]) -> int:
    """Build a uint8 anchor bitmask (bit SET = FREE) from FREE DOF names.

    Args:
        free: DOF names to mark FREE (a subset of :data:`_ANCHOR_DOF_NAMES`; the legacy
            ``x`` / ``y`` / ``z`` spellings in :data:`_ANCHOR_DOF_ALIASES` are accepted).

    Returns:
        The integer bitmask.
    """
    mask = 0
    for name in free:
        canon = _ANCHOR_DOF_ALIASES.get(name, name)
        if canon not in _ANCHOR_DOF_INDEX:
            raise ValueError(f"unknown anchor DOF '{name}'; expected one of {_ANCHOR_DOF_NAMES} "
                             f"(or {sorted(_ANCHOR_DOF_ALIASES)})")
        mask |= 1 << _ANCHOR_DOF_INDEX[canon]
    return mask


def anchor_mask_free_names(mask: int) -> list[str]:
    """The FREE DOF names encoded in ``mask`` (inverse of :func:`anchor_mask_from_free`).

    Returns the canonical terrain-frame names, never the legacy aliases.
    """
    return [n for n, i in _ANCHOR_DOF_INDEX.items() if (int(mask) >> i) & 1]


# Named masks (computed, not hand-typed, to stay in lock-step with the bit order).
_ANCHOR_MASK_LEGACY: int = anchor_mask_from_free(       # 0b100111 = 39; today's ego_placement
    {"downrange", "lateral", "height", "yaw"})
_ANCHOR_MASK_FLAT: int = anchor_mask_from_free(         # 0b100011 = 35; flat, height clip-fixed
    {"downrange", "lateral", "yaw"})
_ANCHOR_MASK_STAIRS: int = anchor_mask_from_free({"lateral"})               # 0b000010 = 2
_ANCHOR_MASK_SLOPE: int = anchor_mask_from_free(        # 0b000011 = 3; yaw FIXED (uniform plane)
    {"downrange", "lateral"})
_ANCHOR_MASK_BOX: int = anchor_mask_from_free({"lateral"})                  # 0b000010 = 2; box MOUNT
# Box clips use a PER-FRAME mask.  This constant is the jump-ON (mount) phase: LATERAL free so the
# landing follows the robot's approach (approach a little off-center -> the reference slides over,
# same as stairs), while downrange/height/yaw stay FIXED (pinned to the box) so the takeoff distance
# + height + heading make the ballistic arc land on the box.  The on-top walk / dismount / recovery
# phases use ``_ANCHOR_MASK_FLAT`` (ego).  Requires the box sized wide enough laterally to absorb the
# free-lateral spread (like stairs spanning the corridor).
# Frame note: "downrange"/"lateral" are resolved in the terrain's own yaw frame via
# ``ego_placement_masked(mask_frame_yaw=...)``, so a rotated tile or a family baked on a rotated
# heading is handled rather than silently transposed.


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


def _placement_core(
    base_pos_cf: torch.Tensor,
    base_ori_cf: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    mask: torch.Tensor,
    cell_anchor_pos: torch.Tensor,
    cell_anchor_yaw: torch.Tensor,
    mask_frame_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat-batch placement solve; every argument carries the same leading dim ``[M]``.

    The one place the FREE/FIXED split is resolved.  Translational bits are masked in the TERRAIN
    frame (rotated by ``mask_frame_yaw`` about world z), so "free lateral" means across the terrain
    whatever the terrain's world heading is; ``height`` is never rotated (gravity-aligned).
    """
    free = {name: ((mask >> i) & 1).bool() for name, i in _ANCHOR_DOF_INDEX.items()}
    if bool(free["roll"].any()) or bool(free["pitch"].any()):
        raise ValueError("free roll/pitch is reserved and unsupported; the mask must FIX roll & pitch")

    # Placement yaw: robot-relative when yaw is FREE, else the terrain-cell heading.
    yaw_free = _yaw_from_quat(target_quat_w) - _yaw_from_quat(base_ori_cf)
    yaw = torch.where(free["yaw"], yaw_free, cell_anchor_yaw)
    half = 0.5 * yaw
    zeros = torch.zeros_like(yaw)
    world_quat = torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)

    # Planar rotation R(yaw) applied to the clip base xy (z is yaw-invariant); this equals
    # quat_apply(world_quat, base_pos_cf)[:, :2] for the pure-yaw world_quat.
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    rot_x = cos_y * base_pos_cf[:, 0] - sin_y * base_pos_cf[:, 1]
    rot_y = sin_y * base_pos_cf[:, 0] + cos_y * base_pos_cf[:, 1]

    # Planar discrepancy between the robot and the clip sitting on its terrain cell, expressed in
    # WORLD axes, then rotated into the terrain frame, masked there, and rotated back.  With
    # mask_frame_yaw == 0 this is componentwise masking of the world discrepancy (the old behavior).
    d_wx = target_pos_w[:, 0] - rot_x - cell_anchor_pos[:, 0]
    d_wy = target_pos_w[:, 1] - rot_y - cell_anchor_pos[:, 1]
    c_t, s_t = torch.cos(mask_frame_yaw), torch.sin(mask_frame_yaw)
    d_down = c_t * d_wx + s_t * d_wy        # R(theta)^T . d_world
    d_lat = -s_t * d_wx + c_t * d_wy
    d_down = torch.where(free["downrange"], d_down, torch.zeros_like(d_down))
    d_lat = torch.where(free["lateral"], d_lat, torch.zeros_like(d_lat))

    world_pos = torch.empty_like(target_pos_w)
    world_pos[:, 0] = cell_anchor_pos[:, 0] + (c_t * d_down - s_t * d_lat)   # R(theta) . d_masked
    world_pos[:, 1] = cell_anchor_pos[:, 1] + (s_t * d_down + c_t * d_lat)
    d_h = target_pos_w[:, 2] - base_pos_cf[:, 2] - cell_anchor_pos[:, 2]
    world_pos[:, 2] = cell_anchor_pos[:, 2] + torch.where(free["height"], d_h, torch.zeros_like(d_h))
    return world_pos, world_quat


def _broadcast_placement_args(
    n: int, device, dtype, anchor_mask, cell_anchor_yaw, mask_frame_yaw,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the three optional/scalar placement inputs to ``[n]`` tensors on ``device``."""
    if not torch.is_tensor(anchor_mask):
        anchor_mask = torch.full((n,), int(anchor_mask), dtype=torch.long, device=device)
    mask = anchor_mask.to(device=device, dtype=torch.long)
    if cell_anchor_yaw is None:
        cell_yaw = torch.zeros(n, device=device, dtype=dtype)
    else:
        cell_yaw = cell_anchor_yaw.to(device=device, dtype=dtype)
    if mask_frame_yaw is None:
        frame_yaw = torch.zeros(n, device=device, dtype=dtype)
    elif not torch.is_tensor(mask_frame_yaw):
        frame_yaw = torch.full((n,), float(mask_frame_yaw), device=device, dtype=dtype)
    else:
        frame_yaw = mask_frame_yaw.to(device=device, dtype=dtype)
    return mask, cell_yaw, frame_yaw


def ego_placement_masked_window(
    base_pos_cf: torch.Tensor,
    base_ori_cf: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    anchor_mask: torch.Tensor,
    cell_anchor_pos: torch.Tensor,
    cell_anchor_yaw: torch.Tensor | None = None,
    mask_frame_yaw: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-FRAME placements for ``K`` anchor masks per env, from ONE re-anchor snapshot.

    The windowed counterpart of :func:`ego_placement_masked`.  A clip's mask varies along its
    timeline (flat approach -> terrain-pinned interaction -> flat exit), and each of those segments
    needs its OWN placement: the terrain-pinned frames must stay welded to the terrain even while
    the robot is still on the ego-anchored approach.  Since a placement is a pure function of the
    re-anchor snapshot and the mask, solving it per frame costs one extra gather.

    The snapshot (``base_*``, ``target_*``) is the SAME for every frame -- it is the pose pair
    captured at the last re-anchor -- so only ``anchor_mask`` carries the per-frame dimension.

    Args:
        base_pos_cf: ``[N, 3]`` clip-world base position at the re-anchor step.
        base_ori_cf: ``[N, 4]`` clip-world base orientation (xyzw) at the re-anchor step.
        target_pos_w: ``[N, 3]`` robot root world position at the re-anchor step.
        target_quat_w: ``[N, 4]`` robot root world orientation (xyzw) at the re-anchor step.
        anchor_mask: ``[N, K]`` uint8 bitmask per frame (bit SET = FREE).
        cell_anchor_pos: ``[N, 3]`` terrain-cell world anchor for FIXED DOFs.
        cell_anchor_yaw: ``[N]`` terrain-cell heading (rad) for a FIXED yaw; ``None`` -> 0.
        mask_frame_yaw: ``[N]`` terrain-frame heading (rad) the translational bits resolve in;
            ``None`` -> 0 (world-axis masking).

    Returns:
        ``(world_pos[N, K, 3], world_quat[N, K, 4] xyzw)``.
    """
    n, k = anchor_mask.shape
    device, dtype = base_pos_cf.device, base_pos_cf.dtype
    mask, cell_yaw, frame_yaw = _broadcast_placement_args(
        n, device, dtype, anchor_mask, cell_anchor_yaw, mask_frame_yaw)

    def _rep3(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(1).expand(n, k, t.shape[-1]).reshape(n * k, t.shape[-1])

    def _rep1(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(1).expand(n, k).reshape(n * k)

    pos, quat = _placement_core(
        _rep3(base_pos_cf), _rep3(base_ori_cf), _rep3(target_pos_w), _rep3(target_quat_w),
        mask.reshape(n * k), _rep3(cell_anchor_pos), _rep1(cell_yaw), _rep1(frame_yaw),
    )
    return pos.reshape(n, k, 3), quat.reshape(n, k, 4)


def ego_placement_masked(
    base_pos_cf: torch.Tensor,
    base_ori_cf: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    anchor_mask: torch.Tensor | int,
    cell_anchor_pos: torch.Tensor,
    cell_anchor_yaw: torch.Tensor | None = None,
    mask_frame_yaw: torch.Tensor | float | None = None,
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
        mask_frame_yaw: ``[N]`` heading (rad) of the TERRAIN's own frame, which the
            ``downrange`` / ``lateral`` bits are resolved in; ``None`` treats it as 0
            (the translational split then falls on world x / y).

    Returns:
        ``(world_pos[N, 3], world_quat[N, 4] xyzw)`` -- the per-env placement.
    """
    n = base_pos_cf.shape[0]
    device, dtype = base_pos_cf.device, base_pos_cf.dtype
    mask, cell_yaw, frame_yaw = _broadcast_placement_args(
        n, device, dtype, anchor_mask, cell_anchor_yaw, mask_frame_yaw)
    return _placement_core(base_pos_cf, base_ori_cf, target_pos_w, target_quat_w,
                           mask, cell_anchor_pos, cell_yaw, frame_yaw)


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

def _flat_anchor(anchor: torch.Tensor, n: int, b: int) -> torch.Tensor:
    """Flatten a batched-lift anchor to ``[N*B, C]``, accepting one anchor per env OR per frame.

    ``[N, C]`` (one anchor for the whole window) is expanded; ``[N, B, C]`` (a per-frame anchor, as
    a per-SEGMENT placement produces) is reshaped.  Lets one lift serve both.
    """
    if anchor.dim() == 2:
        return anchor.unsqueeze(1).expand(n, b, anchor.shape[-1]).reshape(n * b, anchor.shape[-1])
    if anchor.shape[:2] != (n, b):
        raise ValueError(f"per-frame anchor must be [{n}, {b}, C]; got {tuple(anchor.shape)}")
    return anchor.reshape(n * b, anchor.shape[-1])


def lift_position_batched(anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_rel: torch.Tensor) -> torch.Tensor:
    """Lift ``B`` anchor-relative positions per env to the world frame.

    Args:
        anchor_pos: ``[N, 3]`` anchor world positions, or ``[N, B, 3]`` per frame.
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw), or ``[N, B, 4]`` per frame.
        pos_rel: ``[N, B, 3]`` anchor-relative positions.

    Returns:
        ``[N, B, 3]`` world positions.
    """
    n, b, _ = pos_rel.shape
    inv_yaw = anchor_inv_yaw(_flat_anchor(anchor_quat, n, b))
    rotated = quat_apply(inv_yaw, pos_rel.reshape(n * b, 3))
    return (_flat_anchor(anchor_pos, n, b) + rotated).reshape(n, b, 3)


def lift_orientation_batched(anchor_quat: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Lift ``B`` anchor-relative orientations per env to the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw), or ``[N, B, 4]`` per frame.
        quat_rel: ``[N, B, 4]`` anchor-relative quaternions (xyzw).

    Returns:
        ``[N, B, 4]`` world quaternions (xyzw).
    """
    n, b, _ = quat_rel.shape
    inv_yaw = anchor_inv_yaw(_flat_anchor(anchor_quat, n, b))
    return quat_mul(inv_yaw, quat_rel.reshape(n * b, 4)).reshape(n, b, 4)


def lift_vector_batched(anchor_quat: torch.Tensor, vec_rel: torch.Tensor) -> torch.Tensor:
    """Rotate ``B`` anchor-relative vectors per env into the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw), or ``[N, B, 4]`` per frame.
        vec_rel: ``[N, B, 3]`` anchor-relative vectors.

    Returns:
        ``[N, B, 3]`` world vectors.
    """
    n, b, _ = vec_rel.shape
    inv_yaw = anchor_inv_yaw(_flat_anchor(anchor_quat, n, b))
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

    Per WORLD output frame ``f``: position channels (``{f}:pos_{x,y,z}``) are
    translated+rotated, orientation channels (``{f}:ori_{x,y,z,w}``, xyzw) are
    quat-composed, and velocity channels (linear ``{f}:pos_*`` and angular
    ``{f}:ori_*`` in ``vel_output_names``) are rotated only.  ``joint:*``
    channels and root-frame (``@root``) channels are frame-invariant and pass
    through untouched.

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

    # WORLD frames only: ``@root`` channels are pelvis-relative, so lifting them through the
    # anchor would corrupt the pose.
    world_pos = set(world_frame_names(pos_output_names))
    world_vel = set(world_frame_names(vel_output_names))

    # Position blocks (translate + rotate), plus orientation where present.
    for f in sorted(f for f in world_pos if f"{f}:pos_x" in pos_output_names):
        pos_i = _idx(pos_output_names, f, ("x", "y", "z"), "pos")
        yf[:, pos_i] = lift_position(ap, aq, yf[:, pos_i])
        if f"{f}:ori_w" in pos_output_names:
            ori_i = _idx(pos_output_names, f, ("x", "y", "z", "w"), "ori")
            yf[:, ori_i] = lift_orientation(aq, yf[:, ori_i])

    # Velocity blocks (rotate only): linear (pos_*) and angular (ori_*).
    for f in sorted(f for f in world_vel if f"{f}:pos_x" in vel_output_names):
        lin_i = _idx(vel_output_names, f, ("x", "y", "z"), "pos")
        dyf[:, lin_i] = lift_vector(aq, dyf[:, lin_i])
    for f in sorted(f for f in world_vel if f"{f}:ori_x" in vel_output_names):
        ang_i = _idx(vel_output_names, f, ("x", "y", "z"), "ori")
        dyf[:, ang_i] = lift_vector(aq, dyf[:, ang_i])

    return y_des, dy_des
