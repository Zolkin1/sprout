# Copyright 2026 Zachary Olkin. All rights reserved.

"""Geometry transforms for the flow-matching state representation.

This module is the single source of truth for two conventions:

1. The **relative-frame state vector** layout (per timestep), order::

       [joint_pos(nj), joint_vel(nj),
        p_rel(3), v_rel(3), ori6d(6), w_rel(3),                 # base (root) block, 15
        <per extra frame: pos_rel(3)> ...]                      # ee block, 3

   where root *and* every extra body frame are expressed relative to a single
   reference root frame. ``num_extra_frames=0`` reproduces the original
   base-and-joints-only vector (``2*nj + 15``).

   The extra (end-effector) frames carry **position only**. The single consumer of
   the predicted ee frames is the teacher's ``reference_window`` observation, whose
   ``ee_pos`` block is a position (``mdp.observations.reference_window``); ee
   orientation and twist were predicted but read by nothing except two FM
   regularizers and the FM tracker's CLF, which no longer grades ee frames. Dropping
   them takes the G1 state width from 133 to 85. If the joint prediction ever needs
   the FK orientation/velocity residuals back, widen :data:`EE_BLOCK_DIM` rather than
   reintroducing the layout by hand -- every offset here is derived from it.

2. The **6D continuous rotation** representation (Zhou et al., 2019): the first
   two *columns* of the rotation matrix. The exact column ordering is reused on
   both the forward and inverse paths so the round-trip is exact.

All quaternions are ``(x, y, z, w)``, matching ``robot.data`` and
``isaaclab.utils.math`` in this repository.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    subtract_frame_transforms,
    yaw_quat,
)


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors to unit length along the last dimension."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the 6D representation (first two columns).

    Args:
        matrix: Rotation matrices, shape ``(..., 3, 3)``.

    Returns:
        6D rotation representation, shape ``(..., 6)`` = concat(column 0, column 1).
    """
    col0 = matrix[..., :, 0]
    col1 = matrix[..., :, 1]
    return torch.cat([col0, col1], dim=-1)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Recover a rotation matrix from the 6D representation via Gram-Schmidt.

    Robust to non-orthonormal inputs (e.g. raw network outputs): the result is
    always a valid rotation matrix.

    Args:
        d6: 6D rotation representation, shape ``(..., 6)``.

    Returns:
        Orthonormal rotation matrices, shape ``(..., 3, 3)`` with the recovered
        vectors as columns.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def quat_to_rot6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``(x, y, z, w)`` to the 6D rotation representation."""
    return matrix_to_rot6d(matrix_from_quat(quat_xyzw))


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    """Convert the 6D rotation representation to quaternions ``(x, y, z, w)``."""
    return quat_from_matrix(rot6d_to_matrix(d6))


def so3_log(r: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle) of rotation matrices ``(..., 3, 3)``.

    The SO(3) log map. Numerically stable near identity -- uses ``atan2`` (not
    ``arccos``, whose gradient is singular at ``cos = 1``) and a finite small-angle
    limit -- so a finite-difference angular velocity between nearly-equal consecutive
    rotations (``so3_log(R[t+1] @ R[t]^T)``) has well-behaved values and gradients.

    Args:
        r: Rotation matrices, shape ``(..., 3, 3)``.

    Returns:
        Rotation vectors ``(..., 3)``.
    """
    v_raw = torch.stack(
        [r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]],
        dim=-1,
    )  # = 2 * sin(theta) * axis
    s = 0.5 * torch.linalg.norm(v_raw, dim=-1)                    # sin(theta) >= 0
    c = 0.5 * (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1.0)  # cos(theta)
    theta = torch.atan2(s, c)                                     # [0, pi]
    # rotation vector = theta * axis = v_raw * theta / (2 sin theta); -> v_raw/2 as theta->0.
    factor = torch.where(s < 1e-6, torch.full_like(s, 0.5), theta / (2.0 * s.clamp_min(1e-12)))
    return v_raw * factor.unsqueeze(-1)


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(..., 3, 3)`` from rotation vectors ``(..., 3)`` (Rodrigues).

    The inverse of :func:`so3_log`.  No longer used by the window: ``base_pose``'s
    orientation half is a 6D rotation now, so :func:`rot6d_to_matrix` is what decodes it.
    This survives for the tests and as the natural partner of ``so3_log``, which the
    angular finite-difference loss still needs to turn a relative rotation into a rate.

    Small-angle safe: the series limits are taken explicitly, so the gradient at ``theta =
    0`` is finite.

    Args:
        v: Rotation vectors (axis * angle), shape ``(..., 3)``.

    Returns:
        Rotation matrices, shape ``(..., 3, 3)``.
    """
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)                 # (..., 1)
    small = theta < 1e-6
    # sin(t)/t and (1-cos t)/t^2, both with their finite t->0 limits.
    t2 = theta * theta
    a = torch.where(small, 1.0 - t2 / 6.0, torch.sin(theta) / theta.clamp_min(1e-12))
    b = torch.where(small, 0.5 - t2 / 24.0, (1.0 - torch.cos(theta)) / t2.clamp_min(1e-24))
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    k = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)                                                          # (..., 3, 3) skew
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(k.shape)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


# Root (base) block width: [pos(3), lin_vel(3), ori6d(6), ang_vel(3)].
ROOT_BLOCK_DIM = 3 + 3 + 6 + 3  # = 15

# Extra (end-effector) frame block width: [pos(3)]. See the module docstring for why
# these are position-only.
EE_BLOCK_DIM = 3


def target_state_dim(num_joints: int, num_extra_frames: int = 0) -> int:
    """Width of the relative-frame state vector.

    Args:
        num_joints: Number of joints ``nj``.
        num_extra_frames: Number of extra body frames beyond the root (each adds
            an :data:`EE_BLOCK_DIM`-wide block).

    Returns:
        ``2*nj + ROOT_BLOCK_DIM + EE_BLOCK_DIM * num_extra_frames``.
    """
    return 2 * num_joints + ROOT_BLOCK_DIM + EE_BLOCK_DIM * num_extra_frames


def frame_block_start(num_joints: int, frame_index: int) -> int:
    """Start offset of a frame block within the state vector.

    Args:
        num_joints: Number of joints ``nj``.
        frame_index: ``0`` for the root/base block, ``1..K`` for extra frames.

    Returns:
        The start column of that frame's block -- ``[pos, lin_vel, ori6d, ang_vel]``
        for the root, ``[pos]`` for an extra frame.
    """
    if frame_index == 0:
        return 2 * num_joints
    return 2 * num_joints + ROOT_BLOCK_DIM + EE_BLOCK_DIM * (frame_index - 1)


def height_above_ground(p_ref: torch.Tensor, ground_z: float = 0.0) -> torch.Tensor:
    """Root height above the ground at the anchor, the FM's height token.

    The xyz+yaw relative frame removes the anchor's absolute z (root pose is
    expressed relative to itself), so the network loses all ground-clearance
    information. This scalar restores it: ``p_ref.z - ground_z``. It is a stand-in
    for a future heightmap -- on flat ground ``ground_z = 0`` and it is just the
    root link's world z at the anchor step.

    Args:
        p_ref: Anchor (reference) positions (world), shape ``(..., 3)``.
        ground_z: World floor height under the anchor (flat ground -> 0.0).

    Returns:
        Height above ground, shape ``(..., 1)``.
    """
    return p_ref[..., 2:3] - ground_z


def orientation_slice(num_joints: int) -> tuple[int, int]:
    """Return ``(start, end)`` of the root's 6D orientation block.

    The root is the only frame carrying an orientation -- extra frames are
    position-only (see the module docstring).

    Args:
        num_joints: Number of joints ``nj``.
    """
    start = frame_block_start(num_joints, 0) + 3 + 3
    return start, start + 6


def orientation_slices(num_joints: int) -> list[tuple[int, int]]:
    """Return every 6D-orientation slice in the state vector (just the root's).

    Kept as a list so the normalizer's identity-mask loop reads the same whether or
    not more oriented frames are ever added back.
    """
    return [orientation_slice(num_joints)]


def _frame_relative_block(
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    p_w: torch.Tensor,
    q_w: torch.Tensor,
    v_w: torch.Tensor,
    w_w: torch.Tensor,
) -> torch.Tensor:
    """Build one ``[pos(3), lin_vel(3), ori6d(6), ang_vel(3)]`` block (any leading dims).

    The reference frame is an **xyz + yaw** ground frame: the anchor's position
    ``p_ref`` (including z) and yaw are removed, but its roll/pitch are NOT (only
    ``yaw_quat(q_ref)`` is factored out). So the block's ``ori6d`` retains the body's
    full tilt relative to gravity, and the FM predicts in a gravity-aligned frame
    that rides the robot's planar pose + heading -- no re-centering needed at deploy.
    Absolute ground clearance (the z that ``p_rel`` removed) is supplied separately
    by the height token (see :func:`height_above_ground`).
    """
    q_yaw = yaw_quat(q_ref)
    p_rel, q_rel = subtract_frame_transforms(p_ref, q_yaw, p_w, q_w)
    ori6d = matrix_to_rot6d(matrix_from_quat(q_rel))
    v_rel = quat_apply_inverse(q_yaw, v_w)
    w_rel = quat_apply_inverse(q_yaw, w_w)
    return torch.cat([p_rel, v_rel, ori6d, w_rel], dim=-1)


def _frame_relative_position(
    p_ref: torch.Tensor, q_ref: torch.Tensor, p_w: torch.Tensor
) -> torch.Tensor:
    """Express a world position in the anchor's **xyz + yaw** ground frame.

    The position half of :func:`_frame_relative_block`, used for the position-only
    extra-frame blocks so the two paths cannot drift apart.
    """
    return quat_apply_inverse(yaw_quat(q_ref), p_w - p_ref)


def relative_state(
    p_w: torch.Tensor,
    q_w: torch.Tensor,
    v_w: torch.Tensor,
    w_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    frame_pos_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the relative-frame state vector for a batch of steps.

    Every input is ``(M, ·)`` where ``M`` is the number of steps. The root pose
    (and each extra frame's position) is expressed relative to the (per-step) **xyz +
    yaw** ground frame at ``(p_ref, q_ref)`` -- the anchor's position (incl. z) and
    yaw are removed, roll/pitch are kept (see :func:`_frame_relative_block`).
    Velocities are rotated into that frame by a **pure rotation** (a stationary
    world snapshot, not a moving body frame, so no ``w x r`` term is added). Each
    extra frame appends a position-only :data:`EE_BLOCK_DIM` block. Absolute
    ground clearance is supplied separately (the height token).

    Args:
        p_w: Root positions (world), shape ``(M, 3)``.
        q_w: Root quaternions (world, xyzw), shape ``(M, 4)``.
        v_w: Root linear velocities (world), shape ``(M, 3)``.
        w_w: Root angular velocities (world), shape ``(M, 3)``.
        joint_pos: Raw joint positions, shape ``(M, nj)``.
        joint_vel: Raw joint velocities, shape ``(M, nj)``.
        p_ref: Reference position (world), shape ``(M, 3)``.
        q_ref: Reference quaternion (world, xyzw), shape ``(M, 4)``.
        frame_pos_w: Extra-frame positions (world), shape ``(M, K, 3)`` or ``None``.

    Returns:
        Relative-state tensor, shape ``(M, 2*nj + 15 + 3*K)``.
    """
    root_block = _frame_relative_block(p_ref, q_ref, p_w, q_w, v_w, w_w)
    parts = [joint_pos, joint_vel, root_block]
    if frame_pos_w is not None:
        m, k = frame_pos_w.shape[0], frame_pos_w.shape[1]
        pr = p_ref.unsqueeze(1).expand(m, k, 3).reshape(m * k, 3)
        qr = q_ref.unsqueeze(1).expand(m, k, 4).reshape(m * k, 4)
        block = _frame_relative_position(
            pr, qr, frame_pos_w.reshape(m * k, 3)
        ).reshape(m, k * EE_BLOCK_DIM)
        parts.append(block)
    return torch.cat(parts, dim=-1)


def relative_state_from_window(
    p_w: torch.Tensor,
    q_w: torch.Tensor,
    v_w: torch.Tensor,
    w_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ref_index: int,
) -> torch.Tensor:
    """Single-window convenience wrapper around :func:`relative_state`.

    Args:
        p_w, q_w, v_w, w_w, joint_pos, joint_vel: Per-step quantities, shape ``(W, ·)``.
        ref_index: Index of the reference step within the window.

    Returns:
        Relative-state tensor, shape ``(W, 2*nj + 15)``.
    """
    w = p_w.shape[0]
    p_ref = p_w[ref_index : ref_index + 1].expand(w, -1)
    q_ref = q_w[ref_index : ref_index + 1].expand(w, -1)
    return relative_state(p_w, q_w, v_w, w_w, joint_pos, joint_vel, p_ref, q_ref)


def _position_to_absolute(
    p_rel: torch.Tensor, p_ref: torch.Tensor, q_ref: torch.Tensor
) -> torch.Tensor:
    """Invert :func:`_frame_relative_position` back to a world position."""
    return p_ref + quat_apply(yaw_quat(q_ref), p_rel)


def _block_to_absolute(
    block: torch.Tensor, p_ref: torch.Tensor, q_ref: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert one ``[pos(3), lin_vel(3), ori6d(6), ang_vel(3)]`` root block to world.

    Uses the same **xyz + yaw** anchor as :func:`_frame_relative_block`
    (``yaw_quat(q_ref)``), so the round-trip is exact (``yaw_quat`` is idempotent).
    """
    q_yaw = yaw_quat(q_ref)
    p_rel = block[..., 0:3]
    v_rel = block[..., 3:6]
    ori6d = block[..., 6:12]
    w_rel = block[..., 12:15]
    q_rel = rot6d_to_quat(ori6d)
    p_w, q_w = combine_frame_transforms(p_ref, q_yaw, p_rel, q_rel)
    v_w = quat_apply(q_yaw, v_rel)
    w_w = quat_apply(q_yaw, w_rel)
    return p_w, q_w, v_w, w_w


def state_to_absolute(
    rel_state: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    num_joints: int,
    num_extra_frames: int = 0,
) -> dict[str, torch.Tensor]:
    """Invert :func:`relative_state` back to world-frame quantities.

    Args:
        rel_state: Relative-state tensor, shape ``(M, 2*nj + 15 + 3*K)``.
        p_ref: Reference position (world), shape ``(M, 3)``.
        q_ref: Reference quaternion (world, xyzw), shape ``(M, 4)``.
        num_joints: Number of joints ``nj``.
        num_extra_frames: Number of extra body frames ``K`` to decode.

    Returns:
        Dict with ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w``,
        ``root_link_lin_vel_w``, ``root_link_ang_vel_w``. When ``K > 0`` also includes
        ``frame_pos_w``, shape ``(..., K, 3)`` in the frame order used by
        :func:`relative_state` (extra frames are position-only).
    """
    nj = num_joints
    joint_pos = rel_state[..., 0:nj]
    joint_vel = rel_state[..., nj : 2 * nj]
    base = frame_block_start(nj, 0)
    p_w, q_w, v_w, w_w = _block_to_absolute(
        rel_state[..., base : base + ROOT_BLOCK_DIM], p_ref, q_ref
    )
    out = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos_w": p_w,
        "root_quat_w": q_w,
        "root_link_lin_vel_w": v_w,
        "root_link_ang_vel_w": w_w,
    }
    if num_extra_frames > 0:
        fp = []
        for k in range(num_extra_frames):
            b = frame_block_start(nj, 1 + k)
            fp.append(_position_to_absolute(
                rel_state[..., b : b + EE_BLOCK_DIM], p_ref, q_ref
            ))
        out["frame_pos_w"] = torch.stack(fp, dim=-2)
    return out


def ori6d_orthonormality_penalty(d6: torch.Tensor) -> torch.Tensor:
    """Penalty encouraging the 6D rep to encode two orthonormal vectors.

    Args:
        d6: 6D rotation representation, shape ``(..., 6)``.

    Returns:
        Scalar penalty: squared dot product plus squared unit-norm deviations.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    dot = (a1 * a2).sum(dim=-1)
    n1 = a1.norm(dim=-1)
    n2 = a2.norm(dim=-1)
    return dot.pow(2).mean() + (n1 - 1.0).pow(2).mean() + (n2 - 1.0).pow(2).mean()


def scan_grid_counts(resolution: float, size: Sequence[float]) -> tuple[int, int]:
    """Sample counts ``(n_x, n_y)`` of an IsaacLab ``GridPatternCfg``.

    Uses ``torch.arange`` with the same inclusive ``+1e-9`` endpoint nudge that
    ``isaaclab.sensors.ray_caster.patterns.grid_pattern`` uses, so the count matches the
    library's ray count exactly -- including at the float-precision edges, where a
    reimplementation with ``int(size / res) + 1`` differs by one.

    Args:
        resolution: Grid spacing in metres.
        size: ``(size_x, size_y)`` full extents in metres; ``size[0]`` is the x/forward axis.

    Returns:
        ``(n_x, n_y)`` -- samples along forward and lateral respectively.
    """
    sx, sy = float(size[0]), float(size[1])
    res = float(resolution)
    n_x = len(torch.arange(start=-sx / 2, end=sx / 2 + 1.0e-9, step=res))
    n_y = len(torch.arange(start=-sy / 2, end=sy / 2 + 1.0e-9, step=res))
    return n_x, n_y


def scan_grid_shape(
    resolution: float, size: Sequence[float], ordering: str = "xy"
) -> tuple[int, int]:
    """The ``(H, W)`` image fold of a FLAT height-scan row.

    This is the single definition of that fold, shared by the FM's CNN encoder, the recorded
    scan contract, and the sagittal-reflection augmentation. Two copies of it can disagree
    silently -- ``num_rays`` alone cannot distinguish a grid from its transpose -- so callers
    must go through here rather than reimplementing the arithmetic.

    ``grid_pattern`` builds the flat row with ``meshgrid(x, y, indexing=...)`` then
    ``flatten()``, so for the default ``ordering="xy"`` the result is y-outer / x-inner
    (``k = iy * n_x + ix``) and folding row-major gives rows = lateral, columns = forward.
    ``"yx"`` transposes both.

    Args:
        resolution: Grid spacing in metres.
        size: ``(size_x, size_y)`` full extents in metres.
        ordering: ``GridPatternCfg.ordering`` -- ``"xy"`` (the IsaacLab default) or ``"yx"``.

    Returns:
        ``(height, width)``, matching the row-major fold ``k = row * width + col``.

    Raises:
        ValueError: On an unknown ``ordering``.
    """
    n_x, n_y = scan_grid_counts(resolution, size)
    if ordering == "xy":
        return n_y, n_x
    if ordering == "yx":
        return n_x, n_y
    raise ValueError(
        f"unknown GridPatternCfg ordering {ordering!r}; expected 'xy' or 'yx'. The fold of a "
        "flat scan row depends on it, and guessing would transpose the image."
    )


def scan_lateral_dim(ordering: str = "xy") -> int:
    """Which dim of a ``(batch, H, W)`` folded scan is the LATERAL (y) axis.

    The axis a sagittal reflection has to flip. ``1`` for ``"xy"`` (rows are lateral), ``2``
    for ``"yx"``. Derived from the same ``ordering`` :func:`scan_grid_shape` reads, so the
    mirror augmentation and the CNN fold cannot disagree about which way is sideways.

    Raises:
        ValueError: On an unknown ``ordering``.
    """
    if ordering == "xy":
        return 1
    if ordering == "yx":
        return 2
    raise ValueError(f"unknown GridPatternCfg ordering {ordering!r}; expected 'xy' or 'yx'.")


from .config import EXTERO_CLIP  # noqa: E402  (kept below the torch imports)

HEIGHT_SCAN_CLIP = EXTERO_CLIP["height_scan"]
"""Default symmetric clip band (metres) for the height-scan conditioning signal.

Matches the ``height_scan`` RL observation term's ``clip=(-1, 1)``. That agreement is the
right default but no longer a rule: when the generator has its OWN scanner the two signals
are different sensors, and :attr:`~robot_rl.flow_matching.config.FlowMatchingConfig.extero_clip`
widens the FM's band without touching the teacher's obs terms (whose distribution the critic
depends on). See :data:`~robot_rl.flow_matching.config.EXTERO_CLIP`, the real home.
"""

DEPTH_CLIP = EXTERO_CLIP["depth"]
"""Default symmetric clip band for the depth conditioning signal, in NORMALISED units.

Matches the ``depth_near`` / ``depth_far`` RL observation terms' ``clip=(-2, 2)``.
``mdp.depth_image`` already applies ``(d - 0.75) * 0.5``, so the recorded depth is in these
normalised units, not metres -- which is why this band differs from :data:`HEIGHT_SCAN_CLIP`.
Clamping depth at the scan's 1.0 would silently discard real readings.
"""


def sanitize_height_scan(height_scan: torch.Tensor, clip: float = HEIGHT_SCAN_CLIP) -> torch.Tensor:
    """Make a raw height scan finite and clamp it to ``[-clip, clip]``.

    Ray-cast rays that miss the (finite) mesh come back as ``-inf`` in Isaac Lab
    (``pos_z - inf - offset``). Left raw, a single such value makes the height normalizer's
    per-dimension mean non-finite and yields a NaN training loss.

    A miss maps to ``+clip``, the DROP-OFF end. This direction is load-bearing rather than
    arbitrary: a missed downward ray means "no ground within range", which is geometrically
    ground very far BELOW -- and the terrain does genuinely reach large positive values where
    the robot stands high on a staircase over a distant floor. Mapping it to ``-clip`` instead
    (as this did originally) encodes a hole as "ground a metre ABOVE my feet", i.e. a wall,
    which is a reading real geometry also produces on ascending stairs. Holes and walls then
    share a number and nothing downstream can separate them.

    Args:
        height_scan: Raw per-ray ground clearance, shape ``(..., K)``.
        clip: Symmetric clip magnitude in metres.

    Returns:
        Finite, clamped height scan with the same shape as the input.
    """
    return torch.nan_to_num(height_scan, nan=0.0, posinf=clip, neginf=clip).clamp(-clip, clip)


def sanitize_extero(
    values: torch.Tensor, source: str, clip: float | None = None
) -> torch.Tensor:
    """Make an exteroception signal finite and clamp it to its band.

    Both sources need the guard: the height scan records ``-inf`` for missed rays, and while
    ``mdp.depth_image`` should already return finite values, a single non-finite entry makes
    the normalizer's per-dimension mean non-finite and yields a NaN loss.

    Args:
        values: Raw exteroception values, shape ``(..., K)``.
        source: One of :data:`~robot_rl.flow_matching.config.EXTERO_CLIP`'s keys.
        clip: Band override (``FlowMatchingConfig.resolve_extero_clip()``). ``None`` uses the
            per-source default, which is what reproduces an older checkpoint exactly.

    Returns:
        Finite, clamped values with the same shape as the input.

    Raises:
        KeyError: On an unknown ``source``.
    """
    band = EXTERO_CLIP[source] if clip is None else float(clip)
    return sanitize_height_scan(values, clip=band)


def geodesic_angle_rot6d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between two 6-D rotation representations, in radians.

    Through the rotation GROUP rather than by differencing the raw channels, so the result is
    a real angle and a prediction's Gram-Schmidt residual is projected out instead of being
    counted as orientation error.

    Lives here rather than in either caller because it has two: the flow-matching trainer's
    ``channel_metrics`` reports it as ``ori_geodesic_rad`` and the token planner's
    reconstruction metrics report the same quantity, and the two are only comparable if they
    are the same function. They were the same function written twice.

    Args:
        a: ``(..., 6)`` rot6d.
        b: ``(..., 6)`` rot6d.

    Returns:
        ``(...)`` angles in ``[0, pi]``.
    """
    rel = rot6d_to_matrix(a).transpose(-1, -2) @ rot6d_to_matrix(b)
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.arccos(cos)
