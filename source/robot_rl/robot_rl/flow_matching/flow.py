# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rectified-flow / conditional-OT training objective and sampler.

Linear interpolant path: ``x_tau = (1 - tau) * x0 + tau * x1`` with ``x0 ~ N(0, I)``
and ``x1`` the (standardized) future window. The target vector field is the constant
``v* = x1 - x0``; the model regresses it with MSE over ALL F future tokens. Sampling
integrates ``dx/dtau = v_theta`` from ``tau = 0`` (noise) to ``tau = 1`` with Euler steps.

Conditioning is passed as SEPARATE tokens to the model (history / extero / command) --
it is NOT part of the denoised window, so there is no inpainting / known-mask / re-clamp.
History (phase) conditioning may be replaced by the model's learned null embedding for a
fraction of samples via ``history_null_mask`` (classifier-free-guidance / phase-agnostic).
"""

from __future__ import annotations

from typing import Sequence

import torch

from .transforms import rot6d_to_matrix, so3_log
from .window import resolve_reference_blocks, window_block_slices


def sample_flow_batch(
    x1: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a rectified-flow training triplet over the full future window.

    Args:
        x1: Target future window (standardized), shape ``(B, F, D)``.
        generator: Optional RNG for reproducibility.

    Returns:
        ``(x_tau, tau, target_v)`` with shapes ``(B, F, D)``, ``(B,)``, ``(B, F, D)``.
    """
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    tau = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=generator)
    t = tau.view(-1, *([1] * (x1.dim() - 1)))
    x_tau = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    return x_tau, tau, target_v


def frame_consistency_terms(
    pred_phys: torch.Tensor,
    data_phys: torch.Tensor,
    dt: float,
    num_joints: int,
    num_extra_frames: int = 0,
    blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """Per-sample VELOCITY finite-difference consistency for the base block.

    Operates on ``base_pose`` / ``base_vel`` only (joints are handled separately, and the ee
    block is position-only so it carries no velocity to tie), in DENORMALIZED (physical)
    units, over the pure F-node future prediction.

    Both halves are exact under the window representation, for the same reason: every node
    shares ONE anchor, so the anchor cancels out of a difference between nodes.

    - position: ``base_pose[0:3]`` is ``R_actT (p_ref - p_act)`` with ``p_act``, ``R_act``
      constant across nodes, so its forward difference is exactly ``R_actT v_ref`` --
      which is ``base_vel[0:3]``.
    - orientation: ``base_pose[3:9]`` is a rotation, and rotations do NOT difference
      additively, so this goes through the group. The block decodes to
      ``R_k = R_actT R_ref_k``, hence
      ``Log(R_{k+1} R_kT) = R_actT Log(R_ref_{k+1} R_ref_kT)`` -- the anchor cancels (it
      conjugates out) and the result is comparable to ``base_vel[3:6]`` directly. The
      transpose is on the RIGHT factor; it was on the left while ``base_pose`` carried the
      opposite sign.

    Args:
        pred_phys / data_phys: ``(B, F, D)`` predicted / target window, physical units.
        dt: Control step period.
        num_joints: ``nj``; num_extra_frames: ``K``; blocks: the packed selection.

    Returns:
        ``vel_per_sample`` ``(B,)`` (physical units).

    Raises:
        ValueError: If the selection lacks ``base_pose`` or ``base_vel`` -- the caller
            gates on that (see :func:`flow_matching_loss`), so reaching here means a
            weight outlived the block it relates.
    """
    slc = window_block_slices(num_joints, num_extra_frames, blocks)
    if "base_pose" not in slc or "base_vel" not in slc:
        raise ValueError(
            "frame_consistency_terms ties base_pose to base_vel, but this window packs "
            f"{sorted(slc)}. Zero vel_consistency_weight for "
            "this block selection."
        )
    bp, bv = slc["base_pose"].start, slc["base_vel"].start
    x = pred_phys
    pos, ori6d = x[..., bp:bp + 3], x[..., bp + 3:bp + 9]
    lin, ang = x[..., bv:bv + 3], x[..., bv + 3:bv + 6]

    lin_fd = (pos[:, 1:] - pos[:, :-1]) / dt
    vel = ((lin_fd - lin[:, :-1]) ** 2).mean(dim=(1, 2))
    # Gram-Schmidt: the prediction is a raw 9-vector, so it is only approximately a
    # rotation, and so3_log needs a real one.
    rmat = rot6d_to_matrix(ori6d)                 # (B, F, 3, 3) = R_actT R_ref
    ang_fd = so3_log(rmat[:, 1:] @ rmat[:, :-1].transpose(-1, -2)) / dt
    vel = vel + ((ang_fd - ang[:, :-1]) ** 2).mean(dim=(1, 2))

    return vel / 2


def joint_vel_consistency_term(
    pred_phys: torch.Tensor,
    dt: float,
    num_joints: int,
    num_extra_frames: int = 0,
    blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """Per-sample joint-velocity consistency: ``(joint_pos[t+1]-joint_pos[t])/dt ~ joint_vel[t]``.

    Returns ``(B,)`` in physical units (rad/s). Operates on the pure F-node prediction.
    Frame-invariant blocks, so the window representation changes nothing but the offsets.

    Raises:
        ValueError: If the selection lacks ``joint_pos`` or ``joint_vel``.
    """
    slc = window_block_slices(num_joints, num_extra_frames, blocks)
    if "joint_pos" not in slc or "joint_vel" not in slc:
        raise ValueError(
            "joint_vel_consistency_term ties joint_pos to joint_vel, but this window packs "
            f"{sorted(slc)}. Zero joint_vel_consistency_weight for this block selection."
        )
    jp = pred_phys[..., slc["joint_pos"]]
    jv = pred_phys[..., slc["joint_vel"]]
    jp_fd = (jp[:, 1:] - jp[:, :-1]) / dt
    return ((jp_fd - jv[:, :-1]) ** 2).mean(dim=(1, 2))


def fk_consistency_terms(
    pred_phys: torch.Tensor,
    num_joints: int,
    num_extra_frames: int,
    fk: object,
    blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """Per-sample forward-kinematics POSITION consistency for the EE frames.

    Ties each predicted end-effector position to the FK of the predicted joints. This is
    what couples the two halves of the prediction: without it the joints and the ee
    positions are free to describe different postures.

    Direct under the window representation. The ``ee_pos`` block is already the position in
    THAT node's own reference base frame, which is exactly the frame ``FKChain`` reports in
    -- so this is a plain residual. It used to have to reconstruct the base frame from the
    predicted root pose first, which also let a root error partially cancel an ee error;
    the coupling this term exists to enforce is tighter now.

    Position-only, because the window's ee block is position-only.

    On the DENORMALIZED F-node prediction. ``fk`` is a
    :class:`~motion_gen.fk.FKChain` exposing ``pose_and_jacobian``.

    Returns:
        ``fk_pose_per_sample``, shape ``(B,)``.

    Raises:
        ValueError: If the selection lacks ``joint_pos`` or ``ee_pos`` -- there is no
            posture to couple without both.
    """
    x = pred_phys
    b_, f_, _ = x.shape
    nj, k = num_joints, num_extra_frames

    slc = window_block_slices(nj, k, blocks)
    if "joint_pos" not in slc or "ee_pos" not in slc:
        raise ValueError(
            "fk_consistency_terms couples joint_pos to ee_pos, but this window packs "
            f"{sorted(slc)}. Zero fk_pose_weight for this block selection."
        )
    q = x[..., slc["joint_pos"]].reshape(b_ * f_, nj)
    fk_pos, _fk_rot, _jv, _jw = fk.pose_and_jacobian(q)            # (M,K,3), ...
    fk_pos = fk_pos.reshape(b_, f_, k, 3)

    ee = x[..., slc["ee_pos"]].reshape(b_, f_, k, 3)
    return ((ee - fk_pos) ** 2).mean(dim=(1, 2, 3))


def body_pose_terms(
    pred_phys: torch.Tensor,
    data_phys: torch.Tensor,
    num_joints: int,
    num_extra_frames: int,
    body_fk: object,
    ori_scale: float = 1.0,
    blocks: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Per-sample body-link pose error against the target -- the CLF's own metric.

    The teacher this generator serves runs ``tracking_mode="body"``: the CLF grades the 13
    ``<body>@root:*`` frames plus the pelvis and the WORLD end-effector frames, and drops
    ``joint:*`` entirely. The flow loss, meanwhile, is a per-joint MSE in normalized angle
    space. So the generator optimises a metric its consumer is not graded on -- a hip error
    and a wrist error count the same, though one moves the foot an order of magnitude
    further. These terms put the prediction back in the CLF's units.

    ACCURACY against the target, not self-consistency: unlike
    :func:`fk_consistency_terms` (which ties the predicted joints to the predicted
    ``ee_pos`` block) this compares ``FK(predicted q)`` with ``FK(target q)``. It therefore
    adds no information the model does not already have -- it is a change of METRIC, the
    Jacobian reweighting -- and needs no extra window block, which matters because the slim
    teacher dropped ``ee_pos``.

    Two terms, because they fail differently:

    * ``body_pose_root`` -- poses in the reference's own base (pelvis) frame. Pure posture,
      and the one that cannot be gamed: it is independent of ``base_pose`` entirely.
    * ``body_pose_world`` -- the same poses composed with the predicted ``base_pose``. This
      is what the CLF's world ee frames see, so it is the term that cares about landing a
      foot on a particular stair tread. It is computed in the ANCHOR frame, but the number
      is the WORLD-frame error exactly: prediction and target share one anchor, and a rigid
      transform preserves distances and angles, so ``|R_aT(p1-p_a) - R_aT(p2-p_a)| =
      |p1-p2|``. No world coordinate enters, which is what keeps the representation
      deploy-safe.

      Because it folds the base error into all 13 bodies, it also re-weights base position
      UP by that multiplicity, in physical metres -- outside the pooled normalizer that
      currently squashes ``base_pos`` (node 0's error/signal is ~0.87, barely better than
      predicting the mean). That is deliberate.

      The cost of composing is that a base error can partially cancel a joint error, which
      is exactly why ``body_pose_root`` is worth keeping alongside rather than instead --
      see :func:`fk_consistency_terms`, which was rewritten to stop doing precisely this.

    Both are squared errors in m^2, with the orientation half converted through
    ``ori_scale`` (metres per radian) so the two halves of one geometric quantity can be
    weighted together. Orientation goes through the group (``so3_log`` of the relative
    rotation), which is stable and finite-gradient at coincidence -- where a pose loss
    spends its time once it is working.

    On the DENORMALIZED prediction, and expensive, so callers should τ-gate it.

    Args:
        pred_phys / data_phys: ``(B, F, D)`` predicted / target window, physical units.
        num_joints: ``nj``.
        num_extra_frames: The window's ee-frame count ``K`` -- for the block layout only.
            ``body_fk`` has its own, larger frame set and its count is read off its output.
        body_fk: An :class:`~motion_gen.fk.FKChain` over the tracked BODY frames (13 for
            the G1, ``motion_gen.joint_names.G1_BODY_TRACK_FRAMES``), exposing
            ``pose_and_jacobian``.
        ori_scale: Metres per radian for the orientation half.
        blocks: The packed selection.

    Returns:
        ``{"body_pose_root": (B,)}``, plus ``"body_pose_world"`` when the selection packs
        ``base_pose`` (there is nothing to compose with otherwise).

    Raises:
        ValueError: If the selection lacks ``joint_pos`` -- there is no posture to grade.
    """
    slc = window_block_slices(num_joints, num_extra_frames, blocks)
    if "joint_pos" not in slc:
        raise ValueError(
            "body_pose_terms grades the forward kinematics of joint_pos, but this window "
            f"packs {sorted(slc)}. Zero body_pose_root_weight / body_pose_world_weight "
            "for this block selection."
        )
    b_, f_, _ = pred_phys.shape

    def fk_of(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Root-frame body poses: ``(B,F,Nb,3)`` positions, ``(B,F,Nb,3,3)`` rotations."""
        q = x[..., slc["joint_pos"]].reshape(b_ * f_, num_joints)
        pos, rot, _jv, _jw = body_fk.pose_and_jacobian(q)
        nb = pos.shape[-2]
        return pos.reshape(b_, f_, nb, 3), rot.reshape(b_, f_, nb, 3, 3)

    def err(p_a, r_a, p_b, r_b) -> tuple[torch.Tensor, torch.Tensor]:
        """``(position m^2, orientation rad^2)`` per sample, averaged over nodes+frames."""
        pos = ((p_a - p_b) ** 2).sum(dim=-1).mean(dim=(1, 2))              # m^2
        ang = so3_log(r_a.transpose(-1, -2) @ r_b)                        # (B,F,Nb,3)
        return pos, (ang ** 2).sum(dim=-1).mean(dim=(1, 2))                # m^2, rad^2

    def combine(name: str, halves: tuple[torch.Tensor, torch.Tensor]) -> dict:
        """The weighted term plus its two halves, kept separately for logging.

        The halves are what makes ``ori_scale`` tunable on evidence instead of by guess:
        they are in m^2 and rad^2, so their ratio says how much the orientation is actually
        contributing at the current setting. Only the combined key enters the loss.
        """
        pos, ori = halves
        return {name: pos + (ori_scale ** 2) * ori,
                f"{name}_pos": pos.detach(), f"{name}_ori": ori.detach()}

    p_pred, r_pred = fk_of(pred_phys)
    p_tgt, r_tgt = fk_of(data_phys)
    out = combine("body_pose_root", err(p_pred, r_pred, p_tgt, r_tgt))

    if "base_pose" in slc:
        bp = slc["base_pose"].start

        def compose(x: torch.Tensor, p_root: torch.Tensor, r_root: torch.Tensor):
            """Lift root-frame body poses through the window's own ``base_pose``."""
            base_p = x[..., bp:bp + 3]                                    # (B,F,3)
            base_r = rot6d_to_matrix(x[..., bp + 3:bp + 9])               # (B,F,3,3)
            return (
                base_p[..., None, :] + torch.einsum("bfij,bfkj->bfki", base_r, p_root),
                base_r[..., None, :, :] @ r_root,
            )

        out.update(combine("body_pose_world",
                           err(*compose(pred_phys, p_pred, r_pred),
                               *compose(data_phys, p_tgt, r_tgt))))
    return out


#: Bit index of each root DOF in the anchor mask (bit SET = that DOF is FREE). Mirrors
#: ``motion_gen.world_lift._ANCHOR_DOF_NAMES``; kept here so ``flow`` needs no motion_gen
#: import, and pinned by a test against the authority.
_ANCHOR_BIT = {"downrange": 0, "lateral": 1, "height": 2, "roll": 3, "pitch": 4, "yaw": 5}


def _planar_terrain_components(
    err: torch.Tensor, base: int, ang: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The base-position error resolved onto the TERRAIN's axes, in normalized units.

    Projects in metres onto the terrain's downrange/lateral directions -- which are exactly
    orthogonal there -- then divides each by the std ALONG that direction, so both come back
    on the same scale as the other normalized channels. Doing it the other way (projecting the
    already-normalized error) is wrong under the 4.3x x/y anisotropy; see
    :func:`pinned_axis_flow_loss`.

    Args:
        err: ``(B, F, D)`` normalized prediction error.
        base: Column where the ``base_pose`` block starts.
        ang: ``(B, F)`` angle from the window frame to the terrain frame.
        std: ``(2,)`` normalizer std for the x and y position channels.

    Returns:
        ``(downrange, lateral)``, each ``(B, F)``, in normalized units.
    """
    e_m = err[..., base:base + 2] * std                    # metres
    cos_a, sin_a = torch.cos(ang), torch.sin(ang)
    std_dr = ((std[0] * cos_a) ** 2 + (std[1] * sin_a) ** 2).sqrt().clamp_min(1e-12)
    std_lat = ((std[0] * sin_a) ** 2 + (std[1] * cos_a) ** 2).sqrt().clamp_min(1e-12)
    downrange = (e_m[..., 0] * cos_a + e_m[..., 1] * sin_a) / std_dr
    lateral = (-e_m[..., 0] * sin_a + e_m[..., 1] * cos_a) / std_lat
    return downrange, lateral


def pinned_axis_split(
    err: torch.Tensor,
    anchor_mask: torch.Tensor,
    mask_frame_yaw_rel: torch.Tensor | None,
    state_norm: object | None,
    num_joints: int,
    num_extra_frames: int,
    blocks: Sequence[str] | None,
) -> tuple[float, float]:
    """Mean squared flow error split by whether the anchor mask PINS that coordinate.

    The diagnostic the whole terrain-registration experiment turns on: the generator's error
    on coordinates the terrain fixes (which it gets wrong) against its error on coordinates
    that follow the robot (which it gets right). A single ``flow_mse`` averages the two and
    hides exactly the effect being chased.

    The split is over COORDINATES, not raw channels: the base-position error is resolved onto
    the terrain's downrange/lateral axes first (:func:`_planar_terrain_components`), because
    "downrange is pinned, lateral is free" is a statement about the terrain frame and the
    window's x/y are the robot's. Height and the 6D rotation are axis-aligned and split
    directly. Every other block -- velocities, joints -- is unpinned by definition: the anchor
    mask describes the root placement only, so they count as free.

    Both halves are per-dimension means, so they are directly comparable to each other and to
    ``flow_mse``.

    Args:
        err: ``(B, F, D)`` normalized prediction error.
        anchor_mask: ``(B, F)`` anchor bitmasks (bit SET = FREE).
        mask_frame_yaw_rel: ``(B,)`` window-to-terrain angle, or ``None`` if aligned.
        state_norm: State normalizer (for the position std); ``None`` = isotropic.
        num_joints: ``nj``; num_extra_frames: ``K``; blocks: the window's block selection.

    Returns:
        ``(masked, unmasked)`` -- mean squared error over pinned and over free coordinates.
        A half with no members reports ``nan``.
    """
    sq = err ** 2
    slices = window_block_slices(num_joints, num_extra_frames, blocks)
    if "base_pose" not in slices:
        return float("nan"), float(sq.mean())
    base = slices["base_pose"].start

    def is_pinned(name: str) -> torch.Tensor:
        return ((anchor_mask >> _ANCHOR_BIT[name]) & 1 == 0).to(err.dtype)

    p_dr, p_lat = is_pinned("downrange"), is_pinned("lateral")
    p_h = is_pinned("height")
    p_rot = is_pinned("roll") * is_pinned("pitch") * is_pinned("yaw")

    std = torch.ones(2, dtype=err.dtype, device=err.device)
    if state_norm is not None:
        std = state_norm.std[base:base + 2].to(err.dtype).to(err.device)
    ang = (torch.zeros_like(p_dr) if mask_frame_yaw_rel is None
           else mask_frame_yaw_rel.to(err.dtype).unsqueeze(-1).expand_as(p_dr))
    dr, lat = _planar_terrain_components(err, base, ang, std)

    z, rot = sq[..., base + 2], sq[..., base + 3:base + 9].sum(dim=-1)
    pin_sq = (p_dr * dr ** 2 + p_lat * lat ** 2 + p_h * z + p_rot * rot).sum()
    pin_n = (p_dr + p_lat + p_h + 6.0 * p_rot).sum()
    # Everything the mask does not describe is free: the two planar axes it leaves free,
    # height/rotation when free, and every other block outright.
    other = sq.sum() - sq[..., base:base + 9].sum()
    other_n = float(err.shape[0] * err.shape[1] * (err.shape[2] - 9))
    free_sq = ((1 - p_dr) * dr ** 2 + (1 - p_lat) * lat ** 2
               + (1 - p_h) * z + (1 - p_rot) * rot).sum() + other
    free_n = ((1 - p_dr) + (1 - p_lat) + (1 - p_h) + 6.0 * (1 - p_rot)).sum() + other_n
    masked = float(pin_sq / pin_n) if float(pin_n) > 0 else float("nan")
    unmasked = float(free_sq / free_n) if float(free_n) > 0 else float("nan")
    return masked, unmasked


def pinned_axis_flow_loss(
    err: torch.Tensor,
    sq: torch.Tensor,
    anchor_mask: torch.Tensor,
    mask_frame_yaw_rel: torch.Tensor | None,
    weight: float,
    state_norm: object | None,
    num_joints: int,
    num_extra_frames: int,
    blocks: Sequence[str] | None,
) -> torch.Tensor:
    """Flow MSE with the anchor mask's PINNED root DOFs weighted up, renormalized to mean 1.

    A pinned DOF is one the reference keeps from the terrain rather than re-anchoring onto the
    robot -- height on flat ground, downrange + height + yaw on stairs and boxes. Those are
    precisely the axes the generator fails on (its predicted stair segment slides along with a
    displaced robot instead of staying on the terrain), so this concentrates gradient there.

    Three things make this less trivial than a channel mask:

    * **The pinned planar axis is not a channel.** ``downrange`` is named in the TERRAIN's
      frame while the window's x/y are the snap robot's yaw frame, so the pinned direction is
      a rotated vector, and the weight is a quadratic form ``I + (w-1) u u^T`` rather than a
      diagonal.
    * **x and y are standardized differently** -- measured 0.442 vs 0.102 on the shipped
      teacher, a 4.3x anisotropy, because the robot travels far downrange and little
      laterally. A metric direction ``u`` therefore appears in normalized coordinates along
      ``S^-1 u``, not ``u``; projecting on ``u`` directly would leak most of a pinned
      downrange weight onto lateral error.
    * **Yaw is not a channel either** -- the orientation block is a 6D rotation. It is
      weighted only when roll, pitch AND yaw are all pinned, which is exactly the terrain
      masks; flat frees yaw, so flat gets no orientation weight and nothing is distorted.

    One residual approximation, measured rather than assumed. Because the base loss is a
    per-channel normalized MSE, a metric displacement along a DIAGONAL pinned axis also shows
    up in the un-weighted channel term, diluting the boost: the realised ratio between a
    pinned-axis and a free-axis error is the nominal ``weight`` at 0/90/180 degrees of
    relative heading and falls to about ``weight / 5`` at 45. On the shipped library that
    barely bites -- over terrain-pinned windows of the depth_v5 recording, 90.7% of headings
    are within 10 degrees of an axis and 96.0% within 20 (the distribution is bimodal at 0 and
    180, the robot traversing the tile forwards or backwards), leaving ~4% in the diagonal
    band. Removing it entirely would mean weighting in metric units throughout, which changes
    the xy/other balance even at ``weight = 1``.

    Weights are renormalized by their own sum, so the returned value has the same scale as the
    unweighted mean and a with/without comparison does not also change the effective LR.

    Args:
        err: ``(B, F, D)`` prediction error in NORMALIZED units.
        sq: ``err ** 2``, precomputed.
        anchor_mask: ``(B, F)`` integer anchor bitmasks.
        mask_frame_yaw_rel: ``(B,)`` angle from the window's frame to the terrain's, or
            ``None`` to treat them as aligned.
        weight: Multiplier on pinned DOFs (``1.0`` would be a no-op).
        state_norm: State normalizer, for the per-channel std used in the anisotropy
            correction. ``None`` falls back to isotropic.
        num_joints: ``nj``.
        num_extra_frames: ``K``.
        blocks: The window's block selection.

    Returns:
        Scalar weighted mean squared error.
    """
    slices = window_block_slices(num_joints, num_extra_frames, blocks)
    if "base_pose" not in slices:
        return sq.mean()
    base = slices["base_pose"].start
    extra = float(weight) - 1.0

    def is_pinned(name: str) -> torch.Tensor:
        """(B, F) float: 1 where that DOF is FIXED (bit CLEAR = pinned)."""
        return ((anchor_mask >> _ANCHOR_BIT[name]) & 1 == 0).to(err.dtype)

    w = torch.ones_like(sq)
    # Height and the orientation block are axis-aligned, so they ARE channel weights.
    w[..., base + 2] = 1.0 + extra * is_pinned("height")
    rot_pinned = is_pinned("roll") * is_pinned("pitch") * is_pinned("yaw")
    w[..., base + 3:base + 9] = (1.0 + extra * rot_pinned).unsqueeze(-1)

    total = (w * sq).sum()
    w_sum = w.sum()

    # The planar part: a quadratic form along the pinned direction(s), taken in normalized
    # coordinates via S^-1 u so the standardization anisotropy above cannot smear it. It is a
    # rank-1 bump on the 2x2 xy block, so it contributes a SCALAR per node -- accumulated into
    # the total rather than into the per-channel tensor, which would broadcast it over every
    # channel in the window.
    p_dr, p_lat = is_pinned("downrange"), is_pinned("lateral")
    if float((p_dr + p_lat).sum()) > 0.0:
        std = torch.ones(2, dtype=err.dtype, device=err.device)
        if state_norm is not None:
            std = state_norm.std[base:base + 2].to(err.dtype).to(err.device)
        ang = (torch.zeros_like(p_dr) if mask_frame_yaw_rel is None
               else mask_frame_yaw_rel.to(err.dtype).unsqueeze(-1).expand_as(p_dr))
        # Project in METRES, then divide by that axis's OWN std. Doing it the other way --
        # projecting the already-normalized error onto the normalized image of the metric axis
        # -- is wrong under anisotropy: at std (0.442, 0.102) the normalized images of the
        # metric downrange and lateral axes are only 26 degrees apart at 45 deg of heading, so
        # a purely LATERAL error would pick up 81% of the downrange weight. Metric projection
        # gives it exactly zero, and reduces to plain channel weighting at zero heading.
        e_m = err[..., base:base + 2] * std                      # (B, F, 2) metres
        for pinned, (ux, uy) in ((p_dr, (torch.cos(ang), torch.sin(ang))),
                                 (p_lat, (-torch.sin(ang), torch.cos(ang)))):
            if float(pinned.sum()) == 0.0:
                continue
            along = e_m[..., 0] * ux + e_m[..., 1] * uy          # (B, F) metres along the axis
            # The std ALONG that direction, so the term is commensurate with the normalized
            # squared errors it is added to.
            std_u = ((std[0] * ux) ** 2 + (std[1] * uy) ** 2).sqrt().clamp_min(1e-12)
            total = total + (extra * pinned * (along / std_u) ** 2).sum()
            # The quadratic form's TRACE is what the mean must divide by: a rank-1 bump of
            # (weight - 1) adds exactly that to the 2x2 block's trace.
            w_sum = w_sum + extra * pinned.sum()
    return total / w_sum.clamp_min(1e-12)


def flow_matching_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    history: torch.Tensor,
    extero: torch.Tensor,
    cmd_tokens: torch.Tensor | None = None,
    history_null_mask: torch.Tensor | None = None,
    state_norm: object | None = None,
    dt: float = 0.02,
    num_joints: int = 0,
    num_extra_frames: int = 0,
    vel_weight: float = 0.0,
    joint_vel_weight: float = 0.0,
    fk: object | None = None,
    fk_pose_weight: float = 0.0,
    body_fk: object | None = None,
    body_pose_root_weight: float = 0.0,
    body_pose_world_weight: float = 0.0,
    body_pose_ori_scale: float = 1.0,
    tau_min: float = 0.0,
    blocks: Sequence[str] | None = None,
    report_consistency: bool = False,
    anchor_mask: torch.Tensor | None = None,
    mask_frame_yaw_rel: torch.Tensor | None = None,
    pinned_axis_weight: float = 1.0,
    advantage: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the flow-matching loss (+ optional consistency / FK terms).

    The flow MSE is over ALL F future tokens (the conditioning is separate tokens, not
    part of the window). All auxiliary terms act on the predicted endpoint ``x1_hat``
    (denormalized). Velocity + joint-velocity consistency are ungated; the FK and body-pose
    terms are **τ-gated** (only ``tau >= tau_min``, where ``x1_hat ≈ x1``).

    Every auxiliary term relates two blocks, so each is skipped unless ``blocks`` packs
    both. ``dataset.resolve_block_selection`` already zeroes the corresponding weight at
    build time; this is the second line of the same defence, for a caller that assembles
    the kwargs itself.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        x1: Target future window (standardized), ``(B, F, D)``.
        history: Clean history (phase) conditioning (standardized), ``(B, N, D)``.
        extero: Exteroception conditioning (standardized), flat ``(B, T*K)``.
        cmd_tokens: Strided future command tokens (normalized) or ``None``.
        history_null_mask: ``(B,)`` bool; where True the history is replaced by the
            model's learned null embedding (classifier-free-guidance / phase-agnostic).
        state_norm: State :class:`Normalizer` (needed to denormalize); ``None`` disables
            all consistency/FK terms.
        dt: Control step period (seconds) for the finite differences.
        num_joints: ``nj``; num_extra_frames: ``K``.
        vel_weight / joint_vel_weight: consistency weights.
        fk: An :class:`~motion_gen.fk.FKChain` (or ``None``).
        fk_pose_weight: FK position-consistency weight.
        tau_min: τ-gate threshold for the FK and body-pose terms (0 disables gating).
        blocks: The window's block selection; defaults to all five.
        report_consistency: Compute ``vel_consistency`` as a METRIC even when its
            weight is 0. For validation; the training step should leave it off.

    Returns:
        Dict with ``loss`` and detached metrics: ``flow_mse`` and, when enabled,
        ``vel_consistency`` / ``joint_vel_consistency`` /
        ``fk_pose``.
    """
    x_tau, tau, target_v = sample_flow_batch(x1)
    v = model(x_tau, tau, history, extero, cmd_tokens, history_null_mask=history_null_mask,
              advantage=advantage)
    err = v - target_v
    sq = err ** 2
    # Reported UNWEIGHTED whatever the weighting, so a weighted run's curves stay directly
    # comparable with an unweighted one's -- that comparison is the point of the option.
    flow_mse = sq.mean()
    train_loss = flow_mse
    if anchor_mask is not None and float(pinned_axis_weight) != 1.0:
        train_loss = pinned_axis_flow_loss(
            err, sq, anchor_mask, mask_frame_yaw_rel, float(pinned_axis_weight),
            state_norm, num_joints, num_extra_frames, blocks,
        )
    out = {"loss": train_loss, "flow_mse": flow_mse.detach()}

    packed = set(resolve_reference_blocks(blocks))
    has_base = {"base_pose", "base_vel"} <= packed
    has_joints = {"joint_pos", "joint_vel"} <= packed
    do_base = state_norm is not None and has_base and (vel_weight > 0.0
                                                       or report_consistency)
    do_jvel = state_norm is not None and has_joints and joint_vel_weight > 0.0
    do_fk = (fk is not None and state_norm is not None and fk_pose_weight > 0.0
             and {"joint_pos", "ee_pos"} <= packed)
    want_body_world = body_pose_world_weight > 0.0 and "base_pose" in packed
    do_body = (body_fk is not None and state_norm is not None and "joint_pos" in packed
               and (body_pose_root_weight > 0.0 or want_body_world))
    if do_base or do_jvel or do_fk or do_body:
        t = tau.view(-1, *([1] * (x1.dim() - 1)))
        x1_hat = x_tau + (1.0 - t) * v  # predicted endpoint along the linear path
        pred_phys = state_norm.denormalize(x1_hat)
        data_phys = state_norm.denormalize(x1)
        gate = tau >= tau_min  # (B,) bool; gates the FK + body-pose terms
        if do_base:
            vel_ps = frame_consistency_terms(
                pred_phys, data_phys, dt, num_joints, num_extra_frames, blocks)
            vel = vel_ps.mean()                                 # ungated
            out["vel_consistency"] = vel.detach()
            # Reported even at zero weight (``report_consistency``), so the pose/velocity
            # drift the term used to suppress stays VISIBLE after switching it off -- but
            # only where it is asked for, so the training step does not pay for a metric.
            if vel_weight > 0.0:
                out["loss"] = out["loss"] + vel_weight * vel
        if do_jvel:
            jvel = joint_vel_consistency_term(
                pred_phys, dt, num_joints, num_extra_frames, blocks).mean()   # ungated
            out["joint_vel_consistency"] = jvel.detach()
            out["loss"] = out["loss"] + joint_vel_weight * jvel
        if do_fk:
            # Run FK only on the τ-gated samples (it is the expensive term).
            idx = gate.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                fkp = fk_consistency_terms(
                    pred_phys[idx], num_joints, num_extra_frames, fk, blocks).mean()
            else:
                fkp = flow_mse.new_zeros(())
            out["fk_pose"] = fkp.detach()
            out["loss"] = out["loss"] + fk_pose_weight * fkp
        if do_body:
            # τ-gated with the FK term, and for the same reason: two FK evaluations (the
            # prediction and the target) are the most expensive thing in this function.
            idx = gate.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                body = body_pose_terms(
                    pred_phys[idx], data_phys[idx], num_joints, num_extra_frames,
                    body_fk, body_pose_ori_scale, blocks)
                body = {k: v.mean() for k, v in body.items()}
            else:
                zero = flow_mse.new_zeros(())
                body = {f"body_pose_{s}{h}": zero for s in ("root", "world")
                        for h in ("", "_pos", "_ori")}
            weights = {"body_pose_root": body_pose_root_weight,
                       "body_pose_world": body_pose_world_weight}
            for name, value in body.items():
                out[name] = value.detach()
                weight = weights.get(name, 0.0)      # the _pos / _ori halves are diagnostics
                if weight > 0.0:
                    out["loss"] = out["loss"] + weight * value
    return out


@torch.no_grad()
def sample_trajectory(
    model: torch.nn.Module,
    history: torch.Tensor,
    extero: torch.Tensor,
    cmd_tokens: torch.Tensor | None,
    future_len: int,
    state_dim: int,
    num_steps: int = 8,
    history_null_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    zero_noise: bool = False,
    guidance_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    advantage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate the F-step future window by Euler-integrating the learned flow.

    The conditioning (history / extero / command) is fed as separate tokens, held fixed
    across integration; there is no inpainting / re-clamp. The output is in the
    standardized space; the caller denormalizes and Gram-Schmidts the orientation block.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        history: History (phase) conditioning (standardized), ``(B, N, D)``.
        extero: Exteroception conditioning (standardized), flat ``(B, T*K)``.
        cmd_tokens: Strided command tokens (normalized) or ``None``, held fixed.
        future_len: Future window length ``F``.
        state_dim: Per-step state width ``D``.
        num_steps: Number of Euler integration steps.
        history_null_mask: Optional ``(B,)`` bool to null the history (guidance).
        generator: Optional RNG for the initial noise.
        zero_noise: Integrate from ``x0 = 0`` instead of Gaussian noise, making the sample a
            deterministic function of the conditioning. Off by default (Gaussian is what
            training matches). Set it to compare two runs of the same conditioning without the
            per-call draw masking the difference -- e.g. sim2sim, where the deploy-side
            ``GenTrackerPolicy`` exposes the same switch. Note this is OFF the training
            distribution: ``x0 = 0`` is not a typical draw, so treat the output as a diagnostic
            trace rather than a representative rollout.

        guidance_scale: Classifier-free guidance weight ``w`` over the HISTORY conditioning.
            ``1.0`` (default) = plain conditional sampling, the historical behaviour.
            ``w > 1`` integrates ``v_null + w (v_cond - v_null)``, extrapolating along the
            direction the history contributes and so making the sample adhere to it more
            strongly. The model already supports this: it was trained with the history
            dropped on ``null_cond_prob`` of samples, so ``v_null`` is learned, not improvised.

            Worth knowing before turning it up in a closed loop: guidance makes the model
            trust the history MORE, and in autoregressive rollout the history is the model's
            own (possibly drifting) output -- so it should help teacher forcing more than the
            closed loop, and past some ``w`` it may make the closed loop worse by committing
            harder to a bad plan. It also samples off the data distribution, which for a
            reference the robot has to physically track is a sharper risk than it is for
            images. Sweep it against TRACKING, not just flow MSE.

    Returns:
        Generated future window (standardized), shape ``(B, F, D)`` -- all F are novel.
    """
    b = history.shape[0]
    device = history.device
    if noise is not None:
        if tuple(noise.shape) != (b, future_len, state_dim):
            raise ValueError(f"noise must be {(b, future_len, state_dim)}, got {tuple(noise.shape)}")
        x = noise.to(device=device, dtype=history.dtype)
    elif zero_noise:
        x = torch.zeros(b, future_len, state_dim, device=device, dtype=history.dtype)
    else:
        x = torch.randn(b, future_len, state_dim, device=device, dtype=history.dtype,
                        generator=generator)
    dt = 1.0 / num_steps
    guided = float(guidance_scale) != 1.0
    if guided:
        # Classifier-free guidance over the HISTORY only (that is the branch the model was
        # trained to drop, at ``null_cond_prob``). Both halves go through ONE forward at 2x
        # batch rather than two sequential passes -- on GPU that is far cheaper than 2x.
        cond_mask = (torch.zeros(b, dtype=torch.bool, device=device)
                     if history_null_mask is None else history_null_mask)
        null_mask = torch.ones(b, dtype=torch.bool, device=device)
        mask2 = torch.cat([cond_mask, null_mask], dim=0)
        hist2, extero2 = torch.cat([history] * 2, 0), torch.cat([extero] * 2, 0)
        cmd2 = None if cmd_tokens is None else torch.cat([cmd_tokens] * 2, 0)

    # Under ``prefix`` attention the conditioning cannot see the noised tokens and never
    # receives the flow-time embedding, so its transformer representation is IDENTICAL at
    # every step -- encode it once here instead of ``num_steps`` times inside the loop. The
    # conditioning tensors were already hoisted above; this hoists their encoding too.
    #
    # In the guided branch the prefix is built once at 2B (the conditional half plus the
    # null-history half), so CFG pays for one doubled prefix pass rather than eight.
    #
    # The ADVANTAGE label must reach this call under `prefix_token` injection, where the label
    # IS a conditioning token and so is part of what gets encoded. Omitting it silently
    # defaulted the prefix to `null` for every arm, making a positive-conditioned sample
    # bit-identical to a negative-conditioned one -- the served windows differed by exactly
    # 0.0000, which is how it was caught. Under `time_embed` the label rides the flow-time
    # embedding instead and the prefix is label-independent, so passing it here is a harmless
    # no-op on that path.
    cache = None
    if getattr(model, "attention_mode", "full") == "prefix":
        adv_pre = (None if advantage is None
                   else (torch.cat([advantage, advantage], 0) if guided else advantage))
        cache = (model.encode_prefix(hist2, extero2, cmd2, mask2, adv_pre) if guided
                 else model.encode_prefix(history, extero, cmd_tokens, history_null_mask,
                                          adv_pre))

    for i in range(num_steps):
        tau = torch.full((b,), i * dt, device=device, dtype=history.dtype)
        if guided:
            x2, tau2 = torch.cat([x, x], 0), torch.cat([tau, tau], 0)
            adv2 = None if advantage is None else torch.cat([advantage, advantage], 0)
            v2 = (model.decode_future(x2, tau2, cache, adv2) if cache is not None
                  else model(x2, tau2, hist2, extero2, cmd2, history_null_mask=mask2,
                             advantage=adv2))
            v_cond, v_null = v2[:b], v2[b:]
            # v_null + w (v_cond - v_null): extrapolate along the direction the HISTORY
            # contributes. Envs already null-masked have v_cond == v_null, so guidance is a
            # no-op for them -- which is what you want for the first N steps after a reset.
            v = v_null + guidance_scale * (v_cond - v_null)
        elif cache is not None:
            v = model.decode_future(x, tau, cache, advantage)
        else:
            v = model(x, tau, history, extero, cmd_tokens,
                      history_null_mask=history_null_mask, advantage=advantage)
        x = x + dt * v
    return x


def sample_trajectory_grad(
    model: torch.nn.Module,
    history: torch.Tensor,
    extero: torch.Tensor,
    cmd_tokens: torch.Tensor | None,
    future_len: int,
    state_dim: int,
    num_steps: int = 8,
    grad_steps: int = 1,
    history_null_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable Euler rollout with truncated backprop, for sample-space losses.

    Same integration as :func:`sample_trajectory`, but gradient-enabled: the first
    ``num_steps - grad_steps`` steps run under ``no_grad`` and only the final
    ``grad_steps`` steps build the autograd graph (truncated backprop through time).

    Args mirror :func:`sample_trajectory`; ``grad_steps`` is clamped to ``[1, num_steps]``.

    Returns:
        Generated future window (standardized), shape ``(B, F, D)``.
    """
    b = history.shape[0]
    device = history.device
    x = torch.randn(b, future_len, state_dim, device=device, dtype=history.dtype, generator=generator)
    step = 1.0 / num_steps
    keep = min(max(grad_steps, 1), num_steps)
    no_grad_steps = num_steps - keep
    # DELIBERATELY not prefix-cached, unlike sample_trajectory. The early steps here run under
    # no_grad and the tail steps do not; a cache built in the no_grad region and reused in the
    # tail would detach the conditioning embeddings from the autograd graph, so those steps
    # would stop backpropagating into history/extero/cmd -- silently, since the shapes and the
    # forward values would all be right. Nothing outside this module and one test calls this
    # function, so correctness beats the speedup.
    for i in range(num_steps):
        tau = torch.full((b,), i * step, device=device, dtype=history.dtype)
        if i < no_grad_steps:
            with torch.no_grad():
                v = model(x, tau, history, extero, cmd_tokens, history_null_mask=history_null_mask)
                x = x + step * v
        else:
            v = model(x, tau, history, extero, cmd_tokens, history_null_mask=history_null_mask)
            x = x + step * v
    return x
