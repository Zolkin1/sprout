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


def _masked_mean(per_sample: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Mean of ``per_sample (B,)`` over the samples where ``gate (B,)`` is True (0 if none)."""
    g = gate.to(per_sample.dtype)
    return (per_sample * g).sum() / g.sum().clamp_min(1.0)


def frame_consistency_terms(
    pred_phys: torch.Tensor,
    data_phys: torch.Tensor,
    dt: float,
    num_joints: int,
    num_extra_frames: int = 0,
    blocks: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample velocity + acceleration finite-difference consistency for the base block.

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
        ``(vel_per_sample, accel_per_sample)`` each ``(B,)`` (physical units).

    Raises:
        ValueError: If the selection lacks ``base_pose`` or ``base_vel`` -- the caller
            gates on that (see :func:`flow_matching_loss`), so reaching here means a
            weight outlived the block it relates.
    """
    slc = window_block_slices(num_joints, num_extra_frames, blocks)
    if "base_pose" not in slc or "base_vel" not in slc:
        raise ValueError(
            "frame_consistency_terms ties base_pose to base_vel, but this window packs "
            f"{sorted(slc)}. Zero vel_consistency_weight / accel_consistency_weight for "
            "this block selection."
        )
    bp, bv = slc["base_pose"].start, slc["base_vel"].start
    x = pred_phys
    pos, ori6d = x[..., bp:bp + 3], x[..., bp + 3:bp + 9]
    lin, ang = x[..., bv:bv + 3], x[..., bv + 3:bv + 6]
    data_lin, data_ang = data_phys[..., bv:bv + 3], data_phys[..., bv + 3:bv + 6]

    lin_fd = (pos[:, 1:] - pos[:, :-1]) / dt
    vel = ((lin_fd - lin[:, :-1]) ** 2).mean(dim=(1, 2))
    # Gram-Schmidt: the prediction is a raw 9-vector, so it is only approximately a
    # rotation, and so3_log needs a real one.
    rmat = rot6d_to_matrix(ori6d)                 # (B, F, 3, 3) = R_actT R_ref
    ang_fd = so3_log(rmat[:, 1:] @ rmat[:, :-1].transpose(-1, -2)) / dt
    vel = vel + ((ang_fd - ang[:, :-1]) ** 2).mean(dim=(1, 2))

    accel = (((lin[:, 1:] - lin[:, :-1]) / dt - (data_lin[:, 1:] - data_lin[:, :-1]) / dt) ** 2).mean(dim=(1, 2))
    accel = accel + (((ang[:, 1:] - ang[:, :-1]) / dt - (data_ang[:, 1:] - data_ang[:, :-1]) / dt) ** 2).mean(dim=(1, 2))
    return vel / 2, accel / 2


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
    accel_weight: float = 0.0,
    joint_vel_weight: float = 0.0,
    fk: object | None = None,
    fk_pose_weight: float = 0.0,
    body_fk: object | None = None,
    body_pose_root_weight: float = 0.0,
    body_pose_world_weight: float = 0.0,
    body_pose_ori_scale: float = 1.0,
    tau_min: float = 0.0,
    blocks: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the flow-matching loss (+ optional consistency / FK terms).

    The flow MSE is over ALL F future tokens (the conditioning is separate tokens, not
    part of the window). All auxiliary terms act on the predicted endpoint ``x1_hat``
    (denormalized). Velocity + joint-velocity consistency are ungated; the acceleration
    and FK terms are **τ-gated** (only ``tau >= tau_min``, where ``x1_hat ≈ x1``).

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
        vel_weight / accel_weight / joint_vel_weight: consistency weights.
        fk: An :class:`~motion_gen.fk.FKChain` (or ``None``).
        fk_pose_weight: FK position-consistency weight.
        tau_min: τ-gate threshold for the acceleration and FK terms (0 disables gating).
        blocks: The window's block selection; defaults to all five.

    Returns:
        Dict with ``loss`` and detached metrics: ``flow_mse`` and, when enabled,
        ``vel_consistency`` / ``accel_consistency`` / ``joint_vel_consistency`` /
        ``fk_pose``.
    """
    x_tau, tau, target_v = sample_flow_batch(x1)
    v = model(x_tau, tau, history, extero, cmd_tokens, history_null_mask=history_null_mask)
    flow_mse = ((v - target_v) ** 2).mean()
    out = {"loss": flow_mse, "flow_mse": flow_mse.detach()}

    packed = set(resolve_reference_blocks(blocks))
    has_base = {"base_pose", "base_vel"} <= packed
    has_joints = {"joint_pos", "joint_vel"} <= packed
    do_base = state_norm is not None and has_base and (vel_weight > 0.0 or accel_weight > 0.0)
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
        gate = tau >= tau_min  # (B,) bool; gates the accel + FK terms
        if do_base:
            vel_ps, accel_ps = frame_consistency_terms(
                pred_phys, data_phys, dt, num_joints, num_extra_frames, blocks)
            vel = vel_ps.mean()                                 # ungated
            accel = _masked_mean(accel_ps, gate)                # τ-gated
            out["vel_consistency"] = vel.detach()
            out["accel_consistency"] = accel.detach()
            out["loss"] = out["loss"] + vel_weight * vel + accel_weight * accel
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
    x = (torch.zeros(b, future_len, state_dim, device=device, dtype=history.dtype) if zero_noise
         else torch.randn(b, future_len, state_dim, device=device, dtype=history.dtype,
                          generator=generator))
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
    for i in range(num_steps):
        tau = torch.full((b,), i * dt, device=device, dtype=history.dtype)
        if guided:
            v2 = model(torch.cat([x, x], 0), torch.cat([tau, tau], 0), hist2, extero2, cmd2,
                       history_null_mask=mask2)
            v_cond, v_null = v2[:b], v2[b:]
            # v_null + w (v_cond - v_null): extrapolate along the direction the HISTORY
            # contributes. Envs already null-masked have v_cond == v_null, so guidance is a
            # no-op for them -- which is what you want for the first N steps after a reset.
            v = v_null + guidance_scale * (v_cond - v_null)
        else:
            v = model(x, tau, history, extero, cmd_tokens, history_null_mask=history_null_mask)
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
