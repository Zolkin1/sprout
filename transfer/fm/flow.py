# Copyright 2026 Zachary Olkin. All rights reserved.

"""Masked rectified-flow / conditional-OT training objective and sampler.

Linear interpolant path: ``x_tau = (1 - tau) * x0 + tau * x1`` with ``x0 ~ N(0, I)``
and ``x1`` the (standardized) target window. The target vector field is the constant
``v* = x1 - x0``; the model regresses it with MSE. Sampling integrates
``dx/dtau = v_theta`` from ``tau = 0`` (noise) to ``tau = 1`` with Euler steps.

History conditioning is *inpainting* (PARC-style): a per-token boolean mask ``known``
marks the recent-past tokens. Their (observed, possibly noised) values ``cond`` are
held fixed at every flow-time and re-clamped after every Euler step, and the loss is
computed only on the unknown tokens. When ``known`` is all-False the whole window is
generated unconditionally (history dropout).
"""

from __future__ import annotations

import torch

from .transforms import (
    frame_block_start,
    matrix_from_quat,
    ori6d_orthonormality_penalty,
    rot6d_to_matrix,
    so3_log,
)


def sample_flow_batch(
    x1: torch.Tensor,
    cond: torch.Tensor,
    known: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a masked rectified-flow training triplet.

    Args:
        x1: Target window (standardized), shape ``(B, F, D)``.
        cond: Observed values for the conditioning tokens (standardized), shape
            ``(B, F, D)``; only its ``known`` tokens are read.
        known: Per-token conditioning mask, shape ``(B, F)`` bool. Known tokens are
            held at ``cond``; unknown tokens follow the noise->data path.
        generator: Optional RNG for reproducibility.

    Returns:
        ``(x_tau, tau, target_v)`` with shapes ``(B, F, D)``, ``(B,)``, ``(B, F, D)``.
    """
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    tau = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=generator)
    t = tau.view(-1, *([1] * (x1.dim() - 1)))
    x_tau = (1.0 - t) * x0 + t * x1
    x_tau = torch.where(known.unsqueeze(-1), cond, x_tau)
    target_v = x1 - x0
    return x_tau, tau, target_v


def _masked_mean(per_sample: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Mean of ``per_sample (B,)`` over the samples where ``gate (B,)`` is True (0 if none)."""
    g = gate.to(per_sample.dtype)
    return (per_sample * g).sum() / g.sum().clamp_min(1.0)


def frame_consistency_terms(
    pred_phys: torch.Tensor,
    data_phys: torch.Tensor,
    known: torch.Tensor,
    dt: float,
    num_joints: int,
    num_extra_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample velocity + acceleration finite-difference consistency over frame blocks.

    Operates on the root + each extra (end-effector) frame block (joints handled
    separately), in DENORMALIZED (physical) relative-frame units. Uses a single
    consistency trajectory: clean *data* at the known (conditioning) tokens and the
    *prediction* at the unknown tokens, so the history->prediction boundary jump is
    penalized like the interior of the predicted window.

    - velocity: forward finite difference of the frame **pose** (positions, and a proper
      SO(3) difference of orientations) vs the predicted linear/angular velocities.
    - acceleration: forward finite difference of the predicted velocities vs the same
      finite difference of the **data** velocities.

    Returns:
        ``(vel_per_sample, accel_per_sample)`` each ``(B,)`` (physical units).
    """
    x = torch.where(known.unsqueeze(-1), data_phys, pred_phys)  # clean history + prediction
    b_ = x.shape[0]
    vel = x.new_zeros(b_)
    accel = x.new_zeros(b_)
    n = 0
    for fi in range(1 + num_extra_frames):
        b = frame_block_start(num_joints, fi)
        pos, lin = x[..., b:b + 3], x[..., b + 3:b + 6]
        ori, ang = x[..., b + 6:b + 12], x[..., b + 12:b + 15]
        data_lin, data_ang = data_phys[..., b + 3:b + 6], data_phys[..., b + 12:b + 15]

        lin_fd = (pos[:, 1:] - pos[:, :-1]) / dt
        vel = vel + ((lin_fd - lin[:, :-1]) ** 2).mean(dim=(1, 2))
        rmat = rot6d_to_matrix(ori)  # (B, F, 3, 3)
        ang_fd = so3_log(rmat[:, 1:] @ rmat[:, :-1].transpose(-1, -2)) / dt
        vel = vel + ((ang_fd - ang[:, :-1]) ** 2).mean(dim=(1, 2))

        accel = accel + (((lin[:, 1:] - lin[:, :-1]) / dt - (data_lin[:, 1:] - data_lin[:, :-1]) / dt) ** 2).mean(dim=(1, 2))
        accel = accel + (((ang[:, 1:] - ang[:, :-1]) / dt - (data_ang[:, 1:] - data_ang[:, :-1]) / dt) ** 2).mean(dim=(1, 2))
        n += 2
    return vel / n, accel / n


def joint_vel_consistency_term(
    pred_phys: torch.Tensor, data_phys: torch.Tensor, known: torch.Tensor, dt: float, num_joints: int
) -> torch.Tensor:
    """Per-sample joint-velocity consistency: ``(joint_pos[t+1]-joint_pos[t])/dt ≈ joint_vel[t]``.

    Returns ``(B,)`` in physical units (rad/s). Linear in the state, so it is left
    ungated like the frame velocity term.
    """
    x = torch.where(known.unsqueeze(-1), data_phys, pred_phys)
    jp = x[..., 0:num_joints]
    jv = x[..., num_joints:2 * num_joints]
    jp_fd = (jp[:, 1:] - jp[:, :-1]) / dt
    return ((jp_fd - jv[:, :-1]) ** 2).mean(dim=(1, 2))


def fk_consistency_terms(
    pred_phys: torch.Tensor,
    data_phys: torch.Tensor,
    known: torch.Tensor,
    dt: float,
    num_joints: int,
    num_extra_frames: int,
    fk: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample forward-kinematics pose + velocity consistency for the EE frames.

    Ties each predicted end-effector frame to the FK of the predicted joints:

    - pose: EE pose expressed relative to the predicted root (base frame) vs ``FK(q)`` --
      position + SO(3) orientation residual.
    - velocity: the joint-induced EE velocity (the FM EE twist with the predicted base
      motion removed, rotated into the base frame) vs ``J(q)·q̇``.

    All on the DENORMALIZED consistency trajectory (clean history + prediction). ``fk``
    is a :class:`~robot_rl.flow_matching.fk.FKChain` exposing ``pose_and_jacobian``.

    Returns:
        ``(fk_pose_per_sample, fk_vel_per_sample)`` each ``(B,)``.
    """
    x = torch.where(known.unsqueeze(-1), data_phys, pred_phys)
    b_, f_, _ = x.shape
    nj, k = num_joints, num_extra_frames

    q = x[..., 0:nj].reshape(b_ * f_, nj)
    qd = x[..., nj:2 * nj].reshape(b_ * f_, nj)
    fk_pos, fk_rot, jv, jw = fk.pose_and_jacobian(q)               # (M,K,3),(M,K,3,3),(M,K,3,nj)x2
    fk_pos = fk_pos.reshape(b_, f_, k, 3)
    fk_rot = fk_rot.reshape(b_, f_, k, 3, 3)
    v_fk = torch.einsum("mkij,mj->mki", jv, qd).reshape(b_, f_, k, 3)
    w_fk = torch.einsum("mkij,mj->mki", jw, qd).reshape(b_, f_, k, 3)

    rb = frame_block_start(nj, 0)
    p_root = x[..., rb:rb + 3]
    v_root = x[..., rb + 3:rb + 6]
    R_root = rot6d_to_matrix(x[..., rb + 6:rb + 12])               # (B,F,3,3) base->ref
    w_root = x[..., rb + 12:rb + 15]

    pose_ps = x.new_zeros(b_)
    vel_ps = x.new_zeros(b_)
    for kf in range(k):
        eb = frame_block_start(nj, 1 + kf)
        p_ee = x[..., eb:eb + 3]
        v_ee = x[..., eb + 3:eb + 6]
        R_ee = rot6d_to_matrix(x[..., eb + 6:eb + 12])
        w_ee = x[..., eb + 12:eb + 15]

        rel = p_ee - p_root
        # EE pose relative to the (predicted) base; R_rootᵀ via "...ji,...j->...i".
        p_ee_base = torch.einsum("bfji,bfj->bfi", R_root, rel)
        R_ee_base = R_root.transpose(-1, -2) @ R_ee
        pose_ps = pose_ps + ((p_ee_base - fk_pos[:, :, kf]) ** 2).mean(dim=(1, 2))
        rot_res = so3_log(R_ee_base.transpose(-1, -2) @ fk_rot[:, :, kf])
        pose_ps = pose_ps + (rot_res ** 2).mean(dim=(1, 2))

        # Joint-induced EE twist (remove base motion), expressed in the base frame.
        v_joints = v_ee - v_root - torch.linalg.cross(w_root, rel)
        w_joints = w_ee - w_root
        v_base = torch.einsum("bfji,bfj->bfi", R_root, v_joints)
        w_base = torch.einsum("bfji,bfj->bfi", R_root, w_joints)
        vel_ps = vel_ps + ((v_base - v_fk[:, :, kf]) ** 2).mean(dim=(1, 2))
        vel_ps = vel_ps + ((w_base - w_fk[:, :, kf]) ** 2).mean(dim=(1, 2))
    return pose_ps / max(k, 1), vel_ps / max(k, 1)


def terrain_penetration_term(
    pred_phys: torch.Tensor,
    data_phys: torch.Tensor,
    known: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    num_joints: int,
    foot_idxs: list[int],
    contact_pts: torch.Tensor | None,
    terrain: object,
) -> torch.Tensor:
    """Per-sample squared terrain penetration of the predicted feet (world frame).

    Lifts each foot frame from the anchor (ref) frame to the world via the per-window
    anchor pose ``(p_ref, q_ref)`` and asks the terrain how far it is below the surface.
    ``contact_pts=None`` -> the ankle-frame origin (ankle mode); otherwise the box-sole
    points ``(P, 3)`` in the foot frame are placed with the predicted foot orientation
    (contact mode). Uses the clean-history / predicted consistency trajectory.

    Args:
        pred_phys/data_phys: Predicted endpoint / target (denormalized), ``(B, F, D)``.
        known: Per-token conditioning mask, ``(B, F)`` bool.
        p_ref: World anchor position ``(B, 3)``; q_ref: anchor quat ``(B, 4 xyzw)``.
        num_joints: ``nj``.
        foot_idxs: Indices (into the extra frames) of the foot frames.
        contact_pts: Box-sole points in the foot frame ``(P, 3)``, or ``None`` (ankle mode).
        terrain: Object with ``penetration(world_xyz) -> depth`` (e.g. :class:`FlatTerrain`).

    Returns:
        Per-sample penetration ``(B,)`` (mean of depth² over feet, [points], frames).
    """
    x = torch.where(known.unsqueeze(-1), data_phys, pred_phys)
    r_ref = matrix_from_quat(q_ref)  # (B, 3, 3) base->world
    pen = x.new_zeros(x.shape[0])
    cpts = None if contact_pts is None else contact_pts.to(x)
    for fi in foot_idxs:
        eb = frame_block_start(num_joints, 1 + fi)
        p_foot_world = torch.einsum("bij,bfj->bfi", r_ref, x[..., eb:eb + 3]) + p_ref[:, None, :]
        if cpts is None:
            depth = terrain.penetration(p_foot_world)            # (B, F)
            pen = pen + (depth ** 2).mean(dim=1)
        else:
            r_foot = torch.einsum("bij,bfjk->bfik", r_ref, rot6d_to_matrix(x[..., eb + 6:eb + 12]))
            pc = p_foot_world[:, :, None, :] + torch.einsum("bfik,pk->bfpi", r_foot, cpts)  # (B,F,P,3)
            depth = terrain.penetration(pc)                       # (B, F, P)
            pen = pen + (depth ** 2).mean(dim=(1, 2))
    return pen / max(len(foot_idxs), 1)


def flow_matching_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    cond: torch.Tensor,
    known: torch.Tensor,
    cmd_tokens: torch.Tensor | None = None,
    ori_slice: tuple[int, int] | None = None,
    aux_weight: float = 0.0,
    state_norm: object | None = None,
    dt: float = 0.02,
    num_joints: int = 0,
    num_extra_frames: int = 0,
    vel_weight: float = 0.0,
    accel_weight: float = 0.0,
    joint_vel_weight: float = 0.0,
    fk: object | None = None,
    fk_pose_weight: float = 0.0,
    fk_vel_weight: float = 0.0,
    tau_min: float = 0.0,
    p_ref: torch.Tensor | None = None,
    q_ref: torch.Tensor | None = None,
    terrain: object | None = None,
    foot_idxs: list[int] | None = None,
    contact_pts: torch.Tensor | None = None,
    terrain_weight: float = 0.0,
    terrain_tau_gated: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the masked flow-matching loss (+ optional consistency / FK terms).

    All auxiliary terms act on the predicted endpoint ``x1_hat`` (denormalized). The
    velocity and joint-velocity consistency are computed over the whole batch (ungated,
    since they are linear and τ-robust). The acceleration and FK terms are **τ-gated**:
    only samples with ``tau >= tau_min`` contribute, because there ``x1_hat ≈ x1`` (at
    low τ ``x1_hat`` is the blurry conditional mean, where these nonlinear / data-matched
    terms are unreliable). The terrain term is **ungated by default**
    (``terrain_tau_gated=False``): unlike accel/FK, its low-τ endpoint penetrates ~as much
    as the deployed multi-step sample while the high-τ endpoint ≈ clean data and barely
    crosses the floor, so gating it would hide nearly all the penetration signal.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        x1: Target window (standardized), ``(B, F, D)``.
        cond: Observed conditioning values (standardized), ``(B, F, D)``.
        known: Per-token conditioning mask, ``(B, F)`` bool.
        cmd_tokens: Strided command tokens (normalized) or ``None``.
        ori_slice: ``(start, end)`` of the root 6D orientation block, for the aux loss.
        aux_weight: Weight of the orientation orthonormality penalty (0 disables).
        state_norm: State :class:`Normalizer` (needed to denormalize); ``None`` disables
            all consistency/FK terms.
        dt: Control step period (seconds) for the finite differences.
        num_joints: ``nj``; num_extra_frames: ``K``.
        vel_weight / accel_weight / joint_vel_weight: consistency weights.
        fk: An :class:`~robot_rl.flow_matching.fk.FKChain` (or ``None``).
        fk_pose_weight / fk_vel_weight: FK consistency weights.
        tau_min: τ-gate threshold for the acceleration and FK terms (0 disables gating).
        terrain / foot_idxs / contact_pts / terrain_weight: terrain-penetration pieces.
        terrain_tau_gated: if True, restrict the terrain term to ``tau >= tau_min`` like
            accel/FK; default False applies it across all flow-times.

    Returns:
        Dict with ``loss`` and detached metrics: ``flow_mse`` and, when enabled,
        ``ori_aux`` / ``vel_consistency`` / ``accel_consistency`` /
        ``joint_vel_consistency`` / ``fk_pose`` / ``fk_vel``.
    """
    x_tau, tau, target_v = sample_flow_batch(x1, cond, known)
    v = model(x_tau, tau, cmd_tokens)
    unknown = (~known).unsqueeze(-1).to(v.dtype)  # (B, F, 1)
    sq = (v - target_v) ** 2 * unknown
    denom = unknown.sum().clamp_min(1.0) * v.shape[-1]
    flow_mse = sq.sum() / denom
    out = {"loss": flow_mse, "flow_mse": flow_mse.detach()}

    do_frame = state_norm is not None and (vel_weight > 0.0 or accel_weight > 0.0 or joint_vel_weight > 0.0)
    do_fk = fk is not None and state_norm is not None and (fk_pose_weight > 0.0 or fk_vel_weight > 0.0)
    do_terrain = (terrain is not None and state_norm is not None and terrain_weight > 0.0
                  and foot_idxs and p_ref is not None and q_ref is not None)
    if (aux_weight > 0.0 and ori_slice is not None) or do_frame or do_fk or do_terrain:
        t = tau.view(-1, *([1] * (x1.dim() - 1)))
        x1_hat = x_tau + (1.0 - t) * v  # predicted endpoint along the linear path
        if aux_weight > 0.0 and ori_slice is not None:
            ori = x1_hat[..., ori_slice[0]:ori_slice[1]][~known]
            aux = ori6d_orthonormality_penalty(ori)
            out["ori_aux"] = aux.detach()
            out["loss"] = out["loss"] + aux_weight * aux
        if do_frame or do_fk or do_terrain:
            pred_phys = state_norm.denormalize(x1_hat)
            data_phys = state_norm.denormalize(x1)
            gate = tau >= tau_min  # (B,) bool; gates the accel + FK + terrain terms
        if do_frame:
            vel_ps, accel_ps = frame_consistency_terms(pred_phys, data_phys, known, dt, num_joints, num_extra_frames)
            jvel_ps = joint_vel_consistency_term(pred_phys, data_phys, known, dt, num_joints)
            vel, jvel = vel_ps.mean(), jvel_ps.mean()           # ungated
            accel = _masked_mean(accel_ps, gate)                # τ-gated
            out["vel_consistency"] = vel.detach()
            out["accel_consistency"] = accel.detach()
            out["joint_vel_consistency"] = jvel.detach()
            out["loss"] = out["loss"] + vel_weight * vel + accel_weight * accel + joint_vel_weight * jvel
        if do_fk:
            # Run FK only on the τ-gated samples (it is the expensive term).
            idx = gate.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                fkp_ps, fkv_ps = fk_consistency_terms(
                    pred_phys[idx], data_phys[idx], known[idx], dt, num_joints, num_extra_frames, fk
                )
                fkp, fkv = fkp_ps.mean(), fkv_ps.mean()
            else:
                fkp = fkv = flow_mse.new_zeros(())
            out["fk_pose"] = fkp.detach()
            out["fk_vel"] = fkv.detach()
            out["loss"] = out["loss"] + fk_pose_weight * fkp + fk_vel_weight * fkv
        if do_terrain:
            pen_ps = terrain_penetration_term(
                pred_phys, data_phys, known, p_ref, q_ref, num_joints, foot_idxs, contact_pts, terrain
            )
            # Ungated by default: the low-τ endpoint penetrates like the deployed sample,
            # whereas the high-τ endpoint ≈ clean data, so gating would hide the signal.
            pen = _masked_mean(pen_ps, gate) if terrain_tau_gated else pen_ps.mean()
            out["terrain_penetration"] = pen.detach()
            out["loss"] = out["loss"] + terrain_weight * pen
    return out


@torch.no_grad()
def sample_trajectory(
    model: torch.nn.Module,
    cond: torch.Tensor,
    known: torch.Tensor,
    future_len: int,
    state_dim: int,
    num_steps: int = 8,
    cmd_tokens: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a state window by Euler-integrating the learned flow with inpainting.

    Known tokens are clamped to ``cond`` at init and after every step. The output is
    in the standardized space; the caller denormalizes and Gram-Schmidts the
    orientation block.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        cond: Observed conditioning values (standardized), shape ``(B, F, D)``;
            only its ``known`` tokens are used.
        known: Per-token conditioning mask, shape ``(B, F)`` bool.
        future_len: Window length ``F``.
        state_dim: Per-step state width ``D``.
        num_steps: Number of Euler integration steps.
        cmd_tokens: Strided command tokens (normalized), shape
            ``(B, num_cmd_tokens, cmd_dim)``, held fixed across integration.
        generator: Optional RNG for the initial noise.

    Returns:
        Generated window (standardized), shape ``(B, F, D)``. Tokens ``0..N-1`` are
        the (re-clamped) conditioning; novel predictions are ``N..F-1``.
    """
    b = cond.shape[0]
    device = cond.device
    km = known.unsqueeze(-1)
    x = torch.randn(b, future_len, state_dim, device=device, dtype=cond.dtype, generator=generator)
    x = torch.where(km, cond, x)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        tau = torch.full((b,), i * dt, device=device)
        v = model(x, tau, cmd_tokens)
        x = x + dt * v
        x = torch.where(km, cond, x)
    return x
