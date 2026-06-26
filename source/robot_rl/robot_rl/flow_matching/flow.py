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
import torch.nn.functional as F

from .transforms import (
    frame_block_start,
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


def frame_consistency_loss(
    pred_phys: torch.Tensor,
    data_phys: torch.Tensor,
    known: torch.Tensor,
    dt: float,
    num_joints: int,
    num_extra_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Velocity + acceleration finite-difference consistency over the frame blocks.

    Operates on the root + each extra (end-effector) frame block (joints excluded),
    in DENORMALIZED (physical) relative-frame units. Builds a single consistency
    trajectory that uses the clean *data* at the known (conditioning) tokens and the
    *prediction* at the unknown tokens, so the history->prediction boundary jump is
    penalized just like the interior of the predicted window.

    - velocity loss: forward finite difference of the frame **pose** (positions, and a
      proper SO(3) difference of orientations) vs the predicted linear/angular
      velocities -- ``pos[t+1] - pos[t] ≈ vel[t]·dt`` and
      ``so3_log(R[t+1] R[t]^T) ≈ ang_vel[t]·dt``. Ties the predicted velocity to the
      predicted pose trajectory (a no-op if velocity outputs are later dropped and
      velocity is taken straight from the pose finite difference).
    - acceleration loss: forward finite difference of the predicted velocities vs the
      same finite difference of the **data** velocities.

    Args:
        pred_phys: Predicted clean endpoint ``x1_hat`` (denormalized), ``(B, F, D)``.
        data_phys: Target ``x1`` (denormalized), ``(B, F, D)``.
        known: Per-token conditioning mask, ``(B, F)`` bool.
        dt: Control step period (seconds).
        num_joints: ``nj``.
        num_extra_frames: ``K`` extra body frames (the root is always included).

    Returns:
        ``(vel_loss, accel_loss)`` scalars (physical units).
    """
    x = torch.where(known.unsqueeze(-1), data_phys, pred_phys)  # clean history + prediction
    vel_terms: list[torch.Tensor] = []
    accel_terms: list[torch.Tensor] = []
    for fi in range(1 + num_extra_frames):
        b = frame_block_start(num_joints, fi)
        pos, lin = x[..., b:b + 3], x[..., b + 3:b + 6]
        ori, ang = x[..., b + 6:b + 12], x[..., b + 12:b + 15]
        data_lin, data_ang = data_phys[..., b + 3:b + 6], data_phys[..., b + 12:b + 15]

        # velocity consistency: d(pose)/dt (forward difference) vs the predicted velocity.
        lin_fd = (pos[:, 1:] - pos[:, :-1]) / dt
        vel_terms.append(F.mse_loss(lin_fd, lin[:, :-1]))
        rmat = rot6d_to_matrix(ori)  # (B, F, 3, 3)
        ang_fd = so3_log(rmat[:, 1:] @ rmat[:, :-1].transpose(-1, -2)) / dt
        vel_terms.append(F.mse_loss(ang_fd, ang[:, :-1]))

        # acceleration consistency: d(predicted vel)/dt vs d(data vel)/dt.
        accel_terms.append(F.mse_loss((lin[:, 1:] - lin[:, :-1]) / dt,
                                      (data_lin[:, 1:] - data_lin[:, :-1]) / dt))
        accel_terms.append(F.mse_loss((ang[:, 1:] - ang[:, :-1]) / dt,
                                      (data_ang[:, 1:] - data_ang[:, :-1]) / dt))
    return torch.stack(vel_terms).mean(), torch.stack(accel_terms).mean()


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
) -> dict[str, torch.Tensor]:
    """Compute the masked flow-matching loss (+ optional consistency terms) for one batch.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        x1: Target window (standardized), shape ``(B, F, D)``.
        cond: Observed conditioning values (standardized), shape ``(B, F, D)``.
        known: Per-token conditioning mask, shape ``(B, F)`` bool.
        cmd_tokens: Strided command tokens (normalized), shape
            ``(B, num_cmd_tokens, cmd_dim)``, or ``None`` when not command-conditioned.
        ori_slice: ``(start, end)`` of the root 6D orientation block, for the optional aux loss.
        aux_weight: Weight of the orientation orthonormality penalty (0 disables).
        state_norm: The state :class:`Normalizer` (needed to denormalize for the
            physical-units consistency losses); ``None`` disables them.
        dt: Control step period (seconds) for the finite differences.
        num_joints: ``nj`` (for locating the frame blocks).
        num_extra_frames: ``K`` extra body frames.
        vel_weight: Weight of the frame velocity finite-difference consistency loss (0 disables).
        accel_weight: Weight of the frame acceleration finite-difference consistency loss (0 disables).

    Returns:
        Dict with ``loss`` (the optimized scalar), ``flow_mse`` (detached), and
        optionally ``ori_aux`` / ``vel_consistency`` / ``accel_consistency`` (detached).
        The flow MSE is over the unknown tokens only.
    """
    x_tau, tau, target_v = sample_flow_batch(x1, cond, known)
    v = model(x_tau, tau, cmd_tokens)
    unknown = (~known).unsqueeze(-1).to(v.dtype)  # (B, F, 1)
    sq = (v - target_v) ** 2 * unknown
    denom = unknown.sum().clamp_min(1.0) * v.shape[-1]
    flow_mse = sq.sum() / denom
    out = {"loss": flow_mse, "flow_mse": flow_mse.detach()}

    do_consistency = state_norm is not None and (vel_weight > 0.0 or accel_weight > 0.0)
    if (aux_weight > 0.0 and ori_slice is not None) or do_consistency:
        # Recover the predicted endpoint along the linear path: x1_hat = x_tau + (1 - tau) * v.
        t = tau.view(-1, *([1] * (x1.dim() - 1)))
        x1_hat = x_tau + (1.0 - t) * v
        if aux_weight > 0.0 and ori_slice is not None:
            ori = x1_hat[..., ori_slice[0]:ori_slice[1]][~known]  # (num_unknown, 6)
            aux = ori6d_orthonormality_penalty(ori)
            out["ori_aux"] = aux.detach()
            out["loss"] = out["loss"] + aux_weight * aux
        if do_consistency:
            vel_loss, accel_loss = frame_consistency_loss(
                state_norm.denormalize(x1_hat), state_norm.denormalize(x1),
                known, dt, num_joints, num_extra_frames,
            )
            out["vel_consistency"] = vel_loss.detach()
            out["accel_consistency"] = accel_loss.detach()
            out["loss"] = out["loss"] + vel_weight * vel_loss + accel_weight * accel_loss
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
