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

from .transforms import ori6d_orthonormality_penalty


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


def flow_matching_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    cond: torch.Tensor,
    known: torch.Tensor,
    cmd_tokens: torch.Tensor | None = None,
    ori_slice: tuple[int, int] | None = None,
    aux_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute the masked flow-matching loss for one batch.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        x1: Target window (standardized), shape ``(B, F, D)``.
        cond: Observed conditioning values (standardized), shape ``(B, F, D)``.
        known: Per-token conditioning mask, shape ``(B, F)`` bool.
        cmd_tokens: Strided command tokens (normalized), shape
            ``(B, num_cmd_tokens, cmd_dim)``, or ``None`` when not command-conditioned.
        ori_slice: ``(start, end)`` of the root 6D orientation block, for the optional aux loss.
        aux_weight: Weight of the orientation orthonormality penalty (0 disables).

    Returns:
        Dict with ``loss`` (the optimized scalar), ``flow_mse`` (detached), and
        optionally ``ori_aux`` (detached). The MSE is over the unknown tokens only.
    """
    x_tau, tau, target_v = sample_flow_batch(x1, cond, known)
    v = model(x_tau, tau, cmd_tokens)
    unknown = (~known).unsqueeze(-1).to(v.dtype)  # (B, F, 1)
    sq = (v - target_v) ** 2 * unknown
    denom = unknown.sum().clamp_min(1.0) * v.shape[-1]
    flow_mse = sq.sum() / denom
    out = {"loss": flow_mse, "flow_mse": flow_mse.detach()}
    if aux_weight > 0.0 and ori_slice is not None:
        # Recover the predicted endpoint along the linear path: x1_hat = x_tau + (1 - tau) * v.
        t = tau.view(-1, *([1] * (x1.dim() - 1)))
        x1_hat = x_tau + (1.0 - t) * v
        ori = x1_hat[..., ori_slice[0]:ori_slice[1]][~known]  # (num_unknown, 6)
        aux = ori6d_orthonormality_penalty(ori)
        out["ori_aux"] = aux.detach()
        out["loss"] = flow_mse + aux_weight * aux
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
