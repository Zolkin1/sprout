# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rectified-flow / conditional-OT training objective and sampler.

Linear interpolant path: ``x_tau = (1 - tau) * x0 + tau * x1`` with ``x0 ~ N(0, I)``
and ``x1`` the (standardized) target trajectory. The target vector field is the
constant ``v* = x1 - x0``; the model regresses it with MSE. Sampling integrates
``dx/dtau = v_theta`` from ``tau = 0`` (noise) to ``tau = 1`` with Euler steps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .transforms import ori6d_orthonormality_penalty


def sample_flow_batch(
    x1: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a rectified-flow training triplet.

    Args:
        x1: Target trajectory (standardized), shape ``(B, F, D)``.
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


def flow_matching_loss(
    model: torch.nn.Module,
    history: torch.Tensor,
    x1: torch.Tensor,
    ori_slice: tuple[int, int] | None = None,
    aux_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute the flow-matching loss for one batch.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        history: History tokens (normalized), shape ``(B, H, history_in_dim)``.
        x1: Target trajectory (standardized), shape ``(B, F, D)``.
        ori_slice: ``(start, end)`` of the 6D orientation block, for the optional aux loss.
        aux_weight: Weight of the orientation orthonormality penalty (0 disables).

    Returns:
        Dict with ``loss`` (the optimized scalar), ``flow_mse`` (detached), and
        optionally ``ori_aux`` (detached).
    """
    memory = model.encode(history)
    x_tau, tau, target_v = sample_flow_batch(x1)
    v = model(x_tau, tau, memory)
    flow_mse = F.mse_loss(v, target_v)
    out = {"loss": flow_mse, "flow_mse": flow_mse.detach()}
    if aux_weight > 0.0 and ori_slice is not None:
        # Recover the predicted endpoint along the linear path: x1_hat = x_tau + (1 - tau) * v.
        t = tau.view(-1, *([1] * (x1.dim() - 1)))
        x1_hat = x_tau + (1.0 - t) * v
        aux = ori6d_orthonormality_penalty(x1_hat[..., ori_slice[0]:ori_slice[1]])
        out["ori_aux"] = aux.detach()
        out["loss"] = flow_mse + aux_weight * aux
    return out


@torch.no_grad()
def sample_trajectory(
    model: torch.nn.Module,
    history: torch.Tensor,
    future_len: int,
    state_dim: int,
    num_steps: int = 8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a future trajectory by Euler-integrating the learned flow.

    The encoder runs once; only the decoder is re-evaluated per step. The output
    is in the standardized space; the caller denormalizes and Gram-Schmidts the
    orientation block.

    Args:
        model: A :class:`FlowMatchingTransformer`.
        history: History tokens (normalized), shape ``(B, H, history_in_dim)``.
        future_len: Number of future steps ``F``.
        state_dim: Per-step state width ``D``.
        num_steps: Number of Euler integration steps.
        generator: Optional RNG for the initial noise.

    Returns:
        Generated trajectory (standardized), shape ``(B, F, D)``.
    """
    b = history.shape[0]
    device = history.device
    memory = model.encode(history)
    x = torch.randn(b, future_len, state_dim, device=device, generator=generator)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        tau = torch.full((b,), i * dt, device=device)
        v = model(x, tau, memory)
        x = x + dt * v
    return x
