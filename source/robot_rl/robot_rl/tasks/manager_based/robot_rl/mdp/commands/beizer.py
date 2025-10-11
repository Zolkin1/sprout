import torch
import math
from typing import Union

def _ncr(n: int, r: int) -> int:
    """Binomial coefficient n choose r"""
    return math.comb(n, r)


def bezier_deg(
    order: int,
    tau: Union[float, torch.Tensor],
    step_dur: Union[float, torch.Tensor],
    control_points: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """
    Computes the Bezier curve (order=0) or its time-derivative (order=1).
    
    Args:
        order: 0 for position, 1 for derivative
        tau: Tensor of shape [batch], clipped to [0,1]
        step_dur: Tensor of shape [batch]
        control_points: Tensor of shape [batch, degree+1]
        degree: polynomial degree
    
    Returns:
        Tensor of shape [batch]
    """
    if isinstance(tau, float):
        tau = torch.tensor([tau], device=control_points.device, dtype=control_points.dtype)
    if isinstance(step_dur, float):
        step_dur = torch.tensor([step_dur], device=control_points.device, dtype=control_points.dtype)
    if control_points.dim() == 1:
        control_points = control_points.unsqueeze(0)  # [1, degree+1]
 
    
    tau = torch.clamp(tau, 0.0, 1.0)
    batch = tau.size(0)

    if order == 1:
        # Derivative: d/dt B(t) = n * sum[(P_{i+1} - P_i) * B_{i,n-1}(tau)] / step_dur
        cp_diff = control_points[:, 1:] - control_points[:, :-1]  # [batch, degree]
        coefs = torch.tensor(
            [_ncr(degree - 1, i) for i in range(degree)],
            dtype=control_points.dtype,
            device=control_points.device
        )  # [degree]
        i = torch.arange(degree, device=control_points.device)
        tau_pow = tau.unsqueeze(1) ** i  # [batch, degree]
        one_minus_pow = (1 - tau.unsqueeze(1)) ** (degree - 1 - i)  # [batch, degree]
        terms = degree * cp_diff * coefs * tau_pow * one_minus_pow  # [batch, degree]
        return terms.sum(dim=1) / step_dur  # [batch]
    else:
        # Position: B(tau) = sum[P_i * B_{i,n}(tau)]
        coefs = torch.tensor(
            [_ncr(degree, i) for i in range(degree + 1)],
            dtype=control_points.dtype,
            device=control_points.device
        )  # [degree+1]
        i = torch.arange(degree + 1, device=control_points.device)
        tau_pow = tau.unsqueeze(1) ** i  # [batch, degree+1]
        one_minus_pow = (1 - tau.unsqueeze(1)) ** (degree - i)  # [batch, degree+1]
        terms = control_points * coefs * tau_pow * one_minus_pow  # [batch, degree+1]
        return terms.sum(dim=1)  # [batch]

