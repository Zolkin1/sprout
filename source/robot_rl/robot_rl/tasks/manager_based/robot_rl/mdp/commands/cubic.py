import torch

def cubic_spline_coeff_batched(y0, dy0, y1, dy1, dtaudt):
    """
    Solve cubic spline coefficients for batched inputs.

    Args:
        y0, dy0, y1, dy1: [num_envs] tensors
        dtaudt: [num_envs] or scalar tensor = 1 / (t_end - t_start)

    Returns:
        coeffs: [num_envs, 4] tensor, each row [a0, a1, a2, a3]
    """
    T = 1.0 / dtaudt  # (t_end - t_start)
    a0 = y0
    a1 = dy0 * T
    a2 = 3 * (y1 - y0) - (2 * dy0 + dy1) * T
    a3 = -2 * (y1 - y0) + (dy0 + dy1) * T
    return torch.stack([a0, a1, a2, a3], dim=-1)  # [num_envs, 4]


def cubic_spline_eval(coeffs, tau, dtaudt):
    """
    Evaluate y(tau) and dy/dt for batched cubic spline.

    Args:
        coeffs: [num_envs, 4]
        tau: [num_envs] normalized time in [0, 1]
        dtaudt: [num_envs] or scalar = 1 / (t_end - t_start)

    Returns:
        y, dy: [num_envs]
    """
    a0, a1, a2, a3 = coeffs.unbind(-1)
    # Normalize dtaudt to [N] for 1-D tau, or [N,1] for 2-D tau
    if dtaudt.dim() == 2 and dtaudt.size(-1) == 1:
        dtaudt = dtaudt.squeeze(-1)  # [N]

    # Evaluate position and derivative (chain rule)
    y = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
    dy = (a1 + 2 * a2 * tau + 3 * a3 * tau**2) * dtaudt
    return y, dy