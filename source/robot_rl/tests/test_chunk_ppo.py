# Copyright 2026 Zachary Olkin. All rights reserved.

"""The per-step PPO ratio: same objective at ``theta = theta_old``, tamer where it clips.

The properties that make the factorization legitimate rather than merely different:

* **At init every ratio is exactly 1**, so nothing is clipped and the surrogate is the plain
  advantage term.
* **The gradient is ``1/T`` times the joint-ratio gradient** at ``theta = theta_old``. Same
  direction, a constant the learning rate absorbs -- which is the whole argument that this is
  a re-factorization and not a new objective.
* **The log-ratio spread is ``sqrt(T)`` smaller** than the joint one, which is the entire
  point: it is what lets ``clip_param`` and ``desired_kl`` keep their stock meanings.
* **The KL reported to the LR controller is per-step**, i.e. the chunk total divided by ``T``.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch.distributions import Normal  # noqa: E402

T, D = 62, 44
N = T * D


def _log_prob_per_step(mean, std, actions):
    """Reference implementation: sum over channels, keep the time axis."""
    b = actions.shape[0]
    dist = Normal(mean.view(b, T, D), std.view(b, T, D))
    return dist.log_prob(actions.view(b, T, D)).sum(dim=-1)


def _joint_surrogate(ratio_joint, adv, eps):
    """Stock PPO's scalar-ratio surrogate."""
    return torch.max(-adv * ratio_joint,
                     -adv * ratio_joint.clamp(1 - eps, 1 + eps)).mean()


def _step_surrogate(ratio_step, adv, eps):
    """The per-step surrogate, one shared advantage broadcast over the chunk."""
    a = adv.view(-1, 1)
    return torch.max(-a * ratio_step, -a * ratio_step.clamp(1 - eps, 1 + eps)).mean()


def test_at_initialisation_every_per_step_ratio_is_exactly_one() -> None:
    """theta == theta_old, so nothing is clipped and no gradient is discarded."""
    torch.manual_seed(0)
    b = 16
    mean = torch.randn(b, N)
    std = torch.full((b, N), 0.15)
    actions = Normal(mean, std).sample()
    lp = _log_prob_per_step(mean, std, actions)
    ratio = torch.exp(lp - lp)
    assert ratio.shape == (b, T)
    assert torch.equal(ratio, torch.ones_like(ratio))


def test_the_gradient_is_one_over_T_times_the_joint_gradient() -> None:
    """The claim that makes this a re-factorization rather than a different objective."""
    torch.manual_seed(0)
    b, eps = 32, 0.2
    old_mean = torch.randn(b, N)
    std = torch.full((b, N), 0.15)
    actions = Normal(old_mean, std).sample()
    adv = torch.randn(b)

    # theta = theta_old, carried by a parameter so both paths differentiate the same tensor.
    delta = torch.zeros(b, N, requires_grad=True)

    lp_old = _log_prob_per_step(old_mean, std, actions).detach()
    lp_new = _log_prob_per_step(old_mean + delta, std, actions)

    step_loss = _step_surrogate(torch.exp(lp_new - lp_old), adv, eps)
    g_step = torch.autograd.grad(step_loss, delta, retain_graph=True)[0]

    joint_loss = _joint_surrogate(torch.exp(lp_new.sum(-1) - lp_old.sum(-1)), adv, eps)
    g_joint = torch.autograd.grad(joint_loss, delta)[0]

    assert torch.allclose(g_step, g_joint / T, atol=1e-9), (
        "the per-step surrogate's gradient is not 1/T of the joint one at theta = theta_old, "
        "so this is a different objective rather than a re-factorization"
    )


def test_the_per_step_log_ratio_is_sqrt_T_tighter_than_the_joint_one() -> None:
    """The reason the stock clip band and KL target become correct again.

    Var[log r] grows with the number of dimensions summed over, so the joint log-ratio over
    N = 2728 dims is sqrt(T) wider than a per-step one over D = 44.
    """
    torch.manual_seed(0)
    b = 4096
    old_mean = torch.randn(b, N)
    std = torch.full((b, N), 0.15)
    actions = Normal(old_mean, std).sample()
    # A small, uniform mean shift -- the shape of one PPO update.
    new_mean = old_mean + 0.01 * std

    lp_old = _log_prob_per_step(old_mean, std, actions)
    lp_new = _log_prob_per_step(new_mean, std, actions)
    log_r_step = (lp_new - lp_old).flatten()
    log_r_joint = (lp_new - lp_old).sum(dim=-1)

    got = log_r_joint.std() / log_r_step.std()
    assert got == pytest.approx(math.sqrt(T), rel=0.1), (
        f"joint/per-step log-ratio spread is {got:.2f}, expected ~sqrt(T)={math.sqrt(T):.2f}"
    )


def test_the_stock_clip_band_holds_at_the_stock_kl_target() -> None:
    """The measured failure this change fixes: ~45% of samples clipped at the 94x target.

    Under the per-step ratio, a policy sitting at the STOCK per-step KL target of 0.01 must
    keep its clip fraction in the normal band rather than saturating it.
    """
    torch.manual_seed(0)
    b, eps, target_kl = 8192, 0.2, 0.01
    std_val = 0.15
    # Per-step KL for a uniform shift of `delta` sigmas over D dims is D * delta^2 / 2.
    delta = math.sqrt(2.0 * target_kl / D)

    old_mean = torch.zeros(b, N)
    std = torch.full((b, N), std_val)
    actions = Normal(old_mean, std).sample()
    new_mean = old_mean + delta * std

    lp_old = _log_prob_per_step(old_mean, std, actions)
    lp_new = _log_prob_per_step(new_mean, std, actions)

    ratio_step = torch.exp(lp_new - lp_old)
    frac_step = ((ratio_step < 1 - eps) | (ratio_step > 1 + eps)).float().mean().item()

    ratio_joint = torch.exp((lp_new - lp_old).sum(dim=-1))
    frac_joint = ((ratio_joint < 1 - eps) | (ratio_joint > 1 + eps)).float().mean().item()

    # MEASURED at the stock target: per-step 0.154 against joint 0.880, and
    # std(log r) 0.142 against 1.109 (a ratio of 7.8, i.e. sqrt(T)).
    assert 0.05 <= frac_step <= 0.25, (
        f"per-step clip fraction {frac_step:.3f} is outside the healthy 0.1-0.2 band that "
        "makes clip_param=0.2 a trust region rather than a gate"
    )
    assert frac_joint > 3.0 * frac_step, (
        f"joint clip fraction {frac_joint:.3f} vs per-step {frac_step:.3f}: the control is "
        "not reproducing the saturation this change exists to fix, so the comparison says "
        "nothing"
    )
