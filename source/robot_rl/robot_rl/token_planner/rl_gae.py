# Copyright 2026 Zachary Olkin. All rights reserved.

"""GAE for an env that AUTO-RESETS, where one ending truncates and the rest terminate.

Extracted from the training loop and given its own module because the arithmetic below is the
single highest-consequence thing in the RL stage and it is invisible when it is wrong: a
mis-specified truncation makes the return stop distinguishing arriving from failing, and a
double-counted one diverges while every conventional health metric keeps reading fine.

## The two conventions, and why only one is available here

An episode ends for two different reasons and they need opposite treatment.
**Termination** means the MDP is over -- there is no future, ``V(s_terminal) = 0``, and
bootstrapping would invent reward that cannot happen. **Truncation** means the episode was cut
short from outside the MDP; the world would have continued, so the value of the severed future
has to be added back or every value near the cut is biased low.

*Convention A* leaves the next-state bootstrap alive at a truncation::

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)

This is the textbook form and it is correct PROVIDED ``s_{t+1}`` is the real next state.

*Convention B* (rsl-rl's) kills the next-state bootstrap at BOTH endings and folds the
truncation's future into the reward, using ``V(s_t)`` as a stand-in::

    r'_t    = r_t + gamma * V(s_t) * truncated_t
    delta_t = r'_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

**Only B is available here.** IsaacLab's ``ManagerBasedRLEnv`` resets inside ``step``, so the
observation returned after an ending belongs to the NEXT episode -- a fresh spawn, possibly on
a different tile. ``V(s_{t+1})`` under Convention A would bootstrap arrival's payoff off an
unrelated restart.

**Applying both at once diverges.** With ``gamma * V(s_t)`` in the reward and
``gamma * V(s_{t+1})`` still alive, a truncation's target is ``~(2*gamma) * V`` rather than
``gamma * V``: the Bellman operator's modulus goes from 0.985 to 1.97 and there is no fixed
point. Measured on this stack, mean reward reached 3.9e8 over 250 iterations while the
critic's explained variance sat at 0.65 -- EV is scale-invariant and cannot see a diverging
scale, which is why ``train_rl_sim`` also gates on the absolute return.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation under Convention B.

    Args:
        rewards: ``[T, N]`` per-macro-step reward.
        values: ``[T + 1, N]`` critic values; ``values[T]`` bootstraps the rollout's tail.
        terminated: ``[T, N]`` bool or float -- the MDP ended (fall, deadline).
        truncated: ``[T, N]`` bool or float -- the episode was cut with a future (arrival).
        gamma: Discount per macro-step.
        lam: GAE lambda.

    Returns:
        ``(advantages [T, N], returns [T, N])``. ``returns`` is the TD(lambda) value target.

    Raises:
        ValueError: On a shape mismatch, or if any step is both terminated and truncated --
            which is not a representable state and would double-count the bootstrap.
    """
    t, n = rewards.shape
    if values.shape != (t + 1, n):
        raise ValueError(f"values must be [T+1, N] = {(t + 1, n)}, got {tuple(values.shape)}")
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("terminated/truncated must match rewards' shape")
    trm = terminated.to(rewards.dtype)
    trc = truncated.to(rewards.dtype)
    if bool(((trm > 0) & (trc > 0)).any()):
        raise ValueError(
            "a step is flagged BOTH terminated and truncated. The two are mutually exclusive "
            "endings and the bootstrap would be counted twice -- see this module's docstring "
            "for what that does to the value scale.")
    # Convention B: the truncation's severed future, added ONCE, here.
    r = rewards + gamma * values[:-1] * trc
    done = torch.clamp(trm + trc, max=1.0)
    adv = torch.zeros_like(rewards)
    last = torch.zeros(n, dtype=rewards.dtype, device=rewards.device)
    for i in reversed(range(t)):
        # Masked by the FULL done, not by `terminated`: the arrival bootstrap is already in
        # `r`, so leaving `values[i + 1]` alive here would be the second count.
        delta = r[i] + gamma * values[i + 1] * (1.0 - done[i]) - values[i]
        last = delta + gamma * lam * (1.0 - done[i]) * last
        adv[i] = last
    return adv, adv + values[:-1]
