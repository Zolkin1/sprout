# Copyright 2026 Zachary Olkin. All rights reserved.

"""The action -> residual map, with ONE producer.

:meth:`AdjusterEnv._residual_from_action` applies this during training.  Anything that drives a
trained adjuster from outside that env -- the eval harness, a viewer -- must apply the IDENTICAL
map, because the lift, the squash order and the per-channel caps together are what the policy's
output means.  A second transcription of it is a silent way for a scored rollout to differ from
the rollouts that trained the policy.

Kept in its own module rather than on the env class so an off-env consumer can import it
without importing :class:`ManagerBasedRLEnv`, which reaches ``pxr`` and cannot be bound before
AppLauncher runs.

**Driving a trained adjuster from the eval harness is NOT implemented here, deliberately.**  It
needs two things ``WindowAdjusterModel`` does not have yet -- a way to rebuild the actor from a
checkpoint (its constructor takes the live observation groups and the borrowed generator) and a
deterministic forward -- and there is no trained adjuster to write them against.  Guessing at
that API now would produce a loader that looks finished, is never exercised, and rots.  When the
first arm trains, the hook is one call in ``record_rollouts.py`` between the re-plan claim and
the chunk grade, masked by ``cmd.fm_reinfer_mask``, using this function.
"""

from __future__ import annotations

import torch


def residual_from_action(action: torch.Tensor, knot_lift: torch.Tensor,
                         residual_scale: torch.Tensor) -> torch.Tensor:
    """``[B, K*D]`` dimensionless action -> ``[B, F, D]`` residual in raw units.

    **Lift, then squash, then scale, and the order is load-bearing.** Catmull-Rom overshoots by
    up to 25% between knots (worst row L1 1.2499), so squashing the knots first would let the
    interpolated window exceed the per-channel cap it is supposed to respect. ``tanh`` rather
    than a clamp so the gradient never vanishes and ``init_std`` stays meaningful.

    At one knot per node the lift is exactly the identity, so this is also the full-width path.

    Args:
        action: ``[B, K*D]``, the policy's output.
        knot_lift: ``[F, K]`` the fixed basis, from :func:`~robot_rl.flow_matching.knot_basis`.
        residual_scale: ``[D]`` per-channel authority in raw units.

    Returns:
        ``[B, F, D]``.

    Raises:
        ValueError: If the action width is not ``K * D``.
    """
    k, d = int(knot_lift.shape[1]), int(residual_scale.shape[-1])
    if action.shape[-1] != k * d:
        raise ValueError(
            f"expected an action of width K * D = {k} * {d} = {k * d}, got "
            f"{int(action.shape[-1])}. The actor must have been built with this K.")
    shaped = action.view(-1, k, d)
    lifted = torch.einsum("fk,bkd->bfd", knot_lift, shaped)
    return torch.tanh(lifted) * residual_scale


