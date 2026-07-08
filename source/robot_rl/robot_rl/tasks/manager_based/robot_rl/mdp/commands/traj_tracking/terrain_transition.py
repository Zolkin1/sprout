# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain-entry "approach" decision core.

When a terrain-aware skill (e.g. ``stair_up``) is about to be committed, the
stance foot may still be far from where that skill wants it.  Committing
directly snaps the stance anchor onto the terrain and drags the foot — an
awkward over-extended step.  Instead, :func:`plan_slowdown_step` decides
whether to insert one shortened "slow-down" step in the *current* skill that
lands the swing foot at the terrain skill's stance reference, deferring the
terrain-skill commit by one stride.

It is a side-effect-free, tensor-in / tensor-out helper (unit-tested without
IsaacLab), consumed by the offline clip generator
(:class:`~.clips.offline_clip_generator.OfflineClipGenerator`) when it stitches
a flat approach onto a staircase.
"""

from __future__ import annotations

from typing import Tuple

import torch


def plan_slowdown_step(
    target_dist: torch.Tensor,
    step_lengths: torch.Tensor,
    eligible_mask: torch.Tensor,
    approach_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decide, per env, whether to take a slow-down step and which traj.

    For each env the slow-down step is taken iff (a) the stance foot is
    farther than ``approach_threshold`` from the terrain skill's stance
    reference (``target_dist`` measures that gap along the travel
    direction), and (b) there exists an eligible (current-skill)
    trajectory whose forward step length does not overshoot ``target_dist``
    — the largest such step is chosen so the swing foot lands as close to
    the stance reference as possible without stepping past it onto the
    terrain.  When no eligible step fits (target shorter than the shortest
    available step), the env declines the slow-down (returns ``-1``) so the
    caller commits the terrain skill normally.

    Args:
        target_dist: ``[K]`` forward distance from the current stance foot
            to the terrain skill's snapped stance reference.
        step_lengths: ``[T]`` per-trajectory forward swing reach
            (``inf`` for gate-less trajectories, which never fit).
        eligible_mask: ``[K, T]`` bool — trajectory ``t`` is in env ``k``'s
            current skill (the only trajectories a same-skill slow-down may
            select).
        approach_threshold: distance (m) below which the stance is treated
            as "already aligned" and no slow-down is inserted.

    Returns:
        ``(do_approach, forced_traj_idx)`` each ``[K]``.  ``do_approach``
        is bool; ``forced_traj_idx`` is the chosen global trajectory index
        where ``do_approach`` else ``-1``.
    """
    K = target_dist.shape[0]
    T = step_lengths.shape[0]
    fits = (step_lengths.unsqueeze(0) <= target_dist.unsqueeze(1)) & eligible_mask  # [K, T]
    neg_inf = torch.full((K, T), float("-inf"), device=step_lengths.device)
    score = torch.where(fits, step_lengths.unsqueeze(0).expand(K, T), neg_inf)
    best_val, best_idx = score.max(dim=1)                                          # [K]
    has_fit = best_val > float("-inf")
    cond2 = target_dist > approach_threshold
    do_approach = cond2 & has_fit
    forced = torch.where(
        do_approach, best_idx, torch.full_like(best_idx, -1)
    )
    return do_approach, forced
