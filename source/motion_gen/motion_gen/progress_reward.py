# Copyright 2026 Zachary Olkin. All rights reserved.

"""The PROGRESS reward's shape and its five constants, in one place.

Two consumers compute this reward and they must agree exactly, because one grades the data the
other trains on:

* ``scripts/rsl_rl/inspect_terrain_reward.py --stage reward --progress`` grades a RECORDING,
  offline, in numpy.  That is what the off-policy fine-tune (AWR and its ablation ladder) fits
  a critic to and weights its updates by.
* ``robot_rl...mdp.rewards.progress`` computes the same quantity LIVE, in torch, once per
  re-plan interval.  That is what an on-policy arm optimises directly.

The two exist so the algorithms can be compared on one MDP.  If the constants drifted apart the
comparison would silently become a comparison of rewards, with nothing in either run's output
to say so -- so they are stated here, imported by both, and pinned against each other by
``source/robot_rl/tests/test_progress_reward_parity.py``.

**Why the reward has this shape.**  The bare terrain grade is a per-re-plan geometric penalty
with no progress term, so standing still is very nearly its optimum: five AWR iterations
against it drove forward progress from 11.50 to 3.95 m per 1000 steps (0.20 m/s against a
commanded 1.35) while survival ROSE and falls fell.  Both added terms are therefore multiplied
by a gate that closes as the plan's terrain interaction worsens, so neither can be earned by
charging through the boxes: the reward asks for speed AND cleanliness, not either.

This module is IsaacLab-free and torch-optional by design -- ``motion_gen`` is imported by
generation scripts and tests that run without a simulator, and the offline grader is numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressRewardSpec:
    """Every constant of the progress reward, with the measurement behind it.

    Attributes:
        tau: Temperature of the penalty-to-goodness map, ``R = exp(-penalty / tau)``.
            ``1.0`` because :class:`~motion_gen.terrain_consistency.TerrainConsistencySpec`'s
            reference weights already put each penalty term near 1.0 at its own p95, so a unit
            temperature spends the reward's range on exactly the band the terms were scaled to.
        gate_sharpness: ``s`` in the PER-CHUNK gate on the velocity term.  ``s`` multiplies the
            penalty before gating (``exp(-s*p/tau) == R**s``), so 2.0 gates as though every
            penalty were twice as large.
        success_gate_sharpness: ``s`` in the EPISODE gate on the success term.  Separate, and
            higher, because that gate reads the episode's WORST chunk: one destructive re-plan
            should cost the success bonus much of its value, where one mediocre chunk among
            many should only dim its own velocity credit.
        vel_sigma_mps: Width of the velocity kernel, ``exp(-|v_err| / sigma)``.
        success_const: ``C``, the arrival bonus before gating, paid ONCE on the last chunk of
            an arriving episode.  Two other shapes were tried and measured: adding ``C`` to
            every chunk made the target near-binary (unsuccessful rows 0.11, successful 0.95,
            with terrain and velocity confined to the bottom sixth of the range), and
            distributing ``C/n`` diluted it to inertia -- mean 0.019 against terrain's 0.965,
            i.e. 2% of the reward, so the thing the reward exists to encode barely appeared in
            it.  A terminal bonus keeps it concentrated where it is earned and lets the value
            function propagate it backwards, which is what bootstrapping is for.
    """

    tau: float = 1.0
    gate_sharpness: float = 2.0
    success_gate_sharpness: float = 3.0
    vel_sigma_mps: float = 0.5
    success_const: float = 10.0

    @property
    def per_chunk_max(self) -> float:
        """Largest value one chunk's reward can take: ``1 + 1 + C``.

        The terrain grade and the gated velocity term are each bounded by 1, and the terminal
        chunk of a successful episode additionally carries ``gate_worst * C``.
        """
        return 2.0 + float(self.success_const)


#: The shipped constants.  Both consumers default to this instance rather than to five literals.
PROGRESS_REWARD_SPEC = ProgressRewardSpec()


def terrain_gate(reward_terrain, sharpness: float):
    """``sin((pi/2) * clip(R, 0, 1) ** s)`` -- the gate both consumers apply.

    Identically ``cos((pi/2) * (1 - R**s))``, the form it is usually stated in, but written as
    a sine of ``R`` because ``R`` is the quantity on hand and the identity saves a subtraction.

    Backend-agnostic on purpose: the offline grader hands it a numpy array and the live reward
    a torch tensor, and the whole point of this module is that they cannot diverge.  Both types
    provide ``.clip`` and ``**``; only the sine needs dispatching.

    Args:
        reward_terrain: ``R = exp(-penalty / tau)`` in ``(0, 1]``, numpy array or torch tensor.
        sharpness: ``s``.  Larger closes the gate faster as the plan's geometry worsens.

    Returns:
        The gate, same type and shape as ``reward_terrain``, in ``[0, 1]``.
    """
    powered = reward_terrain.clip(0.0, 1.0) ** float(sharpness)
    # `torch` is an optional import here: motion_gen must stay importable in a numpy-only
    # environment, and the offline grader is one.
    if type(powered).__module__.startswith("torch"):
        import torch                                                    # noqa: PLC0415
        return torch.sin(0.5 * math.pi * powered)
    import numpy as np                                                  # noqa: PLC0415
    return np.sin(0.5 * np.pi * powered)
