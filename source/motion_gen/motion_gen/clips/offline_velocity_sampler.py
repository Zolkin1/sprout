# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline, flat-ground re-implementation of the live velocity sampler.

The live :class:`MultiskillVelocityTrackingCommand` samples a skill from a
terrain cell's per-cell probabilities, draws a velocity uniformly within
that skill's bucket, and ramps a held target toward it at a max
acceleration every control step (see ``multiskill_velocity_commands.py``
and the base ``velocity_commands.py``).  For offline clip generation there
is no terrain and no robot, so this class reproduces the same *formulas*
and *distribution* with two flat-ground stand-ins:

- the per-cell ``terrain.skill_probs_at(xy)`` is replaced by a single
  fixed ``skill_probs`` vector, and
- the closed-loop / heading-PD / guidance overrides are dropped (pure
  open-loop sampling).

It reuses the real, side-effect-free helpers
(:func:`clamp_pd_channels_to_buckets`) so the per-skill clamp is identical
to live.  It is fully batched over a set of clips being generated at once.

This reproduces the live velocity *distribution*, not the live env's exact
RNG stream (which is unreachable headless): the sampling, ramp, and clamp
formulas match, and a fixed ``torch.Generator`` seed makes generation
reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..skill_bucket import clamp_pd_channels_to_buckets


class OfflineVelocitySampler:
    """Batched flat-ground velocity sampler matching the live sampler.

    Args:
        velocity_buckets: Mapping ``skill_name -> VelocityBucketCfg`` (the
            same dict that lives on the trajectory command cfg).
        skill_list: Ordered skill names; defines the integer skill ids.
            Must be a subset of ``velocity_buckets`` keys.
        skill_probs: Per-skill sampling probabilities in ``skill_list``
            order (normalised internally).  Flat-ground stand-in for the
            terrain cell distribution.
        resampling_time_range: ``(min, max)`` seconds between target
            resamples (mirrors ``cfg.resampling_time_range``).
        max_acc: Max acceleration for the ramp (m/s^2 and rad/s^2).  Use
            ``float('inf')`` to disable the ramp (instant target).
        command_dt: Control-step dt the ramp/timer advance by (seconds).
        skill_transition_prob: If set, on resample each clip keeps its
            current skill with prob ``1 - p`` or draws a different one
            (prob ``p``) -- matches ``cfg.skill_transition_prob``.  ``None``
            draws independently every resample.
        default_heading: Heading range used for buckets whose ``heading``
            is ``None``.
        device: Torch device.
        seed: Optional seed for the internal ``torch.Generator``.
    """

    def __init__(
        self,
        velocity_buckets: Mapping[str, "VelocityBucketCfg"],
        skill_list: Sequence[str],
        skill_probs: Sequence[float],
        *,
        resampling_time_range: tuple[float, float],
        max_acc: float,
        command_dt: float,
        skill_transition_prob: float | None = None,
        default_heading: tuple[float, float] = (0.0, 0.0),
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.skill_list = list(skill_list)
        self.num_skills = len(self.skill_list)
        self.device = torch.device(device)
        self.command_dt = float(command_dt)
        self.max_acc = float(max_acc)
        self.resampling_time_range = (float(resampling_time_range[0]), float(resampling_time_range[1]))
        self.skill_transition_prob = skill_transition_prob

        missing = [s for s in self.skill_list if s not in velocity_buckets]
        if missing:
            raise KeyError(f"skill_list entries missing from velocity_buckets: {missing}")
        if len(skill_probs) != self.num_skills:
            raise ValueError(
                f"skill_probs has length {len(skill_probs)} but skill_list has {self.num_skills}."
            )

        def _ranges(attr: str) -> torch.Tensor:
            return torch.tensor(
                [getattr(velocity_buckets[s], attr) for s in self.skill_list],
                dtype=torch.float, device=self.device,
            )

        self.lin_x = _ranges("lin_vel_x")  # [S, 2]
        self.lin_y = _ranges("lin_vel_y")
        self.ang_z = _ranges("ang_vel_z")
        self.heading = torch.stack([
            torch.tensor(
                velocity_buckets[s].heading if velocity_buckets[s].heading is not None else default_heading,
                dtype=torch.float, device=self.device,
            )
            for s in self.skill_list
        ])  # [S, 2]

        probs = torch.tensor(skill_probs, dtype=torch.float, device=self.device)
        if (probs < 0).any() or probs.sum() <= 0:
            raise ValueError(f"skill_probs must be non-negative with positive sum, got {skill_probs}.")
        self.skill_probs = probs / probs.sum()  # [S]

        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        # State (allocated in reset()).
        self.n = 0
        self.sampled_skill: torch.Tensor | None = None       # [n] long
        self.vel_target_sampled: torch.Tensor | None = None  # [n, 3]
        self.vel_target: torch.Tensor | None = None          # [n, 3] ramped
        self.vel_command: torch.Tensor | None = None         # [n, 3] ramped + per-skill clamp
        self.heading_target: torch.Tensor | None = None      # [n]
        self.time_left: torch.Tensor | None = None           # [n]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self._gen, device=self.device)

    def _uniform_range(self, rng: torch.Tensor) -> torch.Tensor:
        """Draw uniform samples in ``rng[..., 0:1]..rng[..., 1:2]``; shape ``rng[..., 0]``."""
        lo, hi = rng[..., 0], rng[..., 1]
        return lo + (hi - lo) * self._rand(lo.shape)

    def _draw_skill_independent(self, n: int) -> torch.Tensor:
        """Draw ``n`` skills i.i.d. from ``skill_probs`` (mirrors force_independent)."""
        probs = self.skill_probs.unsqueeze(0).expand(n, self.num_skills)
        return torch.multinomial(probs, num_samples=1, generator=self._gen).squeeze(-1)

    def _draw_skill_transition(self, prev_skill: torch.Tensor) -> torch.Tensor:
        """Resample skills honoring ``skill_transition_prob`` (mirrors `_sample_skill`)."""
        n = prev_skill.shape[0]
        if self.skill_transition_prob is None:
            return self._draw_skill_independent(n)

        new_skill = prev_skill.clone()
        is_transition = self._rand((n,)) < self.skill_transition_prob
        idx = is_transition.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return new_skill

        probs = self.skill_probs.unsqueeze(0).expand(idx.numel(), self.num_skills).clone()
        # Exclude the current skill, renormalise (degenerate rows keep current).
        probs[torch.arange(idx.numel(), device=self.device), prev_skill[idx]] = 0.0
        sums = probs.sum(dim=1)
        degenerate = sums <= 0.0
        if degenerate.any():
            deg = degenerate.nonzero(as_tuple=False).flatten()
            probs[deg] = 0.0
            probs[deg, prev_skill[idx][deg]] = 1.0
            sums = probs.sum(dim=1)
        probs = probs / sums.unsqueeze(1)
        new_skill[idx] = torch.multinomial(probs, num_samples=1, generator=self._gen).squeeze(-1)
        return new_skill

    def _sample_velocity(self, skill: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``(vx, vy, wz)`` [n,3] and heading [n] uniformly from each skill's bucket."""
        vx = self._uniform_range(self.lin_x[skill])
        vy = self._uniform_range(self.lin_y[skill])
        wz = self._uniform_range(self.ang_z[skill])
        heading = self._uniform_range(self.heading[skill])
        return torch.stack([vx, vy, wz], dim=-1), heading

    def _clamp_to_buckets(self) -> torch.Tensor:
        """Per-skill clamp of the ramped target via the real live helper."""
        return clamp_pd_channels_to_buckets(self.vel_target, self.sampled_skill, self.lin_y, self.ang_z)

    def _new_timer(self, n: int) -> torch.Tensor:
        lo, hi = self.resampling_time_range
        return lo + (hi - lo) * self._rand((n,))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, n: int) -> torch.Tensor:
        """Initialise ``n`` clips: draw skill + velocity, snap the ramp target.

        Mirrors ``reset_for_episode``: the held target ``vel_target`` is
        snapped straight to the freshly sampled value (no ramp across the
        episode boundary).

        Returns:
            ``[n, 3]`` initial velocity command (ramped + per-skill clamped).
        """
        self.n = int(n)
        self.sampled_skill = self._draw_skill_independent(self.n)
        self.vel_target_sampled, self.heading_target = self._sample_velocity(self.sampled_skill)
        self.vel_target = self.vel_target_sampled.clone()  # snap past the ramp at reset
        self.time_left = self._new_timer(self.n)
        self.vel_command = self._clamp_to_buckets()
        return self.vel_command

    def step(self) -> torch.Tensor:
        """Advance one control step: timer + resample + ramp + per-skill clamp.

        Returns:
            ``[n, 3]`` velocity command after this step.
        """
        if self.vel_target is None:
            raise RuntimeError("OfflineVelocitySampler.step() called before reset().")

        # Resample expired timers.
        self.time_left = self.time_left - self.command_dt
        expired = self.time_left <= 0.0
        if expired.any():
            ids = expired.nonzero(as_tuple=False).flatten()
            new_skill = self._draw_skill_transition(self.sampled_skill[ids])
            self.sampled_skill[ids] = new_skill
            vel, heading = self._sample_velocity(new_skill)
            self.vel_target_sampled[ids] = vel
            self.heading_target[ids] = heading
            self.time_left[ids] = self._new_timer(ids.numel())

        # Ramp the held target toward the sampled target at max_acc.
        limit = self.max_acc * self.command_dt
        self.vel_target = torch.clamp(
            self.vel_target_sampled,
            min=self.vel_target - limit,
            max=self.vel_target + limit,
        )
        self.vel_command = self._clamp_to_buckets()
        return self.vel_command
