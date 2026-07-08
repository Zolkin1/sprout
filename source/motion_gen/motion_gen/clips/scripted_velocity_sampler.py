# Copyright 2026 Zachary Olkin. All rights reserved.

"""Deterministic, hand-authored velocity profile for offline clip generation.

Drop-in replacement for :class:`OfflineVelocitySampler` that follows a fixed
``(time, velocity)`` keyframe schedule (linearly interpolated) instead of
random sampling.  It is used to bake a single *parity* clip -- e.g.
``stand -> ramp to 1 m/s walk -> hold -> ramp to 3 m/s run -> hold -> ramp
down to stand`` -- so the exact same fixed reference can be played in IsaacLab
and the MuJoCo sim2sim transfer for behaviour comparison.

It subclasses :class:`OfflineVelocitySampler` only to reuse its per-skill
bucket-table setup (``lin_x`` / ``lin_y`` / ``ang_z`` / ``skill_list``), which
:class:`OfflineClipGenerator` reads and validates.  The generator selects the
skill (stand / walk / run) per step from the commanded velocity via
``bucket_for_velocity``, so this sampler only has to emit the right
``(vx, vy, wz)`` each control step.  The profile is forward-only (``vy = wz =
0``); the per-skill clamp is therefore a no-op and is skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .offline_velocity_sampler import OfflineVelocitySampler


class ScriptedVelocitySampler(OfflineVelocitySampler):
    """Emit a fixed, time-indexed velocity profile (linearly interpolated).

    Args:
        velocity_buckets: Mapping ``skill_name -> VelocityBucketCfg`` (same dict
            the trajectory command cfg holds); only used to build the bucket
            tables the generator reads.
        skill_list: Ordered skill names (must match the manager's skill order).
        skill_probs: Per-skill probabilities -- unused for sampling here, kept
            for interface compatibility with the base class.
        keyframes: Ordered ``[(t_seconds, (vx, vy, wz)), ...]`` waypoints. The
            commanded velocity is linearly interpolated between consecutive
            keyframes and held flat outside the first/last keyframe times.
        command_dt: Control-step dt the internal clock advances by (seconds).
        max_acc: Forwarded to the base class (unused: the ramp is baked into the
            keyframe schedule, not applied on top of it).
        device: Torch device.
        seed: Forwarded to the base class (unused: the profile is deterministic).
    """

    def __init__(
        self,
        velocity_buckets: Mapping[str, "VelocityBucketCfg"],
        skill_list: Sequence[str],
        skill_probs: Sequence[float],
        *,
        keyframes: Sequence[tuple[float, Sequence[float]]],
        command_dt: float,
        max_acc: float = 2.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            velocity_buckets, skill_list, skill_probs,
            resampling_time_range=(1.0, 1.0), max_acc=max_acc,
            command_dt=command_dt, skill_transition_prob=None, device=device, seed=seed,
        )
        if len(keyframes) < 2:
            raise ValueError("ScriptedVelocitySampler needs at least two keyframes.")
        ts = [float(t) for t, _ in keyframes]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError(f"keyframe times must be strictly increasing, got {ts}.")
        self._kf_t = ts
        self._kf_v = [[float(c) for c in v] for _, v in keyframes]
        self.clip_length_s = ts[-1]
        self._step_count = 0

    # ------------------------------------------------------------------
    # Profile evaluation
    # ------------------------------------------------------------------

    def _target_vec(self, t: float) -> torch.Tensor:
        """Linearly-interpolated ``[n, 3]`` command at time ``t`` (held at the ends)."""
        ts, vs = self._kf_t, self._kf_v
        if t <= ts[0]:
            vel = vs[0]
        elif t >= ts[-1]:
            vel = vs[-1]
        else:
            i = max(j for j, tj in enumerate(ts) if tj <= t)  # ts[i] <= t < ts[i+1]
            alpha = (t - ts[i]) / (ts[i + 1] - ts[i])
            vel = [a + (b - a) * alpha for a, b in zip(vs[i], vs[i + 1])]
        out = torch.zeros(self.n, 3, device=self.device)
        out[:] = torch.tensor(vel, dtype=torch.float, device=self.device)
        return out

    def _clamp_to_buckets(self) -> torch.Tensor:
        """No-op clamp: the scripted profile is forward-only and within-bucket."""
        return self.vel_target

    # ------------------------------------------------------------------
    # Public API (mirrors OfflineVelocitySampler)
    # ------------------------------------------------------------------

    def reset(self, n: int) -> torch.Tensor:
        """Initialise ``n`` (identical) clips at the profile's ``t = 0`` command."""
        self.n = int(n)
        self._step_count = 0
        self.vel_target_sampled = self._target_vec(0.0)
        self.vel_target = self.vel_target_sampled.clone()
        self.vel_command = self.vel_target.clone()
        return self.vel_command

    def step(self) -> torch.Tensor:
        """Advance one control step and return the scripted command ``[n, 3]``."""
        if self.vel_target is None:
            raise RuntimeError("ScriptedVelocitySampler.step() called before reset().")
        self._step_count += 1
        self.vel_target = self._target_vec(self._step_count * self.command_dt)
        self.vel_command = self.vel_target.clone()
        return self.vel_command
