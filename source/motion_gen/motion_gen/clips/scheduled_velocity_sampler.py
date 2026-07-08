# Copyright 2026 Zachary Olkin. All rights reserved.

"""Target-distribution velocity sampler for offline clip generation.

Drop-in replacement for :class:`OfflineVelocitySampler` that builds each clip's
velocity profile to hit an *explicitly specified* distribution of time, rather
than relying on the stochastic resample-timer + ramp knobs.  You give it:

- ``skill_time_fractions``: how much of the *steady* (held-velocity) time each
  skill should occupy (e.g. ``{standing: 0.2, walk_forward: 0.5, running: 0.3}``);
- ``transition_fraction``: the fraction of total time the command should be
  *ramping* between settled velocities (the rest is steady); and
- ``standing_transition_fraction``: how often a gap between two moving holds is
  turned into a stop (a standing hold inserted between them, i.e. a stop/go
  cycle) -- the recovery-relevant start/stop motions.

Per clip it plans an alternating ``[hold] -> [ramp] -> [hold] -> ...`` schedule
that meets those budgets, samples a representative velocity within each hold's
skill bucket, and linearly interpolates the ramps.  The targets are hit in
aggregate over a library (individual clips vary in skill order, durations, and
sampled velocities); verify the achieved distribution with
``scripts/rsl_rl/command_distribution_stats.py`` and tune.

It subclasses :class:`OfflineVelocitySampler` purely to reuse the per-skill
bucket-table setup (``lin_x`` / ``lin_y`` / ``ang_z`` / ``skill_list``) that
:class:`OfflineClipGenerator` reads and validates.  As with
:class:`ScriptedVelocitySampler`, the generator derives the active skill per
step from the commanded velocity (``bucket_for_velocity``), so this sampler only
has to emit the right ``(vx, vy, wz)`` each control step; the per-skill clamp is
therefore a no-op.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .offline_velocity_sampler import OfflineVelocitySampler

# Name used to identify the standing skill in a skill list (case-insensitive),
# matching command_distribution_stats.STANDING_NAME.
STANDING_NAME = "standing"


class ScheduledVelocitySampler(OfflineVelocitySampler):
    """Emit a per-clip velocity schedule planned to hit target time fractions.

    Args:
        velocity_buckets: Mapping ``skill_name -> VelocityBucketCfg`` (same dict
            the trajectory command cfg holds); used to build the bucket tables
            the generator reads and to sample a representative velocity per hold.
        skill_list: Ordered skill names (must match the manager's skill order).
        skill_probs: Per-skill probabilities -- unused for planning here (kept
            for base-class interface compatibility); the planner uses
            ``skill_time_fractions`` instead.
        skill_time_fractions: Target fraction of *steady* time per skill (keys
            are skill names; missing skills default to 0). Normalised internally.
        transition_fraction: Target fraction of total time spent ramping between
            settled velocities (``0..1``). The remainder is steady (held) time.
        standing_transition_fraction: Controls how the standing-time budget
            (``skill_time_fractions[standing]``) is chunked into stops -- the number
            of standing holds inserted between moving holds is
            ``max(1, round(standing_transition_fraction * num_gaps))`` whenever the
            standing budget is positive. Higher -> more, shorter stops (more stop/go
            cycles); lower -> fewer, longer stands. The standing *occupancy* is set
            by ``skill_time_fractions[standing]``, not this knob.
        hold_duration_range: ``(min, max)`` seconds; sets the average held-segment
            length (drives how many holds, hence transitions, a clip contains).
        command_dt: Control-step dt the schedule is sampled at (seconds).
        clip_length_s: Clip length (seconds); the schedule covers
            ``round(clip_length_s / command_dt)`` steps and MUST match the value
            passed to :class:`OfflineClipGenerator`.
        max_acc: Forwarded to the base class (unused: ramps are baked into the
            schedule, not applied on top of it).
        standing_name: Skill name treated as "standing" (case-insensitive).
        device: Torch device.
        seed: Seeds the internal numpy planner RNG (and the base torch generator).
    """

    def __init__(
        self,
        velocity_buckets: Mapping[str, "VelocityBucketCfg"],
        skill_list: Sequence[str],
        skill_probs: Sequence[float],
        *,
        skill_time_fractions: Mapping[str, float],
        transition_fraction: float,
        standing_transition_fraction: float = 0.0,
        hold_duration_range: tuple[float, float] = (2.0, 5.0),
        command_dt: float,
        clip_length_s: float,
        max_acc: float = 2.0,
        standing_name: str = STANDING_NAME,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            velocity_buckets, skill_list, skill_probs,
            resampling_time_range=(1.0, 1.0), max_acc=max_acc,
            command_dt=command_dt, skill_transition_prob=None, device=device, seed=seed,
        )
        if not 0.0 <= transition_fraction < 1.0:
            raise ValueError(f"transition_fraction must be in [0, 1), got {transition_fraction}.")
        if not 0.0 <= standing_transition_fraction <= 1.0:
            raise ValueError(
                f"standing_transition_fraction must be in [0, 1], got {standing_transition_fraction}."
            )
        if hold_duration_range[0] <= 0.0 or hold_duration_range[1] < hold_duration_range[0]:
            raise ValueError(f"hold_duration_range must be positive and ordered, got {hold_duration_range}.")

        self.clip_length_s = float(clip_length_s)
        self.num_steps = int(round(self.clip_length_s / self.command_dt))
        if self.num_steps < 1:
            raise ValueError(f"clip_length_s/command_dt yields {self.num_steps} steps; need >= 1.")
        self.transition_fraction = float(transition_fraction)
        self.standing_transition_fraction = float(standing_transition_fraction)
        self.hold_duration_range = (float(hold_duration_range[0]), float(hold_duration_range[1]))

        # Resolve the target steady-time fractions onto skill_list order.
        unknown = [k for k in skill_time_fractions if k not in self.skill_list]
        if unknown:
            raise KeyError(f"skill_time_fractions keys not in skill_list {self.skill_list}: {unknown}")
        frac = np.array([float(skill_time_fractions.get(s, 0.0)) for s in self.skill_list], dtype=np.float64)
        if (frac < 0).any() or frac.sum() <= 0:
            raise ValueError(f"skill_time_fractions must be non-negative with positive sum, got {dict(skill_time_fractions)}.")
        self._skill_frac = frac / frac.sum()  # [S]

        # Standing index + moving-skill split (skills with positive weight).
        self._standing_idx = next(
            (i for i, s in enumerate(self.skill_list) if s.lower() == standing_name.lower()), None
        )
        self._moving_idx = np.array(
            [i for i in range(self.num_skills) if i != self._standing_idx and self._skill_frac[i] > 0.0],
            dtype=np.int64,
        )
        if self._moving_idx.size == 0 and self._standing_idx is None:
            raise ValueError("skill_time_fractions selects no usable skills.")
        # Bucket tables as numpy for the planner (the base class holds them on device).
        self._bx = self.lin_x.cpu().numpy()  # [S, 2]
        self._by = self.lin_y.cpu().numpy()
        self._bw = self.ang_z.cpu().numpy()

        self._np_rng = np.random.default_rng(seed)
        self._schedule: torch.Tensor | None = None  # [n, T, 3]
        self._step_count = 0

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    def _partition(self, total: int, k: int) -> list[int]:
        """Split ``total`` steps into ``k`` positive-ish integers summing to ``total``.

        Weights are jittered for per-clip variety; the sum is corrected exactly.
        """
        if k <= 0:
            return []
        if total <= 0:
            return [0] * k
        w = self._np_rng.uniform(0.6, 1.4, size=k)
        parts = np.maximum(1, np.round(w / w.sum() * total)).astype(np.int64)
        # Correct the rounding drift so the parts sum to exactly ``total``.
        drift = int(parts.sum() - total)
        i = 0
        while drift != 0 and k > 0:
            j = i % k
            if drift > 0 and parts[j] > 1:
                parts[j] -= 1
                drift -= 1
            elif drift < 0:
                parts[j] += 1
                drift += 1
            i += 1
        return parts.tolist()

    def _sample_vel(self, skill_idx: int) -> np.ndarray:
        """Sample a representative ``(vx, vy, wz)`` uniformly within a skill bucket."""
        return np.array(
            [
                self._np_rng.uniform(self._bx[skill_idx, 0], self._bx[skill_idx, 1]),
                self._np_rng.uniform(self._by[skill_idx, 0], self._by[skill_idx, 1]),
                self._np_rng.uniform(self._bw[skill_idx, 0], self._bw[skill_idx, 1]),
            ],
            dtype=np.float32,
        )

    def _plan_clip(self) -> np.ndarray:
        """Build one clip's ``[T, 3]`` per-step velocity profile to hit the budgets."""
        T = self.num_steps
        dt = self.command_dt
        n_trans = int(round(self.transition_fraction * T))
        n_steady = T - n_trans

        p_stand = float(self._skill_frac[self._standing_idx]) if self._standing_idx is not None else 0.0
        moving_steady = int(round(n_steady * (1.0 - p_stand)))
        standing_steady = n_steady - moving_steady

        # Number of moving holds from the average hold duration.
        avg_hold_s = 0.5 * (self.hold_duration_range[0] + self.hold_duration_range[1])
        num_moving_holds = max(1, int(round(moving_steady * dt / max(avg_hold_s, dt))))

        # Place the standing-time budget across one or more stops. Standing OCCUPANCY
        # (skill_time_fractions[standing]) is the budget and is realized whenever it is
        # positive (>=1 stop, so short clips with few gaps still get their standing
        # time); standing_transition_fraction scales the NUMBER of separate stops
        # (chunking): higher -> more, shorter stops spread over more gaps.
        num_gaps = max(0, num_moving_holds - 1)
        num_stops = 0
        if self._standing_idx is not None and standing_steady > 0 and num_gaps > 0:
            num_stops = max(1, int(round(self.standing_transition_fraction * num_gaps)))
            num_stops = min(num_stops, num_gaps)
        if num_stops == 0:
            # No room for a stop (single hold) or no standing requested: all steady moving.
            moving_steady, standing_steady = n_steady, 0
        stop_gaps = set()
        if num_stops > 0:
            stop_gaps = set(self._np_rng.choice(num_gaps, size=num_stops, replace=False).tolist())

        # Assign moving-hold skills + durations.
        if self._moving_idx.size > 0:
            p_moving = self._skill_frac[self._moving_idx]
            p_moving = p_moving / p_moving.sum()
            move_skills = self._np_rng.choice(self._moving_idx, size=num_moving_holds, p=p_moving)
        else:
            # Degenerate: only standing is eligible -- everything stands.
            move_skills = np.full(num_moving_holds, self._standing_idx, dtype=np.int64)
        move_durs = self._partition(moving_steady, num_moving_holds)
        stand_durs = self._partition(standing_steady, num_stops)

        # Interleave moving holds with inserted standing stops.
        holds: list[tuple[int, int]] = []
        si = 0
        for i in range(num_moving_holds):
            holds.append((int(move_skills[i]), int(move_durs[i])))
            if i in stop_gaps:
                holds.append((int(self._standing_idx), int(stand_durs[si])))
                si += 1

        num_ramps = len(holds) - 1
        ramp_durs = self._partition(n_trans, num_ramps) if num_ramps > 0 else []
        reps = [self._sample_vel(skill) for skill, _ in holds]

        # Lay out the profile: hold 0, then (ramp_j, hold_{j+1}) for each subsequent hold.
        prof = np.empty((T, 3), dtype=np.float32)
        pos = 0

        def _fill(seg: np.ndarray) -> None:
            nonlocal pos
            take = min(seg.shape[0], T - pos)
            if take > 0:
                prof[pos : pos + take] = seg[:take]
                pos += take

        _fill(np.broadcast_to(reps[0], (holds[0][1], 3)))
        for j in range(1, len(holds)):
            rd = ramp_durs[j - 1]
            if rd > 0:
                # Linear ramp toward reps[j] (endpoint inclusive), excluding the
                # held start so every ramp step registers as "changing".
                alphas = (np.arange(1, rd + 1, dtype=np.float32) / rd)[:, None]
                _fill(reps[j - 1] + (reps[j] - reps[j - 1]) * alphas)
            _fill(np.broadcast_to(reps[j], (holds[j][1], 3)))

        # Pad any rounding shortfall by holding the final velocity.
        if pos < T:
            prof[pos:T] = reps[-1]
        return prof

    # ------------------------------------------------------------------
    # Public API (mirrors OfflineVelocitySampler)
    # ------------------------------------------------------------------

    def _clamp_to_buckets(self) -> torch.Tensor:
        """No-op clamp: scheduled velocities are built from within-bucket samples."""
        return self.vel_target

    def reset(self, n: int) -> torch.Tensor:
        """Plan ``n`` independent clip schedules and return the ``t = 0`` command."""
        self.n = int(n)
        self._step_count = 0
        sched = np.stack([self._plan_clip() for _ in range(self.n)], axis=0)  # [n, T, 3]
        self._schedule = torch.from_numpy(sched).to(self.device)
        self.vel_target_sampled = self._schedule[:, 0].clone()
        self.vel_target = self.vel_target_sampled.clone()
        self.vel_command = self.vel_target.clone()
        return self.vel_command

    def step(self) -> torch.Tensor:
        """Advance one control step and return the scheduled command ``[n, 3]``."""
        if self._schedule is None:
            raise RuntimeError("ScheduledVelocitySampler.step() called before reset().")
        self._step_count = min(self._step_count + 1, self.num_steps - 1)
        self.vel_target = self._schedule[:, self._step_count].clone()
        self.vel_command = self.vel_target.clone()
        return self.vel_command
