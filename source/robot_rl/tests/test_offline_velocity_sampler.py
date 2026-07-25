# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the offline flat-ground velocity sampler.

Pure tensor-level tests on
``robot_rl...traj_tracking.clips.offline_velocity_sampler``.  No IsaacLab,
no YAML.  These assert the offline sampler reproduces the live sampler's
*formulas and distribution*: skill mix matches ``skill_probs``, velocities
are uniform within the per-skill buckets, the ramp never exceeds
``max_acc * dt`` per step, and the per-skill clamp uses the real
``clamp_pd_channels_to_buckets`` helper (so its behaviour is identical to
live by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motion_gen.clips.offline_velocity_sampler import (
    OfflineVelocitySampler,
)
from motion_gen.skill_bucket import (
    clamp_pd_channels_to_buckets,
)

DEVICE = "cpu"


@dataclass
class _Bucket:
    """Minimal stand-in for VelocityBucketCfg (same attribute names)."""

    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    heading: tuple[float, float] | None = None


def _buckets() -> dict[str, _Bucket]:
    return {
        "walk_forward": _Bucket(lin_vel_x=(0.1, 1.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.6, 0.6)),
        "running": _Bucket(lin_vel_x=(1.5, 3.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.3, 0.3)),
        "walk_backwards": _Bucket(lin_vel_x=(-1.0, -0.1), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.4, 0.4)),
        "standing": _Bucket(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    }


SKILLS = ["walk_forward", "running", "walk_backwards", "standing"]


def _make(skill_probs, *, max_acc=2.0, dt=0.02, transition=None, seed=0) -> OfflineVelocitySampler:
    return OfflineVelocitySampler(
        _buckets(), SKILLS, skill_probs,
        resampling_time_range=(0.5, 1.5), max_acc=max_acc, command_dt=dt,
        skill_transition_prob=transition, device=DEVICE, seed=seed,
    )


def test_reset_snaps_target_and_respects_buckets():
    """After reset, vel_command lies inside the sampled skill's bucket."""
    s = _make([0.25, 0.25, 0.25, 0.25], seed=1)
    cmd = s.reset(2048)
    assert cmd.shape == (2048, 3)
    # vx within the sampled skill's x bucket (reset snaps the ramp, no clamp on x needed)
    lo_x = s.lin_x[s.sampled_skill, 0]
    hi_x = s.lin_x[s.sampled_skill, 1]
    assert torch.all(cmd[:, 0] >= lo_x - 1e-5) and torch.all(cmd[:, 0] <= hi_x + 1e-5)
    # vy / wz clamped into the per-skill bucket
    assert torch.all(cmd[:, 1] >= s.lin_y[s.sampled_skill, 0] - 1e-5)
    assert torch.all(cmd[:, 1] <= s.lin_y[s.sampled_skill, 1] + 1e-5)
    assert torch.all(cmd[:, 2] >= s.ang_z[s.sampled_skill, 0] - 1e-5)
    assert torch.all(cmd[:, 2] <= s.ang_z[s.sampled_skill, 1] + 1e-5)


def test_skill_mix_matches_probs():
    """Reset skill draw is i.i.d. from skill_probs (within sampling noise)."""
    probs = [0.5, 0.2, 0.2, 0.1]
    s = _make(probs, seed=7)
    s.reset(40000)
    counts = torch.bincount(s.sampled_skill, minlength=4).float() / 40000.0
    assert torch.allclose(counts, torch.tensor(probs), atol=0.02), counts


def test_standing_skill_is_zero_velocity():
    """The standing bucket is (0,0) on every channel -> zero command."""
    s = _make([0.0, 0.0, 0.0, 1.0], seed=3)  # force standing
    cmd = s.reset(64)
    assert torch.all(s.sampled_skill == SKILLS.index("standing"))
    assert torch.allclose(cmd, torch.zeros_like(cmd), atol=1e-6)


def test_ramp_never_exceeds_max_acc():
    """Per-step change in the held target is bounded by max_acc * dt."""
    max_acc, dt = 1.5, 0.02
    s = _make([0.4, 0.3, 0.2, 0.1], max_acc=max_acc, dt=dt, seed=5)
    s.reset(512)
    prev = s.vel_target.clone()
    limit = max_acc * dt
    for _ in range(300):
        s.step()
        delta = (s.vel_target - prev).abs()
        assert torch.all(delta <= limit + 1e-6), delta.max().item()
        prev = s.vel_target.clone()


def test_ramp_converges_to_sampled_target():
    """With enough steps and no resample, the held target reaches the sampled target."""
    # Long resample window so no resample fires during the test.
    s = OfflineVelocitySampler(
        _buckets(), SKILLS, [1.0, 0.0, 0.0, 0.0],  # always walk_forward
        resampling_time_range=(1e6, 1e6), max_acc=5.0, command_dt=0.02, device=DEVICE, seed=2,
    )
    s.reset(128)
    target = s.vel_target_sampled.clone()
    for _ in range(2000):
        s.step()
    assert torch.allclose(s.vel_target, target, atol=1e-4)


def test_transition_prob_excludes_current_skill():
    """With transition prob 1.0, every resample moves to a different skill."""
    s = _make([0.25, 0.25, 0.25, 0.25], transition=1.0, seed=11)
    s.reset(4096)
    before = s.sampled_skill.clone()
    # Force every timer to expire on the next step.
    s.time_left[:] = 0.0
    s.step()
    # Every clip must have changed skill (transition prob = 1.0).
    assert torch.all(s.sampled_skill != before)


def test_clamp_matches_live_helper():
    """vel_command equals clamp_pd_channels_to_buckets applied to vel_target."""
    s = _make([0.4, 0.3, 0.2, 0.1], seed=9)
    s.reset(256)
    for _ in range(50):
        s.step()
    expected = clamp_pd_channels_to_buckets(s.vel_target, s.sampled_skill, s.lin_y, s.ang_z)
    assert torch.equal(s.vel_command, expected)


def test_reproducible_with_seed():
    """Same seed -> identical sampling stream."""
    a = _make([0.4, 0.3, 0.2, 0.1], seed=42)
    b = _make([0.4, 0.3, 0.2, 0.1], seed=42)
    ca, cb = a.reset(256), b.reset(256)
    assert torch.equal(ca, cb)
    for _ in range(100):
        assert torch.equal(a.step(), b.step())
