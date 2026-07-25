# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the target-distribution (scheduled) offline velocity sampler.

Pure tensor-level tests on
``robot_rl...traj_tracking.clips.scheduled_velocity_sampler`` (no IsaacLab,
no YAML).  They roll the sampler exactly the way ``OfflineClipGenerator``
does (``reset(n)`` then ``step()`` ``T-1`` times) and assert the achieved
distribution matches the requested budgets: the changing-step fraction
tracks ``transition_fraction``, the per-skill steady occupancy tracks
``skill_time_fractions``, raising ``standing_transition_fraction`` injects
standing stops, and the stream is reproducible under a fixed seed.  Skill
labels are derived from the commanded velocity with the same
``bucket_for_velocity`` rule the generator uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motion_gen.clips.scheduled_velocity_sampler import (
    ScheduledVelocitySampler,
)
from motion_gen.skill_bucket import (
    bucket_for_velocity,
)

DEVICE = "cpu"
DT = 0.02
CLIP_S = 20.0
T = int(round(CLIP_S / DT))


@dataclass
class _Bucket:
    """Minimal stand-in for VelocityBucketCfg (same attribute names)."""

    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    heading: tuple[float, float] | None = None


# Forward-only buckets (vy/wz pinned to 0): only lin_vel_x is partitioned, so a
# velocity maps cleanly to one skill -- mirrors the G1 walk/run setup.
def _buckets() -> dict[str, _Bucket]:
    return {
        "standing": _Bucket(lin_vel_x=(0.0, 0.1)),
        "walk_forward": _Bucket(lin_vel_x=(0.1, 1.5)),
        "running": _Bucket(lin_vel_x=(1.5, 3.7)),
    }


SKILLS = ["standing", "walk_forward", "running"]


def _make(skill_time_fractions, *, transition_fraction=0.3, standing_transition_fraction=0.0,
          hold_duration_range=(2.0, 5.0), seed=0) -> ScheduledVelocitySampler:
    return ScheduledVelocitySampler(
        _buckets(), SKILLS, [1.0 / 3] * 3,
        skill_time_fractions=skill_time_fractions,
        transition_fraction=transition_fraction,
        standing_transition_fraction=standing_transition_fraction,
        hold_duration_range=hold_duration_range,
        command_dt=DT, clip_length_s=CLIP_S, device=DEVICE, seed=seed,
    )


def _roll(sampler: ScheduledVelocitySampler, n: int) -> torch.Tensor:
    """Roll the sampler like the generator: returns the [n, T, 3] command stream."""
    vels = [sampler.reset(n)]
    for _ in range(T - 1):
        vels.append(sampler.step())
    out = torch.stack(vels, dim=1)  # [n, T, 3]
    assert out.shape == (n, T, 3)
    return out


def _label_skills(vels: torch.Tensor) -> torch.Tensor:
    """[n*T] skill id per step via the generator's bucket rule."""
    s = _buckets()
    lin_x = torch.tensor([s[k].lin_vel_x for k in SKILLS])
    lin_y = torch.tensor([s[k].lin_vel_y for k in SKILLS])
    ang_z = torch.tensor([s[k].ang_vel_z for k in SKILLS])
    flat = vels.reshape(-1, 3)
    eligible = torch.ones(len(SKILLS), flat.shape[0], dtype=torch.bool)
    return bucket_for_velocity(flat, eligible, lin_x, lin_y, ang_z).reshape(vels.shape[:2])


def _changing_mask(vels: torch.Tensor, thresh: float = 1e-4) -> torch.Tensor:
    """[n, T] bool: any command component moved since the previous step (step 0 False)."""
    dv = (vels[:, 1:] - vels[:, :-1]).abs().amax(dim=-1)  # [n, T-1]
    changing = dv > thresh
    return torch.cat([torch.zeros(vels.shape[0], 1, dtype=torch.bool), changing], dim=1)


def test_transition_fraction_matches_target():
    """Fraction of changing steps tracks transition_fraction across the library."""
    for target in (0.15, 0.35, 0.55):
        s = _make({"walk_forward": 0.6, "running": 0.4}, transition_fraction=target, seed=3)
        vels = _roll(s, 300)
        changing = _changing_mask(vels)
        # Exclude step 0 of each clip (no predecessor) from the denominator.
        frac = changing[:, 1:].float().mean().item()
        assert abs(frac - target) < 0.05, (target, frac)


def test_steady_occupancy_matches_moving_fractions():
    """Among steady steps, per-skill time tracks skill_time_fractions (moving-only)."""
    target = {"walk_forward": 0.7, "running": 0.3}
    s = _make(target, transition_fraction=0.3, seed=5)
    vels = _roll(s, 400)
    skills = _label_skills(vels)
    steady = ~_changing_mask(vels)
    sid = {k: SKILLS.index(k) for k in SKILLS}
    n_walk = int(((skills == sid["walk_forward"]) & steady).sum())
    n_run = int(((skills == sid["running"]) & steady).sum())
    occ_walk = n_walk / (n_walk + n_run)
    assert abs(occ_walk - 0.7) < 0.06, occ_walk


def test_standing_occupancy_realized_and_chunked():
    """Standing time is set by skill_time_fractions (realized at any stf); stf chunks it."""
    frac = {"standing": 0.2, "walk_forward": 0.5, "running": 0.3}
    stand_id = SKILLS.index("standing")

    def standing_stats(stf):
        s = _make(frac, transition_fraction=0.3, standing_transition_fraction=stf, seed=7)
        vels = _roll(s, 200)
        steady = ~_changing_mask(vels)
        is_stand = (_label_skills(vels) == stand_id) & steady  # [n, T]
        occ = is_stand.float().mean().item()
        prev = torch.cat([torch.zeros(vels.shape[0], 1, dtype=torch.bool), is_stand[:, :-1]], dim=1)
        n_segments = int((is_stand & ~prev).sum())  # contiguous standing runs (stops)
        return occ, n_segments

    occ_lo, seg_lo = standing_stats(0.0)
    occ_hi, seg_hi = standing_stats(1.0)
    # Occupancy is realized at BOTH stf values (the budget, ~0.2 of steady ~= 0.14 of
    # total, is always placed) -- this is the bug the per-gap Bernoulli used to miss.
    assert occ_lo > 0.08 and occ_hi > 0.08, (occ_lo, occ_hi)
    # ...and it's roughly stable across stf (set by the budget, not by stf).
    assert abs(occ_lo - occ_hi) < 0.05, (occ_lo, occ_hi)
    # Higher stf chunks the same budget into MORE (shorter) stops.
    assert seg_hi > seg_lo, (seg_lo, seg_hi)


def test_reproducible_with_seed():
    """Same seed -> identical schedule stream."""
    a = _make({"walk_forward": 0.6, "running": 0.4}, seed=42)
    b = _make({"walk_forward": 0.6, "running": 0.4}, seed=42)
    assert torch.equal(_roll(a, 64), _roll(b, 64))


def test_velocities_finite_and_in_range():
    """Commands stay finite and within the overall vx span of the buckets."""
    s = _make({"standing": 0.2, "walk_forward": 0.5, "running": 0.3},
              transition_fraction=0.4, standing_transition_fraction=0.5, seed=1)
    vels = _roll(s, 50)
    assert torch.isfinite(vels).all()
    assert vels[..., 0].min() >= -1e-4 and vels[..., 0].max() <= 3.7 + 1e-4
