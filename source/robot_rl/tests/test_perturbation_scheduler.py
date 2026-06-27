# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the windowed perturbation scheduler used by record_data.py.

Pure tensor-level tests on the group / window / noise logic (the push and OOD
DR methods touch an Isaac Lab env and are not exercised here). They assert the
clean/perturbed split tracks ``clean_fraction``, windows start OFF and flip with
the right duty cycle, noise touches only active rows, and the stream is
reproducible under a fixed seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# perturbation_scheduler lives next to record_data.py under scripts/rsl_rl.
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "rsl_rl"
sys.path.insert(0, str(_SCRIPTS))

from perturbation_scheduler import PerturbationScheduler  # noqa: E402

DT = 0.02


def _make(n=4000, *, clean_fraction=0.4, on=(0.3, 1.0), off=(1.5, 4.0),
          action_noise_scale=0.3, obs_noise_scale=0.0, seed=0) -> PerturbationScheduler:
    s = PerturbationScheduler(
        num_envs=n, dt=DT, device="cpu", clean_fraction=clean_fraction,
        on_range_s=on, off_range_s=off, action_noise_scale=action_noise_scale,
        push_vel_range={"x": (-0.75, 0.75), "y": (-0.75, 0.75)},
        obs_noise_scale=obs_noise_scale, seed=seed,
    )
    s.assign_groups(torch.arange(n))
    return s


def test_group_fraction_matches_clean_fraction():
    """Perturbed fraction tracks 1 - clean_fraction."""
    s = _make(clean_fraction=0.4, seed=1)
    assert abs(float(s.is_perturbed.float().mean()) - 0.6) < 0.03


def test_windows_start_off():
    """All envs begin OFF (a clean lead-in before the first push)."""
    s = _make(seed=2)
    assert not bool(s.window_on.any())
    active, newly_on = s.step_windows()
    # First step cannot turn anything ON unless an off-window was 1 step (rare); the
    # duty cycle test below covers steady-state behavior.
    assert active.shape == (s.num_envs,)


def test_on_duty_cycle_matches_ranges():
    """Steady-state ON fraction among perturbed envs ~ on/(on+off)."""
    on, off = (0.3, 1.0), (1.5, 4.0)
    s = _make(on=on, off=off, seed=3)
    on_steps = tot = 0
    for _ in range(4000):
        active, _ = s.step_windows()
        on_steps += int(active.sum())
        tot += int(s.is_perturbed.sum())
    duty = on_steps / max(tot, 1)
    expected = (sum(on) / 2) / (sum(on) / 2 + sum(off) / 2)
    assert abs(duty - expected) < 0.05, (duty, expected)


def test_clean_envs_never_active():
    """Clean envs are never marked active, ever."""
    s = _make(seed=4)
    clean = ~s.is_perturbed
    for _ in range(2000):
        active, _ = s.step_windows()
        assert not bool((active.cpu() & clean).any())


def test_action_noise_only_on_active_rows():
    """corrupt_actions perturbs exactly the active rows; clean rows untouched."""
    s = _make(seed=5)
    for _ in range(50):
        active, _ = s.step_windows()
    a = torch.zeros(s.num_envs, 12)
    out = s.corrupt_actions(a, active)
    touched = out.abs().sum(dim=1) > 0
    assert torch.equal(touched, active.cpu())


def test_noise_noop_when_scale_zero_or_inactive():
    """Zero scale (or no active envs) returns the input unchanged."""
    s = _make(action_noise_scale=0.0, seed=6)
    a = torch.randn(s.num_envs, 12)
    active, _ = s.step_windows()
    assert torch.equal(s.corrupt_actions(a, active), a)
    # Obs noise off by default -> no-op too.
    o = torch.randn(s.num_envs, 30)
    assert torch.equal(s.corrupt_obs(o, active), o)


def test_reproducible_with_seed():
    """Same seed -> identical group draw and window stream."""
    a, b = _make(seed=42), _make(seed=42)
    assert torch.equal(a.is_perturbed, b.is_perturbed)
    for _ in range(500):
        aa, _ = a.step_windows()
        bb, _ = b.step_windows()
        assert torch.equal(aa, bb)
