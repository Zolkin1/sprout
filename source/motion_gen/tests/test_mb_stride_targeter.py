# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the Motion Bricks stride targeter's phase clock.

The targeter feeds Motion Bricks the TARGET POSE window of each generation: ``nfpt`` frames of one of our
optimized traj-opt strides.  Two rates meet here and getting either wrong is silent -- the model still
produces plausible motion, it just disagrees with the root it was told to track, and the disagreement comes
out as foot slide rather than an error:

* the bank is stored at the RL control period (50 Hz) while Motion Bricks generates at 30 Hz, so a window's
  samples are ``mb_dt / bank_dt`` BANK frames apart, not one;
* a stride is one CLOSED cycle whose last frame duplicates the first, so the distinct period is ``F - 1``.

Both were wrong before: the phase advanced one bank frame per MB frame (0.6x cadence) and wrapped on
``F - nfpt``, skipping ``nfpt - 1`` frames every cycle.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "rsl_rl"))

mb_env = pytest.importorskip("mb_env")

BANK_DT = 0.02          # RL control period, what export_stride_bank.py writes
MB_DT = 1.0 / 30.0      # Motion Bricks' own frame rate
NFPT = 4


def _ramp_stride(n_frames: int, *, per_period: float = 1.0) -> np.ndarray:
    """A synthetic stride whose root x advances linearly and CLOSES: frame ``n-1`` duplicates frame 0.

    ``qpos[:, 0]`` is a clean ramp over the distinct period, so a fractional sample position can be read
    straight back out of it -- that is what makes the tap spacing assertions exact.  The pose channels carry
    a periodic signal so a wrap can be checked for continuity.
    """
    q = np.zeros((n_frames, 36), np.float32)
    p = n_frames - 1                                    # distinct period
    q[:, 0] = np.arange(n_frames) * (per_period / p)    # frame p lands exactly one period on
    q[:, 3] = 1.0                                       # identity quat (wxyz)
    q[:, 7] = np.sin(2.0 * np.pi * np.arange(n_frames) / p)   # periodic: equal at frames 0 and p
    q[:, 2] = 0.78
    return q


def _targeter(strides, vel=None):
    vel = np.zeros((len(strides), 3), np.float32) if vel is None else np.asarray(vel, np.float32)
    return mb_env.StrideTargeter(strides, vel, NFPT, bank_dt=BANK_DT, mb_dt=MB_DT)


def test_window_taps_are_one_mb_frame_apart_in_time():
    """Consecutive samples of a window must be ``mb_dt`` apart, i.e. 5/3 bank frames -- not 1."""
    tg = _targeter([_ramp_stride(43)])
    w = tg.window(0, advance=0)
    dx = np.diff(w[:, 0]) / (1.0 / 42.0)                # in bank frames (per_period=1 over p=42)
    assert np.allclose(dx, MB_DT / BANK_DT, atol=1e-4), dx
    assert w.shape == (NFPT, 36)


def test_phase_advances_one_period_per_period_of_mb_time():
    """Advancing a full period's worth of MB frames must return the phase to where it started."""
    p = 42
    tg = _targeter([_ramp_stride(p + 1)])
    tg.reset_phase(0.0)
    n_mb = (p * BANK_DT) / MB_DT                        # MB frames spanning one period
    assert abs(n_mb - 25.2) < 1e-6                      # 42*0.02 = 0.84 s at 30 fps
    for _ in range(int(round(n_mb))):
        tg.window(0, advance=1)
    # Phase is a fraction on a circle, so compare circularly: 25 of the 25.2 frames is 0.008 SHORT of a
    # full turn, which reads as 0.992, not 0.0.
    assert min(tg._u, 1.0 - tg._u) < 1.0 / n_mb         # inside one MB frame of the turn

    # And the old behaviour, for contrast: one BANK frame per MB frame would have needed 42 MB frames to
    # come round, i.e. it walked the stride at bank_dt/mb_dt = 0.6x the commanded cadence.
    assert n_mb / p == pytest.approx(BANK_DT / MB_DT, rel=1e-9)


def test_window_wraps_through_the_closure_frame():
    """A window straddling the seam must stay continuous in the periodic channels."""
    tg = _targeter([_ramp_stride(43)])
    tg.reset_phase(1.0 - 0.5 / 42.0)                    # half a bank frame short of the closure
    w = tg.window(0, advance=0)
    # The pose channel is periodic, so its step across the seam must match a within-period step.
    steps = np.abs(np.diff(w[:, 7]))
    assert steps.max() < 0.5, w[:, 7]                   # no jump back to the start of the cycle
    assert np.all(np.isfinite(w))


def test_old_wrap_span_would_have_skipped_frames():
    """Guard the regression directly: the period is F-1, and F-nfpt is NOT it."""
    tg = _targeter([_ramp_stride(43)])
    assert tg.periods[0] == 42                          # F-1, the traj-opt closure convention
    assert tg.periods[0] != 43 - NFPT                   # the old span, 3 frames short


def test_phase_is_shared_across_strides_not_keyed_per_index():
    """Switching stride and coming back must NOT resume a stale per-index phase."""
    tg = _targeter([_ramp_stride(43), _ramp_stride(43)])
    tg.reset_phase(0.0)
    for _ in range(10):
        tg.window(0, advance=1)
    u_after_a = tg._u
    tg.window(1, advance=1)                             # a different stride advances the SAME clock
    assert tg._u > u_after_a
    back = tg.window(0, advance=0)                      # returning must continue, not rewind
    assert back[0, 0] > 0.0


def test_phase_carries_as_a_fraction_across_different_periods():
    """A speed change must preserve gait phase as a FRACTION, since the periods differ."""
    tg = _targeter([_ramp_stride(43), _ramp_stride(31)])
    tg.reset_phase(0.25)
    a = tg.window(0, advance=0)
    b = tg.window(1, advance=0)
    # Both sampled at 25% through their own cycle -> same normalized root progress.
    assert a[0, 0] == pytest.approx(0.25, abs=0.02)
    assert b[0, 0] == pytest.approx(0.25, abs=0.02)


def test_quaternions_stay_normalized_through_interpolation():
    """Interpolated windows must return unit quats (the override builds a rotation from them)."""
    q = _ramp_stride(43)
    ang = np.linspace(0.0, 0.7, q.shape[0])
    q[:, 3], q[:, 6] = np.cos(ang / 2.0), np.sin(ang / 2.0)   # yaw sweep, wxyz
    tg = _targeter([q])
    tg.reset_phase(0.31)
    w = tg.window(0, advance=1)
    assert np.allclose(np.linalg.norm(w[:, 3:7], axis=-1), 1.0, atol=1e-5)


def test_sign_flipped_quats_do_not_blow_up_the_interpolation():
    """Double-cover sign flips in the bank must not produce a near-zero (unnormalizable) quat."""
    q = _ramp_stride(43)
    ang = np.linspace(0.0, 0.7, q.shape[0])
    q[:, 3], q[:, 6] = np.cos(ang / 2.0), np.sin(ang / 2.0)
    q[10:, 3:7] *= -1.0                                 # same rotations, flipped hemisphere
    tg = _targeter([q])
    tg.reset_phase(9.5 / 42.0)                          # straddle the flip
    w = tg.window(0, advance=0)
    assert np.allclose(np.linalg.norm(w[:, 3:7], axis=-1), 1.0, atol=1e-5)
