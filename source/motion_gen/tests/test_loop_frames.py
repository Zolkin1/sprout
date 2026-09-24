# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for exact looped replay of a periodic stride -- the steady-state half of flat clip generation.

Motion Bricks in-betweens the TRANSIENTS; steady state is an exact replay of an optimized traj-opt
stride, looped. That is what makes the steady velocity exact instead of whatever duration the model
happened to choose. The two properties these tests protect:

* the replay is EXACT -- no blending, no drift, and the achieved velocity is the stride's own measured
  per-period displacement (not the manifest's commanded ``steady_vel``, which differs by the solver's
  closure residual);
* it is exact for an ARBITRARY frame count and an ARBITRARY entry phase, because a commanded duration
  is not a whole number of gait cycles and a transient hands off wherever it was aimed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from motion_gen.motionbricks.stride_bank import period_net_se2, yaw_of
from motion_gen.terrain.box_clip import loop_frames, loop_periodic

DT = 0.02


def _stride(period: int = 40, *, vx: float = 1.0, wz: float = 0.0):
    """A synthetic CLOSED stride: pose periodic, root advancing at ``(vx, wz)``.

    ``F = period + 1`` frames, with frame ``F-1`` duplicating frame 0 in pose -- the traj-opt closure
    convention the real bank satisfies bit-exactly.
    """
    f = period + 1
    th = np.arange(f) * wz * DT
    x = np.zeros(f)
    y = np.zeros(f)
    for k in range(1, f):                              # integrate the body-frame command
        x[k] = x[k - 1] + math.cos(th[k - 1]) * vx * DT
        y[k] = y[k - 1] + math.sin(th[k - 1]) * vx * DT
    rp = torch.tensor(np.stack([x, y, np.full(f, 0.78)], 1), dtype=torch.float32)
    rq = torch.tensor(np.stack([np.cos(th / 2), np.zeros(f), np.zeros(f), np.sin(th / 2)], 1),
                      dtype=torch.float32)
    ph = 2.0 * np.pi * np.arange(f) / period           # periodic: frame `period` == frame 0
    jt = torch.tensor(np.stack([np.sin(ph), np.cos(ph)] + [np.zeros(f)] * 27, 1), dtype=torch.float32)
    return rp, rq, jt


def test_returns_exactly_the_requested_frame_count():
    """Any duration, including mid-period and many periods."""
    rp, rq, jt = _stride(40)
    for n in (1, 7, 40, 41, 100, 999):
        out = loop_frames(rp, rq, jt, None, n)
        assert out[0].shape[0] == n, n
        assert out[1].shape[0] == n and out[2].shape[0] == n


def test_pose_is_periodic_across_the_seam_with_no_blending():
    """A looped joint trajectory must repeat exactly every ``period`` frames."""
    rp, rq, jt = _stride(40)
    _, _, ljt, _ = loop_frames(rp, rq, jt, None, 200)
    for k in range(0, 200 - 40, 40):
        assert torch.allclose(ljt[k], ljt[k + 40], atol=1e-6), k


def test_no_step_at_the_period_seam():
    """The frame-to-frame joint step at a seam must look like any other frame-to-frame step.

    This is the direct detector for a stride whose last frame is NOT the closure frame: the seam step
    would then be ~2x a normal step (one real step plus the un-closed residual).
    """
    rp, rq, jt = _stride(40)
    _, _, ljt, _ = loop_frames(rp, rq, jt, None, 160)
    steps = torch.linalg.norm(torch.diff(ljt, dim=0), dim=1)
    seams = [39, 79, 119]                              # frames straddling each wrap
    assert max(float(steps[s]) for s in seams) < 1.5 * float(steps.median())


def test_root_velocity_is_the_strides_own_measured_velocity():
    """The looped root must advance at ``net/(P*dt)`` -- the exactness the whole redesign is for."""
    rp, rq, jt = _stride(40, vx=1.3)
    net = period_net_se2(np.concatenate(
        [rp.numpy(), rq.numpy(), jt.numpy()], axis=1))
    v_expected = net[0] / (40 * DT)
    lrp, _, _, _ = loop_frames(rp, rq, jt, None, 401)
    v_actual = float(torch.linalg.norm(lrp[-1, :2] - lrp[0, :2])) / (400 * DT)
    assert v_actual == pytest.approx(v_expected, rel=2e-3)


def test_turning_stride_keeps_turning_across_seams():
    """A turning stride must accumulate yaw per period, not reset to the first period's heading."""
    rp, rq, jt = _stride(40, vx=0.8, wz=0.5)
    lrp, lrq, _, _ = loop_frames(rp, rq, jt, None, 401)
    yaw = yaw_of(lrq.numpy())
    total = float(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])
    assert total == pytest.approx(0.5 * 400 * DT, rel=5e-3)
    # ... and the path must curve, not run straight.
    chord = float(torch.linalg.norm(lrp[-1, :2] - lrp[0, :2]))
    arc = float(torch.linalg.norm(torch.diff(lrp[:, :2], dim=0), dim=1).sum())
    assert chord < 0.8 * arc


def test_entry_phase_shifts_the_replay_without_breaking_it():
    """Entering mid-cycle must start at that pose and stay periodic from there."""
    rp, rq, jt = _stride(40)
    for start in (0, 7, 23, 39):
        _, _, ljt, _ = loop_frames(rp, rq, jt, None, 120, start=start)
        assert torch.allclose(ljt[0], jt[start], atol=1e-6), start
        assert torch.allclose(ljt[0], ljt[40], atol=1e-6), start


def test_contact_is_carried_and_cut_to_length():
    rp, rq, jt = _stride(40)
    contact = torch.zeros(41, 2, dtype=torch.uint8)
    contact[:20, 0] = 1
    contact[20:, 1] = 1
    _, _, _, lc = loop_frames(rp, rq, jt, contact, 95, start=3)
    assert lc is not None and lc.shape == (95, 2)


def test_drift_over_many_periods_is_negligible():
    """Chained SE(2) alignment accumulates float error; over a 20 s clip it must stay irrelevant."""
    rp, rq, jt = _stride(40, vx=1.0)
    n = 1000                                            # 20 s at 50 Hz, i.e. 25 periods
    lrp, _, _, _ = loop_frames(rp, rq, jt, None, n)
    step = torch.linalg.norm(torch.diff(lrp[:, :2], dim=0), dim=1)
    assert float(step.max() - step.min()) < 1e-4        # uniform advance, no seam hiccup


def test_offset_mode_refuses_a_turning_stride():
    """The offset path repeats orientation verbatim, so it is wrong for a turn -- and must say so."""
    rp, rq, jt = _stride(40, vx=0.8, wz=0.5)
    with pytest.raises(ValueError, match="translation-only"):
        loop_periodic(rp, rq, jt, None, 3, offset=[1.0, 0.0, 0.0])
    # A straight stride is still allowed through it (the stair path relies on that).
    srp, srq, sjt = _stride(40, vx=0.8, wz=0.0)
    out = loop_periodic(srp, srq, sjt, None, 3, offset=[0.8 * 40 * DT, 0.0, 0.0])
    assert out[0].shape[0] == 41 + 2 * 40
