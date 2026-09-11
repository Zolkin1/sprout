# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the stride-bank contract: the per-period geometry both halves of generation depend on.

The bank is read by the Motion Bricks side (to pin a stride's pose as the model's target keyframe) and
by the ``motion_gen`` side (to replay that SAME stride exactly as the clip's steady state). The number
that matters is the per-period displacement, because it is what the looped replay actually tracks --
and it is deliberately MEASURED from the stored motion rather than taken from the manifest's commanded
``steady_vel``, which differs by the solver's closure residual.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from motion_gen.motionbricks.stride_bank import (  # noqa: I001
    StrideBank,
    closure_residual,
    load_stride_bank,
    period_net_se2,
    save_stride_bank,
    twist_from_net,
    yaw_of,
)

DT = 0.02
_REAL_BANK = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                          "mb_raw", "quick_v6e", "stride_bank.npz")


def _qpos(period: int = 40, *, vx: float = 1.0, vy: float = 0.0, wz: float = 0.0,
          yaw0: float = 0.0, xy0=(0.0, 0.0), closed: bool = True) -> np.ndarray:
    """A synthetic stride advancing at body-frame ``(vx, vy, wz)`` from an arbitrary world pose."""
    f = period + 1
    th = yaw0 + np.arange(f) * wz * DT
    x = np.full(f, float(xy0[0]))
    y = np.full(f, float(xy0[1]))
    for k in range(1, f):
        # MIDPOINT, not explicit Euler: integrating a turn with the heading at k-1 biases the path
        # sideways by vx*wz*dt/2, which would show up as a phantom lateral velocity in the recovered
        # twist and mask whether twist_from_net is right.
        m = 0.5 * (th[k - 1] + th[k])
        x[k] = x[k - 1] + (math.cos(m) * vx - math.sin(m) * vy) * DT
        y[k] = y[k - 1] + (math.sin(m) * vx + math.cos(m) * vy) * DT
    q = np.zeros((f, 36), np.float32)
    q[:, 0], q[:, 1], q[:, 2] = x, y, 0.78
    q[:, 3], q[:, 6] = np.cos(th / 2), np.sin(th / 2)
    ph = 2.0 * np.pi * np.arange(f) / period
    q[:, 7] = np.sin(ph)                                  # periodic: frame `period` == frame 0
    if not closed:
        q[-1, 7] += 0.3                                   # break the closure
    return q


def test_yaw_of_matches_the_quaternion_it_was_built_from():
    for a in (0.0, 0.4, -1.2, 3.0):
        q = np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        assert float(yaw_of(q)) == pytest.approx(a, abs=1e-9)


def test_net_is_in_the_body_frame_so_world_placement_is_irrelevant():
    """The same gait placed anywhere / facing anywhere must report the same per-period displacement."""
    base = period_net_se2(_qpos(40, vx=1.2, wz=0.3))
    for yaw0, xy0 in ((1.1, (5.0, -3.0)), (-2.4, (-1.0, 8.0)), (math.pi, (0.0, 0.0))):
        moved = period_net_se2(_qpos(40, vx=1.2, wz=0.3, yaw0=yaw0, xy0=xy0))
        assert np.allclose(base, moved, atol=1e-5), (yaw0, xy0)


def test_twist_recovers_the_commanded_velocity_including_turns():
    """``twist_from_net`` must return the (vx, vy, wz) the stride was integrated at.

    The turning cases are the point: the naive ``net/T`` under-reports forward speed because the
    body-frame displacement is the arc's chord, so only the SE(2) log recovers the command.
    """
    for vx, vy, wz in ((1.0, 0.0, 0.0), (0.6, 0.4, 0.0), (0.9, 0.0, 0.5), (-0.7, 0.0, -0.3),
                       (1.4, 0.0, 0.8)):
        net = period_net_se2(_qpos(40, vx=vx, vy=vy, wz=wz))
        v = twist_from_net(net, 40 * DT)
        assert v[0] == pytest.approx(vx, abs=2e-3), (vx, vy, wz, v)
        assert v[1] == pytest.approx(vy, abs=2e-3), (vx, vy, wz, v)
        assert v[2] == pytest.approx(wz, abs=1e-6)


def test_naive_division_is_wrong_for_a_turn_and_the_log_is_not():
    """Pin the mechanism, so nobody 'simplifies' twist_from_net back into a division."""
    net = period_net_se2(_qpos(40, vx=0.9, wz=0.5))
    naive = net[[0, 1, 3]] / (40 * DT)
    exact = twist_from_net(net, 40 * DT)
    assert naive[0] < 0.89 < exact[0]                       # the chord under-reads
    assert exact[0] == pytest.approx(0.9, abs=2e-3)


def test_closure_residual_flags_an_unclosed_stride():
    assert closure_residual(_qpos(40))[0] == pytest.approx(0.0, abs=1e-6)
    assert closure_residual(_qpos(40, closed=False))[0] > 0.2


def test_save_load_round_trip(tmp_path):
    strides = [_qpos(40, vx=1.0), _qpos(31, vx=0.5, wz=0.2)]
    sv = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.2]], np.float32)
    out = str(tmp_path / "bank.npz")
    save_stride_bank(out, strides, sv, np.ones(2, np.uint8),
                     {"dt": DT, "names": ["a", "b"], "n_strides": 2})
    b = load_stride_bank(out)
    assert len(b) == 2 and b.dt == DT and b.names == ["a", "b"]
    assert [s.shape for s in b.strides] == [(41, 36), (32, 36)]
    assert list(b.period_frames) == [40, 31]
    assert np.allclose(b.period_s, [40 * DT, 31 * DT])
    assert np.allclose(b.achieved_vel[0], [1.0, 0.0, 0.0], atol=1e-5)


def test_achieved_vel_is_derived_not_the_commanded_value():
    """A bank whose manifest lies about the velocity must still report the motion's real velocity."""
    strides = [_qpos(40, vx=1.0)]
    lying = np.array([[2.5, 0.0, 0.0]], np.float32)        # manifest says 2.5, motion does 1.0
    b = StrideBank(strides=strides, steady_vel=lying, periodic=np.ones(1, np.uint8),
                   net_se2=np.stack([period_net_se2(strides[0])]), dt=DT, names=["x"], meta={})
    assert b.achieved_vel[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert b.steady_vel[0, 0] == pytest.approx(2.5)


@pytest.mark.skipif(not os.path.exists(_REAL_BANK), reason="no local stride bank")
def test_real_bank_closes_bit_exactly():
    """The property the whole exact-loop design rests on, asserted against the real bank.

    If this ever fails, a looped replay steps at every period seam and the loop needs blending -- which
    is the thing the design gets to skip.
    """
    b = load_stride_bank(_REAL_BANK)
    assert len(b) > 0 and b.dt == pytest.approx(0.02)
    gaps = np.array([closure_residual(s) for s in b.strides])
    assert gaps[:, 0].max() == 0.0, f"joint closure gap max {gaps[:, 0].max()}"
    assert gaps[:, 1].max() == 0.0, f"height closure gap max {gaps[:, 1].max()}"
    # Achieved vs commanded: a real, small discrepancy -- the reason achieved_vel is measured.
    err = np.abs(b.achieved_vel - b.steady_vel)
    assert np.percentile(err[:, 0], 95) < 0.05, np.percentile(err[:, 0], 95)
    assert err[:, 2].max() < 1e-5, err[:, 2].max()        # yaw rate is exact to float precision
