# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the closed-loop FM velocity command + bucket rolling.

Pure tensor-level tests on
``robot_rl...traj_tracking.closed_loop_command`` (no IsaacLab, no simulator).
They cover the feedforward + proportional position controller
(:func:`compute_closed_loop_command`) -- zero-error/zero-gain reduces to the
feedforward, a position error grows the command, the heading-frame rotation maps
world error onto the right body axis, the yaw error sign is correct, and the
clamps hold -- and the per-env bucket assignment (:func:`roll_buckets`):
``clip_fraction`` and ``closed_loop_fraction`` partition the envs and
``closed_loop_mask`` is always a subset of ``fm_mask``.
"""

from __future__ import annotations

import math

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.closed_loop_command import (
    compute_closed_loop_command,
    roll_buckets,
)

DEVICE = "cpu"


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw-only ``(x, y, z, w)`` quaternion for each angle in ``yaw`` ``[N]``."""
    half = 0.5 * yaw
    z = torch.sin(half)
    w = torch.cos(half)
    zeros = torch.zeros_like(z)
    return torch.stack([zeros, zeros, z, w], dim=-1)


def _call(
    v_ff,
    clip_pos,
    clip_yaw,
    robot_pos,
    robot_yaw,
    *,
    kp_pos=1.0,
    kd_pos=0.0,
    kp_yaw=1.0,
    kd_yaw=0.0,
    max_lin=10.0,
    max_ang=10.0,
    robot_lin_b=None,
    robot_ang_b=None,
):
    """Thin wrapper building the quaternions/zeros for a single test scenario."""
    n = v_ff.shape[0]
    if robot_lin_b is None:
        robot_lin_b = torch.zeros(n, 3, device=DEVICE)
    if robot_ang_b is None:
        robot_ang_b = torch.zeros(n, 3, device=DEVICE)
    return compute_closed_loop_command(
        v_ff,
        clip_pos,
        _yaw_quat(clip_yaw),
        robot_pos,
        _yaw_quat(robot_yaw),
        robot_lin_b,
        robot_ang_b,
        kp_pos=kp_pos,
        kd_pos=kd_pos,
        kp_yaw=kp_yaw,
        kd_yaw=kd_yaw,
        max_lin=max_lin,
        max_ang=max_ang,
    )


def test_zero_error_zero_gain_is_feedforward():
    """No position/yaw error and zero gains => command == feedforward."""
    v_ff = torch.tensor([[1.0, -0.3, 0.5]], device=DEVICE)
    pos = torch.tensor([[2.0, 5.0, 0.9]], device=DEVICE)
    yaw = torch.tensor([0.7], device=DEVICE)
    out = _call(v_ff, pos, yaw, pos.clone(), yaw.clone(), kp_pos=2.0, kp_yaw=3.0)
    torch.testing.assert_close(out, v_ff)


def test_forward_error_increases_forward_command():
    """A clip reference ahead in +x (robot facing +x) grows the forward command."""
    v_ff = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    robot_pos = torch.zeros(1, 3, device=DEVICE)
    clip_pos = torch.tensor([[0.5, 0.0, 0.0]], device=DEVICE)  # 0.5 m ahead in x
    yaw = torch.zeros(1, device=DEVICE)
    out = _call(v_ff, clip_pos, yaw, robot_pos, yaw, kp_pos=2.0)
    # vx = 1.0 (ff) + 2.0 * 0.5 = 2.0; vy, wz unchanged.
    torch.testing.assert_close(out, torch.tensor([[2.0, 0.0, 0.0]], device=DEVICE))


def test_heading_frame_rotation():
    """World +x error with the robot yawed +90 deg maps onto body -y."""
    v_ff = torch.zeros(1, 3, device=DEVICE)
    robot_pos = torch.zeros(1, 3, device=DEVICE)
    clip_pos = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)  # +x world
    yaw = torch.tensor([math.pi / 2], device=DEVICE)           # facing +y world
    out = _call(v_ff, clip_pos, yaw, robot_pos, yaw, kp_pos=1.0, kp_yaw=0.0)
    # Robot facing +y: world +x is the robot's body -y axis.
    torch.testing.assert_close(out, torch.tensor([[0.0, -1.0, 0.0]], device=DEVICE), atol=1e-6, rtol=0)


def test_yaw_error_sign():
    """A clip yaw ahead of the robot yaw yields a positive yaw-rate command."""
    v_ff = torch.zeros(1, 3, device=DEVICE)
    pos = torch.zeros(1, 3, device=DEVICE)
    robot_yaw = torch.tensor([0.0], device=DEVICE)
    clip_yaw = torch.tensor([0.4], device=DEVICE)
    out = _call(v_ff, pos, clip_yaw, pos.clone(), robot_yaw, kp_yaw=1.0)
    torch.testing.assert_close(out[:, 2], torch.tensor([0.4], device=DEVICE), atol=1e-6, rtol=0)


def test_yaw_error_wraps():
    """Yaw error wraps to [-pi, pi): clip just past -pi vs robot just below pi."""
    v_ff = torch.zeros(1, 3, device=DEVICE)
    pos = torch.zeros(1, 3, device=DEVICE)
    robot_yaw = torch.tensor([math.pi - 0.1], device=DEVICE)
    clip_yaw = torch.tensor([-math.pi + 0.1], device=DEVICE)
    out = _call(v_ff, pos, clip_yaw, pos.clone(), robot_yaw, kp_yaw=1.0)
    # Shortest path is +0.2 rad (across the wrap), not ~-2*pi.
    torch.testing.assert_close(out[:, 2], torch.tensor([0.2], device=DEVICE), atol=1e-5, rtol=0)


def test_clamps_respected():
    """Large errors saturate at the configured lin/ang limits."""
    v_ff = torch.zeros(1, 3, device=DEVICE)
    robot_pos = torch.zeros(1, 3, device=DEVICE)
    clip_pos = torch.tensor([[100.0, -100.0, 0.0]], device=DEVICE)
    robot_yaw = torch.zeros(1, device=DEVICE)
    clip_yaw = torch.tensor([math.pi], device=DEVICE)
    out = _call(v_ff, clip_pos, clip_yaw, robot_pos, robot_yaw,
                kp_pos=10.0, kp_yaw=10.0, max_lin=1.5, max_ang=2.0)
    assert out[0, 0].item() == 1.5
    assert out[0, 1].item() == -1.5
    assert abs(out[0, 2].item()) == 2.0


def test_kd_damping_opposes_velocity():
    """The Kd term subtracts the robot's body-frame velocity from the command."""
    v_ff = torch.zeros(1, 3, device=DEVICE)
    pos = torch.zeros(1, 3, device=DEVICE)
    yaw = torch.zeros(1, device=DEVICE)
    robot_lin_b = torch.tensor([[0.5, 0.0, 0.0]], device=DEVICE)
    out = _call(v_ff, pos, yaw, pos.clone(), yaw, kp_pos=0.0, kd_pos=2.0,
                robot_lin_b=robot_lin_b)
    torch.testing.assert_close(out[:, 0], torch.tensor([-1.0], device=DEVICE))


# ---------------------------------------------------------------------------
# Bucket rolling
# ---------------------------------------------------------------------------

def test_roll_clip_fraction_one_has_no_fm():
    """clip_fraction == 1 => no env tracks the FM (and none is closed-loop)."""
    n = 1000
    rand_fm = torch.rand(n, device=DEVICE)
    rand_cl = torch.rand(n, device=DEVICE)
    fm, cl = roll_buckets(rand_fm, rand_cl, clip_fraction=1.0, closed_loop_fraction=1.0)
    assert not bool(fm.any())
    assert not bool(cl.any())


def test_roll_all_fm_all_closed():
    """clip_fraction == 0 and closed_loop_fraction == 1 => every env closed-loop."""
    n = 1000
    rand_fm = torch.rand(n, device=DEVICE)
    rand_cl = torch.rand(n, device=DEVICE)
    fm, cl = roll_buckets(rand_fm, rand_cl, clip_fraction=0.0, closed_loop_fraction=1.0)
    assert bool(fm.all())
    assert bool(cl.all())


def test_roll_closed_is_subset_of_fm():
    """closed_loop_mask is always a subset of fm_mask, for any fractions."""
    n = 5000
    for clip_f, cl_f in [(0.2, 0.5), (0.5, 0.3), (0.0, 0.5), (0.8, 0.9)]:
        rand_fm = torch.rand(n, device=DEVICE)
        rand_cl = torch.rand(n, device=DEVICE)
        fm, cl = roll_buckets(rand_fm, rand_cl, clip_f, cl_f)
        assert bool((cl & ~fm).sum() == 0)  # no closed-loop env outside fm


def test_roll_fractions_approximate_budget():
    """Empirical bucket sizes track the requested fractions (large N)."""
    n = 200_000
    clip_f, cl_f = 0.25, 0.4
    g = torch.Generator(device=DEVICE).manual_seed(0)
    rand_fm = torch.rand(n, generator=g, device=DEVICE)
    rand_cl = torch.rand(n, generator=g, device=DEVICE)
    fm, cl = roll_buckets(rand_fm, rand_cl, clip_f, cl_f)
    frac_fm = fm.float().mean().item()
    frac_cl_in_fm = cl.float().sum().item() / fm.float().sum().item()
    assert abs(frac_fm - (1.0 - clip_f)) < 0.01
    assert abs(frac_cl_in_fm - cl_f) < 0.01
