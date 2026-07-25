# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the 2D goal-directed PD steering law.

Pure tensor tests on :func:`goal_pd_commands` (no IsaacLab): it must reduce to
the 1-D :func:`guidance_pd_commands` for a robot at heading 0 chasing a goal
straight ahead, turn toward an off-axis goal, and clamp its outputs.
"""

from __future__ import annotations

import math

import torch

from motion_gen.guidance_math import (
    goal_pd_commands,
    guidance_pd_commands,
)

DEVICE = "cpu"
GAINS = torch.tensor([[1.5, 0.3, 0.5]], device=DEVICE)   # (y_kp, y_kd, heading_kp)
VY = torch.tensor([[-0.4, 0.4]], device=DEVICE)
WZ = torch.tensor([[-0.5, 0.5]], device=DEVICE)


def test_reduces_to_1d_guidance_at_heading_zero():
    """heading=0, goal straight ahead (+x) -> matches the 1-D guidance law."""
    anchor = torch.tensor([[0.0, 0.2]], device=DEVICE)     # off-center in y
    goal = torch.tensor([[5.0, 0.6]], device=DEVICE)       # ahead, +y target
    heading = torch.tensor([0.0], device=DEVICE)
    vy, wz = goal_pd_commands(anchor, heading, goal, GAINS, VY, WZ, lat_vel_b=0.0)
    # 1-D law: target_y=0.6, root_y=0.2, heading target = bearing ~ small +y.
    vy_ref, wz_ref = guidance_pd_commands(
        target_y=goal[:, 1], yaw_target=torch.atan2(goal[:, 1] - anchor[:, 1], goal[:, 0] - anchor[:, 0]),
        root_y=anchor[:, 1], lat_vel_w=torch.zeros(1, device=DEVICE), heading_w=heading,
        gains=GAINS, vy_range=VY, wz_range=WZ,
    )
    assert torch.allclose(vy, vy_ref, atol=1e-6)
    assert torch.allclose(wz, wz_ref, atol=1e-6)


def test_turns_toward_off_axis_goal():
    """A goal to the robot's left produces a positive (left) yaw command."""
    anchor = torch.tensor([[0.0, 0.0]], device=DEVICE)
    goal = torch.tensor([[2.0, 2.0]], device=DEVICE)       # 45 deg to the left
    heading = torch.tensor([0.0], device=DEVICE)
    _, wz = goal_pd_commands(anchor, heading, goal, GAINS, VY, WZ)
    assert wz.item() > 0.0                                  # turn left toward goal
    # A goal to the right -> negative yaw.
    goal_r = torch.tensor([[2.0, -2.0]], device=DEVICE)
    _, wz_r = goal_pd_commands(anchor, heading, goal_r, GAINS, VY, WZ)
    assert wz_r.item() < 0.0


def test_goal_behind_and_left_turns_left_hard():
    """A goal behind-left yaws left and saturates (bearing > 90 deg)."""
    anchor = torch.tensor([[0.0, 0.0]], device=DEVICE)
    goal = torch.tensor([[-1.0, 1.0]], device=DEVICE)      # behind, left
    heading = torch.tensor([0.0], device=DEVICE)
    _, wz = goal_pd_commands(anchor, heading, goal, GAINS, VY, WZ)
    assert abs(wz.item() - 0.5) < 1e-5                      # clamped to +wz max


def test_clamps_outputs():
    """Large errors clamp vy/wz to their ranges."""
    anchor = torch.tensor([[0.0, 0.0]], device=DEVICE)
    goal = torch.tensor([[0.01, 100.0]], device=DEVICE)    # far to the left
    heading = torch.tensor([0.0], device=DEVICE)
    vy, wz = goal_pd_commands(anchor, heading, goal, GAINS, VY, WZ)
    assert abs(vy.item() - 0.4) < 1e-5
    assert abs(wz.item() - 0.5) < 1e-5


def test_heading_aligned_with_goal_no_turn():
    """Already facing the goal -> ~zero yaw, lateral error ~0 -> ~zero vy."""
    anchor = torch.tensor([[0.0, 0.0]], device=DEVICE)
    goal = torch.tensor([[math.cos(0.7), math.sin(0.7)]], device=DEVICE)
    heading = torch.tensor([0.7], device=DEVICE)           # facing the goal
    vy, wz = goal_pd_commands(anchor, heading, goal, GAINS, VY, WZ)
    assert abs(wz.item()) < 1e-6
    assert abs(vy.item()) < 1e-6
