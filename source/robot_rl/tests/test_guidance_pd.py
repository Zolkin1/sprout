# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the terrain velocity-guidance PD helper."""

from __future__ import annotations

import math

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.guidance import (
    guidance_pd_commands,
)

# Wide clamps so the un-clamped PD math is visible.
_WIDE = torch.tensor([[-10.0, 10.0]])
_GAINS = torch.tensor([[1.5, 0.3, 0.5]])  # (y_kp, y_kd, heading_kp)


def _pd(target_y, yaw_target, root_y, lat_vel, heading, gains=_GAINS,
        vy_range=_WIDE, wz_range=_WIDE):
    return guidance_pd_commands(
        target_y=torch.tensor([target_y]),
        yaw_target=torch.tensor([yaw_target]),
        root_y=torch.tensor([root_y]),
        lat_vel_w=torch.tensor([lat_vel]),
        heading_w=torch.tensor([heading]),
        gains=gains,
        vy_range=vy_range,
        wz_range=wz_range,
    )


class TestGuidancePdCommands:
    def test_proportional_signs(self):
        """Target to the left (+y) commands +vy; heading error commands wz
        toward the target heading."""
        vy, wz = _pd(target_y=1.0, yaw_target=0.0, root_y=0.0, lat_vel=0.0, heading=-0.2)
        assert vy.item() == pytest.approx(1.5 * 1.0)
        assert wz.item() == pytest.approx(0.5 * 0.2)

    def test_damping_opposes_lateral_drift(self):
        """Zero position error but lateral drift (e.g. from a misaligned
        heading at forward speed) still produces a counteracting vy."""
        vy, _ = _pd(target_y=0.0, yaw_target=0.0, root_y=0.0, lat_vel=0.5, heading=0.0)
        assert vy.item() == pytest.approx(-0.3 * 0.5)

    def test_yaw_error_wraps(self):
        """Heading error is wrapped to [-pi, pi) — a heading just past pi
        relative to the target commands a small correction, not ~2*pi."""
        _, wz = _pd(target_y=0.0, yaw_target=0.0, root_y=0.0, lat_vel=0.0,
                    heading=math.pi - 0.1)
        # wrap(0 - (pi - 0.1)) = -(pi - 0.1): large error the short way.
        assert wz.item() == pytest.approx(0.5 * -(math.pi - 0.1))
        _, wz = _pd(target_y=0.0, yaw_target=0.0, root_y=0.0, lat_vel=0.0,
                    heading=2.0 * math.pi - 0.1)
        # One full turn collapses: wrap(0.1) = 0.1.
        assert wz.item() == pytest.approx(0.5 * 0.1, abs=1e-5)

    def test_outputs_clamped_to_ranges(self):
        vy, wz = _pd(
            target_y=100.0, yaw_target=3.0, root_y=0.0, lat_vel=0.0, heading=0.0,
            vy_range=torch.tensor([[-0.4, 0.4]]),
            wz_range=torch.tensor([[-0.5, 0.5]]),
        )
        assert vy.item() == pytest.approx(0.4)
        assert wz.item() == pytest.approx(0.5)
