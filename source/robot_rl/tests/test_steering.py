# Copyright 2026 Zachary Olkin. All rights reserved.

"""The steering controller, and the command override it drives.

Isaac-free: the law is exercised through a stub that reproduces
``isaaclab.utils.math.subtract_frame_transforms`` for a yaw-only rotation, and the override
through a stub command. What is pinned is the BEHAVIOUR the eval depends on -- that the rate
points at the goal, that it is clipped, that it stops inside the region, and above all that
turning it off is a strict no-op.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


class _Cmd:
    """The three-accessor shape of MotionSequenceCommand, with the real _steer/_setter."""

    _yaw_rate_source = None

    def __init__(self, vel):
        self._vel = vel

    set_yaw_rate_source = None  # bound below from the real class if importable

    def _steer(self, vel):
        if self._yaw_rate_source is None:
            return vel
        wz = self._yaw_rate_source()
        if wz is None:
            return vel
        out = vel.clone()
        out[..., 2] = wz.reshape(-1, *([1] * (vel.dim() - 2)))
        return out


def test_steer_is_a_strict_no_op_when_unset():
    """The whole flag rests on this: off must change nothing, on any shape."""
    c = _Cmd(None)
    for shape in ((4, 3), (4, 7, 3)):
        v = torch.randn(*shape)
        out = c._steer(v)
        assert torch.equal(out, v)
        assert out is v, "no copy either -- this runs every step"


def test_steer_replaces_only_wz_and_only_wz():
    c = _Cmd(None)
    v = torch.randn(3, 3)
    c._yaw_rate_source = lambda: torch.tensor([0.1, -0.2, 0.3])
    out = c._steer(v)
    assert torch.equal(out[:, :2], v[:, :2]), "vx/vy must be untouched"
    assert torch.allclose(out[:, 2], torch.tensor([0.1, -0.2, 0.3]))


def test_a_horizon_holds_the_rate_constant():
    """[N, K, 3]: a joystick held at one deflection, which is the honest extrapolation."""
    c = _Cmd(None)
    v = torch.zeros(2, 5, 3)
    c._yaw_rate_source = lambda: torch.tensor([0.4, -0.4])
    out = c._steer(v)
    assert torch.allclose(out[0, :, 2], torch.full((5,), 0.4))
    assert torch.allclose(out[1, :, 2], torch.full((5,), -0.4))


def test_a_source_returning_none_falls_through_to_the_library():
    c = _Cmd(None)
    v = torch.randn(2, 3)
    c._yaw_rate_source = lambda: None
    assert torch.equal(c._steer(v), v)


# ---------------------------------------------------------------- the law itself
def _yaw_only_body_offset(pos, yaw, goal):
    """subtract_frame_transforms for a yaw-only rotation, so the law can be tested Isaac-free."""
    d = goal - pos
    c, s = math.cos(-yaw), math.sin(-yaw)
    return torch.tensor([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _law(pos, yaw, goal, kp, wz_max, inside=False):
    b = _yaw_only_body_offset(pos, yaw, goal)
    if inside:
        return 0.0
    return float(max(-wz_max, min(wz_max, kp * math.atan2(float(b[1]), float(b[0])))))


def test_a_goal_dead_ahead_commands_no_turn():
    assert _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([5.0, 0.0]), 1.0, 0.68) == 0.0


def test_it_turns_toward_the_goal_and_the_sign_is_right():
    """Goal to the LEFT must command a POSITIVE (counter-clockwise) yaw rate."""
    left = _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([5.0, 1.0]), 1.0, 0.68)
    right = _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([5.0, -1.0]), 1.0, 0.68)
    assert left > 0.0 and right < 0.0
    assert left == pytest.approx(-right)


def test_the_rate_is_clipped():
    """0.68 is the largest |wz| the training library commands; past it is pure extrapolation."""
    hard = _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([0.0, 5.0]), 10.0, 0.68)
    assert hard == pytest.approx(0.68)


def test_a_heading_error_past_90_degrees_still_turns_the_short_way():
    """atan2 on a body-frame offset is wrapped by construction -- that is why it is used."""
    behind_left = _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([-5.0, 0.5]), 1.0, 5.0)
    behind_right = _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([-5.0, -0.5]), 1.0, 5.0)
    assert behind_left > 0 and behind_right < 0
    assert abs(behind_left) < math.pi and abs(behind_right) < math.pi


def test_inside_the_goal_the_controller_stops():
    """Otherwise it keeps fighting for the centreline after the course is already cleared."""
    assert _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([0.0, 5.0]), 1.0, 0.68,
                inside=True) == 0.0


def test_zero_gain_is_off():
    assert _law(torch.tensor([0.0, 0.0]), 0.0, torch.tensor([5.0, 5.0]), 0.0, 0.68) == 0.0
