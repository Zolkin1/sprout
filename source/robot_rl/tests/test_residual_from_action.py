# Copyright 2026 Zachary Olkin. All rights reserved.

"""The action -> residual map, pinned where it can be exercised without a simulator.

It is the contract between a trained adjuster and whatever drives it, so the properties below
are the ones an off-env consumer relies on: the caps bind, a zero action is exactly zero, and
one knot per node is the full-width path rather than an approximation of it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_rl.flow_matching.knot_basis import knot_basis
from robot_rl.rl.adjuster_policy import residual_from_action


def _lift(f: int, k: int) -> torch.Tensor:
    return torch.as_tensor(knot_basis(f, k, "catmull"), dtype=torch.float32)


def test_a_zero_action_is_exactly_a_zero_residual() -> None:
    """Bit-exact, because it is what makes `zero_residual` the bare-generator control."""
    f, k, d = 62, 13, 44
    scale = torch.ones(d)
    out = residual_from_action(torch.zeros(4, k * d), _lift(f, k), scale)
    assert out.shape == (4, f, d)
    assert torch.equal(out, torch.zeros_like(out))


def test_the_per_channel_cap_binds_after_the_lift_not_before() -> None:
    """Catmull-Rom overshoots ~25% between knots; squashing knots first would leak past the cap."""
    f, k, d = 62, 13, 44
    scale = torch.full((d,), 0.2)
    out = residual_from_action(torch.full((1, k * d), 50.0), _lift(f, k), scale)
    assert out.abs().max().item() <= 0.2 + 1e-6, "the residual exceeded its declared authority"


def test_the_scale_is_per_channel() -> None:
    """A zeroed channel must be INERT -- `joint_vel` and `ee_pos` ship at 0.0."""
    f, k, d = 62, 13, 8
    scale = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    out = residual_from_action(torch.full((2, k * d), 3.0), _lift(f, k), scale)
    assert torch.equal(out[..., 1::2], torch.zeros_like(out[..., 1::2]))
    assert (out[..., 0::2].abs() > 0).all()


def test_one_knot_per_node_is_the_full_width_path() -> None:
    """`knot_basis(F, F)` is the identity, so the lift is a no-op and the action IS the window."""
    f, d = 62, 44
    np.testing.assert_array_equal(knot_basis(f, f, "catmull"), np.eye(f))
    scale = torch.ones(d)
    a = torch.randn(3, f * d)
    out = residual_from_action(a, _lift(f, f), scale)
    torch.testing.assert_close(out, torch.tanh(a.view(3, f, d)))


def test_a_wrong_width_is_refused_by_name() -> None:
    """The actor and the env each state K; a mismatch must not be silently reshaped."""
    with pytest.raises(ValueError, match="width K \\* D"):
        residual_from_action(torch.zeros(1, 7), _lift(62, 13), torch.ones(44))
