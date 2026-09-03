# Copyright 2026 Zachary Olkin. All rights reserved.

"""The adjuster's residual: its per-channel authority, and the ANGLE bound on its rotation.

The RL adjuster emits a dimensionless action and the window's residual tables turn it into
physical units. Two things here are load-bearing and neither is visible in a shape:

* **The rotation bound is an angle, not a box.** ``base_pose``'s orientation half is two unit
  columns, so bounding the six components individually bounds the frame's tilt only to first
  order -- and the residual exists precisely to be large enough to matter. The measured
  relationship is ~0.020 of component displacement per 1.6 deg, so a component box calibrated
  at small angles under-constrains a 15 deg one. These tests measure the realised
  ``tilt_between`` rather than trusting the linearisation.
* **The residual must not be able to move its own grader further than the grader's bound.**
  The failure termination (``mdp.tracking_error.violates``) is measured against the reference
  the adjuster controls, so the residual's reach on ``base_pose`` decides whether the policy
  can cancel its own termination. The invariants below are derived from
  ``TrackingErrorSpec``'s own fields rather than restating its numbers, so tightening the spec
  cannot silently leave the residual over-powered.

The position cap is deliberately EQUAL to the violation threshold in v1 -- the point of the
first run is to measure how much authority the adjuster uses before tightening -- so
:func:`test_position_authority_does_not_exceed_the_violation_bound` asserts a ratio of 1.0
rather than a strict inequality. It is the test to change when that decision changes.
"""

from __future__ import annotations

import math

import pytest
import torch

from robot_rl.flow_matching.transforms import (
    quat_to_rot6d,
    rot6d_to_matrix,
    so3_log,
)
from robot_rl.flow_matching.window import (
    REFERENCE_WINDOW_RESIDUAL_SCALE,
    RESIDUAL_MAX_TILT_RAD,
    clamp_window_rotation_residual,
    project_window_rotations,
    residual_scale_vector,
    window_block_slices,
    window_dim,
)

#: The deployed layout: 29 joints, 4 end-effector frames, the three predicted blocks.
NJ = 29
NEE = 4
BLOCKS = ("base_pose", "base_vel", "joint_pos")


def _random_window(batch: int, nodes: int, dim: int, seed: int = 0) -> torch.Tensor:
    """A window whose ``base_pose`` rotation block holds real, random rotations."""
    gen = torch.Generator().manual_seed(seed)
    quat = torch.randn(batch, nodes, 4, generator=gen)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    window = torch.zeros(batch, nodes, dim)
    start = window_block_slices(NJ, NEE, BLOCKS)["base_pose"].start + 3
    window[..., start:start + 6] = quat_to_rot6d(quat)
    return window


def _induced_tilt(adjusted: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Angle between each node's adjusted and original ``base_pose`` rotation, radians."""
    start = window_block_slices(NJ, NEE, BLOCKS)["base_pose"].start + 3
    r_new = rot6d_to_matrix(adjusted[..., start:start + 6])
    r_old = rot6d_to_matrix(original[..., start:start + 6])
    return torch.linalg.norm(so3_log(r_new @ r_old.transpose(-1, -2)), dim=-1)


def _apply(action: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """The full residual path: scale, add, project, then bound the rotation."""
    scale = torch.tensor(residual_scale_vector(NJ, NEE, BLOCKS))
    adjusted = project_window_rotations(original + action * scale, NJ, NEE, BLOCKS)
    return clamp_window_rotation_residual(adjusted, original, NJ, NEE, BLOCKS)


def test_scale_vector_matches_the_layout() -> None:
    """One value per channel, in packing order, for the deployed selection."""
    dim = window_dim(NJ, NEE, BLOCKS)
    assert dim == 44
    scale = residual_scale_vector(NJ, NEE, BLOCKS)
    assert len(scale) == dim

    slices = window_block_slices(NJ, NEE, BLOCKS)
    base = slices["base_pose"].start
    assert scale[base:base + 3] == [0.20, 0.20, 0.20]
    assert scale[base + 3:base + 9] == [0.20] * 6
    assert scale[slices["base_vel"]] == [0.50] * 3 + [0.50] * 3
    assert scale[slices["joint_pos"]] == [0.30] * NJ


def test_unpredicted_blocks_are_inert() -> None:
    """``joint_vel`` and ``ee_pos`` get zero authority until someone calibrates them.

    No shipped generator predicts either, so they never reach the layout -- but a future
    selection that includes one must produce an inert residual rather than a free hand on a
    channel whose error has never been measured.
    """
    assert REFERENCE_WINDOW_RESIDUAL_SCALE["joint_vel"] == 0.0
    assert REFERENCE_WINDOW_RESIDUAL_SCALE["ee_pos"] == 0.0

    full = ("base_pose", "base_vel", "joint_pos", "joint_vel", "ee_pos")
    scale = residual_scale_vector(NJ, NEE, full)
    slices = window_block_slices(NJ, NEE, full)
    assert scale[slices["joint_vel"]] == [0.0] * NJ
    assert scale[slices["ee_pos"]] == [0.0] * (3 * NEE)


@pytest.mark.parametrize("amplitude", [0.5, 1.0, 3.0])
def test_the_rotation_bound_holds_as_a_measured_angle(amplitude: float) -> None:
    """The realised tilt never exceeds the bound, at any drive level.

    ``amplitude`` over-drives the pre-squash action; 3.0 saturates ``tanh`` on essentially
    every channel, which is the worst case the policy can reach.
    """
    original = _random_window(256, 62, window_dim(NJ, NEE, BLOCKS), seed=1)
    gen = torch.Generator().manual_seed(2)
    action = torch.tanh(torch.randn(original.shape, generator=gen) * amplitude)

    limited = _apply(action, original)
    tilt = _induced_tilt(limited, original)
    assert torch.isfinite(tilt).all()
    assert float(tilt.max()) <= RESIDUAL_MAX_TILT_RAD + 1e-5


def test_a_component_box_alone_would_not_have_held_the_bound() -> None:
    """The clamp is doing real work -- the linearisation genuinely under-constrains.

    This is the test that justifies enforcing an angle instead of a per-component box: if the
    small-angle equivalence held at 15 deg, the unclamped tilt would already be inside the
    bound and the clamp would be decoration.
    """
    original = _random_window(256, 62, window_dim(NJ, NEE, BLOCKS), seed=3)
    gen = torch.Generator().manual_seed(4)
    action = torch.tanh(torch.randn(original.shape, generator=gen))

    scale = torch.tensor(residual_scale_vector(NJ, NEE, BLOCKS))
    unclamped = project_window_rotations(original + action * scale, NJ, NEE, BLOCKS)
    assert float(_induced_tilt(unclamped, original).max()) > RESIDUAL_MAX_TILT_RAD


def test_the_clamp_only_ever_shrinks_and_is_idempotent() -> None:
    """A node inside the bound comes back unchanged, exactly, and re-applying changes nothing.

    Exactly rather than approximately: the log/exp round trip costs ~1e-7 in float32, and this
    runs on every node of every re-plan, so the implementation selects instead of relying on a
    unit scale factor. That is what lets the caller apply it blindly.
    """
    original = _random_window(128, 62, window_dim(NJ, NEE, BLOCKS), seed=5)
    gen = torch.Generator().manual_seed(6)
    small = torch.tanh(torch.randn(original.shape, generator=gen) * 0.02)

    scale = torch.tensor(residual_scale_vector(NJ, NEE, BLOCKS))
    adjusted = project_window_rotations(original + small * scale, NJ, NEE, BLOCKS)
    inside = _induced_tilt(adjusted, original) < RESIDUAL_MAX_TILT_RAD
    assert bool(inside.all()), "the fixture is meant to stay inside the bound"

    limited = clamp_window_rotation_residual(adjusted, original, NJ, NEE, BLOCKS)
    assert torch.equal(limited, adjusted)

    # And on a window that DOES bind, applying twice is the same as applying once.
    big = torch.tanh(torch.randn(original.shape, generator=gen) * 3.0)
    once = _apply(big, original)
    twice = clamp_window_rotation_residual(once, original, NJ, NEE, BLOCKS)
    assert torch.equal(twice, once)


def test_the_clamp_touches_nothing_but_the_rotation_block() -> None:
    """Position, velocity and joint channels pass through the rotation clamp untouched."""
    original = _random_window(64, 62, window_dim(NJ, NEE, BLOCKS), seed=7)
    gen = torch.Generator().manual_seed(8)
    action = torch.tanh(torch.randn(original.shape, generator=gen) * 3.0)

    scale = torch.tensor(residual_scale_vector(NJ, NEE, BLOCKS))
    adjusted = project_window_rotations(original + action * scale, NJ, NEE, BLOCKS)
    limited = clamp_window_rotation_residual(adjusted, original, NJ, NEE, BLOCKS)

    start = window_block_slices(NJ, NEE, BLOCKS)["base_pose"].start + 3
    assert torch.equal(limited[..., :start], adjusted[..., :start])
    assert torch.equal(limited[..., start + 6:], adjusted[..., start + 6:])


def test_the_clamp_is_a_no_op_without_base_pose() -> None:
    """A selection with no orientation to bound is returned as-is."""
    blocks = ("base_vel", "joint_pos")
    dim = window_dim(NJ, NEE, blocks)
    gen = torch.Generator().manual_seed(9)
    adjusted = torch.randn(8, 62, dim, generator=gen)
    original = torch.randn(8, 62, dim, generator=gen)
    assert clamp_window_rotation_residual(adjusted, original, NJ, NEE, blocks) is adjusted


def test_the_rotation_bound_leaves_the_tilt_check_real_margin() -> None:
    """The residual can move the reference's tilt by at most half the fatal bound.

    Derived from ``TrackingErrorSpec``'s own field, not from a restated 30 deg, so tightening
    the spec cannot leave this stale.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import BOOTSTRAP_SPEC

    ratio = RESIDUAL_MAX_TILT_RAD / BOOTSTRAP_SPEC.max_tilt_rad
    assert ratio == pytest.approx(0.5, abs=1e-9)
    assert math.degrees(RESIDUAL_MAX_TILT_RAD) == pytest.approx(15.0)


def test_position_authority_does_not_exceed_the_violation_bound() -> None:
    """The position cap is at, and never past, the lateral/height failure thresholds.

    v1 deliberately sits AT the bound (ratio 1.0) so the first run can measure how much
    authority is actually used. Change the ratio here in the same commit as the cap, and read
    the hazard note in ``REFERENCE_WINDOW_RESIDUAL_SCALE`` before loosening it further --
    past 1.0 the adjuster can move the graded node clean through its own termination.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import BOOTSTRAP_SPEC

    cap = REFERENCE_WINDOW_RESIDUAL_SCALE["base_pos"]
    assert max(cap) <= min(BOOTSTRAP_SPEC.max_lateral_m, BOOTSTRAP_SPEC.max_height_m)
