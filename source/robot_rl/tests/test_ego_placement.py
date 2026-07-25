# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the ego-centric (receding-horizon) clip re-anchor placement.

Pure tensor-level tests on
``robot_rl...traj_tracking.world_lift.ego_placement`` (no IsaacLab, no
simulator).  ``ego_placement`` returns the per-env yaw+xy *placement* that puts a
clip's current-step base onto a target root pose.  The runtime then lifts the
clip-world base through that placement; for the pure-yaw placement this reduces to
``base_w = world_pos + R(world_quat) @ base_cf`` (position) and
``yaw(base_w) = yaw(world_quat) + yaw(base_cf)`` (heading) -- exactly what
``world_lift`` does (``anchor_inv_yaw(Q) == Q`` for pure-yaw ``Q``).  The tests
re-implement that lift independently (small local quaternion helpers) and assert
the lifted base reproduces the target's planar position + heading AND height (z)
-- the current-step base z is snapped onto the target root (matching the FM deploy
anchor), not kept at the clip's natural height -- that the returned quaternion is
pure-yaw, and that the re-anchor is a round-trip no-op when the target already
equals the reference.
"""

from __future__ import annotations

import math

import torch

from motion_gen.world_lift import (
    anchor_mask_from_free,
    anchor_mask_free_names,
    _ANCHOR_MASK_LEGACY,
    _ANCHOR_MASK_FLAT,
    _ANCHOR_MASK_STAIRS,
    _ANCHOR_MASK_SLOPE,
)
# The placement/lift helpers are quaternion-order sensitive, so Isaac Lab-side code
# (these tests included) must reach them through the scalar-last adapter.
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.world_lift_isaac import (
    ego_placement,
    ego_placement_masked,
    yaw_from_quat as _yaw_from_quat,
)

DEVICE = "cpu"


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``(x, y, z, w)`` quaternions ``[..., 4]``."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Active rotation of vector ``v`` ``[..., 3]`` by quaternion ``q`` (xyzw)."""
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def _quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """``(x, y, z, w)`` quaternion from intrinsic roll/pitch/yaw angles ``[N]``."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dim=-1,
    )


def _lift_base(
    world_pos: torch.Tensor,
    world_quat: torch.Tensor,
    base_pos_cf: torch.Tensor,
    base_ori_cf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent runtime lift for a pure-yaw placement (mirrors world_lift)."""
    return world_pos + _quat_apply(world_quat, base_pos_cf), _quat_mul(world_quat, base_ori_cf)


def _wrap(angles: torch.Tensor) -> torch.Tensor:
    """Wrap to ``[-pi, pi)`` for angle comparisons."""
    return torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi


def test_lifted_base_reproduces_target_pos_and_heading():
    """Lifting the clip base through the returned placement hits the target xyz + yaw."""
    torch.manual_seed(0)
    n = 32
    base_pos_cf = torch.randn(n, 3, device=DEVICE) * 2.0
    base_ori_cf = _quat_from_euler(  # non-trivial tilt so yaw extraction matters
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-math.pi, math.pi),
    )
    target_pos = torch.randn(n, 3, device=DEVICE) * 5.0
    target_quat = _quat_from_euler(
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-math.pi, math.pi),
    )

    world_pos, world_quat = ego_placement(base_pos_cf, base_ori_cf, target_pos, target_quat)
    lifted_pos, lifted_quat = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)

    # Full position (xy AND z) matches the target root exactly.
    assert torch.allclose(lifted_pos, target_pos, atol=1e-5)
    # Heading matches the target root (full tilt is *not* matched -- planar only).
    assert torch.allclose(_wrap(_yaw_from_quat(lifted_quat) - _yaw_from_quat(target_quat)),
                          torch.zeros(n), atol=1e-5)


def test_height_snapped_to_target_root():
    """Lifted base z equals the target root z (robot-anchored height, not natural).

    This is the train/deploy alignment: the ego re-anchor pins the current-step base
    height onto the target root just like the FM deploy anchor, so an arbitrary target
    z is reproduced exactly (independent of the clip's own natural base height).
    """
    torch.manual_seed(1)
    n = 16
    base_pos_cf = torch.randn(n, 3, device=DEVICE)
    base_ori_cf = _quat_from_euler(
        torch.zeros(n), torch.zeros(n), torch.empty(n).uniform_(-math.pi, math.pi)
    )
    target_pos = torch.randn(n, 3, device=DEVICE) * 3.0  # arbitrary target z
    target_quat = _quat_from_euler(
        torch.zeros(n), torch.zeros(n), torch.empty(n).uniform_(-math.pi, math.pi)
    )

    world_pos, world_quat = ego_placement(base_pos_cf, base_ori_cf, target_pos, target_quat)
    lifted_pos, _ = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)

    # Placement z is offset so the lifted current-step base sits exactly at the target z.
    assert torch.allclose(world_pos[:, 2], target_pos[:, 2] - base_pos_cf[:, 2], atol=1e-6)
    assert torch.allclose(lifted_pos[:, 2], target_pos[:, 2], atol=1e-5)


def test_returned_placement_is_pure_yaw():
    """The placement quaternion has no roll/pitch (x == y == 0), a ground frame."""
    torch.manual_seed(2)
    n = 16
    base_ori_cf = _quat_from_euler(
        torch.empty(n).uniform_(-0.5, 0.5),
        torch.empty(n).uniform_(-0.5, 0.5),
        torch.empty(n).uniform_(-math.pi, math.pi),
    )
    target_quat = _quat_from_euler(
        torch.empty(n).uniform_(-0.5, 0.5),
        torch.empty(n).uniform_(-0.5, 0.5),
        torch.empty(n).uniform_(-math.pi, math.pi),
    )
    _, world_quat = ego_placement(
        torch.randn(n, 3), base_ori_cf, torch.randn(n, 3), target_quat
    )
    assert torch.allclose(world_quat[:, 0], torch.zeros(n), atol=1e-6)
    assert torch.allclose(world_quat[:, 1], torch.zeros(n), atol=1e-6)
    assert torch.allclose(torch.linalg.norm(world_quat, dim=-1), torch.ones(n), atol=1e-6)


def test_reanchor_roundtrip_is_planar_noop():
    """Re-anchoring to the reference's own current lifted pose preserves the lift.

    Build a target by lifting the base through an existing pure-yaw placement, then
    re-anchor onto that target: the new placement must reproduce the same position
    (xyz) + heading, i.e. re-planning while already on the reference does not move
    the reference.
    """
    torch.manual_seed(3)
    n = 16
    base_pos_cf = torch.randn(n, 3, device=DEVICE)
    base_ori_cf = _quat_from_euler(
        torch.empty(n).uniform_(-0.3, 0.3),
        torch.empty(n).uniform_(-0.3, 0.3),
        torch.empty(n).uniform_(-math.pi, math.pi),
    )
    # Existing placement (pure-yaw) the reference is currently on.
    yaw0 = torch.empty(n).uniform_(-math.pi, math.pi)
    quat0 = _quat_from_euler(torch.zeros(n), torch.zeros(n), yaw0)
    pos0 = torch.randn(n, 3, device=DEVICE)
    target_pos, target_quat = _lift_base(pos0, quat0, base_pos_cf, base_ori_cf)

    world_pos, world_quat = ego_placement(
        base_pos_cf, base_ori_cf, target_pos, target_quat
    )
    lifted_pos, lifted_quat = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)

    assert torch.allclose(lifted_pos, target_pos, atol=1e-5)
    assert torch.allclose(_wrap(_yaw_from_quat(lifted_quat) - _yaw_from_quat(target_quat)),
                          torch.zeros(n), atol=1e-5)


# ----------------------------------------------------------------------
# Per-clip anchor mask (ego_placement_masked).
# ----------------------------------------------------------------------

def _random_inputs(n, seed):
    torch.manual_seed(seed)
    base_pos_cf = torch.randn(n, 3, device=DEVICE) * 2.0
    base_ori_cf = _quat_from_euler(
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-math.pi, math.pi),
    )
    target_pos = torch.randn(n, 3, device=DEVICE) * 5.0
    target_quat = _quat_from_euler(
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-0.4, 0.4),
        torch.empty(n).uniform_(-math.pi, math.pi),
    )
    return base_pos_cf, base_ori_cf, target_pos, target_quat


def test_mask_named_constants_bit_patterns():
    """The named masks encode exactly the intended FREE DOF sets."""
    assert set(anchor_mask_free_names(_ANCHOR_MASK_LEGACY)) == {"x", "y", "z", "yaw"}
    assert set(anchor_mask_free_names(_ANCHOR_MASK_FLAT)) == {"x", "y", "yaw"}
    assert set(anchor_mask_free_names(_ANCHOR_MASK_STAIRS)) == {"y"}
    assert set(anchor_mask_free_names(_ANCHOR_MASK_SLOPE)) == {"x", "y"}
    assert (_ANCHOR_MASK_LEGACY, _ANCHOR_MASK_FLAT, _ANCHOR_MASK_STAIRS, _ANCHOR_MASK_SLOPE) == (39, 35, 2, 3)


def test_masked_legacy_equivalence():
    """Legacy mask + zero cell anchor reproduces ego_placement exactly (z-change guard)."""
    n = 32
    base_pos_cf, base_ori_cf, target_pos, target_quat = _random_inputs(n, seed=10)
    ref_pos, ref_quat = ego_placement(base_pos_cf, base_ori_cf, target_pos, target_quat)
    m_pos, m_quat = ego_placement_masked(
        base_pos_cf, base_ori_cf, target_pos, target_quat,
        anchor_mask=torch.full((n,), _ANCHOR_MASK_LEGACY, dtype=torch.uint8),
        cell_anchor_pos=torch.zeros(n, 3),
    )
    assert torch.allclose(m_pos, ref_pos, atol=1e-6)
    assert torch.allclose(m_quat, ref_quat, atol=1e-6)
    # And a scalar-int mask broadcasts identically.
    s_pos, s_quat = ego_placement_masked(
        base_pos_cf, base_ori_cf, target_pos, target_quat,
        anchor_mask=_ANCHOR_MASK_LEGACY, cell_anchor_pos=torch.zeros(n, 3),
    )
    assert torch.allclose(s_pos, ref_pos, atol=1e-6) and torch.allclose(s_quat, ref_quat, atol=1e-6)


def test_flat_mask_fixes_z_to_clip():
    """FLAT mask: xy/yaw hit the robot, but z is the clip's own base z (not robot z)."""
    n = 24
    base_pos_cf, base_ori_cf, target_pos, target_quat = _random_inputs(n, seed=11)
    world_pos, world_quat = ego_placement_masked(
        base_pos_cf, base_ori_cf, target_pos, target_quat,
        anchor_mask=_ANCHOR_MASK_FLAT, cell_anchor_pos=torch.zeros(n, 3),
    )
    lifted_pos, lifted_quat = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)
    # FREE x, y, yaw -> reproduce the robot's planar pose + heading.
    assert torch.allclose(lifted_pos[:, :2], target_pos[:, :2], atol=1e-5)
    assert torch.allclose(_wrap(_yaw_from_quat(lifted_quat) - _yaw_from_quat(target_quat)),
                          torch.zeros(n), atol=1e-5)
    # FIXED z (cell z = 0) -> lifted z is the clip's absolute base z, not the robot z.
    assert torch.allclose(lifted_pos[:, 2], base_pos_cf[:, 2], atol=1e-5)


def test_stairs_mask_pins_to_cell_except_y():
    """STAIRS mask: only y re-anchors to the robot; x/z/yaw pinned to the terrain cell."""
    n = 20
    base_pos_cf, base_ori_cf, target_pos, target_quat = _random_inputs(n, seed=12)
    cell_pos = torch.randn(n, 3, device=DEVICE) * 3.0
    cell_yaw = torch.empty(n).uniform_(-math.pi, math.pi)
    world_pos, world_quat = ego_placement_masked(
        base_pos_cf, base_ori_cf, target_pos, target_quat,
        anchor_mask=_ANCHOR_MASK_STAIRS, cell_anchor_pos=cell_pos, cell_anchor_yaw=cell_yaw,
    )
    lifted_pos, _ = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)
    # Placement heading is the fixed cell yaw; x/z origin is the cell anchor.
    assert torch.allclose(_wrap(_yaw_from_quat(world_quat) - cell_yaw), torch.zeros(n), atol=1e-5)
    assert torch.allclose(world_pos[:, 0], cell_pos[:, 0], atol=1e-6)
    assert torch.allclose(world_pos[:, 2], cell_pos[:, 2], atol=1e-6)
    # FREE y still lands on the robot y; x/z ride the clip motion off the cell.
    assert torch.allclose(lifted_pos[:, 1], target_pos[:, 1], atol=1e-5)
    assert torch.allclose(lifted_pos[:, 2], cell_pos[:, 2] + base_pos_cf[:, 2], atol=1e-5)


def test_masked_placement_is_pure_yaw_for_every_mask():
    """The returned quaternion is a ground frame (x==y==0) regardless of the mask."""
    n = 12
    base_pos_cf, base_ori_cf, target_pos, target_quat = _random_inputs(n, seed=13)
    for mask in (_ANCHOR_MASK_LEGACY, _ANCHOR_MASK_FLAT, _ANCHOR_MASK_STAIRS, _ANCHOR_MASK_SLOPE):
        _, world_quat = ego_placement_masked(
            base_pos_cf, base_ori_cf, target_pos, target_quat,
            anchor_mask=mask, cell_anchor_pos=torch.randn(n, 3),
            cell_anchor_yaw=torch.empty(n).uniform_(-math.pi, math.pi),
        )
        assert torch.allclose(world_quat[:, :2], torch.zeros(n, 2), atol=1e-6)
        assert torch.allclose(torch.linalg.norm(world_quat, dim=-1), torch.ones(n), atol=1e-6)


def test_free_roll_pitch_rejected():
    """Free roll/pitch is reserved (breaks the pure-yaw invariant) and must raise."""
    n = 4
    base_pos_cf, base_ori_cf, target_pos, target_quat = _random_inputs(n, seed=14)
    for bad in ("roll", "pitch"):
        mask = anchor_mask_from_free({"x", "y", "yaw", bad})
        try:
            ego_placement_masked(
                base_pos_cf, base_ori_cf, target_pos, target_quat,
                anchor_mask=mask, cell_anchor_pos=torch.zeros(n, 3),
            )
            assert False, f"expected free '{bad}' to raise"
        except ValueError:
            pass
