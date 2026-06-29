# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the ego-centric (receding-horizon) clip re-anchor placement.

Pure tensor-level tests on
``robot_rl...traj_tracking.closed_loop_command.ego_placement`` (no IsaacLab, no
simulator).  ``ego_placement`` returns the per-env yaw+xy *placement* that puts a
clip's current-step base onto a target root pose.  The runtime then lifts the
clip-world base through that placement; for the pure-yaw placement this reduces to
``base_w = world_pos + R(world_quat) @ base_cf`` (position) and
``yaw(base_w) = yaw(world_quat) + yaw(base_cf)`` (heading) -- exactly what
``world_lift`` does (``anchor_inv_yaw(Q) == Q`` for pure-yaw ``Q``).  The tests
re-implement that lift independently (small local quaternion helpers) and assert
the lifted base reproduces the target's planar position + heading while keeping
the clip's natural height, that the returned quaternion is pure-yaw, and that the
re-anchor is a round-trip no-op when the target already equals the reference.
"""

from __future__ import annotations

import math

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.closed_loop_command import (
    ego_placement,
    _yaw_from_quat,
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


def test_lifted_base_reproduces_target_planar_pose():
    """Lifting the clip base through the returned placement hits the target xy + yaw."""
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
    ground_z = torch.empty(n, device=DEVICE).uniform_(-0.5, 0.5)

    world_pos, world_quat = ego_placement(base_pos_cf, base_ori_cf, target_pos, target_quat, ground_z)
    lifted_pos, lifted_quat = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)

    # Planar position matches the target root exactly.
    assert torch.allclose(lifted_pos[:, :2], target_pos[:, :2], atol=1e-5)
    # Heading matches the target root (full tilt is *not* matched -- planar only).
    assert torch.allclose(_wrap(_yaw_from_quat(lifted_quat) - _yaw_from_quat(target_quat)),
                          torch.zeros(n), atol=1e-5)


def test_height_kept_natural_not_snapped_to_target():
    """Lifted base z is ground_z + the clip's own base z, independent of target z."""
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
    ground_z = torch.empty(n, device=DEVICE).uniform_(-1.0, 1.0)

    world_pos, world_quat = ego_placement(base_pos_cf, base_ori_cf, target_pos, target_quat, ground_z)
    lifted_pos, _ = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)

    assert torch.allclose(world_pos[:, 2], ground_z, atol=1e-6)
    assert torch.allclose(lifted_pos[:, 2], ground_z + base_pos_cf[:, 2], atol=1e-5)


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
        torch.randn(n, 3), base_ori_cf, torch.randn(n, 3), target_quat, torch.zeros(n)
    )
    assert torch.allclose(world_quat[:, 0], torch.zeros(n), atol=1e-6)
    assert torch.allclose(world_quat[:, 1], torch.zeros(n), atol=1e-6)
    assert torch.allclose(torch.linalg.norm(world_quat, dim=-1), torch.ones(n), atol=1e-6)


def test_reanchor_roundtrip_is_planar_noop():
    """Re-anchoring to the reference's own current lifted pose preserves the planar lift.

    Build a target by lifting the base through an existing pure-yaw placement, then
    re-anchor onto that target: the new placement must reproduce the same planar
    position + heading (and natural height), i.e. re-planning while already on the
    reference does not move the planar reference.
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
        base_pos_cf, base_ori_cf, target_pos, target_quat, pos0[:, 2]
    )
    lifted_pos, lifted_quat = _lift_base(world_pos, world_quat, base_pos_cf, base_ori_cf)

    assert torch.allclose(lifted_pos[:, :2], target_pos[:, :2], atol=1e-5)
    assert torch.allclose(_wrap(_yaw_from_quat(lifted_quat) - _yaw_from_quat(target_quat)),
                          torch.zeros(n), atol=1e-5)
