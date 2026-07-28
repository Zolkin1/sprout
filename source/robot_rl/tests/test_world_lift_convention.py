# Copyright 2026 Zachary Olkin. All rights reserved.

"""Quaternion-convention guards for the clip -> Isaac Lab seam.

Clip libraries store orientation scalar-FIRST (``{frame}:ori_w, ori_x, ori_y,
ori_z``) and :mod:`motion_gen.math_utils` is scalar-first, but Isaac Lab 3.0 is
scalar-LAST: ``body_quat_w`` returns ``(x, y, z, w)`` and
``write_root_pose_to_sim_index`` expects ``(x, y, z, w)``.  Feeding an Isaac Lab
quaternion into the scalar-first lift put the scalar in slot 0, which the sim
read as ``x`` -- a ~180 deg roll about X.  The robot spawned upside down
(``projected_gravity_b_z = +0.997`` where upright is ``-1.0``) from a reference
that was upright to within 10.6 deg, and the mis-ordered anchor also made
``anchor_inv_yaw`` throw the anchor yaw away, displacing the spawn in xy.

These tests pin the seam: the invariant (an upright reference must lift to an
upright world pose), the yaw the lift must preserve, and the rule that Isaac
Lab-side code may not bypass the adapter.
"""

from __future__ import annotations

import math
import pathlib
import re

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.world_lift_isaac import (
    lift_base_world_state,
    lift_position,
    to_wxyz,
    to_xyzw,
)

DEVICE = "cpu"

# The convention-bearing names: anything taking or returning a quaternion.  The
# anchor-MASK helpers are bitmask-only, so they are fine to import directly.
_ORDER_SENSITIVE = (
    "anchor_inv_yaw", "ego_placement", "ego_placement_masked", "lift_base_world_state",
    "lift_orientation", "lift_orientation_batched", "lift_position", "lift_position_batched",
    "lift_vector", "lift_vector_batched", "_yaw_from_quat",
)


def _grav_b_z(quat_xyzw: torch.Tensor) -> float:
    """Body-frame z of world gravity for an ``(x, y, z, w)`` quaternion (-1 = upright)."""
    x, y, z, w = quat_xyzw.reshape(4).tolist()
    # third row of R^T applied to (0, 0, -1) == -(R[2, 2])
    return -(1.0 - 2.0 * (x * x + y * y))


def _idx(nj: int) -> dict[str, torch.Tensor]:
    """Channel indices for a minimal clip layout: base pos, base ori (xyzw), vels, joints."""
    return {
        "pos": torch.tensor([0, 1, 2]),
        "ori": torch.tensor([3, 4, 5, 6]),          # xyzw, as _reference_world_indices builds it
        "lin_vel": torch.tensor([0, 1, 2]),
        "ang_vel": torch.tensor([3, 4, 5]),
        "joint_pos": torch.arange(7, 7 + nj),
        "joint_vel": torch.arange(6, 6 + nj),
    }


# --------------------------------------------------------------------- reorder

def test_reorder_round_trips_and_moves_the_scalar():
    q_xyzw = torch.tensor([[0.0088889, -0.04429148, -0.08012316, 0.99576074]])
    assert torch.allclose(to_xyzw(to_wxyz(q_xyzw)), q_xyzw)
    assert to_wxyz(q_xyzw)[0, 0].item() == pytest.approx(0.99576074)   # scalar first
    assert to_xyzw(to_wxyz(q_xyzw))[0, 3].item() == pytest.approx(0.99576074)   # ... and last again


# ------------------------------------------------ the invariant that was broken

def test_an_upright_reference_lifts_to_an_upright_world_pose():
    """The spawn-upside-down regression: a near-upright clip pose must stay upright.

    The stored quaternion below is a real clip frame (a 10.6 deg tilt).  Read in
    the wrong order it becomes a 171 deg roll, which is what reached the sim.
    """
    nj = 2
    y = torch.zeros(1, 7 + nj)
    dy = torch.zeros(1, 6 + nj)
    y[0, :3] = torch.tensor([0.3, -0.1, 0.78])
    # clip channels are scalar-FIRST on disk; idx["ori"] selects them scalar-last
    ori_wxyz = [0.99576074, 0.0088889, -0.04429148, -0.08012316]
    y[0, 3:7] = torch.tensor([ori_wxyz[1], ori_wxyz[2], ori_wxyz[3], ori_wxyz[0]])

    anchor = torch.zeros(1, 7)
    anchor[0, 6] = 1.0                                    # xyzw identity placement

    state = lift_base_world_state(y, dy, anchor, _idx(nj))
    quat = state[:, 3:7]
    assert quat[0, 3].abs().item() > 0.99, f"scalar must land in w, got {quat[0].tolist()}"
    assert _grav_b_z(quat) < -0.98, f"reference is upright; got grav_b_z={_grav_b_z(quat):+.3f}"


def test_the_lift_preserves_the_anchor_yaw():
    """``anchor_inv_yaw`` fed a mis-ordered quat returns a constant, discarding the yaw.

    That silently rotated every lifted position by a fixed 180 deg instead of the
    anchor's heading, so clips were placed off their terrain cell.
    """
    for yaw in (0.0, 0.7, -1.9):
        anchor_quat = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
        out = lift_position(torch.zeros(1, 3), anchor_quat, torch.tensor([[1.0, 0.0, 0.0]]))
        want = torch.tensor([[math.cos(yaw), math.sin(yaw), 0.0]])
        assert torch.allclose(out, want, atol=1e-5), \
            f"yaw={yaw}: lifted {out[0].tolist()}, expected {want[0].tolist()}"


def test_identity_anchor_is_a_pure_translation():
    anchor_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    pos = torch.tensor([[0.4, -0.2, 0.9]])
    out = lift_position(torch.tensor([[1.0, 2.0, 0.0]]), anchor_quat, pos)
    assert torch.allclose(out, pos + torch.tensor([[1.0, 2.0, 0.0]]), atol=1e-6)


# ------------------------------------------------------------------ import guard

def test_isaac_side_code_does_not_bypass_the_adapter():
    """Isaac Lab code must import the lift from ``world_lift_isaac``, not ``motion_gen``.

    Without this, the next direct ``from motion_gen.world_lift import lift_*``
    silently reintroduces the scalar-first/scalar-last mismatch -- a docstring
    cannot prevent that, a failing test can.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "robot_rl"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "world_lift_isaac.py" or "build/" in str(path):
            continue
        text = path.read_text()
        for match in re.finditer(r"from\s+motion_gen\.world_lift\s+import\s+\(?([^)\n]*(?:\n[^)]*)?)\)?",
                                 text):
            imported = {n.strip().rstrip(",") for n in match.group(1).replace("\n", ",").split(",")}
            bad = imported & set(_ORDER_SENSITIVE)
            if bad:
                offenders.append(f"{path.relative_to(root.parent)}: {sorted(bad)}")
    assert not offenders, (
        "these files import order-sensitive lift helpers directly from motion_gen.world_lift; "
        "import them from world_lift_isaac instead:\n  " + "\n  ".join(offenders)
    )
