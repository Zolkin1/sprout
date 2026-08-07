# Copyright 2026 Zachary Olkin. All rights reserved.

"""Lifting a predicted reference WINDOW back to world coordinates, for diagnostics.

Why this file exists: the MuJoCo runner's viewer ribbon and ``--log`` sidecars kept reading
a retired world-frame ``_pred`` dict (``policy._pred["root_pos"]``) long after ``_pred``
became a packed ``[1, F, D]`` reference window. Nothing caught it because ``draw_viz``
returns early when there is no viewer, so every headless run -- including every survival
sweep -- skipped the code entirely; it only surfaced when a human opened the viewer.

The lift is the one place off-sim that turns a window back into a world coordinate, and it
must agree with ``flow_matching.window.window_to_world`` exactly, so the position and
orientation halves are both checked against that function rather than against local algebra.
Those checks need torch + ``robot_rl`` and skip without them; the pure-numpy ones do not.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import GenTrackerPolicy, wxyz_to_xyzw, xyzw_to_wxyz  # noqa: E402
from deploy.quat_math_np import (  # noqa: E402
    quat_box_minus, quat_box_plus, quat_from_axis_angle, quat_unique,
)
from deploy.window_np import matrix_from_quat, window_dim, world_to_window  # noqa: E402

NJ, K, F = 29, 4, 7
SLIM = ("base_pose", "base_vel", "joint_pos")


def _unit(a: np.ndarray) -> np.ndarray:
    """Normalise along the last axis."""
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _rot_err(got: np.ndarray, want: np.ndarray) -> float:
    """Largest elementwise difference between two quaternion sets AS ROTATIONS.

    ``q`` and ``-q`` are the same rotation, and the lift returns whichever sign
    ``quat_from_matrix`` produces, so raw components cannot be compared. (The old log-map
    lift went through ``quat_box_plus``, which sign-normalised, and could be.)
    """
    return float(np.abs(matrix_from_quat(got) - matrix_from_quat(want)).max())


def _policy(pred: np.ndarray, blocks, valid: bool = True, anchor=None,
            cursor: int = 0) -> GenTrackerPolicy:
    """A bare policy carrying just what the lift accessors read.

    Args:
        pred: The ``[1, F, D]`` predicted window to serve.
        blocks: The block selection ``pred`` is packed with.
        valid: Whether a live prediction exists.
        anchor: ``(pos, quat)`` the window is expressed relative to -- the pose captured at
            the re-plan. Defaults to the identity anchor.
        cursor: ``_pred_idx``, the node being served.

    Returns:
        An unloaded :class:`GenTrackerPolicy` with the serving state populated.
    """
    pol = object.__new__(GenTrackerPolicy)
    pol._pred = pred
    pol._pred_valid = valid
    pol._pred_idx = cursor
    pos, quat = anchor if anchor is not None else (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    pol._anchor_pos = np.asarray(pos, dtype=np.float64)
    pol._anchor_quat = np.asarray(quat, dtype=np.float64)
    pol.reference_blocks = tuple(blocks)
    pol._fm_to_isaac = np.arange(NJ)
    pol.ctx = SimpleNamespace(target_num_extra_frames=K, F=pred.shape[1], D=pred.shape[2])
    return pol


def _case(seed: int = 0, blocks=SLIM):
    """A random anchor + world reference path, and the window that encodes it."""
    rng = np.random.default_rng(seed)
    anchor_p, anchor_q = rng.normal(size=3), _unit(rng.normal(size=4))
    ref_p, ref_q = rng.normal(size=(F, 3)), _unit(rng.normal(size=(F, 4)))
    jp = rng.uniform(-1.0, 1.0, size=(F, NJ))
    nodes = world_to_window(
        anchor_p[None, :], anchor_q[None, :], ref_p, ref_q,
        rng.normal(size=(F, 3)), rng.normal(size=(F, 3)),
        jp, rng.normal(size=(F, NJ)), rng.normal(size=(F, K, 3)), blocks=blocks,
    )
    return anchor_p, anchor_q, ref_p, ref_q, jp, nodes[None, ...]


# ---------------------------------------------------------------------------
# The numpy exp/log map pair the orientation lift rests on.
# ---------------------------------------------------------------------------
def test_box_plus_inverts_box_minus():
    """``q1 (-) q2`` then ``(+) q2`` returns to ``q1``, up to the documented sign."""
    rng = np.random.default_rng(11)
    q1, q2 = _unit(rng.normal(size=(64, 4))), _unit(rng.normal(size=(64, 4)))
    back = quat_box_plus(q2, quat_box_minus(q1, q2))
    assert np.abs(back - quat_unique(q1)).max() < 1e-12


def test_box_plus_of_zero_is_identity_not_a_divide_by_zero():
    """A zero increment is the common case at convergence; it must not produce NaN."""
    q = _unit(np.array([[0.1, -0.2, 0.3, 0.9]]))
    out = quat_box_plus(q, np.zeros((1, 3)))
    assert np.isfinite(out).all()
    assert np.abs(out - quat_unique(q)).max() < 1e-9


def test_exp_map_matches_isaaclab_quat_box_plus():
    """Parity against the real isaaclab op, since a convention slip rotates a reference.

    Skips without isaaclab -- the numpy port is the thing that must run torch-free, not the
    oracle it is checked against.
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    iml = pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
    rng = np.random.default_rng(12)
    q, delta = _unit(rng.normal(size=(32, 4))), rng.normal(size=(32, 3)) * 0.4
    ref = iml.quat_box_plus(torch.as_tensor(q, dtype=torch.float64),
                            torch.as_tensor(delta, dtype=torch.float64)).numpy()
    assert np.abs(quat_box_plus(q, delta) - ref).max() < 1e-10


def test_quat_from_axis_angle_is_unit_norm():
    """A non-unit quaternion silently corrupts every downstream rotation."""
    rng = np.random.default_rng(13)
    q = quat_from_axis_angle(rng.normal(size=(48, 3)) * 2.0)
    assert np.abs(np.linalg.norm(q, axis=-1) - 1.0).max() < 1e-12


# ---------------------------------------------------------------------------
# The lift itself.
# ---------------------------------------------------------------------------
def test_lift_recovers_the_world_reference_pose_exactly():
    """World pose -> window -> world pose, to float64.

    Not an approximation: ``base_pose`` is a rotate-and-translate plus an exp/log pair, so
    any real error means the convention drifted.
    """
    anchor_p, anchor_q, ref_p, ref_q, _, pred = _case(0)
    pol = _policy(pred, SLIM, anchor=(anchor_p, anchor_q))
    got_p, got_q = pol.predicted_base_pose_w()
    assert got_p.shape == (F, 3) and got_q.shape == (F, 4)
    assert np.abs(got_p - ref_p).max() < 1e-12
    assert _rot_err(got_q, ref_q) < 1e-12


def test_root_path_is_the_position_half_of_the_pose():
    """The narrow accessor must not drift from the one that returns both halves."""
    anchor_p, anchor_q, _, _, _, pred = _case(1)
    pol = _policy(pred, SLIM, anchor=(anchor_p, anchor_q))
    assert np.array_equal(pol.predicted_root_path_w(), pol.predicted_base_pose_w()[0])


def test_joint_pos_is_served_verbatim_and_frame_invariant():
    """Joints need no lift, so the anchor must not touch them."""
    _, _, _, _, jp, pred = _case(2)
    pol = _policy(pred, SLIM)
    assert np.abs(pol.predicted_joint_pos() - jp).max() < 1e-12


def test_lift_matches_the_torch_window_to_world_it_mirrors():
    """The authoritative inverse lives on the Isaac side; numpy must equal it, not merely be
    self-consistent. This is the check that catches a convention change made over there.
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    win = pytest.importorskip("robot_rl.flow_matching.window", exc_type=ImportError)
    anchor_p, anchor_q, _, _, _, pred = _case(3)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    out = win.window_to_world(t(pred[0]), t(anchor_p), t(anchor_q), NJ, K, blocks=SLIM)
    got_p, got_q = _policy(pred, SLIM, anchor=(anchor_p, anchor_q)).predicted_base_pose_w()
    assert np.abs(got_p - out["base_pos_w"].numpy()).max() < 1e-12
    assert _rot_err(got_q, out["base_quat_w"].numpy()) < 1e-12


# ---------------------------------------------------------------------------
# Degenerate selections and warmup.
# ---------------------------------------------------------------------------
def test_lift_is_none_before_the_first_prediction():
    """The viewer draws nothing rather than a ribbon of zeros during warmup."""
    pol = _policy(np.zeros((1, F, window_dim(NJ, K, SLIM))), SLIM, valid=False)
    assert pol.predicted_base_pose_w() is None
    assert pol.predicted_root_path_w() is None
    assert pol.predicted_joint_pos() is None


def test_lift_is_none_without_a_base_pose_block():
    """A selection with no anchor-referencing block has no world pose to recover.

    ``joint_pos`` alone is frame-invariant, so the honest answer is None -- not the first
    three joint angles reinterpreted as a position.
    """
    pol = _policy(np.zeros((1, F, NJ)), ("joint_pos",))
    assert pol.predicted_base_pose_w() is None
    assert pol.predicted_joint_pos() is not None       # this one IS available


def test_joint_pos_is_none_when_the_window_drops_it():
    """Symmetric to the above: FK of the predicted joints needs the block to exist."""
    anchor_p, anchor_q, _, _, _, pred = _case(4, blocks=("base_pose", "base_vel"))
    pol = _policy(pred, ("base_pose", "base_vel"), anchor=(anchor_p, anchor_q))
    assert pol.predicted_joint_pos() is None
    assert pol.predicted_base_pose_w() is not None


@pytest.mark.parametrize("blocks", [
    ("base_pose",),
    ("base_pose", "joint_pos"),
    ("base_pose", "base_vel", "joint_pos"),
    ("base_pose", "base_vel", "joint_pos", "joint_vel", "ee_pos"),
])
def test_lift_finds_its_blocks_at_every_selection_width(blocks):
    """``base_pose`` leads every selection today, but the slice must come from the layout.

    Hardcoding ``[:, 0:6]`` happens to work and would break silently the moment a block is
    ordered ahead of it, so this pins the lookup rather than the offset.
    """
    anchor_p, anchor_q, ref_p, ref_q, jp, pred = _case(5, blocks=blocks)
    pol = _policy(pred, blocks, anchor=(anchor_p, anchor_q))
    got_p, got_q = pol.predicted_base_pose_w()
    assert np.abs(got_p - ref_p).max() < 1e-12
    assert _rot_err(got_q, ref_q) < 1e-12
    if "joint_pos" in blocks:
        assert np.abs(pol.predicted_joint_pos() - jp).max() < 1e-12


# ---------------------------------------------------------------------------
# The quaternion order swap the MuJoCo FK hand-off needs.
# ---------------------------------------------------------------------------
def test_quat_order_helpers_round_trip():
    """``qpos[3:7]`` is wxyz and every FM quaternion is xyzw; the swap must be an involution."""
    rng = np.random.default_rng(14)
    q = _unit(rng.normal(size=4))
    assert np.abs(wxyz_to_xyzw(xyzw_to_wxyz(q)) - q).max() < 1e-15
    assert np.abs(xyzw_to_wxyz(q) - np.array([q[3], q[0], q[1], q[2]])).max() == 0.0


if __name__ == "__main__":
    test_box_plus_inverts_box_minus()
    test_box_plus_of_zero_is_identity_not_a_divide_by_zero()
    test_quat_from_axis_angle_is_unit_norm()
    test_lift_recovers_the_world_reference_pose_exactly()
    test_root_path_is_the_position_half_of_the_pose()
    test_joint_pos_is_served_verbatim_and_frame_invariant()
    test_lift_is_none_before_the_first_prediction()
    test_lift_is_none_without_a_base_pose_block()
    test_quat_order_helpers_round_trip()
    print("PREDICTED-WINDOW LIFT CHECKS PASSED")
