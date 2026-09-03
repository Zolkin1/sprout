# Copyright 2026 Zachary Olkin. All rights reserved.

"""The MEASURED history arm: what it puts in a row, and what it refuses to invent.

``history_from="measured"`` fills the generator's ``N`` context rows with the robot's own
achieved state instead of the nodes the runtime served.  Three things have to hold, and none
of them is visible in a shape:

1. **The fallback columns are exactly the ones a robot cannot measure.**  The G1 has no
   odometry and no base linear-velocity estimate, so ``base_pose``'s position half and
   ``base_vel``'s linear half come from the served node.  Both runtimes carry that list --
   torch in ``robot_rl.flow_matching.window``, numpy here -- and a row built under one list
   and read under another is the same width as a correct one.
2. **The measured half really is measured.**  Built against the robot's own pose as the
   anchor, the position block is identically zero and the rot6d block is the robot's own
   tilt; that is the degenerate case that says the builder is reading the state it claims
   to.
3. **A row is all-or-nothing.**  Before the generator has served anything, three of the
   row's columns can come from nowhere, and a row that is part measurement and part zero is
   not one the model has ever seen.  Zeros go in and the null-history embedding covers it.

Artifact-free: everything under test is arithmetic that runs long before a graph is
consulted, so the policy is built attribute-by-attribute like
``test_window_block_agreement`` does.

The ``_np`` suffix follows ``quat_math_np`` / ``transforms_np``: this is the numpy runtime's
half, and ``source/robot_rl/tests/test_measured_history.py`` is the torch command's. The
basenames have to differ -- neither directory is a package, so pytest cannot collect two
files of one name in a single run.

Run: ``pytest transfer/deploy/tests/test_measured_history_np.py``
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.window_np import (  # noqa: E402
    MEASURED_HISTORY_FALLBACK,
    measured_history_fallback_spans,
    window_block_slices,
    window_dim,
)

#: The DEPLOYED generator's layout: no ``joint_vel``, no ``ee_pos``.  Stated here so the
#: assertions below are against the artifact that actually ships rather than the widest
#: layout the code can express -- and the ``ee_pos`` case gets its own test.
BLOCKS = ("base_pose", "base_vel", "joint_pos")
NJ, K = 29, 4


def _policy(blocks=BLOCKS, n_ee=K, history_from="measured"):
    """A bare policy carrying just what the measured-history path reads.

    Args:
        blocks: The window block selection both artifacts agree on.
        n_ee: ``target_num_extra_frames``.
        history_from: The arm to configure.

    Returns:
        An unloaded :class:`GenTrackerPolicy`.
    """
    pol = object.__new__(GenTrackerPolicy)
    pol.fm_enabled = True
    pol.history_from = history_from
    pol.reference_blocks = tuple(blocks)
    pol.ctx = SimpleNamespace(D=window_dim(NJ, n_ee, blocks), nj=NJ,
                              target_num_extra_frames=n_ee, reference_blocks=tuple(blocks))
    pol._isaac_to_fm = np.arange(NJ)
    pol._anchor_pos = np.zeros(3)
    pol._anchor_quat = np.array([0.0, 0.0, 0.0, 1.0])
    pol._prev_measured_node = None
    pol._prev_served_node = None
    pol._measured_fallback_spans = measured_history_fallback_spans(NJ, n_ee, blocks)
    return pol


def _state(pos=(1.0, 2.0, 0.8), quat=(0.0, 0.0, 0.0, 1.0), n_ee=K):
    """A minimal state dict for :meth:`GenTrackerPolicy._measured_history_node`.

    Args:
        pos: World base position.
        quat: World base orientation, xyzw.
        n_ee: End-effector frames to supply.

    Returns:
        The dict.
    """
    return {
        "root_pos": np.array(pos, dtype=np.float64),
        "root_quat": np.array(quat, dtype=np.float64),
        "root_lin_vel_w": np.array([0.7, -0.1, 0.02]),
        "root_ang_vel_w": np.array([0.03, -0.04, 0.11]),
        "joint_pos": np.linspace(-0.4, 0.4, NJ),
        "joint_vel": np.linspace(0.9, -0.9, NJ),
        "frame_pos": np.arange(3 * n_ee, dtype=np.float64).reshape(n_ee, 3),
        "command": np.array([0.8, 0.0, 0.0]),
    }


# ---------------------------------------------------------------------------------------
# 1. the fallback list, on both sides of the port
# ---------------------------------------------------------------------------------------
def test_the_two_runtimes_name_the_same_unmeasurable_columns():
    """The torch and numpy fallback lists must resolve to identical spans.

    Skips without IsaacLab, which is what ``robot_rl.flow_matching.window`` imports for its
    quaternion helpers -- the same ``importorskip`` guard ``transforms_np``'s parity test
    uses.  The numpy half is still covered unconditionally by the tests below.
    """
    torch_window = pytest.importorskip(
        "robot_rl.flow_matching.window",
        reason="needs isaaclab (not isaacsim) for the torch window module")
    assert tuple(torch_window.MEASURED_HISTORY_FALLBACK) == MEASURED_HISTORY_FALLBACK
    for blocks in (BLOCKS, ("base_pose", "base_vel", "joint_pos", "joint_vel", "ee_pos")):
        assert (torch_window.measured_history_fallback_spans(NJ, K, blocks)
                == measured_history_fallback_spans(NJ, K, blocks)), blocks


def test_the_fallback_spans_are_the_position_and_linear_velocity_halves():
    """The spans must land on the halves they are named for, not on whole blocks.

    ``base_pose`` is 9 wide and only its first 3 are unmeasurable; ``base_vel`` is 6 and
    only its first 3.  Falling back over a whole block would overwrite the robot's own
    orientation and angular velocity with the prediction -- leaving an arm that is barely
    measured at all, at exactly the same width.
    """
    slices = window_block_slices(NJ, K, BLOCKS)
    spans = measured_history_fallback_spans(NJ, K, BLOCKS)
    assert spans == ((slices["base_pose"].start, slices["base_pose"].start + 3),
                     (slices["base_vel"].start, slices["base_vel"].start + 3))
    assert sum(b - a for a, b in spans) == 6


def test_a_layout_without_base_vel_has_nothing_to_fall_back_on():
    """A block that is not in the selection contributes no span.

    Indexing a fallback by its position in the FULL layout would write into ``joint_pos``
    on a checkpoint that dropped ``base_vel`` -- in range, so nothing would raise.
    """
    spans = measured_history_fallback_spans(NJ, K, ("base_pose", "joint_pos"))
    assert spans == ((0, 3),)


# ---------------------------------------------------------------------------------------
# 2. the measured half really is measured
# ---------------------------------------------------------------------------------------
def test_against_its_own_pose_the_position_block_is_zero_and_the_rotation_is_the_tilt():
    """The degenerate case that proves the builder reads the state it claims to.

    With the anchor set to the robot's own pose, ``p_meas - p_anchor`` is identically zero
    and ``R_yaw^T R_meas`` is the robot's tilt with its heading divided out.  Checked at a
    non-trivial orientation -- at identity every rot6d is the identity 6-vector and the test
    would pass on a builder that ignored the quaternion entirely.
    """
    roll = 0.21
    # A pure roll about x, so the yaw frame is the identity and R_yaw^T R_meas == R_meas.
    quat = (math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2))
    pol = _policy()
    st = _state(pos=(3.5, -1.25, 0.83), quat=quat)
    pol._anchor_pos = st["root_pos"].copy()
    pol._anchor_quat = st["root_quat"].copy()

    node = pol._measured_history_node(st)
    slices = window_block_slices(NJ, K, BLOCKS)
    bp = node[0, slices["base_pose"]]
    np.testing.assert_allclose(bp[0:3], 0.0, atol=1e-15)
    # rot6d = the first two COLUMNS of R_x(roll).
    np.testing.assert_allclose(bp[3:9], [1.0, 0.0, 0.0,
                                         0.0, math.cos(roll), math.sin(roll)], atol=1e-12)
    np.testing.assert_allclose(node[0, slices["joint_pos"]], st["joint_pos"], atol=1e-15)


def test_the_position_block_is_the_displacement_since_the_anchor():
    """Under the ``plan`` anchor the position half is travel since the re-plan.

    This is what makes the arm deployable in principle: the number needed is a
    displacement over at most ``fm_update_every`` control steps, not a global base position.
    Measured in the anchor's yaw frame, so a heading offset rotates it -- checked at a
    90-degree anchor yaw, where a purely forward world displacement must read as lateral.
    """
    pol = _policy()
    pol._anchor_pos = np.array([1.0, 2.0, 0.8])
    pol._anchor_quat = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    st = _state(pos=(1.3, 2.0, 0.8))          # +0.30 m along world +x

    node = pol._measured_history_node(st)
    bp = node[0, window_block_slices(NJ, K, BLOCKS)["base_pose"]]
    np.testing.assert_allclose(bp[0:3], [0.0, -0.30, 0.0], atol=1e-12)


def test_an_ee_layout_refuses_to_run_without_the_measured_frames():
    """A node carrying ``ee_pos`` must be handed ``frame_pos``, not zero-filled.

    A zeroed posture block is a plausible number -- it is what a robot folded in half would
    produce -- so this is the one case where raising is strictly better than defaulting.
    """
    blocks = ("base_pose", "base_vel", "joint_pos", "ee_pos")
    pol = _policy(blocks=blocks)
    st = _state()
    del st["frame_pos"]
    with pytest.raises(KeyError):
        pol._measured_history_node(st)


# ---------------------------------------------------------------------------------------
# 3. a row is all-or-nothing
# ---------------------------------------------------------------------------------------
def test_nothing_served_yet_gives_an_all_zero_row_not_a_half_measured_one():
    """Before the first plan, the row must be entirely zeros.

    Three of its columns can only come from a served node.  Emitting the measured 38 beside
    3 zeros would hand the model a row it has never seen, with ``history_null`` off -- the
    same defect ``advance_dormant_step`` was stripped down to avoid.
    """
    pol = _policy()
    pol._prev_measured_node = np.arange(pol.ctx.D, dtype=np.float64).reshape(1, -1)
    pol._prev_served_node = None                       # nothing served yet
    row = pol._make_snapshot(_state())["node"]
    assert row.shape == (1, pol.ctx.D)
    np.testing.assert_array_equal(row, 0.0)


def test_the_row_is_measured_everywhere_except_the_fallback_columns():
    """The substitution must hit exactly the fallback spans and nothing else."""
    pol = _policy()
    d = pol.ctx.D
    meas = np.full((1, d), 2.0)
    served = np.full((1, d), 9.0)
    pol._prev_measured_node, pol._prev_served_node = meas, served

    row = pol._make_snapshot(_state())["node"]
    expect = np.full((1, d), 2.0)
    for a, b in pol._measured_fallback_spans:
        expect[:, a:b] = 9.0
    np.testing.assert_array_equal(row, expect)
    assert int((row == 9.0).sum()) == 6, "exactly the two 3-wide halves come from the plan"


def test_the_generator_arm_is_untouched_by_any_of_this():
    """``history_from='generator'`` must still push the served node verbatim.

    The default arm is what deploys; a measured-history change that quietly altered it
    would be a regression in the only configuration that runs on a robot.
    """
    pol = _policy(history_from="generator")
    served = np.arange(pol.ctx.D, dtype=np.float64).reshape(1, -1)
    pol._prev_served_node = served
    pol._prev_measured_node = np.full((1, pol.ctx.D), -1.0)     # must be ignored
    np.testing.assert_array_equal(pol._make_snapshot(_state())["node"], served)


def test_an_unknown_arm_is_refused_at_construction():
    """A typo must not fall through to the closed loop.

    ``history_from='clip'`` in particular is a real arm on the Isaac side, so it is the
    plausible thing to type here -- and silently getting the generator arm instead would
    make a teacher-forcing run look like it produced closed-loop numbers.
    """
    for bad in ("clip", "Measured", "robot"):
        with pytest.raises(ValueError, match="history_from"):
            GenTrackerPolicy("t.onnx", "f.onnx", history_from=bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
