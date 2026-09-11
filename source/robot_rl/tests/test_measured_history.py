# Copyright 2026 Zachary Olkin. All rights reserved.

"""``history_from="measured"``: the row the generator eats when it eats reality.

``FmTrackingCommand`` has three history producers.  Two say "the reference applied at that
step" -- the node it served (closed loop) and the clip's node (teacher forcing).  This third
one says "where the robot actually got to", which is a different kind of quantity and the
reason the arm exists: it is real, it is deployable in principle, and it is off-distribution
against a model fitted on reference rows.

Pure arithmetic, so the tests drive the real methods on a bare instance carrying only the
buffers each reads -- the same pattern ``test_fm_tracking_lookahead`` uses, and for the same
reason: the arithmetic under test is then the shipped arithmetic.

What is pinned here, in order of how quietly it would fail:

* **The push pairs the two stashes at the SAME step.**  ``_prev_measured_node`` and
  ``_prev_served_node`` both describe step ``t-1`` at push time, and the fallback columns are
  taken from the latter.  Substituting at stash time instead would pair a measured row for
  step ``t`` with a served node for ``t-1`` -- off by one control step of travel, in the
  three columns that carry where the robot is.
* **A row is all-or-nothing.**  Part measurement and part zero is not a row the model has
  ever seen, and it would go in with ``history_null`` off.
* **A reset does not leak the previous tile.**  The measured stash is a world pose; pushed
  across a reset it is a metres-wide step in ``base_pose`` that reads as a fall.
* **The frame is the PLAN's**, not the robot's, which is what every history row uses and
  what makes the position half a short-horizon displacement rather than an odometry reading.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp", exc_type=ImportError)
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
wp.init()

from robot_rl.flow_matching.window import (  # noqa: E402
    MEASURED_HISTORY_FALLBACK,
    measured_history_fallback_spans,
    window_block_slices,
    window_dim,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402
    FmTrackingCommand,
)

N_ENVS, NJ = 3, 6
#: The DEPLOYED generator's block selection: no ``joint_vel``, no ``ee_pos``.
BLOCKS = ("base_pose", "base_vel", "joint_pos")
D = window_dim(NJ, 0, BLOCKS)
SLICES = window_block_slices(NJ, 0, BLOCKS)


class _BareFmCommand(FmTrackingCommand):
    """``FmTrackingCommand`` with ``device`` / ``num_envs`` as plain attributes.

    The base class exposes both as read-only properties off the env; these shadow them so a
    bare instance needs no simulator.  Every method under test is inherited unchanged.
    """

    device = torch.device("cpu")
    num_envs = N_ENVS
    _debug_vis_handle = None


def _stub(history_from: str = "measured", root_pos=None, root_quat=None):
    """A bare command carrying only what the measured-history path reads.

    Args:
        history_from: The arm to configure.
        root_pos: ``[N, 3]`` measured base positions, world frame.
        root_quat: ``[N, 4]`` measured base orientations, xyzw world frame.

    Returns:
        The bare :class:`FmTrackingCommand`.
    """
    cmd = object.__new__(_BareFmCommand)
    cmd.cfg = SimpleNamespace(history_from=history_from, fm_update_every=12,
                              reference_source="fm")
    cmd.ctx = SimpleNamespace(N=4, F=62, D=D, nj=NJ, num_extra_frames=0,
                              frame_names=[], reference_blocks=BLOCKS)
    cmd._fm_blocks = BLOCKS
    cmd._measured_ee_idx = None
    cmd._measured_fallback_spans = measured_history_fallback_spans(NJ, 0, BLOCKS)
    cmd._arange_n = torch.arange(N_ENVS)
    cmd._pred_window = torch.zeros(N_ENVS, 62, D)
    cmd._pred_valid = torch.ones(N_ENVS, dtype=torch.bool)
    cmd._prev_served_node = None
    cmd._prev_measured_node = None
    cmd._prev_clip_node = None
    cmd._clip_node_valid = torch.zeros(N_ENVS, dtype=torch.bool)
    cmd._measured_node_valid = torch.zeros(N_ENVS, dtype=torch.bool)
    # The PLAN anchor -- the pose at the last re-plan, which is the frame every history row
    # lives in. Deliberately NOT the robot's current pose; several tests turn on the two
    # being different.
    cmd._anchor_pos = torch.zeros(N_ENVS, 3)
    cmd._anchor_quat = torch.zeros(N_ENVS, 4)
    cmd._anchor_quat[:, 3] = 1.0

    rp = torch.zeros(N_ENVS, 3) if root_pos is None else root_pos
    rq = torch.zeros(N_ENVS, 4) if root_quat is None else root_quat
    if root_quat is None:
        rq[:, 3] = 1.0
    cmd.robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=wp.from_torch(rp.contiguous()),
        root_quat_w=wp.from_torch(rq.contiguous()),
        root_link_lin_vel_w=wp.from_torch(
            torch.full((N_ENVS, 3), 0.5).contiguous()),
        root_link_ang_vel_w=wp.from_torch(
            torch.full((N_ENVS, 3), 0.1).contiguous()),
        joint_pos=wp.from_torch(
            torch.arange(NJ).float().repeat(N_ENVS, 1).contiguous() * 0.1),
        joint_vel=wp.from_torch(torch.zeros(N_ENVS, NJ).contiguous()),
        body_pos_w=wp.from_torch(torch.zeros(N_ENVS, 4, 3).contiguous()),
    ))
    # Captured on the way past, so a test can assert the ring content without a real
    # BatchedHistory (whose own behaviour is covered by test_history_null).
    cmd._history = SimpleNamespace(pushed=[], append=lambda row: cmd._history.pushed.append(row))
    return cmd


def test_the_position_half_is_travel_since_the_plan_anchor_not_a_world_pose():
    """The row is built in the PLAN's yaw frame, so its position half is a displacement.

    That is the whole reason this arm is deployable in principle: the quantity needed is
    the base displacement over at most ``fm_update_every`` control steps, which leg odometry
    can supply, rather than a global base position, which nothing on the robot can.

    A world pose would read as ~metres here; a displacement reads as the 0.3 m stepped.
    """
    pos = torch.tensor([[5.3, 2.0, 0.8], [5.3, 2.0, 0.8], [5.3, 2.0, 0.8]])
    cmd = _stub(root_pos=pos)
    cmd._anchor_pos = torch.tensor([[5.0, 2.0, 0.8]] * N_ENVS)

    node = FmTrackingCommand._measured_history_node(cmd)
    assert node.shape == (N_ENVS, D)
    torch.testing.assert_close(node[:, SLICES["base_pose"]][:, 0:3],
                               torch.tensor([[0.3, 0.0, 0.0]] * N_ENVS), atol=1e-6, rtol=0)


def test_the_joint_block_is_the_measured_joints_verbatim():
    """``joint_pos`` is frame-free, so it must arrive unmodified and unpermuted.

    ``_post_init`` refuses a checkpoint whose joint order differs from the robot's, so there
    is no permutation to apply here -- and applying one anyway would scramble every joint
    with no shape change to reveal it.
    """
    cmd = _stub()
    node = FmTrackingCommand._measured_history_node(cmd)
    torch.testing.assert_close(node[:, SLICES["joint_pos"]],
                               torch.arange(NJ).float().repeat(N_ENVS, 1) * 0.1)


def test_the_push_takes_the_fallback_columns_from_the_node_served_at_the_SAME_step():
    """The two stashes both describe the previous step at push time.

    Pinning the pairing, not just the substitution: the values are chosen so that taking
    the fallback from the wrong stash would be visible.  ``_prev_measured_node`` and
    ``_prev_served_node`` are what the previous step left behind; the row pushed now must
    be the former with exactly the fallback spans replaced by the latter.
    """
    cmd = _stub()
    cmd._prev_measured_node = torch.full((N_ENVS, D), 2.0)
    cmd._prev_served_node = torch.full((N_ENVS, D), 9.0)
    cmd._measured_node_valid = torch.ones(N_ENVS, dtype=torch.bool)

    FmTrackingCommand._push_served_node(cmd)
    row = cmd._history.pushed[-1]["node"]
    expect = torch.full((N_ENVS, D), 2.0)
    for a, b in cmd._measured_fallback_spans:
        expect[:, a:b] = 9.0
    torch.testing.assert_close(row, expect)
    assert int((row == 9.0).sum()) == 6 * N_ENVS, "two 3-wide halves per env, and no more"


def test_the_push_stashes_the_current_step_for_next_time():
    """After the push, the stash must hold a row describing THIS step.

    Without the one-step stash the ring would carry rows describing the step they were
    pushed on, which is one step newer than the dataset's convention (the history block is
    the N rows BEFORE node 0).
    """
    cmd = _stub(root_pos=torch.tensor([[1.0, 0.0, 0.8]] * N_ENVS))
    cmd._anchor_pos = torch.zeros(N_ENVS, 3)
    assert cmd._prev_measured_node is None
    FmTrackingCommand._push_served_node(cmd)
    assert cmd._prev_measured_node is not None
    torch.testing.assert_close(cmd._prev_measured_node[:, 0], torch.ones(N_ENVS),
                               atol=1e-6, rtol=0)


def test_a_row_with_nothing_served_yet_is_all_zeros():
    """Part measurement and part zero is not a row the model has ever seen.

    Three of the row's columns can come only from a served node, so before the first plan
    the whole row goes in as zeros and the learned ``null_history`` embedding covers it --
    the same contract the other two arms follow.
    """
    cmd = _stub()
    cmd._prev_measured_node = torch.full((N_ENVS, D), 2.0)
    cmd._prev_served_node = None
    FmTrackingCommand._push_served_node(cmd)
    torch.testing.assert_close(cmd._history.pushed[-1]["node"], torch.zeros(N_ENVS, D))


def test_an_env_whose_plan_is_not_live_gets_a_zero_row():
    """``_pred_valid`` gates the measured row too, per env.

    It has to: the fallback columns come from that env's served node, so an env with no
    live plan has no way to fill them even though its own state is perfectly available.
    """
    cmd = _stub()
    cmd._prev_measured_node = torch.full((N_ENVS, D), 2.0)
    cmd._prev_served_node = torch.full((N_ENVS, D), 9.0)
    cmd._measured_node_valid = torch.ones(N_ENVS, dtype=torch.bool)
    cmd._pred_valid = torch.tensor([True, False, True])

    FmTrackingCommand._push_served_node(cmd)
    row = cmd._history.pushed[-1]["node"]
    torch.testing.assert_close(row[1], torch.zeros(D))
    assert not bool((row[0] == 0).all()) and not bool((row[2] == 0).all())


def test_a_just_reset_env_does_not_push_the_previous_tiles_pose():
    """The measured stash is invalidated on reset, per env.

    It is a world pose taken on the previous tile: pushed across a reset it is a
    metres-wide step in ``base_pose``, which reads as a fall rather than as a reset. The
    ring's own null-masking would cover it, but leaving stale data to be masked is the kind
    of thing that stops being masked the moment the warmup changes.
    """
    cmd = _stub()
    cmd._prev_measured_node = torch.full((N_ENVS, D), 2.0)
    cmd._prev_served_node = torch.full((N_ENVS, D), 9.0)
    cmd._measured_node_valid = torch.ones(N_ENVS, dtype=torch.bool)
    # What _snapshot_and_infer does for the envs that just reset.
    just_reset = torch.tensor([False, True, False])
    cmd._measured_node_valid = cmd._measured_node_valid & ~just_reset

    FmTrackingCommand._push_served_node(cmd)
    row = cmd._history.pushed[-1]["node"]
    torch.testing.assert_close(row[1], torch.zeros(D))


def test_the_generator_arm_still_pushes_the_served_node_verbatim():
    """The default arm is the one that deploys and must be untouched by all of this."""
    cmd = _stub(history_from="generator")
    served = torch.arange(N_ENVS * D, dtype=torch.float32).reshape(N_ENVS, D)
    cmd._prev_served_node = served
    cmd._prev_measured_node = torch.full((N_ENVS, D), -1.0)      # must be ignored
    FmTrackingCommand._push_served_node(cmd)
    torch.testing.assert_close(cmd._history.pushed[-1]["node"], served)
    assert cmd._prev_measured_node.eq(-1.0).all(), "the generator arm must not stash"


def test_the_fallback_list_names_halves_and_only_halves():
    """The unmeasurable columns are the position and linear-velocity HALVES.

    Falling back over the whole parent blocks would replace the robot's own orientation and
    angular velocity with the prediction too -- an arm that is barely measured, at exactly
    the same width. Those two blocks are also the ones the gap probe measures as moving
    LEAST, so getting this wrong would discard the signal and keep the noise.
    """
    assert MEASURED_HISTORY_FALLBACK == ("base_pos", "base_lin_vel")
    spans = measured_history_fallback_spans(NJ, 0, BLOCKS)
    assert spans == ((SLICES["base_pose"].start, SLICES["base_pose"].start + 3),
                     (SLICES["base_vel"].start, SLICES["base_vel"].start + 3))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
