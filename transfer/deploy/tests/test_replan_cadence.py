# Copyright 2026 Zachary Olkin. All rights reserved.

"""Receding-horizon serving at ``fm_update_every > 1``, matched to the Isaac side.

The deploy runtime used to refuse ``K != 1`` outright. It now mirrors
``FmTrackingCommand._snapshot_and_infer``, where the step ORDER is the algorithm: push the
node served last step, decide whether to re-plan, otherwise advance the cursor, re-plan
(resetting cursor + anchor), then stash the node served this step re-anchored onto the robot
as it is now.

What these tests pin, in rough order of how badly getting it wrong would hurt:

* the cursor advances exactly ``K-1`` times between re-plans and resets on each one;
* the served nodes come from the CURSOR, not from the start of the window;
* a stale anchor is corrected -- serving a plan ``k`` steps late must not shear the window
  by the distance travelled since the re-plan;
* the history row is the node served LAST step, in that step's frame;
* the horizon rule is ``F >= max_offset + K``, so K=12 fits F=62 exactly and K=13 does not;
* K=1 behaviour is unchanged (the re-anchor is an identity there).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.history import BatchedHistory  # noqa: E402
from deploy.window_np import reanchor_window, window_dim, world_to_window  # noqa: E402

NJ, EE, F, N = 29, 4, 62, 4
SLIM = ("base_pose", "base_vel", "joint_pos")
D = window_dim(NJ, EE, SLIM)
OFFSETS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def _unit(a):
    """Normalise along the last axis."""
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _state(pos, quat, cmd=(0.5, 0.0, 0.0)):
    """The subset of the runner's state dict the generator path reads."""
    return {"root_pos": np.asarray(pos, dtype=np.float64),
            "root_quat": np.asarray(quat, dtype=np.float64),
            "command": np.asarray(cmd, dtype=np.float64),
            "depth": np.zeros((2, 780)), "height_scan": np.zeros(780)}


class _FakePolicy(GenTrackerPolicy):
    """A ``GenTrackerPolicy`` whose FM sample is a recorded, deterministic stand-in.

    Everything on the serving path -- cursor, cadence, anchor, stash, re-anchor -- is real;
    only ``_run_fm``'s network call is replaced, so the ONNX sessions are not needed. Each
    plan is a distinct constant per re-plan, which is what lets a test tell "served node 3 of
    plan 2" from "node 0 of plan 3".
    """

    def __init__(self, k: int, offsets=OFFSETS, f: int = F):
        """Build the fake with re-plan period ``k`` and window ``offsets``."""
        self.fm_enabled = True
        self.fm_update_every = k
        # The learned null-history embedding, as every real fm.onnx has: the generator runs
        # from step 0 instead of waiting N steps for the history to fill. Without it the
        # legacy path withholds inference during the warmup and the cadence starts late.
        self.null_history_warmup = True
        self.zero_noise = True
        self.window_offsets = list(offsets)
        self.reference_blocks = SLIM
        self._fm_to_isaac = np.arange(NJ)
        self.ctx = SimpleNamespace(F=f, D=D, N=N, target_num_extra_frames=EE,
                                   has_history_null=True, cmd_dim=3, extero_source="depth")
        self._history = BatchedHistory(1, N)
        self._pred = np.zeros((1, f, D))
        self._pred_valid = False
        self._pred_idx = 0
        self._anchor_pos = np.zeros(3)
        self._anchor_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._prev_served_node = None
        self._ctrl_step = -1
        self._plan_count = 0
        self.replans: list[int] = []          # _ctrl_step at each re-plan
        self.anchors: list[tuple] = []        # (pos, quat) captured at each re-plan
        self.pushed: list[np.ndarray] = []    # history rows, in order

    def get_dt(self):
        """Control period, for the K>1 informational message."""
        return 0.02

    def _run_fm(self, state, history_null=False):
        """Stand in for the network: a distinct constant plan, then the real bookkeeping."""
        self._plan_count += 1
        self.replans.append(self._ctrl_step)
        # node n of plan p = p + n/1000, so both are recoverable from any served value.
        self._pred = (self._plan_count
                      + np.arange(self.ctx.F)[None, :, None] / 1000.0
                      + np.zeros((1, self.ctx.F, D)))
        self._pred_idx = 0
        self._anchor_pos = np.asarray(state["root_pos"], dtype=np.float64).copy()
        self._anchor_quat = np.asarray(state["root_quat"], dtype=np.float64).copy()
        self.anchors.append((self._anchor_pos.copy(), self._anchor_quat.copy()))
        self._pred_valid = True

    def _make_snapshot(self, state):
        """Record what gets pushed, then defer to the real implementation."""
        row = super()._make_snapshot(state)
        self.pushed.append(row["node"].copy())
        return row


# ---------------------------------------------------------------------------
# Cadence + cursor.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 5, 12])
def test_replans_exactly_every_k_steps(k):
    """The cadence is the absolute step counter mod K, as on the Isaac side."""
    pol = _FakePolicy(k)
    for i in range(48):
        pol._update_generator(_state([0.01 * i, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]))
    assert pol.replans == [s for s in range(48) if s % k == 0]


@pytest.mark.parametrize("k", [1, 3, 12])
def test_cursor_walks_the_plan_between_replans(k):
    """``_pred_idx`` is 0 on a re-plan and +1 per step, never past ``F``."""
    pol = _FakePolicy(k)
    seen = []
    for i in range(3 * k):
        pol._update_generator(_state([0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]))
        seen.append(pol.served_node_index)
    assert seen == [i % k for i in range(3 * k)]


def test_cursor_overrun_raises_rather_than_freezing_the_tail():
    """A cadence that stops firing must fail loudly; a clamp would look like a tracking bug."""
    pol = _FakePolicy(k=4, f=6)
    pol._update_generator(_state([0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]))   # plan, cursor 0
    pol.fm_update_every = 10_000                                            # cadence goes quiet
    with pytest.raises(RuntimeError, match="ran past the horizon"):
        for _ in range(10):
            pol._update_generator(_state([0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]))


def test_first_segment_after_a_forced_replan_is_short_not_misaligned():
    """Isaac's absolute modulo means an off-phase forced re-plan yields a SHORT first segment.

    Mirrored deliberately: the alternative (counting from the re-plan) would drift out of step
    with the Isaac side, which is the reference implementation.
    """
    pol = _FakePolicy(k=5)
    pol._ctrl_step = 1          # pretend a warmup consumed some steps
    for _ in range(12):
        pol._update_generator(_state([0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]))
    # First re-plan is forced at step 2 (no live plan), then the cadence takes over at 5, 10.
    assert pol.replans == [2, 5, 10]


# ---------------------------------------------------------------------------
# Serving: the cursor, and the stale anchor.
# ---------------------------------------------------------------------------
def test_served_nodes_come_from_the_cursor_not_the_window_start():
    """At K>1 the served window is ``_pred_idx + offsets``; reading from 0 would serve a plan
    that is up to K-1 steps stale."""
    pol = _FakePolicy(k=6, offsets=[0, 5, 10])
    st = _state([0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0])
    for _ in range(4):
        pol._update_generator(st)
    assert pol.served_node_index == 3
    served = pol._serve_nodes(st, pol.window_offsets)
    # node index is the fractional part * 1000; plan 1 is the only plan so far.
    got = np.round((served[:, 2 * 6] % 1.0) * 1000).astype(int)     # a joint_pos column
    assert got.tolist() == [3, 8, 13]


def test_a_stale_anchor_is_corrected_when_the_robot_has_moved():
    """The core of K>1: serving must not shear the window by the distance travelled.

    Build a window that encodes a KNOWN world reference path against the re-plan pose, then
    serve it from a moved robot. Lifting the served (re-anchored) nodes back through the NEW
    pose must recover the same world path.
    """
    rng = np.random.default_rng(0)
    old_p, old_q = np.array([1.0, -2.0, 0.8]), _unit(rng.normal(size=4))
    new_p, new_q = old_p + np.array([0.24, 0.05, 0.01]), _unit(rng.normal(size=4))
    ref_p, ref_q = rng.normal(size=(F, 3)), _unit(rng.normal(size=(F, 4)))

    pol = _FakePolicy(k=12, offsets=[0, 5, 10])
    pol._pred = world_to_window(
        old_p[None, :], old_q[None, :], ref_p, ref_q,
        rng.normal(size=(F, 3)), rng.normal(size=(F, 3)),
        rng.normal(size=(F, NJ)), rng.normal(size=(F, NJ)),
        rng.normal(size=(F, EE, 3)), blocks=SLIM)[None, ...]
    pol._pred_valid, pol._pred_idx = True, 7
    pol._anchor_pos, pol._anchor_quat = old_p, old_q

    served = pol._serve_nodes(_state(new_p, new_q), pol.window_offsets)
    # Undo the serve by re-anchoring back to the old pose; must return the stored nodes.
    back = reanchor_window(served, new_p[None, :], new_q[None, :], old_p[None, :],
                           old_q[None, :], NJ, EE, SLIM)
    assert np.abs(back - pol._pred[0, [7, 12, 17]]).max() < 1e-11


def test_k1_serving_is_an_identity_on_the_window():
    """At K=1 the anchor IS the current pose, so the re-anchor must not perturb anything.

    This is what makes applying it unconditionally (as Isaac does) free rather than a
    regression on the K=1 path that was already verified exact.
    """
    rng = np.random.default_rng(1)
    p, q = rng.normal(size=3), _unit(rng.normal(size=4))
    pol = _FakePolicy(k=1, offsets=[0, 5])
    pol._pred = world_to_window(
        p[None, :], q[None, :], rng.normal(size=(F, 3)), _unit(rng.normal(size=(F, 4))),
        rng.normal(size=(F, 3)), rng.normal(size=(F, 3)),
        rng.normal(size=(F, NJ)), rng.normal(size=(F, NJ)),
        rng.normal(size=(F, EE, 3)), blocks=SLIM)[None, ...]
    pol._pred_valid, pol._pred_idx = True, 0
    pol._anchor_pos, pol._anchor_quat = p, q
    served = pol._serve_nodes(_state(p, q), pol.window_offsets)
    assert np.abs(served - pol._pred[0, [0, 5]]).max() < 1e-12


# ---------------------------------------------------------------------------
# The autoregressive history row.
# ---------------------------------------------------------------------------
def test_history_row_is_the_node_served_last_step():
    """The context is the generator's OWN outputs, one per step -- at K>1 those are
    consecutive nodes of the same plan, which is what "N frames from the last solve" means."""
    pol = _FakePolicy(k=8, offsets=[0])
    st = _state([0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0])
    for _ in range(5):
        pol._update_generator(st)
    # pushed[i] is what step i pushed = the node served at step i-1 (zeros at step 0).
    assert np.abs(pol.pushed[0]).max() == 0.0
    for step in range(1, 5):
        node_idx = np.round((pol.pushed[step][0, 2 * 6] % 1.0) * 1000).astype(int)
        assert node_idx == step - 1, f"step {step} pushed node {node_idx}"


def test_reset_clears_the_anchor_and_the_stash():
    """A stash from before a reset describes another episode's pose; keeping it would push a
    row from that episode into the fresh history."""
    pol = _FakePolicy(k=4)
    pol._loaded = True
    pol._obs_history = {}
    for _ in range(3):
        pol._update_generator(_state([0.5, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]))
    assert pol._prev_served_node is not None and pol._pred_idx > 0
    pol.reset()
    assert pol._prev_served_node is None
    assert pol._pred_idx == 0
    assert not pol._pred_valid
    assert np.array_equal(pol._anchor_pos, np.zeros(3))


# ---------------------------------------------------------------------------
# The horizon rule.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,ok", [(1, True), (11, True), (12, True), (13, False), (20, False)])
def test_horizon_rule_is_max_offset_plus_k(k, ok):
    """``F >= max_offset + K``. With F=62 and offset 50 that is exactly K <= 12 -- which is
    why F=62 was chosen, and why K=13 needs a re-record rather than a config change."""
    pol = _FakePolicy(k)
    if ok:
        pol._check_horizon()
    else:
        with pytest.raises(ValueError, match="needs F >="):
            pol._check_horizon()


def test_horizon_error_names_the_two_ways_out():
    """Lower K or lengthen the horizon -- a bare 'too short' would not say which."""
    pol = _FakePolicy(k=13)
    with pytest.raises(ValueError) as ei:
        pol._check_horizon()
    msg = str(ei.value)
    assert "fm_update_every" in msg and "--future_len" in msg and "--ref_future_len" in msg


if __name__ == "__main__":
    test_replans_exactly_every_k_steps(5)
    test_cursor_walks_the_plan_between_replans(12)
    test_served_nodes_come_from_the_cursor_not_the_window_start()
    test_a_stale_anchor_is_corrected_when_the_robot_has_moved()
    test_k1_serving_is_an_identity_on_the_window()
    test_history_row_is_the_node_served_last_step()
    print("REPLAN CADENCE CHECKS PASSED")
