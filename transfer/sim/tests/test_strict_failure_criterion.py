# Copyright 2026 Zachary Olkin. All rights reserved.

"""The strict, reference-relative failure criterion on the MuJoCo side.

``test_fall_criterion.py`` covers the reference-FREE backstop -- pelvis below the scene's
lowest terrain surface, or 70 deg from world upright.  That test exists because a wrong
scoring rule "does not crash, it publishes a number", and this one exists for the same reason
one level up: the backstop says nothing about a robot that stumbles down a flight and barely
catches itself, which is the behaviour a selector must not be taught to like.

What is pinned here, each because the alternative is a plausible wrong number:

* the LATCH does not end the episode, and fires on the checks it should;
* the opening command hold is **latched but not scored**, so the spawn transient cannot drag
  every episode's reward toward a common value -- and it drags SHORT episodes hardest, which
  are the failures;
* :meth:`TrackingProbe.reset` rewinds it, which is what a rejected best-of-K candidate needs
  or the first branch to stumble would condemn the whole episode;
* the trace carries the CONTROL-step index of each row, because it starts at the generator's
  first plan and not at step 0;
* ``schema.derive_success``'s strict and loose arms differ exactly on "violated on the way to
  a goal it then reached", and agree everywhere else.

No MuJoCo model is built: the probe reads a handful of fields off ``robot`` and ``policy``, so
small fakes make the behaviour testable without a 20 MB scene and without a GPU.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(_SIM))
for _p in (_SIM, os.path.dirname(_SIM), os.path.join(_SIM, "mjeval"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deploy.tracking_error import BOOTSTRAP_SPEC, TrackingErrorSpec  # noqa: E402
from mjeval.tracking import TrackingProbe  # noqa: E402

_NJ = 29
_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


class _FakeData:
    """The two ``mj_data`` fields the probe reads."""

    def __init__(self) -> None:
        self.time = 0.0
        # [pos(3), quat WXYZ(4), joints(nj)] -- MuJoCo's layout, standing at the origin.
        self.qpos = np.concatenate([np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
                                    np.zeros(_NJ)])


class _FakeRobot:
    """Enough of ``sim.robot.Robot`` for :class:`TrackingProbe`."""

    def __init__(self) -> None:
        self.mj_data = _FakeData()
        self.joint_names = [f"j{i}" for i in range(_NJ)]

    def get_base_world_velocity(self, _body):
        """Frame-origin twist; zero, so only the pose drives the error."""
        return np.zeros(3), np.zeros(3)


class _FakePolicy:
    """A policy whose served reference the test moves by hand."""

    base_frame_name = "pelvis_link"

    def __init__(self) -> None:
        self.ref = {"base_pos_w": np.zeros(3), "base_quat_w": _IDENTITY.copy(),
                    "base_lin_vel_w": np.zeros(3), "base_ang_vel_w": np.zeros(3),
                    "joint_pos": np.zeros(_NJ)}

    def get_joint_names(self):
        """Isaac joint order; identical to the robot's here, so no permutation."""
        return [f"j{i}" for i in range(_NJ)]

    def served_reference_w(self):
        """The reference the tracker was told to be at this step."""
        return self.ref


def _rig(hold: float = 0.0, spec: TrackingErrorSpec = BOOTSTRAP_SPEC):
    """A probe with its fake robot and policy."""
    return TrackingProbe(spec, keep_trace=True, hold_until_s=hold), _FakeRobot(), _FakePolicy()


def test_the_latch_fires_on_the_checks_and_never_on_downrange():
    """Lateral, height and tilt end an episode; downrange and yaw never do."""
    spec = TrackingErrorSpec(max_lateral_m=0.2, max_height_m=0.2,
                             max_tilt_rad=math.radians(30.0))
    for offset, expect in (([5.0, 0.0, 0.0], None),        # downrange: never fatal
                           ([0.0, 0.5, 0.0], "lateral"),
                           ([0.0, 0.0, 0.5], "height")):
        probe, robot, policy = _rig(spec=spec)
        robot.mj_data.qpos[:3] = offset
        probe.fold(robot, policy, 7)
        assert (probe.first_violation_step == 7) == (expect is not None), offset
        for name in ("lateral", "height", "tilt"):
            assert (probe.first_check_step[name] >= 0) == (name == expect), (offset, name)

    # ...and tilt, which is measured against the REFERENCE rather than world upright.
    probe, robot, policy = _rig(spec=spec)
    half = math.radians(40.0) / 2.0
    robot.mj_data.qpos[3:7] = [math.cos(half), 0.0, math.sin(half), 0.0]      # WXYZ pitch
    probe.fold(robot, policy, 3)
    assert probe.first_check_step["tilt"] == 3
    assert probe.first_violation_step == 3


def test_the_latch_records_the_FIRST_crossing_and_does_not_end_the_episode():
    """Latched, not terminating -- the caller keeps folding and the step index is the first."""
    probe, robot, policy = _rig()
    for step in range(10):
        robot.mj_data.qpos[1] = 0.0 if step < 4 else 0.5      # lateral, from step 4
        robot.mj_data.time = step * 0.02
        probe.fold(robot, policy, step)
    assert probe.first_violation_step == 4, "latched on the wrong step"
    assert probe.n_folded == 10, "folding stopped at the violation; it must not"
    assert probe.violated


def test_the_opening_hold_is_latched_but_not_scored():
    """The spawn transient must not enter the reward mean, but must still be able to fail.

    Measured on the real pair, the hold sits near 0.80 whatever the episode goes on to do, so
    folding it in drags every episode toward that value -- up on the bad ones, down on the
    good -- and hardest on SHORT episodes, where 25 steps is a larger share.
    """
    hold = 0.10                                              # 5 steps at dt = 0.02
    probe, robot, policy = _rig(hold=hold)
    for step in range(10):
        robot.mj_data.time = step * 0.02
        # Perfect tracking during the hold, then a constant small error after it.
        robot.mj_data.qpos[1] = 0.0 if robot.mj_data.time < hold else 0.05
        probe.fold(robot, policy, step)
    assert probe.n_folded == 10
    assert probe.n_scored == 5, "the hold's steps were scored"
    # The reward mean is over the SCORED steps only, so it reflects the 0.05 m error alone.
    assert probe.mean_step_reward < 1.0
    lone = TrackingProbe(BOOTSTRAP_SPEC)
    r, rb, pl = lone, _FakeRobot(), _FakePolicy()
    rb.mj_data.qpos[1] = 0.05
    r.fold(rb, pl, 0)
    assert probe.mean_step_reward == pytest.approx(r.mean_step_reward, abs=1e-12)

    # ...but a violation DURING the hold still latches: a robot that cannot stand has failed.
    probe2, robot2, policy2 = _rig(hold=hold)
    robot2.mj_data.time = 0.02
    robot2.mj_data.qpos[1] = 0.5
    probe2.fold(robot2, policy2, 1)
    assert probe2.first_violation_step == 1
    assert probe2.n_scored == 0


def test_reset_rewinds_the_latch_for_a_rejected_branch():
    """A rejected best-of-K candidate must not leave its stumble on the episode."""
    probe, robot, policy = _rig()
    robot.mj_data.qpos[1] = 0.5
    probe.fold(robot, policy, 2)
    assert probe.violated
    probe.reset()
    assert not probe.violated and probe.first_violation_step == -1
    assert probe.n_folded == 0 and probe.trace_array().shape[0] == 0
    assert all(v == -1 for v in probe.first_check_step.values())


def test_the_trace_carries_the_control_step_index_not_a_row_count():
    """The trace starts at the generator's first plan, so row i is NOT step i.

    A plot or a join that assumed otherwise would be shifted by the length of the opening
    hold -- about 25 steps, and entirely plausible-looking on every curve.
    """
    probe, robot, policy = _rig()
    for step in (17, 18, 19, 20):
        probe.fold(robot, policy, step)
    assert probe.step_index_array().tolist() == [17, 18, 19, 20]
    assert probe.trace_array().shape[0] == 4


def test_no_served_reference_is_a_state_not_an_error():
    """Before the first plan there is nothing to measure, and folding must be a no-op."""
    probe, robot, policy = _rig()
    policy.ref = None
    assert probe.fold(robot, policy, 0) is None
    assert probe.n_folded == 0 and probe.first_violation_step == -1


def test_derive_success_strict_and_loose_differ_only_on_violate_then_arrive():
    """The two arms of `mjeval.schema.derive_success`, on the case that separates them."""
    import terrains
    from mjeval import schema as mjschema

    n = 4
    cols = {
        "verdict": np.array([terrains.VERDICT_CODES["success"],
                             terrains.VERDICT_CODES["success"],
                             terrains.VERDICT_CODES["fall"],
                             terrains.VERDICT_CODES["timeout"]], dtype=np.int32),
        "reached_target": np.array([1, 1, 1, 0], dtype=np.uint8),
        "traversed": np.array([1, 1, 1, 0], dtype=np.uint8),
        #                  clean | violated then arrived | fell | timed out
        "first_violation_step": np.array([-1, 40, 12, -1], dtype=np.int32),
    }
    assert mjschema.derive_success(cols, strict=False).tolist() == [1.0, 1.0, 0.0, 0.0]
    assert mjschema.derive_success(cols, strict=True).tolist() == [1.0, 0.0, 0.0, 0.0]

    # A table written before the checks existed reads under the loose rule either way, rather
    # than having every episode silently pass the strict one.
    old = {k: v for k, v in cols.items() if k != "first_violation_step"}
    assert mjschema.derive_success(old, strict=True).tolist() == [1.0, 1.0, 0.0, 0.0]


def test_the_two_harnesses_agree_on_the_success_rule():
    """MuJoCo's `derive_success` and Isaac's `arrived_before_failing` are one rule.

    Stated as a test because the two live in different trees and were, until this change, two
    different rules -- `report.py` failed any episode whose fall termination bit was set
    regardless of ordering, on 51 of 298 episodes of one measured rollout.
    """
    # Loaded BY PATH, not by name. `scripts/rsl_rl/eval/` and `transfer/sim/mjeval/` both
    # contain `schema.py` (and `runner.py`, `accumulate.py`, `report.py`, `evaluate.py`), and
    # both directories go on `sys.path`, so a bare `import schema` resolves to whichever suite
    # imported first -- which is a real, pre-existing collision in this tree and would make
    # this test silently skip in one run order and pass in another.
    import importlib.util

    _path = os.path.join(_ROOT, "scripts", "rsl_rl", "eval", "schema.py")
    if not os.path.exists(_path):
        pytest.skip("the Isaac eval harness is not in this tree")
    _spec = importlib.util.spec_from_file_location("_isaac_eval_schema", _path)
    eval_schema = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(eval_schema)
    cases = [  # (arrived, goal_step, fall_step, viol_step, expected_strict)
        (1, 10, -1, -1, True),
        (1, 10, 40, -1, True),      # arrived THEN failed -> success on both sides
        (1, 40, 10, -1, False),
        (1, 40, -1, 10, False),     # violated first
        (0, -1, -1, -1, False),     # never arrived
    ]
    cols = {k: np.array([c[i] for c in cases]) for i, k in enumerate(
        ("reached_goal", "first_goal_step", "first_fall_step", "first_violation_step"))}
    got = eval_schema.arrived_before_failing(cols, strict=True).tolist()
    assert got == [c[4] for c in cases]
