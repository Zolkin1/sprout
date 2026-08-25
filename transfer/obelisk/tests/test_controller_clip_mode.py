# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the controller's debug-clip tick, off-robot.

``G1Controller._clip_tick`` is the whole point of debug-clip mode and is unreachable from
the normal suite: importing ``controller`` needs rclpy and the obelisk message packages,
which only exist inside the deploy container. Same approach as
``test_controller_guards.py`` -- the ROS imports are stubbed and the real methods are bound
to a stand-in holding only the attributes they touch.

What is actually pinned here, in the order it can break:

* the clip advances one step per tick, and STOPS at the real length rather than running
  into the repeat-last-frame padding;
* a latched depth watchdog FREEZES the clip instead of advancing it (in clip mode there is
  no velocity command to zero, so freezing is the equivalent safe response);
* the velocity-command watchdog is NOT consulted, so an absent joystick cannot stall a run;
* the reference handed to the policy has one entry per tracker window offset;
* the command handed to the policy is the CLIP's, not the joystick's;
* the clip drives from the FIRST tick -- there is no PD warm-up to sit through.

Run: ``pytest transfer/obelisk/tests/test_controller_clip_mode.py``
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

_OBELISK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRANSFER = os.path.dirname(_OBELISK)
sys.path.insert(0, os.path.join(_OBELISK, "g1_control"))
sys.path.insert(0, _TRANSFER)

from test_controller_guards import _stub_ros_modules  # noqa: E402

_stub_ros_modules()
controller = pytest.importorskip("g1_control.controller", exc_type=ImportError)
G1Controller = controller.G1Controller

NJ = 6
OFFSETS = [0, 1, 2, 5]
REAL_STEPS = 8


class _Logger:
    """Collects log lines so a test can assert the controller reported an edge."""

    def __init__(self):
        self.lines: list[str] = []

    def _add(self, msg, **kw):
        self.lines.append(str(msg))

    error = warn = info = _add


class _Clip:
    """Minimal :class:`deploy.clip_reference.ClipReference` stand-in.

    Real ``ClipReference`` needs an on-disk ``.h5``; this exposes only what ``_clip_tick``
    calls, and records the steps it was asked for so the advance/freeze logic is testable.
    """

    def __init__(self):
        self.dt = 0.02
        self.start_step = 0
        self.real_steps = REAL_STEPS
        self.real_length_s = REAL_STEPS * self.dt
        self.served: list[int] = []
        self.anchored: list[int] = []

    def __repr__(self) -> str:
        return f"_Clip(real_steps={self.real_steps})"

    def command_at(self, step: int) -> np.ndarray:
        # Nonzero, so the exhaustion path reports "ends mid-stride" -- the case that is NOT
        # safe to hold and therefore the one worth pinning.
        return np.array([0.5, 0.0, 0.0])

    def state_at(self, step: int) -> dict:
        return {"jp": np.zeros(NJ)}

    def reanchor_ego(self, step, pos, quat) -> None:
        self.anchored.append(int(step))

    def reference_at(self, step: int, offsets) -> dict:
        self.served.append(int(step))
        return {"window": [{"offset": o} for o in offsets]}


class _Fake:
    """A stand-in exposing exactly what ``_clip_tick`` and its helpers read."""

    def __init__(self, ego_anchor: bool = True, depth_stale: bool = False):
        self._clip = _Clip()
        self._clip_step = 0
        # Last step the clip PLAYS. The real controller derives this in _setup_clip from
        # debug_clip_duration; the whole-clip default is the last real frame.
        self._clip_stop_step = self._clip.real_steps - 1
        self._clip_started = False
        self._clip_exhausted = False
        self.clip_ego_anchor = ego_anchor
        self.clip_ego_anchor_every = 1
        self._depth_stale = False
        self._vel_cmd_stale = False
        self._stale_depth = depth_stale
        self.vel_cmd_asked = False
        self.published: list[np.ndarray] = []
        self.states: list[dict] = []
        self._infer_ms: list[float] = []
        self._last_report = 1e9        # suppress the periodic latency report
        self.t = 0.0
        self._logger = _Logger()
        self.policy = types.SimpleNamespace(
            window_offsets=OFFSETS,
            get_dt=lambda: 0.02,
            get_joint_names=lambda: [f"j{i}" for i in range(NJ)],
            create_obs_from_reference=self._create_obs,
            get_action=lambda obs, names: np.zeros(NJ),
        )
        self.hw_joint_names = [f"j{i}" for i in range(NJ)]
        self.last_reference = None
        self.debug_records: list[tuple] = []
        #: Every clip_tick the policy was handed. Under window_anchor='plan' the policy
        #: latches its window anchor on this number and the controller re-places the clip on
        #: it, so the two cannot fire on different steps -- pinned below.
        self.clip_ticks: list[int] = []

    # --- collaborators the tick calls into ---
    def _create_obs(self, state, reference, clip_tick=None):
        self.last_reference = reference
        self.clip_ticks.append(clip_tick)
        return np.zeros((1, 4))

    def _build_state(self, command=None):
        state = {"root_pos": np.zeros(3), "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                 "joint_pos": np.zeros(NJ), "command": command}
        self.states.append(state)
        return state

    def _publish(self, targets):
        self.published.append(np.asarray(targets))
        return targets

    def _publish_debug(self, state, obs):
        # Message construction is covered by the diagnostics tests; what matters here is
        # that the tick records one sample and does so AFTER the command goes out.
        assert self.published, "diagnostics must be published after the command, not before"
        self.debug_records.append((state, obs))

    def _depth_is_stale(self) -> bool:
        return self._stale_depth

    def _vel_cmd_is_stale(self) -> bool:
        self.vel_cmd_asked = True
        return True                    # would freeze the clip if it were ever consulted

    def get_logger(self):
        return self._logger

    # The real methods under test.
    _clip_tick = G1Controller._clip_tick
    _start_clip = G1Controller._start_clip
    _report_clip_exhausted = G1Controller._report_clip_exhausted
    _update_watchdogs = G1Controller._update_watchdogs
    _record_latency = G1Controller._record_latency


def test_clip_advances_one_step_per_tick():
    """Each tick serves the next clip step, so the motion plays at the control rate."""
    f = _Fake()
    for _ in range(4):
        f._clip_tick()
    assert f._clip.served == [0, 1, 2, 3], f._clip.served
    assert len(f.published) == 4


def test_reference_matches_the_tracker_window():
    """The served reference has one entry per tracker offset, which the policy requires."""
    f = _Fake()
    f._clip_tick()
    assert [e["offset"] for e in f.last_reference["window"]] == OFFSETS


def test_command_comes_from_the_clip_not_the_joystick():
    """The policy is told the clip's authored command, matching the motion it tracks."""
    f = _Fake()
    f._clip_tick()
    assert np.array_equal(f.states[-1]["command"], np.array([0.5, 0.0, 0.0]))


def test_velocity_command_watchdog_is_not_consulted():
    """An absent joystick must not stall a clip run -- nothing publishes vel_cmd here."""
    f = _Fake()
    for _ in range(3):
        f._clip_tick()
    assert not f.vel_cmd_asked
    assert f._clip.served == [0, 1, 2]


def test_the_anchor_tick_freezes_with_the_clip():
    """The window-anchor latch is counted on the CLIP's tick, not on control ticks.

    Under ``window_anchor='plan'`` the reference window's ``base_pose`` is measured from the
    robot as it was at the last latch, and the clip's placement is solved from the robot at
    the same instant -- which is the only reason the served window is a pure function of the
    clip and needs no odometry. A depth dropout freezes the clip step but not the control
    loop, so a latch counted on control ticks would slide by exactly the length of the freeze
    and quietly put the robot's egomotion back into the observation.
    """
    f = _Fake(depth_stale=True)
    for _ in range(6):
        G1Controller._clip_tick(f)
    assert f.clip_ticks == [0, 0, 0, 0, 0, 0], (
        f"a frozen clip must hand the policy a frozen tick, got {f.clip_ticks}")

    g = _Fake()
    for _ in range(6):
        G1Controller._clip_tick(g)
    assert g.clip_ticks == [0, 1, 2, 3, 4, 5], \
        f"a running clip must advance it one per tick, got {g.clip_ticks}"


def test_the_anchor_tick_is_relative_to_the_start_step():
    """Starting partway into a clip must not shift the latch phase.

    ``--start_time`` puts ``start_step`` anywhere, and the cadence origin has to move with it:
    keyed off the raw clip index instead, a start_step that is not a multiple of the cadence
    would latch on different steps than the clip is re-placed on, for the whole run.
    """
    f = _Fake()
    f._clip.real_steps = 20                 # room to run past the start step
    f._clip_stop_step = f._clip.real_steps - 1
    f._clip.start_step = 7
    f._clip_step = 7
    for _ in range(4):
        G1Controller._clip_tick(f)
    assert f.clip_ticks == [0, 1, 2, 3], \
        f"the tick must be measured from start_step, got {f.clip_ticks}"


def test_depth_dropout_freezes_the_clip():
    """A frozen image means a frozen reference, not a reference that walks on regardless."""
    f = _Fake(depth_stale=True)
    for _ in range(4):
        f._clip_tick()
    assert f._clip.served == [0, 0, 0, 0], f._clip.served
    assert any("DEPTH DROPOUT" in line for line in f._logger.lines)
    # It still commands every tick -- freezing the reference must not stop control.
    assert len(f.published) == 4


def test_clip_saturates_at_the_real_length_and_says_so_once():
    """Playback stops at the real end, not in the repeat-last-frame padding."""
    f = _Fake()
    for _ in range(REAL_STEPS + 5):
        f._clip_tick()
    assert max(f._clip.served) == REAL_STEPS - 1
    assert f._clip_exhausted
    exhausted = [line for line in f._logger.lines if "CLIP EXHAUSTED" in line]
    assert len(exhausted) == 1, exhausted
    # This clip's final frame is commanded at 0.5 m/s, i.e. it ends MID-STRIDE. Holding that
    # is not a safe state, and saying "it is a stand" there would be actively misleading.
    assert "mid-stride" in exhausted[0] and "DISARM" in exhausted[0], exhausted[0]


def test_handover_reports_the_joint_offset_once():
    """The tracker's starting joint error is measured and logged, not silently blended."""
    f = _Fake()
    for _ in range(3):
        f._clip_tick()
    handover = [line for line in f._logger.lines if "CLIP HANDOVER" in line]
    assert len(handover) == 1, handover
    assert "joint offset" in handover[0]


def test_world_anchored_mode_never_reanchors():
    """With ego anchoring off the clip keeps its own placement (the MuJoCo parity mode)."""
    f = _Fake(ego_anchor=False)
    for _ in range(3):
        f._clip_tick()
    assert f._clip.anchored == []


def test_ego_anchor_runs_every_tick_at_cadence_one():
    """Cadence 1 re-places the clip every step, which is what hardware defaults to."""
    f = _Fake(ego_anchor=True)
    for _ in range(3):
        f._clip_tick()
    # One extra from _start_clip's handover placement on the first tick.
    assert f._clip.anchored == [0, 0, 1, 2], f._clip.anchored


def test_every_tick_records_a_diagnostic_sample():
    """A recorded run needs one sample per tick, including while the clip is frozen."""
    f = _Fake(depth_stale=True)
    for _ in range(4):
        f._clip_tick()
    assert len(f.debug_records) == 4
    assert all(obs is not None for _, obs in f.debug_records)


def test_a_clip_ending_in_a_stand_is_reported_as_safe():
    """The stand cases DO end standing, and the operator should be told so plainly."""
    f = _Fake()
    f._clip.command_at = lambda step: np.zeros(3)
    for _ in range(REAL_STEPS + 2):
        f._clip_tick()
    line = next(x for x in f._logger.lines if "CLIP EXHAUSTED" in x)
    assert "safe state" in line and "DISARM" not in line, line



def test_the_clip_drives_from_the_very_first_tick():
    """No PD warm-up: whatever feeds the tracker does so immediately, as in MuJoCo.

    A stiffened hold would put the robot in a state no training episode started from, and
    would delay the only thing that can actually keep it upright.
    """
    f = _Fake()
    f._clip_tick()
    assert f._clip.served == [0], "the first tick must already serve clip step 0"
    assert len(f.published) == 1, "the first tick must already command the policy's output"
