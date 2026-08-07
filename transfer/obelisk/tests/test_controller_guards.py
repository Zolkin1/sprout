# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the controller's non-finite-command guard, off-robot.

``G1Controller._finite_targets`` is the last thing between a NaN and a live robot's PD
targets, and it is unreachable from the normal test suite: importing ``controller`` needs
rclpy and the obelisk message packages, which only exist inside the deploy container. So
the ROS imports are stubbed and the guard is exercised as a plain method against a
stand-in holding only the attributes it touches -- the real code, not a reimplementation.

Lives outside ``g1_control/test/`` on purpose: that directory is the ament lint package
and its files import ``ament_*``, which is not installed here.
"""

from __future__ import annotations

import os
import sys
import types

from unittest.mock import MagicMock

import numpy as np
import pytest

_OBELISK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRANSFER = os.path.dirname(_OBELISK)
sys.path.insert(0, os.path.join(_OBELISK, "g1_control"))
sys.path.insert(0, _TRANSFER)


def _msg_init(self) -> None:
    """Give a stubbed message a ``header`` with a settable ``stamp``, like the real one."""
    self.header = types.SimpleNamespace(stamp=None)


def _stub_ros_modules() -> None:
    """Install minimal stand-ins for the ROS packages ``controller`` imports at module scope.

    Only their names are needed -- the guard under test touches none of them -- but the
    import must succeed for the module to load at all.
    """
    def mod(name: str, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules.setdefault(name, m)
        return m

    # MagicMock rather than a bare class: module scope reads enum members off these
    # (HistoryPolicy.KEEP_LAST, ...), and a mock conjures whatever is asked for.
    cls = MagicMock
    # First-party diagnostic messages. Real classes, not mocks: the tests assign typed
    # fields on them, and a MagicMock would accept literally anything -- including the
    # field-name typos this is meant to catch.
    mod("g1_msgs")
    mod("g1_msgs.msg",
        ControllerDebug=type("ControllerDebug", (), {"__init__": _msg_init}),
        FmPlan=type("FmPlan", (), {"__init__": _msg_init}))
    mod("obelisk_control_msgs")
    mod("obelisk_control_msgs.msg", ExecutionFSM=cls(), PDFeedForward=cls(),
        VelocityCommand=cls())
    mod("obelisk_estimator_msgs")
    mod("obelisk_estimator_msgs.msg", EstimatedState=cls())
    mod("obelisk_py")
    mod("obelisk_py.core")
    # Must be a real class -- G1Controller subclasses it.
    mod("obelisk_py.core.control", ObeliskController=type("ObeliskController", (), {}))
    mod("obelisk_py.core.utils")
    mod("obelisk_py.core.utils.ros", spin_obelisk=lambda *a, **k: None)
    mod("rclpy")
    mod("rclpy.executors", MultiThreadedExecutor=cls())
    mod("rclpy.qos", DurabilityPolicy=cls(), HistoryPolicy=cls(), QoSProfile=lambda **k: None,
        ReliabilityPolicy=cls())
    mod("sensor_msgs")
    mod("sensor_msgs.msg", Image=cls())


_stub_ros_modules()
controller = pytest.importorskip("g1_control.controller", exc_type=ImportError)
G1Controller = controller.G1Controller
DISARM_TICKS = controller._NONFINITE_DISARM_TICKS

NJ = 6


class _Logger:
    """Captures log lines so the tests can assert the guard actually reported."""

    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg, **kw):
        self.errors.append(msg)

    def warn(self, msg, **kw):
        pass

    def info(self, msg, **kw):
        pass


class _Fake:
    """A stand-in exposing exactly the attributes ``_finite_targets`` reads."""

    def __init__(self, hold_pose=None):
        self._last_good_targets = None
        self._nonfinite_streak = 0
        self.hold_pose = hold_pose
        self.armed = True
        self.q_base = np.zeros(7)
        self.v_base = np.zeros(6)
        self.joint_pos = np.zeros(NJ)
        self.joint_vel = np.zeros(NJ)
        self.vel_cmd = np.zeros(3)
        self.depth_raw: list = []
        self._logger = _Logger()
        self.policy = types.SimpleNamespace(get_dt=lambda: 0.02)

    def get_logger(self):
        return self._logger

    # The methods under test, bound to this stand-in.
    _finite_targets = G1Controller._finite_targets
    _diagnose_nonfinite = G1Controller._diagnose_nonfinite


def test_finite_command_passes_through_and_is_latched():
    """A clean command publishes unchanged and becomes the fallback for the next tick."""
    f = _Fake()
    good = np.arange(NJ, dtype=np.float64)
    assert np.array_equal(f._finite_targets(good), good)
    assert np.array_equal(f._last_good_targets, good)
    assert f._nonfinite_streak == 0
    assert not f._logger.errors
    # The latch must be a COPY: the caller's array is reused across ticks, so storing a
    # view would make the fallback track the corruption it is supposed to survive.
    good[0] = np.nan
    assert np.isfinite(f._last_good_targets).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_falls_back_to_last_good(bad):
    """NaN and both infinities are rejected in favour of the previous tick's command."""
    f = _Fake()
    good = np.arange(NJ, dtype=np.float64)
    f._finite_targets(good)
    dirty = good.copy()
    dirty[3] = bad
    out = f._finite_targets(dirty)
    assert np.array_equal(out, good), "must publish the last good target, not the bad one"
    assert f._nonfinite_streak == 1
    assert f._logger.errors and "NON-FINITE COMMAND" in f._logger.errors[0]


def test_fallback_is_hold_pose_before_any_good_command():
    """With no good tick yet, the arm-time pose is the only fallback available."""
    hold = np.full(NJ, 0.25)
    f = _Fake(hold_pose=hold)
    out = f._finite_targets(np.full(NJ, np.nan))
    assert np.array_equal(out, hold)
    assert "arm-time pose" in f._logger.errors[0]


def test_publishes_nothing_when_no_finite_fallback_exists():
    """No good tick and no hold pose -> publish nothing; the interface holds."""
    f = _Fake(hold_pose=None)
    assert f._finite_targets(np.full(NJ, np.nan)) is None
    assert f.armed, "a single bad tick must not disarm"


def test_sustained_fault_disarms():
    """A persistent non-finite input escalates rather than freezing on a stale target."""
    f = _Fake()
    f._finite_targets(np.arange(NJ, dtype=np.float64))
    dirty = np.full(NJ, np.nan)
    for i in range(1, DISARM_TICKS):
        assert f._finite_targets(dirty) is not None
        assert f.armed, f"disarmed early, at tick {i}"
    assert f._finite_targets(dirty) is None
    assert not f.armed
    assert any("DISARMING" in e for e in f._logger.errors)


def test_streak_resets_after_recovery():
    """A transient fault must not accumulate towards the disarm threshold."""
    f = _Fake()
    good = np.arange(NJ, dtype=np.float64)
    for _ in range(DISARM_TICKS - 1):
        f._finite_targets(good)
        f._finite_targets(np.full(NJ, np.nan))
        assert f.armed
    assert f._nonfinite_streak == 1


def test_diagnosis_names_the_offending_input():
    """The error must say WHICH input went bad -- estimator death vs policy divergence."""
    f = _Fake()
    f.joint_vel = np.full(NJ, np.nan)
    assert "joint_vel" in f._diagnose_nonfinite()
    f.joint_vel = np.zeros(NJ)
    f.q_base = np.full(7, np.inf)
    assert "q_base" in f._diagnose_nonfinite()
    f.q_base = np.zeros(7)
    assert "policy itself diverged" in f._diagnose_nonfinite()
