# Copyright 2026 Zachary Olkin. All rights reserved.

"""The re-plan cadence must be carried by the tracker export, not defaulted.

``fm_update_every`` decides how often the generator is re-sampled. It is baked into
``tracker.onnx`` at export, and both the MuJoCo runner and the hardware stack treat a missing
key as "use my own default" -- 1 in the GenTrackerPolicy constructor, and 2 from the MuJoCo
runner's CLI, which is what actually reaches it. A pair trained at K=12 then re-plans 6x too
often and merely looks like a wobbly policy.

The key goes missing easily: ``export_parameters`` reads it off the ``traj_ref`` command term
inside a try/except and omits it when the term has no such field, so ANY ``play.py`` run without
``--fm_checkpoint`` rewrites ``tracker.onnx`` without it while leaving the older ``fm.onnx``
alone. That happened in practice and cost a mis-run sim2sim session.

No ONNX files needed: ``TrackerSession`` is stubbed, since the guard runs before anything is
read off the graph.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy import gen_tracker as gt  # noqa: E402


class _StubTrackerSession:
    """Minimal stand-in exposing the ``params`` dict the guard consults."""

    params: dict = {}

    def __init__(self, path, num_threads=0, providers=None):  # noqa: D107 - signature match
        self.path = path

    def get_robot_asset(self) -> dict | None:
        """No fingerprint: the robot-asset pair check is not what these tests exercise."""
        return None


class _StubCtx:
    """Enough of an FmOnlineContext for load() to get past the FM section."""

    cmd_dim = 3
    N = 4
    robot_asset = None


def _policy(monkeypatch, params: dict, fm_onnx: str | None):
    """Build an unloaded policy whose TrackerSession yields ``params``.

    ``load_fm_context`` is stubbed as well: on this branch load() opens the FM BEFORE it
    resolves the cadence, so an unstubbed call would die on the missing file and never reach
    the guard under test.
    """
    stub = type("S", (_StubTrackerSession,), {"params": params})
    monkeypatch.setattr(gt, "TrackerSession", stub)
    monkeypatch.setattr(gt, "load_fm_context", lambda *a, **k: _StubCtx())
    return gt.GenTrackerPolicy("tracker.onnx", fm_onnx=fm_onnx)


def test_missing_cadence_with_an_fm_is_refused(monkeypatch):
    """An fm.onnx plus a cadence-less tracker export must raise, not default."""
    policy = _policy(monkeypatch, {}, fm_onnx="fm.onnx")
    with pytest.raises(ValueError, match="fm_update_every"):
        policy.load()


def test_the_error_explains_the_cause_and_the_fix(monkeypatch):
    """The message has to name the cause, or the next person re-derives it from mtimes."""
    policy = _policy(monkeypatch, {}, fm_onnx="fm.onnx")
    with pytest.raises(ValueError) as e:
        policy.load()
    msg = str(e.value)
    assert "--fm_checkpoint" in msg, "must name the run that drops the key"
    assert "fm.onnx" in msg, "must point at the mtime tell"


def test_tracker_only_is_unaffected(monkeypatch):
    """With no generator the cadence is meaningless, so its absence must NOT raise.

    Guarding unconditionally would break every tracker-only deploy, which is a legitimate
    configuration -- the clip drives the tracker and nothing is ever re-planned.
    """
    policy = _policy(monkeypatch, {}, fm_onnx=None)
    try:
        policy.load()
    except ValueError as exc:
        assert "fm_update_every" not in str(exc), "tracker-only must not trip the cadence guard"
    except Exception:
        pass  # any later failure is about the stub, not the guard


def test_present_cadence_is_taken_from_the_metadata(monkeypatch):
    """A carried cadence overrides the constructor default and does not raise."""
    policy = _policy(monkeypatch, {"fm_update_every": 12}, fm_onnx="fm.onnx")
    assert policy.fm_update_every == 1, "the constructor default, before load() consults metadata"
    try:
        policy.load()
    except ValueError as exc:
        assert "fm_update_every" not in str(exc)
    except Exception:
        pass  # later failures come from the stubbed FM, not the guard
    assert policy.fm_update_every == 12, "metadata must win over the constructor default"
