# Copyright 2026 Zachary Olkin. All rights reserved.

"""``fm_lead_ticks``: does the generator actually see STALE inputs?

This emulates the async C++ runtime's generator lead inside the synchronous one, to answer
whether the lead alone explains the async stack tracking worse. That makes it a measurement
instrument, and the failure mode to guard is not a crash -- it is an emulation that quietly
delays nothing, runs clean, and reports "the lead makes no difference". A null result from a
broken instrument is worse than no result, because it closes the question.

So the checks here are mostly POWER checks: with a lead, the generator's inputs must actually
differ from the current tick's, and the sample must survive the caller reusing its buffers.

What is delayed and what is not, per ``plan_scheduler.h`` and
``GenTrackerCore::BuildRequest``: the exteroception and the held command are sampled at
REQUEST time (early), while the history is projected forward exactly, so it is NOT stale.
The tracker's own observation is not delayed either -- on a robot it reads its sensors every
tick regardless of when the generator was asked.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from collections import deque  # noqa: E402

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402


def _bare(lead: int) -> GenTrackerPolicy:
    """A policy with only the lead machinery initialised.

    ``__new__`` without ``__init__``, the same shape as this suite's other fakes: the two
    helpers under test read nothing else, and loading real ONNX artifacts to exercise a deque
    would make the test need a GPU box.

    Args:
        lead: Ticks of lead to emulate.

    Returns:
        The bare policy.
    """
    p = GenTrackerPolicy.__new__(GenTrackerPolicy)
    p.fm_lead_ticks = lead
    p._lead_buf = deque(maxlen=lead + 1)
    return p


def _state(tick: int) -> dict:
    """A state dict whose every field encodes the tick, so staleness is visible.

    Args:
        tick: The tick number to encode.

    Returns:
        A state dict with depth, camera poses and a command.
    """
    return {
        "depth": np.full((2, 4), float(tick)),
        "command": np.array([float(tick), 0.0, 0.0]),
        "fm_camera_pose": [(np.full(3, float(tick)), np.eye(3) * float(tick))],
        "root_pos": np.zeros(3),
    }


def test_the_class_default_is_synchronous():
    """An instance built without __init__ must not accidentally acquire a lead.

    The test fakes in this suite skip the constructor on purpose, so the default lives on the
    class. If this ever fails, every one of those fakes silently started delaying its inputs.
    """
    p = GenTrackerPolicy.__new__(GenTrackerPolicy)
    assert p.fm_lead_ticks == 0


def test_no_lead_conditions_on_the_current_tick():
    """The default path must be byte-for-byte the old behaviour: the state passed straight in."""
    p = _bare(0)
    s = _state(7)
    cond, held = p._fm_conditioning_state(s)
    assert cond is s, "with no lead the generator must see the caller's own state object"
    assert held is None, "with no lead nothing may override the held command"


def test_the_lead_actually_delays_the_exteroception_and_command():
    """THE power check. With a lead of L, tick t must be conditioned on tick t - L."""
    lead = 3
    p = _bare(lead)
    seen = []
    for tick in range(10):
        p._lead_buf.append(p._lead_sample(_state(tick)))
        cond, held = p._fm_conditioning_state(_state(tick))
        seen.append((tick, float(cond["depth"][0, 0]), None if held is None else float(held[0])))

    # Until the window fills there is nothing older to serve, so the current tick is used --
    # the same situation the real runtime is in before it has a plan to project from.
    for tick, depth_tick, held_tick in seen[:lead]:
        assert depth_tick == tick, f"tick {tick}: expected no delay yet, got {depth_tick}"

    # Once full, EXACTLY `lead` ticks of delay, on both inputs.
    for tick, depth_tick, held_tick in seen[lead:]:
        assert depth_tick == tick - lead, f"tick {tick}: depth came from {depth_tick}"
        assert held_tick == tick - lead, f"tick {tick}: command came from {held_tick}"


@pytest.mark.parametrize("lead", [1, 3, 12])
def test_the_delay_is_the_requested_length(lead):
    """No off-by-one at any lead: the buffer holds lead+1 so entry 0 is lead ticks old."""
    p = _bare(lead)
    for tick in range(3 * lead + 5):
        p._lead_buf.append(p._lead_sample(_state(tick)))
    cond, held = p._fm_conditioning_state(_state(3 * lead + 4))
    assert float(cond["depth"][0, 0]) == float(3 * lead + 4 - lead)


def test_the_sample_survives_the_caller_reusing_its_buffers():
    """Copies, not references.

    The runner builds a fresh state dict each step but the arrays inside can be reused
    buffers -- the MuJoCo depth cast in particular. Aliasing one would make the "stale"
    reading identical to the current one, so the emulation would delay nothing while looking
    like it did, and every comparison drawn from it would be wrong in the safe-looking
    direction.
    """
    p = _bare(2)
    s = _state(1)
    sample = p._lead_sample(s)
    s["depth"][...] = 999.0                       # the caller reuses its buffer
    s["command"][...] = 999.0
    s["fm_camera_pose"][0][0][...] = 999.0
    assert float(sample["depth"][0, 0]) == 1.0
    assert float(sample["command"][0]) == 1.0
    assert float(sample["fm_camera_pose"][0][0][0]) == 1.0


def test_a_height_scan_reading_is_delayed_too():
    """The scan is the other exteroception source and must not be left behind."""
    p = _bare(1)
    for tick in range(4):
        p._lead_buf.append(p._lead_sample({"height_scan": np.full(5, float(tick)),
                                           "command": np.zeros(3)}))
    cond, _ = p._fm_conditioning_state({"height_scan": np.full(5, 3.0),
                                        "command": np.zeros(3)})
    assert float(cond["height_scan"][0]) == 2.0
