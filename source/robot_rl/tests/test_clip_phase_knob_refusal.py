# Copyright 2026 Zachary Olkin. All rights reserved.
"""Which history arms a CLIP-FREE library admits.

``_refuse_clip_phase_knobs`` exists to refuse settings that are claims about a motion the
library does not contain. It demanded ``history_from == "generator"``, giving as its reason
"the only other value is 'clip'" -- which was true when there were two arms and wrong from the
moment ``measured`` was added. ``measured`` fills the history from the ROBOT's own achieved
state and reads no clip at all, so a clip-free library is no obstacle to it.

The consequence was not cosmetic: the measured arm was IMPOSSIBLE on every free-form course,
which is every course this pipeline uses. It failed at command construction, after the
simulator had launched, with a message asserting the arm needed a clip.
"""

from __future__ import annotations

import types

import pytest

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (
    FmTrackingCommand,
)


class _Cfg:
    """Only the knobs the refusal reads, at their clip-free-legal values."""

    def __init__(self, history_from: str) -> None:
        """Set every knob legal except the arm under test."""
        self.history_from = history_from
        self.seed_history_from_clip = "never"
        self.random_start_phase = False
        self.random_start_before_terrain_start = False
        self.path = "clips/box2_v2spawn_train.h5"


def _check(arm: str) -> None:
    """Run the refusal for one arm; raises ValueError when it refuses."""
    obj = types.SimpleNamespace(cfg=_Cfg(arm),
                                _CLIP_PHASE_KNOBS=FmTrackingCommand._CLIP_PHASE_KNOBS)
    FmTrackingCommand._refuse_clip_phase_knobs(obj)


@pytest.mark.parametrize("arm", ["generator", "measured"])
def test_the_clip_free_arms_are_admitted(arm: str) -> None:
    """Both read only things a clip-free library has: the generator's own output, or the robot."""
    _check(arm)


def test_teacher_forcing_is_still_refused() -> None:
    """``clip`` feeds the CLIP's window as history, a per-step gather from a file with none."""
    with pytest.raises(ValueError, match="history_from"):
        _check("clip")


@pytest.mark.parametrize("knob,bad", [("seed_history_from_clip", "auto"),
                                      ("random_start_phase", True),
                                      ("random_start_before_terrain_start", True)])
def test_the_other_three_knobs_are_unchanged(knob: str, bad) -> None:
    """Narrowing one entry must not loosen the rest -- each is still a single required value."""
    obj = types.SimpleNamespace(cfg=_Cfg("generator"),
                                _CLIP_PHASE_KNOBS=FmTrackingCommand._CLIP_PHASE_KNOBS)
    setattr(obj.cfg, knob, bad)
    with pytest.raises(ValueError, match=knob):
        FmTrackingCommand._refuse_clip_phase_knobs(obj)
