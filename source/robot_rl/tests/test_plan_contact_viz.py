# Copyright 2026 Zachary Olkin. All rights reserved.

"""The ribbon must draw the contact flag the REWARD gates on, from the same producer.

The whole value of colouring contacts in the debug viz is that you are looking at the thing
being graded.  If the ribbon computed its own flag -- from the ankle frame's speed, say,
rather than the slowest sole point -- it would show a different contact set from the one the
terrain-consistency reward uses, and the viz would quietly stop being evidence.

So ``FmTrackingCommand.plan_foot_contact`` is the single producer, and these are source-level
guards on that. They are source-level because ``_post_init`` needs a live scene: constructing
the term to test its wiring would need Isaac Sim, which is exactly what the rest of this file
tree avoids.
"""

from __future__ import annotations

import inspect
import pathlib


def _cmd_source() -> str:
    """The command term's source, located via the MODULE rather than by counting hops.

    ``..``-counting from a test file has silently broken paths in this repo before, and it is
    what CLAUDE.md asks not to do -- so ask the import system where the module actually is.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
        fm_tracking_cmd)
    return pathlib.Path(inspect.getsourcefile(fm_tracking_cmd)).read_text()


def test_the_ribbon_reads_the_shared_contact_producer():
    """Not its own finite difference, and not the ankle frame's speed."""
    src = _cmd_source()
    draw = src[src.index("def _draw_fm_prediction_ribbon"):]
    draw = draw[:draw.index("\n    def ")]
    assert "self.plan_foot_contact()" in draw, (
        "the ribbon must call plan_foot_contact, so the flag drawn is the flag graded")
    for forbidden in ("plan_contact(", "node_speed(", "torch.diff"):
        assert forbidden not in draw, (
            f"the ribbon computes {forbidden} itself; that is a second contact definition")


def test_the_producer_derives_its_dt_from_the_env():
    """A restated 0.02 would silently mis-scale every speed after a decimation change."""
    src = _cmd_source()
    fn = src[src.index("def plan_foot_contact"):]
    fn = fn[:fn.index("\n    def ")]
    assert "self._env.step_dt" in fn
    assert "0.02" not in fn


def test_the_producer_reads_the_soles_not_the_ankle_frames():
    """The collider model is feet-only and surface-sampled, which is what has soles at all."""
    src = _cmd_source()
    fn = src[src.index("def plan_collider_model"):]
    fn = fn[:fn.index("\n    def ")]
    assert "body_names=list(FOOT_BODIES)" in fn
    assert "PlanColliderModel" in fn


def test_the_contact_colour_is_distinguishable_from_every_frame_colour():
    """It is read AGAINST the frame colours, so it must not be one of them."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd \
        import DEBUG_CONTACT_COLOR
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking \
        .motion_sequence_cmd import _DEBUG_FRAME_COLOR_MAP, _DEBUG_FRAME_FALLBACK_COLOR

    others = list(_DEBUG_FRAME_COLOR_MAP.values()) + [_DEBUG_FRAME_FALLBACK_COLOR]
    for c in others:
        dist = sum((a - b) ** 2 for a, b in zip(DEBUG_CONTACT_COLOR, c)) ** 0.5
        assert dist > 0.35, f"contact colour is too close to {c} (distance {dist:.3f})"


def test_the_foot_body_names_come_from_one_place():
    """The ribbon indexes ``contact`` by FOOT_BODIES order, so it must import that order."""
    src = _cmd_source()
    assert "from motion_gen.plan_geometry import FOOT_BODIES" in src
    assert 'FOOT_BODIES.index(name)' in src
    # ...and it must not restate the names, which would let the two orders diverge.
    assert '"left_ankle_roll_link"' not in src.split("FOOT_BODIES")[-1]
