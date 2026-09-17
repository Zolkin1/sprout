# Copyright 2026 Zachary Olkin. All rights reserved.
"""``height_tol_m`` means the same thing at both ends, and at zero.

It used to mean OPPOSITE things: `terminations.freeform_goal_reached` guards with
``if height_tol_m > 0.0``, so 0 skips the test; `goal_region.in_goal` applied
``|height_err| <= height_tol_m`` unconditionally, so 0 was an EXACT equality on a continuous
quantity and never satisfiable. One field, two readings, both reachable from a single run.

What that cost, once: a varied-terrain spec declaring 0 read 0.0000 over 8400 episodes and
zeroed the progress reward's success term for five whole training iterations. Both tells
pointed away from the cause -- episodes ended at ~474 steps rather than timing out, so the
TERMINATION was plainly firing while `in_goal` could not fire at all.

Zero now means "off" in both places. This pins that, and pins that a positive tolerance is
still a real band.
"""

from __future__ import annotations

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region as gr


def _at(height_err: float, tol_h: float) -> bool:
    """Whether `in_goal` admits a base at this height error under this tolerance."""
    tol = gr.GoalTolerances(approach_margin_m=1.0, half_width_m=2.5, height_tol_m=tol_h)
    out = gr.in_goal(torch.zeros(1), torch.zeros(1), torch.tensor([height_err]), tol)
    return bool(out[0])


def test_zero_turns_the_height_test_OFF() -> None:
    """Not an exact-equality test, which nothing could ever satisfy."""
    assert _at(0.0, 0.0), "even an exact match failed, so the test is an equality"
    assert _at(0.79, 0.0), (
        "0 must disable the height test; 0.79 m is a standing robot's pelvis above a "
        "surface-height goal, which is the case that read 0.0000 over 8400 episodes")


def test_a_positive_tolerance_is_still_a_BAND() -> None:
    """Relaxing zero must not relax everything."""
    assert _at(0.4, 0.5)
    assert not _at(0.6, 0.5)


def test_the_two_ends_AGREE_at_zero() -> None:
    """The termination's guard and `in_goal`'s are the same predicate now.

    Read off the termination's source rather than duplicated, so a change to one that does not
    reach the other fails here.
    """
    import inspect

    from robot_rl.tasks.manager_based.robot_rl.mdp.terminations import terminations as term

    src = inspect.getsource(term.freeform_goal_reached)
    assert "float(height_tol_m) > 0.0" in src, (
        "the termination no longer treats 0 as OFF, so the two ends have diverged again")
    gsrc = inspect.getsource(gr.in_goal)
    assert "(height <= 0)" in gsrc, "in_goal no longer treats 0 as OFF"


#: Shortest box the shipped libraries ask the robot to climb onto, metres. A default band at
#: or above this cannot tell "on the box" from "on the ground beside it", which is the one
#: thing the height test exists for.
SHORTEST_WALKABLE_BOX_M: float = 0.250


def test_the_default_is_a_real_band_not_zero() -> None:
    """A spec that omits the field must not silently disable the test it exists for.

    Pinned as a RANGE rather than as a literal, because the default has a job at both ends and
    the number between them is a fitted quantity that has moved once already (0.5 -> 0.15, when
    the pad it was built from turned out to be 4-30x the tracker's own height error). Above the
    box it discriminates nothing; at zero it is off.
    """
    default = gr.GoalTolerances().height_tol_m
    assert default > 0.0, "a spec omitting the field would silently disable the height test"
    assert default < SHORTEST_WALKABLE_BOX_M, (
        f"default band {default} m does not discriminate a {SHORTEST_WALKABLE_BOX_M} m box")
