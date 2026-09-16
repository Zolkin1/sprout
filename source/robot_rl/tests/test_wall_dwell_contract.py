# Copyright 2026 Zachary Olkin. All rights reserved.

"""The RUNTIME dwell rule and the OFFLINE dwell rule are the same rule.

Three things decide whether the wall course measures what it claims: the termination that ends
the episode (``freeform_goal_dwell``), the goal SPEC that ``eval.accumulate`` scores
``reached_goal`` from, and the debug marker that draws the region. If any two disagree, the arm
trains toward one definition and is reported against another -- and nothing in the numbers says
so, because both definitions produce a plausible success rate.

This is the same failure the repo has paid for twice already, both recorded in
``source/motion_gen/motion_gen/terrain/CLAUDE.md``: ``height_tol_m`` meaning opposite things at
zero in the termination and in ``in_goal``, which scored an arm 0.0000 over 8400 episodes AND
zeroed the reward's success term for five training iterations; and the goal anchor being a
pelvis height in one place and a surface height in the other.

Parses the source rather than constructing the env: building one needs a simulator, and what
must agree here are declared constants.
"""

from __future__ import annotations

import ast
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TERRAIN = _ROOT / "motion_gen/motion_gen/terrain/wall_field.py"
_ENV_CFG = (_ROOT / "robot_rl/robot_rl/tasks/manager_based/robot_rl/g1"
            / "g1_finetune_env_cfg.py")
_SPEC = _ROOT.parent / "logs/ft_walls/goal_regions.json"


def _module_constants(path: pathlib.Path) -> dict[str, float]:
    """Numeric module-level constants declared as ``NAME: type = <number>``.

    Args:
        path: Module to parse.

    Returns:
        ``{name: value}``.
    """
    out: dict[str, float] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                out[node.target.id] = float(ast.literal_eval(node.value))
            except (ValueError, TypeError, AttributeError):
                continue
    return out


def test_the_terrain_module_owns_the_dwell_numbers() -> None:
    """``wall_field`` declares both, and they are positive."""
    c = _module_constants(_TERRAIN)
    assert c.get("DWELL_RADIUS_M", 0.0) > 0.0, "wall_field must declare DWELL_RADIUS_M"
    assert c.get("DWELL_S", 0.0) > 0.0, "wall_field must declare DWELL_S"


def test_the_task_imports_the_dwell_numbers_rather_than_restating_them() -> None:
    """``g1_finetune_env_cfg`` binds its dwell constants to the terrain module's.

    A literal here would be a second declaration of the region, and the two would drift the
    first time either moved -- which is precisely what happened when the radius went from
    0.5 to 1.0 and three docstrings kept the old number.
    """
    src = _ENV_CFG.read_text()
    assert "from motion_gen.terrain.wall_field import DWELL_RADIUS_M" in src, (
        "the task must import the radius from the terrain module")
    assert "WALLS_GOAL_RADIUS_M: float = WALL_DWELL_RADIUS_M" in src, (
        "WALLS_GOAL_RADIUS_M must BE the terrain module's radius, not a copy of its value")
    assert "WALLS_DWELL_S: float = WALL_DWELL_S" in src, (
        "WALLS_DWELL_S must BE the terrain module's dwell time, not a copy of its value")


def test_the_controller_aims_at_the_CENTRE_and_the_steering_stops_at_the_DISC() -> None:
    """Two radii, and the ordering between them is the whole contract.

    This test used to pin ``WALLS_STOP_RADIUS_M == WALL_DWELL_RADIUS_M`` on the argument that
    the command should say one thing: the moment you are in the circle, stand. The argument was
    sound and the consequence was not measured. With the two equal, the P term's setpoint is
    ``to_go == stop_r`` and the hard cut fires at the same range, so the law's equilibrium is a
    SHELL on the scoring boundary. Over 12,598 uncensored episodes the closest approach came out
    as a tight annulus -- p10 0.66, median 0.80, p90 0.89 m against a 1.0 m disc -- with ~1% of
    episodes ever reaching within 0.5 m of the goal, and the ~8% that stopped in the outer 0.1 m
    held the dwell 0.611 of the time at 1.50 m/s against 0.884 for those that stopped deeper.

    It also meant the course skipped its own drill: half the goals sit 0.5-1.5 m from a wall and
    the robot arrives from the near side, so stopping 1.0 m short left it ~1.5 m off the wall it
    is supposed to be learning to stop for.

    So the stop radius is now the COAST distance and the robot rests near the centre. What must
    hold, and what this pins:

    * the stop radius is STRICTLY INSIDE the disc -- otherwise nothing changed;
    * it is not zero, because the floor-to-zero step is what makes a stop a stop rather than a
      crawl through the 0.05-0.3 m/s band the generator barely trained on;
    * the YAW-quiet radius is the disc, not the stop radius. Left equal to 0.25 m, an env that
      comes to rest off-centre has ``vx`` clamped to zero by the along-track term while ``rng``
      is still above the stop radius -- so it would stand inside the scored region and be told
      to keep turning, one drift from resetting a dwell it had earned.

    ``freeform_goal_dwell.__init__`` refuses the other direction (a stop radius LARGER than the
    disc), which would leave a band where the robot is told to stand and scored as not arrived.
    """
    src = _ENV_CFG.read_text()
    c = _module_constants(_TERRAIN)
    stop = _module_constants(_ENV_CFG).get("WALLS_STOP_RADIUS_M")
    assert stop is not None, "WALLS_STOP_RADIUS_M is gone"
    assert 0.0 < stop < c["DWELL_RADIUS_M"], (
        f"WALLS_STOP_RADIUS_M is {stop} against a {c['DWELL_RADIUS_M']} m disc. It must be "
        "strictly inside the disc (so the robot aims at the centre) and strictly positive (so "
        "the command is a ramp-then-step rather than a decay into the walking deadband).")
    assert "WALLS_YAW_QUIET_RADIUS_M: float = WALL_DWELL_RADIUS_M" in src, (
        "the yaw-quiet radius must BE the terrain module's dwell radius, not a copy of its "
        "value and not the stop radius -- see this test's docstring for what happens when the "
        "steering is allowed to stay live inside the scored region.")
    assert _module_constants(_ENV_CFG).get("WALLS_APPROACH_TARGET_R_M") == 0.0, (
        "WALLS_APPROACH_TARGET_R_M must be 0.0 -- the controller aims at the goal POINT. Set "
        "equal to the stop radius it aims at a shell around the goal, which is the arrangement "
        "that produced the 0.80 m median closest approach this docstring describes.")
    assert "freeform_approach_target_r_m = WALLS_APPROACH_TARGET_R_M" in src, (
        "G1WallsFinetuneEnvCfg must wire the aim point; declaring the constant without "
        "assigning it leaves the P term's setpoint on the stop radius.")
    assert "freeform_yaw_quiet_radius_m = WALLS_YAW_QUIET_RADIUS_M" in src, (
        "G1WallsFinetuneEnvCfg must actually wire the yaw-quiet radius; declaring the constant "
        "and not assigning it leaves the steering gated on the 0.25 m stop radius.")


def test_the_goal_spec_declares_the_same_dwell_as_the_task() -> None:
    """The written ``goal_regions.json`` carries the terrain module's numbers.

    This is the offline half. ``eval.accumulate`` reads these two fields and, when they are
    present, scores ``reached_goal`` as "held the disc" instead of "was ever inside the
    half-plane". If the spec lost them, the reward would silently go back to paying for
    crossing the goal LINE -- the pass-through behaviour the course exists to remove -- while
    the episode still ended on the dwell.
    """
    if not _SPEC.exists():
        import pytest
        pytest.skip(f"{_SPEC} absent; build it with make_walls_terrain.py")
    default = json.loads(_SPEC.read_text())["default"]
    c = _module_constants(_TERRAIN)
    assert default.get("dwell_radius_m") == c["DWELL_RADIUS_M"], (
        "goal_regions.json disagrees with wall_field.DWELL_RADIUS_M -- rebuild the library")
    assert default.get("dwell_s") == c["DWELL_S"], (
        "goal_regions.json disagrees with wall_field.DWELL_S -- rebuild the library")


def test_a_partial_dwell_rule_is_refused() -> None:
    """``accumulate`` raises when only one of the two dwell fields is set.

    One alone is the dangerous case: a radius with no duration, or a duration with no radius,
    would silently fall through to the pass-through rule while the spec reads like a stopping
    course to anyone who opens it.
    """
    src = (_ROOT.parent / "scripts/rsl_rl/eval/accumulate.py").read_text()
    assert "partial dwell rule" in src, (
        "accumulate must refuse a half-declared dwell rule rather than falling back to the "
        "half-plane latch")
