# Copyright 2026 Zachary Olkin. All rights reserved.

"""The approach controller and the dwell goal are OPT-IN, and every existing course is unchanged.

The whole point of adding them as new knobs rather than as edits to the existing ones is that a
wall-field course cannot change what any other arm measures. That is a claim about defaults, and
defaults are exactly the kind of thing that drifts silently -- a later edit that flips one of
these on in a shared base class would re-point every fine-tune at a different command law and
nothing in the numbers would say so, because the commanded speed is not an artifact anyone
diffs.

These parse the SOURCE rather than constructing the env cfgs: building one needs IsaacLab and a
simulator app, so a test that did would not run in CI and would not run here. What can be
checked without a simulator is the declared default and the absence of any override, and that is
precisely where this drift would appear.
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "robot_rl"
_CMD_CFG = (_ROOT / "tasks/manager_based/robot_rl/mdp/commands/traj_tracking"
            / "fm_tracking_cmd_cfg.py")
_G1 = _ROOT / "tasks/manager_based/robot_rl/g1"

#: The knobs that must be inert unless a course opts in, and the value that means "off".
OFF_DEFAULTS: dict[str, float] = {
    "freeform_approach_kp": 0.0,
    "freeform_approach_kd": 0.0,
    "freeform_stop_radius_m": 0.0,
    "freeform_approach_min_speed": 0.0,
}


def _annotated_defaults(path: pathlib.Path) -> dict[str, float]:
    """Every ``name: type = <number>`` declared at class scope in a module.

    Args:
        path: The module to parse.

    Returns:
        ``{name: value}`` for numeric defaults.
    """
    out: dict[str, float] = {}
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        if not isinstance(node.target, ast.Name):
            continue
        try:
            out[node.target.id] = float(ast.literal_eval(node.value))
        except (ValueError, TypeError):
            continue
    return out


def test_the_approach_controller_is_declared_off() -> None:
    """Each new knob exists and its declared default disables it."""
    declared = _annotated_defaults(_CMD_CFG)
    for name, off in OFF_DEFAULTS.items():
        assert name in declared, (
            f"{name} is gone from FmTrackingCommandCfg. If it was renamed, this test and "
            "freeform_goal_dwell's constructor check both need to follow it.")
        assert declared[name] == off, (
            f"{name} now defaults to {declared[name]}, not {off}. That silently switches the "
            "range-driven forward command on for EVERY free-form course -- boxes, updown, "
            "indist, multi -- whose numbers were all measured with a constant per-episode vx.")


def test_no_g1_task_turns_the_approach_controller_on() -> None:
    """No task cfg assigns the approach knobs.

    Only ``g1_finetune_env_cfg.py`` may, and only in ``G1WallsFinetuneEnvCfg`` -- checked
    below. The point is that switching them on is a visible edit to this test, not something a
    base class can do on a course's behalf.
    """
    #: The wall course is the ONE task that opts in, and it is listed here so that switching
    #: the controller on anywhere else is a visible edit to this line.
    allowed: set[str] = {"g1_finetune_env_cfg.py"}
    offenders: list[str] = []
    for path in sorted(_G1.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                name = t.attr if isinstance(t, ast.Attribute) else getattr(t, "id", None)
                if name in OFF_DEFAULTS and path.name not in allowed:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, (
        "these task cfgs switch the approach controller on: " + ", ".join(offenders)
        + ". Every free-form arm measured so far used the constant per-episode vx, so this "
        "changes what they mean. Add the file to ALLOWED here if it is deliberate.")


def test_the_dwell_goal_is_a_separate_term() -> None:
    """``freeform_goal_reached`` is still a plain function, and dwell is its own class.

    The separation IS the guarantee: existing courses keep calling the function, so they cannot
    be reached by anything the dwell term does. A refactor that merged them -- say by giving
    ``freeform_goal_reached`` a ``dwell_s`` parameter -- would put every course back on one
    code path and this test is what should stop it.
    """
    src = (_ROOT / "tasks/manager_based/robot_rl/mdp/terminations/terminations.py").read_text()
    tree = ast.parse(src)
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    assert "freeform_goal_reached" in funcs, (
        "freeform_goal_reached is no longer a module-level function; every existing course "
        "wires it and a change of kind changes how the manager calls it.")
    assert "freeform_goal_dwell" in classes, (
        "freeform_goal_dwell must be a ManagerTermBase subclass -- it needs per-env state that "
        "survives across steps and is cleared on reset, which a plain function cannot hold.")
    # The function's OWN source, bounded by the AST rather than by splitting on "\ndef " --
    # which ran straight into the class that follows it and failed on its docstring.
    reached = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "freeform_goal_reached")
    body = ast.get_source_segment(src, reached) or ""
    assert "dwell" not in body, (
        "freeform_goal_reached has grown dwell logic. Keep the two terms separate so an "
        "opt-in course cannot change what the traverse courses measure.")


def test_only_the_walls_cfg_opts_in() -> None:
    """The approach knobs are assigned in ``G1WallsFinetuneEnvCfg`` and nowhere else.

    The file-level allow-list above is necessarily coarse -- the walls course lives in the same
    module as every other free-form cfg -- so this narrows it to the class. A base class in
    that file switching the controller on would reach boxes, updown, indist and multi, whose
    numbers were all measured with a constant per-episode vx.
    """
    path = _G1 / "g1_finetune_env_cfg.py"
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in ast.walk(cls):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                name = t.attr if isinstance(t, ast.Attribute) else getattr(t, "id", None)
                if name in OFF_DEFAULTS and cls.name != "G1WallsFinetuneEnvCfg":
                    offenders.append(f"{cls.name}.{name}")
    assert not offenders, (
        "the approach controller is switched on outside G1WallsFinetuneEnvCfg: "
        + ", ".join(sorted(set(offenders))))
