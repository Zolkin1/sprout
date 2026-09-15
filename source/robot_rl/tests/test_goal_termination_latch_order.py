# Copyright 2026 Zachary Olkin. All rights reserved.

"""The env's goal TERMINATION is latched before the episode is harvested, not after.

``attach_goal_termination`` makes the ENVIRONMENT the single source of ``reached_goal`` on a
course with a dwell rule, because a "stay in the disc for N consecutive steps" rule cannot be
re-derived offline as safely as a latch can. That was the right call and it fixed a 0% arrival
rate. It left a second, quieter off-by-one behind.

The harvest runs INSIDE ``env.step``: ``RolloutEpisodeTracker.__call__`` wraps the reset event,
and the reset event is dispatched by the env while stepping. A latch applied after ``env.step``
RETURNS is therefore applied to the accumulator of the episode that has just begun. The ending
episode was written one call earlier, with the flag still false.

What that looked like, measured on ``eval_ft_walls_on_walls`` at 0.40 m/s:

    ordinal 1: n=384 reached=0.0000        <- no predecessor to inherit from
    ordinal 2: n=384 reached=0.8906
    ordinal 3: n=384 reached=0.9115

with ``min_dist_to_goal_m`` (0.779 vs 0.775), ``steps_survived`` (2030 vs 1971) and the fall
rate all matching to within noise between ordinals 1 and 2. NOTHING ELSE IN THE ROW MOVED. It
reads as a policy that walks to the goal and cannot stop, rather than as a bookkeeping error --
and the campaign's own reduction, FIRST-PER-ENV, is exactly the one with no predecessor to
inherit from, so it read a flat 0.0000 and the paired delta came out +0.0000 with 1536 unchanged
pairs and zero discordant. A null result, from a clerical error, at the point of publication.

These are source-order checks. Exercising the real ordering needs a simulator, a scene and a
policy; the ORDER is the thing that was wrong and it is visible without any of that.
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TRACKER = _ROOT / "scripts/rsl_rl/rollout_outcomes.py"
_RECORD = _ROOT / "scripts/rsl_rl/record_rollouts.py"


def _method(path: pathlib.Path, cls_name: str, fn_name: str) -> ast.FunctionDef:
    """One method's AST node.

    Args:
        path: Module to parse.
        cls_name: Enclosing class.
        fn_name: Method name.

    Returns:
        Its ``ast.FunctionDef``.
    """
    tree = ast.parse(path.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == fn_name)


def _calls_in_order(fn: ast.FunctionDef) -> list[str]:
    """The method names called in the body, in source order.

    Args:
        fn: A function node.

    Returns:
        Attribute names of every ``self.x(...)`` style call.
    """
    out: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            out.append((node.func.attr, node.lineno))
    return [name for name, _ in sorted(out, key=lambda p: p[1])]


def test_the_latch_runs_before_the_harvest() -> None:
    """In ``__call__``, ``_latch_goal_termination`` precedes ``_harvest``.

    This is the whole fix. Reversing the two lines restores an arrival rate that is wrong by
    exactly one episode per env and looks entirely plausible.
    """
    calls = _calls_in_order(_method(_TRACKER, "RolloutEpisodeTracker", "__call__"))
    assert "_latch_goal_termination" in calls, (
        "RolloutEpisodeTracker.__call__ no longer latches the env's goal termination; the "
        "accumulator would then score its own re-derivation of a dwell, which is the rule "
        "attach_goal_termination exists to refuse.")
    assert "_harvest" in calls, "the reset hook must still harvest"
    assert calls.index("_latch_goal_termination") < calls.index("_harvest"), (
        "the goal termination must be latched BEFORE the episode is harvested -- the harvest "
        "writes reached_goal and clears the accumulator, so a latch after it lands on the "
        "NEXT episode.")


def test_the_latch_is_not_also_applied_after_env_step() -> None:
    """``record_rollouts``'s step loop does not call ``note_goal_termination``.

    Where it used to. Leaving it there in addition would latch the same step twice: once
    correctly inside the hook, and once onto the successor episode -- which restores the
    original bug for every episode after the first while making the first one look fixed.
    """
    tree = ast.parse(_RECORD.read_text())
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "note_goal_termination"
    ]
    assert not offenders, (
        f"record_rollouts.py calls note_goal_termination at line(s) {offenders}. The latch "
        "belongs in RolloutEpisodeTracker's reset hook, which runs inside env.step and "
        "therefore before the harvest; anything after env.step returns is one episode late.")


def test_the_latch_reads_the_term_dones_buffer() -> None:
    """It goes through ``termination_manager.get_term``, which the auto-reset does not clear.

    ``TerminationManager.reset`` calls each CLASS term's own ``reset`` -- which is what zeroes
    ``freeform_goal_dwell``'s consecutive counter -- but never touches ``_term_dones``. So the
    value read here still describes the step just taken, even for the envs that have already
    been respawned. Reading the term OBJECT's internal counter instead would get a zero.
    """
    fn = _method(_TRACKER, "RolloutEpisodeTracker", "_latch_goal_termination")
    body = ast.get_source_segment(_TRACKER.read_text(), fn) or ""
    assert "termination_manager.get_term" in body, (
        "the latch must read the manager's per-step term buffer, not the term object's own "
        "state -- the env resets the latter before this hook returns.")
    assert "_goal_term_name" in body, (
        "the latch must be a no-op unless a course attached a goal termination; every "
        "non-dwell course still folds the goal in the accumulator.")
