# Copyright 2026 Zachary Olkin. All rights reserved.

"""Every buffer the trajectory commands INDEX must be one they allocate.

`FmTrackingCommand.reference_window_obs`'s `plan` arm read `self._pred[ar, idx]`. The deploy
twin, `GenTrackerPolicy._serve_nodes`, calls its buffer `_pred` -- and the line was written from
it -- but on the Isaac side the buffer is `_pred_window`, so the attribute simply did not exist.
It raised `AttributeError` the moment the observation manager sized the term, i.e. before any
policy ran, and nothing caught it: the `window_anchor` A/B exercised the arm through the TEACHER
(`MotionSequenceCommand`), which does not take that branch, and `test_window_anchor.py` never
constructs an `FmTrackingCommand` at all.

Instantiating one needs a full IsaacLab env, which is far too heavy for what is really a
name-resolution question. So this reads the source instead: every `self._x[...]` SUBSCRIPT in
these classes must name an attribute the class (or its base) assigns somewhere. That is exactly
the shape of the bug -- a typo'd buffer name on a rarely-taken branch -- and it is checked
without a simulator.

Deliberately limited to subscripted reads. `self._x` in any other position is usually a method
or a scalar and would need real type inference to judge; a subscript is almost always a buffer,
and a buffer is the thing that gets renamed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_CMD_DIR = (pathlib.Path(__file__).resolve().parent.parent
            / "robot_rl" / "tasks" / "manager_based" / "robot_rl"
            / "mdp" / "commands" / "traj_tracking")

#: The modules CHECKED. Both are large, both were edited by the merge, and both take branches
#: that only some configurations reach.
_CHECK_MODULES = ["fm_tracking_cmd.py", "motion_sequence_cmd.py"]

#: The modules that CONTRIBUTE vocabulary -- the whole inheritance chain, since an attribute
#: assigned in a base class is legitimately indexed in a subclass. Kept separate from the list
#: above so widening the chain does not silently widen what is asserted on.
_VOCAB_MODULES = _CHECK_MODULES + ["base_trajectory_cmd.py"]


def _assigned_attrs(tree: ast.AST) -> set[str]:
    """Every ``self.<name>`` ever assigned, annotated, or augmented in a tree."""
    out: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            # Unpacking: a, self.b = ...
            stack = [t]
            while stack:
                cur = stack.pop()
                if isinstance(cur, (ast.Tuple, ast.List)):
                    stack.extend(cur.elts)
                elif (isinstance(cur, ast.Attribute) and isinstance(cur.value, ast.Name)
                      and cur.value.id == "self"):
                    out.add(cur.attr)
    return out


def _defined_names(tree: ast.AST) -> set[str]:
    """Method and property names, plus class-level names -- readable but not 'assigned'."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
    return out


def _subscripted_self_reads(tree: ast.AST) -> list[tuple[str, int]]:
    """``(attr, lineno)`` for every ``self.<attr>[...]`` read."""
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        v = node.value
        if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                and v.value.id == "self"):
            out.append((v.attr, v.lineno))
    return out


@pytest.fixture(scope="module")
def vocabulary() -> set[str]:
    """Everything legitimately readable as ``self.<name>`` across the command modules."""
    known: set[str] = set()
    for name in _VOCAB_MODULES:
        tree = ast.parse((_CMD_DIR / name).read_text(encoding="utf-8"))
        known |= _assigned_attrs(tree) | _defined_names(tree)
    # Base-class machinery these commands inherit from IsaacLab and legitimately index.
    known |= {"cfg", "robot", "ctx", "num_envs", "device", "command", "metrics"}
    return known


@pytest.mark.parametrize("module", _CHECK_MODULES)
def test_every_indexed_buffer_is_one_the_class_allocates(module, vocabulary):
    """A subscripted ``self._x`` must name something assigned somewhere in these classes."""
    tree = ast.parse((_CMD_DIR / module).read_text(encoding="utf-8"))
    unknown = sorted({(a, ln) for a, ln in _subscripted_self_reads(tree)
                      if a not in vocabulary})
    assert not unknown, (
        f"{module} indexes attributes that are never assigned: "
        + ", ".join(f"self.{a} (line {ln})" for a, ln in unknown)
        + ". A typo'd buffer name on a rarely-taken branch raises AttributeError only when "
          "that branch runs, which for an observation term is at env construction -- and for "
          "the window_anchor=plan arm that was never in CI.")


def test_the_plan_arm_indexes_the_prediction_buffer_by_its_real_name():
    """The specific regression: the plan arm must read ``_pred_window``, not ``_pred``.

    Pinned by name as well as by the general rule above, because this one has a plausible
    wrong answer sitting in another file -- the deploy runtime's buffer really is called
    ``_pred`` -- so the next person porting a line between the two can make it again.
    """
    src = (_CMD_DIR / "fm_tracking_cmd.py").read_text(encoding="utf-8")
    assert "self._pred[" not in src, (
        "fm_tracking_cmd.py indexes self._pred, which this class does not define. The Isaac "
        "buffer is self._pred_window; self._pred is the DEPLOY runtime's name for it "
        "(transfer/deploy/gen_tracker.py).")
    assert "self._pred_window[ar, idx]" in src, "the plan arm's gather is missing"
