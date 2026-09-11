# Copyright 2026 Zachary Olkin. All rights reserved.

"""The best-of-K selection score, where the two critic heads are combined.

``live_select.selection_score`` is the one place ``P(fail)`` and ``G_hat`` become a single
orderable number, and it is worth pinning for the same reason the strict criterion is: a wrong
ranking rule does not crash, it publishes a success rate.

What is pinned, each because the alternative is a plausible wrong number:

* ``w_fail = 1`` returns ``p_fail`` **exactly**, ignoring every other argument.  That arm is
  the control every pre-quality-head number was measured under, so it has to REPRODUCE rather
  than approximate -- and the short-circuit is what makes that true by construction instead of
  by two expressions happening to agree at a boundary;
* the n-step return follows ``G_t = (1-g) R_t + g G_{t+1}``, so a candidate's MEASURED chunk
  enters exactly and only its tail is a model output;
* the ``+/-1.0`` terminal sentinels strictly BOUND the survivor range, which is
  ``[-w_track, w_fail]``.  A strict violation is scored with those sentinels, exactly as a
  fall is, because it IS a failure -- it is a hard TERMINATION rather than a graded penalty,
  so nothing about it appears in this function.  Grading it would have made the failure
  definition partly a reward and left two things deciding what "failed" means;
* a missing quality head RAISES below ``w_fail = 1`` rather than substituting a zero, because
  a zero ``G_hat`` is the worst possible return and would silently rank every candidate as
  maximally bad -- reading as a broken selector rather than an absent head.

Also recorded here as a measurement rather than an assertion: at ``gamma = 0.95`` the measured
chunk is worth ``w_track * (1 - gamma)`` of the score against the predicted tail's
``w_track * gamma``, a ratio of ``gamma / (1 - gamma) = 19``.  That is what a 1-step lookahead
IS, not a weighting choice -- but it does mean the measured term is a small exact correction
rather than the thing deciding a ranking.

No MuJoCo model, no critic and no torch: the function takes floats, so its algebra is testable
without a GPU or a 20 MB scene.
"""

from __future__ import annotations

import importlib.util
import math
import os

import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loaded BY PATH rather than by name. `live_select` sits in `transfer/sim`, which is not a
# package, and importing it by bare name depends on whether some other suite has already put
# that directory on `sys.path` -- which makes the import order decide whether this file runs.
_spec = importlib.util.spec_from_file_location(
    "_live_select_under_test", os.path.join(_SIM, "live_select.py"))
_ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ls)

_GAMMA = 0.95


def _score(g, rewards):
    """``selection_score`` with the fixed discount, to keep the cases readable."""
    return _ls.selection_score(g, rewards, gamma=_GAMMA)


def test_the_return_follows_the_one_step_identity():
    """``G_hat_n = (1-g) R + g G_hat`` for one measured chunk, and ``G_hat`` for none.

    The identity survives the cut intact -- only the survival blend around it is gone, so the
    score is now `-G_hat_n` rather than `-(w_track * G_hat_n - w_fail * P_fail)`.
    """
    g = 0.9
    assert _score(g, []) == pytest.approx(-g, abs=1e-12)

    one = _score(g, [0.5])
    assert one == pytest.approx(-((1.0 - _GAMMA) * 0.5 + _GAMMA * g), abs=1e-12)

    # Two chunks discount the second, which is what makes this an n-step estimate rather than
    # a mean over whatever happened to be rolled.
    two = _score(g, [0.5, 0.5])
    want2 = -((1.0 - _GAMMA) * (0.5 + _GAMMA * 0.5) + _GAMMA ** 2 * g)
    assert two == pytest.approx(want2, abs=1e-12)


def test_the_measured_term_is_small_exact_and_the_size_the_discount_says():
    """A MEASUREMENT: the measured chunk is worth ``(1 - gamma)`` of the score.

    It was ``w_track * (1 - gamma)`` while a survival term took the rest of the weight; with
    one term the factor is just ``1 - gamma``. Small is not pointless -- it is exact where
    ``G_hat`` is a prediction -- but it does mean the measured term cannot make or break a
    ranking on its own.
    """
    lo, hi = _score(0.9, [0.0]), _score(0.9, [1.0])
    assert abs(hi - lo) == pytest.approx(1.0 - _GAMMA, abs=1e-12)
    # ...and the predicted tail is gamma/(1-gamma) times it, exactly.
    tlo, thi = _score(0.0, [0.5]), _score(1.0, [0.5])
    assert abs(thi - tlo) / abs(hi - lo) == pytest.approx(_GAMMA / (1.0 - _GAMMA), rel=1e-9)


def test_the_terminal_sentinels_strictly_bound_the_survivor_range():
    """A failed candidate must outrank no survivor, and an arrival be outranked by none.

    The caller scores a fall AND a strict violation at +1.0 and an arrival at -1.0, so those
    have to bound what the survivor branch can return. With the survival blend gone the range
    is simply `-G_hat_n` on the reward's normalized scale, which is a TIGHTER bound than the
    old ``[-w_track, w_fail]`` -- so the sentinels still hold, with equality at the endpoints.
    """
    best = _score(1.0, [1.0])       # nothing can beat this
    worst = _score(0.0, [0.0])
    assert -1.0 <= best <= worst <= 1.0


def test_a_violation_is_not_a_term_in_the_score():
    """The strict latch reaches the ranking as a TERMINATION, never as a graded penalty.

    Pinned as a signature test because the graded version existed and was removed: a penalty
    argument would make the failure definition partly a reward, and then the score and
    ``schema.derive_success`` would each hold half of what "failed" means.
    """
    import inspect

    params = set(inspect.signature(_ls.selection_score).parameters)
    assert not params & {"violated", "violation_penalty"}, (
        "selection_score has regrown a violation term; a violation is a termination, and the "
        "caller scores it with the fall sentinel")


def test_the_k1_arm_is_scored_by_the_same_rule_as_the_candidates():
    """The K=1 baseline must fold a tracking probe, or the paired delta measures the RULE.

    A SOURCE check, because the behaviour needs a MuJoCo rig and this defect needs catching
    without one. It is the same heuristic-pin idiom as
    ``test_stage_grid_forwards_every_flag_the_cell_stage_reads``.

    This fired for real. The ``k <= 1`` fast path called ``_chunk`` without ``probe=``, so the
    baseline was scored on falls alone while every ``k > 1`` candidate was scored on the strict
    criterion. Measured over 63 cells x 40 episodes: K=1 came back ``fall 0.211,
    violation 0.000`` and K=4 ``violation 0.255, fall 0.000`` -- zero of each in the other arm,
    which is impossible under one rule -- and it published best-of-K as COSTING 0.042 with a
    tile-clustered interval that excluded zero. A plausible, significant, entirely wrong number.
    """
    import inspect
    import re

    src = inspect.getsource(_ls.run_episode_selected)
    m = re.search(r"if k <= 1:(.*?)\n        snap = ", src, re.S)
    assert m, "the k <= 1 fast path moved; this pin needs updating"
    assert "probe=" in m.group(1), (
        "the K=1 arm calls _chunk without a probe, so it is scored on falls alone while every "
        "k > 1 candidate is scored on the strict criterion -- the paired K=1-vs-K=4 delta then "
        "measures the failure RULE, not the selection")


def test_both_episode_drivers_return_the_same_arity():
    """Every return of both drivers hands back 5 elements, and the caller unpacks 5.

    There is ONE call site and it dispatches to either driver, so a return that drops an
    element ``ValueError``s at run time -- after the scene is built and the policy loaded,
    minutes into a cell. No unit test can see it, because none of them call ``run()``; this
    reads the AST instead. It caught a real miss: adding the quality element to both drivers
    left the caller unpacking four, so every cell would have died on its first episode.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(os.path.join(_SIM, "live_select.py")).read_text())
    drivers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name in ("run_episode_selected", "run_episode_q_selected")]
    assert len(drivers) == 2, "an episode driver was renamed; this pin needs updating"
    for fn in drivers:
        # Only the driver's OWN returns. `ast.walk` descends into nested helpers, and both
        # drivers define per-step callbacks that legitimately `return` something of their own
        # -- counting those made this pin fail on a helper rather than on the contract.
        nested = {id(n) for d in ast.walk(fn)
                  if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d is not fn
                  for n in ast.walk(d)}
        for r in [n for n in ast.walk(fn)
                  if isinstance(n, ast.Return) and id(n) not in nested]:
            width = len(r.value.elts) if isinstance(r.value, ast.Tuple) else 1
            assert width == 5, f"{fn.name} line {r.lineno} returns {width}, not 5"
    # ...and the single call site unpacks exactly that many.
    src = pathlib.Path(os.path.join(_SIM, "live_select.py")).read_text()
    assert "v, f, nr, sc, q = fn(" in src, "the dispatch no longer unpacks 5 elements"

