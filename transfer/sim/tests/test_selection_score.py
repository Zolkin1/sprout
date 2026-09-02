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


def _score(p, g, rewards, *, w_fail):
    """``selection_score`` with the fixed discount, to keep the cases readable."""
    return _ls.selection_score(p, g, rewards, w_fail=w_fail, gamma=_GAMMA)


@pytest.mark.parametrize("p_fail", [0.0, 0.3, 0.5, 1.0])
def test_w_fail_one_is_the_control_arm_and_ignores_everything_else(p_fail):
    """The survival-only arm is bit-identical to ``crit.p_fail``, whatever else is passed."""
    # Deliberately hostile arguments: a great G_hat and measured rewards. Neither may perturb
    # the control arm, or "the control reproduces" stops being true.
    assert _score(p_fail, 0.9, [0.8, 0.7], w_fail=1.0) == p_fail
    assert _score(p_fail, None, [], w_fail=1.0) == p_fail


def test_the_return_follows_the_one_step_identity():
    """``G_hat_n = (1-g) R + g G_hat`` for one measured chunk, and ``G_hat`` for none."""
    w_fail, p, g = 0.8, 0.2, 0.9
    w_track = 1.0 - w_fail

    no_reward = _score(p, g, [], w_fail=w_fail)
    assert no_reward == pytest.approx(-(w_track * g - w_fail * p), abs=1e-12)

    one_chunk = _score(p, g, [0.5], w_fail=w_fail)
    want = -(w_track * ((1.0 - _GAMMA) * 0.5 + _GAMMA * g) - w_fail * p)
    assert one_chunk == pytest.approx(want, abs=1e-12)

    # Two chunks discount the second, which is what makes this an n-step estimate rather than
    # a mean over whatever happened to be rolled.
    two = _score(p, g, [0.5, 0.5], w_fail=w_fail)
    want2 = -(w_track * ((1.0 - _GAMMA) * (0.5 + _GAMMA * 0.5) + _GAMMA ** 2 * g) - w_fail * p)
    assert two == pytest.approx(want2, abs=1e-12)


def test_the_measured_term_is_small_exact_and_the_size_the_discount_says():
    """A MEASUREMENT: the measured chunk is worth ``w_track * (1 - gamma)`` of the score.

    Small is not the same as pointless -- it is exact where ``G_hat`` is a prediction -- but it
    does mean the measured term cannot be what makes or breaks a ranking. Measured on the
    first real sweep, it also has the LARGEST raw spread across candidates of the three inputs
    (0.0315 against 0.0175 for ``p_fail`` and 0.0133 for ``G_hat``), so it discriminates well
    and is simply weighted down by the discount.
    """
    w_fail = 0.8
    lo = _score(0.2, 0.9, [0.0], w_fail=w_fail)
    hi = _score(0.2, 0.9, [1.0], w_fail=w_fail)
    assert abs(hi - lo) == pytest.approx((1.0 - w_fail) * (1.0 - _GAMMA), abs=1e-12)
    # ...and the predicted tail is gamma/(1-gamma) times it, exactly.
    tlo = _score(0.2, 0.0, [0.5], w_fail=w_fail)
    thi = _score(0.2, 1.0, [0.5], w_fail=w_fail)
    assert abs(thi - tlo) / abs(hi - lo) == pytest.approx(_GAMMA / (1.0 - _GAMMA), rel=1e-9)


def test_the_terminal_sentinels_strictly_bound_the_survivor_range():
    """A failed candidate must outrank no survivor, and an arrival be outranked by none.

    The caller scores a fall AND a strict violation at +1.0 and an arrival at -1.0, so those
    have to bound ``[-w_track, w_fail]``. They do for every ``w_fail`` in [0, 1], with equality
    only at the endpoints -- which is what lets the violation reuse the fall's sentinel instead
    of needing a wider one, and is why this function no longer takes a penalty at all.
    """
    for w_fail in (0.0, 0.5, 0.75, 0.8, 0.833, 1.0):
        best = _score(0.0, 1.0, [1.0], w_fail=w_fail)              # nothing can beat this
        worst = _score(1.0, 0.0, [0.0], w_fail=w_fail)
        assert -1.0 <= best <= worst <= 1.0, w_fail


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


def test_a_missing_quality_head_raises_rather_than_scoring_zero():
    """A zero ``G_hat`` is the WORST return, so a silent one is not a safe fallback."""
    with pytest.raises(ValueError, match="no quality head"):
        _score(0.2, None, [], w_fail=0.8)


def test_lower_is_better_in_both_heads():
    """Monotonicity, so ``argmin`` means what the caller thinks it means."""
    assert _score(0.1, 0.5, [], w_fail=0.8) < _score(0.9, 0.5, [], w_fail=0.8)
    assert _score(0.5, 0.9, [], w_fail=0.8) < _score(0.5, 0.1, [], w_fail=0.8)
    assert math.isfinite(_score(0.5, 0.5, [0.5], w_fail=0.8))
