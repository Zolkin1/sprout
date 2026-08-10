# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generator is never handed a fabricated history.

A history row is a reference-window node this runtime SERVED -- the generator's own past
output. Before the first serve there is nothing to push, so a zero row goes in as a
placeholder, and that row must not count towards the window being complete: training never
produced a zero node inside a valid history, so feeding one is off-distribution, and it would
arrive with ``history_null`` OFF, silently, for exactly the steps the learned null-history
embedding exists to cover.

Two paths got this wrong before:

* ``_update_generator`` -- the window went valid at step ``N-1`` with its OLDEST row still
  the placeholder;
* ``observe`` (the stable-stand warmup) -- nothing is ever served there, so the window went
  valid at step ``N-1`` with EVERY row a placeholder, and the first live FM sample was
  conditioned on an all-zero history with the null flag off.

Neither was covered by a test, which is why both survived. The warmup paths are now gone
entirely -- there is no stand hold, no static fill, and ``advance_dormant_step`` (what
``observe`` became) touches only the cadence counter -- so no code path can put a placeholder
into the history at all. That is a stronger property than "placeholders do not count", and it
is what the last three tests here pin. The C++ port
(``transfer/deploy_cpp``) carries the same rule and the same expectations; the two have to
agree or sim2sim and hardware condition the generator differently for the first N steps.

Run: ``python transfer/deploy/tests/test_history_null.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_replan_cadence import _FakePolicy, _state  # noqa: E402


def _null_flags(policy, steps):
    """Drive `steps` control steps and return ``history_null`` as seen at each one."""
    flags = []
    for _ in range(steps):
        policy._update_generator(_state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        flags.append(bool(policy._history.null_mask()[0]))
    return flags


def test_window_stays_null_until_it_holds_n_real_served_nodes():
    """``history_null`` clears at step N, not N-1: the first row is a placeholder."""
    p = _FakePolicy(k=1)
    n = p.ctx.N
    flags = _null_flags(p, n + 2)
    assert flags[:n] == [True] * n, f"expected null through step {n - 1}, got {flags}"
    assert flags[n] is False, "the window must go valid once N real nodes are in it"
    assert flags[n + 1] is False


def test_no_row_is_a_placeholder_on_the_step_the_flag_clears():
    """Inspect the window on the FIRST step ``history_null`` clears, not a fixed step count.

    Checking at a fixed count is what makes this kind of test useless: with the old behaviour
    the flag cleared a step EARLIER, and by any later step the placeholder had already been
    evicted, so the window looked clean and the test passed anyway.
    """
    p = _FakePolicy(k=1)
    cleared_at = None
    for step in range(2 * p.ctx.N + 2):
        p._update_generator(_state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        if not bool(p._history.null_mask()[0]):
            cleared_at = step
            break
    assert cleared_at is not None, "history_null never cleared"

    stacked = p._history.stack_padded("node")          # (1, N, D)
    zero_rows = [i for i in range(p.ctx.N) if bool(np.all(stacked[0, i] == 0.0))]
    assert not zero_rows, (
        f"history_null cleared at step {cleared_at} while row(s) {zero_rows} are still the "
        "zero placeholder")
    assert np.isfinite(stacked).all()


def test_a_dormant_step_does_not_touch_the_history_at_all():
    """``advance_dormant_step`` moves the cadence counter and nothing else.

    It used to append a row, and since nothing is served while the generator is dormant every
    row it appended was a zero placeholder. Enough of them and the window declared itself
    complete, so the first live FM sample was conditioned on an all-zero history with
    ``history_null`` OFF. Now there is no path at all that puts a placeholder into the history,
    which is a stronger property than "the placeholders do not count".
    """
    p = _FakePolicy(k=1)
    before = len(p._history._steps)
    for _ in range(p.ctx.N + 4):
        p.advance_dormant_step()
    assert len(p._history._steps) == before, "a dormant step must not append a history row"
    assert bool(p._history.null_mask()[0]), "an untouched history cannot be complete"


def test_dormant_steps_advance_the_cadence_counter():
    """The clip path relies on this: the counter must keep moving while the clip drives.

    Otherwise the first re-plan after a clip->FM handoff lands on the wrong cadence phase.
    """
    p = _FakePolicy(k=1)
    start = p._ctrl_step
    for _ in range(5):
        p.advance_dormant_step()
    assert p._ctrl_step == start + 5


def test_dormant_steps_do_not_stop_a_later_real_window_becoming_valid():
    """After a dormant stretch, real served nodes must still fill the window."""
    p = _FakePolicy(k=1)
    for _ in range(p.ctx.N + 2):
        p.advance_dormant_step()
    assert bool(p._history.null_mask()[0])
    later = _null_flags(p, p.ctx.N + 1)
    assert later[-1] is False, "a dormant stretch must not permanently invalidate the history"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all history_null tests passed")
