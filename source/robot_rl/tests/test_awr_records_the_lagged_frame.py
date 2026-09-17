# Copyright 2026 Zachary Olkin. All rights reserved.

"""The AWR recorder must store the frame the generator WAS CONDITIONED ON, not the live one.

With depth latency on, ``FmTrackingCommand`` serves the generator a frame a few control steps
old (``default_depth_lag_steps`` of the trained distribution -- 2 steps, 40 ms, for the shipped
tuple) while the teacher's own depth observation stays fresh. The fine-tune then trains the
generator to reproduce the chunk it emitted, so the row's ``extero`` has to be the input that
produced that chunk.

**Writing the live frame instead is a one-word change and completely silent.** Every shape is
identical, every count matches, the loss goes down -- and the model is being taught to map a
FRESH image onto a plan that was made from a STALE one, i.e. to undo exactly the latency
tolerance the lag exists to build. Nothing downstream could report it: ``depth_lag_used`` would
still say 2, because it is computed from the command's configured lag rather than from the row.

That is why this is a source-level assertion rather than a behavioural one. Reproducing it at
runtime needs Isaac, a loaded pair and a lag-trained parent; the property is a single
expression, and pinning the expression is what keeps it.

See also ``test_depth_latency_default.py`` for the field that switches this on, and
``rollout_dataset`` for why the buffer must NOT re-apply a lag that is already baked in.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REC = (pathlib.Path(__file__).resolve().parents[3]
        / "scripts" / "rsl_rl" / "record_rollouts.py")


def _extero_value_source() -> str:
    """The expression assigned to ``"extero"`` in the recorder's batch dict.

    Returns:
        Its source text.
    """
    tree = ast.parse(_REC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "extero":
                return ast.unparse(v)
    pytest.fail("no \"extero\" key in any dict literal in record_rollouts.py")


def test_the_recorded_extero_is_the_lagged_frame() -> None:
    """The property the whole latency story rests on."""
    src = _extero_value_source()
    assert "_lagged_extero" in src, (
        f'record_rollouts stores "extero" as `{src}`, which is not the lagged frame. The '
        "generator is conditioned on a stale image; training on the fresh one teaches it to "
        "map a frame it never saw onto the plan, undoing the latency tolerance. Nothing "
        "downstream reports this -- shapes, counts and depth_lag_used are all unchanged.")


def test_the_served_lag_is_recorded_per_row() -> None:
    """``depth_lag_used`` must exist, since the served lag is capped early in an episode.

    The command holds a ring of frames, so before it fills, fewer steps of lag are available
    than configured. A row therefore cannot be assumed to carry the nominal lag.
    """
    src = _REC.read_text(encoding="utf-8")
    assert '"depth_lag_used"' in src
    assert "min(nominal, max(held - 1, 0))" in src, (
        "the served lag must be the CAPPED one -- a ring that has not filled yet cannot "
        "supply the nominal lag, and recording the nominal would misdescribe those rows")


def test_the_rollout_dataset_does_not_re_apply_the_lag() -> None:
    """Double-lagging would be invisible: the shapes are identical either way."""
    rd = (pathlib.Path(__file__).resolve().parents[1]
          / "robot_rl" / "flow_matching" / "rollout_dataset.py").read_text(encoding="utf-8")
    assert "_select_depth_lag" not in rd, (
        "the rollout buffer's extero is ALREADY the lagged frame, baked in at collection. "
        "Applying the dataset-side latency model on top would age it twice, and every shape "
        "would still agree.")
