# Copyright 2026 Zachary Olkin. All rights reserved.
"""No comment sits inside a backslash-continued command in the launch scripts.

A comment after a line ending in ``\\`` ENDS the command: the continuation is consumed by the
comment, and the next line runs as a command of its own. The result is valid shell, so
``bash -n`` accepts it, and the failure surfaces as::

    run_finetune_loop.sh: line 341: --history_from: command not found

with the flag simply absent from the invocation -- so the callee silently uses its default.
That cost a GPU launch on 2026-09-09: the measured-history arm ran without ``--history_from``,
served the generator arm to a measured-trained checkpoint, and was refused by
``assert_history_source_matches`` after the simulator had come up.

These scripts run under ``set -uo pipefail`` with NO ``-e``, which is why a stray command that
merely fails does not stop the run.
"""

from __future__ import annotations

import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]

#: Every shell script that launches or evaluates a run.
SCRIPTS: tuple[str, ...] = (
    "scripts/flow_matching/run_finetune_loop.sh",
    "scripts/rsl_rl/run_quick_eval.sh",
)


@pytest.mark.parametrize("rel", SCRIPTS)
def test_no_comment_breaks_a_line_continuation(rel: str) -> None:
    """A ``#`` line may not follow a line ending in a backslash."""
    p = _REPO / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present")
    lines = p.read_text().splitlines()
    bad = []
    for i, line in enumerate(lines[:-1]):
        if not line.rstrip().endswith("\\"):
            continue
        nxt = lines[i + 1].strip()
        if nxt.startswith("#"):
            bad.append((i + 2, nxt[:70]))
    assert not bad, (
        f"{rel}: a comment follows a continued line, which ENDS the command and makes every "
        f"argument after it a separate command:\n"
        + "\n".join(f"    line {n}: {t}" for n, t in bad))


@pytest.mark.parametrize("rel", SCRIPTS)
def test_no_blank_line_breaks_a_line_continuation(rel: str) -> None:
    """Same failure, same silence: a blank line after ``\\`` also ends the command."""
    p = _REPO / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present")
    lines = p.read_text().splitlines()
    bad = [i + 2 for i, line in enumerate(lines[:-1])
           if line.rstrip().endswith("\\") and not lines[i + 1].strip()]
    assert not bad, f"{rel}: a blank line follows a continued line at {bad}"
