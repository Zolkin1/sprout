# Copyright 2026 Zachary Olkin. All rights reserved.

"""The queue must evaluate the checkpoint the loop actually built, and it is not ``best.pt``.

``run_finetune_loop.sh`` chains iteration *k+1* from ``iter_k/ft/last.pt``, and its closing
advice tells you to evaluate ``iter_N/ft/last.pt``. ``run_finetune_queue.sh`` picks the file
its eval jobs score. If those two disagree, the reported number describes a model that was
never part of the chain -- and nothing in any log says so, because both files exist, both load,
and both are the right size.

**The disagreement has a SIGN, which is why this is a test and not a style note.**
``finetune_generator.py`` states it: ``best.pt`` is selected on UNWEIGHTED ``val_flow_mse``,
"which AWR moves away from by design, so it favours the checkpoint that changed least". The
checkpoint that changed least is the one nearest the PARENT -- and the parent is the control
arm every eval in this queue is measured against. Scoring ``best.pt`` therefore pulls every
fine-tune's delta toward zero and manufactures a null result shaped exactly like "fine-tuning
this course does nothing".

That is not hypothetical: the queue shipped pointing at ``best.pt``, and on ``ft_boxes``
iteration 5 the two files are the same size with different md5s, so four evals would have
answered the wrong question with no visible symptom.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_LOOP = _ROOT / "scripts" / "flow_matching" / "run_finetune_loop.sh"
_QUEUE = _ROOT / "scripts" / "flow_matching" / "run_finetune_queue.sh"


def _chained_checkpoint() -> str:
    """The checkpoint basename the LOOP trains each next iteration from.

    Returns:
        e.g. ``"last.pt"``.

    Raises:
        AssertionError: If the assignment cannot be found.
    """
    text = _LOOP.read_text(encoding="utf-8")
    m = re.search(r"POLICY=\"\$ROOT/\$\(printf 'iter_%03d'[^\"]*?/ft/([A-Za-z0-9_]+\.pt)\"", text)
    assert m, "could not find the loop's POLICY= chaining assignment; has it been renamed?"
    return m.group(1)


def _evaluated_checkpoint() -> str:
    """The checkpoint basename the QUEUE hands to its eval jobs.

    Returns:
        e.g. ``"last.pt"``.

    Raises:
        AssertionError: If ``ft_result``'s echo cannot be found.
    """
    text = _QUEUE.read_text(encoding="utf-8")
    body = re.search(r"ft_result\(\)\s*\{(.*?)\n\}", text, re.S)
    assert body, "could not find ft_result() in the queue script"
    m = re.search(r"/ft/([A-Za-z0-9_]+\.pt)\"?\s*$", body.group(1).strip().splitlines()[-1])
    assert m, f"ft_result() does not end in a /ft/<name>.pt path:\n{body.group(1)}"
    return m.group(1)


def test_the_queue_evaluates_what_the_loop_chained() -> None:
    """The one assert this file exists for."""
    chained, evaluated = _chained_checkpoint(), _evaluated_checkpoint()
    assert evaluated == chained, (
        f"the loop chains from {chained!r} but the queue evaluates {evaluated!r}. The reported "
        "number would then describe a model that was never in the chain, and both files exist "
        "so nothing downstream would notice.")


def test_the_queue_does_not_evaluate_best_pt() -> None:
    """Stated separately, because it is the direction of the bias that matters.

    If someone ever changes the loop to chain from ``best.pt``, the test above would go on
    passing while both scripts were wrong together.
    """
    assert _evaluated_checkpoint() != "best.pt", (
        "best.pt is selected on unweighted val_flow_mse, which AWR moves away from by design, "
        "so it favours the checkpoint that changed LEAST -- i.e. the one nearest the parent, "
        "which is the control arm. Scoring it shrinks every fine-tune delta toward zero.")


@pytest.mark.parametrize("path", [_LOOP, _QUEUE])
def test_the_scripts_are_syntactically_valid(path: pathlib.Path) -> None:
    """Cheap guard: these are parsed by regex above, so a broken edit should fail loudly."""
    import subprocess  # noqa: PLC0415
    assert path.exists(), f"{path} is missing"
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, f"{path.name} does not parse:\n{r.stderr}"
