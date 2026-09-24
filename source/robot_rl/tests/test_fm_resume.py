# Copyright 2026 Zachary Olkin. All rights reserved.

"""``train.py --resume_from`` is a TRUE resume, and it refuses the cases that would lie.

A generator training run is many GPU-hours, and until now a killed one could only be restarted
from scratch -- which is what a full disk turned into a lost 86 epochs. The checkpoint already
carried everything stateful (optimizer moments, the EMA shadow AND its step, the epoch), so the
resume was a plumbing gap rather than a missing capability.

The thing these tests protect is the DISTINCTION between a resume and a warm restart. Reloading
only the weights would reset Adam's moments and replay the LR warmup: a different optimisation
problem that produces a checkpoint indistinguishable from a properly-resumed one, and whose
numbers would silently not be comparable with an uninterrupted run. That is the failure this
repo's CLAUDE.md files catalogue over and over -- a plausible number from the wrong procedure.

Source-level, because running the real thing needs a GPU, a dataset and hours.
"""

from __future__ import annotations

import ast
import pathlib

#: parents[3] is the repo root: tests/ -> robot_rl/ -> source/ -> repo. Written as [2] first,
#: which resolved to source/scripts/... and made every test here fail on a missing file.
_TRAIN = (pathlib.Path(__file__).resolve().parents[3]
          / "scripts/flow_matching/train.py")


def _func(name: str) -> ast.FunctionDef:
    """The named top-level function's AST node.

    Args:
        name: Function name.

    Returns:
        Its ``ast.FunctionDef``.
    """
    tree = ast.parse(_TRAIN.read_text())
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name} is not defined in train.py")


def test_the_flag_exists_and_defaults_to_off() -> None:
    """``--resume_from`` defaults to empty, so no existing invocation changes behaviour."""
    src = _TRAIN.read_text()
    assert '"--resume_from"' in src, "train.py must expose --resume_from"
    assert 'parser.add_argument(\n        "--resume_from", type=str, default=""' in src, (
        "--resume_from must default to empty; anything else would change what every "
        "existing training command does.")


def test_resume_restores_the_optimizer_and_the_ema_not_just_the_weights() -> None:
    """The restore touches optimizer moments and the EMA shadow AND step.

    Weights alone is a warm restart. The optimizer carries Adam's first and second moments; the
    EMA carries the shadow that ``best.pt`` is selected on plus the step count that
    ``scheduled_lr`` needs. Dropping any of them changes the optimisation while leaving the
    output looking identical.
    """
    body = ast.get_source_segment(_TRAIN.read_text(), _func("_restore_for_resume")) or ""
    for needed, why in (
        ('opt.load_state_dict', "Adam's moments would be reset -- a warm restart"),
        ('model.load_state_dict', "the weights themselves"),
        ('ema.shadow', "best.pt is selected on the EMA, so its shadow must come back"),
        ('ema.step', "scheduled_lr is closed form in (epoch, step); the step must come back"),
    ):
        assert needed in body, f"_restore_for_resume must restore {needed}: {why}"


def test_a_checkpoint_without_optimizer_state_is_refused() -> None:
    """No moments means the only honest option is a fresh run, so it raises rather than warm-starts."""
    body = ast.get_source_segment(_TRAIN.read_text(), _func("_restore_for_resume")) or ""
    assert '"optimizer" not in ck' in body, (
        "a checkpoint with no optimizer state must be refused; silently warm-restarting from "
        "one produces a run that is not comparable with an uninterrupted one and says nothing")
    assert "WARM RESTART" in body, "the refusal should name what it is avoiding"


def test_an_architecture_or_data_mismatch_is_refused() -> None:
    """Resuming across a config change trains weights that no longer mean what they did.

    This is the dangerous direction: the result loads, trains, and reports plausible numbers.
    """
    body = ast.get_source_segment(_TRAIN.read_text(), _func("_restore_for_resume")) or ""
    assert "REFUSING to resume" in body, "a mismatched resume must raise, not warn"
    for field in ("d_model", "extero_sensor_count", "attention_mode", "history_source"):
        assert field in body, f"the architecture check must cover {field}"
    assert "data_sha256" in body, (
        "resuming onto different data must be refused -- the weights were fitted to the old "
        "set and nothing downstream records which one a resumed run actually saw")


def test_the_loop_starts_at_the_resumed_epoch() -> None:
    """``for epoch in range(start_epoch, cfg.num_epochs)``, not from 0."""
    src = _TRAIN.read_text()
    assert "for epoch in range(start_epoch, cfg.num_epochs):" in src, (
        "the epoch loop must start at the resumed epoch; starting at 0 would re-run epochs "
        "the checkpoint has already done, against an optimizer state that is past them")


def test_the_rng_caveat_is_stated() -> None:
    """The one thing a resume cannot restore is said out loud, not left implied."""
    body = ast.get_source_segment(_TRAIN.read_text(), _func("_restore_for_resume")) or ""
    assert "RNG" in body and "bit-identical" in body, (
        "a resumed run's data order differs from an uninterrupted one; that belongs in the "
        "docstring and in the run's own stdout, or someone will report it as reproducible")
