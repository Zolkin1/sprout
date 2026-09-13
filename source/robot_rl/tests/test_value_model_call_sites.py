# Copyright 2026 Zachary Olkin. All rights reserved.

"""Every ``FlowValueModel(...)`` call must pass arguments the model actually takes.

``dcea58f0`` deleted the critic's scan and proprio token streams. It rewrote the model, it
rewrote five comments describing what the critic reads -- and it left ``proprio_dim=`` and
``proprio_steps=`` at the one place that CONSTRUCTS the model, in ``fit_value.py``. Nothing
caught it: the module imports, the arrays it reads to fill those arguments still exist in the
recording, and the first thing to execute that line is the critic fit itself. So a fine-tune
collected 2000 episodes, graded them, built and verified a 55k-window buffer -- twenty minutes
of an H100 -- and then died on::

    TypeError: FlowValueModel.__init__() got an unexpected keyword argument 'proprio_dim'

The AWR loop had run many times before that: the last healthy critic fit is
``awr_vm30_fbc/iter_005``, 39 minutes before the first of the two refactor commits. Nothing
has fitted a critic between them and now, which is why both of that pair's defects surfaced
together, two days later, inside a queued run.

A keyword argument list is drift that no import, no shape check and no type hint can see, and
the call site is deep enough inside a long pipeline that only a real run reaches it. Parsing
it is the cheap way to keep the two honest, in the same spirit as
``test_finetune_loop_algos.py`` parsing the shell script.

The scan is over the AST rather than a regex so a multi-line call, a nested call or a comment
mentioning the name cannot fool it either way.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Where the model may be constructed. Listed rather than globbed so a NEW construction site
#: has to be added here deliberately -- adding one is exactly when this check matters.
_CALL_SITE_FILES: tuple[str, ...] = (
    "scripts/flow_matching/fit_value.py",
)


def _accepted_keywords() -> tuple[set[str], bool]:
    """The keyword names ``FlowValueModel.__init__`` accepts, and whether it takes ``**kwargs``.

    Returns:
        ``(names, has_var_keyword)``.
    """
    value = pytest.importorskip(
        "robot_rl.flow_matching.value", reason="the critic model imports torch")
    sig = inspect.signature(value.FlowValueModel.__init__)
    names = {n for n, p in sig.parameters.items()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n != "self"}
    var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    return names, var_kw


def _calls_in(path: pathlib.Path) -> list[ast.Call]:
    """Every ``FlowValueModel(...)`` call in one file, however it is spelled.

    Args:
        path: The file to parse.

    Returns:
        The matching call nodes.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == "FlowValueModel":
            out.append(node)
    return out


@pytest.mark.parametrize("rel", _CALL_SITE_FILES)
def test_every_call_site_passes_arguments_the_model_takes(rel: str) -> None:
    """One assert per file, naming the argument that drifted."""
    path = _ROOT / rel
    assert path.exists(), f"{rel} has moved; update _CALL_SITE_FILES"
    accepted, var_kw = _accepted_keywords()
    if var_kw:
        pytest.skip("FlowValueModel takes **kwargs, so no keyword can be wrong here")
    calls = _calls_in(path)
    assert calls, f"no FlowValueModel(...) call found in {rel}; has it moved?"
    for call in calls:
        passed = {kw.arg for kw in call.keywords if kw.arg is not None}
        unknown = sorted(passed - accepted)
        assert not unknown, (
            f"{rel}:{call.lineno} passes {unknown} to FlowValueModel, which accepts "
            f"{sorted(accepted)}. This is a TypeError at the first critic fit -- twenty "
            "minutes into a fine-tune, after the rollout has already been collected.")


def test_the_deleted_token_streams_are_gone_from_the_model() -> None:
    """The refactor's own claim, pinned: no scan and no proprio arguments survive."""
    accepted, _ = _accepted_keywords()
    for dead in ("proprio_dim", "proprio_steps", "scan_dim", "scan_shape"):
        assert dead not in accepted, (
            f"FlowValueModel grew {dead} back; dcea58f0 deleted that token stream, so either "
            "the refactor was reverted or this test is now describing the wrong model")
