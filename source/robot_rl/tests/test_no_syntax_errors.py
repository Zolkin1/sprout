# Copyright 2026 Zachary Olkin. All rights reserved.

"""Every tracked ``.py`` under ``scripts/`` / ``source/`` / ``transfer/`` must COMPILE.

Most modules here are only reachable through IsaacLab or MuJoCo, so nothing in the test suite
imports them -- a committed syntax error can therefore reach ``main`` with a green suite and only
surface when someone runs ``play.py`` or the sim2sim runner. That happened: a merge left two
stacked copies of the same call in ``scripts/flow_matching/export_fm_onnx.py`` (duplicate
``output_names=``) and in ``transfer/sim/gen_tracker_runner.py`` (duplicate ``run()`` signature +
call site), which broke ``play.py`` at import.

This is a SYNTAX check only: :func:`compile` parses and compiles but does not execute or import,
so it needs neither isaaclab nor MuJoCo and stays fast.

NOTE it must use :func:`compile`, NOT :func:`ast.parse`. ``ast.parse`` accepts a duplicate keyword
argument (``f(a=1, a=2)``) -- exactly the error that shipped -- because that is diagnosed after the
parse tree is built. Do not "simplify" this to ``ast.parse``.
"""

from __future__ import annotations

import pathlib

import pytest

# Directories to sweep, relative to the repo root.
_ROOTS = ("scripts", "source", "transfer")

# Path fragments to skip: build artifacts, caches, and vendored virtualenvs. ``build`` covers the
# stale ``source/robot_rl/build/lib`` copy a local ``setup.py build`` leaves behind (gitignored,
# and it holds a snapshot of pre-refactor code).
_SKIP_PARTS = ("__pycache__", "build", ".venv", "venv", "env_isaaclab", ".git", "site-packages")

# A floor on how many files the sweep must see, so a bad glob or a moved directory fails loudly
# instead of passing vacuously with zero files compiled.
_MIN_FILES = 200


def _repo_root() -> pathlib.Path:
    """Walk up from this file to the repo root, identified by its own marker layout.

    Derived rather than counting ``..`` hops, so moving this test between packages cannot
    silently retarget the sweep at the wrong directory.

    Returns:
        The repository root.

    Raises:
        AssertionError: If no ancestor looks like the repo root.
    """
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / ".git").exists() and (parent / "scripts").is_dir():
            return parent
    raise AssertionError("could not locate the repo root above this test file")


def _python_files() -> list[pathlib.Path]:
    """Every ``.py`` under :data:`_ROOTS`, excluding :data:`_SKIP_PARTS`."""
    root = _repo_root()
    out: list[pathlib.Path] = []
    for rel in _ROOTS:
        base = root / rel
        if not base.is_dir():                                # pragma: no cover - layout guard
            continue
        out.extend(
            p for p in base.rglob("*.py")
            if not any(part in _SKIP_PARTS for part in p.parts)
        )
    return sorted(out)


def test_sweep_sees_enough_files() -> None:
    """Guard the guard: a sweep that matches nothing would pass every other check."""
    files = _python_files()
    assert len(files) >= _MIN_FILES, (
        f"only {len(files)} python files found under {_ROOTS} -- the sweep is probably "
        f"mis-rooted (root={_repo_root()}), so it would pass vacuously"
    )


def test_every_python_file_compiles() -> None:
    """No tracked module may contain a syntax error, even if nothing imports it."""
    root = _repo_root()
    failures: list[str] = []
    for path in _python_files():
        try:
            compile(path.read_text(encoding="utf-8", errors="replace"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(root)}:{exc.lineno}: {exc.msg}")
        except ValueError as exc:      # e.g. source containing null bytes
            failures.append(f"{path.relative_to(root)}: {exc}")
    assert not failures, "syntax errors in tracked python files:\n  " + "\n  ".join(failures)


def test_compile_catches_duplicate_kwargs() -> None:
    """Pin the reason this uses ``compile`` and not ``ast.parse``.

    ``ast.parse`` does NOT reject a repeated keyword argument, so a sweep built on it would have
    missed the very error that motivated this file.
    """
    import ast

    src = "f(a=1, a=2)\n"
    ast.parse(src)                                    # accepted -- the weaker check
    with pytest.raises(SyntaxError, match="keyword argument repeated"):
        compile(src, "<dup>", "exec")
