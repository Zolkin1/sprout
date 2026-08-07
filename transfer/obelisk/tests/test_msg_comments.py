# Copyright 2026 Zachary Olkin. All rights reserved.

"""Guard the ``.msg`` comments against sequences that break the generated C.

rosidl copies a ``.msg`` file's comments VERBATIM into C block comments in the generated
headers. A literal ``*/`` anywhere -- for instance in a topic glob like
``/depth/<star>/image_processed`` -- therefore closes that comment early, and every line of
prose after it is handed to the C compiler as code. The build then fails with a wall of
errors naming types nobody wrote (``unknown type name 'image_processed'``), several
translation units away from the comment that caused it.

Cheap to check, essentially impossible to diagnose from the error, and it only shows up
inside the container where colcon runs -- so it is worth a test that runs anywhere.

Run: ``pytest transfer/obelisk/tests/test_msg_comments.py``
"""

from __future__ import annotations

import pathlib

import pytest

MSG_DIR = pathlib.Path(__file__).resolve().parent.parent / "g1_msgs" / "msg"


def _msg_files() -> list[pathlib.Path]:
    """Every message definition in the package."""
    return sorted(MSG_DIR.glob("*.msg"))


def test_there_are_message_files():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert _msg_files(), f"no .msg files under {MSG_DIR}"


@pytest.mark.parametrize("path", _msg_files(), ids=lambda p: p.name)
def test_no_block_comment_terminator(path: pathlib.Path):
    """No line may contain ``*/``, which would end the generated C comment early."""
    bad = [(i, line) for i, line in enumerate(path.read_text().splitlines(), 1)
           if "*/" in line]
    assert not bad, (
        f"{path.name} line(s) {[i for i, _ in bad]} contain '*/', which terminates the "
        "generated C block comment and makes the following prose compile as code:\n  "
        + "\n  ".join(f"{i}: {line}" for i, line in bad)
        + "\nWrite topic globs as /depth/<cam>/... instead."
    )


@pytest.mark.parametrize("path", _msg_files(), ids=lambda p: p.name)
def test_fields_are_parseable(path: pathlib.Path):
    """Every non-comment line is a ``<type> <name>`` pair.

    Catches the other way a hand-written .msg fails late: a typo'd type or a stray word
    only surfaces when colcon runs the generator inside the container.
    """
    for i, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        assert len(parts) >= 2, f"{path.name}:{i}: not a '<type> <name>' pair: {raw!r}"
        type_name = parts[0]
        assert type_name[0].isalpha(), f"{path.name}:{i}: odd type {type_name!r}"
