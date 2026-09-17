# Copyright 2026 Zachary Olkin. All rights reserved.
"""Single keypresses from a terminal while a MuJoCo viewer is open.

The viewer owns the GL context and blocks, so anything that wants to be steered WHILE a replay
runs -- pausing at a latch, labelling a trial as it plays -- has to read the keyboard without
waiting for Enter and without blocking the render loop.  Two consumers now need that (the mode
probe's labeller and ``view_cell.replay``'s latch pause), which is why it lives here rather
than in either of them.

Every function degrades to "no terminal" rather than failing: a piped or redirected stdin has
nothing to configure, and a caller must then fall back to line-oriented prompts instead of
silently reading nothing.
"""

from __future__ import annotations

import contextlib
import select
import sys
import termios
import tty


@contextlib.contextmanager
def cbreak_stdin():
    """Single keypresses without Enter, restored on the way out.

    Yields whether it took effect: a piped or redirected stdin has no terminal to configure,
    and the caller then falls back to line-oriented prompts rather than silently reading
    nothing.  ``cbreak`` rather than ``raw`` on purpose -- it clears ECHO and ICANON but leaves
    output processing alone, so ``print`` still ends lines properly.

    Yields:
        ``True`` when the terminal was put into cbreak mode, ``False`` when there is none.
    """
    if not sys.stdin.isatty():
        yield False
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def poll_key() -> str | None:
    """The next keypress if one is waiting, else ``None``. Never blocks.

    Returns:
        The lower-cased character, or ``None`` when nothing is waiting.
    """
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        return ch.lower() if ch else None
    return None


def drain_keys() -> None:
    """Discard anything typed before the caller started listening.

    Without this, a key pressed during the previous prompt arrives as this one's answer -- so
    an impatient double-tap silently answers the NEXT question.
    """
    while poll_key() is not None:
        pass
