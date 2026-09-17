# Copyright 2026 Zachary Olkin. All rights reserved.
"""The latch pause must always be releasable.

``view_cell.replay`` freezes on the first strict tracking violation so the pose at the crossing
can be inspected rather than watched go past.  A pause is only useful if it ends, and the first
version of this could not: the keypress check was gated on ``cbreak`` having taken effect, so
wherever it silently did not, every keystroke was discarded and the sole advertised escape did
nothing -- an unkillable hold in the middle of a walkthrough, with the console still inviting a
key.

So the hold has FOUR independent exits and every one of them is pinned here. The assertion that
matters in each case is not the return value, it is that the call TERMINATES.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SIM, os.path.join(_SIM, "mjeval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

view_cell = pytest.importorskip("view_cell")
keys = pytest.importorskip("keys")

#: Any hold longer than this is treated as a hang. Generous against a loaded CI box; the real
#: failure mode is unbounded, so the exact number does not matter.
_HANG_S: float = 5.0


class _Viewer:
    """Passive-viewer stand-in: counts syncs and can report itself closed."""

    def __init__(self, close_after: int | None = None) -> None:
        """Store the sync count at which ``is_running`` starts returning False."""
        self.n = 0
        self.close_after = close_after

    def is_running(self) -> bool:
        """Whether the window is still open."""
        return self.close_after is None or self.n < self.close_after

    def sync(self) -> None:
        """Count a render."""
        self.n += 1


def _hold(viewer, cap_s: float, feed=None) -> tuple[bool, float]:
    """Run the hold, optionally feeding it keystrokes, and time it."""
    orig = keys.poll_key
    if feed is not None:
        it = iter(feed)
        keys.poll_key = lambda: next(it, None)
    try:
        t0 = time.time()
        skip = view_cell._hold_until_resumed(viewer, cap_s=cap_s)
        return skip, time.time() - t0
    finally:
        keys.poll_key = orig


def test_the_cap_releases_a_hold_nobody_answers() -> None:
    """The backstop. Without it a keystroke sent to the wrong window is a dead session."""
    skip, dt = _hold(_Viewer(), cap_s=0.4)
    assert dt < _HANG_S, f"held {dt:.1f}s -- the cap did not fire"
    assert skip is False


def test_closing_the_viewer_releases_the_hold() -> None:
    """The escape that needs no terminal at all, which is why it is not the cap's job."""
    skip, dt = _hold(_Viewer(close_after=3), cap_s=999.0)
    assert dt < _HANG_S, f"held {dt:.1f}s -- closing the window did not release it"
    assert skip is False


@pytest.mark.parametrize("key,want_skip", [("s", True), ("n", False), ("x", False)])
def test_a_keystroke_releases_the_hold_and_only_s_skips(key: str, want_skip: bool) -> None:
    """``s`` abandons the rest of the episode; anything else resumes it."""
    skip, dt = _hold(_Viewer(), cap_s=999.0, feed=[None, key])
    assert dt < _HANG_S, f"held {dt:.1f}s on {key!r}"
    assert skip is want_skip


@pytest.mark.parametrize("code,want_skip", [(ord("S"), True), (ord("N"), False),
                                            (32, False)])
def test_a_key_pressed_IN_THE_VIEWER_releases_the_hold(code: int, want_skip: bool) -> None:
    """The primary channel, and the one the first two versions did not have.

    A keystroke goes to whichever window has focus; the operator is looking at the viewer, so
    a hold releasable only from the terminal was releasable by nothing in practice. ``S``
    abandons the episode, any other key resumes it.
    """
    box: dict = {"key": None}
    v = _Viewer()

    class _Feeder(_Viewer):
        """Presses the key on the second render, the way a person would mid-hold."""

        def sync(self) -> None:
            super().sync()
            if self.n == 2:
                box["key"] = code

    t0 = time.time()
    skip = view_cell._hold_until_resumed(_Feeder(), cap_s=999.0, pressed=box)
    dt = time.time() - t0
    assert dt < _HANG_S, f"held {dt:.1f}s -- a viewer keypress did not release it"
    assert skip is want_skip
    assert box["key"] is None, "the key was not consumed, so the next hold would see it"
    assert v.n == 0


def test_a_key_pressed_BEFORE_the_hold_is_discarded() -> None:
    """Otherwise a stray press during the replay releases the pause the instant it starts.

    Same argument as ``drain_keys`` on the terminal side: an impatient tap must not answer the
    next question.
    """
    box: dict = {"key": ord("N")}

    class _Feeder(_Viewer):
        """Presses nothing; the only key on offer is the stale one."""

        def sync(self) -> None:
            super().sync()

    t0 = time.time()
    skip = view_cell._hold_until_resumed(_Feeder(), cap_s=0.4, pressed=box)
    dt = time.time() - t0
    assert dt > 0.3, "a stale keypress released the hold immediately"
    assert skip is False


def test_the_viewer_is_launched_WITH_a_key_callback() -> None:
    """Without it the box is never written and the primary escape silently does not exist."""
    body = open(view_cell.__file__).read()
    assert "key_callback=_on_key" in body, (
        "launch_passive is called without a key_callback, so keys pressed in the viewer "
        "window reach nothing")


def test_the_key_check_is_NOT_gated_on_cbreak_succeeding() -> None:
    """The regression itself, asserted on the source.

    ``poll_key`` already returns ``None`` without a terminal, so gating the call added nothing
    and removed the only escape wherever ``cbreak`` did not take. Pinned by reading the loop
    rather than by behaviour, because the behaviour under a failed cbreak is exactly the
    behaviour under "no key was pressed" -- indistinguishable from the outside.
    """
    body = open(view_cell.__file__).read()
    fn = body[body.index("def _hold_until_resumed"):body.index("def replay(")]
    assert "if live:" not in fn, (
        "the keypress check is gated on cbreak again; where cbreak fails the hold becomes "
        "unreleasable by key")
    assert "keys.poll_key()" in fn


def test_an_uncapped_hold_is_still_releasable_by_key() -> None:
    """``cap_s <= 0`` means "wait", not "wait forever with no way out"."""
    skip, dt = _hold(_Viewer(), cap_s=0.0, feed=[None, "q"])
    assert dt < _HANG_S
    assert skip is False
