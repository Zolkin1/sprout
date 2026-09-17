# Copyright 2026 Zachary Olkin. All rights reserved.

"""The RL command is laid onto the SPLICED PIECES, and it must not double-apply the family's sign.

Two things are pinned here, and the second was a live bug in the shipped path.

1. The command track comes from ``splice_pieces``' own ``(start, end)`` bounds -- one value per
   piece. A piece is a loop or a transient, so a boundary in the command IS a boundary in the
   reference. What it replaces rasterised a time-indexed schedule and re-coalesced it, which could
   place a change INSIDE an exact looped stride -- a command the loop provably cannot follow, since
   a loop travels command x duration by construction.

2. ``family_jump_off`` walks BACKWARDS (``exit_sign = -1``), and its commanded velocity had the
   WRONG SIGN. ``_clip_schedule``'s constant branch returned ``steady_vx = exit_sign * |vx|``,
   already signed; ``_rl_command_track`` then multiplied by ``sign`` again, cancelling it. Measured
   on a rebuilt library: the clip's ``rl_vx`` is -0.908, its pelvis travels a net -1.45 m, and
   ``target_vel`` said **+0.908** -- the teacher was told "walk forward" over a reference walking
   backward, for the whole family. The FM never saw it (``family_jump_off`` is in
   ``DEFAULT_SKIP_FAMILIES``, because no forward-facing camera can see a backwards dismount), but
   the teacher trains on it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch


def _repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "scripts" / "rsl_rl").is_dir() and (parent / "source" / "motion_gen").is_dir():
            return parent
    raise RuntimeError(f"no checkout above {__file__}")


sys.path.insert(0, str(_repo_root() / "scripts" / "rsl_rl"))

from mb_terrain_clips import _cmd_track_from_bounds  # noqa: E402


def test_each_piece_gets_its_own_command():
    out = _cmd_track_from_bounds([(0, 10), (10, 15), (15, 40)], [1.0, 2.0, 3.0], 40)
    assert out.shape == (40,)
    assert torch.all(out[:10] == 1.0) and torch.all(out[10:15] == 2.0) and torch.all(out[15:] == 3.0)


def test_a_change_lands_exactly_on_a_piece_boundary():
    """The property the rasterised schedule could only approximate."""
    out = _cmd_track_from_bounds([(0, 20), (20, 50)], [0.5, 2.5], 50)
    change = (out[1:] != out[:-1]).nonzero().flatten().tolist()
    assert change == [19], f"the command changed at {change}, not at the piece boundary"


def test_trailing_frames_keep_the_last_command():
    """A clip whose pieces end before its frame count must not fall to a silent zero -- a
    zero-velocity command reads as "stand", which is a different instruction, not a missing one."""
    out = _cmd_track_from_bounds([(0, 10), (10, 20)], [1.0, 2.0], 30)
    assert torch.all(out[20:] == 2.0)


def test_an_empty_piece_is_skipped_without_shifting_the_rest():
    """``splice_pieces`` can return a zero-width span (a piece the trim consumed)."""
    out = _cmd_track_from_bounds([(0, 10), (10, 10), (10, 20)], [1.0, 9.0, 3.0], 20)
    assert torch.all(out[:10] == 1.0) and torch.all(out[10:] == 3.0)
    assert 9.0 not in set(out.tolist())


@pytest.mark.parametrize("exit_sign,mag", [(-1.0, 0.908), (1.0, 1.4)])
def test_the_family_sign_is_applied_exactly_once(exit_sign, mag):
    """A backwards family's command must be NEGATIVE, and it must match the plan's own rl_vx.

    The sign lives in ``rl_vx`` and in the holds derived from it; nothing downstream re-applies it.
    This is the regression: the old track multiplied an already-signed value by the sign again, so
    ``jump_off`` -- the one family with ``exit_sign = -1`` -- came out commanded forwards.
    """
    rl_vx = exit_sign * mag
    holds = {"approach": [(3.0, rl_vx)], "interaction": rl_vx, "exit": [(5.5, rl_vx)]}
    cmds = ([c for _d, c in holds["approach"]] + [holds["interaction"]] * 2
            + [c for _d, c in holds["exit"]])
    out = _cmd_track_from_bounds([(0, 10), (10, 15), (15, 25), (25, 40)], cmds, 40)
    assert torch.all(torch.sign(out) == torch.sign(torch.tensor(rl_vx)))
    assert out[0].item() == pytest.approx(rl_vx, abs=1e-6)
    assert out[-1].item() == pytest.approx(rl_vx, abs=1e-6)
