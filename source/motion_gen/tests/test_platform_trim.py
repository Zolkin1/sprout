# Copyright 2026 Zachary Olkin. All rights reserved.

"""``_platform_trim`` must cut EVERY per-frame channel, not just the motion.

A terrain clip whose exit walks off the end of its platform is cut back to where it leaves the
surface -- a shorter reference is honest, a reference standing on nothing is not. The cut has three
consumers: the motion (``Wp``/``Wq``/``Wj``), the segment ``bounds``, and the per-frame diagnostic
channels. The third was missed, and the miss was silent: ``stride_idx_t`` (which marks the exact-loop
frames, and is what ``mb_qc`` uses to find loop<->transient seams and to build ``_cycle_avg_err``'s
steady mask) kept its PRE-trim length, so it was shifted by ``head`` and ran too long -- on exactly
the clips the trim exists to salvage, i.e. the ones already in trouble.

Nothing downstream compares a per-frame channel's length against the clip's, so there was no error
to notice. Hence this file plus the length assertion in ``clip_from_plan``.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest
import torch


def _repo_root() -> pathlib.Path:
    """The checkout this test file lives in.

    Searched for rather than counted in ``..`` hops (which break silently when a file moves), and
    NOT derived from ``motion_gen.__file__``: the editable install points at the MAIN checkout, so
    in a git worktree that resolves to the wrong tree and would test the other copy's scripts.
    """
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "scripts" / "rsl_rl").is_dir() and (parent / "source" / "motion_gen").is_dir():
            return parent
    raise RuntimeError(f"no checkout with scripts/rsl_rl + source/motion_gen above {__file__}")


sys.path.insert(0, os.path.join(str(_repo_root()), "scripts", "rsl_rl"))

from mb_terrain_clips import apply_platform_trim  # noqa: E402

N = 40


def _motion(n=N):
    """Per-frame tensors whose value IS their frame index, so a shift is visible."""
    idx = torch.arange(n, dtype=torch.float32)
    return [idx[:, None].repeat(1, 3), idx[:, None].repeat(1, 4), idx[:, None].repeat(1, 29)]


def _sidx(n=N):
    return torch.arange(n, dtype=torch.int32)


@pytest.mark.parametrize("head,tail", [(0, N), (5, N), (0, 30), (5, 30), (12, 13)])
def test_every_per_frame_channel_gets_the_same_cut(head, tail):
    arrays, bounds, pf = apply_platform_trim(
        head, tail, _motion(), [(0, 10), (10, 25), (25, N)], {"stride_idx_t": _sidx()})
    kept = tail - head
    for a in arrays:
        assert a.shape[0] == kept
    assert pf["stride_idx_t"].shape[0] == kept, "per-frame channel not sliced with the motion"
    # ...and cut at the SAME offset: frame k of the result must be frame head+k of the input.
    assert int(pf["stride_idx_t"][0]) == head
    assert float(arrays[0][0, 0]) == float(head)
    assert bounds[-1][1] == kept, "bounds must be re-based onto the cut"


def test_a_no_op_trim_returns_the_inputs_untouched():
    m, b, pf = _motion(), [(0, 10), (10, 25), (25, N)], {"stride_idx_t": _sidx()}
    arrays, bounds, per_frame = apply_platform_trim(0, N, m, b, pf)
    assert arrays is m and bounds is b and per_frame is pf


def test_a_missing_per_frame_channel_is_tolerated():
    """A plan with no loop pieces has no ``stride_idx_t``; that must not be a crash."""
    _, _, pf = apply_platform_trim(5, 30, _motion(), [(0, N)], {"stride_idx_t": None})
    assert pf["stride_idx_t"] is None


def test_bounds_clamp_rather_than_go_negative():
    """A head past a segment's end collapses it to an empty span, never a negative one."""
    _, bounds, _ = apply_platform_trim(20, 30, _motion(), [(0, 10), (10, 25), (25, N)],
                                      {"stride_idx_t": _sidx()})
    assert bounds[0] == (0, 0), bounds
    for a, b in bounds:
        assert 0 <= a <= b <= 10
