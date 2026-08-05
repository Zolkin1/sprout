# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the N-ary piece splicer.

A flat clip is ``[stand][transient][loop][transient][loop]...``, so the splicer has to handle an
arbitrary alternation rather than the fixed ``approach + terrain + exit`` triple. Two properties carry
the design:

* the RIGID pieces (exact looped strides, played terrain) come through untouched -- easing them would
  throw away the exactness the loops exist to provide;
* seams are eased independently. With many short pieces two seams can share one free piece, and easing
  them lazily would let the second read frames the first already moved and reopen it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motion_gen.motionbricks.terrain_splice import Piece, splice_pieces

DT = 0.02


def _seg(n: int, *, x0: float = 0.0, vx: float = 1.0, joint: float = 0.0, z: float = 0.78):
    """A straight, constant-velocity segment with a constant joint vector (so offsets are readable)."""
    p = torch.zeros(n, 3)
    p[:, 0] = x0 + torch.arange(n, dtype=torch.float32) * vx * DT
    p[:, 2] = z
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    j = torch.full((n, 29), float(joint))
    return (p, q, j)


def _pieces(*specs):
    return [Piece(seg=s, kind=k, rigid=r) for s, k, r in specs]


def test_single_piece_passes_through():
    p, q, j = _seg(10)
    wp, wq, wj, bounds, seams, diag = splice_pieces([Piece(seg=(p, q, j), kind="loop", rigid=True)])
    assert torch.allclose(wp, p) and bounds == [(0, 10)] and seams == []
    assert diag["n_frames"] == 10


def test_pieces_chain_c0_in_root_position():
    """Each piece is welded onto the previous last frame, so the root never jumps."""
    pieces = _pieces((_seg(20, vx=1.0), "loop", True), (_seg(15, vx=1.0), "transient", False),
                     (_seg(20, vx=1.0), "loop", True))
    wp, _, _, bounds, _, _ = splice_pieces(pieces, blend_n=0, dt=DT)
    step = torch.linalg.norm(torch.diff(wp[:, :2], dim=0), dim=1)
    assert float(step.max() - step.min()) < 1e-5          # uniform advance across both seams
    assert bounds[0][1] == bounds[1][0] and bounds[1][1] == bounds[2][0]
    assert bounds[-1][1] == wp.shape[0]


def test_rigid_pieces_are_never_modified_by_easing():
    """The whole point of the loops: a seam ease must consume the TRANSIENT, not the loop."""
    loop_a, trans, loop_b = _seg(30, joint=0.0), _seg(20, joint=0.5), _seg(30, joint=0.0)
    pieces = _pieces((loop_a, "loop", True), (trans, "transient", False), (loop_b, "loop", True))
    _, _, wj, bounds, _, _ = splice_pieces(pieces, blend_n=6, dt=DT)
    for i in (0, 2):
        s, e = bounds[i]
        assert torch.allclose(wj[s:e], torch.tensor(0.0)), f"rigid piece {i} was modified"
    s, e = bounds[1]
    assert not torch.allclose(wj[s:e], torch.tensor(0.5))  # the transient absorbed the blend


def test_two_seams_sharing_a_free_piece_do_not_overlap():
    """Both blends must fit inside the transient, so neither reopens the other."""
    pieces = _pieces((_seg(40, joint=0.0), "loop", True), (_seg(10, joint=0.6), "transient", False),
                     (_seg(40, joint=0.0), "loop", True))
    _, _, _, bounds, seams, _ = splice_pieces(pieces, blend_n=12, dt=DT)
    eased = [s for s in seams if s.get("eased")]
    assert len(eased) == 2
    n_free = bounds[1][1] - bounds[1][0]
    assert all(s["blend_n_used"] <= n_free // 2 for s in eased), eased
    assert all(s["clamped"] for s in eased)               # 12 does not fit twice in 10 frames


def test_easing_is_order_independent():
    """Capturing all offsets up front is what makes this true; capturing lazily would not."""
    def build():
        return _pieces((_seg(40, joint=0.0), "loop", True), (_seg(24, joint=0.4), "transient", False),
                       (_seg(40, joint=0.1), "loop", True))
    a = splice_pieces(build(), blend_n=8, dt=DT)
    b = splice_pieces(build()[::-1][::-1], blend_n=8, dt=DT)   # same list, rebuilt
    assert torch.allclose(a[2], b[2], atol=0) and torch.allclose(a[0], b[0], atol=0)


def test_seam_joint_discontinuity_is_reduced_by_easing():
    pieces = _pieces((_seg(40, joint=0.0), "loop", True), (_seg(30, joint=0.7), "transient", False))
    raw = splice_pieces(pieces, blend_n=0, dt=DT)
    eased = splice_pieces(pieces, blend_n=10, dt=DT)
    i = raw[3][1][0]
    d_raw = float(torch.linalg.norm(raw[2][i] - raw[2][i - 1]))
    d_eased = float(torch.linalg.norm(eased[2][i] - eased[2][i - 1]))
    assert d_eased < 0.1 * d_raw, (d_raw, d_eased)


def test_anchored_piece_is_never_moved():
    """The terrain replay defines the frame: the run before it is pulled back onto it."""
    replay = _seg(20, x0=100.0, vx=1.0, joint=0.3)
    pieces = [Piece(seg=_seg(25, vx=1.0), kind="transient"),
              Piece(seg=replay, kind="replay", rigid=True, anchored=True)]
    wp, _, _, bounds, _, diag = splice_pieces(pieces, blend_n=0, dt=DT)
    s, e = bounds[1]
    assert torch.allclose(wp[s:e], replay[0], atol=1e-5)   # untouched, still out at x=100
    assert wp[0, 0] < 100.0                                # the approach was moved back to reach it
    assert diag["anchored"] == 1


def test_two_anchored_pieces_are_refused():
    pieces = [Piece(seg=_seg(10), kind="replay", rigid=True, anchored=True),
              Piece(seg=_seg(10), kind="transient"),
              Piece(seg=_seg(10), kind="replay", rigid=True, anchored=True)]
    with pytest.raises(NotImplementedError, match="over-constrained"):
        splice_pieces(pieces, blend_n=4, dt=DT)


def test_many_alternating_pieces():
    """The shape a real 20 s flat clip has -- this is what the 3-slot splicer could not express."""
    pieces = []
    for k in range(9):
        loop = k % 2 == 0
        pieces.append(Piece(seg=_seg(40 if loop else 18, joint=0.05 * k),
                            kind="loop" if loop else "transient", rigid=loop,
                            stride_idx=k if loop else -1))
    wp, _, _, bounds, seams, diag = splice_pieces(pieces, blend_n=8, dt=DT)
    assert len(bounds) == 9 and len(seams) == 8
    assert bounds[-1][1] == wp.shape[0] == diag["n_frames"]
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(8))     # contiguous, no gaps
    assert all(s.get("eased") for s in seams)                          # loop<->transient throughout


def test_trim_is_applied_before_placement():
    """A trim changes WHICH frame meets the neighbour, so it has to precede alignment."""
    full = _seg(30, joint=0.2)
    a = splice_pieces([Piece(seg=full, kind="loop", rigid=True)], blend_n=0)
    b = splice_pieces([Piece(seg=full, kind="loop", rigid=True, trim=(5, 4))], blend_n=0)
    assert a[0].shape[0] == 30 and b[0].shape[0] == 21
    assert torch.allclose(b[0], full[0][5:26])


def test_trim_that_empties_a_piece_is_refused():
    with pytest.raises(ValueError, match="trim"):
        splice_pieces([Piece(seg=_seg(6), kind="transient", trim=(4, 4))], blend_n=0)
