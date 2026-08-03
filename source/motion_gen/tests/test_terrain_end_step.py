# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the terrain-pinned run and the reset-start cap derived from it.

``MotionClipManager.terrain_end_step`` is what stops a terrain reset from spawning the
robot AFTER a clip's stair climb / box mount, on the flat top or walking away.  It is
derived from the per-step anchor mask, so the two easy ways to get it wrong -- reading the
RAW ``anchor_mask_t`` (where the ``255`` sentinel looks like "all DOFs free") and forgetting
to trim the repeat-last padding tail -- are covered explicitly.
"""

from __future__ import annotations

import torch

from motion_gen.clips.clip_io import ClipLibraryData
from motion_gen.clips.clip_manager import MotionClipManager
from motion_gen.world_lift import (
    _ANCHOR_MASK_BOX,
    _ANCHOR_MASK_FLAT,
    _ANCHOR_MASK_LEGACY,
    _ANCHOR_MASK_SLOPE,
    _ANCHOR_MASK_STAIRS,
    anchor_mask_is_pinned,
)

_UNSET = 255       # ClipLibraryData's "no per-step mask here" sentinel
_T = 12            # padded clip length used by the fixtures


def _library(rows: list[tuple[list[int], int, int]]) -> ClipLibraryData:
    """Build a minimal in-memory library from ``(per_step_mask, clip_len, per_clip_mask)``.

    Only the fields ``MotionClipManager`` touches at construction are populated; the pose
    tensors are zeros of the right rank.

    Args:
        rows: One tuple per clip. ``per_step_mask`` is padded/truncated to ``_T``.

    Returns:
        The assembled library.
    """
    n = len(rows)
    amt = torch.tensor([r[0] for r in rows], dtype=torch.uint8)
    return ClipLibraryData(
        y_des=torch.zeros(n, _T, 7),
        dy_des=torch.zeros(n, _T, 6),
        contact=torch.zeros(n, _T, 2, dtype=torch.uint8),
        skill_id=torch.zeros(n, _T, dtype=torch.uint8),
        stride_idx=torch.zeros(n, _T, dtype=torch.int32),
        target_vel=torch.zeros(n, _T, 3),
        terrain_id=torch.zeros(n, dtype=torch.int32),
        terrain_type_code=torch.zeros(n, dtype=torch.uint8),
        anchor_mask=torch.tensor([r[2] for r in rows], dtype=torch.uint8),
        clip_len=torch.tensor([r[1] for r in rows], dtype=torch.int32),
        anchor_mask_t=amt,
        meta={
            "pos_output_names": [f"p{i}" for i in range(7)],
            "vel_output_names": [f"v{i}" for i in range(6)],
            "ref_frame": "pelvis_link",
            "contact_frames": ["l", "r"],
            "dt": 0.02,
            "num_steps": _T,
        },
        attrs={},
    )


def test_is_pinned_matches_the_named_masks():
    masks = torch.tensor(
        [_ANCHOR_MASK_FLAT, _ANCHOR_MASK_LEGACY, _ANCHOR_MASK_STAIRS,
         _ANCHOR_MASK_BOX, _ANCHOR_MASK_SLOPE, 63],
        dtype=torch.uint8,   # the dtype the sentinel-resolved masks actually carry
    )
    # flat / legacy / fully-free are ego; stairs, box and slope are registered to terrain.
    expected = torch.tensor([False, False, True, True, True, False])
    assert torch.equal(anchor_mask_is_pinned(masks), expected)


def test_terrain_end_step_over_the_clip_families():
    f, s = _ANCHOR_MASK_FLAT, _ANCHOR_MASK_STAIRS
    lib = _library([
        # flat: ego throughout -> no pinned run, so the cap is clip_len - 1 (a no-op).
        ([f] * _T, _T, f),
        # pure terrain: pinned throughout -> ends at the last valid frame.
        ([s] * _T, _T, s),
        # two-sided transition: approach / CLIMB / walk on the raised top.
        ([f, f, f, s, s, s, f, f, f, f, f, f], _T, s),
        # one-sided box: the clip ENDS pinned, mid-mount.
        ([f, f, f, f, s, s, s, s, f, f, f, f], 8, s),
        # distractor: no per-step schedule, per-clip mask is ego -> no pinned run.
        ([_UNSET] * _T, _T, f),
    ])
    mgr = MotionClipManager(lib)
    assert mgr.terrain_end_step.tolist() == [_T - 1, _T - 1, 5, 7, _T - 1]


def test_padding_tail_does_not_extend_the_pinned_run():
    """A clip that ends pinned must stop at clip_len - 1, not run out to num_steps - 1.

    Clips are repeat-last padded, so frames past ``clip_len`` echo the final mask. An
    unmasked reverse scan would report ``_T - 1`` here instead of 7.
    """
    f, s = _ANCHOR_MASK_FLAT, _ANCHOR_MASK_STAIRS
    lib = _library([([f, f, f, f, s, s, s, s] + [s] * (_T - 8), 8, s)])
    assert int(MotionClipManager(lib).terrain_end_step[0]) == 7


def test_unset_sentinel_resolves_to_the_per_clip_mask():
    """255 must resolve via ``anchor_mask_per_step``, not be read as "all DOFs free".

    In its low 6 bits the sentinel is indistinguishable from 63, so a manager reading the
    RAW dataset would call these clips ego and report no pinned run.
    """
    lib = _library([
        ([_UNSET] * _T, _T, _ANCHOR_MASK_STAIRS),   # pinned via the per-clip mask
        ([_UNSET] * _T, _T, _ANCHOR_MASK_FLAT),     # ego via the per-clip mask
    ])
    assert MotionClipManager(lib).terrain_end_step.tolist() == [_T - 1, _T - 1]
    # The distinction is real even though both land on clip_len - 1 here: the first clip is
    # pinned end-to-end, the second has no pinned frame at all.
    pinned = anchor_mask_is_pinned(MotionClipManager(lib).anchor_mask_t)
    assert bool(pinned[0].all()) and not bool(pinned[1].any())


def test_gather_terrain_end_step_is_device_safe_and_per_env():
    f, s = _ANCHOR_MASK_FLAT, _ANCHOR_MASK_STAIRS
    lib = _library([
        ([f, f, s, s, f, f, f, f, f, f, f, f], _T, s),
        ([f] * _T, _T, f),
    ])
    mgr = MotionClipManager(lib)
    clip_id = torch.tensor([0, 1, 0])
    out = mgr.gather_terrain_end_step(clip_id)
    assert out.tolist() == [3, _T - 1, 3]
    assert out.device == clip_id.device
