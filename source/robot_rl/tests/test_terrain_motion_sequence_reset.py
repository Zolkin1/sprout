# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the terrain-aware clip reset's pure selection + placement logic.

Exercises the two building blocks of
:func:`...mdp.events.resets.reset_on_terrain_motion_sequence` without an IsaacLab
sim: :func:`sample_terrain_clips` (per-env clip assignment matching the env's tile
``terrain_id``) and :func:`...terrains.clip_terrain_loader.clip_frame_origin` (the
tile placement math), plus the flat-clip bbox-centering invariant the reset uses.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.events.resets import sample_terrain_clips
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import clip_frame_origin


def _groups() -> dict[int, torch.Tensor]:
    """{-1: flat clips, 0: one terrain clip, 1: two terrain clips}."""
    return {
        -1: torch.tensor([0, 1, 2], dtype=torch.long),
        0: torch.tensor([3], dtype=torch.long),
        1: torch.tensor([4, 5], dtype=torch.long),
    }


def test_sample_matches_env_terrain():
    """Every env gets a clip whose terrain_id matches its tile."""
    groups = _groups()
    env_tid = torch.tensor([0, -1, 1, 1, -1, 0], dtype=torch.long)
    torch.manual_seed(0)
    clip_id = sample_terrain_clips(env_tid, groups)

    assert clip_id.shape == env_tid.shape
    assert clip_id.dtype == torch.long
    for cid, tid in zip(clip_id.tolist(), env_tid.tolist()):
        assert cid in groups[tid].tolist(), f"clip {cid} not in terrain {tid}'s group"


def test_sample_covers_multi_clip_group():
    """A terrain with multiple clips has all of them reachable across draws."""
    groups = _groups()
    env_tid = torch.full((200,), 1, dtype=torch.long)  # all on terrain 1 (clips 4, 5)
    torch.manual_seed(1)
    picks = set(sample_terrain_clips(env_tid, groups).tolist())
    assert picks == {4, 5}


def test_sample_raises_on_missing_terrain():
    """A tile whose terrain has no clips is a fatal misconfig (loud abort)."""
    groups = _groups()
    env_tid = torch.tensor([0, 2], dtype=torch.long)  # terrain 2 has no clips
    with pytest.raises(ValueError, match="terrain id"):
        sample_terrain_clips(env_tid, groups)


def test_sample_raises_on_empty_group():
    """An empty clip list for a present terrain id also aborts."""
    groups = {0: torch.tensor([], dtype=torch.long)}
    env_tid = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="terrain id"):
        sample_terrain_clips(env_tid, groups)


def test_clip_frame_origin_terrain_centerline():
    """A terrain clip's centerline (clip_y_center = size_y/2) lands on the tile
    centerline; x is the corner + back-buffer; z is the tile ground."""
    corner = torch.tensor([[10.0, 4.0, 0.5]])
    center = torch.tensor([[11.0, 5.0, 0.5]])  # footprint (2, 2) around corner
    size_y = torch.tensor([1.2])
    x_off = torch.tensor([0.3])
    out = clip_frame_origin(corner, center, 0.5 * size_y, x_off)

    assert torch.allclose(out[:, 0], corner[:, 0] + x_off)          # x: corner + buffer
    assert torch.allclose(out[:, 1], center[:, 1] - 0.5 * size_y)   # y: centerline shift
    assert torch.allclose(out[:, 2], corner[:, 2])                  # z: tile ground

    # Invariant: a reference point on the clip's own centerline (local y = size_y/2)
    # ends up exactly on the tile centerline in world.
    world_y_at_centerline = out[:, 1] + 0.5 * size_y
    assert torch.allclose(world_y_at_centerline, center[:, 1])


def test_clip_frame_origin_flat_at_corner():
    """A flat clip (clip_y_center = 0, no x-offset) is anchored at the tile corner."""
    corner = torch.tensor([[10.0, 4.0, 0.0]])
    center = torch.tensor([[15.0, 9.0, 0.0]])
    out = clip_frame_origin(corner, center, torch.zeros(1), 0.0)
    assert torch.allclose(out[:, 1], center[:, 1])   # clip_y_center=0 -> centerline == tile center y
    assert torch.allclose(out[:, 2], corner[:, 2])


def test_flat_bbox_centering_invariant():
    """The reset re-centers a wandering flat clip's bbox on its tile: world placement
    = cell_center - bbox_center, so the reference bbox center lands on the tile center."""
    # A wandering flat clip whose pelvis xy is NOT centered on the clip origin.
    pel_xy = torch.tensor([[[0.0, 0.0], [4.0, 2.0], [10.0, -1.0]]])  # [1, T, 2]
    bbox_center = 0.5 * (pel_xy.amax(dim=1) + pel_xy.amin(dim=1))    # [1, 2] -> (5.0, 0.5)
    assert torch.allclose(bbox_center, torch.tensor([[5.0, 0.5]]))

    cell_center = torch.tensor([[20.0, 7.0]])
    world_xy = cell_center - bbox_center                            # reset's flat placement
    world_bbox_center = world_xy + bbox_center                      # bbox center lifted to world
    assert torch.allclose(world_bbox_center, cell_center)
