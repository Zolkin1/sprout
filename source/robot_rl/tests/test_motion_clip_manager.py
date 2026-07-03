# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the dense-clip gather manager (MotionClipManager).

Synthetic ClipLibraryData (no IsaacLab env, no real library) checks the two
pure operations the runtime relies on: output-dimension reordering
(``order_outputs``) and per-env ``(clip_id, step)`` gather.
"""

from __future__ import annotations

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_io import (
    ClipLibraryData,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_manager import (
    MotionClipManager,
)

DEVICE = "cpu"


def _synthetic_lib(num_clips=4, T=10) -> ClipLibraryData:
    pos_names = ["pelvis_link:pos_x", "pelvis_link:pos_y", "joint:a", "joint:b"]
    vel_names = ["pelvis_link:pos_x", "pelvis_link:pos_y", "joint:a", "joint:b"]
    P, V, C = len(pos_names), len(vel_names), 2
    # y_des[c, t, k] = c*1000 + t*10 + k  -> unambiguous for gather/permute checks.
    base = (
        torch.arange(num_clips)[:, None, None] * 1000
        + torch.arange(T)[None, :, None] * 10
        + torch.arange(P)[None, None, :]
    ).float()
    meta = {
        "dt": 0.02, "clip_length_s": T * 0.02, "num_steps": T,
        "pos_output_names": pos_names, "vel_output_names": vel_names,
        "contact_frames": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "ref_frame": "right_ankle_roll_link",
        "skill_list": ["walk_forward"],
    }
    return ClipLibraryData(
        y_des=base.clone(),
        dy_des=base.clone() + 0.5,
        contact=torch.zeros(num_clips, T, C, dtype=torch.uint8),
        skill_id=torch.zeros(num_clips, T, dtype=torch.uint8),
        stride_idx=torch.zeros(num_clips, T, dtype=torch.int32),
        target_vel=torch.zeros(num_clips, T, 3),
        terrain_id=torch.full((num_clips,), -1, dtype=torch.int32),
        terrain_type_code=torch.zeros(num_clips, dtype=torch.uint8),
        meta=meta, attrs={},
    )


def test_metadata_surface():
    m = MotionClipManager(_synthetic_lib())
    assert m.get_pos_output_names == ["pelvis_link:pos_x", "pelvis_link:pos_y", "joint:a", "joint:b"]
    assert m.get_num_pos_outputs() == 4
    assert m.get_reference_frames() == ["right_ankle_roll_link"]
    assert m.num_steps == 10 and m.num_clips == 4


def test_order_outputs_permutes_columns():
    m = MotionClipManager(_synthetic_lib())
    new_order = ["joint:b", "pelvis_link:pos_x", "joint:a", "pelvis_link:pos_y"]
    before = m.lib.y_des.clone()
    m.order_outputs(new_order, new_order)
    assert m.get_pos_output_names == new_order
    # Column k of the reordered tensor == column (old index of new_order[k]).
    old = ["pelvis_link:pos_x", "pelvis_link:pos_y", "joint:a", "joint:b"]
    for k, name in enumerate(new_order):
        assert torch.equal(m.lib.y_des[:, :, k], before[:, :, old.index(name)])


def test_gather_desired_indexes_per_env():
    m = MotionClipManager(_synthetic_lib())
    clip_id = torch.tensor([0, 3, 1])
    step = torch.tensor([0, 5, 9])
    y, dy = m.gather_desired(clip_id, step)
    assert y.shape == (3, 4)
    # y_des[c, t, k] == c*1000 + t*10 + k
    assert torch.equal(y[0], torch.tensor([0.0, 1, 2, 3]))          # c0 t0
    assert torch.equal(y[1], torch.tensor([3050.0, 3051, 3052, 3053]))  # c3 t5
    assert torch.equal(y[2], torch.tensor([1090.0, 1091, 1092, 1093]))  # c1 t9
    assert torch.allclose(dy, y + 0.5)
