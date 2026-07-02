# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the shared forward swing-reach table.

Exercises :func:`traj_step_length_table` on the on-disk flat library (skipped
if absent).  The online command now delegates to this same function, so this
also covers the offline path used by the terrain-aware clip generator:
building a manager with ``env=None``, calling ``build_ref_frame_map``, and
constructing ``ref_frame_to_pos_xy_idx`` from the manager.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_LIB = Path("hf/trajectories/retargeted/2026-06-13_15-10-58_merged")
pytestmark = pytest.mark.skipif(not _LIB.exists(), reason=f"stride library {_LIB} not present")

DEVICE = "cpu"


@pytest.fixture(scope="module")
def manager_and_table():
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
        MultiSkillManager,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.step_length_table import (
        build_ref_frame_to_pos_xy_idx,
        expand_ref_frames,
        traj_step_length_table,
    )

    m = MultiSkillManager(str(_LIB), DEVICE, env=None)
    ref_frames = expand_ref_frames(list(m.ref_frames))
    m.build_ref_frame_map(ref_frames)
    idx = build_ref_frame_to_pos_xy_idx(m, ref_frames, DEVICE)
    table = traj_step_length_table(m, idx, DEVICE)
    return m, table


def test_shape_matches_num_trajectories(manager_and_table):
    m, table = manager_and_table
    assert table.shape == (m.num_trajectories,)


def test_gated_finite_gateless_inf(manager_and_table):
    """Trajectories with contact gates get a finite reach; gate-less get inf."""
    m, table = manager_and_table
    gated = m._num_gates_per_traj > 0
    assert torch.isfinite(table[gated]).all(), "gated trajectory has non-finite reach"
    assert torch.isinf(table[~gated]).all(), "gate-less trajectory should be inf"


def test_gated_reach_is_bounded(manager_and_table):
    """Forward swing reach is a plausible single-stride distance.

    Loose upper bound only — running strides legitimately reach ~2.4 m; the
    point is to catch garbage (NaN-ish blowups), not to pin the value.
    """
    m, table = manager_and_table
    gated = m._num_gates_per_traj > 0
    reach = table[gated]
    assert (reach.abs() < 4.0).all(), f"implausible step reach: max {reach.abs().max():.3f} m"
