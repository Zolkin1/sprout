# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the shared forward swing-reach table.

Exercises :func:`traj_step_length_table` on the on-disk flat library (skipped
if absent).  The online command now delegates to this same function, so this
also covers the offline path used by the terrain-aware clip generator.  After
the two-gate full-periodic change (gate 0 = the mid-cycle strike) the table
reads the LANDING foot (from the contact schedule) at that gate, so it returns
the single first-stride reach — a positive forward distance for walk_forward.
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
    from motion_gen.multiskill_manager import (
        MultiSkillManager,
    )
    from motion_gen.clips.step_length_table import (
        expand_ref_frames,
        traj_step_length_table,
    )

    m = MultiSkillManager(str(_LIB), DEVICE, env=None)
    ref_frames = expand_ref_frames(list(m.ref_frames))
    m.build_ref_frame_map(ref_frames)
    table = traj_step_length_table(m, DEVICE)
    return m, table


def _skill_ids(m, needle: str) -> torch.Tensor:
    """Global trajectory indices whose skill label contains ``needle``."""
    labels = m.skill_labels
    return torch.tensor(
        [i for i, s in enumerate(labels) if needle in s], dtype=torch.long
    )


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


def test_walk_forward_first_step_reach_is_positive(manager_and_table):
    """Regression guard: the first-stride reach for walk_forward is forward (+x).

    Before the landing-foot fix the table resolved the swing foot via the
    (all-right-ankle) ``_ref_frame_map`` and read the wrong foot at the
    whole-cycle phase, which produced a non-forward / backward reach for
    walk_forward.  Reading the actual landing (left) foot at the mid-cycle gate
    must give a positive forward reach.
    """
    m, table = manager_and_table
    wf = _skill_ids(m, "walk_forward")
    if wf.numel() == 0:
        pytest.skip("no walk_forward trajectories in the library")
    reach = table[wf]
    assert torch.isfinite(reach).all()
    assert (reach > 0.0).all(), (
        f"walk_forward first-step reach must be forward; got min {reach.min():.3f} m"
    )


def test_full_periodic_two_gates(manager_and_table):
    """A1: full-periodic gaits get 2 gates (mid-cycle LEFT strike + cycle-wrap
    RIGHT strike); perpetual standing gets 0.
    """
    m, _ = manager_and_table
    for needle in ("walk_forward", "walk_backwards", "running"):
        ids = _skill_ids(m, needle)
        if ids.numel() == 0:
            continue
        t = int(ids[0])
        assert int(m._num_gates_per_traj[t]) == 2, f"{needle}: expected 2 gates"
        p0, p1 = float(m._gate_phi_table[t, 0]), float(m._gate_phi_table[t, 1])
        assert 0.0 < p0 < 1.0, f"{needle}: gate 0 should be the mid-cycle strike, got {p0}"
        assert p1 == 1.0, f"{needle}: gate 1 should be the cycle wrap, got {p1}"
        assert any("left" in b for b in m._gate_body_names_per_gate[t][0])
        assert any("right" in b for b in m._gate_body_names_per_gate[t][1])
    st = _skill_ids(m, "standing")
    if st.numel():
        assert int(m._num_gates_per_traj[int(st[0])]) == 0, "standing should be gate-less"


def test_landing_foot_at_first_gate_is_left_for_walk(manager_and_table):
    """The foot that lands at gate 0 of a full-periodic walk is the LEFT ankle.

    Gate 0 is the mid-cycle strike; its expected-contact body (A1) is the left
    ankle, and the step-length table's landing-foot resolution must agree.
    """
    from motion_gen.clips.step_length_table import (
        _landing_foot_pos_x_cols,
    )

    m, _ = manager_and_table
    wf = _skill_ids(m, "walk_forward")
    if wf.numel() == 0:
        pytest.skip("no walk_forward trajectories in the library")
    t = wf[:1]
    eps = m._EPS_PHI
    gate_phi = m._gate_phi_table[t, 0]
    post_dom = m._get_domain_indices(torch.remainder(gate_phi + eps, 1.0), t)
    col = _landing_foot_pos_x_cols(m, t, post_dom, DEVICE)
    assert m.pos_output_names[int(col[0])] == "left_ankle_roll_link:pos_x"
    # A1: gate 0 body names name the left ankle.
    assert m._gate_body_names_per_gate[int(t[0])][0] == ["left_ankle_roll_link"]
