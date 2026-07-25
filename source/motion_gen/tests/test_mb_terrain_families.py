# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the Motion-Bricks terrain family table + the boundary-frame gates."""

from __future__ import annotations

import numpy as np
import pytest

from motion_gen.motionbricks.terrain_families import (
    FAMILIES,
    SOLE_Z,
    boundary_candidates,
    episodic_boundaries,
    family_for,
    pick_boundary,
    sole_heights,
)


def _stride(T: int = 40, stance_z: float = 0.15, apex: float = 0.20, stance: str = "left"):
    """A synthetic single-support stride: one sole planted, the other arcing up to ``apex`` and back."""
    soles = np.full((T, 2), stance_z, dtype=np.float64)
    swing = stance_z + apex * np.sin(np.linspace(0.0, np.pi, T))
    contact = np.zeros((T, 2), dtype=np.uint8)
    if stance == "left":
        soles[:, 1] = swing
        contact[:, 0] = 1
    else:
        soles[:, 0] = swing
        contact[:, 1] = 1
    return soles, contact


def test_registry_covers_the_five_terrain_skills_with_consistent_levels():
    assert set(FAMILIES) == {"stairs_up", "stairs_down", "jump_on", "jump_off", "jump_off_front"}
    for skill, fam in FAMILIES.items():
        assert fam.skill == skill
        assert fam.geom in ("stairs", "box")
        assert fam.approach_level in ("lower", "upper") and fam.exit_level in ("lower", "upper")
        # Every family CHANGES level (that is what the terrain is for) and knows how to leave it.
        assert fam.approach_level != fam.exit_level
        assert fam.approach_mode in ("walk", "stand") and fam.exit_mode in ("walk", "stand")
        assert fam.exit_sign in (-1.0, 1.0)
        assert fam.terrain_tvel in ("steady", "hold_approach", "exit_walk")
    assert FAMILIES["stairs_up"].going_up and not FAMILIES["stairs_down"].going_up
    # Only an ASCENT pins the replay's future (measured: it halves the boundary-pose error there, and
    # doubles it on a descent, whose future drops below the platform the approach is still walking on).
    assert FAMILIES["stairs_up"].pin_future
    assert not any(f.pin_future for k, f in FAMILIES.items() if k != "stairs_up")
    assert FAMILIES["jump_off"].exit_sign == -1.0            # backward dismount walks away backwards
    assert FAMILIES["jump_off_front"].exit_sign == 1.0
    # Only the stair families loop; the box interactions are played once.
    assert [f.periodic for f in FAMILIES.values()].count(True) == 2


def test_family_for_unknown_skill_lists_the_options():
    with pytest.raises(KeyError, match="stairs_up"):
        family_for("moonwalk")


def test_sole_heights_subtracts_the_sole_offset():
    pos_names = ["pelvis:pos_z", "left_ankle_roll_link:pos_z", "right_ankle_roll_link:pos_z"]
    y = np.array([[0.8, 0.234, 0.534]], dtype=np.float32)
    s = sole_heights(y, pos_names)
    assert s.shape == (1, 2)
    assert s[0, 0] == pytest.approx(0.234 - SOLE_Z)
    assert s[0, 1] == pytest.approx(0.534 - SOLE_Z)


def test_gate_accepts_single_support_and_rejects_apex_and_ground_contact():
    soles, contact = _stride(T=40, stance_z=0.15, apex=0.20)
    g = boundary_candidates(soles, contact, [0.0, 0.15, 0.30], swing_clear=0.05, swing_band=0.15)
    assert g.strict and g.frames.size > 0
    swing = soles[g.frames, 1] - 0.15
    assert swing.min() > 0.05 and swing.max() <= 0.15      # only the band, never the apex
    assert 20 not in g.frames.tolist()                     # frame 20 is the apex (0.20 above the tread)


def test_gate_needs_contact_not_just_height():
    soles, contact = _stride()
    g = boundary_candidates(soles, np.zeros_like(contact), [0.15], swing_clear=0.05, swing_band=0.15)
    assert g.frames.size == 0 and not g.strict and g.n_planted == 0


def test_gate_falls_back_to_contact_only_when_no_swing_frame_qualifies():
    # Both feet flat on the tread: planted, but the swing foot never lifts -> no single-support frame.
    soles = np.full((12, 2), 0.15)
    contact = np.ones((12, 2), np.uint8)
    g = boundary_candidates(soles, contact, [0.15], swing_clear=0.05, swing_band=0.15)
    assert not g.strict and g.frames.size == 12 and g.n_planted == 12
    assert "fell back" in g.reason


def test_gate_matches_the_nearest_level_of_a_staircase():
    # A stride two treads up: the stance sole is at 0.45, so only that level may accept it.
    soles, contact = _stride(stance_z=0.45, apex=0.10)
    levels = [0.15, 0.30, 0.45, 0.60]
    assert boundary_candidates(soles, contact, levels, swing_clear=0.02, swing_band=0.10).frames.size > 0
    # Drop 0.45 from the levels: the nearest is now 0.30 / 0.60, both 0.15 away -> nothing is planted.
    g = boundary_candidates(soles, contact, [0.15, 0.30, 0.60], swing_clear=0.02, swing_band=0.10)
    assert g.frames.size == 0 and g.n_planted == 0


def test_gate_window_restricts_to_the_first_period():
    soles, contact = _stride(T=60)
    full = boundary_candidates(soles, contact, [0.15], swing_clear=0.05, swing_band=0.15)
    win = boundary_candidates(soles, contact, [0.15], swing_clear=0.05, swing_band=0.15, window=(0, 20))
    assert win.frames.size and win.frames.max() < 20
    assert set(win.frames.tolist()) <= set(full.frames.tolist())


def test_gate_accepts_either_foot_as_the_stance():
    for stance in ("left", "right"):
        soles, contact = _stride(stance=stance)
        assert boundary_candidates(soles, contact, [0.15], swing_clear=0.05, swing_band=0.15).strict


def test_pick_boundary_prefers_first_middle_last():
    soles, contact = _stride()
    g = boundary_candidates(soles, contact, [0.15], swing_clear=0.05, swing_band=0.15)
    f = g.frames.tolist()
    assert pick_boundary(g, prefer="first") == f[0]
    assert pick_boundary(g, prefer="last") == f[-1]
    assert pick_boundary(g, prefer="middle") == f[len(f) // 2]
    with pytest.raises(ValueError):
        pick_boundary(g, prefer="whenever")


def test_pick_boundary_on_empty_gate_reports_the_reason():
    empty = boundary_candidates(np.zeros((5, 2)), np.zeros((5, 2), np.uint8), [1.0])
    with pytest.raises(ValueError, match="no boundary candidates"):
        pick_boundary(empty)


def test_episodic_boundaries_are_first_and_last():
    assert episodic_boundaries(87) == (0, 86)
    with pytest.raises(ValueError):
        episodic_boundaries(1)


def test_gate_validates_input_shapes():
    with pytest.raises(ValueError):
        boundary_candidates(np.zeros((5, 3)), np.zeros((5, 2), np.uint8), [0.0])
    with pytest.raises(ValueError):
        boundary_candidates(np.zeros((5, 2)), np.zeros((4, 2), np.uint8), [0.0])
