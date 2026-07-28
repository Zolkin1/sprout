# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the TerrainPlan hand-off contract: round-trip, replay cut, and the MB-rate window samplers."""

from __future__ import annotations

import numpy as np
import pytest

from motion_gen.motionbricks.terrain_plan import (
    TerrainPlan,
    load_terrain_plan,
    sample_qpos,
    save_terrain_plan,
)

_DT = 0.02


def _plan(F: int = 100, entry: int = 10, exit_: int = 80, **kw) -> TerrainPlan:
    """A synthetic plan: root climbs +x and +z linearly, identity quat, joints ramp per index."""
    q = np.zeros((F, 36), dtype=np.float32)
    q[:, 0] = np.arange(F) * 0.01                       # 0.5 m/s at dt=0.02
    q[:, 2] = 0.8 + np.arange(F) * 0.005                # rising 0.25 m/s
    q[:, 3] = 1.0
    q[:, 7:36] = (np.arange(F)[:, None] * 0.001) * np.arange(29)[None, :]
    contact = np.zeros((F, 2), np.uint8)
    contact[:, 0] = 1
    args = dict(skill="stairs_up", dt=_DT, terrain_qpos=q, terrain_contact=contact,
                entry_frame=entry, exit_frame=exit_, approach_surf_z=0.15, exit_surf_z=0.90,
                approach_vx=0.7, exit_vx=1.0, entry_vx=0.6, exit_speed=0.65,
                approach_mode="walk", exit_mode="walk", approach_s=2.5, exit_s=2.5,
                entry_candidates=np.array([10, 20, 30]), exit_candidates=np.array([80, 90]),
                meta={"boxes": [{"pos": [1.0, 0.0, 0.1], "size": [0.25, 1.4, 0.2], "yaw": 0.0}],
                      "geom": "stairs"})
    args.update(kw)
    return TerrainPlan(**args)


def test_replay_is_cut_to_the_boundary_frames():
    p = _plan(F=100, entry=10, exit_=80)
    assert p.n_frames == 100
    assert p.replay.shape == (71, 36) and p.replay_contact.shape == (71, 2)
    np.testing.assert_allclose(p.replay[0], p.terrain_qpos[10])
    np.testing.assert_allclose(p.replay[-1], p.terrain_qpos[80])
    assert len(p.boxes) == 1


def test_round_trip_through_npz(tmp_path):
    p = _plan()
    path = str(tmp_path / "plan.npz")
    save_terrain_plan(path, p)
    q = load_terrain_plan(path)
    for f in ("skill", "dt", "entry_frame", "exit_frame", "approach_surf_z", "exit_surf_z",
              "approach_vx", "exit_vx", "entry_vx", "exit_speed", "approach_mode", "exit_mode",
              "approach_s", "exit_s"):
        assert getattr(q, f) == pytest.approx(getattr(p, f)) if isinstance(getattr(p, f), float) \
            else getattr(q, f) == getattr(p, f)
    np.testing.assert_allclose(q.terrain_qpos, p.terrain_qpos)
    np.testing.assert_array_equal(q.terrain_contact, p.terrain_contact)
    np.testing.assert_array_equal(q.entry_candidates, p.entry_candidates)
    np.testing.assert_array_equal(q.exit_candidates, p.exit_candidates)
    assert q.meta["boxes"] == p.meta["boxes"]


def test_validation_rejects_bad_shapes_and_frame_order():
    with pytest.raises(ValueError, match="terrain_qpos"):
        _plan(terrain_qpos=np.zeros((10, 35), np.float32))
    with pytest.raises(ValueError, match="terrain_contact"):
        _plan(terrain_contact=np.zeros((99, 2), np.uint8))
    with pytest.raises(ValueError, match="outside the replay"):
        _plan(exit_=100)
    with pytest.raises(ValueError, match="must be after"):
        _plan(entry=50, exit_=50)


def test_sample_qpos_interpolates_and_clamps():
    q = np.zeros((5, 36), np.float32)
    q[:, 0] = np.arange(5)
    q[:, 3] = 1.0
    out = sample_qpos(q, np.array([-2.0, 0.5, 2.25, 99.0]))
    assert out.shape == (4, 36)
    np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 2.25, 4.0], atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=-1), 1.0, atol=1e-5)


def test_approach_pin_window_is_surface_relative_and_time_aligned():
    p = _plan(F=100, entry=10, exit_=80)
    dt_mb = 1.0 / 30.0
    win = p.approach_pin_window(0.0, 4, dt_mb)          # first frame == the entry
    assert win.shape == (4, 36)
    assert win[0, 2] == pytest.approx(p.terrain_qpos[10, 2] - p.approach_surf_z, abs=1e-5)
    np.testing.assert_allclose(win[:, 0:2], 0.0)        # xy zeroed (the root pin supplies it)
    assert win[3, 2] > win[0, 2]                        # the window walks INTO the replay (rising)
    # a positive offset reads further into the replay; a negative one clamps at the entry frame
    later = p.approach_pin_window(0.5, 4, dt_mb)
    assert later[0, 2] > win[0, 2]
    early = p.approach_pin_window(-5.0, 4, dt_mb)
    assert early[0, 2] == pytest.approx(p.terrain_qpos[0, 2] - p.approach_surf_z, abs=1e-5)


def test_exit_seed_window_ends_at_the_exit_frame_at_the_mb_rate():
    p = _plan(F=100, entry=10, exit_=80)
    dt_mb = 1.0 / 30.0
    win = p.exit_seed_window(4, dt_mb)
    assert win.shape == (4, 36)
    np.testing.assert_allclose(win[-1], p.terrain_qpos[80], atol=1e-5)
    # Spacing must be the MB frame period, not the plan's dt: 1/30 s of a 0.5 m/s root = ~1.67 cm.
    dx = np.diff(win[:, 0])
    np.testing.assert_allclose(dx, 0.5 * dt_mb, atol=1e-4)


def test_approach_pin_window_hold_is_one_repeated_entry_keyframe():
    """``hold`` pins exactly the entry frame, so the target does not move between replans.

    Without it, clamping ``t_offset`` to <= 0 still let the window's TAIL span forward into the replay,
    which asks the approach to start generating the interaction it is meant to hand over to.
    """
    plan = _plan()
    for t_offset in (-1.0, -0.3, 0.0):
        win = plan.approach_pin_window(t_offset, 8, 1.0 / 30.0, hold=True)
        assert win.shape == (8, 36)
        assert np.allclose(win, win[0])                      # one keyframe, repeated
    # ...and it is the ENTRY, surface-relative like the sliding window.
    held = plan.approach_pin_window(-1.0, 8, 1.0 / 30.0, hold=True)
    entry = plan.terrain_qpos[plan.entry_frame]
    assert held[0, 2] == pytest.approx(entry[2] - plan.approach_surf_z, abs=1e-5)
    assert np.allclose(held[0, 7:], entry[7:], atol=1e-5)
    assert np.allclose(held[:, 0:2], 0.0)
    # The sliding window does slide; the held one does not.
    slid = plan.approach_pin_window(0.0, 8, 1.0 / 30.0)
    assert not np.allclose(slid, slid[0])
