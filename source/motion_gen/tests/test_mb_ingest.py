# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for Motion Bricks raw-motion ingest: resample, joint reindex, fit-to-length, round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motion_gen.motion_matching.features import G1_JOINT_NAMES
from motion_gen.motionbricks.mb_ingest import (
    fit_to_length,
    raw_to_world,
    resample_rows,
    resample_times,
)
from motion_gen.motionbricks.raw_schema import RawMotion, load_raw_motion, save_raw_motion


def _make_raw(T: int = 61, fps: float = 30.0, joint_names: list[str] | None = None) -> RawMotion:
    """A synthetic RawMotion: x ramps 0->1, fixed pelvis height + identity quat, joints ramp per-index."""
    joint_names = joint_names or list(G1_JOINT_NAMES)
    q = np.zeros((T, 36), dtype=np.float32)
    q[:, 0] = np.linspace(0.0, 1.0, T)              # root x ramp
    q[:, 2] = 0.75                                  # pelvis height
    q[:, 3] = 1.0                                   # wxyz identity quaternion
    q[:, 7:36] = np.linspace(0.0, 1.0, T)[:, None] * np.arange(29)[None, :] * 0.01
    target_vel = np.tile(np.array([0.5, 0.0, 0.0], np.float32), (T, 1))
    return RawMotion(mujoco_qpos=q, fps=fps, joint_names=joint_names, target_vel=target_vel, kind="flat")


def test_resample_times_length_and_endpoints():
    t = resample_times(61, 30.0, 50.0)              # 2.0 s -> 101 frames at 50 Hz
    assert len(t) == 101
    assert t[0] == 0.0 and t[-1] == 60.0
    assert np.all(np.diff(t) > 0)


def test_resample_times_single_frame():
    assert resample_times(1, 30.0, 50.0).tolist() == [0.0]


def test_raw_to_world_shapes_quat_norm_and_duration():
    raw = _make_raw(T=61, fps=30.0)
    root_pos, root_quat, joints, tvel = raw_to_world(raw, 0.02)
    L = root_pos.shape[0]
    assert L == 101                                 # 60/30 s preserved at 50 Hz
    assert root_pos.shape == (L, 3) and root_quat.shape == (L, 4)
    assert joints.shape == (L, 29) and tvel.shape == (L, 3)
    assert torch.allclose(root_quat.norm(dim=-1), torch.ones(L), atol=1e-5)
    assert root_pos[0, 0].item() == pytest.approx(0.0, abs=1e-4)
    assert root_pos[-1, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert root_pos[:, 2].mean().item() == pytest.approx(0.75, abs=1e-4)


def test_raw_to_world_reindexes_joints_by_name():
    # Raw file stores joints in REVERSED order; ingest must reorder to G1_JOINT_NAMES.
    src = list(reversed(G1_JOINT_NAMES))
    raw = _make_raw(T=30, fps=30.0, joint_names=src)
    _, _, joints, _ = raw_to_world(raw, 1.0 / 30.0)
    last_src_order = raw.mujoco_qpos[-1, 7:36]      # values as stored (src order)
    np.testing.assert_allclose(joints[-1].numpy(), last_src_order[::-1], atol=1e-4)


def test_fit_to_length_pad_truncate_exact():
    x = torch.arange(10 * 3, dtype=torch.float32).reshape(10, 3)
    xp, clen = fit_to_length(x, 15)
    assert xp.shape == (15, 3) and clen == 10
    assert torch.equal(xp[10:], x[-1:].expand(5, 3))    # repeat-last padding
    xt, clen2 = fit_to_length(x, 6)
    assert xt.shape == (6, 3) and clen2 == 6
    xe, clen3 = fit_to_length(x, 10)
    assert clen3 == 10 and torch.equal(xe, x)


def test_raw_schema_roundtrip(tmp_path):
    raw = _make_raw()
    path = str(tmp_path / "m.npz")
    save_raw_motion(path, raw)
    r2 = load_raw_motion(path)
    np.testing.assert_allclose(r2.mujoco_qpos, raw.mujoco_qpos)
    np.testing.assert_allclose(r2.target_vel, raw.target_vel)
    assert r2.joint_names == raw.joint_names
    assert r2.fps == raw.fps and r2.kind == "flat"
    assert r2.target_vel_snapped is None                 # absent channel stays absent, not zeros


def test_raw_schema_roundtrips_snapped_command(tmp_path):
    raw = _make_raw()
    raw.target_vel_snapped = raw.target_vel + np.array([0.13, 0.0, 0.0], np.float32)
    path = str(tmp_path / "m.npz")
    save_raw_motion(path, raw)
    r2 = load_raw_motion(path)
    assert r2.target_vel_snapped is not None
    np.testing.assert_allclose(r2.target_vel_snapped, raw.target_vel_snapped)
    np.testing.assert_allclose(r2.target_vel, raw.target_vel)      # the two channels stay distinct


def test_raw_schema_rejects_mismatched_snapped_command():
    raw = _make_raw(T=20)
    with pytest.raises(ValueError, match="target_vel_snapped"):
        RawMotion(mujoco_qpos=raw.mujoco_qpos, fps=raw.fps, joint_names=raw.joint_names,
                  target_vel=raw.target_vel, target_vel_snapped=np.zeros((19, 3), np.float32))


def test_resample_rows_matches_raw_to_world_clock():
    """A second command channel resampled standalone lands on the same clock as raw_to_world's."""
    raw = _make_raw(T=61)
    _, _, _, tvel = raw_to_world(raw, 0.02)
    rows = resample_rows(raw.target_vel, raw.fps, 50.0)
    assert rows.shape == tuple(tvel.shape)
    np.testing.assert_allclose(rows, tvel.numpy(), atol=1e-6)
