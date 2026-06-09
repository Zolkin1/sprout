# Copyright 2026 Zachary Olkin. All rights reserved.

"""Unit tests for the visualization-recording I/O (numpy only, no Isaac Sim).

Run with: ``pytest source/robot_rl/robot_rl/flow_matching/tests`` inside the
``env_isaaclab`` venv.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from robot_rl.flow_matching.viz.recording import (
    VizRecordingMeta,
    find_newest_recording,
    load_viz_recording,
    save_viz_recording,
    unique_output_path,
)


def _make_recording(num_scan_rays: int, with_terrain_colors: bool) -> tuple[dict[str, np.ndarray], VizRecordingMeta]:
    """Build a small synthetic recording (T=6, N=2, nj=4, F=5, P=3)."""
    t, n, nj, f, p, c = 6, 2, 4, 5, 3, 3
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {
        "root_pos": rng.normal(size=(t, n, 3)).astype(np.float32),
        "root_quat_xyzw": rng.normal(size=(t, n, 4)).astype(np.float32),
        "joint_pos": rng.normal(size=(t, n, nj)).astype(np.float32),
        "vel_cmd": rng.normal(size=(t, n, c)).astype(np.float32),
        "done": rng.integers(0, 2, size=(t, n)).astype(np.uint8),
        "pred_index": np.full((t, n), -1, dtype=np.int32),
        "pred_root_pos": rng.normal(size=(p, f, 3)).astype(np.float32),
        "pred_root_quat_xyzw": rng.normal(size=(p, f, 4)).astype(np.float32),
        "pred_joint_pos": rng.normal(size=(p, f, nj)).astype(np.float32),
        "terrain_vertices": rng.normal(size=(8, 3)).astype(np.float32),
        "terrain_faces": rng.integers(0, 8, size=(4, 3)).astype(np.int32),
    }
    data["pred_index"][4:, 0] = 2
    if num_scan_rays > 0:
        scan = rng.normal(size=(t, n, num_scan_rays, 3)).astype(np.float32)
        scan[0, 0, 0] = np.inf  # missed rays must survive the round trip
        data["scan_hits"] = scan
    if with_terrain_colors:
        data["terrain_vertex_colors"] = rng.integers(0, 256, size=(8, 4)).astype(np.uint8)
    meta = VizRecordingMeta(
        urdf_path="robot_assets/g1/g1_21j/g1_21j.urdf",
        task_name="Test-Task",
        fm_checkpoint="",
        recorded_at="2026-06-09T00:00:00",
        quat_format="xyzw",
        dry_run=True,
        dt=0.02,
        update_every=2,
        num_envs=n,
        num_steps=t,
        history_len=3,
        future_len=f,
        num_joints=nj,
        num_scan_rays=num_scan_rays,
        joint_names=[f"joint_{i}" for i in range(nj)],
        terrain_source="generator",
    )
    return data, meta


@pytest.mark.parametrize("num_scan_rays,with_colors", [(7, True), (0, False)])
def test_save_load_round_trip(tmp_path, num_scan_rays: int, with_colors: bool):
    """Arrays and metadata survive a save/load round trip, optional keys included."""
    data, meta = _make_recording(num_scan_rays, with_colors)
    path = str(tmp_path / "viz_test.npz")
    save_viz_recording(path, data, meta)

    loaded, loaded_meta = load_viz_recording(path)
    assert loaded_meta == meta
    assert set(loaded.keys()) == set(data.keys())
    for key, arr in data.items():
        np.testing.assert_array_equal(loaded[key], arr, err_msg=key)
        assert loaded[key].dtype == arr.dtype
    assert ("scan_hits" in loaded) == (num_scan_rays > 0)
    assert ("terrain_vertex_colors" in loaded) == with_colors


def test_find_newest_recording(tmp_path):
    """Picks the most recently modified viz_*.npz and errors on an empty dir."""
    with pytest.raises(FileNotFoundError, match="No viz_"):
        find_newest_recording(str(tmp_path))

    data, meta = _make_recording(0, False)
    old = str(tmp_path / "viz_old.npz")
    new = str(tmp_path / "viz_new.npz")
    save_viz_recording(old, data, meta)
    save_viz_recording(new, data, meta)
    os.utime(old, (1.0, 1.0))
    os.utime(new, (2.0, 2.0))
    assert find_newest_recording(str(tmp_path)) == os.path.abspath(new)


def test_unique_output_path(tmp_path):
    """Colliding timestamped names get numeric suffixes instead of overwriting."""
    first = unique_output_path(str(tmp_path))
    assert first.endswith(".npz") and os.path.basename(first).startswith("viz_")
    open(first, "w").close()
    second = unique_output_path(str(tmp_path))
    assert second != first
