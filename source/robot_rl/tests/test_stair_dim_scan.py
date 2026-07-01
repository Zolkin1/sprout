# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the stair-dimension library scan helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_rl.tasks.manager_based.robot_rl.terrains.config.stair_dim_scan import (
    scan_stair_step_dims,
)


def _write_traj(folder: Path, name: str, height: float | None, length: float | None) -> None:
    """Write a minimal trajectory YAML with a conditioner block."""
    lines = ["conditioner:", "  terrain: stairs"]
    if height is not None:
        lines.append(f"  terrain_height: {height}")
    if length is not None:
        lines.append(f"  terrain_length: {length}")
    lines += [
        "  terrain_width: 0.8",
        "  vel_x: 0.4",
        "  vel_y: 0.0",
        "  vel_yaw: 0.0",
        "domain_sequence: [single_support]",
        "name: stairs_test",
    ]
    (folder / name).write_text("\n".join(lines) + "\n")


@pytest.fixture()
def stair_library(tmp_path: Path) -> Path:
    """Library root with a stair_up folder: 3 unique dims + 1 duplicate."""
    stair_dir = tmp_path / "stair_up"
    stair_dir.mkdir()
    _write_traj(stair_dir, "a.yaml", 0.10, 0.18)
    _write_traj(stair_dir, "b.yaml", 0.14, 0.26)
    _write_traj(stair_dir, "c.yaml", 0.10, 0.26)
    # Duplicate of a.yaml's dims with float noise below the rounding digit.
    _write_traj(stair_dir, "d.yaml", 0.1000000001, 0.18)
    return tmp_path


class TestScanStairStepDims:
    def test_unique_sorted_pairs(self, stair_library):
        dims = scan_stair_step_dims(str(stair_library), hf_repo=None)
        assert dims == [(0.10, 0.18), (0.10, 0.26), (0.14, 0.26)]

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Stair trajectory folder"):
            scan_stair_step_dims(str(tmp_path / "nope"), hf_repo=None)

    def test_missing_keys_raises(self, tmp_path):
        stair_dir = tmp_path / "stair_up"
        stair_dir.mkdir()
        _write_traj(stair_dir, "bad.yaml", None, 0.20)
        with pytest.raises(ValueError, match="terrain_height"):
            scan_stair_step_dims(str(tmp_path), hf_repo=None)

    def test_empty_folder_raises(self, tmp_path):
        (tmp_path / "stair_up").mkdir()
        with pytest.raises(ValueError, match="No trajectory YAML"):
            scan_stair_step_dims(str(tmp_path), hf_repo=None)
