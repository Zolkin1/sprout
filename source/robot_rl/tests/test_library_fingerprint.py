# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the portable trajectory-library fingerprint.

The fingerprint must be: deterministic, independent of the absolute path
(so two machines hashing the same library agree), sensitive to any content or
layout change, and report accurate per-skill counts.
"""

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_fingerprint import (
    discover_skill_folders,
    fingerprint_library,
    fingerprint_skill_folders,
)


def _make_library(root, layout: dict[str, dict[str, str]]) -> None:
    """Write a ``{skill: {filename: content}}`` library tree under ``root``."""
    for skill, files in layout.items():
        skill_dir = root / skill
        skill_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (skill_dir / name).write_text(content)


_LAYOUT = {
    "walk_forward": {"w_20.yaml": "a: 1\n", "w_40.yaml": "a: 2\n"},
    "running": {"r_200.yaml": "b: 3\n"},
}


def test_deterministic(tmp_path):
    """Re-fingerprinting the same library yields an identical hash."""
    _make_library(tmp_path, _LAYOUT)
    assert fingerprint_library(tmp_path)["hash"] == fingerprint_library(tmp_path)["hash"]


def test_path_independent(tmp_path):
    """The same contents under a different absolute path hash identically."""
    a, b = tmp_path / "machine_a", tmp_path / "machine_b"
    _make_library(a, _LAYOUT)
    _make_library(b, _LAYOUT)
    assert fingerprint_library(a)["hash"] == fingerprint_library(b)["hash"]


def test_manifest_counts(tmp_path):
    """``skill_counts`` and ``num_trajectories`` reflect the library."""
    _make_library(tmp_path, _LAYOUT)
    fp = fingerprint_library(tmp_path)
    assert fp["skill_counts"] == {"running": 1, "walk_forward": 2}
    assert fp["num_trajectories"] == 3
    assert fp["short"] == fp["hash"][:12]


def test_content_change_changes_hash(tmp_path):
    """Editing a trajectory file changes the hash."""
    _make_library(tmp_path, _LAYOUT)
    before = fingerprint_library(tmp_path)["hash"]
    (tmp_path / "running" / "r_200.yaml").write_text("b: 999\n")
    assert fingerprint_library(tmp_path)["hash"] != before


def test_added_file_changes_hash(tmp_path):
    """Adding a trajectory file changes the hash."""
    _make_library(tmp_path, _LAYOUT)
    before = fingerprint_library(tmp_path)["hash"]
    (tmp_path / "walk_forward" / "w_60.yaml").write_text("a: 3\n")
    assert fingerprint_library(tmp_path)["hash"] != before


def test_rename_changes_hash(tmp_path):
    """Renaming a file (same content) changes the hash — layout is part of it."""
    _make_library(tmp_path, _LAYOUT)
    before = fingerprint_library(tmp_path)["hash"]
    src = tmp_path / "walk_forward" / "w_20.yaml"
    src.rename(tmp_path / "walk_forward" / "w_25.yaml")
    assert fingerprint_library(tmp_path)["hash"] != before


def test_discover_single_default_skill(tmp_path):
    """A flat folder of YAMLs is discovered as one ``default`` skill."""
    (tmp_path / "a.yaml").write_text("x: 1\n")
    (tmp_path / "b.yaml").write_text("x: 2\n")
    folders = discover_skill_folders(tmp_path)
    assert set(folders) == {"default"}
    assert fingerprint_skill_folders(folders)["skill_counts"] == {"default": 2}
