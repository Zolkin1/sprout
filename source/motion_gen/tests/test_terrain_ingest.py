# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the terrain traj-opt ingest + the include-by-name filter.

Uses a synthetic terrain ``.npz`` + manifest (no external HF data) so it runs in CI.
"""

from __future__ import annotations

import json

import numpy as np

from motion_gen.ingest.bones_ingest import filter_rows
from motion_gen.ingest.terrain_ingest import ingest_terrain_library

_ANKLES = ["robot/left_ankle_roll_link", "robot/right_ankle_roll_link"]
_WRISTS = ["robot/left_wrist_yaw_link", "robot/right_wrist_yaw_link"]


# --------------------------------------------------------------------------- #
# include_name_substrings filter
# --------------------------------------------------------------------------- #
def _rows(*names):
    return [{"filename": n, "package": "Locomotion", "is_mirror": "False"} for n in names]


def test_include_name_substrings_selects_families():
    rows = _rows("jump_over_obstacle_0_5m_R_002", "walking_neutral_001", "jump_on_box_R_001",
                "ladder_jump_on_R_001", "come_up_50cm_box_R_003")
    kept = [r["filename"] for r in filter_rows(
        rows, filters={"package": "Locomotion"},
        include_name_substrings=["jump_over_obstacle", "jump_on", "come_up"],
        exclude_name_substrings=["ladder"])]
    assert "jump_over_obstacle_0_5m_R_002" in kept
    assert "jump_on_box_R_001" in kept
    assert "come_up_50cm_box_R_003" in kept
    assert "walking_neutral_001" not in kept          # not in include list
    assert "ladder_jump_on_R_001" not in kept          # excluded wins over include


def test_include_empty_is_noop():
    rows = _rows("walking_001", "jump_on_002")
    kept = [r["filename"] for r in filter_rows(rows, include_name_substrings=None)]
    assert kept == ["walking_001", "jump_on_002"]


# --------------------------------------------------------------------------- #
# terrain ingest
# --------------------------------------------------------------------------- #
def _make_terrain_clip(path, *, skill, terrain, provenance, T=80, box_size_half=(0.4, 0.22, 0.23)):
    """Write a synthetic terrain .npz (state[T,71] + time + contacts + meta)."""
    state = np.zeros((T, 71), dtype=np.float64)
    state[:, 0] = 0.0
    state[:, 1] = -np.linspace(0.0, 0.5, T)           # travel along -y (like the real clips)
    state[:, 2] = 0.8
    state[:, 3] = 1.0                                  # quat w (wxyz identity)
    time = np.arange(T) * 0.005
    flags = np.zeros((T, 4), dtype=np.int8)
    flags[: T // 2, 0] = 1                             # left foot down first half
    flags[T // 2:, 1] = 1                              # right foot down second half
    meta = {"skill": skill, "terrain": terrain, "provenance": provenance}
    np.savez(path, state=state, time=time, contact_flags=flags,
             contact_bodies=np.array(_ANKLES + _WRISTS), meta=json.dumps(meta))


def _write_lib(tmp_path):
    (tmp_path / "jump_on").mkdir()
    terrain = {"type": "box", "box_height": 0.46,
               "boxes": [{"pos": [0.0, -0.5, 0.23], "size": [0.4, 0.22, 0.23], "yaw": 0.0}]}
    _make_terrain_clip(tmp_path / "jump_on" / "clip.npz", skill="jump_on",
                       terrain=terrain, provenance={"source_csv": "jump_on_x.csv", "crop": [99, 306]})
    (tmp_path / "manifest.csv").write_text(
        "skill,vel_x,vel_y,vel_yaw,terrain,box_height,tread,periodic,status,clip,file\n"
        "jump_on,0.0,0.0,0.0,box,0.46,,False,ok,jump_on_x.csv,jump_on/clip.npz\n")


def test_ingest_terrain_attaches_contacts_geometry_provenance(tmp_path):
    _write_lib(tmp_path)
    src = ingest_terrain_library(str(tmp_path), hf_repo=None, dt=0.02, device="cpu")
    assert len(src) == 1
    s = src[0]
    # steady clean clip, one-shot (periodic=False -> episodic)
    assert s.steady is True and s.wrap == "episodic"
    # contacts: two feet, aligned to the resampled frame count, binary
    assert s.contact is not None
    assert s.contact.shape == (s.n, 2)
    assert set(s.contact.unique().tolist()) <= {0, 1}
    # geometry: MuJoCo half-extents converted to FULL extents (x2)
    box = s.terrain_meta["terrain"]["boxes"][0]
    assert box["size"] == [0.8, 0.44, 0.46]
    assert s.terrain_meta["terrain"]["type"] == "box"
    # provenance carried through (bones lead-up handoff)
    assert s.provenance["crop"] == [99, 306]
    # heading recorded for later +x normalization
    assert "heading_yaw0" in s.terrain_meta


def test_ingest_terrain_skill_whitelist(tmp_path):
    _write_lib(tmp_path)
    assert ingest_terrain_library(str(tmp_path), skills=["stairs_down"]) == []
    assert len(ingest_terrain_library(str(tmp_path), skills=["jump_on"])) == 1
