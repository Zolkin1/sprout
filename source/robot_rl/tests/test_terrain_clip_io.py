# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the terrain fields of the clip-library HDF5 format.

Round-trips per-clip ``terrain_id`` / ``terrain_type_code`` and the embedded
``terrain_manifest`` root attr through :class:`ClipLibraryWriter` /
:func:`load_clip_library`, and checks a pre-terrain library still loads (empty
terrain placeholders).  Pure HDF5 + numpy — no IsaacLab.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from motion_gen.clips.clip_io import (
    ClipLibraryWriter,
    load_clip_library,
)

T, P, V, C = 5, 4, 3, 2


def _meta():
    return {
        "dt": 0.02,
        "clip_length_s": T * 0.02,
        "num_steps": T,
        "pos_output_names": [f"f:pos_{a}" for a in "xyz"] + ["j:0"],
        "vel_output_names": [f"f:vel_{a}" for a in "xyz"],
        "contact_frames": ["l", "r"],
        "ref_frame": "r",
        "skill_list": ["walk", "stair_up"],
    }


def _batch(n, terrain_id, terrain_type):
    return {
        "y_des": torch.zeros(n, T, P),
        "dy_des": torch.zeros(n, T, V),
        "contact": torch.zeros(n, T, C, dtype=torch.uint8),
        "skill_id": torch.zeros(n, T, dtype=torch.uint8),
        "stride_idx": torch.zeros(n, T, dtype=torch.int32),
        "target_vel": torch.zeros(n, T, 3),
        "terrain_id": torch.full((n,), terrain_id, dtype=torch.int32),
        "terrain_type": terrain_type,
    }


def test_terrain_tags_and_manifest_roundtrip(tmp_path):
    """Per-clip tags + manifest survive a write/read cycle."""
    path = tmp_path / "lib.h5"
    manifest = [
        {"terrain_id": 0, "type": "terrain", "dims": {"h": 0.16, "d": 0.30, "w": 0.777}, "num_steps": 6},
        {"terrain_id": 1, "type": "transition", "dims": {"h": 0.12, "d": 0.25, "w": 0.777}, "num_steps": 4},
    ]
    with ClipLibraryWriter(str(path), num_clips=6, meta=_meta(), compression=None) as w:
        w.write_batch(0, _batch(2, -1, "flat"))       # flat clips
        w.write_batch(2, _batch(2, 0, "terrain"))     # terrain 0
        w.write_batch(4, _batch(2, 1, "transition"))  # terrain 1
        w.set_attr("terrain_manifest", manifest)

    data = load_clip_library(str(path))
    assert data.terrain_id.tolist() == [-1, -1, 0, 0, 1, 1]
    # codes: flat=0, terrain=1, transition=2
    assert data.terrain_type_code.tolist() == [0, 0, 1, 1, 2, 2]
    assert data.attrs["terrain_type_names"] == ["flat", "terrain", "transition"]
    assert data.attrs["terrain_manifest"] == manifest


def test_pre_terrain_library_loads_with_empty_tags(tmp_path):
    """A library written without the terrain datasets still loads."""
    path = tmp_path / "old.h5"
    # Hand-write a minimal library missing the terrain datasets.
    with h5py.File(str(path), "w") as f:
        f.create_dataset("y_des", data=np.zeros((3, T, P), np.float32))
        f.create_dataset("dy_des", data=np.zeros((3, T, V), np.float32))
        f.attrs["num_clips"] = 3

    data = load_clip_library(str(path))
    assert data.num_clips == 3
    assert data.terrain_id.numel() == 0            # empty placeholder
    assert data.terrain_type_code.numel() == 0
