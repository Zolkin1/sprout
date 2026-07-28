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
from motion_gen.world_lift import (
    _ANCHOR_MASK_BOX,
    _ANCHOR_MASK_FLAT,
    _ANCHOR_MASK_LEGACY,
    _ANCHOR_MASK_STAIRS,
    anchor_mask_free_names,
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
    assert data.attrs["terrain_type_names"] == ["flat", "terrain", "transition", "slope", "box"]
    assert data.attrs["terrain_manifest"] == manifest
    # Batches omitted anchor_mask -> every clip gets the legacy default.
    assert data.anchor_mask.tolist() == [_ANCHOR_MASK_LEGACY] * 6


def test_anchor_mask_roundtrip(tmp_path):
    """An explicit per-clip anchor_mask survives write/read; omitting it => legacy default."""
    path = tmp_path / "masked.h5"
    with ClipLibraryWriter(str(path), num_clips=4, meta=_meta(), compression=None) as w:
        flat = _batch(2, -1, "flat")
        flat["anchor_mask"] = torch.full((2,), _ANCHOR_MASK_FLAT, dtype=torch.uint8)
        w.write_batch(0, flat)
        stairs = _batch(2, 0, "terrain")
        stairs["anchor_mask"] = torch.full((2,), _ANCHOR_MASK_STAIRS, dtype=torch.uint8)
        w.write_batch(2, stairs)

    data = load_clip_library(str(path))
    assert data.anchor_mask.tolist() == [_ANCHOR_MASK_FLAT, _ANCHOR_MASK_FLAT,
                                         _ANCHOR_MASK_STAIRS, _ANCHOR_MASK_STAIRS]
    # anchor_mask_or_legacy is a pass-through when the library carries the dataset.
    assert data.anchor_mask_or_legacy().tolist() == data.anchor_mask.tolist()


def test_box_type_code_and_anchor_mask_roundtrip(tmp_path):
    """The ``box`` terrain type + box mount anchor mask survive round-trip."""
    # Box clips use per-frame masks; the mount (jump-on) phase is LATERAL-free (the landing follows
    # the robot) with downrange/height/yaw pinned to the terrain so the jump lands on the box --
    # same DOF pattern as stairs.
    assert anchor_mask_free_names(_ANCHOR_MASK_BOX) == ["lateral"]
    assert _ANCHOR_MASK_BOX == _ANCHOR_MASK_STAIRS
    path = tmp_path / "box.h5"
    manifest = [{
        "terrain_id": 0, "type": "box", "size_x": 6.0, "size_y": 2.0,
        "overlays": [{"kind": "box", "pos_xy": [3.0, 1.0],
                      "size_xyz": [0.6, 0.8, 0.4], "base_z": 0.0, "walkable_top": True}],
    }]
    with ClipLibraryWriter(str(path), num_clips=2, meta=_meta(), compression=None) as w:
        b = _batch(2, 0, "box")
        b["anchor_mask"] = torch.full((2,), _ANCHOR_MASK_BOX, dtype=torch.uint8)
        w.write_batch(0, b)
        w.set_attr("terrain_manifest", manifest)

    data = load_clip_library(str(path))
    # box is code 4 (index in _TERRAIN_TYPE_NAMES after flat/terrain/transition/slope).
    assert data.terrain_type_code.tolist() == [4, 4]
    assert data.anchor_mask.tolist() == [_ANCHOR_MASK_BOX, _ANCHOR_MASK_BOX]
    assert data.attrs["terrain_manifest"][0]["overlays"][0]["kind"] == "box"


def test_per_step_anchor_mask_roundtrip(tmp_path):
    """Per-step anchor_mask_t round-trips; a MIXED library falls back per-clip via the sentinel."""
    path = tmp_path / "perstep.h5"
    with ClipLibraryWriter(str(path), num_clips=3, meta=_meta(), compression=None) as w:
        # box clip: first half jump-on (BOX y-free), second half on-top-walk (FLAT ego)
        box = _batch(2, 0, "box")
        box["anchor_mask"] = torch.full((2,), _ANCHOR_MASK_BOX, dtype=torch.uint8)
        amt = torch.empty(2, T, dtype=torch.uint8)
        amt[:, : T // 2] = _ANCHOR_MASK_BOX
        amt[:, T // 2:] = _ANCHOR_MASK_FLAT
        box["anchor_mask_t"] = amt
        w.write_batch(0, box)
        # flat clip WITHOUT a per-step mask -> sentinel -> broadcast its per-clip mask
        flat = _batch(1, -1, "flat")
        flat["anchor_mask"] = torch.full((1,), _ANCHOR_MASK_FLAT, dtype=torch.uint8)
        w.write_batch(2, flat)

    data = load_clip_library(str(path))
    ps = data.anchor_mask_per_step()
    assert ps.shape == (3, T)
    assert (ps[0, : T // 2] == _ANCHOR_MASK_BOX).all()     # per-step pattern preserved
    assert (ps[0, T // 2:] == _ANCHOR_MASK_FLAT).all()
    assert (ps[1] == ps[0]).all()
    assert (ps[2] == _ANCHOR_MASK_FLAT).all()              # sentinel -> per-clip FLAT everywhere


def test_no_per_step_mask_broadcasts_per_clip(tmp_path):
    """A library with NO anchor_mask_t dataset broadcasts the per-clip mask across steps."""
    path = tmp_path / "noperstep.h5"
    with ClipLibraryWriter(str(path), num_clips=2, meta=_meta(), compression=None) as w:
        b = _batch(2, 0, "terrain")
        b["anchor_mask"] = torch.full((2,), _ANCHOR_MASK_STAIRS, dtype=torch.uint8)
        w.write_batch(0, b)
    data = load_clip_library(str(path))
    assert data.anchor_mask_t.numel() == 0
    ps = data.anchor_mask_per_step()
    assert ps.shape == (2, T) and (ps == _ANCHOR_MASK_STAIRS).all()


def test_pre_terrain_library_loads_with_empty_tags(tmp_path):
    """A library written without the terrain/anchor datasets still loads."""
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
    assert data.anchor_mask.numel() == 0           # pre-mask library
    # ... but the interpretation helper fills the legacy mask for all clips.
    assert data.anchor_mask_or_legacy().tolist() == [_ANCHOR_MASK_LEGACY] * 3
