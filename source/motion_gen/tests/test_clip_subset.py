# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the play-sized ``_small`` clip-library subset (:mod:`motion_gen.clips.subset`).

Covers the three properties the feature rests on:

* the subset is LOADABLE -- ``terrain_id`` remapped to ``0..K-1`` with a manifest that matches,
  because the terrain importer builds its tile rows by walking the manifest;
* the tile pick SPANS FAMILIES -- generators lay families out in contiguous id blocks, so a
  prefix would give a subset that is entirely one family;
* a STALE small is detected -- play defaults to the small, so a small cut from a
  since-regenerated parent would silently put the robot on terrain that no longer exists.
"""

from __future__ import annotations

import json
import os
import time

import h5py

import pytest
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.clips.subset import (
    check_small_is_current,
    is_small_path,
    resolve_play_clip,
    select_clips,
    small_path_for,
    write_small_library,
)

_NT, _T, _P, _V = 12, 6, 4, 3          # tiles, steps, pos-, vel-channels
_CLIPS_PER_TILE = 4
_FLAT = 5


def _make_lib(path: str) -> str:
    """Write a synthetic terrain library: 12 tiles x 4 clips + 5 flat, families in id blocks."""
    n = _NT * _CLIPS_PER_TILE + _FLAT
    meta = {
        "dt": 0.02, "clip_length_s": _T * 0.02, "num_steps": _T,
        "pos_output_names": [f"p{i}" for i in range(_P)],
        "vel_output_names": [f"v{i}" for i in range(_V)],
        "contact_frames": ["l", "r"], "ref_frame": "pelvis_link", "skill_list": ["s"],
    }
    w = ClipLibraryWriter(path, n, meta)
    manifest = []
    row = 0
    for t in range(_NT):
        # 3 contiguous id blocks -> a prefix pick would return only "famA".
        fam = ("famA", "famB", "famC")[t // (_NT // 3)]
        manifest.append({"terrain_id": t, "type": "box", "family": f"family_{fam}.h5",
                         "size_x": 4.0, "size_y": 2.0, "overlays": []})
        for _ in range(_CLIPS_PER_TILE):
            w.write_batch(row, _batch(t, row)); row += 1
    for _ in range(_FLAT):
        w.write_batch(row, _batch(-1, row)); row += 1
    w.set_attr("generator", "synthetic")
    w.set_attr("terrain_manifest", manifest)
    w.close()
    return path


def _batch(tid: int, row: int) -> dict:
    """One clip's batch dict, values keyed to ``row`` so we can trace which rows survived."""
    return {
        "y_des": torch.full((1, _T, _P), float(row)),
        "dy_des": torch.full((1, _T, _V), float(row)),
        "contact": torch.ones(1, _T, 2, dtype=torch.uint8),
        "skill_id": torch.zeros(1, _T, dtype=torch.uint8),
        "stride_idx": torch.full((1, _T), -1, dtype=torch.int32),
        "target_vel": torch.zeros(1, _T, 3),
        "terrain_id": torch.tensor([tid], dtype=torch.int32),
        "terrain_type": "flat" if tid < 0 else "box",
        "anchor_mask": torch.tensor([35 if tid < 0 else 2], dtype=torch.uint8),
        "clip_len": torch.tensor([_T], dtype=torch.int32),
    }


@pytest.fixture()
def lib(tmp_path) -> str:
    """A synthetic full library on disk."""
    return _make_lib(str(tmp_path / "syn.h5"))


def test_path_helpers() -> None:
    """``foo.h5`` <-> ``foo_small.h5``, and a small is recognised as one."""
    assert small_path_for("a/b/foo.h5") == "a/b/foo_small.h5"
    assert is_small_path("a/b/foo_small.h5") and not is_small_path("a/b/foo.h5")


def test_tile_pick_spans_families(lib: str) -> None:
    """The pick must cover all three family blocks, not just the first."""
    with h5py.File(lib) as f:
        tid = f["terrain_id"][:]
        man = {int(e["terrain_id"]): e for e in json.loads(f.attrs["terrain_manifest"])}
    _, tiles = select_clips(tid, num_tiles=6, clips_per_tile=1, flat_clips=1)
    fams = {man[t]["family"] for t in tiles}
    assert len(fams) == 3, f"pick covered only {fams}"


def test_subset_is_loadable_with_remapped_ids(lib: str) -> None:
    """terrain_ids are 0..K-1 and the manifest matches -- what the terrain importer needs."""
    info = write_small_library(lib, num_tiles=5, clips_per_tile=2, flat_clips=3)
    assert info["num_clips"] == 5 * 2 + 3
    small = load_clip_library(info["out"], device="cpu")
    tids = sorted({int(t) for t in small.terrain_id.tolist() if int(t) != -1})
    assert tids == list(range(5)), tids
    man_ids = sorted(int(e["terrain_id"]) for e in small.attrs["terrain_manifest"])
    assert man_ids == tids, f"manifest {man_ids} != clips {tids}"
    # time axis and per-clip length preserved verbatim
    assert small.num_steps == _T and set(small.clip_len.tolist()) == {_T}


def test_subset_keeps_whole_clips(lib: str) -> None:
    """A kept clip's data is copied verbatim (rows carry their source index as the value)."""
    info = write_small_library(lib, num_tiles=3, clips_per_tile=1, flat_clips=1, seed=7)
    small = load_clip_library(info["out"], device="cpu")
    for i in range(small.num_clips):
        vals = torch.unique(small.y_des[i])
        assert vals.numel() == 1, "a clip's frames came from more than one source row"


def test_refuses_to_subset_a_subset(lib: str) -> None:
    """Guard against ``foo_small_small.h5``."""
    info = write_small_library(lib)
    with pytest.raises(ValueError, match="already a _small"):
        write_small_library(info["out"])


def test_staleness_detected_when_parent_changes(lib: str) -> None:
    """A small cut from a since-regenerated parent must not be silently trusted."""
    info = write_small_library(lib, num_tiles=2, clips_per_tile=1, flat_clips=1)
    ok, why = check_small_is_current(info["out"], lib)
    assert ok and why == "", why
    time.sleep(0.01)
    os.utime(lib, ns=(time.time_ns(), time.time_ns()))       # simulate a regenerate
    ok, why = check_small_is_current(info["out"], lib)
    assert not ok and "different" in why, why


def test_unstamped_small_is_tolerated(lib: str, tmp_path) -> None:
    """A hand-made small (no ``derived_from``) is used, with a note -- not rejected."""
    info = write_small_library(lib, num_tiles=2, clips_per_tile=1, flat_clips=1)
    with h5py.File(info["out"], "a") as f:
        del f.attrs["derived_from"]
    ok, why = check_small_is_current(info["out"], lib)
    assert ok and "no derived_from" in why


class TestResolvePlayClip:
    """The play-side substitution decision."""

    def test_prefers_a_current_small(self, lib: str) -> None:
        """The whole point: play loads the small."""
        info = write_small_library(lib, num_tiles=2, clips_per_tile=1, flat_clips=1)
        got, msgs = resolve_play_clip(lib)
        assert got == info["out"] and any("play default" in m for m in msgs)

    def test_full_clip_override(self, lib: str) -> None:
        """``--full_clip`` loads exactly what was asked for."""
        write_small_library(lib, num_tiles=2, clips_per_tile=1, flat_clips=1)
        got, msgs = resolve_play_clip(lib, prefer_small=False)
        assert got == lib and any("full_clip" in m for m in msgs)

    def test_missing_small_warns_and_falls_back(self, lib: str) -> None:
        """No sibling -> load the full library, but say so and name the fix."""
        got, msgs = resolve_play_clip(lib)
        assert got == lib
        assert any("make_clip_subset" in m for m in msgs) and any(m.startswith("[WARN]") for m in msgs)

    def test_stale_small_falls_back_to_full(self, lib: str) -> None:
        """A stale small must NOT be loaded -- wrong terrain is worse than slow."""
        write_small_library(lib, num_tiles=2, clips_per_tile=1, flat_clips=1)
        time.sleep(0.01)
        os.utime(lib, ns=(time.time_ns(), time.time_ns()))
        got, msgs = resolve_play_clip(lib)
        assert got == lib, "a STALE small was loaded"
        assert any("STALE" in m for m in msgs)

    def test_small_passed_directly_is_left_alone(self, lib: str) -> None:
        """Asking for the small explicitly is honoured without comment."""
        info = write_small_library(lib, num_tiles=2, clips_per_tile=1, flat_clips=1)
        got, msgs = resolve_play_clip(info["out"])
        assert got == info["out"] and msgs == []
