# Copyright 2026 Zachary Olkin. All rights reserved.

"""Subsetting a clip library, and picking the ``fm_debug`` case clips out of one.

Two things have to hold for the FM debugger to be reproducible:

* :func:`~motion_gen.clips.clip_io.write_clip_subset` keeps exactly the requested clips, in
  the requested order, with only their tiles in the manifest and a ``counts.flat`` that
  matches -- the importer sizes its FLAT tile count from that field in preference to the
  clip count, so a copied ``flat: 60`` builds 60 flat tiles for a 3-clip library and the
  terrain reset then aborts;
* the case selection is deterministic, so re-prepping the same library gives the same
  three clips without a seed.

Pure HDF5 + numpy -- no IsaacLab, no simulator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library, write_clip_subset

# fm_debug lives next to the other drivers under scripts/rsl_rl (they are scripts, not an
# installed package), and its prep stage is IsaacLab-free -- so the selector is testable here.
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "rsl_rl"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fm_debug import select_case_clips  # noqa: E402

T, P, V, C = 6, 4, 3, 2


def _meta() -> dict:
    """Minimal layout metadata for the writer."""
    return {
        "dt": 0.02, "clip_length_s": T * 0.02, "num_steps": T,
        "pos_output_names": [f"f:pos_{a}" for a in "xyz"] + ["j:0"],
        "vel_output_names": [f"f:vel_{a}" for a in "xyz"],
        "contact_frames": ["l", "r"], "ref_frame": "r", "skill_list": ["walk"],
    }


def _batch(n: int, terrain_id: int, terrain_type: str, tag: float, clip_len: int) -> dict:
    """A batch whose per-clip contents are identifiable by ``tag``."""
    return {
        "y_des": torch.full((n, T, P), tag),
        "dy_des": torch.full((n, T, V), tag),
        "contact": torch.zeros(n, T, C, dtype=torch.uint8),
        "skill_id": torch.zeros(n, T, dtype=torch.uint8),
        "stride_idx": torch.zeros(n, T, dtype=torch.int32),
        "target_vel": torch.full((n, T, 3), tag),
        "terrain_id": torch.full((n,), terrain_id, dtype=torch.int32),
        "clip_len": torch.full((n,), clip_len, dtype=torch.int32),
        "terrain_type": terrain_type,
    }


@pytest.fixture
def library(tmp_path) -> str:
    """A 6-clip library: 2 flat, 2 on a transition tile, 2 on a box tile."""
    path = tmp_path / "src.h5"
    manifest = [
        {"terrain_id": 0, "type": "transition", "family": "family_stairs_up.h5",
         "size_x": 6.0, "size_y": 2.0, "overlays": []},
        {"terrain_id": 1, "type": "box", "family": "family_jump_on.h5",
         "size_x": 6.0, "size_y": 2.0, "overlays": []},
    ]
    with ClipLibraryWriter(str(path), num_clips=6, meta=_meta(), compression=None) as w:
        w.write_batch(0, _batch(2, -1, "flat", 10.0, clip_len=4))
        w.write_batch(2, _batch(2, 0, "transition", 20.0, clip_len=5))
        w.write_batch(4, _batch(2, 1, "box", 30.0, clip_len=6))
        w.set_attr("terrain_manifest", manifest)
        w.set_attr("generator_config", {"counts": {"flat": 60}})
    return str(path)


def test_subset_keeps_the_requested_clips_in_order(library, tmp_path):
    """Per-clip data follows the requested index list, not the source order."""
    dst = str(tmp_path / "subset.h5")
    write_clip_subset(library, dst, [4, 0, 2])

    lib = load_clip_library(dst)
    assert lib.terrain_id.tolist() == [1, -1, 0]
    assert lib.clip_len.tolist() == [6, 4, 5]
    # y_des was tagged per family, so the ordering is checkable on the data itself.
    assert [float(lib.y_des[i, 0, 0]) for i in range(3)] == [30.0, 10.0, 20.0]


def test_subset_is_byte_identical_to_the_source_rows(library, tmp_path):
    """Every dataset is a straight gather -- no re-derivation, no padding change."""
    dst = str(tmp_path / "subset.h5")
    idx = [5, 1]
    write_clip_subset(library, dst, idx)
    with h5py.File(library, "r") as src, h5py.File(dst, "r") as out:
        assert set(out.keys()) == set(src.keys())
        for key in out.keys():
            want = np.stack([np.asarray(src[key][i]) for i in idx])
            assert np.array_equal(np.asarray(out[key]), want), key


def test_subset_keeps_only_the_tiles_its_clips_stand_on(library, tmp_path):
    """A tile with no clip left would be built by the importer and never reset onto."""
    dst = str(tmp_path / "subset.h5")
    manifest = write_clip_subset(library, dst, [0, 2])       # one flat + one transition
    assert [m["terrain_id"] for m in manifest] == [0]
    with h5py.File(dst, "r") as f:
        assert json.loads(f.attrs["terrain_manifest"]) == manifest
        assert int(f.attrs["num_clips"]) == 2
        assert int(f.attrs["num_clips_written"]) == 2


def test_subset_rewrites_the_flat_tile_count(library, tmp_path):
    """``counts.flat`` must describe THIS library, not the one it was cut from.

    The importer prefers that field to the actual flat-clip count
    (``clip_terrain_loader.default_num_flat_tiles``), so carrying the source's 60 over
    builds 60 flat tiles for a 3-clip library and the terrain reset aborts with
    "terrain id(s) [-1] have an env tile but NO clips".
    """
    for indices, expected in (([0, 2, 4], 1), ([2, 4], 0), ([0, 1], 2)):
        dst = str(tmp_path / f"subset_{len(indices)}_{expected}.h5")
        write_clip_subset(library, dst, indices)
        with h5py.File(dst, "r") as f:
            gcfg = json.loads(f.attrs["generator_config"])
        assert gcfg["counts"]["flat"] == expected, indices
        assert gcfg["subset_of"]["clip_indices"] == indices


@pytest.mark.parametrize("indices", [[], [1, 1]])
def test_subset_rejects_an_empty_or_duplicated_selection(library, tmp_path, indices):
    """Both would produce a library whose clip count lies about its contents."""
    with pytest.raises(ValueError):
        write_clip_subset(library, str(tmp_path / "bad.h5"), indices)


# ---------------------------------------------------------------------------
# Case selection (scripts/rsl_rl/fm_debug.py)
# ---------------------------------------------------------------------------

def _select(library, cases, overrides=None):
    """Call the debugger's case selector."""
    return select_case_clips(library, cases, overrides or {})


def test_case_selection_picks_the_longest_match_per_family(library):
    """Longest first, so the rollout has as much of each terrain family as the library has."""
    cases = [
        {"name": "stairs", "type": "transition", "family": "family_stairs_up.h5"},
        {"name": "flat", "type": "flat", "family": None},
        {"name": "box", "type": "box", "family": "family_jump_on.h5"},
    ]
    chosen = _select(library, cases)
    assert [c["name"] for c in chosen] == ["stairs", "flat", "box"]
    assert [c["terrain_id"] for c in chosen] == [0, -1, 1]
    assert [c["clip_len"] for c in chosen] == [5, 4, 6]
    assert [c["type"] for c in chosen] == ["transition", "flat", "box"]
    # Deterministic: no seed, same answer every call.
    assert _select(library, cases) == chosen


def test_case_selection_honours_an_explicit_index(library):
    """``--clip_index name=i`` wins over the automatic pick."""
    cases = [{"name": "box", "type": "box", "family": "family_jump_on.h5"}]
    assert _select(library, cases)[0]["clip_index"] == 4       # first of the tied pair
    assert _select(library, cases, {"box": 5})[0]["clip_index"] == 5


def test_case_selection_fails_loudly_on_an_absent_family(library):
    """Silently returning two cases instead of three would skew every later comparison."""
    cases = [{"name": "downstairs", "type": "transition", "family": "family_stairs_down.h5"}]
    with pytest.raises(SystemExit, match="matches no clip"):
        _select(library, cases)
