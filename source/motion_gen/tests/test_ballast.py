# Copyright 2026 Zachary Olkin. All rights reserved.
"""BALLAST: a fixed slice of the pre-training set, carried at weight 1 and never graded.

AWR reweights the generator toward one course; ballast is what stops that quietly costing the
distribution it was pre-trained on. Everything that can go wrong with it is silent:

* a short draw makes the declared ``ballast_flat_frac`` a fiction while every log line reads
  normally, so a stratum that cannot fill its share must be an ERROR;
* ballast leaking into the critic or the advantage would make ungraded rows look graded, which
  is why it stays in its own FILE rather than being merged into the buffer -- the merge would
  need a reserved uid range that nothing checks;
* ballast in the validation split would change which checkpoint ``best.pt`` selects, for a
  reason unrelated to the course being fitted.
"""

from __future__ import annotations

import json
import pathlib
import sys

import h5py
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (_ROOT / "scripts" / "flow_matching",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bal = pytest.importorskip("ballast")

_FAMILIES = ["flat", "family_stairs_up.h5", "family_jump_on.h5"]
_PER_FAMILY = 40


def _recording(path: pathlib.Path, families=None, n_per=_PER_FAMILY) -> pathlib.Path:
    """A window file carrying only the columns the ballast selection reads."""
    families = families or _FAMILIES
    code = np.repeat(np.arange(len(families), dtype=np.int64), n_per)
    with h5py.File(path, "w") as h:
        h.create_dataset("family_code", data=code)
        h.attrs["num_windows_written"] = code.size
        h.attrs["family_names"] = json.dumps(families)
    return path


@pytest.fixture()
def rec(tmp_path):
    """A 120-window recording: 40 flat, 80 terrain."""
    return _recording(tmp_path / "pre.h5")


# ---------------------------------------------------------------------------------------------
# The declared mixture
# ---------------------------------------------------------------------------------------------
def test_zero_chunks_disables_it(rec) -> None:
    assert bal.select_ballast_rows(str(rec), 0).size == 0


def test_the_flat_share_is_exact(rec) -> None:
    """The number is DECLARED, so it has to be honoured rather than approached."""
    rows = bal.select_ballast_rows(str(rec), 40, flat_frac=0.5, seed=0)
    flat = bal.flat_mask(str(rec))
    assert rows.size == 40
    assert int(flat[rows].sum()) == 20


@pytest.mark.parametrize("frac,want_flat", [(0.0, 0), (0.25, 10), (0.5, 20), (1.0, 40)])
def test_the_share_holds_across_the_range(rec, frac, want_flat) -> None:
    rows = bal.select_ballast_rows(str(rec), 40, flat_frac=frac, seed=1)
    got = int(bal.flat_mask(str(rec))[rows].sum())
    # frac 0 draws uniformly and never reads the family table, so its flat count is incidental.
    if frac > 0.0:
        assert got == want_flat


def test_a_stratum_that_cannot_fill_its_share_is_REFUSED(rec) -> None:
    """Returning fewer would make the declared mixture a fiction, silently."""
    with pytest.raises(SystemExit, match="holds only"):
        bal.select_ballast_rows(str(rec), 100, flat_frac=0.9, seed=0)   # needs 90 flat, has 40


def test_more_chunks_than_the_recording_holds_is_refused(rec) -> None:
    with pytest.raises(SystemExit, match="holds"):
        bal.select_ballast_rows(str(rec), 10_000)


def test_a_recording_with_no_family_table_is_refused_when_a_share_is_declared(tmp_path) -> None:
    """Otherwise the flat share would be unverifiable and silently unmet."""
    p = tmp_path / "nofam.h5"
    with h5py.File(p, "w") as h:
        h.attrs["num_windows_written"] = 50
    with pytest.raises(SystemExit, match="no family table"):
        bal.select_ballast_rows(str(p), 10, flat_frac=0.5)


def test_no_family_table_is_FINE_when_no_share_is_declared(tmp_path) -> None:
    """frac 0 never reads the table, so a family-less recording is still usable."""
    p = tmp_path / "nofam.h5"
    with h5py.File(p, "w") as h:
        h.attrs["num_windows_written"] = 50
    assert bal.select_ballast_rows(str(p), 10, flat_frac=0.0).size == 10


def test_the_draw_is_reproducible_and_the_seed_moves_it(rec) -> None:
    a = bal.select_ballast_rows(str(rec), 30, 0.5, seed=0)
    assert np.array_equal(a, bal.select_ballast_rows(str(rec), 30, 0.5, seed=0))
    assert not np.array_equal(a, bal.select_ballast_rows(str(rec), 30, 0.5, seed=5))


def test_rows_are_unique_and_sorted(rec) -> None:
    """Duplicated rows would train on one window twice and quietly double its weight."""
    rows = bal.select_ballast_rows(str(rec), 60, 0.5, seed=3)
    assert rows.size == np.unique(rows).size
    assert np.array_equal(rows, np.sort(rows))


def test_flat_is_matched_by_NAME_not_by_substring(tmp_path) -> None:
    """A substring test would silently swallow any future flat_* family."""
    p = _recording(tmp_path / "s.h5", families=["flat_room_decor", "flat"], n_per=10)
    flat = bal.flat_mask(str(p))
    assert int(flat[:10].sum()) == 0, "'flat_room_decor' is not the flat family"
    assert int(flat[10:].sum()) == 10


def test_the_h5_suffix_is_stripped_before_matching(tmp_path) -> None:
    """record_rollouts normalises family names that way, so this must agree."""
    p = _recording(tmp_path / "s.h5", families=["family_flat.h5", "family_stairs_up.h5"],
                   n_per=10)
    assert int(bal.flat_mask(str(p))[:10].sum()) == 10


def test_describe_says_it_is_ungraded_and_weight_one(rec) -> None:
    """The banner is the only place a reader learns the run carried ballast at all."""
    rows = bal.select_ballast_rows(str(rec), 20, 0.5, seed=0)
    text = bal.describe(str(rec), rows)
    assert "weight 1" in text and "ungraded" in text and "10 flat" in text


# ---------------------------------------------------------------------------------------------
# The config contract
# ---------------------------------------------------------------------------------------------
def _write_cfg(tmp_path, **over):
    """A minimal valid config with overrides."""
    from finetune_config import config_keys

    base = {k: 0 for k in config_keys()}
    base.update({
        "name": "t", "task": "T", "clip": "c.h5", "goals": "g.json",
        "buffer": 10, "refresh": 5, "iters": 1, "num_envs": 8,
        "ballast_chunks": 0, "ballast_flat_frac": 0.0, "ballast_rollout": "",
        "updown_cells": [], "updown_measured_cells": [], "updown_measured_frac": 0.0,
        "updown_steps_range": [1, 2], "updown_width_range_m": [1.0, 2.0],
        "updown_platform_range_m": [1.0, 2.0], "updown_wall_height_range_m": [1.0, 2.0],
        "boxes_per_tile_range": [1, 2], "box_height_range_m": [0.2, 0.5],
        "box_depth_range_m": [0.4, 1.0], "box_width_range_m": [1.0, 2.0],
        "box_gap_range_m": [1.0, 2.0], "commanded_speed_range_mps": [0.6, 2.5],
        "tiles_boxes": 1, "tiles_updown": 0, "tiles_indist": 0, "tiles_entry_stairs": 0,
        "seed": 0,
    })
    base.update(over)
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(base))
    return str(p)


def test_a_ballast_count_with_no_source_is_refused(tmp_path) -> None:
    """Which recording is part of what the run WAS, and it is invisible everywhere else."""
    from finetune_config import load_finetune_config

    with pytest.raises(SystemExit, match="ballast_rollout is empty"):
        load_finetune_config(_write_cfg(tmp_path, ballast_chunks=100))


def test_a_source_with_no_count_is_refused(tmp_path) -> None:
    from finetune_config import load_finetune_config

    with pytest.raises(SystemExit, match="ballast_chunks is 0"):
        load_finetune_config(_write_cfg(tmp_path, ballast_rollout="x.h5"))


def test_a_flat_share_with_no_ballast_is_refused(tmp_path) -> None:
    """It declares a mixture of nothing."""
    from finetune_config import load_finetune_config

    with pytest.raises(SystemExit, match="mixture of nothing"):
        load_finetune_config(_write_cfg(tmp_path, ballast_flat_frac=0.5))


def test_the_loop_receives_the_source_path(tmp_path) -> None:
    """as_env is the whole bridge to the shell; a key missing here is a silent no-ballast run."""
    from finetune_config import as_env, load_finetune_config

    cfg = load_finetune_config(_write_cfg(tmp_path, ballast_chunks=7,
                                          ballast_flat_frac=0.5,
                                          ballast_rollout="pre.h5"))
    env = as_env(cfg)
    assert env["BALLAST_ROLLOUT"] == "pre.h5"
    assert env["BALLAST_CHUNKS"] == "7"
    assert env["BALLAST_FLAT_FRAC"] == "0.5"


# ---------------------------------------------------------------------------------------------
# ConcatWindowDataset -- how the ballast reaches the trainer
# ---------------------------------------------------------------------------------------------
class _Part:
    """A stand-in for RolloutWindowDataset with only what the concat wrapper touches."""

    def __init__(self, n: int, tag: str, extero=(2, 3), cmd=4, labels=None) -> None:
        """Hold a length and a tag so items are identifiable."""
        self.n, self.tag = n, tag
        self.extero_shape, self.cmd_dim = extero, cmd
        self.path = f"{tag}.h5"
        self._labels = labels or {}

    def __len__(self) -> int:
        """Windows."""
        return self.n

    def __getitem__(self, i: int):
        """A tagged item."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        return (self.tag, i)

    def label_counts(self) -> dict:
        """Label histogram."""
        return dict(self._labels)


def _concat(parts):
    """Build the wrapper."""
    from robot_rl.flow_matching.rollout_dataset import ConcatWindowDataset

    return ConcatWindowDataset(parts)


def test_the_concat_length_is_the_sum() -> None:
    assert len(_concat([_Part(7, "buf"), _Part(3, "bal")])) == 10


def test_every_index_maps_to_the_right_part_and_offset() -> None:
    """An off-by-one here trains on the wrong window and nothing downstream could tell."""
    ds = _concat([_Part(3, "buf"), _Part(2, "bal")])
    assert [ds[i] for i in range(5)] == [
        ("buf", 0), ("buf", 1), ("buf", 2), ("bal", 0), ("bal", 1)]


def test_a_single_part_is_a_passthrough() -> None:
    ds = _concat([_Part(4, "buf")])
    assert len(ds) == 4 and [ds[i] for i in range(4)] == [("buf", i) for i in range(4)]


def test_negative_indices_work() -> None:
    ds = _concat([_Part(3, "buf"), _Part(2, "bal")])
    assert ds[-1] == ("bal", 1) and ds[-5] == ("buf", 0)


def test_an_empty_part_is_skipped_cleanly() -> None:
    """A ballast slice of 0 rows must not shift the buffer's indices."""
    ds = _concat([_Part(2, "buf"), _Part(0, "bal"), _Part(1, "tail")])
    assert [ds[i] for i in range(3)] == [("buf", 0), ("buf", 1), ("tail", 0)]


def test_label_counts_are_summed_across_parts() -> None:
    ds = _concat([_Part(1, "a", labels={"positive": 2, "null": 1}),
                  _Part(1, "b", labels={"null": 4})])
    assert ds.label_counts() == {"positive": 2, "null": 5}


def test_parts_from_different_recordings_are_refused() -> None:
    """Two window files from different checkpoints differ in WIDTH, and the mismatch would
    otherwise surface as a shape error inside a DataLoader worker."""
    with pytest.raises(ValueError, match="two different recordings"):
        _concat([_Part(2, "buf", extero=(2, 3)), _Part(2, "bal", extero=(5, 9))])


def test_no_parts_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        _concat([])
