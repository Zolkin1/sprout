# Copyright 2026 Zachary Olkin. All rights reserved.

"""Merging rollout stems must renumber ``episode_uid``, and must refuse a contract mismatch.

The renumbering is the whole reason a merge is a script rather than a concatenation.
``episode_uid`` is assigned per RUN starting at 0, so every file contains an episode 0.
``fit_value`` splits BY EPISODE precisely so that near-duplicate windows from one episode cannot
straddle the split -- and a collision defeats exactly that: two unrelated episodes become one
group, their windows are dealt to both sides, and the held-out metrics measure memorisation
while looking entirely clean.  That is the leak this project already paid for once on the FM's
own episode split, where 86.5% of held-out clips also appeared in training, so it is worth a
test that fails loudly rather than a comment.

The files here are built by hand rather than recorded: the merger reads through
``rollout_outcomes``' own readers, so a fixture only has to satisfy those -- the schema version,
the window count attribute, and a window/episode join on ``episode_uid``.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts" / "rsl_rl"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

ro = pytest.importorskip("rollout_outcomes")


def _load_merger():
    """Import ``merge_rollouts`` by path -- ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "merge_rollouts", str(_SCRIPTS / "merge_rollouts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: A window contract two files must share to be mergeable.
_CONTRACT = {"num_history": 4, "state_dim": 44, "future_len": 64, "cmd_dim": 3,
             "extero_shape": [2, 3900], "proprio_hw": [10, 40], "scan_dim": 176}


def _write_stem(stem: str, n_eps: int, per_ep: int, contract: dict | None = None,
                first_uid: int = 0) -> None:
    """Write a minimal but READABLE rollout pair at ``stem``.

    Args:
        stem: Output stem (no suffix).
        n_eps: Episodes in the outcome table.
        per_ep: Windows per episode.
        contract: Window attrs; defaults to :data:`_CONTRACT`.
        first_uid: The uid the file starts numbering at -- 0 in every real recording, which is
            the collision this merge exists to resolve.
    """
    import json

    uids = np.arange(first_uid, first_uid + n_eps, dtype=np.int64)
    win_uid = np.repeat(uids, per_ep)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    with h5py.File(stem + ".h5", "w") as f:
        f.create_dataset("episode_uid", data=win_uid)
        f.create_dataset("payload", data=np.arange(win_uid.size, dtype=np.float32))
        f.attrs["schema_version"] = ro.SCHEMA_VERSION
        f.attrs["num_windows_written"] = win_uid.size
        for k, v in (contract or _CONTRACT).items():
            f.attrs[k] = json.dumps(v)
    with h5py.File(stem + ".episodes.h5", "w") as f:
        f.create_dataset("episode_uid", data=uids)
        f.create_dataset("reached_goal", data=np.zeros(n_eps, dtype=np.int64))
        f.attrs["schema_version"] = ro.SCHEMA_VERSION


def test_colliding_uids_are_renumbered_so_no_two_episodes_share_a_group(tmp_path):
    """Two files that both number from 0 come out with disjoint uids."""
    merge = _load_merger()
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _write_stem(a, n_eps=3, per_ep=2)
    _write_stem(b, n_eps=4, per_ep=2)
    out = str(tmp_path / "merged")
    merge.merge([a, b], out)

    E, _ = ro.read_rollout_episodes(out + ".episodes.h5")
    W, _ = ro.read_rollout_windows(out + ".h5")
    assert len(E["episode_uid"]) == 7
    assert len(np.unique(E["episode_uid"])) == 7, "a collision survived the merge"
    assert sorted(E["episode_uid"].tolist()) == list(range(7))
    # Every window still joins to an outcome row, and to the RIGHT one: the payload was
    # numbered per file, so a window's payload must stay with its own episode.
    assert set(np.unique(W["episode_uid"])) <= set(E["episode_uid"].tolist())
    assert len(W["episode_uid"]) == (3 + 4) * 2


def test_the_second_files_windows_move_with_its_episodes(tmp_path):
    """Renumbering must apply to BOTH tables, or the join silently loses a file."""
    merge = _load_merger()
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _write_stem(a, n_eps=2, per_ep=3)
    _write_stem(b, n_eps=2, per_ep=3)
    out = str(tmp_path / "merged")
    merge.merge([a, b], out)
    W, _ = ro.read_rollout_windows(out + ".h5")
    E, _ = ro.read_rollout_episodes(out + ".episodes.h5")
    # 4 episodes x 3 windows, and each uid carries exactly its own three.
    counts = {int(u): int((W["episode_uid"] == u).sum()) for u in E["episode_uid"]}
    assert counts == {0: 3, 1: 3, 2: 3, 3: 3}, counts


def test_a_differing_contract_is_refused_by_name(tmp_path):
    """A 1-camera row must not be concatenated onto a 2-camera one."""
    merge = _load_merger()
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _write_stem(a, n_eps=2, per_ep=1)
    _write_stem(b, n_eps=2, per_ep=1, contract=dict(_CONTRACT, extero_shape=[1, 3900]))
    with pytest.raises(SystemExit, match="extero_shape"):
        merge.merge([a, b], str(tmp_path / "merged"))


def test_a_contract_field_present_on_only_one_side_is_refused(tmp_path):
    """'Missing' and 'different' are both reasons not to merge."""
    merge = _load_merger()
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _write_stem(a, n_eps=2, per_ep=1)
    thin = {k: v for k, v in _CONTRACT.items() if k != "scan_dim"}
    _write_stem(b, n_eps=2, per_ep=1, contract=thin)
    with pytest.raises(SystemExit, match="scan_dim"):
        merge.merge([a, b], str(tmp_path / "merged"))


def test_an_orphan_window_is_refused_rather_than_dropped(tmp_path):
    """A window whose episode has no outcome row would be silently lost in the join."""
    merge = _load_merger()
    a = str(tmp_path / "a")
    _write_stem(a, n_eps=2, per_ep=2)
    # Add a window for an episode the outcome table does not carry.
    with h5py.File(a + ".h5", "a") as f:
        uid = f["episode_uid"][...]
        del f["episode_uid"]
        f.create_dataset("episode_uid", data=np.append(uid, 99))
        pay = f["payload"][...]
        del f["payload"]
        f.create_dataset("payload", data=np.append(pay, 0.0).astype(np.float32))
        f.attrs["num_windows_written"] = uid.size + 1
    with pytest.raises(SystemExit, match="no outcome row"):
        merge.merge([a], str(tmp_path / "merged"))


def test_a_missing_stem_names_itself(tmp_path):
    """A typo'd stem must not read as an empty merge."""
    merge = _load_merger()
    with pytest.raises(SystemExit, match="not found"):
        merge.merge([str(tmp_path / "nope")], str(tmp_path / "merged"))
