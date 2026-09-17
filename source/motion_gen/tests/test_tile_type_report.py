# Copyright 2026 Zachary Olkin. All rights reserved.
"""The per-tile-type reduction of multi-terrain rollouts.

The reduction itself is arithmetic; what this file is aimed at is the ATTRIBUTION. Every
failure mode here produces a full, plausible table in which the rates belong to the wrong
terrain:

* the family map taken from a different library than the rollouts ran on -- every id resolves,
  nothing is empty, and each number is filed under the wrong type;
* a pooled rate quoted across libraries, which moves with the tile MIX and not the policy;
* a binomial interval over pooled episodes, which is too narrow because episodes on one tile
  share its geometry.

A synthetic rollout is written rather than a recorded one reduced, so the expected answer is
known exactly and the test does not need a GPU or a multi-hour recording.
"""

from __future__ import annotations

import json
import pathlib
import sys

import h5py
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (_ROOT / "scripts" / "rsl_rl", _ROOT / "scripts" / "rsl_rl" / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ttr = pytest.importorskip("tile_type_report")

#: Two families, four tiles each, with DIFFERENT success rates so a mislabelled map shows up.
_FAMS = {0: "varied_boxes", 1: "varied_boxes", 2: "varied_boxes", 3: "varied_boxes",
         4: "entry_stairs_up", 5: "entry_stairs_up", 6: "entry_stairs_up",
         7: "entry_stairs_up"}
#: Per-tile arrival rate: boxes succeed often, the entry stairs rarely.
_RATE = {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2}
_EPS_PER_TILE = 25


def _write_library(path: pathlib.Path) -> None:
    """A manifest-only library carrying just the family map this test needs."""
    manifest = [{"terrain_id": t, "type": "terrain", "family": f,
                 "size_x": 10.0, "size_y": 6.0, "has_floor": False, "frame_yaw": 0.0,
                 "overlays": [], "route": {}} for t, f in _FAMS.items()]
    with h5py.File(path, "w") as h:
        h.attrs["terrain_manifest"] = json.dumps(manifest)


def _write_rollout(run_dir: pathlib.Path, lib: pathlib.Path, seed: int = 0) -> None:
    """One synthetic episode table with a known per-tile arrival rate."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tid, ok = [], []
    for t in sorted(_FAMS):
        tid += [t] * _EPS_PER_TILE
        ok += list(rng.random(_EPS_PER_TILE) < _RATE[t])
    tid = np.array(tid, dtype=np.int64)
    ok = np.array(ok, dtype=bool)
    n = tid.size
    cols = {
        "terrain_id": tid,
        "episode_uid": np.arange(n, dtype=np.int64),
        # One episode per env, so the first-per-env correction keeps everything and the
        # expected rate is exactly the drawn one.
        "env_id": np.arange(n, dtype=np.int64),
        "censored": np.zeros(n, dtype=bool),
        "reached_goal": ok,
        "first_goal_step": np.where(ok, 100, -1).astype(np.int64),
        "first_fall_step": np.where(ok, -1, 50).astype(np.int64),
        "first_violation_step": np.full(n, -1, dtype=np.int64),
        "truncated": ~ok,
        "steps_survived": np.where(ok, 200, 50).astype(np.int64),
        "track_reward_mean": np.full(n, 0.5),
    }
    with h5py.File(run_dir / "rollout_x.episodes.h5", "w") as h:
        for k, v in cols.items():
            h.create_dataset(k, data=v)
        # JSON-ENCODED, exactly as record_rollouts writes its whole provenance block. Writing
        # it raw is what let the resolver's quote bug live: the attr read back as
        # '"clips/x.h5"', every path built from it missed, and the default resolution -- the
        # only thing that checks the map against the recording -- was dead code.
        h.attrs["clip_library_path"] = json.dumps(str(lib))


@pytest.fixture()
def rollouts(tmp_path):
    """A library plus one run of rollouts on it."""
    lib = tmp_path / "lib.h5"
    _write_library(lib)
    _write_rollout(tmp_path / "root" / "run_a", lib)
    return tmp_path / "root", lib


def _recs(root, lib):
    """Reduce."""
    from box2_ic_report import load_cells

    return ttr.by_family(load_cells(str(root), rounds=1), ttr.family_map(str(lib)))


# ---------------------------------------------------------------------------------------------
def test_each_family_gets_its_own_rate(rollouts) -> None:
    """The point of the report: two families whose rates differ by 0.6 must not be pooled."""
    root, lib = rollouts
    recs = _recs(root, lib)
    got = {f: r["success"]["value"] for (_run, f), r in recs.items()}
    assert set(got) == {"varied_boxes", "entry_stairs_up"}
    assert got["varied_boxes"] == pytest.approx(0.8, abs=0.12)
    assert got["entry_stairs_up"] == pytest.approx(0.2, abs=0.12)


def test_the_tile_and_episode_counts_are_right(rollouts) -> None:
    root, lib = rollouts
    for (_run, _f), r in _recs(root, lib).items():
        assert r["n_tiles"] == 4
        assert r["n_episodes"] == 4 * _EPS_PER_TILE


def test_the_interval_is_tile_clustered_not_binomial(rollouts) -> None:
    """Episodes on one tile share its geometry, so a pooled binomial is too narrow.

    With 4 tiles per family the clustered interval must be materially wider than the binomial
    one over the same 100 episodes -- if it is not, the clustering is not happening.
    """
    from stats import wilson

    root, lib = rollouts
    r = _recs(root, lib)[("run_a", "varied_boxes")]
    s = r["success"]
    n = r["n_episodes"]
    _, blo, bhi = wilson(int(round(s["value"] * n)), n)
    assert s["method"] in ("cluster_bootstrap", "wilson_clusters")
    assert (s["ci_hi"] - s["ci_lo"]) > (bhi - blo), (
        f"clustered width {s['ci_hi'] - s['ci_lo']:.3f} is no wider than the binomial "
        f"{bhi - blo:.3f}; the interval is not clustered")


def test_a_family_map_from_the_WRONG_library_is_refused(rollouts, tmp_path) -> None:
    """The failure this report exists to make impossible.

    Every id would resolve, the table would be full, and every rate would be filed under the
    wrong terrain -- so an unknown id has to be an error rather than an 'unknown' bucket.
    """
    root, _lib = rollouts
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as h:
        h.attrs["terrain_manifest"] = json.dumps(
            [{"terrain_id": 0, "family": "something_else"}])
    from box2_ic_report import load_cells

    with pytest.raises(SystemExit, match="wrong library"):
        ttr.by_family(load_cells(str(root), rounds=1), ttr.family_map(str(other)))


def test_the_library_is_resolved_from_the_recording(rollouts) -> None:
    """So the map cannot be chosen by a flag that happened to be stale."""
    root, lib = rollouts
    assert ttr.library_from_recording(str(root)) == str(lib)


def test_two_libraries_under_one_root_are_refused(rollouts, tmp_path) -> None:
    """One family map cannot be correct for two experiments."""
    root, lib = rollouts
    other = tmp_path / "lib2.h5"
    _write_library(other)
    _write_rollout(root / "run_b", other, seed=1)
    with pytest.raises(SystemExit, match="DIFFERENT clip libraries"):
        ttr.library_from_recording(str(root))


def test_a_library_with_no_manifest_is_refused(tmp_path) -> None:
    """Reducing per tile type on a flat-only library is a question with no answer."""
    p = tmp_path / "flat.h5"
    with h5py.File(p, "w"):
        pass
    with pytest.raises(SystemExit, match="no terrain_manifest"):
        ttr.family_map(str(p))


def test_the_report_names_the_rules_and_labels_the_pooled_row(rollouts) -> None:
    """The rule is part of the number, and the pooled row is a property of the tile MIX."""
    root, lib = rollouts
    text = ttr.format_report(_recs(root, lib), metric="terminal_reward", strict=True,
                             rounds=1, lib_path=str(lib))
    assert "arrival BEFORE the first failure" in text
    assert "tracking bounds" in text
    assert "pooled over this mix" in text and "mix-dependent" in text
    assert "TILE MIX" in text
    for fam in ("varied_boxes", "entry_stairs_up"):
        assert fam in text


def test_the_pooled_row_moves_with_the_MIX_alone(rollouts) -> None:
    """Quantifies the warning the report prints, so it is a measurement and not a caution.

    The same per-family rates under a different tile mix give a different pooled number, which
    is why a multi-terrain arm can only be compared per family.
    """
    root, lib = rollouts
    recs = _recs(root, lib)
    rates = {f: r["success"]["value"] for (_run, f), r in recs.items()}
    even = np.mean([rates["varied_boxes"], rates["entry_stairs_up"]])
    box_heavy = 0.75 * rates["varied_boxes"] + 0.25 * rates["entry_stairs_up"]
    assert abs(even - box_heavy) > 0.1, (
        "the two mixes should give materially different pooled rates from identical "
        "per-family rates")


def test_the_stamp_is_JSON_decoded_the_way_the_recorder_wrote_it(rollouts) -> None:
    """A raw read returns the path with its quotes still on and every lookup misses."""
    root, lib = rollouts
    with h5py.File(next((root / "run_a").glob("*.episodes.h5")), "r") as h:
        raw = h.attrs["clip_library_path"]
    assert raw.startswith('"'), "the fixture no longer matches how the recorder writes it"
    assert ttr.library_from_recording(str(root)) == str(lib)
    assert pathlib.Path(ttr.library_from_recording(str(root))).is_file()


def test_a_lib_that_disagrees_with_the_stamp_is_refused(rollouts, tmp_path) -> None:
    """A SUPERSET library resolves every id and files every rate under the wrong terrain.

    The unknown-id check cannot catch it -- there are no unknown ids -- so the disagreement
    has to be caught at the source, by comparing against what the recording says it ran on.
    """
    root, _lib = rollouts
    bigger = tmp_path / "superset.h5"
    _write_library(bigger)      # same ids, different file
    import subprocess

    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "rsl_rl" / "tile_type_report.py"),
         str(root), "--lib", str(bigger)],
        capture_output=True, text=True)
    assert r.returncode != 0 and "was made on" in (r.stdout + r.stderr), (
        f"a disagreeing --lib was accepted:\n{r.stdout}\n{r.stderr}")
