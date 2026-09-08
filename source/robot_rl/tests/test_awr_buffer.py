# Copyright 2026 Zachary Olkin. All rights reserved.

"""The AWR replay buffer's churn, and the ways a wrong buffer would be SILENT.

Every test here targets a failure that produces a plausible, trainable artifact rather than an
error: a reused uid pairs an episode with another episode's reward, a truncated merge reads as
an empty table, an unnamed stem is merged whole.  The happy path is one test; the rest are the
silent ones.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys

import h5py
import numpy as np
import pytest

_TESTS = pathlib.Path(__file__).resolve().parent
_REPO = _TESTS.parents[2]
_SCRIPTS = _REPO / "scripts"
for _p in (_REPO / "source" / "motion_gen", _REPO / "source" / "robot_rl",
           _SCRIPTS / "rsl_rl", _SCRIPTS / "flow_matching"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rollout_outcomes as ro  # noqa: E402

_CONTRACT = {"num_history": 4, "state_dim": 44, "future_len": 64, "cmd_dim": 3,
             "extero_shape": [2, 3900], "proprio_hw": [10, 40], "scan_dim": 176}


def _load(name: str):
    """Import a script module by path -- ``scripts/`` is not a package.

    Args:
        name: Module stem, resolved under ``scripts/rsl_rl`` then ``scripts/flow_matching``.

    Returns:
        The imported module.
    """
    for parent in ("rsl_rl", "flow_matching"):
        path = _SCRIPTS / parent / f"{name}.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise AssertionError(f"{name}.py not found")


def _write_increment(root: str, iteration: int, n_eps: int, per_ep: int = 3,
                     first_uid: int = 0, contract: dict | None = None,
                     reward_meta: dict | None = None, allocated: int | None = None,
                     n_censored: int = 2, n_envs: int = 1) -> str:
    """Write a readable rollout pair plus its terrain-reward sidecar into ``inc_<NNN>/``.

    Args:
        root: Run root.
        iteration: 1-based iteration whose increment this is.
        n_eps: USABLE (uncensored) episodes to write.
        per_ep: Re-plan windows per episode.
        n_envs: Envs to spread the episodes over, for the per-env round selection.
        n_censored: Extra rows marked ``censored``, on top of ``n_eps``. Defaults to 2 rather
            than 0 because every real recording has exactly ``num_envs`` of them --
            ``record_rollouts`` ends by flushing whatever is still in flight -- and a fixture
            without them cannot see the buffer inflation that costs.
        first_uid: Starting uid -- 0 in every real recording, which is the collision the
            global uid space exists to resolve.
        contract: Window contract attrs.
        reward_meta: Extra sidecar metadata fields.
        allocated: ``num_windows``; defaults to twice what is written (cap did not bind).

    Returns:
        The stem written.
    """
    d = os.path.join(root, f"inc_{iteration:03d}")
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, "rollout_trk")
    n_all = n_eps + n_censored
    uids = np.arange(first_uid, first_uid + n_all, dtype=np.int64)
    censored = np.zeros(n_all, dtype=np.uint8)
    censored[n_eps:] = 1           # the end-of-run flush, exactly as the recorder writes it
    win_uid = np.repeat(uids, per_ep)
    ridx = np.tile(np.arange(per_ep, dtype=np.int64), n_all)

    with h5py.File(stem + ".h5", "w") as f:
        f.create_dataset("episode_uid", data=win_uid)
        f.create_dataset("replan_index", data=ridx)
        f.create_dataset("payload", data=np.arange(win_uid.size, dtype=np.float32))
        f.attrs["schema_version"] = ro.SCHEMA_VERSION
        f.attrs["num_windows_written"] = win_uid.size
        f.attrs["num_windows"] = win_uid.size * 2 if allocated is None else allocated
        for k, v in (contract or _CONTRACT).items():
            f.attrs[k] = json.dumps(v)
    with h5py.File(stem + ".episodes.h5", "w") as f:
        f.create_dataset("episode_uid", data=uids)
        f.create_dataset("censored", data=censored)
        # Envs cycle, so uid order interleaves them exactly as a real rollout does.
        f.create_dataset("env_id", data=(np.arange(n_all) % max(1, n_envs)).astype(np.int64))
        f.create_dataset("n_replans", data=np.full(n_all, per_ep, dtype=np.int64))
        f.attrs["schema_version"] = ro.SCHEMA_VERSION
        f.attrs["tracking_spec"] = json.dumps({"mode": "body"})
        f.attrs["history_from"] = "generator"

    meta = {"reward_source": "terrain_consistency", "tau": 0.5, "thresh": 0.1, "dilate": 2,
            "max_gap": 3, "spacing": "served", "contact_weight": 68.0,
            "penetration_weight": 1.0, "rollout": os.path.abspath(stem),
            "n_windows": int(win_uid.size)}
    meta.update(reward_meta or {})
    payload = {
        "episode_uid": win_uid.copy(), "replan_index": ridx.astype(np.int32),
        "reward": np.full(win_uid.size, 0.9), "penalty": np.full(win_uid.size, 0.1),
        "meta_json": json.dumps(meta, sort_keys=True),
    }
    payload["reward_hash"] = "0" * 16
    np.savez_compressed(os.path.join(d, "terrain_reward_sidecar.npz"), **payload)
    return stem


def _run(root: str, iteration: int, capacity: int, refresh: int, n_new: int,
         seed: int = 0, **inc_kw):
    """Write, adopt, plan and materialise one iteration.

    Args:
        root: Run root.
        iteration: 1-based iteration number.
        capacity: Stated buffer capacity.
        refresh: Episodes to drop (0 on iteration 1).
        n_new: Episodes in the new increment.
        seed: Drop seed.
        **inc_kw: Forwarded to :func:`_write_increment`.

    Returns:
        The ``awr_buffer`` module, for further calls.
    """
    ab = _load("awr_buffer")
    _write_increment(root, iteration, n_new, **inc_kw)
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=iteration, expect_episodes=n_new, policy="ckpt.pt",
                    rounds=0, trim_seed=0))
    ab.cmd_plan(ns(root=root, iteration=iteration, capacity=capacity, refresh=refresh,
                   seed=seed))
    ab.cmd_materialise(ns(root=root, iteration=iteration))
    return ab


def test_iteration_one_holds_exactly_the_named_episodes_and_no_censored_rows(tmp_path):
    """The buffer is what the manifest says -- not the increment, which has censored rows.

    `record_rollouts` always flushes its in-flight envs, so an increment carries `num_envs`
    extra `censored` rows that `fit_value` drops. Using the increment in place would put
    episodes in the buffer that the manifest does not name and the critic will not use.
    """
    root = str(tmp_path)
    ab = _run(root, 1, capacity=6, refresh=0, n_new=6, n_censored=4)
    man = json.loads((tmp_path / "iter_001" / "buffer.json").read_text())
    assert man["n_episodes"] == 6
    assert man["stems"][0]["n_censored_excluded"] == 4
    with h5py.File(man["buffer"]["stem"] + ".episodes.h5", "r") as f:
        assert f["episode_uid"].shape[0] == 6                 # not 10
        assert int(f["censored"][...].sum()) == 0
    assert ab.verify_buffer(root, 1) == 0


def test_the_censored_rows_do_not_inflate_the_buffer_across_iterations(tmp_path):
    """THE RUN-KILLER, and then the silent version of it.

    Every increment brings `num_envs` censored rows. If they were counted, `--capacity` could
    never be satisfied (the run dies at stage 4); if the capacity check were then relaxed, the
    USABLE buffer would grow monotonically instead of holding steady, and the loop's premise
    -- a fixed-size window over ~4 policy generations -- would quietly stop being true.
    """
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6, n_censored=5)
    for it in (2, 3):
        _run(root, it, capacity=6, refresh=3, n_new=3, n_censored=5)
        man = json.loads((pathlib.Path(root) / f"iter_{it:03d}" / "buffer.json").read_text())
        assert man["n_episodes"] == 6, f"iteration {it} drifted to {man['n_episodes']}"
        with h5py.File(man["buffer"]["stem"] + ".episodes.h5", "r") as f:
            assert f["episode_uid"].shape[0] == 6
            assert int(f["censored"][...].sum()) == 0


def test_an_over_collection_is_trimmed_without_favouring_short_episodes(tmp_path):
    """A uid is assigned when an episode ENDS, so a prefix would be the failures."""
    root = str(tmp_path)
    ab = _load("awr_buffer")
    _write_increment(root, 1, 40, n_censored=3)
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=10, policy="", rounds=0, trim_seed=0))
    rec = json.loads((tmp_path / "inc_001" / "adopt.json").read_text())
    assert rec["n_episodes"] == 10
    local = [u - rec["uid_base"] for u in rec["uids"]]
    # A prefix would be 0..9; a random draw over 40 essentially never is.
    assert local != list(range(10))
    assert max(local) < 40 and min(local) >= 0


def test_the_uid_space_is_global_so_a_dropped_uid_is_never_reused(tmp_path):
    """The silent one: renumbering would give a dropped uid to a fresh episode.

    Both increments number their episodes from 0 on disk -- that IS what a real recording
    does. If the merge renumbered them, iteration 2's offset would depend on which of
    iteration 1's episodes the drop removed, so a reward sidecar written before the drop would
    join to the WRONG episodes and `_reward_sidecar` would report partial coverage and fit.
    """
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    ab = _run(root, 2, capacity=6, refresh=3, n_new=3)

    man = json.loads((tmp_path / "iter_002" / "buffer.json").read_text())
    with h5py.File(man["buffer"]["stem"] + ".episodes.h5", "r") as f:
        got = np.sort(f["episode_uid"][...])
    want = np.array(sorted(u for r in man["stems"] for u in r["uids"]), dtype=np.int64)
    assert np.array_equal(got, want)
    # Every uid still carries its generating iteration, and the two bands cannot overlap.
    assert set(int(u) // ab.UID_STRIDE for u in got) == {1, 2}
    assert len(set(got.tolist())) == got.size
    assert ab.verify_buffer(root, 2) == 0


def test_the_sidecar_still_joins_after_a_drop(tmp_path):
    """Coverage stays complete -- this is the end-to-end detector for uid drift."""
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    _run(root, 2, capacity=6, refresh=3, n_new=3)
    man = json.loads((tmp_path / "iter_002" / "buffer.json").read_text())
    with h5py.File(man["buffer"]["stem"] + ".h5", "r") as f:
        n = int(f.attrs["num_windows_written"])
        key = (f["episode_uid"][:n].astype(np.int64) << np.int64(20)) \
            + f["replan_index"][:n].astype(np.int64)
    with np.load(man["buffer"]["reward_npz"], allow_pickle=False) as z:
        have = (z["episode_uid"].astype(np.int64) << np.int64(20)) \
            + z["replan_index"].astype(np.int64)
    assert float(np.isin(key, have).mean()) == 1.0


def test_adopt_is_idempotent_because_rebasing_twice_would_move_the_band(tmp_path):
    """A second adopt must not shift the uids again."""
    root = str(tmp_path)
    ab = _load("awr_buffer")
    stem = _write_increment(root, 1, 4)
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=4, policy="",
                    rounds=0, trim_seed=0))
    with h5py.File(stem + ".episodes.h5", "r") as f:
        first = f["episode_uid"][...].copy()
    ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=4, policy="",
                    rounds=0, trim_seed=0))
    with h5py.File(stem + ".episodes.h5", "r") as f:
        assert np.array_equal(f["episode_uid"][...], first)


def test_plan_never_redraws_the_drop(tmp_path):
    """The manifest is the commit point: a redraw would not match a half-written merge."""
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    ab = _load("awr_buffer")
    _write_increment(root, 2, 3)
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=2, expect_episodes=3, policy="",
                    rounds=0, trim_seed=0))
    ab.cmd_plan(ns(root=root, iteration=2, capacity=6, refresh=3, seed=0))
    first = (tmp_path / "iter_002" / "buffer.json").read_text()
    ab.cmd_plan(ns(root=root, iteration=2, capacity=6, refresh=3, seed=1))
    assert (tmp_path / "iter_002" / "buffer.json").read_text() == first


def test_a_truncated_collection_is_refused_rather_than_shrinking_the_buffer(tmp_path):
    """`--num_windows` binding before `--num_episodes` must not pass silently."""
    root = str(tmp_path)
    ab = _load("awr_buffer")
    _write_increment(root, 1, 4, per_ep=3, allocated=12)      # written == allocated
    ns = __import__("argparse").Namespace
    with pytest.raises(SystemExit, match="HARD CAP"):
        ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=4, policy="",
                    rounds=0, trim_seed=0))


def test_a_short_collection_is_refused(tmp_path):
    """Fewer episodes than asked for silently shrinks the buffer below its capacity."""
    root = str(tmp_path)
    ab = _load("awr_buffer")
    _write_increment(root, 1, 3)
    ns = __import__("argparse").Namespace
    with pytest.raises(SystemExit, match="3 USABLE episodes"):
        ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=6, policy="",
                    rounds=0, trim_seed=0))


def test_a_moved_tracking_spec_is_refused_even_though_the_merge_allows_it(tmp_path):
    """`_CONTRACT_ATTRS` omits `tracking_spec` on purpose; across AWR iterations it matters."""
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    ab = _load("awr_buffer")
    d = os.path.join(root, "inc_002")
    _write_increment(root, 2, 3)
    with h5py.File(os.path.join(d, "rollout_trk.episodes.h5"), "r+") as f:
        f.attrs["tracking_spec"] = json.dumps({"mode": "joint"})
    ns = __import__("argparse").Namespace
    with pytest.raises(SystemExit, match="moved since iteration 1"):
        ab.cmd_adopt(ns(root=root, iteration=2, expect_episodes=3, policy="",
                    rounds=0, trim_seed=0))


def test_two_graders_are_refused_rather_than_becoming_one_quantity(tmp_path):
    """`fit_value` reads ONE meta_json for the whole file, so a mixed grade is invisible."""
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    ab = _load("awr_buffer")
    _write_increment(root, 2, 3, reward_meta={"tau": 0.25})
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=2, expect_episodes=3, policy="",
                    rounds=0, trim_seed=0))
    ab.cmd_plan(ns(root=root, iteration=2, capacity=6, refresh=3, seed=0))
    with pytest.raises(SystemExit, match="reward field 'tau'"):
        ab.cmd_materialise(ns(root=root, iteration=2))


def test_a_manifest_whose_size_claim_is_wrong_is_not_committed(tmp_path):
    """Every later size check reads the manifest, so its claim must be true when written."""
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    ab = _load("awr_buffer")
    _write_increment(root, 2, 3)
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=2, expect_episodes=3, policy="",
                    rounds=0, trim_seed=0))
    with pytest.raises(SystemExit, match="not the stated capacity"):
        ab.cmd_plan(ns(root=root, iteration=2, capacity=6, refresh=1, seed=0))


def test_a_merge_killed_before_it_stamped_its_count_reads_as_empty_and_is_refused(tmp_path):
    """The dangerous one: `read_rollout_windows` returns ZERO rows, not an error."""
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    ab = _run(root, 2, capacity=6, refresh=3, n_new=3)
    man = json.loads((tmp_path / "iter_002" / "buffer.json").read_text())
    # BOTH attrs. `read_rollout_windows` falls back to `num_windows` when the written count
    # is absent, so deleting only the latter would leave the table reading as a plausible
    # full one -- the earlier version of this test passed on `verify`'s own `get(..., 0)`
    # default rather than on the mechanism its name describes.
    with h5py.File(man["buffer"]["stem"] + ".h5", "r+") as f:
        del f.attrs["num_windows_written"]
        del f.attrs["num_windows"]
    import rollout_outcomes as _ro
    cols, _ = _ro.read_rollout_windows(man["buffer"]["stem"] + ".h5")
    assert cols["episode_uid"].size == 0, "the reader should see an empty table here"
    with pytest.raises(SystemExit, match="EMPTY"):
        ab.verify_buffer(root, 2)


def test_a_rebase_killed_between_the_two_tables_is_completed_not_repeated(tmp_path):
    """`adopt` mutates three artifacts and writes its marker LAST.

    A kill in between used to leave a partially shifted increment that a resumed `adopt`
    shifted AGAIN -- detected only downstream, as a coverage failure or an opaque h5py
    broadcast error. Each artifact now records its own uid base, so a resume finishes the
    half-done work.
    """
    root = str(tmp_path)
    ab = _load("awr_buffer")
    stem = _write_increment(root, 1, 6)
    # Simulate the crash: the WINDOW table is shifted and stamped, nothing else is.
    with h5py.File(stem + ".h5", "r+") as f:
        f["episode_uid"][...] = f["episode_uid"][...].astype(np.int64) + ab.UID_STRIDE
        f.attrs["awr_uid_base"] = ab.UID_STRIDE
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=6, policy="", rounds=0, trim_seed=0))
    with h5py.File(stem + ".h5", "r") as f:
        win = np.unique(f["episode_uid"][...])
    with h5py.File(stem + ".episodes.h5", "r") as f:
        eps = np.unique(f["episode_uid"][...])
    # The window table must NOT have been shifted twice, and the two tables must agree.
    assert win.min() == ab.UID_STRIDE and win.max() < 2 * ab.UID_STRIDE
    assert set(win.tolist()) == set(eps.tolist())


def test_a_local_uid_at_the_stride_is_refused_because_the_bands_would_overlap(tmp_path):
    """The precondition the entire band scheme rests on, asserted rather than assumed."""
    root = str(tmp_path)
    ab = _load("awr_buffer")
    _write_increment(root, 1, 3, first_uid=ab.UID_STRIDE - 1)
    ns = __import__("argparse").Namespace
    with pytest.raises(SystemExit, match="UID_STRIDE"):
        ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=3, policy="", rounds=0, trim_seed=0))


def test_verify_catches_an_episode_set_that_drifts_from_the_manifest(tmp_path):
    """A buffer holding other episodes than the manifest names must not pass."""
    root = str(tmp_path)
    _run(root, 1, capacity=6, refresh=0, n_new=6)
    ab = _run(root, 2, capacity=6, refresh=3, n_new=3)
    path = tmp_path / "iter_002" / "buffer.json"
    man = json.loads(path.read_text())
    man["stems"][0]["uids"] = man["stems"][0]["uids"][:-1] + [999999999]
    path.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="does not match the manifest"):
        ab.verify_buffer(root, 2)


def test_the_drop_is_reproducible_from_the_seed(tmp_path):
    """The manifest records the seed pair, so the churn can be replayed."""
    def run(where):
        _run(where, 1, capacity=6, refresh=0, n_new=6)
        _run(where, 2, capacity=6, refresh=3, n_new=3, seed=7)
        return json.loads((pathlib.Path(where) / "iter_002" / "buffer.json").read_text())
    a = run(str(tmp_path / "a"))
    b = run(str(tmp_path / "b"))
    assert a["drop"]["uids"] == b["drop"]["uids"]
    assert a["drop"]["seed"] == [7, 2]


def test_only_the_first_round_per_env_is_kept(tmp_path):
    """The surplus a round-based stop leaves behind is the SHORT tail, so it is discarded.

    A rollout stops when the slowest env finishes its k-th episode, by which time the fast
    envs have finished more -- and an env finishes more precisely when its episodes were
    short. Keeping the surplus would tilt the sample back toward the short episodes the
    round-based stop exists to stop selecting.
    """
    root = str(tmp_path)
    ab = _load("awr_buffer")
    # 4 envs, 12 completed episodes: env 0 finished 3, the others 3 each by construction,
    # so asking for 1 round must keep exactly 4 -- one per env.
    _write_increment(root, 1, 12, n_censored=4, n_envs=4)
    ns = __import__("argparse").Namespace
    ab.cmd_adopt(ns(root=root, iteration=1, expect_episodes=0, policy="", rounds=1,
                    trim_seed=0))
    rec = json.loads((tmp_path / "inc_001" / "adopt.json").read_text())
    assert rec["n_episodes"] == 4, rec["n_episodes"]
    local = sorted(u - rec["uid_base"] for u in rec["uids"])
    assert local == [0, 1, 2, 3], local           # the FIRST episode of each env


def test_first_per_env_is_a_complete_design(tmp_path):
    """Two rounds over three envs keeps exactly six, two from each."""
    ab = _load("awr_buffer")
    uids = np.arange(12, dtype=np.int64)
    env = np.arange(12, dtype=np.int64) % 3
    keep = ab.first_per_env(uids, env, rounds=2)
    assert len(keep) == 6
    per = {}
    for u in keep:
        per[int(env[u])] = per.get(int(env[u]), 0) + 1
    assert per == {0: 2, 1: 2, 2: 2}


def test_rounds_zero_keeps_everything(tmp_path):
    """A recording made before rounds existed must still adopt whole."""
    ab = _load("awr_buffer")
    uids = np.arange(5, dtype=np.int64)
    assert ab.first_per_env(uids, np.zeros(5, dtype=np.int64), rounds=0) == [0, 1, 2, 3, 4]
