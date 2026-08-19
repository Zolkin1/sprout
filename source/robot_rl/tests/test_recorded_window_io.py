# Copyright 2026 Zachary Olkin. All rights reserved.

"""Round-trip tests for the streaming windowed HDF5 recorder (RecordedWindowWriter).

Numpy + h5py only (no IsaacLab): write a few batches of uniquely-valued windows, read
them back with load_recorded_windows, and assert bit-exact arrays, trimming to
num_windows_written, optional-array omission, and metadata/content-hash round-trip.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# ``recorded_data_io`` lives in scripts/rsl_rl, which is not a package on the path. Insert it
# HERE rather than inheriting it from whichever other test module happened to be collected
# first: without this the file passes inside the full suite and ERRORS when run alone, which
# is the same collection-order fragility that made test_train_cli_defaults import the wrong
# train.py. Off the repo root, never a count of ".." hops (see CLAUDE.md).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          *[os.pardir] * 3))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "rsl_rl"))

from recorded_data_io import (
    RecordedWindowWriter,
    RecordingMeta,
    fingerprint_windows,
    is_windowed_recording,
    load_recorded_windows,
)


def _meta(W: int, H: int, F: int, K: int) -> RecordingMeta:
    """A minimal but complete RecordingMeta for the windowed format."""
    return RecordingMeta(
        checkpoint_path="ckpt.pt", checkpoint_sha256="abc", task_name="G1-x",
        experiment_name="g1", run_name="run", runner_class="OnPolicyRunner",
        rsl_rl_version="4.0.0", recorded_at="2026-01-01", stochastic=False, dt=0.02,
        num_envs=64, num_pairs=W,
        obs_term_names=["a"], obs_term_dims=[1], action_joint_names=["j0", "j1", "j2"],
        state_term_names=["root_pos_w", "root_quat_w", "joint_pos"], state_term_dims=[3, 4, 3],
        state_quat_format="xyzw", state_joint_names=["j0", "j1", "j2"], state_vel_frame="link",
        state_frame_names=["left_ankle", "right_ankle"],
        ref_source="ego_anchored_clip_future", ref_future_len=F,
        ref_future_term_names=["root_pos_w", "root_quat_w", "joint_pos"],
        ref_future_term_dims=[3, 4, 3], height_scan_dim=K,
        format="windowed_h5", history_len=H,
        reference_blocks=["base_pose", "base_vel", "joint_pos"],
        history_anchor="plan",
    )


def _rand(W, H, F, node, K, cmd, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "history_state": rng.standard_normal((W, H, node)).astype(np.float32),
        "ref_future": rng.standard_normal((W, F, node)).astype(np.float32),
        "height_scan": rng.standard_normal((W, K)).astype(np.float32),
        "cmd_future": rng.standard_normal((W, F, cmd)).astype(np.float32),
        "episode_id": np.arange(W, dtype=np.int64),
    }


def test_writer_roundtrip_bit_exact():
    """Full write (in two batches) -> load returns bit-exact arrays + meta + content hash."""
    W, H, F, node, K, cmd = 12, 8, 40, 107, 176, 3
    arrs = _rand(W, H, F, node, K, cmd)
    meta = _meta(W, H, F, K)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded_x.h5")
        assert is_windowed_recording(p)
        with RecordedWindowWriter(p, W, H, node, F, K, cmd, meta) as wr:
            wr.write_batch({k: v[:5] for k, v in arrs.items()})
            wr.write_batch({k: v[5:] for k, v in arrs.items()})
            wr.set_attr("content_sha256", fingerprint_windows(arrs))
        data, m2 = load_recorded_windows(p)
        for k, v in arrs.items():
            assert np.array_equal(data[k], v), f"{k} not bit-exact"
        assert m2.history_len == H and m2.ref_future_len == F and m2.height_scan_dim == K
        assert m2.format == "windowed_h5"
        assert m2.state_term_names == meta.state_term_names
        assert m2.state_frame_names == meta.state_frame_names
        # The recording teacher's window selection: it is what the FM dataset resolves the
        # generator's WIDTH from, so losing it in the round trip would silently train the
        # next generator on all five blocks again.
        assert m2.reference_blocks == meta.reference_blocks
        assert m2.content_sha256 == fingerprint_windows(arrs)


def test_writer_trims_to_written():
    """An under-filled file (fewer windows written than pre-sized) trims on load."""
    W, H, F, node, K, cmd = 10, 4, 8, 22, 5, 3
    arrs = _rand(W, H, F, node, K, cmd)
    meta = _meta(W, H, F, K)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded_y.h5")
        with RecordedWindowWriter(p, W, H, node, F, K, cmd, meta) as wr:
            n = wr.write_batch({k: v[:7] for k, v in arrs.items()})  # only 7 of 10
            assert n == 7 and wr.num_written == 7
        data, _ = load_recorded_windows(p)
        assert data["history_state"].shape[0] == 7
        assert np.array_equal(data["ref_future"], arrs["ref_future"][:7])


def test_writer_clamps_overflow():
    """write_batch clamps to the remaining capacity and reports the count taken."""
    W, H, F, node, K, cmd = 5, 2, 4, 22, 0, 3  # K=0 -> no height dataset
    arrs = _rand(W, H, F, node, K=1, cmd=cmd)  # build then drop height
    arrs.pop("height_scan")
    meta = _meta(W, H, F, 0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded_z.h5")
        with RecordedWindowWriter(p, W, H, node, F, 0, cmd, meta) as wr:
            took = wr.write_batch({k: np.tile(v[:1], (8, 1, 1) if v.ndim == 3 else (8,))
                                   for k, v in arrs.items()})  # ask 8, cap 5
            assert took == 5
        data, _ = load_recorded_windows(p)
        assert "height_scan" not in data  # optional dataset omitted when K==0
        assert data["history_state"].shape[0] == 5
        assert "cmd_future" in data


def test_optional_arrays_omitted():
    """With cmd_dim==0 and K==0, only the mandatory datasets exist."""
    W, H, F, node = 4, 2, 4, 22
    meta = _meta(W, H, F, 0)
    rng = np.random.default_rng(1)
    batch = {
        "history_state": rng.standard_normal((W, H, node)).astype(np.float32),
        "ref_future": rng.standard_normal((W, F, node)).astype(np.float32),
        "episode_id": np.arange(W, dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded_w.h5")
        with RecordedWindowWriter(p, W, H, node, F, 0, 0, meta) as wr:
            wr.write_batch(batch)
        data, _ = load_recorded_windows(p)
        assert set(data) == {"history_state", "ref_future", "episode_id"}


def test_clip_family_needs_tag_families_to_be_stored():
    """``clip_family`` requires ``tag_families``, the analogue of ``tag_distractors``.

    Regression: the writer declared ``is_distractor`` behind a flag but had no families flag,
    so ``write_batch`` silently DROPPED the key it was handed and ``train.py --skip_families``
    could never find a tag to filter on. Asserting the round trip AND the omission, because
    the failure mode was a silent drop rather than an error.
    """
    W, H, F, node = 6, 2, 4, 22
    meta = _meta(W, H, F, 0)
    rng = np.random.default_rng(2)
    fam = np.arange(W, dtype=np.uint8) % 3
    batch = {
        "history_state": rng.standard_normal((W, H, node)).astype(np.float32),
        "ref_future": rng.standard_normal((W, F, node)).astype(np.float32),
        "episode_id": np.arange(W, dtype=np.int64),
        "clip_family": fam,
    }
    with tempfile.TemporaryDirectory() as d:
        on = os.path.join(d, "recorded_on.h5")
        with RecordedWindowWriter(on, W, H, node, F, 0, 0, meta, tag_families=True) as wr:
            wr.write_batch(batch)
        data, _ = load_recorded_windows(on)
        assert np.array_equal(data["clip_family"], fam)

        off = os.path.join(d, "recorded_off.h5")
        with RecordedWindowWriter(off, W, H, node, F, 0, 0, meta) as wr:
            wr.write_batch(batch)
        data_off, _ = load_recorded_windows(off)
        assert "clip_family" not in data_off


def test_clip_index_needs_tag_clip_index_to_be_stored():
    """``clip_id``/``clip_step`` need ``tag_clip_index``, like ``clip_family`` needs its flag.

    Same silent-drop failure mode as ``test_clip_family_needs_tag_families_to_be_stored``:
    ``write_batch`` iterates the DECLARED keys, so a key the recorder assembles but the
    writer never declared vanishes without an error. Asserting the round trip AND the
    omission, because a missing index reads as "this recording predates the tag" rather
    than as a bug.
    """
    W, H, F, node = 6, 2, 4, 22
    meta = _meta(W, H, F, 0)
    rng = np.random.default_rng(3)
    cid = np.arange(W, dtype=np.int32) * 7
    cstep = np.arange(W, dtype=np.int32) * 13 + 1
    batch = {
        "history_state": rng.standard_normal((W, H, node)).astype(np.float32),
        "ref_future": rng.standard_normal((W, F, node)).astype(np.float32),
        "episode_id": np.arange(W, dtype=np.int64),
        "clip_id": cid,
        "clip_step": cstep,
        "env_id": np.arange(W, dtype=np.int32) % 4,
    }
    with tempfile.TemporaryDirectory() as d:
        on = os.path.join(d, "recorded_on.h5")
        with RecordedWindowWriter(on, W, H, node, F, 0, 0, meta, tag_clip_index=True) as wr:
            wr.write_batch(batch)
        data, _ = load_recorded_windows(on)
        assert np.array_equal(data["clip_id"], cid)
        assert np.array_equal(data["clip_step"], cstep)
        assert np.array_equal(data["env_id"], np.arange(W, dtype=np.int32) % 4)

        off = os.path.join(d, "recorded_off.h5")
        with RecordedWindowWriter(off, W, H, node, F, 0, 0, meta) as wr:
            wr.write_batch(batch)
        data_off, _ = load_recorded_windows(off)
        assert not ({"clip_id", "clip_step", "env_id"} & set(data_off))


def test_clip_library_identity_roundtrips():
    """The library path AND its hash survive, so a clip_id join can prove its source.

    ``clip_id`` is a position in one particular library and libraries are regenerated in
    place under the same filename, so the path alone cannot detect a mismatch.
    """
    W, H, F, node = 4, 2, 4, 22
    meta = _meta(W, H, F, 0)
    meta.clip_library_path = "clips/g1_mb_v5_body.h5"
    meta.clip_library_sha256 = "deadbeef" * 8
    rng = np.random.default_rng(4)
    batch = {
        "history_state": rng.standard_normal((W, H, node)).astype(np.float32),
        "ref_future": rng.standard_normal((W, F, node)).astype(np.float32),
        "episode_id": np.arange(W, dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded.h5")
        with RecordedWindowWriter(p, W, H, node, F, 0, 0, meta) as wr:
            wr.write_batch(batch)
        _, got = load_recorded_windows(p)
        assert got.clip_library_path == "clips/g1_mb_v5_body.h5"
        assert got.clip_library_sha256 == "deadbeef" * 8


def test_robot_asset_roundtrips():
    """The recording must say WHICH ROBOT produced its rows.

    "No re-record" for the depth xyz channels holds only while the camera and the FK model
    are fixed: the offline unprojection rebuilds the camera pose from the recorded joint
    angles through motion_gen's FKChain, so a recording replayed against a changed model is
    silently mis-registered. Every other identity here pins the data or the sensor.
    """
    W, H, F, node = 4, 2, 4, 22
    meta = _meta(W, H, F, 0)
    meta.robot_asset = {"hash": "f" * 64, "short": "ffffffffffff", "version": 1,
                        "joint_names": ["j0", "j1", "j2"], "total_mass_kg": 36.1683}
    rng = np.random.default_rng(11)
    batch = {
        "history_state": rng.standard_normal((W, H, node)).astype(np.float32),
        "ref_future": rng.standard_normal((W, F, node)).astype(np.float32),
        "episode_id": np.arange(W, dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded.h5")
        with RecordedWindowWriter(p, W, H, node, F, 0, 0, meta) as wr:
            wr.write_batch(batch)
        _, got = load_recorded_windows(p)
    assert got.robot_asset == meta.robot_asset


def test_a_recording_without_a_robot_asset_still_loads():
    """Every recording made before the field exists has none; that must not be an error."""
    W, H, F, node = 2, 2, 4, 22
    meta = _meta(W, H, F, 0)
    assert meta.robot_asset is None
    rng = np.random.default_rng(12)
    batch = {
        "history_state": rng.standard_normal((W, H, node)).astype(np.float32),
        "ref_future": rng.standard_normal((W, F, node)).astype(np.float32),
        "episode_id": np.arange(W, dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded.h5")
        with RecordedWindowWriter(p, W, H, node, F, 0, 0, meta) as wr:
            wr.write_batch(batch)
        _, got = load_recorded_windows(p)
    assert got.robot_asset is None


def test_depth_roundtrips_alongside_the_height_scan():
    """Both exteroception arrays survive one recording, so either FM can train from it.

    Recording BOTH is what makes ``cfg.extero_source`` a train-time choice instead of a
    re-record; the depth array keeps its ``(W, num_cameras, per_camera)`` token axis so
    the dataset can resolve the token count from the shape.
    """
    W, H, F, node, K, cmd = 6, 4, 8, 22, 5, 3
    cams, per_cam = 2, 15
    arrs = _rand(W, H, F, node, K, cmd)
    rng = np.random.default_rng(1)
    arrs["depth"] = rng.standard_normal((W, cams, per_cam)).astype(np.float32)
    meta = _meta(W, H, F, K)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded_depth.h5")
        with RecordedWindowWriter(
            p, W, H, node, F, K, cmd, meta, depth_shape=(cams, per_cam)
        ) as wr:
            wr.write_batch(arrs)
        data, _ = load_recorded_windows(p)
        assert np.array_equal(data["depth"], arrs["depth"])
        assert data["depth"].shape == (W, cams, per_cam)
        assert np.array_equal(data["height_scan"], arrs["height_scan"])


def test_depth_dataset_is_omitted_when_not_recorded():
    """A scan-only recording must not gain an empty depth dataset."""
    W, H, F, node, K, cmd = 4, 3, 5, 22, 5, 3
    arrs = _rand(W, H, F, node, K, cmd)
    meta = _meta(W, H, F, K)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recorded_nodepth.h5")
        with RecordedWindowWriter(p, W, H, node, F, K, cmd, meta) as wr:
            wr.write_batch(arrs)
        data, _ = load_recorded_windows(p)
        assert "depth" not in data
        assert "height_scan" in data


##
# Phase C additions: per-key dtype, byte-capped chunks, the new meta fields
##

def test_scan_can_be_stored_as_float16():
    """Opt-in f16 for the dense scan: halves the file and the dataset's RAM.

    1 mm quantisation is far below a 0.025 m grid cell, and lzf compresses f16 sensor data
    much better than f32 (whose low mantissa bits are effectively noise). The override is
    per-writer, never a change to the module table, so the depth baseline's arrays stay
    bit-identical and the A/B is not confounded by a storage change.
    """
    W, H, F, node, K, cmd = 8, 4, 5, 10, 121 * 121, 3
    batch = _rand(W, H, F, node, K, cmd, seed=3)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "f16.h5")
        with RecordedWindowWriter(path, W, H, node, F, K, cmd, _meta(W, H, F, K),
                                  dtypes={"height_scan": "float16"}) as w:
            w.write_batch(batch)
        data, _ = load_recorded_windows(path)
        assert data["height_scan"].dtype == np.float16
        # Everything else keeps its declared dtype.
        assert data["history_state"].dtype == np.float32
        assert data["ref_future"].dtype == np.float32
        # f16 round-trips the value to ~1e-3, which is 25x finer than the grid cell.
        assert np.allclose(data["height_scan"], batch["height_scan"], atol=2e-3)


def test_an_unknown_dtype_override_is_refused():
    """A typo'd key would silently do nothing, which is how a "f16 run" ends up f32."""
    W, H, F, node, K, cmd = 4, 4, 5, 10, 16, 3
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(KeyError, match="unknown window array"):
            RecordedWindowWriter(os.path.join(td, "x.h5"), W, H, node, F, K, cmd,
                                 _meta(W, H, F, K), dtypes={"heightscan": "float16"})


def test_chunks_are_capped_by_bytes_not_just_window_count():
    """A fixed 256-window block is a 23 MB chunk on a dense scan.

    That is far past HDF5's 1 MB default chunk cache, so every read pushes the whole chunk
    through uncached. The block must fall as the row grows -- and must never reach 0.
    """
    from recorded_data_io import _CHUNK_TARGET_BYTES, _window_chunk

    # A narrow row still blocks the full 256 windows.
    assert _window_chunk((10_000, 8, 10), itemsize=4)[0] == 256
    # A dense scan row must be capped by bytes.
    dense = _window_chunk((10_000, 121 * 121), itemsize=4)
    assert dense[0] < 256
    assert dense[0] * 121 * 121 * 4 <= _CHUNK_TARGET_BYTES
    # f16 halves the row, so it fits ~twice the windows per chunk (floor division makes the
    # ratio inexact by one, hence the bound rather than equality).
    half = _window_chunk((10_000, 121 * 121), itemsize=2)
    assert 2 * dense[0] <= half[0] <= 2 * dense[0] + 1
    assert half[0] * 121 * 121 * 2 <= _CHUNK_TARGET_BYTES
    # Never zero, however wide the row.
    assert _window_chunk((10, 50_000_000), itemsize=4)[0] == 1
    # And never more windows than the dataset holds.
    assert _window_chunk((3, 8, 10), itemsize=4)[0] == 3


def test_scan_geometry_and_noise_flag_round_trip():
    """``height_scan_spec`` / ``extero_noise_free`` survive the JSON attr round-trip.

    The spec is what lets the FM dataset recompute the fold from the physical fields and
    refuse a file whose statements of the grid disagree, so it has to arrive intact.
    """
    W, H, F, node, K, cmd = 4, 4, 5, 10, 121 * 121, 3
    meta = _meta(W, H, F, K)
    meta.height_scan_spec = [{
        "sensor": "fm_height_scanner", "resolution": 0.025, "size": [3.0, 3.0],
        "ordering": "xy", "height": 121, "width": 121, "num_rays": K,
        "sensor_offset": 0.5, "privileged": True,
    }]
    meta.extero_noise_free = True
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "spec.h5")
        with RecordedWindowWriter(path, W, H, node, F, K, cmd, meta) as w:
            w.write_batch(_rand(W, H, F, node, K, cmd, seed=4))
        _, got = load_recorded_windows(path)
    assert got.extero_noise_free is True
    assert got.height_scan_spec == meta.height_scan_spec
    assert got.height_scan_spec[0]["privileged"] is True


def test_a_recording_without_the_new_fields_still_loads():
    """Backward compatible: the loader filters attrs against the dataclass fields.

    An older file simply has no ``height_scan_spec`` attr, and must come back as the
    defaults rather than raising -- every existing recording on disk is in that state.
    """
    W, H, F, node, K, cmd = 4, 4, 5, 10, 16, 3
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "old.h5")
        with RecordedWindowWriter(path, W, H, node, F, K, cmd, _meta(W, H, F, K)) as w:
            w.write_batch(_rand(W, H, F, node, K, cmd, seed=5))
        import h5py
        with h5py.File(path, "a") as f:          # simulate a pre-field recording
            del f.attrs["height_scan_spec"]
            del f.attrs["extero_noise_free"]
        _, got = load_recorded_windows(path)
    assert got.height_scan_spec is None
    assert got.extero_noise_free is False


##
# Selective loading: read only the arrays and rows a run will actually use
##

def test_meta_can_be_read_without_touching_arrays():
    """Deciding WHAT to load must not require loading it."""
    from recorded_data_io import read_recording_meta

    W, H, F, node, K, cmd = 6, 4, 5, 10, 32, 3
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.h5")
        with RecordedWindowWriter(path, W, H, node, F, K, cmd, _meta(W, H, F, K)) as w:
            w.write_batch(_rand(W, H, F, node, K, cmd, seed=11))
        meta, n = read_recording_meta(path)
    assert n == W and meta.height_scan_dim == K


def test_only_the_requested_arrays_are_loaded():
    """A recording holds BOTH exteroception signals so one file trains either generator.

    A given run reads exactly one, and on a dense scan the unused array is the largest thing
    in the file -- so not loading it is the biggest single saving available.
    """
    W, H, F, node, K, cmd = 6, 4, 5, 10, 64, 3
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "both.h5")
        with RecordedWindowWriter(path, W, H, node, F, K, cmd, _meta(W, H, F, K),
                                  depth_shape=(2, 40)) as w:
            w.write_batch(_rand(W, H, F, node, K, cmd, seed=12) | {
                "depth": np.zeros((W, 2, 40), np.float32)})
        full, _ = load_recorded_windows(path)
        scan_run, _ = load_recorded_windows(
            path, keys=[k for k in full if k != "depth"])
    assert "depth" in full and "height_scan" in full
    assert "depth" not in scan_run and "height_scan" in scan_run
    assert sum(v.nbytes for v in scan_run.values()) < sum(v.nbytes for v in full.values())


def test_row_subset_preserves_the_requested_order():
    """h5py needs STRICTLY increasing indices, so unsorted or repeated rows must be handled.

    The train/val split hands over arbitrary index arrays; silently returning them in sorted
    order would pair every window with the wrong label.
    """
    W, H, F, node, K, cmd = 12, 4, 5, 10, 16, 3
    batch = _rand(W, H, F, node, K, cmd, seed=13)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rows.h5")
        with RecordedWindowWriter(path, W, H, node, F, K, cmd, _meta(W, H, F, K)) as w:
            w.write_batch(batch)
        rows = np.array([9, 0, 5, 0, 11])          # unsorted, with a duplicate
        sub, _ = load_recorded_windows(path, keys=["height_scan", "episode_id"], rows=rows)
    assert sub["height_scan"].shape == (rows.shape[0], K)
    for i, r in enumerate(rows):
        assert np.array_equal(sub["height_scan"][i], batch["height_scan"][r])
    assert np.array_equal(sub["episode_id"], batch["episode_id"][rows])


def test_out_of_range_rows_are_refused():
    """Silently clipping would train on the wrong windows; h5py alone would not catch it."""
    W, H, F, node, K, cmd = 4, 4, 5, 10, 16, 3
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "oor.h5")
        with RecordedWindowWriter(path, W, H, node, F, K, cmd, _meta(W, H, F, K)) as w:
            w.write_batch(_rand(W, H, F, node, K, cmd, seed=14))
        with pytest.raises(IndexError, match="out of range"):
            load_recorded_windows(path, rows=np.array([0, W]))
