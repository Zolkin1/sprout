# Copyright 2026 Zachary Olkin. All rights reserved.

"""Round-trip tests for the streaming windowed HDF5 recorder (RecordedWindowWriter).

Numpy + h5py only (no IsaacLab): write a few batches of uniquely-valued windows, read
them back with load_recorded_windows, and assert bit-exact arrays, trimming to
num_windows_written, optional-array omission, and metadata/content-hash round-trip.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

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
    }
    with tempfile.TemporaryDirectory() as d:
        on = os.path.join(d, "recorded_on.h5")
        with RecordedWindowWriter(on, W, H, node, F, 0, 0, meta, tag_clip_index=True) as wr:
            wr.write_batch(batch)
        data, _ = load_recorded_windows(on)
        assert np.array_equal(data["clip_id"], cid)
        assert np.array_equal(data["clip_step"], cstep)

        off = os.path.join(d, "recorded_off.h5")
        with RecordedWindowWriter(off, W, H, node, F, 0, 0, meta) as wr:
            wr.write_batch(batch)
        data_off, _ = load_recorded_windows(off)
        assert "clip_id" not in data_off and "clip_step" not in data_off


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
