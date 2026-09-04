# Copyright 2026 Zachary Olkin. All rights reserved.

"""What :mod:`robot_rl.token_planner.data` claims about the chunks it hands the tokenizer.

The module builds nothing itself -- it runs
:func:`~robot_rl.flow_matching.dataset.build_train_val` and keeps the ``state`` half -- so
what is testable here is not the window construction (that is the FM's, and tested there)
but the four things this module adds on top of it, each of which is silent when wrong:

* it loads the CHEAPEST exteroception array present rather than the one a generator would
  train on, purely to satisfy the FM dataset's mandatory-array check. That is only safe
  because the choice cannot reach the targets, which is a claim about ``build_train_val``
  and is therefore pinned against ``build_train_val``, bit for bit;
* it joins a per-window terrain-family label onto the chunks through ``window_sel``, which
  indexes the FILTERED window array. Joining off the recording instead is off by however
  many windows the distractor and family filters removed -- and every count still lines up,
  so the only symptom is wrong labels. That bug was real;
* the split must be the GENERATOR's split, or an AR-vs-FM comparison is not on the same
  held-out data;
* the cache must refuse a key it does not match, since a cache quietly serving the wrong
  split makes every number in a run wrong with nothing to say so.

The fixture is the FM suite's own ``windowed_recording``
(:mod:`robot_rl.flow_matching.tests.recording_fixtures`, which builds in the four window
invariants the dataset enforces), written to a real ``.h5`` by
:class:`recorded_data_io.RecordedWindowWriter` -- the module under test reads files, and the
writer is where ``history_anchor`` and the opt-in family tag are settled.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("motion_gen")
# ``import robot_rl`` registers every gym env, so the whole package needs IsaacLab -- and so
# does the dataset's invalid-pixel report on the depth arm below.
pytest.importorskip("isaaclab")

from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import build_train_val  # noqa: E402
from robot_rl.flow_matching.tests.recording_fixtures import (  # noqa: E402
    camera_spec,
    windowed_recording,
)
from robot_rl.token_planner.data import (  # noqa: E402
    _recorded_data_io,
    build_chunk_data,
    cache_key,
    load_chunk_cache,
    load_recordings,
    save_chunk_cache,
)

#: The recording's family table, indexed by the per-window ``clip_family`` code (0 =
#: ``unknown``, as ``record_data._clip_family_codes`` writes it).
_FAMILY_NAMES: tuple[str, ...] = (
    "unknown", "family_flat", "family_stairs_up", "family_jump_on", "family_distractor",
)

#: The family the tests drop, and the one the distractor windows belong to.
_DROPPED_FAMILY: str = "family_stairs_up"
_DISTRACTOR_FAMILY: str = "family_distractor"

#: Family assignment cycle. Its length is coprime with both the windows per episode and the
#: per-episode rotation below, which is what makes the tag aperiodic -- see
#: :func:`_family_codes`.
_FAMILY_CYCLE: tuple[str, ...] = (
    "family_flat", "family_flat", "family_stairs_up", "family_jump_on",
    "family_jump_on", "family_stairs_up", "family_flat", "family_jump_on",
)

# Fixture geometry. Small on purpose: nothing here is about capacity, and the recording is
# written to disk and rebuilt several times per test.
_N_EP: int = 4
_WINDOWS_PER_EP: int = 6
_HISTORY_LEN: int = 4
_FUTURE_LEN: int = 8
_NUM_JOINTS: int = 4
_NUM_EE_FRAMES: int = 2
_NUM_SCAN_RAYS: int = 6
#: ``(num_cameras, height, width)`` -- a second, DIFFERENTLY shaped exteroception array, so
#: the extero-source pin has something to be invariant to.
_DEPTH: tuple[int, int, int] = (2, 3, 4)

#: The chunk horizon asked of the builder; ``<= _FUTURE_LEN``, which the recording must hold.
_CHUNK_LEN: int = 6
_SEED: int = 0
_VAL_FRACTION: float = 0.25

#: The three-block window the g1 teacher packs today, stated by the recording's teacher.
_REFERENCE_BLOCKS: list[str] = ["base_pose", "base_vel", "joint_pos"]


def _family_codes(episode_id: np.ndarray, is_distractor: np.ndarray) -> np.ndarray:
    """Per-window family codes: several families, interleaved with the distractor flag.

    Deliberately aperiodic. The codes come from a fixed cycle ROTATED per episode, so no two
    episodes keep the same mix of families and none of them alternates -- a uniform tag, or
    one with period 2 after the filters have taken a family out, would let a join over the
    wrong index space come out right by luck, and this fixture exists precisely to make such
    a join produce wrong labels.

    Args:
        episode_id: ``(W,)`` per-window episode id, ascending in blocks.
        is_distractor: ``(W,)`` per-window distractor flag; those windows are tagged with
            the distractor family, as the recorder does.

    Returns:
        ``(W,)`` uint8 codes indexing :data:`_FAMILY_NAMES`.
    """
    episode_id = np.asarray(episode_id).reshape(-1)
    step = np.zeros(episode_id.shape[0], dtype=np.int64)
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        step[rows] = np.arange(rows.size)
    codes = np.array(
        [
            _FAMILY_NAMES.index(
                _FAMILY_CYCLE[(int(w) + 3 * int(e)) % len(_FAMILY_CYCLE)]
            )
            for w, e in zip(step, episode_id)
        ],
        dtype=np.uint8,
    )
    codes[np.asarray(is_distractor).reshape(-1).astype(bool)] = _FAMILY_NAMES.index(
        _DISTRACTOR_FAMILY
    )
    return codes


def _fixture_recording(seed: int = 0) -> tuple[dict[str, np.ndarray], Any]:
    """A windowed recording carrying BOTH exteroception arrays, plus its metadata.

    The arrays are the FM suite's own fixture, extended with the three opt-in per-window
    tags ``RecordedWindowWriter`` declares for a real recording (``clip_family`` and the
    ``clip_id``/``clip_step``/``env_id`` index). Both ``height_scan`` and ``depth`` are
    present, as the shipped recorder writes them, so one file can build either arm.

    ``history_anchor="plan"`` and ``reference_blocks`` are load-bearing rather than
    decorative: the dataset refuses any other anchor outright, and the block selection is
    what fixes the node width the tokenizer trains on.

    Args:
        seed: RNG seed for the trajectories; the window/episode LAYOUT is identical across
            seeds, so two seeds give two recordings whose episode ids collide.

    Returns:
        ``(data, meta)`` ready for :func:`_write_recording`.
    """
    recording_meta = _recorded_data_io().RecordingMeta
    data, term_names, term_dims, frame_names = windowed_recording(
        n_ep=_N_EP, windows_per_ep=_WINDOWS_PER_EP, history_len=_HISTORY_LEN,
        future_len=_FUTURE_LEN, nj=_NUM_JOINTS, k=_NUM_EE_FRAMES,
        num_scan_rays=_NUM_SCAN_RAYS, depth=_DEPTH, seed=seed,
    )
    num_windows = int(data["history_state"].shape[0])
    data["clip_family"] = _family_codes(data["episode_id"], data["is_distractor"])
    # Where each window came from in the library. Not read by anything under test (the
    # pinned-axis weighting is off), but the writer declares these three together and
    # requires every declared key on every batch.
    data["clip_id"] = (np.arange(num_windows) % 7).astype(np.int64)
    data["clip_step"] = (np.arange(num_windows) % 11).astype(np.int64)
    data["env_id"] = (np.arange(num_windows) % 3).astype(np.int64)
    joint_names = [f"j{i}" for i in range(_NUM_JOINTS)]
    meta = recording_meta(
        checkpoint_path="<fixture>", checkpoint_sha256="0" * 64,
        task_name="G1-teacher-clf-terrain-depth-play", experiment_name="fixture",
        run_name="fixture", runner_class="OnPolicyRunner", rsl_rl_version="0.0.0",
        recorded_at="2026-01-01T00:00:00", stochastic=False, dt=0.02, num_envs=1,
        num_pairs=num_windows, obs_term_names=[], obs_term_dims=[],
        action_joint_names=list(joint_names), state_term_names=list(term_names),
        state_term_dims=list(term_dims), state_quat_format="xyzw",
        state_joint_names=list(joint_names), state_vel_frame="link",
        state_frame_names=list(frame_names),
        ref_source="ego_anchored_clip_future", ref_future_len=_FUTURE_LEN,
        ref_future_term_names=list(term_names), ref_future_term_dims=list(term_dims),
        height_scan_dim=_NUM_SCAN_RAYS, depth_camera_spec=camera_spec(*_DEPTH),
        format="windowed_h5", history_len=_HISTORY_LEN, fm_snap_every=4,
        family_names=list(_FAMILY_NAMES), reference_blocks=list(_REFERENCE_BLOCKS),
        history_anchor="plan",
    )
    return data, meta


def _write_recording(path: Any, data: dict[str, np.ndarray], meta: Any) -> str:
    """Write a fixture recording to a real windowed ``.h5``, as ``record_data.py`` does.

    The module under test reads FILES, so the fixture has to go through the writer -- which
    is also where the per-window family tag is opted into: without ``tag_families`` the
    writer never declares the dataset and silently drops the column, and then no family
    filter can match anything.

    Args:
        path: Destination (anything ``str()`` accepts, e.g. a ``tmp_path`` child).
        data: Window arrays from :func:`_fixture_recording`.
        meta: Its metadata.

    Returns:
        The path written, as a string.
    """
    writer_cls = _recorded_data_io().RecordedWindowWriter
    depth = data["depth"]
    with writer_cls(
        str(path),
        num_windows=int(data["history_state"].shape[0]),
        history_len=int(data["history_state"].shape[1]),
        node_dim=int(data["history_state"].shape[-1]),
        future_len=int(data["ref_future"].shape[1]),
        height_scan_dim=int(data["height_scan"].shape[-1]),
        cmd_dim=int(data["cmd_future"].shape[-1]),
        meta=meta,
        depth_shape=(int(depth.shape[1]), int(depth.shape[2])),
        tag_distractors=True,
        tag_families=True,
        tag_clip_index=True,
        record_history_ref=True,
    ) as writer:
        written = writer.write_batch(data)
    assert written == int(data["history_state"].shape[0])
    return str(path)


def _fm_config(**overrides: Any) -> FlowMatchingConfig:
    """The flow-matching config ``build_chunk_data`` resolves, for a direct comparison build.

    Restated rather than reached for, because these tests exist to check ``build_chunk_data``
    against ``build_train_val`` -- reading the config back off the module would make both
    sides of the comparison the same statement. Only the fields that can move the split or
    the targets are set; the rest stay at their shipped defaults.

    Args:
        **overrides: Fields to change (``extero_source`` is what the pin varies).

    Returns:
        A fresh config; ``build_train_val`` writes resolved values back onto it, so it must
        not be shared between arms.
    """
    settings: dict[str, Any] = dict(
        future_len=_CHUNK_LEN,
        num_history_states=1,
        stride=1,
        val_fraction=_VAL_FRACTION,
        seed=_SEED,
        skip_distractors=True,
        skip_families=[_DROPPED_FAMILY],
        reference_blocks=None,
        extero_source="height_scan",
        extero_norm_mode="metric",
        # One channel and the flat path on BOTH arms: the 5-channel depth default rebuilds
        # the camera pose through motion_gen's FK chain, which this four-joint synthetic
        # robot ("j0".."j3") cannot resolve. The channel widening is the FM's own concern and
        # is tested there; what is under test here is the state half, which no extero knob
        # may touch.
        extero_channels=1,
        extero_encoder="mlp",
        depth_latency_probs=None,
        pinned_axis_loss_weight=1.0,
        pinned_axis_metrics=False,
        cmd_array=None,
        obs_noise=False,
    )
    settings.update(overrides)
    return FlowMatchingConfig(**settings)


def _kept_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    """Which windows the distractor + family filters keep, from the fixture's own tags.

    Computed straight off the recorded columns rather than by calling the dataset's mask
    helpers, so it is an INDEPENDENT statement of the filtered index space the family join
    has to live in -- reusing those helpers would make the oracle and the code under test
    the same answer.

    Args:
        data: The fixture's window arrays.

    Returns:
        ``(W,)`` bool keep-mask.
    """
    families = np.asarray(data["clip_family"]).reshape(-1)
    distractor = np.asarray(data["is_distractor"]).reshape(-1).astype(bool)
    return ~distractor & (families != _FAMILY_NAMES.index(_DROPPED_FAMILY))


def _counts(values: Any) -> Counter:
    """Multiset of int codes, for comparing labels against an independently derived set."""
    return Counter(int(v) for v in np.asarray(values).reshape(-1).tolist())


def test_the_extero_source_does_not_move_the_state_targets(tmp_path) -> None:
    """The cheapest-array shortcut is only safe if the choice cannot reach the targets.

    ``build_chunk_data`` loads ``height_scan`` over ``depth`` -- 176 floats a window against
    7800 -- because a chunk-local autoencoder conditions on nothing and the FM dataset merely
    REQUIRES an exteroception array. That is a 6.2 GB -> 141 MB saving on the shipped
    recording and it is indefensible if the source can perturb ``state``, since a tokenizer
    trained on the cheap array would then be reconstructing something the generator never
    regresses. So: same recording, same everything, one field different, and the targets must
    be equal bit for bit rather than merely close.
    """
    data, meta = _fixture_recording()
    path = _write_recording(tmp_path / "both_extero.h5", data, meta)
    loaded, loaded_meta = load_recordings([path])
    assert loaded.get("height_scan") is not None
    assert loaded.get("depth") is not None

    scan_train, scan_val = build_train_val(
        dict(loaded), loaded_meta, _fm_config(extero_source="height_scan")
    )
    depth_train, depth_val = build_train_val(
        dict(loaded), loaded_meta, _fm_config(extero_source="depth")
    )
    assert scan_val is not None and depth_val is not None

    # The pin is only meaningful if the two arms really did read different sensors...
    assert scan_train.extero.shape != depth_train.extero.shape
    assert scan_train.extero_sensor_count != depth_train.extero_sensor_count
    # ...and if the targets are a real signal rather than a tensor two builds could match by
    # both being degenerate.
    for target in (scan_train.state, scan_val.state):
        assert target.numel() > 0 and bool(torch.isfinite(target).all())
        assert float(target.std()) > 0.1

    assert torch.equal(scan_train.state, depth_train.state)
    assert torch.equal(scan_val.state, depth_val.state)
    # ...and the same windows, in the same order, so "equal" is not two different subsets
    # that happen to share a tensor shape.
    np.testing.assert_array_equal(scan_train.window_sel, depth_train.window_sel)
    np.testing.assert_array_equal(scan_val.window_sel, depth_val.window_sel)
    # The state normalizer travels into the checkpoint so reconstruction error can be
    # reported in physical units; it must not move either.
    scan_norm, depth_norm = (
        scan_train.state_normalizer.state_dict(),
        depth_train.state_normalizer.state_dict(),
    )
    assert scan_norm.keys() == depth_norm.keys()
    for name, value in scan_norm.items():
        assert torch.equal(value, depth_norm[name]), name


def test_the_family_join_is_in_the_filtered_index_space(tmp_path) -> None:
    """``window_sel`` indexes the FILTERED windows, and joining off the recording is silent.

    The dataset applies the distractor and family masks BEFORE splitting, so a label taken
    from the recording at ``window_sel[i]`` is off by however many windows the filters
    removed. Nothing about that is visible from the shapes: the label tensor is the right
    length, every count adds up, and the codes are all legal. It was caught only because a
    dropped family turned up among the labels of a run that drops it.
    """
    data, meta = _fixture_recording()
    path = _write_recording(tmp_path / "families.h5", data, meta)
    chunks = build_chunk_data(
        [path], chunk_len=_CHUNK_LEN, seed=_SEED, val_fraction=_VAL_FRACTION,
        skip_distractors=True, skip_families=[_DROPPED_FAMILY],
    )
    dropped = _FAMILY_NAMES.index(_DROPPED_FAMILY)
    distractor = _FAMILY_NAMES.index(_DISTRACTOR_FAMILY)
    keep = _kept_mask(data)
    assert chunks.family_names == list(_FAMILY_NAMES)

    # One label per chunk, on both splits.
    assert chunks.train_family.shape[0] == chunks.train_nodes.shape[0]
    assert chunks.val_family.shape[0] == chunks.val_nodes.shape[0]
    assert chunks.train_nodes.shape[0] > 0 and chunks.val_nodes.shape[0] > 0
    labels = torch.cat([chunks.train_family, chunks.val_family])
    assert int(labels.numel()) == int(keep.sum())

    # Nothing the build dropped may appear as a label.
    assert dropped not in set(labels.tolist())
    assert distractor not in set(labels.tolist())
    # And the labels ARE the kept windows' families, not merely legal codes.
    assert _counts(labels) == _counts(data["clip_family"][keep])

    # Per split, too -- which is the half a shifted join breaks even when the totals survive,
    # because the fixture gives every episode a different mix of families.
    episodes = np.asarray(data["episode_id"])[keep]
    families = np.asarray(data["clip_family"])[keep]
    in_val = np.isin(episodes, list(chunks.val_episode_ids))
    assert _counts(chunks.val_family) == _counts(families[in_val])
    assert _counts(chunks.train_family) == _counts(families[~in_val])

    # The fixture has resolving power: the filters removed windows (so the two index spaces
    # really do differ), and the recording's own first ``n_kept`` windows -- the slice a join
    # over the unfiltered array reaches into -- still contain both filtered families, so a
    # wrong join cannot come out right by accident.
    assert int(keep.sum()) < keep.size
    naive = set(np.asarray(data["clip_family"])[: int(keep.sum())].tolist())
    assert dropped in naive and distractor in naive


def test_the_split_is_by_episode_and_matches_the_generators(tmp_path) -> None:
    """The tokenizer's held-out set must be the generator's, or no AR-vs-FM number compares.

    The whole reason this module delegates to ``build_train_val`` instead of cutting its own
    windows is that the split then cannot drift. So it is checked against that function
    directly, and separately checked to be a split by EPISODE -- windows of one episode are
    near-duplicates of each other, so an episode straddling the two splits leaks.
    """
    data, meta = _fixture_recording()
    path = _write_recording(tmp_path / "split.h5", data, meta)
    chunks = build_chunk_data(
        [path], chunk_len=_CHUNK_LEN, seed=_SEED, val_fraction=_VAL_FRACTION,
        skip_distractors=True, skip_families=[_DROPPED_FAMILY],
    )
    loaded, loaded_meta = load_recordings([path])
    train_ds, val_ds = build_train_val(dict(loaded), loaded_meta, _fm_config())
    assert val_ds is not None

    assert chunks.val_episode_ids, "an empty val split would not test the claim"
    assert chunks.val_episode_ids == train_ds.val_episode_ids
    assert chunks.train_nodes.shape[0] == len(train_ds)
    assert chunks.val_nodes.shape[0] == len(val_ds)

    keep = _kept_mask(data)
    episodes = np.asarray(data["episode_id"])[keep]
    train_eps = {int(e) for e in episodes[train_ds.window_sel]}
    val_eps = {int(e) for e in episodes[val_ds.window_sel]}
    assert not (train_eps & val_eps)
    assert val_eps == set(chunks.val_episode_ids)
    assert train_eps | val_eps == {int(e) for e in np.unique(episodes)}


def test_the_chunk_cache_round_trips_and_refuses_a_mismatch(tmp_path) -> None:
    """A cache that rebuilt, or served, on a key mismatch would make a run's numbers lie.

    The build reads and decompresses tens of GB while the tokenizer's own sweeps change none
    of it, so the cache is what makes the ablations cheap -- and that is exactly why it has to
    refuse loudly: one silently wrong split and every number in the report is about data the
    report does not name.
    """
    data, meta = _fixture_recording()
    path = _write_recording(tmp_path / "cache_source.h5", data, meta)
    chunks = build_chunk_data(
        [path], chunk_len=_CHUNK_LEN, seed=_SEED, val_fraction=_VAL_FRACTION,
        skip_distractors=True, skip_families=[_DROPPED_FAMILY],
    )
    key = cache_key(chunks.provenance, chunks.chunk_len, chunks.reference_blocks)
    cache = str(tmp_path / "nested" / "chunks.pt")
    save_chunk_cache(cache, chunks)

    back = load_chunk_cache(cache, key)
    assert back.val_nodes.numel() > 0, "a degenerate split would not exercise the val half"
    for name in ("train_nodes", "val_nodes", "train_family", "val_family"):
        assert torch.equal(getattr(back, name), getattr(chunks, name)), name
    assert back.state_normalizer_state.keys() == chunks.state_normalizer_state.keys()
    for name, value in chunks.state_normalizer_state.items():
        assert torch.equal(back.state_normalizer_state[name], value), name
    assert back.family_names == chunks.family_names
    assert back.reference_blocks == chunks.reference_blocks
    assert back.val_episode_ids == chunks.val_episode_ids
    assert (back.chunk_len, back.node_dim, back.num_joints, back.num_ee_frames) == (
        chunks.chunk_len, chunks.node_dim, chunks.num_joints, chunks.num_ee_frames
    )

    with pytest.raises(ValueError) as excinfo:
        load_chunk_cache(cache, {**key, "seed": _SEED + 7})
    message = str(excinfo.value)
    # It must name the field that differs, and only that one -- a refusal that does not say
    # what disagreed sends the reader to delete a cache that was fine.
    assert f"seed: cached {_SEED} != requested {_SEED + 7}" in message
    assert "chunk_len:" not in message and "skip_families:" not in message


def test_a_second_recording_offsets_the_episode_ids(tmp_path) -> None:
    """Two recordings both number their episodes from 0, and the split is BY episode.

    Concatenating them without an offset puts one id in both splits, which is a leak that
    shows up as nothing at all -- the val loss just reads better than it should.
    """
    first, first_meta = _fixture_recording(seed=0)
    second, second_meta = _fixture_recording(seed=1)
    # The premise: two independently recorded files really do collide on episode id.
    np.testing.assert_array_equal(first["episode_id"], second["episode_id"])
    assert not np.array_equal(first["ref_future"], second["ref_future"])

    path_a = _write_recording(tmp_path / "first.h5", first, first_meta)
    path_b = _write_recording(tmp_path / "second.h5", second, second_meta)
    data, meta = load_recordings([path_a, path_b])

    n_first = int(first["episode_id"].shape[0])
    ids = np.asarray(data["episode_id"])
    assert ids.shape[0] == n_first + int(second["episode_id"].shape[0])
    np.testing.assert_array_equal(ids[:n_first], first["episode_id"])
    assert not set(ids[:n_first].tolist()) & set(ids[n_first:].tolist())
    assert int(ids[n_first:].min()) > int(ids[:n_first].max())
    # A pure shift: the second file's own episode STRUCTURE has to survive, or its windows
    # get regrouped and the split stops being a split by episode.
    np.testing.assert_array_equal(
        ids[n_first:] - (int(ids[:n_first].max()) + 1), second["episode_id"]
    )
    # Every other array is concatenated in the same order, and the metadata is the FIRST
    # file's -- which is what the FM does, so the two stages agree on what they read.
    np.testing.assert_array_equal(
        data["ref_future"][n_first:], second["ref_future"]
    )
    assert meta.family_names == list(first_meta.family_names)
