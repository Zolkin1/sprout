# Copyright 2026 Zachary Olkin. All rights reserved.

"""Serialization for recorded (observation, action) datasets.

This module is intentionally lightweight: it only depends on ``numpy`` and the
standard library so the downstream supervised-learning script can
``from recorded_data_io import load_recorded_data, RecordingMeta`` without
pulling in Isaac Sim. The producer side lives in ``record_data.py``.

Datasets are stored as a single compressed ``.npz`` file holding the per-pair
arrays plus the provenance/layout metadata needed to interpret them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields

import numpy as np


@dataclass
class RecordingMeta:
    """Provenance and layout metadata for a recorded dataset.

    Attributes:
        checkpoint_path: Absolute path to the checkpoint that produced the data.
        checkpoint_sha256: SHA-256 of the checkpoint file (unambiguous policy identity).
        task_name: Gym task name the rollout was collected on.
        experiment_name: RSL-RL experiment name from the agent config.
        run_name: Name of the run directory the checkpoint lives in.
        runner_class: ``OnPolicyRunner`` or ``DistillationRunner``.
        rsl_rl_version: Installed ``rsl-rl-lib`` version at recording time.
        recorded_at: ISO-8601 timestamp of when the dataset was recorded.
        stochastic: Whether actions were sampled stochastically (vs deterministic).
        dt: Environment step period in seconds.
        num_envs: Number of parallel environments used during collection.
        num_pairs: Total number of (obs, action) pairs stored.
        obs_term_names: Ordered names of the policy observation terms.
        obs_term_dims: Per-term flattened dimensions; sums to the obs width.
        action_joint_names: Ordered joint names the action vector maps onto.
        state_term_names: Ordered names of the raw robot-state terms.
        state_term_dims: Per-term dimensions of the state vector; sums to its width.
        state_quat_format: Component order of ``root_quat_w`` in the state (e.g. "xyzw").
        state_joint_names: Ordered joint names for the state's joint_pos/joint_vel terms.
        state_vel_frame: Frame/accessor used for the recorded root velocities (e.g. "link").
        state_frame_names: Ordered extra body frames appended to the state vector beyond
            root + joints (each contributes ``pos_w(3), quat_w(4), lin_vel_w(3), ang_vel_w(3)``).
            Empty when only root + joints were recorded.
        ref_source: Kind of recorded reference. "" (none/legacy), "reference" (per-step
            tracked ``ref_state``), or "ego_anchored_clip_future" (the robot-state-conditioned
            FM target: a per-step F-node future of the clip ego-anchored onto the robot, in
            ``ref_future``, with snap events flagged by ``reanchor_active``).
        ref_future_len: Number of future nodes ``F`` stored per step in ``ref_future`` (0 if absent).
        ref_future_term_names: Per-node layout names of ``ref_future`` (== ``state_term_names``).
        ref_future_term_dims: Per-node layout dims of ``ref_future`` (== ``state_term_dims``).
        height_scan_dim: Number of rays ``K`` in the per-step ``height_scan`` array (0 if
            absent). The height scan is the FM's height-conditioning token (per-ray ground
            clearance from the ``height_scanner`` sensor), captured when the recording task
            has one active (e.g. ``G1-teacher-clf-terrain-play``).
        depth_camera_spec: Deployment contract of the cameras that produced the ``depth``
            array -- mount, intrinsics, range and normalisation, one entry per camera in
            token order (``mdp.depth_camera_specs``). ``None`` when the recording task had
            no depth cameras. Carried through the FM checkpoint into ``fm.onnx`` so a
            deployed generator states which cameras it was trained on and can be checked
            against the tracker driving the same physical pair.
        content_sha256: Content hash of the recorded arrays (see :func:`fingerprint_arrays`),
            set after collection. Compression-robust dataset identity; empty for older files.
        format: Storage layout. "per_step_npz" (the legacy every-step ``.npz``) or
            "windowed_h5" (pre-formed FM snap windows in a streaming ``.h5``; see
            :class:`RecordedWindowWriter`). Defaulted so legacy ``.npz`` files still load.
        history_len: For "windowed_h5", the number of history-state rows ``H`` stored per
            window (must be >= the trainable ``num_history_states``). 0 for per-step files.
    """

    checkpoint_path: str
    checkpoint_sha256: str
    task_name: str
    experiment_name: str
    run_name: str
    runner_class: str
    rsl_rl_version: str
    recorded_at: str
    stochastic: bool
    dt: float
    num_envs: int
    num_pairs: int
    obs_term_names: list[str]
    obs_term_dims: list[int]
    action_joint_names: list[str]
    state_term_names: list[str]
    state_term_dims: list[int]
    state_quat_format: str
    state_joint_names: list[str]
    state_vel_frame: str
    state_frame_names: list[str] = field(default_factory=list)
    ref_source: str = ""
    ref_future_len: int = 0
    ref_future_term_names: list[str] = field(default_factory=list)
    ref_future_term_dims: list[int] = field(default_factory=list)
    height_scan_dim: int = 0
    depth_camera_spec: list[dict] | None = None
    content_sha256: str = ""
    format: str = "per_step_npz"
    history_len: int = 0
    fm_snap_every: int = 0
    """Control steps between snap (re-plan) events in this recording, from the command's
    ``fm_snap_every``. The FM tracker's ``fm_update_every`` must equal it -- the horizon is
    sized from the period and the targets were cut at that cadence. 0 = not recorded."""


def fingerprint_arrays(data: dict[str, np.ndarray]) -> str:
    """Content hash over the recorded arrays (independent of npz compression).

    Hashes each present array's name + dtype + shape + raw bytes in sorted key
    order, so identical recorded content yields the same digest regardless of how
    the ``.npz`` was written. This identifies a dataset by *content* (complementing
    :func:`sha256_file`, which hashes the exact file). ``None`` entries are skipped.

    Args:
        data: The per-pair arrays passed to :func:`save_recorded_data`.

    Returns:
        The hex-encoded SHA-256 content digest.
    """
    h = hashlib.sha256()
    for key in sorted(data):
        arr = data[key]
        if arr is None:
            continue
        a = np.ascontiguousarray(arr)
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, reading in ``chunk_size`` blocks.

    Args:
        path: File to hash.
        chunk_size: Read block size in bytes (default 1 MiB).

    Returns:
        The hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_output_path(out_dir: str, prefix: str = "recorded", ext: str = ".npz") -> str:
    """Build a timestamped output path that does not collide with existing files.

    A numeric suffix (``_1``, ``_2``, ...) is appended if a file with the
    timestamped name already exists, so back-to-back runs within the same second
    never overwrite previous data.

    Args:
        out_dir: Directory the file will be written to (created if missing).
        prefix: Filename prefix.
        ext: File extension, including the leading dot.

    Returns:
        An absolute, currently-nonexistent file path inside ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(out_dir, f"{prefix}_{stamp}{ext}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(out_dir, f"{prefix}_{stamp}_{counter}{ext}")
        counter += 1
    return os.path.abspath(candidate)


def save_recorded_data(path: str, data: dict[str, np.ndarray], meta: RecordingMeta) -> None:
    """Persist recorded per-pair arrays and metadata to a compressed ``.npz`` file.

    Only string/number arrays are stored so the file loads with
    ``allow_pickle=False``.

    Args:
        path: Destination ``.npz`` path.
        data: Per-pair arrays (``obs``, ``action``, ``state``, ``terminated``,
            ``done``, ``time_out``, ``episode_id``, ``step_in_episode``). May also
            include ``ref_state`` (desired reference in the ``state`` layout); it
            is written only when present.
        meta: Provenance/layout metadata to embed alongside the arrays.
    """
    # ``ref_state`` shares the ``state`` layout (meta.state_term_*); written only
    # when the recording task exposed a trajectory reference.
    optional_arrays = {}
    if data.get("ref_state") is not None:
        optional_arrays["ref_state"] = data["ref_state"].astype(np.float32)
    # Per-step commanded velocity (vx, vy, wz) -- the upstream command that drove the
    # reference. Written only when the recording task exposed a command velocity;
    # used by the flow-matching dataset for future-horizon command conditioning.
    if data.get("target_vel") is not None:
        optional_arrays["target_vel"] = data["target_vel"].astype(np.float32)
    # Per-step perturbation flag (1 while action-noise/push was active for that env);
    # written only for recovery recordings (record_data.py --perturb). Consumed by the
    # flow-matching dataset's recovery-aware window filter.
    if data.get("perturb_active") is not None:
        optional_arrays["perturb_active"] = data["perturb_active"].astype(np.uint8)
    # Ego-anchored clip FUTURE target (robot-state-conditioned FM): per step an F-node
    # window ``[num_pairs, F, node_dim]`` (node layout == state layout), the future of
    # the clip ego-anchored onto the robot. ``cmd_future`` is the aligned future command
    # ``[num_pairs, F, 3]``; ``reanchor_active`` flags the snap (re-anchor) steps the FM
    # trains on. Written only when the recording task exposed the look-ahead accessors.
    if data.get("ref_future") is not None:
        optional_arrays["ref_future"] = data["ref_future"].astype(np.float32)
    if data.get("cmd_future") is not None:
        optional_arrays["cmd_future"] = data["cmd_future"].astype(np.float32)
    if data.get("reanchor_active") is not None:
        optional_arrays["reanchor_active"] = data["reanchor_active"].astype(np.uint8)
    # Per-step height scan ``[num_pairs, K]`` (per-ray ground clearance) -- the FM's
    # height-conditioning token. Written only when a height_scanner sensor was active.
    if data.get("height_scan") is not None:
        optional_arrays["height_scan"] = data["height_scan"].astype(np.float32)
    # Per-step depth ``[num_pairs, T, H*W]`` (normalised, one token row per camera) --
    # the FM's other exteroception source. Written only when the recording task had
    # depth observation terms; the geometry that produced it is in
    # ``meta.depth_camera_spec``.
    if data.get("depth") is not None:
        optional_arrays["depth"] = data["depth"].astype(np.float32)
    np.savez_compressed(
        path,
        # per-pair arrays
        obs=data["obs"].astype(np.float32),
        action=data["action"].astype(np.float32),
        state=data["state"].astype(np.float32),
        **optional_arrays,
        terminated=data["terminated"].astype(np.uint8),
        done=data["done"].astype(np.uint8),
        time_out=data["time_out"].astype(np.uint8),
        episode_id=data["episode_id"].astype(np.int64),
        step_in_episode=data["step_in_episode"].astype(np.int64),
        # provenance metadata
        checkpoint_path=np.array(meta.checkpoint_path),
        checkpoint_sha256=np.array(meta.checkpoint_sha256),
        task_name=np.array(meta.task_name),
        experiment_name=np.array(meta.experiment_name),
        run_name=np.array(meta.run_name),
        runner_class=np.array(meta.runner_class),
        rsl_rl_version=np.array(meta.rsl_rl_version),
        recorded_at=np.array(meta.recorded_at),
        stochastic=np.bool_(meta.stochastic),
        dt=np.float64(meta.dt),
        num_envs=np.int64(meta.num_envs),
        num_pairs=np.int64(meta.num_pairs),
        # layout metadata (obs/action ordering)
        obs_term_names=np.array(meta.obs_term_names),
        obs_term_dims=np.array(meta.obs_term_dims, dtype=np.int64),
        action_joint_names=np.array(meta.action_joint_names),
        # layout metadata (raw state ordering)
        state_term_names=np.array(meta.state_term_names),
        state_term_dims=np.array(meta.state_term_dims, dtype=np.int64),
        state_quat_format=np.array(meta.state_quat_format),
        state_joint_names=np.array(meta.state_joint_names),
        state_vel_frame=np.array(meta.state_vel_frame),
        state_frame_names=np.array(meta.state_frame_names),
        ref_source=np.array(meta.ref_source),
        ref_future_len=np.int64(meta.ref_future_len),
        ref_future_term_names=np.array(meta.ref_future_term_names),
        ref_future_term_dims=np.array(meta.ref_future_term_dims, dtype=np.int64),
        height_scan_dim=np.int64(meta.height_scan_dim),
        # Nested structure, so JSON rather than an array -- the file stays
        # ``allow_pickle=False`` loadable.
        depth_camera_spec=np.array(json.dumps(meta.depth_camera_spec)),
        content_sha256=np.array(meta.content_sha256),
    )


def load_recorded_data(path: str) -> tuple[dict[str, np.ndarray], RecordingMeta]:
    """Load a dataset produced by :func:`save_recorded_data`.

    Args:
        path: Path to a ``.npz`` file written by :func:`save_recorded_data`.

    Returns:
        A ``(data, meta)`` tuple where ``data`` maps array names to numpy arrays
        and ``meta`` is the reconstructed :class:`RecordingMeta`.
    """
    z = np.load(path, allow_pickle=False)
    data = {
        "obs": z["obs"],
        "action": z["action"],
        "state": z["state"],
        # Present only for datasets recorded with a trajectory reference; older
        # files lack it, so default to None.
        "ref_state": z["ref_state"] if "ref_state" in z.files else None,
        # Present only when recorded on a clip task exposing a command velocity.
        "target_vel": z["target_vel"] if "target_vel" in z.files else None,
        # Present only for recovery recordings (record_data.py --perturb).
        "perturb_active": z["perturb_active"] if "perturb_active" in z.files else None,
        # Present only for robot-state-conditioned FM recordings (ref_source=
        # "ego_anchored_clip_future"): the per-step ego-anchored clip FUTURE target,
        # the aligned future command, and the snap-event flag.
        "ref_future": z["ref_future"] if "ref_future" in z.files else None,
        "cmd_future": z["cmd_future"] if "cmd_future" in z.files else None,
        "reanchor_active": z["reanchor_active"] if "reanchor_active" in z.files else None,
        # Present only when a height_scanner sensor was active during recording.
        "height_scan": z["height_scan"] if "height_scan" in z.files else None,
        # Present only when the recording task had depth observation terms.
        "depth": z["depth"] if "depth" in z.files else None,
        "terminated": z["terminated"],
        "done": z["done"],
        "time_out": z["time_out"],
        "episode_id": z["episode_id"],
        "step_in_episode": z["step_in_episode"],
    }
    meta = RecordingMeta(
        checkpoint_path=str(z["checkpoint_path"]),
        checkpoint_sha256=str(z["checkpoint_sha256"]),
        task_name=str(z["task_name"]),
        experiment_name=str(z["experiment_name"]),
        run_name=str(z["run_name"]),
        runner_class=str(z["runner_class"]),
        rsl_rl_version=str(z["rsl_rl_version"]),
        recorded_at=str(z["recorded_at"]),
        stochastic=bool(z["stochastic"]),
        dt=float(z["dt"]),
        num_envs=int(z["num_envs"]),
        num_pairs=int(z["num_pairs"]),
        obs_term_names=[str(x) for x in z["obs_term_names"]],
        obs_term_dims=[int(x) for x in z["obs_term_dims"]],
        action_joint_names=[str(x) for x in z["action_joint_names"]],
        state_term_names=[str(x) for x in z["state_term_names"]],
        state_term_dims=[int(x) for x in z["state_term_dims"]],
        state_quat_format=str(z["state_quat_format"]),
        state_joint_names=[str(x) for x in z["state_joint_names"]],
        state_vel_frame=str(z["state_vel_frame"]),
        state_frame_names=(
            [str(x) for x in z["state_frame_names"]] if "state_frame_names" in z.files else []
        ),
        ref_source=str(z["ref_source"]) if "ref_source" in z.files else "",
        ref_future_len=int(z["ref_future_len"]) if "ref_future_len" in z.files else 0,
        ref_future_term_names=(
            [str(x) for x in z["ref_future_term_names"]] if "ref_future_term_names" in z.files else []
        ),
        ref_future_term_dims=(
            [int(x) for x in z["ref_future_term_dims"]] if "ref_future_term_dims" in z.files else []
        ),
        height_scan_dim=int(z["height_scan_dim"]) if "height_scan_dim" in z.files else 0,
        depth_camera_spec=(
            json.loads(str(z["depth_camera_spec"])) if "depth_camera_spec" in z.files else None
        ),
        content_sha256=str(z["content_sha256"]) if "content_sha256" in z.files else "",
    )
    return data, meta


# ---------------------------------------------------------------------------
# Windowed (snap-only) recordings -- streaming HDF5
# ---------------------------------------------------------------------------
#
# The FM dataset only ever trains on "snap" windows (flow_matching/dataset.py
# snap_windows). Recording every step and storing the F-node ``ref_future`` at EVERY
# step -- then discarding ~92% of it -- is what makes the per-step ``.npz`` path OOM.
# The windowed format instead stores ONE pre-formed snap window per row (every stored
# byte is used) and streams to a chunked, compressed HDF5 with bounded RAM (np.savez
# cannot append). Layout per window (raw world-frame; the dataset applies relative_state
# + normalization at build time):
#   history_state [W, H, node]  the H most-recent MEASURED state rows (last row == the
#                               snap; its root pose is the ego-anchor). H >= trainable N.
#   history_ref   [W, H, node]  the reference APPLIED at each of those H steps, same
#                               layout. Feeds ``history_source="reference_partial"``;
#                               stored unconditionally so that stays a train-time A/B.
#   ref_future    [W, F, node]  the ego-anchored clip future at the snap.
#   height_scan   [W, K]        the snap-row height scan (optional).
#   cmd_future    [W, F, cmd]   the snap-row future command (optional; dataset strides it).
#   episode_id    [W]           the snap-row episode id (leakage-free train/val split).

_WINDOW_DTYPES: dict[str, str] = {
    "history_state": "float32",
    "history_ref": "float32",
    "ref_future": "float32",
    "height_scan": "float32",
    "depth": "float32",
    "cmd_future": "float32",
    "episode_id": "int64",
    # Per-window: 1 if the window's clip belongs to the DISTRACTOR family (a box the robot walks
    # around rather than onto). Recorded unconditionally so training can drop those windows with
    # ``train.py --skip_distractors`` without a re-record. Absent in older recordings.
    "is_distractor": "uint8",
}


def is_windowed_recording(path: str) -> bool:
    """True if ``path`` is a windowed HDF5 recording (by extension ``.h5``/``.hdf5``)."""
    return os.path.splitext(str(path))[1].lower() in (".h5", ".hdf5")


def fingerprint_windows(data: dict[str, np.ndarray]) -> str:
    """Content hash over the windowed arrays (see :func:`fingerprint_arrays`)."""
    return fingerprint_arrays(data)


def _window_chunk(shape: tuple[int, ...], block: int = 256) -> tuple[int, ...]:
    """Chunk shape for a windowed dataset: block windows along the first axis.

    Blocking many windows per chunk keeps the chunk count small (millions of
    single-window chunks bloat the HDF5 B-tree), while a ~1-4 MB chunk stays
    efficient for the sequential full-load reads the FM dataset does.
    """
    return (min(shape[0], block),) + shape[1:]


class RecordedWindowWriter:
    """Streaming HDF5 writer for pre-formed FM snap windows.

    Datasets are pre-sized to ``num_windows`` and each collected batch is written into
    its slice, so recording millions of windows never holds more than one batch in RAM
    (unlike ``np.savez_compressed``). ``height_scan``/``cmd_future`` datasets are created
    only when their dim is > 0. :class:`RecordingMeta` is stored as JSON-encoded root
    attrs; :meth:`close` records ``num_windows_written`` and :func:`load_recorded_windows`
    slices to it (so an under-filled file trims cleanly).

    Args:
        path: Output ``.h5`` path.
        num_windows: Total windows the file will hold (upper bound).
        history_len: ``H``, history-state rows per window.
        node_dim: Per-row raw state width (== ref_future node width).
        future_len: ``F``, future nodes per window.
        height_dim: ``K`` rays (0 to omit the ``height_scan`` dataset).
        depth_shape: ``(num_cameras, H*W)`` for the depth dataset, or ``None`` to omit it.
            Recorded alongside the height scan so one recording can train either FM.
        cmd_dim: Future-command width (0 to omit the ``cmd_future`` dataset).
        meta: Provenance/layout metadata (stored as JSON attrs).
        compression: h5py filter ("lzf" fast; "gzip" smaller/slower; None off).
        record_history_ref: Also store the applied-reference history ring (``history_ref``),
            which ``history_source="reference_partial"`` trains on.
    """

    def __init__(
        self, path: str, num_windows: int, history_len: int, node_dim: int,
        future_len: int, height_dim: int, cmd_dim: int, meta: RecordingMeta,
        compression: str | None = "lzf", depth_shape: tuple[int, int] | None = None,
        tag_distractors: bool = False, record_history_ref: bool = False,
    ) -> None:
        import h5py

        self.path = str(path)
        self.num_windows = int(num_windows)
        shapes: dict[str, tuple[int, ...]] = {
            "history_state": (num_windows, history_len, node_dim),
            "ref_future": (num_windows, future_len, node_dim),
            "episode_id": (num_windows,),
        }
        # The applied-reference history ring, for history_source="reference_partial".
        # ~H/F extra bytes per window (12% at H=8, F=65) buys a train-time A/B instead of
        # a second multi-hour recording, so it is not made optional.
        if record_history_ref:
            shapes["history_ref"] = (num_windows, history_len, node_dim)
        # Opt-in per-window distractor tag: declared only when the caller will supply it every
        # batch (``write_batch`` requires every declared key), so older writers and the window
        # tests stay valid. ``load_recorded_windows`` treats it as optional on read.
        if tag_distractors:
            shapes["is_distractor"] = (num_windows,)
        if height_dim > 0:
            shapes["height_scan"] = (num_windows, height_dim)
        if depth_shape is not None:
            shapes["depth"] = (num_windows, int(depth_shape[0]), int(depth_shape[1]))
        if cmd_dim > 0:
            shapes["cmd_future"] = (num_windows, future_len, cmd_dim)
        self._keys = list(shapes)
        self._f = h5py.File(self.path, "w")
        for key, shape in shapes.items():
            self._f.create_dataset(
                key, shape=shape, dtype=_WINDOW_DTYPES[key],
                chunks=_window_chunk(shape), compression=compression,
            )
        # RecordingMeta as JSON attrs (reconstructed by load_recorded_windows).
        for k, v in asdict(meta).items():
            self._f.attrs[k] = json.dumps(v)
        self._f.attrs["num_windows"] = int(num_windows)
        self._written = 0

    def write_batch(self, batch: dict[str, np.ndarray]) -> int:
        """Append a batch of windows; returns the number actually written.

        Clamps to the remaining capacity. ``batch`` maps each present dataset key to a
        ``[n, ...]`` array; all keys must share the leading ``n``.
        """
        n = int(batch["history_state"].shape[0])
        start = self._written
        take = min(n, self.num_windows - start)
        if take <= 0:
            return 0
        end = start + take
        for key in self._keys:
            arr = np.asarray(batch[key])[:take]
            self._f[key][start:end] = arr.astype(_WINDOW_DTYPES[key], copy=False)
        self._written = end
        return take

    def set_attr(self, name: str, value) -> None:
        """Store an extra JSON-encoded root attribute (e.g. the content fingerprint)."""
        self._f.attrs[name] = json.dumps(value)

    @property
    def num_written(self) -> int:
        """Number of windows written so far."""
        return self._written

    def close(self) -> None:
        self._f.attrs["num_windows_written"] = self._written
        self._f.close()

    def __enter__(self) -> "RecordedWindowWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_recorded_windows(path: str) -> tuple[dict[str, np.ndarray], RecordingMeta]:
    """Load a windowed recording written by :class:`RecordedWindowWriter`.

    Returns ``(data, meta)`` where ``data`` holds the pre-formed window arrays
    (``history_state``, ``ref_future``, and -- when present -- ``height_scan``,
    ``cmd_future``, plus ``episode_id``), each trimmed to ``num_windows_written``.
    ``meta`` is reconstructed from the JSON root attrs.
    """
    import h5py

    data: dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        for key in _WINDOW_DTYPES:
            if key in f:
                data[key] = f[key][:n]
        raw_attrs = {k: json.loads(v) for k, v in f.attrs.items()
                     if k not in ("num_windows", "num_windows_written")}
    field_names = {fld.name for fld in fields(RecordingMeta)}
    meta = RecordingMeta(**{k: v for k, v in raw_attrs.items() if k in field_names})
    return data, meta
