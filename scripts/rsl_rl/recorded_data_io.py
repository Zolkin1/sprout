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
import os
from dataclasses import dataclass, field

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
        content_sha256: Content hash of the recorded arrays (see :func:`fingerprint_arrays`),
            set after collection. Compression-robust dataset identity; empty for older files.
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
    content_sha256: str = ""


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
        content_sha256=str(z["content_sha256"]) if "content_sha256" in z.files else "",
    )
    return data, meta
