# Copyright 2026 Zachary Olkin. All rights reserved.

"""Serialization for flow-matching visualization recordings.

This module only depends on ``numpy`` and the standard library so the viser
viewer (``scripts/rsl_rl/view_debug_episodes.py``) can load recordings without
pulling in Isaac Sim.

A recording is a single compressed ``.npz`` file holding, for ``T`` recorded
steps of ``N`` environments with ``nj`` joints and a future horizon ``F``:

Per-step arrays (frame ``t`` is the state snapshot taken BEFORE ``env.step``):
    - ``root_pos``        (T, N, 3)    float32, world frame
    - ``root_quat_xyzw``  (T, N, 4)    float32
    - ``joint_pos``       (T, N, nj)   float32, in ``joint_names`` order
    - ``vel_cmd``         (T, N, C)    float32, velocity_commands obs slice
    - ``done``            (T, N)       uint8, dones from the step taken at t
    - ``scan_hits``       (T, N, R, 3) float32, world-frame ray hits (optional;
      omitted entirely when the task has no height scanner; may contain inf)
    - ``pred_index``      (T, N)       int32, index into the prediction arrays;
      -1 means no valid prediction. Carries the latest prediction forward
      between inference steps and resets to -1 on an env reset.

Prediction arrays (flat, absolute world frame; ``P`` predictions total):
    - ``pred_root_pos``        (P, F, 3)  float32
    - ``pred_root_quat_xyzw``  (P, F, 4)  float32
    - ``pred_joint_pos``       (P, F, nj) float32

Terrain (recorded once):
    - ``terrain_vertices``       (V, 3)  float32
    - ``terrain_faces``          (Fc, 3) int32
    - ``terrain_vertex_colors``  (V, 4)  uint8 (optional)
"""

from __future__ import annotations

import datetime
import glob
import os
from dataclasses import dataclass

import numpy as np

VIZ_DIR_DEFAULT = os.path.join("logs", "flow_matching", "viz")

# Per-step / prediction array keys and their dtypes; scan_hits and
# terrain_vertex_colors are optional.
_ARRAY_DTYPES: dict[str, type] = {
    "root_pos": np.float32,
    "root_quat_xyzw": np.float32,
    "joint_pos": np.float32,
    "vel_cmd": np.float32,
    "done": np.uint8,
    "pred_index": np.int32,
    "pred_root_pos": np.float32,
    "pred_root_quat_xyzw": np.float32,
    "pred_joint_pos": np.float32,
    "terrain_vertices": np.float32,
    "terrain_faces": np.int32,
    "scan_hits": np.float32,
    "terrain_vertex_colors": np.uint8,
}
_OPTIONAL_KEYS = ("scan_hits", "terrain_vertex_colors")


@dataclass
class VizRecordingMeta:
    """Provenance and layout metadata for a visualization recording.

    Attributes:
        urdf_path: Path to the robot URDF the recording should be viewed with.
        task_name: Gym task name the rollout was collected on.
        fm_checkpoint: Path to the flow-matching checkpoint ("" for dry runs).
        recorded_at: ISO-8601 timestamp of when the recording was made.
        quat_format: Component order of recorded quaternions (always "xyzw").
        dry_run: Whether the predictions came from a randomly-initialized model.
        dt: Environment control step period in seconds.
        update_every: Flow-matching inference cadence in control steps.
        num_envs: Number of recorded environments N.
        num_steps: Number of recorded steps T.
        history_len: Flow-matching history window H.
        future_len: Flow-matching future horizon F.
        num_joints: Number of robot joints nj.
        num_scan_rays: Height-scan rays per env R (0 when no scanner).
        joint_names: Ordered joint names for the recorded ``joint_pos``.
        terrain_source: "generator" (real terrain mesh) or "plane_fallback".
    """

    urdf_path: str
    task_name: str
    fm_checkpoint: str
    recorded_at: str
    quat_format: str
    dry_run: bool
    dt: float
    update_every: int
    num_envs: int
    num_steps: int
    history_len: int
    future_len: int
    num_joints: int
    num_scan_rays: int
    joint_names: list[str]
    terrain_source: str


def unique_output_path(out_dir: str, prefix: str = "viz", ext: str = ".npz") -> str:
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


def save_viz_recording(path: str, data: dict[str, np.ndarray], meta: VizRecordingMeta) -> None:
    """Persist recording arrays and metadata to a compressed ``.npz`` file.

    Only string/number arrays are stored so the file loads with
    ``allow_pickle=False``.

    Args:
        path: Destination ``.npz`` path.
        data: Arrays as documented in the module docstring. ``scan_hits`` and
            ``terrain_vertex_colors`` may be omitted.
        meta: Provenance/layout metadata to embed alongside the arrays.
    """
    arrays: dict[str, np.ndarray] = {}
    for key, dtype in _ARRAY_DTYPES.items():
        if key in _OPTIONAL_KEYS and key not in data:
            continue
        arrays[key] = data[key].astype(dtype)
    np.savez_compressed(
        path,
        **arrays,
        # provenance metadata
        urdf_path=np.array(meta.urdf_path),
        task_name=np.array(meta.task_name),
        fm_checkpoint=np.array(meta.fm_checkpoint),
        recorded_at=np.array(meta.recorded_at),
        quat_format=np.array(meta.quat_format),
        dry_run=np.bool_(meta.dry_run),
        dt=np.float64(meta.dt),
        update_every=np.int64(meta.update_every),
        # layout metadata
        num_envs=np.int64(meta.num_envs),
        num_steps=np.int64(meta.num_steps),
        history_len=np.int64(meta.history_len),
        future_len=np.int64(meta.future_len),
        num_joints=np.int64(meta.num_joints),
        num_scan_rays=np.int64(meta.num_scan_rays),
        joint_names=np.array(meta.joint_names),
        terrain_source=np.array(meta.terrain_source),
    )


def load_viz_recording(path: str) -> tuple[dict[str, np.ndarray], VizRecordingMeta]:
    """Load a recording produced by :func:`save_viz_recording`.

    Args:
        path: Path to a ``.npz`` file written by :func:`save_viz_recording`.

    Returns:
        A ``(data, meta)`` tuple where ``data`` maps array names to numpy arrays
        (optional keys present only if they were recorded) and ``meta`` is the
        reconstructed :class:`VizRecordingMeta`.
    """
    z = np.load(path, allow_pickle=False)
    data = {key: z[key] for key in _ARRAY_DTYPES if key in z.files}
    meta = VizRecordingMeta(
        urdf_path=str(z["urdf_path"]),
        task_name=str(z["task_name"]),
        fm_checkpoint=str(z["fm_checkpoint"]),
        recorded_at=str(z["recorded_at"]),
        quat_format=str(z["quat_format"]),
        dry_run=bool(z["dry_run"]),
        dt=float(z["dt"]),
        update_every=int(z["update_every"]),
        num_envs=int(z["num_envs"]),
        num_steps=int(z["num_steps"]),
        history_len=int(z["history_len"]),
        future_len=int(z["future_len"]),
        num_joints=int(z["num_joints"]),
        num_scan_rays=int(z["num_scan_rays"]),
        joint_names=[str(x) for x in z["joint_names"]],
        terrain_source=str(z["terrain_source"]),
    )
    return data, meta


def find_newest_recording(dir_path: str = VIZ_DIR_DEFAULT) -> str:
    """Return the most recently modified ``viz_*.npz`` recording in a directory.

    Args:
        dir_path: Directory to search (non-recursive).

    Returns:
        Absolute path to the newest recording.

    Raises:
        FileNotFoundError: If the directory contains no ``viz_*.npz`` files.
    """
    candidates = glob.glob(os.path.join(dir_path, "viz_*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No viz_*.npz recordings found in '{dir_path}'. Pass --data <path>."
        )
    return os.path.abspath(max(candidates, key=os.path.getmtime))
