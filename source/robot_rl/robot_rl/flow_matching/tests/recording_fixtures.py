# Copyright 2026 Zachary Olkin. All rights reserved.

"""Synthetic recordings for the FM dataset tests.

Shared because the dataset now VALIDATES the recording rather than just reshaping it
(``FutureStateDataset._check_window_invariants``), so a fixture has to be a plausible
rollout instead of white noise. Three properties the old per-file builders did not have,
each pinned by a guard:

* **Smooth.** Consecutive reference rows are one control step apart, so joints move by
  ~0.01 rad between them, not by a full standard normal.
* **Tracked.** The measured robot sits within a few centimetres of the reference, so the
  ego-anchor invariant (target node 0's base offset) holds.
* **Near-upright, unit quaternions.** The base_pose orientation channel is a
  ``quat_box_minus`` log map that saturates at pi; uniformly random rotations sail past it
  and trip the branch-cut guard. (Random *unnormalised* 4-vectors, which is what the old
  fixtures used, are not rotations at all.)

* ``ref_state[t] == ref_future[t, 0]`` -- the same step, which is the alignment the
  recorder guarantees and the dataset checks.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise along the last axis."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _small_quats(rng: np.random.Generator, t: int, tilt: float = 0.08) -> np.ndarray:
    """``(t, 4)`` xyzw unit quaternions near identity, drifting smoothly.

    Args:
        rng: Source of randomness.
        t: Number of steps.
        tilt: Scale of the rotation-vector random walk.

    Returns:
        Near-upright unit quaternions, well inside the ``quat_box_minus`` branch cut.
    """
    axis = np.cumsum(rng.standard_normal((t, 3)) * tilt * 0.1, axis=0) * tilt
    return _unit(np.concatenate([axis, np.ones((t, 1))], axis=-1))


def reference_trajectory(
    rng: np.random.Generator, t: int, nj: int, k: int
) -> np.ndarray:
    """A smooth world-frame reference trajectory in the recorded node layout.

    Layout matches ``record_data._assemble_state``: ``root_pos(3), root_quat(4 xyzw),
    root_lin(3), root_ang(3), joint_pos(nj), joint_vel(nj)``, then per extra frame
    ``pos(3), quat(4), lin(3), ang(3)``.

    Args:
        rng: Source of randomness.
        t: Number of steps.
        nj: Joint count.
        k: Extra (ee) frame count.

    Returns:
        ``(t, 13 + 2*nj + 13*k)`` float64 rows.
    """
    pos = np.cumsum(rng.standard_normal((t, 3)) * 0.01, axis=0) + np.array([0.0, 0.0, 0.75])
    quat = _small_quats(rng, t)
    lin = np.gradient(pos, axis=0) / 0.02
    ang = rng.standard_normal((t, 3)) * 0.05
    jp = np.cumsum(rng.standard_normal((t, nj)) * 0.01, axis=0)
    jv = np.gradient(jp, axis=0) / 0.02
    parts = [pos, quat, lin, ang, jp, jv]
    for i in range(k):
        offset = np.array([0.1 * (i + 1), 0.05 * (i + 1), -0.4])
        parts += [pos + offset, quat, lin, ang]
    return np.concatenate(parts, axis=-1)


def episode_arrays(
    rng: np.random.Generator, length: int, future_len: int, nj: int, k: int,
    snap_every: int, track_err: float = 0.02,
) -> dict[str, np.ndarray]:
    """One episode's per-step arrays: measured state, applied reference, and futures.

    Args:
        rng: Source of randomness.
        length: Steps in the episode.
        future_len: ``F`` nodes per ``ref_future`` row.
        nj: Joint count.
        k: Extra (ee) frame count.
        snap_every: Period of the ``reanchor_active`` flag.
        track_err: Metres of position error between the robot and the reference.

    Returns:
        ``state``, ``ref_state``, ``ref_future``, ``reanchor_active``, ``done``.
    """
    ref = reference_trajectory(rng, length + future_len, nj, k)
    state = ref[:length].copy()
    # The robot tracks the reference with a small error -- so node 0's base offset is the
    # live tracking error, small but non-zero, exactly as in a real rollout.
    state[:, 0:3] += rng.standard_normal((length, 3)) * track_err
    state[:, 13:13 + 2 * nj] += rng.standard_normal((length, 2 * nj)) * 0.01
    # ref_future[t] is the reference from step t forward, so node 0 IS ref_state[t].
    future = np.stack([ref[t:t + future_len] for t in range(length)])
    done = np.zeros(length, np.uint8)
    done[-1] = 1
    return {
        "state": state.astype(np.float32),
        "ref_state": ref[:length].astype(np.float32),
        "ref_future": future.astype(np.float32),
        "reanchor_active": (np.arange(length) % snap_every == 0).astype(np.uint8),
        "done": done,
    }


def state_layout(nj: int, k: int) -> tuple[list[str], list[int], list[str]]:
    """``(term_names, term_dims, frame_names)`` for the recorded node layout."""
    names = ["root_pos_w", "root_quat_w", "root_link_lin_vel_w", "root_link_ang_vel_w",
             "joint_pos", "joint_vel"]
    dims = [3, 4, 3, 3, nj, nj]
    frames = [f"f{i}" for i in range(k)]
    for fr in frames:
        names += [f"{fr}:pos_w", f"{fr}:quat_w", f"{fr}:lin_vel_w", f"{fr}:ang_vel_w"]
        dims += [3, 4, 3, 3]
    return names, dims, frames


def per_step_recording(
    n_ep: int = 6, length: int = 40, nj: int = 4, k: int = 2, future_len: int = 8,
    snap_every: int = 4, num_scan_rays: int = 6, depth: tuple[int, int] | None = None,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[str], list[int], list[str]]:
    """A multi-episode per-step (``.npz``-shaped) recording.

    Args:
        n_ep: Episodes.
        length: Steps per episode.
        nj: Joint count.
        k: Extra (ee) frame count.
        future_len: ``F``.
        snap_every: Period of the re-anchor flag.
        num_scan_rays: Height-scan width.
        depth: ``(num_cameras, per_camera_dim)`` to add a ``depth`` array, or ``None``.
        seed: RNG seed.

    Returns:
        ``(data, state_term_names, state_term_dims, frame_names)``.
    """
    rng = np.random.default_rng(seed)
    names, dims, frames = state_layout(nj, k)
    per_ep = [episode_arrays(rng, length, future_len, nj, k, snap_every) for _ in range(n_ep)]
    data = {key: np.concatenate([ep[key] for ep in per_ep]) for key in per_ep[0]}
    total = n_ep * length
    data["cmd_future"] = rng.standard_normal((total, future_len, 3)).astype(np.float32)
    data["height_scan"] = rng.standard_normal((total, num_scan_rays)).astype(np.float32)
    data["episode_id"] = np.repeat(np.arange(n_ep, dtype=np.int64), length)
    data["step_in_episode"] = np.tile(np.arange(length, dtype=np.int64), n_ep)
    if depth is not None:
        data["depth"] = rng.standard_normal((total, depth[0], depth[1])).astype(np.float32)
    return data, names, dims, frames
