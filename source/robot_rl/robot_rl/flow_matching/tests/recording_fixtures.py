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

:func:`windowed_recording` is the one the dataset actually accepts. The per-step
(``.npz``) shape below is kept only for the tests that assert the dataset REFUSES it:
``FutureStateDataset`` requires ``meta.history_anchor == "plan"``, which only the
pre-formed windowed writer produces, so a per-step recording cannot build a dataset at
all (see :func:`per_step_recording`).
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
        Near-upright unit quaternions.
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
    snap_every: int = 4, num_scan_rays: int = 6,
    depth: tuple[int, int, int] | None = None, seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[str], list[int], list[str]]:
    """A multi-episode per-step (``.npz``-shaped) recording.

    Kept for the tests that assert the dataset REFUSES this shape -- it carries no
    ``history_state``, so it selects the per-step branch, which
    ``meta.history_anchor == "plan"`` now rejects. Use :func:`windowed_recording` to build
    a dataset.

    Args:
        n_ep: Episodes.
        length: Steps per episode.
        nj: Joint count.
        k: Extra (ee) frame count.
        future_len: ``F``.
        snap_every: Period of the re-anchor flag.
        num_scan_rays: Height-scan width.
        depth: ``(num_cameras, height, width)`` to add a ``depth`` array, or ``None``.
            The array is flat per camera (``height * width``), as the recorder writes it.
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
        data["depth"] = depth_images(rng, total, depth).astype(np.float32)
    return data, names, dims, frames


def depth_images(
    rng: np.random.Generator, rows: int, depth: tuple[int, int, int]
) -> np.ndarray:
    """``(rows, num_cameras, height*width)`` synthetic depth, flat per camera.

    Flat per camera because that is what the recorder writes: ``mdp.depth_image``
    reshapes the sensor's ``(B, H, W)`` to ``(B, H*W)`` row-major, so the 2-D shape
    survives only in the camera contract. A CNN encoder folds it back, which is why the
    contract's ``width``/``height`` are load-bearing rather than descriptive.

    Args:
        rng: Source of randomness.
        rows: Leading dimension (steps or windows).
        depth: ``(num_cameras, height, width)``.

    Returns:
        The flattened images.
    """
    num_cameras, height, width = depth
    return rng.standard_normal((rows, num_cameras, height * width))


def windowed_recording(
    n_ep: int = 6, windows_per_ep: int = 5, history_len: int = 4, future_len: int = 8,
    nj: int = 4, k: int = 2, num_scan_rays: int = 6,
    depth: tuple[int, int, int] | None = None, track_err: float = 0.02, seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[str], list[int], list[str]]:
    """A pre-formed windowed (``.h5``-shaped) recording -- the format the dataset accepts.

    One window per snap (re-plan) event, already cut, matching
    ``recorded_data_io.RecordedWindowWriter``. The four invariants
    ``FutureStateDataset._check_window_invariants`` enforces are built in rather than
    hoped for:

    * ``history_ref[H-1] == ref_future[0]`` -- both are step ``H-1`` of one smooth
      reference, so the applied-reference ring and the look-ahead agree at the snap
      (guard 3, the off-by-one catcher).
    * ``history_ref[H-2]`` is exactly one control step before ``ref_future[0]``
      (guard 4).
    * ``history_state`` rows ``0..H-2`` all carry ONE pose -- the robot at the last
      re-plan, which is what the recorder writes there and what makes the history
      PLAN-anchored. Only row ``H-1`` is the robot at the snap (hence
      ``history_anchor="plan"``; see the module docstring).
    * that snap row sits ``track_err`` from the reference, so the target's node-0 base
      offset is a small live tracking error rather than zero or a metre (guard 1).

    Args:
        n_ep: Episodes (the val split is drawn over these, so >= 3 for a nonempty split).
        windows_per_ep: Snap windows per episode.
        history_len: ``H`` rows per window. Needs ``H >= num_history_states + 1``.
        future_len: ``F`` nodes per window.
        nj: Joint count.
        k: Extra (ee) frame count.
        num_scan_rays: Height-scan width.
        depth: ``(num_cameras, height, width)`` to add a ``depth`` array, or ``None``.
        track_err: Metres between the robot and its reference at the snap.
        seed: RNG seed.

    Returns:
        ``(data, state_term_names, state_term_dims, frame_names)``.
    """
    rng = np.random.default_rng(seed)
    names, dims, frames = state_layout(nj, k)
    total = n_ep * windows_per_ep
    hist_ref, hist_state, futures = [], [], []
    for _ in range(total):
        # H + F - 1 steps: the H history rows, then the F future nodes starting at the
        # LAST history row (which is the snap, shared between the two).
        ref = reference_trajectory(rng, history_len + future_len - 1, nj, k)
        h_ref = ref[:history_len]
        future = ref[history_len - 1:history_len - 1 + future_len]
        measured = ref[:history_len].copy()
        measured[:, 0:3] += rng.standard_normal((history_len, 3)) * track_err
        measured[:, 13:13 + 2 * nj] += rng.standard_normal((history_len, 2 * nj)) * 0.01
        # Rows 0..H-2 collapse onto the plan anchor: ONE pose, the robot as it was when
        # this plan was made. Row H-1 stays the robot at the snap.
        measured[:history_len - 1] = measured[0]
        hist_ref.append(h_ref)
        hist_state.append(measured)
        futures.append(future)
    data = {
        "history_ref": np.stack(hist_ref).astype(np.float32),      # (W, H, node)
        "history_state": np.stack(hist_state).astype(np.float32),  # (W, H, node)
        "ref_future": np.stack(futures).astype(np.float32),        # (W, F, node)
        "cmd_future": rng.standard_normal((total, future_len, 3)).astype(np.float32),
        "height_scan": rng.standard_normal((total, num_scan_rays)).astype(np.float32),
        # Per-WINDOW episode id (the snap row's), which is what the windowed split reads.
        "episode_id": np.repeat(np.arange(n_ep, dtype=np.int64), windows_per_ep),
        # Every window tagged, as the current recorder does. Not all-False: skip_distractors
        # is ON by default now, and the dataset refuses a filter that would drop nothing as
        # readily as one that would drop everything -- a fixture of all-False would pass by
        # being rejected for the wrong reason. One distractor per episode leaves the split
        # non-empty either way.
        "is_distractor": np.tile(
            np.arange(windows_per_ep) == windows_per_ep - 1, n_ep).astype(bool),
    }
    if depth is not None:
        data["depth"] = depth_images(rng, total, depth).astype(np.float32)
    return data, names, dims, frames

def camera_spec(n_cameras: int, height: int, width: int) -> list[dict]:
    """A minimal depth-camera contract for a ``(n_cameras, height*width)`` depth array.

    Field set and values mirror :func:`mdp.depth_camera_specs` (the real producer) closely
    enough to satisfy both the dataset's validation and
    ``deploy.depth_obs.parse_depth_spec``, so a spec built here would survive the real
    export -> deploy path.

    The geometry is REAL 2-D, not a ``token_dim x 1`` placeholder. It used to be the
    latter, which was harmless while the exteroception embedding was a flat MLP -- and
    silently degenerate once it became a convolution, since every "image" was one pixel
    tall and every conv test a 1-D one. Intrinsics are still nominal; nothing here asserts
    the numbers describe a real lens.

    Args:
        n_cameras: Number of cameras, one per depth sensor slot.
        height: Image rows.
        width: Image columns.

    Returns:
        One contract dict per camera, in token order.
    """
    return [
        {
            "obs_term": f"depth_cam{i}",
            "sensor": f"cam{i}",
            "mount_body": "torso_link",
            # Forward of and above the mount, pitched 45 deg DOWN -- not the identity pose it
            # used to be. A camera whose boresight is horizontal never meets the ground, so it
            # has no geometric xyz offset and the contract normalization refuses it; the real
            # pair looks 65 and 35 deg down. The numbers are nominal, but the posture is not
            # optional.
            "offset_pos": [0.1, 0.0, 0.1],
            "offset_quat_xyzw": [0.0, 0.3826834, 0.0, 0.9238795],
            "width": width,
            "height": height,
            "fx": 1.0,
            "fy": 1.0,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "max_distance": 9.0,
            "update_period": 0.0,
            "data_type": "distance_to_image_plane",
            "depth_clipping_behavior": "max",
            "clip": None,
            # normalize_depth's defaults, as _normalization_params would resolve them.
            "offset": 0.75,
            "scale": 0.5,
            "min_dist": 0.2,
            "max_dist": 9.0,
            "min_clip_val": 0.0,
            "max_clip_val": 0.0,
        }
        for i in range(n_cameras)
    ]
