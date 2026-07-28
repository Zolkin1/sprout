# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage 2: turn a :class:`~motion_gen.motionbricks.raw_schema.RawMotion` into the world
trajectory the existing :class:`~motion_gen.clip_output.ClipOutputBuilder` consumes.

Motion Bricks emits ``mujoco_qpos`` at 30 Hz; the clip library must be at the RL control dt
(50 Hz). :func:`raw_to_world` reindexes the 29 joints to ``G1_JOINT_NAMES`` order and resamples
root position / orientation / joints / command to the target dt, returning exactly the
``(root_pos, root_quat, joints)`` triple + ``target_vel`` that ``ClipOutputBuilder.build`` expects.
Everything downstream (FK end-effector frames, contacts, the writer) is the shared clip machinery --
this module is the *only* Motion-Bricks-specific conversion on the isaaclab-free side.
"""

from __future__ import annotations

import numpy as np
import torch

from .joint_map import reindex_to, split_qpos
from .raw_schema import RawMotion


def resample_times(n_src: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """Fractional source-frame indices ``[n_dst]`` that resample ``n_src`` frames from ``src_fps`` to ``dst_fps``.

    Preserves the clip's wall-clock duration ``(n_src - 1) / src_fps``; the returned indices span
    ``[0, n_src - 1]`` and are used to linearly (or spherically, for quaternions) interpolate rows.
    """
    if n_src < 1:
        raise ValueError("need at least one source frame")
    if n_src == 1:
        return np.zeros(1, dtype=np.float64)
    duration = (n_src - 1) / float(src_fps)
    n_dst = max(1, int(round(duration * float(dst_fps))) + 1)
    return np.linspace(0.0, n_src - 1, n_dst, dtype=np.float64)


def _lerp_rows(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linearly interpolate rows of ``arr[n_src, D]`` at fractional indices ``t[n_dst]``."""
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, arr.shape[0] - 1)
    w = (t - i0)[:, None]
    return arr[i0] * (1.0 - w) + arr[i1] * w


def resample_rows(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly resample rows of ``arr[n_src, D]`` from ``src_fps`` to ``dst_fps``, preserving duration.

    The row-wise counterpart of :func:`raw_to_world` for any per-frame quantity that is not a rotation
    (e.g. a second command channel), so it lands on exactly the same clock as the converted motion without
    re-running the quaternion/joint conversion.
    """
    a = np.asarray(arr, dtype=np.float32)
    return _lerp_rows(a, resample_times(a.shape[0], src_fps, dst_fps))


def _slerp_quat(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherically interpolate wxyz quaternions ``q[n_src, 4]`` at fractional indices ``t[n_dst]``.

    Neighbour signs are aligned (shortest arc) and near-parallel pairs fall back to normalised lerp.
    """
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, q.shape[0] - 1)
    w = (t - i0)
    q0 = q[i0]
    q1 = q[i1].copy()
    dot = (q0 * q1).sum(axis=-1)
    q1[dot < 0.0] *= -1.0                       # shortest-arc: flip sign of the far neighbour
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    close = sin_t < 1e-4
    safe = np.where(close, 1.0, sin_t)
    a = (np.sin((1.0 - w) * theta) / safe)[:, None]
    b = (np.sin(w * theta) / safe)[:, None]
    slerp = a * q0 + b * q1
    nlerp = q0 * (1.0 - w)[:, None] + q1 * w[:, None]
    out = np.where(close[:, None], nlerp, slerp)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def raw_to_world(
    raw: RawMotion, target_dt: float, device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a :class:`RawMotion` to ``(root_pos[T,3], root_quat_wxyz[T,4], joints[T,29], target_vel[T,3])``.

    ``T`` is the resampled length at ``1/target_dt`` Hz; joints come out in ``G1_JOINT_NAMES`` order.
    Returns torch float tensors on ``device`` ready for ``ClipOutputBuilder.build`` / the writer.
    """
    root_pos, root_quat, joints_mb = split_qpos(raw.mujoco_qpos)
    idx = reindex_to(raw.joint_names)                       # MB joint order -> G1_JOINT_NAMES
    joints = joints_mb[:, idx]
    t = resample_times(raw.num_frames, raw.fps, 1.0 / target_dt)
    root_pos_r = _lerp_rows(root_pos, t)
    root_quat_r = _slerp_quat(root_quat, t)
    joints_r = _lerp_rows(joints, t)
    target_vel_r = _lerp_rows(raw.target_vel, t)

    def _to(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    return _to(root_pos_r), _to(root_quat_r), _to(joints_r), _to(target_vel_r)


def fit_to_length(x: torch.Tensor, T: int) -> tuple[torch.Tensor, int]:
    """Pad (repeat last frame) or truncate ``x[L, ...]`` to exactly ``T`` frames.

    Returns ``(x_fit[T, ...], clip_len)`` where ``clip_len = min(L, T)`` marks the real frames before
    repeat-last padding (the clip library's ``clip_len`` convention -- the RL ends the episode there).
    """
    L = x.shape[0]
    if L == T:
        return x, T
    if L > T:
        return x[:T].contiguous(), T
    pad = x[-1:].expand(T - L, *x.shape[1:])
    return torch.cat([x, pad], dim=0).contiguous(), L
