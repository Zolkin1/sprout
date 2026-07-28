# Copyright 2026 Zachary Olkin. All rights reserved.

"""onnxruntime flow-matching generator session (numpy glue + ORT net).

Numpy + onnxruntime counterpart of ``transfer/fm/online.py``. The ONNX
(``fm.onnx``, exported by ``scripts/flow_matching/export_fm_onnx.py``) folds in the
normalizers and the unrolled masked-Euler sampler, so this builds the **raw**
conditioning window (numpy ``relative_state``) + known mask + raw strided command
tokens, runs one ``session.run`` for the whole sampler, and lifts the predicted
relative state to the world frame (numpy ``state_to_absolute``). No torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .tracker_session import make_session
from .transforms_np import matrix_from_quat, relative_state, so3_log, state_to_absolute


@dataclass
class FmContext:
    """Loaded ``fm.onnx`` session + metadata (mirrors ``transfer/fm`` FmOnlineContext)."""

    sess: ort.InferenceSession
    N: int
    F: int
    D: int
    nj: int
    target_num_extra_frames: int
    num_steps: int
    dt: float
    cmd_dim: int
    cmd_steps: list[int]
    num_cmd_tokens: int
    joint_names: list[str]
    target_frame_names: list[str]

    @property
    def frame_names(self) -> list[str]:
        """Predicted (target) extra-frame order -- what ``ee_ref`` consumes."""
        return self.target_frame_names

    @property
    def num_extra_frames(self) -> int:
        """Alias for the predicted (target) extra-frame count."""
        return self.target_num_extra_frames


def load_fm_context(onnx_path: str, num_threads: int = 0) -> FmContext:
    """Load ``fm.onnx`` and parse its ``fm_meta`` metadata into an :class:`FmContext`."""
    sess = make_session(onnx_path, num_threads)
    raw = sess.get_modelmeta().custom_metadata_map
    if "fm_meta" not in raw:
        raise ValueError(f"{onnx_path} has no 'fm_meta' metadata; re-export with metadata.")
    m = json.loads(raw["fm_meta"])
    return FmContext(
        sess=sess, N=m["N"], F=m["F"], D=m["D"], nj=m["nj"],
        target_num_extra_frames=len(m["target_frame_names"]),
        num_steps=m["num_steps"], dt=float(m["dt"]), cmd_dim=m["cmd_dim"],
        cmd_steps=list(m["cmd_steps"]), num_cmd_tokens=m["num_cmd_tokens"],
        joint_names=list(m["joint_names"]), target_frame_names=list(m["target_frame_names"]),
    )


def build_cond_states(ctx, root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel,
                      frames=None) -> tuple[np.ndarray, np.ndarray]:
    """Build the **raw** conditioning window ``(V, F, D)`` and known mask ``(V, F)``.

    Unlike the torch ``transfer/fm`` builder, this does NOT normalize -- the
    ``fm.onnx`` graph folds in the state normalization. Inputs are the ``N``
    most-recent steps ``(V, N, ·)`` in FM joint/frame order, world frame, xyzw quats;
    ``frames`` is ``(V,N,K,13)`` (required when the model predicts extra frames).
    The N relative states fill the first N tokens (anchored at step N-1); the rest
    are zeros with ``known = 0``.
    """
    v, n = root_pos.shape[0], root_pos.shape[1]
    nj, k, m = ctx.nj, ctx.target_num_extra_frames, root_pos.shape[0] * root_pos.shape[1]
    p_ref = np.broadcast_to(root_pos[:, n - 1 : n], (v, n, 3)).reshape(m, 3)
    q_ref = np.broadcast_to(root_quat[:, n - 1 : n], (v, n, 4)).reshape(m, 4)

    fpos = fquat = None
    if k > 0:
        if frames is None:
            raise ValueError(f"Context predicts {k} extra frames but no history frames were provided.")
        fr = frames.reshape(m, k, 13)
        fpos, fquat = fr[..., 0:3], fr[..., 3:7]  # velocities (7:13) unused (position-only state)

    rel = relative_state(
        root_pos.reshape(m, 3), root_quat.reshape(m, 4),
        joint_pos.reshape(m, nj), p_ref, q_ref,
        frame_pos_w=fpos, frame_quat_w=fquat,
    ).reshape(v, n, ctx.D)

    cond = np.zeros((v, ctx.F, ctx.D), dtype=rel.dtype)
    cond[:, :n] = rel
    known = np.zeros((v, ctx.F), dtype=np.float32)
    known[:, :n] = 1.0
    return cond, known


def build_cmd_tokens(ctx: FmContext, future_cmd: np.ndarray) -> np.ndarray:
    """Stride the per-step commanded-velocity profile into **raw** command tokens.

    Args:
        ctx: Loaded FM context.
        future_cmd: Raw command profile over the window ``(V, F, cmd_dim)``.

    Returns:
        Raw strided command tokens ``(V, num_cmd_tokens, cmd_dim)`` (the ``fm.onnx``
        graph folds in the command normalization).
    """
    return future_cmd[:, ctx.cmd_steps, :]


def _finite_diff(seq: np.ndarray, dt: float) -> np.ndarray:
    """Forward finite difference along the time axis (axis 1); backward-filled at the last step.

    ``seq`` is ``(V, F, ·)`` or ``(V, F, K, ·)``; returns the same shape.
    """
    diff = (seq[:, 1:] - seq[:, :-1]) / dt
    return np.concatenate([diff, diff[:, -1:]], axis=1)


def _angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a quaternion sequence ``(V, F, [K,] 4)``.

    ``so3_log(R[t+1] @ R[t]^T) / dt`` (matches the torch ``online.sample_world`` and the
    training-time velocity-match loss), backward-filled at the last step.
    """
    r = matrix_from_quat(quat)  # (V, F, [K,] 3, 3)
    w = so3_log(r[:, 1:] @ np.swapaxes(r[:, :-1], -1, -2)) / dt
    return np.concatenate([w, w[:, -1:]], axis=1)


def sample_world(ctx: FmContext, cond, known, p_ref, q_ref, cmd_tokens, noise) -> dict:
    """Run the ONNX sampler and lift the predicted relative state to the world frame.

    The state is position-only, so the velocity fields are reconstructed here by
    finite-differencing the predicted world-frame position / orientation sequence
    (``ctx.dt`` spacing) -- matching the torch ``transfer/fm`` ``sample_world``.

    Args:
        ctx: Loaded FM context.
        cond: Raw conditioning window ``(V, F, D)``.
        known: Known mask ``(V, F)`` (1.0 at conditioning tokens).
        p_ref: World anchor position ``(V, 3)``.
        q_ref: World anchor quaternion ``(V, 4, xyzw)``.
        cmd_tokens: Raw strided command tokens ``(V, num_cmd_tokens, cmd_dim)``.
        noise: Sampling noise ``(V, F, D)`` (caller-supplied for determinism).

    Returns:
        World-frame prediction dict (same keys as ``transfer/fm`` ``sample_world``).
        Novel (predicted) frames are ``N..F-1``. The velocity fields are finite
        differences of the position/orientation fields.
    """
    feed = {
        "cond": cond.astype(np.float32),
        "known": known.astype(np.float32),
        "cmd_tokens": cmd_tokens.astype(np.float32),
        "noise": noise.astype(np.float32),
    }
    pred = ctx.sess.run(["trajectory"], feed)[0].astype(np.float64)  # (V, F, D) denormalized rel-state

    v, f, k = pred.shape[0], ctx.F, ctx.target_num_extra_frames
    p_ref_f = np.broadcast_to(p_ref[:, None, :], (v, f, 3)).reshape(v * f, 3)
    q_ref_f = np.broadcast_to(q_ref[:, None, :], (v, f, 4)).reshape(v * f, 4)
    world = state_to_absolute(pred.reshape(v * f, ctx.D), p_ref_f, q_ref_f, ctx.nj, num_extra_frames=k)

    root_pos = world["root_pos_w"].reshape(v, f, 3)
    root_quat = world["root_quat_w"].reshape(v, f, 4)
    joint_pos = world["joint_pos"].reshape(v, f, ctx.nj)
    dt = ctx.dt
    out = {
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "root_link_lin_vel_w": _finite_diff(root_pos, dt),
        "root_link_ang_vel_w": _angular_velocity(root_quat, dt),
        "joint_pos": joint_pos,
        "joint_vel": _finite_diff(joint_pos, dt),
    }
    if k > 0:
        frame_pos = world["frame_pos_w"].reshape(v, f, k, 3)
        frame_quat = world["frame_quat_w"].reshape(v, f, k, 4)
        out["frame_pos_w"] = frame_pos
        out["frame_quat_w"] = frame_quat
        out["frame_lin_vel_w"] = _finite_diff(frame_pos, dt)
        out["frame_ang_vel_w"] = _angular_velocity(frame_quat, dt)
    return out
