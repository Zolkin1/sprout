# Copyright 2026 Zachary Olkin. All rights reserved.

"""onnxruntime flow-matching generator session (numpy glue + ORT net).

Numpy + onnxruntime counterpart of ``transfer/fm/online.py`` (robot-state
conditioned). The ONNX (``fm.onnx``, exported by
``scripts/flow_matching/export_fm_onnx.py``) folds in the normalizers and the
unrolled Euler sampler, so this builds the **raw** N history (phase) tokens (numpy
``relative_state``), the **raw** root-height token, and the raw strided command
tokens, runs one ``session.run`` for the whole sampler, and lifts the predicted
F-step future to the world frame (numpy ``state_to_absolute``). There is no
inpainting / known mask -- conditioning is separate tokens. No torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .tracker_session import make_session
from .transforms_np import relative_state, state_to_absolute


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
    cmd_dim: int
    cmd_steps: list[int]
    num_cmd_tokens: int
    joint_names: list[str]
    target_frame_names: list[str]
    ground_z: float
    height_dim: int

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
        num_steps=m["num_steps"], cmd_dim=m["cmd_dim"],
        cmd_steps=list(m["cmd_steps"]), num_cmd_tokens=m["num_cmd_tokens"],
        joint_names=list(m["joint_names"]), target_frame_names=list(m["target_frame_names"]),
        ground_z=float(m.get("ground_z", 0.0)), height_dim=int(m.get("height_dim", 1)),
    )


def build_history_tokens(ctx, root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel,
                         frames=None) -> np.ndarray:
    """Build the **raw** ``N`` history (phase) tokens ``(V, N, D)`` (relative-state).

    Unlike the torch ``transfer/fm`` builder, this does NOT normalize -- the
    ``fm.onnx`` graph folds in the state normalization. Inputs are the ``N``
    most-recent real-robot steps ``(V, N, ·)`` in FM joint/frame order, world frame,
    xyzw quats; ``frames`` is ``(V,N,K,13)`` (required when the model predicts extra
    frames). The relative frame is the **xyz+yaw** ground frame at the most recent
    step (index ``N-1``), matching training. Returns ONLY the N history tokens (no
    length-F window, no known mask -- history is separate conditioning).
    """
    v, n = root_pos.shape[0], root_pos.shape[1]
    nj, k, m = ctx.nj, ctx.target_num_extra_frames, root_pos.shape[0] * root_pos.shape[1]
    p_ref = np.broadcast_to(root_pos[:, n - 1 : n], (v, n, 3)).reshape(m, 3)
    q_ref = np.broadcast_to(root_quat[:, n - 1 : n], (v, n, 4)).reshape(m, 4)

    fpos = fquat = flin = fang = None
    if k > 0:
        if frames is None:
            raise ValueError(f"Context predicts {k} extra frames but no history frames were provided.")
        fr = frames.reshape(m, k, 13)
        fpos, fquat, flin, fang = fr[..., 0:3], fr[..., 3:7], fr[..., 7:10], fr[..., 10:13]

    return relative_state(
        root_pos.reshape(m, 3), root_quat.reshape(m, 4),
        root_lin_vel.reshape(m, 3), root_ang_vel.reshape(m, 3),
        joint_pos.reshape(m, nj), joint_vel.reshape(m, nj), p_ref, q_ref,
        frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
    ).reshape(v, n, ctx.D)


def build_height_token(ctx: FmContext, height_scan: np.ndarray) -> np.ndarray:
    """Build the **raw** height-scan token ``(V, height_dim)`` from per-ray ground clearance.

    The ``fm.onnx`` graph folds in the height normalization, so this just shapes the raw
    scan. ``height_scan`` is the live per-ray clearance from the height_scanner sensor
    (body-relative / yaw-aligned), shape ``(V, height_dim)`` or ``(height_dim,)``.
    """
    scan = np.asarray(height_scan, dtype=np.float64)
    if scan.ndim == 1:
        scan = scan[None, :]
    if scan.shape[-1] != ctx.height_dim:
        raise ValueError(f"height_scan width {scan.shape[-1]} != fm height_dim {ctx.height_dim}.")
    return scan


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


def sample_world(ctx: FmContext, history, height, cmd_tokens, p_ref, q_ref, noise) -> dict:
    """Run the ONNX sampler and lift the predicted F-step future to the world frame.

    All F frames are predicted FUTURE (node 0 = the current step, base offset ~= 0),
    expressed relative to the **xyz+yaw** ground frame at ``p_ref``/``q_ref`` (the
    current robot base pose = the anchor at step N-1).

    Args:
        ctx: Loaded FM context.
        history: Raw N history (phase) tokens ``(V, N, D)`` (from :func:`build_history_tokens`).
        height: Raw root-height token ``(V, 1)`` (from :func:`build_height_token`).
        cmd_tokens: Raw strided command tokens ``(V, num_cmd_tokens, cmd_dim)``.
        p_ref: World anchor position ``(V, 3)`` (the current robot base position).
        q_ref: World anchor quaternion ``(V, 4, xyzw)``.
        noise: Sampling noise ``(V, F, D)`` (caller-supplied for determinism).

    Returns:
        World-frame prediction dict (same keys as ``transfer/fm`` ``sample_world``).
    """
    feed = {
        "history": history.astype(np.float32),
        "height": height.astype(np.float32),
        "cmd_tokens": cmd_tokens.astype(np.float32),
        "noise": noise.astype(np.float32),
    }
    pred = ctx.sess.run(["trajectory"], feed)[0].astype(np.float64)  # (V, F, D) denormalized rel-state

    v, f, k = pred.shape[0], ctx.F, ctx.target_num_extra_frames
    p_ref_f = np.broadcast_to(p_ref[:, None, :], (v, f, 3)).reshape(v * f, 3)
    q_ref_f = np.broadcast_to(q_ref[:, None, :], (v, f, 4)).reshape(v * f, 4)
    world = state_to_absolute(pred.reshape(v * f, ctx.D), p_ref_f, q_ref_f, ctx.nj, num_extra_frames=k)

    out = {
        "root_pos_w": world["root_pos_w"].reshape(v, f, 3),
        "root_quat_w": world["root_quat_w"].reshape(v, f, 4),
        "root_link_lin_vel_w": world["root_link_lin_vel_w"].reshape(v, f, 3),
        "root_link_ang_vel_w": world["root_link_ang_vel_w"].reshape(v, f, 3),
        "joint_pos": world["joint_pos"].reshape(v, f, ctx.nj),
        "joint_vel": world["joint_vel"].reshape(v, f, ctx.nj),
    }
    if k > 0:
        out["frame_pos_w"] = world["frame_pos_w"].reshape(v, f, k, 3)
        out["frame_quat_w"] = world["frame_quat_w"].reshape(v, f, k, 4)
        out["frame_lin_vel_w"] = world["frame_lin_vel_w"].reshape(v, f, k, 3)
        out["frame_ang_vel_w"] = world["frame_ang_vel_w"].reshape(v, f, k, 3)
    return out
