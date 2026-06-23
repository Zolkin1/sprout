# Copyright 2026 Zachary Olkin. All rights reserved.

"""onnxruntime flow-matching generator session (numpy glue + ORT net).

Numpy + onnxruntime counterpart of ``transfer/fm/online.py``. The ONNX
(``fm.onnx``, exported by ``scripts/flow_matching/export_fm_onnx.py``) folds in the
normalizers and the unrolled Euler sampler, so this builds the **raw** history
token (numpy ``relative_state``), runs one ``session.run`` for the whole sampler,
and lifts the predicted relative state to the world frame (numpy
``state_to_absolute``). No torch.
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
    H: int
    F: int
    D: int
    history_D: int
    Din: int
    nj: int
    history_num_extra_frames: int
    target_num_extra_frames: int
    num_steps: int
    cmd_dim: int
    future_cmd_dim: int
    scan_dim: int
    joint_names: list[str]
    history_frame_names: list[str]
    target_frame_names: list[str]

    @property
    def frame_names(self) -> list[str]:
        """Predicted (target) extra-frame order -- what ``ee_ref`` consumes."""
        return self.target_frame_names


def load_fm_context(onnx_path: str, num_threads: int = 0) -> FmContext:
    """Load ``fm.onnx`` and parse its ``fm_meta`` metadata into an :class:`FmContext`."""
    sess = make_session(onnx_path, num_threads)
    raw = sess.get_modelmeta().custom_metadata_map
    if "fm_meta" not in raw:
        raise ValueError(f"{onnx_path} has no 'fm_meta' metadata; re-export with metadata.")
    m = json.loads(raw["fm_meta"])
    return FmContext(
        sess=sess, H=m["H"], F=m["F"], D=m["D"], history_D=m["history_D"], Din=m["Din"], nj=m["nj"],
        history_num_extra_frames=len(m["history_frame_names"]),
        target_num_extra_frames=len(m["target_frame_names"]),
        num_steps=m["num_steps"], cmd_dim=m["cmd_dim"], future_cmd_dim=m["future_cmd_dim"],
        scan_dim=m["scan_dim"], joint_names=list(m["joint_names"]),
        history_frame_names=list(m["history_frame_names"]), target_frame_names=list(m["target_frame_names"]),
    )


def build_history_token(ctx, root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel,
                        cmd, frames=None) -> np.ndarray:
    """Build the **raw** (un-normalized) history token ``(V, H, Din)``.

    Unlike the torch ``transfer/fm`` builder, this does NOT normalize -- the
    ``fm.onnx`` graph folds in the history normalization. Inputs ``(V, H, ·)`` in
    FM joint/frame order, world frame, xyzw quats; ``frames`` is ``(V,H,K_hist,13)``
    or None.
    """
    v, h = root_pos.shape[0], root_pos.shape[1]
    nj, k, m = ctx.nj, ctx.history_num_extra_frames, root_pos.shape[0] * root_pos.shape[1]
    p_ref = np.broadcast_to(root_pos[:, h - 1 : h], (v, h, 3)).reshape(m, 3)
    q_ref = np.broadcast_to(root_quat[:, h - 1 : h], (v, h, 4)).reshape(m, 4)

    fpos = fquat = flin = fang = None
    if k > 0:
        if frames is None:
            raise ValueError(f"Context expects {k} history extra frames but none were provided.")
        fr = frames.reshape(m, k, 13)
        fpos, fquat, flin, fang = fr[..., 0:3], fr[..., 3:7], fr[..., 7:10], fr[..., 10:13]

    rel = relative_state(
        root_pos.reshape(m, 3), root_quat.reshape(m, 4),
        root_lin_vel.reshape(m, 3), root_ang_vel.reshape(m, 3),
        joint_pos.reshape(m, nj), joint_vel.reshape(m, nj), p_ref, q_ref,
        frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
    ).reshape(v, h, ctx.history_D)
    return np.concatenate([rel, cmd], axis=-1)


def sample_world(ctx: FmContext, history_token, p_ref, q_ref, future_cmd, noise) -> dict:
    """Run the ONNX sampler and lift the predicted relative state to the world frame.

    Args:
        ctx: Loaded FM context.
        history_token: Raw history token ``(V, H, Din)``.
        p_ref: World anchor position ``(V, 3)``.
        q_ref: World anchor quaternion ``(V, 4, xyzw)``.
        future_cmd: Raw future command ``(V, F, future_cmd_dim)``.
        noise: Sampling noise ``(V, F, D)`` (caller-supplied for determinism).

    Returns:
        World-frame prediction dict (same keys as ``transfer/fm`` ``sample_world``).
    """
    feed = {
        "history_token": history_token.astype(np.float32),
        "future_cmd": future_cmd.astype(np.float32),
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
