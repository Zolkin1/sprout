# Copyright 2026 Zachary Olkin. All rights reserved.

"""onnxruntime flow-matching generator session (numpy glue + ORT net).

Numpy + onnxruntime counterpart of ``robot_rl.flow_matching.online`` (robot-state
conditioned). The ONNX (``fm.onnx``, exported by
``scripts/flow_matching/export_fm_onnx.py``) folds in the normalizers and the
unrolled Euler sampler, so this builds the **raw** N history (phase) tokens (numpy
``relative_state``), the **raw** exteroception tokens, and the raw strided command
tokens, runs one ``session.run`` for the whole sampler, and lifts the predicted
F-step future to the world frame (numpy ``state_to_absolute``). There is no
inpainting / known mask -- conditioning is separate tokens. No torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from .depth_obs import parse_depth_spec, sanitize_extero_np
from .tracker_session import make_session
from .transforms_np import relative_state, state_to_absolute


@dataclass
class FmContext:
    """Loaded ``fm.onnx`` session + metadata (mirrors ``flow_matching.online`` FmOnlineContext)."""

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
    extero_source: str = "height_scan"
    """Which sensor feeds the exteroception tokens ("height_scan" or "depth")."""
    extero_tokens: int = 1
    """Number of exteroception tokens: 1 for the scan, one per camera for depth."""
    extero_spec: list[dict] = field(default_factory=list)
    """Camera contract behind those tokens (depth only); see ``depth_obs.py``."""

    @property
    def extero_dim(self) -> int:
        """Flat width of the exteroception input, ``extero_tokens * height_dim``."""
        return self.extero_tokens * self.height_dim
    has_history_null: bool = False
    """Whether the graph accepts the ``history_null`` input (learned null-history embedding).

    False for an ``fm.onnx`` exported before that input existed; the caller must then fall back
    to withholding the generator until its history window fills, and cannot match the RL side's
    step-0 behaviour. Re-export with ``export_fm_onnx.py`` to enable it."""

    @property
    def frame_names(self) -> list[str]:
        """Predicted (target) extra-frame order -- what ``ee_ref`` consumes."""
        return self.target_frame_names

    @property
    def num_extra_frames(self) -> int:
        """Alias for the predicted (target) extra-frame count."""
        return self.target_num_extra_frames


_WARNED_NO_NULL = False


def load_fm_context(onnx_path: str, num_threads: int = 0) -> FmContext:
    """Load ``fm.onnx`` and parse its ``fm_meta`` metadata into an :class:`FmContext`."""
    sess = make_session(onnx_path, num_threads)
    raw = sess.get_modelmeta().custom_metadata_map
    if "fm_meta" not in raw:
        raise ValueError(f"{onnx_path} has no 'fm_meta' metadata; re-export with metadata.")
    m = json.loads(raw["fm_meta"])
    source = str(m.get("extero_source", "height_scan"))
    return FmContext(
        sess=sess, N=m["N"], F=m["F"], D=m["D"], nj=m["nj"],
        target_num_extra_frames=len(m["target_frame_names"]),
        num_steps=m["num_steps"], cmd_dim=m["cmd_dim"],
        cmd_steps=list(m["cmd_steps"]), num_cmd_tokens=m["num_cmd_tokens"],
        joint_names=list(m["joint_names"]), target_frame_names=list(m["target_frame_names"]),
        ground_z=float(m.get("ground_z", 0.0)), height_dim=int(m.get("height_dim", 1)),
        extero_source=source, extero_tokens=int(m.get("extero_tokens", 1)),
        extero_spec=parse_depth_spec(m.get("extero_spec"), source=f"{onnx_path} fm_meta"),
        # Trust the graph itself over the metadata flag: an input either exists or it does not.
        has_history_null=any(i.name == "history_null" for i in sess.get_inputs()),
    )


def build_history_tokens(ctx, root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel,
                         frames=None) -> np.ndarray:
    """Build the **raw** ``N`` history (phase) tokens ``(V, N, D)`` (relative-state).

    Unlike the torch ``flow_matching.online`` builder, this does NOT normalize -- the
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


def build_extero_tokens(ctx: FmContext, values: np.ndarray) -> np.ndarray:
    """Build the **raw** exteroception tokens ``(V, extero_tokens * height_dim)``.

    Source-agnostic, mirroring ``flow_matching.online.build_extero_tokens``: the context
    knows which sensor the checkpoint was trained on, so the caller passes whichever
    reading that names -- the live per-ray height scan (body-relative, yaw-aligned) or the
    per-camera normalised depth rows -- and the clip band follows from
    ``ctx.extero_source``. Both are already in the FM's xyz+yaw ego frame, so there is no
    re-anchoring. The ``fm.onnx`` graph folds in the normalization, so this only shapes
    and sanitises.

    Args:
        ctx: Loaded FM context.
        values: Raw reading, shape ``(V, T, dim)``, ``(V, T*dim)`` or ``(T*dim,)``.

    Returns:
        Raw tokens flattened to ``(V, T*dim)`` -- the graph reshapes internally.

    Raises:
        ValueError: If the flat width does not match the model's exteroception input.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    flat = arr.reshape(arr.shape[0], -1)
    if flat.shape[-1] != ctx.extero_dim:
        raise ValueError(
            f"exteroception width {flat.shape[-1]} != fm extero_dim {ctx.extero_dim} "
            f"({ctx.extero_tokens} x {ctx.height_dim}, source '{ctx.extero_source}')."
        )
    # Missed rays / bad sensor reads arrive as +/-inf or NaN; the fm.onnx graph folds in
    # the normalization, so a non-finite value here would poison it. The band is
    # per-source: metres for the scan, normalised units for depth.
    return sanitize_extero_np(flat, ctx.extero_source)


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


def sample_world(ctx: FmContext, history, extero, cmd_tokens, p_ref, q_ref, noise, history_null=None) -> dict:
    """Run the ONNX sampler and lift the predicted F-step future to the world frame.

    All F frames are predicted FUTURE (node 0 = the current step, base offset ~= 0),
    expressed relative to the **xyz+yaw** ground frame at ``p_ref``/``q_ref`` (the
    current robot base pose = the anchor at step N-1).

    Args:
        ctx: Loaded FM context.
        history: Raw N history (phase) tokens ``(V, N, D)`` (from :func:`build_history_tokens`).
        extero: Raw exteroception tokens ``(V, T*height_dim)`` (from
            :func:`build_extero_tokens`).
        cmd_tokens: Raw strided command tokens ``(V, num_cmd_tokens, cmd_dim)``.
        p_ref: World anchor position ``(V, 3)`` (the current robot base position).
        q_ref: World anchor quaternion ``(V, 4, xyzw)``.
        noise: Sampling noise ``(V, F, D)`` (caller-supplied for determinism).
        history_null: Optional ``(V,)`` bool/float; where set, the graph swaps the history
            content for the learned ``null_history`` embedding, so the generator can be sampled
            before its history window has filled. Silently ignored (with a one-time warning) on
            a graph exported without the ``history_null`` input.

    Returns:
        World-frame prediction dict (same keys as ``flow_matching.online.sample_world``).
    """
    feed = {
        "history": history.astype(np.float32),
        "extero": extero.astype(np.float32),
        "cmd_tokens": cmd_tokens.astype(np.float32),
        "noise": noise.astype(np.float32),
    }
    if ctx.has_history_null:
        nz = (np.zeros((history.shape[0], 1), dtype=np.float32) if history_null is None
              else np.asarray(history_null, dtype=np.float32).reshape(-1, 1))
        feed["history_null"] = nz
    elif history_null is not None and np.any(history_null):
        global _WARNED_NO_NULL
        if not _WARNED_NO_NULL:
            print("[WARN fm_session] This fm.onnx has no 'history_null' input, so the learned "
                  "null-history embedding cannot be used; the generator will condition on a "
                  "not-yet-full history window. Re-export with export_fm_onnx.py.")
            _WARNED_NO_NULL = True
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
