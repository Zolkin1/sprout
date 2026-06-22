# Copyright 2026 Zachary Olkin. All rights reserved.

"""Batched online inference for the flow-matching generator.

Shared by the offline visualizer (``scripts/flow_matching/record_viz.py``) and
the online RL tracking command (``GeneratorTrackingCommand``): both maintain a
rolling per-env history of the robot state, reconstruct the exact training-time
history token (relative-frame state + command [+ height scan]), sample a future
trajectory, and lift it back to the world frame.

The history token must match ``robot_rl.flow_matching.dataset.FutureStateDataset``
exactly: per past step ``[relative_state | command | height_scan]`` with the
reference frame taken from the most recent step. ``relative_state`` now carries
extra body frames (feet/hands), and the command may be a future-horizon
``target_vel`` (``cmd_array`` checkpoints), whose future steps additionally
condition the decoder.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .config import FlowMatchingConfig
from .flow import sample_trajectory
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .transforms import relative_state as _relative_state, state_to_absolute


# ----------------------------------------------------------------------
# Rolling history buffer (per-env, fixed length, reset-aware).
# ----------------------------------------------------------------------
class BatchedHistory:
    """Fixed-length rolling history of raw world-frame state for ALL envs.

    Holds the most recent ``maxlen`` steps as ``(num_envs, ·)`` tensors. Env
    resets are handled with a per-env validity counter instead of physically
    clearing slots: a window is only used once an env has ``maxlen``
    uninterrupted steps since its last reset.
    """

    def __init__(self, num_envs: int, maxlen: int, device: torch.device):
        """Create an empty history for ``num_envs`` envs of length ``maxlen``."""
        self.maxlen = maxlen
        self._steps: deque[dict[str, torch.Tensor]] = deque(maxlen=maxlen)
        self._steps_since_reset = torch.zeros(num_envs, dtype=torch.long, device=device)

    def append(self, step: dict[str, torch.Tensor]) -> None:
        """Append one step. ``step`` maps each tracked key to a ``(N, ·)`` tensor."""
        self._steps.append(step)
        self._steps_since_reset += 1

    def invalidate(self, env_mask: torch.Tensor) -> None:
        """Restart the validity counter for envs that just reset (``(N,)`` bool)."""
        self._steps_since_reset[env_mask] = 0

    def valid(self) -> torch.Tensor:
        """Per-env mask of complete, reset-free history windows (``(N,)`` bool)."""
        if len(self._steps) < self.maxlen:
            return torch.zeros_like(self._steps_since_reset, dtype=torch.bool)
        return self._steps_since_reset >= self.maxlen

    def stack(self, key: str) -> torch.Tensor:
        """Stack a tracked quantity over the window into ``(N, maxlen, ·)``."""
        return torch.stack([s[key] for s in self._steps], dim=1)


# ----------------------------------------------------------------------
# Loaded-checkpoint context.
# ----------------------------------------------------------------------
@dataclass
class FmOnlineContext:
    """Everything needed to inference a trained flow-matching checkpoint online.

    Attributes:
        model: The loaded :class:`FlowMatchingTransformer` (eval mode).
        target_norm: Normalizer for the future state target (denormalizes samples).
        history_norm: Normalizer for the history token (its command columns also
            normalize the decoder ``future_cmd``).
        H, F, D, Din: history length, future length, state width, history-token width.
        nj: number of joints.
        num_extra_frames: number of extra body frames in the state representation.
        num_steps: Euler sampling steps.
        cmd_dim: command width inside the history token (vel cmd / target_vel).
        future_cmd_dim: per-future-step decoder command width (0 if not conditioned).
        scan_dim: height-scan width inside the history token (0 if none).
        joint_names: FM training joint order.
        frame_names: FM training extra-frame order.
        device: torch device the model/normalizers live on.
    """

    model: FlowMatchingTransformer
    target_norm: Normalizer
    history_norm: Normalizer
    H: int
    F: int
    D: int
    Din: int
    nj: int
    num_extra_frames: int
    num_steps: int
    cmd_dim: int
    future_cmd_dim: int
    scan_dim: int
    joint_names: list[str]
    frame_names: list[str]
    device: torch.device


def load_fm_context(
    checkpoint: str | dict,
    device: torch.device | str,
    num_sampling_steps: int | None = None,
    scan_dim: int = 0,
    map_location: str = "cpu",
) -> FmOnlineContext:
    """Load a flow-matching checkpoint into an inference-ready context.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint or an already-loaded checkpoint dict.
        device: Device to place the model and normalizers on.
        num_sampling_steps: Override the checkpoint's Euler step count (or ``None``).
        scan_dim: Height-scan width present in the history token (0 for flat tasks).
        map_location: ``torch.load`` map location when ``checkpoint`` is a path.

    Returns:
        A populated :class:`FmOnlineContext`.
    """
    ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False) if isinstance(checkpoint, str) else checkpoint
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    meta = ckpt["meta"]
    nj = int(ckpt["num_joints"])
    frame_names = list(meta.get("state_frame_names", []) or [])
    joint_names = list(meta["state_joint_names"])

    dev = torch.device(device)
    target_norm = Normalizer.from_state_dict(ckpt["target_normalizer"]).to(dev)
    history_norm = Normalizer.from_state_dict(ckpt["history_normalizer"]).to(dev)

    model = FlowMatchingTransformer(
        history_in_dim=cfg.history_in_dim, state_dim=cfg.state_dim,
        history_len=cfg.history_len, future_len=cfg.future_len,
        d_model=cfg.d_model, nhead=cfg.nhead, num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers, dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout, cmd_dim=cfg.cmd_dim,
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cmd_dim = cfg.history_in_dim - cfg.state_dim - scan_dim
    if cmd_dim < 0:
        raise ValueError(
            f"Resolved negative command width: history_in_dim={cfg.history_in_dim}, "
            f"state_dim={cfg.state_dim}, scan_dim={scan_dim}."
        )
    return FmOnlineContext(
        model=model, target_norm=target_norm, history_norm=history_norm,
        H=cfg.history_len, F=cfg.future_len, D=cfg.state_dim, Din=cfg.history_in_dim,
        nj=nj, num_extra_frames=len(frame_names),
        num_steps=int(num_sampling_steps or cfg.num_sampling_steps),
        cmd_dim=cmd_dim, future_cmd_dim=cfg.cmd_dim, scan_dim=scan_dim,
        joint_names=joint_names, frame_names=frame_names, device=dev,
    )


# ----------------------------------------------------------------------
# History-token construction + world-frame sampling.
# ----------------------------------------------------------------------
def build_history_token(
    ctx: FmOnlineContext,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    cmd: torch.Tensor,
    frames: torch.Tensor | None = None,
    scan: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the normalized history token ``(V, H, Din)`` for a batch of windows.

    All raw inputs are ``(V, H, ·)`` in the FM's joint/frame order, world frame.
    The reference frame is the most recent step (index ``H-1``), matching training.

    Args:
        ctx: The loaded context.
        root_pos/root_quat/root_lin_vel/root_ang_vel: Root world quantities ``(V,H,·)``.
        joint_pos/joint_vel: Joint quantities ``(V,H,nj)`` in FM joint order.
        cmd: Command columns ``(V,H,cmd_dim)`` (vel cmd / past target_vel).
        frames: Extra-frame world state ``(V,H,K,13)`` =
            ``[pos(3), quat(4 xyzw), lin_vel(3), ang_vel(3)]`` in FM frame order, or None.
        scan: Height-scan columns ``(V,H,scan_dim)`` or None.

    Returns:
        Normalized history token ``(V, H, Din)``.
    """
    v, h = root_pos.shape[0], root_pos.shape[1]
    nj, k = ctx.nj, ctx.num_extra_frames
    m = v * h
    p_ref = root_pos[:, h - 1 : h].expand(v, h, 3).reshape(m, 3)
    q_ref = root_quat[:, h - 1 : h].expand(v, h, 4).reshape(m, 4)

    fpos = fquat = flin = fang = None
    if k > 0:
        if frames is None:
            raise ValueError(f"Context expects {k} extra frames but none were provided.")
        fr = frames.reshape(m, k, 13)
        fpos, fquat, flin, fang = fr[..., 0:3], fr[..., 3:7], fr[..., 7:10], fr[..., 10:13]

    rel = _relative_state(
        root_pos.reshape(m, 3), root_quat.reshape(m, 4),
        root_lin_vel.reshape(m, 3), root_ang_vel.reshape(m, 3),
        joint_pos.reshape(m, nj), joint_vel.reshape(m, nj),
        p_ref, q_ref,
        frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
    ).reshape(v, h, ctx.D)

    parts = [rel, cmd]
    if scan is not None:
        parts.append(scan)
    token = torch.cat(parts, dim=-1)
    return ctx.history_norm.normalize(token)


def normalize_future_cmd(ctx: FmOnlineContext, future_cmd: torch.Tensor) -> torch.Tensor:
    """Normalize the decoder ``future_cmd`` with the history token's command columns.

    Mirrors ``FutureStateDataset``: the future command shares the standardization
    of the past command columns ``[D : D + future_cmd_dim]`` of the history token.

    Args:
        ctx: The loaded context.
        future_cmd: Raw future command ``(V, F, future_cmd_dim)``.

    Returns:
        Normalized future command ``(V, F, future_cmd_dim)``.
    """
    c0 = ctx.D
    c1 = ctx.D + ctx.future_cmd_dim
    mean = ctx.history_norm.mean[c0:c1]
    std = ctx.history_norm.std[c0:c1]
    return (future_cmd - mean) / std


@torch.no_grad()
def sample_world(
    ctx: FmOnlineContext,
    history_token: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    future_cmd: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sample a future trajectory and lift it to the world frame.

    Args:
        ctx: The loaded context.
        history_token: Normalized history token ``(V, H, Din)``.
        p_ref: World reference position ``(V, 3)`` (the anchor each prediction is
            expressed relative to -- the current base position).
        q_ref: World reference quaternion ``(V, 4, xyzw)``.
        future_cmd: Raw future command ``(V, F, future_cmd_dim)`` or None.

    Returns:
        World-frame prediction dict with ``root_pos_w (V,F,3)``,
        ``root_quat_w (V,F,4)``, ``root_link_lin_vel_w (V,F,3)``, ``root_link_ang_vel_w (V,F,3)``,
        ``joint_pos (V,F,nj)``, ``joint_vel (V,F,nj)`` and, when the model predicts
        extra frames, ``frame_pos_w``/``frame_quat_w``/``frame_lin_vel_w``/
        ``frame_ang_vel_w`` each ``(V,F,K,·)``.
    """
    v = history_token.shape[0]
    f, d, nj, k = ctx.F, ctx.D, ctx.nj, ctx.num_extra_frames

    fc = None
    if ctx.future_cmd_dim > 0:
        if future_cmd is None:
            raise ValueError("Checkpoint is future-command conditioned but future_cmd is None.")
        fc = normalize_future_cmd(ctx, future_cmd)

    pred_std = sample_trajectory(ctx.model, history_token, f, d, ctx.num_steps, future_cmd=fc)
    pred = ctx.target_norm.denormalize(pred_std)  # (V, F, D)

    p_ref_f = p_ref[:, None, :].expand(v, f, 3).reshape(v * f, 3)
    q_ref_f = q_ref[:, None, :].expand(v, f, 4).reshape(v * f, 4)
    world = state_to_absolute(pred.reshape(v * f, d), p_ref_f, q_ref_f, nj, num_extra_frames=k)

    out = {
        "root_pos_w": world["root_pos_w"].reshape(v, f, 3),
        "root_quat_w": world["root_quat_w"].reshape(v, f, 4),
        "root_link_lin_vel_w": world["root_link_lin_vel_w"].reshape(v, f, 3),
        "root_link_ang_vel_w": world["root_link_ang_vel_w"].reshape(v, f, 3),
        "joint_pos": world["joint_pos"].reshape(v, f, nj),
        "joint_vel": world["joint_vel"].reshape(v, f, nj),
    }
    if k > 0:
        out["frame_pos_w"] = world["frame_pos_w"].reshape(v, f, k, 3)
        out["frame_quat_w"] = world["frame_quat_w"].reshape(v, f, k, 4)
        out["frame_lin_vel_w"] = world["frame_lin_vel_w"].reshape(v, f, k, 3)
        out["frame_ang_vel_w"] = world["frame_ang_vel_w"].reshape(v, f, k, 3)
    return out
