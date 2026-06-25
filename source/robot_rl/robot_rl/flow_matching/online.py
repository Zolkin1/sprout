# Copyright 2026 Zachary Olkin. All rights reserved.

"""Batched online inference for the flow-matching generator.

Shared by the offline visualizer (``scripts/flow_matching/record_viz.py``) and
the online RL tracking command (``GeneratorTrackingCommand``): both maintain a
rolling per-env history of the robot state, build the clean conditioning tokens
(the N most-recent measured states, in the relative-frame representation) plus the
strided command tokens, sample the future window by masked inpainting, and lift it
back to the world frame.

The conditioning must match ``robot_rl.flow_matching.dataset.FutureStateDataset``
exactly: the N most-recent states occupy the first N tokens of a length-F window,
anchored at the most recent step (index N-1); the predicted novel frames are
``N..F-1``. History and target share one full-state representation (every recorded
extra frame), so the conditioning carries the same end-effector frames the model
predicts. This file is kept in lock-step with the deployment-side
``transfer/fm/online.py``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .config import FlowMatchingConfig
from .flow import sample_trajectory
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .transforms import relative_state as _relative_state, state_to_absolute, target_state_dim


# ----------------------------------------------------------------------
# Rolling history buffer (per-env, fixed length, reset-aware).
# ----------------------------------------------------------------------
class BatchedHistory:
    """Fixed-length rolling history of raw world-frame state for ALL envs.

    Holds the most recent ``maxlen`` steps as ``(num_envs, ·)`` tensors. Env
    resets are handled with a per-env validity counter instead of physically
    clearing slots: a window is only used once an env has ``maxlen``
    uninterrupted steps since its last reset. For the PARC-style generator
    ``maxlen`` is the number of conditioning states ``N`` (nominally 2).
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

    def fill(self, step: dict[str, torch.Tensor]) -> None:
        """Warm-start: fill every slot with ``step`` and mark all windows valid.

        Used at deployment where there is no clip-tracking warmup -- pre-loading a
        standing snapshot makes the generator usable from the first control step.
        Kept in lock-step with the vendored ``transfer/fm/online.py``.
        """
        self._steps.clear()
        for _ in range(self.maxlen):
            self._steps.append({k: v.clone() for k, v in step.items()})
        self._steps_since_reset[:] = self.maxlen

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
        state_norm: Normalizer for the state tokens (conditioning and prediction).
        cmd_norm: Normalizer for the command tokens (``None`` if not conditioned).
        N: Number of conditioning (history) states (the first N tokens).
        F: Window length (number of state tokens); novel frames are ``N..F-1``.
        D: State width (= ``2*nj + 15*(1 + target_num_extra_frames)``).
        nj: Number of joints.
        target_num_extra_frames: Extra body frames the generator predicts (and that
            the conditioning carries).
        num_steps: Euler sampling steps.
        cmd_dim: Per-step command width (0 if not command-conditioned).
        cmd_steps: Strided window indices that become command tokens.
        num_cmd_tokens: Number of command tokens (``len(cmd_steps)`` or 0).
        joint_names: FM training joint order.
        target_frame_names: Predicted extra-frame order (== ``frame_names``).
        device: torch device the model/normalizers live on.
    """

    model: FlowMatchingTransformer
    state_norm: Normalizer
    cmd_norm: Normalizer | None
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
    device: torch.device

    @property
    def frame_names(self) -> list[str]:
        """The predicted (target) extra-frame order -- what ``ee_ref`` consumes."""
        return self.target_frame_names

    @property
    def num_extra_frames(self) -> int:
        """Alias for the predicted (target) extra-frame count."""
        return self.target_num_extra_frames


def load_fm_context(
    checkpoint: str | dict,
    device: torch.device | str,
    num_sampling_steps: int | None = None,
    map_location: str = "cpu",
) -> FmOnlineContext:
    """Load a flow-matching checkpoint into an inference-ready context.

    Reads the predicted extra-frame order from the checkpoint ``meta``
    (``target_frame_names``, with the legacy ``state_frame_names`` as a fallback).
    Kept in lock-step with the deployment-side ``transfer/fm/online.py``.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint or an already-loaded checkpoint dict.
        device: Device to place the model and normalizers on.
        num_sampling_steps: Override the checkpoint's Euler step count (or ``None``).
        map_location: ``torch.load`` map location when ``checkpoint`` is a path.

    Returns:
        A populated :class:`FmOnlineContext`.
    """
    ckpt = (
        torch.load(checkpoint, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, str)
        else checkpoint
    )
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    meta = ckpt["meta"]
    nj = int(ckpt["num_joints"])
    joint_names = list(meta["state_joint_names"])

    legacy = list(meta.get("state_frame_names", []) or [])
    target_frame_names = list(meta.get("target_frame_names", legacy) or [])
    target_k = len(target_frame_names)

    dev = torch.device(device)
    state_norm = Normalizer.from_state_dict(ckpt["state_normalizer"]).to(dev)
    cmd_norm = (
        Normalizer.from_state_dict(ckpt["cmd_normalizer"]).to(dev)
        if ckpt.get("cmd_normalizer") is not None
        else None
    )

    model = FlowMatchingTransformer(
        state_dim=cfg.state_dim, future_len=cfg.future_len, cmd_dim=cfg.cmd_dim,
        cmd_stride=cfg.cmd_stride, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    target_d = target_state_dim(nj, target_k)
    if target_d != cfg.state_dim:
        raise ValueError(
            f"target_frame_names implies state_dim {target_d} (nj={nj}, K={target_k}) "
            f"but the checkpoint config has state_dim={cfg.state_dim}."
        )
    if cfg.cmd_dim > 0 and cmd_norm is None:
        raise ValueError("Checkpoint is command-conditioned (cmd_dim>0) but has no cmd_normalizer.")
    cmd_steps = list(range(0, cfg.future_len, cfg.cmd_stride))
    return FmOnlineContext(
        model=model, state_norm=state_norm, cmd_norm=cmd_norm,
        N=cfg.num_history_states, F=cfg.future_len, D=target_d, nj=nj,
        target_num_extra_frames=target_k,
        num_steps=int(num_sampling_steps or cfg.num_sampling_steps),
        cmd_dim=cfg.cmd_dim, cmd_steps=cmd_steps,
        num_cmd_tokens=(len(cmd_steps) if cfg.cmd_dim > 0 else 0),
        joint_names=joint_names, target_frame_names=target_frame_names, device=dev,
    )


# ----------------------------------------------------------------------
# Conditioning construction + world-frame sampling.
# ----------------------------------------------------------------------
def build_cond_states(
    ctx: FmOnlineContext,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    frames: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the conditioning window ``(V, F, D)`` and its known mask ``(V, F)``.

    All raw inputs are the ``N`` most-recent steps ``(V, N, ·)`` in the FM's
    joint/frame order, world frame. The reference frame is the most recent step
    (index ``N-1``), matching training. The N relative states fill tokens ``0..N-1``
    (clean conditioning); tokens ``N..F-1`` are left zero (overwritten by noise at
    sampling) with ``known=False``.

    Args:
        ctx: The loaded context.
        root_pos/root_quat/root_lin_vel/root_ang_vel: Root world quantities ``(V,N,·)``.
        joint_pos/joint_vel: Joint quantities ``(V,N,nj)`` in FM joint order.
        frames: Extra-frame world state ``(V,N,K,13)`` =
            ``[pos(3), quat(4 xyzw), lin_vel(3), ang_vel(3)]`` (required when the model
            predicts extra frames), or None.

    Returns:
        ``(cond, known)`` with shapes ``(V, F, D)`` (normalized) and ``(V, F)`` bool.
    """
    v, n = root_pos.shape[0], root_pos.shape[1]
    nj, k = ctx.nj, ctx.target_num_extra_frames
    m = v * n
    p_ref = root_pos[:, n - 1 : n].expand(v, n, 3).reshape(m, 3)
    q_ref = root_quat[:, n - 1 : n].expand(v, n, 4).reshape(m, 4)

    fpos = fquat = flin = fang = None
    if k > 0:
        if frames is None:
            raise ValueError(f"Context predicts {k} extra frames but no history frames were provided.")
        fr = frames.reshape(m, k, 13)
        fpos, fquat, flin, fang = fr[..., 0:3], fr[..., 3:7], fr[..., 7:10], fr[..., 10:13]

    rel = _relative_state(
        root_pos.reshape(m, 3), root_quat.reshape(m, 4),
        root_lin_vel.reshape(m, 3), root_ang_vel.reshape(m, 3),
        joint_pos.reshape(m, nj), joint_vel.reshape(m, nj),
        p_ref, q_ref,
        frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
    ).reshape(v, n, ctx.D)
    rel = ctx.state_norm.normalize(rel)

    cond = torch.zeros(v, ctx.F, ctx.D, device=rel.device, dtype=rel.dtype)
    cond[:, :n] = rel
    known = torch.zeros(v, ctx.F, dtype=torch.bool, device=rel.device)
    known[:, :n] = True
    return cond, known


def build_cmd_tokens(ctx: FmOnlineContext, future_cmd: torch.Tensor) -> torch.Tensor | None:
    """Build the strided, normalized command tokens from a window command profile.

    Args:
        ctx: The loaded context.
        future_cmd: Raw command profile over the window ``(V, F, cmd_dim)`` (at
            deployment the current command is held constant across the horizon).

    Returns:
        Normalized command tokens ``(V, num_cmd_tokens, cmd_dim)``, or ``None`` when
        the model is not command-conditioned.
    """
    if ctx.cmd_dim == 0:
        return None
    strided = future_cmd[:, ctx.cmd_steps, :]
    return ctx.cmd_norm.normalize(strided)


@torch.no_grad()
def sample_world(
    ctx: FmOnlineContext,
    cond: torch.Tensor,
    known: torch.Tensor,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    cmd_tokens: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sample the future window by masked inpainting and lift it to the world frame.

    The novel (predicted) frames are ``N..F-1``; tokens ``0..N-1`` reproduce the
    conditioning. All frames are expressed relative to ``p_ref``/``q_ref`` (the
    current base pose = the anchor at step N-1).

    Args:
        ctx: The loaded context.
        cond: Conditioning window ``(V, F, D)`` (from :func:`build_cond_states`).
        known: Per-token known mask ``(V, F)`` bool.
        p_ref: World reference position ``(V, 3)`` (the current base position).
        q_ref: World reference quaternion ``(V, 4, xyzw)``.
        cmd_tokens: Normalized command tokens ``(V, num_cmd_tokens, cmd_dim)`` or None.

    Returns:
        World-frame prediction dict with ``root_pos_w (V,F,3)``,
        ``root_quat_w (V,F,4)``, ``root_link_lin_vel_w (V,F,3)``,
        ``root_link_ang_vel_w (V,F,3)``, ``joint_pos (V,F,nj)``,
        ``joint_vel (V,F,nj)`` and, when the model predicts extra frames,
        ``frame_pos_w``/``frame_quat_w``/``frame_lin_vel_w``/``frame_ang_vel_w``
        each ``(V,F,K_target,·)``.
    """
    v = cond.shape[0]
    f, d, nj, k = ctx.F, ctx.D, ctx.nj, ctx.target_num_extra_frames
    if ctx.cmd_dim > 0 and cmd_tokens is None:
        raise ValueError("Checkpoint is command-conditioned but cmd_tokens is None.")

    pred_std = sample_trajectory(ctx.model, cond, known, f, d, ctx.num_steps, cmd_tokens=cmd_tokens)
    pred = ctx.state_norm.denormalize(pred_std)  # (V, F, D)

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
