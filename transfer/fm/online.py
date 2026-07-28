# Copyright 2026 Zachary Olkin. All rights reserved.

"""Batched online inference for the flow-matching generator (isaaclab-free).

Vendored, deployment-side copy of ``robot_rl.flow_matching.online`` for the
sim2sim / hardware tracker runner: it carries **no** isaaclab or robot_rl import
(the frame math comes from the local :mod:`quat_math` / :mod:`transforms`).

It maintains a rolling per-env history of the robot state, builds the ``N`` history
(phase) conditioning tokens (in the xyz+yaw relative-frame representation), the
height-scan token, and the strided future command tokens, samples the ``F``-step
future window, and lifts it back to the world frame.

The conditioning matches ``robot_rl.flow_matching.dataset.FutureStateDataset``:
``N`` real-robot-state history tokens anchored at the most recent step (index N-1,
the xyz+yaw ground frame), a single height-scan token, the strided command tokens,
and the ``F`` predicted FUTURE states (node 0 = current step, offset ≈ 0). There is
no inpainting / known-mask: conditioning is separate tokens. Keep in lock-step with
``source/robot_rl/robot_rl/flow_matching/online.py``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .config import FlowMatchingConfig
from .flow import sample_trajectory
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .transforms import (
    relative_state as _relative_state,
    sanitize_height_scan,
    state_to_absolute,
    target_state_dim,
)


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

    def null_mask(self) -> torch.Tensor:
        """Per-env ``(N,)`` bool: True where the history window is NOT yet complete.

        Hand this to the model as ``history_null_mask``: training nulled ``null_cond_prob`` of
        its samples, so the learned ``null_history`` embedding covers exactly this case and the
        FM can be sampled from step 0 -- no clip warmup, no fabricated static history.
        """
        if len(self._steps) < self.maxlen:
            return torch.ones_like(self._steps_since_reset, dtype=torch.bool)
        return self._steps_since_reset < self.maxlen

    def stack_padded(self, key: str) -> torch.Tensor:
        """Like :meth:`stack`, but always ``maxlen`` long -- front-padded by repeating the oldest.

        Only the global deque underfill (the first steps of a run) is padded. Per-env staleness
        is not repaired: those envs are the ones :meth:`null_mask` flags, whose history content
        the model replaces with the learned null embedding. The newest step stays at index -1,
        so the caller's anchor is always the live state.
        """
        if not self._steps:
            raise RuntimeError("stack_padded on an empty history: append at least one step first.")
        steps = list(self._steps)
        pad = [steps[0]] * (self.maxlen - len(steps))
        return torch.stack([s[key] for s in (pad + steps)], dim=1)


# ----------------------------------------------------------------------
# Loaded-checkpoint context.
# ----------------------------------------------------------------------
@dataclass
class FmOnlineContext:
    """Everything needed to inference a trained flow-matching checkpoint online.

    Attributes:
        model: The loaded :class:`FlowMatchingTransformer` (eval mode).
        state_norm: Normalizer for the state tokens (history conditioning + future).
        cmd_norm: Normalizer for the command tokens (``None`` if not conditioned).
        height_norm: Normalizer for the height-scan token.
        ground_z: World floor height for the terrain-penetration loss only (no longer feeds any FM token).
        N: Number of history (phase) conditioning tokens.
        F: Future window length (all F are predicted; node 0 = current step).
        D: State width (= ``2*nj + 15*(1 + target_num_extra_frames)``); also history_dim.
        nj: Number of joints.
        target_num_extra_frames: Extra body frames the generator predicts (and that
            the history carries).
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
    height_norm: Normalizer
    ground_z: float
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
    height_norm = Normalizer.from_state_dict(ckpt["height_normalizer"]).to(dev)

    model = FlowMatchingTransformer(
        state_dim=cfg.state_dim, future_len=cfg.future_len,
        num_history_states=cfg.num_history_states, history_dim=cfg.state_dim,
        height_dim=cfg.height_dim, cmd_dim=cfg.cmd_dim,
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
        height_norm=height_norm, ground_z=float(cfg.ground_z),
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
def build_history_tokens(
    ctx: FmOnlineContext,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    frames: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the ``N`` history (phase) conditioning tokens ``(V, N, D)`` (normalized).

    All raw inputs are the ``N`` most-recent real-robot steps ``(V, N, ·)`` in the FM's
    joint/frame order, world frame. The relative frame is the **xyz+yaw** ground frame
    at the most recent step (index ``N-1``), matching training. Returns ONLY the N
    history tokens (no length-F window, no known mask -- history is separate conditioning).

    Args:
        ctx: The loaded context.
        root_pos/root_quat/root_lin_vel/root_ang_vel: Root world quantities ``(V,N,·)``.
        joint_pos/joint_vel: Joint quantities ``(V,N,nj)`` in FM joint order.
        frames: Extra-frame world state ``(V,N,K,13)`` =
            ``[pos(3), quat(4 xyzw), lin_vel(3), ang_vel(3)]`` (required when the model
            predicts extra frames), or None.

    Returns:
        Normalized history tokens ``(V, N, D)``.
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
    return ctx.state_norm.normalize(rel)


def build_height_token(ctx: FmOnlineContext, height_scan: torch.Tensor) -> torch.Tensor:
    """Build the normalized height-scan token ``(V, height_dim)`` from a raw scan.

    The height scan is a vector of ray-cast ground-clearance values (body-relative,
    yaw-aligned -- already in the FM's xyz+yaw ego frame, so no re-anchoring), embedded
    downstream through one MLP into a single token. It replaces the old root-z scalar.

    Args:
        ctx: The loaded context.
        height_scan: Raw height scan at the anchor step, shape ``(V, height_dim)``. At
            deployment this is the live per-ray ground clearance for each env.

    Returns:
        Normalized height-scan token ``(V, height_dim)``.
    """
    return ctx.height_norm.normalize(sanitize_height_scan(height_scan))


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
    history: torch.Tensor,
    height: torch.Tensor,
    cmd_tokens: torch.Tensor | None,
    p_ref: torch.Tensor,
    q_ref: torch.Tensor,
    history_null_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sample the ``F``-step future window and lift it to the world frame.

    All F frames are predicted FUTURE (node 0 = the current step, base offset ≈ 0).
    Every frame is expressed relative to the **xyz+yaw** ground frame at
    ``p_ref``/``q_ref`` (the current robot base pose = the anchor at step N-1), so no
    re-centering is needed.

    Args:
        ctx: The loaded context.
        history: History (phase) tokens ``(V, N, D)`` (from :func:`build_history_tokens`).
        height: Normalized height-scan token ``(V, height_dim)`` (from :func:`build_height_token`).
        cmd_tokens: Normalized command tokens ``(V, num_cmd_tokens, cmd_dim)`` or None.
        p_ref: World reference position ``(V, 3)`` (the current robot base position).
        q_ref: World reference quaternion ``(V, 4, xyzw)``.
        history_null_mask: Optional ``(V,)`` bool; where True the history content is replaced by
            the model's learned ``null_history`` embedding. Pass
            :meth:`BatchedHistory.null_mask` to sample before the window has filled instead of
            falling back to a clip or a fabricated static history.

    Returns:
        World-frame prediction dict with ``root_pos_w (V,F,3)``,
        ``root_quat_w (V,F,4)``, ``root_link_lin_vel_w (V,F,3)``,
        ``root_link_ang_vel_w (V,F,3)``, ``joint_pos (V,F,nj)``,
        ``joint_vel (V,F,nj)`` and, when the model predicts extra frames,
        ``frame_pos_w``/``frame_quat_w``/``frame_lin_vel_w``/``frame_ang_vel_w``
        each ``(V,F,K_target,·)``.
    """
    v = history.shape[0]
    f, d, nj, k = ctx.F, ctx.D, ctx.nj, ctx.target_num_extra_frames
    if ctx.cmd_dim > 0 and cmd_tokens is None:
        raise ValueError("Checkpoint is command-conditioned but cmd_tokens is None.")

    pred_std = sample_trajectory(ctx.model, history, height, cmd_tokens, f, d, ctx.num_steps,
                                 history_null_mask=history_null_mask)
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
