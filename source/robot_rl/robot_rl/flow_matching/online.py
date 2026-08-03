# Copyright 2026 Zachary Olkin. All rights reserved.

"""Batched online inference for the flow-matching generator.

Used by the online RL tracking command (``FmTrackingCommand``): maintains a
rolling per-env history of the
robot state, build the ``N`` history (phase) conditioning tokens (in the xyz+yaw
relative-frame representation), the height-scan token, and the strided future
command tokens, sample the ``F``-step future window, and lift it back to the world.

The conditioning must match ``robot_rl.flow_matching.dataset.FutureStateDataset``
exactly: ``N`` real-robot-state history tokens anchored at the most recent step
(index N-1, the xyz+yaw ground frame), a single height-scan token, the strided
command tokens, and the ``F`` predicted FUTURE states (node 0 = current step,
offset ≈ 0). There is no inpainting / known-mask: conditioning is separate tokens.
The torch-free deployment counterpart is ``transfer/deploy/fm_session.py`` (numpy +
onnxruntime); it ports this layout rather than duplicating this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .config import FlowMatchingConfig
from .flow import sample_trajectory
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .transforms import sanitize_extero
from .window import (
    REFERENCE_WINDOW_BLOCK_NAMES,
    project_window_rotations,
    resolve_reference_blocks,
    window_dim,
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
        Ported to numpy in ``transfer/deploy/history.BatchedHistory``.
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

        This is the mask to hand the model as ``history_null_mask``: the generator was trained
        with ``null_cond_prob`` of its samples history-nulled, so a learned ``null_history``
        embedding covers exactly this case and the FM can be sampled from step 0 -- no clip
        warmup, and no fabricated static history.
        """
        if len(self._steps) < self.maxlen:
            return torch.ones_like(self._steps_since_reset, dtype=torch.bool)
        return self._steps_since_reset < self.maxlen

    def stack_padded(self, key: str) -> torch.Tensor:
        """Like :meth:`stack`, but always ``maxlen`` long -- front-padded by repeating the oldest.

        Only the global deque underfill (the first steps of a run) is padded here. Per-env
        staleness -- an env that just reset, whose slots still hold pre-reset states -- is NOT
        repaired, because it does not need to be: those envs are exactly the ones
        :meth:`null_mask` flags, and the model discards their history content in favour of the
        learned null embedding. The newest step stays at index ``-1``, so the caller's anchor
        (``stack(...)[:, -1]``) is always the live state.
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
        N: Number of history (context) tokens -- the reference-window nodes this model
            emitted at the N previous control steps.
        F: Future window length (all F are predicted; node 0 = current step).
        D: Window-node width (``window_dim(nj, target_num_extra_frames,
            reference_blocks)``); history and future share it.
        nj: Number of joints.
        target_num_extra_frames: Extra body frames the generator predicts (and that
            the history carries).
        num_steps: Euler sampling steps.
        zero_noise: Integrate the flow from ``x0 = 0`` rather than Gaussian noise, making
            sampling deterministic given the conditioning (see
            :func:`~robot_rl.flow_matching.flow.sample_trajectory`). Default False =
            training-matched stochastic sampling. Intended for sim2sim / A-B diagnostics,
            where the deploy-side ``GenTrackerPolicy`` exposes the same switch.
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
    reference_blocks: tuple[str, ...] = REFERENCE_WINDOW_BLOCK_NAMES
    """Which reference-window blocks this checkpoint PREDICTS, in packing order.

    The generator's output is the policy's observation, so this is a contract rather than
    a description: a consumer whose ``reference_window`` term packs a different selection
    is being handed a scrambled vector of (possibly) the right width. Defaults to all five,
    which is what a checkpoint predating the selector trained on."""
    extero_source: str = "height_scan"
    """Which sensor the checkpoint was TRAINED on ("height_scan" or "depth").

    Carried on the context so a consumer feeds the right reading based on the
    checkpoint rather than on whatever the task config happens to say.  Defaults to the
    height scan, which is also what a checkpoint predating the selector resolves to.
    """
    fm_snap_every: int | None = None
    """Control steps between snap (re-plan) events in the recording this model trained on.

    Carried so a tracker can verify its own ``fm_update_every`` matches: the horizon ``F`` is
    sized from the re-plan period and the training targets were cut at that cadence, so a
    mismatch is both off-distribution and possibly short of horizon. ``None`` for a
    checkpoint recorded before the field existed.
    """
    extero_spec: list[dict] | None = None
    """Camera contract behind the exteroception tokens, when :attr:`extero_source` is depth.

    Mount, intrinsics, range and normalisation per camera (``mdp.depth_camera_specs``),
    carried from the recording through the checkpoint.  A deployment can therefore check
    this generator against the tracker sharing the same physical cameras.  ``None`` for
    the height scan, whose grid travels in the tracker's ``height_scan_spec`` instead.
    """
    zero_noise: bool = False
    guidance_scale: float = 1.0
    """Classifier-free guidance weight over the HISTORY conditioning (1.0 = off).

    See :func:`~robot_rl.flow_matching.flow.sample_trajectory` for what it does and why it
    is expected to help teacher forcing more than a closed-loop rollout.
    """

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
    zero_noise: bool = False,
    guidance_scale: float = 1.0,
    use_ema: bool = True,
) -> FmOnlineContext:
    """Load a flow-matching checkpoint into an inference-ready context.

    Reads the predicted extra-frame order from the checkpoint ``meta``
    (``target_frame_names``, with the legacy ``state_frame_names`` as a fallback).
    The deploy side reads the same order from the exported ONNX ``fm_meta``.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint or an already-loaded checkpoint dict.
        device: Device to place the model and normalizers on.
        num_sampling_steps: Override the checkpoint's Euler step count (or ``None``).
        map_location: ``torch.load`` map location when ``checkpoint`` is a path.
        zero_noise: Sample deterministically from ``x0 = 0`` (see :class:`FmOnlineContext`).
        guidance_scale: Classifier-free guidance weight over the history (1.0 = off).
        use_ema: Load the averaged ``model_ema`` weights when the checkpoint has them
            (the default, since they are what training selected ``best.pt`` on and what
            would be deployed). Set False to inspect the raw weights instead. Checkpoints
            written before the EMA existed have only raw weights and are unaffected.

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
        height_dim=cfg.height_dim, extero_tokens=cfg.extero_tokens, cmd_dim=cfg.cmd_dim,
        cmd_stride=cfg.cmd_stride, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
    ).to(dev)
    weights, which = ckpt["model"], "raw"
    if use_ema and ckpt.get("model_ema") is not None:
        weights, which = ckpt["model_ema"], "EMA"
    model.load_state_dict(weights)
    model.eval()
    if which == "EMA":
        st = ckpt.get("ema_state") or {}
        print(f"[INFO load_fm_context] using EMA weights (decay {st.get('decay', '?')}, "
              f"{st.get('step', '?')} updates). Pass use_ema=False for the raw ones.")

    # A checkpoint predating the selector has no reference_blocks and was necessarily
    # trained on all five, which is what resolve_reference_blocks(None) returns.
    blocks = resolve_reference_blocks(cfg.reference_blocks)
    target_d = window_dim(nj, target_k, blocks)
    if target_d != cfg.state_dim:
        raise ValueError(
            f"blocks={list(blocks)} + target_frame_names imply state_dim {target_d} "
            f"(nj={nj}, K={target_k}) but the checkpoint config has "
            f"state_dim={cfg.state_dim}."
        )
    if cfg.cmd_dim > 0 and cmd_norm is None:
        raise ValueError("Checkpoint is command-conditioned (cmd_dim>0) but has no cmd_normalizer.")
    cmd_steps = list(range(0, cfg.future_len, cfg.cmd_stride))
    return FmOnlineContext(
        model=model, state_norm=state_norm, cmd_norm=cmd_norm,
        height_norm=height_norm,
        N=cfg.num_history_states, F=cfg.future_len, D=target_d, nj=nj,
        target_num_extra_frames=target_k,
        num_steps=int(num_sampling_steps or cfg.num_sampling_steps),
        cmd_dim=cfg.cmd_dim, cmd_steps=cmd_steps,
        num_cmd_tokens=(len(cmd_steps) if cfg.cmd_dim > 0 else 0),
        joint_names=joint_names, target_frame_names=target_frame_names, device=dev,
        reference_blocks=blocks,
        zero_noise=bool(zero_noise),
        guidance_scale=float(guidance_scale),
        fm_snap_every=(int(meta["fm_snap_every"]) or None) if meta.get("fm_snap_every") else None,
        extero_source=str(cfg.extero_source),
        extero_spec=(meta.get("extero_spec") or None),
    )


# ----------------------------------------------------------------------
# Conditioning construction + world-frame sampling.
# ----------------------------------------------------------------------
def build_history_tokens(ctx: FmOnlineContext, nodes: torch.Tensor) -> torch.Tensor:
    """Normalize the ``N`` history (context) tokens ``(V, N, D)``.

    There is no geometry left to do here. The history is the ``N`` reference-window nodes
    this model emitted at the ``N`` previous control steps, each already expressed relative
    to the robot at its own step -- so a caller pushes back the node it just served and this
    only applies the shared normalizer.

    That is what makes the deploy loop odometry-free: the previous transform-based version
    took world-frame robot state and re-expressed it against the newest step, which needed a
    base pose in some fixed frame.

    Args:
        ctx: The loaded context.
        nodes: Raw (unnormalized) window nodes ``(V, N, D)``, oldest first.

    Returns:
        Normalized history tokens ``(V, N, D)``.

    Raises:
        ValueError: If ``nodes`` does not match the checkpoint's ``(N, D)``.
    """
    if nodes.shape[1:] != (ctx.N, ctx.D):
        raise ValueError(
            f"history must be (V, {ctx.N}, {ctx.D}) window nodes for this checkpoint, "
            f"got {tuple(nodes.shape)}."
        )
    return ctx.state_norm.normalize(nodes)


def build_extero_tokens(ctx: FmOnlineContext, values: torch.Tensor) -> torch.Tensor:
    """Build the normalized exteroception tokens from a raw sensor reading.

    Source-agnostic: the FM's context knows which sensor it was trained on
    (``ctx.extero_source``), so callers pass whichever reading that names and the
    clip band and normalizer follow from it.  Both sources are body-relative and
    yaw-aligned already -- in the FM's xyz+yaw ego frame -- so there is no re-anchoring.

    Args:
        ctx: The loaded context.
        values: Raw reading at the anchor step, shape ``(V, T*dim)`` or ``(V, T, dim)``.
            At deployment this is the live ``mdp.height_scan`` or ``mdp.depth_image``
            reading; in the test bench it is the recorded array at the snap row.

    Returns:
        Normalized tokens, flattened to ``(V, T*dim)`` -- the model reshapes internally.
    """
    flat = values.reshape(values.shape[0], -1)
    return ctx.height_norm.normalize(sanitize_extero(flat, ctx.extero_source))


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
def sample_window(
    ctx: FmOnlineContext,
    history: torch.Tensor,
    height: torch.Tensor,
    cmd_tokens: torch.Tensor | None,
    history_null_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample the ``F``-node future window, denormalized.

    The return value IS the reference window: node 0 is the current step and every node is
    already expressed relative to the robot at this re-plan, in the layout
    ``mdp.observations.reference_window`` feeds the policy. There is no world lift, which is
    the point -- nothing between the network and the policy needs a world pose, in
    simulation or at deploy. Use ``window.window_to_world`` if a DIAGNOSTIC needs world
    coordinates and has the anchor to supply.

    Args:
        ctx: The loaded context.
        history: History tokens ``(V, N, D)`` (from :func:`build_history_tokens`).
        height: Normalized exteroception tokens ``(V, T*height_dim)``.
        cmd_tokens: Normalized command tokens ``(V, num_cmd_tokens, cmd_dim)`` or None.
        history_null_mask: Optional ``(V,)`` bool; where True the history content is
            replaced by the model's learned ``null_history`` embedding. Pass
            :meth:`BatchedHistory.null_mask` to sample before the window has filled --
            which is also the first ``N`` steps after a reset, when the generator has
            emitted nothing to condition on yet.

    Returns:
        Raw (denormalized) window nodes ``(V, F, D)``.
    """
    if ctx.cmd_dim > 0 and cmd_tokens is None:
        raise ValueError("Checkpoint is command-conditioned but cmd_tokens is None.")
    pred_std = sample_trajectory(ctx.model, history, height, cmd_tokens, ctx.F, ctx.D,
                                 ctx.num_steps, history_null_mask=history_null_mask,
                                 zero_noise=ctx.zero_noise,
                                 guidance_scale=ctx.guidance_scale)
    # Project the rotation block back onto SO(3): the model emits 6 free numbers, and the
    # served window is both a reference pose and the next step's history token. The exported
    # ONNX module does the same, so sim and deploy agree.
    return project_window_rotations(
        ctx.state_norm.denormalize(pred_std),
        ctx.nj, ctx.target_num_extra_frames, ctx.reference_blocks,
    )
