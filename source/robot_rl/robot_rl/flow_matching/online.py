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
        extero_norm: Normalizer for the height-scan token.
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
    extero_norm: Normalizer
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
    extero_sensor_count: int = 1
    """How many exteroception SENSOR readings the checkpoint conditions on (rows).

    1 for the height scan, 2 for the camera pair. Carried beside :attr:`extero_source` for
    the same reason: a consumer checks what it is about to feed the model against the
    checkpoint, not against what the task config happens to build. Defaults, with
    :attr:`extero_sensor_dim`, to the legacy scan a pre-selector checkpoint resolves to.
    """
    extero_sensor_dim: int = 176
    """Width of ONE exteroception reading (rays for the scan, pixels for a camera)."""
    extero_spec: list[dict] | None = None
    """Camera contract behind the exteroception tokens, when :attr:`extero_source` is depth.

    Mount, intrinsics, range and normalisation per camera (``mdp.depth_camera_specs``),
    carried from the recording through the checkpoint.  A deployment can therefore check
    this generator against the tracker sharing the same physical cameras.  ``None`` for
    the height scan, whose grid travels in the tracker's ``height_scan_spec`` instead.
    """
    extero_clip: float | None = None
    """Symmetric clip band this checkpoint was TRAINED under, or ``None`` for the default.

    ``sanitize_extero`` must clamp the live reading exactly as the dataset clamped the
    recorded one, so the band travels with the checkpoint rather than being re-derived. The
    scan's 1.0 m default is too tight for a large grid, and a live reading clipped harder than
    the training data is silently a different sensor.
    """
    extero_image_hw: list[tuple[int, int]] | None = None
    """Per-sensor ``(height, width)`` the exteroception row folds into, or ``None``.

    Only meaningful for the convolutional encoder, and only depth has it. Carried so the
    exported ONNX can state the geometry it was traced with: the flat width alone cannot
    distinguish a 30x26 camera from a 26x30 one, and a convolution folded with the axes
    swapped reads scrambled neighbourhoods with every tensor the right size.
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


def read_fm_extero_meta(checkpoint: str, map_location: str = "cpu") -> dict:
    """Read a checkpoint's exteroception contract WITHOUT building the model.

    For the callers that must know what a generator expects to be fed before the thing that
    would feed it exists -- ``play.py`` has to size the env's camera pair before
    ``gym.make``, and building the full context there would mean loading 400 MB of weights
    to learn two integers.

    DERIVED, never declared: the resolution comes from the artifact rather than a flag, so
    a generator and the cameras built for it cannot disagree by a typo.

    Args:
        checkpoint: Path to an FM ``.pt``.
        map_location: ``torch.load`` map location.

    Returns:
        ``{"extero_source", "extero_sensor_count", "extero_sensor_dim", "extero_encoder",
        "extero_spec", "width", "height"}``. ``width``/``height`` are ``None`` unless the
        checkpoint conditions on depth and states a per-camera geometry.

    Raises:
        ValueError: If a depth checkpoint's cameras disagree on resolution, or if the stated
            geometry does not unfold its own sensor row -- both of which would otherwise
            surface much later as a shape error inside the graph.
    """
    ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    spec = list((ckpt.get("meta") or {}).get("extero_spec") or [])
    out = {
        "extero_source": cfg.extero_source,
        "extero_sensor_count": int(cfg.extero_sensor_count),
        "extero_sensor_dim": int(cfg.extero_sensor_dim),
        "extero_encoder": cfg.resolve_extero_encoder(),
        "extero_spec": spec,
        "width": None,
        "height": None,
    }
    if cfg.extero_source != "depth" or not spec:
        return out
    sizes = {(int(c["width"]), int(c["height"])) for c in spec}
    if len(sizes) != 1:
        raise ValueError(
            f"{checkpoint}: its cameras sample at {sorted(sizes)}; one generator's cameras "
            "share one resolution (they are stacked into one array)."
        )
    width, height = sizes.pop()
    if width * height != out["extero_sensor_dim"]:
        raise ValueError(
            f"{checkpoint}: cameras are {width}x{height} = {width * height} px but the "
            f"sensor row is {out['extero_sensor_dim']} wide."
        )
    out["width"], out["height"] = width, height
    return out


def fm_scan_geometry(spec: list[dict] | None) -> dict | None:
    """The scan geometry a checkpoint was trained on, for building a matching sensor.

    The scan counterpart of :func:`read_fm_extero_meta`'s ``width``/``height``: what
    ``play.py`` needs to construct the generator's own scanner from the artifact rather than
    from a flag, so the two cannot disagree by a typo.

    Args:
        spec: A checkpoint's ``meta["extero_spec"]``, or ``None``.

    Returns:
        ``{"size", "resolution", "offset_xy", "ordering", "height", "width", "privileged"}``,
        or ``None`` when the checkpoint states no scan geometry -- which is the case for every
        pre-field height-scan checkpoint, and correctly means "build nothing, use the
        policy's scanner as before".

    Raises:
        ValueError: If several sensors disagree on the geometry; they would need one sensor.
    """
    if not spec or "resolution" not in spec[0]:
        return None
    keys = {(float(s["resolution"]), tuple(float(v) for v in s["size"]),
             str(s.get("ordering", "xy"))) for s in spec}
    if len(keys) != 1:
        raise ValueError(
            f"the checkpoint states {len(keys)} different scan geometries {sorted(keys)}; "
            "one generator conditions on one grid."
        )
    resolution, size, ordering = keys.pop()
    offset = [float(v) for v in (spec[0].get("offset_pos") or [0.0, 0.0, 0.0])]
    return {
        "size": list(size),
        "resolution": resolution,
        "offset_xy": (offset[0], offset[1]),
        "ordering": ordering,
        "height": int(spec[0]["height"]),
        "width": int(spec[0]["width"]),
        "privileged": bool(spec[0].get("privileged", False)),
    }


#: State-dict prefixes renamed with the exteroception contract: old -> new.
_LEGACY_WEIGHT_PREFIXES: dict[str, str] = {
    "height_in.": "extero_in.",
    "height_type": "extero_type",
}


def _rename_legacy_extero_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map pre-rename exteroception parameter names onto the current ones.

    ``load_state_dict`` is strict, so without this every checkpoint trained before the
    rename becomes unloadable -- including the ones any new encoder has to be A/B'd
    against. One-time and narrow: only these prefixes, and only when the new name is
    absent.

    Args:
        weights: A model or EMA state dict, as saved.

    Returns:
        The same dict with legacy exteroception keys renamed (a copy if anything changed).
    """
    hits = [k for k in weights if any(k.startswith(p) for p in _LEGACY_WEIGHT_PREFIXES)]
    if not hits:
        return weights
    out = dict(weights)
    for key in hits:
        for old, new in _LEGACY_WEIGHT_PREFIXES.items():
            if key.startswith(old):
                out[new + key[len(old):]] = out.pop(key)
                break
    print(f"[INFO load_fm_context] remapped {len(hits)} pre-rename exteroception weight "
          f"key(s) ({', '.join(sorted(hits))}) onto their current names.")
    return out


def _check_encoder_matches_weights(weights: dict[str, torch.Tensor], encoder: str) -> None:
    """Fail with the flag to flip when a checkpoint's weights are the other encoder's.

    ``load_state_dict`` is strict and would catch this on its own -- as a wall of
    missing/unexpected keys that does not say which switch produced it. This says it.

    Args:
        weights: The state dict about to be loaded (already legacy-renamed).
        encoder: The resolved encoder the model was built with.

    Raises:
        ValueError: On a config/weights encoder disagreement.
    """
    has_mlp = any(k.startswith("extero_in.") for k in weights)
    has_cnn = any(k.startswith("extero_cnn.") for k in weights)
    if encoder == "cnn" and has_mlp and not has_cnn:
        raise ValueError(
            "this checkpoint's exteroception weights are the FLAT MLP's (extero_in.*) but the "
            "config says extero_encoder='cnn'. Train with --extero_encoder cnn, or load it "
            "with --extero_encoder mlp -- the two are different architectures, not two "
            "spellings of one."
        )
    if encoder == "mlp" and has_cnn and not has_mlp:
        raise ValueError(
            "this checkpoint's exteroception weights are the CONVOLUTIONAL encoder's "
            "(extero_cnn.*) but the config says extero_encoder='mlp'. Load it with "
            "--extero_encoder cnn."
        )


def _check_extero_image_hw(cfg: FlowMatchingConfig, meta: dict, encoder: str) -> None:
    """Cross-check the config's per-sensor image shapes against the recorded camera contract.

    ``cfg.extero_image_hw`` is derived from ``meta["extero_spec"]`` at dataset build, so at
    load the two must still agree exactly -- ``(H, W)`` and not merely ``H*W``, because a
    transposed camera has the same pixel count and would fold every convolution's
    neighbourhood wrong while leaving all the shapes right.

    Args:
        cfg: The checkpoint's config.
        meta: The checkpoint's ``meta`` dict (``extero_spec`` is read when present).
        encoder: The resolved encoder name.

    Raises:
        ValueError: If the shapes disagree with the contract.
    """
    spec = list(meta.get("extero_spec") or [])
    if encoder != "cnn" or not spec or not cfg.extero_image_hw:
        return
    contract = [(int(c["height"]), int(c["width"])) for c in spec]
    stated = [(int(h), int(w)) for h, w in cfg.extero_image_hw]
    if contract != stated:
        raise ValueError(
            f"config extero_image_hw={stated} disagrees with the recorded camera contract "
            f"{contract}. These must match exactly, not just in pixel count -- a transposed "
            "camera folds every convolution's neighbourhood wrong with every shape intact."
        )


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
    # ``extero_normalizer`` was ``height_normalizer`` before the rename; see _LEGACY_KEYS.
    extero_norm = Normalizer.from_state_dict(
        ckpt["extero_normalizer"] if "extero_normalizer" in ckpt else ckpt["height_normalizer"]
    ).to(dev)

    # The exteroception architecture travels in the config, and is re-verified against the
    # recording's camera contract here -- the same "derive once, check at every boundary"
    # treatment window_dim/state_dim get below. A hand-edited config cannot misdescribe the
    # image shape without this firing.
    encoder = cfg.resolve_extero_encoder()
    cfg.check_extero_encoder()
    cfg.check_attention_mode()
    _check_extero_image_hw(cfg, meta, encoder)
    # Said out loud because nothing else can reveal it: the two attention modes share one
    # parameter set, so a checkpoint carries no structural evidence of which it was trained
    # under. If this line disagrees with what you expect, the config is what to look at.
    print(f"[INFO load_fm_context] attention_mode={cfg.attention_mode} "
          f"(extero_encoder={encoder})")
    # ONE construction site, shared with the trainer. ``encoder`` is the resolved name and is
    # already written back onto cfg by the dataset; assert rather than pass it separately, so
    # there is no second path by which the two could describe different models.
    cfg.extero_encoder = encoder
    model = FlowMatchingTransformer.from_config(cfg).to(dev)
    weights, which = ckpt["model"], "raw"
    if use_ema and ckpt.get("model_ema") is not None:
        weights, which = ckpt["model_ema"], "EMA"
    weights = _rename_legacy_extero_weights(weights)
    _check_encoder_matches_weights(weights, encoder)
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
        extero_norm=extero_norm,
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
        extero_sensor_count=int(cfg.extero_sensor_count),
        extero_sensor_dim=int(cfg.extero_sensor_dim),
        extero_spec=(meta.get("extero_spec") or None),
        extero_image_hw=([(int(h), int(w)) for h, w in cfg.extero_image_hw]
                         if cfg.extero_image_hw else None),
        # The band this checkpoint was TRAINED under, so the live token builder clamps the way
        # the dataset did. Absent in an old checkpoint -> None -> the per-source default, which
        # is what that checkpoint used.
        extero_clip=cfg.extero_clip,
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
    return ctx.extero_norm.normalize(
        sanitize_extero(flat, ctx.extero_source, clip=ctx.extero_clip))


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
    extero: torch.Tensor,
    cmd_tokens: torch.Tensor | None,
    history_null_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
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
        extero: Normalized exteroception tokens ``(V, T*extero_sensor_dim)``.
        cmd_tokens: Normalized command tokens ``(V, num_cmd_tokens, cmd_dim)`` or None.
        history_null_mask: Optional ``(V,)`` bool; where True the history content is
            replaced by the model's learned ``null_history`` embedding. Pass
            :meth:`BatchedHistory.null_mask` to sample before the window has filled --
            which is also the first ``N`` steps after a reset, when the generator has
            emitted nothing to condition on yet.

    Args:
        generator: Optional RNG for the initial noise. The default (None) draws from the
            GLOBAL torch stream, which the environment shares -- so a diagnostic sample taken
            alongside a rollout would shift every later draw in it (observation noise, DR) and
            silently change the trajectory. Pass a dedicated generator for any sample that must
            not perturb the run it is annotating.

    Returns:
        Raw (denormalized) window nodes ``(V, F, D)``.
    """
    if ctx.cmd_dim > 0 and cmd_tokens is None:
        raise ValueError("Checkpoint is command-conditioned but cmd_tokens is None.")
    pred_std = sample_trajectory(ctx.model, history, extero, cmd_tokens, ctx.F, ctx.D,
                                 ctx.num_steps, history_null_mask=history_null_mask,
                                 zero_noise=ctx.zero_noise,
                                 guidance_scale=ctx.guidance_scale,
                                 generator=generator)
    # Project the rotation block back onto SO(3): the model emits 6 free numbers, and the
    # served window is both a reference pose and the next step's history token. The exported
    # ONNX module does the same, so sim and deploy agree.
    return project_window_rotations(
        ctx.state_norm.denormalize(pred_std),
        ctx.nj, ctx.target_num_extra_frames, ctx.reference_blocks,
    )
