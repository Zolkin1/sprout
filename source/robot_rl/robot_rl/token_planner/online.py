# Copyright 2026 Zachary Olkin. All rights reserved.

"""Serving the token planner online -- the duck-type of the flow-matching generator.

``FmTrackingCommand`` does not know what a flow model is. It loads a CONTEXT, asks three
shared builders to normalize its three conditioning streams, and calls one function that
returns ``(V, F, D)`` raw reference-window nodes per re-plan. Everything else in that command
-- the history ring, the exteroception sourcing and camera lag, the command horizon, the
cursor and anchor machinery, the strictness masks, the world lift, the viz -- reads plain
numbers off the context (``N``, ``F``, ``D``, ``nj``, ``reference_blocks``, ``extero_*``,
``joint_names``) and cares not at all how the window was produced.

So this module is a context with the SAME FIELD NAMES and a ``sample_window`` with the same
signature. Nothing in the command needed rewriting; only which of two loaders and which of
two samplers it uses.

.. admonition:: What has no analogue, and is refused rather than faked

    * **``noise``.** A flow model integrates from an ``x0`` draw, and simulation tooling
      (the variance probe, the best-of-K harness) injects one to hold the draw fixed across
      arms. An autoregressive planner has no ``x0`` -- its randomness is the categorical draw
      at each of the ``N`` steps, reached through :attr:`TokenPlannerContext.generator`. A
      supplied ``noise`` therefore RAISES: accepting and ignoring it would let a probe
      designed to vary one thing silently vary nothing and report the difference as zero.
    * **``return_flow``.** The RL adjuster reads the final Euler step's velocity field beside
      the window. There is no such quantity here -- the planner's intermediate state is a
      categorical distribution over tokens, not a direction in window space. Also raises.

.. admonition:: ``num_steps`` means something different

    On the flow context it is the Euler step count. Here it is ``N``, the number of
    autoregressive decode steps -- the analogous "how many sequential passes per plan", and
    the quantity per-plan latency is linear in. Carried under the same name because
    ``fm_debug.py`` reports it off the context, and a reader is better served by the real
    number than by a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from robot_rl.flow_matching.normalizer import Normalizer
from robot_rl.flow_matching.window import (
    REFERENCE_WINDOW_BLOCK_NAMES,
    resolve_reference_blocks,
)

from .config import PlannerConfig, TokenizerConfig
from .planner import TokenPlanner, _resolve_extero_channels
from .tokenizer import TrajectoryTokenizer


@dataclass
class TokenPlannerContext:
    """Everything needed to run a trained token planner online.

    Field names mirror :class:`~robot_rl.flow_matching.online.FmOnlineContext` exactly, because
    that is the interface ``FmTrackingCommand`` and the three token builders are written
    against. The docstrings there apply; only the entries below differ in meaning.

    Attributes:
        model: The loaded :class:`~robot_rl.token_planner.planner.TokenPlanner`, in EVAL mode
            -- which its own ``sample`` refuses to run without, since the shipped dropout
            would otherwise make the returned log-probabilities disagree with a later scoring
            pass.
        num_steps: ``N``, the autoregressive decode steps per plan. See the module docstring.
            Read by ``fm_debug.py``, not by the command.
        temperature: Softmax temperature for the categorical draw.
        greedy: Take the argmax instead of sampling. The deployment analogue of the flow
            model's ``zero_noise``: it makes a plan a deterministic function of the
            conditioning, which is what the hardware controller runs.
        generator: Explicit RNG for the draw, so a rollout is replayable. Must live on the
            same device as the conditioning.
        zero_noise / guidance_scale: Carried INERT, for the diagnostics that read them off a
            CONTEXT rather than off a config (``fm_debug.py`` reports both). The command reads
            its own cfg fields, never these. ``zero_noise`` mirrors :attr:`greedy`; guidance
            has no analogue and stays 1.0.
    """

    model: TokenPlanner
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
    extero_source: str = "depth"
    history_source: str = "reference"
    fm_snap_every: int | None = None
    extero_sensor_count: int = 1
    extero_sensor_dim: int = 176
    extero_channels: int = 1
    extero_spec: list[dict] | None = None
    robot_asset: dict | None = None
    extero_clip: float | None = None
    recording_meta: dict | None = None
    depth_latency_probs: tuple[float, ...] | None = None
    advantage_label: int | None = None
    extero_image_hw: list[list[int]] | None = None
    zero_noise: bool = True
    guidance_scale: float = 1.0
    # -- planner-specific --
    temperature: float = 1.0
    greedy: bool = True
    generator: torch.Generator | None = None
    planner_config: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_names(self) -> list[str]:
        """Predicted extra-frame order; the alias the MuJoCo runner reads."""
        return list(self.target_frame_names)

    @property
    def num_extra_frames(self) -> int:
        """Count of those frames."""
        return int(self.target_num_extra_frames)


def _meta_from_recording(path: str) -> dict:
    """The recording contract, read from an ``.h5``'s ATTRS -- no array data.

    Args:
        path: A windowed recording.

    Returns:
        The metadata as a dict, or ``{}`` if the file is not there.
    """
    import os  # noqa: PLC0415

    if not os.path.isfile(path):
        return {}
    from robot_rl.token_planner.data import _recorded_data_io  # noqa: PLC0415

    rdio = _recorded_data_io()
    meta, _ = rdio.read_recording_meta(path)      # (RecordingMeta, num_windows)
    out = dict(vars(meta)) if hasattr(meta, "__dict__") else dict(meta)
    out.setdefault("extero_spec", out.get("depth_camera_spec"))
    out.setdefault("target_frame_names", list(out.get("state_frame_names") or []))
    return out


def _check_context(ctx: "TokenPlannerContext", tok_cfg: TokenizerConfig) -> None:
    """Refuse a context that cannot be served, at LOAD rather than on a live simulator.

    Each of these has a counterpart in
    :func:`~robot_rl.flow_matching.online.load_fm_context`, and each was missing here -- so a
    planner artifact could get further than a flow one on the same defect and fail somewhere
    that named the wrong cause.

    Args:
        ctx: The assembled context.
        tok_cfg: The embedded tokenizer config.

    Raises:
        ValueError: On any disagreement, naming what to look at.
    """
    from robot_rl.flow_matching.window import window_dim  # noqa: PLC0415

    if not ctx.joint_names:
        raise ValueError(
            "the checkpoint states no joint names, so nothing can map the generator's joint "
            "order onto the robot's. They come from the recording contract: re-train with a "
            "current `train_planner.py` (which saves `meta`), or pass `meta_from=<the "
            "recording .h5>`.")
    if len(ctx.target_frame_names) != ctx.target_num_extra_frames:
        raise ValueError(
            f"{len(ctx.target_frame_names)} predicted frame name(s) against "
            f"target_num_extra_frames={ctx.target_num_extra_frames}. The two are one fact; "
            "downstream slicing uses the count and the world lift uses the names.")
    expect = window_dim(ctx.nj, ctx.target_num_extra_frames, ctx.reference_blocks)
    if expect != ctx.D:
        raise ValueError(
            f"node width {ctx.D} disagrees with the layout: {list(ctx.reference_blocks)} at "
            f"nj={ctx.nj}, K={ctx.target_num_extra_frames} is {expect} wide. The tokenizer "
            "and the recording describe different robots.")
    if ctx.D != int(tok_cfg.node_dim):
        raise ValueError(
            f"the planner's node width is {ctx.D} but its tokenizer decodes "
            f"{tok_cfg.node_dim}; the served window would be the wrong shape.")
    if ctx.cmd_dim > 0 and ctx.cmd_norm is None:
        raise ValueError(
            "checkpoint is command-conditioned (cmd_dim > 0) but carries no cmd_normalizer; "
            "the command tokens could not be normalized and the failure would surface inside "
            "a builder at the first re-plan.")
    if ctx.extero_source == "depth" and ctx.extero_clip is None:
        # The flow loader refuses this by name for the same reason: the per-source default
        # resolves to +-5.575 in training and +-2.0 on the deploy side, so an unstated band
        # is ambiguous rather than merely missing.
        raise ValueError(
            "a depth checkpoint must STATE its extero_clip band; this one says None, which "
            "two different fallback tables resolve differently (5.575 here, 2.0 at deploy). "
            "Train with --fm_config so the band is inherited from the generator's config.")


def load_planner_context(
    checkpoint: str | dict,
    device: torch.device | str,
    *,
    meta_from: str | None = None,
    temperature: float = 1.0,
    greedy: bool = True,
    seed: int | None = None,
    use_ema: bool = True,
    map_location: str = "cpu",
) -> TokenPlannerContext:
    """Load a ``planner.pt`` into a servable context.

    Args:
        checkpoint: Path to a checkpoint written by ``scripts/token_planner/train_planner.py``,
            or an already-loaded dict.
        device: Where the model and normalizers should live.
        meta_from: A recording ``.h5`` to take the recording contract from, overriding
            whatever the checkpoint carries. Only needed for a checkpoint written before the
            trainer saved it whose own recording has since moved.
        temperature: Softmax temperature for the token draw.
        greedy: Argmax instead of sampling. Defaults True, matching what a deployment runs.
        seed: Seed for the sampling RNG. ``None`` leaves the generator unset, which
            ``TokenPlanner.sample`` treats as "use the global RNG" -- fine for a greedy run,
            and worth setting for anything stochastic that has to be replayable.
        use_ema: Prefer the EMA weights when the checkpoint carries them.
        map_location: Passed to ``torch.load``.

    Returns:
        The context, with the model frozen and in eval mode.

    Raises:
        KeyError: If the checkpoint is missing a field the context cannot be built without.
        ValueError: If the planner and its embedded tokenizer disagree on the node layout.
    """
    ckpt = (torch.load(checkpoint, map_location=map_location, weights_only=False)
            if isinstance(checkpoint, str) else checkpoint)
    dev = torch.device(device)

    cfg = PlannerConfig.from_dict(ckpt["config"])
    cfg.validate()
    tok_cfg = TokenizerConfig.from_dict(cfg.tokenizer)

    tokenizer = TrajectoryTokenizer(tok_cfg)
    model = TokenPlanner(cfg, tokenizer)
    state = ckpt.get("model_ema") if (use_ema and ckpt.get("model_ema")) else ckpt["model"]
    model.load_state_dict(state)
    model = model.to(dev).eval().requires_grad_(False)

    def _norm(key: str) -> Normalizer | None:
        """Rebuild a saved normalizer onto ``dev``, or ``None`` when absent."""
        blob = ckpt.get(key)
        return None if blob is None else Normalizer.from_state_dict(blob).to(dev)

    state_norm = _norm("state_normalizer")
    if state_norm is None:
        raise KeyError("checkpoint carries no 'state_normalizer'; the window cannot be "
                       "denormalized and every served node would be in the wrong units.")
    extero_norm = _norm("extero_normalizer")
    if extero_norm is None:
        raise KeyError("checkpoint carries no 'extero_normalizer'.")

    # The generator-agnostic half of the contract comes from the FM config the planner was
    # trained under -- which `train_planner.py --fm_config` copied from the generator's own
    # saved config, so these are the numbers the paired generator used.
    fm = dict(ckpt.get("fm_config") or {})
    meta = dict(ckpt.get("meta") or {})
    if not meta and meta_from is None:
        # SELF-HEAL for checkpoints written before the trainer saved `meta`. Everything
        # needed is in the recording's HDF5 ATTRS -- no array is read -- and the checkpoint
        # already records which recording it was, so this needs nothing from the caller. It
        # is a recovery path, not a design: a checkpoint that states its own provenance is
        # the point, and the trainer now writes one.
        first = str(ckpt.get("data", "")).split(",")[0].strip()
        if first:
            meta = _meta_from_recording(first)
            if meta:
                print(f"[tp] this checkpoint carries no 'meta'; recovered the recording "
                      f"contract from {first}")
    if meta_from is not None:
        meta = _meta_from_recording(str(meta_from))
    cmd_steps = list(range(0, int(cfg.future_len), int(cfg.cmd_stride))) if cfg.cmd_dim else []

    snap = meta.get("fm_snap_every")
    if snap is None:
        # Same reading the flow context gives an absent value: unknown, not zero. The
        # cadence check downstream is skipped rather than passed, and says so.
        print("[tp] no 'fm_snap_every' in this checkpoint's recording contract, so the "
              "re-plan cadence cannot be checked against the tracker's. Verify "
              "--fm_update_every matches the recording by hand.")

    ctx = TokenPlannerContext(
        model=model,
        state_norm=state_norm,
        cmd_norm=_norm("cmd_normalizer"),
        extero_norm=extero_norm,
        N=int(cfg.num_history_states),
        F=int(cfg.future_len),
        D=int(cfg.history_dim),
        nj=int(ckpt.get("num_joints", tok_cfg.num_joints)),
        target_num_extra_frames=len(list(ckpt.get("target_frame_names")
                                        or meta.get("target_frame_names") or [])),
        num_steps=int(cfg.num_tokens),
        cmd_dim=int(cfg.cmd_dim),
        cmd_steps=cmd_steps,
        num_cmd_tokens=len(cmd_steps),
        joint_names=list(meta.get("state_joint_names") or []),
        # ONE source for the names, and `target_num_extra_frames` is derived from it below --
        # the flow context guarantees `len(frame_names) == num_extra_frames` by construction
        # and nothing downstream survives them disagreeing.
        target_frame_names=list(ckpt.get("target_frame_names")
                                or meta.get("target_frame_names") or []),
        device=dev,
        reference_blocks=resolve_reference_blocks(
            list(ckpt.get("reference_blocks") or tok_cfg.reference_blocks)),
        extero_source=str(cfg.extero_source),
        history_source=str(fm.get("history_source", "reference")),
        fm_snap_every=(None if snap is None else int(snap)),
        extero_sensor_count=int(cfg.extero_sensor_count),
        extero_sensor_dim=int(cfg.extero_sensor_dim),
        extero_channels=_resolve_extero_channels(cfg),
        extero_spec=meta.get("extero_spec") or meta.get("depth_camera_spec"),
        robot_asset=meta.get("robot_asset"),
        extero_clip=fm.get("extero_clip"),
        recording_meta=meta or None,
        depth_latency_probs=(tuple(fm["depth_latency_probs"])
                             if fm.get("depth_latency_probs") else None),
        extero_image_hw=cfg.extero_image_hw,
        zero_noise=bool(greedy),
        temperature=float(temperature),
        greedy=bool(greedy),
        generator=(None if seed is None
                   else torch.Generator(device=dev.type).manual_seed(int(seed))),
        planner_config=cfg.to_dict(),
    )
    _check_context(ctx, tok_cfg)
    print(
        f"[tp] token planner: N={ctx.N} history, F={ctx.F} nodes, D={ctx.D} wide, "
        f"{ctx.num_steps} decode steps over V={cfg.vocab_size}\n"
        f"[tp] blocks={list(ctx.reference_blocks)} nj={ctx.nj} K={ctx.target_num_extra_frames} "
        f"extero={ctx.extero_source} ({ctx.extero_sensor_count} x {ctx.extero_sensor_dim}, "
        f"{ctx.extero_channels} ch)\n"
        f"[tp] history_from={ctx.history_source} snap_every={ctx.fm_snap_every} "
        f"sampling={'greedy' if ctx.greedy else f'T={ctx.temperature}'} "
        f"weights={'EMA' if (use_ema and ckpt.get('model_ema')) else 'raw'} "
        f"epoch={ckpt.get('epoch')}"
    )
    return ctx


@torch.no_grad()
def sample_window(
    ctx: TokenPlannerContext,
    history: torch.Tensor,
    extero: torch.Tensor,
    cmd_tokens: torch.Tensor | None,
    history_null_mask: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    return_flow: bool = False,
) -> torch.Tensor:
    """Sample the ``F``-node future window, denormalized -- the flow sampler's twin.

    The return value IS the reference window, in the layout
    ``mdp.observations.reference_window`` feeds the policy: node 0 is the current step and
    every node is already relative to the robot at this re-plan. No world lift. The rotation
    block is projected back onto SO(3) inside
    :meth:`~robot_rl.token_planner.planner.TokenPlanner.decode`, at exactly the point the
    flow sampler projects, so the served pose is a real pose on both paths.

    Args:
        ctx: The loaded context.
        history: History tokens ``(V, N, D)``, normalized.
        extero: Normalized exteroception ``(V, T * extero_sensor_dim)``.
        cmd_tokens: Normalized command tokens ``(V, num_cmd_tokens, cmd_dim)`` or ``None``.
        history_null_mask: ``(V,)`` bool; where True the history is replaced by the learned
            null embedding -- the first ``N`` steps after a reset, when nothing has been
            emitted to condition on.
        noise: Must be ``None``. Present only so this is signature-compatible with the flow
            sampler; see below.
        return_flow: Must be ``False``. Likewise.

    Returns:
        Raw (denormalized) window nodes ``(V, F, D)``.

    Raises:
        ValueError: If ``cmd_tokens`` is missing on a command-conditioned checkpoint, or if
            ``noise`` / ``return_flow`` are used. Both are refused rather than ignored: a
            variance probe that injects a fixed ``noise`` to hold the draw constant across
            arms would otherwise vary nothing and report the difference as zero, and the
            adjuster reading a flow field would silently get whatever was in the buffer.
    """
    if ctx.cmd_dim > 0 and cmd_tokens is None:
        raise ValueError("Checkpoint is command-conditioned but cmd_tokens is None.")
    if noise is not None:
        raise ValueError(
            "the token planner takes no `noise`: it integrates nothing, and its randomness "
            "is the categorical draw at each of the N decode steps. Hold a draw fixed with "
            "`greedy=True` or with an explicit `generator` on the context instead -- "
            "accepting a noise tensor and ignoring it would make a probe that varies it "
            "measure nothing and report zero.")
    if return_flow:
        raise ValueError(
            "the token planner has no flow field: its intermediate state is a categorical "
            "distribution over tokens, not a direction in window space. The RL adjuster's "
            "`plan_flow` input is unavailable against this generator.")
    tokens, _, _ = ctx.model.sample(
        history, extero, cmd_tokens, history_null_mask,
        temperature=ctx.temperature, greedy=ctx.greedy, generator=ctx.generator,
    )
    return ctx.model.decode(tokens, ctx.state_norm)
