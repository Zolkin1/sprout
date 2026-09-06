# Copyright 2026 Zachary Olkin. All rights reserved.

"""Train the trajectory tokenizer -- stage A of the tokenized autoregressive planner.

Fits a chunk-local autoencoder with an FSQ bottleneck to the SAME future-window targets the
flow-matching generator regresses, on the same recordings and the same episode-wise split,
so the AR planner that consumes its tokens can be compared with the FM like for like.

Usage mirrors ``scripts/flow_matching/train.py``::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/train_tokenizer.py \\
        --data <recorded>.h5,<augmented>.h5 --run_name d4_v1

Every flag that shadows a :class:`~robot_rl.token_planner.config.TokenizerConfig` field
renders its default from that dataclass, never from a literal, so the config is the single
place a default lives. That is the FM trainer's convention and its
``test_train_cli_defaults.py`` exists because the two drifted once.

Isaac-free apart from the dataset build (``robot_rl.flow_matching`` imports
``isaaclab.utils.math``, though nothing from ``isaacsim``), so this runs wherever the FM
trainer runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from motion_gen.features import _REPO_ROOT

sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))

from provenance import dump_params_yaml, git_snapshot  # noqa: E402

from robot_rl.flow_matching.ema import WeightEMA  # noqa: E402
from robot_rl.flow_matching.normalizer import Normalizer  # noqa: E402
from robot_rl.token_planner.config import TokenizerConfig  # noqa: E402
from robot_rl.token_planner.data import (  # noqa: E402
    STATE_SCALE_NAMES,
    ChunkDataset,
    build_chunk_data,
    cache_key,
    load_chunk_cache,
    save_chunk_cache,
)
from robot_rl.token_planner.metrics import (  # noqa: E402
    ReconAccumulator,
    channel_group_labels,
    format_report,
)
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer  # noqa: E402


def parse_args() -> tuple[TokenizerConfig, argparse.Namespace]:
    """Build the CLI and return ``(config, runtime_args)``.

    Returns:
        The tokenizer config (data-dependent fields still unresolved) and the runtime-only
        arguments, which deliberately do NOT live on the config: they describe this
        invocation rather than the artifact.
    """
    d = TokenizerConfig()
    p = argparse.ArgumentParser(description="Train the trajectory tokenizer.")
    p.add_argument("--data", default="",
                   help="Comma-separated windowed .h5 recordings, concatenated in order. "
                        "Use the SAME set the generator was trained on. Optional when "
                        "--cache names an existing cache, which carries its own record of "
                        "what it was built from; given anyway, the two must agree.")
    p.add_argument("--build_cache_only", action="store_true",
                   help="Build --cache and exit without training. The point of the split is "
                        "that the build belongs where the recordings are and the training "
                        "belongs where you iterate; running an epoch just to get the cache "
                        "would leave a tokenizer.pt and a gating report that mean nothing "
                        "and could later be mistaken for a result.")
    p.add_argument("--verify_data_sha256", action="store_true",
                   help="Re-hash the --data files and require them to match what the cache "
                        "was built from. Off by default because it is a full read of tens of "
                        "GB, and impossible at all on a box the recordings were never copied "
                        "to -- which is the normal case, since the cache is the portable "
                        "artifact and the recordings are not.")
    p.add_argument("--logdir", default="logs/token_planner")
    p.add_argument("--run_name", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache", default="",
                   help="Path to a chunk cache (.pt). Written if absent, reused if present "
                        "and matching. Skips minutes of h5 decompression per ablation run.")
    # Opt-OUT, matching scripts/flow_matching/train.py (and the RL side, whose agent cfgs
    # ship `logger: wandb`). A quantization run's decisive numbers are the codebook
    # utilisation and the per-group physical RMSEs, and comparing those ACROSS the D / C /
    # level-table ablations is the whole point of stage A -- which is precisely what a
    # per-run TensorBoard directory is bad at.
    p.add_argument("--wandb", action="store_true", default=True,
                   help="Log to Weights & Biases (default; use --no_wandb to disable).")
    p.add_argument("--no_wandb", dest="wandb", action="store_false",
                   help="Disable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="robot_rl_token_planner")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--residual", action="store_true",
                   help="Train on FM RESIDUALS rather than on the clip's own future: the "
                        "chunks are `clip - generator_plan` and the normalizer is the "
                        "residual's. --cache must name a build from "
                        "scripts/token_planner/build_residual_cache.py, and the two arms are "
                        "refused in BOTH directions -- their tensors have identical shape and "
                        "completely different meaning, so a mix-up would train to completion "
                        "and report plausible numbers about the wrong quantity.")
    p.add_argument("--fm_config", default="",
                   help="A flow-matching run's params/fm_config.yaml. Supplies the DECLARED "
                        "STATE SCALES, the only config values that reach the state "
                        "normalizer -- and the normalizer is what makes a reconstruction "
                        "error mean the same thing in both stages. Without it the scales are "
                        "whatever FlowMatchingConfig currently defaults to, which is the trap "
                        "CLAUDE.md records four times over for other fields. The resolved "
                        "values are recorded and keyed either way.")
    p.add_argument("--match_val_episodes", default="",
                   help="Path to a flow-matching run's params/data.yaml. Asserts this "
                        "build's held-out episode ids are IDENTICAL to that run's, which is "
                        "what 'the same split as the generator' has to mean if the AR-vs-FM "
                        "comparison is to be paired. Cheap, and it turns the claim into a "
                        "check -- the split depends on the seed, the val fraction, the "
                        "filters AND the order the recordings were concatenated in, and a "
                        "mismatch in any of them leaves every tensor the right shape.")
    p.add_argument("--load_rows", type=int, default=0,
                   help="SMOKE TEST ONLY: load only the first N windows of each recording. "
                        "This CHANGES THE SPLIT, so a run using it is not comparable with "
                        "anything; it is stamped into the provenance and the cache key and "
                        "printed as a warning for exactly that reason.")

    # -- data / layout --
    p.add_argument("--with_anchor_mask", action="store_true",
                   help="Join the per-node anchor mask and the window-to-terrain frame angle "
                        "back from the clip library at BUILD time, so --pinned_axis_weight "
                        "has something to weight. Costs one library open and its sha256 "
                        "check, and makes a cache that older runs can still read.")
    p.add_argument("--chunk_len", type=int, default=d.chunk_len,
                   help="Nodes per chunk H; must equal the generator's future_len.")
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Held-out fraction of EPISODES. Match the generator's.")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--split_seed", type=int, default=0,
                   help="Episode-split seed. Match the generator's cfg.seed.")
    p.add_argument("--skip_distractors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip_families", default=None,
                   help="Comma-separated families to drop; unset takes the FM's own default "
                        "(family_jump_off). Pass '' to drop none.")
    p.add_argument("--reference_blocks", default="",
                   help="Comma-separated block selection; empty takes the recording's.")

    # -- bottleneck / architecture --
    p.add_argument("--downsample", type=int, default=d.downsample,
                   help="Nodes per token D; must divide --chunk_len.")
    p.add_argument("--fsq_levels", default=",".join(str(v) for v in d.fsq_levels))
    p.add_argument("--channels", type=int, default=d.channels)
    p.add_argument("--num_blocks_high", type=int, default=d.num_blocks_high)
    p.add_argument("--num_blocks_low", type=int, default=d.num_blocks_low)
    p.add_argument("--kernel_size", type=int, default=d.kernel_size)
    p.add_argument("--pad_mode", default=d.pad_mode, choices=("replicate", "reflect", "zeros"))
    p.add_argument("--upsample_mode", default=d.upsample_mode, choices=("nearest", "transpose"))

    # -- loss / training --
    p.add_argument("--lambda_vel", type=float, default=d.lambda_vel)
    p.add_argument("--lambda_acc", type=float, default=d.lambda_acc)
    p.add_argument("--pinned_axis_weight", type=float, default=d.pinned_axis_weight,
                   help="Reconstruction weight on the terrain-PINNED root DOFs -- the "
                        "generator's pinned_axis_loss_weight, ported. Height on flat, "
                        "downrange + height + yaw on stairs and boxes. Needs a chunk build "
                        "carrying the anchor mask (--with_anchor_mask); a cache without one "
                        "is refused rather than silently unweighted. Renormalized to mean 1 "
                        "together with --base_pose_weight.")
    p.add_argument("--base_pose_weight", type=float, default=d.base_pose_weight,
                   help="Reconstruction weight on the base_pose block (position + 6-D "
                        "orientation, 9 channels); every other channel stays 1. The vector "
                        "is renormalized to mean 1, so this is a pure ratio and does not "
                        "also move the effective learning rate.")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--lr_min", type=float, default=d.lr_min)
    p.add_argument("--weight_decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup_steps", type=int, default=d.warmup_steps)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--num_epochs", type=int, default=d.num_epochs)
    p.add_argument("--grad_clip", type=float, default=d.grad_clip)
    p.add_argument("--ema_decay", type=float, default=d.ema_decay)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=d.bf16,
                   help="bf16 autocast. OFF by default, and that is a correctness choice: "
                        "under autocast the encoder emits bf16 pre-activations, and rounding "
                        "those to the FSQ lattice moves 5.9%% of the emitted TOKENS against "
                        "the fp32 answer -- which are exactly stage B's targets, produced "
                        "outside autocast. The model is 5.5M parameters, so the speed is not "
                        "worth it, and the FM trainer is fp32 throughout as well.")

    args = p.parse_args()
    cfg = TokenizerConfig(
        bf16=args.bf16,
        chunk_len=args.chunk_len,
        fsq_levels=[int(v) for v in args.fsq_levels.split(",") if v.strip()],
        downsample=args.downsample,
        channels=args.channels,
        num_blocks_high=args.num_blocks_high,
        num_blocks_low=args.num_blocks_low,
        kernel_size=args.kernel_size,
        pad_mode=args.pad_mode,
        upsample_mode=args.upsample_mode,
        lambda_vel=args.lambda_vel,
        lambda_acc=args.lambda_acc,
        base_pose_weight=args.base_pose_weight,
        pinned_axis_weight=args.pinned_axis_weight,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        seed=args.seed,
    )
    # BEFORE anything reads a recording: a bad (H, D) pair, or an empty level table, would
    # otherwise cost the whole build -- minutes, tens of GB, or a multi-gigabyte cache load
    # -- before raising. TrajectoryTokenizer validates at construction for the same reason,
    # and building first defeats it.
    cfg.validate()
    return cfg, args


def scheduled_lr(cfg: TokenizerConfig, epoch: int, step: int) -> float:
    """Per-step warmup into a per-epoch cosine decay to :attr:`TokenizerConfig.lr_min`.

    Closed form rather than a stateful ``lr_scheduler``, matching the FM trainer: the
    learning rate is then a pure function of ``(epoch, step)`` and cannot be desynced by a
    resume or by an extra ``scheduler.step()``.

    Args:
        cfg: The config (``lr``, ``lr_min``, ``num_epochs``, ``warmup_steps``).
        epoch: Zero-based epoch.
        step: Zero-based GLOBAL step.

    Returns:
        The learning rate for this step.
    """
    progress = min(max(float(epoch) / max(int(cfg.num_epochs), 1), 0.0), 1.0)
    lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * progress))
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        lr *= float(step + 1) / float(cfg.warmup_steps)
    return lr


def pinned_axis_terms(
    err: torch.Tensor,
    sq: torch.Tensor,
    w: torch.Tensor,
    anchor_mask: torch.Tensor,
    mask_frame_yaw: torch.Tensor | None,
    weight: float,
    state_std: torch.Tensor | None,
    cfg: TokenizerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The terrain-pinned root-DOF weighting, as ``(extra_total, extra_w_sum)``.

    A port of :func:`robot_rl.flow_matching.flow.pinned_axis_flow_loss` onto the tokenizer's
    reconstruction error. It is the same quantity computed on the same chunk layout, so the
    generator's reasoning carries over verbatim and is not restated here beyond what a reader
    needs; three things stop it being a channel mask:

    * **The pinned planar axis is not a channel.** ``downrange`` is named in the TERRAIN's
      frame while the chunk's x/y are the snap robot's yaw frame, so the weight is a
      quadratic form ``I + (w-1) u u^T``.
    * **x and y are standardized differently** (0.442 vs 0.102 on the shipped teacher, 4.3x),
      so the error is projected in METRES and divided by the std ALONG that direction.
      Projecting the already-normalized error onto the normalized image of the metric axis
      would leak most of a downrange weight onto lateral error.
    * **Yaw is not a channel** -- the orientation block is a 6-D rotation, so it is weighted
      only when roll, pitch AND yaw are all pinned. Flat frees yaw and gets no orientation
      weight.

    The height and orientation parts ARE axis-aligned, so they are folded into ``w`` in
    place. The planar part is a rank-1 bump on the 2x2 xy block and contributes a SCALAR per
    node, returned separately rather than broadcast across every channel.

    Args:
        err: ``(B, H, D)`` reconstruction error in NORMALIZED units.
        sq: ``err ** 2``, precomputed.
        w: ``(B, H, D)`` weight tensor, MUTATED in place for the axis-aligned parts.
        anchor_mask: ``(B, H)`` int64 anchor bitmasks; bit CLEAR = that DOF is pinned.
        mask_frame_yaw: ``(B,)`` angle from the chunk's frame to the terrain's, or ``None``
            to treat them as aligned.
        weight: Multiplier on pinned DOFs.
        state_std: ``(D,)`` per-channel std, for the anisotropy correction. ``None`` falls
            back to isotropic.
        cfg: The config, for the block layout.

    Returns:
        ``(extra_total, extra_w_sum)`` -- the planar term's contribution to the weighted sum
        of squares and to the trace the mean divides by. Both are zero when nothing is
        planar-pinned.
    """
    from robot_rl.flow_matching.flow import _ANCHOR_BIT  # noqa: PLC0415
    from robot_rl.flow_matching.window import window_block_slices  # noqa: PLC0415

    slices = window_block_slices(int(cfg.num_joints), int(cfg.num_ee_frames),
                                 list(cfg.reference_blocks))
    zero = torch.zeros((), dtype=err.dtype, device=err.device)
    if "base_pose" not in slices:
        return zero, zero
    base = slices["base_pose"].start
    extra = float(weight) - 1.0

    def is_pinned(name: str) -> torch.Tensor:
        """``(B, H)`` float: 1 where that DOF is FIXED (bit CLEAR = pinned)."""
        return ((anchor_mask >> _ANCHOR_BIT[name]) & 1 == 0).to(err.dtype)

    w[..., base + 2] = w[..., base + 2] * (1.0 + extra * is_pinned("height"))
    rot_pinned = is_pinned("roll") * is_pinned("pitch") * is_pinned("yaw")
    w[..., base + 3:base + 9] = w[..., base + 3:base + 9] * (
        1.0 + extra * rot_pinned).unsqueeze(-1)

    p_dr, p_lat = is_pinned("downrange"), is_pinned("lateral")
    if float((p_dr + p_lat).sum()) == 0.0:
        return zero, zero
    std = torch.ones(2, dtype=err.dtype, device=err.device)
    if state_std is not None:
        std = state_std[base:base + 2].to(err.dtype).to(err.device)
    ang = (torch.zeros_like(p_dr) if mask_frame_yaw is None
           else mask_frame_yaw.to(err.dtype).to(err.device).unsqueeze(-1).expand_as(p_dr))
    e_m = err[..., base:base + 2] * std                       # (B, H, 2) metres
    total, w_sum = zero, zero
    for pinned, (ux, uy) in ((p_dr, (torch.cos(ang), torch.sin(ang))),
                             (p_lat, (-torch.sin(ang), torch.cos(ang)))):
        if float(pinned.sum()) == 0.0:
            continue
        along = e_m[..., 0] * ux + e_m[..., 1] * uy           # (B, H) metres along the axis
        std_u = ((std[0] * ux) ** 2 + (std[1] * uy) ** 2).sqrt().clamp_min(1e-12)
        total = total + (extra * pinned * (along / std_u) ** 2).sum()
        # A rank-1 bump of (weight - 1) adds exactly that to the 2x2 block's trace, and the
        # trace is what the mean must divide by.
        w_sum = w_sum + extra * pinned.sum()
    return total, w_sum


#: ``(node_dim,)`` loss weights per config+device, so the vector is built once rather than
#: per batch. Keyed on everything that changes it.
_CHANNEL_WEIGHT_CACHE: dict[tuple, torch.Tensor] = {}


def channel_weights(cfg: TokenizerConfig, device: torch.device) -> torch.Tensor | None:
    """Per-channel reconstruction weights, or ``None`` when every channel is weighted 1.

    ``base_pose_weight`` upweights the base position and the 6-D orientation TOGETHER -- one
    block, because they are one pose and the tracker reads them as one. The vector is
    normalized to mean 1 (see :attr:`TokenizerConfig.base_pose_weight` for why), so the plain
    unweighted path and a weighted one put the same total scale into the gradient.

    Args:
        cfg: The config; the weight, the block selection and the joint/frame counts.
        device: Where the loss is computed.

    Returns:
        ``(node_dim,)`` weights, or ``None`` at weight 1 -- which keeps the untouched
        ``F.mse_loss`` path rather than multiplying by a vector of ones.

    Raises:
        ValueError: If the resolved block widths do not sum to ``node_dim``, which would
            silently weight the wrong channels.
    """
    w = float(cfg.base_pose_weight)
    if w == 1.0:
        return None
    key = (w, int(cfg.node_dim), int(cfg.num_joints), int(cfg.num_ee_frames),
           tuple(cfg.reference_blocks), str(device))
    hit = _CHANNEL_WEIGHT_CACHE.get(key)
    if hit is not None:
        return hit
    from robot_rl.flow_matching.window import window_blocks  # noqa: PLC0415

    blocks = window_blocks(int(cfg.num_joints), int(cfg.num_ee_frames),
                           list(cfg.reference_blocks))
    total = sum(width for _, width in blocks)
    if total != int(cfg.node_dim):
        raise ValueError(
            f"the reference blocks {cfg.reference_blocks} are {total} channels wide but "
            f"node_dim is {cfg.node_dim}; the weights would land on the wrong channels."
        )
    vec = torch.ones(int(cfg.node_dim), device=device)
    i = 0
    for name, width in blocks:
        if name == "base_pose":
            vec[i:i + width] = w
        i += width
    vec = vec / vec.mean()
    _CHANNEL_WEIGHT_CACHE[key] = vec
    return vec


def _mse(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """Mean squared error over all axes, optionally weighted along the CHANNEL axis.

    Args:
        a: Prediction, ``(B, T, node_dim)``.
        b: Target, same shape.
        w: ``(node_dim,)`` weights, or ``None`` for the plain mean.

    Returns:
        The scalar loss.
    """
    if w is None:
        return torch.nn.functional.mse_loss(a, b)
    return (((a - b) ** 2) * w).mean()


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, cfg: TokenizerConfig,
    batch: dict[str, torch.Tensor] | None = None,
    state_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalized-space reconstruction loss, plus its terms for logging.

    ``L_rec`` alone is v1. The finite-difference terms exist because a quantization
    bottleneck's error is biased towards high frequency -- it tends to produce a *jittery*
    trajectory rather than an offset one -- and the tracker feels jitter far more than it
    feels a constant bias. They stay OFF by default so the first gating number is
    attributable to one thing.

    Note on the rotation block: ``base_ori`` is the 6-D representation, two orthonormal
    columns, exempt from standardization and O(1) by construction. A plain MSE on it is the
    Frobenius error of those columns, which is a monotone surrogate for the geodesic angle at
    the errors that matter here; the spec's rotation-aware term becomes worth adding only if
    ``base_ori`` turns out to be the group that fails the gate.

    Args:
        recon: ``(B, H, D)`` reconstruction.
        target: ``(B, H, D)`` ground truth.
        cfg: The config, for the two lambdas and the two weights.
        batch: The loader's item, for ``anchor_mask`` / ``mask_frame_yaw``. Only read when
            ``pinned_axis_weight != 1``.
        state_std: ``(D,)`` per-channel std, for the pinned planar term's anisotropy
            correction.

    Returns:
        ``(loss, terms)``.

    Raises:
        ValueError: If ``pinned_axis_weight != 1`` but the batch carries no anchor mask --
            the run would otherwise train unweighted under a name that says otherwise.
    """
    w = channel_weights(cfg, recon.device)
    if float(cfg.pinned_axis_weight) != 1.0:
        if batch is None or "anchor_mask" not in batch:
            raise ValueError(
                "pinned_axis_weight != 1 needs the per-node anchor mask, which this chunk "
                "build does not carry. Rebuild the cache with --with_anchor_mask."
            )
        err = recon - target
        sq = err ** 2
        wt = (torch.ones_like(sq) if w is None
              else w.to(sq.dtype).expand_as(sq).clone())
        pin_total, pin_wsum = pinned_axis_terms(
            err, sq, wt, batch["anchor_mask"].to(recon.device),
            batch.get("mask_frame_yaw"), float(cfg.pinned_axis_weight), state_std, cfg)
        # ONE renormalization over the COMBINED weights, so base_pose_weight and
        # pinned_axis_weight compose without either one moving the loss scale.
        rec = ((wt * sq).sum() + pin_total) / (wt.sum() + pin_wsum).clamp_min(1e-12)
        loss = rec
        terms = {"rec": float(rec.detach()), "total": float(rec.detach())}
        return loss, terms
    rec = _mse(recon, target, w)
    loss = rec
    terms = {"rec": float(rec.detach())}
    if cfg.lambda_vel > 0.0:
        v = _mse(recon.diff(dim=1), target.diff(dim=1), w)
        loss = loss + cfg.lambda_vel * v
        terms["vel"] = float(v.detach())
    if cfg.lambda_acc > 0.0:
        a = _mse(
            recon.diff(dim=1).diff(dim=1), target.diff(dim=1).diff(dim=1), w)
        loss = loss + cfg.lambda_acc * a
        terms["acc"] = float(a.detach())
    terms["total"] = float(loss.detach())
    return loss, terms


@torch.no_grad()
def evaluate(
    model: TrajectoryTokenizer, loader: DataLoader, device: torch.device,
    acc_factory, cfg: TokenizerConfig, state_std: torch.Tensor | None = None,
    residual: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run a full pass and return ``(report, scalars)``.

    Args:
        model: The tokenizer, already in the weights to be measured.
        loader: A validation loader over :class:`ChunkDataset`.
        device: Where to run.
        acc_factory: Zero-argument factory for a fresh :class:`ReconAccumulator`.
        cfg: The config, for the loss terms.
        state_std: ``(D,)`` per-channel std, for the pinned planar term.
        residual: The chunks are FM residuals, so ``ori_geodesic_rad`` is dropped -- the
            angle between two DIFFERENCES of rotation frames is not a rotation angle, and
            reporting one would put a plausible number on a meaningless quantity. The
            orientation error that does mean something is measured in PLAN space, after the
            residual is composed back onto the plan; see
            :func:`~robot_rl.token_planner.residual.plan_space_report`.

    Returns:
        ``(report, scalars)`` -- the full nested report, and the flat scalars for
        TensorBoard and checkpoint selection.
    """
    model.eval()
    acc = acc_factory()
    loss_sum, n = 0.0, 0
    for batch in loader:
        nodes = batch["nodes"].to(device, non_blocking=True)
        recon, ids = model(nodes)
        loss, _ = reconstruction_loss(recon.float(), nodes, cfg, batch, state_std)
        loss_sum += float(loss) * nodes.shape[0]
        n += int(nodes.shape[0])
        acc.update(nodes, recon.float(), ids, batch["family"])
    rep = acc.report()
    scalars: dict[str, float] = {"loss": loss_sum / max(n, 1)}
    for g, v in rep["rmse"].items():
        scalars[f"rmse/{g}"] = v
    for g, v in rep["node_rmse_p99"].items():
        scalars[f"p99/{g}"] = v
    for g, ends in rep["boundary_penalty"].items():
        for end, v in ends.items():
            scalars[f"edge/{g}/{end}"] = v
    for k, v in rep["codebook"].items():
        scalars[f"codebook/{k}"] = v
    if residual:
        rep.pop("ori_geodesic_rad", None)
    else:
        scalars["ori_geodesic_rad"] = rep["ori_geodesic_rad"]
    # ONE selection scalar, so "best" is not a judgement call made differently each run.
    # Mean of the group RMSEs in their own units: the groups are already O(1)-comparable
    # (metres, rad, rad/s all land in the same decade on this data) and weighting them would
    # be an unmeasured choice smuggled into checkpoint selection.
    scalars["rmse_mean"] = float(np.mean(list(rep["rmse"].values())))
    return rep, scalars


def resolve_chunks(cfg: TokenizerConfig, args: argparse.Namespace):
    """Load or build the chunk data, honouring ``--cache``.

    Args:
        cfg: The config; ``node_dim``, ``num_joints``, ``num_ee_frames``, ``dt``,
            ``replan_period`` and ``reference_blocks`` are written back from the data.
        args: Runtime args.

    Returns:
        The :class:`~robot_rl.token_planner.data.ChunkData`.
    """
    paths = [p for p in args.data.split(",") if p.strip()]
    blocks = ([b for b in args.reference_blocks.split(",") if b.strip()]
              if args.reference_blocks else None)
    # Resolved here, not left as None: the cache key records the concrete list, so an
    # unresolved None would mismatch a perfectly good cache -- and, worse, a later change to
    # DEFAULT_SKIP_FAMILIES would silently re-resolve what an old cache is compared against.
    from robot_rl.flow_matching.config import DEFAULT_SKIP_FAMILIES

    families = (list(DEFAULT_SKIP_FAMILIES) if args.skip_families is None
                else [f for f in args.skip_families.split(",") if f.strip()])
    if args.residual:
        return _resolve_residual_chunks(cfg, args, blocks)
    scales = _state_scales_from(args.fm_config)

    def _build():
        if args.load_rows > 0:
            print(f"[tok] *** --load_rows {args.load_rows}: SMOKE TEST. The episode split is "
                  "computed over a truncated recording, so nothing measured here is "
                  "comparable with a full run. ***")
            _patch_row_limit(args.load_rows)
        chunks = build_chunk_data(
            paths, chunk_len=cfg.chunk_len, seed=args.split_seed,
            val_fraction=args.val_fraction, stride=args.stride,
            skip_distractors=args.skip_distractors, skip_families=families,
            reference_blocks=blocks, state_scales=scales,
            with_anchor_mask=bool(args.with_anchor_mask),
        )
        chunks.provenance["load_rows"] = int(args.load_rows)
        return chunks

    if args.cache and os.path.exists(args.cache):
        chunks = _load_cache(args.cache, cfg, args, blocks, families)
        print(f"[tok] chunk cache {args.cache}: "
              f"{tuple(chunks.train_nodes.shape)} train / "
              f"{tuple(chunks.val_nodes.shape)} val")
    else:
        if not paths:
            raise SystemExit(
                "--data is required unless --cache names an existing cache file "
                f"(--cache {args.cache!r} does not exist)."
            )
        chunks = _build()
        if args.cache:
            save_chunk_cache(args.cache, chunks)
            print(f"[tok] wrote chunk cache {args.cache}")

    if args.match_val_episodes:
        _assert_split_matches(chunks, args.match_val_episodes)

    cfg.node_dim = chunks.node_dim
    cfg.reference_blocks = list(chunks.reference_blocks)
    cfg.num_joints = chunks.num_joints
    cfg.num_ee_frames = chunks.num_ee_frames
    cfg.dt = chunks.dt
    cfg.replan_period = int(chunks.provenance.get("recording_replan_period", 0)) or \
        cfg.replan_period
    return chunks


def _resolve_residual_chunks(cfg: TokenizerConfig, args: argparse.Namespace,
                             blocks: list[str] | None):
    """Load a residual build and write its layout back onto ``cfg``.

    There is no build path here on purpose. Forming a residual needs the GENERATOR -- its
    weights, its normalizers and its own held-out split -- which is a different set of inputs
    from anything this trainer takes, so it lives in
    ``scripts/token_planner/build_residual_cache.py`` and its output is the only way in.

    What this checks, and why each one:

    * the payload's ARM, so an absolute cache cannot be trained as a residual one. Both hold
      ``(Nw, H, D)`` float tensors; only their meaning differs.
    * ``--fm_config``, which is REFUSED. Its declared state scales are the absolute
      normalizer, and in this arm the cache's own residual normalizer is the only one that
      may reach the model -- silently applying the wrong affine would rescale every channel
      and every reported metre.
    * the layout the tokenizer will build against, so a 62-node cache cannot feed a 64-node
      config.

    Args:
        cfg: The config, written through with the cache's layout.
        args: Runtime args.
        blocks: An explicit ``--reference_blocks``, or ``None`` to take the cache's.

    Returns:
        The :class:`~robot_rl.token_planner.residual.ResidualChunkData`.

    Raises:
        SystemExit: If ``--cache`` is missing or absent, or ``--fm_config`` was given.
    """
    from robot_rl.token_planner.residual import RESIDUAL_CACHE_VERSION, load_residual_cache

    if not args.cache or not os.path.exists(args.cache):
        raise SystemExit(
            "--residual needs --cache to name an existing residual build. Make one with "
            "scripts/token_planner/build_residual_cache.py --stage build; it is not built "
            "here because forming a residual needs the generator, not just a recording.")
    if args.fm_config:
        raise SystemExit(
            "--residual and --fm_config are mutually exclusive. --fm_config supplies the "
            "declared state scales, which are the ABSOLUTE normalizer; a residual build "
            "carries its own, fitted on the residual with the mean pinned to zero. Applying "
            "the absolute one here would rescale every channel and every reported metre "
            "while every shape stayed right.")
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    # Only what the TRAINER can independently assert. The generator fields in the full key
    # exist so a rebuild can detect its own staleness; this side has no way to know them and
    # would only be restating whatever the file said.
    want = {
        "residual_version": RESIDUAL_CACHE_VERSION,
        "chunk_len": int(cfg.chunk_len),
        "reference_blocks": (blocks if blocks is not None
                             else dict(payload.get("key", {})).get("reference_blocks")),
    }
    chunks = load_residual_cache(args.cache, want, payload=payload)

    fm = chunks.fm_provenance
    print(f"[tok] RESIDUAL arm: chunks are (clip - plan) from {fm.get('fm_run')}")
    print(f"[tok]   checkpoint {fm.get('checkpoint')} epoch {fm.get('fm_epoch')} "
          f"sha {str(fm.get('fm_sha256'))[:16]}")
    print(f"[tok]   draw={fm.get('draw')}  history_source={fm.get('history_source')}  "
          f"honesty={fm.get('honesty')}")
    print(f"[tok]   {tuple(chunks.train_nodes.shape)} train / "
          f"{tuple(chunks.val_nodes.shape)} val"
          + (f", {chunks.provenance.get('zero_chunks_injected', 0)} of the train chunks are "
             "injected all-zero residuals"
             if chunks.provenance.get("zero_chunks_injected") else ""))
    _print_residual_band(chunks)

    cfg.node_dim = chunks.node_dim
    cfg.reference_blocks = list(chunks.reference_blocks)
    cfg.num_joints = chunks.num_joints
    cfg.num_ee_frames = chunks.num_ee_frames
    cfg.dt = chunks.dt
    return chunks


def _print_residual_band(chunks) -> None:
    """Print the fitted residual scale beside the adjuster's DECLARED authority.

    The ratio is the number worth seeing: it says how much of the correction the RL adjuster
    is allowed to make the generator's typical error actually occupies. It also makes the
    fitted mean visible -- which is NOT subtracted (the origin has to keep meaning "the
    generator was right") but IS the generator's systematic bias, and a large one is a finding
    rather than a nuisance.

    Args:
        chunks: The loaded residual build.
    """
    from robot_rl.flow_matching.normalizer import Normalizer
    from robot_rl.token_planner.metrics import channel_group_labels
    from robot_rl.token_planner.residual import declared_residual_normalizer

    norm = Normalizer.from_state_dict(chunks.state_normalizer_state)
    names, labels = channel_group_labels(chunks.num_joints, chunks.num_ee_frames,
                                         chunks.reference_blocks)
    try:
        declared = declared_residual_normalizer(chunks.num_joints, chunks.num_ee_frames,
                                                chunks.reference_blocks).std
    except ValueError:
        declared = None
    mean = torch.tensor(chunks.provenance.get("residual_fitted_mean", [0.0] * norm.std.numel()))
    print("[tok]   residual scale (fitted std | declared adjuster band | fitted mean, "
          "per group):")
    for g, name in enumerate(names):
        sel = labels == g
        fit = float(norm.std[sel].mean())
        dec = float(declared[sel].mean()) if declared is not None else float("nan")
        bias = float(mean[sel].abs().max()) if mean.numel() == norm.std.numel() else float("nan")
        print(f"[tok]     {name:<14} {fit:8.4f} | {dec:8.4f} ({fit / dec:5.2f}x) | "
              f"max|bias| {bias:.4f}")


def _load_cache(path: str, cfg: TokenizerConfig, args: argparse.Namespace,
                blocks: list[str] | None, families: list[str] | None):
    """Load a chunk cache, checking every field that could make it the wrong data.

    The key is assembled from :data:`~robot_rl.token_planner.data.CACHE_VERSION` and the
    caller's OWN arguments. An earlier version of this function took ``version`` off the
    cached payload and left ``data_sha256`` and ``skip_families`` out of the comparison
    entirely, which defeated three of the four guards
    :func:`~robot_rl.token_planner.data.cache_key` exists to provide: a cache built from a
    different recording, or under a different family exclusion, loaded silently, and bumping
    the format version could never refuse a stale one.

    Two fields cannot be checked the same way and are handled explicitly:

    * ``reference_blocks`` -- only a build resolves it from the recording, so an unset
      ``--reference_blocks`` is accepted from the cache and an explicit one is compared.
    * ``data_sha256`` -- verifying it means re-reading tens of GB, and on the box this cache
      was COPIED to the recordings do not exist at all. So it is accepted from the cache and
      PRINTED, which puts the provenance in the run's log and in the checkpoint;
      ``--verify_data_sha256`` re-hashes where the files are actually present. The path list
      is compared whenever ``--data`` was given, which is the cheap half of the same check.

    Args:
        path: The cache file.
        cfg: The config, for ``chunk_len``.
        args: Runtime args.
        blocks: Explicit block selection, or ``None``.
        families: Resolved family exclusions.

    Returns:
        The cached chunk data.

    Raises:
        SystemExit: If the cache was built from different recordings than ``--data`` names,
            or if ``--verify_data_sha256`` finds a hash mismatch.
    """
    from robot_rl.token_planner.data import CACHE_VERSION

    # ONE read. This file is the built chunks -- 4.16 GB on the shipped recordings -- so
    # loading it to inspect the key and then loading it again to use it is a second full
    # read and a second full peak allocation on every run start.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cached_key = dict(payload.get("key", {}))
    want = {
        "version": CACHE_VERSION,
        "chunk_len": int(cfg.chunk_len),
        "seed": int(args.split_seed),
        "val_fraction": float(args.val_fraction),
        "stride": int(args.stride),
        "skip_distractors": bool(args.skip_distractors),
        "skip_families": list(families),
        "reference_blocks": (blocks if blocks is not None
                             else cached_key.get("reference_blocks")),
        # A full run must not accept a cache built from a TRUNCATED recording -- that is a
        # different episode split, so nothing measured against it is comparable.
        "load_rows": int(args.load_rows),
        "data_sha256": cached_key.get("data_sha256"),
        # The declared per-block scales ARE the normalizer on the blocks they cover.
        "state_scales": cached_key.get("state_scales", {}),
    }
    chunks = load_chunk_cache(path, want, payload=payload)

    cached_paths = [p for p in str(chunks.provenance.get("data", "")).split(",") if p]
    cached_shas = [h for h in str(chunks.provenance.get("data_sha256", "")).split(",") if h]
    print("[tok] cache was built from:")
    for p_, h in zip(cached_paths, cached_shas + [""] * len(cached_paths)):
        print(f"[tok]   {h[:16]}  {p_}")
    if int(chunks.provenance.get("load_rows", 0) or 0):
        print(f"[tok] *** this cache was built with --load_rows "
              f"{chunks.provenance['load_rows']}: a TRUNCATED recording and therefore a "
              "different episode split. Nothing measured from it is comparable. ***")

    requested = [p for p in args.data.split(",") if p.strip()]
    if requested and requested != cached_paths:
        listing = "\n       ".join
        raise SystemExit(
            f"[tok] the cache {path} was built from\n       "
            + listing(cached_paths)
            + "\n       but --data names\n       "
            + listing(requested)
            + "\n       Drop --data to use the cache's own inputs, or point --cache "
              "elsewhere. Loading it anyway would train on one recording and record "
              "another."
        )

    if args.verify_data_sha256:
        from robot_rl.token_planner import data as tpd

        sha256_file = tpd._recorded_data_io().sha256_file
        for p_, h in zip(cached_paths, cached_shas):
            if not os.path.exists(p_):
                raise SystemExit(
                    f"[tok] --verify_data_sha256: {p_} is not on this machine, so there is "
                    "nothing to hash. The cache is the portable artifact; drop the flag."
                )
            have = sha256_file(p_)
            if have != h:
                raise SystemExit(
                    f"[tok] --verify_data_sha256: {p_}\n"
                    f"       cache built from {h}\n"
                    f"       on disk now      {have}\n"
                    "       The recording changed under the cache."
                )
        print(f"[tok] --verify_data_sha256: {len(cached_paths)} file(s) match.")
    return chunks


def _state_scales_from(fm_config_path: str) -> dict[str, float | None] | None:
    """The declared per-block state scales a generator trained under.

    Args:
        fm_config_path: A run's ``params/fm_config.yaml``, or ``""``.

    Returns:
        ``{block: scale}`` for every entry present, or ``None`` when no path is given (which
        takes ``FlowMatchingConfig``'s current defaults -- see ``build_chunk_data``).
    """
    if not fm_config_path:
        print("[tok] no --fm_config: the declared state scales are FlowMatchingConfig's "
              "current defaults. They define the normalizer, so pass the generator's own "
              "config if this tokenizer is meant to pair with it.")
        return None
    import yaml

    with open(fm_config_path) as fh:
        raw = yaml.safe_load(fh) or {}
    out = {n: raw[f"state_scale_{n}"] for n in STATE_SCALE_NAMES
           if f"state_scale_{n}" in raw}
    print(f"[tok] state scales from {fm_config_path}: {out}")
    return out


def _assert_split_matches(chunks, params_data_yaml: str) -> None:
    """Require this build's held-out episodes to be exactly a reference run's.

    Args:
        chunks: The built chunk data.
        params_data_yaml: A flow-matching run's ``params/data.yaml``, which records its
            ``val_episode_ids``.

    Raises:
        SystemExit: On a mismatch, reporting the sizes and the symmetric difference. Fatal
            rather than a warning: a tokenizer trained on windows the planner will be
            validated against is a leak, and one validated on windows the FM trained on
            makes the comparison meaningless in the other direction.
        ValueError: If the reference file records no ``val_episode_ids``.
    """
    import yaml

    ref = yaml.safe_load(open(params_data_yaml))
    want = ref.get("val_episode_ids")
    if not want:
        raise ValueError(f"{params_data_yaml} records no val_episode_ids to match against.")
    want_set = {int(v) for v in want}
    have_set = {int(v) for v in chunks.val_episode_ids}
    if want_set != have_set:
        only_ref = sorted(want_set - have_set)[:8]
        only_new = sorted(have_set - want_set)[:8]
        raise SystemExit(
            f"[tok] the episode split does NOT match {params_data_yaml}:\n"
            f"       reference held out {len(want_set)} episodes, this build "
            f"{len(have_set)}; {len(want_set ^ have_set)} differ.\n"
            f"       only in the reference: {only_ref}\n"
            f"       only in this build:    {only_new}\n"
            "       Check --split_seed, --val_fraction, --skip_distractors, "
            "--skip_families, and the ORDER of --data (the episode ids are offset per file, "
            "so swapping two recordings renumbers every episode)."
        )
    print(f"[tok] episode split matches {params_data_yaml}: "
          f"{len(have_set)} held-out episodes, identical set.")


def _patch_row_limit(n: int) -> None:
    """Restrict every recording read to its first ``n`` windows (smoke tests only).

    Args:
        n: Row cap.
    """
    from robot_rl.token_planner import data as tpd

    rdio = tpd._recorded_data_io()
    original = rdio.load_recorded_windows

    def limited(path, keys=None, rows=None):
        return original(path, keys=keys, rows=np.arange(n))

    rdio.load_recorded_windows = limited


@torch.no_grad()
def residual_gate(model: TrajectoryTokenizer, chunks, cfg: TokenizerConfig,
                  device: torch.device, batch_size: int = 256) -> dict[str, Any]:
    """The two things a residual tokenizer is gated on, measured on the selected weights.

    **ZERO_CODE.** An adjuster has to be able to say "the generator was right here", and under
    a shared codebook that means one id whose constant chunk decodes to ~nothing. FSQ is
    parameter-free and its lattice has no exact centre on an even-levelled axis, so the id is
    FOUND after training rather than reserved -- and if the build injected no all-zero chunks
    there simply will not be a good one, which is what the recorded norm is for.

    **The plan-space gate.** Reconstruction error on the residual would answer the wrong
    question. What matters is whether ``P (+) delta_hat`` is closer to the clip than ``P``
    alone, composed through the SAME rule the robot serves -- Gram-Schmidt and the 15 deg tilt
    bound included. On the linear channels the two are identical, since
    ``(P + d_hat) - gt == d_hat - d``; on the rotation half they are not, and that is exactly
    where a correction can be clamped away.

    Args:
        model: The tokenizer in the weights being selected (swap the EMA in first).
        chunks: The residual build; its ``val_plan`` is what makes this measurable.
        cfg: The config, for the layout.
        device: Where to run.
        batch_size: Chunks per decode.

    Returns:
        ``{"zero_code", "zero_code_decode", "zero_code_rms", "plan_space", "zero_code_frac"}``.
    """
    from robot_rl.flow_matching.normalizer import Normalizer
    from robot_rl.token_planner.metrics import channel_group_labels
    from robot_rl.token_planner.residual import find_zero_code, plan_space_report

    # Two copies rather than one moved back and forth: ``Normalizer.to`` mutates in place, so
    # a single instance shuttled between devices would silently leave the caller's on whichever
    # side it was last used.
    norm = Normalizer.from_state_dict(chunks.state_normalizer_state).to(device)
    norm_cpu = Normalizer.from_state_dict(chunks.state_normalizer_state)
    names, labels = channel_group_labels(cfg.num_joints, cfg.num_ee_frames,
                                         cfg.reference_blocks)
    zero_id, per_group, rms = find_zero_code(model, norm, labels, names)

    recon, ids = [], []
    for i in range(0, int(chunks.val_nodes.shape[0]), batch_size):
        nodes = chunks.val_nodes[i:i + batch_size].to(device)
        out, tok = model(nodes)
        recon.append(norm.denormalize(out.float()).cpu())
        ids.append(tok.cpu())
    delta_hat = torch.cat(recon)
    tok_ids = torch.cat(ids)

    plan = chunks.val_plan
    # ``target = plan + delta`` reconstructs the clip exactly, because the residual was formed
    # as a plain elementwise difference. Re-deriving it here rather than storing it keeps the
    # cache smaller and makes the identity explicit.
    target = plan + norm_cpu.denormalize(chunks.val_nodes)
    rep = plan_space_report(plan, target, delta_hat, cfg.num_joints, cfg.num_ee_frames,
                            cfg.reference_blocks, chunks.val_family, chunks.family_names,
                            model.vocab_size)
    frac = float((tok_ids == zero_id).float().mean())
    out = {
        "zero_code": int(zero_id),
        "zero_code_decode": per_group,
        "zero_code_rms": rms,
        "zero_code_frac": frac,
        "plan_space": rep,
    }
    print(f"[tok] ZERO_CODE {zero_id}: decode rms {rms:.5f}, per group "
          + "  ".join(f"{g} {v:.5f}" for g, v in per_group.items()))
    print(f"[tok]   {100 * frac:.1f}% of held-out tokens ARE that id"
          + ("  <-- over ~60% means the residuals are too small to be worth tokenizing"
             if frac > 0.6 else ""))
    print("[tok] PLAN SPACE (metres/rad, vs the clip): "
          "arm            " + "  ".join(f"{g:>12}" for g in rep["plan"]["rmse"]))
    for arm in ("plan", "plan_plus_residual"):
        print(f"[tok]   {arm:<22}" + "  ".join(f"{v:12.4f}"
                                               for v in rep[arm]["rmse"].values()))
    better = {g: rep["plan_plus_residual"]["rmse"][g] < rep["plan"]["rmse"][g]
              for g in rep["plan"]["rmse"]}
    print("[tok]   the correction HELPS on: "
          + ", ".join(g for g, ok in better.items() if ok)
          + (" | HURTS on: " + ", ".join(g for g, ok in better.items() if not ok)
             if not all(better.values()) else ""))
    return out


def save_checkpoint(path: str, model: TrajectoryTokenizer, opt, cfg: TokenizerConfig,
                    chunks, epoch: int, scalars: dict[str, float],
                    report: dict[str, Any], ema: WeightEMA | None, step: int = 0) -> None:
    """Write a self-describing tokenizer checkpoint.

    Self-describing is the requirement, not a nicety: the AR planner's targets are token ids,
    and a token id means nothing without the FSQ table, ``D``, ``H`` and the normalizer that
    says what the decoded numbers are. So all of it travels here, and the planner's own
    checkpoint will embed this config rather than re-deriving it.

    Args:
        path: Destination.
        model: The tokenizer.
        opt: The optimizer, for a resume.
        cfg: The resolved config.
        chunks: The chunk data, for the normalizer and provenance.
        epoch: Zero-based epoch just finished.
        scalars: Flat validation scalars.
        report: The full validation report.
        ema: The weight EMA, or ``None``.
        step: The global optimizer step. ``scheduled_lr`` is a pure function of
            ``(epoch, step)``, so without it even a hand-written resume would restart
            the warmup. There is no resume path yet; this is what one would need.
        step: The global optimizer step. ``scheduled_lr`` is a pure function of
            ``(epoch, step)``, so without it even a hand-written resume would restart the
            warmup. There is no resume path yet; this is what one would need.
    """
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg.to_dict(),
        "state_normalizer": chunks.state_normalizer_state,
        "num_joints": int(chunks.num_joints),
        "num_ee_frames": int(chunks.num_ee_frames),
        "reference_blocks": list(chunks.reference_blocks),
        "family_names": list(chunks.family_names),
        "val_episode_ids": list(chunks.val_episode_ids),
        "provenance": dict(chunks.provenance),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": dict(scalars),
        "report": report,
    }
    if ema is not None:
        payload["model_ema"] = ema.shadow
        payload["ema_state"] = {k: v for k, v in ema.state_dict().items() if k != "shadow"}
    torch.save(payload, path)


def main() -> None:
    """Train the tokenizer and write ``tokenizer.pt`` plus a gating report."""
    cfg, args = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(args.device)

    chunks = resolve_chunks(cfg, args)

    if args.build_cache_only:
        if not args.cache:
            raise SystemExit("--build_cache_only needs --cache to say where to write it.")
        print(f"[tok] --build_cache_only: {chunks.train_nodes.shape[0]} train / "
              f"{chunks.val_nodes.shape[0]} val chunks of "
              f"({cfg.chunk_len}, {cfg.node_dim}) -> {args.cache}")
        return

    run = (f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
           if args.run_name else time.strftime("%Y-%m-%d_%H-%M-%S"))
    out_dir = os.path.join(args.logdir, run)
    os.makedirs(out_dir, exist_ok=True)

    model = TrajectoryTokenizer(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    # The self-describing banner. Same reason the FM trainer has one: several of these
    # choices leave every tensor width unchanged, so the log and the saved config are the
    # only witnesses to what a checkpoint actually trained under.
    print(
        f"[tok] tokenizer: H={cfg.chunk_len} nodes ({cfg.chunk_len * cfg.dt:.2f} s) "
        f"-> N={cfg.num_tokens} tokens, D={cfg.downsample} "
        f"({cfg.token_seconds * 1000:.0f} ms/token)\n"
        f"[tok] node_dim={cfg.node_dim} blocks={cfg.reference_blocks} "
        f"nj={cfg.num_joints} K={cfg.num_ee_frames} dt={cfg.dt} replan_K={cfg.replan_period}\n"
        f"[tok] FSQ levels {cfg.fsq_levels} -> V={cfg.vocab_size} "
        f"(L_DIM={cfg.latent_dim}); {math.log2(cfg.vocab_size):.1f} bits/token, "
        f"{cfg.num_tokens * math.log2(cfg.vocab_size):.0f} bits/plan\n"
        f"[tok] trunk C={cfg.channels} blocks {cfg.num_blocks_high}+{cfg.num_blocks_low} "
        f"k={cfg.kernel_size} pad={cfg.pad_mode} up={cfg.upsample_mode} "
        f"-> {n_par / 1e6:.2f}M params\n"
        f"[tok] loss: rec + {cfg.lambda_vel} * vel + {cfg.lambda_acc} * acc "
        f"(normalized space); base_pose weight {cfg.base_pose_weight}, "
        f"pinned_axis weight {cfg.pinned_axis_weight} (mean-1 normalized)\n"
        f"[tok] data: {chunks.train_nodes.shape[0]} train / {chunks.val_nodes.shape[0]} val "
        f"chunks over {len(chunks.val_episode_ids)} held-out episodes\n"
        f"[tok] out: {out_dir}"
    )

    git_snapshot(out_dir)
    dump_params_yaml(os.path.join(out_dir, "params", "tokenizer_config.yaml"), cfg.to_dict())
    dump_params_yaml(os.path.join(out_dir, "params", "data.yaml"), {
        **chunks.provenance,
        "train_chunks": int(chunks.train_nodes.shape[0]),
        "val_chunks": int(chunks.val_nodes.shape[0]),
        "val_episode_ids": list(chunks.val_episode_ids),
        "family_names": list(chunks.family_names),
        "reference_blocks": list(chunks.reference_blocks),
    })

    train_loader = DataLoader(
        ChunkDataset(chunks.train_nodes, chunks.train_family,
                     chunks.train_anchor_mask, chunks.train_mask_frame_yaw),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(ChunkDataset(chunks.val_nodes, chunks.val_family,
                                chunks.val_anchor_mask, chunks.val_mask_frame_yaw),
                   batch_size=cfg.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        if chunks.val_nodes.shape[0] else None
    )

    group_names, labels = channel_group_labels(
        cfg.num_joints, cfg.num_ee_frames, cfg.reference_blocks)
    _norm = Normalizer.from_state_dict(chunks.state_normalizer_state)
    std, mean = _norm.std.clone(), _norm.mean.clone()
    # The pinned planar term projects in METRES and divides by the std along the pinned
    # direction, so it needs the same normalizer the chunks were built under -- not a fresh
    # one, and not the accumulator's copy by coincidence.
    state_std = std.to(device)
    if float(cfg.pinned_axis_weight) != 1.0 and chunks.train_anchor_mask is None:
        raise SystemExit(
            f"[tok] --pinned_axis_weight {cfg.pinned_axis_weight:g} needs the per-node anchor "
            "mask, and this chunk build carries none.\n"
            "      Rebuild with --with_anchor_mask (and a --cache path of its own, so the "
            "maskless cache other runs use is left alone)."
        )

    def acc_factory() -> ReconAccumulator:
        """A fresh accumulator wired to this run's layout."""
        return ReconAccumulator(
            group_names=group_names, labels=labels, std=std, mean=mean,
            chunk_len=cfg.chunk_len,
            vocab_size=cfg.vocab_size, num_families=len(chunks.family_names),
            family_names=list(chunks.family_names),
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(float(cfg.betas[0]), float(cfg.betas[1])),
    )
    ema = WeightEMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0.0 else None
    writer = SummaryWriter(log_dir=out_dir)
    wandb_run = None
    if args.wandb:
        # Never let telemetry take down a training run: an uninstalled wandb, a missing
        # login or a network blip would otherwise kill the job before its first epoch.
        # Degrade to TensorBoard (always written) and say so loudly enough to notice.
        try:
            import wandb  # noqa: PLC0415

            wandb_run = wandb.init(
                project=args.wandb_project, name=run,
                config={**cfg.to_dict(), **chunks.provenance,
                        "train_chunks": int(chunks.train_nodes.shape[0]),
                        "val_chunks": int(chunks.val_nodes.shape[0]),
                        "num_tokens": cfg.num_tokens, "vocab_size": cfg.vocab_size,
                        "token_seconds": cfg.token_seconds, "params": n_par},
            )
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb logging is OFF: {type(exc).__name__}: {exc}\n"
                  f"       Training continues; scalars still go to TensorBoard in {out_dir}.\n"
                  "       Pass --no_wandb to silence this, or fix the login (wandb login).")
    use_bf16 = bool(args.bf16 and device.type == "cuda")

    best = math.inf
    step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        t0 = time.time()
        run_loss, run_n, clipped = 0.0, 0, 0
        for batch in train_loader:
            nodes = batch["nodes"].to(device, non_blocking=True)
            lr = scheduled_lr(cfg, epoch, step)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                recon, _ = model(nodes)
            loss, terms = reconstruction_loss(recon.float(), nodes, cfg, batch,
                                               state_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            clipped += int(gn > cfg.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            run_loss += terms["total"] * nodes.shape[0]
            run_n += int(nodes.shape[0])
            if step % 50 == 0:
                writer.add_scalar("train/loss_step", terms["total"], step)
                writer.add_scalar("train/lr_step", lr, step)
                writer.add_scalar("train/grad_norm_step", gn, step)
            step += 1

        writer.add_scalar("train/loss_epoch", run_loss / max(run_n, 1), epoch)
        writer.add_scalar("train/grad_clipped_frac", clipped / max(len(train_loader), 1), epoch)
        writer.add_scalar("train/epoch_time_s", time.time() - t0, epoch)
        line = (f"[tok] epoch {epoch:>4d}  train {run_loss / max(run_n, 1):.6f}  "
                f"{time.time() - t0:.1f}s")

        epoch_scalars = {
            "train/loss_epoch": run_loss / max(run_n, 1),
            "train/grad_clipped_frac": clipped / max(len(train_loader), 1),
            "train/epoch_time_s": time.time() - t0,
            "train/lr": scheduled_lr(cfg, epoch, step),
        }

        do_val = val_loader is not None and (
            epoch % max(args.val_every, 1) == 0 or epoch == cfg.num_epochs - 1)
        if do_val:
            rep, scal = evaluate(model, val_loader, device, acc_factory, cfg, state_std,
                                 residual=args.residual)
            _val_raw = dict(scal)
            for k, v in scal.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            sel, sel_rep, tag = scal["rmse_mean"], rep, "val"
            if ema is not None:
                with ema.swapped_in(model):
                    rep_e, scal_e = evaluate(model, val_loader, device, acc_factory,
                                             cfg, state_std, residual=args.residual)
                for k, v in scal_e.items():
                    writer.add_scalar(f"val_ema/{k}", v, epoch)
                sel, sel_rep, tag = scal_e["rmse_mean"], rep_e, "val_ema"
                epoch_scalars.update({f"val_ema/{k}": v for k, v in scal_e.items()})
                scal = scal_e
            # The RAW-weight numbers, always -- ``scal`` has been rebound to the EMA's by
            # the branch above, and logging those under both names would make the two series
            # identical and hide whatever the EMA is actually buying.
            epoch_scalars.update({f"val/{k}": v for k, v in _val_raw.items()})
            # The per-terrain-family split is where a quantization failure shows up first --
            # the *_pure families compress far better than the transition ones -- so it goes
            # to wandb as its own series rather than only into the report file.
            for fam, vals in sel_rep["rmse_by_family"].items():
                for g, v in vals.items():
                    epoch_scalars[f"family/{fam}/{g}"] = v
            line += (f"  {tag} rmse_mean {sel:.5f}  "
                     f"pos {scal['rmse/base_pos']:.4f} m  "
                     f"jnt {scal['rmse/joint_pos']:.4f} rad  "
                     f"cb {100 * scal['codebook/utilisation']:.1f}% "
                     f"ppl {scal['codebook/perplexity']:.0f}")
            if sel < best:
                best = sel
                if args.residual:
                    # Under the SELECTED weights: `sel` came from the EMA arm when there is
                    # one, and gating the raw weights instead would report a checkpoint
                    # nobody serves.
                    ctx = ema.swapped_in(model) if ema is not None else contextlib.nullcontext()
                    with ctx:
                        sel_rep["residual_gate"] = residual_gate(model, chunks, cfg, device)
                    gate = sel_rep["residual_gate"]
                    scal["zero_code_rms"] = gate["zero_code_rms"]
                    scal["zero_code_frac"] = gate["zero_code_frac"]
                    for arm in ("plan", "plan_plus_residual"):
                        for g, v in gate["plan_space"][arm]["rmse"].items():
                            scal[f"{arm}/{g}"] = v
                save_checkpoint(os.path.join(out_dir, "tokenizer.pt"), model, opt, cfg,
                                chunks, epoch, scal, sel_rep, ema, step)
                with open(os.path.join(out_dir, "gating_report.json"), "w") as fh:
                    json.dump({"epoch": epoch, "scalars": scal, "report": sel_rep}, fh,
                              indent=2)
                with open(os.path.join(out_dir, "gating_report.txt"), "w") as fh:
                    fh.write(format_report(sel_rep) + "\n")
                line += "  *"
        print(line, flush=True)
        if wandb_run is not None:
            # step=epoch on EVERY call. Without it wandb increments its own counter once per
            # log() CALL, so a run with several logs per epoch ends up with an x-axis that is
            # not the epoch -- the same trap scripts/flow_matching/train.py records.
            wandb_run.log({**epoch_scalars, "epoch": epoch}, step=epoch)

        if epoch % max(args.ckpt_every, 1) == 0 or epoch == cfg.num_epochs - 1:
            save_checkpoint(os.path.join(out_dir, f"ckpt_{epoch}.pt"), model, opt, cfg,
                            chunks, epoch, {}, {}, ema, step)

    if val_loader is None:
        # Without a validation split nothing above ever fired, so the run would finish
        # having printed the path of a file it never wrote.
        save_checkpoint(os.path.join(out_dir, "tokenizer.pt"), model, opt, cfg, chunks,
                        cfg.num_epochs - 1, {}, {}, ema, step)
        print("[tok] no validation split: wrote tokenizer.pt from the final weights, with "
              "no metrics and no model selection.")
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    if os.path.exists(os.path.join(out_dir, "gating_report.txt")):
        print("\n" + pathlib.Path(out_dir, "gating_report.txt").read_text())
    print(f"[tok] done. best val rmse_mean {best:.6f} -> {out_dir}/tokenizer.pt")


if __name__ == "__main__":
    main()
