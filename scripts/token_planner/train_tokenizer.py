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
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--ckpt_every", type=int, default=10)
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
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--lr_min", type=float, default=d.lr_min)
    p.add_argument("--weight_decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup_steps", type=int, default=d.warmup_steps)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--num_epochs", type=int, default=d.num_epochs)
    p.add_argument("--grad_clip", type=float, default=d.grad_clip)
    p.add_argument("--ema_decay", type=float, default=d.ema_decay)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast. On by default, unlike the FM trainer, which is fp32 "
                        "throughout -- this is a fresh model with no checkpoint to stay "
                        "numerically comparable with.")

    args = p.parse_args()
    cfg = TokenizerConfig(
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


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, cfg: TokenizerConfig
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
        cfg: The config, for the two lambdas.

    Returns:
        ``(loss, terms)``.
    """
    rec = torch.nn.functional.mse_loss(recon, target)
    loss = rec
    terms = {"rec": float(rec.detach())}
    if cfg.lambda_vel > 0.0:
        v = torch.nn.functional.mse_loss(recon.diff(dim=1), target.diff(dim=1))
        loss = loss + cfg.lambda_vel * v
        terms["vel"] = float(v.detach())
    if cfg.lambda_acc > 0.0:
        a = torch.nn.functional.mse_loss(
            recon.diff(dim=1).diff(dim=1), target.diff(dim=1).diff(dim=1))
        loss = loss + cfg.lambda_acc * a
        terms["acc"] = float(a.detach())
    terms["total"] = float(loss.detach())
    return loss, terms


@torch.no_grad()
def evaluate(
    model: TrajectoryTokenizer, loader: DataLoader, device: torch.device,
    acc_factory, cfg: TokenizerConfig,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run a full pass and return ``(report, scalars)``.

    Args:
        model: The tokenizer, already in the weights to be measured.
        loader: A validation loader over :class:`ChunkDataset`.
        device: Where to run.
        acc_factory: Zero-argument factory for a fresh :class:`ReconAccumulator`.
        cfg: The config, for the loss terms.

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
        loss, _ = reconstruction_loss(recon.float(), nodes, cfg)
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
            reference_blocks=blocks,
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

    cached_key = torch.load(path, map_location="cpu", weights_only=False).get("key", {})
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
        "data_sha256": cached_key.get("data_sha256"),
    }
    chunks = load_chunk_cache(path, want)

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


def save_checkpoint(path: str, model: TrajectoryTokenizer, opt, cfg: TokenizerConfig,
                    chunks, epoch: int, scalars: dict[str, float],
                    report: dict[str, Any], ema: WeightEMA | None) -> None:
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
    cfg.validate()

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
        f"(normalized space)\n"
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
        ChunkDataset(chunks.train_nodes, chunks.train_family),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(ChunkDataset(chunks.val_nodes, chunks.val_family),
                   batch_size=cfg.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        if chunks.val_nodes.shape[0] else None
    )

    group_names, labels = channel_group_labels(
        cfg.num_joints, cfg.num_ee_frames, cfg.reference_blocks)
    std = Normalizer.from_state_dict(chunks.state_normalizer_state).std.clone()

    def acc_factory() -> ReconAccumulator:
        """A fresh accumulator wired to this run's layout."""
        return ReconAccumulator(
            group_names=group_names, labels=labels, std=std, chunk_len=cfg.chunk_len,
            vocab_size=cfg.vocab_size, num_families=len(chunks.family_names),
            family_names=list(chunks.family_names),
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(float(cfg.betas[0]), float(cfg.betas[1])),
    )
    ema = WeightEMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0.0 else None
    writer = SummaryWriter(log_dir=out_dir)
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
            loss, terms = reconstruction_loss(recon.float(), nodes, cfg)
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

        do_val = val_loader is not None and (
            epoch % max(args.val_every, 1) == 0 or epoch == cfg.num_epochs - 1)
        if do_val:
            rep, scal = evaluate(model, val_loader, device, acc_factory, cfg)
            for k, v in scal.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            sel, sel_rep, tag = scal["rmse_mean"], rep, "val"
            if ema is not None:
                with ema.swapped_in(model):
                    rep_e, scal_e = evaluate(model, val_loader, device, acc_factory, cfg)
                for k, v in scal_e.items():
                    writer.add_scalar(f"val_ema/{k}", v, epoch)
                sel, sel_rep, tag = scal_e["rmse_mean"], rep_e, "val_ema"
                scal = scal_e
            line += (f"  {tag} rmse_mean {sel:.5f}  "
                     f"pos {scal['rmse/base_pos']:.4f} m  "
                     f"jnt {scal['rmse/joint_pos']:.4f} rad  "
                     f"cb {100 * scal['codebook/utilisation']:.1f}% "
                     f"ppl {scal['codebook/perplexity']:.0f}")
            if sel < best:
                best = sel
                save_checkpoint(os.path.join(out_dir, "tokenizer.pt"), model, opt, cfg,
                                chunks, epoch, scal, sel_rep, ema)
                with open(os.path.join(out_dir, "gating_report.json"), "w") as fh:
                    json.dump({"epoch": epoch, "scalars": scal, "report": sel_rep}, fh,
                              indent=2)
                with open(os.path.join(out_dir, "gating_report.txt"), "w") as fh:
                    fh.write(format_report(sel_rep) + "\n")
                line += "  *"
        print(line, flush=True)

        if epoch % max(args.ckpt_every, 1) == 0 or epoch == cfg.num_epochs - 1:
            save_checkpoint(os.path.join(out_dir, f"ckpt_{epoch}.pt"), model, opt, cfg,
                            chunks, epoch, {}, {}, ema)

    writer.close()
    if os.path.exists(os.path.join(out_dir, "gating_report.txt")):
        print("\n" + pathlib.Path(out_dir, "gating_report.txt").read_text())
    print(f"[tok] done. best val rmse_mean {best:.6f} -> {out_dir}/tokenizer.pt")


if __name__ == "__main__":
    main()
