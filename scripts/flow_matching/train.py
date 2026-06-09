# Copyright 2026 Zachary Olkin. All rights reserved.

"""Train a flow-matching transformer to predict future robot states.

Loads one recorded ``.npz`` dataset (produced by ``scripts/rsl_rl/record_data.py``
with the state-recording extension), builds episode-aware (history, future)
windows, and trains a conditional flow-matching transformer. Logs to TensorBoard
(and optionally Weights & Biases) and checkpoints the model together with the
normalizers, config, and dataset metadata.

Example:
    python scripts/flow_matching/train.py \
        --data logs/rsl_rl/g1/<run>/recorded_data/recorded_<ts>.npz \
        --logdir logs/flow_matching --num_epochs 200
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make the recording I/O module importable (it lives in scripts/rsl_rl).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, os.pardir, "rsl_rl"))
from recorded_data_io import load_recorded_data  # noqa: E402  isort: skip

from robot_rl.flow_matching import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingTransformer,
    build_train_val,
    flow_matching_loss,
    orientation_slice,
    rot6d_to_matrix,
    sample_trajectory,
)


def parse_args() -> tuple[FlowMatchingConfig, argparse.Namespace]:
    """Parse CLI arguments into a config plus runtime options."""
    parser = argparse.ArgumentParser(description="Train a flow-matching future-state predictor.")
    parser.add_argument("--data", type=str, required=True, help="Path to a recorded .npz dataset.")
    parser.add_argument("--logdir", type=str, default="logs/flow_matching", help="Log/checkpoint root.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (default: timestamp).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--val_every", type=int, default=5, help="Validate every N epochs.")
    parser.add_argument("--ckpt_every", type=int, default=20, help="Checkpoint every N epochs.")
    parser.add_argument("--cosine", action="store_true", help="Use cosine LR annealing.")
    parser.add_argument("--wandb", action="store_true", help="Also log to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="robot_rl_flow_matching")
    # config fields
    parser.add_argument("--history_len", type=int, default=25)
    parser.add_argument("--future_len", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_sampling_steps", type=int, default=8)
    parser.add_argument("--ori6d_aux_weight", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = FlowMatchingConfig(
        history_len=args.history_len, future_len=args.future_len, stride=args.stride,
        val_fraction=args.val_fraction, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_sampling_steps=args.num_sampling_steps, ori6d_aux_weight=args.ori6d_aux_weight,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        num_epochs=args.num_epochs, grad_clip=args.grad_clip, seed=args.seed,
    )
    return cfg, args


def channel_metrics(pred: torch.Tensor, tgt: torch.Tensor, num_joints: int) -> dict[str, float]:
    """Per-channel physical-units error between predicted and target futures.

    Args:
        pred: Predicted future states (denormalized), shape ``(B, F, D)``.
        tgt: Target future states (denormalized), shape ``(B, F, D)``.
        num_joints: Number of joints ``nj``.

    Returns:
        Dict of RMSEs per channel plus the orientation geodesic error (radians).
    """
    nj = num_joints

    def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.sqrt(torch.mean((a - b) ** 2)).item()

    o0, o1 = orientation_slice(nj)
    r_pred = rot6d_to_matrix(pred[..., o0:o1])
    r_tgt = rot6d_to_matrix(tgt[..., o0:o1])
    rel = r_pred.transpose(-1, -2) @ r_tgt
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    geodesic = torch.arccos(cos).mean().item()

    return {
        "rmse_joint_pos": _rmse(pred[..., 0:nj], tgt[..., 0:nj]),
        "rmse_joint_vel": _rmse(pred[..., nj:2 * nj], tgt[..., nj:2 * nj]),
        "rmse_root_pos": _rmse(pred[..., 2 * nj:2 * nj + 3], tgt[..., 2 * nj:2 * nj + 3]),
        "rmse_root_lin_vel": _rmse(pred[..., 2 * nj + 3:2 * nj + 6], tgt[..., 2 * nj + 3:2 * nj + 6]),
        "rmse_root_ang_vel": _rmse(pred[..., 2 * nj + 12:2 * nj + 15], tgt[..., 2 * nj + 12:2 * nj + 15]),
        "ori_geodesic_rad": geodesic,
    }


@torch.no_grad()
def evaluate(model, loader, cfg, target_normalizer, num_joints, device) -> dict[str, float]:
    """Compute mean flow loss and per-channel sampling metrics over a loader."""
    model.eval()
    flow_sum, n_batches = 0.0, 0
    metric_sums: dict[str, float] = {}
    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        flow_sum += flow_matching_loss(model, history, future)["flow_mse"].item()
        n_batches += 1
        pred = sample_trajectory(model, history, cfg.future_len, cfg.state_dim, cfg.num_sampling_steps)
        pred_d = target_normalizer.denormalize(pred)
        tgt_d = target_normalizer.denormalize(future)
        for k, v in channel_metrics(pred_d, tgt_d, num_joints).items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v
    out = {"flow_mse": flow_sum / max(n_batches, 1)}
    for k, v in metric_sums.items():
        out[k] = v / max(n_batches, 1)
    return out


def main() -> None:
    """Entry point: load data, build datasets, train, validate, checkpoint."""
    cfg, args = parse_args()
    torch.manual_seed(cfg.seed)

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.logdir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[INFO] Loading dataset: {args.data}")
    data, meta = load_recorded_data(args.data)
    train_ds, val_ds = build_train_val(data, meta, cfg)
    num_joints = train_ds.num_joints
    print(f"[INFO] state_dim={cfg.state_dim}, history_in_dim={cfg.history_in_dim}, "
          f"train_windows={len(train_ds)}, val_windows={0 if val_ds is None else len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = FlowMatchingTransformer(
        history_in_dim=cfg.history_in_dim, state_dim=cfg.state_dim,
        history_len=cfg.history_len, future_len=cfg.future_len,
        d_model=cfg.d_model, nhead=cfg.nhead, num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers, dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Normalizers are fit on CPU; move their stats to the model device so
    # (de)normalization during evaluation/sampling stays on one device.
    train_ds.target_normalizer.to(device)
    train_ds.history_normalizer.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs) if args.cosine else None
    )
    ori_slice = orientation_slice(num_joints)

    writer = SummaryWriter(log_dir=out_dir)
    wandb_run = None
    if args.wandb:
        import wandb  # noqa: PLC0415
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=cfg.to_dict())

    best_val = math.inf
    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss, n = 0.0, 0
        for batch in train_loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            opt.zero_grad()
            out = flow_matching_loss(model, history, future, ori_slice, cfg.ori6d_aux_weight)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += out["loss"].item()
            n += 1
            global_step += 1
            writer.add_scalar("train/loss_step", out["loss"].item(), global_step)
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        train_loss = epoch_loss / max(n, 1)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/epoch_time_s", epoch_time, epoch)
        log_line = f"[epoch {epoch}] train_loss={train_loss:.4f}  train_time={epoch_time:.1f}s"

        if val_loader is not None and (epoch % args.val_every == 0 or epoch == cfg.num_epochs - 1):
            val = evaluate(model, val_loader, cfg, train_ds.target_normalizer, num_joints, device)
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            log_line += "  " + "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            if val["flow_mse"] < best_val:
                best_val = val["flow_mse"]
                _save_ckpt(os.path.join(out_dir, "best.pt"), model, opt, cfg, train_ds, meta, epoch)
            if wandb_run is not None:
                wandb_run.log({**{f"val/{k}": v for k, v in val.items()}, "epoch": epoch})

        print(log_line)
        if wandb_run is not None:
            wandb_run.log({"train/loss_epoch": train_loss, "train/epoch_time_s": epoch_time, "epoch": epoch})
        if epoch % args.ckpt_every == 0 or epoch == cfg.num_epochs - 1:
            _save_ckpt(os.path.join(out_dir, f"ckpt_{epoch}.pt"), model, opt, cfg, train_ds, meta, epoch)

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[INFO] Done. Checkpoints in {out_dir}")


def _save_ckpt(path, model, opt, cfg, train_ds, meta, epoch) -> None:
    """Save model weights, optimizer, config, normalizers, and dataset metadata."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg.to_dict(),
            "target_normalizer": train_ds.target_normalizer.state_dict(),
            "history_normalizer": train_ds.history_normalizer.state_dict(),
            "num_joints": train_ds.num_joints,
            "meta": asdict(meta) if hasattr(meta, "__dataclass_fields__") else vars(meta),
            "epoch": epoch,
        },
        path,
    )


if __name__ == "__main__":
    main()
