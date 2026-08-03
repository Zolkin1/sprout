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
        --logdir logs/flow_matching --num_epochs 100
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make the recording I/O module importable (it lives in scripts/rsl_rl).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_THIS_DIR, os.pardir, "rsl_rl"))
from recorded_data_io import (  # noqa: E402  isort: skip
    load_recorded_data,
    load_recorded_windows,
    is_windowed_recording,
)
from provenance import dump_params_yaml, git_snapshot, sha256_file  # noqa: E402  isort: skip

from robot_rl.flow_matching import (  # noqa: E402
    FKChain,
    FlowMatchingConfig,
    EXTERO_SOURCES,
    FlowMatchingTransformer,
    FutureStateDataset,
    build_train_val,
    flow_matching_loss,
    REFERENCE_WINDOW_BLOCK_NAMES,
    REFERENCE_WINDOW_NOISE,
    REFERENCE_BLOCK_KEYS,
    so3_exp,
    window_block_slices,
    sample_trajectory,
    snap_windows,
)


def parse_args() -> tuple[FlowMatchingConfig, argparse.Namespace]:
    """Parse CLI arguments into a config plus runtime options."""
    parser = argparse.ArgumentParser(description="Train a flow-matching future-state predictor.")
    parser.add_argument("--data", type=str, required=True, help="Path to a recorded .npz dataset.")
    parser.add_argument(
        "--eval_data", type=str, default=None,
        help="Optional separate held-out recording (.npz) for eval metrics (logged under "
             "eval/*). Never trained on; evaluated with the train normalizers. Must share "
             "the train dataset's layout (state_dim / cmd_dim / extra frames).",
    )
    parser.add_argument("--logdir", type=str, default="logs/flow_matching", help="Log/checkpoint root.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (default: timestamp).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--val_every", type=int, default=5, help="Validate every N epochs.")
    parser.add_argument("--ckpt_every", type=int, default=20, help="Checkpoint every N epochs.")
    parser.add_argument("--cosine", action="store_true", help="Use cosine LR annealing.")
    # Opt-OUT, matching the RL side (its agent cfgs ship `logger: wandb`). Every FM run is
    # worth comparing against the others, and a run you have to remember to instrument is
    # a run whose curve you do not have when you want it. TensorBoard is written either way.
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Log to Weights & Biases (default; use --no_wandb to disable).")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false",
                        help="Disable Weights & Biases; TensorBoard logging is unaffected.")
    parser.add_argument("--wandb_project", type=str, default="robot_rl_flow_matching")
    # config fields
    parser.add_argument("--future_len", type=int, default=FlowMatchingConfig().future_len,
                        help="Window length F (state tokens). Effective novel horizon is "
                             "F - num_history_states (default from config.py, sized to cover the "
                             "deploy-time policy lookahead; see FlowMatchingConfig.future_len).")
    parser.add_argument("--num_history_states", type=int,
                        default=FlowMatchingConfig().num_history_states,
                        help="N autoregressive context tokens: the reference-window nodes the "
                             "generator emitted at the N previous control steps. Needs a "
                             "recording with history_len >= N + 1.")
    parser.add_argument("--extero_source", type=str, default=FlowMatchingConfig().extero_source,
                        choices=list(EXTERO_SOURCES),
                        help="Which recorded exteroception feeds the conditioning tokens. "
                             "record_data.py stores BOTH the height scan and the depth images "
                             "whenever the task has depth obs terms, so this is a train-time "
                             "A/B over one recording. 'depth' is the deployable choice on a "
                             "robot with cameras and no heightmap; the tracker it is paired "
                             "with must agree (checked at load in deploy/gen_tracker.py).")
    parser.add_argument(
        "--reference_blocks", type=str, default=None,
        help="Comma-separated reference-window blocks the generator predicts, out of "
             f"{','.join(REFERENCE_WINDOW_BLOCK_NAMES)}. Default: whatever the RECORDING's "
             "teacher observed (RecordingMeta.reference_blocks), which is what the "
             "generator has to predict to drive that policy. Set this only to train for a "
             "DIFFERENT teacher than the one recorded -- the agreement is enforced at play "
             "time and at deploy load, not here.",
    )
    parser.add_argument("--null_cond_prob", type=float, default=FlowMatchingConfig().null_cond_prob,
                        help="Probability a training sample's history (phase) tokens are replaced by "
                             "the model's learned null embedding (classifier-free guidance / phase-agnostic).")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument(
        "--target_source", type=str, default="auto",
        choices=["auto", "ego_anchored_clip_future"],
        help="Future target. 'ego_anchored_clip_future': the clip ego-anchored onto the robot at "
             "the snap (recorded ref_future) -- the robot-state-conditioned paradigm. Default 'auto' "
             "resolves to it when the recording has ref_future.",
    )
    parser.add_argument(
        "--cmd_array", type=str, default="auto",
        help="Commanded-velocity profile over the window, sampled every --cmd_stride steps into "
             "conditioning command tokens: a recorded array name (e.g. 'target_vel'). 'auto' -> "
             "'target_vel' if present in the recording, else no command. 'none' -> no command.",
    )
    parser.add_argument("--cmd_stride", type=int, default=5,
                        help="Keep every cmd_stride-th step of the command profile as a token.")
    parser.add_argument(
        "--recovery_mode", type=str, default="clean_prediction",
        choices=["clean_prediction", "all"],
        help="For perturbed recordings (record_data.py --perturb, which logs 'perturb_active'): "
             "'clean_prediction' (default) keeps only windows whose prediction region is clean while "
             "the history may be perturbed (learn recovery refs from a bad-state history); 'all' uses "
             "every window. Recordings without 'perturb_active' always use all windows.",
    )
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_sampling_steps", type=int, default=8)
    # Loss-weight CLI flags default to the FlowMatchingConfig() values, so config.py is the
    # single source of truth -- editing a default there changes a plain `train.py` run (a
    # bare `default=0.0` here would silently override the config and disable the loss).
    _wc = FlowMatchingConfig()
    # Frame finite-difference consistency regularizers (root + ee frames). dt defaults
    # to the recording's control period. The consistency losses are in physical units
    # (m/s, rad/s and m/s^2, rad/s^2).
    parser.add_argument("--dt", type=float, default=None,
                        help="Control dt (s) for the consistency finite differences; default: recording meta.")
    parser.add_argument("--vel_consistency_weight", type=float, default=_wc.vel_consistency_weight,
                        help="Weight on the frame velocity finite-difference consistency loss "
                             "(predicted pose d/dt vs predicted velocity).")
    parser.add_argument("--accel_consistency_weight", type=float, default=_wc.accel_consistency_weight,
                        help="Weight on the frame acceleration finite-difference consistency loss "
                             "(predicted velocity d/dt vs data velocity d/dt). tau-gated.")
    parser.add_argument("--joint_vel_consistency_weight", type=float, default=_wc.joint_vel_consistency_weight,
                        help="Weight on the joint-velocity finite-difference consistency "
                             "(predicted joint_pos d/dt vs predicted joint_vel). Ungated.")
    parser.add_argument("--fk_pose_weight", type=float, default=_wc.fk_pose_weight,
                        help="Weight on the FK position consistency (predicted ee position, "
                             "base-relative, vs FK of predicted joints). tau-gated.")
    parser.add_argument("--consistency_tau_min", type=float, default=FlowMatchingConfig().consistency_tau_min,
                        help="Only score the acceleration + FK terms on samples with flow-time "
                             "tau >= this (velocity/joint-velocity stay ungated).")
    parser.add_argument("--fk_model_path", type=str, default=FlowMatchingConfig().fk_model_path,
                        help="MuJoCo MJCF defining the kinematics for the FK losses (repo-relative).")
    parser.add_argument("--skip_distractors", action="store_true", default=_wc.skip_distractors,
                        help="Drop DISTRACTOR-family windows (boxes the robot walks AROUND, not "
                             "onto). Train-time switch -- recordings tag every window with "
                             "'is_distractor', so A/B-ing needs no re-record. Errors if the "
                             "recording predates the tag.")
    # Observation noise (train-time augmentation on the conditioning/history tokens only;
    # mirrors the RL teacher/tracker obs corruption). Defaults to ON. Amplitudes are
    # physical uniform half-widths; the command (target_vel) is never noised.
    _dn = FlowMatchingConfig()
    parser.add_argument(
        "--obs_noise", action=argparse.BooleanOptionalAction, default=_dn.obs_noise,
        help="Add observation noise to the conditioning tokens during training (default: on; "
             "disable with --no-obs_noise). Off for val/eval/deployment regardless.",
    )
    parser.add_argument("--noise_scale", type=float, default=_dn.noise_scale,
                        help="Global multiplier on every noise amplitude.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = FlowMatchingConfig(
        future_len=args.future_len, num_history_states=args.num_history_states,
        extero_source=args.extero_source,
        reference_blocks=(
            [b.strip() for b in args.reference_blocks.split(",") if b.strip()]
            if args.reference_blocks else None
        ),
        null_cond_prob=args.null_cond_prob, stride=args.stride,
        val_fraction=args.val_fraction, target_source=args.target_source,
        recovery_mode=args.recovery_mode,
        cmd_array=args.cmd_array, cmd_stride=args.cmd_stride,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_sampling_steps=args.num_sampling_steps,
        vel_consistency_weight=args.vel_consistency_weight,
        accel_consistency_weight=args.accel_consistency_weight,
        joint_vel_consistency_weight=args.joint_vel_consistency_weight,
        fk_pose_weight=args.fk_pose_weight,
        consistency_tau_min=args.consistency_tau_min, fk_model_path=args.fk_model_path,
        skip_distractors=args.skip_distractors,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        num_epochs=args.num_epochs, grad_clip=args.grad_clip, seed=args.seed,
        obs_noise=args.obs_noise, noise_scale=args.noise_scale,
    )
    return cfg, args


def channel_metrics(
    pred: torch.Tensor, tgt: torch.Tensor, num_joints: int, num_extra_frames: int = 0,
    blocks=None,
) -> dict[str, float]:
    """Per-channel physical-units error between predicted and target futures.

    Reports only the blocks the window actually packs -- a metric for an absent block
    would either index a neighbour's columns or invent a zero, and both read as "that
    channel is fine". Extra frames are position-only, so there is no per-frame
    orientation geodesic.

    Args:
        pred: Predicted future states (denormalized), shape ``(B, F, D)``.
        tgt: Target future states (denormalized), shape ``(B, F, D)``.
        num_joints: Number of joints ``nj``.
        num_extra_frames: Number of extra body frames ``K`` appended to the state.
        blocks: The window's block selection; defaults to all five.

    Returns:
        Dict of RMSEs per selected channel plus, with ``base_pose``, the orientation
        geodesic error (radians).
    """
    def _rmse(slc: slice) -> float:
        return torch.sqrt(torch.mean((pred[..., slc] - tgt[..., slc]) ** 2)).item()

    def _geodesic(slc: slice) -> float:
        """Angle between the predicted and target base orientations, in radians.

        Through ``Exp`` rather than differencing the log values, so the metric stays
        meaningful (and wrap-immune) however far the reference has turned from the anchor.
        """
        rel = so3_exp(pred[..., slc]).transpose(-1, -2) @ so3_exp(tgt[..., slc])
        cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        return torch.arccos(cos).mean().item()

    b = window_block_slices(num_joints, num_extra_frames, blocks)
    out: dict[str, float] = {}
    if "base_pose" in b:
        start = b["base_pose"].start
        out["rmse_base_pos"] = _rmse(slice(start, start + 3))
        out["ori_geodesic_rad"] = _geodesic(slice(start + 3, start + 6))
    if "base_vel" in b:
        start = b["base_vel"].start
        out["rmse_base_lin_vel"] = _rmse(slice(start, start + 3))
        out["rmse_base_ang_vel"] = _rmse(slice(start + 3, start + 6))
    for name, key in (("joint_pos", "rmse_joint_pos"), ("joint_vel", "rmse_joint_vel")):
        if name in b:
            out[key] = _rmse(b[name])
    if "ee_pos" in b and num_extra_frames > 0:
        out["rmse_ee_pos"] = _rmse(b["ee_pos"])
    return out


_CONS_KEYS = ("vel_consistency", "accel_consistency", "joint_vel_consistency", "fk_pose")


@torch.no_grad()
def _fk_sanity_check(fk, train_ds, num_joints, device, n_windows: int = 64) -> None:
    """Sanity baseline: FK(recorded joints) vs the recorded ee pose (base frame).

    On clean recorded data the FK of the joints should reproduce the recorded
    end-effector frames (~0), validating the joint/frame name+order mapping and that
    the MJCF kinematics match the recording. A large value warns of a model mismatch
    that would floor the FK loss.
    """
    nj, k = num_joints, train_ds.num_extra_frames
    s = train_ds.state[:n_windows].to(device)
    x = train_ds.state_normalizer.denormalize(s)  # (W, F, D) physical
    w, f = x.shape[0], x.shape[1]
    blocks = window_block_slices(nj, k, train_ds.blocks)
    fk_pos, _, _, _ = fk.pose_and_jacobian(x[..., blocks["joint_pos"]].reshape(w * f, nj))
    fk_pos = fk_pos.reshape(w, f, k, 3)
    # The window's ee block is ALREADY base-relative, so this is a direct residual.
    ee = x[..., blocks["ee_pos"]].reshape(w, f, k, 3)
    errs = [float((ee[:, :, kf] - fk_pos[:, :, kf]).norm(dim=-1).mean()) for kf in range(k)]
    mean_mm = 1000.0 * sum(errs) / max(len(errs), 1)
    print(f"[INFO] FK sanity (FK(recorded joints) vs recorded ee, base frame): "
          f"mean {mean_mm:.1f} mm | per-frame mm {[round(1000 * e, 1) for e in errs]}")
    if mean_mm > 20.0:
        print("[WARN] FK baseline > 20 mm: the MJCF kinematics may not match the recording "
              "(joint order/names, base link, or model mismatch); the FK loss will have this floor.")


def consistency_kwargs(cfg, num_joints, num_extra_frames, fk) -> dict:
    """Shared flow_matching_loss kwargs for the consistency + FK regularizers."""
    return dict(
        dt=cfg.dt, num_joints=num_joints, num_extra_frames=num_extra_frames,
        vel_weight=cfg.vel_consistency_weight, accel_weight=cfg.accel_consistency_weight,
        joint_vel_weight=cfg.joint_vel_consistency_weight,
        fk=fk, fk_pose_weight=cfg.fk_pose_weight,
        tau_min=cfg.consistency_tau_min, blocks=cfg.reference_blocks,
    )


@torch.no_grad()
def evaluate(model, loader, cfg, state_normalizer, num_joints, device, num_extra_frames=0,
             fk=None) -> dict[str, float]:
    """Compute conditioned/null-history flow loss and per-channel sampling metrics.

    Metrics are over the full F-step prediction; ``flow_mse`` is the history-conditioned
    loss and ``flow_mse_null`` replaces the history with the learned null embedding.
    """
    model.eval()
    flow_sum, flow_null_sum, n_batches = 0.0, 0.0, 0
    cons_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    ckw = consistency_kwargs(cfg, num_joints, num_extra_frames, fk)
    for batch in loader:
        state = batch["state"].to(device)
        history = batch["history"].to(device)
        height = batch["height"].to(device)
        cmd_tokens = batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None
        b = state.shape[0]
        cond_out = flow_matching_loss(model, state, history, height, cmd_tokens,
                                      state_norm=state_normalizer, **ckw)
        flow_sum += cond_out["flow_mse"].item()
        for k in _CONS_KEYS:
            if k in cond_out:
                cons_sums[k] = cons_sums.get(k, 0.0) + cond_out[k].item()
        null_mask = torch.ones(b, dtype=torch.bool, device=device)
        flow_null_sum += flow_matching_loss(
            model, state, history, height, cmd_tokens, history_null_mask=null_mask
        )["flow_mse"].item()
        n_batches += 1
        pred = sample_trajectory(
            model, history, height, cmd_tokens, cfg.future_len, cfg.state_dim, cfg.num_sampling_steps,
        )
        pred_full = state_normalizer.denormalize(pred)
        tgt_d = state_normalizer.denormalize(state)
        for k, v in channel_metrics(pred_full, tgt_d, num_joints, num_extra_frames,
                                    cfg.reference_blocks).items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v
    out = {
        "flow_mse": flow_sum / max(n_batches, 1),
        "flow_mse_null": flow_null_sum / max(n_batches, 1),
    }
    for k, v in cons_sums.items():
        out[k] = v / max(n_batches, 1)
    for k, v in metric_sums.items():
        out[k] = v / max(n_batches, 1)
    return out


def main() -> None:
    """Entry point: load data, build datasets, train, validate, checkpoint."""
    cfg, args = parse_args()
    torch.manual_seed(cfg.seed)

    # Run-dir naming matches RL: <timestamp>[_<run_name>].
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{ts}_{args.run_name}" if args.run_name else ts
    out_dir = os.path.join(args.logdir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    data_path = _resolve_data_path(args.data)
    print(f"[INFO] Loading dataset: {data_path}")
    if is_windowed_recording(data_path):
        data, meta = load_recorded_windows(data_path)
    else:
        data, meta = load_recorded_data(data_path)
    # Auto-config from the recording's content (only the 'auto' sentinel infers;
    # an explicit --cmd_array / --target_source always wins).
    cfg.cmd_array = _resolve_cmd_array(cfg.cmd_array, data)
    cfg.target_source = _resolve_target_source(cfg.target_source, data)
    # Control dt for the consistency finite differences: --dt wins, else the recording's.
    cfg.dt = float(args.dt) if args.dt is not None else float(meta.dt)
    print(f"[auto] target_source={cfg.target_source}, cmd_array={cfg.cmd_array or 'none'}")
    train_ds, val_ds = build_train_val(data, meta, cfg)
    num_joints = train_ds.num_joints
    print(f"[INFO] window blocks={list(train_ds.blocks)} -> {cfg.state_dim}/node "
          f"({cfg.future_len * cfg.state_dim} over the horizon); the policy this drives "
          "must pack exactly these.")
    print(f"[INFO] target_source={cfg.target_source}, state_dim={cfg.state_dim}, "
          f"cmd_dim={cfg.cmd_dim}, num_cmd_tokens={cfg.num_cmd_tokens}, "
          f"extero={cfg.extero_source} ({cfg.extero_tokens}x{cfg.height_dim}), "
          f"F={cfg.future_len}, N={cfg.num_history_states}, "
          f"null_cond_prob={cfg.null_cond_prob}, train_windows={len(train_ds)}, "
          f"val_windows={0 if val_ds is None else len(val_ds)}")
    if cfg.obs_noise:
        # Only the keys feeding a SELECTED block -- the others are not in the window and
        # so are not applied, and printing them would suggest otherwise.
        live = [k for b in train_ds.blocks for k in REFERENCE_BLOCK_KEYS[b]]
        print(f"[INFO] obs_noise=on (conditioning tokens, train split only), "
              f"scale={cfg.noise_scale} x window.REFERENCE_WINDOW_NOISE "
              f"(the same table the policy's reference obs is corrupted with): "
              + " ".join(f"{k}={REFERENCE_WINDOW_NOISE[k]}" for k in live))
    else:
        print("[INFO] obs_noise=off (clean conditioning tokens)")

    # Optional held-out eval recording (separate dataset, never trained on; reuses the
    # train normalizers so eval/* metrics are leakage-free and comparable to val/*).
    eval_path = _resolve_data_path(args.eval_data) if args.eval_data else None
    eval_ds = _build_eval_dataset(eval_path, cfg, train_ds) if eval_path else None
    eval_sha = sha256_file(eval_path) if eval_path else None
    if eval_ds is not None:
        print(f"[INFO] eval_data: {len(eval_ds)} windows from {eval_path} (sha {eval_sha[:12]})")

    # -- run provenance (parity with RL: git snapshot + params yaml + data identity) --
    # cfg dims (state_dim/cmd_dim) are resolved by build_train_val above,
    # so fm_config.yaml captures the *resolved* config.
    data_path_abs = os.path.abspath(data_path)
    data_sha256 = sha256_file(data_path)
    git_snapshot(out_dir)
    dump_params_yaml(os.path.join(out_dir, "params", "fm_config.yaml"), cfg.to_dict())
    dump_params_yaml(os.path.join(out_dir, "params", "data.yaml"), {
        "data": data_path_abs,
        "data_sha256": data_sha256,
        "recording_meta": asdict(meta) if hasattr(meta, "__dataclass_fields__") else dict(vars(meta)),
        "eval_data": os.path.abspath(eval_path) if eval_path else None,
        "eval_data_sha256": eval_sha,
        # Held-out val episode ids (by episode_id) so the test bench can label the
        # forecast_step --forecast-split eval set deterministically.
        "val_episode_ids": getattr(train_ds, "val_episode_ids", []),
    })
    print(f"[INFO] Provenance -> {out_dir}/params/{{fm_config,data}}.yaml + git/  (data sha {data_sha256[:12]})")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = FlowMatchingTransformer(
        state_dim=cfg.state_dim, future_len=cfg.future_len,
        num_history_states=cfg.num_history_states, history_dim=cfg.state_dim,
        height_dim=cfg.height_dim, extero_tokens=cfg.extero_tokens, cmd_dim=cfg.cmd_dim,
        cmd_stride=cfg.cmd_stride, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
    ).to(device)
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"(command conditioning: {'on' if cfg.cmd_dim > 0 else 'off'})")

    # Normalizers are fit on CPU; move their stats to the model device so
    # (de)normalization during evaluation/sampling stays on one device.
    train_ds.state_normalizer.to(device)
    train_ds.height_normalizer.to(device)
    if train_ds.cmd_normalizer is not None:
        train_ds.cmd_normalizer.to(device)

    # Forward-kinematics chain for the FK consistency losses (only when weighted).
    # ``resolve_block_selection`` has already zeroed this weight if the window does not
    # pack joint_pos + ee_pos, so reaching here means the blocks are there.
    fk = None
    if cfg.fk_pose_weight > 0.0:
        if train_ds.num_extra_frames == 0:
            print("[WARN] fk_pose_weight set but the dataset has no extra (ee) "
                  "frames; skipping the FK loss.")
        else:
            fk_path = cfg.fk_model_path if os.path.isabs(cfg.fk_model_path) else os.path.join(_REPO_ROOT, cfg.fk_model_path)
            fk = FKChain(fk_path, list(meta.state_joint_names), list(train_ds.target_frame_names)).to(device)
            _fk_sanity_check(fk, train_ds, num_joints, device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs) if args.cosine else None
    )

    writer = SummaryWriter(log_dir=out_dir)
    wandb_run = None
    if args.wandb:
        # Never let telemetry take down a training run. Now that this is on by default, an
        # uninstalled wandb, a missing login or a network blip would otherwise kill a
        # multi-hour job before its first epoch. Degrade to TensorBoard (which is always
        # written) and say so loudly enough to notice at the top of the log.
        try:
            import wandb  # noqa: PLC0415
            wandb_run = wandb.init(project=args.wandb_project, name=run_name,
                                   config=cfg.to_dict())
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb logging is OFF: {type(exc).__name__}: {exc}\n"
                  f"       Training continues; scalars still go to TensorBoard in {out_dir}.\n"
                  "       Pass --no_wandb to silence this, or fix the login (wandb login).")

    best_val = math.inf
    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss, n = 0.0, 0
        for batch in train_loader:
            state = batch["state"].to(device)
            history = batch["history"].to(device)
            height = batch["height"].to(device)
            cmd_tokens = batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None
            # Per-sample history null-out (classifier-free guidance / phase-agnostic).
            null_mask = torch.rand(state.shape[0], device=device) < cfg.null_cond_prob
            opt.zero_grad()
            out = flow_matching_loss(
                model, state, history, height, cmd_tokens,
                history_null_mask=null_mask,
                state_norm=train_ds.state_normalizer,
                **consistency_kwargs(cfg, num_joints, train_ds.num_extra_frames, fk),
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += out["loss"].item()
            n += 1
            global_step += 1
            writer.add_scalar("train/loss_step", out["loss"].item(), global_step)
            for k in ("flow_mse", *_CONS_KEYS):
                if k in out:
                    writer.add_scalar(f"train/{k}_step", out[k].item(), global_step)
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        train_loss = epoch_loss / max(n, 1)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/epoch_time_s", epoch_time, epoch)
        log_line = f"[epoch {epoch}] train_loss={train_loss:.4f}  train_time={epoch_time:.1f}s"

        if val_loader is not None and (epoch % args.val_every == 0 or epoch == cfg.num_epochs - 1):
            val = evaluate(
                model, val_loader, cfg, train_ds.state_normalizer, num_joints, device,
                num_extra_frames=train_ds.num_extra_frames, fk=fk,
            )
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            log_line += "  " + "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            if val["flow_mse"] < best_val:
                best_val = val["flow_mse"]
                _save_ckpt(os.path.join(out_dir, "best.pt"), model, opt, cfg, train_ds, meta, epoch,
                           data_path=data_path_abs, data_sha256=data_sha256)
            if wandb_run is not None:
                wandb_run.log({**{f"val/{k}": v for k, v in val.items()}, "epoch": epoch})

        if eval_loader is not None and (epoch % args.val_every == 0 or epoch == cfg.num_epochs - 1):
            ev = evaluate(
                model, eval_loader, cfg, train_ds.state_normalizer, num_joints, device,
                num_extra_frames=train_ds.num_extra_frames, fk=fk,
            )
            for k, v in ev.items():
                writer.add_scalar(f"eval/{k}", v, epoch)
            log_line += "  " + "  ".join(f"eval_{k}={v:.4f}" for k, v in ev.items())
            if wandb_run is not None:
                wandb_run.log({**{f"eval/{k}": v for k, v in ev.items()}, "epoch": epoch})

        print(log_line)
        if wandb_run is not None:
            wandb_run.log({"train/loss_epoch": train_loss, "train/epoch_time_s": epoch_time, "epoch": epoch})
        if epoch % args.ckpt_every == 0 or epoch == cfg.num_epochs - 1:
            _save_ckpt(os.path.join(out_dir, f"ckpt_{epoch}.pt"), model, opt, cfg, train_ds, meta, epoch,
                       data_path=data_path_abs, data_sha256=data_sha256)

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[INFO] Done. Checkpoints in {out_dir}")


def _resolve_data_path(path: str) -> str:
    """Accept a recorded ``.npz``/``.h5`` file or a directory.

    For a directory (e.g. a run dir or its ``recorded_data/``), return the newest
    ``recorded_*.npz`` or ``recorded_*.h5`` (windowed) found under it — so you can pass a
    run name instead of the full timestamped filename.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        import glob  # noqa: PLC0415
        cands = glob.glob(os.path.join(path, "**", "recorded_*.npz"), recursive=True)
        cands += glob.glob(os.path.join(path, "**", "recorded_*.h5"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"No recorded_*.npz/.h5 found under directory: {path}")
        return max(cands, key=os.path.getmtime)
    raise FileNotFoundError(f"--data not found: {path}")


def _resolve_cmd_array(value, data) -> str | None:
    """Resolve the ``--cmd_array`` sentinel against the recording's arrays.

    ``auto`` -> ``target_vel`` if recorded, else no command; ``none`` -> no command;
    any other value is taken as an explicit array name.
    """
    if value in (None, "none", "None"):
        return None
    if value == "auto":
        # Per-step recordings carry the command as ``target_vel``; windowed recordings
        # carry it pre-strided as ``cmd_future``. Either enables command conditioning
        # (the dataset reads the command from ``cmd_future`` in both cases).
        has_cmd = data.get("target_vel") is not None or data.get("cmd_future") is not None
        return "target_vel" if has_cmd else None
    return value


def _resolve_target_source(value, data) -> str:
    """Resolve the ``--target_source`` sentinel: ``auto`` -> 'ego_anchored_clip_future'
    when the recording has ``ref_future`` (the robot-state-conditioned FM target)."""
    if value == "auto":
        if data.get("ref_future") is None:
            raise ValueError(
                "Recording has no 'ref_future'; the robot-state-conditioned FM needs it. "
                "Re-record with scripts/rsl_rl/record_data.py (ego_anchor_fraction=1.0)."
            )
        return "ego_anchored_clip_future"
    return value


def _build_eval_dataset(eval_path: str, cfg, train_ds) -> FutureStateDataset:
    """Build a held-out eval dataset from a *separate* recording.

    Uses ALL of its valid windows (no train/val split) and reuses the train
    normalizers, so eval metrics are leakage-free and directly comparable to val.
    The eval recording must share the train layout (validated below).
    """
    if is_windowed_recording(eval_path):
        # Pre-formed windowed (.h5) eval set: use ALL windows (no split).
        data, meta = load_recorded_windows(eval_path)
        windows = np.arange(int(data["history_state"].shape[0]), dtype=np.int64)
        if windows.size == 0:
            raise ValueError(f"--eval_data has no windows: {eval_path}")
    else:
        data, meta = load_recorded_data(eval_path)
        if data.get("reanchor_active") is None:
            raise ValueError(f"--eval_data has no 'reanchor_active' (snap flag): {eval_path}")
        windows = snap_windows(
            data["episode_id"], data["step_in_episode"], data["done"],
            data["reanchor_active"], cfg.num_history_states, cfg.stride,
        )
        if not windows:
            raise ValueError(f"--eval_data produced no valid snap windows: {eval_path}")
    ds = FutureStateDataset(
        data, meta, cfg, windows,
        state_normalizer=train_ds.state_normalizer,
        cmd_normalizer=train_ds.cmd_normalizer,
        height_normalizer=train_ds.height_normalizer,
    )
    got = (ds.state_dim, ds.cmd_dim, ds.num_extra_frames)
    want = (train_ds.state_dim, train_ds.cmd_dim, train_ds.num_extra_frames)
    if got != want:
        raise ValueError(
            f"--eval_data layout (state_dim, cmd_dim, extra_frames)={got} "
            f"differs from train={want}; record eval data with the same task/flags."
        )
    return ds


def _save_ckpt(path, model, opt, cfg, train_ds, meta, epoch, data_path="", data_sha256="") -> None:
    """Save model weights, optimizer, config, normalizers, and dataset provenance.

    ``data_path``/``data_sha256`` record exactly which recorded dataset trained this
    checkpoint, so a ``.pt`` is self-describing without the separate params YAML.

    The predicted extra-frame order is injected into the saved ``meta`` so
    ``online.load_fm_context`` (and the deploy-side ``transfer/deploy/fm_session.py``)
    reconstruct the exact token layout.
    """
    meta_dict = dict(asdict(meta) if hasattr(meta, "__dataclass_fields__") else vars(meta))
    meta_dict["target_frame_names"] = list(train_ds.target_frame_names)
    # The contract of the sensor this model ACTUALLY conditions on (None for the height
    # scan). Distinct from the recording's ``depth_camera_spec``, which is present
    # whenever the cameras were recorded even if this model trained on the scan instead.
    meta_dict["extero_spec"] = train_ds.extero_spec
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg.to_dict(),
        "state_normalizer": train_ds.state_normalizer.state_dict(),
        "height_normalizer": train_ds.height_normalizer.state_dict(),
        "num_joints": train_ds.num_joints,
        "meta": meta_dict,
        "data_path": data_path,
        "data_sha256": data_sha256,
        "epoch": epoch,
    }
    if train_ds.cmd_normalizer is not None:
        ckpt["cmd_normalizer"] = train_ds.cmd_normalizer.state_dict()
    torch.save(ckpt, path)


if __name__ == "__main__":
    main()
