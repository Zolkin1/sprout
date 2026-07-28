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
    FlatTerrain,
    FlowMatchingConfig,
    FlowMatchingTransformer,
    FutureStateDataset,
    box_sole_points,
    build_train_val,
    flow_matching_loss,
    frame_block_start,
    matrix_from_quat,
    orientation_slice,
    rot6d_to_matrix,
    sample_trajectory,
    snap_windows,
    terrain_penetration_term,
    yaw_quat,
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
    parser.add_argument("--wandb", action="store_true", help="Also log to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="robot_rl_flow_matching")
    # config fields
    parser.add_argument("--future_len", type=int, default=FlowMatchingConfig().future_len,
                        help="Window length F (state tokens). Effective novel horizon is "
                             "F - num_history_states (default from config.py, sized to cover the "
                             "deploy-time policy lookahead; see FlowMatchingConfig.future_len).")
    parser.add_argument("--num_history_states", type=int, default=2,
                        help="N most-recent real-robot states used as history (phase) conditioning tokens.")
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
    parser.add_argument("--ori6d_aux_weight", type=float, default=_wc.ori6d_aux_weight)
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
                        help="Weight on the FK pose consistency (predicted ee pose vs FK of "
                             "predicted joints). tau-gated.")
    parser.add_argument("--fk_vel_weight", type=float, default=_wc.fk_vel_weight,
                        help="Weight on the FK velocity consistency (predicted ee velocity vs "
                             "J(q)*qdot of predicted joints/joint-vels). tau-gated.")
    parser.add_argument("--consistency_tau_min", type=float, default=FlowMatchingConfig().consistency_tau_min,
                        help="Only score the acceleration + FK terms on samples with flow-time "
                             "tau >= this (velocity/joint-velocity stay ungated).")
    parser.add_argument("--fk_model_path", type=str, default=FlowMatchingConfig().fk_model_path,
                        help="MuJoCo MJCF defining the kinematics for the FK losses (repo-relative).")
    # Terrain penetration (penalize predicted feet below the ground). tau-gated.
    parser.add_argument("--terrain_penetration_weight", type=float, default=_wc.terrain_penetration_weight,
                        help="Weight on the terrain-penetration loss (predicted feet below ground).")
    parser.add_argument("--terrain_penetration_mode", type=str, default=_wc.terrain_penetration_mode,
                        choices=["ankle", "contact"],
                        help="'ankle' (penalize ankle-frame origin) or 'contact' (box-sole grid points).")
    parser.add_argument("--terrain_ground_z", type=float, default=None,
                        help="World ground height (m); default None -> auto-calibrate from the recording.")
    parser.add_argument("--terrain_contact_grid", type=int, nargs=2, default=list(FlowMatchingConfig().terrain_contact_grid),
                        help="(nx ny) sole-grid resolution for the 'contact' mode.")
    parser.add_argument("--terrain_floor_percentile", type=float, default=FlowMatchingConfig().terrain_floor_percentile,
                        help="Low percentile of recorded foot z used as the auto-calibrated ground.")
    parser.add_argument("--terrain_tau_gated", action="store_true", default=_wc.terrain_tau_gated,
                        help="Restrict the terrain-penetration loss to tau >= consistency_tau_min "
                             "(like accel/FK). Default off: the low-tau endpoint penetrates like the "
                             "deployed sample, so gating hides the signal.")
    parser.add_argument("--terrain_sample_weight", type=float, default=_wc.terrain_sample_weight,
                        help="Weight on the sampled-rollout terrain penalty (penalizes the actual "
                             "multi-step sample, not x1_hat). >0 enables it; same m^2 units as "
                             "--terrain_penetration_weight (expect a similar ~1e2 scale).")
    parser.add_argument("--terrain_sample_size", type=int, default=_wc.terrain_sample_size,
                        help="Sub-batch size for the differentiable sampled rollout (caps cost).")
    parser.add_argument("--terrain_sample_steps", type=int, default=_wc.terrain_sample_steps,
                        help="Euler steps for the training-time sampled rollout.")
    parser.add_argument("--terrain_sample_grad_steps", type=int, default=_wc.terrain_sample_grad_steps,
                        help="Truncated-backprop depth: only the last N rollout steps carry gradient.")
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
    parser.add_argument("--noise_joint_pos", type=float, default=_dn.noise_joint_pos)
    parser.add_argument("--noise_joint_vel", type=float, default=_dn.noise_joint_vel)
    parser.add_argument("--noise_pos", type=float, default=_dn.noise_pos)
    parser.add_argument("--noise_lin_vel", type=float, default=_dn.noise_lin_vel)
    parser.add_argument("--noise_ori", type=float, default=_dn.noise_ori)
    parser.add_argument("--noise_ang_vel", type=float, default=_dn.noise_ang_vel)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = FlowMatchingConfig(
        future_len=args.future_len, num_history_states=args.num_history_states,
        null_cond_prob=args.null_cond_prob, stride=args.stride,
        val_fraction=args.val_fraction, target_source=args.target_source,
        recovery_mode=args.recovery_mode,
        cmd_array=args.cmd_array, cmd_stride=args.cmd_stride,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_sampling_steps=args.num_sampling_steps, ori6d_aux_weight=args.ori6d_aux_weight,
        vel_consistency_weight=args.vel_consistency_weight,
        accel_consistency_weight=args.accel_consistency_weight,
        joint_vel_consistency_weight=args.joint_vel_consistency_weight,
        fk_pose_weight=args.fk_pose_weight, fk_vel_weight=args.fk_vel_weight,
        consistency_tau_min=args.consistency_tau_min, fk_model_path=args.fk_model_path,
        terrain_penetration_weight=args.terrain_penetration_weight,
        terrain_penetration_mode=args.terrain_penetration_mode,
        terrain_ground_z=args.terrain_ground_z,
        terrain_contact_grid=tuple(args.terrain_contact_grid),
        terrain_floor_percentile=args.terrain_floor_percentile,
        terrain_tau_gated=args.terrain_tau_gated,
        terrain_sample_weight=args.terrain_sample_weight,
        terrain_sample_size=args.terrain_sample_size,
        terrain_sample_steps=args.terrain_sample_steps,
        terrain_sample_grad_steps=args.terrain_sample_grad_steps,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        num_epochs=args.num_epochs, grad_clip=args.grad_clip, seed=args.seed,
        obs_noise=args.obs_noise, noise_scale=args.noise_scale,
        noise_joint_pos=args.noise_joint_pos, noise_joint_vel=args.noise_joint_vel,
        noise_pos=args.noise_pos, noise_lin_vel=args.noise_lin_vel,
        noise_ori=args.noise_ori, noise_ang_vel=args.noise_ang_vel,
    )
    return cfg, args


def channel_metrics(
    pred: torch.Tensor, tgt: torch.Tensor, num_joints: int, num_extra_frames: int = 0
) -> dict[str, float]:
    """Per-channel physical-units error between predicted and target futures.

    The base (root) + joint channels occupy a fixed prefix unchanged by extra
    frames, so their offsets are stable. When ``num_extra_frames > 0`` the extra
    body-frame position RMSE and orientation geodesic are aggregated (mean over
    frames) into ``rmse_frame_pos`` / ``frame_ori_geodesic_rad``.

    Args:
        pred: Predicted future states (denormalized), shape ``(B, F, D)``.
        tgt: Target future states (denormalized), shape ``(B, F, D)``.
        num_joints: Number of joints ``nj``.
        num_extra_frames: Number of extra body frames ``K`` appended to the state.

    Returns:
        Dict of RMSEs per channel plus the orientation geodesic error (radians).
    """
    nj = num_joints

    def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.sqrt(torch.mean((a - b) ** 2)).item()

    def _geodesic(o0: int, o1: int) -> float:
        r_pred = rot6d_to_matrix(pred[..., o0:o1])
        r_tgt = rot6d_to_matrix(tgt[..., o0:o1])
        rel = r_pred.transpose(-1, -2) @ r_tgt
        cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        return torch.arccos(cos).mean().item()

    bo0, bo1 = orientation_slice(nj)
    out = {
        "rmse_joint_pos": _rmse(pred[..., 0:nj], tgt[..., 0:nj]),
        "rmse_joint_vel": _rmse(pred[..., nj:2 * nj], tgt[..., nj:2 * nj]),
        "rmse_root_pos": _rmse(pred[..., 2 * nj:2 * nj + 3], tgt[..., 2 * nj:2 * nj + 3]),
        "rmse_root_lin_vel": _rmse(pred[..., 2 * nj + 3:2 * nj + 6], tgt[..., 2 * nj + 3:2 * nj + 6]),
        "rmse_root_ang_vel": _rmse(pred[..., 2 * nj + 12:2 * nj + 15], tgt[..., 2 * nj + 12:2 * nj + 15]),
        "ori_geodesic_rad": _geodesic(bo0, bo1),
    }
    if num_extra_frames > 0:
        pos_rmse, geo = 0.0, 0.0
        for k in range(num_extra_frames):
            b = frame_block_start(nj, 1 + k)
            pos_rmse += _rmse(pred[..., b:b + 3], tgt[..., b:b + 3])
            geo += _geodesic(b + 6, b + 12)
        out["rmse_frame_pos"] = pos_rmse / num_extra_frames
        out["frame_ori_geodesic_rad"] = geo / num_extra_frames
    return out


_CONS_KEYS = ("vel_consistency", "accel_consistency", "joint_vel_consistency", "fk_pose", "fk_vel",
              "terrain_penetration", "terrain_penetration_sample")


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
    fk_pos, _, _, _ = fk.pose_and_jacobian(x[..., :nj].reshape(w * f, nj))
    fk_pos = fk_pos.reshape(w, f, k, 3)
    rb = frame_block_start(nj, 0)
    p_root = x[..., rb:rb + 3]
    r_root = rot6d_to_matrix(x[..., rb + 6:rb + 12])
    errs = []
    for kf in range(k):
        eb = frame_block_start(nj, 1 + kf)
        p_ee_base = torch.einsum("wfji,wfj->wfi", r_root, x[..., eb:eb + 3] - p_root)
        errs.append(float((p_ee_base - fk_pos[:, :, kf]).norm(dim=-1).mean()))
    mean_mm = 1000.0 * sum(errs) / max(len(errs), 1)
    print(f"[INFO] FK sanity (FK(recorded joints) vs recorded ee, base frame): "
          f"mean {mean_mm:.1f} mm | per-frame mm {[round(1000 * e, 1) for e in errs]}")
    if mean_mm > 20.0:
        print("[WARN] FK baseline > 20 mm: the MJCF kinematics may not match the recording "
              "(joint order/names, base link, or model mismatch); the FK loss will have this floor.")


@torch.no_grad()
def _terrain_calibrate(train_ds, foot_idxs, contact_pts, num_joints, device, percentile, n_windows: int = 128) -> float:
    """Calibrate the world ground height from the recording (a low percentile of the
    lifted recorded foot z), so the clean recorded motion has ~0 penetration.

    Lifts the recorded foot frames (ankle origin, or the box-sole ``contact_pts``) to the
    world via the per-window anchor ``(p_ref, q_ref)`` and returns ``percentile``-th
    percentile of their world z.
    """
    s = train_ds.state[:n_windows].to(device)
    x = train_ds.state_normalizer.denormalize(s)            # (W, F, D) physical (ref frame)
    p_ref = train_ds.p_ref[:n_windows].to(device)
    q_ref = train_ds.q_ref[:n_windows].to(device)
    r_ref = matrix_from_quat(yaw_quat(q_ref))                # (W, 3, 3) yaw-ground->world (matches rel frame)
    cpts = None if contact_pts is None else contact_pts.to(x)
    zs = []
    for fi in foot_idxs:
        eb = frame_block_start(num_joints, 1 + fi)
        p_foot = torch.einsum("wij,wfj->wfi", r_ref, x[..., eb:eb + 3]) + p_ref[:, None, :]
        if cpts is None:
            zs.append(p_foot[..., 2].reshape(-1))
        else:
            r_foot = torch.einsum("wij,wfjk->wfik", r_ref, rot6d_to_matrix(x[..., eb + 6:eb + 12]))
            pc = p_foot[:, :, None, :] + torch.einsum("wfik,pk->wfpi", r_foot, cpts)
            zs.append(pc[..., 2].reshape(-1))
    z = torch.cat(zs)
    floor = float(torch.quantile(z, percentile / 100.0))
    mode = "contact" if contact_pts is not None else "ankle"
    print(f"[INFO] terrain floor calibrated (mode={mode}): ground_z={floor:.4f} m "
          f"(recorded foot z: min={float(z.min()):.4f}, p{percentile:g}={floor:.4f}, "
          f"median={float(z.median()):.4f})")
    return floor


def consistency_kwargs(cfg, num_joints, num_extra_frames, fk,
                       terrain=None, foot_idxs=None, contact_pts=None) -> dict:
    """Shared flow_matching_loss kwargs for the consistency + FK + terrain regularizers."""
    return dict(
        dt=cfg.dt, num_joints=num_joints, num_extra_frames=num_extra_frames,
        vel_weight=cfg.vel_consistency_weight, accel_weight=cfg.accel_consistency_weight,
        joint_vel_weight=cfg.joint_vel_consistency_weight,
        fk=fk, fk_pose_weight=cfg.fk_pose_weight, fk_vel_weight=cfg.fk_vel_weight,
        tau_min=cfg.consistency_tau_min,
        terrain=terrain, foot_idxs=foot_idxs, contact_pts=contact_pts,
        terrain_weight=cfg.terrain_penetration_weight,
        terrain_tau_gated=cfg.terrain_tau_gated,
    )


@torch.no_grad()
def evaluate(model, loader, cfg, state_normalizer, num_joints, device, num_extra_frames=0,
             fk=None, terrain=None, foot_idxs=None, contact_pts=None) -> dict[str, float]:
    """Compute conditioned/null-history flow loss and per-channel sampling metrics.

    Metrics are over the full F-step prediction; ``flow_mse`` is the history-conditioned
    loss and ``flow_mse_null`` replaces the history with the learned null embedding.
    """
    model.eval()
    flow_sum, flow_null_sum, n_batches = 0.0, 0.0, 0
    cons_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    ckw = consistency_kwargs(cfg, num_joints, num_extra_frames, fk, terrain, foot_idxs, contact_pts)
    for batch in loader:
        state = batch["state"].to(device)
        history = batch["history"].to(device)
        height = batch["height"].to(device)
        cmd_tokens = batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None
        p_ref = batch["p_ref"].to(device)
        q_ref = batch["q_ref"].to(device)
        b = state.shape[0]
        cond_out = flow_matching_loss(model, state, history, height, cmd_tokens,
                                      state_norm=state_normalizer, p_ref=p_ref, q_ref=q_ref, **ckw)
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
        for k, v in channel_metrics(pred_full, tgt_d, num_joints, num_extra_frames).items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v
        # Terrain penetration of the deployed multi-step sample (reported regardless of
        # whether the sampled-rollout penalty is weighted).
        if terrain is not None and foot_idxs:
            pen_s = terrain_penetration_term(
                pred_full, p_ref, q_ref, num_joints, foot_idxs, contact_pts, terrain,
            ).mean()
            metric_sums["terrain_penetration_sample"] = (
                metric_sums.get("terrain_penetration_sample", 0.0) + pen_s.item()
            )
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
    print(f"[INFO] target_source={cfg.target_source}, state_dim={cfg.state_dim}, "
          f"cmd_dim={cfg.cmd_dim}, num_cmd_tokens={cfg.num_cmd_tokens}, "
          f"extero={cfg.extero_source} ({cfg.extero_tokens}x{cfg.height_dim}), "
          f"F={cfg.future_len}, N={cfg.num_history_states}, "
          f"null_cond_prob={cfg.null_cond_prob}, train_windows={len(train_ds)}, "
          f"val_windows={0 if val_ds is None else len(val_ds)}")
    if cfg.obs_noise:
        print(f"[INFO] obs_noise=on (conditioning tokens, train split only), scale={cfg.noise_scale} | "
              f"joint_pos={cfg.noise_joint_pos} joint_vel={cfg.noise_joint_vel} "
              f"pos={cfg.noise_pos} lin_vel={cfg.noise_lin_vel} ori={cfg.noise_ori} "
              f"ang_vel={cfg.noise_ang_vel}")
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
    fk = None
    if cfg.fk_pose_weight > 0.0 or cfg.fk_vel_weight > 0.0:
        if train_ds.num_extra_frames == 0:
            print("[WARN] fk_pose_weight/fk_vel_weight set but the dataset has no extra (ee) "
                  "frames; skipping the FK losses.")
        else:
            fk_path = cfg.fk_model_path if os.path.isabs(cfg.fk_model_path) else os.path.join(_REPO_ROOT, cfg.fk_model_path)
            fk = FKChain(fk_path, list(meta.state_joint_names), list(train_ds.target_frame_names)).to(device)
            _fk_sanity_check(fk, train_ds, num_joints, device)

    # Terrain-penetration loss: foot frames + contact points + flat-ground terrain.
    # Built when either the x1_hat term or the sampled-rollout term is active (and so the
    # eval-time sample-penetration metric can be reported).
    terrain = foot_idxs = contact_pts = None
    if cfg.terrain_penetration_weight > 0.0 or cfg.terrain_sample_weight > 0.0:
        foot_idxs = [i for i, fn in enumerate(train_ds.target_frame_names)
                     if "ankle" in fn.lower() or "foot" in fn.lower()]
        if not foot_idxs:
            print("[WARN] terrain_penetration_weight set but no foot frames (ankle/foot) in the "
                  "target frames; skipping the terrain loss.")
        else:
            contact_pts = (box_sole_points(grid=tuple(cfg.terrain_contact_grid)).to(device)
                           if cfg.terrain_penetration_mode == "contact" else None)
            ground_z = (cfg.terrain_ground_z if cfg.terrain_ground_z is not None
                        else _terrain_calibrate(train_ds, foot_idxs, contact_pts, num_joints, device,
                                                cfg.terrain_floor_percentile))
            terrain = FlatTerrain(ground_z).to(device)
            terms = []
            if cfg.terrain_penetration_weight > 0.0:
                terms.append(f"x1_hat(w={cfg.terrain_penetration_weight:g}, "
                             f"{'gated' if cfg.terrain_tau_gated else 'ungated'})")
            if cfg.terrain_sample_weight > 0.0:
                terms.append(f"sample(w={cfg.terrain_sample_weight:g}, size={cfg.terrain_sample_size}, "
                             f"steps={cfg.terrain_sample_steps}, grad_steps={cfg.terrain_sample_grad_steps})")
            print(f"[INFO] terrain loss: mode={cfg.terrain_penetration_mode}, foot frames="
                  f"{[train_ds.target_frame_names[i] for i in foot_idxs]}, ground_z={ground_z:.4f} | "
                  f"terms: {', '.join(terms) if terms else 'metric-only'}")

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
            state = batch["state"].to(device)
            history = batch["history"].to(device)
            height = batch["height"].to(device)
            cmd_tokens = batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None
            p_ref = batch["p_ref"].to(device)
            q_ref = batch["q_ref"].to(device)
            # Per-sample history null-out (classifier-free guidance / phase-agnostic).
            null_mask = torch.rand(state.shape[0], device=device) < cfg.null_cond_prob
            opt.zero_grad()
            out = flow_matching_loss(
                model, state, history, height, cmd_tokens,
                history_null_mask=null_mask, ori_slice=ori_slice, aux_weight=cfg.ori6d_aux_weight,
                state_norm=train_ds.state_normalizer, p_ref=p_ref, q_ref=q_ref,
                terrain_sample_weight=cfg.terrain_sample_weight,
                terrain_sample_size=cfg.terrain_sample_size,
                terrain_sample_steps=cfg.terrain_sample_steps,
                terrain_sample_grad_steps=cfg.terrain_sample_grad_steps,
                **consistency_kwargs(cfg, num_joints, train_ds.num_extra_frames, fk,
                                     terrain, foot_idxs, contact_pts),
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
                terrain=terrain, foot_idxs=foot_idxs, contact_pts=contact_pts,
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
                terrain=terrain, foot_idxs=foot_idxs, contact_pts=contact_pts,
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
    ``online.load_fm_context`` (and the deployment-side ``transfer/fm/online.py``)
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
