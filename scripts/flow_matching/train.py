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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Make the recording I/O module importable (it lives in scripts/rsl_rl).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_THIS_DIR, os.pardir, "rsl_rl"))
from recorded_data_io import (  # noqa: E402  isort: skip
    WINDOW_ARRAY_KEYS,
    load_recorded_data,
    load_recorded_windows,
    is_windowed_recording,
)
from provenance import dump_params_yaml, git_snapshot, sha256_file  # noqa: E402  isort: skip

from robot_rl.flow_matching import (  # noqa: E402
    FKChain,
    DEPTH_LATENCY_PROBS,
    FlowMatchingConfig,
    ATTENTION_MODES,
    EXTERO_ENCODERS,
    EXTERO_NORM_MODES,
    EXTERO_SOURCES,
    FlowMatchingTransformer,
    FutureStateDataset,
    build_train_val,
    flow_matching_loss,
    pinned_axis_split,
    sample_flow_batch,
    REFERENCE_WINDOW_BLOCK_NAMES,
    REFERENCE_WINDOW_NOISE,
    REFERENCE_BLOCK_KEYS,
    rot6d_to_matrix,
    window_block_slices,
    sample_trajectory,
    snap_windows,
)
from robot_rl.flow_matching.ema import WeightEMA  # noqa: E402
from motion_gen.joint_names import G1_BODY_TRACK_FRAMES  # noqa: E402


def parse_args() -> tuple[FlowMatchingConfig, argparse.Namespace]:
    """Parse CLI arguments into a config plus runtime options."""
    parser = argparse.ArgumentParser(description="Train a flow-matching future-state predictor.")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Recorded dataset. Accepts SEVERAL comma-separated paths, concatenated "
             "window-wise -- e.g. an original recording plus its registration-augmented "
             "counterpart, without writing a merged copy to disk. Episode ids are offset per "
             "file so the episode-wise train/val split cannot leak across them.")
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
        "--extero_channels", type=int, default=FlowMatchingConfig().extero_channels,
        help="Planes per exteroception PIXEL. 1 = depth alone. 4 adds the pixel's xyz in the "
             "root's yaw-aligned frame, which is what tells the model a step edge is '0.8 m "
             "ahead and 0.15 m up' rather than 'row 14, column 8, 2.1 m'. It matters because "
             "the generator sees no joint angles and no root orientation, so without it a "
             "waist turn or a pelvis pitch is indistinguishable from the terrain moving. "
             "DERIVED at dataset build from the recorded root quaternion, joint angles and "
             "the NOMINAL mount -- no re-record, and the same three inputs a robot has. "
             "Depth only; a scan's rays are already ground points.")
    parser.add_argument(
        "--extero_encoder", type=str, default=FlowMatchingConfig().extero_encoder,
        choices=["auto", *EXTERO_ENCODERS],
        help="How ONE exteroception reading becomes transformer tokens. 'cnn' = a 3-layer "
             "stride-2 conv trunk PER CAMERA, so each camera contributes a grid of tokens "
             "(16 on the 30x26 pair) each carrying a learned spatial slot -- the flat "
             "alternative is permutation-invariant over pixels, so nothing in the model can "
             "attend to WHERE in the frame a step edge is. 'mlp' = the original one token "
             "per reading, and the A/B baseline. 'auto' picks by source (depth -> cnn, "
             "scan -> mlp) -- 'auto' does NOT convolve a scan, because that would silently "
             "re-architect every existing scan run; pass 'cnn' explicitly, on a recording "
             "whose height_scan_spec states the grid's (H, W) fold.")
    parser.add_argument(
        "--extero_cnn_channels", type=str,
        default=",".join(str(c) for c in FlowMatchingConfig().extero_cnn_channels),
        help="Comma-separated output channels per stride-2 convolution; the COUNT sets the "
             "depth, so it sets how many times the image is halved. Three entries take "
             "26x30 to 4x4; a fourth reaches 2x2 natively, which is the cheaper way to "
             "shrink the sequence (pooling pays for the resolution and then discards it).")
    parser.add_argument(
        "--extero_cnn_grid", type=str,
        # Rendered from the dataclass for the same reason --extero_cnn_strides is: a literal
        # default here SHADOWS the field. It happens to agree today (both mean "no pooling"),
        # which is exactly what would make the next divergence silent.
        default=",".join(str(g) for g in (FlowMatchingConfig().extero_cnn_grid or [])),
        help="Optional 'gh,gw' to average-pool the conv trunk down to, i.e. tokens per "
             "camera. Empty keeps the trunk's native grid. This is the knob that separates "
             "'a CNN beats an MLP' from '16 tokens beat 1': '1,1' gives the legacy token "
             "count through a conv trunk.")
    parser.add_argument(
        "--attention_mode", type=str, default=FlowMatchingConfig().attention_mode,
        choices=list(ATTENTION_MODES),
        help="Which way the conditioning/future dependency runs. 'prefix' (default) has the "
             "conditioning attend only to ITSELF and keeps the flow-time embedding off it, so "
             "its representation is a fixed statement rather than a function of the noise "
             "level -- and is therefore computed once per re-plan instead of every Euler step. "
             "'full' is what every checkpoint before this flag was trained under; pass it for "
             "a controlled comparison against one.")
    parser.add_argument(
        "--state_scale_base_pos", type=float,
        default=FlowMatchingConfig().state_scale_base_pos,
        help="ONE declared scale (metres) for the base POSITION channels, replacing the fitted "
             "per-axis std. Fitted, x/y/z get (0.500, 0.111, 0.145) -- x is 4.5x y -- so a "
             "metre forward and a metre sideways reach the model as very different numbers. "
             "One scale keeps a metre a metre, and makes zero displacement map to zero "
             "(standardizing subtracts the ~0.354 m the robot advances over the horizon). "
             "0.4 gives a pooled std of 0.87 on the shipped library. Pass 0 to fit this block per channel instead.")
    parser.add_argument(
        "--state_scale_base_lin_vel", type=float,
        default=FlowMatchingConfig().state_scale_base_lin_vel,
        help="ONE declared scale (m/s) for the base LINEAR velocity channels. Fitted per axis "
             "gives (0.691, 0.228, 0.316); 0.5 gives a pooled std of 1.05. None = fit.")
    parser.add_argument(
        "--state_scale_base_ang_vel", type=float,
        default=FlowMatchingConfig().state_scale_base_ang_vel,
        help="ONE declared scale (rad/s) for the base ANGULAR velocity channels. Fitted per "
             "axis gives (0.426, 0.354, 0.428); 0.4 gives a pooled std of 1.01. None = fit.")
    parser.add_argument(
        "--state_scale_joint_pos", type=float,
        default=FlowMatchingConfig().state_scale_joint_pos,
        help="ONE declared scale (rad) for the joint-position channels. Unlike the geometric "
             "blocks there is no cross-channel metric to preserve -- different joints have "
             "genuinely different ranges (fitted std spans 0.067 to 0.504) -- so fitting per "
             "channel is usually right here. None = fit.")
    parser.add_argument(
        "--state_scale_joint_vel", type=float,
        default=FlowMatchingConfig().state_scale_joint_vel,
        help="ONE declared scale (rad/s) for the joint-velocity channels. None = fit.")
    parser.add_argument(
        "--state_scale_ee_pos", type=float,
        default=FlowMatchingConfig().state_scale_ee_pos,
        help="ONE declared scale (metres) for the end-effector position channels. None = fit.")
    parser.add_argument(
        "--cmd_scale", type=str,
        default=",".join(str(v) for v in (FlowMatchingConfig().cmd_scale or [])),
        help="Comma-separated per-channel DECLARED max for the command tokens "
             "'vx,vy,wz', e.g. '3.5,1.0,0.8'. Divides by the designed maximum with mean 0, so "
             "commands land in [-1, 1] by construction and a ZERO command stays zero -- "
             "standardizing does not (the fitted vx mean is 1.25 m/s, so 'stand still' would "
             "arrive as -1.29). Measured on clips/g1_mb_v5_body.h5, the library's actual "
             "extremes are vx 3.4989, vy 0.9998, wz 0.7999. Pass '' or 'none' to fit instead.")
    parser.add_argument(
        "--pinned_axis_loss_weight", type=float,
        default=FlowMatchingConfig().pinned_axis_loss_weight,
        help="Relative flow-loss weight on root DOFs the anchor mask PINS to the terrain -- "
             "height on flat nodes, downrange + height + yaw on stair/box nodes. 1.0 (default) "
             "is off. These are exactly the axes the generator gets wrong: it slides its "
             "predicted terrain segment along with a displaced robot instead of leaving it "
             "where the terrain put it. Weights are RENORMALIZED to mean 1, so the loss scale "
             "and the effective learning rate match an unweighted run and a with/without A/B "
             "differs in one thing only. Needs the recorded clip index and the clip library "
             "(joined and hash-checked, as --transition_fraction is).")
    parser.add_argument(
        "--extero_cnn_strides", type=str,
        # Rendered from the dataclass, as --cmd_scale does, so the CLI cannot shadow the
        # config's default with one of its own: a bare "" default here parsed to None and
        # OVERRODE the (1, 2, 2) field, which is how a run meant to be 56 tokens/camera
        # silently trained at 16 (sequence 111 rather than 191).
        default=",".join(str(s) for s in (FlowMatchingConfig().extero_cnn_strides or [])),
        help="Comma-separated per-layer convolution stride, one per --extero_cnn_channels "
             "entry. A LEADING stride-1 layer keeps the first stage at full resolution "
             "instead of throwing three quarters of the image away before the first "
             "non-linearity: '1,2,2' takes the G1's 26x30 depth pair to 7x8 = 56 tokens per "
             "camera, and '1,2,2,2' takes a dense 121x121 scan to 16x16 = 256. Pass '' for "
             "stride 2 everywhere, which is what every checkpoint before this flag trained "
             "under (16 tokens/camera on the depth pair). Must be as long as "
             "--extero_cnn_channels; a run that deepens the trunk states both. The grid "
             "target and this list are coupled, so changing one alone changes the "
             "transformer's sequence length.")
    parser.add_argument(
        "--extero_xyz_scale", type=float, default=FlowMatchingConfig().extero_xyz_scale,
        help="Declared divisor for the xyz channels under --extero_norm_mode contract, in "
             "metres. ONE constant for every camera and every axis -- that sharing is the "
             "point, since a per-camera or per-axis divisor makes a metre mean different "
             "things in different parts of the same input.")
    parser.add_argument(
        "--extero_nominal_ground_z", type=float,
        default=FlowMatchingConfig().extero_nominal_ground_z,
        help="Nominal ground plane, metres below the pelvis at zero joints, used to place "
             "each camera's declared xyz offset where its boresight meets the ground. "
             "Default: derived by FK from --fk_model_path (the ankle roll link, less the "
             "sole). Lowering it pushes the offsets FARTHER along each boresight, so it is "
             "one way to answer a large declared-vs-measured residual -- but it declares a "
             "ground plane that is not the ground; prefer stating --extero_xyz_offset.")
    parser.add_argument(
        "--extero_xyz_offset", type=str, default="",
        help="Declared per-camera xyz offsets in metres, overriding the derivation: "
             "'x,y,z;x,y,z' in sensor order (semicolons between cameras). The escape hatch "
             "when the boresight-on-nominal-ground default does not centre the data -- the "
             "dataset prints declared-vs-measured per camera at build, so set this from that "
             "line. Empty derives them. Stating them keeps the constant DECLARED and recorded "
             "in the run's config, which fitting the normalizer would not.")
    parser.add_argument(
        "--extero_norm_mode", type=str, default=FlowMatchingConfig().extero_norm_mode,
        choices=list(EXTERO_NORM_MODES),
        help="How the exteroception is scaled. Every mode is uniform over PIXELS, which is "
             "what a convolution's weight sharing needs. 'per_channel' (default) fits one "
             "(mean, std) per sensor per channel group -- d alone, xyz together so a metre "
             "stays a metre in every direction, and v left as an identity because it is "
             "already 0/1. 'per_sensor' pools across channels too, which lets xyz dominate the "
             "std and squashes validity. 'metric' uses a DECLARED affine and no fit at all, so "
             "the input is metres/scale and is identical across recordings -- both fitted modes "
             "make a checkpoint's notion of '1 m' a property of its training set. Irrelevant "
             "to --extero_encoder mlp.")
    parser.add_argument(
        "--extero_norm_scale", type=float, default=FlowMatchingConfig().extero_norm_scale,
        help="Divisor for --extero_norm_mode metric. Default 1.0, i.e. the network input IS "
             "metres.")
    parser.add_argument(
        "--extero_clip", type=float, default=None,
        help="Symmetric clip band applied before normalization, in the source's units. "
             "Default: 1.0 m for the scan, 2.0 (normalised) for depth. The scan default is "
             "too tight for a large grid -- measured on the real recording, 33%% of "
             "stairs_dn_pure rays already exceed +1 m at 176 rays, and a 3 m window destroys "
             "~11%% of the image (42%% on descents). ~2.0 leaves 0.5%% clipped. Safe to widen "
             "because the generator has its own scanner, so the teacher's clip=(-1,1) obs "
             "terms and the critic's input distribution are untouched. The dataset PRINTS the "
             "saturation and miss fractions it measured, so pick this from a run's own log.")
    parser.add_argument(
        "--depth_latency_probs", type=str, default=None,
        help="Camera-latency distribution: comma-separated probabilities, index = lag in "
             "CONTROL STEPS, so '0,0.10,0.40,0.50' is P(0 ms), P(20 ms), P(40 ms), P(60 ms) "
             "-- the camera and control both run at 50 Hz, so a step is exactly 20 ms. "
             "'shipped' uses DEPTH_LATENCY_PROBS (that same tuple, mean 48 ms, matching the "
             "measured depth-compute-plus-ROS lag). Omit for NO latency: the frame captured "
             "at the re-plan, which is what every recording before 2026-08-21 supports. "
             "Needs a recording that carries 'depth_lag'; it raises rather than silently "
             "falling back. Keep lag 0 at zero -- a frame cannot arrive in the step it was "
             "captured in, and it is the easiest case.")
    # --noise_height_scan / --noise_depth are GONE: there is no train-time exteroception noise
    # any more. The sensor model belongs to the RL env, whose DR is already in the recording.
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
    parser.add_argument("--body_pose_root_weight", type=float,
                        default=FlowMatchingConfig().body_pose_root_weight,
                        help="Weight on the body-link pose error in the reference's own "
                             "pelvis frame (pure posture, cannot be satisfied by moving "
                             "the base). 0 = off. See flow.body_pose_terms.")
    parser.add_argument("--body_pose_world_weight", type=float,
                        default=FlowMatchingConfig().body_pose_world_weight,
                        help="Weight on the body-link pose error composed with the "
                             "predicted base_pose -- numerically the WORLD-frame error, "
                             "which is what the body-mode CLF grades and what stair "
                             "precision needs. 0 = off. Pair it with "
                             "--body_pose_root_weight, which stops a base error "
                             "cancelling a joint error.")
    parser.add_argument("--body_pose_ori_scale", type=float,
                        default=FlowMatchingConfig().body_pose_ori_scale,
                        help="Metres per radian for the orientation half of both "
                             "body-pose terms.")
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
    # Rendered from the dataclass, never restated: these six sat here as literals and drifted
    # the moment the shipped architecture grew to 27M, so the CLI silently built a 256-wide
    # 4-layer model while config.py documented 512/8. Same rule as every other flag.
    _arch = FlowMatchingConfig()
    parser.add_argument("--d_model", type=int, default=_arch.d_model)
    parser.add_argument("--nhead", type=int, default=_arch.nhead)
    parser.add_argument("--num_layers", type=int, default=_arch.num_layers)
    parser.add_argument("--dim_feedforward", type=int, default=_arch.dim_feedforward)
    parser.add_argument("--dropout", type=float, default=_arch.dropout)
    parser.add_argument("--num_sampling_steps", type=int, default=_arch.num_sampling_steps)
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
    parser.add_argument("--joint_vel_consistency_weight", type=float, default=_wc.joint_vel_consistency_weight,
                        help="Weight on the joint-velocity finite-difference consistency "
                             "(predicted joint_pos d/dt vs predicted joint_vel). Ungated.")
    parser.add_argument("--fk_pose_weight", type=float, default=_wc.fk_pose_weight,
                        help="Weight on the FK position consistency (predicted ee position, "
                             "base-relative, vs FK of predicted joints). tau-gated.")
    parser.add_argument("--consistency_tau_min", type=float, default=FlowMatchingConfig().consistency_tau_min,
                        help="Only score the FK + body-pose terms on samples with flow-time "
                             "tau >= this (velocity/joint-velocity stay ungated).")
    parser.add_argument("--fk_model_path", type=str, default=FlowMatchingConfig().fk_model_path,
                        help="MuJoCo MJCF defining the kinematics for the FK losses (repo-relative).")
    parser.add_argument(
        "--skip_families", type=str, default=None,
        help="Comma-separated clip FAMILIES to drop (see the recording's family_names, e.g. "
             "'family_jump_off'). A train-time switch like --skip_distractors: the recording "
             "tags every window with its family, so A/B-ing one costs no re-record. The case "
             "this exists for is family_jump_off -- the BACKWARDS jump-off-a-box family, "
             "which the forward-facing depth cameras cannot observe at all. Errors if the "
             "recording predates the tag or a named family is not in it.",
    )
    parser.add_argument(
        "--transition_fraction", type=float, default=_wc.transition_fraction,
        help="Oversample windows CROSSING a terrain transition until they are this fraction "
             "of each training epoch (e.g. 0.5; natural rate on the v5 library is 0.178). A "
             "sampler weight, not a filter: epoch size and per-window loss are unchanged, "
             "only the draw rate. 'Crosses a transition' is an anchor_mask_t boundary inside "
             "[clip_step, clip_step+F), joined via the recorded clip_id/clip_step -- ground "
             "truth, not a height heuristic. Needs a recording carrying that index.",
    )
    parser.add_argument(
        "--clip_library_path", type=str, default=_wc.clip_library_path,
        help="Clip library to join clip_id/clip_step against for --transition_fraction. "
             "Default: the path recorded in the .h5. Its sha256 is verified either way.",
    )
    # BooleanOptionalAction, not store_true: the default is now ON, and a store_true flag can
    # only ever turn a knob on -- so the off position would be unreachable from the CLI, which
    # is the same "the config says one thing and the command line cannot say the other" bug
    # test_train_cli_defaults exists to catch.
    parser.add_argument("--skip_distractors", action=argparse.BooleanOptionalAction,
                        default=_wc.skip_distractors,
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
    parser.add_argument(
        "--ema_decay", type=float, default=0.999,
        help="Exponential moving average of the weights, updated EVERY optimizer step and "
             "saved into each checkpoint beside the raw ones (0 disables). Validation runs "
             "through BOTH, logged as val/* and val_ema/*, so what EMA bought is visible in "
             "the run rather than assumed -- and best.pt is selected on the EMA metric when "
             "this is on, since the EMA weights are what would be deployed.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=FlowMatchingConfig().lr_warmup_steps,
        help="Ramp the LR linearly to --lr over this many optimizer steps (0 = off). Composes "
             "with --cosine. Cheap insurance against the predict-the-mean plateau becoming "
             "permanent -- see FlowMatchingConfig.lr_warmup_steps.",
    )
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
        joint_vel_consistency_weight=args.joint_vel_consistency_weight,
        fk_pose_weight=args.fk_pose_weight,
        body_pose_root_weight=args.body_pose_root_weight,
        body_pose_world_weight=args.body_pose_world_weight,
        body_pose_ori_scale=args.body_pose_ori_scale,
        consistency_tau_min=args.consistency_tau_min, fk_model_path=args.fk_model_path,
        skip_distractors=args.skip_distractors,
        skip_families=([f.strip() for f in args.skip_families.split(",") if f.strip()]
                       if args.skip_families else None),
        transition_fraction=args.transition_fraction,
        clip_library_path=args.clip_library_path,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        num_epochs=args.num_epochs, grad_clip=args.grad_clip,
        lr_warmup_steps=args.lr_warmup_steps, seed=args.seed,
        obs_noise=args.obs_noise, noise_scale=args.noise_scale,
        extero_encoder=args.extero_encoder, extero_channels=args.extero_channels,
        extero_cnn_channels=[int(c) for c in args.extero_cnn_channels.split(",") if c.strip()],
        extero_cnn_grid=([int(g) for g in args.extero_cnn_grid.split(",") if g.strip()]
                         or None),
        extero_cnn_strides=([int(s) for s in args.extero_cnn_strides.split(",") if s.strip()]
                            or None),
        extero_norm_mode=args.extero_norm_mode,
        extero_norm_scale=args.extero_norm_scale,
        extero_xyz_scale=args.extero_xyz_scale,
        extero_nominal_ground_z=args.extero_nominal_ground_z,
        # ``None`` (not ``[]``) when unset: the dataset derives the offsets only when the
        # field is None, and an empty list would read as "zero cameras stated" and trip the
        # per-camera count check instead.
        extero_xyz_offset=([[float(v) for v in cam.split(",")]
                            for cam in args.extero_xyz_offset.split(";") if cam.strip()]
                           or None),
        pinned_axis_loss_weight=args.pinned_axis_loss_weight,
        state_scale_base_pos=args.state_scale_base_pos,
        state_scale_base_lin_vel=args.state_scale_base_lin_vel,
        state_scale_base_ang_vel=args.state_scale_base_ang_vel,
        state_scale_joint_pos=args.state_scale_joint_pos,
        state_scale_joint_vel=args.state_scale_joint_vel,
        state_scale_ee_pos=args.state_scale_ee_pos,
        cmd_scale=([float(v) for v in args.cmd_scale.split(",") if v.strip()]
                   if args.cmd_scale and args.cmd_scale.lower() != "none" else None),
        attention_mode=args.attention_mode,
    )
    # Noise amplitudes and the clip band keep their per-source defaults unless named, so an
    # unspecified flag reproduces the previous behaviour exactly rather than overwriting it
    # with an argparse default.
    if args.extero_clip is not None:
        cfg.extero_clip = float(args.extero_clip)
    if args.depth_latency_probs is not None:
        cfg.depth_latency_probs = (
            DEPTH_LATENCY_PROBS if args.depth_latency_probs == "shipped"
            else tuple(float(x) for x in args.depth_latency_probs.split(",") if x.strip()))
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

        Through the rotation group rather than differencing the raw channels, so the metric
        is a real angle -- and so the prediction's Gram-Schmidt residual is projected out
        rather than counted as orientation error.
        """
        rel = rot6d_to_matrix(pred[..., slc]).transpose(-1, -2) @ rot6d_to_matrix(tgt[..., slc])
        cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        return torch.arccos(cos).mean().item()

    b = window_block_slices(num_joints, num_extra_frames, blocks)
    out: dict[str, float] = {}
    if "base_pose" in b:
        start = b["base_pose"].start
        out["rmse_base_pos"] = _rmse(slice(start, start + 3))
        out["ori_geodesic_rad"] = _geodesic(slice(start + 3, start + 9))
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


_CONS_KEYS = ("vel_consistency", "joint_vel_consistency", "fk_pose",
              "body_pose_root", "body_pose_root_pos", "body_pose_root_ori",
              "body_pose_world", "body_pose_world_pos", "body_pose_world_ori")


def scheduled_lr(base_lr: float, epoch: int, num_epochs: int, step: int,
                 warmup_steps: int, cosine: bool) -> float:
    """The learning rate for one optimizer step: cosine over epochs x linear warmup.

    Both factors are CLOSED FORM in ``(epoch, step)`` rather than a stateful
    ``lr_scheduler``, because the two have to compose at different granularities -- warmup
    is per step, the cosine per epoch -- and ``CosineAnnealingLR.get_lr`` is recursive in
    ``param_group["lr"]``. Mutating the group's LR mid-epoch for the warmup would therefore
    corrupt the cosine's own state instead of scaling it. The cosine factor reproduces
    ``CosineAnnealingLR(T_max=num_epochs, eta_min=0)`` exactly at epoch granularity, so
    ``--cosine`` alone behaves as it always did.

    Args:
        base_lr: The configured peak LR (``cfg.lr``).
        epoch: Current epoch index.
        num_epochs: Total epochs (the cosine's ``T_max``).
        step: Global optimizer-step index (0-based).
        warmup_steps: Steps over which to ramp linearly to ``base_lr`` (0 = no warmup).
        cosine: Whether cosine annealing is on.

    Returns:
        The LR to use for this step.
    """
    lr = base_lr
    if cosine and num_epochs > 1:
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / num_epochs))
    if warmup_steps > 0 and step < warmup_steps:
        lr *= (step + 1) / warmup_steps
    return lr


@torch.no_grad()
def adam_update_snr(opt: torch.optim.Optimizer) -> float:
    """Median over parameter tensors of ``|exp_avg| / sqrt(exp_avg_sq)`` -- Adam's own SNR.

    The direct "is this run still learning?" number, and the one the gradient norm cannot
    give: a frozen run's gradient is not zero, it is zero-MEAN, so the norm stays small but
    finite while the accumulated direction cancels. This ratio is what Adam actually
    multiplies the LR by, so it separates the two cases outright.

    MEDIAN, not mean, measured on the 2026-08-12 pinned-axis A/B: over the stalled arm's 131
    parameter tensors the median read 3.8e-5 against the healthy arm's 7.5e-2 (2000x), while
    the means were 7.0e-3 vs 7.7e-2 (11x) -- a handful of tensors (biases, LayerNorm gains)
    keep a healthy ratio even in a frozen network and drag the mean up by 200x. The median
    is what makes a threshold like 1e-3 mean anything.

    Cheap enough to take every epoch (one pass over ~130 state tensors, no backward).

    Args:
        opt: An Adam/AdamW optimizer that has taken at least one step.

    Returns:
        The median ratio, or ``0.0`` before the first step (no state yet).
    """
    ratios = []
    for state in opt.state.values():
        m, v = state.get("exp_avg"), state.get("exp_avg_sq")
        if m is None or v is None:
            continue
        ratios.append(float(m.abs().mean()) / max(float(v.mean()) ** 0.5, 1e-30))
    return float(np.median(ratios)) if ratios else 0.0


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


def consistency_kwargs(cfg, num_joints, num_extra_frames, fk, body_fk=None) -> dict:
    """Shared flow_matching_loss kwargs for the consistency + FK regularizers."""
    return dict(
        dt=cfg.dt, num_joints=num_joints, num_extra_frames=num_extra_frames,
        vel_weight=cfg.vel_consistency_weight,        joint_vel_weight=cfg.joint_vel_consistency_weight,
        fk=fk, fk_pose_weight=cfg.fk_pose_weight,
        body_fk=body_fk, body_pose_root_weight=cfg.body_pose_root_weight,
        body_pose_world_weight=cfg.body_pose_world_weight,
        body_pose_ori_scale=cfg.body_pose_ori_scale,
        tau_min=cfg.consistency_tau_min, blocks=cfg.reference_blocks,
    )


@torch.no_grad()
def evaluate(model, loader, cfg, state_normalizer, num_joints, device, num_extra_frames=0,
             fk=None, body_fk=None, rollout_metrics: bool = True) -> dict[str, float]:
    """Compute conditioned/null-history flow loss and per-channel sampling metrics.

    Metrics are over the full F-step prediction; ``flow_mse`` is the history-conditioned
    loss and ``flow_mse_null`` replaces the history with the learned null embedding.

    Args:
        rollout_metrics: Also compute the per-channel RMSEs, which need a full
            ``sample_trajectory`` per batch and dominate the cost. ``False`` returns the flow
            metrics alone, which is cheap enough to run every epoch.
    """
    model.eval()
    flow_sum, flow_null_sum, n_batches = 0.0, 0.0, 0
    cons_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    split_sums: dict[str, float] = {}
    ckw = consistency_kwargs(cfg, num_joints, num_extra_frames, fk, body_fk)
    for batch in loader:
        state = batch["state"].to(device)
        history = batch["history"].to(device)
        extero = batch["extero"].to(device)
        cmd_tokens = batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None
        b = state.shape[0]
        cond_out = flow_matching_loss(model, state, history, extero, cmd_tokens,
                                      state_norm=state_normalizer,
                                      report_consistency=True, **ckw)
        flow_sum += cond_out["flow_mse"].item()
        # Split the SAME flow error by whether the terrain pins that coordinate. A single
        # flow_mse averages the two halves and hides the effect the whole experiment is
        # chasing -- the generator is accurate on coordinates that follow the robot and
        # wrong on the ones the terrain fixes. Recomputed from a fresh flow batch rather
        # than threaded out of the loss, so it is independent of whether the loss is
        # weighted and both A/B arms report the same quantity.
        if "anchor_mask" in batch:
            x_tau, tau, target_v = sample_flow_batch(state)
            with torch.no_grad():
                v = model(x_tau, tau, history, extero, cmd_tokens)
            m, u = pinned_axis_split(
                v - target_v, batch["anchor_mask"].to(device),
                batch["mask_frame_yaw_rel"].to(device), state_normalizer,
                num_joints, num_extra_frames, cfg.reference_blocks)
            split_sums["flow_mse_masked"] = split_sums.get("flow_mse_masked", 0.0) + m
            split_sums["flow_mse_unmasked"] = split_sums.get("flow_mse_unmasked", 0.0) + u
        for k in _CONS_KEYS:
            if k in cond_out:
                cons_sums[k] = cons_sums.get(k, 0.0) + cond_out[k].item()
        null_mask = torch.ones(b, dtype=torch.bool, device=device)
        flow_null_sum += flow_matching_loss(
            model, state, history, extero, cmd_tokens, history_null_mask=null_mask
        )["flow_mse"].item()
        n_batches += 1
        if not rollout_metrics:
            # The channel RMSEs below need a full 8-step sample_trajectory per batch, which
            # is what makes validation expensive. The flow metrics above are 2-3 forward
            # passes, so they can run every epoch while these stay on the slower cadence.
            continue
        pred = sample_trajectory(
            model, history, extero, cmd_tokens, cfg.future_len, cfg.state_dim, cfg.num_sampling_steps,
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
    for k, v in split_sums.items():
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

    # --data accepts SEVERAL comma-separated recordings, concatenated window-wise. The case
    # this exists for is registration augmentation: the augmented windows land in their own
    # .h5 next to the original, and merging them on disk would mean a second full copy of a
    # ~10 GB file just to train once.
    data_paths = [_resolve_data_path(p) for p in str(args.data).split(",") if p.strip()]
    parts = []
    # Which arrays this run will actually READ. A recording carries both exteroception
    # signals so one file can train either generator, and at a dense scan the unused one is
    # the largest thing in the file -- so not loading it is the single biggest saving
    # available, and it costs nothing to know in advance.
    unused_extero = [s for s in EXTERO_SOURCES if s != cfg.extero_source]
    load_keys = [k for k in WINDOW_ARRAY_KEYS if k not in unused_extero]
    for dp in data_paths:
        print(f"[INFO] Loading dataset: {dp}")
        if is_windowed_recording(dp):
            parts.append(load_recorded_windows(dp, keys=load_keys))
        else:
            parts.append(load_recorded_data(dp))
    data, meta = parts[0]
    if len(parts) > 1:
        # episode_id must stay UNIQUE across files or the train/val split leaks: the split is
        # by episode, and two files independently numbering from 0 would put the same id in
        # both halves. Offset each subsequent file past the running maximum.
        data = {k: v for k, v in data.items()}
        ep_off = int(np.asarray(data["episode_id"]).max()) + 1
        for extra, extra_meta in parts[1:]:
            if set(extra) != set(data):
                raise SystemExit(
                    f"recordings carry different arrays and cannot be concatenated:\n"
                    f"  {sorted(set(data) - set(extra))} missing from a later file\n"
                    f"  {sorted(set(extra) - set(data))} missing from the first")
            # Only the FIRST file's meta survives (see below), so a geometry difference would
            # be invisible: two scans of the same WIDTH but different extent or resolution
            # concatenate silently and train as if they were one sensor. Array widths already
            # fail loudly in np.concatenate; this catches the case they cannot see.
            for field in ("height_scan_spec", "depth_camera_spec"):
                mine = getattr(meta, field, None)
                theirs = getattr(extra_meta, field, None)
                if _sensor_geometry_key(mine) != _sensor_geometry_key(theirs):
                    raise SystemExit(
                        f"recordings disagree on {field}:\n"
                        f"  first : {_sensor_geometry_key(mine)}\n"
                        f"  later : {_sensor_geometry_key(theirs)}\n"
                        "Same-width sensors with different geometry would concatenate "
                        "silently and train as one. Record them with matching geometry, or "
                        "train them separately.")
            for k in data:
                if k == "episode_id":
                    data[k] = np.concatenate([data[k], np.asarray(extra[k]) + ep_off])
                else:
                    data[k] = np.concatenate([data[k], np.asarray(extra[k])])
            ep_off = int(np.asarray(data["episode_id"]).max()) + 1
        n_each = [int(np.asarray(p[0]["episode_id"]).shape[0]) for p in parts]
        print(f"[INFO] concatenated {len(parts)} recordings: {n_each} -> "
              f"{int(np.asarray(data['episode_id']).shape[0])} windows "
              "(episode ids offset per file so the val split cannot leak between them)")
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
          f"extero={cfg.extero_source} ({cfg.extero_sensor_count}x{cfg.extero_sensor_dim}), "
          f"F={cfg.future_len}, N={cfg.num_history_states}, "
          f"null_cond_prob={cfg.null_cond_prob}, train_windows={len(train_ds)}, "
          f"val_windows={0 if val_ds is None else len(val_ds)}")
    # The exteroception embedding and, from it, the SEQUENCE LENGTH -- which nothing printed
    # before and which is what the transformer's cost actually scales with.
    shapes = (", ".join(f"{h}x{w}" for h, w in (train_ds.extero_image_hw or []))
              or "(no image shape)")
    strides = (tuple(cfg.extero_cnn_strides) if cfg.extero_cnn_strides
               else (2,) * len(cfg.extero_cnn_channels))
    detail = (f"grid={tuple(cfg.extero_cnn_grid) if cfg.extero_cnn_grid else 'native'}, "
              f"channels={list(cfg.extero_cnn_channels)}, strides={list(strides)}, "
              f"images {shapes}, "
              if cfg.extero_encoder == "cnn" else "")
    print(f"[INFO] extero_encoder={cfg.extero_encoder} ({detail}"
          f"{cfg.tokens_per_sensor} token(s)/sensor -> {cfg.extero_seq_tokens} extero "
          f"tokens), extero_norm_mode={cfg.extero_norm_mode}; SEQUENCE LENGTH "
          f"{cfg.sequence_len} = {cfg.num_history_states} history + {cfg.extero_seq_tokens} "
          f"extero + {cfg.num_cmd_tokens} cmd + {cfg.future_len} future.")
    # The positional information, stated explicitly because it is the point of the metric
    # path and a checkpoint carries no obvious evidence of it: a convolution given coordinate
    # channels is a structurally different model from one that is not.
    if cfg.extero_coord_extent_m is not None:
        print(f"[INFO] metric grid {cfg.extero_coord_extent_m[0]}x"
              f"{cfg.extero_coord_extent_m[1]} m, ordering={cfg.extero_grid_ordering}: "
              f"pixel (x, y) offsets in METRES as CoordConv channels + a "
              f"{cfg.extero_fourier_bands}-band Fourier embedding of each token's cell centre"
              + (f", validity channel on (miss -> {cfg.extero_miss_sentinel:+.3f})"
                 if cfg.extero_miss_sentinel is not None else "")
              + ". A shared kernel is translation-equivariant, so without this it cannot "
                "represent WHERE an edge is.")
    # The band and the scaling, stated because BOTH are resolved rather than typed: the clip
    # comes from the per-source default when a run does not name one, so a log line is the only
    # place a checkpoint's band is attributable after the fact.
    print(f"[INFO] extero={cfg.extero_source} clip=+-{cfg.resolve_extero_clip():g} "
          f"({'declared' if cfg.extero_clip is not None else 'per-source default'}), "
          f"norm_mode={cfg.extero_norm_mode}, channels={cfg.extero_channels} "
          f"{list(cfg.extero_channel_names)}. NO train-time exteroception noise: the sensor "
          f"model is the RL env's DR, already in the recording (extero_corrupted="
          f"{bool(getattr(meta, 'extero_corrupted', False))}, extero_noise_free="
          f"{bool(getattr(meta, 'extero_noise_free', False))}).")
    # Stated in the log because a checkpoint carries no structural evidence of it -- the two
    # attention modes share one parameter set, so this line is how a run's architecture stays
    # attributable after the fact.
    cfg.check_attention_mode()
    prefix_note = ("conditioning attends only to itself and is encoded once per re-plan"
                   if cfg.attention_mode == "prefix"
                   else "every token attends to every other (the pre-flag architecture)")
    print(f"[INFO] attention_mode={cfg.attention_mode}: {prefix_note}; "
          f"prefix {cfg.sequence_len - cfg.future_len} + future {cfg.future_len} tokens.")
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
    # Provenance for ALL inputs, not just the first: a checkpoint trained on an original plus
    # its augmented counterpart must name both, or the run cannot be reproduced from what it
    # records. Comma-joined to match how --data took them.
    data_path_abs = ",".join(os.path.abspath(p) for p in data_paths)
    data_sha256 = ",".join(sha256_file(p) for p in data_paths)
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

    # cfg.transition_fraction oversamples terrain-transition windows via sampler weights
    # (see dataset.transition_sampler_weights). `sampler` and `shuffle` are mutually
    # exclusive in torch, and a WeightedRandomSampler already shuffles. Epoch size is held
    # at len(train_ds) so "epoch" means the same thing weighted or not, and replacement is
    # required for a weight >1 to draw a window more than once.
    _w = getattr(train_ds, "sampler_weights", None)
    train_sampler = (
        WeightedRandomSampler(torch.as_tensor(_w, dtype=torch.double),
                              num_samples=len(train_ds), replacement=True)
        if _w is not None else None
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
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

    # ONE construction site, shared with ``online.load_fm_context`` -- see
    # ``FlowMatchingTransformer.from_config`` for why spelling the arguments out here twice
    # was a bug rather than a duplication.
    model = FlowMatchingTransformer.from_config(cfg).to(device)
    extero_params = sum(p.numel() for n, p in model.named_parameters() if "extero" in n)
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"(exteroception branch: {extero_params:,}; "
          f"command conditioning: {'on' if cfg.cmd_dim > 0 else 'off'})")

    # Normalizers are fit on CPU; move their stats to the model device so
    # (de)normalization during evaluation/sampling stays on one device.
    #
    # NOT the exteroception one. It is applied inside ``Dataset.__getitem__``, on host-side
    # rows in a DataLoader worker, and moving it in place made every fetch raise "at least two
    # devices" -- while nothing on the device ever reads it, because exteroception is only an
    # INPUT (the state and command stats are the ones the loss and the sampler denormalize
    # with). The checkpoint is unaffected either way: ``state_dict`` stores stats on CPU.
    train_ds.state_normalizer.to(device)
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

    # The BODY-pose chain: a separate FKChain over the tracked body frames, which are a
    # superset of the window's ee frames (13 vs 4 on the G1). Kept separate rather than
    # slicing one chain, because fk_consistency_terms indexes its output by the WINDOW's
    # ee-frame count and would silently mis-slice a wider one.
    body_fk = None
    if cfg.body_pose_root_weight > 0.0 or cfg.body_pose_world_weight > 0.0:
        frames = list(cfg.body_fk_frames) if cfg.body_fk_frames else list(G1_BODY_TRACK_FRAMES)
        fk_path = cfg.fk_model_path if os.path.isabs(cfg.fk_model_path) else os.path.join(_REPO_ROOT, cfg.fk_model_path)
        body_fk = FKChain(fk_path, list(meta.state_joint_names), frames).to(device)
        cfg.body_fk_frames = frames
        print(f"[INFO] body-pose loss on {len(frames)} frames "
              f"(root {cfg.body_pose_root_weight}, world {cfg.body_pose_world_weight}, "
              f"ori_scale {cfg.body_pose_ori_scale} m/rad): {frames}")

    # EMA of the weights. Built AFTER .to(device) so the shadow lives on the same device,
    # and after any warning prints so the enable line lands next to them.
    ema = None
    if args.ema_decay > 0.0:
        ema = WeightEMA(model, decay=args.ema_decay)
        print(f"[INFO] weight EMA on, decay={args.ema_decay} (~{1/(1-args.ema_decay):.0f} "
              f"step window, warmed in). best.pt will be selected on val_ema/flow_mse.")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_warmup_steps > 0:
        print(f"[INFO] LR warmup: linear over the first {cfg.lr_warmup_steps} optimizer steps "
              f"to lr={cfg.lr:g}" + (" (then cosine over epochs)." if args.cosine else "."))

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
        epoch_gnorm, n_clipped = 0.0, 0
        lr_now = scheduled_lr(cfg.lr, epoch, cfg.num_epochs, global_step,
                              cfg.lr_warmup_steps, args.cosine)   # also the empty-loader value
        for batch in train_loader:
            # Closed-form schedule, set before the step it applies to. ``global_step`` is
            # 0-based here (it is incremented after the step, below).
            lr_now = scheduled_lr(cfg.lr, epoch, cfg.num_epochs, global_step,
                                  cfg.lr_warmup_steps, args.cosine)
            for group in opt.param_groups:
                group["lr"] = lr_now
            state = batch["state"].to(device)
            history = batch["history"].to(device)
            extero = batch["extero"].to(device)
            cmd_tokens = batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None
            # Per-sample history null-out (classifier-free guidance / phase-agnostic).
            null_mask = torch.rand(state.shape[0], device=device) < cfg.null_cond_prob
            opt.zero_grad()
            out = flow_matching_loss(
                model, state, history, extero, cmd_tokens,
                history_null_mask=null_mask,
                state_norm=train_ds.state_normalizer,
                # Present only when the pinned-axis weighting is on; the loss treats an
                # absent mask as "every channel equal", which is the historical behaviour.
                anchor_mask=(batch["anchor_mask"].to(device)
                             if "anchor_mask" in batch else None),
                mask_frame_yaw_rel=(batch["mask_frame_yaw_rel"].to(device)
                                    if "mask_frame_yaw_rel" in batch else None),
                pinned_axis_weight=cfg.pinned_axis_loss_weight,
                **consistency_kwargs(cfg, num_joints, train_ds.num_extra_frames, fk, body_fk),
            )
            out["loss"].backward()
            # The PRE-clip total norm, which ``clip_grad_norm_`` returns and this used to
            # throw away. Logged because it is the cheapest early warning there is: a run
            # that has stalled at the plateau reads two orders of magnitude below a healthy
            # one within the first epoch, while every loss curve still looks merely flat.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            opt.step()
            if ema is not None:
                ema.update(model)
            epoch_loss += out["loss"].item()
            epoch_gnorm += grad_norm
            n_clipped += int(grad_norm > cfg.grad_clip)
            n += 1
            global_step += 1
            writer.add_scalar("train/loss_step", out["loss"].item(), global_step)
            writer.add_scalar("train/grad_norm_step", grad_norm, global_step)
            writer.add_scalar("train/lr_step", lr_now, global_step)
            for k in ("flow_mse", *_CONS_KEYS):
                if k in out:
                    writer.add_scalar(f"train/{k}_step", out[k].item(), global_step)

        epoch_time = time.perf_counter() - epoch_start
        train_loss = epoch_loss / max(n, 1)
        grad_norm_mean = epoch_gnorm / max(n, 1)
        clipped_frac = n_clipped / max(n, 1)
        # Adam's own signal-to-noise. Together with the norm this distinguishes "small
        # gradients" from "gradients that cancel", which the norm alone cannot.
        snr = adam_update_snr(opt)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/grad_norm", grad_norm_mean, epoch)
        writer.add_scalar("train/grad_clipped_frac", clipped_frac, epoch)
        writer.add_scalar("train/adam_snr", snr, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)
        writer.add_scalar("train/epoch_time_s", epoch_time, epoch)
        log_line = (f"[epoch {epoch}] train_loss={train_loss:.4f}  train_time={epoch_time:.1f}s"
                    f"  grad_norm={grad_norm_mean:.3f} (clipped {clipped_frac:.0%})"
                    f"  adam_snr={snr:.2e}  lr={lr_now:.2e}")

        # FLOW metrics every epoch, the sampled-rollout RMSEs on the slower cadence: the
        # flow numbers cost 2-3 forward passes while each RMSE needs a full 8-step
        # sample_trajectory per batch, so this buys a per-epoch curve for the cheap half
        # without paying for the expensive one.
        full_val = (epoch % args.val_every == 0 or epoch == cfg.num_epochs - 1)
        if val_loader is not None:
            val = evaluate(
                model, val_loader, cfg, train_ds.state_normalizer, num_joints, device,
                num_extra_frames=train_ds.num_extra_frames, fk=fk, body_fk=body_fk,
                rollout_metrics=full_val,
            )
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            log_line += "  " + "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            # The same validation through the AVERAGED weights. Logged separately rather
            # than replacing val/*, so a run answers "did EMA help?" on its own evidence.
            val_ema = None
            if ema is not None:
                with ema.swapped_in(model):
                    val_ema = evaluate(
                        model, val_loader, cfg, train_ds.state_normalizer, num_joints, device,
                        num_extra_frames=train_ds.num_extra_frames, fk=fk, body_fk=body_fk,
                        rollout_metrics=full_val,
                    )
                for k, v in val_ema.items():
                    writer.add_scalar(f"val_ema/{k}", v, epoch)
                log_line += "  " + "  ".join(f"valema_{k}={v:.4f}" for k, v in val_ema.items())
            # Select on whichever weights would actually be deployed.
            selector = val_ema["flow_mse"] if val_ema is not None else val["flow_mse"]
            if selector < best_val:
                best_val = selector
                _save_ckpt(os.path.join(out_dir, "best.pt"), model, opt, cfg, train_ds, meta, epoch,
                           data_path=data_path_abs, data_sha256=data_sha256, ema=ema)
            if wandb_run is not None:
                # step=epoch on EVERY call. Without it wandb increments its own counter
                # once per log() CALL, and there are up to three per epoch (val, eval,
                # train) -- so a 100-epoch run with val_every=5 made 121 calls and the
                # x-axis ran to 120. With an explicit step the calls merge into one row
                # per epoch and the axis is the epoch.
                wandb_run.log({**{f"val/{k}": v for k, v in val.items()},
                               **{f"val_ema/{k}": v for k, v in (val_ema or {}).items()},
                               "epoch": epoch}, step=epoch)

        if eval_loader is not None and full_val:
            ev = evaluate(
                model, eval_loader, cfg, train_ds.state_normalizer, num_joints, device,
                num_extra_frames=train_ds.num_extra_frames, fk=fk, body_fk=body_fk,
            )
            for k, v in ev.items():
                writer.add_scalar(f"eval/{k}", v, epoch)
            log_line += "  " + "  ".join(f"eval_{k}={v:.4f}" for k, v in ev.items())
            if wandb_run is not None:
                wandb_run.log({**{f"eval/{k}": v for k, v in ev.items()}, "epoch": epoch},
                               step=epoch)

        print(log_line)
        if wandb_run is not None:
            wandb_run.log({"train/loss_epoch": train_loss, "train/epoch_time_s": epoch_time,
                           "train/grad_norm": grad_norm_mean,
                           "train/grad_clipped_frac": clipped_frac,
                           "train/adam_snr": snr, "train/lr": lr_now,
                           "epoch": epoch}, step=epoch)
        if epoch % args.ckpt_every == 0 or epoch == cfg.num_epochs - 1:
            _save_ckpt(os.path.join(out_dir, f"ckpt_{epoch}.pt"), model, opt, cfg, train_ds, meta, epoch, ema=ema,
                       data_path=data_path_abs, data_sha256=data_sha256)

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[INFO] Done. Checkpoints in {out_dir}")


def _sensor_geometry_key(spec: list[dict] | None) -> tuple | None:
    """The geometry-identifying fields of a sensor contract, for cross-file comparison.

    Deliberately a SUBSET: two recordings from different runs legitimately differ in
    per-episode noise state or update period, but must agree on what the pixels MEAN. Missing
    keys compare as ``None`` so an old contract and a new one differ only where they really do.

    Args:
        spec: A ``height_scan_spec`` / ``depth_camera_spec`` list, or ``None``.

    Returns:
        A hashable key, or ``None`` when there is no contract.
    """
    if not spec:
        return None
    keys = ("sensor", "resolution", "size", "ordering", "height", "width", "num_rays",
            "offset_pos", "ray_alignment", "sensor_offset", "fx", "fy", "cx", "cy",
            "offset", "scale")
    out = []
    for entry in spec:
        out.append(tuple(
            (k, tuple(entry[k]) if isinstance(entry.get(k), list) else entry.get(k))
            for k in keys if k in entry
        ))
    return tuple(out)


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
        extero_normalizer=train_ds.extero_normalizer,
    )
    got = (ds.state_dim, ds.cmd_dim, ds.num_extra_frames)
    want = (train_ds.state_dim, train_ds.cmd_dim, train_ds.num_extra_frames)
    if got != want:
        raise ValueError(
            f"--eval_data layout (state_dim, cmd_dim, extra_frames)={got} "
            f"differs from train={want}; record eval data with the same task/flags."
        )
    return ds


def _save_ckpt(path, model, opt, cfg, train_ds, meta, epoch, data_path="", data_sha256="",
               ema=None) -> None:
    """Save model weights, optimizer, config, normalizers, and dataset provenance.

    When an EMA is running its shadow weights go in as ``model_ema`` ALONGSIDE the raw
    ``model``, so one file serves either and nothing has to be decided at training time.
    ``online.load_fm_context`` prefers ``model_ema`` when present.

    ``data_path``/``data_sha256`` record exactly which recorded dataset trained this
    checkpoint, so a ``.pt`` is self-describing without the separate params YAML.

    The predicted extra-frame order is injected into the saved ``meta`` so
    ``online.load_fm_context`` (and the deploy-side ``transfer/deploy/fm_session.py``)
    reconstruct the exact token layout.
    """
    meta_dict = dict(asdict(meta) if hasattr(meta, "__dataclass_fields__") else vars(meta))
    meta_dict["target_frame_names"] = list(train_ds.target_frame_names)
    # The contract of the sensor this model ACTUALLY conditions on (None for the extero
    # scan). Distinct from the recording's ``depth_camera_spec``, which is present
    # whenever the cameras were recorded even if this model trained on the scan instead.
    meta_dict["extero_spec"] = train_ds.extero_spec
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg.to_dict(),
        "state_normalizer": train_ds.state_normalizer.state_dict(),
        "extero_normalizer": train_ds.extero_normalizer.state_dict(),
        "num_joints": train_ds.num_joints,
        "meta": meta_dict,
        "data_path": data_path,
        "data_sha256": data_sha256,
        "epoch": epoch,
    }
    if train_ds.cmd_normalizer is not None:
        ckpt["cmd_normalizer"] = train_ds.cmd_normalizer.state_dict()
    if ema is not None:
        ckpt["model_ema"] = ema.shadow
        ckpt["ema_state"] = {k: v for k, v in ema.state_dict().items() if k != "shadow"}
    torch.save(ckpt, path)


if __name__ == "__main__":
    main()
