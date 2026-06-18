# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play N trained policies the same way and emit cross-policy comparison plots.

This script makes it easy to compare several RSL-RL checkpoints head-to-head at
one or a few commanded forward velocities. For each policy it rolls out the
SHARED currently-registered env config (plus eval overrides), pins the forward
velocity command, and aggregates, per velocity: the CLF Lyapunov value ``V``
(overall + per body subgroup), the forward velocity tracking error, the
per-cause termination fractions, and the per-output / per-frame position
tracking error (pelvis & ankles highlighted). The cross-policy comparison
plots + stats are written into EACH policy's ``<log_dir>/policy_comparison/``.

Two things are taken PER POLICY from its saved ``params/`` so policies trained
differently still load and are scored fairly:
  * the network architecture, from ``agent.yaml`` (so checkpoints with different
    layer sizes load against the shared env), and
  * the reference trajectory library, from ``env.yaml`` (``commands.traj_ref.path``),
    so each policy is scored on the references it actually trained on. Override
    with ``--traj_source shared`` / ``--shared_traj_path`` to score everyone on
    one library. Per-output tracking stays comparable because the tracked-output
    set comes from the shared env cfg, not the library.

Architecture (IsaacLab launches one ``SimulationApp`` per process, so policies
cannot share a process):

    orchestrator (no sim) -> N worker subprocesses (1 sim app each, 1 policy)
       each worker          -> writes a pre-aggregated dump (npz + meta.json)
    orchestrator            -> compare_plots.aggregate_and_plot reads the dumps
                               and writes comparison figures into each log dir

The orchestrator and worker are the SAME file; the hidden ``--_worker`` flag
selects worker mode. Run the orchestrator from the repo root (so ``logs/`` and
trajectory paths resolve), exactly like ``play.py`` / ``eval_per_trajectory.py``.

Velocity pinning caveat: on a multiskill terrain task the terrain still gates
which trajectory *variant* is eligible for a given ``vx``. A flat / single-skill
``--task`` gives the cleanest per-velocity reading; the default play task uses a
flat terrain so the gating is mild.

Examples
--------
Compare two runs at three forward speeds (run from the repo root)::

    python scripts/rsl_rl/eval_compare_policies.py \
        --runs 2026-06-13_15-10-58_a 2026-06-14_09-00-00_b \
        --experiment_name g1_multiskill \
        --velocities 0.0 1.0 2.5 --num_envs 128 --steps 1000 --headless
"""

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tempfile

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
import compare_plots  # isort: skip
from episode_stats import EpisodeStatsAccumulator  # isort: skip
from per_traj_stats import DEFAULT_TRACK_FRAMES, resolve_frame_pos_indices  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compare N RSL-RL policies at fixed forward velocities."
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=128,
                    help="Number of parallel envs per policy (shared across all).")
parser.add_argument("--task", type=str, default="G1-multiskill-terrain-clf-play",
                    help="Shared registered task played by every policy.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=0,
                    help="Shared deterministic seed so every policy sees identical conditions.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--velocities", type=float, nargs="+", default=[0.0, 1.0, 2.0],
                    help="Forward velocities (m/s) to evaluate at (vy=0, yaw=0).")
parser.add_argument("--steps", type=int, default=1000,
                    help="Rollout steps collected per velocity (after warmup).")
parser.add_argument("--warmup", type=int, default=10,
                    help="Steps skipped at the start of each velocity segment (and after each "
                         "episode reset) before counting samples — lets the robot settle at the "
                         "newly pinned velocity. Increase when sweeping widely separated velocities.")
parser.add_argument("--cmd_name", type=str, default="traj_ref",
                    help="Trajectory command term name (carries CLF V + tracking outputs).")
parser.add_argument("--vel_cmd_name", type=str, default="base_velocity",
                    help="Velocity command term name to pin.")
parser.add_argument("--traj_source", choices=["per_policy", "shared"], default="per_policy",
                    help="'per_policy' (default): each policy tracks its OWN saved reference "
                         "trajectory library (commands.traj_ref.path from its env.yaml), so "
                         "policies trained on different libraries are scored on what they learned. "
                         "'shared': all policies use the current-registry library (or --shared_traj_path).")
parser.add_argument("--shared_traj_path", type=str, default=None,
                    help="Force every policy onto this trajectory library path "
                         "(overrides --traj_source; e.g. to score all policies on one library).")
parser.add_argument("--plots", type=str, default="all",
                    help=f"Comparison plots: 'all' or comma-list of "
                         f"{', '.join(compare_plots.PLOT_REGISTRY.keys())}.")
# -- orchestrator-only
parser.add_argument("--runs", type=str, nargs="+", default=None,
                    help="Policy specs: a run-folder name under "
                         "logs/rsl_rl/<experiment_name>, a run dir, or a .pt path.")
parser.add_argument("--checkpoint_regex", type=str, default="model_.*.pt",
                    help="Regex for the checkpoint file within each run (latest matched is used).")
parser.add_argument("--out_root", type=str, default=None,
                    help="Scratch dir for worker dumps. Defaults to a tempdir.")
# -- worker-only (internal)
parser.add_argument("--_worker", action="store_true", default=False,
                    help=argparse.SUPPRESS)
parser.add_argument("--out_dump", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--policy_label", type=str, default=None, help=argparse.SUPPRESS)

cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Worker mode launches a sim app and consumes hydra overrides; the orchestrator
# never launches a sim app, but keep argv clean for both.
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_torch(data) -> torch.Tensor:
    """Convert warp arrays / array-likes to torch tensors (passthrough for torch)."""
    if isinstance(data, torch.Tensor):
        return data
    try:
        import warp as wp
        return wp.to_torch(data)
    except (ImportError, TypeError):
        return torch.as_tensor(data)


# ---------------------------------------------------------------------------
# Worker-side online accumulator
# ---------------------------------------------------------------------------

class _SegmentMetrics:
    """Online per-env accumulator for one (policy, velocity) rollout segment.

    Sums are gated by a per-step ``active`` mask (warm, non-terminal envs) and
    reduced as: per-env time-mean, then mean/std over envs — matching
    ``play_plots._time_mean_per_env`` so the numbers line up with ``stats.txt``.
    """

    def __init__(self, n_envs: int, d_pos: int, d_vel: int, n_frames: int,
                 n_sub: int, device: torch.device) -> None:
        """Allocate zero accumulators sized to the metric dimensions."""
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.cnt = z(n_envs)
        self.sum_v = z(n_envs)
        self.sum_vx_act = z(n_envs)
        self.sum_abs_err = z(n_envs)
        self.sum_pos_sq = z(n_envs, d_pos) if d_pos else None
        self.sum_vel_sq = z(n_envs, d_vel) if d_vel else None
        self.sum_frame = z(n_envs, n_frames) if n_frames else None
        self.sum_sub = z(n_envs, n_sub) if n_sub else None
        self.has_v = False

    def add(self, active, v, vx_act, abs_err, pos_sq, vel_sq, frame, sub) -> None:
        """Fold one step's per-env metrics in, gated by ``active`` ([N] bool)."""
        zeros = torch.zeros_like(self.cnt)
        am = active
        amf = active.unsqueeze(1)
        self.cnt += active.float()
        if v is not None:
            self.has_v = True
            self.sum_v += torch.where(am, torch.nan_to_num(v), zeros)
        self.sum_vx_act += torch.where(am, vx_act, zeros)
        self.sum_abs_err += torch.where(am, abs_err, zeros)
        if pos_sq is not None and self.sum_pos_sq is not None:
            self.sum_pos_sq += torch.where(amf, pos_sq, torch.zeros_like(pos_sq))
        if vel_sq is not None and self.sum_vel_sq is not None:
            self.sum_vel_sq += torch.where(amf, vel_sq, torch.zeros_like(vel_sq))
        if frame is not None and self.sum_frame is not None:
            self.sum_frame += torch.where(amf, frame, torch.zeros_like(frame))
        if sub is not None and self.sum_sub is not None:
            self.sum_sub += torch.where(amf, sub, torch.zeros_like(sub))

    def finalize(self) -> dict:
        """Reduce to per-velocity scalars/arrays (mean & std over envs)."""
        cnt = self.cnt.cpu().numpy()
        valid = cnt > 0

        def pe_vec(sum_t):
            s = sum_t.cpu().numpy()
            out = np.full_like(s, np.nan, dtype=np.float64)
            out[valid] = s[valid] / cnt[valid]
            return out

        def pe_mat(sum_t):
            s = sum_t.cpu().numpy()
            out = np.full(s.shape, np.nan, dtype=np.float64)
            out[valid] = s[valid] / cnt[valid, None]
            return out

        def red_vec(pe):
            return float(np.nanmean(pe)), float(np.nanstd(pe))

        def red_mat(pe):
            return np.nanmean(pe, axis=0), np.nanstd(pe, axis=0)

        res: dict = {}
        if self.has_v:
            res["clf_v_mean"], res["clf_v_std"] = red_vec(pe_vec(self.sum_v))
        else:
            res["clf_v_mean"], res["clf_v_std"] = float("nan"), float("nan")
        res["vx_actual_mean"], res["vx_actual_std"] = red_vec(pe_vec(self.sum_vx_act))
        res["vel_track_err_mean"], res["vel_track_err_std"] = red_vec(pe_vec(self.sum_abs_err))
        res["pos_err_mse"], res["pos_err_mse_std"] = (
            red_mat(pe_mat(self.sum_pos_sq)) if self.sum_pos_sq is not None else (np.zeros(0), np.zeros(0)))
        res["vel_err_mse"], res["vel_err_mse_std"] = (
            red_mat(pe_mat(self.sum_vel_sq)) if self.sum_vel_sq is not None else (np.zeros(0), np.zeros(0)))
        res["frame_track_err_mean"], res["frame_track_err_std"] = (
            red_mat(pe_mat(self.sum_frame)) if self.sum_frame is not None else (np.zeros(0), np.zeros(0)))
        res["clf_subgroup_mean"], res["clf_subgroup_std"] = (
            red_mat(pe_mat(self.sum_sub)) if self.sum_sub is not None else (np.zeros(0), np.zeros(0)))
        return res


# ---------------------------------------------------------------------------
# Worker mode
# ---------------------------------------------------------------------------

def _resolve_worker_checkpoint(agent_cfg, log_root_path: str, train_task_name: str) -> str:
    """Resolve the checkpoint path for the worker (orchestrator passes --checkpoint)."""
    if args_cli.use_pretrained_checkpoint:
        path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not path:
            raise SystemExit("[worker] No pre-trained checkpoint available for this task.")
        return path
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _read_saved_traj_ref(log_dir: str) -> dict:
    """Read the saved ``commands.traj_ref`` sub-config from a run's env.yaml.

    Uses a tolerant loader that drops Python-object tags (e.g. ``builtins.slice``
    / ``class_type``) the safe loader rejects, so the plain scalar fields we need
    (``path``, ``hf_repo``, ``library_fingerprint``, ``library_manifest``) still
    parse. Returns an empty dict when the file is absent or unreadable.
    """
    path = os.path.join(log_dir, "params", "env.yaml")
    if not os.path.isfile(path):
        return {}
    try:
        import yaml

        class _TolerantLoader(yaml.SafeLoader):
            pass

        # Map any python/* tag to None instead of erroring (we only read scalars).
        _TolerantLoader.add_multi_constructor(
            "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
        )
        with open(path) as fh:
            data = yaml.load(fh, Loader=_TolerantLoader)
        return (data.get("commands") or {}).get("traj_ref") or {}
    except Exception as exc:  # noqa: BLE001 - never block the rollout
        print(f"[worker] Could not read saved traj_ref from {path}: {exc}")
        return {}


def _apply_trajectory_library(env_cfg, log_dir: str) -> tuple[str | None, object]:
    """Point the env at the right reference trajectory library for this policy.

    Policies are often trained against different trajectory libraries. To score
    each policy on the references it actually learned, the worker overrides
    ``commands.traj_ref.path`` (and ``hf_repo``) with the policy's own saved
    value by default (``--traj_source per_policy``). ``--shared_traj_path`` forces
    every policy onto one library instead; ``--traj_source shared`` (without an
    explicit path) keeps the current registry default. The library fingerprint /
    manifest are also injected for the loader's divergence check (parity with
    play.py).

    Returns:
        ``(path_used, fingerprint)`` for the dump metadata. ``path_used`` is the
        registry default (``None`` here) when nothing was overridden.
    """
    if not (hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "traj_ref")):
        return None, None
    traj = env_cfg.commands.traj_ref
    saved = _read_saved_traj_ref(log_dir)
    fingerprint = saved.get("library_fingerprint")
    if fingerprint is not None:
        traj.library_fingerprint = fingerprint
        traj.library_manifest = saved.get("library_manifest")

    path_used: str | None = None
    if args_cli.shared_traj_path:
        path_used = args_cli.shared_traj_path
        traj.path = path_used
        print(f"[worker] Trajectory library (shared override): {path_used}")
    elif args_cli.traj_source == "per_policy":
        saved_path = saved.get("path")
        if saved_path:
            path_used = saved_path
            traj.path = saved_path
            if saved.get("hf_repo") is not None:
                traj.hf_repo = saved.get("hf_repo")
            print(f"[worker] Trajectory library (policy's own): {saved_path}")
        else:
            print("[worker] No saved traj_ref.path found; using current-registry library.")
    else:
        print("[worker] Using current-registry trajectory library (traj_source=shared).")
    return (path_used if path_used is not None else getattr(traj, "path", None)), fingerprint


def _load_runner_cfg(agent_cfg, log_dir: str, device) -> tuple[dict, str, object]:
    """Build the runner cfg dict, preferring the policy's own saved agent.yaml.

    The saved ``params/agent.yaml`` records the exact network architecture the
    checkpoint was trained with; using it lets policies with differing
    architectures load against the shared current env config. Falls back to the
    current-registry ``agent_cfg`` (with the deprecation shim) when absent.

    Args:
        agent_cfg: The hydra-provided (current-registry) agent cfg, used only as
            a fallback when the run has no saved agent.yaml.
        log_dir: The checkpoint's run directory (holds ``params/agent.yaml``).
        device: Device to run the policy on (overrides the saved value).

    Returns:
        ``(runner_cfg_dict, runner_class_name, clip_actions)``.
    """
    saved = os.path.join(log_dir, "params", "agent.yaml")
    if os.path.isfile(saved):
        from isaaclab.utils.io import load_yaml
        cfg = dict(load_yaml(saved))
        print(f"[worker] Using saved agent cfg (network arch): {saved}")
    else:
        cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version).to_dict()
        print("[worker] No saved agent.yaml found; using current-registry agent cfg.")
    cfg["seed"] = int(args_cli.seed) if args_cli.seed is not None else cfg.get("seed")
    cfg["device"] = device
    return cfg, cfg.get("class_name", "OnPolicyRunner"), cfg.get("clip_actions", None)


def _make_velocity_pin(vel_cmd, orig_update):
    """Build a per-segment ``_update_command`` post-hook factory pinning vx.

    Returns a function ``set_vx(vx)`` that re-wraps ``orig_update`` so that after
    the ramp/PD/guidance run, the forward velocity is hard-pinned to ``vx`` and
    lateral/yaw are zeroed. Re-wrapping from the captured original each call
    avoids stacking nested wrappers across velocity segments.
    """
    def set_vx(vx: float):
        standing = abs(vx) < 1e-6

        def patched():
            orig_update()
            vel_cmd.vel_command_b[:, 0] = vx
            vel_cmd.vel_command_b[:, 1:] = 0.0
            vel_cmd.vel_target_b[:, 0] = vx
            vel_cmd.vel_target_b[:, 1:] = 0.0
            if hasattr(vel_cmd, "vel_target_sampled_b"):
                vel_cmd.vel_target_sampled_b[:, 0] = vx
                vel_cmd.vel_target_sampled_b[:, 1:] = 0.0
            if hasattr(vel_cmd, "is_standing_env"):
                vel_cmd.is_standing_env[:] = standing

        vel_cmd._update_command = patched

    return set_vx


def _run_worker(env_cfg, agent_cfg) -> None:
    """Roll out one policy across all requested velocities and write its dump."""
    import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # Shared eval conditions (deterministic): the ENV uses the current registry
    # cfg + the shared seed/device; the policy network comes from its own saved
    # agent.yaml (resolved below) so the checkpoint loads regardless of how the
    # registered architecture has since changed.
    env_cfg.seed = int(args_cli.seed) if args_cli.seed is not None else env_cfg.seed
    device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.device = device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = _resolve_worker_checkpoint(agent_cfg, log_root_path, train_task_name)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    traj_path_used, traj_fingerprint = _apply_trajectory_library(env_cfg, log_dir)

    # Build the runner cfg from the policy's OWN saved agent.yaml so the network
    # architecture matches the checkpoint. Fall back to the current registry cfg
    # (with the deprecation shim) if the run has no saved agent.yaml.
    runner_cfg, runner_class, clip_actions = _load_runner_cfg(agent_cfg, log_dir, device)

    # --- eval overrides on the velocity command cfg (before env build) ---
    velocity_pinned = True
    vc = getattr(env_cfg.commands, args_cli.vel_cmd_name, None)
    if vc is not None:
        vc.resampling_time_range = (1.0e9, 1.0e9)
        if hasattr(vc, "skill_transition_prob"):
            vc.skill_transition_prob = 0.0
    else:
        velocity_pinned = False
        print(f"[worker] WARNING: no '{args_cli.vel_cmd_name}' command term in cfg; "
              f"forward velocity will NOT be pinned.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
        from isaaclab.envs import multi_agent_to_single_agent
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    print(f"[worker] Loading checkpoint: {resume_path}")
    if runner_class == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=device)
    elif runner_class == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=None, device=device)
    else:
        raise ValueError(f"Unsupported runner class: {runner_class}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    dev = unwrapped.device
    n_envs = unwrapped.num_envs
    dt = float(unwrapped.step_dt)
    tm = unwrapped.termination_manager
    term_names = list(tm.active_terms)

    # --- resolve the trajectory command + its CLF / output structure ---
    try:
        traj = unwrapped.command_manager.get_term(args_cli.cmd_name)
    except (KeyError, Exception):  # noqa: BLE001
        traj = None
    had_clf = traj is not None and getattr(traj, "clf", None) is not None
    clf = getattr(traj, "clf", None) if traj is not None else None
    pos_names = list(getattr(traj, "ordered_pos_output_names", [])) if traj is not None else []
    vel_names = list(getattr(traj, "ordered_vel_output_names", [])) if traj is not None else []

    subgroup_names: list[str] = []
    if clf is not None and hasattr(clf, "subgroup_indices"):
        subgroup_names = [k for k, idx in clf.subgroup_indices.items() if idx.numel() > 0]
    had_subgroups = len(subgroup_names) > 0

    # Tracking frames: prefer the frame_drift termination's frames (parity with eval_per_trajectory).
    track_frames = DEFAULT_TRACK_FRAMES
    with contextlib.suppress(Exception):
        track_frames = tuple(tm.get_term_cfg("frame_drift").params.get("frame_names", DEFAULT_TRACK_FRAMES))
    pos_idx, frame_names = (None, [])
    if pos_names:
        pos_idx, frame_names = resolve_frame_pos_indices(pos_names, track_frames, dev)

    mgr = getattr(traj, "manager", None) if traj is not None else None
    is_multiskill = mgr is not None and hasattr(mgr, "get_current_trajectory_indices")
    # skill_labels is per-trajectory; the distinct count is the number of skills.
    num_skills = len(set(getattr(mgr, "skill_labels", []) or [])) if is_multiskill else 1

    # --- velocity pin setup ---
    set_vx = None
    if velocity_pinned:
        vel_cmd = unwrapped.command_manager.get_term(args_cli.vel_cmd_name)
        if not hasattr(vel_cmd, "vel_command_b"):
            velocity_pinned = False
            print(f"[worker] WARNING: '{args_cli.vel_cmd_name}' has no vel_command_b buffer; "
                  f"cannot pin forward velocity.")
        else:
            set_vx = _make_velocity_pin(vel_cmd, vel_cmd._update_command)

    d_pos, d_vel, n_frames = len(pos_names), len(vel_names), len(frame_names)
    n_sub = len(subgroup_names)

    # --- roll out each velocity ---
    # We never call env.reset() (matching play.py / eval_per_trajectory.py):
    # episodes reset naturally inside env.step, so we avoid mutating the
    # velocity command's inference tensors outside inference mode. Each velocity
    # segment is gated by its own warmup (steps_since_reset reset to 0), which
    # excludes the transient while the robot settles at the newly pinned vx.
    per_vel: list[dict] = []
    velocities = list(args_cli.velocities)
    print(f"[worker] {n_envs} envs x {args_cli.steps}(+{args_cli.warmup}) steps "
          f"per velocity over {velocities}; multiskill={is_multiskill}, clf={had_clf}.")
    obs = env.get_observations()
    # Record the actor obs dimension for the cross-policy compatibility check.
    obs_dim = 0
    try:
        actor_obs = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
        obs_dim = int(actor_obs.shape[-1])
    except (KeyError, AttributeError, TypeError, IndexError):
        obs_dim = int(getattr(env, "num_obs", 0) or 0)

    for vx in velocities:
        if set_vx is not None:
            set_vx(vx)
        seg = _SegmentMetrics(n_envs, d_pos, d_vel, n_frames, n_sub, dev)
        ep = EpisodeStatsAccumulator()
        episode_ends = 0
        steps_since_reset = torch.zeros(n_envs, dtype=torch.long, device=dev)
        printed_pin = False

        for t in range(args_cli.warmup + args_cli.steps):
            # Snapshot current-state metrics BEFORE stepping (parity with eval_per_trajectory).
            v = None
            if had_clf and isinstance(getattr(traj, "v", None), torch.Tensor):
                v = traj.v.detach()
            pos_sq = vel_sq = frame = sub = None
            if traj is not None:
                pos_sq = (traj.y_des - traj.y_act) ** 2 if d_pos else None
                vel_sq = (traj.dy_des - traj.dy_act) ** 2 if d_vel else None
                if pos_idx is not None:
                    err = (traj.y_act - traj.y_des)[:, pos_idx]          # [N, F, 3]
                    frame = torch.linalg.norm(err, dim=-1)               # [N, F]
                if had_subgroups:
                    vsg = getattr(clf, "v_subgroups", {}) or {}
                    sub = torch.stack(
                        [vsg[name] if name in vsg else torch.zeros(n_envs, device=dev)
                         for name in subgroup_names], dim=1)              # [N, G]
            vx_act = _to_torch(unwrapped.scene["robot"].data.root_lin_vel_b)[:, 0]
            abs_err = torch.abs(vx_act - vx)

            with torch.inference_mode():
                actions = policy(obs, stochastic_output=False)
                obs, _, dones, infos = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

            dones_b = dones.to(dev).bool()
            term_fired = {n: tm.get_term(n) for n in term_names}
            active = (steps_since_reset >= args_cli.warmup) & ~dones_b
            seg.add(active, v, vx_act, abs_err, pos_sq, vel_sq, frame, sub)
            ep.collect_step(infos, dones_b, term_fired=term_fired)
            episode_ends += int(dones_b.sum().item())
            steps_since_reset = torch.where(
                dones_b, torch.zeros_like(steps_since_reset), steps_since_reset + 1)

            if not printed_pin and t == args_cli.warmup and velocity_pinned:
                vcb = unwrapped.command_manager.get_term(args_cli.vel_cmd_name).vel_command_b
                print(f"[worker] vx={vx:.3f}: pinned vel_command_b[:,0] "
                      f"min={float(vcb[:, 0].min()):.4f} max={float(vcb[:, 0].max()):.4f} "
                      f"| lateral|yaw max={float(vcb[:, 1:].abs().max()):.4f}", flush=True)
                printed_pin = True

        seg_res = seg.finalize()
        fracs = ep.termination_fractions()  # {name: (frac, count)}
        seg_res["term_frac"] = np.array([fracs.get(n, (0.0, 0))[0] for n in term_names], dtype=np.float64)
        seg_res["term_counts"] = np.array([fracs.get(n, (0.0, 0))[1] for n in term_names], dtype=np.int64)
        seg_res["episode_ends"] = episode_ends
        per_vel.append(seg_res)
        print(f"[worker] vx={vx:.3f} done: clf_v={seg_res['clf_v_mean']:.4f}, "
              f"vx_actual={seg_res['vx_actual_mean']:.4f}, episode_ends={episode_ends}")

    env.close()

    # --- write dump (npz arrays + json metadata) ---
    def stack(key):
        return np.stack([pv[key] for pv in per_vel], axis=0)

    dump = dict(
        velocities=np.asarray(velocities, dtype=np.float64),
        clf_v_mean=np.asarray([pv["clf_v_mean"] for pv in per_vel], dtype=np.float64),
        clf_v_std=np.asarray([pv["clf_v_std"] for pv in per_vel], dtype=np.float64),
        clf_subgroup_mean=stack("clf_subgroup_mean"),
        clf_subgroup_std=stack("clf_subgroup_std"),
        vel_track_err_mean=np.asarray([pv["vel_track_err_mean"] for pv in per_vel], dtype=np.float64),
        vel_track_err_std=np.asarray([pv["vel_track_err_std"] for pv in per_vel], dtype=np.float64),
        vx_actual_mean=np.asarray([pv["vx_actual_mean"] for pv in per_vel], dtype=np.float64),
        vx_actual_std=np.asarray([pv["vx_actual_std"] for pv in per_vel], dtype=np.float64),
        term_frac=stack("term_frac"),
        term_counts=stack("term_counts"),
        episode_ends=np.asarray([pv["episode_ends"] for pv in per_vel], dtype=np.int64),
        pos_err_mse=stack("pos_err_mse"),
        pos_err_mse_std=stack("pos_err_mse_std"),
        vel_err_mse=stack("vel_err_mse"),
        vel_err_mse_std=stack("vel_err_mse_std"),
        frame_track_err_mean=stack("frame_track_err_mean"),
        frame_track_err_std=stack("frame_track_err_std"),
    )
    out_dump = args_cli.out_dump or os.path.join(log_dir, "policy_comparison", "self_dump.npz")
    os.makedirs(os.path.dirname(out_dump), exist_ok=True)
    np.savez(out_dump, **dump)

    meta = dict(
        policy_label=args_cli.policy_label or os.path.basename(log_dir),
        log_dir=os.path.abspath(log_dir),
        checkpoint=os.path.abspath(resume_path),
        term_names=term_names,
        pos_names=pos_names,
        vel_names=vel_names,
        clf_subgroup_names=subgroup_names,
        frame_names=list(frame_names),
        dt=dt,
        task=args_cli.task,
        seed=int(agent_cfg.seed),
        num_envs=int(n_envs),
        steps=int(args_cli.steps),
        warmup=int(args_cli.warmup),
        had_clf=bool(had_clf),
        had_subgroups=bool(had_subgroups),
        is_multiskill=bool(is_multiskill),
        num_skills=int(num_skills),
        velocity_pinned=bool(velocity_pinned),
        vel_cmd_name=args_cli.vel_cmd_name,
        traj_path=str(traj_path_used) if traj_path_used is not None else None,
        traj_fingerprint=str(traj_fingerprint) if traj_fingerprint is not None else None,
        traj_source=args_cli.traj_source if not args_cli.shared_traj_path else "shared_override",
        obs_dim=int(obs_dim),
        action_dim=int(getattr(env, "num_actions", 0) or 0),
    )
    with open(os.path.splitext(out_dump)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[worker] dump written: {out_dump}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Worker entry: launch one sim app and roll out one policy."""
    with launch_simulation(env_cfg, args_cli):
        _run_worker(env_cfg, agent_cfg)


# ---------------------------------------------------------------------------
# Orchestrator mode
# ---------------------------------------------------------------------------

def _resolve_spec(spec: str, experiment_name: str | None, checkpoint_regex: str) -> tuple[str, str]:
    """Resolve a --runs spec to ``(log_dir, checkpoint_path)`` (filesystem only)."""
    spec = os.path.expanduser(spec)
    if os.path.isfile(spec) and spec.endswith(".pt"):
        ckpt = os.path.abspath(spec)
        return os.path.dirname(ckpt), ckpt
    if os.path.isdir(spec):
        run_dir = os.path.abspath(spec)
        ckpt = get_checkpoint_path(os.path.dirname(run_dir), os.path.basename(run_dir), checkpoint_regex)
        return os.path.dirname(ckpt), ckpt
    if not experiment_name:
        raise SystemExit(
            f"[orchestrator] '{spec}' is not a path and --experiment_name was not given; "
            f"cannot resolve a bare run name.")
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    ckpt = get_checkpoint_path(log_root, spec, checkpoint_regex)
    return os.path.dirname(ckpt), ckpt


def _worker_argv(task: str, ckpt: str, dump: str, label: str) -> list[str]:
    """Build the argv for one worker subprocess."""
    argv = [
        sys.executable, os.path.abspath(__file__), "--_worker",
        "--task", task, "--agent", args_cli.agent,
        "--checkpoint", ckpt,
        "--num_envs", str(args_cli.num_envs),
        "--steps", str(args_cli.steps), "--warmup", str(args_cli.warmup),
        "--seed", str(args_cli.seed),
        "--cmd_name", args_cli.cmd_name, "--vel_cmd_name", args_cli.vel_cmd_name,
        "--traj_source", args_cli.traj_source,
        "--velocities", *[str(v) for v in args_cli.velocities],
        "--out_dump", dump, "--policy_label", label,
        "--headless",
    ]
    if args_cli.shared_traj_path:
        argv += ["--shared_traj_path", args_cli.shared_traj_path]
    if args_cli.device is not None:
        argv += ["--device", str(args_cli.device)]
    if args_cli.disable_fabric:
        argv += ["--disable_fabric"]
    # Forward any hydra-style overrides the user passed through.
    argv += hydra_args
    return argv


def orchestrate() -> None:
    """Resolve runs, spawn one worker per policy, then aggregate into each folder."""
    if not args_cli.runs:
        raise SystemExit("[orchestrator] --runs is required (one or more policy specs).")

    # Resolve every spec up front (fail fast, before any sim launch).
    resolved: list[tuple[str, str, str]] = []  # (label, log_dir, checkpoint)
    for spec in args_cli.runs:
        log_dir, ckpt = _resolve_spec(spec, args_cli.experiment_name, args_cli.checkpoint_regex)
        label = os.path.basename(log_dir) if os.path.isdir(log_dir) else spec
        resolved.append((label, log_dir, ckpt))
        print(f"[orchestrator] {spec}  ->  {ckpt}")

    out_root = args_cli.out_root or tempfile.mkdtemp(prefix="policy_compare_")
    os.makedirs(out_root, exist_ok=True)
    print(f"[orchestrator] worker dumps -> {out_root}")

    # Run one worker subprocess per policy (sequential: one sim app at a time).
    succeeded: list[tuple[str, str]] = []  # (dump_path, log_dir)
    for i, (label, log_dir, ckpt) in enumerate(resolved):
        dump = os.path.join(out_root, f"policy_{i}.npz")
        argv = _worker_argv(args_cli.task, ckpt, dump, label)
        print(f"\n[orchestrator] === policy {i + 1}/{len(resolved)}: {label} ===")
        ret = subprocess.run(argv).returncode
        if ret != 0 or not os.path.isfile(dump):
            print(f"[orchestrator] WARNING: worker for '{label}' failed "
                  f"(exit {ret}, dump present={os.path.isfile(dump)}); dropping it.")
            continue
        succeeded.append((dump, log_dir))

    if not succeeded:
        raise SystemExit("[orchestrator] No policies produced a dump; nothing to compare.")

    dumps = [d for d, _ in succeeded]
    log_dirs = [ld for _, ld in succeeded]
    print(f"\n[orchestrator] aggregating {len(dumps)} policies -> comparison plots/stats")
    compare_plots.aggregate_and_plot(dumps, log_dirs, which=args_cli.plots)
    print("[orchestrator] done.")


if __name__ == "__main__":
    if args_cli._worker:
        main()
    else:
        orchestrate()
