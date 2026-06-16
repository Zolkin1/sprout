# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless per-trajectory eval: CLF/tracking + termination stats.

Runs a checkpoint deterministically across many parallel envs (no video,
JIT/ONNX, MoE hooks, or real-time pacing) and accumulates, per global
trajectory, the mean Lyapunov value ``V``, the frame tracking error, and the
per-termination-term episode-end fractions. Writes a full CSV, a triage text
report, and per-skill metric-vs-conditioning plots next to the checkpoint.

Coverage scales with ``--num_envs`` x ``--steps`` — use a large batch so the
~1300 trajectories each get enough samples (see the low-coverage section of the
text report).
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys

import gymnasium as gym
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
from per_traj_stats import (  # isort: skip
    DEFAULT_TRACK_FRAMES,
    PerTrajectoryStats,
    build_traj_names,
    frame_tracking_error,
    resolve_frame_pos_indices,
)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(
    description="Per-trajectory CLF/tracking + termination eval for an RSL-RL checkpoint."
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--steps", type=int, default=3000, help="Number of rollout steps.")
parser.add_argument(
    "--cmd_name", type=str, default="traj_ref",
    help="Trajectory command term name (multiskill).",
)
parser.add_argument(
    "--warmup", type=int, default=3,
    help="Steps to skip after each trajectory change/reset before counting CLF "
         "and tracking-error samples (drops the transition transient).",
)
parser.add_argument("--top_k", type=int, default=20, help="Rows in the worst-trajectory tables.")
parser.add_argument(
    "--min_samples", type=int, default=20,
    help="Min CLF samples for a trajectory to rank in the mean_v table.",
)
parser.add_argument(
    "--min_eps", type=int, default=5,
    help="Min episode-ends for a trajectory to rank in the termination table.",
)
parser.add_argument(
    "--out_dir", type=str, default=None,
    help="Output directory. Defaults to <checkpoint_dir>/per_traj.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out the policy and write the per-trajectory report."""
    with launch_simulation(env_cfg, args_cli):
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] No pre-trained checkpoint available for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            if agent_cfg.class_name == "DistillationRunner" and getattr(args_cli, "teacher_experiment", None):
                play_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment))
            else:
                play_log_path = log_root_path
            resume_path = get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading checkpoint: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        unwrapped = env.unwrapped
        dev = unwrapped.device
        n_envs = unwrapped.num_envs

        # Resolve the trajectory command + its multiskill manager.
        cmd = unwrapped.command_manager.get_term(args_cli.cmd_name)
        mgr = getattr(cmd, "manager", None)
        if mgr is None or not hasattr(mgr, "get_current_trajectory_indices"):
            print(f"[ERROR] Command '{args_cli.cmd_name}' is not a multiskill command "
                  f"with per-trajectory indices. Is this a multiskill task?")
            env.close()
            return

        tm = unwrapped.termination_manager
        term_names = list(tm.active_terms)

        # Trajectory tracking-error frames: prefer the frame_drift cfg frames.
        track_frames = DEFAULT_TRACK_FRAMES
        with contextlib.suppress(Exception):
            fd_cfg = tm.get_term_cfg("frame_drift")
            track_frames = tuple(fd_cfg.params.get("frame_names", DEFAULT_TRACK_FRAMES))
        pos_idx, used_frames = resolve_frame_pos_indices(
            cmd.ordered_pos_output_names, track_frames, dev
        )
        if pos_idx is None:
            print("[WARN] No tracking-error frames resolved; tracking error disabled.")
        else:
            print(f"[INFO] Tracking-error frames: {used_frames}")

        stats = PerTrajectoryStats(
            num_trajectories=int(mgr.num_trajectories),
            term_names=term_names,
            device=dev,
            skill_labels=list(getattr(mgr, "skill_labels", []) or []) or None,
            conditioning=getattr(mgr, "_global_conditioning", None),
            traj_names=build_traj_names(mgr),
        )

        obs = env.get_observations()
        prev_traj: torch.Tensor | None = None
        steps_since_change = torch.zeros(n_envs, dtype=torch.long, device=dev)

        print(f"[INFO] Rolling out {args_cli.steps} steps across {n_envs} envs "
              f"({mgr.num_trajectories} trajectories, {len(term_names)} term terms)...")
        for t in range(args_cli.steps):
            # Snapshot BEFORE the step: this is the trajectory/state active during
            # the step, so terminations attribute to the live traj (not a
            # post-_reset_idx resample).
            traj_idx = mgr.get_current_trajectory_indices().clone()
            v = cmd.v.clone() if hasattr(cmd, "v") and isinstance(getattr(cmd, "v"), torch.Tensor) else None
            track_err = frame_tracking_error(cmd, pos_idx) if pos_idx is not None else None

            if prev_traj is None:
                changed = torch.ones(n_envs, dtype=torch.bool, device=dev)
            else:
                changed = traj_idx != prev_traj
            steps_since_change = torch.where(
                changed, torch.zeros_like(steps_since_change), steps_since_change + 1
            )
            prev_traj = traj_idx

            with torch.inference_mode():
                actions = policy(obs, stochastic_output=False)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

            term_fired = {n: tm.get_term(n) for n in term_names}
            active = (steps_since_change >= args_cli.warmup) & ~dones.to(dev).bool()
            stats.collect_step(traj_idx, v, track_err, active, term_fired, dones.to(dev).bool())

        stats.finalize()
        env.close()

        out_dir = args_cli.out_dir or os.path.join(log_dir, "per_traj")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "per_traj_stats.csv")
        txt_path = os.path.join(out_dir, "per_traj_stats.txt")
        plots_dir = os.path.join(out_dir, "plots")

        stats.write_csv(csv_path)
        stats.write_text_report(
            txt_path, num_envs=n_envs, top_k=args_cli.top_k,
            min_samples=args_cli.min_samples, min_eps=args_cli.min_eps,
        )
        written = stats.make_plots(plots_dir)

        # Echo the triage report to stdout for convenience.
        print("\n" + stats.format_text_report(
            num_envs=n_envs, top_k=args_cli.top_k,
            min_samples=args_cli.min_samples, min_eps=args_cli.min_eps,
        ))
        print(f"[INFO] CSV:   {csv_path}")
        print(f"[INFO] Text:  {txt_path}")
        print(f"[INFO] Plots: {len(written)} written to {plots_dir}")


if __name__ == "__main__":
    main()
