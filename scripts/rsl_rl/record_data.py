# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play a trained RSL-RL policy and record (observation, action) pairs to disk.

The recorded dataset is intended for downstream supervised learning. Each row is
one (policy observation, action) pair, where the action is exactly what the
policy produced from that observation. Alongside the pairs we record:
  - ``terminated``: 1 if the pair led to a true termination (failure), excluding
    time-outs -- i.e. the last pair before the robot terminated.
  - ``done`` / ``time_out``: the combined episode-end flag and the time-out flag,
    so the consumer can distinguish failures from episode-length cutoffs.
  - ``episode_id`` / ``step_in_episode``: bookkeeping for episode-aware splits.
Plus provenance metadata (which checkpoint, its SHA-256, task, obs/action layout)
so the data is self-describing. See ``recorded_data_io.py`` for the on-disk
format and the loader used by the learning script.

Specify the total number of pairs with ``--num_pairs``; every run writes a new,
non-overwriting file under ``<log_dir>/recorded_data/`` (override with
``--output_dir``).

Example:
    python scripts/rsl_rl/record_data.py --task <Task> --num_envs 64 \
        --num_pairs 100000 --load_run <run> --headless
"""

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from export_parameters import export_policy_parameters  # isort: skip
from recorded_data_io import (  # isort: skip
    RecordingMeta,
    sha256_file,
    unique_output_path,
    save_recorded_data,
)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Play a trained RSL-RL policy and record (obs, action) pairs for supervised learning."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument(
    "--num_pairs", type=int, required=True,
    help="Total number of (observation, action) pairs to record.",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Where to write the dataset. Default: <log_dir>/recorded_data.",
)
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions stochastically. Default is deterministic (mean) actions.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_resume_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the checkpoint path the same way play.py does, without the simulator.

    Args:
        agent_cfg: The RSL-RL agent configuration (already CLI-updated).

    Returns:
        Absolute path to the checkpoint file to load.
    """
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    if getattr(args_cli, "teacher_experiment", None):
        play_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment))
    else:
        play_log_path = log_root_path
    return get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _build_recording_meta(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
    log_dir: str,
    num_pairs: int,
) -> RecordingMeta:
    """Capture provenance and obs/action layout metadata once at the start.

    Args:
        env: The wrapped Isaac Lab environment.
        agent_cfg: The RSL-RL agent configuration.
        resume_path: Absolute path to the loaded checkpoint.
        log_dir: Directory the checkpoint lives in.
        num_pairs: Final number of pairs that will be saved.

    Returns:
        A populated :class:`RecordingMeta`.
    """
    obs_mgr = env.unwrapped.observation_manager
    obs_term_names = list(obs_mgr.active_terms["policy"])
    obs_term_dims = [int(td[0]) for td in obs_mgr.group_obs_term_dim["policy"]]

    robot = env.unwrapped.scene.articulations["robot"]
    action_joint_names = list(robot.data.joint_names)

    # Raw robot-state layout recorded by collect_pairs (world frame, absolute).
    # Order and dims MUST match the torch.cat in collect_pairs.
    num_joints = len(action_joint_names)
    state_term_names = [
        "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel",
    ]
    state_term_dims = [3, 4, 3, 3, num_joints, num_joints]

    return RecordingMeta(
        checkpoint_path=os.path.abspath(resume_path),
        checkpoint_sha256=sha256_file(resume_path),
        task_name=args_cli.task,
        experiment_name=agent_cfg.experiment_name,
        run_name=os.path.basename(log_dir),
        runner_class=agent_cfg.class_name,
        rsl_rl_version=installed_version,
        recorded_at=datetime.now().isoformat(timespec="seconds"),
        stochastic=bool(args_cli.stochastic),
        dt=float(env.unwrapped.step_dt),
        num_envs=int(env.num_envs),
        num_pairs=int(num_pairs),
        obs_term_names=obs_term_names,
        obs_term_dims=obs_term_dims,
        action_joint_names=action_joint_names,
        state_term_names=state_term_names,
        state_term_dims=state_term_dims,
        state_quat_format="xyzw",
        state_joint_names=action_joint_names,
        state_vel_frame="com",
    )


def collect_pairs(
    env: RslRlVecEnvWrapper,
    policy,
    policy_nn,
    initial_obs,
    num_steps: int,
    num_pairs: int,
    stochastic: bool,
) -> dict[str, np.ndarray]:
    """Roll out the policy and collect flattened (obs, action) pairs.

    The action recorded for an observation is ``policy(obs)`` computed *before*
    stepping the environment, so each pair is correctly aligned. The
    ``terminated``/``done``/``time_out`` flags are recorded *after* the step and
    describe the transition out of that pair: ``terminated`` excludes time-outs,
    so it marks the last pair before a true (failure) termination.

    Args:
        env: The wrapped Isaac Lab environment.
        policy: Inference policy callable returning actions from observations.
        policy_nn: The policy network for ``reset`` on rsl-rl < 4.0.0; ``None``
            when the inference policy itself exposes ``reset`` (rsl-rl >= 4.0.0).
        initial_obs: The starting observation TensorDict (from ``get_observations``).
        num_steps: Number of rollout steps (``ceil(num_pairs / num_envs)``).
        num_pairs: Total pairs to keep; arrays are trimmed to this length.
        stochastic: Whether to sample actions stochastically.

    Returns:
        A dict of arrays each with leading dimension ``num_pairs``:
        ``obs``, ``action``, ``terminated``, ``done``, ``time_out``,
        ``episode_id``, ``step_in_episode``.
    """
    n_envs = env.num_envs
    device = env.unwrapped.device
    use_policy_reset = version.parse(installed_version) >= version.parse("4.0.0")
    robot = env.unwrapped.scene.articulations["robot"]

    obs = initial_obs
    # Distinct ids per env so step-0 pairs from different envs aren't conflated.
    episode_id = torch.arange(n_envs, dtype=torch.int64, device=device)
    step_in_ep = torch.zeros(n_envs, dtype=torch.int64, device=device)
    next_episode_id = int(n_envs)

    buf: dict[str, list[torch.Tensor]] = {
        "obs": [], "action": [], "state": [],
        "terminated": [], "done": [], "time_out": [],
        "episode_id": [], "step_in_episode": [],
    }

    print(f"[INFO] Collecting {num_steps} steps on {n_envs} envs "
          f"({num_steps * n_envs} pairs, trimming to {num_pairs})...")
    for step in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs, stochastic_output=stochastic)

        # Snapshot the (obs, action) pair *before* stepping. Clone because the
        # TensorDict tensors are overwritten by the next env.step.
        buf["obs"].append(obs["policy"].detach().cpu().clone())
        buf["action"].append(actions.detach().cpu().clone())
        # Raw robot world-state at the same timestep as obs. Order/dims MUST match
        # state_term_names/state_term_dims in _build_recording_meta.
        state = torch.cat([
            wp.to_torch(robot.data.root_pos_w),      # [N, 3]
            wp.to_torch(robot.data.root_quat_w),     # [N, 4] (xyzw)
            wp.to_torch(robot.data.root_lin_vel_w),  # [N, 3] (com vel)
            wp.to_torch(robot.data.root_ang_vel_w),  # [N, 3] (com ang vel)
            wp.to_torch(robot.data.joint_pos),       # [N, num_joints] (raw)
            wp.to_torch(robot.data.joint_vel),       # [N, num_joints] (raw)
        ], dim=-1)
        buf["state"].append(state.detach().cpu().clone())
        buf["episode_id"].append(episode_id.detach().cpu().clone())
        buf["step_in_episode"].append(step_in_ep.detach().cpu().clone())

        with torch.inference_mode():
            obs, _, dones, extras = env.step(actions)
            # Reset recurrent/transformer state for terminated envs.
            if use_policy_reset:
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # Disambiguate termination (failure) from truncation (time-out). The
        # wrapper returns dones = terminated | truncated and exposes truncation
        # via extras["time_outs"] (present when not is_finite_horizon).
        time_out = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        terminated = dones.bool() & ~time_out.bool()
        buf["done"].append(dones.detach().cpu().clone())
        buf["terminated"].append(terminated.detach().cpu().clone())
        buf["time_out"].append(time_out.detach().cpu().clone())

        # Episode bookkeeping for the *next* step's observation: envs that just
        # finished get a fresh id and a reset step counter.
        done_mask = dones.to(torch.bool)
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            new_ids = torch.arange(
                next_episode_id, next_episode_id + n_done, device=device, dtype=torch.int64
            )
            episode_id = episode_id.clone()
            episode_id[done_mask] = new_ids
            next_episode_id += n_done
            step_in_ep = step_in_ep.clone()
            step_in_ep[done_mask] = 0
            step_in_ep[~done_mask] += 1
        else:
            step_in_ep = step_in_ep + 1

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{num_steps}")

    # Stack to [T, N, ...], flatten time and env into one sample axis, trim.
    out: dict[str, np.ndarray] = {}
    for key, tensors in buf.items():
        stacked = torch.stack(tensors, dim=0).numpy()  # [T, N, ...]
        if stacked.ndim >= 3:
            flat = stacked.reshape(-1, *stacked.shape[2:])
        else:
            flat = stacked.reshape(-1)
        out[key] = flat[:num_pairs]
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load a checkpoint, roll it out, and save recorded (obs, action) pairs."""
    if args_cli.num_pairs <= 0:
        raise ValueError(f"--num_pairs must be positive; got {args_cli.num_pairs}.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    resume_path = _resolve_resume_path(agent_cfg)
    log_dir = os.path.dirname(resume_path)
    out_dir = args_cli.output_dir or os.path.join(log_dir, "recorded_data")

    with launch_simulation(env_cfg, args_cli):
        # Recursive contact sensor patch (matches play.py / analyze_distillation.py).
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        runner.alg.eval_mode()

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # For rsl-rl < 4.0.0 the inference policy doesn't expose reset(); fetch
        # the underlying network to reset recurrent state instead (see play.py).
        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy_nn = None
        elif version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        num_pairs = args_cli.num_pairs
        num_steps = math.ceil(num_pairs / env.num_envs)
        meta = _build_recording_meta(env, agent_cfg, resume_path, log_dir, num_pairs)

        # Single observation fetch reused for the memory estimate, the provenance
        # export, and the start of collection. compute() defaults to
        # update_history=False, so this does not mutate observation history.
        obs = env.get_observations()

        # Warn (don't fail) on very large in-memory datasets: npz buffers
        # everything in RAM before writing.
        obs_dim = int(obs["policy"].shape[1])
        est_bytes = num_pairs * (obs_dim + int(env.num_actions)) * 4
        if est_bytes > 8 * (1 << 30):
            print(f"[WARN] Estimated ~{est_bytes / (1 << 30):.1f} GiB of obs+action data will be "
                  f"buffered in RAM before writing. Consider fewer pairs.")

        # Export richer provenance YAML (gains, scales, default joint angles).
        # Optional: skip if the task lacks the terms it needs.
        with torch.inference_mode():
            first_actions = policy(obs, stochastic_output=args_cli.stochastic)
        try:
            export_policy_parameters(env, obs, first_actions, log_dir)
        except (KeyError, ValueError) as exc:
            print(f"[WARN] export_policy_parameters skipped: {exc}")

        # Clear any recurrent/transformer state advanced by the export-time
        # forward pass so the first recorded pair starts from a clean state.
        if policy_nn is None:
            policy.reset()
        else:
            policy_nn.reset()

        data = collect_pairs(env, policy, policy_nn, obs, num_steps, num_pairs, args_cli.stochastic)

        out_path = unique_output_path(out_dir, prefix="recorded", ext=".npz")
        save_recorded_data(out_path, data, meta)
        print(f"[INFO] Saved {meta.num_pairs} (obs, action) pairs to: {out_path}")
        print(f"[INFO]   obs shape={data['obs'].shape}, action shape={data['action'].shape}, "
              f"state shape={data['state'].shape}, terminations flagged={int(data['terminated'].sum())}")

        env.close()


if __name__ == "__main__":
    main()
