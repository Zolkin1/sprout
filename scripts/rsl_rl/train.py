# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import os
import platform
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments (before hydra_task_config)
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

RSL_RL_VERSION = "3.0.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# -- argparse ----------------------------------------------------------------
# allow_abbrev=False: never accept a prefix of a long option as that option, so
# ``--clip <h5>`` can't silently bind to ``--clip_device`` (feeding a clip PATH
# as a torch device string). The explicit ``--clip`` flag below is the only
# meaning of ``--clip``; any other ambiguous prefix falls through to hydra.
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.", allow_abbrev=False)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--profile", action="store_true", default=False,
    help="Enable the per-phase profiling harness (CUDA-event timers + manager "
         "breakdown). Run with a small --max_iterations (e.g. 3). Zero cost when off.",
)
parser.add_argument(
    "--profile_trace", action="store_true", default=False,
    help="Also run torch.profiler during the (bounded) learn and dump a Chrome "
         "trace + top-op table. Use with a small --max_iterations.",
)
parser.add_argument(
    "--profile_report_every", type=int, default=50,
    help="With --profile, print a cumulative per-phase snapshot every N env "
         "steps during the run (also written to logs so it survives a hard "
         "kill). 0 disables periodic snapshots (report only at the end).",
)
parser.add_argument(
    "--profile_sync", action="store_true", default=False,
    help="Profile in synchronous mode: drain the whole device at each phase "
         "boundary for true, correctly-nested per-phase costs (incl. Warp "
         "physics on its own stream). Slower than the default async timers; "
         "use it to resolve cross-stream mis-attribution.",
)
parser.add_argument(
    "--clip", type=str, default=None,
    help="Path to the clip / trajectory library (.h5) to track, overriding the "
         "cfg default. Applied to any command term exposing a `path` field "
         "(traj_ref). Equivalent to env.commands.traj_ref.path=<h5> without hydra.",
)
parser.add_argument(
    "--clip_device", type=str, default=None,
    help="Device to hold a trajectory/clip library on (e.g. 'cpu' to keep a multi-GB "
         "library off a small GPU). Applied to any command term exposing a clip_device "
         "field. Set here (not via hydra) because the cfg default is None.",
)
parser.add_argument(
    "--fm_checkpoint", type=str, default=None,
    help="Path to the flow-matching generator checkpoint (.pt) to track. Applied to any "
         "command term exposing an fm_checkpoint field (the GeneratorTrackingCommand), "
         "overriding the cfg default.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# -- check RSL-RL version ----------------------------------------------------
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # Patch activate_contact_sensors for recursive traversal (must be after SimulationApp)
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # Optionally override the clip/trajectory library path (the file each
        # command term tracks); equivalent to env.commands.traj_ref.path=<h5>.
        if args_cli.clip is not None and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
            for name, term in vars(env_cfg.commands).items():
                if hasattr(term, "path"):
                    term.path = args_cli.clip
                    print(f"[INFO] clip path='{args_cli.clip}' for commands.{name}.")

        # Optionally hold a clip/trajectory library off-GPU (set in Python, not via
        # hydra: the cfg field default is None so hydra rejects a str override).
        if args_cli.clip_device is not None and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
            for name, term in vars(env_cfg.commands).items():
                if hasattr(term, "clip_device"):
                    term.clip_device = args_cli.clip_device
                    print(f"[INFO] clip_device='{args_cli.clip_device}' for commands.{name}.")

        # Optionally override the flow-matching generator checkpoint to track
        # (mirrors --clip_device; the cfg default is a str so hydra works too,
        # but the flag is the convenient way to pick a freshly-trained FM).
        if args_cli.fm_checkpoint is not None and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
            for name, term in vars(env_cfg.commands).items():
                if hasattr(term, "fm_checkpoint"):
                    term.fm_checkpoint = args_cli.fm_checkpoint
                    print(f"[INFO] fm_checkpoint='{args_cli.fm_checkpoint}' for commands.{name}.")
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # check for invalid combination of CPU device with distributed training
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError(
                "Distributed training is not supported when using CPU device. "
                "Please use GPU device (e.g., --device cuda) for distributed training."
            )

        # multi-gpu training configuration
        if args_cli.distributed:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            global_rank = int(os.getenv("RANK", "0"))
            env_cfg.sim.device = f"cuda:{local_rank}"
            agent_cfg.device = f"cuda:{local_rank}"

            # use global rank for seed diversity across all nodes
            seed = agent_cfg.seed + global_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
        # change it (see PR #2346, comment-2819298849)
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # set the IO descriptors export flag if requested
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning(
                "IO descriptors are only supported for manager based RL environments."
                " No IO descriptors will be exported."
            )

        # set the log directory for the environment (works for all environment types)
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # save resume path before creating a new log_dir
        if agent_cfg.class_name == "DistillationRunner":
            # Resolve teacher checkpoint from a (potentially different) experiment directory
            teacher_experiment = getattr(args_cli, "teacher_experiment", None) or agent_cfg.teacher_experiment_name
            teacher_run = getattr(args_cli, "teacher_run", None) or agent_cfg.teacher_load_run
            teacher_checkpoint = getattr(args_cli, "teacher_checkpoint", None) or agent_cfg.teacher_load_checkpoint
            teacher_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", teacher_experiment))
            resume_path = get_checkpoint_path(teacher_log_path, teacher_run, teacher_checkpoint)
        elif agent_cfg.resume:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner from rsl-rl
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        # Don't upload model checkpoints to W&B (they eat a lot of cloud storage and
        # we keep/use the local copies instead). Checkpoints are still written to
        # ``log_dir`` by the runner, and code/config state still uploads via save_file.
        if agent_cfg.logger == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            WandbSummaryWriter.save_model = lambda self, model_path, it: None

        # print parameter counts
        if isinstance(runner, DistillationRunner):
            student_params = sum(p.numel() for p in runner.alg.student.parameters())
            teacher_params = sum(p.numel() for p in runner.alg.teacher.parameters())
            print(f"Student Parameters: {student_params:,}")
            print(f"Teacher Parameters: {teacher_params:,}")
            if hasattr(runner.alg, "critic"):
                critic_params = sum(p.numel() for p in runner.alg.critic.parameters())
                print(f"Critic Parameters: {critic_params:,}")
                print(f"Total Parameters: {student_params + teacher_params + critic_params:,}")
        else:
            actor_params = sum(p.numel() for p in runner.alg.actor.parameters())
            critic_params = sum(p.numel() for p in runner.alg.critic.parameters())
            print(f"Actor Parameters: {actor_params:,}")
            print(f"Critic Parameters: {critic_params:,}")
            print(f"Total Parameters: {actor_params + critic_params:,}")

        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        # load the checkpoint
        if agent_cfg.resume or agent_cfg.class_name == "DistillationRunner":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # optional profiling harness — instrument the manager computes + env.step
        # so the rollout time (rsl_rl's logged Perf/collection_time) is broken
        # down per manager, with the remainder ("unaccounted") ≈ physics.
        phase_timer = None
        profile_report_path = None
        profile_report_kwargs = None
        if args_cli.profile:
            from profiling import PhaseTimer, TorchProfilerSession, instrument_env_managers, instrument_frame_step

            profile_report_path = os.path.join(log_dir, "profile_report.txt")
            profile_report_kwargs = {
                "total_key": "frame_total",
                "top_level": ["obs", "reward", "termination", "command", "events"],
                "title": "TRAIN per-step breakdown",
                "out_path": profile_report_path,
            }
            phase_timer = PhaseTimer(agent_cfg.device, sync=args_cli.profile_sync)
            instrument_env_managers(env, phase_timer)
            # Periodic cumulative snapshots during the run (live + kill-safe);
            # the same snapshot is appended to ``profile_report.txt``.
            instrument_frame_step(
                env, phase_timer,
                report_every=(args_cli.profile_report_every or None),
                report_kwargs=profile_report_kwargs,
            )
            print(
                f"[profiling] enabled  |  sync mode: "
                f"{'ON (drains device at each phase; slower but exact nesting)' if phase_timer.sync else 'OFF (async CUDA-event timers)'}"
                f"  |  requested --profile_sync={args_cli.profile_sync}, device={agent_cfg.device}",
                flush=True,
            )
            print(
                f"[profiling] periodic snapshots every "
                f"{args_cli.profile_report_every} steps -> console + "
                f"{profile_report_path}. Also read rsl_rl's per-iteration "
                "Perf/collection_time, Perf/learning_time, Perf/total_fps for "
                "the rollout-vs-update split. To stop early use Ctrl-C (not "
                "kill) so the final report prints.",
                flush=True,
            )
            if args_cli.max_iterations is None or agent_cfg.max_iterations > 10:
                print(
                    "[profiling] WARNING: profiling is meant for a short run; "
                    "pass a small --max_iterations (e.g. 3).",
                    flush=True,
                )

        # run training
        try:
            # Debug NaN
            obs = runner.env.get_observations()
            for key, tensor in obs.items():
                if not isinstance(tensor, torch.Tensor):
                    # Non-concatenated obs groups arrive as TensorDicts (e.g. CNN-shape
                    # depth_image groups). torch.isnan doesn't dispatch on them; skip.
                    print(f"[DEBUG] Pre-learn obs '{key}': non-tensor type={type(tensor).__name__}")
                    continue
                nan_count = torch.isnan(tensor).sum().item()
                print(f"[DEBUG] Pre-learn obs '{key}': shape={tensor.shape}, NaN count={nan_count}")

            profiler_ctx = (
                TorchProfilerSession(log_dir) if (args_cli.profile and args_cli.profile_trace)
                else contextlib.nullcontext()
            )
            with profiler_ctx:
                runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        except KeyboardInterrupt:
            pass
        finally:
            if phase_timer is not None:
                phase_timer.report(**profile_report_kwargs)
            # close the simulator
            env.close()


if __name__ == "__main__":
    main()