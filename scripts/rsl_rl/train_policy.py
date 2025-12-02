import argparse
import os
import sys
from datetime import datetime

import cli_args
from isaaclab.app import AppLauncher

# Environment names
ENVIRONMENTS = {
    "vanilla": "G1-flat-vel",
    "lip_ref_tracking": "G1-flat-ref-tracking",
    "lip_clf_vdot": "G1-flat-clf-vdot",
    "lip_clf": "G1-lip-clf-custom",
    "mlip_clf": "G1-mlip-clf-custom",
    "hzd_clf": "G1-hzd-clf",
    "hzd_clf_custom": "G1-hzd-clf-custom",
    "hzd_clf_minimum": "G1-hzd-clf-minimum",
    "stepping_stone": "G1-steppingstone",
    "stepping_stone_distillation": "G1-steppingstone-distillation",
    "stepping_stone_finetune": "G1-steppingstone-finetune",
    "stepping_stone_noheightmap": "G1-steppingstone-testing-no-heightmap",
    "stepping_stone_noheightmapdistill": "G1-steppingstone-testing-no-heightmapdistill",
    "stepping_stone_noheightmapfinetune": "G1-steppingstone-testing-no-heightmapfinetune",
    "stepping_stone_baseline": "G1-steppingstone-baseline",
}

EXPERIMENT_NAMES = {
    "vanilla": "baseline",
    "basic": "baseline",
    "lip_clf": "lip",
    "mlip_clf": "mlip",
    "lip_ref_tracking": "lip",
    "lip_clf_vdot": "lip",
    "lip_ref_play": "lip",
    "hzd_clf": "hzd",
    "hzd_clf_play": "hzd",
    "hzd_clf_custom": "hzd",
    "hzd_clf_minimum": "hzd",
    "stepping_stone": "stepping_stone",
    "stepping_stone_distillation": "stepping_stone",
    "stepping_stone_finetune": "stepping_stone",
    "stepping_stone_noheightmap": "stepping_stone",
    "stepping_stone_noheightmapdistill": "stepping_stone",
    "stepping_stone_noheightmapfinetune": "stepping_stone",
    "stepping_stone_baseline": "stepping_stone_baseline",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policies for different environments.")
    parser.add_argument(
        "--env_type",
        type=str,
        choices=list(ENVIRONMENTS.keys()),
        help="Type of environment to train on (vanilla/custom/clf)",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    parser.add_argument(
        "--agent", 
        type=str, 
        default="rsl_rl_cfg_entry_point", 
        help="Name of the RL agent configuration entry point."
    )
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()
import re
def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", checkpoint: str = ".*", other_dirs: list[str] = None, sort_alpha: bool = True
) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: ``<log_path>/<run_dir>/<*other_dirs>/<checkpoint>``, where the
    :attr:`other_dirs` are intermediate folder names to concatenate. These cannot be regex expressions.

    If :attr:`run_dir` and :attr:`checkpoint` are regex expressions then the most recent (highest alphabetical order)
    run and checkpoint are selected. To disable this behavior, set the flag :attr:`sort_alpha` to False.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The regex expression for the name of the directory containing the run. Defaults to the most
            recent directory created inside :attr:`log_path`.
        other_dirs: The intermediate directories between the run directory and the checkpoint file. Defaults to
            None, which implies that checkpoint file is directly under the run directory.
        checkpoint: The regex expression for the model checkpoint file. Defaults to the most recent
            torch-model saved in the :attr:`run_dir` directory.
        sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
            If False, the folders in :attr:`run_dir` are sorted by the last modified time.

    Returns:
        The path to the model checkpoint.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in the input directory.

    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        if other_dirs is not None:
            run_path = os.path.join(runs[-1], *other_dirs)
        else:
            run_path = runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)

def main():
    args_cli, hydra_args = parse_args()

    if not args_cli.env_type:
        print("Please specify an environment type using --env_type")
        print("Available options:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    # Set the task based on environment type
    args_cli.task = ENVIRONMENTS[args_cli.env_type]
    args_cli.logger = "wandb"
    args_cli.log_project_name = "g1_rl"

    # always enable cameras to record video
    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import necessary modules after app launch
    import gymnasium as gym
    import torch
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
    import omni
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import robot_rl.tasks

    from rsl_rl.runners import OnPolicyRunner, DistillationRunner

    # Configure PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, args_cli.agent)
    def train(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        """Train with RSL-RL agent."""
        # Override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # Set the environment seed
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Multi-gpu training configuration
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # Create organized directory structure for logging

        base_log_path = os.path.join("logs", "g1_policies", EXPERIMENT_NAMES[args_cli.env_type])
        log_root_path = os.path.join(base_log_path, EXPERIMENT_NAMES[args_cli.env_type])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

        # Create timestamp-based run directory
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

 

        # set the log directory for the environment (works for all environment types)
        env_cfg.log_dir = log_dir
        # Create environment
        if hasattr(env_cfg, "__prepare_tensors__") and callable(getattr(env_cfg, "__prepare_tensors__")):
            env_cfg.__prepare_tensors__()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # Convert to single-agent if needed
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # Handle resume path
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # Setup video recording if enabled
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

        # Wrap environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner from rsl-rl
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # write git state to logs
        runner.add_git_repo_to_log(__file__)

        # Load checkpoint if resuming
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        # Save configurations
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        # Run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

        # Cleanup
        env.close()

    # Run training
    train()
    # Close sim app
    simulation_app.close()


if __name__ == "__main__":
    main()
