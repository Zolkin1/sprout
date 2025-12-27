import argparse
import glob
import os
import pickle
import sys
import time
import numpy as np

import cli_args
import torch
from isaaclab.app import AppLauncher

# Import plot_trajectories functions
from train_policy import ENVIRONMENTS, EXPERIMENT_NAMES
# Experiment names mapping for different environments

SIM_ENVIRONMENTS = {
    "vanilla": "G1-flat-vel",
    "lip_clf": "G1-LIP-ref-play",
    "mlip_clf": "G1-MLIP-ref-play",
    "stepping_stone": "G1-steppingstone-play",
    "stepping_stone_distillation": "G1-steppingstone-distillation-play",
    "stepping_stone_finetune": "G1-steppingstone-finetune-play",
    "stepping_stone_noheightmap": "G1-steppingstone-testing-no-heightmap",
    "stepping_stone_noheightmapdistill": "G1-steppingstone-testing-no-heightmapdistill",
    "stepping_stone_noheightmapfinetune": "G1-steppingstone-testing-no-heightmapfinetune",
    "stepping_stone_baseline": "G1-steppingstone-baseline-play",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Play trained RL policies for different environments.")
    parser.add_argument(
        "--env_type",
        type=str,
        choices=list(ENVIRONMENTS.keys()),
        help="Type of environment to play (vanilla/custom/clf)",
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Override the default experiment name for the environment type"
    )
    parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )

    parser.add_argument("--play_log_dir", type=str, default=None, help="export directory ")

    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


def find_latest_checkpoint(log_root_path):
    """Find the latest checkpoint in the given directory."""
    # Find all run directories
    run_dirs = glob.glob(os.path.join(log_root_path, "*"))
    if not run_dirs:
        return None, None

    # Get the latest run directory
    latest_run = max(run_dirs, key=os.path.getmtime)

    # Find all checkpoint files in the latest run
    checkpoint_files = glob.glob(os.path.join(latest_run, "model_*.pt"))
    if not checkpoint_files:
        return None, None

    # Get the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    checkpoint_num = int(os.path.basename(latest_checkpoint).split("_")[1].split(".")[0])
    run_name = os.path.basename(latest_run)

    return run_name, checkpoint_num


def collect_data(terrain_difficulty: float, policy_str: str):
    args_cli, hydra_args = parse_args()

    if not args_cli.env_type:
        print("Please specify an environment type using --env_type")
        print("Available options:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    print("[DEBUG] Starting main function")
    # Set the task based on environment type
    args_cli.task = SIM_ENVIRONMENTS[args_cli.env_type]
    print(f"[DEBUG] Using task: {args_cli.task}")

    # Get experiment name (use override if provided, otherwise use default)
    experiment_name = args_cli.exp_name or EXPERIMENT_NAMES[args_cli.env_type]
    print(f"[DEBUG] Using experiment name: {experiment_name}")

    # clear out sys.argv for Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    print("[DEBUG] Launching Omniverse app")
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import necessary modules after app launch
    import gymnasium as gym
    import torch
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab_rl.rsl_rl import (
        RslRlBaseRunnerCfg,
        RslRlVecEnvWrapper,
    )
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    import robot_rl.tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config


    from rsl_rl.runners import OnPolicyRunner,DistillationRunner
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.stones_output_cmd import YIdx
    
    FOOT_IDX = [YIdx.swing_foot_x, YIdx.swing_foot_y, YIdx.swing_foot_z,
                    YIdx.swing_foot_roll, YIdx.swing_foot_pitch, YIdx.swing_foot_yaw]
    COM_IDX = [YIdx.comx, YIdx.comy, YIdx.comz,
                YIdx.pelvis_roll, YIdx.pelvis_pitch, YIdx.pelvis_yaw]

    print("[DEBUG] Modules imported successfully")

    # Configure PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    print("[DEBUG] Parsing configurations")
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print("[DEBUG] Setting the difficulty range to {terrain_difficulty}")
    env_cfg.scene.terrain.terrain_generator.difficulty_range = (terrain_difficulty, terrain_difficulty)
    # print("[DEBUG] Removing curriculum")
    # env_cfg.scene.terrain.terrain_generator.curriculum = False
    # env_cfg.curriculum = None
    print("[DEBUG] Setting terrain to only 'stones'")
    env_cfg.scene.terrain.terrain_generator.sub_terrains["upstairs"].proportion = 0.
    env_cfg.scene.terrain.terrain_generator.sub_terrains["downstairs"].proportion = 0.
    env_cfg.scene.terrain.terrain_generator.sub_terrains["flat_stones"].proportion = 0.
    env_cfg.scene.terrain.terrain_generator.sub_terrains["stones"].proportion = 1.
    env_cfg.scene.terrain.terrain_generator.border_width = 0.0
    env_cfg.scene.terrain.terrain_generator.num_cols = np.ceil(np.sqrt(env_cfg.scene.num_envs)).astype(int)*8
    # env_cfg.scene.terrain.terrain_generator.num_cols = env_cfg.scene.num_envs
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    
    agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    print("[DEBUG] Configurations parsed")

    # specify directory for logging experiments
    base_log_path = os.path.join("logs", "g1_policies", EXPERIMENT_NAMES[args_cli.env_type])
    log_root_path = os.path.join(base_log_path, EXPERIMENT_NAMES[args_cli.env_type])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[DEBUG] Log root path: {log_root_path}")
    
    # create isaac environment
    if hasattr(env_cfg, "__prepare_tensors__") and callable(getattr(env_cfg, "__prepare_tensors__")):
        env_cfg.__prepare_tensors__()
        
        
    ##
    # Environment has been created successfully. Now, loop through the policies.
    ##
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    
    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # If no checkpoint is specified, find the latest one
    agent_cfg.load_run = policy_str

    # Get checkpoint path from the training directory
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[DEBUG] Checkpoint path: {resume_path}")

    # Use the checkpoint directory for saving results
    if not args_cli.play_log_dir:
        play_log_dir = os.path.dirname(resume_path)
    else:
        play_log_dir = args_cli.play_log_dir

    print(f"[DEBUG] Play log directory: {play_log_dir}")

    print(f"[DEBUG] Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    ref_gen = env.unwrapped.command_manager.get_term("hlip_ref")
    term_mng = env.unwrapped.termination_manager
    # reset environment
    # env.reset()
    obs = env.get_observations()
    timestep = 0
    print("[DEBUG] Starting simulation loop")
    
    # Variables to track
    N = env.num_envs
    total_foot_des = []
    total_foot_act = []
    total_com_des = []
    total_com_act = []
    total_done = []
    success = torch.zeros((N,), dtype=torch.bool, device=env.device)
    envs_finished = torch.zeros((N,), dtype=torch.bool, device=env.device)
    
    # simulate environment
    while simulation_app.is_running() and not torch.all(envs_finished):
        if timestep % 100 == 0:
            print(f"[DEBUG] Timestep: {timestep}, Proportion done: {torch.mean(envs_finished.float()).item():.2f}")
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rew, done, extras = env.step(actions)
            
            foot_des = ref_gen.y_out[:, FOOT_IDX]
            foot_des[envs_finished] = torch.nan
            foot_act = ref_gen.y_act[:, FOOT_IDX]
            foot_act[envs_finished] = torch.nan
            com_des = ref_gen.y_out[:, COM_IDX]
            com_des[envs_finished] = torch.nan
            com_act = ref_gen.y_act[:, COM_IDX]
            com_act[envs_finished] = torch.nan
            
            total_foot_des.append(foot_des.clone().detach().cpu().numpy())
            total_foot_act.append(foot_act.clone().detach().cpu().numpy())
            total_com_des.append(com_des.clone().detach().cpu().numpy())
            total_com_act.append(com_act.clone().detach().cpu().numpy())
            total_done.append(done.clone().detach().cpu().numpy())
            
            succ = term_mng.get_term("finished_long_stones")
            success[done.bool() & ~envs_finished] = succ[done.bool() & ~envs_finished]
            
            envs_finished |= done.bool()

        timestep += 1
    print(f"[DEBUG] Timestep: {timestep}, Proportion done: {torch.mean(envs_finished.float()).item():.2f}")
    # env.reset()
    # close the simulator
    env.close()
    
    from scipy.io import savemat
    os.makedirs("logs/paper_results", exist_ok=True)
    savemat(
        f"logs/paper_results/{policy_name}_d{int(100 * difficulty)}.mat",
        {
            "foot_des": np.stack(total_foot_des, axis=1),
            "foot_act": np.stack(total_foot_act, axis=1),
            "com_des": np.stack(total_com_des, axis=1),
            "com_act": np.stack(total_com_act, axis=1),
            "done": np.stack(total_done, axis=1),
            "success": success.cpu().numpy()
        }
    )

    # Ensure simulation app is closed
    if simulation_app is not None:
        simulation_app.close()
        print("[DEBUG] Simulation app closed")

if __name__ == "__main__":
    difficulty = 1.0
    policy_name = "hardware"; policy_str = "2025-12-03_10-16-29"
    # policy_name = "fixedDZ"; policy_str = "2025-12-07_14-30-23"
    # policy_name = "fixedT"; policy_str = "2025-12-06_15-29-36"
    
    # policy_name = "baseline"; policy_str = "2025-12-08_12-15-47"
    collect_data(difficulty, policy_str)
