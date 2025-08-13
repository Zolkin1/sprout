
import sys
import os
import argparse
import cli_args
from train_policy import ENVIRONMENTS, EXPERIMENT_NAMES
from play_policy import SIM_ENVIRONMENTS

from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Export the policy to be used on hardware using hugging face.")

    # Load in the hugging face repo
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="Hugging Face repository ID (e.g., 'username/repo-name'). If provided, exported policy will be uploaded."
    )

    # Name to export the policy as
    parser.add_argument(
        "--policy_name",
        type=str,
        default="policy",
        help="Name for the exported policy files (default: 'policy')."
    )

    # The environment to load
    parser.add_argument(
        "--env_type",
        type=str,
        choices=list(ENVIRONMENTS.keys()),
        help="Type of environment to play (vanilla/custom/clf)"
    )

    cli_args.add_rsl_rl_args(parser)
    args_cli = parser.parse_args()

    # Set the task based on environment type
    args_cli.task = SIM_ENVIRONMENTS[args_cli.env_type]

    # Upload to Hugging Face if repo_id provided
    if args_cli.hf_repo_id:
        from huggingface_hub import HfApi, repo_exists, login, whoami

        try:
            whoami()
            print("HF already logged in.")
        except Exception:
            print("Not authenticated, logging in...")
            login()

        print(f"[DEBUG] Uploading policy to Hugging Face repository: {args_cli.hf_repo_id}")
        api = HfApi()

        # Check if repository exists
        if not repo_exists(args_cli.hf_repo_id):
            print(f"[ERROR] Repository {args_cli.hf_repo_id} does not exist. Please create it first.")
            sys.exit(1)

        # Get the location of the exported file
        base_log_path = os.path.join("logs", "g1_policies", EXPERIMENT_NAMES[args_cli.env_type])
        log_root_path = os.path.join(base_log_path, args_cli.env_type)
        log_root_path = os.path.abspath(log_root_path)

        print("[DEBUG] Launching Omniverse app")
        # launch omniverse app
        app_launcher = AppLauncher(args_cli)

        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
        import robot_rl.tasks  # noqa: F401

        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

        checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        export_model_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")

        print(f"[INFO] Looking for the exported policy in {export_model_dir}")

        # Upload only the JIT policy file
        policy_file = os.path.join(export_model_dir, "policy.pt")

        # Upload files
        if os.path.exists(policy_file):
            # Upload policy with custom name to env_type folder
            repo_path = f"{args_cli.env_type}/{args_cli.policy_name}.pt"
            api.upload_file(
                path_or_fileobj=policy_file,
                path_in_repo=repo_path,
                repo_id=args_cli.hf_repo_id,
                commit_message=f"Upload {args_cli.policy_name}.pt from {args_cli.env_type} training"
            )
            print(f"[DEBUG] Uploaded {repo_path} to {args_cli.hf_repo_id}")

        print(f"[INFO] Successfully uploaded policy to https://huggingface.co/{args_cli.hf_repo_id}")

if __name__ == "__main__":
    main()