# Copyright 2026 Zachary Olkin. All rights reserved.

"""Upload a deploy ONNX pair (``tracker.onnx`` + ``fm.onnx``) to HuggingFace for the robot.

The robot pulls these two files and nothing else: every deployment parameter -- joint
order, gains, observation terms and their clips, the depth-camera contract -- is embedded
in their metadata, so there is no sidecar YAML to keep in sync (and no ``policy.pt``; the
hardware runtime is torch-free).  ``transfer/deploy/hardware.load_gen_tracker_from_hf``
is the consumer, and it expects exactly this ``<folder>/{tracker.onnx,fm.onnx}`` layout.

Produce the artifacts first with ``export_deploy_onnx.py`` (which itself wants a
``play.py``-exported ``policy.pt``), then::

    python scripts/rsl_rl/export_to_hardware.py \\
        --run_dir logs/rsl_rl/g1/<run> --hf_repo_id user/repo --folder_name <name>

The pair is LOADED before anything is uploaded, which is the point of doing this here
rather than with ``huggingface-cli upload``: a tracker and a generator that disagree
about the depth cameras, or a generator whose exteroception the tracker cannot supply,
are rejected on the dev box instead of on the robot.  Isaac-free and torch-free.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "transfer"))

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402

#: What the robot downloads. Both are required: the tracker alone cannot walk.
ARTIFACTS = ("tracker.onnx", "fm.onnx")


def validate_pair(exported_dir: str) -> GenTrackerPolicy:
    """Load the artifact pair and return the policy, so a bad pair never ships.

    Args:
        exported_dir: Directory holding ``tracker.onnx`` and ``fm.onnx``.

    Returns:
        The loaded policy, for reporting what is about to be uploaded.

    Raises:
        FileNotFoundError: If either artifact is missing.
        ValueError: Propagated from the load when the two disagree about the depth
            cameras, or the generator conditions on exteroception the contract lacks.
    """
    missing = [n for n in ARTIFACTS if not os.path.exists(os.path.join(exported_dir, n))]
    if missing:
        raise FileNotFoundError(
            f"{exported_dir} is missing {missing}. Run export_deploy_onnx.py first "
            "(and play.py before that, to produce exported/policy.pt)."
        )
    policy = GenTrackerPolicy(*(os.path.join(exported_dir, n) for n in ARTIFACTS))
    policy.load()
    return policy


def describe(policy: GenTrackerPolicy) -> str:
    """A one-block summary of what the robot will be running.

    Args:
        policy: The loaded policy.

    Returns:
        Human-readable lines covering the observation size, joint count, control period
        and the exteroception the deployment must supply.
    """
    spec = policy.get_depth_camera_spec()
    if spec:
        extero = f"{len(spec)} depth camera(s): " + ", ".join(
            f"{c.get('sensor') or c['obs_term']} {int(c['width'])}x{int(c['height'])} "
            f"on {c['mount_body']}" for c in spec
        )
    elif policy.needs_height_scan():
        extero = f"height scan ({policy.ctx.height_dim} rays)"
    else:
        extero = "none (blind)"
    return (
        f"  obs {policy.get_num_obs()}  actions {policy.get_num_actions()}  "
        f"dt {policy.get_dt():.4f}s\n"
        f"  exteroception: {extero}"
    )


def main() -> None:
    """Validate the ONNX pair in a run's ``exported/`` and upload it to HuggingFace."""
    parser = argparse.ArgumentParser(
        description="Upload tracker.onnx + fm.onnx to HuggingFace for hardware deployment.")
    parser.add_argument("--run_dir", required=True,
                        help="Run directory; uploads from <run_dir>/exported.")
    parser.add_argument("--hf_repo_id", required=True,
                        help="HuggingFace repository ID (e.g. 'username/repo-name').")
    parser.add_argument("--folder_name", required=True,
                        help="Folder within the repo to upload the two ONNX files to.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate the pair and report, but upload nothing.")
    args = parser.parse_args()

    exported = os.path.join(os.path.abspath(args.run_dir), "exported")
    print(f"[INFO] Validating {exported}")
    policy = validate_pair(exported)
    print(f"[INFO] Artifact pair is self-consistent:\n{describe(policy)}")

    if args.dry_run:
        print("[INFO] --dry_run: nothing uploaded.")
        return

    from huggingface_hub import HfApi, login, repo_exists, whoami

    try:
        whoami()
        print("[INFO] HF already logged in.")
    except Exception:
        print("[INFO] Not authenticated, logging in...")
        login()

    if not repo_exists(args.hf_repo_id):
        print(f"[ERROR] Repository {args.hf_repo_id} does not exist. Create it first.")
        sys.exit(1)

    api = HfApi()
    for name in ARTIFACTS:
        repo_path = f"{args.folder_name}/{name}"
        api.upload_file(
            path_or_fileobj=os.path.join(exported, name),
            path_in_repo=repo_path,
            repo_id=args.hf_repo_id,
            commit_message=f"Upload {name} to {args.folder_name}",
        )
        print(f"[INFO] Uploaded {repo_path}")

    print(f"[INFO] Done: https://huggingface.co/{args.hf_repo_id}/tree/main/{args.folder_name}")
    print(f"[INFO] On the robot, point the controller at folder '{args.folder_name}'.")


if __name__ == "__main__":
    main()
