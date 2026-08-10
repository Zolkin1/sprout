# Copyright 2026 Zachary Olkin. All rights reserved.

"""Upload a deploy ONNX pair (``tracker.onnx`` + ``fm.onnx``) to HuggingFace for the robot.

The robot pulls these two files and nothing else: every deployment parameter -- joint
order, gains, observation terms and their clips, the depth-camera contract -- is embedded
in their metadata, so there is no sidecar YAML to keep in sync (and no ``policy.pt``; the
hardware runtime is torch-free).  ``transfer/deploy/hardware.load_gen_tracker_from_hf``
is the consumer, and it expects exactly this ``<folder>/{tracker.onnx,fm.onnx}`` layout.

Produce the artifacts first with ``play.py`` (which writes ``exported/tracker.onnx``,
and ``exported/fm.onnx`` when given ``--fm_checkpoint``), then::

    python scripts/rsl_rl/export_to_hardware.py \\
        --run_dir logs/rsl_rl/g1/<run> --hf_repo_id user/repo

The repo folder defaults to the run directory's own name, so a deployed policy names the
run that produced it; pass --folder_name only to override that deliberately.

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
from deploy.hardware import DEFAULT_HF_REPO_ID  # noqa: E402

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
            f"{exported_dir} is missing {missing}. Run play.py --fm_checkpoint <ckpt> "
            "first; it writes both files into <run>/exported/."
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
        extero = f"height scan ({policy.ctx.extero_sensor_dim} rays)"
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
    parser.add_argument("--hf_repo_id", default=DEFAULT_HF_REPO_ID,
                        help=f"HuggingFace repository ID (e.g. 'username/repo-name'). "
                             f"Defaults to {DEFAULT_HF_REPO_ID}, the same constant the "
                             f"on-robot prefetch reads, so upload and download cannot "
                             f"drift apart.")
    parser.add_argument("--folder_name", default=None,
                        help="Folder within the repo to upload the two ONNX files to. "
                             "Defaults to the RUN DIRECTORY's own name (timestamp + run "
                             "name), so the repo folder and the local run are the same "
                             "string and a deployed policy can be traced back to the run "
                             "that produced it without a lookup table. Override only to "
                             "publish under a different name deliberately.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate the pair and report, but upload nothing.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir.rstrip("/"))
    folder_name = args.folder_name or os.path.basename(run_dir)
    exported = os.path.join(run_dir, "exported")
    print(f"[INFO] Validating {exported}")
    policy = validate_pair(exported)
    print(f"[INFO] Artifact pair is self-consistent:\n{describe(policy)}")

    if args.dry_run:
        # Name the destination even when uploading nothing: the folder is now DERIVED from
        # the run dir unless overridden, so a dry run has to show what it resolved to.
        print(f"[INFO] --dry_run: nothing uploaded. Would upload to "
              f"{args.hf_repo_id}/{folder_name}/{{tracker.onnx,fm.onnx}}"
              f"{'' if args.folder_name else '  (folder derived from --run_dir)'}")
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
        repo_path = f"{folder_name}/{name}"
        api.upload_file(
            path_or_fileobj=os.path.join(exported, name),
            path_in_repo=repo_path,
            repo_id=args.hf_repo_id,
            commit_message=f"Upload {name} to {folder_name}",
        )
        print(f"[INFO] Uploaded {repo_path}")

    print(f"[INFO] Done: https://huggingface.co/{args.hf_repo_id}/tree/main/{folder_name}")
    print(f"[INFO] On the robot, point the controller at folder '{folder_name}'.")


if __name__ == "__main__":
    main()
