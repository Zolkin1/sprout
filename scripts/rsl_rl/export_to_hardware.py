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

from deploy.artifacts import DEPLOY_FM_NAME, resolve_fm_onnx  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.hardware import DEFAULT_HF_REPO_ID, policy_repo_dir  # noqa: E402

#: What the robot downloads. Both are required: the tracker alone cannot walk.
#:
#: The generator is uploaded UNDER THIS NAME whatever it is called locally. Local exports
#: are named for their FM checkpoint so several can sit in one ``exported/`` without
#: overwriting each other (``deploy.artifacts``); a robot installs exactly one pair and its
#: controller reads a fixed filename, so the rename happens here, at the boundary.
ARTIFACTS = ("tracker.onnx", DEPLOY_FM_NAME)


def validate_pair(exported_dir: str, fm_onnx: str | None = None,
                  allow_robot_asset_mismatch: bool = False) -> GenTrackerPolicy:
    """Load the artifact pair and return the policy, so a bad pair never ships.

    Args:
        exported_dir: Directory holding ``tracker.onnx`` and the generator.
        fm_onnx: Which generator to ship, when the directory holds several.
        allow_robot_asset_mismatch: Upload a pair whose ``robot_asset`` fingerprints
            disagree. Off by default. This is the UPLOAD path, so the bar is the same as
            the controller's: someone has read the printed diff and knows why it is
            harmless. It exists because ``robot_asset`` is baked into an FM checkpoint at
            TRAINING time and is not recomputed at export, so a change to the robot
            description made after a generator was trained cannot be cleared by
            re-exporting it -- and where that change is confined to fields neither model
            reads (actuator gains, say), the pair is compatible though the hashes are not.
            The controller has the matching ``allow_robot_asset_mismatch`` parameter, and
            BOTH must be set: this one only gets the pair uploaded.

    Returns:
        The loaded policy, for reporting what is about to be uploaded.

    Raises:
        FileNotFoundError: If either artifact is missing.
        ValueError: Propagated from the load when the two disagree about the depth
            cameras, or the generator conditions on exteroception the contract lacks.
    """
    tracker = os.path.join(exported_dir, "tracker.onnx")
    if not os.path.exists(tracker):
        raise FileNotFoundError(
            f"{tracker} is missing. Run play.py --fm_checkpoint <ckpt> first; it writes "
            "the tracker and the generator into <run>/exported/."
        )
    policy = GenTrackerPolicy(tracker, resolve_fm_onnx(exported_dir, fm_onnx),
                              allow_robot_asset_mismatch=allow_robot_asset_mismatch)
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
    parser.add_argument("--fm_onnx", default=None,
                        help="Which generator to ship: a path, or a bare filename inside "
                             "the run's exported/. Local exports are named for their FM "
                             "checkpoint, so this is REQUIRED when the directory holds "
                             "several. It is uploaded as 'fm.onnx' whatever it is called "
                             "here -- the robot installs one pair under a fixed name.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate the pair and report, but upload nothing.")
    parser.add_argument("--allow_robot_asset_mismatch", action="store_true",
                        help="Upload a pair whose robot_asset fingerprints disagree, warning "
                             "instead of refusing. The guard exists because a tracker and a "
                             "generator describing different robots is invisible to every "
                             "shape check, so read the printed diff before reaching for this. "
                             "It only gets the pair UPLOADED -- the controller has its own "
                             "allow_robot_asset_mismatch parameter and both must be set.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir.rstrip("/"))
    folder_name = args.folder_name or os.path.basename(run_dir)
    # The `policies/` prefix is applied HERE and never typed by a caller: --folder_name and
    # the controller's hf_policy_folder both name a policy by its run name alone. Imported
    # from deploy.hardware rather than spelled again, so the uploader and the on-robot fetch
    # cannot disagree about where a policy lives -- the failure that would otherwise show up
    # as a successful upload the robot cannot find.
    repo_dir = policy_repo_dir(folder_name)
    exported = os.path.join(run_dir, "exported")
    print(f"[INFO] Validating {exported}")
    policy = validate_pair(exported, args.fm_onnx, args.allow_robot_asset_mismatch)
    print(f"[INFO] Artifact pair is self-consistent:\n{describe(policy)}")

    if args.dry_run:
        # Name the destination even when uploading nothing: the folder is now DERIVED from
        # the run dir unless overridden, so a dry run has to show what it resolved to.
        print(f"[INFO] --dry_run: nothing uploaded. Would upload to "
              f"{args.hf_repo_id}/{repo_dir}/{{tracker.onnx,fm.onnx}}"
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
    # (local path, name on the robot). The generator is renamed on the way up; see
    # ARTIFACTS for why the fixed name lives only at this boundary. Local exports are named
    # for their FM checkpoint, so the source has to be RESOLVED rather than joined by name.
    uploads = ((os.path.join(exported, "tracker.onnx"), "tracker.onnx"),
               (resolve_fm_onnx(exported, args.fm_onnx), DEPLOY_FM_NAME))
    for local, name in uploads:
        # repo_dir, not folder_name: this branch puts the pair under the `policies/` prefix.
        repo_path = f"{repo_dir}/{name}"
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=repo_path,
            repo_id=args.hf_repo_id,
            commit_message=f"Upload {name} to {repo_dir}",
        )
        print(f"[INFO] Uploaded {repo_path}")

    print(f"[INFO] Done: https://huggingface.co/{args.hf_repo_id}/tree/main/{repo_dir}")
    print(f"[INFO] On the robot, point the controller at folder '{folder_name}' "
          f"(bare, with no '{repo_dir.rsplit('/', 1)[0]}/' prefix -- the runtime adds it).")


if __name__ == "__main__":
    main()
