# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared run-provenance helpers: git snapshots and params dumps.

Used by both the (Isaac-side) recording script and the (Isaac-free) flow-matching
trainer so every run records exactly how it was produced -- matching the git
snapshot + ``params/*.yaml`` that RL training already saves. Intentionally
dependency-light (stdlib + PyYAML) so the Isaac-free trainer can import it without
pulling in Isaac Sim.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

# Re-export so callers have a single provenance import surface.
from recorded_data_io import sha256_file  # noqa: F401


def git_snapshot(out_dir: str, repo_root: str | None = None) -> bool:
    """Write a git snapshot (commit SHA + status + diff) to ``<out_dir>/git/<repo>.diff``.

    Mirrors rsl_rl's code-state capture so recording / flow-matching runs carry the
    same reproducibility info as RL training. Best-effort: any failure (not a git
    checkout, git missing) is reported and skipped, never fatal.

    Args:
        out_dir: Run directory; the snapshot is written under ``<out_dir>/git/``.
        repo_root: Repo to snapshot (default: current working directory).

    Returns:
        True if a snapshot was written, False otherwise.
    """
    repo_root = os.path.abspath(repo_root or os.getcwd())
    try:
        def _git(*a: str) -> str:
            return subprocess.run(
                ["git", "-C", repo_root, *a], capture_output=True, text=True, check=True
            ).stdout.rstrip()

        commit = _git("rev-parse", "HEAD")
        status = _git("status")
        diff = _git("diff", "HEAD")
    except Exception as exc:  # not a git repo, git missing, detached worktree, etc.
        print(f"[WARN] git_snapshot skipped: {exc}")
        return False

    name = os.path.basename(repo_root) or "repo"
    git_dir = os.path.join(out_dir, "git")
    os.makedirs(git_dir, exist_ok=True)
    with open(os.path.join(git_dir, f"{name}.diff"), "w") as f:
        f.write(
            f"--- git commit ---\n{commit}\n\n"
            f"--- git status ---\n{status}\n\n"
            f"--- git diff ---\n{diff}\n"
        )
    return True


def auto_clip_device(env_cfg, explicit: str | None, size_threshold_gb: float = 1.0) -> None:
    """Set ``clip_device`` on any command term that exposes it.

    Uses ``explicit`` when given; otherwise defaults *large* clip libraries
    (> ``size_threshold_gb``) to ``'cpu'`` -- they OOM a small GPU alongside the sim
    + renderer, and the per-env gather from host RAM is cheap. Small libraries are
    left on the env device. Mutates ``env_cfg.commands.*`` in place before env build.

    Args:
        env_cfg: The (configclass) env config; its ``commands`` are scanned.
        explicit: The user's ``--clip_device`` value, or ``None`` to auto-decide.
        size_threshold_gb: Library-size cutoff above which CPU is chosen.
    """
    for name, term in vars(env_cfg.commands).items():
        if not hasattr(term, "clip_device"):
            continue
        if explicit is not None:
            term.clip_device = explicit
            print(f"[INFO] clip_device='{explicit}' for commands.{name}.")
            continue
        path = getattr(term, "path", "") or ""
        size_gb = os.path.getsize(path) / 1e9 if os.path.isfile(path) else 0.0
        if size_gb > size_threshold_gb:
            term.clip_device = "cpu"
            print(f"[auto] clip_device='cpu' for commands.{name} (library {size_gb:.1f} GB).")


def dump_params_yaml(path: str, obj: Any) -> None:
    """Dump a plain dict/list to YAML at ``path`` (creating parent dirs).

    Uses PyYAML directly (not ``isaaclab.utils.io``) so the Isaac-free flow-matching
    trainer can call it without importing Isaac Sim.

    Args:
        path: Destination ``.yaml`` path.
        obj: A YAML-serializable object (dict/list of plain scalars).
    """
    import yaml  # noqa: PLC0415

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, default_flow_style=False, sort_keys=False)


#: Filename, under a run's ``params/``, recording the CODE-level conventions its checkpoints
#: are only meaningful against. Separate from ``env.yaml`` because these are constants in the
#: source, not configclass fields -- ``env.yaml`` is byte-identical across a convention change.
RUN_CONVENTIONS_FILE = "conventions.yaml"


def run_conventions(env_cfg: Any = None) -> dict:
    """The conventions a trained checkpoint is only valid under.

    ``base_pose_convention`` is the load-bearing one: it names the frame and direction of the
    reference window's ``base_pose`` block. Changing it alters what the observation MEANS
    without changing its width, so a stale policy loads clean and steers wrong -- which is
    exactly the failure this file exists to make loud.

    The re-plan cadences are recorded alongside it because the generator's history is a slice
    of one plan's tail, so they are part of what a checkpoint assumes.

    Args:
        env_cfg: The environment config, if available; its ``commands.traj_ref`` cadences are
            read when present. ``None`` records the code constants alone.

    Returns:
        A YAML-serializable dict.
    """
    from robot_rl.flow_matching.window import (  # noqa: PLC0415
        DEFAULT_WINDOW_ANCHOR, base_pose_convention)

    traj = getattr(getattr(env_cfg, "commands", None), "traj_ref", None)
    # Read the convention off the RUN's config, not off a module constant: two arms of the
    # same branch now train under different ones simultaneously, so a global would stamp
    # whichever the code happened to default to and hand the other arm a wrong record.
    # A cfg with no ``window_anchor`` -- every run trained before the field existed -- falls
    # back to the default, which reproduces the original string exactly.
    anchor = getattr(traj, "window_anchor", None) or DEFAULT_WINDOW_ANCHOR
    out: dict = {"base_pose_convention": base_pose_convention(anchor),
                 "window_anchor": anchor}
    for field in ("fm_snap_every", "ego_anchor_every", "fm_update_every"):
        value = getattr(traj, field, None)
        if value is not None:
            out[field] = int(value)
    return out


def dump_run_conventions(log_dir: str, env_cfg: Any = None) -> dict:
    """Write :func:`run_conventions` into ``<log_dir>/params/``. Returns what was written."""
    conv = run_conventions(env_cfg)
    dump_params_yaml(os.path.join(log_dir, "params", RUN_CONVENTIONS_FILE), conv)
    return conv


def check_run_conventions(log_dir: str, env_cfg: Any = None) -> None:
    """Refuse to load a run whose recorded conventions disagree with this code.

    HARD refusal, including when the file is absent: a run written before this existed was
    trained under an older convention, and "no record" is not evidence of agreement. The
    alternative -- defaulting to the current value when the key is missing -- is what let a
    stale artifact through on the deploy path until it was tightened, and it is safe only
    while exactly one convention has ever existed.

    Cadence differences are REPORTED, not refused: overriding the re-plan period at play time
    is a legitimate experiment, and it changes behaviour visibly rather than silently.

    Args:
        log_dir: The run directory being loaded (the one holding ``params/``).
        env_cfg: The environment config about to be built, for the cadence comparison.

    Raises:
        ValueError: If the record is missing, unreadable, or names a different
            ``base_pose_convention``.
    """
    import yaml  # noqa: PLC0415

    path = os.path.join(log_dir, "params", RUN_CONVENTIONS_FILE)
    want = run_conventions(env_cfg)
    if not os.path.isfile(path):
        raise ValueError(
            f"{path} is missing, so this run does not record which observation conventions it "
            f"was trained under. It predates the record, which means it was trained under an "
            f"older convention -- not the current "
            f"{want['base_pose_convention']!r}. The reference window's base_pose block has the "
            "same WIDTH under every convention, so loading it anyway would run a policy whose "
            "observation means something different, with no visible symptom.\n"
            "  * To play a pre-convention run, check out the commit it was trained at "
            "(git tag pre-yaw-frame is the last full-rotation one).\n"
            "  * To adopt it deliberately, write the file yourself with the convention that "
            "run actually used."
        )
    try:
        got = yaml.safe_load(open(path)) or {}
    except Exception as exc:  # noqa: BLE001 - a corrupt record is a refusal, not a warning
        raise ValueError(f"{path} could not be read ({exc}); refusing to guess the convention.")

    if got.get("base_pose_convention") != want["base_pose_convention"]:
        raise ValueError(
            f"This run was trained with base_pose_convention "
            f"{got.get('base_pose_convention')!r}, but this code implements "
            f"{want['base_pose_convention']!r}. The observation has the same width under both, "
            "so the policy would load clean and steer wrong. Check out the commit the run was "
            "trained at (git tag pre-yaw-frame is the last full-rotation one), or retrain."
        )
    drift = {k: (got.get(k), v) for k, v in want.items()
             if k not in ("base_pose_convention", "window_anchor")
             and got.get(k) is not None and got.get(k) != v}
    if drift:
        print("[WARN] re-plan cadences differ from the trained run (trained -> now): "
              + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in drift.items())
              + ". Behaviour will differ; this is allowed as a deliberate override.")


def policy_far_camera_in_run(log_dir: str) -> bool | None:
    """Did the POLICY of the run at ``log_dir`` read the far depth camera?

    The policy reads the near camera alone as of ``efe93b4e``, so a near-only task builds 2222
    inputs where every earlier tracker expects 3002 -- measured on the actual checkpoints, and
    the difference is exactly 780, one 30x26 camera. A run trained before that change cannot be
    loaded at all without putting the term back, and the far block is the WHOLE difference: the
    ``height_scan`` those runs also list under ``observations.policy`` does not reach the
    vector, which the arithmetic settles (3002 - 2222 = 780, not 780 + a scan).

    Read off the run's OWN ``params/env.yaml`` rather than taken as a flag, deliberately: a
    flag added to the eval's parser is not forwarded to the cell subprocess unless
    ``stage_grid`` restates it, and reverts to its argparse default in the child SILENTLY --
    measured twice on this harness. A property of the artifact cannot be dropped that way.

    Args:
        log_dir: The run directory holding ``params/env.yaml``.

    Returns:
        ``True`` / ``False`` when the record exists, ``None`` when it does not -- an old run
        with no ``params/`` is unknowable here and the caller decides what that means.
    """
    import yaml  # noqa: PLC0415

    path = os.path.join(log_dir, "params", "env.yaml")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        cfg = yaml.unsafe_load(fh)
    policy = ((cfg or {}).get("observations") or {}).get("policy") or {}
    if "depth_far" not in policy:
        return None
    return policy["depth_far"] is not None
