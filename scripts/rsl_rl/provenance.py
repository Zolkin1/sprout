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
