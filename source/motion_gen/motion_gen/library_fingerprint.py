# Copyright 2026 Zachary Olkin. All rights reserved.

"""Portable content fingerprint for trajectory libraries.

The merged-tensor cache key (``MultiSkillManager._library_cache_key``) folds in
absolute paths and file mtimes, so it is *machine-specific* — useful for local
cache invalidation, useless for comparing the library across machines. This
module computes a **portable** digest over the library's relative layout and
file *contents*, so the same set of trajectory YAMLs hashes identically on any
machine (talos, laptop, CI).

It is used to:
  * record the loaded library's fingerprint into ``params/env.yaml`` at train
    time (via the command cfg, dumped after env construction); and
  * verify at play / resume that a policy is being run against the exact
    trajectory library it trained on.

The module is intentionally dependency-free (stdlib only) so the standalone
checker ``scripts/rsl_rl/traj_library_fingerprint.py`` can import it under a
bare ``python3`` without activating the IsaacLab environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Bump if the digest algorithm below changes in a way that should invalidate
# previously recorded fingerprints (e.g. switching what bytes feed the hash).
_FINGERPRINT_VERSION = 1


def discover_skill_folders(root: str | Path) -> dict[str, Path]:
    """Map skill name -> folder for an already-resolved local library root.

    Mirrors ``MultiSkillManager._discover_skill_folders`` for the local-path
    case (no HuggingFace download): each immediate subdirectory containing YAML
    files is a skill; failing that, a root that directly holds YAMLs is the
    single ``"default"`` skill.

    Args:
        root: Resolved local path to the library's top-level folder.

    Returns:
        Dict mapping skill name to its folder ``Path``, ordered by name.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
        ValueError: If no YAML trajectory files are found anywhere under ``root``.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Library root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Library root is not a directory: {root}")

    skill_folders: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and _yaml_files(child):
            skill_folders[child.name] = child

    if not skill_folders:
        if _yaml_files(root):
            skill_folders["default"] = root
        else:
            raise ValueError(f"No skill subfolders with YAML files found under: {root}")

    return skill_folders


def fingerprint_skill_folders(skill_folders: dict[str, Path]) -> dict:
    """Compute a portable content fingerprint over a set of skill folders.

    The digest is order-independent of the host filesystem: skills are visited
    in sorted name order, files within a skill in sorted *basename* order, and
    only the relative ``"<skill>/<filename>"`` key plus the file's content hash
    feed the digest (never absolute paths or mtimes).

    Args:
        skill_folders: Mapping of skill name to folder ``Path`` (as produced by
            :func:`discover_skill_folders` or
            ``MultiSkillManager._discover_skill_folders``).

    Returns:
        Dict with:
            ``hash``: full sha256 hex digest of the library contents.
            ``short``: first 12 hex chars (for log lines).
            ``num_trajectories``: total YAML file count across all skills.
            ``skill_counts``: ``{skill_name: file_count}`` ordered by name.
    """
    h = hashlib.sha256()
    h.update(f"traj-lib-fingerprint;version={_FINGERPRINT_VERSION};".encode())

    skill_counts: dict[str, int] = {}
    for skill_name in sorted(skill_folders):
        files = sorted(_yaml_files(skill_folders[skill_name]), key=lambda p: p.name)
        skill_counts[skill_name] = len(files)
        h.update(f"skill={skill_name}:{len(files)};".encode())
        for p in files:
            h.update(f"{skill_name}/{p.name}=".encode())
            h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
            h.update(b";")

    digest = h.hexdigest()
    return {
        "hash": digest,
        "short": digest[:12],
        "num_trajectories": sum(skill_counts.values()),
        "skill_counts": skill_counts,
    }


def fingerprint_library(root: str | Path) -> dict:
    """Discover skill folders under a local ``root`` and fingerprint them.

    Args:
        root: Resolved local path to the library's top-level folder.

    Returns:
        The fingerprint dict from :func:`fingerprint_skill_folders`.
    """
    return fingerprint_skill_folders(discover_skill_folders(root))


def _yaml_files(folder: Path) -> list[Path]:
    """Return the ``*.yaml`` / ``*.yml`` files directly inside ``folder``."""
    return list(folder.glob("*.yaml")) + list(folder.glob("*.yml"))
