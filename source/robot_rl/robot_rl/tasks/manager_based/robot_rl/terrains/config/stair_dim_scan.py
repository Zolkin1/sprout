# Copyright 2026 Zachary Olkin. All rights reserved.

"""Scan a trajectory library for the stair step dimensions it was built with.

The gait library conditions its stair trajectories on the stair geometry —
each YAML carries ``conditioner.terrain_height`` (step height) and
``conditioner.terrain_length`` (step depth).  The terrain generator must
build stairs with exactly those dimensions so the per-env stair-size
trajectory filter always has a matching trajectory.  This module extracts
the unique (height, depth) grid from the library so the terrain cfg can be
derived from the data instead of hardcoded.

Kept dependency-light (no torch / IsaacLab imports) so terrain cfgs can call
it at construction time without dragging in the runtime stack.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_HEIGHT_RE = re.compile(r"^\s*terrain_height:\s*([-+0-9.eE]+)\s*$", re.MULTILINE)
_LENGTH_RE = re.compile(r"^\s*terrain_length:\s*([-+0-9.eE]+)\s*$", re.MULTILINE)

# Conditioner blocks sit at the top of the trajectory YAMLs (keys are dumped
# alphabetically); reading the head of the file is enough and avoids pulling
# ~120 KB of spline data per file.
_HEAD_BYTES = 4096


def _resolve_skill_folder(traj_path: str, hf_repo: str | None, skill_folder: str) -> Path:
    """Resolve the on-disk stair skill folder for a trajectory library.

    Mirrors ``MultiSkillManager`` path resolution: with ``hf_repo`` set the
    library lives in the ``<cwd>/hf/<traj_path>`` cache (downloaded on
    demand); without it ``traj_path`` is used as-is.

    Args:
        traj_path: Library root (HF-relative or local).
        hf_repo: Optional HuggingFace repo ID for download fallback.
        skill_folder: Name of the stair skill subfolder.

    Returns:
        Path to the skill folder (existing).

    Raises:
        FileNotFoundError: The folder doesn't exist locally and couldn't be
            downloaded.
    """
    if hf_repo is None:
        folder = Path(traj_path) / skill_folder
    else:
        cache_dir = Path(os.path.abspath(os.getcwd())) / "hf"
        folder = cache_dir / traj_path / skill_folder
        if not folder.is_dir():
            try:
                from huggingface_hub import snapshot_download

                print(f"Downloading {traj_path}/{skill_folder} from {hf_repo}...")
                snapshot_download(
                    repo_id=hf_repo,
                    allow_patterns=[f"{traj_path}/{skill_folder}/*"],
                    local_dir=str(cache_dir),
                )
            except Exception as e:  # noqa: BLE001 — re-raised with context below
                raise FileNotFoundError(
                    f"Stair trajectory folder not found at '{folder}' and the "
                    f"HuggingFace download from '{hf_repo}' failed ({e}). The "
                    "terrain stair sizes are derived from the library — "
                    "merge/upload the library (or place it locally under hf/) "
                    "before constructing this env."
                ) from e

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Stair trajectory folder not found: '{folder}'. The terrain "
            "stair sizes are derived from the library — merge/upload the "
            "library (or place it locally) before constructing this env."
        )
    return folder


def scan_stair_step_dims(
    traj_path: str,
    hf_repo: str | None = None,
    skill_folder: str = "stair_up",
    round_decimals: int = 4,
) -> list[tuple[float, float]]:
    """Extract the unique stair (step_height, step_depth) pairs from a library.

    Scans every trajectory YAML in ``<library>/<skill_folder>`` for its
    ``conditioner.terrain_height`` / ``conditioner.terrain_length`` values
    (step height and step depth in meters) and returns the deduplicated grid.

    Args:
        traj_path: Library root, e.g. ``"trajectories/retargeted/<merged>"``.
            Resolved against the ``hf/`` cache when ``hf_repo`` is set,
            otherwise used as a local path.
        hf_repo: Optional HuggingFace repo ID; enables download when the
            folder isn't cached locally.
        skill_folder: Stair skill subfolder name.
        round_decimals: Decimal places used when deduplicating, to absorb
            float noise in the YAML values.

    Returns:
        Sorted list of unique ``(step_height, step_depth)`` tuples — the
        format ``step_dim_options`` expects on the stair terrain cfgs.

    Raises:
        FileNotFoundError: Skill folder missing (and not downloadable).
        ValueError: A YAML lacks the conditioner terrain keys, or no
            trajectory files were found.
    """
    folder = _resolve_skill_folder(traj_path, hf_repo, skill_folder)

    yaml_files = sorted(folder.glob("*.yaml")) + sorted(folder.glob("*.yml"))
    if len(yaml_files) == 0:
        raise ValueError(f"No trajectory YAML files in: {folder}")

    dims: set[tuple[float, float]] = set()
    for yf in yaml_files:
        with open(yf, "r") as f:
            head = f.read(_HEAD_BYTES)
        h_match = _HEIGHT_RE.search(head)
        l_match = _LENGTH_RE.search(head)
        if h_match is None or l_match is None:
            raise ValueError(
                f"Trajectory '{yf}' has no conditioner terrain_height/"
                "terrain_length — stair trajectories must declare the stair "
                "geometry they were optimized for."
            )
        dims.add((
            round(float(h_match.group(1)), round_decimals),
            round(float(l_match.group(1)), round_decimals),
        ))

    if len(dims) == 0:
        raise ValueError(f"No stair dimensions found in: {folder}")
    return sorted(dims)
