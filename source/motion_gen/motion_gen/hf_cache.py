# Copyright 2026 Zachary Olkin. All rights reserved.

"""Resolve a reference-motion library path, fetching from HuggingFace into ``./hf`` if needed.

Lifted out of the trajectory-manager stack so the ingest and metadata readers can share one
cache convention without pulling in the managers.

The cache is keyed only by path, and a cached folder is never re-checked against the remote --
so two machines can end up holding *different* libraries at the same path. That is what
``library_fingerprint`` / ``library_metadata.read_library_metadata(verify_fingerprint=...)`` and
``clips.clip_fingerprint.fingerprint_solved_library`` exist to catch; use them rather than
trusting the path.

The cache lives at ``<repo>/hf``, anchored on THIS package's location rather than on
``os.getcwd()``. It used to follow the cwd, which silently built a second full copy whenever
anything ran from a subdirectory -- measured: one call from ``source/motion_gen`` downloaded all
308 files of ``library_v2`` again into ``source/motion_gen/hf`` (103 MB), and then reported a
different fingerprint than the same call from the repo root would have. A cache that moves with
the cwd is a cache that cannot be reasoned about.
"""

from __future__ import annotations

import os
import pathlib
from pathlib import Path

# <repo>/source/motion_gen/motion_gen/hf_cache.py -> parents[3] == <repo>. Derived from the package
# location, not counted from the cwd; see motion_gen.features._REPO_ROOT for the same idiom and
# CLAUDE.md for why a hop count from __file__ is the lesser evil against a cwd-relative path.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def resolve_library_path(source: str | Path, hf_repo: str | None = None) -> Path:
    """Return a local path for ``source``, downloading from ``hf_repo`` when given.

    Args:
        source: A local library root, or the in-repo path when ``hf_repo`` is set.
        hf_repo: Optional HuggingFace repo id (e.g. ``"zolkin/robot_rl"``).

    Returns:
        Local ``Path`` to the library root.
    """
    if hf_repo is None:
        return Path(source)
    return download_from_hf(hf_repo, str(source))


def download_from_hf(hf_repo: str, hf_path: str) -> Path:
    """Download a folder from HuggingFace into the repo-local ``hf/`` cache.

    Args:
        hf_repo: HuggingFace repo ID (e.g. ``"zolkin/robot_rl"``).
        hf_path: Path within the repo to download.

    Returns:
        Local ``Path`` to the downloaded folder.
    """
    cache_dir = str(_REPO_ROOT / "hf")
    os.makedirs(cache_dir, exist_ok=True)
    local_folder = os.path.join(cache_dir, hf_path)

    # Already cached? A YAML library has *.yaml; an npz library (clean / terrain, e.g.
    # library_v2) has a manifest.csv instead -- accept either.
    if os.path.isdir(local_folder):
        has_content = (
            list(Path(local_folder).glob("*.yaml"))
            + list(Path(local_folder).glob("*.yml"))
            + list(Path(local_folder).glob("**/*.yaml"))
            + list(Path(local_folder).glob("manifest.csv"))
        )
        if has_content:
            print(f"Using cached trajectories from {local_folder}")
            return Path(local_folder)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for downloading trajectories. "
            "Install with: pip install huggingface_hub"
        )

    print(f"Downloading {hf_path} from {hf_repo}...")
    try:
        # fnmatch's ``*`` does NOT cross ``/``, so one glob per directory depth. npz terrain
        # libraries (e.g. library_v2) nest clips several levels deep
        # (skill/param/variant/solved.npz), so match through 4 levels, not just 2.
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=[
                f"{hf_path}/*",          # flat files (manifest.csv)
                f"{hf_path}/*/*",        # skill/clip.npz (flat locomotion)
                f"{hf_path}/*/*/*",      # skill/param/clip.npz
                f"{hf_path}/*/*/*/*",    # skill/param/variant/solved.npz (retarget grids)
            ],
            local_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download from HuggingFace: {e}")

    print(f"Successfully downloaded to {local_folder}")
    return Path(local_folder)
