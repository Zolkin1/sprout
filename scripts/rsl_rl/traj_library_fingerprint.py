#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Print the portable content fingerprint of a trajectory library.

Use this to verify that two machines (e.g. the talos training server and a
laptop) hold the *same* trajectory library — the loader silently uses a local
``hf/`` cache and never re-checks HuggingFace, so caches can diverge without
any warning.  Run it on both machines and compare the printed hash::

    python3 scripts/rsl_rl/traj_library_fingerprint.py \\
        hf/trajectories/retargeted/2026-06-13_15-10-58_merged

Identical hashes => identical libraries.  Different hashes => the references
differ (the per-skill manifest shows where).

This is the manual counterpart to the automatic check baked into training /
play: new runs record the fingerprint in ``params/env.yaml`` and ``play.py``
warns on a mismatch.  This script exists for older runs whose ``env.yaml``
predates that feature.  It imports only the dependency-free fingerprint module, so it runs
under a bare ``python3`` (no IsaacLab environment needed) as long as ``source/motion_gen``
is importable -- e.g. ``PYTHONPATH=source/motion_gen python3 scripts/rsl_rl/...``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The canonical (stdlib-only) fingerprint module. It used to be loaded by file path, from
# a location under robot_rl/.../traj_tracking/, to avoid dragging torch / IsaacLab in via
# the package import chain. It moved to the IsaacLab-free ``motion_gen`` package in 6b3aae3
# and that hard-coded path was never updated, so this script had been dying on
# FileNotFoundError ever since. ``motion_gen`` imports no IsaacLab, so a plain import both
# fixes it and cannot silently rot the same way.
from motion_gen import library_fingerprint as _fp  # noqa: E402


def _resolve(path: str) -> Path:
    """Resolve a library path, falling back to the local ``hf/`` cache.

    Args:
        path: A direct filesystem path, or an HF-relative path such as
            ``trajectories/retargeted/<lib>`` resolved under ``./hf/``.

    Returns:
        The first existing candidate path.

    Raises:
        FileNotFoundError: If neither the path nor its ``hf/`` form exists.
    """
    candidates = [Path(path)]
    if not os.path.isabs(path):
        candidates.append(Path("hf") / path)
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Library not found. Tried: {', '.join(str(c) for c in candidates)}"
    )


def main() -> int:
    """Parse args, compute the fingerprint, and print it (optionally comparing)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="Library root (e.g. hf/trajectories/retargeted/<lib>), or an "
        "HF-relative path resolved under ./hf/.",
    )
    parser.add_argument(
        "--compare",
        metavar="HASH",
        default=None,
        help="A fingerprint hash to compare against; exits non-zero on mismatch.",
    )
    args = parser.parse_args()

    root = _resolve(args.path)
    fp = _fp.fingerprint_library(root)

    print(f"library : {root}")
    print(f"hash    : {fp['hash']}")
    print(f"short   : {fp['short']}")
    print(f"total   : {fp['num_trajectories']} trajectories")
    print("skills  :")
    for skill, count in fp["skill_counts"].items():
        print(f"    {skill:<16} {count}")

    if args.compare is not None:
        match = args.compare == fp["hash"]
        print(f"\ncompare : {'MATCH' if match else 'MISMATCH'} (vs {args.compare})")
        return 0 if match else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
