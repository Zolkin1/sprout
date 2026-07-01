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
predates that feature.  It imports only the dependency-free fingerprint module,
so it runs under a bare ``python3`` (no IsaacLab environment needed).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Load the canonical (stdlib-only) fingerprint module directly by file path so
# this script does not drag in torch / IsaacLab via the package import chain.
_FP_MODULE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "source/robot_rl/robot_rl/tasks/manager_based/robot_rl"
    / "mdp/commands/traj_tracking/library_fingerprint.py"
)
_spec = importlib.util.spec_from_file_location("traj_library_fingerprint_impl", _FP_MODULE_PATH)
_fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fp)


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
