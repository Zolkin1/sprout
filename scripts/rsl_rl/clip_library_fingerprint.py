#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Print the fingerprints recorded in a dense motion-clip library.

Reads the JSON-encoded ``clip_fingerprint`` / ``source_library_fingerprint``
/ ``generator_config`` attrs from a clip library ``.h5`` and prints them, so
two machines can confirm they hold the *same* generated clip set (the
counterpart to ``traj_library_fingerprint.py`` for stride libraries).

With ``--verify`` it also RE-HASHES a solved library and compares, which answers the
question the recorded hash exists for: "was this library built from the solved
library I have?". The local ``hf/`` cache is never re-checked against the remote, so
two machines can hold different libraries at one path and nothing else would notice.

Only depends on ``h5py`` + stdlib, so it runs under a bare ``python3``
without the IsaacLab environment::

    python3 scripts/rsl_rl/clip_library_fingerprint.py clips/g1_flat_v1.h5
    python3 scripts/rsl_rl/clip_library_fingerprint.py clips/g1_mb_v6.h5 \
        --verify trajectories/skills/library_v2

Exit code is non-zero on a --verify mismatch, so it can gate a build.
"""

from __future__ import annotations

import argparse
import json

import h5py


def main() -> int:
    """Parse args, read the clip library attrs, and print the fingerprints."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="Path to a clip library .h5 file.")
    parser.add_argument("--config", action="store_true", help="Also print the full generator config.")
    parser.add_argument("--verify", default=None, metavar="SOLVED_LIBRARY",
                        help="Re-hash this solved traj-opt library and compare against the one the "
                             "clip library records. Non-zero exit on a mismatch.")
    parser.add_argument("--hf-repo", default="zolkin/robot_rl",
                        help="HuggingFace repo for --verify; '' for a purely local path.")
    args = parser.parse_args()

    with h5py.File(args.library, "r") as f:
        def attr(name):
            v = f.attrs.get(name)
            if isinstance(v, (str, bytes)):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return v
            return v

        clip_fp = attr("clip_fingerprint") or {}
        src_fp = attr("source_library_fingerprint") or {}
        n = f.attrs.get("num_clips_written", f.attrs.get("num_clips"))
        print(f"library: {args.library}")
        print(f"  clips written     : {n}")
        print(f"  clip fingerprint  : {clip_fp.get('hash', '<none>')}")
        print(f"  clip short        : {clip_fp.get('short', '<none>')}")
        print(f"  source lib hash   : {src_fp.get('hash', '<none>')}")
        # num_clips is what fingerprint_solved_library records; num_trajectories was the older
        # stride-library field name, and asking only for it printed "?" on every current library.
        _n_src = src_fp.get("num_clips", src_fp.get("num_trajectories", "?"))
        print(f"  source lib short  : {src_fp.get('short', '<none>')} ({_n_src} clips)")
        gen_cfg = attr("generator_config") or {}
        recipe = gen_cfg.get("recipe") if isinstance(gen_cfg, dict) else None
        if recipe:
            print(f"  recipe            : {gen_cfg.get('recipe_path', '<unnamed>')} "
                  f"({len(recipe)} keys, seed {recipe.get('seed', '?')})")
        if args.config:
            print("  generator config  :")
            print(json.dumps(gen_cfg, indent=4, sort_keys=True))

    if args.verify is None:
        return 0

    # Imported here so the plain print path keeps working without motion_gen importable.
    # The checkout is SEARCHED for rather than counted in ``..`` hops: a hop count breaks
    # silently when a file moves, which CLAUDE.md records as having already happened once.
    import pathlib
    import sys
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "source" / "motion_gen" / "motion_gen").is_dir():
            sys.path.insert(0, str(parent / "source" / "motion_gen"))
            break
    else:
        raise SystemExit("--verify needs source/motion_gen, which is not above this script")
    from motion_gen.clips.clip_fingerprint import (  # noqa: E402
        fingerprint_solved_library, verify_solved_fingerprint)

    current = fingerprint_solved_library(args.verify, args.hf_repo or None)
    ok, msg = verify_solved_fingerprint(src_fp, current)
    print(f"\n  verify            : {msg}")
    if not ok:
        return 1
    if not src_fp:
        print("  (nothing was compared -- this library records no solved-library fingerprint)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
