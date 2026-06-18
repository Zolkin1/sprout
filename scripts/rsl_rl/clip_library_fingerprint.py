#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Print the fingerprints recorded in a dense motion-clip library.

Reads the JSON-encoded ``clip_fingerprint`` / ``source_library_fingerprint``
/ ``generator_config`` attrs from a clip library ``.h5`` and prints them, so
two machines can confirm they hold the *same* generated clip set (the
counterpart to ``traj_library_fingerprint.py`` for stride libraries).

Only depends on ``h5py`` + stdlib, so it runs under a bare ``python3``
without the IsaacLab environment::

    python3 scripts/rsl_rl/clip_library_fingerprint.py clips/g1_flat_v1.h5
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
        print(f"  source lib short  : {src_fp.get('short', '<none>')} "
              f"({src_fp.get('num_trajectories', '?')} strides)")
        if args.config:
            print("  generator config  :")
            print(json.dumps(attr("generator_config"), indent=4, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
