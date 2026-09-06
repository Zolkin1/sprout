# Copyright 2026 Zachary Olkin. All rights reserved.

"""Write the TWO-BOX fine-tune terrain as manifest-only clip libraries.

One tile per approach distance, identical box geometry on every tile -- a clone of the MuJoCo
eval course ``mjeval_box2_h040``.  See :mod:`motion_gen.terrain.box2` for why this course and
why the geometry is fixed.

**Two files, not one, and that is the holdout.**  The training distances and the held-out
distance go into SEPARATE libraries, so a recording run cannot reach the held-out tile by
accident: it is not in the file it was pointed at.  A single library with a "do not sample this
tile" flag would put the guarantee in a runtime filter that any later caller could drop, which
is the shape of leak this repo has already paid for once (the FM's episode split, where 86.5%
of held-out clips also appeared in training).

Usage::

    python scripts/rsl_rl/make_box2_terrain.py --out_dir clips \\
        --like clips/g1_mb_v5_body_mode11_decor.h5

writes ``clips/box2_train.h5`` (5 tiles) and ``clips/box2_holdout.h5`` (1 tile).

**Pass ``--like``.**  The FM checkpoint's ``frame_names`` are checked against the ee frames
resolved from the library's OWN channel names, so the minimal 7-channel layout admits only a
generator with no extra frames.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# <repo>/scripts/rsl_rl/make_box2_terrain.py -> parents[2] == <repo>.  Derived rather than
# counting ``..`` hops, so moving this script cannot silently break the path.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from motion_gen.clips.clip_fingerprint import fingerprint_terrain_manifest  # noqa: E402
from motion_gen.clips.clip_io import ClipLibraryWriter  # noqa: E402
from motion_gen.terrain.box2 import (  # noqa: E402
    BOX2_FAMILY,
    HOLDOUT_SPAWN_DISTANCE_M,
    HOLDOUT_SPEED_MPS,
    TRAIN_SPAWN_DISTANCES_M,
    TRAIN_SPEEDS_MPS,
    generate_box2_manifest,
)
from motion_gen.terrain.freeform import GENERATOR_TAG, validate_manifest  # noqa: E402

# The layout helper and the load-back check are generic over what is in the manifest, so they
# are taken from the free-form writer rather than restated.  Only the CONFIG block below differs.
from make_freeform_terrain import library_meta, verify  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="clips", help="Directory for the two .h5 files.")
    p.add_argument("--prefix", default="box2", help="Filename stem: <prefix>_{train,holdout}.h5")
    p.add_argument("--like", default=None,
                   help="Copy the channel layout from this library. Motion is never copied.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the load-back + assemble_clip_terrain round trip.")
    return p.parse_args()


def write_box2_library(path: str, manifest: list[dict], meta: dict, role: str,
                       distances: tuple[float, ...], speeds: tuple[float, ...]) -> None:
    """Write one manifest-only library.

    Args:
        path: Output file.
        manifest: The terrain manifest.
        meta: Channel-layout metadata.
        role: ``"train"`` or ``"holdout"``, recorded in the file.
        distances: The approach distances this file's tiles were built at (m).
        speeds: The commanded speeds this file is INTENDED to be recorded at (m/s). Recorded
            for provenance only -- the speed is a command and is pinned per run, so nothing
            reads this back.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as writer:
        writer.set_attr("terrain_manifest", manifest)
        writer.set_attr("generator", GENERATOR_TAG)
        writer.set_attr("generator_config", {
            "seed": 0,
            # No synthesised flat CLIP tiles; every tile here is a manifest tile.
            "counts": {"flat": 0},
            "box2": {"role": role, "family": BOX2_FAMILY,
                     "spawn_distances_m": [float(d) for d in distances],
                     "intended_speeds_mps": [float(s) for s in speeds],
                     "mjeval_course": "mjeval_box2_h040"},
        })


def main() -> int:
    """Write both libraries and check them."""
    args = parse_args()
    meta = library_meta(args.like)
    out_dir = pathlib.Path(args.out_dir)

    for role, distances, speeds in (
        ("train", TRAIN_SPAWN_DISTANCES_M, TRAIN_SPEEDS_MPS),
        ("holdout", (HOLDOUT_SPAWN_DISTANCE_M,), (HOLDOUT_SPEED_MPS,)),
    ):
        manifest = generate_box2_manifest(distances)
        validate_manifest(manifest, families=(BOX2_FAMILY,))
        path = str(out_dir / f"{args.prefix}_{role}.h5")
        write_box2_library(path, manifest, meta, role, distances, speeds)
        fp = fingerprint_terrain_manifest(manifest)
        print(f"\n[box2] {role}: {len(manifest)} tile(s) -> {path}")
        print(f"[box2]   approach distances {list(distances)} m, "
              f"intended speeds {list(speeds)} m/s")
        print(f"[box2]   terrain fingerprint {fp}")
        for t in manifest:
            r, b = t["route"], t["box2"]
            print(f"[box2]   tile {t['terrain_id']}: spawn {b['spawn_distance_m']:.2f} m out, "
                  f"tile {t['size_x']:.2f} x {t['size_y']:.2f} m, "
                  f"boxes x={b['box_faces_x'][0][0]:.2f}-{b['box_faces_x'][0][1]:.2f} and "
                  f"{b['box_faces_x'][1][0]:.2f}-{b['box_faces_x'][1][1]:.2f}, "
                  f"goal x={r['goal_x']:.2f} +-{r['goal_half_width']:.2f} m")
        if not args.no_verify:
            verify(path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
