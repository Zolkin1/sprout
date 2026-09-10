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
    BOX2_V2_FAMILY,
    COMMANDED_SPEED_RANGE_MPS,
    GOAL_SHIFT_V2_M,
    TARGET_LEAD_M,
    TARGET_LEAD_V2_M,
    HOLDOUT_SPAWN_DISTANCE_M,
    HOLDOUT_SPEED_MPS,
    TRAIN_SPAWN_DISTANCES_M,
    TRAIN_SPEEDS_MPS,
    generate_box2_manifest,
)
from motion_gen.terrain.freeform import GENERATOR_TAG, validate_manifest  # noqa: E402

# The layout helper and the load-back check are generic over what is in the manifest, so they
# are taken from the free-form writer rather than restated.  Only the CONFIG block below differs.
from terrain_library import library_meta, verify  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="clips", help="Directory for the two .h5 files.")
    p.add_argument("--variant", default="v1", choices=("v1", "v2"),
                   help="v1 is the mjeval course exactly. v2 moves the GOAL "
                        f"{GOAL_SHIFT_V2_M:.1f} m further out and grows the tile to match -- "
                        "the boxes and the spawn do not move, so it is the same obstacle with "
                        "a longer run-out. It gets its OWN family (box2_v2) and its own files "
                        "because the two are not comparable: more distance after box 2 is "
                        "strictly more chances to fall, so a v2 rate is lower than a v1 rate "
                        "on the same policy for a reason that is not the policy.")
    p.add_argument("--prefix", default=None,
                   help="Filename stem: <prefix>_{train,holdout}.h5. Defaults to `box2` for "
                        "v1 and `box2_v2` for v2.")
    p.add_argument("--like", default=None,
                   help="Copy the channel layout from this library. Motion is never copied.")
    p.add_argument("--train_distances", type=float, nargs="+", default=None,
                   help="Override the training approach distances (m). Defaults to the full "
                        "grid. The point of overriding is a SMALLER library for a fast "
                        "comparator -- `box2_quick` is --prefix box2_quick --train_distances "
                        "1.5 3.0, which concentrates every episode on two tiles instead of "
                        "spreading them over five, so a 30-minute run reaches a usable "
                        "per-cell n. The geometry of each tile is unchanged, so a quick cell "
                        "and a full-grid cell at the same distance are the same course.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the load-back + assemble_clip_terrain round trip.")
    return p.parse_args()


def write_box2_library(path: str, manifest: list[dict], meta: dict, role: str,
                       distances: tuple[float, ...], speeds: tuple[float, ...],
                       *, family: str = BOX2_FAMILY, variant: str = "v1",
                       target_lead_m: float = TARGET_LEAD_M) -> None:
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
        family: The manifest family the tiles were built with.
        variant: ``"v1"`` or ``"v2"``, recorded so a library states which course it is.
        target_lead_m: The goal lead the tiles were built with (m).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as writer:
        writer.set_attr("terrain_manifest", manifest)
        writer.set_attr("generator", GENERATOR_TAG)
        writer.set_attr("generator_config", {
            "seed": 0,
            # No synthesised flat CLIP tiles; every tile here is a manifest tile.
            "counts": {"flat": 0},
            "box2": {"role": role, "family": family,
                     "variant": variant, "target_lead_m": float(target_lead_m),
                     "spawn_distances_m": [float(d) for d in distances],
                     # The eval LADDER, provenance only -- a recording pins one rung per run.
                     "intended_speeds_mps": [float(s) for s in speeds],
                     # The band a TRAINING run samples continuously, under the key
                     # `clip_io.read_commanded_speed_range` actually reads. Before this the
                     # library declared only `intended_speeds_mps`, which NOTHING reads, so
                     # the whole declared-band path was a no-op here and the loop fell back
                     # to its own 0.5-2.2 -- the same numbers, stated in a shell script
                     # instead of in the artifact.
                     "commanded_speed_range_mps": [float(v) for v in COMMANDED_SPEED_RANGE_MPS],
                     # ONLY v1 is the mjeval clone. v2 moves the goal 3 m further out with the
                     # boxes unmoved, so it is a different course; claiming the clone on both
                     # let a v2 recording read as reproducing an mjeval number.
                     **({"mjeval_course": "mjeval_box2_h040"} if variant == "v1" else {})},
        })


def main() -> int:
    """Write both libraries and check them."""
    args = parse_args()
    meta = library_meta(args.like)
    out_dir = pathlib.Path(args.out_dir)
    v2 = args.variant == "v2"
    lead = TARGET_LEAD_V2_M if v2 else TARGET_LEAD_M
    family = BOX2_V2_FAMILY if v2 else BOX2_FAMILY
    prefix = args.prefix or ("box2_v2" if v2 else "box2")
    print(f"[box2] variant {args.variant}: target lead {lead:.2f} m, family {family!r}"
          + (f"  (goal {GOAL_SHIFT_V2_M:.1f} m further out than v1; rates NOT comparable "
             "with v1)" if v2 else "  (the mjeval course exactly)"))

    train_d = (tuple(float(d) for d in args.train_distances) if args.train_distances
               else TRAIN_SPAWN_DISTANCES_M)
    for role, distances, speeds in (
        ("train", train_d, TRAIN_SPEEDS_MPS),
        ("holdout", (HOLDOUT_SPAWN_DISTANCE_M,), (HOLDOUT_SPEED_MPS,)),
    ):
        manifest = generate_box2_manifest(distances, target_lead_m=lead, family=family)
        validate_manifest(manifest, families=(family,))
        path = str(out_dir / f"{prefix}_{role}.h5")
        write_box2_library(path, manifest, meta, role, distances, speeds,
                           family=family, variant=args.variant, target_lead_m=lead)
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
