# Copyright 2026 Zachary Olkin. All rights reserved.

"""DEPRECATED 2026-09-09 -- writes the randomised free-form section library.

**Nothing has ever been fine-tuned on it.**  Every result on this branch came from the FIXED
courses instead (`make_box2_terrain.py`, `make_step_courses_terrain.py`,
`make_varied_terrain.py`, `make_indist_routes.py`), because a fixed course can be overfitted
to, re-evaluated on and compared against MuJoCo, and a distribution cannot.  No library it
writes exists on disk.

Kept, unused, because the reasoning below is still true about how to author terrain.  The two
helpers four LIVE builders used to import from here -- `library_meta` and `verify` -- moved to
`scripts/rsl_rl/terrain_library.py`, so a deprecated script is no longer a load-bearing
dependency of every course that replaced it.

Write a MANIFEST-ONLY clip library: free-form terrain with no clips in it.

The clip library is the spine -- it carries both the reference motions and the terrain they were
generated on -- and everything downstream reaches terrain through it: the RL importer, both
MuJoCo emitters, both viser viewers, ``terrain_fingerprint`` and the eval's tile sampler.  The
FM-driven fine-tune needs the second half and none of the first: its reference is generated
perceptively (``reference_source="fm"``), so no clip is ever read.

So this writes a library with a ``terrain_manifest`` and ZERO CLIPS -- the clip datasets are
present with their full channel layout and zero extent -- and it is passed as ``--clip`` exactly
like any other.  Most of the terrain side needed no special case: ``assert_manifest_covers_clips``
no-ops on an empty ``terrain_id`` and ``default_num_flat_tiles`` returns 0 with no flat clips and
``counts.flat == 0``.  What DID need stating is what "zero clips" means to the RL runtime, and it
is stated in exactly two places: ``ClipLibraryData.has_clips`` (the predicate) and
``MotionClipManager._require_clips`` (the refusal every gather goes through).  A clip-consuming
term left wired on this task therefore fails by name instead of quietly indexing nothing.

Usage::

    python scripts/rsl_rl/make_freeform_terrain.py --out clips/freeform_v1.h5 \\
        --seed 0 --flat 8 --boxes 16 --stairs 24

    # copy an existing library's channel layout, for a consumer that inspects it
    python scripts/rsl_rl/make_freeform_terrain.py --out clips/freeform_v1.h5 \\
        --like clips/g1_mb_v5_body_mode11_decor.h5

Then::

    python scripts/rsl_rl/view_terrain_viser.py --clip clips/freeform_v1.h5

.. note::
    The file goes through :class:`~motion_gen.clips.clip_io.ClipLibraryWriter` like every other
    library, at ``num_clips=0``.  It used to be written with raw ``h5py`` because that writer
    chunks each clip dataset ``(1, ...)``, which h5py refuses against a zero-extent clip axis
    ("Chunk shape must not be greater than data shape") -- so the datasets were simply OMITTED.
    That was the wrong half to give up: the loader's placeholder for an absent dataset is rank-1,
    and ``num_steps`` is read off ``y_des.shape[1]``, so ``MotionClipManager`` raised
    ``IndexError`` on the first line that touched it and the task could not be constructed at
    all.  The writer now drops the CHUNKING at zero clips (there is nothing to chunk) and keeps
    the SHAPES, which is what makes the library well-defined rather than merely loadable.

.. note::
    The terrain FINGERPRINT is printed and deliberately NOT stored.  ``ClipTerrainImporter``
    recomputes it from the manifest at load, so a stored copy would be a second statement of the
    same thing that any later manifest edit (``decorate_terrain.py``, ``author_clutter.py``)
    would leave stale.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py

# <repo>/scripts/rsl_rl/make_freeform_terrain.py -> parents[2] == <repo>.  Derived rather than
# counting ``..`` hops, so moving this script cannot silently break the path.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Its OWN directory, so the sibling `terrain_library` import below resolves however this module
# was loaded. Running it as a script works without this -- python puts the script's directory on
# the path itself -- so the gap is invisible from the CLI and appears only when something
# IMPORTS this module, which `test_freeform_env_cfg` does.
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from motion_gen.clips.clip_fingerprint import fingerprint_terrain_manifest  # noqa: E402
from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library  # noqa: E402
from motion_gen.clips.clip_manager import MotionClipManager  # noqa: E402
# The ONE copy. These were duplicated here when they moved out; two byte-identical
# definitions of `verify` is exactly the drift the move existed to prevent.
from terrain_library import (  # noqa: E402,F401
    _MINIMAL_META, library_meta, schema_only_layout, verify,
)

from motion_gen.terrain.freeform import (  # noqa: E402
    COMMANDED_SPEED_RANGE_MPS,
    FAMILIES,
    GENERATOR_TAG,
    FreeformConfig,
    generate_freeform_manifest,
    validate_manifest,
)

#: Channel layout for a library that carries no motion.  It is pure bookkeeping here -- nothing
#: reads it without clips to index -- but it has to be a coherent, non-empty layout, because the
#: loader's metadata keys are not optional and a zero-width name list would make the shapes
#: degenerate.  ``--like`` copies a real library's layout for a consumer that inspects it.



def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output .h5 (a manifest-only clip library).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed. A tile's geometry depends only on (seed, terrain_id).")
    p.add_argument("--flat", type=int, default=8, help="Flat sections.")
    p.add_argument("--boxes", type=int, default=16, help="Box-corridor sections.")
    p.add_argument("--stairs", type=int, default=24, help="Stair sections.")
    p.add_argument("--like", default=None,
                   help="Copy the channel layout (pos/vel names, dt, contact frames) from this "
                        "library instead of the minimal one. Motion is never copied.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the load-back + assemble_clip_terrain round trip.")
    return p.parse_args()


def write_terrain_library(path: str, manifest: list[dict], meta: dict, cfg: FreeformConfig) -> None:
    """Write the manifest-only ``.h5``.

    Through the shared :class:`ClipLibraryWriter` at ``num_clips=0``, so the channel layout,
    the ``terrain_type_names`` list and the clip datasets' RANK are all stated by the one
    writer every other library uses -- and there is no second place stating what a library
    looks like.

    Args:
        path: Output file.
        manifest: The terrain manifest.
        meta: Channel-layout metadata (see :func:`library_meta`).
        cfg: The generation config, recorded so the file states how it was made.
    """
    counts = {"flat": int(cfg.num_flat), "boxes": int(cfg.num_boxes),
              "stairs": int(cfg.num_stairs)}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as writer:
        writer.set_attr("terrain_manifest", manifest)
        writer.set_attr("generator", GENERATOR_TAG)
        writer.set_attr("generator_config", {
            "seed": int(cfg.seed),
            # ``counts.flat`` is what ``default_num_flat_tiles`` reads, and it means flat CLIP
            # tiles -- the synthesised patches a flat clip is dropped onto.  There are none: the
            # flat SECTIONS here are manifest tiles like any other, so this is 0 and the
            # free-form counts live under their own key.
            "counts": {"flat": 0},
            # The band, under the key `clip_io.read_commanded_speed_range` reads. Without it
            # a loop pointed here falls back to box2's 0.5-2.2, whose floor is below this
            # command's own 0.6 -- a speed it was never built to be recorded at.
            "freeform": {"section_counts": counts,
                         "commanded_speed_range_mps":
                             [float(v) for v in COMMANDED_SPEED_RANGE_MPS]},
        })


def main() -> int:
    """Generate, validate, write and verify."""
    args = parse_args()
    cfg = FreeformConfig(seed=args.seed, num_flat=args.flat, num_boxes=args.boxes,
                         num_stairs=args.stairs)
    manifest = generate_freeform_manifest(cfg)
    validate_manifest(manifest)
    write_terrain_library(args.out, manifest, library_meta(args.like), cfg)

    fp = fingerprint_terrain_manifest(manifest)
    per_family = {fam: sum(1 for m in manifest if m["family"] == fam) for fam in FAMILIES}
    arms = {a: sum(1 for m in manifest if m.get("stair_sampling_arm") == a)
            for a in ("regulation", "trained")}
    print(f"[freeform] wrote {args.out}: {len(manifest)} tiles {per_family}, "
          f"stair arms {arms}, {fp['num_overlays']} overlays, "
          f"terrain fingerprint {fp['short']} (recomputed at load; not stored)")
    if not args.no_verify:
        verify(args.out, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
