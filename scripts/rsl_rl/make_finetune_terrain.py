#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Build the terrain library ONE fine-tune config asks for.

    python scripts/rsl_rl/make_finetune_terrain.py --config configs/finetune/multi.yaml

One driver for all four configs, because they differ only in their tile COUNTS: boxes.yaml is
30/0/0/0, updown.yaml 0/30/0/0, indist.yaml 0/0/0/60 and multi.yaml 20/20/6/20. Writing four
builders would put the same join logic in four places and let them drift, which is the thing
the shared ``motion_gen.terrain.multi_courses`` exists to stop.

**The config is the single source of truth, and this closes its loop.** ``configs/finetune``
already declared every tile count and geometry band; until now nothing consumed them, so the
files described libraries that had to be built by hand with matching flags. This reads the
same file the fine-tune loop reads, writes the ``clip:`` path it names and the ``goals:`` spec
beside it, so a library and the run that uses it cannot disagree about what was built.

**A HOLDOUT is written too, from a different seed rather than a different grid.** A tile index
selects geometry here, so the equivalent of a held-out initial condition is unseen GEOMETRY:
same distributions, same counts, a seed the training library never drew from. The
in-distribution tiles are the exception -- they are a fixed set of real tiles, so the holdout
draws a different SAMPLE of them rather than different geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "source" / "motion_gen"), str(_ROOT / "source" / "robot_rl"),
           str(_HERE), str(_ROOT / "scripts" / "flow_matching")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py  # noqa: E402

from motion_gen.clips.clip_fingerprint import fingerprint_terrain_manifest  # noqa: E402
from motion_gen.clips.clip_io import ClipLibraryWriter  # noqa: E402
from motion_gen.terrain.freeform import GENERATOR_TAG  # noqa: E402
from motion_gen.terrain.multi_courses import (  # noqa: E402
    FAMILY_ORDER,
    MultiCounts,
    family_blocks,
    generate_multi_manifest,
    multi_generator_config,
    multi_goal_region_spec,
    tile_key,
    validate_multi_manifest,
)
from motion_gen.terrain.varied_courses import VariedCoursesCfg  # noqa: E402

from finetune_config import load_finetune_config  # noqa: E402
from terrain_library import library_meta  # noqa: E402

#: Seed offset for the holdout library. Large and fixed, so a holdout can never coincide with
#: a training seed someone picked by hand.
HOLDOUT_SEED_OFFSET: int = 100_000

#: Config keys that map straight onto a ``VariedCoursesCfg`` field of the same name.
_GEOMETRY_KEYS: tuple[str, ...] = (
    "updown_cells", "updown_measured_cells", "updown_measured_frac", "updown_steps_range",
    "updown_width_range_m", "updown_platform_range_m", "updown_wall_height_range_m",
    "boxes_per_tile_range", "box_height_range_m", "box_depth_range_m", "box_width_range_m",
    "box_gap_range_m",
)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="A configs/finetune/*.yaml.")
    p.add_argument("--indist_lib", default="clips/indist_train.h5",
                   help="Library to COPY in-distribution tiles from. Only read when the "
                        "config asks for some; they are copied rather than regenerated so the "
                        "control arm is the same question on the same ground.")
    p.add_argument("--no_holdout", action="store_true",
                   help="Write only the training library.")
    p.add_argument("--like", default=None,
                   help="Library whose channel-layout metadata to copy.")
    return p.parse_args()


def course_cfg(cfg, seed: int) -> VariedCoursesCfg:
    """A :class:`VariedCoursesCfg` carrying the config's declared geometry.

    Args:
        cfg: A loaded fine-tune config.
        seed: The seed for this role.

    Returns:
        The course config the drawn tiles are generated from.
    """
    kw = {k: getattr(cfg, k) for k in _GEOMETRY_KEYS}
    # The cells arrive as lists of lists; the dataclass holds tuples of pairs, and a tile's
    # draw indexes them either way -- but tuples keep the cfg hashable and un-mutated by a
    # caller that keeps a reference.
    kw["updown_cells"] = tuple(tuple(c) for c in kw["updown_cells"])
    kw["updown_measured_cells"] = tuple(tuple(c) for c in kw["updown_measured_cells"])
    for k in ("updown_steps_range", "updown_width_range_m", "updown_platform_range_m",
              "updown_wall_height_range_m", "boxes_per_tile_range", "box_height_range_m",
              "box_depth_range_m", "box_width_range_m", "box_gap_range_m"):
        kw[k] = tuple(kw[k])
    return VariedCoursesCfg(seed=int(seed), **kw)


def read_manifest(path: str) -> list[dict]:
    """The ``terrain_manifest`` of a library, without loading its clips.

    Args:
        path: The ``.h5``.

    Returns:
        The manifest.

    Raises:
        SystemExit: If the file is missing or carries no manifest.
    """
    if not os.path.isfile(path):
        raise SystemExit(
            f"in-distribution library not found: {path}. The config asks for in-distribution "
            "tiles, which are COPIED from a library built by make_indist_routes.py.")
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("terrain_manifest")
        if raw is None:
            raise SystemExit(f"{path} carries no terrain_manifest.")
        return json.loads(raw if isinstance(raw, str) else raw.tobytes().decode())


def write_library(path: str, manifest: list[dict], meta: dict, role: str,
                  counts: MultiCounts, cc: VariedCoursesCfg, cfg,
                  indist_source: str | None) -> None:
    """Write one manifest-only fine-tune library.

    Args:
        path: Output file.
        manifest: The terrain manifest.
        meta: Channel-layout metadata.
        role: ``"train"`` or ``"holdout"``, recorded in the file.
        counts: The tile counts it was built from.
        cc: The course config the drawn tiles came from.
        cfg: The loaded fine-tune config, for its declared command band.
        indist_source: Where the copied tiles came from.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    gen_cfg = multi_generator_config(
        counts, cc, indist_source=indist_source,
        commanded_speed_range_mps=tuple(cfg.commanded_speed_range_mps))
    gen_cfg["multi_courses"]["role"] = role
    gen_cfg["multi_courses"]["config"] = cfg.name
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as writer:
        writer.set_attr("terrain_manifest", manifest)
        writer.set_attr("generator", GENERATOR_TAG)
        writer.set_attr("generator_config", gen_cfg)


def main() -> int:
    """Build the library (and its holdout) the named config asks for."""
    args = parse_args()
    cfg = load_finetune_config(args.config)
    counts = MultiCounts(boxes=cfg.tiles_boxes, updown=cfg.tiles_updown,
                         entry_stairs=cfg.tiles_entry_stairs, indist=cfg.tiles_indist)
    meta = library_meta(args.like)

    indist_src = read_manifest(args.indist_lib) if counts.indist else []
    lo, hi = cfg.commanded_speed_range_mps
    print(f"[ft-terrain] config {args.config} ({cfg.name})")
    print(f"[ft-terrain] tiles: {counts.boxes} box + {counts.updown} up-into-down + "
          f"{counts.entry_up} entry-up + {counts.entry_down} entry-down + "
          f"{counts.indist} in-distribution = {counts.total}")
    print(f"[ft-terrain] commanded speed {lo}-{hi} m/s continuous per episode")
    if counts.indist:
        print(f"[ft-terrain] in-distribution tiles copied from {args.indist_lib} "
              f"({len(indist_src)} available)")

    spec = multi_goal_region_spec(counts, course_cfg(cfg, cfg.seed))
    os.makedirs(os.path.dirname(os.path.abspath(cfg.goals)) or ".", exist_ok=True)
    with open(cfg.goals, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    d = spec["default"]
    print(f"[ft-terrain] goal regions -> {cfg.goals}  (half_width {d['half_width_m']:.2f} m, "
          f"height band {d['height_tol_m']:.2f} m)")

    train_path = pathlib.Path(cfg.clip)
    roles = [("train", cfg.seed, str(train_path))]
    if not args.no_holdout:
        holdout = train_path.with_name(train_path.name.replace("_train", "") .replace(".h5", "")
                                       + "_holdout.h5")
        roles.append(("holdout", cfg.seed + HOLDOUT_SEED_OFFSET, str(holdout)))

    # The holdout EXCLUDES the training library's copied tiles. Without that the two are
    # independent samples of one fixed pool and overlap badly -- at 60 of 75 they shared 80%,
    # and jump_off_front (15 tiles, all of them taken by both) shared 100%. Drawn geometry does
    # not need this: a different seed is genuinely different terrain.
    spent: set[tuple[str, int]] = set()
    for role, seed, path in roles:
        cc = course_cfg(cfg, seed)
        try:
            manifest = generate_multi_manifest(counts, cc, indist_src,
                                               exclude_indist=spent or None)
        except ValueError as e:
            if role == "holdout" and "already spent" in str(e):
                print(f"\n[ft-terrain] NO HOLDOUT WRITTEN: {e}")
                print("[ft-terrain] Nothing partial is written -- an overlapping holdout "
                      "measures the tiles it was fitted on, which is worse than none.")
                break
            raise
        validate_multi_manifest(manifest)
        spent |= {tile_key(t) for t in manifest if t.get("family") not in FAMILY_ORDER}
        write_library(path, manifest, meta, role, counts, cc, cfg,
                      args.indist_lib if counts.indist else None)
        print(f"\n[ft-terrain] {role} (seed {seed}): {len(manifest)} tile(s) -> {path}")
        print(f"[ft-terrain]   terrain fingerprint {fingerprint_terrain_manifest(manifest)}")
        for fam, (a, b) in family_blocks(manifest).items():
            mark = "" if fam in FAMILY_ORDER else "   (copied)"
            print(f"[ft-terrain]   {fam:28s} ids {a:3d}..{b:3d}  "
                  f"({b - a + 1:2d} tiles){mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
