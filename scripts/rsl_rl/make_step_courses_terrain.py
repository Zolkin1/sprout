# Copyright 2026 Zachary Olkin. All rights reserved.

"""Write the CURB and UP-INTO-DOWN fine-tune terrains as manifest-only clip libraries.

Two fixed courses, siblings of ``make_box2_terrain.py`` and built the same way -- see
:mod:`motion_gen.terrain.step_courses` for what each one probes and why every constant is what
it is.

**Two files per course, and that is the holdout.**  The training grid and the held-out cell go
into SEPARATE libraries, so a recording run cannot reach the held-out tile by accident: it is
not in the file it was pointed at.  A single library with a "do not sample this tile" flag
would put the guarantee in a runtime filter any later caller could drop, which is the shape of
leak this repo has already paid for once (the FM's episode split, where 86.5% of held-out clips
also appeared in training).

**The two courses grid over different things**, which is a property of the questions and not an
inconsistency: the curb's tiles are step HEIGHTS (the axis that makes it out of distribution)
at a pinned approach, while up-into-down's tiles are approach DISTANCES on fixed geometry, as
box2's are.

Usage::

    python scripts/rsl_rl/make_step_courses_terrain.py --course curb \\
        --out_dir clips --like clips/g1_mb_v5_body_mode11_decor.h5

    python scripts/rsl_rl/make_step_courses_terrain.py --course updown --platform_depth 2.0 \\
        --out_dir clips --like clips/g1_mb_v5_body_mode11_decor.h5

writes ``clips/curb_{train,holdout}.h5`` (4 + 1 tiles) and
``clips/updown_p020_{train,holdout}.h5`` (5 + 1 tiles).

**Pass ``--like``.**  The FM checkpoint's ``frame_names`` are checked against the ee frames
resolved from the library's OWN channel names, so the minimal 7-channel layout admits only a
generator with no extra frames.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# <repo>/scripts/rsl_rl/make_step_courses_terrain.py -> parents[2] == <repo>.  Derived rather
# than counting ``..`` hops, so moving this script cannot silently break the path.
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
from motion_gen.terrain.freeform import GENERATOR_TAG, validate_manifest  # noqa: E402
from motion_gen.terrain import step_courses as sc  # noqa: E402

# The layout helper and the load-back check are generic over what is in the manifest, so they
# come from the free-form writer rather than being restated.  Only the CONFIG block differs.
from make_freeform_terrain import library_meta, verify  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--course", required=True, choices=("curb", "updown"),
                   help="curb = one wide step onto a surface that keeps going (authored here, "
                        "deliberately out of distribution). updown = mjeval's staircase walked "
                        "up, across a platform and back down.")
    p.add_argument("--out_dir", default="clips", help="Directory for the two .h5 files.")
    p.add_argument("--prefix", default=None,
                   help="Filename stem: <prefix>_{train,holdout}.h5. Defaults to the course's "
                        "family name (`curb`, `updown_p020`, ...).")
    p.add_argument("--like", default=None,
                   help="Copy the channel layout from this library. Motion is never copied.")
    p.add_argument("--platform_depth", type=float, default=None,
                   help="updown only: flat platform between the two flights (m). mjeval sweeps "
                        f"{list(sc.UPDOWN_PLATFORM_DEPTHS_M)} -- 8.0 is the in-distribution "
                        "control (trained transition platforms are 6.2-10.1 m) and 2.0 is short "
                        "enough to present as the top of a BOX rather than as a landing. Each "
                        "gets its OWN family and files: they are courses of different lengths, "
                        "so their success rates are not comparable.")
    p.add_argument("--runout", type=float, default=sc.UPDOWN_RUNOUT_M,
                   help="updown only: EXTRA flat ground past the bottom riser (m). The default "
                        "0.0 is mjeval's up_into_down exactly; mjeval's own stairs_down class "
                        "adds 3.0, because at 0.0 the goal band starts 0.5 m past the last "
                        "riser and a robot still stumbling off the flight scores the same as "
                        "one that recovered. A non-zero value takes its own family.")
    p.add_argument("--visual_walls", action="store_true",
                   help="Author the stairwell walls as DECOR (no collider), which is mjeval's "
                        "configuration and what makes a tile an mjeval clone. Default is solid: "
                        "the reason mjeval calls visual-only non-negotiable is `ee_pose_error`, "
                        "a termination the fine-tune task does not have. Solid walls enter the "
                        "collider SDF, so the terrain reward charges a plan that puts a limb "
                        "inside one.")
    p.add_argument("--no_walls", action="store_true",
                   help="updown only: omit the visual-only stairwell walls. They are what "
                        "mjeval has, and a wall is the vertical surface the generator is known "
                        "to misread as a box -- so dropping them changes what the cameras see "
                        "and a run without them is not comparable to one with them.")
    p.add_argument("--train_heights", type=float, nargs="+", default=None,
                   help="curb only: override the training step heights (m). Defaults to the "
                        f"full grid {list(sc.CURB_TRAIN_HEIGHTS_M)}.")
    p.add_argument("--train_approaches", type=float, nargs="+", default=None,
                   help="updown only: override the training approach distances (m). Defaults "
                        f"to {list(sc.UPDOWN_TRAIN_APPROACHES_M)}. The point of overriding is a "
                        "SMALLER library for a fast comparator: fewer tiles concentrates the "
                        "same episode budget, and a tile at a given distance is the same course "
                        "either way.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the load-back + assemble_clip_terrain round trip.")
    return p.parse_args()


def _resolve(args: argparse.Namespace) -> dict:
    """Turn the CLI into one course spec: family, manifests and what to record.

    Kept apart from :func:`main` because it is the half that decides what a file MEANS, and it
    refuses a cross-course flag rather than ignoring it -- a `--platform_depth` silently
    dropped on a curb run would write a file whose name and contents disagreed.

    Args:
        args: Parsed arguments.

    Returns:
        ``prefix``, ``family``, ``config`` (the generator_config block), and ``roles``: a
        ``(role, manifest, grid, speeds)`` tuple per file to write.

    Raises:
        SystemExit: On a flag that does not belong to the chosen course.
    """
    wrong = []
    if args.course == "curb":
        for name in ("platform_depth", "train_approaches"):
            if getattr(args, name) is not None:
                wrong.append(f"--{name}")
        if args.runout != sc.UPDOWN_RUNOUT_M:
            wrong.append("--runout")
        if args.no_walls:
            wrong.append("--no_walls")
    else:
        if args.train_heights is not None:
            wrong.append("--train_heights")
    if wrong:
        raise SystemExit(f"REFUSING: {', '.join(wrong)} does not apply to --course "
                         f"{args.course}; a flag that is silently dropped writes a library "
                         "whose provenance block does not describe it.")

    if args.course == "curb":
        heights = (tuple(float(h) for h in args.train_heights) if args.train_heights
                   else sc.CURB_TRAIN_HEIGHTS_M)
        family = sc.CURB_FAMILY
        return {
            "prefix": args.prefix or family,
            "family": family,
            "config": {"course": "curb", "family": family,
                       "approach_m": sc.APPROACH_M,
                       "top_depth_m": sc.CURB_TOP_DEPTH_M,
                       "corridor_width_m": 2.0 * sc.CURB_HALF_WIDTH_M
                                           + 2.0 * sc.SECTION_MARGIN_M,
                       "mjeval_course": ""},
            "roles": (
                ("train", sc.generate_curb_manifest(heights), heights,
                 sc.CURB_TRAIN_SPEEDS_MPS),
                ("holdout", sc.generate_curb_manifest((sc.CURB_HOLDOUT_HEIGHT_M,)),
                 (sc.CURB_HOLDOUT_HEIGHT_M,), (sc.CURB_HOLDOUT_SPEED_MPS,)),
            ),
            "speed_range": None,
        "grid_name": "step_heights_m",
        }

    depth = args.platform_depth
    if depth is None:
        raise SystemExit("REFUSING: --course updown needs --platform_depth; mjeval builds "
                         f"{list(sc.UPDOWN_PLATFORM_DEPTHS_M)} and they are different courses "
                         "with incomparable rates, so there is no safe default.")
    approaches = (tuple(float(a) for a in args.train_approaches) if args.train_approaches
                  else sc.UPDOWN_TRAIN_APPROACHES_M)
    walls = not args.no_walls
    family = sc.updown_family(depth, args.runout)
    kw = {"runout_m": args.runout, "walls": walls,
          "wall_collidable": not bool(args.visual_walls)}
    return {
        "prefix": args.prefix or family,
        "family": family,
        "config": {"course": "updown", "family": family,
                   "platform_depth_m": float(depth), "runout_m": float(args.runout),
                   "rise_m": sc.UPDOWN_RISE_M, "run_m": sc.UPDOWN_RUN_M,
                   "steps": int(sc.UPDOWN_STEPS), "width_m": sc.UPDOWN_WIDTH_M,
                   "walls": bool(walls),
                   "wall_collidable": bool(walls and not args.visual_walls),
                   # Agrees with `make_updown_tile`'s own claim, which needs BOTH the
                   # default runout and visual walls -- solid walls are different physics.
                   "mjeval_course": (f"mjeval_updn_p{int(round(float(depth) * 10)):03d}"
                                     if (args.runout == sc.UPDOWN_RUNOUT_M
                                         and (args.visual_walls or not walls)) else "")},
        # The commanded band is a property of the COURSE and is stamped, so a runner reads
        # it rather than restating it -- see `clip_io.read_commanded_speed_range`. The
        # holdout keeps a single pinned speed: it is a held-out INITIAL CONDITION, and
        # varying its command too would confound the two axes.
        "speed_range": sc.UPDOWN_SPEED_RANGE_MPS,
        "roles": (
            ("train", sc.generate_updown_manifest(approaches, depth, **kw), approaches,
             sc.UPDOWN_SPEED_RANGE_MPS),
            ("holdout", sc.generate_updown_manifest((sc.UPDOWN_HOLDOUT_APPROACH_M,), depth,
                                                    **kw),
             (sc.UPDOWN_HOLDOUT_APPROACH_M,), (sc.UPDOWN_HOLDOUT_SPEED_MPS,)),
        ),
        "grid_name": "approach_distances_m",
    }


def write_library(path: str, manifest: list[dict], meta: dict, role: str,
                  grid_name: str, grid: tuple[float, ...], speeds: tuple[float, ...],
                  config: dict, speed_range: tuple[float, float] | None = None) -> None:
    """Write one manifest-only library.

    Args:
        path: Output file.
        manifest: The terrain manifest.
        meta: Channel-layout metadata.
        role: ``"train"`` or ``"holdout"``, recorded in the file.
        grid_name: What this course's tiles vary (``step_heights_m`` / ``approach_distances_m``).
        grid: The values those tiles were built at, in TILE ORDER -- so ``terrain_id`` is an
            index into this and the geometry or IC a row was collected at is recoverable.
        speeds: Representative commanded speeds, for the build banner. Provenance only.
        speed_range: The commanded-speed interval this course is to be recorded over (m/s),
            stamped so a runner READS it instead of restating it --
            :func:`motion_gen.clips.clip_io.read_commanded_speed_range` is the reader and
            ``run_awr_loop.sh`` uses it. ``None`` stamps nothing, which every library written
            before this did, and a reader must then fall back to its own default.
        config: The course's own constants, from :func:`_resolve`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as writer:
        writer.set_attr("terrain_manifest", manifest)
        writer.set_attr("generator", GENERATOR_TAG)
        writer.set_attr("generator_config", {
            "seed": 0,
            # No synthesised flat CLIP tiles; every tile here is a manifest tile.
            "counts": {"flat": 0},
            "step_course": {**config, "role": role,
                            grid_name: [float(v) for v in grid],
                            "intended_speeds_mps": [float(s) for s in speeds],
                            **({} if speed_range is None else
                               {"commanded_speed_range_mps": [float(v) for v in speed_range]})},
        })


def _describe(tile: dict) -> str:
    """One line naming what a tile's geometry is, for the build log.

    Args:
        tile: A manifest entry from either course.

    Returns:
        The line, without the leading tile id.
    """
    r = tile["route"]
    common = (f"tile {tile['size_x']:.2f} x {tile['size_y']:.2f} m, "
              f"spawn x={r['start_x']:.2f}, goal x={r['goal_x']:.2f} "
              f"+-{r['goal_half_width']:.2f} m at z={r['goal_z']:.3f}")
    if "curb" in tile:
        c = tile["curb"]
        return (f"step {c['height_m']:.3f} m at x={c['riser_x'][0]:.2f}, top to "
                f"x={c['riser_x'][1]:.2f} ({c['top_depth_m']:.1f} m deep, full "
                f"{c['width_m']:.2f} m width); {common}")
    u = tile["updown"]
    return (f"spawn {u['approach_m']:.2f} m out, {u['steps']} x {u['rise_m']:.3f} m up to "
            f"z={u['flight_height_m']:.2f} (x={u['first_riser_x']:.2f}), platform "
            f"{u['platform_x'][0]:.2f}-{u['platform_x'][1]:.2f}, down to floor at "
            f"x={u['bottom_x']:.2f}; walls {u['walls']}; {common}")


def main() -> int:
    """Write both libraries for the chosen course and check them."""
    args = parse_args()
    spec = _resolve(args)
    meta = library_meta(args.like)
    out_dir = pathlib.Path(args.out_dir)
    print(f"[step] course {args.course!r}, family {spec['family']!r}"
          + (f"  (== mjeval {spec['config']['mjeval_course']})"
             if spec["config"]["mjeval_course"] else "  (authored here; no mjeval twin)"))

    for role, manifest, grid, speeds in spec["roles"]:
        validate_manifest(manifest, families=(spec["family"],))
        path = str(out_dir / f"{spec['prefix']}_{role}.h5")
        # The HOLDOUT stamps no range: its speed is pinned, so a band there would say the
        # opposite of what the file is for.
        write_library(path, manifest, meta, role, spec["grid_name"], grid, speeds,
                      spec["config"],
                      speed_range=(spec.get("speed_range") if role == "train" else None))
        print(f"\n[step] {role}: {len(manifest)} tile(s) -> {path}")
        print(f"[step]   {spec['grid_name']} {list(grid)}, "
              f"intended speeds {list(speeds)} m/s")
        print(f"[step]   terrain fingerprint {fingerprint_terrain_manifest(manifest)}")
        for t in manifest:
            print(f"[step]   tile {t['terrain_id']}: {_describe(t)}")
        if not args.no_verify:
            verify(path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
