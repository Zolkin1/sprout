#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""DEPRECATED: part of the motion-matching / DP clip generator, retained for ablation studies
only -- the live top-level driver is generate_mb_library.py.  Not on the live pipeline; see
``source/motion_gen/motion_gen/motion_matching/README.md``.

Top-level driver: generate every terrain-clip FAMILY, then compose ONE RL training library.

Each family has its own generator (they build their own motion-matching DB and emit an ``.h5`` with
an embedded terrain manifest); this driver runs them in sequence as subprocesses, then merges the
results into a single training library at controlled per-family tile proportions via
:mod:`compose_terrain_library`.

Scale is expressed as PER-FAMILY TILE COUNTS + one uniform ``--clips-per-tile`` (the RL importer
round-robins envs over tiles, so a family's training share = its tile count / total tiles):

* **flat**        -- ``generate_mm_clips`` free flat locomotion (``--flat-tiles`` tiles worth of clips).
* **boxes**       -- ``generate_box_clips`` box-tiles path, run once per jump skill (jump_on / jump_off /
  jump_off_front), a fixed box per tile shared by ``--clips-per-tile`` approach speeds.
* **stairs**      -- ``generate_box_clips`` tiles path, run once per (descent/ascent) x (transition/pure);
  clips-per-tile == number of ``--transition-vels``, so the driver builds ``--clips-per-tile`` evenly
  spaced approach speeds over ``--stair-vel-range``.
* **distractors** -- ``generate_distractor_clips`` (a box is seen but NOT jumped: stop / turn / between).

The terrain families share one big with-lead-in motion DB (``--bones-limit``); flat + distractors share a
LEANER no-lead-in DB (``--flat-bones-limit``) so the flat DP solves -- the bulk of the wall-clock -- stay
fast.  The candidate graph is content-addressed + cached, so each DB variant is built once and reused.

    # end-to-end VERIFICATION (small + fast):
    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/generate_terrain_library.py \
        --source hf/trajectories/retargeted/<merged> --bones-cache clips/bones_cache_locomotion.h5 \
        --out clips/g1_train_verify.h5 --work-dir clips/_famv --quick

Drop ``--quick`` for the full (~10k) library.  Runs with NO IsaacLab.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))          # <repo>/scripts/rsl_rl -> <repo>


def _run(name: str, script: str, argv: list[str]) -> None:
    """Run one generator/compose step as a subprocess (repo-root cwd, PYTHONPATH set); raise on failure."""
    env = dict(os.environ)
    src = f"{_REPO}/source/robot_rl:{_REPO}/source/motion_gen"
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-u", os.path.join(_HERE, script)] + argv
    print(f"\n===== [{name}] " + " ".join(argv), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=_REPO, env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"[{name}] FAILED (rc={r.returncode}) after {dt:.0f}s")
    print(f"===== [{name}] done in {dt:.0f}s", flush=True)


def _linspace_str(lo: float, hi: float, n: int) -> str:
    """``';'``-joined ``n`` evenly-spaced velocities -- the stair ``--transition-vels`` (one clip per
    value, so #values == clips-per-tile for both pure and transition stair tiles)."""
    if n <= 1:
        return f"{float(lo):.3f}"
    return ";".join(f"{lo + (hi - lo) * i / (n - 1):.3f}" for i in range(n))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Retargeted merged library (output channel layout).")
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--terrain-lib", default="trajectories/skills/library_v2",
                   help="The ONE unified library every family matches against: terrain skills "
                        "(stairs/jump geometry) AND clean locomotion strides (walk/run/side_step/turn/"
                        "backwards).  Drives both --terrain-lib (geometry) and --clean-source (motion).")
    p.add_argument("--out", required=True, help="Final composed training library .h5.")
    p.add_argument("--work-dir", default="clips/_families", help="Where the per-family .h5 are written.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="Per-family parallel workers (0 = all cores - 2).")
    p.add_argument("--quick", action="store_true", help="Small/fast scale for end-to-end verification.")
    # --- SCALE: per-family tile counts + uniform clips-per-tile (the ~10k spec) ---
    p.add_argument("--clips-per-tile", type=int, default=50, help="Uniform clips per tile across all families.")
    p.add_argument("--flat-tiles", type=int, default=60, help="Flat tiles (share = tiles / total).")
    p.add_argument("--box-tiles", default="20,15,15",
                   help="jump_on,jump_off,jump_off_front tile counts (comma-sep); sum = the box share.")
    p.add_argument("--transition-tiles", type=int, default=40, help="Stair TRANSITION tiles (split evenly descent/ascent).")
    p.add_argument("--pure-tiles", type=int, default=30, help="PURE stair tiles (split evenly descent/ascent).")
    p.add_argument("--distractor-tiles", type=int, default=20)
    # --- DB sizes: terrain approaches vs the (leaner, faster) flat DP ---
    p.add_argument("--bones-limit", type=int, default=4000, help="Bones cap for the TERRAIN families' DB (stairs/boxes).")
    p.add_argument("--flat-bones-limit", type=int, default=1500,
                   help="Bones cap for the FLAT + DISTRACTOR DB (smaller -> much faster flat DP solves; the "
                        "clean library already spans the full velocity range).")
    # --- ranges ---
    p.add_argument("--stair-vel-range", type=float, nargs=2, default=(0.6, 2.6), metavar=("MIN", "MAX"),
                   help="Stair approach speeds: clips-per-tile evenly-spaced speeds in this band.")
    p.add_argument("--box-vel-range", type=float, nargs=2, default=(0.5, 3.5), metavar=("MIN", "MAX"),
                   help="Approach forward-speed range (m/s) sampled per box-tile clip.")
    p.add_argument("--flat-clip-length", type=float, default=9.0)
    p.add_argument("--stair-floor-frac", type=float, default=0.5,
                   help="Fraction of stair tiles (pure + transition) that keep a flat ground plane; the "
                        "rest are the bare stairs (+ platforms) with no ground under them.")
    p.add_argument("--stair-float-frac", type=float, default=0.0,
                   help="Fraction of stair TILES with FLOATING (riser-less) treads -- thin slabs with an "
                        "open gap beneath, same tread tops (so the same clips are reused).")
    p.add_argument("--stair-float-thick-range", type=float, nargs=2, default=(0.03, 0.06),
                   metavar=("MIN", "MAX"), help="Floating-tread slab thickness band (m), per tile.")
    p.add_argument("--skip", default="", help="Comma-sep families to skip (stairs,boxes,distractors,flat).")
    args = p.parse_args()

    if args.quick:                                          # shrink everything for a fast pipeline check
        args.clips_per_tile = 4
        args.flat_tiles = 4
        args.box_tiles = "2,1,1"
        args.transition_tiles, args.pure_tiles = 2, 2
        args.distractor_tiles = 3
        args.flat_clip_length = 6.0
        args.bones_limit, args.flat_bones_limit = 1500, 800

    cpt = int(args.clips_per_tile)
    tvels = _linspace_str(args.stair_vel_range[0], args.stair_vel_range[1], cpt)   # stair clips-per-tile = #vels
    box_on, box_off, box_off_front = (int(x) for x in args.box_tiles.split(","))
    trans_dn = args.transition_tiles // 2; trans_up = args.transition_tiles - trans_dn
    pure_dn = args.pure_tiles // 2; pure_up = args.pure_tiles - pure_dn

    os.makedirs(os.path.join(_REPO, args.work_dir), exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    # Shared base argv: every family matches the SAME unified library (--clean-source = --terrain-lib).
    base = ["--source", args.source, "--bones-cache", args.bones_cache, "--device", args.device,
            "--clean-source", args.terrain_lib, "--workers", str(args.workers)]
    terr_bl = ["--bones-limit", str(args.bones_limit)]          # stairs/boxes DB (WITH lead-ins)
    flat_bl = ["--bones-limit", str(args.flat_bones_limit)]     # flat/distractor DB (NO lead-ins)
    box_vr = ["--box-vel-range", str(args.box_vel_range[0]), str(args.box_vel_range[1])]
    fam = {k: os.path.join(args.work_dir, f"family_{k}.h5") for k in (
        "stairs_dn_trans", "stairs_up_trans", "stairs_dn_pure", "stairs_up_pure",
        "jump_on", "jump_off", "jump_off_front", "distractors", "flat")}

    t0 = time.time()
    # --- STAIRS: (descent/ascent) x (transition/pure), each with cpt evenly-spaced approach vels ---
    # transition + pure MUST be separate runs: --transition-vels drives clips-per-tile for both.
    stair_base = base + terr_bl + ["--terrain-lib", args.terrain_lib, "--transition-vels", tvels,
                                   "--tile-seed", str(args.seed), "--stair-floor-frac", str(args.stair_floor_frac),
                                   "--stair-float-frac", str(args.stair_float_frac),
                                   "--stair-float-thick-range",
                                   *(str(v) for v in args.stair_float_thick_range)]
    stair_runs = [
        ("stairs_dn_trans", "stairs_down", ["--n-transition-tiles", str(trans_dn), "--n-pure-tiles", "0"]),
        ("stairs_up_trans", "stairs_up", ["--n-transition-tiles", str(trans_up), "--n-pure-tiles", "0"]),
        ("stairs_dn_pure", "stairs_down", ["--n-pure-tiles", str(pure_dn), "--n-transition-tiles", "0"]),
        ("stairs_up_pure", "stairs_up", ["--n-pure-tiles", str(pure_up), "--n-transition-tiles", "0"]),
    ]
    if "stairs" not in skip:
        for key, skill, extra in stair_runs:
            _run(key, "generate_box_clips.py",
                 stair_base + ["--terrain-skills", skill] + extra + ["--out", fam[key]])
    # --- BOXES: one run per jump skill, cpt clips per box tile ---
    if "boxes" not in skip:
        for key, skill, n in (("jump_on", "jump_on", box_on), ("jump_off", "jump_off", box_off),
                              ("jump_off_front", "jump_off_front", box_off_front)):
            _run(key, "generate_box_clips.py", base + terr_bl + [
                "--terrain-lib", args.terrain_lib, "--terrain-skills", skill,
                "--num-box-tiles", str(n), "--box-clips-per-tile", str(cpt),
                "--tile-seed", str(args.seed), "--seed", str(args.seed), "--out", fam[key]] + box_vr)
    # --- DISTRACTORS: cpt clips per tile (leaner flat DB) ---
    if "distractors" not in skip:
        _run("distractors", "generate_distractor_clips.py", base + flat_bl + [
            "--num-tiles", str(args.distractor_tiles), "--clips-per-tile", str(cpt),
            "--seed", str(args.seed), "--out", fam["distractors"]])
    # --- FLAT: flat_tiles * cpt varied clips (own leaner DB; fans across the fork pool) ---
    if "flat" not in skip:
        _run("flat", "generate_mm_clips.py", base + flat_bl + [
            "--clip-length", str(args.flat_clip_length), "--n-clips", str(args.flat_tiles * cpt),
            "--seed", str(args.seed), "--out", fam["flat"]])

    # --- compose to the target distribution (each terrain tile capped at cpt clips) ---
    mix_spec = [("stairs_dn_trans", trans_dn), ("stairs_up_trans", trans_up),
                ("stairs_dn_pure", pure_dn), ("stairs_up_pure", pure_up),
                ("jump_on", box_on), ("jump_off", box_off), ("jump_off_front", box_off_front),
                ("distractors", args.distractor_tiles)]
    skip_keys: set[str] = set()
    if "stairs" in skip:
        skip_keys |= {"stairs_dn_trans", "stairs_up_trans", "stairs_dn_pure", "stairs_up_pure"}
    if "boxes" in skip:
        skip_keys |= {"jump_on", "jump_off", "jump_off_front"}
    if "distractors" in skip:
        skip_keys |= {"distractors"}
    mix = ", ".join(f"{fam[k]}:{n}" for k, n in mix_spec if k not in skip_keys)
    compose = ["--mix", mix, "--max-clips-per-tile", str(cpt), "--out", args.out, "--device", args.device]
    if "flat" not in skip:
        compose += ["--flat-from", fam["flat"], "--flat-tiles", str(args.flat_tiles),
                    "--flat-clips", str(args.flat_tiles * cpt)]
    _run("compose", "compose_terrain_library.py", compose)

    print(f"\n===== terrain library written: {args.out}  (total {time.time() - t0:.0f}s) =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
