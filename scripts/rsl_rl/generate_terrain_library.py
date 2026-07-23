#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Top-level driver: generate every terrain-clip FAMILY, then compose ONE RL training library.

Each family has its own generator (they build their own motion-matching DB and emit an ``.h5`` with
an embedded terrain manifest); this driver runs them in sequence as subprocesses, then merges the
results into a single training library at controlled per-family tile proportions via
:mod:`compose_terrain_library`.  Families:

* **stairs**      -- ``generate_box_clips`` tiles path (pure looped descents + flat->stairs->flat
  transition tiles), many approach speeds sharing each tile.
* **boxes**       -- ``generate_box_clips`` box-tiles path (a fixed jump-on box per tile, many approach
  speeds sharing each tile -- the box analog of the stair transition tiles).
* **distractors** -- ``generate_distractor_clips`` (a box is seen but NOT jumped: stop / turn / between).
* **flat**        -- ``generate_mm_clips`` free flat locomotion (the negative-space majority class).

The composer draws stairs/boxes/distractors from ``--mix`` (one tile per terrain_id, replicated to the
target count) and the flat pool from ``--flat-from``, so the training distribution is dialed directly.

    # end-to-end VERIFICATION (small + fast):
    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/generate_terrain_library.py \
        --source hf/trajectories/retargeted/<merged> --bones-cache clips/bones_cache_locomotion.h5 \
        --out clips/g1_train_verify.h5 --work-dir clips/_famv --quick

Drop ``--quick`` and raise the per-family counts / ``--bones-limit`` for the full (~10k) library.
Runs with NO IsaacLab.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Retargeted merged library (output channel layout).")
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--terrain-lib", default="trajectories/skills/library_v2")
    p.add_argument("--out", required=True, help="Final composed training library .h5.")
    p.add_argument("--work-dir", default="clips/_families", help="Where the per-family .h5 are written.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="Small/fast scale for end-to-end verification.")
    # --- per-family SCALE (production defaults; --quick shrinks them) ---
    p.add_argument("--bones-limit", type=int, default=8000, help="Cap bones clips per family matcher build.")
    p.add_argument("--pure-tiles", type=int, default=8)
    p.add_argument("--transition-tiles", type=int, default=8)
    p.add_argument("--transition-vels", default="0.8;1.5;2.5")
    p.add_argument("--stair-floor-frac", type=float, default=0.5,
                   help="Fraction of stair tiles (pure + transition) that keep a flat ground plane; the "
                        "rest are the bare stairs (+ platforms) with no ground under them.")
    p.add_argument("--box-tiles", type=int, default=8, help="Jump-on box tiles (one fixed box each).")
    p.add_argument("--box-clips-per-tile", type=int, default=4,
                   help="Approach speeds per box tile (all share the tile's box + terrain_id).")
    p.add_argument("--box-vel-range", type=float, nargs=2, default=(0.5, 3.5), metavar=("MIN", "MAX"),
                   help="Approach forward-speed range (m/s) sampled per box-tile clip.")
    p.add_argument("--distractor-clips-per-tile", type=int, default=6)
    p.add_argument("--flat-clip-length", type=float, default=9.0)
    # --- composed distribution (per-family tile counts + clips/tile cap) ---
    p.add_argument("--mix-stairs", type=int, default=40)
    p.add_argument("--mix-boxes", type=int, default=20)
    p.add_argument("--mix-distractors", type=int, default=30)
    p.add_argument("--mix-flat", type=int, default=60)
    p.add_argument("--flat-clips", type=int, default=300)
    p.add_argument("--max-clips-per-tile", type=int, default=0, help="0 = all (full training).")
    p.add_argument("--skip", default="", help="Comma-sep families to skip (stairs,boxes,distractors,flat).")
    args = p.parse_args()

    if args.quick:                                          # shrink everything for a fast pipeline check
        args.bones_limit = 1500
        args.pure_tiles, args.transition_tiles = 1, 1
        args.transition_vels = "0.8;1.5"
        args.box_tiles, args.box_clips_per_tile = 2, 2
        args.distractor_clips_per_tile = 3
        args.flat_clip_length = 6.0
        args.mix_stairs = args.mix_boxes = args.mix_distractors = args.mix_flat = 4
        args.flat_clips, args.max_clips_per_tile = 40, 4

    os.makedirs(os.path.join(_REPO, args.work_dir), exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    common = ["--source", args.source, "--bones-cache", args.bones_cache, "--device", args.device]
    bl = ["--bones-limit", str(args.bones_limit)]
    fam = {                                                 # family -> output .h5
        k: os.path.join(args.work_dir, f"family_{k}.h5")
        for k in ("stairs", "stairs_up", "boxes", "jump_off", "jump_off_front", "distractors", "flat")
    }
    box_vr = ["--box-vel-range", str(args.box_vel_range[0]), str(args.box_vel_range[1])]

    t0 = time.time()
    if "stairs" not in skip:
        _run("stairs", "generate_box_clips.py", common + bl + [
            "--terrain-lib", args.terrain_lib, "--terrain-skills", "stairs_down",
            "--n-pure-tiles", str(args.pure_tiles), "--n-transition-tiles", str(args.transition_tiles),
            "--transition-vels", args.transition_vels, "--tile-seed", str(args.seed),
            "--stair-floor-frac", str(args.stair_floor_frac),
            "--out", fam["stairs"]])
    if "stairs_up" not in skip:
        # Ascent: PURE tiles only (transition-ascent peel-off is a TODO -> raises); the riser x tread
        # grid gives the geometry variety.  Pure is matcher-free.
        _run("stairs_up", "generate_box_clips.py", common + bl + [
            "--terrain-lib", args.terrain_lib, "--terrain-skills", "stairs_up",
            "--n-pure-tiles", str(args.pure_tiles), "--n-transition-tiles", "0",
            "--tile-seed", str(args.seed), "--stair-floor-frac", str(args.stair_floor_frac),
            "--out", fam["stairs_up"]])
    if "boxes" not in skip:
        _run("boxes", "generate_box_clips.py", common + bl + [
            "--terrain-lib", args.terrain_lib, "--terrain-skills", "jump_on",
            "--num-box-tiles", str(args.box_tiles), "--box-clips-per-tile", str(args.box_clips_per_tile),
            "--tile-seed", str(args.seed), "--seed", str(args.seed), "--out", fam["boxes"]] + box_vr)
    for off in ("jump_off", "jump_off_front"):              # DISMOUNTS: spawn-on-box -> play -> DP walk-away
        if off not in skip:
            _run(off, "generate_box_clips.py", common + bl + [
                "--terrain-lib", args.terrain_lib, "--terrain-skills", off,
                "--num-box-tiles", str(args.box_tiles), "--box-clips-per-tile", str(args.box_clips_per_tile),
                "--tile-seed", str(args.seed), "--seed", str(args.seed), "--out", fam[off]] + box_vr)
    if "distractors" not in skip:
        _run("distractors", "generate_distractor_clips.py", common + bl + [
            "--clips-per-tile", str(args.distractor_clips_per_tile), "--seed", str(args.seed),
            "--out", fam["distractors"]])
    if "flat" not in skip:
        _run("flat", "generate_mm_clips.py", [
            "--source", args.source, "--bones-cache", args.bones_cache, "--device", args.device,
            "--bones-limit", str(args.bones_limit), "--clip-length", str(args.flat_clip_length),
            "--out", fam["flat"]])

    # --- compose the generated families into ONE training library ---
    # Each new stair/box skill mixes at its cousin's proportion (stairs_up ~ stairs; dismounts ~ boxes).
    mix = ", ".join(f"{fam[k]}:{c}" for k, c in
                    (("stairs", args.mix_stairs), ("stairs_up", args.mix_stairs),
                     ("boxes", args.mix_boxes), ("jump_off", args.mix_boxes),
                     ("jump_off_front", args.mix_boxes),
                     ("distractors", args.mix_distractors)) if k not in skip)
    compose = ["--mix", mix, "--max-clips-per-tile", str(args.max_clips_per_tile),
               "--out", args.out, "--device", args.device]
    if "flat" not in skip:
        compose += ["--flat-from", fam["flat"], "--flat-tiles", str(args.mix_flat),
                    "--flat-clips", str(args.flat_clips)]
    _run("compose", "compose_terrain_library.py", compose)

    print(f"\n===== terrain library written: {args.out}  (total {time.time() - t0:.0f}s) =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
