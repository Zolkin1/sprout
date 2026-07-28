#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Top-level driver: build a whole Motion Bricks training library, across both conda environments.

The MB generator is split over two environments that cannot share a process -- ``motionbricks`` (GPU, drives
the model) and ``env_isaaclab`` (the clip machinery) -- so a library is four passes over one work dir:

===== ===================================== ============== =========================================
stage script                                env            product
===== ===================================== ============== =========================================
0     ``export_stride_bank.py``             env_isaaclab   stride bank (skipped if it exists)
a     ``mb_plan_library.py``                env_isaaclab   sampled tiles + per-clip plans + requests
b     ``mb_terrain_segments.py --requests``  motionbricks   every unique approach / exit segment
b     ``mb_encode.py --n-clips``             motionbricks   the flat family's raw motions
c     ``mb_assemble_library.py``             env_isaaclab   per-family terrain ``.h5``
c     ``generate_mb_clips.py``               env_isaaclab   the flat ``.h5``
d     ``compose_terrain_library.py``         env_isaaclab   ONE training library
===== ===================================== ============== =========================================

Every stage is skip-if-done, so a killed run resumes where it stopped, and ``--stage`` runs one stage alone
(e.g. to run stage b on the machine that has the GPU). Stage b is invoked through ``conda run``; each of its
two scripts loads the model ONCE and then generates everything it owns, which is the difference between one
checkpoint load and thousands.

PURE stair tiles and distractors need no Motion Bricks (a pure tile is a bare looped traversal, matcher-free)
-- generate those with ``generate_box_clips.py`` / ``generate_distractor_clips.py`` and fold them in with
``--extra-mix "path.h5:tiles, ..."``. Example::

    python scripts/rsl_rl/generate_mb_library.py \
        --work-dir mb_raw/lib_v1 --out clips/g1_mb_v1.h5 \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --mb-root ../GR00T-WholeBodyControl/motionbricks --quick
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))          # <repo>/scripts/rsl_rl -> <repo>


def _run(name: str, script: str, argv: list[str], conda_env: str | None = None) -> None:
    """Run one stage as a subprocess (repo-root cwd, PYTHONPATH set); raise on failure."""
    env = dict(os.environ)
    env["PYTHONPATH"] = (f"{_REPO}/source/robot_rl:{_REPO}/source/motion_gen"
                         + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""))
    path = os.path.join(_HERE, script)
    cmd = (["conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u", path]
           if conda_env else [sys.executable, "-u", path]) + argv
    print(f"\n===== [{name}] {'(' + conda_env + ') ' if conda_env else ''}" + " ".join(argv), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=_REPO, env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"[{name}] FAILED (rc={r.returncode}) after {dt:.0f}s")
    print(f"===== [{name}] done in {dt:.0f}s", flush=True)


def _resolve_mb_root(arg: str) -> str:
    """Absolute path to the Motion Bricks clone, given a possibly repo-relative ``--mb-root``.

    The clone is a SIBLING of the repo, so a relative default silently resolves to the wrong place when the
    repo is a git worktree (``<repo>/.claude/worktrees/<branch>``). Try the argument as given, then walk up
    from the repo looking for the clone, and pass the absolute path on -- the Motion Bricks harness chdirs
    into it, so a relative path is a trap either way.
    """
    cand = arg if os.path.isabs(arg) else os.path.join(_REPO, arg)
    if os.path.isdir(cand):
        return os.path.abspath(cand)
    tail = os.path.join("GR00T-WholeBodyControl", "motionbricks")
    here = _REPO
    for _ in range(6):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, tail)):
            return os.path.join(here, tail)
    raise SystemExit(f"[mb] cannot find the Motion Bricks clone: --mb-root {arg!r} does not exist and no "
                     f"{tail} was found above {_REPO}. Pass --mb-root explicitly.")


def _n_schedules(path: str) -> int:
    """How many distractor motions the schedule file asks for (0 if it has not been written yet)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return len(json.load(f))


def _segments_pending(work: str) -> int:
    """How many unique segments are still missing (0 = stage b's terrain half is done)."""
    req = os.path.join(work, "requests.json")
    if not os.path.exists(req):
        return -1
    with open(req) as f:
        reqs = json.load(f)["requests"]
    return sum(0 if os.path.exists(os.path.join(work, r["out"])) else 1 for r in reqs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True, help="Shared work dir (plans/, segments/, raw_flat/).")
    p.add_argument("--out", required=True, help="Final composed training library .h5.")
    p.add_argument("--source", required=True, help="Traj-opt library root (channel layout).")
    p.add_argument("--mb-root", default="../GR00T-WholeBodyControl/motionbricks",
                   help="GR00T motionbricks package root (stage b only).")
    p.add_argument("--conda-env", default="motionbricks", help="Conda env that has Motion Bricks.")
    p.add_argument("--library", default="trajectories/skills/library_v2")
    p.add_argument("--stride-bank", default=None, help="Stride bank .npz (default <work-dir>/stride_bank.npz).")
    p.add_argument("--families-dir", default=None, help="Where the per-family .h5 go (default <work-dir>/families).")
    p.add_argument("--stage", default="all", choices=["all", "0", "a", "b", "c", "d"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="Small/fast scale for an end-to-end check.")
    # --- scale (DP parity by default: 200 tiles x 50 clips = 10k) ---
    p.add_argument("--clips-per-tile", type=int, default=50)
    p.add_argument("--stair-tiles", type=int, default=20, help="TRANSITION tiles per direction (up/down).")
    p.add_argument("--box-tiles", default="20,15,15", help="jump_on,jump_off,jump_off_front tile counts.")
    p.add_argument("--flat-tiles", type=int, default=60)
    p.add_argument("--distractor-tiles", type=int, default=20,
                   help="Distractor tiles (a box SEEN but not touched). MB-generated: no DP anywhere.")
    p.add_argument("--distractor-s", type=float, default=6.0, help="Distractor clip duration (s).")
    p.add_argument("--flat-clip-length", type=float, default=20.0,
                   help="Flat clip duration (s). 20 s matches the teacher's episode_length_s, so one clip "
                        "is one episode; it is passed to BOTH the MB generation and the Stage-2 ingest.")
    p.add_argument("--stair-vel-range", type=float, nargs=2, default=(0.6, 2.6), metavar=("MIN", "MAX"))
    p.add_argument("--box-vel-range", type=float, nargs=2, default=(0.5, 3.5), metavar=("MIN", "MAX"),
                   help="GROUND-level speed band for the box families (the run-up / walk-away).")
    p.add_argument("--box-top-s-range", type=float, nargs=2, default=(2.0, 3.0), metavar=("MIN", "MAX"),
                   help="Duration (s) of a box clip's ELEVATED segment -- the walk along the box top after "
                        "a mount / before a dismount. Sizes the platform authored flush with the box top; "
                        "do not go below ~2 s (a mount's exit needs that long to get walking from its "
                        "standing seed -- see mb_plan_library for the measurement).")
    p.add_argument("--box-top-vx-range", type=float, nargs=2, default=(0.3, 0.7), metavar=("MIN", "MAX"),
                   help="Speed band (m/s) of a box clip's ELEVATED segment (see --box-top-s-range); "
                        "platform length is roughly their product.")
    p.add_argument("--box-nav-width", type=float, default=-1.0,
                   help="y-width (m) the box + its top platform are widened to; -1 = the tile's corridor "
                        "width, 0 = the reference box's own narrow extents.")
    p.add_argument("--backward-vx-max", type=float, default=1.0,
                   help="Cap (m/s) on any BACKWARD command (the jump_off dismount), where the clean "
                        "library's walk_backwards strides stop.")
    p.add_argument("--stair-floor-frac", type=float, default=0.5)
    p.add_argument("--steady-approach-frac", type=float, default=0.5,
                   help="Fraction of terrain clips whose approach starts ALREADY AT SPEED rather than from "
                        "a stand (see mb_terrain_segments --approach-start).")
    p.add_argument("--pin-lead", type=float, default=1.0,
                   help="FLOOR (s) on when the terrain pose pin engages. The primary trigger is now "
                        "proximity (mb_terrain_segments --pin-dist), so this is only the safety net for a "
                        "robot that never closes the distance. It used to be 2.0, which combined with "
                        "--approach-s-range (1.5,3.5) meant the pin was on from frame ONE on most clips -- "
                        "no stride keyframes at all, and the model holding a standing pre-jump crouch as "
                        "its target through the whole run-up.")
    p.add_argument("--rl-vels", default="raw", choices=["raw", "snapped"],
                   help="Which command channel the flat clips expose to the RL (see generate_mb_clips.py).")
    p.add_argument("--extra-mix", default="",
                   help="Extra 'path.h5:tiles, ...' folded into the compose (pure stairs, distractors).")
    args = p.parse_args()

    if args.quick:
        args.clips_per_tile, args.stair_tiles, args.box_tiles = 4, 2, "2,1,1"
        args.flat_tiles, args.flat_clip_length = 4, 20.0
        args.distractor_tiles = 3

    work = args.work_dir
    fam_dir = args.families_dir or os.path.join(work, "families")
    bank = args.stride_bank or os.path.join(work, "stride_bank.npz")
    raw_flat = os.path.join(work, "raw_flat")
    raw_dis = os.path.join(work, "raw_distractor")
    dis_sched = os.path.join(work, "distractor_schedules.json")
    flat_h5 = os.path.join(fam_dir, "family_flat.h5")
    dis_h5 = os.path.join(fam_dir, "family_distractor.h5")
    os.makedirs(os.path.join(_REPO, work), exist_ok=True)
    os.makedirs(os.path.join(_REPO, fam_dir), exist_ok=True)
    stage, cpt = args.stage, int(args.clips_per_tile)
    dis_argv = ["--source", args.source, "--num-tiles", str(args.distractor_tiles),
                "--clips-per-tile", str(cpt), "--short-s", str(args.distractor_s),
                "--seed", str(args.seed), "--device", args.device]
    box_on, box_off, box_off_front = (int(x) for x in args.box_tiles.split(","))
    n_flat_clips = args.flat_tiles * cpt
    t0 = time.time()

    # --- stage 0: the stride bank the MB pose pin is driven from -------------------------------------
    if stage in ("all", "0") and not os.path.exists(os.path.join(_REPO, bank)):
        _run("bank", "export_stride_bank.py",
             ["--library", args.library, "--out", bank, "--dt", "0.02"])

    # --- stage a: tiles + plans + deduped segment requests (env_isaaclab) ---------------------------
    if stage in ("all", "a") and not os.path.exists(os.path.join(_REPO, work, "jobs.json")):
        _run("plan", "mb_plan_library.py", [
            "--work-dir", work, "--source", args.source, "--library", args.library,
            "--stair-tiles", str(args.stair_tiles), "--box-tiles", args.box_tiles,
            "--clips-per-tile", str(cpt), "--seed", str(args.seed), "--device", args.device,
            "--stair-vel-range", *(str(v) for v in args.stair_vel_range),
            "--box-vel-range", *(str(v) for v in args.box_vel_range),
            "--box-top-s-range", *(str(v) for v in args.box_top_s_range),
            "--box-top-vx-range", *(str(v) for v in args.box_top_vx_range),
            "--box-nav-width", str(args.box_nav_width),
            "--backward-vx-max", str(args.backward_vx_max),
            "--stair-floor-frac", str(args.stair_floor_frac),
            "--steady-approach-frac", str(args.steady_approach_frac)])
    if stage in ("all", "a") and args.distractor_tiles > 0 \
            and not os.path.exists(os.path.join(_REPO, dis_sched)):
        _run("distract-plan", "mb_distractor_clips.py", dis_argv + ["--dump-schedules", dis_sched])

    # --- stage b: EVERYTHING that needs the model, one load per script (motionbricks env) -----------
    if stage in ("all", "b"):
        mb_root = _resolve_mb_root(args.mb_root)
        args.mb_root = mb_root
        print(f"[mb] motionbricks root: {mb_root}")
        pending = _segments_pending(os.path.join(_REPO, work))
        if pending < 0:
            raise SystemExit(f"[segments] no requests.json in {work}; run stage a first")
        if pending:
            _run("segments", "mb_terrain_segments.py", [
                "--mb-root", args.mb_root, "--stride-bank", bank,
                "--requests", os.path.join(work, "requests.json"), "--work-dir", work,
                "--pin-lead", str(args.pin_lead)],
                conda_env=args.conda_env)
        else:
            print(f"\n===== [segments] all present, skipping")
        n_have = len(glob.glob(os.path.join(_REPO, raw_flat, "*.npz")))
        if n_have < n_flat_clips:
            _run("flat-gen", "mb_encode.py", [
                "--mb-root", args.mb_root, "--stride-bank", bank, "--out-dir", raw_flat,
                "--n-clips", str(n_flat_clips), "--clip-length", str(args.flat_clip_length),
                "--seed", str(args.seed)], conda_env=args.conda_env)
        else:
            print(f"\n===== [flat-gen] {n_have} raw flat motions present, skipping")
        if args.distractor_tiles > 0:
            n_want = _n_schedules(os.path.join(_REPO, dis_sched))
            n_got = len(glob.glob(os.path.join(_REPO, raw_dis, "*.npz")))
            if n_got < n_want:
                _run("distract-gen", "mb_encode.py", [
                    "--mb-root", args.mb_root, "--stride-bank", bank, "--out-dir", raw_dis,
                    "--schedules", dis_sched, "--seed", str(args.seed)], conda_env=args.conda_env)
            else:
                print(f"\n===== [distract-gen] {n_got} raw distractor motions present, skipping")

    # --- stage c: splice / ingest into per-family libraries (env_isaaclab) --------------------------
    if stage in ("all", "c"):
        if not glob.glob(os.path.join(_REPO, fam_dir, "family_stairs_*.h5")):
            _run("assemble", "mb_assemble_library.py", [
                "--work-dir", work, "--source", args.source, "--out-dir", fam_dir,
                "--device", args.device, "--workers", str(args.workers)])
        if not os.path.exists(os.path.join(_REPO, flat_h5)):
            _run("flat-clips", "generate_mb_clips.py", [
                "--raw-dir", raw_flat, "--source", args.source, "--out", flat_h5,
                "--clip-length", str(args.flat_clip_length), "--device", args.device,
                "--rl-vels", args.rl_vels, "--workers", str(args.workers)])
        if args.distractor_tiles > 0 and not os.path.exists(os.path.join(_REPO, dis_h5)):
            _run("distract-clips", "mb_distractor_clips.py",
                 dis_argv + ["--mb-raw-dir", raw_dis, "--out", dis_h5, "--workers", str(args.workers)])

    # --- stage d: one training library at the requested tile proportions ---------------------------
    if stage in ("all", "d"):
        counts = {"stairs_up": args.stair_tiles, "stairs_down": args.stair_tiles,
                  "jump_on": box_on, "jump_off": box_off, "jump_off_front": box_off_front,
                  "distractor": args.distractor_tiles}
        mix = [f"{fam_dir}/family_{k}.h5:{n}" for k, n in counts.items()
               if n > 0 and os.path.exists(os.path.join(_REPO, fam_dir, f"family_{k}.h5"))]
        if args.extra_mix.strip():
            mix += [e.strip() for e in args.extra_mix.split(",") if e.strip()]
        if not mix:
            raise SystemExit("[compose] no family libraries found; run stage c first")
        compose = ["--mix", ", ".join(mix), "--max-clips-per-tile", str(cpt),
                   "--out", args.out, "--device", args.device]
        if os.path.exists(os.path.join(_REPO, flat_h5)):
            compose += ["--flat-from", flat_h5, "--flat-tiles", str(args.flat_tiles),
                        "--flat-clips", str(n_flat_clips)]
        _run("compose", "compose_terrain_library.py", compose)

    print(f"\n===== MB library: {args.out}  (total {time.time() - t0:.0f}s) =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
