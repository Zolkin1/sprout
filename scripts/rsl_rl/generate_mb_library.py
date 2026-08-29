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
c     ``mb_assemble_library.py``             env_isaaclab   per-family TRANSITION + box ``.h5``
c     ``mb_pure_stair_clips.py``             env_isaaclab   the PURE stair families (no MB needed)
c     ``generate_mb_clips.py``               env_isaaclab   the flat ``.h5``
d     ``compose_terrain_library.py``         env_isaaclab   ONE training library
===== ===================================== ============== =========================================

Every stage is skip-if-done, so a killed run resumes where it stopped, and ``--stage`` runs one stage alone
(e.g. to run stage b on the machine that has the GPU). Stage b is invoked through ``conda run``; each of its
two scripts loads the model ONCE and then generates everything it owns, which is the difference between one
checkpoint load and thousands.

The DISTRACTOR family is reachable from no recipe and no stage here. Its 20 tiles went to terrain
(see configs/clip_gen/g1_mb_v6.yaml): the visual-clutter pass now delivers "a box you see but do
not touch" on every tile rather than on 10% of them, at higher diversity and with no motion
generation at all, and the family's clips were anchor_mask = 0 (fully world-pinned) besides.
``mb_distractor_clips.py`` / ``motionbricks/distractor_plan.py`` / ``generate_distractor_clips.py``
stay on disk: the first is the second consumer of ``flat_plan.plan_from_holds``, so deleting it
would take that function's only non-flat test coverage with it.

PURE stair tiles need no Motion Bricks -- a pure tile is one long platform-less staircase with no approach
or exit, so there is nothing to in-between and ``mb_plan_library`` hardcodes ``n_pure=0``. They are built in
stage c by ``mb_pure_stair_clips.py`` (matcher-free, reusing the same ``place_periodic_on_tile`` the
transition path uses) and folded into stage d automatically. They previously came from the DEPRECATED
``generate_box_clips.py`` and had to be stapled on with ``--extra-mix``, which is how the first v3 build
silently shipped ZERO pure stairs; ``--extra-mix`` is still there for genuinely external families.
Example::

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
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_gen_config import load_clip_gen_config  # noqa: E402

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


def _segments_planned(work: str) -> int:
    """How many unique segments stage a asked for (-1 = stage a has not run).

    Deliberately NOT "how many are missing". This used to count absent files and skip the whole
    segment stage when none were, which silently defeated ``mb_terrain_segments._is_stale``: that
    check compares each segment's recorded generation knobs (and ``_GEN_VERSION``) against the
    current ones and regenerates on a mismatch, and it can only do that if it RUNS. Gating on
    existence meant a knob change -- or a bump of the very version counter that exists to force
    regeneration -- reused every cached segment without a word. Measured: bumping _GEN_VERSION
    4 -> 5 regenerated nothing.

    So the stage is now always invoked when stage a has run, and the batch decides for itself
    what to reuse. A pass with nothing to do costs one model load (~5 s), which is the price of
    the staleness guard being real."""
    req = os.path.join(work, "requests.json")
    if not os.path.exists(req):
        return -1
    with open(req) as f:
        return len(json.load(f)["requests"])


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
    p.add_argument("--config", required=True,
                   help="Clip-library recipe (configs/clip_gen/*.yaml). Holds every scale and "
                        "distribution parameter; see clip_gen_config.py. Loading is STRICT -- an "
                        "unknown or missing key is an error, not a default.")
    args = p.parse_args()
    cfg = load_clip_gen_config(args.config)
    print(f"[mb] recipe {cfg.config_path}")

    work = args.work_dir
    fam_dir = args.families_dir or os.path.join(work, "families")
    bank = args.stride_bank or os.path.join(work, "stride_bank.npz")
    raw_flat = os.path.join(work, "raw_flat")
    flat_h5 = os.path.join(fam_dir, "family_flat.h5")
    os.makedirs(os.path.join(_REPO, work), exist_ok=True)
    # Copy the recipe next to the artifacts it produced. A work dir is resumable and long-lived, so
    # "which parameters built this" has to travel with it rather than living only in a shell history.
    shutil.copyfile(cfg.config_path, os.path.join(_REPO, work, "clip_gen_config.yaml"))
    os.makedirs(os.path.join(_REPO, fam_dir), exist_ok=True)
    stage, cpt = args.stage, int(cfg.clips_per_tile)
    box_on, box_off, box_off_front = (int(x) for x in cfg.box_tiles)
    n_flat_clips = cfg.flat_tiles * cpt
    t0 = time.time()

    # --- stage 0: the stride bank the MB pose pin is driven from -------------------------------------
    if stage in ("all", "0") and not os.path.exists(os.path.join(_REPO, bank)):
        _run("bank", "export_stride_bank.py",
             ["--library", args.library, "--out", bank, "--dt", "0.02"])

    # --- stage a: tiles + plans + deduped segment requests (env_isaaclab) ---------------------------
    if stage in ("all", "a") and not os.path.exists(os.path.join(_REPO, work, "jobs.json")):
        _run("plan", "mb_plan_library.py", [
            "--work-dir", work, "--source", args.source, "--library", args.library,
            # Pass A needs the BANK now: a terrain clip's lead-up is a loop-per-command alternation,
            # and choosing each loop's stride and gait phase is a pass-A decision.
            "--stride-bank", bank,
            "--terrain-cmd-changes-per-side", str(cfg.terrain_cmd_changes_per_side),
            "--terrain-cmd-segments", str(cfg.terrain_cmd_segments),
            "--terrain-cmd-vy-range", *(str(v) for v in cfg.terrain_cmd_vy_range),
            "--terrain-cmd-wz-range", *(str(v) for v in cfg.terrain_cmd_wz_range),
            "--min-hold-m", str(cfg.min_hold_m),
            # The SAME three the segment stage gets: pass A predicts a transient's travel with them,
            # pass B generates with them, and one recipe key feeds both.
            "--approach-a-max", str(cfg.approach_a_max),
            "--approach-min-transient-s", str(cfg.approach_min_transient_s),
            "--approach-hop-slack", str(cfg.approach_hop_slack),
            "--flat-ramp-dv", str(cfg.flat_ramp_dv),
            "--stair-tiles", str(cfg.stair_tiles), "--box-tiles", ",".join(str(x) for x in cfg.box_tiles),
            "--clips-per-tile", str(cpt), "--seed", str(cfg.seed), "--device", args.device,
            *([] if cfg.segment_reuse else ["--no-segment-reuse"]),
            "--stair-vel-range", *(str(v) for v in cfg.stair_vel_range),
            "--box-vel-range", *(str(v) for v in cfg.box_vel_range),
            "--box-top-vx-range", *(str(v) for v in cfg.box_top_vx_range),
            "--box-two-sided-frac", str(cfg.box_two_sided_frac),
            "--box-nav-width", str(cfg.box_nav_width),
            "--backward-vx-max", str(cfg.backward_vx_max),
            "--stair-floor-frac", str(cfg.stair_floor_frac),
            "--stair-float-frac", str(cfg.stair_float_frac),
            "--stair-float-thick-range", *(str(v) for v in cfg.stair_float_thick_range),
            "--steady-approach-frac", str(cfg.steady_approach_frac),
            "--approach-flat-s-range", *(str(v) for v in cfg.approach_flat_s_range),
            "--exit-flat-s-range", *(str(v) for v in cfg.exit_flat_s_range),
            "--flat-dist-clamp", *(str(v) for v in cfg.flat_dist_clamp),
            "--box-top-dist-clamp", *(str(v) for v in cfg.box_top_dist_clamp),
            "--dist-step", str(cfg.dist_step),
            "--platform-buffer-m", str(cfg.platform_buffer_m),
            "--n-steps-range", *(str(int(v)) for v in cfg.n_steps_range),
            "--tile-width-range", *(str(v) for v in cfg.tile_width_range),
            "--entry-pick", cfg.entry_pick, "--exit-pick", cfg.exit_pick,
            "--contact-tol", str(cfg.contact_tol),
            "--footprint-margin", str(cfg.footprint_margin),
            # null means "derive from the riser", which is the argparse default -- so pass the flag
            # only when the recipe states a number. Sending "None" would be a parse error.
            *([] if cfg.swing_clear is None else ["--swing-clear", str(cfg.swing_clear)]),
            *([] if cfg.swing_band is None else ["--swing-band", str(cfg.swing_band)])])

    # --- stage b: EVERYTHING that needs the model, one load per script (motionbricks env) -----------
    if stage in ("all", "b"):
        mb_root = _resolve_mb_root(args.mb_root)
        args.mb_root = mb_root
        print(f"[mb] motionbricks root: {mb_root}")
        planned = _segments_planned(os.path.join(_REPO, work))
        if planned < 0:
            raise SystemExit(f"[segments] no requests.json in {work}; run stage a first")
        _run("segments", "mb_terrain_segments.py", [
            "--mb-root", args.mb_root, "--stride-bank", bank,
            "--requests", os.path.join(work, "requests.json"), "--work-dir", work,
            "--a-max", str(cfg.approach_a_max),
            "--min-transient-s", str(cfg.approach_min_transient_s),
            "--hop-slack", str(cfg.approach_hop_slack),
            # The INTERIOR lead-up transients are ordinary loop-to-loop hops and take the FLAT
            # family's chaining bound, not the terrain boundary's -- same generator, same knob.
            "--ramp-dv", str(cfg.flat_ramp_dv),
            # The interior lead-up transients are flat-style hops, so they take the FLAT family's
            # acceleration bound rather than the terrain boundary's -- measured cleaner there
            # (interior peak max 10.87 -> 3.88 m/s, one blow-up -> none).
            "--flat-a-max", str(cfg.flat_a_max),
            "--gen-attempts", str(cfg.gen_attempts),
            "--pin-future", cfg.pin_future],
            conda_env=args.conda_env)
        flat_plans = os.path.join(work, "flat_plans.json")
        flat_tr = os.path.join(work, "flat_transients")
        if cfg.flat_loops:
            # Plan the flat clips (pass A) then generate ONLY their transients (pass B). Steady state is
            # an exact looped stride replayed in pass C, so Motion Bricks runs ~6 times per clip instead
            # of ~37 and never produces the steady stretches it tracks poorly.
            if not os.path.exists(os.path.join(_REPO, flat_plans)):
                _run("flat-plan", "mb_flat_clips.py", [
                    "--stride-bank", bank, "--dump-plan", flat_plans,
                    "--n-clips", str(n_flat_clips), "--clip-length", str(cfg.flat_clip_length),
                    "--hold-s-range", *(str(v) for v in cfg.flat_hold_s_range),
                    # The DRIVER'S ENVELOPE, sampled continuously -- one range per channel, not a set
                    # of mode bands. The motion snaps to the nearest bank stride; the command does not.
                    "--vx-range", *(str(v) for v in cfg.flat_vx_range),
                    "--vy-range", *(str(v) for v in cfg.flat_vy_range),
                    "--wz-range", *(str(v) for v in cfg.flat_wz_range),
                    "--mass-stand", str(cfg.flat_mass_stand),
                    "--mass-sidestep", str(cfg.flat_mass_sidestep),
                    "--strata-width", str(cfg.flat_strata_width),
                    "--speed-quota", cfg.flat_speed_quota,
                    "--seed", str(cfg.seed)])
            _run("flat-transients", "mb_flat_transients.py", [
                "--mb-root", args.mb_root, "--stride-bank", bank, "--plans", flat_plans,
                "--out-dir", flat_tr, "--ramp-dv", str(cfg.flat_ramp_dv),
                "--a-max", str(cfg.flat_a_max), "--alpha-max", str(cfg.flat_a_max),
                "--min-transient-s", str(cfg.flat_min_transient_s),
                "--hop-slack", str(cfg.flat_hop_slack), "--seed", str(cfg.seed)],
                conda_env=args.conda_env)
        if not cfg.flat_loops:
            # The free-running mb_encode path is GONE, not merely off. It generated the flat family
            # as one long Motion Bricks rollout instead of exact looped strides plus bounded
            # transients, and its own knobs (flat_seg_dur_range, rl_vels) went with the command
            # envelope rewrite -- the schema no longer accepts them, so a recipe that asked for this
            # path would fail on a missing attribute somewhere far from the cause. Say so here.
            raise SystemExit(
                "flat_loops: false selects the mb_encode rollout path, which no longer exists. The "
                "flat family is exact looped strides plus bounded in-betweenings; there is no other "
                "arm. Remove the key or set it true.")
    # --- stage c: splice / ingest into per-family libraries (env_isaaclab) --------------------------
    if stage in ("all", "c"):
        if not glob.glob(os.path.join(_REPO, fam_dir, "family_stairs_*.h5")):
            _run("assemble", "mb_assemble_library.py", [
                "--work-dir", work, "--source", args.source, "--out-dir", fam_dir,
                "--device", args.device, "--workers", str(args.workers),
                # The bank is how pass C replays the loop that covers the approach's travel budget.
                "--stride-bank", bank,
                "--seams", cfg.seams, "--seam-blend-s", str(cfg.seam_blend_s),
                "--weld-search", str(cfg.weld_search),
                "--arrival-search", str(cfg.arrival_search)])
        # PURE stair tiles: no approach or exit, so Motion Bricks has nothing to in-between and
        # mb_plan_library hardcodes n_pure=0. They used to come from the DEPRECATED
        # generate_box_clips.py and be stapled on by hand with --extra-mix, which is how the first
        # v3 build silently shipped ZERO pure stairs. Generated here instead, from the same
        # --stair-vel-range as the transition tiles so the two stair families cannot drift apart.
        for direction in ("stairs_up", "stairs_down"):
            pure_h5 = os.path.join(fam_dir, f"family_pure_{direction}.h5")
            if cfg.pure_stair_tiles > 0 and not os.path.exists(os.path.join(_REPO, pure_h5)):
                _run(f"pure-{direction}", "mb_pure_stair_clips.py", [
                    "--source", args.source, "--terrain-lib", args.library,
                    "--terrain-skills", direction, "--out", pure_h5,
                    "--n-tiles", str(cfg.pure_stair_tiles),
                    "--clips-per-tile", str(cpt),
                    "--stair-vel-range", *(str(v) for v in cfg.stair_vel_range),
                    "--pure-seconds", str(cfg.pure_seconds),
                    "--stair-floor-frac", str(cfg.stair_floor_frac),
                    "--stair-float-frac", str(cfg.stair_float_frac),
                    "--stair-float-thick-range", *(str(v) for v in cfg.stair_float_thick_range),
                    "--tile-seed", str(cfg.seed), "--device", args.device,
                    "--end-buffer", str(cfg.pure_end_buffer),
                    "--tile-width-range", *(str(v) for v in cfg.pure_tile_width_range),
                    "--n-steps-range", *(str(int(v)) for v in cfg.pure_n_steps_range),
                    "--footprint-margin", str(cfg.pure_footprint_margin)])
        if cfg.flat_loops and not os.path.exists(os.path.join(_REPO, flat_h5)):
            _run("flat-clips", "mb_flat_clips.py", [
                "--assemble", "--stride-bank", bank,
                "--plans", os.path.join(work, "flat_plans.json"),
                "--transient-dir", os.path.join(work, "flat_transients"),
                "--source", args.source, "--out", flat_h5,
                "--clip-length", str(cfg.flat_clip_length),
                "--device", args.device, "--seed", str(cfg.seed),
                "--seam-blend-s", str(cfg.flat_seam_blend_s),
                "--flat-anchor", cfg.flat_anchor])
        # (the non-loop arm is refused in stage b, above -- there is no second flat path)
    # --- stage d: one training library at the requested tile proportions ---------------------------
    if stage in ("all", "d"):
        counts = {"stairs_up": cfg.stair_tiles, "stairs_down": cfg.stair_tiles,
                  "jump_on": box_on, "jump_off": box_off, "jump_off_front": box_off_front,
                  # PURE stair tiles, from mb_pure_stair_clips in stage c. Folded in HERE rather
                  # than left to --extra-mix: a whole terrain class silently missing from the
                  # library is not something a caller should have to remember to prevent.
                  "pure_stairs_up": cfg.pure_stair_tiles,
                  "pure_stairs_down": cfg.pure_stair_tiles}
        mix = [f"{fam_dir}/family_{k}.h5:{n}" for k, n in counts.items()
               if n > 0 and os.path.exists(os.path.join(_REPO, fam_dir, f"family_{k}.h5"))]
        for k, n in (("pure_stairs_up", cfg.pure_stair_tiles),
                     ("pure_stairs_down", cfg.pure_stair_tiles)):
            if n > 0 and not os.path.exists(os.path.join(_REPO, fam_dir, f"family_{k}.h5")):
                print(f"[compose][WARN] {k} requested ({n} tiles) but "
                      f"{fam_dir}/family_{k}.h5 is missing -- the library will have NO {k}.")
        if cfg.extra_mix.strip():
            mix += [e.strip() for e in cfg.extra_mix.split(",") if e.strip()]
        if not mix:
            raise SystemExit("[compose] no family libraries found; run stage c first")
        compose = ["--mix", ", ".join(mix), "--max-clips-per-tile", str(cpt),
                   "--out", args.out, "--device", args.device,
                   # Provenance: which solved library, which recipe, which seed. Without these the
                   # finished .h5 cannot say what produced it, and the local hf/ cache is never
                   # re-checked against the remote.
                   "--library", args.library, "--recipe", cfg.config_path,
                   "--seed", str(cfg.seed)]
        if os.path.exists(os.path.join(_REPO, flat_h5)):
            compose += ["--flat-from", flat_h5, "--flat-tiles", str(cfg.flat_tiles),
                        "--flat-clips", str(n_flat_clips)]
        _run("compose", "compose_terrain_library.py", compose)

    print(f"\n===== MB library: {args.out}  (total {time.time() - t0:.0f}s) =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
