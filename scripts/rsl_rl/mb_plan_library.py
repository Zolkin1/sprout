#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""PASS A (env_isaaclab): sample terrain TILES, plan every clip on them, and dedupe the MB segment work.

``mb_terrain_clips.py`` plans ONE traversal per invocation; a training library needs thousands, spread over
sampled tile geometries, with many clips SHARING each tile (the RL importer round-robins envs over tiles, so
a tile is a reuse unit). This pass does the whole env_isaaclab half of that up front:

1. **Sample tiles** with the same samplers + ranges the DP generator uses
   (:func:`~motion_gen.motion_matching.terrain_sampler.sample_stair_tiles` /
   :func:`~motion_gen.motion_matching.terrain_sampler.sample_box_tiles`), so the MB library's tile
   population is statistically the DP library's.
2. **Plan every clip** via ``mb_terrain_clips.plan_from_spec`` (one reference selection + placement +
   boundary gate per clip), writing one :class:`~motion_gen.motionbricks.terrain_plan.TerrainPlan` each.
3. **Dedupe the segment requests.** A Motion Bricks approach / exit does NOT depend on the tile -- only on
   the boundary pose it must reach or leave, the commands around it, and the surface it walks on. So two
   clips on different tiles that share a reference, boundary frame, speed and duration need the SAME
   segment generated ONCE. The key hashes the replay window MB will actually be conditioned on plus those
   scalars, so it can never over-merge: identical key == identical MB input.

Emits ``jobs.json`` (clip -> tile + plan + segment keys), ``requests.json`` (unique segments for pass B,
each naming a representative plan + which side to generate) and ``plans/``. Run pass B in the Motion Bricks
env, then pass C (``mb_assemble_library.py``) back here. Example::

    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" \
    env_isaaclab/bin/python scripts/rsl_rl/mb_plan_library.py \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --work-dir mb_raw/lib_v1 --stair-tiles 20 --box-tiles 20,15,15 --clips-per-tile 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mb_terrain_clips import (                                            # noqa: E402
    ingest_family_sources,
    make_builder,
    plan_from_spec,
)

from motion_gen.motion_matching.terrain_sampler import sample_box_tiles, sample_stair_tiles  # noqa: E402
from motion_gen.motionbricks import family_for, save_terrain_plan                            # noqa: E402

# Frames of the replay hashed into a segment key: MB is conditioned on a ~1 s window around the boundary
# frame (its planning horizon / context), so a window this long covers everything the model can see.
_KEY_WINDOW = 64
_STAIR_SKILLS = ("stairs_up", "stairs_down")


def _quantize(v: float, step: float) -> float:
    """Snap a sampled duration onto a grid.

    Segment reuse hinges on this: an MB approach depends on (reference, speed, DURATION), so drawing the
    duration continuously would make every tile's approach unique and the bank pointless. A grid keeps the
    duration variety that matters (platform length / lead-in feel) while letting tiles share segments.
    """
    return float(round(float(v) / float(step)) * float(step)) if step > 0 else float(v)


def _skill_seed(skill: str) -> int:
    """Stable per-family RNG stream id. ``hash(str)`` is salted per process, so it must NOT be used here --
    the sampled tiles have to be identical for the same ``--seed`` on every run."""
    return int.from_bytes(hashlib.sha1(skill.encode()).digest()[:4], "big") % (2 ** 31)


def _round_key(*vals) -> str:
    """Stable short hash of the given strings / scalars / arrays (rounded, so float noise cannot split a key).

    Deliberately does NOT use ``hash()``: Python salts string hashing per process, so a key built that way
    would change between runs and break both segment reuse and resume.
    """
    h = hashlib.sha1()
    for v in vals:
        if isinstance(v, str):
            h.update(v.encode())
        else:
            h.update(np.ascontiguousarray(np.round(np.asarray(v, dtype=np.float64), 4)).tobytes())
    return h.hexdigest()[:16]


def _segment_keys(plan, approach_pin: str, exit_pin: str) -> tuple[str, str]:
    """``(approach_key, exit_key)`` -- identical keys iff Motion Bricks would see identical input.

    Keyed on exactly what ``mb_terrain_segments`` feeds the model, no more:

    * A **hard** pin conditions the model on a replay window, so that window is hashed. Both sides are
      SURFACE-RELATIVE in the generator (``approach_pin_window`` subtracts ``approach_surf_z``;
      ``_generate_exit`` subtracts ``exit_surf_z`` from the seed, because Motion Bricks' height prior is
      absolute), so tile HEIGHT is deliberately not part of either key -- two tiles differing only in step
      count share a segment.
    * A **soft** pin sees no replay at all (a free steady walk from the stride bank), so only the commands,
      duration and mode matter.

    Anything the tile varies that the model never sees -- width, floor, lateral offset, ``terrain_id``,
    placement -- is absent from both keys by construction.
    """
    rep, ef, xf = plan.terrain_qpos, plan.entry_frame, plan.exit_frame
    a_scalars = [plan.approach_vx, plan.entry_vx, plan.approach_s]
    x_scalars = [plan.exit_vx, plan.exit_speed, plan.exit_s]
    a_parts = [f"{plan.skill}|{plan.approach_mode}|{approach_pin}|approach", a_scalars]
    x_parts = [f"{plan.skill}|{plan.exit_mode}|{exit_pin}|exit", x_scalars]
    if approach_pin == "hard":
        a_win = rep[max(0, ef - 1):ef + _KEY_WINDOW].copy()
        a_win[:, 2] -= plan.approach_surf_z                 # as approach_pin_window supplies it
        a_win[:, 0:2] = 0.0                                 # planar target comes from the exact-velocity pin
        a_parts.append(a_win)
    if exit_pin == "hard":
        x_win = rep[max(0, xf - _KEY_WINDOW):xf + 1].copy()
        x_win[:, 0:2] -= x_win[-1:, 0:2]                    # the seed is used relative to its own end
        x_win[:, 2] -= plan.exit_surf_z                     # as _generate_exit shifts it
        x_parts.append(x_win)
    return _round_key(*a_parts), _round_key(*x_parts)


def _stair_tile_specs(rng, skill: str, sources, args) -> list[dict]:
    """Sampled TRANSITION stair tiles for one direction -> per-tile arg dicts.

    Only transition tiles come through the MB pipeline: a PURE stair tile is a bare looped traversal with no
    approach or exit, so it needs no Motion Bricks at all and is generated directly by the DP generator's
    matcher-free path. The ``(riser, tread)`` grid is the set actually SOLVED in the reference library --
    feet only land on treads whose geometry the played stride was solved for.
    """
    hd_grid = sorted({(round(float(s.terrain_meta["terrain"]["riser"]), 4),
                       round(float(s.terrain_meta["terrain"]["tread"]), 4)) for s in sources})
    tiles = sample_stair_tiles(
        rng, hd_grid=hd_grid, n_pure=0, n_transition=int(args.stair_tiles),
        pure_length=1.0, transition_length=12.0,
        width_range=tuple(args.tile_width_range), num_steps_range=tuple(args.n_steps_range),
        lead_in_range=(1.5, 4.0), end_platform=1.0, inter_flat_range=(3.0, 5.0),
        nav_width_range=tuple(args.tile_width_range), flat_width_margin=0.3,
        single_step_frac=0.0, floor_frac=float(args.stair_floor_frac),
    )
    print(f"[mb_plan] {skill}: hd_grid={hd_grid} -> {len(tiles)} transition tiles")
    return [{"riser": float(t.step_height), "tread": float(t.step_depth),
             "n_steps": int(t.stair_steps[0]), "width": float(t.width),
             "nav_width": float(t.nav_width), "has_floor": bool(t.has_floor)} for t in tiles]


def _box_tile_specs(rng, skill: str, sources, args, n_tiles: int) -> list[dict]:
    """Sampled box tiles for one jump family -> per-tile arg dicts (a target box height each).

    The MB tile author places the single walkable box at the reference's SOLVED extents, so a tile varies
    only by which box HEIGHT it targets (the nearest reference is then selected). The DP path's support
    variants and lateral box offsets have no MB tile-author equivalent yet -- see the module TODO.
    """
    heights = sorted({round(float(s.terrain_meta["terrain"].get("box_height", 0.0)), 4) for s in sources})
    heights = [h for h in heights if h > 0.0] or [0.45]
    specs = sample_box_tiles(rng, n_tiles=n_tiles, n_sources=len(heights), variant_fracs=None,
                             nav_width_range=tuple(args.tile_width_range))
    print(f"[mb_plan] {skill}: box heights={heights} -> {len(specs)} tiles")
    return [{"box_height": heights[int(s.source_idx) % len(heights)]} for s in specs]


def _clip_speeds(rng, family_kind: str, args, cpt: int) -> list[float]:
    """Per-clip traversal/approach speeds for one tile.

    Stairs get EVENLY SPACED speeds over the band (mirroring the DP ``--transition-vels``, so a tile's clips
    tile the speed axis); boxes get uniform samples (mirroring the DP box path).
    """
    if family_kind == "stairs":
        lo, hi = args.stair_vel_range
        return [lo] if cpt <= 1 else [lo + (hi - lo) * i / (cpt - 1) for i in range(cpt)]
    lo, hi = args.box_vel_range
    return [float(rng.uniform(lo, hi)) for _ in range(cpt)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True, help="Shared work dir: plans/, requests.json, jobs.json.")
    p.add_argument("--source", required=True, help="Traj-opt library root (channel layout).")
    p.add_argument("--library", default="trajectories/skills/library_v2", help="Terrain/clean library folder.")
    p.add_argument("--hf-repo", default="zolkin/robot_rl")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--device", default="cpu")
    p.add_argument("--families", default="stairs_up,stairs_down,jump_on,jump_off,jump_off_front",
                   help="Comma-separated terrain families to plan.")
    p.add_argument("--stair-tiles", type=int, default=20, help="TRANSITION tiles per stair direction.")
    p.add_argument("--box-tiles", default="20,15,15", help="jump_on,jump_off,jump_off_front tile counts.")
    p.add_argument("--clips-per-tile", type=int, default=50)
    p.add_argument("--stair-vel-range", type=float, nargs=2, default=(0.6, 2.6), metavar=("MIN", "MAX"))
    p.add_argument("--box-vel-range", type=float, nargs=2, default=(0.5, 3.5), metavar=("MIN", "MAX"))
    p.add_argument("--approach-s-range", type=float, nargs=2, default=(1.5, 3.5), metavar=("MIN", "MAX"),
                   help="MB approach duration (s), sampled PER TILE (see --help note on geometry).")
    p.add_argument("--exit-s-range", type=float, nargs=2, default=(1.5, 3.5), metavar=("MIN", "MAX"))
    p.add_argument("--duration-step", type=float, default=0.5,
                   help="Quantize sampled segment durations onto this grid (s) so tiles can SHARE MB "
                        "segments; 0 disables (every tile then needs its own segments).")
    p.add_argument("--n-steps-range", type=int, nargs=2, default=(3, 7), metavar=("MIN", "MAX"))
    p.add_argument("--tile-width-range", type=float, nargs=2, default=(0.8, 1.2), metavar=("MIN", "MAX"))
    p.add_argument("--stair-floor-frac", type=float, default=0.5,
                   help="Fraction of stair tiles keeping a ground plane (rest are bare stairs).")
    p.add_argument("--approach-pin", default="hard", choices=["hard", "soft"])
    p.add_argument("--exit-pin", default="hard", choices=["hard", "soft"])
    p.add_argument("--entry-pick", default="first", choices=["first", "middle", "last"])
    p.add_argument("--exit-pick", default="first", choices=["first", "middle", "last"])
    p.add_argument("--contact-tol", type=float, default=0.02)
    p.add_argument("--swing-clear", type=float, default=None)
    p.add_argument("--swing-band", type=float, default=None)
    p.add_argument("--footprint-margin", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    work = args.work_dir
    os.makedirs(os.path.join(work, "plans"), exist_ok=True)
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    box_counts = dict(zip(("jump_on", "jump_off", "jump_off_front"),
                          (int(x) for x in args.box_tiles.split(","))))
    cpt = int(args.clips_per_tile)
    builder = make_builder(args, args.device)                 # ONE builder for the whole run
    jobs: list[dict] = []
    requests: dict[str, dict] = {}
    next_tid = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}

    for skill in fams:
        kind = "stairs" if skill in _STAIR_SKILLS else "box"
        family = family_for(skill)
        sources = ingest_family_sources(skill, args, args.device)
        tile_rng = np.random.default_rng([args.seed, _skill_seed(skill)])
        if kind == "stairs":
            tile_specs = _stair_tile_specs(tile_rng, skill, sources, args)
        else:
            tile_specs = _box_tile_specs(tile_rng, skill, sources, args, box_counts.get(skill, 0))

        for spec in tile_specs:
            tid = next_tid; next_tid += 1
            # Durations are fixed PER TILE, not per clip: the tile author sizes its platforms from them, and
            # every clip on a tile must emit byte-identical geometry (the reuse invariant the RL importer
            # relies on).  Speed is what varies per clip; duration variety lives across tiles.
            d_rng = np.random.default_rng([args.seed + 1, tid])
            a_s = _quantize(d_rng.uniform(*args.approach_s_range), args.duration_step)
            x_s = _quantize(d_rng.uniform(*args.exit_s_range), args.duration_step)
            v_rng = np.random.default_rng([args.seed + 2, tid])
            speeds = _clip_speeds(v_rng, kind, args, cpt)
            # The tile's walking surfaces are sized for the FASTEST clip on it, and every clip authors the
            # geometry with that same speed -- otherwise a slow clip authors a short platform, the tile
            # meshes some other clip's geometry, and the slow clip stands metres away from the staircase.
            # (Platforms are long because the approach accelerates from a stand; see
            # TODO(steady-approach) in mb_terrain_segments.py -- a steady start would shorten both.)
            plat_vx = max(abs(v) for v in speeds) if speeds else 1.0
            for ci, vx in enumerate(speeds):
                job_args = SimpleNamespace(
                    skill=skill, source=args.source, library=args.library, hf_repo=args.hf_repo,
                    dt=args.dt, device=args.device, terrain_id=tid,
                    riser=spec.get("riser", 0.15), tread=spec.get("tread", 0.25),
                    n_steps=spec.get("n_steps", 6), width=spec.get("width", 1.4),
                    nav_width=spec.get("nav_width"), box_height=spec.get("box_height", 0.45),
                    has_floor=spec.get("has_floor", True),
                    # The commanded RUN-UP speed varies per clip (the DP path's --transition-vels role), so
                    # a tile's clips differ in how fast the robot arrives, not just in which reference was
                    # matched.  A family whose approach is a STANDING settle on a box top has no run-up:
                    # leave it None so the planner resolves it to 0.
                    terrain_speed=vx, exit_vx=vx, platform_vx=plat_vx,
                    approach_vx=(vx if family.approach_mode == "walk" else None),
                    approach_s=a_s, exit_s=x_s,
                    entry_pick=args.entry_pick, exit_pick=args.exit_pick,
                    contact_tol=args.contact_tol, swing_clear=args.swing_clear,
                    swing_band=args.swing_band, footprint_margin=args.footprint_margin,
                )
                rel = os.path.join("plans", f"{skill}_t{tid:04d}_c{ci:03d}.npz")
                try:
                    plan = plan_from_spec(sources, job_args, args.device, builder)
                # One unreachable geometry must cost one clip, not the run.  SystemExit is included on
                # purpose: the planner reports "the traversal never lands on the exit surface" that way, and
                # it is BaseException, so a bare `except Exception` would let it kill a whole library run --
                # which is exactly how the DP path lost a 4.6 h build.
                except (Exception, SystemExit) as e:
                    geom = (f"riser={job_args.riser:.3f} tread={job_args.tread:.3f} n_steps={job_args.n_steps}"
                            if kind == "stairs" else f"box_h={job_args.box_height:.3f}")
                    print(f"[mb_plan][SKIP] {skill} tile {tid} clip {ci} (vx={vx:.2f} {geom}): "
                          f"{type(e).__name__}: {e}")
                    skipped += 1
                    skip_reasons[f"{skill}:{type(e).__name__}"] = \
                        skip_reasons.get(f"{skill}:{type(e).__name__}", 0) + 1
                    continue
                save_terrain_plan(os.path.join(work, rel), plan)
                a_key, x_key = _segment_keys(plan, args.approach_pin, args.exit_pin)
                for key, side, pin in ((a_key, "approach", args.approach_pin),
                                       (x_key, "exit", args.exit_pin)):
                    requests.setdefault(key, {"key": key, "side": side, "pin": pin, "plan": rel,
                                              "out": os.path.join("segments", f"{key}.npz"),
                                              "n_uses": 0})
                    requests[key]["n_uses"] += 1
                jobs.append({"family": skill, "kind": kind, "terrain_id": tid, "clip": ci,
                             "plan": rel, "approach_key": a_key, "exit_key": x_key,
                             "vx": float(vx), "approach_s": a_s, "exit_s": x_s,
                             "approach_pin": args.approach_pin, "exit_pin": args.exit_pin,
                             "footprint_margin": float(args.footprint_margin)})

    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    with open(os.path.join(work, "jobs.json"), "w") as f:
        json.dump({"config": cfg, "n_tiles": next_tid, "jobs": jobs}, f, indent=1)
    with open(os.path.join(work, "requests.json"), "w") as f:
        json.dump({"config": cfg, "requests": list(requests.values())}, f, indent=1)

    n_seg_naive = 2 * len(jobs)
    tiles_with_clips = len({j["terrain_id"] for j in jobs})
    print(f"\n[mb_plan] {len(jobs)} clips over {tiles_with_clips}/{next_tid} tiles"
          + (f" ({skipped} clips SKIPPED)" if skipped else ""))
    for reason, n in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"[mb_plan]   skipped {n:5d}: {reason}")
    print(f"[mb_plan] MB segments: {len(requests)} unique of {n_seg_naive} requested "
          f"({n_seg_naive / max(1, len(requests)):.1f}x reuse)")
    for side in ("approach", "exit"):
        n = sum(1 for r in requests.values() if r["side"] == side)
        print(f"[mb_plan]   {side}: {n} unique")
    print(f"[mb_plan] wrote {work}/jobs.json, {work}/requests.json, {work}/plans/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
