#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""PASS A (env_isaaclab): sample terrain TILES, plan every clip on them, and dedupe the MB segment work.

``mb_terrain_clips.py`` plans ONE traversal per invocation; a training library needs thousands, spread over
sampled tile geometries, with many clips SHARING each tile (the RL importer round-robins envs over tiles, so
a tile is a reuse unit). This pass does the whole env_isaaclab half of that up front:

1. **Sample tiles** with the same samplers + ranges the DP generator uses
   (:func:`~motion_gen.terrain.terrain_sampler.sample_stair_tiles` /
   :func:`~motion_gen.terrain.terrain_sampler.sample_box_tiles`), so the MB library's tile
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

from motion_gen.command_schedule import (as_json, sample_command_schedule,  # noqa: E402
                                        total_duration)
from motion_gen.terrain.terrain_sampler import sample_box_tiles, sample_stair_tiles  # noqa: E402
from motion_gen.motionbricks import (elevated_side, family_for, ground_side,   # noqa: E402
                                    save_terrain_plan)

# Frames of the replay hashed into a segment key: MB is conditioned on a ~1 s window around the boundary
# frame (its planning horizon / context), so a window this long covers everything the model can see.
_KEY_WINDOW = 64
_STAIR_SKILLS = ("stairs_up", "stairs_down")
# Slack added to a clip's schedule duration to cover the terrain interaction, whose length is
# only known once the replay is placed. cmd_at saturates past the end, so over-long is safe.
_INTERACTION_BUDGET_S = 12.0


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


def _segment_keys(plan, approach_pin: str, exit_pin: str, approach_start: str = "stand",
                  schedule=None, unique_salt: str | None = None) -> tuple[str, str]:
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
    # ``approach_start`` is part of the key: a run-up from a stand and an arrival already at speed are
    # different motions for the same command, so they must not share a generated segment.
    a_parts = [f"{plan.skill}|{plan.approach_mode}|{approach_pin}|{approach_start}|approach", a_scalars]
    x_parts = [f"{plan.skill}|{plan.exit_mode}|{exit_pin}|exit", x_scalars]
    # The SCHEDULE is part of what Motion Bricks is conditioned on now: pass B tracks it to generate
    # both segments, so two clips whose schedules differ are two different motions and must not share
    # a cached segment. Sampled per clip, so in practice this alone drops reuse to ~0.
    if schedule is not None:
        flat = [v for seg in schedule for v in seg]
        a_parts.append(flat)
        x_parts.append(flat)
    # ``unique_salt`` forces a segment per clip even if two schedules DID collide -- an explicit
    # guarantee rather than one inferred from the sampler's variety.
    if unique_salt is not None:
        a_parts.append(f"|salt|{unique_salt}")
        x_parts.append(f"|salt|{unique_salt}")
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
        # FLOATING treads ride the shared stair-decor policy, resolved ONCE per tile here (one band per
        # transition tile) and carried as an explicit thickness -- the generator and the RL importer must
        # not re-draw it independently.  All-zero fracs consume no rng, so an undecorated run reproduces
        # earlier tile populations exactly.
        decor_params={"float_frac": float(args.stair_float_frac),
                      "float_thick_range": tuple(args.stair_float_thick_range)},
    )
    n_float = sum(1 for t in tiles if t.stair_decor and t.stair_decor[0]["float_thickness"] > 0.0)
    print(f"[mb_plan] {skill}: hd_grid={hd_grid} -> {len(tiles)} transition tiles "
          f"({n_float} floating-tread / {len(tiles) - n_float} solid)")
    return [{"riser": float(t.step_height), "tread": float(t.step_depth),
             "n_steps": int(t.stair_steps[0]), "width": float(t.width),
             "nav_width": float(t.nav_width), "has_floor": bool(t.has_floor),
             "float_thickness": (float(t.stair_decor[0]["float_thickness"]) if t.stair_decor else 0.0)}
            for t in tiles]


def _box_tile_specs(rng, skill: str, sources, args, n_tiles: int) -> list[dict]:
    """Sampled box tiles for one jump family -> per-tile arg dicts.

    The MB tile author places the walkable box at the reference's SOLVED extents, so a tile varies by which
    box HEIGHT it targets (the nearest reference is then selected), by the corridor width it widens the box
    + top platform to, and by how much ELEVATED locomotion its platform must hold. The DP path's support
    variants (floating tops / legs) and lateral box offsets still have no MB tile-author equivalent.

    The box-top duration and speed are fixed PER TILE (like the ground-side durations below): the platform
    is authored from them, and every clip on a tile must emit byte-identical geometry. Per-clip variety
    lives on the GROUND side -- how fast the robot arrives at / leaves the box.
    """
    heights = sorted({round(float(s.terrain_meta["terrain"].get("box_height", 0.0)), 4) for s in sources})
    heights = [h for h in heights if h > 0.0] or [0.45]
    specs = sample_box_tiles(rng, n_tiles=n_tiles, n_sources=len(heights), variant_fracs=None,
                             nav_width_range=tuple(args.tile_width_range))
    print(f"[mb_plan] {skill}: box heights={heights} -> {len(specs)} tiles")
    out = []
    # ONE-SIDED vs TWO-SIDED is a TILE property: a one-sided tile is the bare jump box (no walkable
    # platform), a two-sided one carries the platform its elevated segment walks on, and every clip on
    # a tile must emit byte-identical geometry.  One-sided tiles teach the interaction on its own --
    # run up and mount, or drop off and walk away -- without also demanding the on-box walk.
    #
    # EXACTLY ``round(frac * n)`` tiles are two-sided, rather than an independent Bernoulli per tile: a
    # family has only ~15-20 tiles, so a coin flip lands well off the requested split (0.5 measured 32/50
    # = 64% two-sided across the three families), and the one-sided tiles exist to fix a specific learning
    # failure -- their count is the point, not its expectation.
    n_two = int(round(float(args.box_two_sided_frac) * len(specs)))
    two_idx = set(np.random.default_rng([args.seed + 7, _skill_seed(skill)])
                  .permutation(len(specs))[:n_two].tolist())
    for i, s in enumerate(specs):
        t_rng = np.random.default_rng([args.seed + 4, _skill_seed(skill), i])
        t_rng.random()          # consumed, not used: keeps the rest of this tile's draws where they were
        two_sided = i in two_idx
        out.append({"box_height": heights[int(s.source_idx) % len(heights)],
                    # 0 keeps the source box's own (narrow) width; anything else widens box + platform.
                    "box_nav_width": (float(s.nav_width) if args.box_nav_width < 0.0
                                      else float(args.box_nav_width)) or None,
                    "two_sided": two_sided,
                    "box_approach_s": _quantize(t_rng.uniform(*args.box_approach_s_range),
                                                args.duration_step),
                    "box_exit_s": _quantize(t_rng.uniform(*args.box_exit_s_range), args.duration_step),
                    "box_top_vx": float(t_rng.uniform(*args.box_top_vx_range))})
    print(f"[mb_plan] {skill}: {n_two} two-sided (box + platform) / {len(specs) - n_two} one-sided "
          f"(bare box, terrain + ground side only)")
    return out


def _cmd_mag(vx: float, sign: float, args) -> float:
    """Command MAGNITUDE for one side, clamped to what the stride bank can actually walk in that direction.

    A BACKWARD side (the ``jump_off`` dismount: it walks backwards along the box top and backwards away
    from the box) is driven from the clean library's ``walk_backwards`` strides, which only reach about
    1 m/s -- so commanding the box speed range's 3.5 m/s asks Motion Bricks for a motion it cannot make
    AND writes an unreachable ``target_vel`` into the clip for the RL to chase. Forward sides are
    unclamped: the bank covers up to 3.6 m/s there.
    """
    return min(abs(float(vx)), float(args.backward_vx_max)) if float(sign) < 0.0 else abs(float(vx))


def _clip_schedule(rng, family_kind: str, approach_s: float, exit_s: float,
                   steady_vx: float, args):
    """Per-clip command schedule for a terrain clip, spanning approach + interaction + exit.

    Forward-only: a traversal is authored along the tile's +x, so a lateral or yaw command would walk
    the robot off it. Everything else comes from the SHARED sampler every family uses -- the flat
    family calls it with the full command space, this calls it with one band.

    The duration covers the whole clip generously; the interaction's own length is not known until
    the replay is placed, and ``cmd_at`` saturates past the end, so an over-long schedule is
    harmless while a short one would leave the tail uncommanded.
    """
    band = tuple(args.stair_vel_range) if family_kind == "stairs" else tuple(args.box_vel_range)
    total = float(approach_s) + float(exit_s) + _INTERACTION_BUDGET_S
    if not args.rl_cmd_vary:
        # Ablation: ONE held value for the whole clip. Still aligned -- pass B generates both
        # segments by tracking it, so a constant schedule reproduces the old behaviour exactly
        # rather than approximately.
        return [(total, float(steady_vx), 0.0, 0.0)]
    return sample_command_schedule(
        rng, duration=total, seg_dur_range=tuple(args.rl_cmd_seg_dur_range), vx_range=band)


def _clip_speeds(rng, family_kind: str, args, cpt: int, vx_max: float | None = None) -> list[float]:
    """Per-clip GROUND-level traversal/approach speeds for one tile.

    Stairs get EVENLY SPACED speeds over the band (mirroring the DP ``--transition-vels``, so a tile's clips
    tile the speed axis); boxes get uniform samples (mirroring the DP box path).

    ``vx_max`` narrows the band's top instead of clamping each draw afterwards, which matters for a
    BACKWARD family (``jump_off``): clamping would collapse most of the band onto the cap and make every
    clip on the tile the same motion, where narrowing keeps the variety inside what the bank can walk.

    The speed picked here is the clip's STEADY command. It is no longer the whole story: pass 2 varies
    it over the approach and exit (``mb_terrain_clips._rl_command_track``) while holding it across the
    terrain interaction, so a terrain clip carries command CHANGES rather than one value end to end.
    Measured on the pre-change ``clips/g1_mb_v5.h5`` with ``scripts/rsl_rl/clip_speed_distribution.py``:
    every terrain/transition clip was a single held speed on a 50-level grid, which left ``flat`` (52% of
    library time) and ``box`` as the only families the generator ever saw a command transition on.

    The band's floor was also 0.6, so the library had no slow terrain traversal at all -- anything below
    that was flat-only. It now starts at 0.2.
    """
    if family_kind == "stairs":
        lo, hi = args.stair_vel_range
        return [lo] if cpt <= 1 else [lo + (hi - lo) * i / (cpt - 1) for i in range(cpt)]
    lo, hi = args.box_vel_range
    if vx_max is not None:
        hi = max(float(lo), min(float(hi), float(vx_max)))
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
    p.add_argument("--rl-cmd-vary", action=argparse.BooleanOptionalAction, default=True,
                   help="Sample a varying command schedule per terrain clip. --no-rl-cmd-vary gives "
                        "ONE held value, which is the pre-schedule behaviour and a clean ablation.")
    p.add_argument("--rl-cmd-seg-dur-range", type=float, nargs=2, default=(1.0, 3.0),
                   metavar=("MIN", "MAX"),
                   help="Seconds per segment of a terrain clip's command schedule. Sampled HERE "
                        "(pass A) because pass B generates the motion by tracking the schedule, so "
                        "it must exist before the motion does.")
    p.add_argument("--segment-reuse", action=argparse.BooleanOptionalAction, default=False,
                   help="Share a generated MB segment between clips whose conditioning is identical. "
                        "OFF by default: with per-clip command schedules two clips are only "
                        "interchangeable if their whole schedules match, and paying ~1.5x the "
                        "segment generation buys the guarantee that every clip's approach and exit "
                        "were generated for THAT clip.")
    p.add_argument("--stair-vel-range", type=float, nargs=2, default=(0.2, 2.6), metavar=("MIN", "MAX"),
                   help="GROUND-level speed band for the stair families (-> the terrain and "
                        "transition clip families). The floor was 0.6, which left the library with "
                        "no slow terrain traversal at all -- everything below it was flat-only.")
    p.add_argument("--box-vel-range", type=float, nargs=2, default=(0.2, 3.5), metavar=("MIN", "MAX"),
                   help="GROUND-level speed band for the box families (the run-up / walk-away). "
                        "Floor lowered 0.5 -> 0.2 to match the stair band, so no family is missing "
                        "slow traversal.")
    p.add_argument("--box-approach-s-range", type=float, nargs=2, default=(2.5, 4.0),
                   metavar=("MIN", "MAX"),
                   help="Duration (s) of a box clip's LEAD-IN -- the segment before the interaction, "
                        "sampled per tile. Whichever LEVEL that runs on depends on the family (jump_on "
                        "leads in on the ground, the dismounts lead in along the box top), and when it is "
                        "the elevated one it also sizes the top platform.")
    p.add_argument("--box-exit-s-range", type=float, nargs=2, default=(2.5, 6.0), metavar=("MIN", "MAX"),
                   help="Duration (s) of a box clip's TRAIL-AWAY -- the segment after the interaction "
                        "(see --box-approach-s-range).")
    p.add_argument("--box-top-vx-range", type=float, nargs=2, default=(0.3, 0.7), metavar=("MIN", "MAX"),
                   help="Speed band (m/s) of the ELEVATED segment on a box tile, sampled per tile. Modest "
                        "on purpose: the top platform is sized from this times the elevated duration, so a "
                        "fast on-box walk turns the box into a raised walkway.")
    p.add_argument("--box-two-sided-frac", type=float, default=0.5,
                   help="Fraction of box TILES that get the walkable top platform and an ELEVATED segment "
                        "(approach + interaction + exit). The rest are the BARE jump box: terrain plus the "
                        "GROUND side only -- run up and mount, or drop off and walk away -- which teaches "
                        "the interaction without also demanding the on-box walk. Per TILE, not per clip: "
                        "every clip on a terrain_id must emit byte-identical geometry.")
    p.add_argument("--box-platform-buffer", type=float, default=1.0,
                   help="Extra metres of top platform beyond the elevated segment's nominal travel, so the "
                        "reference cannot walk off the far edge (the ground side needs none -- the tile "
                        "keeps its floor there).")
    p.add_argument("--box-nav-width", type=float, default=-1.0,
                   help="y-width (m) the box AND its top platform are widened to. -1 (default) uses the "
                        "tile's sampled corridor width, 0 keeps the reference box's own narrow extents. "
                        "The landing is laterally free under the box anchor mask, so a walkable top wants "
                        "the corridor width the same way a stair platform does.")
    p.add_argument("--backward-vx-max", type=float, default=1.0,
                   help="Cap (m/s) on any BACKWARD command (the jump_off dismount). The clean library's "
                        "walk_backwards strides stop near 1 m/s, so a larger command is unreachable.")
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
    p.add_argument("--stair-float-frac", type=float, default=0.0,
                   help="Fraction of stair TILES built with FLOATING (riser-less) treads: each tread is a "
                        "thin slab with an open gap beneath instead of a solid pillar to the floor. The "
                        "tread-top surface is identical, so the same played traversals are reused -- this "
                        "changes what the height scan and the feet-adjacent geometry look like, not the "
                        "motion. Independent of --stair-floor-frac, which drops the GROUND under the "
                        "whole structure.")
    p.add_argument("--stair-float-thick-range", type=float, nargs=2, default=(0.03, 0.06),
                   metavar=("MIN", "MAX"),
                   help="Floating-tread slab thickness band (m), sampled per floating tile.")
    p.add_argument("--steady-approach-frac", type=float, default=0.5,
                   help="Fraction of clips whose approach starts ALREADY AT SPEED (context-seeded from the "
                        "matched stride) rather than from a stand. 0 = every clip runs up from rest.")
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
        # Which end runs on the ground and which on the elevated surface, and which way each of them
        # travels -- fixed per family, so resolve it once here rather than per tile.
        ground_sign = family.approach_sign if family.approach_level == "lower" else family.exit_sign
        top_sign = family.approach_sign if family.approach_level == "upper" else family.exit_sign
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
            # A box family's two ends get their own BOX-SPECIFIC durations, keyed by ORDER (the lead-in
            # into the interaction vs the trail-away from it) rather than by level.  Order is what a caller
            # actually reasons about, and level cannot express it: jump_on leads in on the GROUND while the
            # dismounts lead in on the BOX TOP, so one shared "ground" range would have to be both the
            # lead-in bound and the trail-away bound at once.  Box-specific because the shared
            # --approach-s-range / --exit-s-range also drive STAIRS, whose platform is duration x speed with
            # speeds to 2.6 m/s -- a 6 s box duration there would author a ~17 m stair platform.
            # Whichever end is ELEVATED still sizes the top platform; _author_tile reads that off the family.
            top_vx = spec.get("box_top_vx")
            two_sided = bool(spec.get("two_sided", True))
            if kind == "box":
                a_s, x_s = spec.get("box_approach_s"), spec.get("box_exit_s")
            v_rng = np.random.default_rng([args.seed + 2, tid])
            s_rng = np.random.default_rng([args.seed + 3, tid])
            speeds = _clip_speeds(v_rng, kind, args, cpt,
                                  vx_max=(args.backward_vx_max if ground_sign < 0 else None))
            # Mix of run-up-from-rest and arrive-at-speed approaches: the library wants both, so the RL
            # sees terrain entered from a stand AND entered while already walking (the deployment case).
            starts = ["steady" if s_rng.random() < args.steady_approach_frac else "stand"
                      for _ in range(len(speeds))]
            # The tile's walking surfaces are sized for the FASTEST clip on it, and every clip authors the
            # geometry with that same speed -- otherwise a slow clip authors a short platform, the tile
            # meshes some other clip's geometry, and the slow clip stands metres away from the staircase.
            # (Platforms are long because the approach accelerates from a stand; see
            # TODO(steady-approach) in mb_terrain_segments.py -- a steady start would shorten both.)
            # Commands are MAGNITUDES here (the family owns every direction), clamped per side to what the
            # stride bank can actually walk that way.
            ground_mags = [_cmd_mag(v, ground_sign, args) for v in speeds]
            plat_vx = max(ground_mags) if ground_mags else 1.0
            top_mag = _cmd_mag(top_vx, top_sign, args) if kind == "box" else None
            for ci, (vx, g_mag, a_start) in enumerate(zip(speeds, ground_mags, starts)):
                if kind != "box":
                    a_mag = x_mag = g_mag
                else:
                    mag = {elevated_side(family): top_mag, ground_side(family): g_mag}
                    a_mag, x_mag = mag["approach"], mag["exit"]
                job_args = SimpleNamespace(
                    skill=skill, source=args.source, library=args.library, hf_repo=args.hf_repo,
                    dt=args.dt, device=args.device, terrain_id=tid,
                    riser=spec.get("riser", 0.15), tread=spec.get("tread", 0.25),
                    n_steps=spec.get("n_steps", 6), width=spec.get("width", 1.4),
                    nav_width=spec.get("nav_width"), box_height=spec.get("box_height", 0.45),
                    has_floor=spec.get("has_floor", True), box_nav_width=spec.get("box_nav_width"),
                    stair_float_thickness=spec.get("float_thickness", 0.0),
                    box_platform_buffer=args.box_platform_buffer,
                    two_sided=two_sided,
                    # The commanded GROUND-level speed varies per clip (the DP path's --transition-vels
                    # role), so a tile's clips differ in how fast the robot arrives at / leaves the terrain,
                    # not just in which reference was matched.  The elevated side is per TILE, because the
                    # platform it walks on is authored from it.
                    terrain_speed=vx, approach_vx=a_mag, exit_vx=x_mag,
                    # The band this clip's steady speed was drawn FROM, carried into the plan so
                    # pass 2 can vary the command within the family's own range. Per family:
                    # the stair band tops out at 2.6 and the box band at 3.5, so a single
                    # pipeline-wide value would let stair clips be commanded faster than any
                    # stair clip is ever generated.
                    rl_vel_range=(tuple(args.stair_vel_range) if kind == "stairs"
                                  else tuple(args.box_vel_range)),
                    # The per-clip command SCHEDULE, sampled here in pass A rather than when the clip
                    # is written in pass C. It has to exist before pass B, because Motion Bricks
                    # GENERATES the approach and exit motion by tracking it -- a schedule invented
                    # afterwards would be a command the motion never followed. The terrain overrides
                    # it only near the interaction (mb_terrain_segments); everywhere else the motion
                    # tracks it, which is the whole point.
                    rl_cmd_schedule=as_json(_clip_schedule(
                        np.random.default_rng([args.seed + 7, _skill_seed(skill), int(tid), ci]),
                        kind, a_s, x_s, float(family.exit_sign) * abs(float(vx)), args)),
                    platform_vx=plat_vx, platform_top_vx=top_mag,
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
                a_key, x_key = _segment_keys(
                    plan, args.approach_pin, args.exit_pin, a_start,
                    schedule=job_args.rl_cmd_schedule,
                    unique_salt=(None if args.segment_reuse else f"{skill}|{tid}|{ci}"))
                # A one-sided box tile has no ELEVATED segment: don't request it and don't record its key,
                # so pass B never generates it and pass C knows to splice that side as None.
                drop = elevated_side(family) if (kind == "box" and not two_sided) else None
                if drop == "approach":
                    a_key = None
                elif drop == "exit":
                    x_key = None
                for key, side, pin in ((a_key, "approach", args.approach_pin),
                                       (x_key, "exit", args.exit_pin)):
                    if key is None:
                        continue
                    requests.setdefault(key, {"key": key, "side": side, "pin": pin, "plan": rel,
                                              "out": os.path.join("segments", f"{key}.npz"),
                                              "start": a_start if side == "approach" else "stand",
                                              "n_uses": 0})
                    requests[key]["n_uses"] += 1
                jobs.append({"family": skill, "kind": kind, "terrain_id": tid, "clip": ci,
                             "plan": rel, "approach_key": a_key, "exit_key": x_key,
                             "approach_start": a_start, "two_sided": two_sided,
                             "vx": float(vx), "approach_s": a_s, "exit_s": x_s,
                             "approach_vx": float(a_mag), "exit_vx": float(x_mag),
                             "top_vx": (None if top_mag is None else float(top_mag)),
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
