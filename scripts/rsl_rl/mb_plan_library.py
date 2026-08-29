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

from motion_gen.motionbricks.approach_zone import APPROACH_ZONE_M  # noqa: E402
from motion_gen.motionbricks import flat_plan as fp  # noqa: E402
from motion_gen.motionbricks.stride_bank import load_stride_bank  # noqa: E402
from motion_gen.terrain.terrain_sampler import sample_box_tiles, sample_stair_tiles  # noqa: E402
from motion_gen.motionbricks import (elevated_side, family_for, ground_side,   # noqa: E402
                                    save_terrain_plan)

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


# Bump when the MEANING of a segment key changes, so the invalidation is one greppable line rather than a
# remembered ``rm -rf segments/``.  Partial reuse across such a change is the dangerous outcome: the
# library silently mixes segments generated under two different contracts.
#   2: the approach is specified by a travel BUDGET (``approach_dist``) rather than a duration, so the same
#      (reference, speed, duration) no longer describes the same motion.
_KEY_VERSION = "v2"


def _round_key(*vals) -> str:
    """Stable short hash of the given strings / scalars / arrays (rounded, so float noise cannot split a key).

    Deliberately does NOT use ``hash()``: Python salts string hashing per process, so a key built that way
    would change between runs and break both segment reuse and resume.
    """
    h = hashlib.sha1()
    h.update(_KEY_VERSION.encode())
    for v in vals:
        if isinstance(v, str):
            h.update(v.encode())
        else:
            h.update(np.ascontiguousarray(np.round(np.asarray(v, dtype=np.float64), 4)).tobytes())
    return h.hexdigest()[:16]


def _segment_keys(plan, approach_pin: str, exit_pin: str, approach_start: str = "stand",
                  holds=None, side_plans=None, unique_salt: str | None = None) -> tuple[str, str]:
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
    # DISTANCE, not duration: the budget is what the generator reaches for, and the duration
    # is now emergent from it and the speed. Keying on both would invalidate every segment
    # whenever the (now-derived) duration moved by a rounding step.
    a_scalars = [plan.approach_vx, plan.entry_vx, plan.approach_dist]
    x_scalars = [plan.exit_vx, plan.exit_speed, plan.exit_dist]
    # ``approach_start`` is part of the key: a run-up from a stand and an arrival already at speed are
    # different motions for the same command, so they must not share a generated segment.
    a_parts = [f"{plan.skill}|{plan.approach_mode}|{approach_pin}|{approach_start}|approach", a_scalars]
    x_parts = [f"{plan.skill}|{plan.exit_mode}|{exit_pin}|exit", x_scalars]
    # The HOLDS are part of what Motion Bricks is conditioned on: pass B tracks them to generate
    # both segments, so two clips whose holds differ are two different motions and must not share a
    # cached segment. Drawn per clip, so in practice this alone drops reuse to ~0.
    if holds is not None:
        # The HOLD LIST, flattened. Pass B generates both segments by tracking these commands, so
        # two clips whose holds differ are two different motions and must not share a segment.
        flat = [v for side in ("approach", "exit") for d, c in holds[side] for v in (d, c)]
        flat.append(float(holds["interaction"]))
        a_parts.append(flat)
        x_parts.append(flat)
    if side_plans:
        # The STRIDES and PHASES too. Pass A now chooses them and pass B is told them, so two clips
        # whose plans differ are seeded from different gait phases and are different motions -- even
        # if their holds happen to match.
        for side, parts in (("approach", a_parts), ("exit", x_parts)):
            pl = side_plans.get(f"{side}_plan")
            if pl:
                parts.append([(pc.get("stride", -1), pc.get("phase0", pc.get("phase_in", -1)),
                               pc.get("phase_out", -1)) for pc in pl["pieces"]])
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

    CONVERGED cells only. The grid used to come from every ingested source, including solves the
    optimizer gave up on -- measured on ``library_v2``, 6 of the 8 ``(riser, tread)`` cells that
    ``stairs_down`` tiles were sampled from are non-converged. A tile authored around one of those is
    a staircase whose reference cannot walk it, and since the tile geometry is what the RL and the
    depth cameras SEE, the defect outlives the clip.

    This is the one place convergence is filtered on rather than reported; the flat families keep
    their non-converged cells deliberately (see ``clean_ingest.solve_converged``). Expect it to
    NARROW the descent population until the solver delivers more cells -- that is the honest state,
    not a regression.
    """
    ok = [s for s in sources if (s.terrain_meta or {}).get("converged", True)]
    if not ok:
        raise SystemExit(
            f"[mb_plan] {skill}: every reference in the library is a non-converged solve, so there "
            f"is no (riser, tread) cell a tile could honestly be authored around. Re-solve, or pass "
            f"a library that has converged {skill} entries.")
    if len(ok) < len(sources):
        # IDENTITY, not `s not in ok`: MotionSource is a dataclass, so `in` invokes __eq__, which
        # compares its tensor fields elementwise and raises on two clips of different lengths
        # ("The size of tensor a (69) must match the size of tensor b (73)").
        _ok_ids = {id(s) for s in ok}
        dropped = sorted({(round(float(s.terrain_meta["terrain"]["riser"]), 4),
                           round(float(s.terrain_meta["terrain"]["tread"]), 4))
                          for s in sources if id(s) not in _ok_ids})
        print(f"[mb_plan] {skill}: {len(sources) - len(ok)}/{len(sources)} references did NOT "
              f"converge; their geometry is excluded from the tile grid. Cells lost: {dropped}")
    sources = ok
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
                    # No per-tile durations any more: the travel budget is per CLIP and in metres,
                    # drawn in the clip loop from one band that serves stairs and boxes alike. The
                    # box family needed its own duration ranges only because a platform was
                    # duration x speed, so a 6 s box duration on a 2.6 m/s stair would have authored
                    # a ~17 m platform.
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


def _split_dist(rng, total: float, mins: list) -> list:
    """Split ``total`` metres into ``len(mins)`` stretches, each at least its own minimum.

    Dirichlet-ish rather than equal, so a tile's clips do not all change command at the same place --
    a fixed split would put every command change at the same distance from the terrain, which is a
    landmark the model could key on instead of the depth image.
    """
    n = len(mins)
    if n == 1:
        return [float(total)]
    free = float(total) - float(sum(mins))
    if free <= 0.0:                                    # caller should have reduced n; be safe anyway
        return [float(total) / n] * n
    w = rng.dirichlet([2.0] * n)                       # alpha=2 avoids the near-zero shares alpha=1 gives
    return [float(m) + float(free) * float(wi) for m, wi in zip(mins, w)]


def _cycle_travel(bank, cmd: float, dt: float) -> float:
    """Metres one full gait cycle of the stride nearest ``cmd`` covers.

    This is the real floor on a loop, and it is not a constant: ``loop_frames_for`` will not emit
    fewer than one period, so a budget below this buys a WHOLE cycle it cannot pay for -- or, once
    ``_loop_piece`` refuses to do that, buys no loop at all. It ranges 0.1 m at 0.2 m/s to ~3 m at
    3.5, so the flat ``min_hold_m`` that preceded it was simultaneously too strict at the slow end and
    too lax at the fast one (measured: one transition clip still lost every loop at ``min_hold_m``
    1.5 m, because a cycle at 3 m/s is 1.8 m).
    """
    if bank is None:
        return 0.0
    i = fp.nearest_stride(bank.achieved_vel, (float(cmd), 0.0, 0.0))
    return float(bank.period_frames[i]) * float(dt) * float(
        np.linalg.norm(bank.achieved_vel[i][:2]))


# A planned hold must be worth planning. ONE gait cycle is the arithmetic floor (``loop_frames_for``
# will not emit less), but sizing a hold to exactly that leaves no margin at all: the transient reserve
# is a bound on what Motion Bricks MAY take and it can legitimately take all of it, at which point the
# loop misses by centimetres and vanishes -- which is how two in-betweenings ended up spliced back to
# back.
#
# 1.5 is chosen against the budget rather than by taste. Fraction of random command pairs from the
# stair band that fit in a given flat lead-in, measured over 4000 draws:
#
#     flat      1.0x    1.5x    2.0x
#      4 m       50%     18%      6%
#      6 m      100%     84%     41%
#      8 m      100%    100%     95%
#
# At the shipped 6 m, 2.0 would refuse 59% of command changes and 1.0 is the zero-margin case that
# produced the defect. A pair that does not fit costs VARIATION, not correctness -- ``_side`` simply
# plans one hold -- so the tighter number would buy nothing but bigger tiles.
_HOLD_CYCLES = 1.5


def _hold_minimums(cmds: list, min_hold: float, args, bank,
                   lead_in_from: float | None = None) -> list:
    """Per-hold minimum travel: enough for a real steady stretch, plus its incoming transient.

    Measured when the lead-up became an alternation: sized any smaller, holds lose their loop and pass
    C is left splicing two in-betweenings back to back -- an acceleration running straight into a
    deceleration with no steady motion between them. On screen that is the robot sprinting into the
    terrain from a long way out, and it is what this function exists to prevent.

    Both terms are per hold and derived rather than declared:

    * the LOOP floor is ``_HOLD_CYCLES`` gait cycles of the stride this command selects
      (:func:`_cycle_travel`) -- one cycle is what ``loop_frames_for`` will not go below, and the
      second is the margin that keeps a tight transient reserve from erasing the loop;
    * the TRANSIENT reserve comes from ``flat_plan.transient_travel_bound``, which mirrors what
      ``mb_transient.generate`` actually does, chained hops included -- ONE definition, so pass A's
      prediction and pass B's generation cannot drift.

    ``min_hold_m`` survives as an extra floor, for a recipe that wants steadier stretches than the
    bare minimum.

    ``lead_in_from`` is the speed the FIRST hold is entered from, and it is what makes the two sides
    symmetric. An approach's first hold opens the clip and has no incoming transient, so it is
    ``None``. An exit's first hold is entered from the TERRAIN via the exit transient -- and forgetting
    that cost one clip every exact loop on its trail-away. The approach's mirror is the approach zone,
    added to its last hold for the same reason.
    """
    ramp = float(getattr(args, "flat_ramp_dv", 1.0))

    def _bound(a, b):
        return fp.transient_travel_bound(
            (a, 0.0, 0.0), (b, 0.0, 0.0), a_max=float(args.approach_a_max),
            floor_s=float(args.approach_min_transient_s), slack=float(args.approach_hop_slack),
            ramp_dv=ramp)

    def _steady(c):
        return max(float(min_hold), _HOLD_CYCLES * _cycle_travel(bank, c, args.dt))

    out = [_steady(cmds[0]) + (0.0 if lead_in_from is None else _bound(lead_in_from, cmds[0]))]
    for j in range(1, len(cmds)):
        out.append(_steady(cmds[j]) + _bound(cmds[j - 1], cmds[j]))
    return out


def _rl_holds(rng, family_kind: str, approach_dist: float, exit_dist: float,
              steady_vx: float, args, bank) -> dict:
    """The RL-visible command as a list of HOLDS, one per steady stretch, plus the interaction's.

    Distance-indexed, and that is the point rather than a tidy-up. This replaced a schedule sampled
    over an absolute clip timeline, which had to be held flat across the approach and the exit --
    which meant knowing where those windows were, so pass A reserved a fixed 12 s for a traversal
    that is really 2-6 s. A hold list has no time axis to be wrong about.

    **The command varies along the lead-up and the trail-away**, which is what this whole branch
    exists for: nothing in the shipped library has ever shown the generator a speed change while a
    terrain feature was in view. The hold ADJOINING the terrain is pinned to the clip's own steady
    command on both sides -- the entry transient departs from it and the exit transient lands on it,
    and the tile's entry/exit candidates were selected for it -- so the variation lives strictly in
    the outer stretches and the terrain interface is untouched.

    The commands are drawn CONTINUOUSLY from the family's band and are not tailored to the strides
    that exist: the motion is whichever stride is nearest (``flat_plan.nearest_stride``), and the
    resulting disagreement is exactly what teaches a generator to SELECT rather than to memorise.

    Forward-only: a traversal is authored along the tile's +x, so a lateral or yaw command would walk
    the robot off it.

    Returns:
        ``{"approach": [(dist, vx)], "interaction": vx, "exit": [(dist, vx)]}``, outermost hold
        first on the approach and innermost first on the exit -- i.e. both lists run in CLIP ORDER.
    """
    band = tuple(args.stair_vel_range) if family_kind == "stairs" else tuple(args.box_vel_range)
    sign = -1.0 if float(steady_vx) < 0 else 1.0
    lo, hi = min(abs(band[0]), abs(band[1])), max(abs(band[0]), abs(band[1]))
    req = int(getattr(args, "terrain_cmd_changes_per_side", 0))
    min_hold = float(getattr(args, "min_hold_m", 0.0))

    def _side(total: float, is_approach: bool) -> list:
        # The zone belongs to the hold that adjoins the terrain, and is NOT available for splitting:
        # it is the stretch the entry transient consumes.
        zone = APPROACH_ZONE_M if is_approach else 0.0
        flat = max(0.0, float(total) - zone)
        # How many holds the travel can actually FUND -- each interior one has to pay for its own
        # in-betweening before it can pay for a loop. The requested count is a ceiling; drop one at a
        # time until the minimums fit. Reducing rather than shrinking the holds is deliberate: a hold
        # too short to contain an exact loop is not a steady stretch, and planning one produces a
        # lead-up made entirely of in-betweenings.
        n = max(1, 1 + int(req))
        while n > 1:
            cmds = [sign * float(rng.uniform(lo, hi)) for _ in range(n)]
            if is_approach:
                cmds[-1] = float(steady_vx)
            else:
                cmds[0] = float(steady_vx)
            mins = _hold_minimums(cmds, min_hold, args, bank,
                                  lead_in_from=(None if is_approach else 0.0))
            if sum(mins) <= flat:
                break
            n -= 1
        else:
            cmds = [float(steady_vx)]
            mins = [min(_hold_minimums(cmds, min_hold, args, bank,
                                       lead_in_from=(None if is_approach else 0.0))[0], flat)]
        shares = _split_dist(rng, flat, mins)
        # Give the zone to the hold that touches the terrain.
        if is_approach:
            shares[-1] += zone
        return [(float(d), float(c)) for d, c in zip(shares, cmds)]

    return {"approach": _side(float(approach_dist), True),
            "interaction": float(steady_vx),
            "exit": _side(float(exit_dist), False)}


def _side_plans(name: str, holds: dict, rng, bank, dt: float, seed: int) -> dict:
    """The approach and exit as ordinary flat plans -- a loop per hold, an in-betweening between.

    This is the whole of step 8's structural claim: **a terrain clip's lead-up is not a special
    case.** It is `flat_plan`'s alternation, budgeted in metres instead of seconds, with a Motion
    Bricks transient welded to each end that happens to target the terrain rather than a stride.

    Pass A owns every discrete choice, which is the property `flat_plan` is built on: the strides and
    the hand-off phases are decided HERE and pass B is told them. Pass B used to pick the entry
    transient's seed stride itself (``targeter.nearest(v0)``) and its phase from the RNG seed
    (``seed % period``); with several loops on the approach the LAST one has to hand off at exactly
    that stride and phase, and two independent 'nearest' implementations agreeing is not something to
    rely on when they can simply not both exist.
    """
    out = {}
    for side in ("approach", "exit"):
        seq = holds.get(side) or []
        if not seq:
            continue
        out[f"{side}_plan"] = fp.plan_to_dict(fp.plan_from_distance_holds(
            f"{name}_{side}", [(d, (c, 0.0, 0.0)) for d, c in seq], rng,
            bank_periods=bank.period_frames, bank_vel=bank.achieved_vel, dt=float(dt),
            seed=int(seed)))
    return out


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
    p.add_argument("--stride-bank", default=None,
                   help="Stride bank .npz. Pass A needs it now: the approach and the exit are ordinary "
                        "flat plans (a loop per commanded speed, an in-betweening between), and "
                        "choosing a loop's stride and gait phase requires the bank's periods and "
                        "achieved velocities. Without it a clip gets ONE hold per side, which is the "
                        "pre-step-8 structure.")
    p.add_argument("--terrain-cmd-changes-per-side", type=int, default=0,
                   help="Command changes during a terrain clip's lead-up and trail-away. 0 reproduces "
                        "the single-hold structure. A CEILING, not a promise -- the travel budget "
                        "decides how many steady stretches actually fit (--min-hold-m). Each change "
                        "costs one extra Motion Bricks generation per clip in pass B, so 2 is roughly "
                        "3x the stage-b cost of 0.")
    p.add_argument("--min-hold-m", type=float, default=1.5,
                   help="Shortest steady stretch (m) a command hold may be. A hold shorter than a gait "
                        "cycle is not steady in any measurable sense, and loop_frames_for rounds it UP "
                        "-- silently overrunning the travel budget and pushing the terrain past the "
                        "platform edge.")
    # The three knobs that decide how long an in-betweening lasts. Pass B applies them; pass A has to
    # PREDICT them, because a command hold must fund its incoming transient before it can fund a loop.
    # Same names and same recipe keys on both sides, so the two cannot be given different values.
    p.add_argument("--approach-a-max", type=float, default=2.0,
                   help="Max planar acceleration (m/s^2) a transition may demand; sets a FLOOR on its "
                        "duration. Must match mb_terrain_segments --a-max.")
    p.add_argument("--approach-min-transient-s", type=float, default=0.35,
                   help="Floor on a transient's duration (s). Must match --min-transient-s in pass B.")
    p.add_argument("--approach-hop-slack", type=float, default=1.3,
                   help="How much longer than its floor a transient may run. Must match --hop-slack "
                        "in pass B.")
    p.add_argument("--flat-ramp-dv", type=float, default=1.0,
                   help="Max speed change one in-betweening is asked for; a bigger change is split into chained hops. Pass A predicts the resulting TRAVEL with it, so it must match --ramp-dv in pass B.")
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
    p.add_argument("--box-nav-width", type=float, default=-1.0,
                   help="y-width (m) the box AND its top platform are widened to. -1 (default) uses the "
                        "tile's sampled corridor width, 0 keeps the reference box's own narrow extents. "
                        "The landing is laterally free under the box anchor mask, so a walkable top wants "
                        "the corridor width the same way a stair platform does.")
    p.add_argument("--backward-vx-max", type=float, default=1.0,
                   help="Cap (m/s) on any BACKWARD command (the jump_off dismount). The clean library's "
                        "walk_backwards strides stop near 1 m/s, so a larger command is unreachable.")
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
    p.add_argument("--entry-pick", default="sample",
                   choices=["first", "middle", "last", "sample"],
                   help="Which gate candidate the terrain entry hands off at. 'sample' varies the "
                        "hand-off gait phase per clip; a fixed preference gives every clip on a "
                        "reference the identical entry pose.")
    p.add_argument("--exit-pick", default="first", choices=["first", "middle", "last"])
    p.add_argument("--contact-tol", type=float, default=0.02)
    p.add_argument("--swing-clear", type=float, default=None)
    p.add_argument("--swing-band", type=float, default=None)
    p.add_argument("--approach-flat-s-range", type=float, nargs=2, default=(1.5, 4.0),
                   metavar=("MIN", "MAX"),
                   help="Seconds of flat lead-in before the approach zone, converted to a DISTANCE "
                        "at each clip's own speed and clamped by --flat-dist-clamp.")
    p.add_argument("--exit-flat-s-range", type=float, nargs=2, default=(1.5, 4.0),
                   metavar=("MIN", "MAX"), help="As --approach-flat-s-range, for the trail-away.")
    p.add_argument("--flat-dist-clamp", type=float, nargs=2, default=(1.0, 4.0),
                   metavar=("MIN", "MAX"),
                   help="Metre bounds on the converted flat distance, so neither end of the speed "
                        "band degenerates (0.2 m/s would walk 6 m over 30 s; 3.5 m/s would cross "
                        "the tile in 1.5 s).")
    p.add_argument("--dist-step", type=float, default=0.25,
                   help="Quantization (m) of the travel budget, so segments can still be shared.")
    p.add_argument("--platform-buffer-m", type=float, default=1.0,
                   help="Walkable surface (m) past the nominal travel, so the reference cannot walk "
                        "off the far edge.")
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
    # Pass A now makes the discrete choices the approach and exit plans are built from -- which
    # stride each hold replays and which gait phase it hands off at -- so it needs the bank. Optional
    # so a plan-only debug run still works; without it every side gets ONE hold, the pre-step-8
    # structure.
    bank = load_stride_bank(args.stride_bank) if args.stride_bank else None
    if bank is None and int(args.terrain_cmd_changes_per_side) > 0:
        raise SystemExit(
            "--terrain-cmd-changes-per-side > 0 needs --stride-bank: a command change IS a loop "
            "boundary, and choosing a loop's stride and gait phase requires the bank. Asking for "
            "changes without it would silently produce none.")
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
            # The travel budget is PER CLIP and in METRES, drawn below from each clip's own speed.
            # It used to be a duration fixed per TILE, and the tile author then sized its platforms
            # as duration x the tile's FASTEST clip -- which is why a 3.5 s budget on a 2.6 m/s
            # stair tile authored a 10.1 m platform on both ends of every transition tile, and why
            # the box families needed their OWN duration ranges (a 6 s box duration applied to a
            # 2.6 m/s stair would have authored a ~17 m platform). In metres that whole coupling
            # goes: one band serves both families, and the tile takes the MAX over its clips, which
            # is the same discipline ``plat_vx`` already used for speed.
            d_rng = np.random.default_rng([args.seed + 1, tid])
            top_vx = spec.get("box_top_vx")
            two_sided = bool(spec.get("two_sided", True))
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
            # (Platforms are long because a STAND start has to accelerate from rest, and any tile may host
            # one -- ``starts`` above mixes the two per clip.  A steady-only library would shorten them;
            # see ``_generate_approach`` in mb_terrain_segments.py for what each start actually does.)
            # Commands are MAGNITUDES here (the family owns every direction), clamped per side to what the
            # stride bank can actually walk that way.
            ground_mags = [_cmd_mag(v, ground_sign, args) for v in speeds]
            plat_vx = max(ground_mags) if ground_mags else 1.0
            top_mag = _cmd_mag(top_vx, top_sign, args) if kind == "box" else None

            # Travel budget per clip, in metres: seconds of flat lead-in at THIS clip's own speed,
            # clamped in metres so neither end of the speed band degenerates (a 0.2 m/s clip would
            # otherwise walk 6 m over 30 s, and a 3.5 m/s clip would cover the whole tile in 1.5 s),
            # then the approach zone on top. APPROACH_ZONE_M is the shared constant the clutter pass
            # vetoes decoys against, so the distance a clip stops tracking the command at and the
            # distance a decoy may not sit inside are the same number by construction.
            lo_d, hi_d = args.flat_dist_clamp

            def _budget(mag: float, s_range) -> float:
                flat = min(max(abs(float(mag)) * float(d_rng.uniform(*s_range)), lo_d), hi_d)
                return _quantize(flat + APPROACH_ZONE_M, args.dist_step)

            a_dists = [_budget(m if kind != "box" else
                               (top_mag if elevated_side(family) == "approach" else m),
                               args.approach_flat_s_range) for m in ground_mags]
            x_dists = [_budget(m if kind != "box" else
                               (top_mag if elevated_side(family) == "exit" else m),
                               args.exit_flat_s_range) for m in ground_mags]
            # The tile's walking surfaces are sized for the LONGEST budget on it, and every clip
            # authors the geometry with that same number -- otherwise a short clip authors a short
            # platform, the tile meshes some other clip's geometry, and the short clip stands metres
            # away from the staircase. Split ground vs elevated, because they are different surfaces.
            _ground_is_approach = ground_side(family) == "approach"
            plat_ground_dist = max((a_dists if _ground_is_approach else x_dists), default=1.0)
            plat_top_dist = max((x_dists if _ground_is_approach else a_dists), default=1.0)
            for ci, (vx, g_mag, a_start, a_dist, x_dist) in enumerate(
                    zip(speeds, ground_mags, starts, a_dists, x_dists)):
                if kind != "box":
                    a_mag = x_mag = g_mag
                else:
                    mag = {elevated_side(family): top_mag, ground_side(family): g_mag}
                    a_mag, x_mag = mag["approach"], mag["exit"]
                _holds_ci = _rl_holds(
                    np.random.default_rng([args.seed + 7, _skill_seed(skill), int(tid), ci]),
                    kind, a_dist, x_dist, float(family.exit_sign) * abs(float(vx)), args, bank)
                _sides_ci = _side_plans(
                    f"{skill}_t{tid:04d}_c{ci:03d}", _holds_ci,
                    np.random.default_rng([args.seed + 13, _skill_seed(skill), int(tid), ci]),
                    bank, args.dt,
                    int(np.random.default_rng(
                        [args.seed + 17, _skill_seed(skill), int(tid), ci]).integers(0, 2 ** 31))
                ) if bank is not None else {}
                job_args = SimpleNamespace(
                    skill=skill, source=args.source, library=args.library, hf_repo=args.hf_repo,
                    dt=args.dt, device=args.device, terrain_id=tid,
                    riser=spec.get("riser", 0.15), tread=spec.get("tread", 0.25),
                    n_steps=spec.get("n_steps", 6), width=spec.get("width", 1.4),
                    nav_width=spec.get("nav_width"), box_height=spec.get("box_height", 0.45),
                    has_floor=spec.get("has_floor", True), box_nav_width=spec.get("box_nav_width"),
                    stair_float_thickness=spec.get("float_thickness", 0.0),
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
                    # The per-clip RL command, as a list of HOLDS, decided here in pass A rather
                    # than when the clip is written in pass C. It has to exist before pass B,
                    # because Motion Bricks GENERATES the approach and exit motion by tracking these
                    # commands -- a command invented afterwards would be one the motion never
                    # followed. The terrain overrides it only near the interaction
                    # (mb_terrain_segments); everywhere else the motion tracks it.
                    rl_holds=_holds_ci,
                    # The approach and the exit as ordinary FLAT plans -- a loop per hold, an
                    # in-betweening between. Pass B generates those in-betweenings and pass C splices
                    # them, so both read this rather than re-deriving anything.
                    side_plans=_sides_ci,
                    # Metres, not seconds. platform_* are TILE-level maxima; approach/exit_dist are
                    # this clip's own budget.
                    platform_ground_dist=plat_ground_dist, platform_top_dist=plat_top_dist,
                    platform_buffer_m=args.platform_buffer_m,
                    approach_dist=a_dist, exit_dist=x_dist,
                    entry_pick=args.entry_pick, exit_pick=args.exit_pick,
                    clip_seed=int(np.random.default_rng(
                        [args.seed + 11, _skill_seed(skill), int(tid), ci]).integers(0, 2 ** 31)),
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
                    holds=job_args.rl_holds, side_plans=job_args.side_plans,
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
                # The INTERIOR transients: one per command change on each side. Ordinary
                # loop-to-loop in-betweenings -- the same job the flat family's transients are, and
                # generated by the same code (``mb_transient.generate``). Keyed by the plan's own
                # piece key, which is unique per clip because ``plan_from_distance_holds`` names them
                # after the clip. NOT deduped across clips: two transients are interchangeable only
                # if they start and end at the EXACT same frame of the same stride, and the phases
                # are drawn per clip, so a match would be both rare and unverifiable.
                for _side in ("approach", "exit"):
                    if (_side == "approach" and a_key is None) or (_side == "exit" and x_key is None):
                        continue                      # a side with no terrain segment has no lead-up
                    _pl = (job_args.side_plans or {}).get(f"{_side}_plan")
                    for _pc in ((_pl or {}).get("pieces") or []):
                        if _pc.get("kind") != "transient":
                            continue
                        _k = _pc["key"]
                        requests[_k] = {"key": _k, "side": f"{_side}_interior", "pin": "stride",
                                        "plan": rel, "piece_key": _k,
                                        "out": os.path.join("segments", f"{_k}.npz"), "n_uses": 1}
                jobs.append({"family": skill, "kind": kind, "terrain_id": tid, "clip": ci,
                             "plan": rel, "approach_key": a_key, "exit_key": x_key,
                             "approach_start": a_start, "two_sided": two_sided,
                             "vx": float(vx), "approach_dist": a_dist, "exit_dist": x_dist,
                             "approach_vx": float(a_mag), "exit_vx": float(x_mag),
                             "top_vx": (None if top_mag is None else float(top_mag)),
                             "approach_pin": args.approach_pin, "exit_pin": args.exit_pin,
                             "footprint_margin": float(args.footprint_margin)})

    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    with open(os.path.join(work, "jobs.json"), "w") as f:
        json.dump({"config": cfg, "n_tiles": next_tid, "jobs": jobs}, f, indent=1)
    with open(os.path.join(work, "requests.json"), "w") as f:
        json.dump({"config": cfg, "requests": list(requests.values())}, f, indent=1)

    # Segment references the jobs ACTUALLY make -- not 2 x clips. A one-sided box tile emits only
    # the ground side, so the naive doubling counts sides that were never requested and then reports
    # the difference as "reuse". Measured on the quick recipe it claimed 1.2x reuse where the true
    # sharing was ZERO (52 references, 52 distinct keys), which is exactly backwards: it would mask
    # a real reuse regression, and it invites the reader to conclude segment_reuse is leaking.
    n_seg_refs = sum(1 for j in jobs for side in ("approach", "exit") if j.get(f"{side}_key"))
    tiles_with_clips = len({j["terrain_id"] for j in jobs})
    print(f"\n[mb_plan] {len(jobs)} clips over {tiles_with_clips}/{next_tid} tiles"
          + (f" ({skipped} clips SKIPPED)" if skipped else ""))
    for reason, n in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"[mb_plan]   skipped {n:5d}: {reason}")
    _shared = n_seg_refs - len(requests)
    print(f"[mb_plan] MB segments: {len(requests)} unique of {n_seg_refs} referenced"
          + (f" ({n_seg_refs / max(1, len(requests)):.2f}x reuse)" if _shared
             else " (no sharing -- one segment per clip side, as segment_reuse=false intends)"))
    for side in ("approach", "exit"):
        n = sum(1 for r in requests.values() if r["side"] == side)
        print(f"[mb_plan]   {side}: {n} unique")
    print(f"[mb_plan] wrote {work}/jobs.json, {work}/requests.json, {work}/plans/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
