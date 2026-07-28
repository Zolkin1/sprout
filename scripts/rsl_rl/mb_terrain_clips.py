#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage 2 (env_isaaclab): plan a terrain tile traversal, then splice the Motion Bricks segments around it.

ONE driver for every terrain family (``stairs_up``, ``stairs_down``, ``jump_on``, ``jump_off``,
``jump_off_front``): a terrain clip is always ``MB approach -> replayed terrain -> MB exit``, the families
differ only by the declarative facts in :mod:`motion_gen.motionbricks.terrain_families`.  The script runs
in two passes because the middle segment must be planned before Motion Bricks (a different conda env) can
generate the two ends:

**Pass 1 --** ``--dump-plan plan.npz``: select the reference nearest the requested geometry + speed, author
the tile, place the replay on it (:mod:`motion_gen.terrain.terrain_place`), FILTER the entry / exit
frames, and write the :class:`~motion_gen.motionbricks.terrain_plan.TerrainPlan`.

**Pass 2 --** ``--plan plan.npz --approach-raw ... --exit-raw ...``: splice the Motion Bricks segments onto
the replay's boundary frames (hard weld or soft ease per side), emit the tile geometry + per-step anchor
mask + RL ``target_vel``, and write the clip ``.h5``.

Boundary frames: an EPISODIC interaction (box mount / dismount) is played whole, so the boundaries are its
first and last frame.  A PERIODIC one (a stair stride looped up/down the treads) has no natural end -- cut
anywhere and a foot starts inside a tread or ends on a phantom stair past the tile -- so candidates are
filtered by contact + tread geometry and one is hand-picked (``--entry-pick`` / ``--exit-pick``).
Example (stairs_up, both passes)::

    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" \
    env_isaaclab/bin/python scripts/rsl_rl/mb_terrain_clips.py --skill stairs_up \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --dump-plan mb_raw/plan_stairs_up.npz --riser 0.15 --tread 0.25 --terrain-speed 0.6 --n-steps 6
    # ... conda run -n motionbricks python scripts/rsl_rl/mb_terrain_segments.py --plan ... --out-dir ...
    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" \
    env_isaaclab/bin/python scripts/rsl_rl/mb_terrain_clips.py --skill stairs_up \
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
        --plan mb_raw/plan_stairs_up.npz --approach-raw mb_raw/seg/approach.npz \
        --exit-raw mb_raw/seg/exit.npz --out clips/g1_stairs_up_mb.h5
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.terrain.box_clip import box_tile_boxes, build_anchor_mask_t, stair_tile_boxes
from motion_gen.clip_output import ClipOutputBuilder
from motion_gen.ingest.terrain_ingest import ingest_terrain_library
from motion_gen.terrain.terrain_place import (
    episodic_box_extents,
    place_episodic_on_tile,
    place_periodic_on_tile,
    select_terrain_reference,
)
from motion_gen.terrain.terrain_sampler import TerrainSpec
from motion_gen.motionbricks import (
    TerrainPlan,
    boundary_candidates,
    box_platform_sign,
    episodic_boundaries,
    family_for,
    footprint_recenter,
    load_raw_motion,
    load_terrain_plan,
    overlays_for,
    pick_boundary,
    raw_to_world,
    save_terrain_plan,
    sole_heights,
    splice_terrain,
)
from motion_gen.multiskill_manager import MultiSkillManager
from motion_gen.world_lift import _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT

_CONTACT_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _heading_speed(Wp: torch.Tensor, Wq: torch.Tensor, f: int, dt: float) -> float:
    """SIGNED forward (body-frame) speed at frame ``f`` -- negative when the motion travels backwards.

    Used for the two ramp targets: the approach decelerates to the replay's speed at the entry frame, and
    the exit accelerates from its speed at the exit frame, so neither MB segment hands off at a velocity
    the replay does not have.
    """
    n = Wp.shape[0]
    a, b = max(0, min(f, n - 2)), min(n - 1, max(f + 1, 1))
    vx, vy = ((Wp[b, :2] - Wp[a, :2]) / (dt * max(1, b - a))).tolist()
    w, x, y, z = (float(c) for c in Wq[min(f, n - 1)])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.cos(yaw) * vx + math.sin(yaw) * vy


def _tread_tops(boxes: list[dict]) -> list[float]:
    """Top-surface heights of the NARROW tread boxes of a stair tile (the levels a foot may rest on)."""
    min_sx = min(float(b["size"][0]) for b in boxes)
    return sorted(float(b["pos"][2]) + float(b["size"][2]) / 2.0
                  for b in boxes if float(b["size"][0]) <= 1.5 * min_sx)


def _off_surface(Wp: torch.Tensor, overlays: list[dict]) -> np.ndarray:
    """Per-frame flag ``[T]``: the root's ``(x, y)`` is outside EVERY walkable box footprint.

    The tile footprint is sized to the union of the motion and the geometry, so a segment that outgrows the
    platform it walks on (a longer ``--approach-s`` than the plan authored the platform for) or drifts
    sideways off it still lands inside the tile -- but with nothing underneath.  Boxes are axis-aligned here
    (the placement yaw-normalizes them), so an AABB containment test suffices.  A ``has_floor`` tile hides
    this at ground level, but on a platform or a box top it is a real hole in the reference.
    """
    xy = Wp[:, :2].detach().cpu().numpy()
    inside = np.zeros(xy.shape[0], dtype=bool)
    for o in overlays:
        cx, cy = float(o["pos_xy"][0]), float(o["pos_xy"][1])
        hx, hy = float(o["size_xyz"][0]) / 2.0, float(o["size_xyz"][1]) / 2.0
        inside |= (np.abs(xy[:, 0] - cx) <= hx) & (np.abs(xy[:, 1] - cy) <= hy)
    return ~inside


def _author_tile(family, ts, args):
    """Author the FIXED tile geometry for ``family`` -> ``(boxes, tile, lower_z, upper_z, tile_meta)``.

    Stairs get a transition tile (approach platform -> treads -> recovery platform) sized to the reference's
    SOLVED riser / tread, because feet only sit on the treads when the played pose matches the geometry it
    was solved for.  A box tile mirrors that shape: the walkable box at the source's solved extents (a
    center-align placement is only valid while depth + height match the source) PLUS a top platform flush
    with it, on the side away from the jump face, so the elevated end of the clip can walk instead of
    standing still (:func:`~motion_gen.terrain.box_clip.box_tile_boxes`).  Only the platform is new
    geometry -- the takeoff / landing footprint is untouched.

    ``lower_z`` / ``upper_z`` are the two walkable levels the family's approach / exit run on: bottom and
    top platform for stairs, ground and box top for a box.

    A walking surface is sized to hold its segment's seconds of travel, i.e. it depends on SPEED -- so a
    tile shared by clips of different speeds MUST be authored with ``args.platform_vx`` (the fastest
    GROUND-level speed on that tile) and ``args.platform_top_vx`` (its single box-top speed). Without them
    each clip authors its own platform length, the geometry stops being fixed per tile, and every clip but
    the longest ends up standing somewhere else than the meshed tile.
    """
    tm = ts.terrain_meta["terrain"]
    plat_vx = getattr(args, "platform_vx", None)      # tile-level speed: one geometry for all its clips
    if family.geom == "stairs":
        riser, tread = float(tm["riser"]), float(tm["tread"])
        n_steps = int(args.n_steps)
        ground_s = max(args.approach_s, args.exit_s)
        plat_speed = (abs(float(plat_vx)) if plat_vx is not None else
                      max(0.3, abs(args.terrain_speed), abs(args.approach_vx or 0.0), abs(args.exit_vx)))
        plat = float(ground_s) * max(0.3, plat_speed) + 1.0
        tile = TerrainSpec(
            terrain_id=int(args.terrain_id), ttype="transition",
            step_height=riser, step_depth=tread, width=float(args.width), lead_in=0.0,
            stair_steps=[n_steps], flat_after=[0.0], nav_width=float(args.nav_width or args.width),
            spawn_y_offset=0.0, stair_decor=[], has_floor=bool(args.has_floor),
        )
        # FLOATING treads: a thickness > 0 tags each tread so the RL importer meshes a thin slab with an
        # open gap beneath instead of a solid pillar.  The tread-TOP surface is unchanged, so the played
        # traversal, its contact filtering and the boundary gate below are all unaffected -- the reference
        # walks on exactly the same steps either way.  Per TILE (the mesh is per terrain_id).
        float_thickness = float(getattr(args, "stair_float_thickness", 0.0) or 0.0)
        boxes = stair_tile_boxes(step_height=riser, step_depth=tread, num_steps=n_steps,
                                 tread_width=tile.width, ttype="transition",
                                 top_platform_len=plat, bottom_platform_len=plat,
                                 nav_width=tile.nav_width, going_up=family.going_up,
                                 float_thickness=float_thickness)
        lower_z, upper_z = riser, n_steps * riser        # bottom platform / top platform tops
        meta = {"geom": "stairs", "riser": riser, "tread": tread, "n_steps": n_steps,
                "platform_len": plat, "width": float(tile.width), "nav_width": float(tile.nav_width),
                "has_floor": bool(args.has_floor), "clip_type": "transition",
                "float_thickness": float_thickness}
        return boxes, tile, lower_z, upper_z, meta

    # The box's extents IN THE TRAVEL FRAME the placement will use (see _place): for every jump reference
    # in the library the robot walks along the source's y, so this is the swap of the raw solved size --
    # sx is the depth the robot crosses, sy the lateral width.
    box_size = episodic_box_extents(ts, align="facing")
    # Which SIDE of the interaction is on the ground and which is on the box top comes from the family's
    # levels (both ends walk now, so the old "which one is a standing hold" test is gone).
    ground_on_approach = family.approach_level == "lower"
    ground_s = args.approach_s if ground_on_approach else args.exit_s
    top_s = args.exit_s if ground_on_approach else args.approach_s
    # Tile-level speeds when a batch driver supplies them (so every clip on the tile authors byte-identical
    # geometry), else this clip's own per-side command.
    top_vx = getattr(args, "platform_top_vx", None) or (args.exit_vx if ground_on_approach
                                                        else args.approach_vx)
    ground_vx = plat_vx or ((args.approach_vx if ground_on_approach else args.exit_vx) or 1.0)
    plat_sign = box_platform_sign(family)
    # A ONE-SIDED tile has no elevated segment, so it needs no platform to walk on: it is the bare jump
    # box -- exactly the geometry this family had before the platform existed.  ``box_tile_boxes``
    # degenerates to a single box at ``platform_len <= 0``, so there is no separate code path.  The split
    # is a TILE property, not a clip one: every clip on a terrain_id must emit byte-identical geometry
    # (the RL importer meshes one mesh per tile), so platformed and bare clips cannot share a tile.
    two_sided = bool(getattr(args, "two_sided", True))
    # BUFFER past the nominal travel so the reference cannot walk off the far edge.  Needed because the
    # nominal is only an estimate in BOTH directions: the elevated segment covers roughly
    # vx * (s - 1.1) (gait initiation eats the first ~1.1 s, so usually LESS), but the exit ramp was
    # measured overshooting its command by ~12%, so it can also be MORE.  The ground side needs no such
    # margin -- a box tile keeps its floor, so running long there is flat ground, not a hole.
    buf = float(getattr(args, "box_platform_buffer", 1.0) or 0.0)
    plat_len = (max(buf, float(top_s) * max(0.2, abs(float(top_vx or 1.0))) + buf) if two_sided else 0.0)
    boxes = box_tile_boxes(box_size=box_size, platform_len=plat_len, platform_sign=plat_sign,
                           approach_len=max(1.0, float(ground_s) * max(0.3, abs(float(ground_vx))) + 1.0),
                           nav_width=(getattr(args, "box_nav_width", None) or None))
    lower_z = 0.0
    upper_z = float(boxes[0]["pos"][2]) + float(boxes[0]["size"][2]) / 2.0
    meta = {"geom": "box", "box_size": box_size, "box_top": upper_z,
            "box_height": float(tm.get("box_height", upper_z)), "clip_type": "box",
            "platform_len": float(plat_len), "platform_sign": float(plat_sign),
            "platform_buffer": buf, "two_sided": two_sided, "nav_width": float(boxes[0]["size"][1])}
    return boxes, None, lower_z, upper_z, meta


def _place(family, ts, tile, boxes, dev, k: int):
    """Place the replay on the tile from source frame ``k`` (looped for periodic, played once for episodic).

    Both families place TRAVEL-ALIGNED, i.e. running along the tile's ``+x`` -- which is what the tile
    layout (run-up, top platform, ``exit_sign``) and the world-frame anchor mask both assume.  For the
    stair path that is automatic; the box path has to ask for it (``align="facing"``), because a solved
    box's own yaw is 0 no matter which way the robot walked, so normalizing by it is a no-op that would
    leave the interaction on its raw source heading.
    """
    if family.periodic:
        return place_periodic_on_tile(ts, tile, boxes, dev, k=k, truncate=False, going_up=family.going_up)
    return place_episodic_on_tile(ts, boxes[0], dev, k=k, align="facing")


def _soles(builder, Wp, Wq, Wj, dt) -> np.ndarray:
    """Per-frame sole heights ``[T,2]`` of a placed segment (FK through the clip-output builder)."""
    y, _dy, _c = builder.build(Wp, Wq, Wj, dt)
    return sole_heights(y.cpu().numpy(), list(builder.pos_names))


def make_builder(args, dev) -> ClipOutputBuilder:
    """The clip-output builder for ``args.source``'s channel layout.

    Split out because constructing it loads the whole traj-opt library (1470 trajectories), so a batch
    driver building many clips must make ONE and pass it in rather than paying that per clip.
    """
    manager = MultiSkillManager(args.source, dev, env=None)
    return ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                             contact_frames=_CONTACT_FRAMES, device=dev)


def ingest_family_sources(skill: str, args, dev):
    """Terrain-library references for one family (the other expensive, per-family-not-per-clip load)."""
    family = family_for(skill)
    sources = ingest_terrain_library(args.library, hf_repo=(args.hf_repo or None), dt=args.dt,
                                     device=dev, skills=[family.skill])
    if not sources:
        raise SystemExit(f"no {family.skill} references in {args.library}")
    return sources


def plan_from_spec(sources, args, dev, builder) -> TerrainPlan:
    """Pass-1 CORE: reference selection -> tile authoring -> replay placement -> boundary frames -> plan.

    Takes the already-ingested ``sources`` and an already-built ``builder`` so a batch driver can plan many
    tiles per family without re-paying either load; ``args`` carries the per-tile spec (geometry, speeds,
    durations, ``terrain_id``) exactly as the CLI supplies it.
    """
    family = family_for(args.skill)
    ri = select_terrain_reference(
        sources,
        riser=(args.riser if family.geom == "stairs" else None),
        tread=(args.tread if family.geom == "stairs" else None),
        box_height=(args.box_height if family.geom == "box" else None),
        speed=args.terrain_speed)
    ts = sources[ri]
    tm = ts.terrain_meta["terrain"]
    ref_speed = float(ts.steady_vel[0]) if ts.steady_vel is not None else 0.0
    print(f"[mb_terrain] {family.skill}: {family.blurb}")
    print(f"[mb_terrain]   reference {ri}/{len(sources)} '{ts.name}' "
          + (f"riser={tm.get('riser', 0):.3f} tread={tm.get('tread', 0):.3f} "
             if family.geom == "stairs" else f"box_h={tm.get('box_height', 0):.3f} ")
          + f"speed={ref_speed:.2f} frames={ts.n} wrap={ts.wrap}")

    boxes, tile, lower_z, upper_z, tile_meta = _author_tile(family, ts, args)
    approach_surf = lower_z if family.approach_level == "lower" else upper_z
    exit_surf = lower_z if family.exit_level == "lower" else upper_z

    Wp, Wq, Wj, tc = _place(family, ts, tile, boxes, dev, k=0)
    contact = (tc.cpu().numpy() if tc is not None else np.zeros((Wp.shape[0], 2), np.uint8))
    entry_gate = exit_gate = None
    k = 0                                          # source frame the replay is (re-)anchored from
    if family.periodic:
        riser = float(tile.step_height)
        swing_clear = args.swing_clear if args.swing_clear is not None else min(0.05, 0.5 * riser)
        swing_band = args.swing_band if args.swing_band is not None else max(0.5 * riser, swing_clear + 0.05)
        levels = _tread_tops(boxes)
        # ENTRY: any frame of the FIRST period that stands on a tread with the swing foot just above it.
        # The placement re-anchors whatever tread the chosen frame stands on onto the tile's first tread,
        # so the candidate is judged against ALL tread levels and the re-placed clip starts at frame 0.
        entry_gate = boundary_candidates(_soles(builder, Wp, Wq, Wj, args.dt), contact, levels,
                                         tol=args.contact_tol, swing_clear=swing_clear,
                                         swing_band=swing_band, window=(0, int(ts.n)))
        k = pick_boundary(entry_gate, prefer=args.entry_pick)
        print(f"[mb_terrain]   entry gate: {entry_gate.reason} -> frame {k} (source period {ts.n} frames, "
              f"{'strict' if entry_gate.strict else 'FALLBACK'})")
        if k != 0:
            Wp, Wq, Wj, tc = _place(family, ts, tile, boxes, dev, k=k)
            contact = (tc.cpu().numpy() if tc is not None else np.zeros((Wp.shape[0], 2), np.uint8))
        entry_frame = 0
        # EXIT: the first clean arrival on the far surface, so the traversal stops as soon as it has landed
        # and never steps onto a phantom tread past the tile.
        exit_gate = boundary_candidates(_soles(builder, Wp, Wq, Wj, args.dt), contact, [exit_surf],
                                        tol=args.contact_tol, swing_clear=swing_clear,
                                        swing_band=swing_band)
        if exit_gate.frames.size == 0:
            raise SystemExit(f"{family.skill}: the traversal never lands a foot on the exit surface "
                             f"z={exit_surf:.3f} -- check --n-steps / the reference geometry")
        exit_frame = pick_boundary(exit_gate, prefer=args.exit_pick)
        print(f"[mb_terrain]   exit gate: {exit_gate.reason} -> frame {exit_frame}/{Wp.shape[0] - 1} "
              f"({'strict' if exit_gate.strict else 'FALLBACK'})")
    else:
        entry_frame, exit_frame = episodic_boundaries(int(Wp.shape[0]))
        print(f"[mb_terrain]   episodic: replaying the whole interaction, frames {entry_frame}..{exit_frame}")

    entry_vx = _heading_speed(Wp, Wq, entry_frame, args.dt)
    exit_speed = _heading_speed(Wp, Wq, exit_frame, args.dt)
    # A walking approach defaults to the reference's own steady speed; a box interaction records no command
    # (its reference is a standing jump), so fall back to the requested traversal speed.  A STANDING end
    # commands zero -- there is nowhere to walk.  Both directions come from the family, never from the sign
    # of the requested magnitude: a backward dismount walks in BEFORE the drop and away AFTER it.
    approach_mag = abs(float(args.approach_vx)) if args.approach_vx is not None \
        else (abs(ref_speed) or abs(args.terrain_speed))
    approach_vx = float(family.approach_sign) * (approach_mag if family.approach_mode == "walk" else 0.0)
    exit_vx = float(family.exit_sign) * (abs(float(args.exit_vx)) if family.exit_mode == "walk" else 0.0)
    # The RL / flow-matching command: ONE steady velocity for the whole clip, the terrain-agnostic thing a
    # user or planner would ask for.  Deliberately NOT the generator's own command, which slows into the
    # interaction and picks back up -- see the module docstring of .terrain_families.  ``terrain_speed`` is
    # the requested speed (it is also what select_terrain_reference matched the reference to, so a 1.2 m/s
    # request can legitimately be served by a 0.9 m/s stair reference); the family supplies the direction.
    rl_vx = float(family.exit_sign) * abs(float(args.terrain_speed))

    qpos = torch.cat([Wp, Wq, Wj], dim=-1).detach().cpu().numpy().astype(np.float32)
    meta = dict(tile_meta)
    meta.update({
        # Coerce the placement-critical fields to plain floats (they may arrive as numpy scalars, which
        # do not survive the JSON round trip), but carry every OTHER authored key through verbatim: a
        # support-structure key like ``variant`` / ``slab_thickness`` is authored here and consumed much
        # later by ``overlays_for``, so dropping unrecognised keys silently downgrades the tile to solid.
        "boxes": [{**{k: v for k, v in b.items() if k not in ("pos", "size", "yaw")},
                   "pos": [float(v) for v in b["pos"]], "size": [float(v) for v in b["size"]],
                   "yaw": float(b.get("yaw", 0.0))} for b in boxes],
        "reference": ts.name, "reference_speed": ref_speed, "rl_vx": float(rl_vx),
        "terrain_id": int(args.terrain_id), "footprint_margin": float(args.footprint_margin),
        "family_blurb": family.blurb, "exit_sign": float(family.exit_sign),
        "lower_z": float(lower_z), "upper_z": float(upper_z),
        "entry_strict": bool(entry_gate.strict) if entry_gate else True,
        "exit_strict": bool(exit_gate.strict) if exit_gate else True,
        "entry_source_frame": int(k),               # which entry candidate the replay was re-anchored from
    })
    plan = TerrainPlan(
        skill=family.skill, dt=float(args.dt), terrain_qpos=qpos,
        terrain_contact=contact.astype(np.uint8),
        entry_frame=int(entry_frame), exit_frame=int(exit_frame),
        approach_surf_z=float(approach_surf), exit_surf_z=float(exit_surf),
        approach_vx=float(approach_vx), exit_vx=float(exit_vx),
        entry_vx=float(entry_vx), exit_speed=float(exit_speed),
        approach_mode=family.approach_mode, exit_mode=family.exit_mode,
        approach_s=float(args.approach_s), exit_s=float(args.exit_s),
        entry_candidates=(entry_gate.frames if entry_gate else np.zeros(0, np.int64)),
        exit_candidates=(exit_gate.frames if exit_gate else np.zeros(0, np.int64)),
        meta=meta,
    )
    return plan


def print_plan(plan: TerrainPlan) -> None:
    """Summarize a planned traversal: replay length / climb, and each MB segment's command + surface."""
    rz = plan.replay[:, 2]
    print(f"[mb_terrain]   replay {plan.replay.shape[0]} frames ({plan.replay.shape[0] * plan.dt:.2f}s) "
          f"pelvis_z {rz[0]:.2f} -> {rz[-1]:.2f} ({rz[-1] - rz[0]:+.2f} m)")
    print(f"[mb_terrain]   approach: {plan.approach_mode} @ {plan.approach_vx:+.2f} m/s on "
          f"z={plan.approach_surf_z:.2f} -> entry speed {plan.entry_vx:+.2f} m/s")
    print(f"[mb_terrain]   exit:     {plan.exit_mode} @ {plan.exit_vx:+.2f} m/s on z={plan.exit_surf_z:.2f} "
          f"<- exit speed {plan.exit_speed:+.2f} m/s")
    print(f"[mb_terrain]   RL target_vel (whole clip, terrain-agnostic) = "
          f"{float(plan.meta.get('rl_vx', 0.0)):+.2f} m/s "
          f"(reference solved at {float(plan.meta.get('reference_speed', 0.0)):+.2f})")


def _dump_plan(args, dev) -> int:
    """Pass 1 (CLI): plan ONE tile traversal and write it out."""
    sources = ingest_family_sources(args.skill, args, dev)
    plan = plan_from_spec(sources, args, dev, make_builder(args, dev))
    save_terrain_plan(args.dump_plan, plan)
    print_plan(plan)
    print(f"[mb_terrain] wrote plan -> {args.dump_plan}")
    return 0


def _load_segment(path: str | None, dt: float, dev):
    """Load an MB segment ``.npz`` -> ``((P, Q, J), provenance)`` resampled to ``dt`` (or Nones).

    The segment's own ``target_vel`` -- the command Motion Bricks was DRIVEN with, ramps and all -- is
    deliberately dropped here.  It used to be copied straight into the clip as the RL command, which gave
    the policy and the flow model a jagged terrain-shaped profile (measured on a jump_off_front clip:
    +0.67 through the approach, 0.00, then a step to +2.05 for the replay) that no planner would ever
    emit.  The RL command is one steady value for the whole clip instead; see ``plan.meta["rl_vx"]``.
    """
    if not path:
        return None, {}
    raw = load_raw_motion(path)
    Wp, Wq, Wj, _tvel = raw_to_world(raw, dt, device=dev)
    return (Wp, Wq, Wj), dict(raw.provenance or {})


def clip_from_plan(plan: TerrainPlan, args, builder, dev) -> dict:
    """Pass-2 CORE: splice ``args.approach_raw`` / ``args.exit_raw`` around the plan's replay.

    Returns the per-clip record a library writer needs -- ``y``, ``dy``, ``contact``, ``anchor_mask_t``,
    ``target_vel``, ``manifest`` (one terrain-manifest entry) plus the geometry/diagnostic fields
    (``bounds``, ``diag``, ``overlays``, ``size_x/size_y``, ``Wp``) -- WITHOUT writing anything, so a batch
    driver can size a tile to the union of its clips before emitting them.
    """
    family = family_for(plan.skill)
    dt = float(plan.dt)
    boxes = plan.boxes
    if not boxes:
        raise SystemExit(f"plan for {plan.skill} carries no tile geometry")

    approach, a_prov = _load_segment(args.approach_raw, dt, dev)
    exit_seg, e_prov = _load_segment(args.exit_raw, dt, dev)
    # The SEAM treatment is a splice-time choice (default: whatever Stage 1 pinned for), but the segment's
    # coordinate FRAME is a fact of how it was generated -- a context-seeded exit already stands on the box.
    a_pin = a_prov.get("pin", "hard") if args.approach_pin in (None, "auto") else args.approach_pin
    e_pin = e_prov.get("pin", "hard") if args.exit_pin in (None, "auto") else args.exit_pin
    a_seam = "weld" if a_pin == "hard" else "ease"
    e_seam = "weld" if e_pin == "hard" else "ease"
    a_frame = str(a_prov.get("frame", "flat"))
    e_frame = str(e_prov.get("frame", "flat"))
    e_seeded = bool(e_prov.get("seeded", False))
    replay = torch.as_tensor(plan.replay, dtype=torch.float32, device=dev)
    terrain = (replay[:, 0:3], replay[:, 3:7], replay[:, 7:36])
    print(f"[mb_terrain] {plan.skill}: replay {replay.shape[0]}f + approach "
          f"{0 if approach is None else approach[0].shape[0]}f ({a_frame}/{a_seam}) + exit "
          f"{0 if exit_seg is None else exit_seg[0].shape[0]}f ({e_frame}/{e_seam}"
          f"{'/seeded' if e_seeded else ''})")

    blend_n = int(round(args.seam_blend_s / dt))
    Wp, Wq, Wj, bounds, diag = splice_terrain(
        terrain, approach, exit_seg,
        approach_surf_z=plan.approach_surf_z, exit_surf_z=plan.exit_surf_z, blend_n=blend_n,
        approach_seam=a_seam, exit_seam=e_seam, approach_frame=a_frame, exit_frame=e_frame,
        exit_seeded=e_seeded, weld_search=args.weld_search, arrival_search=args.arrival_search, dt=dt)
    for side in ("approach", "exit"):
        if f"{side}_joint_res" not in diag:
            continue
        extra = (f" arrival_cut={diag['approach_arrival_cut']}f" if side == "approach"
                 and "approach_arrival_cut" in diag else "")
        extra += f" overlap_skip={diag['exit_overlap_skip']}" if side == "exit" else ""
        print(f"[mb_terrain]   {side} seam: {diag[side + '_frame']}/{diag[side + '_seam']} "
              f"surface_err={diag[side + '_surface_err']:+.3f} m "
              f"joint_res={diag[side + '_joint_res']:.3f} rad "
              f"seam_speed={diag.get(side + '_seam_speed', float('nan')):.2f} m/s{extra}")

    margin = float(plan.meta.get("footprint_margin", 1.0))
    Wp, corner, size_x, size_y = footprint_recenter(Wp, boxes, margin)
    overlays = overlays_for(boxes, corner)
    if plan.meta.get("geom") == "stairs":
        # Widen the WIDE approach / recovery platforms to the tile width so a laterally drifting MB segment
        # cannot walk off them onto the ground a riser below (the treads keep their authored width).
        min_sx = min(o["size_xyz"][0] for o in overlays)
        for o in overlays:
            if o["size_xyz"][0] > 1.5 * min_sx:
                o["size_xyz"][1] = size_y
                o["pos_xy"][1] = size_y / 2.0

    y, dy, contact = builder.build(Wp, Wq, Wj, dt)
    ta, tb = bounds[1]
    contact[ta:tb] = torch.as_tensor(plan.replay_contact, dtype=contact.dtype, device=contact.device)
    T = int(Wp.shape[0])
    seg_masks = [_ANCHOR_MASK_FLAT, _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT]
    amt = build_anchor_mask_t(T, bounds, seg_masks, device=dev)

    # RL-visible command: ONE steady value for the whole clip -- what a user or planner asked for, with no
    # knowledge of the terrain.  The motion underneath does whatever the terrain requires (slow into the
    # takeoff, jump, pick back up); teaching the flow model to bridge that gap is the entire point, and it
    # can only learn it from clips that pair a terrain-agnostic command with a terrain-aware reference.
    tvel = torch.zeros(T, 3, dtype=torch.float32, device=dev)
    if "rl_vx" not in plan.meta:
        # NEVER default this. A plan written before the RL-command split carries the old per-family
        # ``terrain_vx`` (the replay's own speed, or the approach command held through it) -- a different
        # quantity, not a stand-in for the user command. Defaulting to 0.0 wrote a SILENT zero-velocity
        # command into every clip, which trains wrong without erroring anywhere.
        raise SystemExit(
            f"plan for {plan.skill} predates the RL-command split: it has "
            f"{'terrain_vx' if 'terrain_vx' in plan.meta else 'neither rl_vx nor terrain_vx'}, not 'rl_vx'. "
            f"Re-run pass 1 (mb_terrain_clips --dump-plan / mb_plan_library) to regenerate the plans.")
    tvel[:, 0] = float(plan.meta["rl_vx"])

    # ``frame_yaw``: the heading of the TERRAIN's own frame within the tile -- the axis the traversal
    # crosses it along.  The RL anchor mask resolves its downrange / lateral bits in this frame
    # (world_lift.ego_placement_masked), so it is AUTHORED here rather than inferred at runtime from
    # the motion (which is degenerate for a standing approach and only approximate otherwise).  0.0
    # because ``_place`` puts every family travel-aligned along the tile's +x: the stair path by
    # construction, the box path via ``align="facing"``.  If a family is ever authored on a rotated
    # heading, this is the ONE number that has to change with it.
    manifest = {"terrain_id": int(plan.meta.get("terrain_id", 0)),
                "type": str(plan.meta.get("clip_type", "transition")),
                "size_x": float(size_x), "size_y": float(size_y), "overlays": overlays,
                "has_floor": bool(plan.meta.get("has_floor", True)),
                "frame_yaw": 0.0}
    return {"y": y, "dy": dy, "contact": contact, "anchor_mask_t": amt, "target_vel": tvel,
            "manifest": manifest, "bounds": bounds, "diag": diag, "overlays": overlays,
            "size_x": float(size_x), "size_y": float(size_y), "Wp": Wp, "n": T,
            # The tile-local origin this clip was recentred onto. Returned because it depends on the CLIP's
            # own extent (``footprint_recenter`` takes the min of motion and geometry), so a batch driver
            # putting several clips on one tile must re-datum them to a common origin -- otherwise each clip
            # would place the same geometry at a different tile-local position.
            "corner": (float(corner[0]), float(corner[1])),
            "seams": {"approach": f"{a_frame}/{a_seam}", "exit": f"{e_frame}/{e_seam}"}}


def _write_keyframe_sidecar(srcs: dict, out: str, diag: dict, corner, dev) -> int:
    """Map the SEGMENTS' pinned keyframes into the spliced clip's frame and write one viser sidecar.

    ``mb_terrain_segments`` records the keyframe Motion Bricks was actually aiming at on each replan, in
    the segment's own MB world frame.  The splice then moves that segment rigidly onto the terrain
    boundary and the tile is re-datumed, so the keyframes have to ride along or they land metres away.
    Both transforms are replayed here: ``diag["<side>_xform"]`` from the splice, then the footprint
    corner shift.

    ``srcs`` maps side ("approach" / "exit") -> that segment's ``_keyframes.npz``; both are merged into
    one sidecar so the overlay shows the whole clip at once.  NOTE the two sides are recorded by
    SEPARATE generator runs (``mb.reset`` clears the buffer), so generate them with ``--side approach``
    / ``--side exit`` if you want both -- a ``--side both`` run leaves only the exit's.

    The ``kind`` tag ("stride" / "terrain") is carried through untouched -- it is what lets the viewer
    show that a TERRAIN keyframe pins a pose but not a place (its root xy is overwritten by the command
    rollout unless ``--pin-place terrain``; see ``mb_env.target_pose_override``).  Exit keyframes are
    always "stride": the exit is context-SEEDED from the replay rather than pose-pinned to it.

    Returns the number of keyframes written.  Note an EASED seam additionally inertializes the segment
    near the join, which the keyframes do not reflect -- they are where the model was AIMED, not where it
    ended up.
    """
    import json

    from motion_gen.math_utils import quat_apply, quat_mul

    qs, ts, kinds, jn, fps = [], [], [], None, None
    for side, src in sorted(srcs.items()):
        if not src or f"{side}_xform" not in diag:
            continue
        d = np.load(src, allow_pickle=False)
        q, tk = d["qpos"], d["t"]
        if not len(q):
            continue
        gq_l, t_l = diag[f"{side}_xform"]
        gq = torch.tensor(gq_l, dtype=torch.float32, device=dev)
        tv = torch.tensor(t_l, dtype=torch.float32, device=dev)
        P = torch.as_tensor(q[:, 0:3], dtype=torch.float32, device=dev)
        Q = torch.as_tensor(q[:, 3:7], dtype=torch.float32, device=dev)
        n = P.shape[0]
        P = quat_apply(gq.unsqueeze(0).expand(n, 4), P) + tv
        Q = quat_mul(gq.unsqueeze(0).expand(n, 4), Q)
        P[:, 0] -= float(corner[0])
        P[:, 1] -= float(corner[1])
        qs.append(np.concatenate([P.cpu().numpy(), Q.cpu().numpy(), q[:, 7:]], 1).astype(np.float32))
        ts.append(tk)
        kinds += json.loads(str(d["kind"])) if "kind" in d else ["stride"] * n
        jn, fps = d["joint_names"], d["fps"]
    if not qs:
        return 0
    out_q = np.concatenate(qs, 0)
    np.savez_compressed(out, clip_idx=np.zeros(len(out_q), np.int32), t=np.concatenate(ts),
                        qpos=out_q, kind=np.array(json.dumps(kinds)), joint_names=jn, fps=fps)
    return len(out_q)


def _splice_clip(args, dev) -> int:
    """Pass 2 (CLI): splice ONE planned traversal and write a one-clip library."""
    plan = load_terrain_plan(args.plan)
    builder = make_builder(args, dev)
    rec = clip_from_plan(plan, args, builder, dev)
    y, dy, contact = rec["y"], rec["dy"], rec["contact"]
    amt, tvel, manifest = rec["anchor_mask_t"], rec["target_vel"], rec["manifest"]
    bounds, diag, overlays = rec["bounds"], rec["diag"], rec["overlays"]
    Wp, T, dt = rec["Wp"], int(rec["n"]), float(plan.dt)
    size_x, size_y = rec["size_x"], rec["size_y"]
    meta = {
        "dt": dt, "clip_length_s": T * dt, "num_steps": T,
        "pos_output_names": list(builder.pos_names), "vel_output_names": list(builder.vel_names),
        "contact_frames": _CONTACT_FRAMES, "ref_frame": "pelvis_link", "skill_list": [plan.skill],
    }
    writer = ClipLibraryWriter(args.out, 1, meta)
    writer.set_attr("generator", "motionbricks_terrain_v1")
    writer.set_attr("terrain_manifest", [manifest])
    writer.set_attr("mb_terrain_plan", {"skill": plan.skill,
                                        "approach": rec["seams"]["approach"], "exit": rec["seams"]["exit"],
                                        "entry_frame": plan.entry_frame, "exit_frame": plan.exit_frame,
                                        "reference": plan.meta.get("reference", ""),
                                        "seam": {k: float(v) for k, v in diag.items()
                                                 if isinstance(v, (int, float))}})
    writer.write_batch(0, {
        "y_des": y.unsqueeze(0), "dy_des": dy.unsqueeze(0), "contact": contact.unsqueeze(0),
        "skill_id": torch.zeros(1, T, dtype=torch.uint8),
        "stride_idx": torch.full((1, T), -1, dtype=torch.int32),
        "target_vel": tvel.unsqueeze(0),
        "terrain_id": torch.tensor([int(plan.meta.get("terrain_id", 0))], dtype=torch.int32),
        "terrain_type": manifest["type"],
        "anchor_mask": torch.tensor([_ANCHOR_MASK_BOX], dtype=torch.uint8),
        "anchor_mask_t": amt.unsqueeze(0),
        "clip_len": torch.tensor([T], dtype=torch.int32),
    })
    writer.close()

    # Report how well each segment sits on its own surface: the sole should touch it (0) and never sink far
    # below it.  A hard pin trades a clean seam for exactly this error, so it is the number to compare.
    # Also count frames whose root has left every walkable box -- a segment longer than the platform it runs
    # on (or one that drifted sideways off it) walks the robot into thin air, which no seam metric shows.
    soles = sole_heights(y.cpu().numpy(), list(builder.pos_names))
    off = _off_surface(Wp, overlays)
    has_floor = bool(manifest["has_floor"])
    for name, (a, b), surf in (("approach", bounds[0], plan.approach_surf_z),
                               ("replay", bounds[1], None),
                               ("exit", bounds[2], plan.exit_surf_z)):
        if b <= a:
            continue
        lo = float(soles[a:b].min())
        extra = "" if surf is None else f" (surface z={surf:.3f}, penetration {min(0.0, lo - surf):+.3f} m)"
        # An ELEVATED segment must stay over a walkable box; a ground-level one is carried by the tile's flat
        # floor (so being "off every box" is normal there), and the replay is world-pinned + partly airborne.
        n_off = int(off[a:b].sum()) if (surf is not None and (surf > 1e-3 or not has_floor)) else 0
        print(f"[mb_terrain]   {name:8s} frames {a:4d}..{b - 1:4d}  min sole z={lo:+.3f}{extra}"
              + (f"  [WARN] {n_off} frames OFF the walkable surface (segment longer than the platform, "
                 f"or drifted off it)" if n_off else ""))
    if args.approach_keyframes or args.exit_keyframes:
        kf_out = os.path.splitext(args.out)[0] + "_keyframes.npz"
        n_kf = _write_keyframe_sidecar({"approach": args.approach_keyframes,
                                        "exit": args.exit_keyframes},
                                       kf_out, diag, rec["corner"], dev)
        print(f"[mb_terrain]   {n_kf} pinned keyframes -> {kf_out} "
              f"(view: view_clip_viser.py --lib {args.out} --keyframes {kf_out})")
    lib = load_clip_library(args.out)
    yl = torch.as_tensor(lib.y_des)
    zc = yl[0, :, 2]
    print(f"[mb_terrain] wrote {args.out}: y_des {tuple(yl.shape)} finite={bool(torch.isfinite(yl).all())} "
          f"pelvis_z[{float(zc.min()):.2f}->{float(zc.max()):.2f}] net={float(zc[-1] - zc[0]):+.2f} m")
    print(f"[mb_terrain]   manifest: type={manifest['type']} overlays={len(overlays)} "
          f"size=({size_x:.1f},{size_y:.1f}) has_floor={manifest['has_floor']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skill", required=True, help="Terrain family: stairs_up | stairs_down | jump_on | "
                                                 "jump_off | jump_off_front.")
    p.add_argument("--source", required=True, help="Traj-opt library root -- robot pos/vel output-name layout.")
    p.add_argument("--dt", type=float, default=0.02, help="RL control dt (clip sampling period).")
    p.add_argument("--device", default="cpu")
    # pass 1 (plan)
    p.add_argument("--dump-plan", default=None, help="PASS 1: write the terrain plan to this .npz and exit.")
    p.add_argument("--library", default="trajectories/skills/library_v2", help="Terrain/clean library folder.")
    p.add_argument("--hf-repo", default="zolkin/robot_rl")
    p.add_argument("--riser", type=float, default=0.15, help="[stairs] target riser (m).")
    p.add_argument("--tread", type=float, default=0.25, help="[stairs] target tread depth (m).")
    p.add_argument("--n-steps", type=int, default=6, help="[stairs] number of treads on the tile.")
    p.add_argument("--width", type=float, default=1.4, help="[stairs] tread y-width (m).")
    p.add_argument("--nav-width", type=float, default=None, help="[stairs] platform y-width (default --width).")
    p.add_argument("--stair-float-thickness", type=float, default=0.0,
                   help="[stairs] FLOATING treads: slab thickness (m) of each riser-less tread, leaving an "
                        "open gap beneath it (0 = solid steps). The tread-top surface is identical either "
                        "way, so the reference motion is unchanged -- only the mesh below it. Clamped per "
                        "tread to that tread's height.")
    p.add_argument("--box-height", type=float, default=0.45, help="[box] target box height (m).")
    p.add_argument("--box-platform-buffer", type=float, default=1.0,
                   help="[box] extra metres of top platform beyond the elevated segment's nominal travel, "
                        "so the reference cannot walk off the far edge.")
    p.add_argument("--box-nav-width", type=float, default=0.0,
                   help="[box] y-width (m) to widen the box AND its top platform to (expand-only); "
                        "0 keeps the reference box's own extents.")
    p.add_argument("--terrain-speed", type=float, default=0.6,
                   help="Target speed for the reference pick (m/s); also the stair traversal command.")
    p.add_argument("--approach-vx", type=float, default=None,
                   help="Steady approach command (m/s); default = the reference speed (walk) or 0 (stand).")
    p.add_argument("--exit-vx", type=float, default=1.0,
                   help="Steady exit walk-away command magnitude (m/s); the family sets the sign.")
    p.add_argument("--approach-s", type=float, default=2.5, help="MB approach duration (s).")
    p.add_argument("--exit-s", type=float, default=2.5, help="MB exit duration (s).")
    p.add_argument("--entry-pick", default="first", choices=["first", "middle", "last"],
                   help="[periodic] which filtered ENTRY candidate to hand-pick.")
    p.add_argument("--exit-pick", default="first", choices=["first", "middle", "last"],
                   help="[periodic] which filtered EXIT candidate to hand-pick.")
    p.add_argument("--contact-tol", type=float, default=0.02,
                   help="How close a planted sole must be to a surface to count as ON it (m).")
    p.add_argument("--swing-clear", type=float, default=None,
                   help="Min swing-foot height above the stance surface at a boundary (default 0.5*riser, "
                        "capped at 0.05 m).")
    p.add_argument("--swing-band", type=float, default=None,
                   help="Max swing-foot height above the stance surface at a boundary (default 0.5*riser).")
    p.add_argument("--terrain-id", type=int, default=0, help="Tile id written to the clip + manifest.")
    p.add_argument("--footprint-margin", type=float, default=1.0, help="Tile margin around the geometry (m).")
    p.add_argument("--has-floor", action="store_true", default=True,
                   help="[stairs] keep the flat ground plane under the tile.")
    p.add_argument("--no-floor", dest="has_floor", action="store_false",
                   help="[stairs] isolated structure: no ground plane under the tile.")
    # pass 2 (splice)
    p.add_argument("--plan", default=None, help="PASS 2: the terrain plan written by --dump-plan.")
    p.add_argument("--approach-raw", default=None, help="PASS 2: MB approach RawMotion .npz.")
    p.add_argument("--exit-raw", default=None, help="PASS 2: MB exit RawMotion .npz.")
    p.add_argument("--approach-keyframes", default=None,
                   help="PASS 2: the approach segment's '_keyframes.npz' (written by mb_terrain_segments "
                        "--plan). Maps the pinned target keyframes through the splice into clip "
                        "coordinates and writes '<out>_keyframes.npz' for view_clip_viser --keyframes.")
    p.add_argument("--exit-keyframes", default=None,
                   help="PASS 2: the exit segment's '_keyframes.npz', merged into the same sidecar. The two "
                        "sides are recorded by separate generator runs, so use --side approach / --side "
                        "exit to capture both (a --side both run keeps only the exit's).")
    p.add_argument("--out", default=None, help="PASS 2: output clip library .h5.")
    p.add_argument("--approach-pin", default="soft", choices=["hard", "soft", "auto"],
                   help="PASS 2: seam treatment for the approach -- 'hard' welds the boundary frame with no "
                        "ease, 'soft' (default) inertializes the residual, 'auto' follows whatever Stage 1 "
                        "pinned for. A PINNED segment spliced SOFT is the hybrid and the measured best: the "
                        "pin buys the boundary POSE, the ease keeps the feet on their own surface. Welding "
                        "instead leaves a stairs_up approach ~0.07 m INSIDE the platform (surface_err is "
                        "exactly 0 when eased) and a nonzero seam speed. This matches "
                        "mb_assemble_library --seams, whose default is likewise ease.")
    p.add_argument("--exit-pin", default="soft", choices=["hard", "soft", "auto"],
                   help="PASS 2: seam treatment for the exit (see --approach-pin).")
    p.add_argument("--seam-blend-s", type=float, default=0.25,
                   help="PASS 2: seam inertialization window (s); only used by SOFT seams.")
    p.add_argument("--weld-search", type=int, default=12,
                   help="PASS 2: leading exit frames searched for the context-seed overlap (hard exit).")
    p.add_argument("--arrival-search", type=int, default=60,
                   help="PASS 2: trailing approach frames searched for the in-betweened arrival pose "
                        "(hard approach); 1 welds the last generated frame as-is.")
    args = p.parse_args()
    if bool(args.dump_plan) == bool(args.plan):
        raise SystemExit("pass exactly one of --dump-plan (pass 1) or --plan (pass 2)")
    if args.plan and not args.out:
        raise SystemExit("--plan (pass 2) needs --out")
    if args.plan and not (args.approach_raw or args.exit_raw):
        raise SystemExit("--plan (pass 2) needs at least one of --approach-raw / --exit-raw")
    for path in (args.dump_plan, args.out):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    return _dump_plan(args, args.device) if args.dump_plan else _splice_clip(args, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
