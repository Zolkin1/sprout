#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate box / terrain TRAVERSAL clips: DP(approach) + PLAY(terrain) + DP(exit).

The terrain interaction (a box jump, a stair stride) is PLAYED from the terrain traj-opt
library (``trajectories/skills/terrain_v1``, ingested with its exported contacts + geometry);
the exact backward DP only fills the free-locomotion approach (pinned to arrive at the terrain
ENTRY state) and the exit (seeded from the terrain EXIT state).  The three world-space segments
are rigid-SE(2)+z spliced (heading-normalized, seam frame dropped) and the exported box / stair
geometry rides along the same transform.  Output is a standard clip ``.h5`` with a per-STEP
anchor mask (jump-on world-pinned, approach / recovery ego) + a ``box`` terrain manifest.

Runs with NO IsaacLab.  The graph DB is flat locomotion (bones + clean strides) PLUS the jump
lead-ins (``jump_on`` / ``jump_over`` / ... families) so the approach can reach the takeoff.

    python scripts/rsl_rl/generate_box_clips.py \
        --source hf/trajectories/retargeted/<merged> --bones-cache clips/bones_cache_locomotion.h5 \
        --terrain-lib trajectories/skills/terrain_v1 --out clips/g1_box_v1.h5 --device cpu
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.motion_matching.bones_ingest import load_cache
from motion_gen.motion_matching.box_clip import (
    align_and_concat,
    build_anchor_mask_t,
    loop_periodic,
    n_periodic_loops,
    periodic_tread_offset,
    se2z_align,
    stair_tile_boxes,
    transform_boxes,
    yaw_normalize,
)
from motion_gen.math_utils import quat_apply, quat_mul
from motion_gen.motion_matching.clean_ingest import ingest_clean_library
from motion_gen.motion_matching.clip_output import ClipOutputBuilder
from motion_gen.motion_matching.dp import DPConfig, DPMatcher
from motion_gen.motion_matching.graph import GraphConfig, build_candidate_graph
from motion_gen.motion_matching.playback import inertialize_seam
from motion_gen.motion_matching.terrain_ingest import ingest_terrain_library
from motion_gen.motion_matching.terrain_sampler import sample_stair_tiles
from motion_gen.motion_matching.terrain_variants import resolve_box_variant
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig
from motion_gen.multiskill_manager import MultiSkillManager
from motion_gen.world_lift import _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT

_CONTACT = ["left_ankle_roll_link", "right_ankle_roll_link"]

# The jump / step-box lead-in families (bones filename prefixes) that make the terrain entry
# reachable from flat locomotion; exclude the ladder / wall / roll look-alikes.
_TERRAIN_INCLUDE = ["jump_over_obstacle", "jump_on", "jump_off", "come_up", "come_down"]
_TERRAIN_EXCLUDE = ["ladder", "wall", "roll", "monkey", "shoulder", "safety", "hook"]
# Flat-locomotion movement labels for the approach/exit steady tracking.
_FLAT_MOVEMENT = ["walking", "jogging", "running", "running, turning", "turning",
                  "standing idle", "transition"]
_FLAT_EXCLUDE = ["injured", "limp", "drunk", "jump", "roll"]

# Per terrain tile TYPE: which bones-source CATEGORIES the approach/recovery DP may draw from.
# The candidate graph is built ONCE over the union (flat + clean + leadin); each tile type filters it
# per-generate (see ``DPMatcher._set_category_filter``).  A stair traversal is PLAYED, so its approach
# and recovery are ordinary flat locomotion -> {flat, clean} only; the box/jump lead-ins (``come_up`` /
# ``jump_on`` ...) would otherwise leak elevated on-box frames into a flat run-up (the entry pop).  A box
# jump must SYNTHESISE a takeoff, so it also needs the lead-ins.  Pure stair tiles are matcher-free.
_APPROACH_FILTERS = {
    "stair_transition": frozenset({"flat", "clean"}),
    "box": frozenset({"flat", "clean", "leadin"}),
}


def _steady_cmd(vx: float):
    return lambda _t: (float(vx), 0.0, 0.0)


def _ramp_cmd(vx0: float, approach_s: float, ramp_s: float, vx1: float = 0.0):
    """Hold ``vx0``, then linearly decelerate to ``vx1`` over the last ``ramp_s`` seconds.

    The DP TRACKS this decelerating command (relax_tail=0), forcing a real stepping deceleration
    into the entry instead of searching a transition (which stitches jarring fast->slow cuts)."""
    t0 = max(0.0, approach_s - ramp_s)
    def cmd(t):
        a = 0.0 if t <= t0 else min(1.0, (t - t0) / max(1e-6, ramp_s))
        return (float(vx0 + (vx1 - vx0) * a), 0.0, 0.0)
    return cmd


def _ramp_up_cmd(vx0: float, ramp_s: float, vx1: float):
    """Ramp ``vx0`` -> ``vx1`` over the FIRST ``ramp_s`` seconds, then HOLD ``vx1``.

    Mirror of :func:`_ramp_cmd` for the RECOVERY: start at the terrain-exit speed (``vx0``, the played
    interaction's slow exit) and accelerate back up to the commanded target (``vx1``) so the exit motion
    matches the held-forward RL ``target_vel`` instead of walking away at the descent speed."""
    def cmd(t):
        a = 1.0 if ramp_s <= 0.0 else min(1.0, t / max(1e-6, ramp_s))
        return (float(vx0 + (vx1 - vx0) * a), 0.0, 0.0)
    return cmd


def _entry_vx(ts, dt: float) -> float:
    """Forward (heading-frame) speed at the START of the terrain clip -- the ramp target.

    The approach decelerates to whatever speed the played interaction begins at: ~0 for a standing
    ``jump_on`` (the clip opens standing), the run speed for a future running vault whose frame 0 is
    already moving.  Finite-differenced from the first two root frames, projected onto frame-0 heading."""
    if int(ts.root_pos.shape[0]) < 2:
        return 0.0
    vx, vy = ((ts.root_pos[1, :2] - ts.root_pos[0, :2]) / dt).tolist()
    w, x, y, z = (float(c) for c in ts.root_quat[0])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return max(0.0, math.cos(yaw) * vx + math.sin(yaw) * vy)


def _ground_z_under(xy, boxes) -> float:
    """The walkable surface height (world z) under a point ``(x, y)`` -- generic ground offset.

    The highest overlapping box TOP covering ``(x, y)``, or ``0.0`` (the flat floor) if none.  A flat
    DB clip is grounded to soles-at-0, so a flat segment placed on this surface is just ``DB_z + this``
    -- 0 on flat ground (no change), the platform height on a stair top platform, etc.  Boxes are
    axis-aligned here (``yaw_normalize`` zeroes their yaw), so an AABB xy-containment test suffices."""
    x, y = float(xy[0]), float(xy[1])
    tops = [float(b["pos"][2]) + float(b["size"][2]) / 2.0
            for b in boxes
            if abs(x - float(b["pos"][0])) <= float(b["size"][0]) / 2.0
            and abs(y - float(b["pos"][1])) <= float(b["size"][1]) / 2.0]
    return max(tops) if tops else 0.0


def _pad_or_trim(t: torch.Tensor, T: int) -> torch.Tensor:
    """Pad (repeat last frame) or trim a ``[t0, ...]`` tensor to length ``T`` along dim 0."""
    n = t.shape[0]
    if n == T:
        return t
    if n > T:
        return t[:T]
    pad = t[-1:].expand((T - n,) + tuple(t.shape[1:]))
    return torch.cat([t, pad], dim=0)


def _build_matcher(args, dev):
    """Flat locomotion (bones + clean) + jump lead-ins -> DPMatcher with a candidate graph."""
    print("[box] loading flat-locomotion bones ...")
    flat, _ = load_cache(args.bones_cache, filters={"package": "Locomotion",
                         "content_type_of_movement": _FLAT_MOVEMENT, "content_body_position": "standing"},
                         exclude_name_substrings=_FLAT_EXCLUDE, limit=args.bones_limit, device=dev)
    print(f"[box] + {len(flat)} flat bones clips")
    print("[box] loading jump/step lead-ins ...")
    leadins, _ = load_cache(args.bones_cache, filters={"package": "Locomotion"},
                            include_name_substrings=_TERRAIN_INCLUDE,
                            exclude_name_substrings=_TERRAIN_EXCLUDE, limit=args.leadin_limit, device=dev)
    print(f"[box] + {len(leadins)} lead-in clips")
    clean = []
    if args.include_clean:
        print(f"[box] ingesting clean library {args.clean_source} ...")
        clean = ingest_clean_library(args.clean_source, hf_repo=args.clean_hf_repo, dt=args.dt, device=dev)
        print(f"[box] + {len(clean)} clean STEADY strides")
    sources = flat + clean + leadins
    # Per-source category, aligned with ``sources`` order -- lets each tile type filter the shared DB
    # per-generate (``_APPROACH_FILTERS``) without rebuilding the candidate graph.
    categories = ["flat"] * len(flat) + ["clean"] * len(clean) + ["leadin"] * len(leadins)
    vcfg = VanillaConfig(dt=args.dt)
    dcfg = DPConfig(dt=args.dt, cut_penalty=args.cut_penalty, w_traj=1.0, w_track=0.0,
                    w_ground=args.ground_penalty, ground_z=0.0, max_pelvis_z=args.max_pelvis_z)
    nframes = sum(s.n for s in sources)
    print(f"[box] building DP matcher DB ({len(sources)} clips, {nframes} frames) ...")
    matcher = DPMatcher(sources, vcfg, dcfg, device=dev, categories=categories)
    gcfg = GraphConfig(k_min=args.graph_k_min, k_max=args.graph_k_max,
                       k_target=args.graph_k_target, dt=args.dt)
    print("[box] building candidate graph ...")
    cost_feat = matcher._norm[:, matcher._traj_dim:]
    select_feat = cost_feat[:, POSE_POSTURAL_IDX]
    graph = build_candidate_graph(select_feat, cost_feat, matcher.feat_raw, matcher.owner,
                                  matcher.src_start, matcher.src_len, matcher.yaw_all, gcfg,
                                  steady_owner_mask=matcher._src_steady,
                                  periodic_owner_mask=matcher._src_periodic)
    matcher.attach_graph(graph)
    return matcher


def _make_box_clip(matcher, builder, ts, tid, approach_vx, args, dev,
                   box_variant_seed=0, box_variant_params=None, approach_s=None, recovery_s=None):
    """One box clip -> (records dict) for padding.

    ``--mode full`` (default): DP(approach @ approach_vx) + PLAY(ts) + exit -- the traversal.
    ``--mode pure``: just PLAY(ts) from frame 0 -- the BARE interaction (spawn-at-box, world-pinned
    BOX throughout), a small-fraction skill anchor that drills the core jump undiluted by a lead-in.

    ``box_variant_seed`` / ``box_variant_params`` resolve the box SUPPORT variant
    (solid / floating-top / center-leg / corner-legs).  The seed is keyed on the
    physical terrain (not the approach speed) so a terrain's box looks identical
    across speeds; the variant is applied only when the terrain has exactly one
    walkable box (a jump box), never to multi-box stair-as-boxes traversals.  ``approach_s`` /
    ``recovery_s`` override the fixed approach/recovery durations (``--box-splice eased`` passes
    per-clip sampled values so the run-up / recovery lengths vary like the stair tiles).
    """
    approach_s = args.approach_s if approach_s is None else float(approach_s)
    recovery_s = args.recovery_s if recovery_s is None else float(recovery_s)
    # Loop a PERIODIC terrain stride (e.g. stairs) enough times to traverse its geometry; an EPISODIC
    # interaction (a box jump) loops once. Geometry-driven, so it stays terrain-generic (no stair code).
    boxes_geom = ts.terrain_meta.get("terrain", {}).get("boxes", [])
    n_loops = n_periodic_loops(ts.root_pos, ts.wrap, boxes_geom)
    # Exact per-period tread offset (prevents drift accumulating down a periodic flight; None = chain).
    period_offset = periodic_tread_offset(ts.root_pos, ts.wrap, boxes_geom)
    if args.mode == "pure":
        k = 0
        rp, rq, jt, terr_contact = loop_periodic(
            ts.root_pos.to(dev), ts.root_quat.to(dev), ts.joint_pos.to(dev),
            None if ts.contact is None else ts.contact.to(dev), n_loops, offset=period_offset)
        terr = (rp, rq, jt)
        segments, box_seg, seg_masks = [terr], 0, [_ANCHOR_MASK_BOX]
    else:
        T_app = max(1, int(round(approach_s / args.dt)))
        T_exit = max(1, int(round(recovery_s / args.dt)))
        tail = max(1, int(round(args.approach_ramp_s / args.dt)))
        # Hand off into ANY of the terrain clip's first ``entry_frac`` frames (a faster run-up can
        # merge into a later moving frame; a standing jump only has frame 0, so it lands there).
        # A DESCENDING interaction (e.g. stairs_down) is the exception: it must hand off at the TOP
        # (highest frame) so the FULL descent plays and the approach sits on the top platform.  With
        # the wide window the DP picks the frame best matching the flat walk -- a MID-descent frame --
        # skipping the top of the stairs and dropping the approach below the platform.  So for a net
        # descent, cap the entry window to the top (``descent_entry_frac``); jumps/ascents keep it wide.
        descends = float(ts.root_pos[0, 2] - ts.root_pos[-1, 2]) > args.descent_z
        efrac = min(args.entry_frac, args.descent_entry_frac) if descends else args.entry_frac
        entry_descs = matcher.entry_descriptors(ts, entry_frac=efrac)
        exit_descs = matcher.exit_descriptors(ts, exit_frac=args.exit_frac)   # exit WINDOW (mirror of entry)
        entry_vx = _entry_vx(ts, args.dt)   # the played interaction's start speed (ramp target)
        # --transition ramp (default): prescribe a velocity decel to entry_vx and TRACK it (real
        # stepping gait into the entry, least foot slide); relax: hold the command + relax tracking.
        if args.transition == "ramp":
            app_cmd = _ramp_cmd(approach_vx, approach_s, args.approach_ramp_s, vx1=entry_vx)
            ra, qa, ja = matcher.generate_approach(app_cmd, T_app, entry_descs,
                                                   boundary_weight=args.boundary_weight, relax_tail=0,
                                                   allowed_categories=_APPROACH_FILTERS["box"])
        else:
            app_cmd = _steady_cmd(approach_vx)
            ra, qa, ja = matcher.generate_approach(app_cmd, T_app, entry_descs,
                                                   boundary_weight=args.boundary_weight, relax_tail=tail,
                                                   allowed_categories=_APPROACH_FILTERS["box"])
        k = int(matcher.matched_entry)      # play the interaction FROM the matched entry frame
        # Periodic: play from the DP-chosen entry `k` (any frame in the first fraction), then WRAP the
        # full period forward -- so the seam matches where the approach actually hands off (no jump),
        # while the loop still descends the full flight. Episodic: play once from k (no wrap).
        rp, rq, jt, terr_contact = loop_periodic(
            ts.root_pos.to(dev), ts.root_quat.to(dev), ts.joint_pos.to(dev),
            None if ts.contact is None else ts.contact.to(dev), n_loops, offset=period_offset, start=k)
        terr = (rp, rq, jt)
        # Interaction-aware exit: ELEVATED end (jumped ONTO a box, no dismount) -> HOLD the
        # stand-on-box pose; else run a flat recovery DP seeded from the exit state.
        elevated = float(ts.root_pos[-1, 2] - ts.root_pos[k, 2]) > 0.15
        if elevated:
            re = ts.root_pos[-1:].to(dev).repeat(T_exit, 1)
            qe = ts.root_quat[-1:].to(dev).repeat(T_exit, 1)
            je = ts.joint_pos[-1:].to(dev).repeat(T_exit, 1)
        else:
            re, qe, je = matcher.generate_exit(_steady_cmd(entry_vx), T_exit, exit_descs,
                                               allowed_categories=_APPROACH_FILTERS["box"])
            # Peel off at the DP-chosen exit frame (mirror of the entry `matched_entry` play-from):
            # `matched_exit` indexes into the last `exit_frac` of the SOURCE period; because the loop is
            # periodic that window equals the played descent's last `m` frames, so truncate the descent
            # there and let the recovery seam in at whatever phase matched cleanest -- no single frame
            # pinned on the exit either.
            m_exit = max(1, int(round(args.exit_frac * ts.root_pos.shape[0])))
            cut = rp.shape[0] - m_exit + int(matcher.matched_exit) + 1
            rp, rq, jt = rp[:cut], rq[:cut], jt[:cut]
            terr = (rp, rq, jt)
            if terr_contact is not None:
                terr_contact = terr_contact[:cut]
        segments, box_seg, seg_masks = ([(ra, qa, ja), terr, (re, qe, je)], 1,
                                        [_ANCHOR_MASK_FLAT, _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT])

    # --- Choose the splice: EASED (jump-anchored, the SAME path the stair tiles use) or the legacy
    # rigid SE(2)+z WELD.  Eased applies only to an EPISODIC full traversal (a box jump); pure mode and
    # periodic (stairs-as-boxes) traversals stay on the weld. ---
    use_eased = (args.box_splice == "eased" and args.mode != "pure"
                 and box_seg == 1 and n_loops == 1)
    if use_eased:
        # Jump is the ANCHOR (kept pristine), so its box geometry rides along UNCHANGED (no per-segment
        # transform) and stays 1:1 with the played contacts (c_off=0).  The flat approach / recovery are
        # placed grounded to their OWN surface and each seam is eased via inertialize_seam (not welded).
        boxes = ts.terrain_meta["terrain"].get("boxes", [])
        jP, jQ = terr[0], terr[1]
        top_surf = _ground_z_under(jP[0, :2], boxes)                          # surface under the jump ENTRY
        bot_surf = 0.0 if elevated else _ground_z_under(jP[-1, :2], boxes)    # held pose is already placed
        blend_n = int(round(args.seam_blend_s / args.dt))
        Wp, Wq, Wj, bounds, box_seg = _splice_around(terr, (ra, qa, ja), (re, qe, je),
                                                     top_surf, bot_surf, blend_n)
        Wp, Wq, boxes = yaw_normalize(Wp, Wq, boxes, jQ[0])   # axis-align the whole traversal by the box yaw
        c_off = 0
    else:
        if args.box_splice == "eased" and args.mode != "pure" and box_seg == 1 and n_loops > 1:
            print("[box][WARN] --box-splice eased is episodic-only; this periodic traversal falls back to weld.")
        Wp, Wq, Wj, bounds, tfs = align_and_concat(segments)

        # The exported box geometry (in the terrain clip's own frame) glued to the played segment via
        # its splice transform -- this is where the CLIP says the box goes, not a heuristic.
        gq_t, t_t = tfs[box_seg]
        boxes = transform_boxes(ts.terrain_meta["terrain"].get("boxes", []), gq_t, t_t)
        # Normalize the whole traversal (clip + boxes) so the terrain is axis-aligned (box yaw -> ~0):
        # the approach heading is arbitrary, and leaving it in yaws the box under the motion.
        Wp, Wq, boxes = yaw_normalize(Wp, Wq, boxes, gq_t)

        # Place a looped periodic traversal at the FIRST tread. The source period is cropped from wherever
        # it happened to sit on the staircase (often MID-flight), so its entry lands on a middle tread and
        # the N loops then run off the end. Detect the first tread (highest for a descent, lowest for a
        # climb) and shift the whole clip so the entry stands on THAT tread instead -- preserving the
        # source's foot->tread offset (which the per-loop check confirmed) but starting at the top/bottom,
        # so every loop lands on a real tread. Generic: needs only the traversal's net z-direction.
        if n_loops > 1 and boxes:
            k0, k1 = bounds[box_seg]
            descends = float(Wp[k0, 2] - Wp[k1 - 1, 2]) > 0.0
            # STEP treads only -- exclude the wide top/bottom PLATFORMS (size_x >> the step depth), or the
            # entry gets centered on a platform instead of standing on the first step.
            min_sx = min(float(b["size"][0]) for b in boxes)
            steps = [i for i, b in enumerate(boxes) if float(b["size"][0]) <= 1.5 * min_sx] or list(range(len(boxes)))
            tread_top = {i: float(boxes[i]["pos"][2]) + float(boxes[i]["size"][2]) / 2.0 for i in steps}
            first = (max if descends else min)(steps, key=lambda i: tread_top[i])
            exy = Wp[k0, :2].tolist()
            cur = min(steps, key=lambda i: (boxes[i]["pos"][0] - exy[0]) ** 2 + (boxes[i]["pos"][1] - exy[1]) ** 2)
            sh = torch.tensor([boxes[first]["pos"][j] - boxes[cur]["pos"][j] for j in range(3)],
                              device=Wp.device, dtype=Wp.dtype)
            Wp = Wp + sh                                 # entry now stands on the first STEP tread
            # Level the top/bottom PLATFORMS (wide boxes) flush with the END treads. The source puts them
            # one riser BEYOND the extreme steps (a real upper/lower floor), but the played motion starts
            # and ends ON the extreme treads -- so a floor one riser off leaves a phantom step at the
            # stair<->platform junction. Flushing them to the end treads makes the approach/recovery level
            # with where the descent actually begins/ends.
            ttops = sorted(float(boxes[i]["pos"][2]) + float(boxes[i]["size"][2]) / 2.0 for i in steps)
            hi_t, lo_t = ttops[-1], ttops[0]
            riser = (hi_t - lo_t) / max(1, len(ttops) - 1)
            for j, b in enumerate(boxes):
                if j not in steps:                       # a platform (wide box)
                    top = float(b["pos"][2]) + float(b["size"][2]) / 2.0
                    # The motion STARTS on the first (highest) step but ENDS one riser BELOW the last step,
                    # on the floor -- so the TOP platform is flush with the first step, the BOTTOM platform
                    # is the floor one riser under the last step (else it duplicates the last step).
                    tgt = hi_t if abs(top - hi_t) < abs(top - lo_t) else (lo_t - riser)
                    b["pos"] = [float(b["pos"][0]), float(b["pos"][1]), tgt - float(b["size"][2]) / 2.0]

        # Ground the FLAT approach on its walkable surface -- generic "DB height + ground height". A flat DB
        # clip is grounded to soles-at-0, so on an elevated surface it just gets that surface height ADDED:
        # 0 on flat ground (unchanged), the platform height on a stair TOP platform. Using the DB-grounded
        # approach root (``ra``, feet~0) + the surface under the terrain ENTRY -- rather than the z the seam
        # chain happened to leave it at -- is what lifts the lead-in walk up onto the platform instead of into it.
        if box_seg > 0 and boxes:
            a0, a1 = bounds[0]                                       # approach segment slice
            entry_gh = _ground_z_under(Wp[a1, :2], boxes)           # surface the terrain entry stands on
            Wp = Wp.clone()
            Wp[a0:a1, 2] = ra[:, 2].to(Wp) + entry_gh
        c_off = 0 if box_seg == 0 else 1                            # align_and_concat drops each non-anchor seam frame

    # Tile footprint = union of the clip PATH and the BOX footprint, with margin on ALL sides, so the
    # box sits fully INSIDE the tile (not overhanging the void gap) and inset from the edges instead of
    # jammed in the (0,0) corner. Re-center so the union-min sits ``footprint_margin`` inside the tile.
    m = args.footprint_margin
    if boxes:
        bxy = torch.tensor([[float(b_["pos"][0]), float(b_["pos"][1])] for b_ in boxes])
        bhf = torch.tensor([[float(b_["size"][0]) / 2.0, float(b_["size"][1]) / 2.0] for b_ in boxes])
        box_lo, box_hi = (bxy - bhf).amin(0), (bxy + bhf).amax(0)
    else:
        box_lo, box_hi = Wp[:, :2].amin(0), Wp[:, :2].amax(0)
    lo = torch.minimum(Wp[:, :2].amin(0), box_lo)
    hi = torch.maximum(Wp[:, :2].amax(0), box_hi)
    corner = lo - m                              # union-min -> margin inside the tile
    Wp = Wp.clone(); Wp[:, 0] -= corner[0]; Wp[:, 1] -= corner[1]

    y, dy, contact = builder.build(Wp, Wq, Wj, args.dt)
    # Splice the terrain segment's REAL exported foot contacts (looped with the motion) over the
    # kinematic ones. Segment 0 keeps its seam frame; later segments dropped theirs in align_and_concat.
    a, b = bounds[box_seg]
    # ``c_off`` set by the splice branch above: weld drops each non-anchor seam frame (1); eased keeps
    # the anchored jump's frames intact (0).
    if terr_contact is not None:
        contact[a:b] = terr_contact[c_off:c_off + (b - a)].to(contact.dtype)

    T = Wp.shape[0]
    # Per-segment anchor: ONLY the box interaction (the jump) is world-pinned (BOX: y-free,
    # x/z/yaw pinned to the box) so the arc lands on the fixed box; the lead-up AND recovery stay
    # FLAT (ego) so the robot handles them as ordinary flat locomotion. The FLAT->BOX seam is a
    # snap-in of the jump relative to the (perceived/fixed) box that the RL learns to absorb --
    # matching deploy, where the FM places the jump from perception, not from a world-pinned path.
    amt = build_anchor_mask_t(T, bounds, seg_masks, device=dev)

    overlays = [{"kind": "box", "pos_xy": [b_["pos"][0] - float(corner[0]), b_["pos"][1] - float(corner[1])],
                 "size_xyz": [float(v) for v in b_["size"]], "base_z": float(b_["pos"][2] - b_["size"][2] / 2.0),
                 "walkable_top": True, "yaw": float(b_["yaw"])} for b_ in boxes]
    # Support-structure variant (solid / floating / legs).  Only a single walkable
    # jump box gets a variant -- multi-box (stair-as-boxes) traversals stay solid so
    # the treads keep their support.  The walkable top height is variant-invariant,
    # so the baked motion / anchor mask are unaffected.
    if box_variant_params is not None and len(overlays) == 1 and overlays[0]["walkable_top"]:
        vp = resolve_box_variant(
            np.random.default_rng(box_variant_seed), overlays[0]["size_xyz"], **box_variant_params
        )
        overlays[0].update(vp)
    size_x = float(hi[0] - lo[0]) + 2 * m       # margin on BOTH sides (was one-sided)
    size_y = float(hi[1] - lo[1]) + 2 * m
    manifest_entry = {"terrain_id": tid, "type": "box", "size_x": size_x, "size_y": size_y,
                      "overlays": overlays}
    # TWO-VELOCITY SPLIT (the RL command != the DP generation command). The DP RAMPS the command
    # down (app_cmd) so the approach decelerates smoothly into the jump, but the RL-visible
    # ``target_vel`` is HELD FORWARD at the approach speed through the approach AND jump -- so
    # "velocity heading to 0" stays an unambiguous STOP signal and the policy learns "hold forward
    # + box => jump" (not "decelerate => jump"); the DP's ramp lives only in the achieved motion
    # (y/dy). The matching stop-before clips carry a target_vel that DOES ramp to 0. Only the on-box
    # recovery reads 0 (arrived; the box is now behind/below in the scan, so this 0 is distinguishable
    # from a stop BEFORE the box).
    tvel = torch.full((T, 3), 0.0, dtype=torch.float32, device=dev)
    jump_end = bounds[box_seg][1]                 # end of the played jump segment (approach + jump)
    tvel[:jump_end, 0] = float(approach_vx)       # held forward; recovery stays 0 (stand on box)
    return {"y": y, "dy": dy, "contact": contact, "anchor_mask_t": amt, "target_vel": tvel,
            "terrain_id": tid, "n": T, "manifest": manifest_entry, "name": ts.name,
            "entry_frame": k, "n_terr": int(ts.n)}


def _place_descent_on_tile(ts, tile, boxes, dev, k=0, truncate=True):
    """Loop the source descent from frame ``k`` and place it on the FIXED tile treads -- SHARED by the
    pure and transition paths so their descent geometry is identical.

    ``truncate`` (pure path): cut the descent once it reaches the lowest tread, so it ends there.  The
    transition path passes ``truncate=False`` and instead peels off via the contact+lowest-tread gate
    (which needs the settled frames ON the lowest tread, so it keeps the full looped descent).

    Loops enough source strides to descend the tile's N treads (``n_loops`` = tile #steps / the source
    stride's treads-per-period; NOT ``n_periodic_loops``, whose box-CENTER z-span under-reads our solid
    floor-to-top tread boxes ~2x).  The per-loop offset is the SOURCE stride's own-frame step
    (``yaw_normalize`` rotates the whole looped descent to +x after).  Then ANCHORS the source's
    frame-``k`` start tread (the tread it stands on, in the source's OWN geometry) onto the tile's
    highest tread by tread-TOP, preserving the source's foot->tread stance -- so the robot starts ON
    the first tread exactly as the trajectory source file has it, not floating mid-staircase.

    Returns ``(Wp, Wq, Wj, terr_contact)`` -- the placed descent segment in the tile frame.
    """
    src_boxes = ts.terrain_meta.get("terrain", {}).get("boxes", [])
    stride_dz = abs(float(ts.root_pos[k, 2] - ts.root_pos[-1, 2]))
    ptreads = max(1, round(stride_dz / float(tile.step_height)))   # treads the source covers per period
    n_loops = max(1, round(int(tile.stair_steps[0]) / ptreads))
    period_offset = periodic_tread_offset(ts.root_pos, "periodic", src_boxes)
    rp, rq, jt, terr_contact = loop_periodic(
        ts.root_pos.to(dev), ts.root_quat.to(dev), ts.joint_pos.to(dev),
        None if ts.contact is None else ts.contact.to(dev), n_loops, offset=period_offset, start=k)
    Wp, Wq, Wj, _, _ = align_and_concat([(rp, rq, jt)])
    Wp, Wq, _ = yaw_normalize(Wp, Wq, [], Wq[0])            # descent heading -> +x (R = -facing)
    src_min_sx = min(float(bb["size"][0]) for bb in src_boxes)
    src_steps = [b for b in src_boxes if float(b["size"][0]) <= 1.5 * src_min_sx]
    rk = ts.root_pos[k]
    src_start = min(src_steps, key=lambda b: (float(b["pos"][0]) - float(rk[0])) ** 2
                                             + (float(b["pos"][1]) - float(rk[1])) ** 2)
    sst = [float(src_start["pos"][0]), float(src_start["pos"][1]),
           float(src_start["pos"][2]) + float(src_start["size"][2]) / 2.0]      # source start-tread TOP
    qw, qx, qy, qz = (float(v) for v in ts.root_quat[k])
    fy = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))  # source facing yaw
    c, s = math.cos(-fy), math.sin(-fy)                                          # R = rot(-facing)
    rst = [c * sst[0] - s * sst[1], s * sst[0] + c * sst[1], sst[2]]             # R(src_start_top)
    # The tile's highest TREAD (top of stairs).  A transition tile's top platform is authored FLUSH with
    # the highest tread (same top z) but WIDE, so an unrestricted argmax(top-z) ties and can pick the wide
    # TOP PLATFORM -- whose center x is far from the tread column -- anchoring the whole descent onto the
    # platform instead of the stairs (the descent then floats over the platform, off the treads).  Restrict
    # to the NARROW tread boxes (size_x ~ the min) the way the rest of the pipeline tells treads apart.
    min_sx = min(float(b["size"][0]) for b in boxes)
    tread_ids = [i for i in range(len(boxes)) if float(boxes[i]["size"][0]) <= 1.5 * min_sx]
    first = max(tread_ids,
                key=lambda i: float(boxes[i]["pos"][2]) + float(boxes[i]["size"][2]) / 2.0)
    t0 = [float(boxes[first]["pos"][0]), float(boxes[first]["pos"][1]),
          float(boxes[first]["pos"][2]) + float(boxes[first]["size"][2]) / 2.0]  # tile highest-tread TOP
    sh = torch.tensor([t0[j] - rst[j] for j in range(3)], device=Wp.device, dtype=Wp.dtype)
    Wp = Wp + sh

    if truncate:
        # Pure path: end the descent once the root has descended (N-1) risers from the highest tread
        # (i.e. it reached the LOWEST tread), so a pure clip doesn't overshoot below the tile geometry.
        N = int(tile.stair_steps[0])
        z_lowest = float(Wp[0, 2]) - (N - 1) * float(tile.step_height)
        reached = (Wp[:, 2] <= z_lowest + 0.02 * float(tile.step_height)).nonzero(as_tuple=False)
        if reached.numel() > 0:
            cut = int(reached[0].item()) + 1
            Wp, Wq, Wj = Wp[:cut], Wq[:cut], Wj[:cut]
            if terr_contact is not None:
                terr_contact = terr_contact[:cut]
    return Wp, Wq, Wj, terr_contact


def _make_pure_tile_clip(builder, ts, tile, args, dev):
    """PURE stair TILE clip -> records dict (phase B, pure path).

    Loops the source descent stride down a FIXED, sampled staircase ``tile`` (its own dims / #steps /
    width) and emits the tile's fixed geometry + the tile's shared ``terrain_id``.  Matcher-free (no
    approach): the long looped-descent clip for a pure-stair terrain tile.  Descent placement is the
    SHARED :func:`_place_descent_on_tile` (same code the transition path uses).
    """
    boxes = stair_tile_boxes(step_height=tile.step_height, step_depth=tile.step_depth,
                             num_steps=int(tile.stair_steps[0]), tread_width=tile.width, ttype="terrain")
    Wp, Wq, Wj, terr_contact = _place_descent_on_tile(ts, tile, boxes, dev, k=0)
    bounds = [(0, Wp.shape[0])]

    # Recenter by the FIXED tile boxes (NOT the motion) so every clip on this tile emits identical
    # geometry -- the reuse invariant.  The tile footprint still spans the motion (it rides the treads).
    m = args.footprint_margin
    bxy = torch.tensor([[float(b["pos"][0]), float(b["pos"][1])] for b in boxes])
    bhf = torch.tensor([[float(b["size"][0]) / 2.0, float(b["size"][1]) / 2.0] for b in boxes])
    box_lo, box_hi = (bxy - bhf).amin(0), (bxy + bhf).amax(0)
    lo = torch.minimum(Wp[:, :2].amin(0), box_lo)
    hi = torch.maximum(Wp[:, :2].amax(0), box_hi)
    corner = box_lo - m                              # datum = fixed boxes only (clip-independent)
    Wp = Wp.clone(); Wp[:, 0] -= corner[0]; Wp[:, 1] -= corner[1]

    y, dy, contact = builder.build(Wp, Wq, Wj, args.dt)
    a, b = bounds[0]
    if terr_contact is not None:
        contact[a:b] = terr_contact[:b - a].to(contact.dtype)
    T = Wp.shape[0]
    amt = build_anchor_mask_t(T, bounds, [_ANCHOR_MASK_BOX], device=dev)
    overlays = [{"kind": "box",
                 "pos_xy": [b_["pos"][0] - float(corner[0]), b_["pos"][1] - float(corner[1])],
                 "size_xyz": [float(v) for v in b_["size"]],
                 "base_z": float(b_["pos"][2] - b_["size"][2] / 2.0),
                 "walkable_top": True, "yaw": float(b_["yaw"])} for b_ in boxes]
    size_x = float(hi[0] - lo[0]) + 2 * m
    size_y = float(hi[1] - lo[1]) + 2 * m
    manifest_entry = {"terrain_id": int(tile.terrain_id), "type": "terrain",
                      "size_x": size_x, "size_y": size_y, "overlays": overlays}
    tvel = torch.zeros(T, 3, dtype=torch.float32, device=dev)      # pure descent: no forward-cmd label
    return {"y": y, "dy": dy, "contact": contact, "anchor_mask_t": amt, "target_vel": tvel,
            "terrain_id": int(tile.terrain_id), "n": T, "manifest": manifest_entry, "name": ts.name,
            "entry_frame": 0, "n_terr": int(ts.n)}


def _splice_around(descent, approach, recovery, top_surf, bot_surf, blend_n):
    """Splice a played DESCENT (the anchored segment, kept pristine on its treads) with a DP approach
    BEFORE and a DP recovery AFTER -- treated like a normal clip transition, NOT a rigid z-weld.

    Each flat segment is yaw+xy aligned to the descent boundary and grounded to its OWN surface
    (approach -> top platform ``top_surf``, recovery -> bottom platform ``bot_surf``; the DP flats are
    grounded to feet~0, so the surface is just added to z -- the same role z=0 plays for a flat clip).
    Their z is NOT welded to the descent's (crouched) boundary height, so their feet stay on their own
    surface.  The seam discontinuity is then eased with :func:`playback.inertialize_seam` -- the exact
    inertialization the matcher uses at a cut -- so the transition eases in like any flat clip cut.

    Returns ``(Wp, Wq, Wj, bounds, box_seg)`` with the descent at ``box_seg == 1``.
    """
    dP, dQ, dJ = descent

    def _place(seg, tgt_p, tgt_q, ground, align_last):
        """Yaw+xy align ``seg``'s boundary frame to ``(tgt_p, tgt_q)`` (drop z), then ground z by adding
        ``ground`` (so the flat clip's feet sit on its surface)."""
        P, Q, J = seg
        bidx = -1 if align_last else 0
        gq, t = se2z_align(P[bidx], Q[bidx], tgt_p, tgt_q)     # gq = pure-yaw; keep only xy of t
        n = P.shape[0]
        txy = torch.tensor([float(t[0]), float(t[1]), 0.0], device=P.device, dtype=P.dtype)
        P = quat_apply(gq.unsqueeze(0).expand(n, 4), P) + txy
        Q = quat_mul(gq.unsqueeze(0).expand(n, 4), Q)
        P = P.clone(); P[:, 2] = P[:, 2] + float(ground)       # feet(~0) -> surface
        return P, Q, J

    aP, aQ, aJ = _place(approach, dP[0], dQ[0], top_surf, align_last=True)
    rP, rQ, rJ = _place(recovery, dP[-1], dQ[-1], bot_surf, align_last=False)
    Wp = torch.cat([aP, dP, rP]); Wq = torch.cat([aQ, dQ, rQ]); Wj = torch.cat([aJ, dJ, rJ])
    na, nd = aP.shape[0], dP.shape[0]
    bounds = [(0, na), (na, na + nd), (na + nd, na + nd + rP.shape[0])]
    box_seg = 1
    if blend_n >= 2:
        # Ease each seam like a matcher cut: the played descent is the ANCHORED side (kept pristine on
        # the treads); the flat approach/recovery ease into it.
        Wp, Wq, Wj = inertialize_seam(Wp, Wq, Wj, na, blend_n, anchored_after=True)        # approach->descent
        Wp, Wq, Wj = inertialize_seam(Wp, Wq, Wj, na + nd, blend_n, anchored_after=False)   # descent->recovery
    return Wp, Wq, Wj, bounds, box_seg


def _make_transition_tile_clip(matcher, builder, ts, tile, approach_vx, args, dev, top_len, bot_len,
                               approach_s, recovery_s):
    """TRANSITION stair TILE clip -> records dict (phase B, full path).

    approach (DP, ``approach_s`` long, ramps DOWN to the descent entry speed) on the top platform ->
    descent (played, SHARED :func:`_place_descent_on_tile`) -> recovery (DP, ``recovery_s`` long, ramps
    UP from the descent exit speed back to ``approach_vx``) on the bottom platform, on a FIXED transition
    tile (flat -> single staircase -> flat).  Many approach speeds share the tile's geometry +
    ``terrain_id`` (the reuse invariant); ``approach_s`` / ``recovery_s`` are sampled per clip (the
    platform is sized to the longest possible).  The RL-visible ``target_vel`` is held at ``approach_vx``
    the whole clip, so the recovery ramp-up makes the exit MOTION match that command instead of walking
    away at the slow descent speed.  Grounding is inherited from the tile-anchored descent via the seam
    splice (approach on the top platform ~N*h, recovery on the bottom platform ~h).
    """
    boxes = stair_tile_boxes(step_height=tile.step_height, step_depth=tile.step_depth,
                             num_steps=int(tile.stair_steps[0]), tread_width=tile.width,
                             ttype="transition", top_platform_len=top_len, bottom_platform_len=bot_len,
                             nav_width=tile.nav_width)
    T_app = max(1, int(round(approach_s / args.dt)))
    T_exit = max(1, int(round(recovery_s / args.dt)))
    # Hand off near the TOP (descent_entry_frac cap) so the FULL descent plays.
    efrac = min(args.entry_frac, args.descent_entry_frac)
    entry_descs = matcher.entry_descriptors(ts, entry_frac=efrac)
    entry_vx = _entry_vx(ts, args.dt)
    app_cmd = _ramp_cmd(approach_vx, approach_s, args.approach_ramp_s, vx1=entry_vx)
    ra, qa, ja = matcher.generate_approach(app_cmd, T_app, entry_descs,
                                           boundary_weight=args.boundary_weight, relax_tail=0,
                                           allowed_categories=_APPROACH_FILTERS["stair_transition"])
    k = int(matcher.matched_entry)
    # Full looped descent (NOT truncated): the exit peel-off needs the settled frames ON the lowest tread.
    Wp_d, Wq_d, Wj_d, terr_contact = _place_descent_on_tile(ts, tile, boxes, dev, k=k, truncate=False)

    # EXIT peel-off gate: the recovery may hand off ONLY from a SINGLE-support frame near the bottom of the
    # descent -- the stance foot in CONTACT with the LAST (lowest) tread while the SWING foot is still clear
    # of it.  That is a mid-stride, pelvis-UP pose; the terminal double-support settle (both feet on the
    # bottom platform) is deliberately excluded because there the pelvis bottoms out and welds low, which is
    # exactly what the descent->recovery seam-ease then has to lift the robot out of (the exit "pop").  FK
    # the placed descent for foot heights; the DP still CHOOSES among the qualifying frames.
    _SOLE = 0.034
    h = float(tile.step_height)                      # lowest tread top (tile-local); tile treads are h..N*h
    tol = 0.02                                        # tight: stance foot must be ON the tread, not past it
    swing_clear = min(0.05, 0.5 * h)                 # swing foot must clear the lowest tread by this (single support)
    yd, _dyd, _cd = builder.build(Wp_d, Wq_d, Wj_d, args.dt)
    lz = builder.pos_names.index("left_ankle_roll_link:pos_z")
    rz = builder.pos_names.index("right_ankle_roll_link:pos_z")
    lsole = yd[:, lz].numpy() - _SOLE
    rsole = yd[:, rz].numpy() - _SOLE
    tc = (terr_contact.cpu().numpy() if terr_contact is not None
          else np.zeros((Wp_d.shape[0], 2), dtype=np.uint8))
    l_on = (tc[:, 0] > 0) & (np.abs(lsole - h) < tol)   # left stance foot planted on the lowest tread
    r_on = (tc[:, 1] > 0) & (np.abs(rsole - h) < tol)   # right stance foot planted on the lowest tread
    on_contact = l_on | r_on                            # some foot down on the lowest tread (the old gate)
    on_single = ((l_on & (rsole > h + swing_clear))     # + the OTHER (swing) foot still clear of the tread
                 | (r_on & (lsole > h + swing_clear)))
    allowed = np.nonzero(on_single)[0]
    if allowed.size == 0:                               # no clean single-support frame -> contact-only fallback
        allowed = np.nonzero(on_contact)[0]
        if allowed.size:
            print(f"[box][WARN] transition tile {tile.terrain_id}: no single-support exit frame "
                  f"(swing_clear={swing_clear:.3f} m); falling back to contact-only peel-off "
                  f"(exit pose may be low).")
    if allowed.size == 0:
        raise ValueError(
            f"transition tile {tile.terrain_id}: the descent never lands a foot on the lowest tread "
            f"(top z={h:.3f}, tol={tol}); cannot pick an exit frame (loop-count / coverage bug -- this "
            f"should not happen).")
    # Descent frame -> source-period frame (loop_periodic offset layout): the first play is source[k:],
    # then wrapped periods are source[1:].  Descriptors are placement/yaw-invariant, so a descent frame's
    # descriptor == its source-period frame's.
    nper = int(ts.root_pos.shape[0])

    def _src_frame(f):
        return (k + f) if f < (nper - k) else (1 + ((f - (nper - k)) % (nper - 1)))

    src_desc = matcher._source_descriptors(ts)       # [nper, dim] normalized descriptors
    exit_descs = src_desc[[_src_frame(int(f)) for f in allowed]].detach().cpu().numpy()
    # Recovery ACCELERATES from the descent exit speed back up to the commanded approach speed (mirror of
    # the approach ramp-down), so the exit motion matches the held-forward RL target_vel.
    rec_cmd = _ramp_up_cmd(entry_vx, args.recovery_ramp_s, approach_vx)
    re, qe, je = matcher.generate_exit(rec_cmd, T_exit, exit_descs,
                                       allowed_categories=_APPROACH_FILTERS["stair_transition"])
    cut = int(allowed[int(matcher.matched_exit)]) + 1   # peel off at the DP-chosen qualifying frame
    cf = cut - 1                                         # the chosen exit frame (index into the full descent)
    # Diagnostic: report the chosen peel-off frame's support state + pelvis height so we can confirm it is a
    # single-support, pelvis-up pose (not a low double-support settle) -- i.e. the seam z-residual is small.
    print(f"[box]   -> exit peel-off frame {cf}/{Wp_d.shape[0] - 1} of {len(allowed)} candidates: "
          f"contact=(L{int(tc[cf, 0])},R{int(tc[cf, 1])}) "
          f"sole-vs-tread L={lsole[cf] - h:+.3f} R={rsole[cf] - h:+.3f} m, "
          f"pelvis-above-tread={float(Wp_d[cf, 2]) - h:.3f} m")
    Wp_d, Wq_d, Wj_d = Wp_d[:cut], Wq_d[:cut], Wj_d[:cut]
    if terr_contact is not None:
        terr_contact = terr_contact[:cut]

    # Splice: approach + recovery grounded to their platforms (top = N*h, bottom = h) and eased into the
    # descent like a normal clip cut -- NOT z-welded (which dropped the flat feet into the platform).
    top_surf = int(tile.stair_steps[0]) * float(tile.step_height)   # top platform (= highest tread)
    bot_surf = float(tile.step_height)                               # bottom platform (= lowest tread)
    blend_n = int(round(args.seam_blend_s / args.dt))
    Wp, Wq, Wj, bounds, box_seg = _splice_around(
        (Wp_d, Wq_d, Wj_d),
        (ra.to(dev), qa.to(dev), ja.to(dev)),
        (re.to(dev), qe.to(dev), je.to(dev)),
        top_surf, bot_surf, blend_n)
    seg_masks = [_ANCHOR_MASK_FLAT, _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT]

    # Recenter by the FIXED tile boxes so every clip on this tile emits identical geometry.
    m = args.footprint_margin
    bxy = torch.tensor([[float(b["pos"][0]), float(b["pos"][1])] for b in boxes])
    bhf = torch.tensor([[float(b["size"][0]) / 2.0, float(b["size"][1]) / 2.0] for b in boxes])
    box_lo, box_hi = (bxy - bhf).amin(0), (bxy + bhf).amax(0)
    lo = torch.minimum(Wp[:, :2].amin(0), box_lo)
    hi = torch.maximum(Wp[:, :2].amax(0), box_hi)
    corner = box_lo - m
    Wp = Wp.clone(); Wp[:, 0] -= corner[0]; Wp[:, 1] -= corner[1]

    y, dy, contact = builder.build(Wp, Wq, Wj, args.dt)
    a, b = bounds[box_seg]
    if terr_contact is not None:
        # The descent is the splice ANCHOR -> keeps ALL its frames (only the approach's last / recovery's
        # first are dropped), so its contacts map 1:1 onto [a:b] with no offset.
        contact[a:b] = terr_contact[:b - a].to(contact.dtype)

    T = Wp.shape[0]
    amt = build_anchor_mask_t(T, bounds, seg_masks, device=dev)
    overlays = [{"kind": "box",
                 "pos_xy": [b_["pos"][0] - float(corner[0]), b_["pos"][1] - float(corner[1])],
                 "size_xyz": [float(v) for v in b_["size"]],
                 "base_z": float(b_["pos"][2] - b_["size"][2] / 2.0),
                 "walkable_top": True, "yaw": float(b_["yaw"])} for b_ in boxes]
    size_x = float(hi[0] - lo[0]) + 2 * m
    size_y = float(hi[1] - lo[1]) + 2 * m
    manifest_entry = {"terrain_id": int(tile.terrain_id), "type": "transition",
                      "size_x": size_x, "size_y": size_y, "overlays": overlays}
    tvel = torch.full((T, 3), 0.0, dtype=torch.float32, device=dev)
    tvel[:, 0] = float(approach_vx)                               # hold forward down the stairs + off
    return {"y": y, "dy": dy, "contact": contact, "anchor_mask_t": amt, "target_vel": tvel,
            "terrain_id": int(tile.terrain_id), "n": T, "manifest": manifest_entry, "name": ts.name,
            "entry_frame": k, "n_terr": int(ts.n)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt library (output channel layout).")
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--terrain-lib", default="trajectories/skills/terrain_v1")
    p.add_argument("--terrain-hf-repo", default="zolkin/robot_rl")
    p.add_argument("--terrain-skills", default=None, help="';'-sep subset of terrain skills (default all).")
    p.add_argument("--out", required=True)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--approach-s", type=float, default=2.5, help="Approach (DP) duration (s) [box path].")
    p.add_argument("--recovery-s", type=float, default=1.0, help="Exit/recovery (DP) duration (s) [box path].")
    p.add_argument("--approach-s-range", type=float, nargs=2, default=(1.0, 4.0), metavar=("MIN", "MAX"),
                   help="[tiles] Sample the transition-tile approach (lead-in) duration (s) uniformly from "
                        "this range, PER CLIP.  Platforms are sized to MAX * the fastest speed.")
    p.add_argument("--recovery-s-range", type=float, nargs=2, default=(1.0, 4.0), metavar=("MIN", "MAX"),
                   help="[tiles] Sample the transition-tile recovery (exit) duration (s) uniformly per clip.")
    p.add_argument("--recovery-ramp-s", type=float, default=0.8,
                   help="[tiles] Seconds over which the recovery accelerates from the descent exit speed "
                        "back up to the commanded (approach) speed at the bottom of the stairs.")
    p.add_argument("--approach-vels", default="0.0",
                   help="';'-separated approach forward speeds (m/s); one CLIP per (terrain x speed). "
                        "e.g. '0.8;1.6;2.4;3.2' for walk->run run-ups into the jump. In --mode pure "
                        "the motion is identical across speeds (no approach); the speed only labels "
                        "the held-forward target_vel (jump intent), so a few values give speed-invariance.")
    p.add_argument("--mode", choices=["full", "pure"], default="full",
                   help="'full' (default): DP(approach) + PLAY(jump) + recovery -- the traversal. "
                        "'pure': just PLAY the bare interaction from frame 0 (spawn-at-box skill "
                        "anchor, world-pinned BOX throughout, no lead-in / no walk-on-box).")
    p.add_argument("--approach-ramp-s", type=float, default=0.8,
                   help="Seconds over which the run-up decelerates to the jump entry speed "
                        "(--transition=ramp) or the tracking-relax window (--transition=relax).")
    p.add_argument("--transition", choices=["relax", "ramp"], default="ramp",
                   help="How the approach reaches the terrain entry. 'ramp' (default): prescribe a "
                        "velocity decel the DP tracks -> a real stepping gait (less foot slide). "
                        "'relax': hold the command + relax tracking near the terminal so the DP "
                        "searches the transition (smoother root vel, but more foot slide at speed).")
    p.add_argument("--box-splice", choices=["weld", "eased"], default="eased",
                   help="How the box approach/recovery join the played jump. 'eased' (default): the SAME "
                        "jump-anchored inertialize_seam splice the stair tiles use, with per-clip SAMPLED "
                        "durations from --approach-s-range/--recovery-s-range (episodic jumps only). 'weld': "
                        "the legacy rigid SE(2)+z C0 weld (align_and_concat) with FIXED --approach-s/--recovery-s.")
    p.add_argument("--clip-length", type=float, default=None,
                   help="Fixed clip length (s); default = the longest traversal.")
    p.add_argument("--boundary-weight", type=float, default=200.0,
                   help="Terminal-pin weight for the approach DP (larger = harder entry pin).")
    p.add_argument("--entry-frac", type=float, default=0.5,
                   help="The approach may hand off into the terrain clip's first this-fraction of "
                        "frames (0..1), picking whichever best matches its incoming speed. 0.5 = "
                        "first half (keeps it from skipping the interaction by entering near the end).")
    p.add_argument("--exit-frac", type=float, default=0.5,
                   help="Mirror of --entry-frac for the peel-OFF: the recovery may seam in at any of the "
                        "terrain clip's LAST this-fraction of frames (0..1), picking whichever best "
                        "matches the resumed locomotion; the played interaction is truncated there so no "
                        "single frame is pinned on the exit.")
    p.add_argument("--descent-entry-frac", type=float, default=0.1,
                   help="entry-frac cap for a net-DESCENT interaction (e.g. stairs_down): hand off near "
                        "the TOP so the full descent plays and the approach is on the top platform "
                        "(with the wide --entry-frac the DP enters mid-descent, skipping the top).")
    p.add_argument("--descent-z", type=float, default=0.1,
                   help="Root-z drop (start - end, m) above which an interaction counts as a DESCENT.")
    p.add_argument("--footprint-margin", type=float, default=2.0, help="Tile footprint margin (m).")
    p.add_argument("--seed", type=int, default=0, help="Base seed for box-variant resolution (per-terrain rng = seed + terrain_index).")
    # -- box support-structure variant knobs (single-jump-box terrains only) --
    p.add_argument("--box-variant-fracs", default="solid=1",
                   help="Comma-separated variant weights, e.g. 'solid=0.25,floating=0.25,center_leg=0.25,corner_legs=0.25'. "
                        "Names: solid, floating, center_leg (=floating_center_leg), corner_legs (=floating_corner_legs). "
                        "Default 'solid=1' reproduces the plain solid box. Weights are normalized.")
    p.add_argument("--slab-thickness", type=float, default=0.1, help="[box] top-slab thickness for floating variants (m).")
    p.add_argument("--leg-size", type=float, nargs=2, default=(0.1, 0.1), metavar=("LX", "LY"), help="[box] support-leg cross-section (m).")
    p.add_argument("--leg-inset", type=float, default=0.05, help="[box] corner-leg inset from the footprint edge (m).")
    p.add_argument("--cut-penalty", type=float, default=3.0)
    p.add_argument("--ground-penalty", type=float, default=12.0,
                   help="Weight penalizing candidate frames whose pelvis is above --max-pelvis-z over "
                        "the flat approach ground (keeps on-box/on-stairs lead-in frames out of the "
                        "run-up). 0 = off.")
    p.add_argument("--max-pelvis-z", type=float, default=0.95,
                   help="Pelvis-above-ground ceiling (m) before the ground penalty kicks in.")
    p.add_argument("--include-clean", action="store_true", help="Fold in the clean STEADY strides.")
    p.add_argument("--clean-source", default="trajectories/skills/library_v1")
    p.add_argument("--clean-hf-repo", default="zolkin/robot_rl")
    p.add_argument("--bones-limit", type=int, default=None)
    p.add_argument("--leadin-limit", type=int, default=None)
    p.add_argument("--graph-k-max", type=int, default=50)
    p.add_argument("--graph-k-min", type=int, default=25)
    p.add_argument("--graph-k-target", type=int, default=40)
    # --- tiles-then-clips (phase B) knobs ---
    p.add_argument("--n-pure-tiles", type=int, default=0,
                   help="[tiles] If >0, generate N PURE stair tiles (sampled #steps/width, long looped "
                        "descent) instead of the source-driven loop. Matcher-free.")
    p.add_argument("--pure-seconds", type=float, default=20.0,
                   help="[tiles] Target duration (s) for a pure looped-descent clip; sets the tile length "
                        "(#steps ~ pure_seconds * stair_vx / tread_depth).")
    p.add_argument("--stair-vx", type=float, default=None,
                   help="[tiles] Descent forward speed (m/s) for the pure-length calc (default: measured "
                        "from the source stride).")
    p.add_argument("--tile-width-min", type=float, default=0.8, help="[tiles] min tread width (m).")
    p.add_argument("--tile-width-max", type=float, default=1.2, help="[tiles] max tread width (m).")
    p.add_argument("--tile-seed", type=int, default=0, help="[tiles] RNG seed for tile sampling.")
    p.add_argument("--n-transition-tiles", type=int, default=0,
                   help="[tiles] If >0, generate N TRANSITION tiles (flat lead-in -> single staircase -> "
                        "flat), each REUSED by all --transition-vels approach speeds (shared terrain_id).")
    p.add_argument("--transition-vels", default="0.8;1.5",
                   help="[tiles] ';'-sep approach speeds per transition tile (all share the tile).")
    p.add_argument("--tile-num-steps-min", type=int, default=3, help="[tiles] min stairs per transition tile.")
    p.add_argument("--tile-num-steps-max", type=int, default=7, help="[tiles] max stairs per transition tile.")
    p.add_argument("--seam-blend-s", type=float, default=0.12,
                   help="[tiles] inertialization blend time (s) at the approach/descent/recovery splice "
                        "seams -- decays the joint discontinuity so transitions aren't snappy (0 = off).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = args.device

    def _csv(s):
        return None if not s else [x.strip() for x in s.split(";") if x.strip()]

    # The matcher (candidate graph) is only needed where an approach/exit DP runs: the transition-tile
    # path and the legacy source-driven full path.  Pure tiles / --mode pure play the bare interaction
    # -> skip the expensive build.
    need_matcher = args.n_transition_tiles > 0 or (
        args.n_pure_tiles == 0 and args.mode == "full")
    matcher = _build_matcher(args, dev) if need_matcher else None
    if matcher is None:
        print("[box] no approach DP needed: skipping candidate-graph build")
    manager = MultiSkillManager(args.source, dev, env=None)
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT, device=dev)

    print(f"[box] ingesting terrain library {args.terrain_lib} ...")
    terr_srcs = ingest_terrain_library(args.terrain_lib, hf_repo=args.terrain_hf_repo, dt=args.dt,
                                       device=dev, skills=_csv(args.terrain_skills))
    print(f"[box] {len(terr_srcs)} terrain interaction clips")

    approach_vels = [float(v) for v in _csv(args.approach_vels)]
    print(f"[box] approach speeds: {approach_vels}  -> {len(terr_srcs) * len(approach_vels)} clips")

    # Parse the box-variant fraction spec once (e.g. "solid=0.5,floating=0.5").
    box_variant_fracs = {}
    for tok in (args.box_variant_fracs or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, weight = tok.partition("=")
        box_variant_fracs[name.strip()] = float(weight)
    box_variant_params = {
        "fracs": box_variant_fracs,
        "slab_thickness": args.slab_thickness,
        "leg_size": tuple(args.leg_size),
        "leg_inset": args.leg_inset,
    }
    print(f"[box] variant fracs: {box_variant_fracs}")

    records = []
    if args.n_pure_tiles > 0 or args.n_transition_tiles > 0:
        # Phase B (tiles-then-clips): sample the tile population + build clips on the FIXED per-tile
        # geometry (each tile's own #steps/width) sharing that tile's terrain_id.  PURE tile = one long
        # looped-descent clip (matcher-free); TRANSITION tile = approach+descent+recovery per
        # --transition-vels, ALL sharing the tile's geometry + id (the reuse invariant).
        src = terr_srcs[0]                                        # the descent stride source
        sb = src.terrain_meta.get("terrain", {}).get("boxes", [])
        msx = min(b["size"][0] for b in sb)
        steps = sorted((b for b in sb if b["size"][0] <= 1.5 * msx), key=lambda b: -b["pos"][2])
        sp = np.array([b["pos"] for b in steps], dtype=float)
        sv = np.mean(np.diff(sp, axis=0), axis=0)                # mean tread->tread step (source frame)
        h, d = float(abs(sv[2])), float(np.hypot(sv[0], sv[1]))  # source riser / tread depth
        dur = max((src.root_pos.shape[0] - 1) * args.dt, 1e-6)
        stair_vx = (float(args.stair_vx) if args.stair_vx
                    else float(torch.norm(src.root_pos[-1, :2] - src.root_pos[0, :2]) / dur))
        pure_len = args.pure_seconds * stair_vx
        tvels = [float(v) for v in _csv(args.transition_vels)] or [0.8]
        print(f"[box] tiles: source h={h:.3f} d={d:.3f} stair_vx={stair_vx:.2f} m/s | "
              f"{args.n_pure_tiles} pure (~{pure_len:.0f}m) + {args.n_transition_tiles} transition x {len(tvels)} vels")
        tiles = sample_stair_tiles(
            np.random.default_rng(args.tile_seed), hd_grid=[(h, d)],
            n_pure=args.n_pure_tiles, n_transition=args.n_transition_tiles,
            pure_length=pure_len, transition_length=max(12.0, pure_len),
            width_range=(args.tile_width_min, args.tile_width_max),
            num_steps_range=(args.tile_num_steps_min, args.tile_num_steps_max),
            lead_in_range=(1.5, 4.0), end_platform=1.0, inter_flat_range=(3.0, 5.0),
            nav_width_range=(args.tile_width_min, args.tile_width_max), flat_width_margin=0.3)
        # Dedicated rng for the per-clip approach/recovery DURATION sampling (distinct seed so it does
        # not perturb the tile-geometry rng above -> both stay reproducible from --tile-seed).
        time_rng = np.random.default_rng(args.tile_seed + 1)
        for tile in tiles:
            if tile.ttype == "terrain":
                print(f"[box] pure tile {tile.terrain_id}: {tile.stair_steps[0]} steps, width {tile.width:.2f}")
                records.append(_make_pure_tile_clip(builder, src, tile, args, dev))
            else:
                mx = max(tvels)
                # Platforms sized to the LONGEST POSSIBLE sampled lead-in / recovery (range max * fastest vx).
                top_len = args.approach_s_range[1] * mx + 1.0
                bot_len = args.recovery_s_range[1] * mx + 1.0
                print(f"[box] transition tile {tile.terrain_id}: {tile.stair_steps[0]} steps, "
                      f"width {tile.width:.2f}, platforms top={top_len:.1f} bot={bot_len:.1f} m, vels {tvels}")
                for vx in tvels:
                    a_s = float(time_rng.uniform(*args.approach_s_range))   # per-clip lead-in / recovery s
                    r_s = float(time_rng.uniform(*args.recovery_s_range))
                    rec = _make_transition_tile_clip(matcher, builder, src, tile, vx, args, dev,
                                                     top_len, bot_len, a_s, r_s)
                    rec["approach_vx"] = vx
                    print(f"[box]   -> tile {tile.terrain_id} vx={vx}: approach_s={a_s:.2f} recovery_s={r_s:.2f}, "
                          f"entered at frame {rec['entry_frame']}/{rec['n_terr']}")
                    records.append(rec)
    else:
        tid = 0
        # Per-clip approach/recovery DURATION sampling for --box-splice eased (distinct seed so it does
        # not perturb the box-variant rng); weld keeps the fixed --approach-s/--recovery-s.
        box_time_rng = np.random.default_rng(args.seed + 7919)
        for ti, ts in enumerate(terr_srcs):
            for vx in approach_vels:
                if args.box_splice == "eased":
                    a_s = float(box_time_rng.uniform(*args.approach_s_range))
                    r_s = float(box_time_rng.uniform(*args.recovery_s_range))
                else:
                    a_s, r_s = args.approach_s, args.recovery_s
                print(f"[box] splicing {ts.name} @ approach vx={vx} (terrain_id={tid}) "
                      f"[{args.box_splice}: approach_s={a_s:.2f} recovery_s={r_s:.2f}] ...")
                rec = _make_box_clip(matcher, builder, ts, tid, vx, args, dev,
                                     box_variant_seed=args.seed + ti,
                                     box_variant_params=box_variant_params,
                                     approach_s=a_s, recovery_s=r_s)
                rec["approach_vx"] = vx
                print(f"[box]   -> entered terrain at frame {rec['entry_frame']}/{rec['n_terr']}")
                for w in getattr(matcher, "warnings", []) or []:
                    print(f"[box][WARN] {w}")
                records.append(rec)
                tid += 1

    T = (int(round(args.clip_length / args.dt)) if args.clip_length
         else max(r["n"] for r in records))
    print(f"[box] padding {len(records)} clips to T={T} steps")

    meta = {"dt": args.dt, "clip_length_s": T * args.dt, "num_steps": int(T),
            "pos_output_names": list(manager.pos_output_names),
            "vel_output_names": list(manager.vel_output_names),
            "contact_frames": _CONTACT, "ref_frame": "pelvis_link",
            "skill_list": [s.name for s in manager.skills]}
    writer = ClipLibraryWriter(args.out, len(records), meta)
    writer.set_attr("generator", "box_clip_splice")
    writer.set_attr("terrain_manifest", [r["manifest"] for r in records])
    for i, r in enumerate(records):
        batch = {
            "y_des": _pad_or_trim(r["y"], T).unsqueeze(0),
            "dy_des": _pad_or_trim(r["dy"], T).unsqueeze(0),
            "contact": _pad_or_trim(r["contact"], T).unsqueeze(0),
            "skill_id": torch.zeros(1, T, dtype=torch.uint8, device=dev),
            "stride_idx": torch.full((1, T), -1, dtype=torch.int32, device=dev),
            "target_vel": _pad_or_trim(r["target_vel"], T).unsqueeze(0),
            "terrain_id": torch.tensor([r["terrain_id"]], dtype=torch.int32, device=dev),
            "terrain_type": "box",
            "anchor_mask": torch.tensor([_ANCHOR_MASK_BOX], dtype=torch.uint8, device=dev),
            "anchor_mask_t": _pad_or_trim(r["anchor_mask_t"], T).unsqueeze(0),
        }
        writer.write_batch(i, batch)
    writer.close()

    lib = load_clip_library(args.out)
    yy = torch.as_tensor(lib.y_des)
    print(f"[box] wrote {args.out}: {lib.num_clips} clips x {lib.num_steps} steps, "
          f"y_des {tuple(yy.shape)} finite={bool(torch.isfinite(yy).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
