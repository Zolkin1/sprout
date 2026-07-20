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

import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.motion_matching.bones_ingest import load_cache
from motion_gen.motion_matching.box_clip import (
    align_and_concat,
    build_anchor_mask_t,
    transform_boxes,
    yaw_normalize,
)
from motion_gen.motion_matching.clean_ingest import ingest_clean_library
from motion_gen.motion_matching.clip_output import ClipOutputBuilder
from motion_gen.motion_matching.dp import DPConfig, DPMatcher
from motion_gen.motion_matching.graph import GraphConfig, build_candidate_graph
from motion_gen.motion_matching.terrain_ingest import ingest_terrain_library
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
    vcfg = VanillaConfig(dt=args.dt)
    dcfg = DPConfig(dt=args.dt, cut_penalty=args.cut_penalty, w_traj=1.0, w_track=0.0,
                    w_ground=args.ground_penalty, ground_z=0.0, max_pelvis_z=args.max_pelvis_z)
    nframes = sum(s.n for s in sources)
    print(f"[box] building DP matcher DB ({len(sources)} clips, {nframes} frames) ...")
    matcher = DPMatcher(sources, vcfg, dcfg, device=dev)
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


def _make_box_clip(matcher, builder, ts, tid, approach_vx, args, dev):
    """One box clip -> (records dict) for padding.

    ``--mode full`` (default): DP(approach @ approach_vx) + PLAY(ts) + exit -- the traversal.
    ``--mode pure``: just PLAY(ts) from frame 0 -- the BARE interaction (spawn-at-box, world-pinned
    BOX throughout), a small-fraction skill anchor that drills the core jump undiluted by a lead-in.
    """
    if args.mode == "pure":
        k = 0
        terr = (ts.root_pos.to(dev), ts.root_quat.to(dev), ts.joint_pos.to(dev))
        segments, box_seg, seg_masks = [terr], 0, [_ANCHOR_MASK_BOX]
    else:
        T_app = max(1, int(round(args.approach_s / args.dt)))
        T_exit = max(1, int(round(args.recovery_s / args.dt)))
        tail = max(1, int(round(args.approach_ramp_s / args.dt)))
        # Hand off into ANY of the terrain clip's first ``entry_frac`` frames (a faster run-up can
        # merge into a later moving frame; a standing jump only has frame 0, so it lands there).
        entry_descs = matcher.entry_descriptors(ts, entry_frac=args.entry_frac)
        _, exit_desc = matcher.boundary_descriptors(ts)
        entry_vx = _entry_vx(ts, args.dt)   # the played interaction's start speed (ramp target)
        # --transition ramp (default): prescribe a velocity decel to entry_vx and TRACK it (real
        # stepping gait into the entry, least foot slide); relax: hold the command + relax tracking.
        if args.transition == "ramp":
            app_cmd = _ramp_cmd(approach_vx, args.approach_s, args.approach_ramp_s, vx1=entry_vx)
            ra, qa, ja = matcher.generate_approach(app_cmd, T_app, entry_descs,
                                                   boundary_weight=args.boundary_weight, relax_tail=0)
        else:
            app_cmd = _steady_cmd(approach_vx)
            ra, qa, ja = matcher.generate_approach(app_cmd, T_app, entry_descs,
                                                   boundary_weight=args.boundary_weight, relax_tail=tail)
        k = int(matcher.matched_entry)      # play the interaction FROM the matched entry frame
        terr = (ts.root_pos[k:].to(dev), ts.root_quat[k:].to(dev), ts.joint_pos[k:].to(dev))
        # Interaction-aware exit: ELEVATED end (jumped ONTO a box, no dismount) -> HOLD the
        # stand-on-box pose; else run a flat recovery DP seeded from the exit state.
        if float(ts.root_pos[-1, 2] - ts.root_pos[k, 2]) > 0.15:
            re = ts.root_pos[-1:].to(dev).repeat(T_exit, 1)
            qe = ts.root_quat[-1:].to(dev).repeat(T_exit, 1)
            je = ts.joint_pos[-1:].to(dev).repeat(T_exit, 1)
        else:
            re, qe, je = matcher.generate_exit(_steady_cmd(entry_vx), T_exit, exit_desc)
        segments, box_seg, seg_masks = ([(ra, qa, ja), terr, (re, qe, je)], 1,
                                        [_ANCHOR_MASK_FLAT, _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT])

    Wp, Wq, Wj, bounds, tfs = align_and_concat(segments)

    # The exported box geometry (in the terrain clip's own frame) glued to the played segment via
    # its splice transform -- this is where the CLIP says the box goes, not a heuristic.
    gq_t, t_t = tfs[box_seg]
    boxes = transform_boxes(ts.terrain_meta["terrain"].get("boxes", []), gq_t, t_t)
    # Normalize the whole traversal (clip + boxes) so the terrain is axis-aligned (box yaw -> ~0):
    # the approach heading is arbitrary, and leaving it in yaws the box under the motion.
    Wp, Wq, boxes = yaw_normalize(Wp, Wq, boxes, gq_t)

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
    # Splice the terrain segment's REAL exported foot contacts over the kinematic ones. Segment 0
    # keeps its seam frame (offset k); later segments dropped theirs in align_and_concat (offset k+1).
    a, b = bounds[box_seg]
    c_off = k if box_seg == 0 else k + 1
    if ts.contact is not None:
        contact[a:b] = ts.contact[c_off:c_off + (b - a)].to(contact.dtype).to(dev)

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt library (output channel layout).")
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--terrain-lib", default="trajectories/skills/terrain_v1")
    p.add_argument("--terrain-hf-repo", default="zolkin/robot_rl")
    p.add_argument("--terrain-skills", default=None, help="';'-sep subset of terrain skills (default all).")
    p.add_argument("--out", required=True)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--approach-s", type=float, default=2.5, help="Approach (DP) duration (s).")
    p.add_argument("--recovery-s", type=float, default=1.0, help="Exit/recovery (DP) duration (s).")
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
    p.add_argument("--clip-length", type=float, default=None,
                   help="Fixed clip length (s); default = the longest traversal.")
    p.add_argument("--boundary-weight", type=float, default=200.0,
                   help="Terminal-pin weight for the approach DP (larger = harder entry pin).")
    p.add_argument("--entry-frac", type=float, default=0.5,
                   help="The approach may hand off into the terrain clip's first this-fraction of "
                        "frames (0..1), picking whichever best matches its incoming speed. 0.5 = "
                        "first half (keeps it from skipping the interaction by entering near the end).")
    p.add_argument("--footprint-margin", type=float, default=2.0, help="Tile footprint margin (m).")
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
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = args.device

    def _csv(s):
        return None if not s else [x.strip() for x in s.split(";") if x.strip()]

    # --mode pure plays the bare interaction (no approach/exit DP), so the candidate graph is unused:
    # skip the expensive build (and the bones-cache load) entirely.
    matcher = _build_matcher(args, dev) if args.mode == "full" else None
    if matcher is None:
        print("[box] --mode pure: skipping candidate-graph build (approach DP not used)")
    manager = MultiSkillManager(args.source, dev, env=None)
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT, device=dev)

    print(f"[box] ingesting terrain library {args.terrain_lib} ...")
    terr_srcs = ingest_terrain_library(args.terrain_lib, hf_repo=args.terrain_hf_repo, dt=args.dt,
                                       device=dev, skills=_csv(args.terrain_skills))
    print(f"[box] {len(terr_srcs)} terrain interaction clips")

    approach_vels = [float(v) for v in _csv(args.approach_vels)]
    print(f"[box] approach speeds: {approach_vels}  -> {len(terr_srcs) * len(approach_vels)} clips")
    records = []
    tid = 0
    for ts in terr_srcs:
        for vx in approach_vels:
            print(f"[box] splicing {ts.name} @ approach vx={vx} (terrain_id={tid}) ...")
            rec = _make_box_clip(matcher, builder, ts, tid, vx, args, dev)
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
