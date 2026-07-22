#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate DISTRACTOR box clips: a box the robot sees but does NOT jump on.

Coherence-first design (each clip's box is placed so the motion's relationship to it is valid --
never "walk into a box"): generate SHORT flat motions by TYPE, then fit a box/wall per type so the
box sits exactly where that motion makes sense:

* ``stop``    -- walk up and STOP; box near-face 0.2-0.5 m ahead of the last (stopped) frame.
* ``turn``    -- steer AROUND; box ahead on the pre-turn heading (outside of the turn) once the
                 heading has swung >= ``--turn-deg``.
* ``between`` -- walk BETWEEN two walls; walls flank the path at +/- (lateral spread + clearance).

Box SIZE is an orthogonal axis (small cube -> big box -> wall); every type works at any size. Each
box (tile) is shared by several motions placed relative to it (few tiles, many clips). The flat
motion + its REAL target_vel are kept (negatives: honest velocity -- stop ramps to 0, straight/turn
hold), so the RL learns "box in view != jump".

ANCHOR: these are WORLD-PINNED (anchor = nothing free), NOT ego. If they ego-re-anchored mid-episode
the motion would float off the fixed box and the baked relationship (stop 0.3 m before the box)
would drift away -- the teacher would then see a stop unrelated to the box. Deploy still re-aligns
from perception via the FM.
    TODO(revisit ego): world-pin is the safe default, but ego-anchoring these may well be fine in
    practice -- ego fires on only a FRACTION of envs per episode (ego_anchor_fraction) and only on a
    cadence, so the incoherent fraction is small and may wash out. Worth A/B-ing once training runs.

Runs with NO IsaacLab.

    python scripts/rsl_rl/generate_distractor_clips.py \
        --source source/robot_rl/tests/test_data/merged_library \
        --bones-cache clips/bones_cache_locomotion.h5 --out clips/g1_distractor_v1.h5 --device cpu
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.math_utils import quat_apply, quat_mul
from motion_gen.motion_matching.bones_ingest import load_cache
from motion_gen.motion_matching.box_clip import _quat_yaw_wxyz, se2z_align
from motion_gen.motion_matching.clean_ingest import ingest_clean_library
from motion_gen.motion_matching.clip_output import ClipOutputBuilder
from motion_gen.motion_matching.distractor import occupancy_grid
from motion_gen.motion_matching.dp import DPConfig, DPMatcher
from motion_gen.motion_matching.graph import GraphConfig, build_candidate_graph
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig
from motion_gen.multiskill_manager import MultiSkillManager
from motion_gen.world_lift import anchor_mask_from_free

_CONTACT = ["left_ankle_roll_link", "right_ankle_roll_link"]
_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
_FLAT_MOVEMENT = ["walking", "jogging", "running", "running, turning", "turning",
                  "standing idle", "transition"]
# Exclude non-flat-locomotion junk that the movement labels don't catch: "in_place" (jog/run/march
# in place -- the "jump rope" bounce), "fall"/"crazy" (falls, erratic), "crouch"/"squat" (low
# postures). SEPARATE from generate_box_clips's terrain-jump include -- does NOT touch the mantle jumps.
_FLAT_EXCLUDE = ["injured", "limp", "drunk", "jump", "roll", "in_place", "fall", "crazy", "crouch", "squat"]
_PINNED = anchor_mask_from_free(set())               # world-pinned: no root DOF re-anchors to the robot


# ---- velocity profiles (command = f(t_seconds) -> (vx, vy, wz)) ----
def _straight(vx):
    return lambda _t: (float(vx), 0.0, 0.0)


def _stop(vx, t_stop, ramp_s):
    t0 = t_stop - ramp_s
    def cmd(t):
        if t <= t0:
            return (float(vx), 0.0, 0.0)
        if t >= t_stop:
            return (0.0, 0.0, 0.0)
        return (float(vx * (1.0 - (t - t0) / ramp_s)), 0.0, 0.0)
    return cmd


def _turn(vx, wz):
    return lambda _t: (float(vx), 0.0, float(wz))


# ---- flat matcher (bones + optional clean, NO terrain lead-ins) ----
def _build_flat_matcher(args, dev):
    print("[distractor] loading flat-locomotion bones ...")
    flat, _ = load_cache(args.bones_cache, filters={"package": "Locomotion",
                         "content_type_of_movement": _FLAT_MOVEMENT, "content_body_position": "standing"},
                         exclude_name_substrings=_FLAT_EXCLUDE, limit=args.bones_limit, device=dev)
    print(f"[distractor] + {len(flat)} flat bones clips")
    clean = ingest_clean_library(args.clean_source, hf_repo=args.clean_hf_repo, dt=args.dt,
                                 device=dev) if args.include_clean else []
    sources = flat + clean
    dcfg = DPConfig(dt=args.dt, cut_penalty=args.cut_penalty, w_traj=1.0, w_track=0.0)
    print(f"[distractor] building DP matcher DB ({len(sources)} clips, {sum(s.n for s in sources)} frames) ...")
    matcher = DPMatcher(sources, VanillaConfig(dt=args.dt), dcfg, device=dev)
    gcfg = GraphConfig(k_min=args.graph_k_min, k_max=args.graph_k_max, k_target=args.graph_k_target, dt=args.dt)
    print("[distractor] building candidate graph ...")
    cost_feat = matcher._norm[:, matcher._traj_dim:]
    graph = build_candidate_graph(cost_feat[:, POSE_POSTURAL_IDX], cost_feat, matcher.feat_raw,
                                  matcher.owner, matcher.src_start, matcher.src_len, matcher.yaw_all, gcfg,
                                  steady_owner_mask=matcher._src_steady, periodic_owner_mask=matcher._src_periodic)
    matcher.attach_graph(graph)
    return matcher


def _start_frame(matcher, cmd0):
    """DB frame whose body velocity is nearest ``cmd0`` -> start already on the steady gait."""
    d = ((matcher._bvel - np.asarray(cmd0, dtype=np.float64)) ** 2).sum(1)
    return int(d.argmin())


def _yaw_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _align(rp, rq, ref_idx, tgt_xy, dev):
    """SE2-map the whole clip so frame ``ref_idx`` sits at world ``tgt_xy`` facing +x (heading 0).

    Returns (rp', rq'). Rotates the ref-frame heading to +x and translates its position to ``tgt_xy``
    (z kept), applied rigidly to every frame -- the box is then placed in this common frame so many
    clips share one box.
    """
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    tgt_pos = torch.tensor([float(tgt_xy[0]), float(tgt_xy[1]), float(rp[ref_idx, 2])], device=dev)
    gq, t = se2z_align(rp[ref_idx], rq[ref_idx], tgt_pos, ident)
    n = rp.shape[0]
    rp2 = quat_apply(gq.unsqueeze(0).expand(n, 4), rp) + t
    rq2 = quat_mul(gq.unsqueeze(0).expand(n, 4), rq)
    return rp2, rq2


def _heading_change(rq) -> np.ndarray:
    """|heading change from frame 0| per frame [T] (radians, wrapped to [0, pi])."""
    y0 = _yaw_wxyz(rq[0])
    return np.array([abs(((_yaw_wxyz(rq[i]) - y0 + math.pi) % (2 * math.pi)) - math.pi)
                     for i in range(rq.shape[0])])


def _clear_box(center, size, direction, occ, origin, cell, clearance, step=0.05, max_push=4.0):
    """Push a box (``center``, full extent ``size``) along ``direction`` until its whole footprint +
    ``clearance`` no longer overlaps the dilated swept-footprint ``occ`` -- so a placed box never
    touches the robot (the earlier center-only guard missed edge overlaps). Returns the new center."""
    cx, cy = float(center[0]), float(center[1])
    dx, dy = direction
    hx, hy = size[0] / 2.0 + clearance, size[1] / 2.0 + clearance

    def hits(px, py):
        i0 = max(0, int((px - hx - origin[0]) / cell)); i1 = min(occ.shape[0], int((px + hx - origin[0]) / cell) + 1)
        j0 = max(0, int((py - hy - origin[1]) / cell)); j1 = min(occ.shape[1], int((py + hy - origin[1]) / cell) + 1)
        return i1 > i0 and j1 > j0 and bool(occ[i0:i1, j0:j1].any())

    pushed = 0.0
    while hits(cx, cy) and pushed < max_push:
        cx += dx * step; cy += dy * step; pushed += step
    return [cx, cy]


def _box_pool(args):
    """(kind, box_xyz, n_clips, cmd-builder) tiles. Size axis: small cube -> big box -> wall."""
    n = args.clips_per_tile
    stop_sizes = [(0.5, 0.5, 0.4), (0.8, 0.45, 0.5), (0.6, 0.6, 0.9), (2.5, 0.4, 1.2)]     # ...last = wall
    turn_sizes = [(0.6, 0.6, 0.6), (0.9, 0.5, 0.9), (2.5, 0.4, 1.3)]                        # ...last = wall
    wall_sizes = [(3.0, 0.3, 1.0), (2.0, 0.3, 1.5)]                                         # between: 2 walls each
    tiles = []
    for sz in stop_sizes:
        vx = float(np.random.uniform(0.9, 2.2))
        cmds = [_stop(vx, ts, args.stop_ramp_s) for ts in np.linspace(0.4 * args.short_s, 0.6 * args.short_s, n)]
        tiles.append(("stop", sz, cmds))
    for sz in turn_sizes:
        wz = float(np.random.choice([-1, 1])) * float(np.random.uniform(0.4, 0.7))
        cmds = [_turn(1.2, wz) for _ in range(n)]
        tiles.append(("turn", sz, cmds))
    for sz in wall_sizes:
        vx = float(np.random.uniform(0.9, 1.8))
        cmds = [_straight(v) for v in np.linspace(0.7 * vx, 1.3 * vx, n)]
        tiles.append(("between", sz, cmds))
    return tiles


def _make_tile(matcher, builder, kind, box_xyz, cmds, tid, T, args, dev):
    """Generate a tile's motions, align each by its box-relevant frame, fit the box per type ->
    (records, manifest). All motions in the tile share the one box, placed coherently."""
    pos = builder.pos_names
    pxy = [pos.index("pelvis_link:pos_x"), pos.index("pelvis_link:pos_y")]
    fxy = [[pos.index(f"{l}:pos_x"), pos.index(f"{l}:pos_y")] for l in _FEET]
    x_ch = [i for i, nm in enumerate(pos) if nm.endswith(":pos_x")]
    y_ch = [i for i, nm in enumerate(pos) if nm.endswith(":pos_y")]
    bx, by, bz = box_xyz

    clips, tvels, foot_pts, turn_x = [], [], [], []
    for i, cmd in enumerate(cmds):
        sf = None if args.stand_start else _start_frame(matcher, cmd(0.0))
        rp, rq, jt = matcher.generate(cmd, T, start_frame=sf)
        if kind == "stop":
            # STOP `gap` behind the box near-face (which sits at x=0); vary gap across the tile so
            # the tile's clips stop at a range of distances. gap is pelvis->near-face (~body+0.2).
            gap = float(np.linspace(0.4, 0.7, len(cmds))[i])
            rp2, rq2 = _align(rp, rq, rp.shape[0] - 1, (-gap, 0.0), dev)
        else:
            rp2, rq2 = _align(rp, rq, 0, (0.0, 0.0), dev)          # start at origin facing +x
            if kind == "turn":                                     # record where the heading swings >= turn_deg
                over = _heading_change(rq2) >= math.radians(args.turn_deg)
                k = int(over.argmax()) if bool(over.any()) else rp.shape[0] - 1
                turn_x.append(float(rp2[k, 0]))
        y, dy, contact = builder.build(rp2, rq2, jt, args.dt)
        clips.append((y, dy, contact))
        tvels.append(torch.tensor([cmd(t * args.dt) for t in range(T)], dtype=torch.float32, device=dev))
        pel = y[:, pxy].cpu().numpy()
        ft = np.concatenate([y[:, ix].cpu().numpy() for ix in fxy], axis=0)
        foot_pts.append(np.concatenate([pel, ft], axis=0))

    allpts = np.concatenate(foot_pts, axis=0)
    # occupancy = the union swept footprint dilated by the robot radius; a box overlapping this HITS.
    origin = allpts.min(0) - 2.0
    shape = (int((allpts[:, 0].max() + 2.0 - origin[0]) / args.cell) + 1,
             int((allpts[:, 1].max() + 2.0 - origin[1]) / args.cell) + 1)
    occ = occupancy_grid(allpts, origin, args.cell, shape, args.robot_radius)

    # --- fit the box(es) per type (pos_xy = box CENTER), each with a push direction (away from path) ---
    if kind == "stop":                                             # near-face at x=0 -> robot stops behind; push +x
        overlays = [{"pos_xy": [bx / 2.0, 0.0], "size": [bx, by, bz], "_dir": (1.0, 0.0)}]
    elif kind == "turn":                                           # ahead on +x, beyond the turn (outside); push +x
        ahead = (max(turn_x) if turn_x else 1.0) + bx / 2.0 + 0.3
        overlays = [{"pos_xy": [ahead, 0.0], "size": [bx, by, bz], "_dir": (1.0, 0.0)}]
    else:                                                          # between: walls flank the path; push each outward
        off = float(np.abs(allpts[:, 1]).max()) + args.wall_clearance + by / 2.0
        xc = float(allpts[:, 0].mean())
        overlays = [{"pos_xy": [xc, off], "size": [bx, by, bz], "_dir": (0.0, 1.0)},
                    {"pos_xy": [xc, -off], "size": [bx, by, bz], "_dir": (0.0, -1.0)}]
    # push each box clear of the footprint so it never hits the robot (whole-footprint + clearance)
    for o in overlays:
        o["pos_xy"] = _clear_box(o["pos_xy"], o["size"], o.pop("_dir"), occ, origin, args.cell, args.box_clearance)

    # Tile bbox over the swept footprint AND every box extent, then pad by ``footprint_margin`` on
    # ALL FOUR sides (corner = min - margin, size = span + 2*margin) so there is always ground around
    # every box/wall -- the height scan sees floor beside an obstacle instead of it sitting flush at
    # the tile edge.  The clips recenter by the same corner (cx0, cy0) below.
    m = float(args.footprint_margin)
    xmin = float(min(allpts[:, 0].min(), min(o["pos_xy"][0] - o["size"][0] / 2 for o in overlays)))
    xmax = float(max(allpts[:, 0].max(), max(o["pos_xy"][0] + o["size"][0] / 2 for o in overlays)))
    ymin = float(min(allpts[:, 1].min(), min(o["pos_xy"][1] - o["size"][1] / 2 for o in overlays)))
    ymax = float(max(allpts[:, 1].max(), max(o["pos_xy"][1] + o["size"][1] / 2 for o in overlays)))
    cx0, cy0 = xmin - m, ymin - m
    ov_manifest = [{"kind": "box", "pos_xy": [o["pos_xy"][0] - cx0, o["pos_xy"][1] - cy0],
                    "size_xyz": o["size"], "base_z": 0.0, "walkable_top": True, "yaw": 0.0} for o in overlays]
    size_x = (xmax - xmin) + 2 * m
    size_y = (ymax - ymin) + 2 * m
    manifest = {"terrain_id": tid, "type": "box", "size_x": size_x, "size_y": size_y, "overlays": ov_manifest}

    records = []
    for (y, dy, contact), tvel in zip(clips, tvels):
        y = y.clone()
        y[:, x_ch] -= cx0
        y[:, y_ch] -= cy0
        records.append({"y": y, "dy": dy, "contact": contact, "target_vel": tvel,
                        "terrain_id": tid, "n": T, "name": f"distractor:{kind}"})
    return records, manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt library (output channel layout).")
    p.add_argument("--bones-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--short-s", type=float, default=3.0, help="Distractor clip length (s).")
    p.add_argument("--clips-per-tile", type=int, default=6, help="Motions sharing each box tile.")
    p.add_argument("--stand-start", action="store_true", help="Start from standing (default: steady gait).")
    p.add_argument("--stop-ramp-s", type=float, default=0.8)
    p.add_argument("--turn-deg", type=float, default=20.0, help="Heading swing that marks the turn point.")
    p.add_argument("--robot-radius", type=float, default=0.25)
    p.add_argument("--wall-clearance", type=float, default=0.4, help="Half-gap margin for walk-between walls.")
    p.add_argument("--box-clearance", type=float, default=0.2,
                   help="Extra margin (m) a box is pushed beyond the robot's swept footprint so it never hits.")
    p.add_argument("--cell", type=float, default=0.1)
    p.add_argument("--footprint-margin", type=float, default=1.0)
    p.add_argument("--cut-penalty", type=float, default=3.0)
    p.add_argument("--include-clean", action="store_true")
    p.add_argument("--clean-source", default="trajectories/skills/library_v1")
    p.add_argument("--clean-hf-repo", default="zolkin/robot_rl")
    p.add_argument("--bones-limit", type=int, default=1500)
    p.add_argument("--graph-k-max", type=int, default=50)
    p.add_argument("--graph-k-min", type=int, default=25)
    p.add_argument("--graph-k-target", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = args.device
    np.random.seed(args.seed)

    matcher = _build_flat_matcher(args, dev)
    manager = MultiSkillManager(args.source, dev, env=None)
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT, device=dev)
    T = max(1, int(round(args.short_s / args.dt)))

    tiles = _box_pool(args)
    print(f"[distractor] {len(tiles)} box tiles x {args.clips_per_tile} clips = {len(tiles) * args.clips_per_tile}")
    records, manifests = [], []
    for tid, (kind, box_xyz, cmds) in enumerate(tiles):
        recs, manifest = _make_tile(matcher, builder, kind, box_xyz, cmds, tid, T, args, dev)
        print(f"[distractor] tile {tid} {kind} box {box_xyz}: {len(recs)} clips, {len(manifest['overlays'])} box(es)")
        records.extend(recs)
        manifests.append(manifest)

    meta = {"dt": args.dt, "clip_length_s": T * args.dt, "num_steps": int(T),
            "pos_output_names": list(manager.pos_output_names),
            "vel_output_names": list(manager.vel_output_names),
            "contact_frames": _CONTACT, "ref_frame": "pelvis_link",
            "skill_list": [s.name for s in manager.skills]}
    writer = ClipLibraryWriter(args.out, len(records), meta)
    writer.set_attr("generator", "distractor")
    writer.set_attr("terrain_manifest", manifests)
    for i, r in enumerate(records):
        writer.write_batch(i, {
            "y_des": r["y"].unsqueeze(0), "dy_des": r["dy"].unsqueeze(0), "contact": r["contact"].unsqueeze(0),
            "skill_id": torch.zeros(1, T, dtype=torch.uint8, device=dev),
            "stride_idx": torch.full((1, T), -1, dtype=torch.int32, device=dev),
            "target_vel": r["target_vel"].unsqueeze(0),
            "terrain_id": torch.tensor([r["terrain_id"]], dtype=torch.int32, device=dev),
            "terrain_type": "box",
            "anchor_mask": torch.tensor([_PINNED], dtype=torch.uint8, device=dev),
        })
    writer.close()

    lib = load_clip_library(args.out)
    yy = torch.as_tensor(lib.y_des)
    print(f"[distractor] wrote {args.out}: {lib.num_clips} clips x {lib.num_steps} steps, "
          f"y_des {tuple(yy.shape)} finite={bool(torch.isfinite(yy).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
