#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""DISTRACTOR family, Motion Bricks sourced: a box the robot SEES but never touches.

A distractor clip is flat locomotion with a box fitted where that motion makes it coherent -- walk up and
STOP short of it, steer AROUND it, walk BETWEEN two walls. There is no terrain interaction, so none of the
terrain plan/segment/splice machinery applies: the only thing Motion Bricks has to supply is the flat motion
for a command profile. Everything that makes a distractor coherent -- SE(2) alignment by the box-relevant
frame, the swept-footprint occupancy grid, per-type box fitting, the push-clear so a box never touches the
robot, the tile footprint and manifest -- is reused VERBATIM from ``generate_distractor_clips._make_tile``
via its ``gen`` hook, so the MB and DP distractor families differ only in the motion source.

Two passes, mirroring the terrain path (the generator runs in the other conda env):

**Pass 1** ``--dump-schedules s.json``: sample the tile pool (``generate_distractor_clips._box_pool``) and
write every clip's command profile as an MB schedule.
**Pass 2** ``--mb-raw-dir <dir> --out f.h5``: fit the boxes to the generated motions and write the family.

Between them: ``mb_encode.py --schedules s.json --out-dir <dir>`` in the ``motionbricks`` env. Example::

    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" \
    env_isaaclab/bin/python scripts/rsl_rl/mb_distractor_clips.py --source <traj-lib> \
        --num-tiles 20 --clips-per-tile 50 --dump-schedules mb_raw/lib_v1/distractor_schedules.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_distractor_clips import _box_pool, _make_tile        # noqa: E402

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library   # noqa: E402
from motion_gen.motion_matching.clip_output import ClipOutputBuilder        # noqa: E402
from motion_gen.motion_matching.gen_parallel import default_workers, unit_pool  # noqa: E402
from motion_gen.motionbricks import load_raw_motion, raw_to_world           # noqa: E402
from motion_gen.motionbricks.distractor_plan import profile_segs, request_key  # noqa: E402
from motion_gen.multiskill_manager import MultiSkillManager                 # noqa: E402
from motion_gen.world_lift import anchor_mask_from_free                      # noqa: E402

_CONTACT = ["left_ankle_roll_link", "right_ankle_roll_link"]
_PINNED = int(anchor_mask_from_free(set()))       # world-pinned, exactly as the DP distractors are
# TODO(distractor-ego-mask): distractors are FLAT walking that merely happens to route around
# obstacle boxes, so their anchor mask should be the FLAT one (x, y, yaw FREE; z clip-fixed) --
# NOT fully world-pinned. Pinned means no root DOF ever re-anchors onto the robot, so a
# distractor env is never a re-anchor event: with the snap mask now tied to ego re-anchoring
# (resets.py), these clips emit no FM windows at all, and under the old
# ``ego_mask | ~is_flat`` they emitted windows that were pure tracking drift. The boxes are
# obstacles to avoid, not surfaces to land on, so nothing needs world-pinning here.
# Fix = use ``anchor_mask_from_free({"x", "y", "yaw"})`` (== _ANCHOR_MASK_FLAT) here and in
# generate_distractor_clips.py, then regenerate the distractor family.
# Until then, EXCLUDE the distractor family from FM recording libraries.
_SHARED: dict = {}


def _pool(args) -> list[tuple]:
    """The sampled tile pool (kind, box size, per-clip profile specs) -- shared by both passes.

    Deterministic in ``--seed``, so pass 2 reconstructs exactly the pool pass 1 wrote schedules for.
    """
    return _box_pool(args, int(args.num_tiles), np.random.default_rng(args.seed))


def _dump_schedules(args) -> int:
    """Pass 1: write one MB schedule per clip of every tile."""
    scheds = []
    for tid, (kind, _sz, specs) in enumerate(_pool(args)):
        for i, spec in enumerate(specs):
            scheds.append({"name": f"t{tid:04d}_{request_key(spec, i)}", "segs": profile_segs(spec)})
    with open(args.dump_schedules, "w") as f:
        json.dump(scheds, f, indent=1)
    secs = sum(sum(d for d, *_ in s["segs"]) for s in scheds)
    print(f"[mb_distract] {len(scheds)} schedules ({secs:.0f}s of motion) -> {args.dump_schedules}")
    return 0


def _tile_worker(tid: int):
    """Fit tile ``tid``'s boxes to its MB motions -> (records, manifest)."""
    S = _SHARED
    args, builder, dev = S["args"], S["builder"], S["dev"]
    kind, box_xyz, specs = S["tiles"][tid]
    T = int(round(args.short_s / args.dt))

    def gen(spec, _cmd, T_, i):
        """The MOTION for clip ``i`` of this tile: the Motion Bricks clip generated for its profile."""
        path = os.path.join(args.mb_raw_dir, f"t{tid:04d}_{request_key(spec, i)}.npz")
        Wp, Wq, Wj, _tvel = raw_to_world(load_raw_motion(path), args.dt, device=dev)
        n = Wp.shape[0]
        if n < T_:                                   # repeat-last if MB produced a hair less than asked
            pad = T_ - n
            Wp = torch.cat([Wp, Wp[-1:].expand(pad, 3)])
            Wq = torch.cat([Wq, Wq[-1:].expand(pad, 4)])
            Wj = torch.cat([Wj, Wj[-1:].expand(pad, Wj.shape[1])])
        return Wp[:T_], Wq[:T_], Wj[:T_]

    try:
        recs, manifest = _make_tile(None, builder, kind, box_xyz, specs, tid, T, args, dev, gen=gen)
    except (Exception, SystemExit) as e:
        print(f"[mb_distract][SKIP] tile {tid} ({kind}): {type(e).__name__}: {e}", flush=True)
        return []
    for r in recs:
        r["manifest"] = manifest
    print(f"[mb_distract] tile {tid} ({kind}): {len(recs)} clips, "
          f"footprint {manifest['size_x']:.1f}x{manifest['size_y']:.1f} m", flush=True)
    return recs


def _assemble(args) -> int:
    """Pass 2: fit boxes to the generated motions and write the distractor family."""
    manager = MultiSkillManager(args.source, args.device, env=None)
    builder = ClipOutputBuilder(manager.pos_output_names, manager.vel_output_names,
                                contact_frames=_CONTACT, device=args.device)
    tiles = _pool(args)
    _SHARED.update(args=args, builder=builder, dev=args.device, tiles=tiles)
    n_workers = default_workers(len(tiles), args.workers)
    print(f"[mb_distract] {len(tiles)} tiles x {args.clips_per_tile} clips, workers={n_workers}")
    recs = unit_pool(_tile_worker, range(len(tiles)), n_workers)
    if not recs:
        raise SystemExit("[mb_distract] no clip survived")

    T = max(int(r["y"].shape[0]) for r in recs)
    meta = {"dt": args.dt, "clip_length_s": T * args.dt, "num_steps": T,
            "pos_output_names": list(builder.pos_names), "vel_output_names": list(builder.vel_names),
            "contact_frames": _CONTACT, "ref_frame": "pelvis_link", "skill_list": ["distractor"]}
    writer = ClipLibraryWriter(args.out, len(recs), meta)
    writer.set_attr("generator", "motionbricks_distractor_v1")
    writer.set_attr("terrain_manifest", [r["manifest"] for r in recs])
    for i, r in enumerate(recs):
        writer.write_batch(i, {
            "y_des": r["y"].unsqueeze(0), "dy_des": r["dy"].unsqueeze(0),
            "contact": r["contact"].unsqueeze(0), "target_vel": r["target_vel"].unsqueeze(0),
            "skill_id": torch.zeros(1, T, dtype=torch.uint8),
            "stride_idx": torch.full((1, T), -1, dtype=torch.int32),
            "terrain_id": torch.tensor([int(r["terrain_id"])], dtype=torch.int32),
            "terrain_type": "box",
            "anchor_mask": torch.tensor([_PINNED], dtype=torch.uint8),
            "clip_len": torch.tensor([int(r["n"])], dtype=torch.int32),
        })
    writer.close()
    lib = load_clip_library(args.out)
    y = torch.as_tensor(lib.y_des)
    n_tiles = len({int(m["terrain_id"]) for m in lib.attrs["terrain_manifest"]})
    print(f"[mb_distract] wrote {args.out}: {lib.num_clips} clips x {lib.num_steps} steps over {n_tiles} "
          f"tiles, finite={bool(torch.isfinite(y).all())}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Traj-opt library root (channel layout).")
    p.add_argument("--dump-schedules", default=None, help="PASS 1: write MB schedules here and exit.")
    p.add_argument("--mb-raw-dir", default=None, help="PASS 2: dir of the MB motions (mb_encode --schedules).")
    p.add_argument("--out", default=None, help="PASS 2: output .h5.")
    p.add_argument("--num-tiles", type=int, default=20)
    p.add_argument("--clips-per-tile", type=int, default=50)
    p.add_argument("--short-s", type=float, default=6.0, help="Distractor clip duration (s).")
    p.add_argument("--stop-ramp-s", type=float, default=0.8, help="Decel window of a 'stop' profile (s).")
    p.add_argument("--turn-deg", type=float, default=35.0, help="Heading swing that counts as the turn.")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    # geometry / placement knobs (same defaults as generate_distractor_clips.py)
    p.add_argument("--cell", type=float, default=0.05, help="Occupancy-grid cell (m).")
    p.add_argument("--robot-radius", type=float, default=0.35, help="Footprint dilation (m).")
    p.add_argument("--box-clearance", type=float, default=0.15, help="Extra gap a box is pushed clear by (m).")
    p.add_argument("--wall-clearance", type=float, default=0.35, help="'between' wall gap beyond the path (m).")
    p.add_argument("--footprint-margin", type=float, default=2.0, help="Ground margin around the tile (m).")
    p.add_argument("--stand-start", action="store_true", help="Unused here (MB clips start from a stand).")
    args = p.parse_args()
    if bool(args.dump_schedules) == bool(args.mb_raw_dir):
        raise SystemExit("pass exactly one of --dump-schedules (pass 1) or --mb-raw-dir (pass 2)")
    if args.mb_raw_dir and not args.out:
        raise SystemExit("--mb-raw-dir (pass 2) needs --out")
    for path in (args.dump_schedules, args.out):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    return _dump_schedules(args) if args.dump_schedules else _assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
