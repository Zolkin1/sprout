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
from motion_gen.clip_output import ClipOutputBuilder        # noqa: E402
from motion_gen.gen_parallel import default_workers, unit_pool  # noqa: E402
from motion_gen.motionbricks import load_raw_motion, raw_to_world           # noqa: E402
from motion_gen.motionbricks.distractor_plan import profile_holds, request_key  # noqa: E402
from motion_gen.motionbricks import flat_plan as fp                       # noqa: E402
from motion_gen.motionbricks.stride_bank import load_stride_bank          # noqa: E402
from mb_flat_clips import assemble_world                                  # noqa: E402
from motion_gen.library_metadata import read_library_metadata
from motion_gen.world_lift import _ANCHOR_MASK_FLAT                          # noqa: E402

_CONTACT = ["left_ankle_roll_link", "right_ankle_roll_link"]
# FLAT mask (downrange/lateral/yaw FREE, height clip-fixed) -- ego-anchored like flat walking, and
# identical to the DP distractors in ``generate_distractor_clips._DISTRACTOR_MASK``. See that
# module's ANCHOR docstring for the trade-off and for the per-step mask that would do better.
#
# DO NOT pin this. Both generators are held to this value by
# ``source/robot_rl/tests/test_distractor_anchor_mask.py``; a pinned distractor silently poisons FM
# recording (the snap fires but the placement cannot move, so the window's target is tracking
# drift). To disable distractors for the FM, use ``train.py --skip_distractors`` instead.
_DISTRACTOR_MASK = int(_ANCHOR_MASK_FLAT)
_SHARED: dict = {}


def _pool(args) -> list[tuple]:
    """The sampled tile pool (kind, box size, per-clip profile specs) -- shared by both passes.

    Deterministic in ``--seed``, so pass 2 reconstructs exactly the pool pass 1 wrote schedules for.
    """
    return _box_pool(args, int(args.num_tiles), np.random.default_rng(args.seed))


def _clip_name(tid: int, spec: dict, i: int) -> str:
    """The plan / transient key for clip ``i`` of tile ``tid`` -- shared by both passes."""
    return f"t{tid:04d}_{request_key(spec, i)}"


def _dump_schedules(args) -> int:
    """Pass 1: one loops-and-in-betweening PLAN per clip of every tile.

    Distractors used to be generated the way everything else was before the redesign -- Motion Bricks
    tracking a piecewise-constant command schedule for the whole six seconds.  They are now built like
    the flat family: exact looped bank strides for the steady phases, Motion Bricks only for the
    changes between them.  The velocities are not sampled here, they come from the clip's command
    profile (:func:`~motion_gen.motionbricks.distractor_plan.profile_holds`), because the box is fitted
    to the motion that profile produces.
    """
    bank = load_stride_bank(args.stride_bank)
    n_frames = int(round(args.short_s / args.dt))
    plans = []
    for tid, (_kind, _sz, specs) in enumerate(_pool(args)):
        for i, spec in enumerate(specs):
            name = _clip_name(tid, spec, i)
            seed = (int(args.seed) * 1000003 + tid * 1009 + i) % (2 ** 31)
            plans.append(fp.plan_from_holds(
                name, profile_holds(spec), np.random.default_rng(seed),
                bank_periods=bank.period_frames, bank_vel=bank.achieved_vel,
                n_frames=n_frames, dt=args.dt, seed=seed))
    fp.to_json(plans, args.dump_schedules)
    n_tr = sum(len(pl.transients) for pl in plans)
    print(f"[mb_distract] {len(plans)} plans, {n_tr} transients "
          f"({len(plans) * args.short_s:.0f}s of motion) -> {args.dump_schedules}")
    return 0


def _tile_worker(tid: int):
    """Fit tile ``tid``'s boxes to its MB motions -> (records, manifest)."""
    S = _SHARED
    args, builder, dev = S["args"], S["builder"], S["dev"]
    kind, box_xyz, specs = S["tiles"][tid]
    T = int(round(args.short_s / args.dt))

    def gen(spec, _cmd, T_, i):
        """The MOTION for clip ``i`` of this tile: its plan's loops spliced with its MB in-betweenings."""
        name = _clip_name(tid, spec, i)
        world, diag = assemble_world(S["bank"], S["plans"][name], args.transient_dir, args.dt, T_,
                                     dev, _DISTRACTOR_MASK, S["blend_n"])
        if world is None:
            raise RuntimeError(f"{name}: {diag.get('skip')}")
        wp, wq, wj, tv, _sidx = world
        return wp, wq, wj, tv

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
    lib_meta = read_library_metadata(args.source)
    builder = ClipOutputBuilder(lib_meta.pos_output_names, lib_meta.vel_output_names,
                                contact_frames=_CONTACT, device=args.device)
    tiles = _pool(args)
    plans = {pl.name: pl for pl in fp.from_json(args.plans)}
    _SHARED.update(args=args, builder=builder, dev=args.device, tiles=tiles,
                   bank=load_stride_bank(args.stride_bank), plans=plans, blend_n=args.blend_n)
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
            "anchor_mask": torch.tensor([_DISTRACTOR_MASK], dtype=torch.uint8),
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
    p.add_argument("--transient-dir", default=None,
                   help="PASS 2: dir of the generated in-betweenings (mb_flat_transients --plans).")
    p.add_argument("--plans", default=None, help="PASS 2: the plan file pass 1 wrote.")
    p.add_argument("--stride-bank", required=True, help="Exact periodic strides (export_stride_bank.py).")
    p.add_argument("--blend-n", type=int, default=12, help="Seam blend length, in frames.")
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
    if bool(args.dump_schedules) == bool(args.transient_dir):
        raise SystemExit("pass exactly one of --dump-schedules (pass 1) or --transient-dir (pass 2)")
    if args.transient_dir and not (args.out and args.plans):
        raise SystemExit("--transient-dir (pass 2) needs --out and --plans")
    for path in (args.dump_schedules, args.out):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    return _dump_schedules(args) if args.dump_schedules else _assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
