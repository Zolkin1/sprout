#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Passes A and C of the loop-based flat family: author the plans, then assemble the clips.

``--dump-plan`` (pass A) samples one :class:`~motion_gen.motionbricks.flat_plan.FlatClipPlan` per clip:
which strides it loops, at which gait phases it enters and leaves each, and therefore which transients
Motion Bricks has to in-between. ``--assemble`` (pass C) replays the loops exactly, splices in the
generated transients, and writes the clip library.

Why the clip is exact in length and in velocity:

* every steady stretch is :func:`~motion_gen.terrain.box_clip.loop_frames` over an optimized stride, so
  its velocity is the stride's own and its contacts are as solved;
* a loop's LENGTH is free, so pass A picks one that lands its last frame on the hand-off phase the
  transient was seeded from -- the requested hold is honoured to within a gait cycle, the phase exactly;
* the transients' durations are Motion Bricks' choice and only known now, so the FINAL loop absorbs
  whatever is left over. That is what makes ``clip_len == T`` by construction rather than by padding.

Runs in ``env_isaaclab`` (needs ``motion_gen`` + torch, no simulator)::

    # pass A
    PYTHONPATH=source/motion_gen:source/robot_rl CUDA_VISIBLE_DEVICES="" env_isaaclab/bin/python \
      scripts/rsl_rl/mb_flat_clips.py --dump-plan mb_raw/x/flat_plans.json \
        --stride-bank mb_raw/x/stride_bank.npz --n-clips 3000 --clip-length 20.0
    # pass C (after mb_flat_transients.py)
    ... --assemble --plans mb_raw/x/flat_plans.json --transient-dir mb_raw/x/flat_transients \
        --stride-bank ... --source hf/trajectories/retargeted/... --out mb_raw/x/family_flat.h5
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from motion_gen.clips.clip_io import ClipLibraryWriter
from motion_gen.clip_output import ClipOutputBuilder
from motion_gen.library_metadata import read_library_metadata
from motion_gen.motionbricks import flat_plan as fp
from motion_gen.motionbricks.mb_ingest import raw_to_world
from motion_gen.motionbricks.raw_schema import load_raw_motion
from motion_gen.motionbricks.stride_bank import load_stride_bank
from motion_gen.motionbricks.terrain_splice import Piece, splice_pieces
from motion_gen.terrain.box_clip import loop_frames
from motion_gen.world_lift import anchor_mask_from_free

_CONTACT_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _split(qpos, dev: str):
    q = torch.as_tensor(np.asarray(qpos, np.float32), dtype=torch.float32, device=dev)
    return q[:, 0:3], q[:, 3:7], q[:, 7:36]


def _loop_piece(bank, lp, n_exact: int, dev: str, mask: int) -> Piece:
    """An exact looped replay contributing ``n_exact`` frames starting at ``lp.phase0``.

    One frame of LEAD-IN is prepended: ``align_and_concat`` drops each piece's first frame as a duplicate
    of the previous piece's last, so without it the loop would lose its own ``phase0`` frame and start one
    frame late.  With it, every surviving frame of this piece is an exact stride frame.
    """
    period = int(bank.period_frames[lp.stride])
    rp, rq, jt = _split(bank.strides[lp.stride], dev)
    lead = (int(lp.phase0) - 1) % period
    lrp, lrq, ljt, _ = loop_frames(rp, rq, jt, None, n_exact + 1, start=lead)
    return Piece(seg=(lrp, lrq, ljt), kind="loop", rigid=True, mask=mask,
                 stride_idx=int(lp.stride))


def _transient_piece(path: str, dt: float, dev: str, mask: int) -> tuple[Piece, np.ndarray]:
    """Load a generated transient, resample it to the control rate, return ``(piece, target_vel)``."""
    raw = load_raw_motion(path)
    wp, wq, wj, tvel = raw_to_world(raw, dt)
    seg = (torch.as_tensor(wp, dtype=torch.float32, device=dev),
           torch.as_tensor(wq, dtype=torch.float32, device=dev),
           torch.as_tensor(wj, dtype=torch.float32, device=dev))
    return Piece(seg=seg, kind="transient", rigid=False, mask=mask), tvel


def _assemble_clip(bank, plan, transient_dir: str, builder, dt: float, t_frames: int, dev: str,
                   mask: int, blend_n: int):
    """Build one clip -> ``(y, dy, contact, target_vel, stride_idx_t, diag)``."""
    world, diag = assemble_world(bank, plan, transient_dir, dt, t_frames, dev, mask, blend_n)
    if world is None:
        return None, diag
    wp, wq, wj, tv, sidx = world
    y, dy, contact = builder.build(wp, wq, wj, dt)
    return (y, dy, contact, tv, sidx), diag


def assemble_world(bank, plan, transient_dir: str, dt: float, t_frames: int, dev: str,
                   mask: int, blend_n: int):
    """Splice one plan into WORLD motion -> ``((root_pos, root_quat, joints, target_vel, stride_idx), diag)``.

    Split out of :func:`_assemble_clip` because the distractor family needs the world motion BEFORE the
    observation build: its box is fitted to the swept footprint of the motion, so the geometry is only
    decidable once the clip exists (``generate_distractor_clips._make_tile``).
    """
    # Load the transients first: their lengths are Motion Bricks' choice, and every loop length below is
    # sized against what is left after them.
    trs, tvels = {}, {}
    for tr in plan.transients:
        piece, tv = _transient_piece(os.path.join(transient_dir, f"{tr.key}.npz"), dt, dev, mask)
        trs[tr.key], tvels[tr.key] = piece, tv

    # Pass A had to RESERVE a worst case for each transient, so the actual total is usually short and the
    # final loop takes up the slack.  When it is instead LONG -- Motion Bricks chose near-maximum horizons
    # throughout -- drop the trailing (transient, loop) pair and re-solve rather than lose the clip.
    plan_pieces = list(plan.pieces)
    while True:
        loops = [pc for pc in plan_pieces if pc.kind == "loop"]
        if not loops:
            return None, {"skip": "no loop survived the length solve"}
        n_tr_pieces = [pc for pc in plan_pieces if pc.kind == "transient"]
        if plan_pieces[0].kind != "transient":
            return None, {"skip": "plan does not start with a transient; the frame accounting assumes it"}
        # Frame accounting.  ``align_and_concat`` drops the FIRST frame of every piece but the first, and
        # each loop already asks for one frame of lead-in to pay for exactly that (see _loop_piece), so a
        # loop contributes precisely its ``want``.  Only the welds BETWEEN TRANSIENTS are unpaid, and the
        # opening transient keeps all its frames -- hence ``n_t - 1``, not ``n_pieces - 1``.
        n_tr = sum(trs[pc.key].seg[0].shape[0] for pc in n_tr_pieces) - max(0, len(n_tr_pieces) - 1)
        want = [fp.loop_frames_for(lp.phase0, lp.phase_out, int(bank.period_frames[lp.stride]),
                                   int(round(lp.hold_s / dt)),
                                   min_frames=max(2 * blend_n + 1, int(bank.period_frames[lp.stride])))
                for lp in loops]
        floor = int(bank.period_frames[loops[-1].stride])
        residual = t_frames - (n_tr + sum(want[:-1]))
        if residual >= floor:
            want[-1] = residual
            break
        if len(plan_pieces) <= 2:
            return None, {"skip": f"even one pair overruns the clip (residual {residual} < {floor})"}
        plan_pieces = plan_pieces[:-2]

    pieces: list[Piece] = []
    seg_vel: list[np.ndarray] = []
    li = 0
    for pc in plan_pieces:
        if pc.kind == "transient":
            pieces.append(trs[pc.key])
            seg_vel.append(tvels[pc.key][0])           # constant over the transient: where it is going
        else:
            pieces.append(_loop_piece(bank, pc, want[li], dev, mask))
            seg_vel.append(bank.achieved_vel[pc.stride])
            li += 1

    wp, wq, wj, bounds, seams, diag = splice_pieces(pieces, blend_n=blend_n, dt=dt)
    if wp.shape[0] != t_frames:
        return None, {"skip": f"assembled {wp.shape[0]} frames, wanted {t_frames}"}

    tv = torch.zeros(t_frames, 3, dtype=torch.float32, device=dev)
    sidx = torch.full((t_frames,), -1, dtype=torch.int32, device=dev)
    for (s, e), v, pc in zip(bounds, seg_vel, pieces):
        tv[s:e] = torch.as_tensor(np.asarray(v, np.float32), dtype=torch.float32, device=dev)
        if pc.stride_idx >= 0:
            sidx[s:e] = int(pc.stride_idx)
    diag["seams"] = seams
    diag["loop_frac"] = float((sidx >= 0).float().mean())
    return (wp, wq, wj, tv, sidx), diag


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stride-bank", required=True)
    p.add_argument("--dump-plan", default=None, help="PASS A: write plans to this .json and exit.")
    p.add_argument("--assemble", action="store_true", help="PASS C: build the clip library.")
    p.add_argument("--plans", default=None, help="Plans .json (pass C).")
    p.add_argument("--transient-dir", default=None, help="Generated transients (pass C).")
    p.add_argument("--source", default=None, help="Traj-opt library root, for the channel layout (pass C).")
    p.add_argument("--out", default=None, help="Output clip library .h5 (pass C).")
    p.add_argument("--n-clips", type=int, default=64, help="Clips to plan (pass A).")
    p.add_argument("--clip-length", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--hold-s-range", type=float, nargs=2, default=(1.5, 4.0), metavar=("MIN", "MAX"),
                   help="Requested STEADY duration per loop, before rounding to the hand-off phase. The "
                        "old flat sampler drew 1.5-3.0 s COMMAND segments, which after a ~1 s transient "
                        "left about one gait cycle of steady -- the hold and the transient are separate "
                        "quantities now.")
    p.add_argument("--seam-blend-s", type=float, default=0.24,
                   help="Seam ease length (s). Applies at loop<->transient seams only; a loop's own "
                        "period seams need nothing (the strides close bit-exactly).")
    p.add_argument("--flat-anchor", default="x,y,yaw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    bank = load_stride_bank(args.stride_bank)
    t_frames = max(1, int(round(args.clip_length / args.dt)))
    blend_n = max(0, int(round(args.seam_blend_s / args.dt)))

    if args.dump_plan:
        rng = np.random.default_rng(args.seed)
        plans = [fp.sample_plan(f"flat{i:05d}", rng, bank_periods=bank.period_frames,
                                bank_vel=bank.achieved_vel, n_frames=t_frames, dt=args.dt,
                                seed=args.seed + i, hold_s_range=tuple(args.hold_s_range),
                                blend_n=blend_n)
                 for i in range(args.n_clips)]
        os.makedirs(os.path.dirname(args.dump_plan) or ".", exist_ok=True)
        fp.to_json(plans, args.dump_plan)
        n_tr = sum(len(pl.transients) for pl in plans)
        n_lp = sum(len(pl.loops) for pl in plans)
        print(f"[flat_plan] {len(plans)} clips -> {args.dump_plan}")
        print(f"[flat_plan]   {n_tr} transients, {n_lp} loops "
              f"({n_lp / len(plans):.1f} steady stretches per clip)")
        print(f"[flat_plan]   MB calls per clip: {n_tr / len(plans):.1f} "
              f"(the replan loop needed ~{args.clip_length * 30 / 16:.0f})")
        return 0

    if not args.assemble:
        raise SystemExit("give --dump-plan (pass A) or --assemble (pass C)")
    for need in ("plans", "transient_dir", "source", "out"):
        if not getattr(args, need):
            raise SystemExit(f"--assemble needs --{need.replace('_', '-')}")

    plans = fp.from_json(args.plans)
    lib_meta = read_library_metadata(args.source)
    builder = ClipOutputBuilder(lib_meta.pos_output_names, lib_meta.vel_output_names,
                                contact_frames=_CONTACT_FRAMES, device=args.device)
    mask = int(anchor_mask_from_free([x.strip() for x in args.flat_anchor.split(",") if x.strip()]))
    meta = {"dt": args.dt, "clip_length_s": args.clip_length, "num_steps": int(t_frames),
            "pos_output_names": list(builder.pos_names), "vel_output_names": list(builder.vel_names),
            "contact_frames": _CONTACT_FRAMES, "ref_frame": "pelvis_link",
            "skill_list": lib_meta.skill_list}

    recs, skips = [], []
    for pl in plans:
        try:
            out, diag = _assemble_clip(bank, pl, args.transient_dir, builder, args.dt, t_frames,
                                       args.device, mask, blend_n)
        except (Exception, SystemExit) as e:            # SystemExit is NOT an Exception subclass
            out, diag = None, {"skip": f"{type(e).__name__}: {e}"}
        if out is None:
            skips.append((pl.name, diag["skip"]))
            continue
        recs.append((pl.name, out, diag))
    if not recs:
        raise SystemExit(f"[flat_clips] every clip failed; first: {skips[:3]}")

    writer = ClipLibraryWriter(args.out, len(recs), meta)
    writer.set_attr("generator", "flat_loops_v1")
    writer.set_attr("generator_config", {"stride_bank": os.path.abspath(args.stride_bank),
                                         "hold_s_range": list(args.hold_s_range),
                                         "seam_blend_s": args.seam_blend_s, "seed": args.seed})
    writer.set_attr("has_regime", True)     # stride_idx >= 0 marks the exact-loop frames, for QC
    for i, (_name, (y, dy, contact, tv, sidx), _d) in enumerate(recs):
        writer.write_batch(i, {
            "y_des": y.unsqueeze(0), "dy_des": dy.unsqueeze(0), "contact": contact.unsqueeze(0),
            "target_vel": tv.unsqueeze(0), "terrain_id": torch.tensor([-1], dtype=torch.int32),
            "terrain_type": "flat", "anchor_mask": torch.tensor([mask], dtype=torch.uint8),
            "clip_len": torch.tensor([t_frames], dtype=torch.int32),
            "stride_idx": sidx.unsqueeze(0)})
    writer.close()

    lf = np.array([d["loop_frac"] for _, _, d in recs])
    sm = np.array([s.get("seam_speed", 0.0) for _, _, d in recs for s in d["seams"]])
    jr = np.array([s.get("joint_res", 0.0) for _, _, d in recs for s in d["seams"]])
    print(f"[flat_clips] wrote {args.out}: {len(recs)} clips ({len(skips)} skipped)")
    print(f"[flat_clips]   exact-loop fraction: p50={np.median(lf):.3f} min={lf.min():.3f}")
    print(f"[flat_clips]   seam joint residual: p50={np.median(jr):.3f} p95={np.percentile(jr,95):.3f} rad")
    print(f"[flat_clips]   seam speed: p50={np.median(sm):.3f} p95={np.percentile(sm,95):.3f} m/s")
    for name, why in skips[:10]:
        print(f"[flat_clips]   SKIP {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
