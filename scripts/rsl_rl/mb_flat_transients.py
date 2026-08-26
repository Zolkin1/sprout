#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pass B (Motion Bricks env): generate ONE segment per flat transient, and nothing else.

In the loop-based flat pipeline Motion Bricks is an in-betweener, not a locomotion generator: every
steady stretch is an exact looped replay of an optimized stride, and the model only fills the transitions
between them. So this driver walks a plan (:mod:`motion_gen.motionbricks.flat_plan`) and generates one
segment per transient, given full freedom over position and timing:

* **context** = the outgoing loop's last ``nfpt`` frames, sampled at the MODEL's frame rate and ending
  exactly on the phase the loop hands off at, so the generated motion continues that gait;
* **target** = the incoming stride's ``nfpt`` frames from its entry phase -- pinned in SPACE (the ordinary
  exact-velocity root pin at the incoming command) but NOT in time. The model's own num-token head
  decides how long the transition takes, which is precisely what it is good at.
* **single-shot**: one generation, then the whole chosen horizon is drained. The MPC replan loop the old
  flat path used re-issued the same relative problem every 16 frames and re-quantized the duration each
  time, which is what turned a one-off timing choice into a sustained velocity bias.

Runs ~6 model calls per clip instead of ~37. Emits the same ``RawMotion`` contract as every other
Stage-1 driver, keyed by the plan's transient ``key`` so pass C can find it::

    PYTHONPATH=source/motion_gen:source/robot_rl conda run -n motionbricks \
    python scripts/rsl_rl/mb_flat_transients.py --mb-root <GR00T>/motionbricks \
        --stride-bank mb_raw/x/stride_bank.npz --plans mb_raw/x/flat_plans.json \
        --out-dir mb_raw/x/flat_transients
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_env  # noqa: E402

# How many leading frames to consider when locating the end of the re-predicted seed.  Motion Bricks
# opens a context-seeded generation by re-predicting its seed (the pre-filter blends the first frames
# back toward the supplied history), so those frames replay the loop we just left; concatenating from
# frame 0 would repeat a step.  Structurally that is ``nfpt`` frames, but the terrain exit measured the
# best match one frame later, so the index is FOUND within this window and reported -- a wrong constant
# here is a seam spike, and it is cheap to just look.
_SEED_SEARCH = 8


def _seed_overlap(frames: np.ndarray, seed_last: np.ndarray, nfpt: int) -> int:
    """Index of the last frame that still replays the seed: drop ``[0, idx]``, keep from ``idx + 1``."""
    k = min(_SEED_SEARCH, frames.shape[0] - 1)
    if k < 1:
        return 0
    d = np.linalg.norm(frames[:k, 7:] - seed_last[None, 7:], axis=1)
    return int(np.argmin(d))


def _min_tokens_for(mb, v0, v1, a_max: float, alpha_max: float, floor_s: float) -> tuple[int, float]:
    """Fewest tokens that give the transition enough TIME to be physically trackable.

    Motion Bricks picks its own horizon and -- measured over 16 real transients -- that choice is
    essentially uncorrelated with the size of the velocity change it is bridging (r = -0.03). Left free it
    produced a 3.62 -> 0.80 m/s change in 0.53 s, i.e. **5.3 m/s^2**, which nothing can track. So bound the
    duration from below by the change itself: ``T >= |dv| / a_max`` and ``T >= |dwz| / alpha_max``, never
    below ``floor_s``.  The model keeps every choice ABOVE the floor, which is where it already sits most
    of the time (median 12-14 tokens), so this trims the infeasible tail rather than dictating a duration.

    Frames survive as ``4N`` minus the seed overlap and the target window, so the floor has to be set on
    the KEPT duration, not the raw horizon.  Returns ``(n_min, t_min_s)``.
    """
    dv = float(np.linalg.norm(np.asarray(v1)[:2] - np.asarray(v0)[:2]))
    dw = float(abs(float(v1[2]) - float(v0[2])))
    t_min = max(float(floor_s), dv / max(a_max, 1e-6), dw / max(alpha_max, 1e-6))
    trim = 2 * mb.nfpt                                     # seed overlap + target window, both discarded
    n = int(np.ceil((t_min * mb.fps + trim) / mb.nfpt))
    return int(np.clip(n, mb.min_tokens, mb.max_tokens)), t_min


def _ramp_waypoints(v0, v1, ramp_dv: float, achieved) -> list:
    """Split a big velocity change into hops, none changing speed by more than ``ramp_dv``.

    Motion Bricks pins only the ENDPOINT of a generation, so nothing asks it for a smooth ramp: measured,
    it coasts near the initial velocity and then converges hard, and the peak acceleration correlates with
    the SIZE of the change (r = +0.50) far more than with the duration it is given (r = +0.26).  Giving it
    more time therefore does not help -- it fills the extra frames by coasting.

    So bound the size instead.  Each hop is a normal 8-frame in-betweening of a small change, which is the
    regime the model already handles well (measured peak 1.70 m/s^2 max for |dv| < 0.8, against 5.08 for
    |dv| in 0.8-1.5).  Intermediate waypoints are velocities, NOT dwells: the robot passes through them
    without ever holding one, so a stand -> run stays a single continuous transition rather than being
    forced to spend time walking.

    Returns ``[(velocity, stride_idx)]``, the last entry being the true destination.
    """
    v0, v1 = np.asarray(v0, np.float32), np.asarray(v1, np.float32)
    dv = float(np.linalg.norm(v1[:2] - v0[:2]))
    k = max(1, int(np.ceil(dv / max(ramp_dv, 1e-6))))
    out = []
    for i in range(1, k + 1):
        w = v0 + (v1 - v0) * (i / k)
        # Nearest bank stride, only to supply a plausible POSE for the waypoint; the velocity is the point.
        j = int(np.argmin(np.linalg.norm(achieved - w[None], axis=1)))
        out.append((w, j))
    return out


def _generate(mb, sb, flat_plan, strides, achieved, bank_dt: float, tr, seed: int, *,
              a_max: float, alpha_max: float, floor_s: float, ramp_dv: float, slack: float):
    """One transient -> ``(qpos[T,36], diag)`` at the MODEL's frame rate, trimmed at both ends.

    A large velocity change is generated as several chained hops (see :func:`_ramp_waypoints`), each
    context-seeded from the previous one's tail so the gait is continuous across them.
    """
    step = mb.dt / bank_dt
    cmd = np.asarray(achieved[tr.stride_in], np.float32)          # where we are going
    v0 = np.asarray(achieved[tr.stride_out] if tr.stride_out >= 0 else np.zeros(3), np.float32)
    way = _ramp_waypoints(v0, cmd, ramp_dv, achieved)

    mb.reset(seed)
    kept_all, toks, mins = [], [], []
    ctx_qpos, v_prev = None, v0
    for j, (w, s_j) in enumerate(way):
        last = (j == len(way) - 1)
        s_use, ph_use = (tr.stride_in, tr.phase_in) if last else (s_j, 0)
        mb.set_target_pose(sb.phase_window(strides[s_use], ph_use, mb.nfpt, step), kind="stride")
        mb.set_regime(1.0, 0.0)                                   # the pose pin stays ON: it is the point
        # A standing start needs real gait initiation (~1 s measured), so it keeps a larger floor than a
        # hop that is already moving.
        f_s = max(floor_s, 1.0) if (j == 0 and tr.stride_out < 0) else floor_s
        n_min, n_max, t_min = mb_env.hop_tokens(mb, v_prev, w, a_max=a_max, alpha_max=alpha_max,
                                               floor_s=f_s, slack=slack)
        allowed = mb_env.allowed_tokens_mask(n_min, mb.min_tokens, mb.max_tokens, n_max=n_max)
        if j == 0 and tr.stride_out < 0:
            # The clip's opening transient: no outgoing loop, so start from the agent's own standing reset.
            n = mb.prime(mb_env.walk_control(mb.context(), w, mb.lookahead_times, standing=True,
                                             allowed_tokens=allowed))
            seed_last = None
        else:
            ctx = (sb.phase_window(strides[tr.stride_out],
                                   flat_plan.transient_context_phase(tr.phase_out, mb.nfpt, step),
                                   mb.nfpt, step) if j == 0 else ctx_qpos)
            seed_ctx = t.as_tensor(np.asarray(ctx, np.float32), dtype=t.float32,
                                   device=mb.agent._device)[None]
            n = mb.prime(mb_env.walk_control(seed_ctx, w, mb.lookahead_times, allowed_tokens=allowed))
            seed_last = np.asarray(ctx[-1], np.float32)
        frames = np.stack([mb.next_frame() for _ in range(int(n))], 0)
        lead = 0 if seed_last is None else _seed_overlap(frames, seed_last, mb.nfpt) + 1
        # The pinned target occupies the LAST nfpt frames.  On the final hop its first frame is the
        # incoming stride's phase_in, which the exact loop supplies itself; on an intermediate hop the
        # next hop re-predicts them from the seed.  Either way, drop them.
        end = frames.shape[0] - mb.nfpt
        if end - lead < 2:
            raise RuntimeError(f"transient {tr.key} hop {j}: horizon {n} leaves {end - lead} frames")
        keep = frames[lead:end]
        kept_all.append(keep)
        ctx_qpos, v_prev = keep[-mb.nfpt:], w
        toks.append(int(n) // mb.nfpt); mins.append(int(n_min))
    out = np.concatenate(kept_all, 0)
    return out, {"tokens": int(np.median(toks)), "horizon": int(sum(toks) * mb.nfpt),
                 "seed_overlap": lead, "hops": len(way), "kept": int(out.shape[0]),
                 "cmd": [float(x) for x in cmd], "n_min": int(min(mins)), "t_min": float(t_min),
                 "accel": float(np.linalg.norm(cmd[:2] - v0[:2]) / max(out.shape[0] / mb.fps, 1e-6))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mb-root", required=True, help="GR00T motionbricks package root.")
    p.add_argument("--stride-bank", required=True, help="Stride bank .npz.")
    p.add_argument("--plans", required=True, help="Flat plans .json (mb_flat_clips.py --dump-plan).")
    p.add_argument("--out-dir", required=True, help="Directory for one RawMotion .npz per transient.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--a-max", type=float, default=2.0,
                   help="Max planar acceleration (m/s^2) a transition may demand. Sets a FLOOR on its "
                        "duration; Motion Bricks still chooses freely above it. Left unbounded it "
                        "produced 5.3 m/s^2 changes.")
    p.add_argument("--alpha-max", type=float, default=2.0,
                   help="Max yaw acceleration (rad/s^2), same role as --a-max.")
    p.add_argument("--ramp-dv", type=float, default=1.0,
                   help="Max planar speed change (m/s) any ONE generation may be asked for. A bigger "
                        "change is split into chained hops through intermediate VELOCITY waypoints (not "
                        "dwells), because peak acceleration tracks the size of the change, not the time "
                        "allowed for it. Set very high to disable.")
    p.add_argument("--hop-slack", type=float, default=1.3,
                   help="How much longer than the acceleration bound requires a hop may take. 1.0 pins "
                        "the duration exactly; higher lets Motion Bricks choose. Caps the horizon from "
                        "ABOVE, which is what stops a multi-hop ramp running long.")
    p.add_argument("--min-transient-s", type=float, default=0.35,
                   help="Absolute floor on a transition's kept duration, so even a tiny velocity change "
                        "gets a plausible gait transition rather than a snap.")
    p.add_argument("--controller", default="random")          # unused; keeps build_demo_args happy
    p.add_argument("--allowed-mode", default=None)
    args = p.parse_args()

    mb_env.enter_mb_root(args, ("stride_bank", "out_dir", "plans"))
    raw_schema = mb_env.load_schema_module("raw_schema")
    sb = mb_env.load_schema_module("stride_bank")
    flat_plan = mb_env.load_schema_module("flat_plan")
    os.makedirs(args.out_dir, exist_ok=True)

    bank = sb.load_stride_bank(args.stride_bank)
    plans = flat_plan.from_json(args.plans)
    mb = mb_env.MBAgent(args)
    print(f"[flat_tr] bank {len(bank)} strides (dt={bank.dt}); {len(plans)} plans; {mb.describe()}")

    todo = [(pl, tr) for pl in plans for tr in pl.transients]
    done = skipped = 0
    tok, kept, overlaps, accels, bound, hops = [], [], [], [], [], []
    for i, (pl, tr) in enumerate(todo):
        out = os.path.join(args.out_dir, f"{tr.key}.npz")
        if os.path.exists(out):
            skipped += 1
            continue
        qpos, diag = _generate(mb, sb, flat_plan, bank.strides, bank.achieved_vel, bank.dt, tr,
                               seed=(args.seed * 2654435761 + i) % (2 ** 31),
                               a_max=args.a_max, alpha_max=args.alpha_max,
                               floor_s=args.min_transient_s, ramp_dv=args.ramp_dv,
                               slack=args.hop_slack)
        tvel = np.tile(np.asarray(diag["cmd"], np.float32), (qpos.shape[0], 1))
        raw_schema.save_raw_motion(out, raw_schema.RawMotion(
            mujoco_qpos=qpos.astype(np.float32), fps=mb.fps, joint_names=list(mb_env.G1_JOINT_NAMES),
            target_vel=tvel, kind="flat_transient",
            provenance={"backend": "motionbricks", "mode": "flat_transient", "clip": pl.name,
                        "key": tr.key, "stride_out": int(tr.stride_out), "phase_out": int(tr.phase_out),
                        "stride_in": int(tr.stride_in), "phase_in": int(tr.phase_in),
                        "stride_bank": os.path.basename(args.stride_bank), **diag}))
        tok.append(diag["tokens"]); kept.append(diag["kept"]); overlaps.append(diag["seed_overlap"])
        accels.append(diag["accel"]); bound.append(diag["tokens"] <= diag["n_min"])
        hops.append(diag["hops"])
        done += 1
        if done % 50 == 0:
            print(f"[flat_tr]   {done}/{len(todo) - skipped} generated")
    print(f"[flat_tr] done: {done} generated, {skipped} already present -> {args.out_dir}")
    if tok:
        u, c = np.unique(overlaps, return_counts=True)
        print(f"[flat_tr]   tokens chosen: p50={np.median(tok):.0f} min={min(tok)} max={max(tok)}"
              f"  (horizon {4*min(tok)}-{4*max(tok)} frames @ {mb.fps:.0f} fps)")
        print(f"[flat_tr]   frames kept: p50={np.median(kept):.0f} min={min(kept)} max={max(kept)}")
        print(f"[flat_tr]   seed overlap dropped: " + ", ".join(f"{a}:{b}" for a, b in zip(u, c))
              + "   (structurally nfpt=%d; a wide spread here means the trim is guessing)" % mb.nfpt)
        ac = np.array(accels)
        print(f"[flat_tr]   implied accel |dv|/T: p50={np.median(ac):.2f} p95={np.percentile(ac,95):.2f} "
              f"max={ac.max():.2f} m/s^2  (bound: {args.a_max})")
        print(f"[flat_tr]   duration floor BOUND on {sum(bound)}/{len(bound)} transients "
              f"-- the rest chose a longer horizon on their own")
        hu, hc = np.unique(hops, return_counts=True)
        print(f"[flat_tr]   hops per transient (--ramp-dv {args.ramp_dv}): "
              + ", ".join(f"{a}:{b}" for a, b in zip(hu, hc)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
