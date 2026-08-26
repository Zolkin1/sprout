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

The generation itself lives in :mod:`mb_transient`, shared with the terrain driver -- a terrain
clip's lead-up contains ordinary loop-to-loop transitions too, and a second implementation of
them would silently lack the multi-hop chaining that keeps a large change trackable.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_env  # noqa: E402
import mb_transient  # noqa: E402


# The generation knobs a cached flat transient is keyed on. Mirrors ``mb_terrain_segments._GEN_KNOBS``
# and exists for the same reason, found the same way: this stage was a bare ``os.path.exists`` skip, so
# a change to what the generator PRODUCES left every already-generated segment in place. Measured on the
# quick build's 47 cached transients -- 38 of them differ in SHAPE from what the same command produces
# today, and nothing said so. A silently stale segment is worse than a slow rebuild: the plan, the clip
# and every QC number are consistent with motion that no current code path would generate.
_GEN_KNOBS = ("gen_version", "stride_bank", "a_max", "alpha_max", "min_transient_s", "ramp_dv",
              "hop_slack")

# Bump when the GENERATOR changes what it produces for identical knobs -- a CLI flag is comparable, code
# is not. Compared like any other knob, so an invalidation is one greppable line.
#   1: the first version to record any of this. Everything cached before it has no ``gen_version`` and
#      is therefore stale by construction, which is correct: those files predate the token UPPER bound
#      (``hop_tokens``' ``n_max`` / ``--hop-slack``), so they are longer than anything generated now.
_GEN_VERSION = 1


def _gen_params(args) -> dict:
    """The generation knobs this run would use. ``stride_bank`` by BASENAME, as on the terrain side:
    the driver absolutizes the path, so comparing the full one would invalidate everything after a
    work-dir move while the basename still catches the case that matters (a different bank)."""
    return {"gen_version": _GEN_VERSION, "stride_bank": os.path.basename(args.stride_bank),
            "a_max": float(args.a_max), "alpha_max": float(args.alpha_max),
            "min_transient_s": float(args.min_transient_s), "ramp_dv": float(args.ramp_dv),
            "hop_slack": float(args.hop_slack)}


def _is_stale(path: str, want: dict, raw_schema) -> str | None:
    """``None`` if the transient on disk was generated with ``want``; else the first differing knob."""
    try:
        prov = dict(raw_schema.load_raw_motion(path).provenance or {})
    except Exception:
        return "unreadable"
    for k in _GEN_KNOBS:
        if k not in prov:
            return f"{k} (not recorded)"
        a, b = prov[k], want[k]
        if isinstance(b, float) and isinstance(a, (int, float)):
            if abs(float(a) - b) > 1e-9:
                return f"{k}: {a} != {b}"
        elif a != b:
            return f"{k}: {a!r} != {b!r}"
    return None


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
    want = _gen_params(args)
    done = skipped = regen = 0
    stale_why: dict[str, int] = {}
    tok, kept, overlaps, accels, bound, hops = [], [], [], [], [], []
    for i, (pl, tr) in enumerate(todo):
        out = os.path.join(args.out_dir, f"{tr.key}.npz")
        if os.path.exists(out):
            why = _is_stale(out, want, raw_schema)
            if why is None:
                skipped += 1
                continue
            # Named rather than counted: "38 stale" says a rebuild happened, "gen_version (not
            # recorded)" says WHY, which is the difference between a knob you moved on purpose and a
            # cache you did not know was old.
            stale_why[why.split(":")[0]] = stale_why.get(why.split(":")[0], 0) + 1
            regen += 1
        qpos, diag = mb_transient.generate(
            mb, sb, flat_plan, bank.strides, bank.achieved_vel, bank.dt, tr,
            seed=(args.seed * 2654435761 + i) % (2 ** 31),
            a_max=args.a_max, alpha_max=args.alpha_max,
            floor_s=args.min_transient_s, ramp_dv=args.ramp_dv, slack=args.hop_slack)
        tvel = np.tile(np.asarray(diag["cmd"], np.float32), (qpos.shape[0], 1))
        raw_schema.save_raw_motion(out, raw_schema.RawMotion(
            mujoco_qpos=qpos.astype(np.float32), fps=mb.fps, joint_names=list(mb_env.G1_JOINT_NAMES),
            target_vel=tvel, kind="flat_transient",
            provenance={"backend": "motionbricks", "mode": "flat_transient", "clip": pl.name,
                        "key": tr.key, "stride_out": int(tr.stride_out), "phase_out": int(tr.phase_out),
                        "stride_in": int(tr.stride_in), "phase_in": int(tr.phase_in),
                        **want, **diag}))
        tok.append(diag["tokens"]); kept.append(diag["kept"]); overlaps.append(diag["seed_overlap"])
        accels.append(diag["accel"]); bound.append(diag["tokens"] <= diag["n_min"])
        hops.append(diag["hops"])
        done += 1
        if done % 50 == 0:
            print(f"[flat_tr]   {done}/{len(todo) - skipped} generated")
    print(f"[flat_tr] done: {done} generated ({regen} of them REGENERATED as stale), "
          f"{skipped} reused -> {args.out_dir}")
    if stale_why:
        print("[flat_tr]   stale because: "
              + ", ".join(f"{k} x{n}" for k, n in sorted(stale_why.items())))
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
