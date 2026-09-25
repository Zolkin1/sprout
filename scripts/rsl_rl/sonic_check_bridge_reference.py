#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Validate ``SonicEnvBridge``'s reference sourcing -- the piece the trace-replay oracle skips.

``sonic_trace_replay.py`` proves our observation *packing* is right, but it feeds the adapter a
reference recorded from NVIDIA's env, so it never exercises the path that actually runs in
production: our clip ``.h5`` -> the tracking command term -> the clip-order->Isaac permutation ->
the future-offset window. A fault there would leave the packing check perfectly green while
feeding SONIC a wrong reference.

Two independent checks, both on the encoder input the bridge really used
(``debug_episodes.py --dump_sonic_obs``):

**1. Cross-validation against their env.** The reference joint blocks depend only on the clip
PHASE, not on the robot state, so our window at clip step 0 must equal theirs at clip step 0 --
even though the two rollouts diverge dynamically. The two sides reach those numbers by completely
separate routes (ours: clip .h5 -> command term -> name-based permutation; theirs: our converted
.pkl -> their motion_lib FK -> their MuJoCo->IsaacLab permutation), so agreement validates the
converter and the bridge together.

**2. Self-consistency of the horizon.** With offsets ``[0, 5, 10, ...]``, future frame 1 at step k
is the same clip frame as future frame 0 at step k+5. If the window does not advance -- e.g. every
frame collapsed onto the current one -- SONIC sees a constant posture over its whole 1 s lookahead,
has no gait to track, and would hold a pose without locomoting.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
      python scripts/rsl_rl/sonic_check_bridge_reference.py \\
        --dump /tmp/sonic_dump.npz --trace /tmp/sonic_their_trace.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np

NJ = 29
# Offsets into the 1762-wide encoder input (validated against the released graph).
JP = (4, 294)          # reference joint positions, 10 frames x 29, frame-major
JV = (294, 584)        # reference joint velocities


def main() -> int:
    """Run both checks and report."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="Bundle from --dump_sonic_obs (our env).")
    p.add_argument("--trace", default=None, help="Trace from their env (enables check 1).")
    p.add_argument("--env", type=int, default=0, help="Captured env index to inspect.")
    p.add_argument("--stride", type=int, default=5, help="Reference frame stride.")
    p.add_argument("--tol", type=float, default=2e-5)
    args = p.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    if "sonic_enc" not in d:
        raise SystemExit(f"{args.dump} has no 'sonic_enc' -- record it with --dump_sonic_obs.")
    enc = d["sonic_enc"][:, args.env]                       # [T, 1762]
    steps = d["sonic_clip_step"][:, args.env]               # [T]
    T = enc.shape[0]
    jp = enc[:, JP[0]:JP[1]].reshape(T, 10, NJ)
    jv = enc[:, JV[0]:JV[1]].reshape(T, 10, NJ)
    print(f"{args.dump}: {T} steps, env {args.env}, clip steps {steps[:8].tolist()} ...")
    print(f"  reference amplitude: joint pos std over the horizon "
          f"{jp.std(axis=1).mean():.4f} rad, over time {jp[:, 0].std(axis=0).mean():.4f} rad")

    # --- check 2: does the horizon advance? -----------------------------------------------------
    ok2 = True
    print(f"\n=== horizon self-consistency (frame 1 at k == frame 0 at k+{args.stride}) ===")
    for k in range(min(5, T - args.stride)):
        dif = float(np.abs(jp[k, 1] - jp[k + args.stride, 0]).max())
        good = dif <= args.tol
        ok2 &= good
        print(f"  k={k:2d}: max|diff| {dif:.3e}  {'OK' if good else 'MISMATCH'}")
    flat = float(np.abs(jp[0, 1:] - jp[0, :1]).max())
    print(f"  horizon is {'CONSTANT (BUG: window never advances)' if flat < 1e-9 else 'varying'}"
          f" -- max|frame_i - frame_0| at step 0 = {flat:.4f} rad")

    # --- check 1: cross-validate against their env ----------------------------------------------
    ok1 = None
    if args.trace:
        t = np.load(args.trace, allow_pickle=True)
        if "their_tok_flat" not in t:
            raise SystemExit(f"{args.trace} has no 'their_tok_flat'.")
        their = np.concatenate([np.zeros((t["their_tok_flat"].shape[0], 1), np.float32),
                                t["their_tok_flat"]], axis=1)      # prepend the mode-id scalar
        # Pair on clip PHASE, not by searching for the best-matching step. Both rollouts start at
        # clip frame 0, so our step at min(clip_step) is the counterpart of their step 0. Do NOT
        # "phase-align by minimising the residual": that picks whichever step happens to agree and
        # so reports success even when the arrangement is wrong -- it would have hidden the real
        # bug this check exists to catch (the bridge double-permuted the reference; same values,
        # wrong slots, 1.32 rad here, SONIC commanding a ~1 rad elbow error from step 1).
        ours_idx = int(np.argmin(np.abs(steps - steps.min())))
        print(f"\n=== cross-validation vs their env (our step {ours_idx}, clip step "
              f"{int(steps[ours_idx])}  vs  their step 0) ===")
        for name, (a, b) in (("ref joint pos", JP), ("ref joint vel", JV)):
            o = enc[ours_idx, a:b].astype(np.float64)
            h = their[0, a:b].astype(np.float64)
            dif = float(np.abs(o - h).max())
            good = dif <= args.tol
            ok1 = good if ok1 is None else (ok1 and good)
            worst = int(np.argmax(np.abs(o - h)))
            print(f"  {name:16s} max|diff| {dif:.4e}  {'OK' if good else 'MISMATCH'}"
                  + ("" if good else f"   worst idx {worst} (frame {worst // NJ}, joint "
                                    f"{worst % NJ}): ours {o[worst]:+.4f} theirs {h[worst]:+.4f}"))
            if not good:
                # Distinguish "wrong values" from "right values, wrong slots". A permutation fault
                # leaves every value present and only moves it, so a magnitude-based check reads
                # as healthy -- name it explicitly, per frame, and solve for the permutation.
                of, hf = o.reshape(-1, NJ), h.reshape(-1, NJ)
                same_set = np.allclose(np.sort(of, axis=1), np.sort(hf, axis=1), atol=args.tol)
                print(f"  {'':16s} -> values as a multiset per frame: "
                      f"{'IDENTICAL => PERMUTATION fault, not a value fault' if same_set else 'differ => genuine value fault'}")
                if same_set:
                    pi = [int(np.flatnonzero(np.abs(of[0] - hf[0, j]) < args.tol)[0])
                          for j in range(NJ)
                          if len(np.flatnonzero(np.abs(of[0] - hf[0, j]) < args.tol))]
                    if len(pi) == NJ and sorted(pi) == list(range(NJ)):
                        print(f"  {'':16s} -> ours[pi] == theirs with pi = {pi}")

    print("\n================ VERDICT ================")
    print(f"  horizon advances correctly : {'YES' if ok2 else 'NO'}")
    if ok1 is not None:
        print(f"  matches their env at phase 0: {'YES' if ok1 else 'NO'}")
        if not ok1:
            print("\n  => the bridge's reference sourcing (or the clip converter) is wrong;\n"
                  "     the worst index above names the frame and joint.")
    return 0 if (ok2 and ok1 is not False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
