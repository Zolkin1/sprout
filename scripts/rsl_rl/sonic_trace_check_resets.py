#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Check whether the proprioception mismatches in a trace replay line up with env resets.

``sonic_trace_replay.py`` compares our 10-frame proprioception stack against the one their env
built. Their env resets whenever a termination fires -- and with the zero-action trace the robot
falls almost immediately, so it resets repeatedly. On reset their ``CircularBuffer`` warm-starts
(the whole depth is filled with the post-reset sample), while our replay's history keeps rolling,
because the recorder does not capture the done flags. Every reset therefore makes up to ``depth``
subsequent steps incomparable through no fault of the observation code.

This detects resets from the trace itself: a reset teleports the robot onto the reference, which
shows up as a step-to-step jump in joint position or root orientation far larger than 0.02 s of
physics can produce. If the count of affected steps matches the replay's disagreement count, the
mismatch is a limitation of the replay harness rather than of the adapter -- which the production
path does not share, since
:class:`~robot_rl.baselines.sonic.env_bridge.SonicEnvBridge` resets its history on
``episode_length_buf == 0``.

Usage::

    python scripts/rsl_rl/sonic_trace_check_resets.py --trace /tmp/sonic_their_trace.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    """Report reset-like discontinuities and how many compared steps they contaminate."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--depth", type=int, default=10, help="History depth used by the layout.")
    # 0.15 rad/step is NOT enough to separate a teleport from ordinary fast motion: the clip's
    # reset state carries ~5 rad/s of joint velocity, which is 0.1 rad per 0.02 s step, and the
    # first few steps of a zero-action fall measured 0.157 / 0.135 / 0.122 -- a smooth decay, not a
    # discontinuity. Real resets in the same trace jumped 0.79 and 0.84, so 0.4 separates them
    # cleanly. Require BOTH the joint and the quaternion jump, since a teleport moves both.
    p.add_argument("--jump_rad", type=float, default=0.40,
                   help="Joint jump per control step that implies a teleport, not dynamics.")
    p.add_argument("--jump_quat", type=float, default=0.10)
    args = p.parse_args()

    d = np.load(args.trace, allow_pickle=True)
    jp, rq = d["joint_pos"], d["root_quat_w_wxyz"]
    n = jp.shape[0]
    djp = np.abs(np.diff(jp, axis=0)).max(1)
    drq = np.abs(np.diff(rq, axis=0)).max(1)

    print(f"{args.trace}: {n} steps, history depth {args.depth}\n")
    print("step-to-step jumps (a jump means their env reset between those steps):")
    jumps = []
    for k in range(len(djp)):
        is_jump = djp[k] > args.jump_rad and drq[k] > args.jump_quat
        if is_jump:
            jumps.append(k + 1)
        if is_jump or k < 3:
            print(f"  {k:2d}->{k + 1:2d}  d|joint|={djp[k]:.4f}  d|quat|={drq[k]:.4f}"
                  + ("   <== RESET" if is_jump else ""))

    print(f"\nreset-like discontinuities at steps: {jumps}")
    # A reset at step j warm-starts THEIR whole depth at j, while ours still holds pre-reset
    # frames. Ours becomes all-post-reset once its window starts at j, i.e. at step j+depth-1 --
    # so the contaminated steps are j .. j+depth-2 inclusive.
    contaminated = sorted({s for j in jumps for s in range(j, min(j + args.depth - 1, n))})
    compared = [s for s in range(args.depth - 1, n)]
    overlap = [s for s in contaminated if s in compared]
    print(f"steps whose history spans a reset: {len(contaminated)}")
    print(f"of the {len(compared)} steps the replay compares (>= depth-1), "
          f"{len(overlap)} are contaminated  -> {overlap}")
    print("\nIf that count equals the replay's proprio disagreement count, the mismatch is the\n"
          "missing reset signal in the replay harness, not the observation construction --\n"
          "note the agreeing steps match to ~3e-08, which a wrong ordering could not produce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
