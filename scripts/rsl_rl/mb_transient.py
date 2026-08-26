#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""The ONE Motion-Bricks in-betweening generator, shared by the flat and the terrain drivers.

Motion Bricks is an in-betweener in this pipeline, never a locomotion generator: every steady stretch
is an exact looped replay of an optimized stride and the model only fills the transitions between
them. This module is that fill, and it is deliberately the only implementation of it.

It used to be two. ``mb_flat_transients`` had this one; ``mb_terrain_segments`` had
``_generate_approach_bounded`` / ``_generate_exit_bounded``, which are a DIFFERENT job (they pin a
terrain boundary as the target pose rather than a stride phase, and they still live there). But a
terrain clip's approach is also going to contain ordinary loop-to-loop transitions once its command
is allowed to change during the lead-up, and those are this job exactly. Generating them with a
second, single-hop implementation would silently drop the multi-hop chaining below -- which is what
keeps a large deceleration trackable, and is the difference between 1.70 and 5.08 m/s^2 of peak
acceleration.

Extracted VERBATIM from ``mb_flat_transients`` and verified bit-identical against the 47 cached
segments of the quick build, so the hoist moved no motion.
"""

from __future__ import annotations

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
SEED_SEARCH = 8


def seed_overlap(frames: np.ndarray, seed_last: np.ndarray, nfpt: int) -> int:
    """Index of the last frame that still replays the seed: drop ``[0, idx]``, keep from ``idx + 1``."""
    k = min(SEED_SEARCH, frames.shape[0] - 1)
    if k < 1:
        return 0
    d = np.linalg.norm(frames[:k, 7:] - seed_last[None, 7:], axis=1)
    return int(np.argmin(d))


def ramp_waypoints(v0, v1, ramp_dv: float, achieved) -> list:
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


def generate(mb, sb, flat_plan, strides, achieved, bank_dt: float, tr, seed: int, *,
             a_max: float, alpha_max: float, floor_s: float, ramp_dv: float, slack: float,
             min_travel_m: float = 0.0):
    """One transient -> ``(qpos[T,36], diag)`` at the MODEL's frame rate, trimmed at both ends.

    A large velocity change is generated as several chained hops (see :func:`ramp_waypoints`), each
    context-seeded from the previous one's tail so the gait is continuous across them.

    Args:
        tr: a :class:`flat_plan.Transient` -- ``(stride_out, phase_out, stride_in, phase_in, key)``.
        min_travel_m: floor on the DISTANCE the transient covers, passed through to
            :func:`mb_env.hop_tokens`. Zero (the default, and what every flat transient uses) leaves
            the duration bounded by the velocity change alone.
    """
    step = mb.dt / bank_dt
    cmd = np.asarray(achieved[tr.stride_in], np.float32)          # where we are going
    v0 = np.asarray(achieved[tr.stride_out] if tr.stride_out >= 0 else np.zeros(3), np.float32)
    way = ramp_waypoints(v0, cmd, ramp_dv, achieved)

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
        # The travel floor rides on the LAST hop only: it exists to make an entry transient span the
        # approach zone, and that is a property of the whole chain's endpoint, not of each hop.
        n_min, n_max, t_min = mb_env.hop_tokens(
            mb, v_prev, w, a_max=a_max, alpha_max=alpha_max, floor_s=f_s, slack=slack,
            min_travel_m=(float(min_travel_m) if last else 0.0), label=f"{tr.key} hop {j}")
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
        lead = 0 if seed_last is None else seed_overlap(frames, seed_last, mb.nfpt) + 1
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
