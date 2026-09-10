#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Drive the STOCK Motion Bricks stack from one of our clips' velocity commands, and compare.

The paper claim is that optimized strides for steady state plus Motion Bricks for the transitions
tracks a velocity command better than Motion Bricks alone. This is the "alone" arm: the same command
profile, the same robot, the same dt, driven the way Motion Bricks itself drives -- and then scored
against our clip on the identical measurement.

**FLAT ONLY.** On terrain our reference deviates from the command on purpose (the command is
terrain-agnostic and the motion is not), so a tracking comparison there would score a design decision
as an error.

Four choices decide whether this is a fair fight, and each is made the way the SHIPPED stack makes it
rather than the way ours does:

* **The root is FREE** (``has_specific_target = 0``). ``scripts/interactive_demo_g1.py`` sends a
  movement direction, a facing direction and a mode -- and no ``specific_target_positions`` at all.
  Our terrain path pins a target position; imposing that here would be testing our harness, not
  Motion Bricks.
* **The horizon is the MODEL'S** (``allowed_tokens=None``). Motion Bricks picks 6..16 tokens from its
  own logit head. We mask that head on the terrain path; masking it here would hand the baseline a
  handicap we invented.
* **The command is OURS, unchanged.** Both arms see the identical ``target_vel`` profile, taken from
  a flat clip -- whose commands are already drawn continuously from the driver envelope, so this is
  the envelope a planner would send rather than a set of speeds either stack is good at.
* **One re-plan per command segment.** The clip's command is piecewise constant by construction, so
  a segment boundary is the only place a re-plan is implied. Re-planning FASTER would manufacture
  Motion Bricks' pin-law overshoot (``flat_plan.pin_travel``: the target sits ``|v| * 1.0 s`` ahead
  regardless of the horizon chosen, so a transient given less time must cover it faster) and would
  make the comparison a statement about our chunking.

THREE stages, and the split is forced rather than tidy: **the motionbricks conda env has no
h5py**, the same constraint the deploy image has, so the command profile crosses the env boundary
as an ``.npz`` exactly like a ``TerrainPlan`` does::

    # 1. EXPORT -- env_isaaclab, reads the .h5
    python scripts/rsl_rl/mb_baseline.py --stage export \\
        --clip clips/g1_mb_v6f_decor.h5 --clip-index 7000 --out /tmp/baseline/profile.npz

    # 2. RUN -- the motionbricks conda env, GPU
    conda run --no-capture-output -n motionbricks python -u scripts/rsl_rl/mb_baseline.py \\
        --stage run --mb-root <mb> --profile /tmp/baseline/profile.npz --out /tmp/baseline/run.npz

    # 3. COMPARE -- env_isaaclab, no GPU
    python scripts/rsl_rl/mb_baseline.py --stage compare \\
        --clip clips/g1_mb_v6f_decor.h5 --clip-index 7000 --run /tmp/baseline/run.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------- the command


def command_segments(cmd: np.ndarray, dt: float) -> list[tuple[np.ndarray, int]]:
    """``[(cmd[3], n_frames)]`` -- the piecewise-constant runs of a clip's ``target_vel``.

    The clip's command changes only on piece boundaries (``mb_flat_clips.assemble_world`` writes one
    command per spliced piece), so this recovers exactly the segments the plan authored rather than
    imposing a chunking of its own.
    """
    key = np.round(np.asarray(cmd, np.float64), 6)
    edges = [0] + (np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1).tolist() + [len(key)]
    return [(key[a], int(b - a)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def read_clip(path: str, index: int) -> dict:
    """The one clip we are comparing against: its command, its achieved root motion, its dt."""
    import h5py
    with h5py.File(path, "r") as f:
        names = json.loads(f.attrs["pos_output_names"])
        n = int(f["clip_len"][index])
        y = np.asarray(f["y_des"][index, :n, :])
        cmd = np.asarray(f["target_vel"][index, :n, :])
        tid = int(f["terrain_id"][index])
        dt = float(f.attrs.get("dt", 0.02))
    pos = y[:, [names.index(f"pelvis_link:pos_{c}") for c in "xyz"]]
    quat = y[:, [names.index(f"pelvis_link:ori_{c}") for c in "wxyz"]]
    return {"pos": pos, "quat": quat, "cmd": cmd, "dt": dt, "terrain_id": tid, "n": n,
            "names": names, "y": y}


# --------------------------------------------------------------------------------- measurement


def body_velocity(pos: np.ndarray, quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """``[T-1, 3]`` body-frame ``(vx, vy, wz)`` from a root trajectory.

    Heading-frame, matching how the solver and ``terrain_ingest._measured_steady_vel`` define a
    command -- so "achieved" means the same thing for both arms and for the library that trained
    them. A world-frame difference would score a turning clip as lateral motion.
    """
    d = (pos[1:, :2] - pos[:-1, :2]) / float(dt)
    w, x, y, z = (quat_wxyz[:, i] for i in range(4))
    psi = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(psi[:-1]), np.sin(psi[:-1])
    dpsi = np.arctan2(np.sin(psi[1:] - psi[:-1]), np.cos(psi[1:] - psi[:-1])) / float(dt)
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c, dpsi], axis=1)


def boxcar(v: np.ndarray, dt: float, window_s: float) -> np.ndarray:
    """Centred moving average over ``window_s``, applied per channel.

    Instantaneous root velocity oscillates within every stride, so an unsmoothed error is mostly
    GAIT RIPPLE and not tracking -- and the two arms have different ripple, so the raw number would
    rank them on gait shape. A fixed-time boxcar is deliberately method-agnostic: the alternative,
    averaging over each arm's own gait CYCLE, needs a stride index that Motion Bricks does not have.
    """
    n = max(1, int(round(window_s / dt)))
    if n <= 1 or len(v) < n:
        return v
    k = np.ones(n) / n
    pad = n // 2
    out = np.empty_like(v)
    for i in range(v.shape[1]):
        e = np.pad(v[:, i], (pad, n - 1 - pad), mode="edge")
        out[:, i] = np.convolve(e, k, mode="valid")[:len(v)]
    return out


def tracking_stats(achieved: np.ndarray, commanded: np.ndarray, dt: float,
                   settle_s: float = 0.4) -> dict:
    """Per-channel tracking error, excluding the first ``settle_s`` after each command change.

    The exclusion is not a courtesy to either arm -- it is what makes the number mean "tracking"
    rather than "how fast can you change speed". A ramp is a transient by definition and both stacks
    are entitled to one; scoring it would mostly measure the acceleration bound, which is a declared
    parameter here and a learned property there.
    """
    n = min(len(achieved), len(commanded))
    a, c = achieved[:n], commanded[:n]
    chg = np.flatnonzero(np.any(np.round(c[1:], 6) != np.round(c[:-1], 6), axis=1)) + 1
    settled = np.ones(n, bool)
    settled[:int(settle_s / dt)] = False
    for k in chg:
        settled[k:k + int(settle_s / dt)] = False
    out: dict = {"n_frames": int(n), "n_settled": int(settled.sum()),
                 "n_segments": int(len(chg) + 1)}
    if not settled.any():
        return out
    for i, ch in enumerate(("vx", "vy", "wz")):
        e = a[settled, i] - c[settled, i]
        out[f"{ch}_rmse"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{ch}_mae"] = float(np.mean(np.abs(e)))
        out[f"{ch}_bias"] = float(np.mean(e))
    planar = np.linalg.norm(a[settled, :2] - c[settled, :2], axis=1)
    out["planar_rmse"] = float(np.sqrt(np.mean(planar ** 2)))
    out["planar_p95"] = float(np.percentile(planar, 95))
    # PER SEGMENT too, because a pooled number over one clip conflates two different things: a
    # command the solved library can serve (where the error is tracking) and one it cannot (where
    # the error is the designed snap -- e.g. a diagonal at jog speed, which has no solved cell at
    # all). Only the first kind is a fair comparison; the second says more about library coverage.
    bounds = [0] + chg.tolist() + [n]
    out["segments"] = []
    for a0, b0 in zip(bounds[:-1], bounds[1:]):
        m = settled[a0:b0]
        if not m.any():
            continue
        e = np.linalg.norm(a[a0:b0][m, :2] - c[a0:b0][m, :2], axis=1)
        out["segments"].append({"cmd": [float(x) for x in c[a0]], "frames": int(b0 - a0),
                                "planar_rmse": float(np.sqrt(np.mean(e ** 2)))})
    return out


# --------------------------------------------------------------------------------- stage: run


def stage_export(args) -> int:
    """Write the clip's command profile where the motionbricks env can read it (no h5py there)."""
    clip = read_clip(args.clip, args.clip_index)
    if clip["terrain_id"] >= 0:
        raise SystemExit(f"clip {args.clip_index} sits on terrain {clip['terrain_id']}; this "
                         f"comparison is FLAT-only (see the module docstring)")
    segs = command_segments(clip["cmd"], clip["dt"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out,
                        cmd=np.asarray([c for c, _ in segs], np.float64),
                        frames=np.asarray([n for _, n in segs], np.int64),
                        dt=np.float64(clip["dt"]),
                        meta=json.dumps({"clip": args.clip, "clip_index": args.clip_index,
                                         "n_frames": clip["n"]}))
    print(f"[mb_base] wrote {args.out}: {len(segs)} segments, {clip['n']} frames "
          f"({clip['n'] * clip['dt']:.1f} s)")
    return 0


def stage_run(args) -> int:
    """Drive Motion Bricks through the exported command profile. Needs the motionbricks env."""
    sys.path.insert(0, _HERE)
    import mb_env

    # Load BEFORE enter_mb_root: it chdir's into the Motion Bricks root (whose configs reference
    # `out/` and `assets/` relatively), so a repo-relative --profile would stop resolving.
    d = np.load(args.profile, allow_pickle=True)
    clip = {"dt": float(d["dt"])}
    segs = [(np.asarray(c, np.float64), int(n)) for c, n in zip(d["cmd"], d["frames"])]
    print(f"[mb_base] profile: {len(segs)} command segments, "
          f"{int(sum(n for _, n in segs))} frames")

    mb_env.enter_mb_root(args, ("out",))
    # pose_override=False: the agent AS SHIPPED. Our override exists to drive a pinned target
    # pose, which is the half of our stack this comparison is meant to be without.
    mb = mb_env.MBAgent(args, pose_override=False)
    print(f"[mb_base] {mb.describe() if hasattr(mb, 'describe') else ''}")
    mb.reset(args.seed)
    frames: list[np.ndarray] = []
    plan_log = []
    for i, (cmd, n_clip) in enumerate(segs):
        # The profile counts CLIP frames at the clip's dt (50 Hz); Motion Bricks runs at its own
        # fps (30). Generating n_clip frames here would run every segment fps*dt = 1.67x too long
        # and the comparison would be against a different command profile entirely.
        want_s = n_clip * clip["dt"]
        n_want = int(round(want_s * mb.fps))
        got = 0
        # One re-plan per command segment, then let the model's own horizon carry it. If a horizon
        # ends before the segment does we re-prime with the SAME command -- that is the model asking
        # for another plan, not us chunking it.
        while got < n_want:
            n_new = int(mb.prime(mb_env.walk_control(
                mb.context(), cmd, mb.lookahead_times,
                free_root=(args.root_mode == "free"))))
            take = min(n_new, n_want - got)
            for _ in range(take):
                frames.append(mb.next_frame())
            for _ in range(max(0, n_new - take)):
                mb.next_frame()          # burn the tail so the context stays consistent
            got += take
            plan_log.append({"segment": i, "cmd": [float(v) for v in cmd],
                             "tokens_frames": int(n_new), "took": int(take)})
        print(f"[mb_base]   seg {i:3d} cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
              f"{want_s:.2f}s -> {got} mb frames ({got / mb.fps:.2f}s)", flush=True)

    q = np.asarray(frames, np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, qpos=q, fps=np.float32(mb.fps), clip_dt=np.float32(clip["dt"]),
                        meta=json.dumps({"profile": args.profile, "seed": args.seed,
                                         "root_mode": args.root_mode, "allowed_tokens": None,
                                         "plans": plan_log}))
    print(f"[mb_base] wrote {args.out}: {q.shape[0]} frames at {mb.fps} fps")
    return 0


# --------------------------------------------------------------------------------- stage: compare


def stage_compare(args) -> int:
    """Score both arms on the identical measurement."""
    clip = read_clip(args.clip, args.clip_index)
    d = np.load(args.run, allow_pickle=True)
    q, fps = np.asarray(d["qpos"]), float(d["fps"])
    dt_mb = 1.0 / fps

    # OURS: pelvis root motion out of the clip.
    ours = body_velocity(clip["pos"], clip["quat"], clip["dt"])
    # BASELINE: MB's own root, resampled onto the clip's grid so both are scored at one rate.
    t_mb = np.arange(len(q)) * dt_mb
    t_cl = np.arange(clip["n"]) * clip["dt"]
    keep = t_cl <= t_mb[-1] + 1e-9
    pos = np.stack([np.interp(t_cl[keep], t_mb, q[:, i]) for i in range(3)], axis=1)
    quat = np.stack([np.interp(t_cl[keep], t_mb, q[:, 3 + i]) for i in range(4)], axis=1)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    base = body_velocity(pos, quat, clip["dt"])

    cmd = clip["cmd"]
    ours = boxcar(ours, clip["dt"], args.smooth_s)
    base = boxcar(base, clip["dt"], args.smooth_s)
    so = tracking_stats(ours, cmd, clip["dt"])
    sb = tracking_stats(base, cmd[:len(base) + 1], clip["dt"])
    print(f"clip {args.clip_index} of {args.clip}   {clip['n']} frames, "
          f"{so['n_segments']} command segments")
    print(f"baseline covered {len(base)} of {len(ours)} frames "
          f"({len(base) / max(len(ours), 1):.0%} of the clip)\n")
    print(f"{'metric':>14} {'OURS':>10} {'MB alone':>10}   better")
    for k in ("vx_rmse", "vy_rmse", "wz_rmse", "planar_rmse", "planar_p95",
              "vx_bias", "vy_bias", "wz_bias"):
        if k not in so or k not in sb:
            continue
        a, b = so[k], sb[k]
        win = "ours" if abs(a) < abs(b) else "MB"
        print(f"{k:>14} {a:10.4f} {b:10.4f}   {win}")
    print(f"\nsettled frames scored: ours {so['n_settled']}, baseline {sb['n_settled']} "
          f"(the first 0.4 s after every command change is excluded from BOTH; "
          f"velocity boxcar {args.smooth_s:.1f} s)")
    print(f"\n{'per-segment planar RMSE':>26}   {'OURS':>8} {'MB alone':>9}")
    for a_, b_ in zip(so.get("segments", []), sb.get("segments", [])):
        c_ = a_["cmd"]
        print(f"  ({c_[0]:+.2f},{c_[1]:+.2f},{c_[2]:+.2f}) {a_['frames']:4d} f   "
              f"{a_['planar_rmse']:8.4f} {b_['planar_rmse']:9.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ours": so, "baseline": sb, "clip": args.clip,
                       "clip_index": args.clip_index}, f, indent=1)
        print(f"wrote {args.json}")
    return 0


def main() -> int:
    """Parse args and run a stage."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["export", "run", "compare"])
    p.add_argument("--clip", default=None, help="Clip library holding the reference clip.")
    p.add_argument("--clip-index", type=int, default=0)
    p.add_argument("--profile", default=None, help="run: the .npz written by --stage export")
    p.add_argument("--out", default=None, help="export / run: output .npz")
    p.add_argument("--run", default=None, help="compare: the .npz written by --stage run")
    p.add_argument("--json", default=None, help="compare: also write the stats here")
    p.add_argument("--mb-root", default=None, help="run: Motion Bricks package root (ABSOLUTE)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--root-mode", default="free", choices=["free", "rollout"],
                   help="free: Motion Bricks owns the root, driven by movement_direction + "
                        "target_vel -- the shipped demo's mode, and the honest 'MB alone'. "
                        "rollout: the root is pinned to the exact-velocity integration of the "
                        "command, which tracks well BY CONSTRUCTION and is therefore an UPPER "
                        "BOUND on MB rather than a baseline -- report both, since the demo has no "
                        "speed input and picking one configuration would be picking a result.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smooth-s", type=float, default=1.0,
                   help="compare: boxcar window on the achieved velocity (s). 0 disables it and "
                        "scores gait ripple as tracking error.")
    args = p.parse_args()
    if args.stage == "export" and not (args.clip and args.out):
        raise SystemExit("--stage export needs --clip and --out")
    if args.stage == "run" and not (args.out and args.mb_root and args.profile):
        raise SystemExit("--stage run needs --profile, --out and --mb-root")
    if args.stage == "compare" and not (args.run and args.clip):
        raise SystemExit("--stage compare needs --clip and --run")
    return {"export": stage_export, "run": stage_run, "compare": stage_compare}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
