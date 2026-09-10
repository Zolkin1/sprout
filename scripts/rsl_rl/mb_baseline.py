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

FIVE stages -- ``export`` / ``run`` / ``compare`` / ``view`` / ``batch``, the last being the one
to reach for (it drives the other four over N clips and writes the paper table). The split is
forced rather than tidy: **the motionbricks conda env has no
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

#: Seconds after a command change that the SECONDARY, settled-only view ignores.
#:
#: The headline number includes the transient, deliberately: reaching a commanded speed promptly is
#: part of tracking it, and excluding the ramp forgives a sluggish stack for exactly the failure a
#: velocity command is meant to expose. But the two effects are worth separating -- a stack can lose
#: on responsiveness while matching on steady-state accuracy, and that is a different story -- so
#: both are reported side by side.
#:
#: 1.5 s is MEASURED rather than chosen: ``verify_command_tracking.py`` records that at 0.6 s a
#: correctly-tracking approach still reads 0.725 m/s of "error" while at 1.5 s it reads 0.188. A
#: command segment shorter than this contributes nothing to the settled view, which is why that
#: view is secondary and not the headline.
SETTLE_S = 1.5


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
                   settle_s: float = SETTLE_S) -> dict:
    """Per-channel tracking error, excluding the first ``settle_s`` after each command change.

    This is the SETTLED view -- steady-state accuracy once both stacks have had their transient.
    It is deliberately NOT the headline: how fast a stack reaches a commanded speed is part of
    tracking that command, not a courtesy owed to it, so the reported number includes the ramp.
    Pass ``settle_s=0.0`` for that all-frames view; ``stage_compare`` prints both side by side and
    ``bucket_mse`` returns both.

    What the settled view is still FOR: it separates "slow to respond" from "wrong once settled",
    which the all-frames number pools into one figure. A command segment shorter than ``settle_s``
    contributes nothing to it, which is the other reason it cannot be the headline.
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


def stage_export(args: argparse.Namespace) -> int:
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


def stage_run(args: argparse.Namespace) -> int:
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


def stage_compare(args: argparse.Namespace) -> int:
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
    # BOTH views, in the same order the batch table prints them -- a single-clip debug run that
    # disagreed with the paper table about which number is the headline was its own trap.
    so_all = tracking_stats(ours, cmd, clip["dt"], settle_s=0.0)
    sb_all = tracking_stats(base, cmd[:len(base) + 1], clip["dt"], settle_s=0.0)
    so = tracking_stats(ours, cmd, clip["dt"], settle_s=args.settle_s)
    sb = tracking_stats(base, cmd[:len(base) + 1], clip["dt"], settle_s=args.settle_s)
    print(f"clip {args.clip_index} of {args.clip}   {clip['n']} frames, "
          f"{so['n_segments']} command segments")
    print(f"baseline covered {len(base)} of {len(ours)} frames "
          f"({len(base) / max(len(ours), 1):.0%} of the clip)\n")
    keys = ("vx_rmse", "vy_rmse", "wz_rmse", "planar_rmse", "planar_p95",
            "vx_bias", "vy_bias", "wz_bias")
    for label, a_s, b_s in (("ALL FRAMES (headline)", so_all, sb_all),
                            (f"SETTLED ONLY (first {args.settle_s} s after each change dropped)",
                             so, sb)):
        print(f"### {label}")
        print(f"{'metric':>14} {'OURS':>10} {'MB alone':>10}   better")
        for k in keys:
            if k not in a_s or k not in b_s:
                continue
            a, b = a_s[k], b_s[k]
            win = "ours" if abs(a) < abs(b) else "MB"
            print(f"{k:>14} {a:10.4f} {b:10.4f}   {win}")
        print()
    print(f"settled frames scored: ours {so['n_settled']}, baseline {sb['n_settled']} "
          f"(the first {args.settle_s} s after every command change is excluded from BOTH; "
          f"velocity boxcar {args.smooth_s:.1f} s)")
    print(f"\n{'per-segment planar RMSE':>26}   {'OURS':>8} {'MB alone':>9}")
    for a_, b_ in zip(so.get("segments", []), sb.get("segments", [])):
        c_ = a_["cmd"]
        print(f"  ({c_[0]:+.2f},{c_[1]:+.2f},{c_[2]:+.2f}) {a_['frames']:4d} f   "
              f"{a_['planar_rmse']:8.4f} {b_['planar_rmse']:9.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ours": so_all, "baseline": sb_all,
                       "ours_settled": so, "baseline_settled": sb,
                       "settle_s": args.settle_s, "smooth_s": args.smooth_s,
                       "clip": args.clip, "clip_index": args.clip_index}, f, indent=1)
        print(f"wrote {args.json}")
    return 0




# --------------------------------------------------------------------------------- stage: view


def _resample_run(run_path: str, dt_clip: float, n_clip: int) -> tuple:
    """The MB run on the CLIP's time grid -> ``(pos[T,3], wxyz[T,4], joints[T,29], n)``.

    Motion Bricks runs at its own fps (30) and the clip at 50 Hz, so one shared frame slider needs
    one grid. Resampling MB onto the clip's is the same choice ``stage_compare`` makes, so the
    picture and the numbers cannot disagree about which frame is which.
    """
    d = np.load(run_path, allow_pickle=True)
    q, fps = np.asarray(d["qpos"]), float(d["fps"])
    t_mb = np.arange(len(q)) / fps
    t_cl = np.arange(n_clip) * dt_clip
    keep = t_cl <= t_mb[-1] + 1e-9
    tt = t_cl[keep]
    pos = np.stack([np.interp(tt, t_mb, q[:, i]) for i in range(3)], axis=1)
    wxyz = np.stack([np.interp(tt, t_mb, q[:, 3 + i]) for i in range(4)], axis=1)
    wxyz /= np.linalg.norm(wxyz, axis=1, keepdims=True)
    joints = np.stack([np.interp(tt, t_mb, q[:, 7 + i]) for i in range(q.shape[1] - 7)], axis=1)
    return pos, wxyz, joints, int(keep.sum())


def _compare_image(cmd: np.ndarray, ours: np.ndarray, base: np.ndarray, dt: float,
                   frame: int, smooth_s: float, ours_raw: np.ndarray | None = None,
                   base_raw: np.ndarray | None = None) -> np.ndarray:
    """Three stacked panels -- vx, vy, wz -- with the command and BOTH arms on each.

    The command is drawn as a step because it IS one (piecewise constant per spliced piece), and
    both achieved traces are boxcar-smoothed identically: unsmoothed they are mostly gait ripple,
    and the two arms have different ripple, so a raw plot would compare gait shape rather than
    tracking. A cursor marks the frame the 3-D scene is showing.

    The RAW traces are drawn faintly underneath, and that is not decoration. Over an exact looped
    stride our smoothed velocity goes EXACTLY constant -- measured on clip 7000, local std 0.0000
    across frames 800-950, all one stride -- because a loop repeats identically and the filter
    annihilates a perfectly periodic ripple. Motion Bricks' ripple is not periodic, so the same
    filter leaves some of it. Reading the smoothed lines alone, ours looks impossibly smooth for a
    reason that is about periodicity rather than tracking; the raw band shows what was averaged.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    n = min(len(ours), len(base), len(cmd) - 1)
    t = np.arange(n) * dt
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True, dpi=110)
    for i, (ax, lab) in enumerate(zip(axes, ("vx (m/s)", "vy (m/s)", "wz (rad/s)"))):
        if ours_raw is not None:
            ax.plot(t, ours_raw[:n, i], color="#1f77b4", lw=0.7, alpha=0.28)
        if base_raw is not None:
            ax.plot(t, base_raw[:n, i], color="#d62728", lw=0.7, alpha=0.28)
        ax.step(t, cmd[:n, i], where="post", color="0.25", lw=1.6, label="commanded")
        ax.plot(t, ours[:n, i], color="#1f77b4", lw=1.7, label="ours")
        ax.plot(t, base[:n, i], color="#d62728", lw=1.7, label="MB alone")
        ax.axvline(min(frame, n - 1) * dt, color="0.6", lw=1.0, ls="--")
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel(f"time (s)   -- bold = boxcar {smooth_s:.1f} s, faint = raw. A flat bold "
                        f"line over a rippling faint one is an EXACT LOOP: perfectly periodic, so "
                        f"the filter removes all of it.", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def stage_view(args: argparse.Namespace) -> int:
    """Both arms in viser: the robots, the world paths, and the tracking plot."""
    import viser
    from viser.extras import ViserUrdf
    sys.path.insert(0, _HERE)
    from robot_rl.flow_matching.viz.robot import (DEFAULT_URDF, load_urdf,
                                                  recording_to_urdf_perm, xyzw_to_wxyz)
    from motion_gen.joint_names import G1_JOINT_NAMES

    repo = os.path.dirname(os.path.dirname(_HERE))
    clip = read_clip(args.clip, args.clip_index)
    names = clip["names"]
    ours_pos = clip["pos"]
    ours_wxyz = xyzw_to_wxyz(clip["y"][:, [names.index(f"pelvis_link:ori_{c}") for c in "xyzw"]])
    jidx = [i for i, nm in enumerate(names) if nm.startswith("joint:")]
    ours_j = clip["y"][:, jidx]
    ours_jn = [names[i].split(":", 1)[1] for i in jidx]
    b_pos, b_wxyz, b_j, n_b = _resample_run(args.run, clip["dt"], clip["n"])

    urdf = load_urdf(os.path.join(repo, DEFAULT_URDF))
    perm_ours = recording_to_urdf_perm(ours_jn, urdf)
    perm_mb = recording_to_urdf_perm(list(G1_JOINT_NAMES), urdf)

    r_ours = body_velocity(ours_pos, ours_wxyz, clip["dt"])
    r_base = body_velocity(b_pos, b_wxyz, clip["dt"])
    v_ours = boxcar(r_ours, clip["dt"], args.smooth_s)
    v_base = boxcar(r_base, clip["dt"], args.smooth_s)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=24.0, height=24.0)
    f_ours = server.scene.add_frame("/ours", show_axes=False)
    u_ours = ViserUrdf(server, urdf, root_node_name="/ours")
    f_mb = server.scene.add_frame("/mb", show_axes=False)
    u_mb = ViserUrdf(server, urdf, root_node_name="/mb")

    def seg(p: np.ndarray, colour: tuple) -> None:
        """Draw one arm's root path as a viser line strip."""
        pts = np.stack([p[:-1, :3], p[1:, :3]], axis=1).astype(np.float32)
        return server.scene.add_line_segments("/path/" + colour[0], points=pts,
                                              colors=colour[1], line_width=4.0)
    seg(ours_pos, ("ours", (0.12, 0.47, 0.71)))
    seg(b_pos, ("mb", (0.84, 0.15, 0.16)))

    gui = server.gui
    t_sl = gui.add_slider("Frame", min=0, max=min(len(ours_pos), n_b) - 1, step=1, initial_value=0)
    overlay = gui.add_checkbox("Overlay at the origin (compare GAIT, not path)", False)
    info = gui.add_markdown("")
    plot = {"h": None}

    def redraw_plot(frame: int) -> None:
        """Re-render the velocity panel with the cursor at ``frame``."""
        if plot["h"] is not None:
            plot["h"].remove()
        img = _compare_image(clip["cmd"], v_ours, v_base, clip["dt"], frame, args.smooth_s,
                             ours_raw=r_ours, base_raw=r_base)
        plot["h"] = (gui.add_image(img, label="body velocity: commanded vs ours vs MB alone")
                     if img is not None else None)

    def draw(t: int) -> None:
        """Place both robots at frame ``t`` and refresh the panel and readout."""
        t = int(np.clip(t, 0, min(len(ours_pos), n_b) - 1))
        o, m = ours_pos[t].copy(), b_pos[t].copy()
        if overlay.value:                      # subtract each arm's own start, so gaits superpose
            o = o - ours_pos[0] + np.array([0.0, 0.0, ours_pos[0, 2]])
            m = m - b_pos[0] + np.array([0.0, 1.0, b_pos[0, 2]])
        f_ours.position, f_ours.wxyz = tuple(float(v) for v in o), tuple(float(v) for v in ours_wxyz[t])
        f_mb.position, f_mb.wxyz = tuple(float(v) for v in m), tuple(float(v) for v in b_wxyz[t])
        u_ours.update_cfg(np.asarray(ours_j[t])[perm_ours])
        u_mb.update_cfg(np.asarray(b_j[t])[perm_mb])
        k = min(t, len(v_ours) - 1, len(v_base) - 1)
        c = clip["cmd"][t]
        info.content = (
            f"### frame {t}  ({t * clip['dt']:.2f} s)\n\n"
            f"| | vx | vy | wz |\n|---|---|---|---|\n"
            f"| commanded | {c[0]:+.3f} | {c[1]:+.3f} | {c[2]:+.3f} |\n"
            f"| **ours** | {v_ours[k,0]:+.3f} | {v_ours[k,1]:+.3f} | {v_ours[k,2]:+.3f} |\n"
            f"| **MB alone** | {v_base[k,0]:+.3f} | {v_base[k,1]:+.3f} | {v_base[k,2]:+.3f} |\n\n"
            f"blue = ours, red = MB alone. Paths are the true world trajectories, so they diverge "
            f"-- that divergence IS the tracking error accumulated. Tick the box to superpose them "
            f"and compare gait instead.")

    @t_sl.on_update
    def _(_e: object) -> None:
        """Viser callback: refresh the view when this control changes."""
        draw(int(t_sl.value))

    @overlay.on_update
    def _(_e: object) -> None:
        """Viser callback: refresh the view when this control changes."""
        draw(int(t_sl.value))

    redraw_plot(0)
    draw(0)
    print(f"[mb_base] viser on http://localhost:{args.port}  "
          f"(ours {len(ours_pos)} frames, MB {n_b})", flush=True)
    import time
    while True:
        time.sleep(1.0)




# --------------------------------------------------------------------------------- stage: batch


#: Commanded-vx buckets the tracking error is reported in. Signed, so backward is its own bucket:
#: the solved library's backward band is narrower (-1.0 .. -0.2) than its forward one and a signed
#: bucket keeps a backward miss from cancelling a forward one inside a mean.
#: The outer edges are ``flat_vx_range`` from ``configs/clip_gen/g1_mb_v6.yaml`` -- the driver
#: envelope the clips were SAMPLED from -- restated rather than imported because this module runs
#: against a finished ``.h5`` and must not need the recipe that built it. If that range moves,
#: these move with it or the outermost bucket silently stops covering the fastest commands.
DEFAULT_BUCKETS = ((-1.0, -0.2), (-0.2, 1.0), (1.0, 2.5))


def bucket_mse(cmd: np.ndarray, achieved: np.ndarray, dt: float, buckets: tuple,
               settle_s: float = SETTLE_S) -> dict:
    """``{bucket_index: {"all": (sum_sq_err, n), "settled": (sum_sq_err, n)}}``, binned by COMMANDED vx.

    BOTH views, because the settled one alone answers a different question than the paper asks:
    ``all`` is the headline (the acceleration transient is part of tracking a command) and
    ``settled`` is the secondary steady-state number.

    Binned per FRAME rather than per segment, so a segment contributes in proportion to how long
    it was held -- which is what "the generator saw this speed for this long" means, and what the
    library's own command marginal is measured in.

    Sums rather than an RMSE, because the caller pools across clips and sqrt(mean) of pooled
    squares is not the mean of per-clip RMSEs.
    """
    n = min(len(achieved), len(cmd) - 1)
    a, c = achieved[:n], cmd[:n]
    chg = np.flatnonzero(np.any(np.round(c[1:], 6) != np.round(c[:-1], 6), axis=1)) + 1
    settled = np.ones(n, bool)
    settled[:int(settle_s / dt)] = False
    for k in chg:
        settled[k:k + int(settle_s / dt)] = False
    err2 = np.sum((a[:, :2] - c[:, :2]) ** 2, axis=1)
    out = {}
    for bi, (lo, hi) in enumerate(buckets):
        # bi == 0, not `lo == buckets[0][0]`: keying on the VALUE made every later bucket that
        # happens to share the first one's low edge inclusive-low too, so overlapping bucket sets
        # double-counted frames.
        inb = ((c[:, 0] >= lo) & (c[:, 0] <= hi) if bi == 0
               else (c[:, 0] > lo) & (c[:, 0] <= hi))
        # ALL frames is the headline: the acceleration transient is part of tracking a command.
        # SETTLED is the secondary view, so responsiveness and steady-state accuracy separate.
        out[bi] = {"all": (float(err2[inb].sum()), int(inb.sum())),
                   "settled": (float(err2[inb & settled].sum()), int((inb & settled).sum()))}
    return out


def bootstrap_rmse(per_clip: list, n_boot: int = 2000, seed: int = 0,
                   ci: float = 0.95) -> tuple:
    """Clip-CLUSTERED bootstrap of a pooled RMSE. ``per_clip`` is ``[(sum_sq, n_frames)]``.

    Resampling CLIPS rather than frames, because frames within a clip are heavily correlated --
    the same stride, the same command, the same tile. The eval harness measured what ignoring that
    costs: a pooled interval there had 14-70% actual coverage against a nominal 95%, and it got
    WORSE with more draws per unit. So the clip is the independent unit here too.
    """
    rows = [(s, n) for s, n in per_clip if n > 0]
    if not rows:
        return (float("nan"), float("nan"), float("nan"), 0, 0)
    if len(rows) < 2:
        # One clip: the point estimate is real, the INTERVAL is not -- a clip-clustered bootstrap
        # over a single cluster resamples the same clip every draw and would report a spuriously
        # tight CI. Report the point and NaN the interval rather than inventing confidence.
        ss = sum(s for s, _ in rows); nn = sum(n for _, n in rows)
        return (float(np.sqrt(ss / max(nn, 1e-9))), float("nan"), float("nan"), int(nn), 1)
    ss = np.array([r[0] for r in rows], float)
    nn = np.array([r[1] for r in rows], float)
    point = float(np.sqrt(ss.sum() / nn.sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(int(n_boot), len(rows)))
    draws = np.sqrt(ss[idx].sum(axis=1) / np.maximum(nn[idx].sum(axis=1), 1e-9))
    lo, hi = np.percentile(draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return point, float(lo), float(hi), int(nn.sum()), len(rows)


def bootstrap_delta(ours: list, base: list, n_boot: int = 2000, seed: int = 0,
                    ci: float = 0.95) -> tuple:
    """PAIRED clip-clustered bootstrap of ``rmse(base) - rmse(ours)``.

    Paired because both arms run the identical clips and the identical command profile, so the
    clip-to-clip variation is shared and differencing removes it. Two independent intervals would
    be wider and would not answer "is one better ON THE SAME INPUT".
    """
    pairs = [(o, b) for o, b in zip(ours, base) if o[1] > 0 and b[1] > 0]
    if len(pairs) < 2:
        return (float("nan"), float("nan"), float("nan"))
    os_, on = np.array([p[0][0] for p in pairs]), np.array([p[0][1] for p in pairs], float)
    bs_, bn = np.array([p[1][0] for p in pairs]), np.array([p[1][1] for p in pairs], float)
    point = float(np.sqrt(bs_.sum() / bn.sum()) - np.sqrt(os_.sum() / on.sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(int(n_boot), len(pairs)))
    d = (np.sqrt(bs_[idx].sum(axis=1) / np.maximum(bn[idx].sum(axis=1), 1e-9))
         - np.sqrt(os_[idx].sum(axis=1) / np.maximum(on[idx].sum(axis=1), 1e-9)))
    lo, hi = np.percentile(d, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return point, float(lo), float(hi)


def _provenance(args: argparse.Namespace, argv: list, clips: list, runs: list) -> str:
    """Everything needed to re-create or audit this run, as a text block.

    Written because a number in a paper whose inputs cannot be named is a number nobody can check
    a year later -- and this repo has been bitten repeatedly by artifacts whose provenance was a
    default path that had since rotted.
    """
    import subprocess as sp
    import datetime

    def git(*a: str) -> str:
        """``git`` in the repo, empty string on any failure (a tarball has no repo)."""
        try:
            return sp.run(["git", *a], cwd=_HERE, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "<unavailable>"

    dirty = git("status", "--porcelain")
    lines = [
        "# Motion Bricks baseline: velocity tracking, ours vs the stock stack",
        "",
        f"generated      : {datetime.datetime.now().astimezone().isoformat()}",
        f"host           : {os.uname().nodename}",
        "",
        "## git",
        f"branch         : {git('rev-parse', '--abbrev-ref', 'HEAD')}",
        f"commit         : {git('rev-parse', 'HEAD')}",
        f"describe       : {git('describe', '--always', '--dirty')}",
        f"working tree   : {'DIRTY -- see below' if dirty else 'clean'}",
    ]
    if dirty:
        # The DIFF, not just the file list. A paper number generated from an uncommitted tree is
        # only auditable if the uncommitted part is recorded -- naming the files says what changed
        # but not to what, and "re-create if needed" then needs a tree nobody kept.
        diff = git("diff", "HEAD")
        lines += ["", "### uncommitted files", "```", dirty, "```"]
        if diff:
            cap = 200000
            lines += ["", "### uncommitted diff (git diff HEAD)", "```diff",
                      diff[:cap] + ("\n... TRUNCATED, re-run against a clean tree"
                                    if len(diff) > cap else ""), "```"]
    lines += [
        "", "## command",
        "```",
        " ".join(argv),
        "```",
        "", "## inputs",
        f"clip library   : {os.path.abspath(args.clip)}",
        f"clips used     : {len(clips)} flat clips, indices {clips}",
        f"seed           : {args.seed}   bootstrap seed {args.boot_seed}, {args.n_boot} resamples",
        f"smoothing      : boxcar {args.smooth_s} s on the achieved velocity",
        f"settle excluded: {args.settle_s} s after every command change (SETTLE_S default "
        f"{SETTLE_S}, measured -- see mb_baseline.SETTLE_S)",
        f"buckets (vx)   : {list(args.buckets_parsed)}",
        f"MB root mode   : {args.root_mode}   (free = the shipped demo's driving mode)",
        f"mb-root        : {args.mb_root}",
        "", "## per-clip Motion Bricks runs",
    ] + [f"  {r}" for r in runs]
    return "\n".join(lines) + "\n"


def _run_matches(run_path: str, args: argparse.Namespace) -> bool:
    """Whether an existing run ``.npz`` was produced under the arm we are about to score.

    ``run_{clip:05d}.npz`` encodes neither ``--root-mode`` nor ``--seed``, so a filename match is
    not an arm match: re-running with ``--root-mode rollout`` would reuse the ``free`` runs and the
    provenance would then record ``rollout`` over motion that is not. Those two configurations are
    the ones this module exists to report SEPARATELY, so the check is worth the one file read --
    the run already stores both fields.
    """
    try:
        d = np.load(run_path, allow_pickle=True)
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    except Exception:
        return False
    ok = (meta.get("root_mode") == args.root_mode and int(meta.get("seed", -1)) == int(args.seed))
    if not ok:
        print(f"[mb_base] {os.path.basename(run_path)} was made with "
              f"root_mode={meta.get('root_mode')} seed={meta.get('seed')}, not "
              f"{args.root_mode}/{args.seed} -- regenerating", flush=True)
    return ok


def stage_batch(args: argparse.Namespace) -> int:
    """Run both arms over many flat clips and report bucketed RMSE with clip-clustered CIs."""
    import h5py
    import subprocess as sp

    with h5py.File(args.clip, "r") as f:
        tid = np.asarray(f["terrain_id"])
    flat = np.flatnonzero(tid < 0)
    rng = np.random.default_rng(args.seed)
    pick = sorted(int(i) for i in rng.choice(flat, size=min(args.n_clips, len(flat)),
                                             replace=False))
    print(f"[mb_base] {len(pick)} flat clips of {len(flat)} available: {pick[:8]}"
          f"{' ...' if len(pick) > 8 else ''}")

    work = os.path.join(args.out_dir, "runs")
    os.makedirs(work, exist_ok=True)
    buckets = args.buckets_parsed
    ours_b = {(i, w): [] for i in range(len(buckets)) for w in ("all", "settled")}
    base_b = {(i, w): [] for i in range(len(buckets)) for w in ("all", "settled")}
    floor_b = {(i, w): [] for i in range(len(buckets)) for w in ("all", "settled")}
    run_paths, failed = [], []

    for j, ci_ in enumerate(pick):
        prof = os.path.join(work, f"profile_{ci_:05d}.npz")
        runp = os.path.join(work, f"run_{ci_:05d}.npz")
        e = argparse.Namespace(**{**vars(args), "clip_index": ci_, "out": prof})
        stage_export(e)
        cmd = ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python", "-u",
               os.path.join(_HERE, "mb_baseline.py"), "--stage", "run",
               "--root-mode", args.root_mode, "--mb-root", args.mb_root,
               "--profile", prof, "--out", runp, "--seed", str(args.seed)]
        if args.reuse_runs and os.path.exists(runp) and _run_matches(runp, args):
            # The MB run is the expensive half and the scoring is free, so iterating on the METRIC
            # must not re-burn the GPU -- and re-running would also change the motion under a
            # comparison that is supposed to differ only in how it is scored.
            r = sp.CompletedProcess(cmd, 0, "", "")
        else:
            r = sp.run(cmd, cwd=os.path.dirname(os.path.dirname(_HERE)),
                       capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(runp):
            print(f"[mb_base] clip {ci_}: MB FAILED\n   "
                  + "\n   ".join((r.stderr or r.stdout).strip().splitlines()[-4:]))
            failed.append(ci_)
            continue
        run_paths.append(runp)

        clip = read_clip(args.clip, ci_)
        from robot_rl.flow_matching.viz.robot import xyzw_to_wxyz
        nm = clip["names"]
        ow = xyzw_to_wxyz(clip["y"][:, [nm.index(f"pelvis_link:ori_{c}") for c in "xyzw"]])
        vo = boxcar(body_velocity(clip["pos"], ow, clip["dt"]), clip["dt"], args.smooth_s)
        bp, bw, _bj, _n = _resample_run(runp, clip["dt"], clip["n"])
        vb = boxcar(body_velocity(bp, bw, clip["dt"]), clip["dt"], args.smooth_s)
        mo = bucket_mse(clip["cmd"], vo, clip["dt"], buckets, settle_s=args.settle_s)
        mb_ = bucket_mse(clip["cmd"], vb, clip["dt"], buckets, settle_s=args.settle_s)
        # The FLOOR: score the command against a SMOOTHED COPY OF ITSELF, i.e. what a perfect
        # tracker measures under this exact instrument. The boxcar is centred, so half its window
        # of post-change motion leaks backwards across every command change -- harmless in the
        # settled view, which masked those frames out, but real in the all-frames headline.
        # Measured rather than caveated: on a synthetic 0.4 -> 1.2 step it is 0.073 m/s, which is
        # not small next to the middle bucket. It is the SCALE of the artifact and NOT a floor --
        # an arm whose achieved velocity leads the command cancels part of it, and ours measurably
        # does (settled mid-bucket 0.0967 against an artifact of 0.1007). It applies identically to
        # both arms, so the paired DELTA is unaffected, which is why the claim rests on the delta.
        mf = bucket_mse(clip["cmd"], boxcar(clip["cmd"][:-1].copy(), clip["dt"], args.smooth_s),
                        clip["dt"], buckets, settle_s=args.settle_s)
        for bi in range(len(buckets)):
            for w in ("all", "settled"):
                ours_b[(bi, w)].append(mo[bi][w])
                base_b[(bi, w)].append(mb_[bi][w])
                floor_b[(bi, w)].append(mf[bi][w])
        print(f"[mb_base] clip {ci_:5d}  ({j + 1}/{len(pick)})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    results, lines = [], []
    for w, title in (("all", "ALL FRAMES -- the headline: the acceleration transient is part of "
                             "tracking a command"),
                     ("settled", f"SETTLED ONLY -- the first {args.settle_s} s after each command "
                                 f"change excluded, i.e. steady-state accuracy alone")):
        lines += ["", f"### {title}",
                  f"{'bucket (cmd vx)':>18} {'n frames':>9} {'clips':>6} {'artifact':>9} "
                  f"{'OURS rmse [95% CI]':>26} {'MB alone rmse [95% CI]':>26} "
                  f"{'delta [95% CI]':>24}"]
        for bi, (lo, hi) in enumerate(buckets):
            # Score both arms over the SAME clips. The baseline covers only the frames Motion
            # Bricks produced, so a bucket reached late in a clip can be populated for us and
            # empty for it -- and then the two columns describe different clip sets while the
            # paired delta silently drops to NaN. Pair first, and say how many clips were lost.
            pairs = [(o, b) for o, b in zip(ours_b[(bi, w)], base_b[(bi, w)])
                     if o[1] > 0 and b[1] > 0]
            n_drop = sum(1 for o, b in zip(ours_b[(bi, w)], base_b[(bi, w)])
                         if (o[1] > 0) != (b[1] > 0))
            po, lo_o, hi_o, nfo, nco = bootstrap_rmse([p[0] for p in pairs],
                                                      args.n_boot, args.boot_seed)
            pb, lo_b, hi_b, nfb, _c = bootstrap_rmse([p[1] for p in pairs],
                                                     args.n_boot, args.boot_seed)
            dp, dlo, dhi = bootstrap_delta([p[0] for p in pairs], [p[1] for p in pairs],
                                           args.n_boot, args.boot_seed)
            # Floor over the SAME paired clips, so it is comparable to the two columns beside it.
            keep = [k for k, (o, b) in enumerate(zip(ours_b[(bi, w)], base_b[(bi, w)]))
                    if o[1] > 0 and b[1] > 0]
            pf, _l, _h, _nf, _nc = bootstrap_rmse([floor_b[(bi, w)][k] for k in keep],
                                                  args.n_boot, args.boot_seed)
            lines.append(f"{f'({lo:+.1f}, {hi:+.1f}]':>18} {nfo:9d} {nco:6d} {pf:9.4f} "
                         f"{f'{po:.4f} [{lo_o:.4f}, {hi_o:.4f}]':>26} "
                         f"{f'{pb:.4f} [{lo_b:.4f}, {hi_b:.4f}]':>26} "
                         f"{f'{dp:+.4f} [{dlo:+.4f}, {dhi:+.4f}]':>24}")
            results.append({"window": w, "bucket": [lo, hi], "n_clips_paired": nco,
                            "n_frames_ours": nfo, "n_frames_baseline": nfb,
                            "n_clips_unpaired_dropped": n_drop,
                            "smoothing_artifact_rmse": pf,
                            "ours": {"rmse": po, "ci": [lo_o, hi_o]},
                            "baseline": {"rmse": pb, "ci": [lo_b, hi_b]},
                            "delta_base_minus_ours": {"point": dp, "ci": [dlo, dhi]}})
    table = "\n".join(lines).lstrip("\n")
    print("\n" + table)
    print(f"\ndelta > 0 means MB alone is WORSE. Clip-clustered bootstrap, "
          f"{args.n_boot} resamples, paired on clip.")
    print(f"'artifact' is what a PERFECT tracker scores here -- the command against a "
          f"{args.smooth_s} s boxcar of\nitself. The boxcar is centred, so half its window of "
          f"post-change motion leaks backwards across\nevery command change. Read it as the SCALE "
          f"of the instrument's own contribution, NOT as a floor:\nit is not a bound, because an "
          f"arm whose achieved velocity leads the command cancels part of it\n(our settled "
          f"mid-bucket does exactly that). It applies identically to both arms, so the paired\n"
          f"delta -- the number the paper claim rests on -- is unaffected either way.")
    if failed:
        print(f"MB failed on {len(failed)} clips: {failed}")

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"buckets": results, "clips": pick, "failed": failed,
                   "smooth_s": args.smooth_s, "root_mode": args.root_mode,
                   "n_boot": args.n_boot}, f, indent=1)
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write(table + "\n\ndelta > 0 means MB alone is WORSE.\n")
    with open(os.path.join(args.out_dir, "PROVENANCE.md"), "w") as f:
        f.write(_provenance(args, sys.argv, pick, run_paths))
    print(f"\nwrote {args.out_dir}/{{results.json,results.txt,PROVENANCE.md}} "
          f"and {len(run_paths)} runs under runs/")
    return 0


def main() -> int:
    """Parse args and run a stage."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["export", "run", "compare", "view", "batch"])
    p.add_argument("--clip", default=None, help="Clip library holding the reference clip.")
    p.add_argument("--clip-index", type=int, default=0)
    p.add_argument("--profile", default=None, help="run: the .npz written by --stage export")
    p.add_argument("--out", default=None, help="export / run: output .npz")
    p.add_argument("--run", default=None, help="compare: the .npz written by --stage run")
    p.add_argument("--json", default=None, help="compare: also write the stats here")
    p.add_argument("--mb-root", default=None, help="run: Motion Bricks package root (ABSOLUTE)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--n-clips", type=int, default=30, help="batch: how many flat clips to sample.")
    p.add_argument("--out-dir", default=None, help="batch: results + provenance go here.")
    p.add_argument("--conda-env", default="motionbricks")
    p.add_argument("--reuse-runs", action="store_true",
                   help="batch: skip the Motion Bricks run for any clip whose run .npz already "
                        "exists in --out-dir/runs. Re-score without regenerating.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--buckets", default="-1.0:-0.2,-0.2:1.0,1.0:2.5",
                   help="batch: commanded-vx buckets as 'lo:hi,lo:hi,...' (signed).")
    p.add_argument("--root-mode", default="free", choices=["free", "rollout"],
                   help="free: Motion Bricks owns the root, driven by movement_direction + "
                        "target_vel -- the shipped demo's mode, and the honest 'MB alone'. "
                        "rollout: the root is pinned to the exact-velocity integration of the "
                        "command, which tracks well BY CONSTRUCTION and is therefore an UPPER "
                        "BOUND on MB rather than a baseline -- report both, since the demo has no "
                        "speed input and picking one configuration would be picking a result.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--settle-s", type=float, default=SETTLE_S,
                   help=f"Seconds ignored after each command change, both arms. Default "
                        f"{SETTLE_S} s is measured; see mb_baseline.SETTLE_S.")
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
    try:
        args.buckets_parsed = tuple(tuple(float(x) for x in b.split(":"))
                                    for b in args.buckets.split(","))
    except ValueError as exc:
        raise SystemExit(f"--buckets: expected 'lo:hi,lo:hi,...', got {args.buckets!r} ({exc})")
    # Validated rather than trusted: an unparsed bucket fails deep inside bucket_mse with a shape
    # error, and a mis-ORDERED or overlapping one fails nowhere at all -- it just double-counts
    # frames into two columns of the paper table. Bucket 0 is inclusive-low by design (so the
    # lowest command is scored), which is exactly what makes an overlap silent.
    for b in args.buckets_parsed:
        if len(b) != 2:
            raise SystemExit(f"--buckets: {b} is not a lo:hi pair")
        if not b[0] < b[1]:
            raise SystemExit(f"--buckets: {b} is empty or inverted (need lo < hi)")
    for a_, b_ in zip(args.buckets_parsed, args.buckets_parsed[1:]):
        if b_[0] < a_[1]:
            raise SystemExit(f"--buckets: {a_} and {b_} overlap; a frame would be counted twice")
    if args.stage == "batch" and not (args.clip and args.out_dir and args.mb_root):
        raise SystemExit("--stage batch needs --clip, --out-dir and --mb-root")
    if args.stage == "view" and not (args.run and args.clip):
        raise SystemExit("--stage view needs --clip and --run")
    return {"export": stage_export, "run": stage_run, "compare": stage_compare,
            "view": stage_view, "batch": stage_batch}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
