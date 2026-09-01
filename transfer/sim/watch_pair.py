# Copyright 2026 Zachary Olkin. All rights reserved.

"""Watch TWO generators run the same tile, same seed, side by side in one viewer.

Answers "what does the fine-tune actually do differently?", which no success rate can.  One
arm is drawn solid and the other translucent, from a single frozen initial condition and a
single episode seed, so the only difference between the two robots is which ``fm.onnx`` drew
the plans.

**Capture then replay, never a live drive.**  ``mjeval.view_cell`` measured that attaching the
MuJoCo viewer perturbs ``mjData`` enough to change the trajectory -- one course survived 16 s
headless and fell reproducibly at 6.04 s with the viewer attached.  A live two-robot sim would
therefore show episodes that were never the ones scored.  Both arms are run HEADLESS with a
per-control-step capture, and playback only writes ``qpos`` and calls ``mj_forward``: no
physics, no policy, nothing the viewer can disturb.

**One rig, two policies.**  Both arms are driven through the SAME ``CourseRig`` -- the same
``mj_model``, the same terrain, the same spawn -- rather than two rigs built from the same
description.  Two rigs would be identical by construction and equal in practice, but sharing
one removes the question entirely, and the ghost is rendered from the main arm's model, which
only makes sense if they really are the same model.

**The episode loop mirrors ``live_select``'s K=1 path exactly** (reset, seed ``policy._rng``
from the episode seed, then repeated ``_chunk``), so the trajectory shown IS the episode that
harness scored at that seed -- not a re-derivation that merely resembles it.  The verdict is
printed for both arms and can be checked against the recorded table.

Run from ``transfer/sim`` with ``env_isaaclab`` active::

    python watch_pair.py --tile 20 --speed 2.4 --seed 4 \\
        --fm_onnx <base>_fm.onnx --fm_onnx_b logs/value/ft_eval/lr3e-5_e20_fm.onnx
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def _rpy_error(ref_wxyz: np.ndarray, act_wxyz: np.ndarray) -> np.ndarray:
    """Signed roll / pitch / yaw of the actual pose RELATIVE to the reference, in degrees.

    The relative rotation is ``q_ref^-1 (x) q_act``, so the result is expressed in the
    REFERENCE frame and reads as "how the robot is turned away from where it should be",
    rather than a world-frame attitude that would move with the reference itself.

    Decomposed ZYX (yaw-pitch-roll), the usual convention for a walking base.  Signed on
    purpose: a single unsigned angle cannot tell pitching forward from pitching back, and on
    stairs those are opposite failures.

    Args:
        ref_wxyz: Reference orientation, WXYZ.
        act_wxyz: Achieved orientation, WXYZ.

    Returns:
        ``[roll, pitch, yaw]`` in degrees, each wrapped to (-180, 180].
    """
    r = ref_wxyz / np.linalg.norm(ref_wxyz)
    a = act_wxyz / np.linalg.norm(act_wxyz)
    rw, rx, ry, rz = r[0], -r[1], -r[2], -r[3]          # conjugate = inverse (unit quat)
    aw, ax, ay, az = a
    w = rw * aw - rx * ax - ry * ay - rz * az
    x = rw * ax + rx * aw + ry * az - rz * ay
    y = rw * ay - rx * az + ry * aw + rz * ax
    z = rw * az + rx * ay - ry * ax + rz * aw
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def roll_capture(rig, policy, fm_lag, spawn, cmd, cfg, seed, mjrunner, cmd_hold_s=0.0):
    """Run one episode headless at ``seed``, capturing every control step.

    Driven through ``runner.run_episode`` rather than a local loop, because the OPENING HOLD is
    a per-control-step decision (``time < hold_until`` -> zero command, then a hard step to the
    commanded speed) and ``live_select._chunk`` takes one command for a whole 12-step chunk --
    a 0.5 s hold is 25 steps and does not divide into chunks.  ``run_episode`` already
    implements it, and with ``cmd_hold_s = 0`` it is the same loop in the same order that
    ``_chunk`` mirrors, so the no-hold arm still reproduces what ``live_select`` scored.

    The episode seed is written into ``policy._rng`` BEFORE the call: ``run_episode`` resets the
    sim and the policy's buffers but does not seed, so this is what makes the trajectory the
    one that seed produced.  Verified by reproducing the recorded verdicts, not assumed.

    Args:
        rig: The shared course rig.
        policy: The arm to drive.
        fm_lag: That arm's camera-lag ring.
        spawn: Initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        seed: Episode seed.
        mjrunner: The ``runner`` module.
        cmd_hold_s: Seconds of zero command before stepping to ``cmd``.

    Returns:
        ``(frames, plans, verdict, failure, err)``. ``frames`` is ``(time, qpos)`` per step;
        ``err`` is ``[N, 7]`` of ``(t, ex, ey, ez, roll, pitch, yaw)`` against the served
        reference -- position in metres, angles in degrees -- NaN before the first plan.
    """
    frames: list[tuple[float, np.ndarray]] = []
    plans: list[tuple[int, np.ndarray]] = []
    err: list[np.ndarray] = []          # [t, ex, ey, ez, ang] per control step
    last_plan = [-1]

    def rec(robot_, policy_) -> None:
        """Per control step: the pose, the fresh plan, and the error against the SERVED node."""
        t = float(robot_.mj_data.time)
        frames.append((t, robot_.mj_data.qpos.copy()))
        p = policy_.last_plan()
        if p is not None and int(p["plan_id"]) != last_plan[0]:
            last_plan[0] = int(p["plan_id"])
            path = policy_.predicted_root_path_w()
            if path is not None:
                plans.append((len(frames) - 1, np.asarray(path, dtype=float).copy()))
        # THE REFERENCE IS THE SERVED NODE, NOT NODE 0. `served_node_index` is the cursor --
        # 0 on a re-plan and +1 per step until the next one -- and its own docstring says a
        # diagnostic wanting "the reference at this step" must index with it. Using node 0
        # would compare the robot against a reference up to fm_update_every-1 steps stale and
        # would show a sawtooth that is the indexing, not the tracker.
        lifted = policy_.predicted_base_pose_w()
        if lifted is None:
            err.append(np.full(7, np.nan)); err[-1][0] = t
            return
        ref_p, ref_q = lifted
        j = int(policy_.served_node_index)
        j = min(max(j, 0), ref_p.shape[0] - 1)
        dp = np.asarray(robot_.mj_data.qpos[:3], dtype=float) - np.asarray(ref_p[j], float)
        # qpos is WXYZ, predicted_base_pose_w returns XYZW -- reordered, not reinterpreted.
        aq = np.asarray(robot_.mj_data.qpos[3:7], dtype=float)
        rq = np.asarray(ref_q[j], dtype=float)[[3, 0, 1, 2]]
        err.append(np.concatenate(([t], dp, _rpy_error(rq, aq))))

    policy._rng = np.random.default_rng(seed)
    verdict, failure, _acc = mjrunner.run_episode(
        rig, policy, fm_lag, spawn, cmd, None, recorder=rec,
        steer_kp=0.0, cfg_hold=float(cmd_hold_s))
    return frames, plans, verdict, failure, np.asarray(err)



def plot_errors(ea, eb, label_a, label_b, va, vb, title, out_png=None, show=True):
    """Six panels: base position error per axis, and roll / pitch / yaw error.

    Both arms on shared axes so the comparison is read off one figure.  The episodes are
    DIFFERENT LENGTHS -- one may fall while the other is still walking -- so each series simply
    ends where its episode did, and the verdict is in the legend.  Padding the shorter one
    would draw a flat tail that reads as "tracking perfectly" at exactly the moment the robot
    is on the floor.

    Args:
        ea: ``[N, 5]`` arm A errors ``(t, ex, ey, ez, deg)``.
        eb: ``[M, 5]`` arm B errors.
        label_a: Arm A name.
        label_b: Arm B name.
        va: Arm A verdict.
        vb: Arm B verdict.
        title: Figure title.
        out_png: Also write the figure here, or ``None`` for a window only.
        show: Open an interactive window and block until it is closed.

    Returns:
        ``out_png`` when one was written, else ``None``.
    """
    import matplotlib  # noqa: PLC0415

    # Agg ONLY when nothing is to be displayed. Forcing it unconditionally is what makes a
    # figure silently never appear -- the backend is process-wide and cannot be switched once
    # pyplot has bound it, so a later plt.show() would be a no-op with no error.
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = (("base position error x  (downrange)", 1, "m"),
            ("base position error y  (lateral)", 2, "m"),
            ("base position error z  (vertical)", 3, "m"),
            ("orientation error: ROLL", 4, "deg"),
            ("orientation error: PITCH", 5, "deg"),
            ("orientation error: YAW", 6, "deg"))
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
    for ax, (name, col, unit) in zip(axes, rows):
        ax.plot(ea[:, 0], ea[:, col], lw=1.4, color="#4a3f6b",
                label=f"{label_a} ({va})")
        ax.plot(eb[:, 0], eb[:, col], lw=1.4, color="#e08a1a",
                label=f"{label_b} ({vb})")
        # Every panel is a signed error now, so zero is meaningful on all six.
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.set_ylabel(f"{unit}")
        ax.set_title(name, fontsize=10, loc="left")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("episode time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=140)
        print(f"[watch] wrote {out_png}")
    if show:
        print("[watch] close the plot window to exit.", flush=True)
        try:
            plt.show()
        except Exception as exc:                                       # noqa: BLE001
            print(f"[WARN] cannot open a plot window ({type(exc).__name__}: {exc}); "
                  "pass --plot <path> to write it instead.")
    return out_png


def main() -> int:
    """Entry point."""
    import terrains as _terrains  # noqa: PLC0415  the shared CMD_HOLD_S default

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile", type=int, default=20)
    p.add_argument("--speed", type=float, default=2.4)
    p.add_argument("--approach", type=float, default=2.5)
    p.add_argument("--klass", default="")
    p.add_argument("--seed", type=int, default=0, help="episode seed; 0..39 are the ones the "
                                                       "eval measured.")
    # Two levels up from transfer/sim is the tree root -- the same derivation
    # gen_tracker_runner.py makes for its own log_root, rather than a second convention.
    p.add_argument("--tracker_onnx", default=os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "logs", "rsl_rl", "g1",
        "2026-08-20_16-36-10_mode11_decor_plan_v1", "exported", "tracker.onnx"),
        help="Defaults to the deployed tracker in THIS tree (derived from the module path, "
             "not a hardcoded hop count).")
    p.add_argument("--fm_onnx", required=True, help="arm A -- drawn SOLID.")
    p.add_argument("--fm_onnx_b", required=True, help="arm B -- drawn translucent.")
    p.add_argument("--label_a", default="base")
    p.add_argument("--label_b", default="fine-tuned")
    p.add_argument("--clip_lib", default="clips/g1_mb_v5_body_mode11_decor.h5")
    p.add_argument("--offset_y", type=float, default=0.0,
                   help="shift arm B sideways (m). 0 overlays them at true position, which is "
                        "what shows the divergence directly; ~1.0 separates them to compare "
                        "gait.")
    p.add_argument("--cmd_hold_s", type=float, default=_terrains.CMD_HOLD_S,
                   help="Seconds of zero command before stepping to --speed; defaults to "
                        "terrains.CMD_HOLD_S. Pass 0 to reproduce anything measured before "
                        "that became the default -- on tile 20 at 2.4 m/s the base generator "
                        "scores 0.275 at hold 0 and 0.675 at hold 0.5, so the two are "
                        "different experiments rather than different views of one.")
    p.add_argument("--rt", type=float, default=1.0, help="playback speed vs real time.")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip the tracking-error figure entirely.")
    p.add_argument("--show_plot", action="store_true",
                   help="With --no_view, still open the figure window (it is opened "
                        "automatically after a replay).")
    p.add_argument("--plot", default="",
                   help="ALSO write the figure to this path (a .png, or a directory); by "
                        "default it is only shown. "
                        "Four panels: base position error x/y/z in WORLD axes -- meaningful "
                        "because these tiles are yaw-free, so x is downrange and y lateral -- "
                        "and the angle to the reference orientation. Both arms overlaid.")
    p.add_argument("--no_ghost_plan", action="store_true",
                   help="Draw only arm A's plan ribbon. Two ribbons overlaid at offset_y=0 can "
                        "be hard to read on a divergent episode; this is the way back to one.")
    p.add_argument("--no_view", action="store_true",
                   help="Capture both arms and print their verdicts, then exit without opening "
                        "the viewer. The smoke test for the capture half, and the way to check "
                        "a seed is worth watching before watching it.")
    p.add_argument("--scene_tag", default="_watch")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--providers", default="auto")
    args = p.parse_args()

    import runner as mjrunner  # noqa: PLC0415
    import view_cell  # noqa: PLC0415

    course = _terrains.isaac_tile_course(args.clip_lib, int(args.tile), speeds=(args.speed,),
                                        approach=args.approach, klass=(args.klass or None))
    course = dataclasses.replace(course, name=f"tile{args.tile}{args.scene_tag}")

    def arm(fm_onnx, label):
        cfg = mjrunner.ArmRun(
            arm=0, label=label, tracker_onnx=args.tracker_onnx, fm_onnx=fm_onnx,
            out_dir="/tmp",
            providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
            fall_height=args.fall_height, fall_tilt_deg=args.fall_tilt_deg,
            cmd_hold_s=0.0, steer_kp=0.0, zero_noise=False, fm_seed=0)
        return cfg, *mjrunner.load_arm(cfg)

    cfg_a, pol_a, lag_a, spawn = arm(args.fm_onnx, args.label_a)
    rig = mjrunner.CourseRig(course, pol_a, cfg_a, spawn)
    cfg_b, pol_b, lag_b, _sp = arm(args.fm_onnx_b, args.label_b)

    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)
    print(f"[watch] tile {args.tile} @ {args.speed} m/s, seed {args.seed}, GAUSSIAN noise "
          "(what the eval measured; gen_tracker_runner's zero_noise default is a DIFFERENT "
          f"sampling regime), cmd_hold {args.cmd_hold_s:g} s")
    fa, pa, va, _, ea = roll_capture(rig, pol_a, lag_a, spawn, cmd, cfg_a, args.seed, mjrunner,
                                     cmd_hold_s=args.cmd_hold_s)
    print(f"[watch] {args.label_a:12s} {va:8s} {len(fa)} steps")
    fb, pb, vb, _, eb = roll_capture(rig, pol_b, lag_b, spawn, cmd, cfg_b, args.seed, mjrunner,
                                     cmd_hold_s=args.cmd_hold_s)
    print(f"[watch] {args.label_b:12s} {vb:8s} {len(fb)} steps")

    out = None
    if args.plot:
        out = (args.plot if args.plot.endswith(".png") else
               os.path.join(args.plot, f"err_t{args.tile}_s{args.speed}_seed{args.seed}"
                                       f"_hold{args.cmd_hold_s:g}.png"))
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    title = (f"tile {args.tile} @ {args.speed} m/s, seed {args.seed}, "
             f"cmd_hold {args.cmd_hold_s:g} s -- error vs the SERVED reference node")

    def draw(show: bool) -> None:
        """Render the error figure once, if it was not disabled."""
        if not args.no_plot:
            plot_errors(ea, eb, args.label_a, args.label_b, va, vb, title,
                        out_png=out, show=show)

    if args.no_view:
        # Nothing to replay, so the figure is the whole output -- but --no_view is also the
        # batch path, so it only opens a window when one was asked for.
        draw(show=bool(args.show_plot))
        print("[watch] --no_view: capture verified, viewer not opened.")
        return 0
    summary = {"course": course, "speed": args.speed, "verdict": va, "failure": "",
               "row": {}, "model": rig.robot.mj_model}
    # Both plan ribbons are drawn: the whole question is whether the two generators PROPOSE
    # different things at the same state, and the bodies alone only show what the tracker made
    # of the proposals. Arm A's ribbon is PLAN_RGBA, arm B's a distinct hue.
    ghost = {"label": args.label_b, "frames": fb, "rgba": (0.95, 0.45, 0.15, 0.45),
             "offset_y": float(args.offset_y), "verdict": vb,
             "plans": ([] if args.no_ghost_plan else pb)}
    view_cell.replay(summary, fa, pa, args.rt, args.loop, ghost=ghost)
    # AFTER the replay: the MuJoCo viewer owns the GL context and blocks until closed, so a
    # figure opened first would sit behind it unseen.
    draw(show=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
