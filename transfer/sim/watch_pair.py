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

**Seeds are captured in BULK, then filtered on the outcome.**  Which episode is worth
watching is exactly the thing that is not knowable in advance -- two arms agree on most seeds
and the informative one is where they disagree -- so ``--seeds`` takes a range, both arms are
rolled for every one of them, and ``--only diverged`` (or ``b_wins`` / ``a_wins``) picks what
to step through.  Watching a hand-picked seed mostly shows two robots on top of each other.

``--only a_wins`` is worth running every time: it is the REGRESSION set, the episodes the
fine-tune lost, and a headline success rate is a difference of two counts that hides them.

Run from ``transfer/sim`` with ``env_isaaclab`` active.  ``--clip_lib`` is repo-root-relative
by default and this script runs from a subdirectory, so pass it::

    python watch_pair.py --tile 6 --speed 0.8 --seeds 0-15 --only diverged \\
        --clip_lib ../../clips/g1_mb_v5_body_mode11_decor.h5 \\
        --fm_onnx <base>_fm.onnx --fm_onnx_b <finetuned>_fm.onnx
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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


from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import (  # noqa: E402
    TrackingErrorSpec as _TrackingErrorSpec,
)

#: Arm A is drawn solid in the viewer, so it is the darker colour in every figure too.
_A_COLOR = "#4a3f6b"
_B_COLOR = "#e08a1a"


def roll_capture(rig, policy, fm_lag, spawn, cmd, cfg, seed, mjrunner, cmd_hold_s=0.0,
                 spec=None):
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
        ``(frames, plans, verdict, failure, err, probe)``. ``frames`` is ``(time, qpos)`` per
        step; ``err`` is ``[N, 7]`` of ``(t, ex, ey, ez, roll, pitch, yaw)`` against the served
        reference -- position in metres, angles in degrees -- NaN before the first plan.
        ``probe`` is the folded :class:`~mjeval.tracking.TrackingProbe` when ``spec`` was
        given, carrying the reward, or ``None``.

    The probe is folded HERE rather than derived from ``err`` afterwards because the reward is
    a function of the full tracking error -- joints included -- and ``err`` keeps only the base
    pose for the figure. Re-deriving a reward from six of its channels would produce a number
    that moves with the real one and equals it nowhere.
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

    probe_seen = [-1]
    probe = None
    if spec is not None:
        from tracking import TrackingProbe  # noqa: PLC0415
        probe = TrackingProbe(spec, keep_trace=True, hold_until_s=float(cmd_hold_s))

    def rec_all(robot_, policy_) -> None:
        """Capture the pose and error, and fold the reward probe on the same step."""
        rec(robot_, policy_)
        if probe is not None:
            is_replan = int(last_plan[0]) != probe_seen[0]
            probe_seen[0] = int(last_plan[0])
            probe.fold(robot_, policy_, len(frames) - 1, is_replan=bool(is_replan))

    policy._rng = np.random.default_rng(seed)
    verdict, failure, _acc = mjrunner.run_episode(
        rig, policy, fm_lag, spawn, cmd, None, recorder=rec_all,
        steer_kp=0.0, cfg_hold=float(cmd_hold_s))
    if probe is not None:
        probe.finish()
    return frames, plans, verdict, failure, np.asarray(err), probe



def plot_errors(ea, eb, label_a, label_b, va, vb, title, out_png=None, show=True,
                pa=None, pb=None, spec=None, dt=0.02):
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
        pa: Arm A's folded ``TrackingProbe``, or ``None`` to omit the reward panel.
        pb: Arm B's.
        spec: The ``TrackingErrorSpec`` the probes were folded under, needed to turn the
            stored trace back into a per-step reward.
        dt: Control period, from ``policy.get_dt()``. The probe stores STEP INDICES, and the
            error panels are in seconds, so the reward panel needs it to share their x-axis.
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
    # THE REWARD PANEL IS FIRST, and it is the only panel that is not an error: the six error
    # traces say what diverged, this says whether it mattered. Drawn from each probe's own
    # trace through `step_reward`, so it is the same quantity the critic's quality head
    # regresses and the same one live_select reports as `track_reward` -- not a re-derivation
    # off the six base-pose channels this figure happens to keep.
    want_reward = pa is not None and pb is not None and spec is not None
    n_panels = 7 if want_reward else 6
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.2 * n_panels), sharex=True)
    axes = list(axes)
    if want_reward:
        from tracking import step_reward  # noqa: PLC0415
        ax = axes.pop(0)
        for probe, lab, verdict, colour in ((pa, label_a, va, _A_COLOR),
                                            (pb, label_b, vb, _B_COLOR)):
            tr = probe.trace_array().astype(np.float64)
            if tr.shape[0] == 0:
                continue
            t = probe.step_index_array().astype(np.float64) * float(dt)
            ax.plot(t, step_reward(tr, spec), lw=1.6, color=colour,
                    label=f"{lab} ({verdict})  mean {probe.mean_step_reward:.3f}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("reward")
        ax.set_title("per-step tracking reward -- the quantity the quality head regresses",
                     fontsize=10, loc="left")
        ax.legend(loc="best", fontsize=9)
    for ax, (name, col, unit) in zip(axes, rows):
        ax.plot(ea[:, 0], ea[:, col], lw=1.4, color=_A_COLOR,
                label=f"{label_a} ({va})")
        ax.plot(eb[:, 0], eb[:, col], lw=1.4, color=_B_COLOR,
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


def _parse_seeds(text: str) -> list[int]:
    """Parse ``"4"`` / ``"0-7"`` / ``"0-3,9"`` into a seed list.

    Args:
        text: The specification.

    Returns:
        Seeds in the order written, duplicates preserved.

    Raises:
        SystemExit: On an unparseable token, rather than silently dropping it -- a typo that
            quietly watched fewer episodes than asked for would be invisible.
    """
    out: list[int] = []
    for tok in (t.strip() for t in text.split(",")):
        if not tok:
            continue
        try:
            if "-" in tok.lstrip("-"):
                lo, hi = tok.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(tok))
        except ValueError:
            raise SystemExit(f"--seeds: cannot parse {tok!r} in {text!r}") from None
    if not out:
        raise SystemExit(f"--seeds {text!r} names no seed")
    return out


def _select(caps: list[dict], only: str) -> list[dict]:
    """Filter captured seeds by how the two arms compared.

    Args:
        caps: One record per seed, carrying ``va`` and ``vb``.
        only: The filter name, or ``""`` for all.

    Returns:
        The kept records, in their capture order.

    Raises:
        SystemExit: On an unknown filter name. Not a silent pass-through: a mistyped filter
            that showed every episode would look exactly like one that matched everything.
    """
    if not only:
        return list(caps)
    a = [bool(c["va"] == "success") for c in caps]
    b = [bool(c["vb"] == "success") for c in caps]
    tests = {
        "diverged": lambda i: a[i] != b[i],
        "b_wins": lambda i: b[i] and not a[i],
        "a_wins": lambda i: a[i] and not b[i],
        "both_fail": lambda i: not a[i] and not b[i],
        "both_ok": lambda i: a[i] and b[i],
    }
    if only not in tests:
        raise SystemExit(f"--only {only!r} is not one of {sorted(tests)}")
    return [c for i, c in enumerate(caps) if tests[only](i)]


def main() -> int:
    """Entry point."""
    import terrains as _terrains  # noqa: PLC0415  the shared CMD_HOLD_S default

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile", type=int, default=20)
    p.add_argument("--speed", type=float, default=2.4)
    p.add_argument("--approach", type=float, default=2.5)
    p.add_argument("--klass", default="")
    p.add_argument("--seeds", default="0",
                   help="Episode seeds: '4', '0-7', '0-3,9'. Seeds 0..39 are the ones "
                        "live_select scores, so a seed named here is the SAME episode that "
                        "harness reported. Both arms are captured for EVERY seed before any "
                        "viewer opens, which is what lets --only select on the outcome -- and "
                        "that is the point: the seed worth watching is the one the two arms "
                        "disagree about, which is not knowable in advance.")
    p.add_argument("--only", default="",
                   help="Which captured seeds to step through: 'diverged' (the arms disagree), "
                        "'b_wins' (B arrives where A fell), 'a_wins' (the REGRESSION -- worth "
                        "looking at, it is the half a headline success rate hides), "
                        "'both_fail', 'both_ok', or '' for all.")
    p.add_argument("--sort", default="seed",
                   choices=("seed", "divergence", "reward_gap", "reward_a"),
                   help="'divergence' orders by |steps_B - steps_A| and 'reward_gap' by "
                        "|mean reward B - A|, largest first, so the most different episode is "
                        "watched first. 'reward_a' is arm A's worst tracking first, which "
                        "finds the episodes it survived BADLY -- a verdict cannot.")
    p.add_argument("--spec", default="",
                   help="tracking_spec.json from measure_tracking_error.py --stage report. "
                        "Without it the BOOTSTRAP spec is used, whose thresholds are "
                        "placeholders -- the reward is still comparable BETWEEN the two arms "
                        "here, since both are folded under the same spec, but its absolute "
                        "level means nothing and must not be compared across runs.")
    p.add_argument("--tracker_onnx", default=os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "logs", "rsl_rl", "g1",
        "2026-08-20_16-36-10_mode11_decor_plan_v1", "exported", "tracker.onnx"),
        help="Defaults to the deployed tracker in THIS tree (derived from the module path, "
             "not a hardcoded hop count).")
    p.add_argument("--fm_onnx", required=True, help="arm A -- drawn SOLID.")
    p.add_argument("--fm_onnx_b", required=True, help="arm B -- drawn translucent.")
    p.add_argument("--label_a", default="base")
    p.add_argument("--label_b", default="fine-tuned")
    p.add_argument("--history_from", default="generator",
                   choices=["generator", "measured"],
                   help="arm A's history source: the nodes it served ('generator') or the "
                        "robot's own achieved state ('measured').")
    p.add_argument("--history_from_b", default="generator",
                   choices=["generator", "measured"],
                   help="arm B's history source. Pass the SAME --fm_onnx twice and differ "
                        "only here to see the measured-history A/B side by side.")
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

    def arm(fm_onnx, label, history_from):
        cfg = mjrunner.ArmRun(
            arm=0, label=label, tracker_onnx=args.tracker_onnx, fm_onnx=fm_onnx,
            out_dir="/tmp",
            providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
            fall_height=args.fall_height, fall_tilt_deg=args.fall_tilt_deg,
            cmd_hold_s=0.0, steer_kp=0.0, zero_noise=False, fm_seed=0,
            history_from=history_from)
        return cfg, *mjrunner.load_arm(cfg)

    # The two arms may differ in the GENERATOR, in the HISTORY SOURCE, or in both. Passing
    # the same fm.onnx twice with different --history_from is the measured-history A/B, and
    # it is the case where a side-by-side earns the most: the success rates are close enough
    # that a number cannot say what changed, while the plans are visibly different.
    cfg_a, pol_a, lag_a, spawn = arm(args.fm_onnx, args.label_a, args.history_from)
    rig = mjrunner.CourseRig(course, pol_a, cfg_a, spawn)
    cfg_b, pol_b, lag_b, _sp = arm(args.fm_onnx_b, args.label_b, args.history_from_b)

    # ONE spec for BOTH arms, always. The reward here is a comparison, and two arms folded
    # under different thresholds would differ by the thresholds rather than by the policy.
    # The BOOTSTRAP default's absolute level is a placeholder, which the banner says out loud
    # so it cannot be quoted as a measurement or compared across runs.
    from mjeval.tracking import BOOTSTRAP_SPEC  # noqa: PLC0415
    spec = BOOTSTRAP_SPEC
    if args.spec:
        with open(args.spec) as fh:
            spec = _TrackingErrorSpec.from_dict(json.load(fh))
        print(f"[watch] spec: {args.spec}")
    else:
        print("[watch] spec: BOOTSTRAP (un-fitted placeholders). The reward is comparable "
              "BETWEEN these two arms, which share it, and nowhere else.")
    dt = float(pol_a.get_dt())

    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)
    seeds = _parse_seeds(args.seeds)
    print(f"[watch] tile {args.tile} @ {args.speed} m/s, {len(seeds)} seed(s), GAUSSIAN noise "
          "(what the eval measured; gen_tracker_runner's zero_noise default is a DIFFERENT "
          f"sampling regime), cmd_hold {args.cmd_hold_s:g} s")

    # EVERY SEED IS CAPTURED BEFORE ANYTHING IS SHOWN. The filters select on the OUTCOME, which
    # is not knowable before the roll; and capture-then-replay is required regardless, since
    # attaching the viewer perturbs mjData enough to change the trajectory.
    caps = []
    hdr = (f"{'seed':>5} | {args.label_a[:14]:>14} {'verdict':>8} {'steps':>6} {'reward':>7}"
           f" | {args.label_b[:14]:>14} {'verdict':>8} {'steps':>6} {'reward':>7} | note")
    print("[watch] " + hdr)
    print("[watch] " + "-" * len(hdr))
    for sd in seeds:
        fa, pla, va, _, ea, pra = roll_capture(rig, pol_a, lag_a, spawn, cmd, cfg_a, sd,
                                               mjrunner, cmd_hold_s=args.cmd_hold_s, spec=spec)
        fb, plb, vb, _, eb, prb = roll_capture(rig, pol_b, lag_b, spawn, cmd, cfg_b, sd,
                                               mjrunner, cmd_hold_s=args.cmd_hold_s, spec=spec)
        ra = pra.mean_step_reward if pra is not None else float("nan")
        rb = prb.mean_step_reward if prb is not None else float("nan")
        caps.append({"seed": sd, "fa": fa, "pa": pla, "va": va, "ea": ea, "pra": pra, "ra": ra,
                     "fb": fb, "pb": plb, "vb": vb, "eb": eb, "prb": prb, "rb": rb,
                     "div": abs(len(fb) - len(fa)),
                     "rgap": (abs(rb - ra) if np.isfinite(ra) and np.isfinite(rb) else -1.0)})
        note = "DIVERGED" if (va == "success") != (vb == "success") else ""
        print(f"[watch] {sd:>5} | {'':>14} {va:>8} {len(fa):>6} {ra:>7.3f}"
              f" | {'':>14} {vb:>8} {len(fb):>6} {rb:>7.3f} | {note}")

    n_a = sum(c["va"] == "success" for c in caps)
    n_b = sum(c["vb"] == "success" for c in caps)
    n_div = sum((c["va"] == "success") != (c["vb"] == "success") for c in caps)
    ma = float(np.nanmean([c["ra"] for c in caps]))
    mb = float(np.nanmean([c["rb"] for c in caps]))
    print(f"[watch] SUCCESS  {args.label_a} {n_a}/{len(caps)} vs {args.label_b} "
          f"{n_b}/{len(caps)};  {n_div} seed(s) disagree")
    print(f"[watch] REWARD   {args.label_a} {ma:.4f} vs {args.label_b} {mb:.4f}  "
          f"(delta {mb - ma:+.4f}), mean over every scored step of every seed")
    print("[watch] The two can move in OPPOSITE directions: reward is graded and scores the "
          "episodes both arms survive, where success is a threshold that most of them clear. "
          "A reward gain with no success gain is cleaner tracking, not a better policy -- and "
          "the reverse is an arm that survives by tracking loosely.")

    keep = _select(caps, args.only)
    if args.sort == "divergence":
        keep.sort(key=lambda c: -c["div"])
    elif args.sort == "reward_gap":
        keep.sort(key=lambda c: -c["rgap"])
    elif args.sort == "reward_a":
        # Worst tracking first; a seed that served no reference sorts LAST rather than opening
        # the walkthrough with a degenerate episode.
        keep.sort(key=lambda c: (float("inf") if not np.isfinite(c["ra"]) else c["ra"]))
    if args.only:
        print(f"[watch] --only {args.only}: {len(keep)} of {len(caps)} seed(s)")
    if not keep:
        print("[watch] nothing matched the filter; nothing to show.")
        return 0

    def _png(c) -> str | None:
        """Where this seed's figure goes, or None."""
        if not args.plot:
            return None
        if args.plot.endswith(".png") and len(keep) == 1:
            return args.plot
        d = os.path.dirname(args.plot) if args.plot.endswith(".png") else args.plot
        return os.path.join(d, f"err_t{args.tile}_s{args.speed:g}_seed{c['seed']}"
                               f"_hold{args.cmd_hold_s:g}.png")

    def _title(c) -> str:
        """Figure title carrying everything needed to identify the episode."""
        return (f"tile {args.tile} @ {args.speed:g} m/s, seed {c['seed']}, "
                f"cmd_hold {args.cmd_hold_s:g} s -- error vs the SERVED reference node")

    def _draw(c, show: bool) -> None:
        """Render one seed's figure, if it was not disabled."""
        if args.no_plot:
            return
        out = _png(c)
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        plot_errors(c["ea"], c["eb"], args.label_a, args.label_b, c["va"], c["vb"],
                    _title(c), out_png=out, show=show,
                    pa=c["pra"], pb=c["prb"], spec=spec, dt=dt)

    def _describe(pos: int, c) -> str:
        """One line naming the episode and both arms' outcome, for the navigator prompt."""
        return (f"[{pos + 1}/{len(keep)}] seed {c['seed']}: "
                f"{args.label_a} {c['va']} r={c['ra']:.3f} ({len(c['fa'])} steps) | "
                f"{args.label_b} {c['vb']} r={c['rb']:.3f} ({len(c['fb'])} steps)")

    if args.no_view:
        for c in keep:
            _draw(c, show=bool(args.show_plot))
        print(f"[watch] --no_view: {len(keep)} capture(s) verified, viewer not opened.")
        return 0

    # The same navigator watch_tracking uses: the walkthrough is a browse, not a queue, because
    # the episode worth a second look is usually identified by having just watched the next one.
    pos = 0
    while 0 <= pos < len(keep):
        c = keep[pos]
        print("\n" + _describe(pos, c))
        summary = {"course": course, "speed": args.speed, "verdict": c["va"], "failure": "",
                   "row": {}, "model": rig.robot.mj_model}
        # Both plan ribbons are drawn: the question is whether the two generators PROPOSE
        # different things at the same state, and the bodies alone only show what the tracker
        # made of the proposals. Arm A's ribbon is PLAN_RGBA, arm B's a distinct hue.
        ghost = {"label": args.label_b, "frames": c["fb"], "rgba": (0.95, 0.45, 0.15, 0.45),
                 "offset_y": float(args.offset_y), "verdict": c["vb"],
                 "plans": ([] if args.no_ghost_plan else c["pb"])}
        view_cell.replay(summary, c["fa"], c["pa"], args.rt, args.loop, ghost=ghost)
        # AFTER the replay: the MuJoCo viewer owns the GL context and blocks until closed, so a
        # figure opened first would sit behind it unseen.
        _draw(c, show=True)
        try:
            ans = input(f"[{pos + 1}/{len(keep)}] enter/n next, p prev, r replay, "
                        "<seed> jump, q quit > ").strip().lower()
        except EOFError:
            break
        if ans in ("q", "quit"):
            break
        if ans in ("p", "prev"):
            pos = max(pos - 1, 0)
        elif ans in ("r", "replay"):
            continue
        elif ans.isdigit():
            # A SEED, not a position: the table above is keyed by seed and that is what a
            # reader has in front of them. An unknown one advances rather than throwing.
            want = int(ans)
            hit = [i for i, cc in enumerate(keep) if cc["seed"] == want]
            pos = hit[0] if hit else pos + 1
        else:
            pos += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
