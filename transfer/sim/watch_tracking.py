# Copyright 2026 Zachary Olkin. All rights reserved.

"""Step through episodes on one tile: watch the motion, then read the reward and the flags.

The instrument for choosing the strict-failure thresholds.  ``measure_tracking_error.py`` says
what the distributions ARE; this says what a given episode LOOKED like while producing them,
which is the only way to tell a bound that catches stumbles from one that catches ordinary stair
descent.

Per episode, in order: the MuJoCo viewer replays the captured trajectory, and closing it opens a
six-panel figure of the same episode --

1. the ROOT POSITION in the world frame with the spawn subtracted, so every panel below can be
   read against where on the terrain the robot actually was;
2. the four DeepMimic reward terms and their weighted total;
3. position vs the served node, signed per axis -- lateral and height carry their bounds,
   **downrange does not**, because a terrain-matched re-plan moves the reference forward on
   purpose and the tracker then cannot be anything but behind it;
4. tilt against its cone, and yaw beside it -- **yaw carries no bound** either;
5. the twist error, linear and angular;
6. the joint RMS.

The LATCH is marked on every panel, together with the step the robot entered the goal, so
"violated then recovered and arrived anyway" is visible as a shape rather than inferred from two
numbers.  That case is common and is exactly why the latch does not end the episode.

**The error trace does not start at step 0** -- nothing is folded before the generator's first
plan -- so its x comes from the recorded control-step indices rather than from ``arange``.  The
difference is about the length of the opening command hold, ~25 steps, and a shift of that size
is entirely plausible-looking on every curve.

**Capture then replay, never watch live.**  Attaching the viewer changes the trajectory
(``mjeval.runner`` documents an episode that survived headless and fell at 6.04 s with the
viewer attached), so every episode is rolled to completion first and the viewer only writes
``qpos``.  Same discipline as ``watch_pair.py``, which this is a sibling of.

Usage::

    python watch_tracking.py --tile 20 --speed 1.6 --seeds 0-7 \\
        --tracker_onnx <...>/exported/tracker.onnx --fm_onnx <...>_fm.onnx \\
        --clip_lib clips/g1_mb_v5_body_mode11_decor.h5

    # batch: write every figure, open nothing
    python watch_tracking.py --tile 20 --seeds 0-19 --no_view --plot logs/tracking/t20 ...

At the prompt: ``<enter>``/``n`` next, ``p`` previous, ``r`` replay this one, ``<number>`` jump,
``s`` skip the viewer for this one, ``q`` quit.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from deploy.tracking_error import (  # noqa: E402
    BOOTSTRAP_SPEC,
    NUM_FIXED_CHANNELS,
    REWARD_TERM_NAMES,
    TrackingErrorSpec,
    step_reward,
    step_reward_terms,
)

#: Panel colours, deliberately distinguishable in greyscale as well as in hue.
_TERM_COLOR = {"pos": "#4a3f6b", "ori": "#e08a1a", "vel": "#3d8b7d", "joint": "#b0413e"}
_AXIS_COLOR = ("#4a3f6b", "#e08a1a", "#3d8b7d")


def parse_seeds(text: str) -> list[int]:
    """Parse ``"0-7"`` / ``"0,3,9"`` / ``"0-3,7"`` into a seed list.

    Args:
        text: The specification.

    Returns:
        Seeds, in the order written, duplicates preserved (repeating one is a legitimate way to
        ask for it twice in a walkthrough).

    Raises:
        ValueError: On an unparseable token, rather than silently dropping it -- a typo that
            quietly watched fewer episodes than asked for would be invisible.
    """
    out: list[int] = []
    for tok in (t.strip() for t in text.split(",")):
        if not tok:
            continue
        if "-" in tok.lstrip("-"):
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    if not out:
        raise ValueError(f"--seeds {text!r} selected nothing")
    return out


def roll_capture(rig, policy, fm_lag, spawn, cmd, cfg, seed, mjrunner, spec, cmd_hold_s):
    """Roll one episode headless, capturing the frames, the plans and the error trace.

    Both hooks are used and they are different things: ``recorder`` gets the pose for the
    viewer, ``probe`` gets the state dict and the re-plan edge for the tracking fold.  Both fire
    at the same point -- immediately after ``acc.fold``, before the physics advance -- so a
    captured frame and its error row describe the same instant the policy acted on.

    Args:
        rig: The shared course rig.
        policy: The loaded ``GenTrackerPolicy``.
        fm_lag: The camera-lag ring.
        spawn: Initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        seed: Episode seed; written into ``policy._rng`` before the call, because
            ``run_episode`` resets the sim and the policy's buffers but does not seed.
        mjrunner: The ``runner`` module.
        spec: Thresholds and kernel widths.
        cmd_hold_s: Seconds of zero command before the commanded speed.

    Returns:
        ``(frames, plans, verdict, failure, probe)``.
    """
    from tracking import TrackingProbe  # noqa: PLC0415

    frames: list[tuple[float, np.ndarray]] = []
    plans: list[tuple[int, np.ndarray]] = []
    probe = TrackingProbe(spec, keep_trace=True, hold_until_s=float(cmd_hold_s))
    last_plan = [-1]

    def _record(robot_, policy_) -> None:
        """Pose per control step, and the fresh plan's root path on a re-plan."""
        frames.append((float(robot_.mj_data.time), robot_.mj_data.qpos.copy()))
        p = policy_.last_plan()
        if p is not None and int(p["plan_id"]) != last_plan[0]:
            last_plan[0] = int(p["plan_id"])
            path = policy_.predicted_root_path_w()
            if path is not None:
                plans.append((len(frames) - 1, np.asarray(path, dtype=float).copy()))

    def _fold(robot_, policy_, _state, is_replan) -> None:
        """Fold this step's error row, stamped with the CONTROL-step index.

        ``recorder`` fires immediately before ``probe``, so ``len(frames) - 1`` is this step's
        index. Not ``probe.n_folded``: the trace starts at the generator's first plan, several
        steps into the episode, so a folded-row count would put every trace row earlier than
        the pose it belongs to.
        """
        probe.fold(robot_, policy_, len(frames) - 1, is_replan=bool(is_replan))

    policy._rng = np.random.default_rng(seed)
    verdict, failure, _acc = mjrunner.run_episode(
        rig, policy, fm_lag, spawn, cmd, None, recorder=_record, probe=_fold,
        steer_kp=0.0, cfg_hold=float(cmd_hold_s))
    probe.finish()
    return frames, plans, verdict, failure, probe


def plot_episode(probe, dt, spec, title, out_png=None, show=True, goal_step=-1, frames=None):
    """Six panels: where the robot IS, then the reward terms, the checks, the twist, the joints.

    The root-position panel comes first on purpose: every other panel is an error against a
    reference, and an error is uninterpretable without knowing whether the robot was still
    approaching, on the flight, or past it.  It is drawn in the WORLD frame with the spawn
    subtracted, so ``z`` reads directly as height gained or lost on the terrain.

    Args:
        probe: A folded :class:`~mjeval.tracking.TrackingProbe` with ``keep_trace``.
        dt: Control step, seconds -- the x axis is episode time, not step index, so it is
            comparable with the viewer's clock and with every other plot in this tree.
        spec: The bounds actually applied, drawn on the panels they bound.
        title: Figure title.
        out_png: Also write the figure here, or ``None``.
        show: Open an interactive window and block until it is closed.
        goal_step: Step the robot entered the goal, or -1.
        frames: ``(time, qpos)`` per CONTROL step, for the root-position panel.  ``None``
            omits that panel rather than faking it.

    Returns:
        ``out_png`` when one was written, else ``None``.
    """
    import matplotlib  # noqa: PLC0415

    # Agg ONLY when nothing is displayed: the backend is process-wide and cannot be switched
    # once pyplot has bound it, so forcing it here would make a later show() a silent no-op.
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    err = probe.trace_array().astype(np.float64)
    if err.shape[0] == 0:
        print("[watch] this episode served no reference -- nothing to plot.")
        return None
    # The trace starts at the generator's FIRST PLAN, not at step 0, so its x comes from the
    # recorded control-step indices. Using arange would slide every error curve earlier by the
    # length of the opening hold -- a shift of ~25 steps that looks entirely plausible.
    t = probe.step_index_array().astype(np.float64) * float(dt)
    terms = step_reward_terms(err, spec)
    total = step_reward(err, spec)

    n_panels = 6 if frames else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.6 * n_panels), sharex=True)
    axes = list(axes)

    if frames:
        ax = axes.pop(0)
        ft = np.array([f[0] for f in frames], dtype=np.float64)
        # Sim time, not step*dt: run_episode's clock is what the viewer shows, and the two
        # differ if the decimation ever changes.
        ft = ft - ft[0]
        root = np.array([f[1][:3] for f in frames], dtype=np.float64)
        root = root - root[0]
        for i, (name, color) in enumerate(zip(("x (downrange)", "y (lateral)", "z (height)"),
                                              _AXIS_COLOR)):
            ax.plot(ft, root[:, i], lw=1.4, color=color, label=name)
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.set_ylabel("m")
        ax.set_title("root position in the WORLD frame, spawn subtracted -- where in the "
                     "motion we are", fontsize=10, loc="left")
        ax.legend(loc="best", fontsize=8, ncol=3)

    ax = axes[0]
    for name in REWARD_TERM_NAMES:
        w = spec.weights[REWARD_TERM_NAMES.index(name)]
        ax.plot(t, terms[name], lw=1.1, alpha=0.75, color=_TERM_COLOR[name],
                label=f"r_{name} (w={w:g})")
    ax.plot(t, total, lw=2.0, color="black", label="r_total")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("reward")
    ax.set_title(f"DeepMimic tracking reward -- episode mean {total.mean():.3f}",
                 fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=8, ncol=5)

    ax = axes[1]
    bounds = (None, float(spec.max_lateral_m), float(spec.max_height_m))
    for i, (name, color) in enumerate(zip(("downrange", "lateral", "height"), _AXIS_COLOR)):
        # Downrange is drawn SOLID and unbounded, the other two dashed against their limit, so
        # the panel shows at a glance which axes can end an episode. A terrain-matched re-plan
        # moves the reference downrange on purpose, so that axis is graded and never fatal.
        label = name if bounds[i] is None else f"{name}  (+/- {bounds[i]:.2f} m)"
        ax.plot(t, err[:, i], lw=1.3, color=color, label=label,
                ls="-" if bounds[i] is None else "-")
        if bounds[i] is not None:
            ax.axhline(bounds[i], color=color, lw=0.8, ls=":", alpha=0.7)
            ax.axhline(-bounds[i], color=color, lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.set_ylabel("m")
    ax.set_title("position vs the served node (its yaw frame) -- lateral/height bounded; "
                 "DOWNRANGE IS NOT (a re-plan moves it forward on purpose)",
                 fontsize=10, loc="left")
    ax.legend(loc="best", fontsize=8, ncol=3)

    ax = axes[2]
    ax.plot(t, np.degrees(err[:, 3]), lw=1.5, color="#b0413e", label="tilt")
    ax.plot(t, np.degrees(np.abs(err[:, 4])), lw=1.1, color="#7a7a7a", ls="--",
            label="|yaw|  (graded, NOT a failure check)")
    ax.axhline(np.degrees(spec.max_tilt_rad), color="#b0413e", lw=0.9, ls=":")
    ax.set_ylabel("deg")
    ax.set_title(f"orientation vs the served node -- tilt cone "
                 f"{np.degrees(spec.max_tilt_rad):.1f} deg (dotted). Tilt is measured against "
                 f"GRAVITY and is not re-zeroed at a re-plan; yaw is.", fontsize=10, loc="left")
    ax.legend(loc="best", fontsize=8)

    ax = axes[3]
    ax.plot(t, np.linalg.norm(err[:, 5:8], axis=-1), lw=1.3, color="#4a3f6b", label="|lin| m/s")
    ax.plot(t, np.linalg.norm(err[:, 8:11], axis=-1), lw=1.3, color="#e08a1a",
            label="|ang| rad/s")
    ax.set_ylabel("twist err")
    ax.set_title("base twist error vs the served node", fontsize=10, loc="left")
    ax.legend(loc="best", fontsize=8)

    ax = axes[4]
    ax.plot(t, np.sqrt(np.mean(err[:, NUM_FIXED_CHANNELS:] ** 2, axis=-1)), lw=1.3,
            color="#3d8b7d")
    ax.set_ylabel("rad")
    ax.set_xlabel("episode time (s)")
    ax.set_title("joint angle error, RMS over the 29 joints", fontsize=10, loc="left")

    # The events, on EVERY panel including the root-position one, so a stumble can be read
    # across all of them -- and against where on the terrain it happened -- at one glance.
    hold_end = float(probe.hold_until_s)
    for ax in fig.axes:
        ax.grid(alpha=0.25)
        if hold_end > 0.0:
            # The opening hold, shaded because it is LATCHED but NOT SCORED. It is a spawn
            # transient -- the generator's history ring is empty, so the reward starts near 0.5
            # and climbs to ~0.90 by the end of the hold, landing near 0.80 whatever the episode
            # goes on to do. Folding it in drags every episode toward that value and compresses
            # the contrast the reward exists to express. The index's `r` excludes it.
            ax.axvspan(0.0, hold_end, color="0.85", alpha=0.45, zorder=0, lw=0)
        if probe.first_violation_step >= 0:
            ax.axvline(probe.first_violation_step * dt, color="red", lw=1.4, alpha=0.8)
        for check, color in (("lateral", "#e08a1a"), ("height", "#3d8b7d"), ("tilt", "#b0413e")):
            step = probe.first_check_step.get(check, -1)
            if step >= 0 and step != probe.first_violation_step:
                ax.axvline(step * dt, color=color, lw=1.0, ls="--", alpha=0.5)
        if goal_step >= 0:
            ax.axvline(goal_step * dt, color="green", lw=1.2, alpha=0.7)

    fig.suptitle(title, fontsize=11)
    # Reserve the top strip for the suptitle explicitly: `tight_layout` does not account for it,
    # so the first panel's own title lands on top of it.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    if out_png:
        os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=140)
        print(f"[watch] wrote {out_png}")
    if show:
        print("[watch] close the plot window to continue.", flush=True)
        try:
            plt.show()
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] cannot open a plot window ({type(exc).__name__}: {exc}); "
                  "pass --plot <dir> to write it instead.")
    plt.close(fig)
    return out_png


#: Peak-excursion channels, in the order the index line prints them.
_PEAK_KEYS = ("down", "lat", "hgt", "tilt")


def episode_peaks(probe) -> dict[str, float]:
    """The largest excursion this episode reached on each check channel.

    The number that decides a bound.  A quantile over a POPULATION says what the policy
    typically does; it cannot say whether a 0.30 m downrange excursion was a stumble or a
    perfectly good stride, and the population it is taken over is "episodes the OLD metric
    passed" -- which is the metric being replaced.  So the workflow is to sort by these, watch
    the extremes, and decide.

    Args:
        probe: A folded :class:`~mjeval.tracking.TrackingProbe` with ``keep_trace``.

    Returns:
        ``{"down", "lat", "hgt"}`` in metres (absolute) and ``"tilt"`` in degrees, all NaN on
        an episode that served no reference.
    """
    err = probe.trace_array().astype(np.float64)
    if err.shape[0] == 0:
        return dict.fromkeys(_PEAK_KEYS, float("nan"))
    return {"down": float(np.abs(err[:, 0]).max()), "lat": float(np.abs(err[:, 1]).max()),
            "hgt": float(np.abs(err[:, 2]).max()),
            "tilt": float(np.degrees(err[:, 3].max()))}


def _describe(idx: int, seed: int, verdict: str, failure: str, probe, dt: float) -> str:
    """One line summarising an episode, for the index and the prompt."""
    latch = probe.first_violation_step
    latch_s = "none" if latch < 0 else f"{latch * dt:.2f}s"
    checks = ",".join(k for k, v in probe.first_check_step.items() if v >= 0) or "-"
    pk = episode_peaks(probe)
    tag = ""
    if latch >= 0 and verdict == "success":
        # The case that motivated latching rather than terminating: it went wrong, recovered,
        # and arrived. Called out by name so it is not read as an inconsistency.
        tag = "  <- VIOLATED THEN RECOVERED"
    return (f"[{idx:>3}] seed {seed:>3} {verdict:<8}{failure:<6}"
            f"steps {probe.n_folded:>4} r {probe.mean_step_reward:.3f}  "
            f"peak down {pk['down']:.3f} lat {pk['lat']:.3f} hgt {pk['hgt']:.3f} "
            f"tilt {pk['tilt']:5.1f}d  latch {latch_s:<7}{checks:<9}{tag}")


def main() -> int:
    """Entry point."""
    import terrains as _terrains  # noqa: PLC0415

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile", type=int, default=20)
    p.add_argument("--speed", type=float, default=1.6)
    p.add_argument("--seeds", default="0-7",
                   help='Which episodes. "0-7", "0,3,9" or "0-3,7".')
    p.add_argument("--tracker_onnx", required=True)
    p.add_argument("--fm_onnx", required=True)
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--spec", default="",
                   help="tracking_spec.json from measure_tracking_error.py --stage report. "
                        "Without one the BOOTSTRAP bounds are used, and the header says so -- "
                        "which is the point of watching: they are placeholders until this "
                        "script and that one have been read together.")
    p.add_argument("--klass", default="",
                   help="Override the course class inferred from the tile's manifest family. "
                        "Needed for jump_off_front, which klass_for_family REFUSES by design "
                        "(it dismounts, which a forward-facing camera cannot see, and "
                        "mjeval.goal_anchor excludes it) -- pass --klass box to watch it "
                        "anyway, and read the traversal semantics as borrowed rather than "
                        "validated for that family.")
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    p.add_argument("--scene_tag", default="_watchtrk",
                   help="Suffix on the generated scene filename. CourseRig writes "
                        "robots/g1/<robot>_<course>.xml before loading it, so two concurrent "
                        "runs on one tile race on the same path and tear it silently.")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--cmd_hold_s", type=float, default=_terrains.CMD_HOLD_S,
                   help="Seconds of zero command before the commanded speed. Defaults to "
                        "terrains.CMD_HOLD_S; 0 reproduces anything measured before 58d0951c "
                        "and is a spawn transient, not a policy failure.")
    p.add_argument("--rt", type=float, default=0.5, help="Viewer real-time multiple.")
    p.add_argument("--loop", action="store_true", help="Loop the replay instead of holding.")
    p.add_argument("--no_view", action="store_true",
                   help="Never open the viewer. With --plot this is the batch path.")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--plot", default="", help="Directory to write one PNG per episode.")
    p.add_argument("--only", default="",
                   help="Filter which episodes are stepped through: 'violated', 'recovered' "
                        "(violated but still arrived), 'failed', 'survived', or '' for all. "
                        "The whole reason for the filter is that a bound is chosen by looking "
                        "at the episodes it DISAGREES with the verdict on.")
    p.add_argument("--sort", default="seed",
                   choices=("seed", "down", "lat", "hgt", "tilt", "reward"),
                   help="Order to step through in. WORST FIRST for the four excursion "
                        "channels, lowest-reward first for 'reward'. This is the point of the "
                        "tool: a quantile over a population cannot say whether a 0.30 m "
                        "downrange excursion was a stumble or a good stride, so sort by it and "
                        "watch the extremes.")
    args = p.parse_args()

    import runner as mjrunner  # noqa: PLC0415
    from fm_noise_attribution import build_arm  # noqa: PLC0415

    spec = BOOTSTRAP_SPEC
    if args.spec:
        import json  # noqa: PLC0415
        with open(args.spec) as f:
            spec = TrackingErrorSpec.from_dict(json.load(f))
    print(f"[watch] spec: {'BOOTSTRAP (un-fitted placeholders)' if spec.is_bootstrap else args.spec}"
          f"  lateral {spec.max_lateral_m:.2f} m, height {spec.max_height_m:.2f} m, "
          f"tilt {np.degrees(spec.max_tilt_rad):.1f} deg  "
          "(downrange and yaw are graded, never fatal)")

    course = _terrains.isaac_tile_course(args.clip_lib, args.tile, speeds=(args.speed,),
                                         klass=(args.klass or None),
                                         goal_anchor_max_offset=args.max_offset)
    course = dataclasses.replace(course, name=f"tile{args.tile}{args.scene_tag}")
    cfg, policy, fm_lag, spawn, rig = build_arm(
        types.SimpleNamespace(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                              out=(args.plot or "."), providers=args.providers,
                              fall_height=args.fall_height, fall_tilt_deg=args.fall_tilt_deg,
                              cmd_hold_s=args.cmd_hold_s), course)
    dt = float(policy.get_dt())
    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)

    seeds = parse_seeds(args.seeds)
    print(f"[watch] tile {args.tile} ({course.klass}) @ {args.speed} m/s, "
          f"{len(seeds)} episode(s), cmd_hold {args.cmd_hold_s:g} s", flush=True)

    episodes = []
    for seed in seeds:
        frames, plans, verdict, failure, probe = roll_capture(
            rig, policy, fm_lag, spawn, cmd, cfg, seed, mjrunner, spec, args.cmd_hold_s)
        # The goal step is not returned by run_episode, and re-deriving it from the course is
        # cheaper and less fragile than widening that signature for a viewer marker.
        goal_step = -1
        for i, (_t, qpos) in enumerate(frames):
            if course.in_goal(float(qpos[0]), float(qpos[1])):
                goal_step = i
                break
        episodes.append({"seed": seed, "frames": frames, "plans": plans, "verdict": verdict,
                         "failure": failure, "probe": probe, "goal_step": goal_step})
        print(_describe(len(episodes) - 1, seed, verdict, failure, probe, dt), flush=True)

    keep = list(range(len(episodes)))
    if args.only:
        def _match(e) -> bool:
            """Whether an episode passes --only."""
            v = e["probe"].first_violation_step >= 0
            if args.only == "violated":
                return v
            if args.only == "recovered":
                return v and e["verdict"] == "success"
            if args.only == "failed":
                return e["verdict"] != "success"
            if args.only == "survived":
                return e["verdict"] == "success"
            raise SystemExit(f"--only {args.only!r} is not one of "
                             "violated / recovered / failed / survived")
        keep = [i for i, e in enumerate(episodes) if _match(e)]
        print(f"[watch] --only {args.only}: {len(keep)} of {len(episodes)} episode(s)")
        if not keep:
            return 0

    if args.sort != "seed":
        # Worst first. NaN (an episode that served no reference) sorts last rather than
        # first, so a degenerate episode never opens the walkthrough.
        def _key(i: int) -> float:
            """Sort key; negated for excursions so the largest comes first."""
            if args.sort == "reward":
                r = episodes[i]["probe"].mean_step_reward
                return float("inf") if np.isnan(r) else r
            v = episode_peaks(episodes[i]["probe"])[args.sort]
            return float("inf") if np.isnan(v) else -v
        keep.sort(key=_key)
        print(f"\n[watch] sorted by {args.sort}, worst first:")
        for pos, i in enumerate(keep):
            e = episodes[i]
            print(f"  {pos:>3}. " + _describe(i, e["seed"], e["verdict"], e["failure"],
                                              e["probe"], dt))

    def _png(e) -> str | None:
        """Where this episode's figure goes, or None."""
        if not args.plot:
            return None
        return os.path.join(args.plot, f"trk_t{args.tile}_s{args.speed:g}"
                                       f"_seed{e['seed']}.png")

    def _title(e) -> str:
        """Figure title carrying everything needed to identify the run."""
        return (f"tile {args.tile} ({course.klass}) @ {args.speed:g} m/s, seed {e['seed']} -- "
                f"{e['verdict']}{(' / ' + e['failure']) if e['failure'] else ''}   "
                f"[red = latch, green = goal]")

    if args.no_view:
        for i in keep:
            e = episodes[i]
            if not args.no_plot:
                plot_episode(e["probe"], dt, spec, _title(e), out_png=_png(e), show=False,
                             goal_step=e["goal_step"], frames=e["frames"])
        print("[watch] --no_view: figures written, viewer not opened.")
        return 0

    import view_cell  # noqa: PLC0415

    pos = 0
    while 0 <= pos < len(keep):
        i = keep[pos]
        e = episodes[i]
        print("\n" + _describe(i, e["seed"], e["verdict"], e["failure"], e["probe"], dt))
        summary = {"course": course, "speed": args.speed, "verdict": e["verdict"],
                   "failure": e["failure"], "row": {}, "model": rig.robot.mj_model}
        view_cell.replay(summary, e["frames"], e["plans"], args.rt, args.loop)
        # AFTER the replay: the MuJoCo viewer owns the GL context and blocks until closed, so a
        # figure opened first would sit behind it unseen.
        if not args.no_plot:
            plot_episode(e["probe"], dt, spec, _title(e), out_png=_png(e), show=True,
                         goal_step=e["goal_step"], frames=e["frames"])
        try:
            ans = input(f"[{pos + 1}/{len(keep)}] enter/n next, p prev, r replay, "
                        "<number> jump, q quit > ").strip().lower()
        except EOFError:
            break
        if ans in ("q", "quit"):
            break
        if ans in ("p", "prev"):
            pos = max(pos - 1, 0)
        elif ans in ("r", "replay"):
            continue
        elif ans.isdigit():
            want = int(ans)
            pos = keep.index(want) if want in keep else pos + 1
        else:
            pos += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
