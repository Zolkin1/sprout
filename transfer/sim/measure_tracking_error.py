# Copyright 2026 Zachary Olkin. All rights reserved.

"""What the tracking error against the SERVED reference actually looks like, per family.

Phase 0 of the strict-failure work, and it exists because the alternative is choosing a
threshold by intuition.  The precedent is ``scripts/rsl_rl/build_goal_regions.py``, which fits
the goal region from where surviving CLIP-arm episodes actually end rather than declaring a
tolerance -- and the counter-precedent is ``base_pose_error_termination``'s 0.25 m, which the
spatial-success work had to call out as "not" the right number for this question.

**"SURVIVOR" DOES NOT MEAN "GOOD", and every number here has to be read with that in mind.**
The split below is by the OLD verdict -- reached the goal without tripping the terrain-height or
70-deg-from-upright test -- which is precisely the criterion being replaced because it passes a
robot that stumbles down a flight and barely catches itself.  So a quantile over survivors says
*what this policy currently does*, never *what is acceptable*.  It is the right thing to size a
soft kernel width against, and it is NOT sufficient to choose a hard bound: for that, sort by the
peak excursion in ``watch_tracking.py`` and look at the motions.  The falls are reported beside
the survivors so the SEPARATION is visible -- a bound that does not separate the two populations
is not measuring what it claims to.

**The three checks are reported separately and must be read separately.**  Under
``window_anchor="plan"``:

* LATERAL and HEIGHT are chunk-local -- the served node is expressed against the robot's own
  root at the last re-plan, so an excursion has to accumulate inside 0.24 s.  Expect a sawtooth
  whose amplitude is the interesting statistic.
* the TILT cone is not.  The anchor contributes YAW ONLY, so the reference's roll and pitch are
  absolute against gravity and the error is un-rezeroed.  It fires far more readily than the
  position checks, and if it fires on ordinary stair descent it is the threshold to move first.
* DOWNRANGE carries no bound at all and is reported for context only.  The generator moves the
  reference along the direction of travel to MATCH THE TERRAIN -- a re-plan that lands it on the
  next tread jumps it forward, and the tracker then cannot be anything but behind it.

Two stages, sharing one JSON:

* ``--stage run`` needs MuJoCo and the ONNX pair; rolls episodes and stores every error row.
* ``--stage report`` needs neither, and prints the per-family quantiles, the proposed spec, and
  how far each candidate bound departs from today's verdict.

Usage::

    python measure_tracking_error.py --stage run --out logs/tracking/v1 \\
        --tracker_onnx <...>/exported/tracker.onnx --fm_onnx <...>_fm.onnx \\
        --clip_lib clips/g1_mb_v5_body_mode11_decor.h5 \\
        --tiles 6,20,40 --speeds 0.8,1.6,2.4 --episodes 10 --device cuda
    python measure_tracking_error.py --stage report --out logs/tracking/v1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from deploy.tracking_error import (  # noqa: E402
    NUM_FIXED_CHANNELS,
    TrackingErrorSpec,
    error_channel_names,
)

#: Quantiles printed for every channel.  p99.9 rather than the max, for the reason
#: ``fm_clutter_sensitivity.py`` records: across two draws of one checkpoint its max moved
#: 0.0066 -> 0.0241 and swung the headline 47% -> 13%, where p99.9 held to five points.
_QUANTILES = (0.5, 0.9, 0.99, 0.999)

#: Candidate bounds swept in the report, so the chosen one is read off a curve rather than
#: asserted.  Metres for the two position checks, degrees for the cone.  **There is no downrange
#: sweep**: that axis carries no bound at all, because a terrain-matched re-plan moves the
#: reference forward on purpose (see ``deploy.tracking_error``).
_LATERAL_CANDIDATES = (0.05, 0.075, 0.10, 0.15, 0.20)
_HEIGHT_CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.30)
_TILT_CANDIDATES_DEG = (15.0, 20.0, 25.0, 30.0, 40.0, 50.0)


def _episode_rows(trace: np.ndarray, num_joints: int) -> dict[str, np.ndarray]:
    """Split an error trace into the summary channels the report reasons about.

    The per-joint block is reduced to its RMS here rather than carried: 29 channels per step is
    what makes a trace large, and no threshold is ever set per joint.

    Args:
        trace: ``[steps, NUM_FIXED_CHANNELS + nj]``.
        num_joints: ``nj``.

    Returns:
        Named ``[steps]`` arrays: the three position axes (SIGNED, because a non-zero mean is
        the thing that would make a symmetric bound wrong), their absolute values, the position
        L2 norm, the tilt and yaw angles, the two twist magnitudes and the joint RMS.
    """
    t = np.asarray(trace, dtype=np.float64)
    if t.shape[-1] != NUM_FIXED_CHANNELS + num_joints:
        raise ValueError(
            f"trace is {t.shape[-1]} wide but nj={num_joints} implies "
            f"{NUM_FIXED_CHANNELS + num_joints}")
    return {
        "d_downrange": t[:, 0], "d_lateral": t[:, 1], "d_height": t[:, 2],
        "abs_downrange": np.abs(t[:, 0]), "abs_lateral": np.abs(t[:, 1]),
        "abs_height": np.abs(t[:, 2]),
        # L2, not an inf-norm: this is what `sigma_pos_m` is the width OF (`r_pos` is
        # `exp(-||dp||^2 / sigma^2)`), so fitting the width against any other reduction would
        # size the kernel for a quantity it never sees. It includes downrange even though
        # downrange carries no BOUND -- the reward grades it, only the latch ignores it.
        "pos_norm": np.linalg.norm(t[:, 0:3], axis=-1),
        "theta_tilt_deg": np.degrees(t[:, 3]),
        "abs_theta_yaw_deg": np.degrees(np.abs(t[:, 4])),
        "lin_vel_err": np.linalg.norm(t[:, 5:8], axis=-1),
        "ang_vel_err": np.linalg.norm(t[:, 8:11], axis=-1),
        "joint_rms": np.sqrt(np.mean(t[:, NUM_FIXED_CHANNELS:] ** 2, axis=-1)),
    }


def run(args) -> int:
    """Roll episodes and store every per-step error row.

    Args:
        args: Parsed CLI.

    Returns:
        Process exit code.
    """
    import terrains  # noqa: PLC0415

    import runner as mjrunner  # noqa: PLC0415
    from fm_noise_attribution import build_arm  # noqa: PLC0415
    from tracking import TrackingProbe  # noqa: PLC0415

    tiles = [int(t) for t in args.tiles.split(",") if t.strip()]
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)

    episodes: list[dict] = []
    traces: list[np.ndarray] = []
    num_joints = -1

    for tile in tiles:
        for speed in speeds:
            course = terrains.isaac_tile_course(
                args.clip_lib, tile, speeds=(speed,),
                goal_anchor_max_offset=args.max_offset)
            # The scene file is written from the course before it is loaded, and two runs on
            # one tile at different speeds produce IDENTICAL geometry -- so concurrent
            # invocations race on the same path and tear it silently. Same reason
            # `fm_variance_probe` grew --scene_tag.
            course = dataclasses.replace(
                course, name=f"tile{tile}_s{speed:g}{args.scene_tag}")
            cfg, policy, fm_lag, spawn, rig = build_arm(
                types.SimpleNamespace(
                    tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx, out=args.out,
                    providers=args.providers, fall_height=args.fall_height,
                    fall_tilt_deg=args.fall_tilt_deg, cmd_hold_s=args.cmd_hold_s), course)
            if num_joints < 0:
                num_joints = len(policy.get_joint_names())
            cmd = np.array([speed, 0.0, 0.0])
            print(f"[run] tile {tile} @ {speed:g} m/s  klass={course.klass}  "
                  f"budget {course.budget_s:.1f} s  hold {args.cmd_hold_s:g} s", flush=True)

            for seed in range(args.episodes):
                probe = TrackingProbe(keep_trace=True,
                                      hold_until_s=float(args.cmd_hold_s))

                step = [0]

                def _hook(robot, pol, state, is_replan, _p=probe, _s=step):
                    """Fold one control step; fired where `acc.fold` fires, pre-advance.

                    Stamped with the CONTROL-step index, not the folded-row count: the trace
                    starts at the generator's first plan, so the two differ by the length of
                    the opening hold and the latch has to be on the episode's own clock.
                    """
                    _p.fold(robot, pol, _s[0], is_replan=is_replan)
                    _s[0] += 1

                # Seeded BEFORE the call: `run_episode` resets the sim and the policy's
                # buffers but does not seed, so this is what makes the trajectory the one that
                # seed produced.
                policy._rng = np.random.default_rng(seed)
                verdict, failure, _acc = mjrunner.run_episode(
                    rig, policy, fm_lag, spawn, cmd, None, steer_kp=cfg.steer_kp,
                    steer_wz_max=cfg.steer_wz_max, cfg_hold=args.cmd_hold_s, probe=_hook)
                probe.finish()
                tr = probe.trace_array()
                episodes.append({
                    "tile": tile, "speed": speed, "seed": seed, "klass": course.klass,
                    "verdict": verdict, "failure": failure,
                    "n_steps": int(tr.shape[0]), "row0": int(sum(t.shape[0] for t in traces)),
                    "first_violation_step": probe.first_violation_step,
                    "first_check_step": dict(probe.first_check_step),
                    "mean_step_reward": probe.mean_step_reward,
                    "n_chunks": len(probe.chunk_rewards),
                })
                traces.append(tr)
                print(f"  seed {seed:>2}: {verdict:8s}{failure:6s} steps={tr.shape[0]:>4} "
                      f"latch={probe.first_violation_step:>5} "
                      f"r={probe.mean_step_reward:.3f}", flush=True)

    if not traces or num_joints < 0:
        print("[run] nothing was rolled", file=sys.stderr)
        return 1

    stacked = np.concatenate(traces, axis=0)
    np.savez_compressed(os.path.join(args.out, "traces.npz"), trace=stacked,
                        num_joints=np.int64(num_joints))
    with open(os.path.join(args.out, "measure.json"), "w") as f:
        json.dump({"episodes": episodes, "args": vars(args), "num_joints": num_joints,
                   "channels": list(error_channel_names(num_joints))}, f, indent=1)
    print(f"\n[run] {len(episodes)} episodes, {stacked.shape[0]} steps -> {args.out}")
    return 0


def report(args) -> int:
    """Print the distributions, sweep the candidate bounds and propose a fitted spec.

    Args:
        args: Parsed CLI.

    Returns:
        Process exit code.
    """
    with open(os.path.join(args.out, "measure.json")) as f:
        doc = json.load(f)
    blob = np.load(os.path.join(args.out, "traces.npz"))
    trace, nj = blob["trace"], int(blob["num_joints"])
    eps = doc["episodes"]

    # Clean = the episode SUCCEEDED. A width fitted over the falls would be fitted over exactly
    # the behaviour the bound exists to exclude.
    surv_rows, fall_rows = [], []
    for ep in eps:
        rows = trace[ep["row0"]:ep["row0"] + ep["n_steps"]]
        if rows.shape[0] == 0:
            continue
        (surv_rows if ep["verdict"] == "success" else fall_rows).append((ep, rows))

    n_surv, n_fall = len(surv_rows), len(fall_rows)
    print(f"[report] {len(eps)} episodes: {n_surv} success, {n_fall} not  "
          f"({trace.shape[0]} steps, nj={nj})")
    if n_surv == 0:
        print("[report] no surviving episode -- there is nothing to fit a width from, "
              "and fitting over the falls would fit the behaviour being excluded.",
              file=sys.stderr)
        return 1

    def _pool(pairs, key=None):
        """Concatenate the summary channels over a set of (episode, rows) pairs."""
        out: dict[str, list[np.ndarray]] = {}
        for ep, rows in pairs:
            if key is not None and ep["klass"] != key:
                continue
            for name, vals in _episode_rows(rows, nj).items():
                out.setdefault(name, []).append(vals)
        return {k: np.concatenate(v) for k, v in out.items() if v}

    surv = _pool(surv_rows)
    falls = _pool(fall_rows) if n_fall else {}

    print("\n--- SURVIVOR distributions (all families pooled) " + "-" * 29)
    print(f"{'channel':>20s}" + "".join(f"{f'p{q * 100:g}':>10s}" for q in _QUANTILES)
          + f"{'mean':>10s}" + (f"{'p99 fall':>10s}" if falls else ""))
    for name in ("abs_downrange", "abs_lateral", "abs_height", "pos_norm",
                 "theta_tilt_deg", "abs_theta_yaw_deg", "lin_vel_err", "ang_vel_err",
                 "joint_rms"):
        v = surv[name]
        line = f"{name:>20s}" + "".join(f"{np.quantile(v, q):10.4f}" for q in _QUANTILES)
        line += f"{v.mean():10.4f}"
        if falls:
            line += f"{np.quantile(falls[name], 0.99):10.4f}"
        print(line)

    # A non-zero MEAN on a signed axis is what would make a symmetric box the wrong shape --
    # it would condemn one direction earlier than the other for a reason that is the
    # generator's node-0 offset rather than the robot's behaviour.
    print("\n--- signed position error: is the box centred? " + "-" * 32)
    for name in ("d_downrange", "d_lateral", "d_height"):
        v = surv[name]
        print(f"{name:>20s}  mean {v.mean():+8.4f}   median {np.median(v):+8.4f}   "
              f"p1 {np.quantile(v, 0.01):+8.4f}   p99 {np.quantile(v, 0.99):+8.4f}")

    print("\n--- per family, the three checks (survivors) " + "-" * 35)
    families = sorted({ep["klass"] for ep, _ in surv_rows})
    print(f"{'family':>16s}{'n_ep':>6s}{'lat p99.9':>11s}{'hgt p99.9':>11s}"
          f"{'tilt p99.9':>12s}{'down p99.9':>12s}")
    for fam in families:
        sub = _pool(surv_rows, fam)
        n = sum(1 for ep, _ in surv_rows if ep["klass"] == fam)
        print(f"{fam:>16s}{n:>6d}"
              f"{np.quantile(sub['abs_lateral'], 0.999):11.4f}"
              f"{np.quantile(sub['abs_height'], 0.999):11.4f}"
              f"{np.quantile(sub['theta_tilt_deg'], 0.999):12.3f}"
              f"{np.quantile(sub['abs_downrange'], 0.999):12.4f}")
    print("  (the downrange column is reported for context only -- that axis carries NO bound)")

    # The sweep. "How many surviving episodes would this bound have condemned" is the number
    # that decides a threshold. Read it knowing what "survivor" means: an episode the OLD
    # metric passed, which is the metric being replaced -- so this says how far a bound departs
    # from today's verdict, NOT how many good motions it rejects. Only watching says that.
    print("\n--- what each candidate bound would condemn " + "-" * 35)
    print("  (share of episodes latching on this check alone, by OLD verdict)")
    print(f"{'bound':>16s}{'survivor latched':>18s}{'fall latched':>15s}")
    for col, name, cands in ((1, "lateral", _LATERAL_CANDIDATES),
                             (2, "height", _HEIGHT_CANDIDATES)):
        for b in cands:
            c = np.mean([np.any(np.abs(r[:, col]) > b) for _, r in surv_rows])
            f_ = (np.mean([np.any(np.abs(r[:, col]) > b) for _, r in fall_rows])
                  if n_fall else float("nan"))
            print(f"  {name:>7s} {b:5.3f} m {c:16.3f}{f_:15.3f}")
    for deg in _TILT_CANDIDATES_DEG:
        rad = np.radians(deg)
        c = np.mean([np.any(r[:, 3] > rad) for _, r in surv_rows])
        f_ = np.mean([np.any(r[:, 3] > rad) for _, r in fall_rows]) if n_fall else float("nan")
        print(f"  {'tilt':>7s} {deg:5.1f} d {c:16.3f}{f_:15.3f}")

    # The proposal. Widths at the p90 -- so a typical step scores ~exp(-1) on that channel and
    # the kernel has resolution where the data is, rather than saturating at 1.
    q90 = {k: float(np.quantile(surv[k], 0.90)) for k in surv}
    proposed = TrackingErrorSpec(
        max_lateral_m=round(float(np.quantile(surv["abs_lateral"], 0.999)), 3),
        max_height_m=round(float(np.quantile(surv["abs_height"], 0.999)), 3),
        max_tilt_rad=float(np.radians(round(np.quantile(surv["theta_tilt_deg"], 0.999), 1))),
        sigma_pos_m=round(max(q90["pos_norm"], 1e-3), 4),
        sigma_ori_rad=round(max(np.radians(q90["theta_tilt_deg"]), 1e-3), 4),
        sigma_lin_vel=round(max(q90["lin_vel_err"], 1e-3), 4),
        sigma_ang_vel=round(max(q90["ang_vel_err"], 1e-3), 4),
        sigma_joint_rad=round(max(q90["joint_rms"], 1e-3), 4),
        provenance={"bootstrap": False, "source": os.path.abspath(args.out),
                    "n_survivor_episodes": n_surv, "n_fall_episodes": n_fall,
                    "n_survivor_steps": int(sum(r.shape[0] for _, r in surv_rows)),
                    "bounds_at": "survivor p99.9", "widths_at": "survivor p90",
                    "families": families},
    )
    path = os.path.join(args.out, "tracking_spec.json")
    with open(path, "w") as f:
        json.dump(proposed.to_dict(), f, indent=1)
    print("\n--- proposed spec (bounds at survivor p99.9, widths at survivor p90) " + "-" * 7)
    print(json.dumps(proposed.to_dict(), indent=1))
    print(f"\n[report] -> {path}")
    print("[report] This is a PROPOSAL. Read it against the sweep above and against tile 20 "
          "in the viewer before adopting it; the point of Phase 0 is that these numbers are "
          "chosen from a distribution, not that a script picked a quantile.")
    return 0


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("run", "report"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracker_onnx", default="")
    p.add_argument("--fm_onnx", default="")
    p.add_argument("--clip_lib", default="")
    p.add_argument("--tiles", default="6,20,40")
    p.add_argument("--speeds", default="0.8,1.6,2.4")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider",
                   help="onnxruntime execution providers, in order. Defaulted to CUDA-first "
                        "rather than left empty because the fallback is SILENT -- a run that "
                        "loses the CUDA provider prints a provider line and then goes ~5x "
                        "slower with no error, which is how a whole sweep was once run on CPU.")
    p.add_argument("--scene_tag", default="_trkerr",
                   help="Suffix on the generated scene filename. CourseRig writes "
                        "robots/g1/<robot>_<course>.xml before loading it, and two concurrent "
                        "runs on one tile produce identical geometry -- so they race on the "
                        "same path and tear it silently rather than failing to load.")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--cmd_hold_s", type=float, default=None,
                   help="Seconds of zero command before the commanded speed. Defaults to "
                        "terrains.CMD_HOLD_S. Anything measured at 0 is measuring a spawn "
                        "transient: tile 20 at 2.4 m/s scores 0.275 at hold 0 and 0.675 at 0.5.")
    args = p.parse_args()

    if args.stage == "run":
        if args.cmd_hold_s is None:
            import terrains  # noqa: PLC0415
            args.cmd_hold_s = float(terrains.CMD_HOLD_S)
        for required in ("tracker_onnx", "fm_onnx", "clip_lib"):
            if not getattr(args, required):
                p.error(f"--{required} is required for --stage run")
        return run(args)
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
