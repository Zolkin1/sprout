#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Mode-selection probe: N noise draws on ONE Isaac tile, then a human labels the first mode.

Four stages, each a separate invocation so the human half is resumable:

    # 1. run both arms, headless.  Only the flow-noise draw differs between trials.
    python transfer/sim/mode_probe.py --stage run --out logs/mode/box2 --arm parent \\
        --clip clips/box2_v2spawn_train.h5 --terrain_id 2 --speed 1.17 --trials 50 \\
        --tracker_onnx <...>/tracker.onnx --fm_onnx <...>/parent_fm.onnx
    python transfer/sim/mode_probe.py --stage run --out logs/mode/box2 --arm finetuned ...

    # 2a. just LOOK at them -- arm shown, fixed order, nothing written
    python transfer/sim/mode_probe.py --stage watch --out logs/mode/box2

    # 2b. label them, BLIND and interleaved, in the viewer
    python transfer/sim/mode_probe.py --stage label --out logs/mode/box2

    # 3. compare
    python transfer/sim/mode_probe.py --stage report --out logs/mode/box2

**The terrain is the ISAAC one, by construction rather than by agreement.**
``terrains.isaac_tile_course`` builds the MuJoCo scene from the clip library's own
``terrain_manifest`` -- the same producer the Isaac ``ClipTerrainImporter`` reads -- so there is
no second description of the geometry to drift.  This is deliberately NOT one of mjeval's
authored courses: those are hand-written to vary one parameter and are a different world from
whatever the Isaac tile currently is.

**The spawn is the tile's OWN route spawn**, not mjeval's default 2.5 m approach: the approach
distance is recovered from the manifest's ``route.start_x`` against the first feature, so the
robot opens where an Isaac episode on this tile would open.

**Capture is headless and replay is passive, and that is not a convenience.**  Attaching the
viewer perturbs ``mjData`` enough to change the episode -- ``view_cell``'s own docstring records
a fall moving to 6.04 s with the viewer attached -- so a human watching a LIVE run would be
labelling a different episode from the one that was scored.  Playback writes ``qpos`` and calls
``mj_forward`` only.

**Labelling is BLIND and interleaved.**  Both arms' trials are shuffled into one sequence and
presented by index; the arm is not shown and is not recoverable from the screen.  A human who
knows which arm they are watching cannot label its mode neutrally, and the whole point of the
measurement is a human judgement.

**Only the FIRST feature is labelled.**  The replay window ends
``--label_margin_m`` past ``Course.obstacle_x_m`` -- the course's own "world x of the first
riser / box face" -- so what is being judged is the first mode decision (onto the first box, up
the first flight) and not the rest of the traverse.  The full trajectory is still saved, and
``v`` in the labeller plays it.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent / "source" / "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mjeval import runner as mjrunner  # noqa: E402
from mjeval.keys import cbreak_stdin, drain_keys, poll_key
from mjeval import terrains  # noqa: E402
from robot import Robot  # noqa: E402

#: The mode labels a human may give.  ``flat`` / ``stairs`` / ``box`` are the three modes the
#: generator can plan; ``unclear`` exists because forcing a three-way choice on a trial that
#: fell before it committed would put a fabricated decision into the data.
MODES = ("flat", "stairs", "box", "unclear")

#: Single-key answers, plus the controls.
KEYS = {"f": "flat", "s": "stairs", "b": "box", "x": "unclear"}

#: Manifest family stem -> mjeval course class.  ``FAMILY_KLASS`` covers the clip-library
#: families only, and these two courses are not clip families, so the mapping lives here and
#: ``--klass`` overrides it.  The class also declares the mode a CORRECT first decision would
#: be, which is what the report scores against.
FAMILY_KLASS: dict[str, str] = {
    "box2": "two_box_course",
    "box2_v2": "two_box_course",
    "varied_boxes": "two_box_course",
    "updown": "up_into_down",
    "varied_updown": "up_into_down",
}

#: The mode a correct FIRST decision is, per course class.
EXPECTED_MODE: dict[str, str] = {
    "two_box_course": "box",
    "up_into_down": "stairs",
    "box": "box",
    "stairs_up": "stairs",
    "stairs_down": "stairs",
}


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=("run", "watch", "label", "report", "migrate"))
    p.add_argument("--out", required=True, help="Probe directory, shared by every stage.")
    # --- run ---
    p.add_argument("--arm", default="", help="Arm name for --stage run, e.g. parent.")
    p.add_argument("--clip", default="",
                   help="Clip library whose manifest defines the tile. Also accepted by "
                        "--stage label to re-point the path when the trials were rolled on "
                        "another machine.")
    p.add_argument("--terrain_id", type=int, default=-1, help="Which tile.")
    p.add_argument("--tracker_onnx", default="", help="Absolute path to tracker.onnx.")
    p.add_argument("--fm_onnx", default="", help="Absolute path to this arm's generator.")
    p.add_argument("--speed", type=float, default=1.17, help="Commanded forward speed (m/s).")
    p.add_argument("--trials", type=int, default=50, help="Noise draws.")
    p.add_argument("--seed_start", type=int, default=0,
                   help="--stage run: roll seeds [seed_start, trials) and APPEND them to the "
                        "arm's existing trials instead of re-rolling from 0. The seed IS the "
                        "trial index, so extending a 25-trial probe to 50 is "
                        "`--trials 50 --seed_start 25`: the first 25 trajectories are left "
                        "untouched on disk and keep their labels.")
    p.add_argument("--obstacle_from", default="raised",
                   choices=("raised", "tread", "drop"),
                   help="What the approach and the label window are measured to. 'raised' "
                        "(default) is the first walkable overlay ABOVE the spawn, which is "
                        "the ascent case and the only one that works on a box course. 'drop' "
                        "is the first one BELOW it -- required where the robot spawns ON a "
                        "platform and what it faces is a DESCENT, as the entry_stairs_down "
                        "family does: 'raised' there finds the flight BEHIND the spawn and "
                        "returns a NEGATIVE approach, which silently collapses the label "
                        "window to a single frame rather than failing. 'tread' is mjeval's "
                        "original first-stair-tread finder, filtered to 0.60 m and therefore "
                        "blind to a 1.20 m box.")
    p.add_argument("--steer_kp", type=float, default=0.0,
                   help="P gain on heading error toward the goal (1/s); 0 (default) leaves "
                        "the probe unsteered, which is what every probe before this flag "
                        "was. Set it to MATCH THE TASK when the probe is being read against "
                        "an Isaac number: the free-form fine-tune runs "
                        "`fm_tracking_cmd.freeform_yaw_rate` at kp=1.0, and it wires a 0.75 m "
                        "goal band BECAUSE that controller runs -- so scoring an unsteered "
                        "rollout against that band charges it for the absence of a "
                        "controller the task has. Still off by default, and still "
                        "off-distribution on the stair families (|wz| max = 0.0000 over 3500 "
                        "stair clips), so this is a boxes/flat instrument.")
    p.add_argument("--steer_law", default="goal", choices=("goal", "centreline"),
                   help="WHICH steering law --steer_kp drives. 'goal' (default, what every "
                        "probe before this flag used) is mjeval's `steer_yaw_rate`: bearing "
                        "straight at the course goal, so the effective cross-track gain is "
                        "kp/distance_to_goal and it STIFFENS as the robot closes. 'centreline' "
                        "is the Isaac free-form law `fm_tracking_cmd.freeform_yaw_rate`: aim at "
                        "a point --steer_lookahead_m ahead of the robot's own projection onto "
                        "the route centreline, so the gain is a constant kp/lookahead. The two "
                        "are NOT the same controller at the same kp -- use 'centreline' when "
                        "the probe is being read against an Isaac number.")
    p.add_argument("--steer_lookahead_m", type=float, default=5.0,
                   help="Look-ahead for --steer_law centreline (m). Isaac's default is 5.0, "
                        "which with kp=1.0 is a 0.2 1/s cross-track gain. Ignored by the "
                        "'goal' law, which has no look-ahead.")
    p.add_argument("--klass", default="", choices=("", *terrains.CLASS_CODES),
                   help="Override the course class inferred from the manifest family.")
    p.add_argument("--label_margin_m", type=float, default=2.0,
                   help="End the label window this far past the first feature (m). 2.0 covers "
                        "a 1.2 m box top and a 6 x 0.25 m flight, so the first decision has "
                        "fully played out either way.")
    p.add_argument("--providers", default="auto", help="onnxruntime providers.")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--scene_tag", default="",
                   help="Suffix for the generated scene file. REQUIRED when two probes run "
                        "concurrently on one tile: `CourseRig` writes "
                        "`robots/g1/<robot>_<course>.xml` before loading it, and identical "
                        "geometry means the race tears the file silently.")
    # --- label ---
    p.add_argument("--rt", type=float, default=1.0,
                   help="Replay speed multiple; 1.0 is real time, 0.5 half speed.")
    p.add_argument("--passes", type=int, default=1,
                   help="Times to play a trial before falling back to a blocking prompt. A "
                        "key pressed DURING playback is taken immediately, so this is only "
                        "the budget for a trial you have not judged yet.")
    p.add_argument("--shuffle_seed", type=int, default=20260909,
                   help="Fixes the blinded presentation order, so a resumed session continues "
                        "the same sequence.")
    p.add_argument("--json", default="",
                   help="--stage report: also write the reduced numbers here, for the figure. "
                        "The plot script reads these rather than importing this module, which "
                        "would drag mjeval and mujoco into a plotting job.")
    p.add_argument("--by_arm", action="store_true",
                   help="--stage watch: walk one arm end to end instead of interleaving the "
                        "arms seed by seed. The default pairs them, because the same seed on "
                        "two arms is the SAME flow-noise draw through a different generator.")
    p.add_argument("--window_only", action="store_true",
                   help="--stage watch: open on the label window instead of the "
                        "whole episode. No effect on --stage label, which is "
                        "window-first already.")
    p.add_argument("--whole", action="store_true",
                   help="--stage label: open every trial on the WHOLE episode instead of the "
                        "label window. The window ends `--label_margin_m` past the FIRST "
                        "feature, which on a two-box course is between the boxes, so a "
                        "decision made at the second one is not in it at all. Note what this "
                        "changes about the DATA: `expected_mode` and the report's "
                        "correct-mode rate are defined on the FIRST decision, so a label "
                        "given for a later one is a different measurement under the same "
                        "column name.")
    p.add_argument("--relabel", action="store_true",
                   help="Re-ask trials that already carry a label.")
    return p


# ---------------------------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------------------------
def tile_entry(clip: str, terrain_id: int) -> dict:
    """The manifest entry for one tile.

    Args:
        clip: Clip library path.
        terrain_id: Tile id.

    Returns:
        The manifest dict for that tile.

    Raises:
        ValueError: If the library carries no manifest or no such tile.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(clip, "r") as f:
        raw = f.attrs.get("terrain_manifest")
    if raw is None:
        raise ValueError(f"{clip} carries no terrain_manifest; there is no tile to build.")
    man = json.loads(raw) if isinstance(raw, (str, bytes)) else list(raw)
    for e in man:
        if int(e.get("terrain_id", -1)) == int(terrain_id):
            return e
    raise ValueError(f"{clip} has no tile {terrain_id} (it has {len(man)}).")


def resolve_klass(entry: dict, override: str) -> str:
    """The mjeval course class for a tile, from ``--klass`` or the manifest family."""
    if override:
        return override
    fam = str(entry.get("family", "")).split(".")[0]
    for stem, k in FAMILY_KLASS.items():
        if fam == stem or fam.startswith(stem):
            return k
    raise ValueError(
        f"family {fam!r} maps to no course class; pass --klass explicitly. Known: "
        f"{sorted(FAMILY_KLASS)}")


def probe_path(out: str) -> str:
    """Path of the probe manifest."""
    return os.path.join(out, "probe.json")


def load_probe(out: str) -> dict:
    """Read the probe manifest, or raise with what to run first."""
    p = probe_path(out)
    if not os.path.isfile(p):
        raise SystemExit(f"no probe manifest at {p}; run --stage run first.")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------
# Stage: run
# ---------------------------------------------------------------------------------------------
def centreline_yaw_rate(course, x: float, y: float, yaw: float, kp: float,
                        wz_max: float, *, lookahead: float) -> float:
    """Isaac's free-form steering law, as a drop-in for mjeval's ``steer_yaw_rate``.

    The port of ``fm_tracking_cmd.freeform_yaw_rate``: aim at the point ``lookahead`` metres
    ahead of the robot's OWN projection onto the route centreline, rather than at the goal.
    That distinction is the whole reason this exists -- bearing-to-goal has an effective
    cross-track gain of ``kp / distance_to_goal``, which stiffens without bound as the robot
    closes on the goal, while this one holds a constant ``kp / lookahead``.  At Isaac's
    defaults (``kp=1.0``, ``lookahead=5.0``) that is 0.2 1/s.

    The route here runs along +x, so the centreline is ``y = course.target_xy[1]`` and the
    robot's own projection onto it is its ``x``; the aim point is therefore
    ``(x + lookahead, cy)`` and the port needs nothing from the tile's frame.  Unlike
    ``steer_yaw_rate`` there is NO in-goal short circuit, because the Isaac law has none.

    Args:
        course: The course, for the centreline's lateral offset.
        x: Root world x (m).
        y: Root world y (m).
        yaw: Root heading (rad).
        kp: Proportional gain (1/s).
        wz_max: Symmetric clip on the output (rad/s).
        lookahead: Distance to the aim point along the centreline (m).

    Returns:
        The commanded yaw rate (rad/s).
    """
    _, cy = course.target_xy
    bearing = math.atan2(cy - y, lookahead)
    err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
    return float(np.clip(kp * err, -wz_max, wz_max))


def stage_run(args) -> int:
    """Roll ``--trials`` noise draws for one arm and save every trajectory."""
    for need, flag in ((args.arm, "--arm"), (args.clip, "--clip"),
                       (args.tracker_onnx, "--tracker_onnx"), (args.fm_onnx, "--fm_onnx")):
        if not need:
            raise SystemExit(f"--stage run needs {flag}.")
    if args.terrain_id < 0:
        raise SystemExit("--stage run needs --terrain_id.")

    # Checked BEFORE anything expensive: the arms of this comparison must differ only in the
    # generator, and a mismatch found after 50 trials is 50 trials wasted.
    pm: dict = {}
    if os.path.isfile(probe_path(args.out)):
        with open(probe_path(args.out), encoding="utf-8") as fh:
            pm = json.load(fh)
        for k, v in (("clip", args.clip), ("terrain_id", int(args.terrain_id)),
                     ("speed", float(args.speed)),
                     ("steer_kp", float(args.steer_kp)),
                     ("steer_law", str(args.steer_law)),
                     ("obstacle_from", str(args.obstacle_from)),
                     ("label_margin_m", float(args.label_margin_m))):
            if k in pm and pm[k] != v:
                raise SystemExit(
                    f"REFUSING: this probe directory was built with {k}={pm[k]!r}, and this "
                    f"run asks for {k}={v!r}. The arms must differ ONLY in the generator, or "
                    "the labels compare two different experiments. Use a new --out.")
        if args.arm in (pm.get("arms") or {}) and int(args.seed_start) == 0:
            print(f"[mode] note: overwriting the existing arm {args.arm!r}")

    entry = tile_entry(args.clip, args.terrain_id)
    klass = resolve_klass(entry, args.klass)
    name = f"modeprobe_t{args.terrain_id}" + (f"_{args.scene_tag}" if args.scene_tag else "")
    # `route` + `raised`: the spawn is the tile's own declared route start, so the robot opens
    # where an Isaac episode opens, and the obstacle datum has no tread-depth filter, without
    # which a 1.20 m box reads as "not a feature course" at all.
    course = terrains.isaac_tile_course(
        args.clip, args.terrain_id, speeds=(args.speed,), klass=klass, name=name,
        spawn_from="route", obstacle_from=args.obstacle_from,
        goal_from="route")
    approach = float(course.obstacle_x_m)
    if approach <= 0.0:
        raise SystemExit(
            f"REFUSING: the first obstacle is {approach:.2f} m AHEAD of the spawn, i.e. "
            f"behind it, under --obstacle_from {args.obstacle_from!r}. The label window "
            f"would end at x={approach + args.label_margin_m:.2f} m, which the first frame "
            "already satisfies, so every trial would be cut to one frame and labelled on "
            "nothing. A tile whose robot spawns ON a platform faces a DESCENT: pass "
            "--obstacle_from drop.")
    label_x = float(course.obstacle_x_m) + float(args.label_margin_m)
    print(f"[mode] tile {args.terrain_id} of {os.path.basename(args.clip)}  "
          f"family {entry.get('family', '?')}  class {klass}")
    print(f"[mode] spawn = the tile's OWN route start; first feature {approach:.3f} m ahead; "
          f"label window ends {args.label_margin_m:.2f} m past it (x={label_x:.3f} m)")
    print(f"[mode] expected first mode: {EXPECTED_MODE.get(klass, '?')}")
    print(f"[mode] arm {args.arm!r}  speed {args.speed} m/s  {args.trials} noise draws  "
          f"steer_kp={args.steer_kp:g}"
          f"{'' if args.steer_kp > 0 else ' (UNSTEERED)'}"
          f"{f' law={args.steer_law}' if args.steer_kp > 0 else ''}"
          f"{f' lookahead={args.steer_lookahead_m:g}m' if args.steer_kp > 0 and args.steer_law == 'centreline' else ''}")

    cfg = mjrunner.ArmRun(
        arm=0, label=f"mode_{args.arm}", tracker_onnx=args.tracker_onnx,
        fm_onnx=args.fm_onnx, out_dir=args.out,
        providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
        fall_height=float(args.fall_height), fall_tilt_deg=float(args.fall_tilt_deg),
        # `--steer_kp` 0 by default: a heading controller is a second thing varying, and it
        # is off-distribution on the stair courses. It is exposed because the Isaac
        # free-form task DOES steer, and its 0.75 m goal band is narrow for that reason.
        steer_kp=float(args.steer_kp), zero_noise=False, fm_seed=0,
    )
    policy, fm_lag, spawn = mjrunner.load_arm(cfg)
    rig = mjrunner.CourseRig(course, policy, cfg, spawn)
    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)
    # None keeps mjeval's own `steer_yaw_rate`, which is what `goal` means; `run_episode`
    # consults either only when steer_kp > 0, so an unsteered probe is unaffected by both.
    steer_fn = None
    if args.steer_law == "centreline":
        steer_fn = functools.partial(centreline_yaw_rate,
                                     lookahead=float(args.steer_lookahead_m))

    arm_dir = os.path.join(args.out, "trials", args.arm)
    os.makedirs(arm_dir, exist_ok=True)
    rows = []
    start = int(args.seed_start)
    if start:
        # Keep the existing trials byte-for-byte rather than trusting the roll to be
        # reproducible: the seeding is deterministic, but the CONTACT dynamics are not
        # invariant to everything (removing a duplicate ground plane once moved a trajectory
        # from 409 to 419 steps), and a silently re-rolled trial would invalidate its label.
        prev = (pm.get("arms") or {}).get(args.arm)
        if prev is None:
            raise SystemExit(f"REFUSING: --seed_start {start} but arm {args.arm!r} has no "
                             "existing trials to append to.")
        rows = [r for r in prev["rows"] if int(r["seed"]) < start]
        if len(rows) != start:
            raise SystemExit(
                f"REFUSING: arm {args.arm!r} has {len(rows)} trial(s) below seed {start}; "
                f"--seed_start needs a contiguous 0..{start - 1} to append to.")
        print(f"[mode] appending: keeping seeds 0..{start - 1}, rolling "
              f"{start}..{int(args.trials) - 1}")
    t0 = time.time()
    for seed in range(start, int(args.trials)):
        frames: list[tuple[float, np.ndarray]] = []

        def record(robot, _policy, _f=frames) -> None:
            """Capture one control step's pose. Physics is untouched."""
            _f.append((float(robot.mj_data.time), robot.mj_data.qpos.copy()))

        # The ONE thing that differs between trials.
        policy._rng = np.random.default_rng(seed)
        mjrunner._reset_episode_state(policy, fm_lag)
        # `stop_on_goal=False`: this is a RECORDING for a human, not a score. mjeval's
        # `in_goal` is a +/- 1 m band, so on the up-into-down course its near edge sits at the
        # descent lip -- an arriving episode was being cut off at the TOP of the descent and
        # the second mode never reached the footage. The verdict is then "how the recording
        # ended" (fall, or ran the budget), NOT "completed the course"; success rates for
        # these courses come from the Isaac evals, which use their own goal line.
        verdict, failure, _acc = mjrunner.run_episode(
            rig, policy, fm_lag, spawn, cmd, None, recorder=record,
            steer_kp=cfg.steer_kp, cfg_hold=cfg.cmd_hold_s, stop_on_goal=False,
            steer_fn=steer_fn)
        q = np.array([f[1] for f in frames], dtype=np.float32)
        t = np.array([f[0] for f in frames], dtype=np.float32)
        # Where the label window ends: the first step past the feature, or the whole episode
        # when it never got there (a fall short of the obstacle is still a mode decision).
        past = np.nonzero(q[:, 0] >= label_x)[0] if q.size else np.array([], dtype=int)
        cut = int(past[0]) + 1 if past.size else int(q.shape[0])
        np.savez_compressed(os.path.join(arm_dir, f"trial_{seed:04d}.npz"),
                            qpos=q, time=t, cut=cut,
                            verdict=str(verdict), failure=str(failure))
        rows.append({"seed": seed, "verdict": str(verdict), "failure": str(failure),
                     "steps": int(q.shape[0]), "cut": cut,
                     "reached_label_x": bool(past.size),
                     "max_x_m": float(q[:, 0].max()) if q.size else 0.0})
        if seed < 3 or (seed + 1) % 10 == 0:
            print(f"[mode] seed {seed:>3}  {verdict:<9}{failure:<7} "
                  f"x_max={rows[-1]['max_x_m']:6.2f}m  {q.shape[0]:4d} steps  "
                  f"cut@{cut:4d}  ({time.time() - t0:.0f}s)", flush=True)
    del rig

    os.makedirs(args.out, exist_ok=True)
    pm.update({"clip": args.clip, "terrain_id": int(args.terrain_id),
               "family": str(entry.get("family", "")), "klass": klass,
               "expected_mode": EXPECTED_MODE.get(klass, ""),
               "speed": float(args.speed), "approach_m": approach,
               "obstacle_x_m": float(course.obstacle_x_m),
               "label_x_m": label_x, "label_margin_m": float(args.label_margin_m),
               "tracker_onnx": args.tracker_onnx, "modes": list(MODES),
               "robot_name": str(cfg.robot_name), "stop_on_goal": False,
               # Stamped so a steered probe and an unsteered one cannot be confused, and so
               # the refusal above can catch two arms that disagree about it.
               "steer_kp": float(args.steer_kp),
               "steer_law": str(args.steer_law),
               "steer_lookahead_m": float(args.steer_lookahead_m),
               "obstacle_from": str(args.obstacle_from),
               "steer_wz_max": float(cfg.steer_wz_max)})
    pm.setdefault("arms", {})[args.arm] = {
        "fm_onnx": args.fm_onnx, "trials": int(args.trials), "rows": rows}
    with open(probe_path(args.out), "w", encoding="utf-8") as fh:
        json.dump(pm, fh, indent=2)
    reached = sum(1 for r in rows if r["reached_label_x"])
    print(f"\n[mode] {args.arm}: {len(rows)} trials -> {arm_dir}")
    print(f"[mode]   {reached}/{len(rows)} reached the label window; the rest ended before it "
          "and are labelled on what they attempted")
    print(f"[mode] manifest {probe_path(args.out)}")
    return 0


# ---------------------------------------------------------------------------------------------
# Stage: label
# ---------------------------------------------------------------------------------------------


def label_key(arm: str, seed: int) -> str:
    """Storage key for one trial's label.

    Keyed by the TRIAL, not by its position in the blinded order. Position keys meant that
    rolling more trials re-shuffled the sequence and silently re-pointed every existing label
    at a different trajectory -- so extending a probe cost a full re-label of work already
    done. The presentation order is still shuffled; only the storage key is stable.

    Args:
        arm: Arm name.
        seed: The trial's noise seed, which is also its index.

    Returns:
        The key used in ``labels.json``.
    """
    return f"{arm}:{int(seed)}"


def trial_fingerprints(pm: dict) -> dict[str, str]:
    """One outcome digest per trial, keyed like the labels.

    Replaces the whole-probe digest, which could not distinguish "this trial was re-rolled and
    moved" from "trials were added". A label describes a TRAJECTORY, so it has to be dropped by
    anything that moves that trajectory -- and only that one.

    Args:
        pm: The probe manifest.

    Returns:
        ``{label_key: digest}``.
    """
    import hashlib  # noqa: PLC0415

    out: dict[str, str] = {}
    for arm, d in sorted(pm["arms"].items()):
        for r in d["rows"]:
            h = hashlib.sha256(
                f"{r['seed']}:{r['verdict']}:{r['failure']}:{r['steps']}:{r['cut']}".encode())
            out[label_key(arm, r["seed"])] = h.hexdigest()[:16]
    return out


def _load_labels(out_dir: str, pm: dict, *, what: str) -> tuple[dict, dict, list]:
    """Labels, the fresh fingerprint map, and the keys whose trial has since been re-rolled.

    Args:
        out_dir: Probe directory.
        pm: The probe manifest.
        what: Verb for the refusal message.

    Returns:
        ``(labels, fingerprints, stale_keys)``.
    """
    import os  # noqa: PLC0415

    lab_path = os.path.join(out_dir, "labels.json")
    labels: dict[str, str] = {}
    if os.path.isfile(lab_path):
        with open(lab_path, encoding="utf-8") as fh:
            labels = json.load(fh)
    if labels and all(k.isdigit() for k in labels):
        raise SystemExit(
            f"REFUSING to {what}: {lab_path} is keyed by POSITION in the blinded order, the "
            "old scheme. Run `--stage migrate` first -- it rewrites the keys as arm:seed "
            "using the shuffle seed the labels were made with, so nothing is re-labelled.")
    fps = trial_fingerprints(pm)
    old: dict[str, str] = {}
    ctx_path = os.path.join(out_dir, "label_context.json")
    if os.path.isfile(ctx_path):
        with open(ctx_path, encoding="utf-8") as fh:
            old = (json.load(fh) or {}).get("trials", {})
    stale = [k for k in labels if k in old and fps.get(k) != old[k]]
    return labels, fps, stale


def order_of(pm: dict, seed: int) -> list[tuple[str, int]]:
    """The blinded presentation order: every arm's trials interleaved by one shuffle.

    Args:
        pm: The probe manifest.
        seed: Shuffle seed, so a resumed session continues the same sequence.

    Returns:
        ``(arm, trial seed)`` pairs.
    """
    keys = [(arm, r["seed"]) for arm, a in sorted(pm["arms"].items()) for r in a["rows"]]
    random.Random(seed).shuffle(keys)
    return keys


def stage_watch(args) -> int:
    """Replay trials for a human to WATCH, writing nothing.

    The labelling stage cannot do this job.  It is blind by construction and it advances only
    on one of :data:`KEYS`, so "just look at them" there means inventing a label per trial and
    leaving it in ``labels.json`` -- which is the one file a probe's conclusion rests on.  This
    stage therefore shows the arm, plays in a fixed order rather than a shuffled one, advances
    when the pass ends, and never opens ``labels.json`` at all.

    Defaults to the WHOLE episode, since watching is usually about how a trial ended rather
    than about the first mode; ``[v]`` still toggles back to the label window.

    Args:
        args: Parsed CLI arguments; ``--out`` and optionally ``--arm`` (one arm instead of
            all), ``--whole`` / ``--rt`` / ``--passes`` as in ``--stage label``.

    Returns:
        Process exit status.
    """
    import mujoco  # noqa: PLC0415
    import mujoco.viewer  # noqa: PLC0415

    from mjeval import view_cell  # noqa: PLC0415

    pm = load_probe(args.out)
    if args.clip:
        pm["clip"] = args.clip
    arms = sorted(pm["arms"])
    if args.arm:
        if args.arm not in pm["arms"]:
            raise SystemExit(f"no arm {args.arm!r} in {args.out}; have {', '.join(arms)}")
        arms = [args.arm]
    # Fixed order, NOT `order_of`: blinding exists to keep a labeller honest, and nothing is
    # recorded here. The default is SEED-MAJOR because the seed IS the noise draw --
    # `stage_run` sets `policy._rng = default_rng(seed)` and nothing else varies within an arm
    # -- so one seed across two arms is a single draw through two generators, which is the only
    # pairing in this data that holds the randomness fixed. Watching those back to back is what
    # makes a difference attributable to the generator rather than to the draw.
    have = {(a, r["seed"]) for a in arms for r in pm["arms"][a]["rows"]}
    seeds = sorted({sd for _, sd in have})
    if args.by_arm:
        order = [(a, sd) for a in arms for sd in seeds if (a, sd) in have]
    else:
        order = [(a, sd) for sd in seeds for a in arms if (a, sd) in have]
    if not order:
        print("[mode] no trials to watch.")
        return 0
    paired = not args.by_arm and len(arms) > 1
    print(f"[mode] {len(order)} trial(s) across {len(arms)} arm(s): {', '.join(arms)}")
    if paired:
        print("[mode] PAIRED: the same seed back to back across arms -- one flow-noise draw, "
              f"{len(arms)} generators. Order: " + " -> ".join(arms))
    incomplete = [sd for sd in seeds if sum((a, sd) in have for a in arms) != len(arms)]
    if paired and incomplete:
        print(f"[mode] note: {len(incomplete)} seed(s) are not on every arm and so are "
              f"unpaired ({', '.join(str(x) for x in incomplete[:8])}"
              f"{', ...' if len(incomplete) > 8 else ''}).")
    print(f"[mode] tile {pm['terrain_id']} ({pm['klass']}), label window ends at "
          f"x={pm['label_x_m']:.2f} m")
    print("[mode] WATCH ONLY -- nothing is written, and the arm IS shown.")
    print("[mode]   [space] pause/resume   [r] replay   [v] window/whole   [q] quit   "
          "any other key: next")

    # Whole episode by default here, the opposite of --stage label's default.
    full = bool(args.whole) or not args.window_only
    course = terrains.isaac_tile_course(
        pm["clip"], int(pm["terrain_id"]), speeds=(float(pm["speed"]),),
        klass=pm["klass"], name=f"modeprobe_t{pm['terrain_id']}",
        spawn_from="route", obstacle_from=str(pm.get("obstacle_from", "raised")),
        goal_from="route")
    terrains.write_scene(course, pm["robot_name"])
    model = Robot(robot_name=pm["robot_name"], scene_name=course.name,
                  joystick_scaling=np.ones(3), input_function=lambda _t: np.zeros(3),
                  fall_height=0.2, fall_tilt_rad=float(np.deg2rad(70.0))).mj_model

    data = mujoco.MjData(model)
    target_z = course.floor_z if course.flight_height_m < 0 else course.flight_height_m
    quit_now = False
    with cbreak_stdin() as live, mujoco.viewer.launch_passive(model, data) as viewer:
        for n_seen, (arm, seed) in enumerate(order, start=1):
            if quit_now or not viewer.is_running():
                break
            z = np.load(os.path.join(args.out, "trials", arm, f"trial_{seed:04d}.npz"),
                        allow_pickle=False)
            q, t, cut = z["qpos"], z["time"], int(z["cut"])
            drain_keys()
            while True:
                if not viewer.is_running():
                    quit_now = True
                    break
                n = q.shape[0] if full else cut
                pair_pos = (f"  [{arms.index(arm) + 1}/{len(arms)} of seed {seed}]"
                            if paired else "")
                print(f"\n[{n_seen}/{len(order)}]  seed {seed}  arm {arm}{pair_pos}  "
                      f"{'WHOLE episode' if full else 'to the first feature'}: "
                      f"{t[n - 1] - t[0]:.2f} s at {args.rt:g}x  "
                      f"({z['verdict']}{'/' + str(z['failure']) if str(z['failure']) else ''})",
                      flush=True)
                ans = _play(viewer, model, data, q[:n], t[:n], course, target_z,
                            float(args.rt), view_cell, live)
                if ans == "__closed__" or ans == "q":
                    quit_now = True
                    break
                if ans == "r":
                    continue
                if ans == "v":
                    full = not full
                    continue
                break           # played out, or any other key -> next trial
    print("[mode] done; nothing was written.")
    return 0


def stage_label(args) -> int:
    """Replay each trial blind and record the human's mode call."""
    import mujoco  # noqa: PLC0415
    import mujoco.viewer  # noqa: PLC0415

    from mjeval import view_cell  # noqa: PLC0415

    pm = load_probe(args.out)
    if args.clip:
        pm["clip"] = args.clip          # --stage run may have been on another machine
    order = order_of(pm, int(args.shuffle_seed))
    lab_path = os.path.join(args.out, "labels.json")
    ctx_path = os.path.join(args.out, "label_context.json")
    # Labels are keyed by TRIAL, so adding trials leaves every existing one valid and only the
    # new ones are asked. A trial that was RE-ROLLED and moved still invalidates its own label,
    # which is the property the old whole-probe digest was protecting.
    labels, fps, stale = _load_labels(args.out, pm, what="label")
    for k in stale:
        labels.pop(k, None)
    if stale:
        print(f"[mode] {len(stale)} trial(s) have been re-rolled since they were labelled; "
              "those labels are dropped and the trials will be asked again.")
        with open(lab_path, "w", encoding="utf-8") as fh:
            json.dump(labels, fh, indent=2)
    with open(ctx_path, "w", encoding="utf-8") as fh:
        json.dump({"trials": fps}, fh, indent=2)
    todo = [i for i, (arm, seed) in enumerate(order)
            if args.relabel or label_key(arm, seed) not in labels]
    print(f"[mode] {len(order)} trials across {len(pm['arms'])} arms; "
          f"{len(order) - len(todo)} already labelled, {len(todo)} to go")
    if not todo:
        print("[mode] nothing to label. --relabel to redo, or --stage report.")
        return 0
    print(f"[mode] tile {pm['terrain_id']} ({pm['klass']}), label window ends at "
          f"x={pm['label_x_m']:.2f} m")
    print("[mode] BLIND: the arm is not shown.")
    print("[mode] Keys are LIVE during playback -- answer as soon as you can see the mode: "
          + "  ".join(f"[{k}] {v}" for k, v in KEYS.items()))
    print("[mode]   [space] pause/resume   [r] replay   [v] whole episode   [q] save+quit")

    # Sticky across trials: `[v]` used to be reset to the window at the top of every trial,
    # so a whole-episode session meant pressing it once per trial.
    full = bool(args.whole)
    course = terrains.isaac_tile_course(
        pm["clip"], int(pm["terrain_id"]), speeds=(float(pm["speed"]),),
        klass=pm["klass"], name=f"modeprobe_t{pm['terrain_id']}",
        spawn_from="route", obstacle_from=str(pm.get("obstacle_from", "raised")),
        goal_from="route")
    # The rig's own two lines, without the policy: it needs one only for the PD gains, and a
    # replay steps no physics.
    terrains.write_scene(course, pm["robot_name"])
    model = Robot(robot_name=pm["robot_name"], scene_name=course.name,
                  joystick_scaling=np.ones(3), input_function=lambda _t: np.zeros(3),
                  fall_height=0.2, fall_tilt_rad=float(np.deg2rad(70.0))).mj_model

    data = mujoco.MjData(model)
    target_z = course.floor_z if course.flight_height_m < 0 else course.flight_height_m
    quit_now = False
    with cbreak_stdin() as live, mujoco.viewer.launch_passive(model, data) as viewer:
        if not live:
            print("[mode] stdin is not a terminal: falling back to typed answers + Enter.")
        for i in todo:
            if quit_now or not viewer.is_running():
                break
            arm, seed = order[i]
            z = np.load(os.path.join(args.out, "trials", arm, f"trial_{seed:04d}.npz"),
                        allow_pickle=False)
            q, t, cut = z["qpos"], z["time"], int(z["cut"])
            done = 0
            drain_keys()
            while True:
                if not viewer.is_running():
                    quit_now = True
                    break
                n = q.shape[0] if full else cut
                print(f"\n[{len(labels) + 1}/{len(order)}]  trial {i}  "
                      f"{'WHOLE episode' if full else 'to the first feature'}: "
                      f"{t[n - 1] - t[0]:.2f} s at {args.rt:g}x  "
                      f"({z['verdict']}{'/' + str(z['failure']) if str(z['failure']) else ''})",
                      flush=True)
                ans = None
                # Every pass is interruptible, so the budget only matters for a trial the
                # operator is still undecided about after watching it through.
                for _ in range(max(int(args.passes), 1)):
                    ans = _play(viewer, model, data, q[:n], t[:n], course, target_z,
                                float(args.rt), view_cell, live)
                    if ans is not None:
                        break
                if ans == "__closed__":
                    quit_now = True
                    break
                if ans is None:
                    ans = _ask(live, viewer)
                if ans == "q":
                    quit_now = True
                    break
                if ans == "r":
                    continue
                if ans == "v":
                    full = not full
                    continue
                if ans in KEYS:
                    labels[label_key(arm, seed)] = KEYS[ans]
                    # Written after EVERY answer: a labelling session is human time, and a
                    # crash at trial 80 must not cost the first 79.
                    with open(lab_path, "w", encoding="utf-8") as fh:
                        json.dump(labels, fh, indent=2)
                    print(f"    -> {KEYS[ans]}", flush=True)
                    break
                print("    unrecognised. [f]lat [s]tairs [b]ox [x]unclear "
                      "[space]pause [r]eplay [v]whole [q]uit", flush=True)
                done += 1
                if done > 20:
                    print("    too many unrecognised keys; skipping this trial.", flush=True)
                    break
    print(f"\n[mode] {len(labels)}/{len(order)} labelled -> {lab_path}")
    return 0


def _ask(live: bool, viewer=None) -> str:
    """Ask for a label after a pass ended untouched.

    Single keypress when stdin is a terminal, a typed line otherwise, so the labeller works
    over a pipe as well as at a console.  The viewer is SYNCED while waiting: without that the
    window stops redrawing the moment the prompt appears, so the last frame cannot be orbited
    -- which is exactly when an operator who watched the whole pass undecided wants to look
    around it.
    """
    print("    mode? [f]lat [s]tairs [b]ox [x]unclear  [r]eplay [v]whole [q]uit: ",
          end="", flush=True)
    if not live:
        try:
            return input().strip().lower()[:1]
        except EOFError:
            # A piped run that ran out of answers: stop and keep what was recorded, rather
            # than tracebacking out and losing the session's own summary line.
            print("q  (end of input)", flush=True)
            return "q"
    while True:
        ch = poll_key()
        if ch is not None and not ch.isspace():
            print(ch, flush=True)
            return ch
        if viewer is not None:
            if not viewer.is_running():
                return "q"
            viewer.sync()
        time.sleep(0.02)


def _play(viewer, model, data, q, t, course, target_z, rt: float, view_cell,
          live: bool) -> str | None:
    """Play one pass, returning the key that interrupted it.

    Writes ``qpos`` and calls ``mj_forward`` only, so the viewer perturbs nothing.

    **Keys are live during playback**, which is the point: the mode is usually obvious well
    before the robot reaches the end of the window, and making the operator wait for the pass
    to finish is a hundred needless seconds per probe.  Space holds the current frame.

    Args:
        viewer: The passive viewer.
        model / data: The scene, and scratch state to write poses into.
        q / t: Poses and times for this pass.
        course: For the goal rectangle.
        target_z: Height the goal rectangle is drawn at.
        rt: Real-time multiple.
        view_cell: Borrowed for its scene-drawing helpers, so the visual conventions match.
        live: Whether single keypresses are available.

    Returns:
        The key pressed, ``"__closed__"`` if the viewer went away, else ``None`` when the pass
        ran to the end untouched.
    """
    import mujoco  # noqa: PLC0415

    wall0 = time.time()
    dt = float(t[1] - t[0]) if t.size > 1 else 0.02
    paused = False
    k = 0
    while k < q.shape[0]:
        if not viewer.is_running():
            return "__closed__"
        if live:
            ch = poll_key()
            if ch == " ":
                paused = not paused
                # Re-datum the wall clock so un-pausing does not fast-forward to catch up.
                wall0 = time.time() - (k * dt) / rt if rt > 0 else time.time()
                print(f"    {'paused' if paused else 'resumed'} at "
                      f"{t[k] - t[0]:.2f} s (space)", flush=True)
            elif ch is not None:
                return ch
        if paused:
            viewer.sync()
            time.sleep(0.03)
            continue
        data.qpos[:] = q[k]
        mujoco.mj_forward(model, data)
        viewer.user_scn.ngeom = 0
        view_cell._goal_rect(viewer.user_scn, course, target_z)
        view_cell._dots(viewer.user_scn, q[k, :3].reshape(1, 3), view_cell.ROOT_RGBA, 0.05)
        viewer.cam.lookat[:] = q[k, :3]
        viewer.sync()
        if rt > 0:
            ahead = (k * dt) / rt - (time.time() - wall0)
            if ahead > 0:
                time.sleep(ahead)
        k += 1
    return None


# ---------------------------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------------------------
def stage_migrate(args) -> int:
    """Rewrite position-keyed labels as arm:seed, in place, labelling nothing new.

    The old scheme keyed a label by its index in the blinded order. That order is
    ``order_of(pm, shuffle_seed)``, and it is reproducible -- so as long as the probe still has
    the trials it had when the labels were made, position i maps back to exactly the trial the
    human watched. The shuffle seed comes from the probe's own ``label_context.json`` rather
    than from the command line, because using a different one here would re-point every label
    silently, which is the failure this whole change exists to remove.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit status.
    """
    pm = load_probe(args.out)
    lab_path = os.path.join(args.out, "labels.json")
    ctx_path = os.path.join(args.out, "label_context.json")
    if not os.path.isfile(lab_path):
        raise SystemExit(f"no labels at {lab_path}; nothing to migrate.")
    with open(lab_path, encoding="utf-8") as fh:
        labels = json.load(fh)
    if not labels:
        raise SystemExit("labels.json is empty; nothing to migrate.")
    if not all(k.isdigit() for k in labels):
        print(f"[mode] {lab_path} is already keyed by arm:seed; nothing to do.")
        return 0

    old_ctx = {}
    if os.path.isfile(ctx_path):
        with open(ctx_path, encoding="utf-8") as fh:
            old_ctx = json.load(fh) or {}
    seed = int(old_ctx.get("shuffle_seed", args.shuffle_seed))
    # The trial COUNTS decide the order, so a probe that has already been extended cannot be
    # migrated -- position i no longer means what it meant. Refuse rather than mis-map.
    now_counts = {a: len(d["rows"]) for a, d in sorted(pm["arms"].items())}
    was_counts = old_ctx.get("trials")
    if was_counts and {k: int(v) for k, v in was_counts.items()} != now_counts:
        raise SystemExit(
            f"REFUSING: the labels were made when the arms held {was_counts} trials and they "
            f"now hold {now_counts}. Position keys only map back against the counts they were "
            "recorded under, so migrate BEFORE rolling more trials.")

    order = order_of(pm, seed)
    out: dict[str, str] = {}
    for i_str, mode in labels.items():
        i = int(i_str)
        if not 0 <= i < len(order):
            raise SystemExit(f"REFUSING: label index {i} is outside the {len(order)} trials "
                             "this probe holds; the labels do not belong to it.")
        arm, tseed = order[i]
        out[label_key(arm, tseed)] = mode
    if len(out) != len(labels):
        raise SystemExit(f"REFUSING: {len(labels)} labels collapsed to {len(out)} keys, so two "
                         "positions mapped to one trial. Not migrating.")

    backup = lab_path + ".position_keyed.bak"
    os.replace(lab_path, backup)
    with open(lab_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(ctx_path, "w", encoding="utf-8") as fh:
        json.dump({"trials": trial_fingerprints(pm)}, fh, indent=2)
    per_arm: dict[str, int] = {}
    for k in out:
        per_arm[k.rsplit(":", 1)[0]] = per_arm.get(k.rsplit(":", 1)[0], 0) + 1
    print(f"[mode] migrated {len(out)} label(s) using shuffle_seed={seed}: "
          + ", ".join(f"{a} {n}" for a, n in sorted(per_arm.items())))
    print(f"[mode] previous file kept at {backup}")
    return 0


def stage_report(args) -> int:
    """Join the blind labels back to their arms and compare."""
    pm = load_probe(args.out)
    order = order_of(pm, int(args.shuffle_seed))
    lab_path = os.path.join(args.out, "labels.json")
    if not os.path.isfile(lab_path):
        raise SystemExit(f"no labels at {lab_path}; run --stage label first.")
    with open(lab_path, encoding="utf-8") as fh:
        labels = json.load(fh)
    _, _, stale = _load_labels(args.out, pm, what="report")
    if stale:
        raise SystemExit(
            f"REFUSING to report: {len(stale)} labelled trial(s) have been re-rolled since "
            f"they were labelled ({', '.join(sorted(stale)[:6])}"
            f"{' ...' if len(stale) > 6 else ''}). The label describes the OLD trajectory. "
            "Re-run `--stage label` -- it drops exactly those and asks them again.")

    want = pm.get("expected_mode", "")
    # Baseline first, whatever it is called, so the delta is always ARM - BASELINE.
    order_hint = ("parent", "baseline", "pretrained", "pre_trained")
    arms = sorted(pm["arms"], key=lambda a: (a.lower() not in order_hint, a))
    counts = {a: dict.fromkeys(MODES, 0) for a in arms}
    verdicts = {a: {} for a in arms}
    for arm, seed in order:
        m = labels.get(label_key(arm, seed))
        if m is None:
            continue
        counts[arm][m] += 1
        row = next(r for r in pm["arms"][arm]["rows"] if r["seed"] == seed)
        verdicts[arm].setdefault(m, []).append(row["verdict"])

    print(f"tile {pm['terrain_id']} ({pm['family']}, class {pm['klass']}) at "
          f"{pm['speed']} m/s;  correct first mode = {want!r}")
    print(f"labelled {sum(sum(c.values()) for c in counts.values())} of {len(order)} trials, "
          f"blind\n")
    head = f"{'arm':<12}" + "".join(f"{m:>10}" for m in MODES) + f"{'n':>6}{'correct':>10}"
    print(head)
    print("-" * len(head))
    for a in arms:
        n = sum(counts[a].values())
        ok = counts[a].get(want, 0)
        print(f"{a:<12}" + "".join(f"{counts[a][m]:>10}" for m in MODES)
              + f"{n:>6}{(ok / n if n else float('nan')):>10.3f}")
    if len(arms) == 2 and want:
        a, b = arms
        na, nb = sum(counts[a].values()), sum(counts[b].values())
        ka, kb = counts[a].get(want, 0), counts[b].get(want, 0)
        if na and nb:
            # a is the baseline by the ordering above, so this reads arm - baseline.
            # Fisher's exact on the 2x2 of correct-mode counts: the trials are independent
            # draws (only the noise differs) and the counts are small, so no normal
            # approximation is warranted.
            p = _fisher(ka, na - ka, kb, nb - kb)
            print(f"\n{b} - {a}: {kb / nb - ka / na:+.3f} correct-mode rate, "
                  f"Fisher exact p={p:.4g}")
    # Survival GIVEN the correct mode. The headline rate answers "did it choose right"; this
    # answers "did choosing right work", and the two dissociate: on one course the gain is all
    # selection and on the other selection was already near ceiling and the gain is execution.
    # Legitimate to condition on, because the mode is chosen BEFORE the outcome and the label
    # was made blind -- it is a mediator, not a post-hoc split on the result.
    if want:
        print(f"\nsurvival GIVEN the correct mode ({want!r}):")
        cells = {}
        for a in arms:
            v = verdicts[a].get(want, [])
            k = sum(1 for x in v if x == "success")
            cells[a] = (k, len(v))
            print(f"  {a:<12}{k:>3}/{len(v):<3} = "
                  f"{(k / len(v) if v else float('nan')):.3f}")
        if len(arms) == 2:
            (ka, na), (kb, nb) = (cells[arms[0]], cells[arms[1]])
            if na and nb:
                print(f"  {arms[1]} - {arms[0]}: {kb / nb - ka / na:+.3f}, "
                      f"Fisher exact p={_fisher(ka, na - ka, kb, nb - kb):.4g}")

    if args.json:
        out = {"course": pm["family"], "klass": pm["klass"], "speed": float(pm["speed"]),
               "terrain_id": int(pm["terrain_id"]), "expected_mode": want,
               "n_labelled": sum(sum(c.values()) for c in counts.values()),
               "arms": {}}
        for a in arms:
            v = verdicts[a].get(want, [])
            out["arms"][a] = {
                "n": sum(counts[a].values()),
                "modes": dict(counts[a]),
                "correct": counts[a].get(want, 0),
                "surv_success": sum(1 for x in v if x == "success"),
                "surv_n": len(v),
            }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")

    print("\nhow the recording ended, by label -- NOT a course-completion rate: the probe "
          "runs\npast the goal on purpose, so 'timeout' means it survived the budget:")
    for a in arms:
        for m in MODES:
            v = verdicts[a].get(m, [])
            if v:
                brk = ", ".join(f"{k}={v.count(k)}" for k in sorted(set(v)))
                print(f"  {a:<12}{m:<10}{brk}")
    return 0


def _fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 ``[[a, b], [c, d]]``."""
    from math import comb  # noqa: PLC0415

    n = a + b + c + d
    r1, c1 = a + b, a + c
    obs = comb(r1, a) * comb(n - r1, c1 - a) if 0 <= c1 - a <= n - r1 else 0
    tot = comb(n, c1)
    p = 0.0
    for k in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        w = comb(r1, k) * comb(n - r1, c1 - k)
        if w <= obs:
            p += w / tot
    return min(p, 1.0)


def main() -> int:
    """Dispatch a stage."""
    args = build_parser().parse_args()
    return {"run": stage_run, "watch": stage_watch, "label": stage_label,
            "report": stage_report, "migrate": stage_migrate}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
