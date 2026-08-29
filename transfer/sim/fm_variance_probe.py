#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""How much does the generator's OWN sampling variance change the outcome?

One initial condition, one velocity command, one in-distribution tile, N repeats.  The only
thing that differs between repeats is the flow-matching noise draw.  Everything else is held:
no domain randomisation, no depth noise, no clip, the same standing spawn, the same command,
and MuJoCo is deterministic, so a repeat at a fixed seed is bit-identical.

**Why this is worth running even though a best-of-K harness exists.**  The Isaac harness
answers the same question per state, and to do that it has to COPY one env's state onto
another -- which Isaac Lab can do for everything the controller reads and cannot do for
PhysX's contact manifolds, friction anchors and solver warm start.  That residue turned out to
account for the whole of the signal it was meant to measure (its same-noise control's
best-of-K rise matched the real arm's at every horizon).  Here there is no restore at all:
every repeat is an independent rollout from a spawn, so the flow noise is the only difference
by construction rather than by correction.

**Why a REAL tile.**  ``mjeval.terrains``' authored courses vary one geometric parameter each
and are deliberately not trained tiles, which is right for a bisect and wrong here: terrain the
generator never saw contributes its own failures, and they would be indistinguishable from
sampling variance.  ``terrains.isaac_tile_course`` reads the clip library's own
``terrain_manifest``, so the tile is the one the pair trained on.

What the answer means:

* **Wide spread in survival or in distance-before-failure** -- the flow's own draw decides the
  outcome from an identical start, so good chunks exist in the conditional and something that
  sharpens the distribution has mass to work with.
* **Narrow spread** -- the outcome is a property of the state and the terrain, not of the
  draw, and sharpening has nothing to select.  The lever is upstream.

Run from ``transfer/sim`` (no GPU, no isaaclab; ``env_transfer`` is enough)::

    python fm_variance_probe.py \\
        --clip ../../logs/headroom/v1/eval_clips.h5 --terrain_id 6 --terrain_id 20 \\
        --tracker_onnx <run>/exported/tracker.onnx --fm_onnx <run>/exported/<fm>.onnx \\
        --speed 0.8 --trials 500 --out ../../logs/fm_variance/v1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, "mjeval"), os.path.dirname(_HERE),
           os.path.dirname(os.path.dirname(_HERE))):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import runner as mjrunner  # noqa: E402  - transfer/sim/mjeval/runner.py
import schema as mjschema  # noqa: E402
import terrains  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True,
                   help="Clip library .h5 carrying the terrain_manifest the tiles come from.")
    p.add_argument("--terrain_id", type=int, action="append", required=True,
                   help="Repeatable. One course per tile; give a stairs-up and a "
                        "stairs-down id.")
    p.add_argument("--tracker_onnx", required=True, help="Absolute path to tracker.onnx.")
    p.add_argument("--fm_onnx", required=True, help="Absolute path to the generator .onnx.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--speed", type=float, default=0.8,
                   help="The commanded forward speed, held constant after the settle hold.")
    p.add_argument("--trials", type=int, default=500,
                   help="Repeats per tile. Seeds are 0..trials-1, so a longer run is a "
                        "superset of a shorter one rather than a different sample.")
    p.add_argument("--cmd_hold_s", type=float, default=terrains.CMD_HOLD_S,
                   help="Seconds of ZERO command before stepping to --speed, so the robot "
                        "settles from the spawn before it is asked to walk.")
    p.add_argument("--approach", type=float, default=2.5,
                   help="Metres before the first riser to spawn.")
    p.add_argument("--zero_noise_control", action="store_true",
                   help="CONTROL: integrate the flow from x0=0 for every trial, so every "
                        "repeat is bit-identical. The spread it reports must be EXACTLY zero; "
                        "anything else means something other than the flow noise is varying "
                        "between repeats, and the main result would be measuring that too.")
    p.add_argument("--scene_tag", default="",
                   help="Suffix for the emitted scene name. REQUIRED to be distinct when two "
                        "of these run concurrently on the same tile: CourseRig writes "
                        "robots/g1/<robot>_<course>.xml from the course before loading it, so "
                        "two processes sharing a course name race on that file -- and for a "
                        "speed sweep the geometry is identical, so the tear would be silent "
                        "rather than a load error.")
    p.add_argument("--providers", default="auto", help="onnxruntime providers.")
    p.add_argument("--fall_height", type=float, default=0.2,
                   help="Pelvis clearance (m) above the scene's lowest terrain below which "
                        "the episode is a fall.")
    p.add_argument("--fall_tilt_deg", type=float, default=70.0, help="Tilt bound from upright.")
    return p


def run_tile(args, terrain_id: int, course, table, meta: dict) -> dict:
    """Run every trial on one tile and return its summary.

    The pair is loaded ONCE per tile and the RNG is reseeded per trial: loading the ONNX pair
    costs seconds, so 500 loads would dominate the run, and reseeding is exactly equivalent --
    ``GenTrackerPolicy._sample_noise`` draws from ``self._rng`` and holds no other state
    across episodes that ``_reset_episode_state`` does not already clear.
    """
    cfg = mjrunner.ArmRun(
        arm=0, label=f"tile{terrain_id}", tracker_onnx=args.tracker_onnx,
        fm_onnx=args.fm_onnx, out_dir=args.out,
        providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
        fall_height=float(args.fall_height), fall_tilt_deg=float(args.fall_tilt_deg),
        cmd_hold_s=float(args.cmd_hold_s),
        # steer_kp stays 0: a heading controller would be a second thing varying between
        # trials and is off-distribution on the stair courses anyway.
        steer_kp=0.0,
        zero_noise=bool(args.zero_noise_control), fm_seed=0,
    )
    policy, fm_lag, spawn = mjrunner.load_arm(cfg)
    rig = mjrunner.CourseRig(course, policy, cfg, spawn)
    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)
    print(f"[probe] {course.name}: {course.note}")
    print(f"[probe] cmd={cmd.tolist()}  hold={cfg.cmd_hold_s}s  budget={course.budget_s:.1f}s  "
          f"zero_noise={cfg.zero_noise}  trials={args.trials}")

    rows: list[dict] = []
    t0 = time.time()
    for seed in range(int(args.trials)):
        # The ONE thing that differs between trials.
        policy._rng = np.random.default_rng(seed)
        mjrunner._reset_episode_state(policy, fm_lag)
        verdict, failure, acc = mjrunner.run_episode(
            rig, policy, fm_lag, spawn, cmd, None,
            steer_kp=cfg.steer_kp, steer_wz_max=cfg.steer_wz_max,
            cfg_hold=cfg.cmd_hold_s)
        row = acc.row(episode_id=terrain_id * 100000 + seed, terrain_idx=terrain_id,
                      speed_idx=0, arm=0, seed=seed, verdict=verdict, failure_code=failure)
        rows.append(row)
        table.append(row)
        if seed < 5 or (seed + 1) % 25 == 0:
            print(f"[probe] {course.name} seed {seed:>4}  {verdict:<9}{failure:<7} "
                  f"x={row['progress_x_m']:6.2f}m  dist={row['distance_m']:6.2f}m  "
                  f"t={row['steps_survived'] * policy.get_dt():5.2f}s "
                  f"({time.time() - t0:.0f}s wall)", flush=True)
    del rig
    return summarise(course, rows, meta)


def summarise(course, rows: list[dict], meta: dict) -> dict:
    """Reduce one tile's trials to the numbers the question turns on."""
    v = np.array([int(r["verdict"]) for r in rows])
    codes = terrains.VERDICT_CODES
    fell = v == codes["fall"]
    success = v == codes["success"]
    x = np.array([float(r["progress_x_m"]) for r in rows])
    d = np.array([float(r["distance_m"]) for r in rows])
    xf = x[fell]
    out = {
        "course": course.name, "terrain_id": course.variant, "klass": course.klass,
        "n": int(v.size),
        "success_rate": float(success.mean()),
        "fall_rate": float(fell.mean()),
        "timeout_rate": float((v == codes["timeout"]).mean()),
        "obstacle_x_m": float(course.obstacle_x_m),
        "course_length_m": float(course.course_length_m),
        "flight_height_m": float(course.flight_height_m),
        "progress_x_m": _dist(x),
        "distance_m": _dist(d),
        "progress_x_m_falls_only": _dist(xf) if xf.size else None,
        # The headline: an outcome that is DECIDED by the draw shows both verdicts from one
        # start. A rate of exactly 0 or 1 means the draw decides nothing here.
        "outcome_is_draw_decided": bool(0 < fell.mean() < 1),
        "meta": meta,
    }
    return out


def _dist(a: np.ndarray) -> dict:
    """Mean, sd and the deciles of one column -- the spread IS the result here."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1))
            if a.size > 1 else 0.0, "min": float(a.min()), "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max()),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25))}


def main() -> int:
    """Build each tile's course, run the trials, write the table and the summary."""
    args = build_parser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    for role, path in (("tracker", args.tracker_onnx), ("generator", args.fm_onnx),
                       ("clip library", args.clip)):
        if not os.path.isfile(path):
            raise SystemExit(f"no {role} at {path}")

    meta = {
        "clip": os.path.abspath(args.clip), "tracker_onnx": os.path.abspath(args.tracker_onnx),
        "fm_onnx": os.path.abspath(args.fm_onnx), "speed": float(args.speed),
        "trials": int(args.trials), "cmd_hold_s": float(args.cmd_hold_s),
        "approach": float(args.approach), "scene_tag": str(args.scene_tag),
        "zero_noise_control": bool(args.zero_noise_control),
        "fall_height": float(args.fall_height), "fall_tilt_deg": float(args.fall_tilt_deg),
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    table = mjschema.EpisodeTable()
    summaries = []
    for tid in args.terrain_id:
        course = terrains.isaac_tile_course(
            args.clip, tid, speeds=(float(args.speed),), approach=float(args.approach),
            name=f"tile{tid}{args.scene_tag}")
        summaries.append(run_tile(args, tid, course, table, meta))
        print(json.dumps(summaries[-1], indent=2, default=float))

    h5 = os.path.join(args.out, "episodes.h5")
    table.write(h5, {**meta, "harness": "fm_variance_probe"})
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"meta": meta, "tiles": summaries}, fh, indent=2, default=float)
    print(f"\n[probe] {sum(t['n'] for t in summaries)} episodes -> {h5}")
    print(f"[probe] summary -> {os.path.join(args.out, 'summary.json')}")
    print("\n[probe] READ: a fall rate strictly between 0 and 1 from ONE initial condition "
          "means the flow's own draw decides the outcome, and a sharpening method has mass "
          "to select. Compare the distance-before-failure spread against the "
          "--zero_noise_control run, whose spread must be exactly zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
