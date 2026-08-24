# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play one debug clip case in BOTH stacks and report whether the robot fell.

The question: the deployed pair tracks well in MuJoCo sim2sim and worse in obelisk, and
nothing said which half that difference lives in. Driving both from the SAME baked clip takes
the generator out of the loop entirely, and scoring both with the SAME criterion
(:mod:`run_metrics`) means the answer cannot be an artefact of two implementations.

**The whole clip is played and the verdict is "did it fall inside it".** No attempt is made to
start or stop on an exact stand: the debug cases contain authored command discontinuities (the
``walk_backward`` case reverses from -0.90 to +0.59 m/s in a single control step, at step 355),
and rather than doctor the motion, both stacks are asked to follow the same one. A
discontinuity is a hard, repeatable stressor at a known step -- which is what a DIFFERENTIAL
test wants. It does mean a fall here is not evidence about the policy in isolation.

MuJoCo is deterministic under ``--zero_noise`` so one run settles it; obelisk is threaded and
is not, so it is repeated. Run the obelisk side with::

    scripts/run_sim_clip.sh --case walk_backward     # brain   (terminal A)
    scripts/run_sim_world.sh                         # world   (terminal B)

then::

    python plot_utils/compare_clip_case.py --case walk_backward --repeats 5

``--repeats N`` scores the N newest matching bags; ``--no-sim`` skips the MuJoCo run and
scores only what is on disk.
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import subprocess
import sys

import numpy as np
import yaml

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))          # transfer/, for deploy + sim

import run_metrics as rm                              # noqa: E402
from bag_reader import read_bag                       # noqa: E402
from bag_run import TOPICS, Run                       # noqa: E402

DEFAULT_BAG_ROOT = _HERE.parent / "log" / "bags"
DEFAULT_CLIP = "transfer/obelisk/g1_control/resource/debug_clips/g1_debug_v1.h5"
REPO_ROOT = _HERE.parents[2]


#: Language suffixes run_sim_clip.sh appends to a bag tag. The empty string covers bags from
#: before the C++ path existed.
_LANG_SUFFIXES = ("", "_cpp", "_py", "_python")


def _is_case(path: str, case: str) -> bool:
    """Whether a bag directory is THIS case and not one whose name extends it.

    ``stand`` is a prefix of ``stand_walk_stand``, so a bare glob silently pools the two --
    and the pooled table looks perfectly reasonable, which is how it went unnoticed until the
    same bag appeared under both cases.
    """
    tail = os.path.basename(path.rstrip("/")).split(f"_clip_{case}", 1)
    return len(tail) == 2 and tail[1] in _LANG_SUFFIXES


def _travel(pos: np.ndarray) -> float:
    """Furthest the base got from where it started, in the PLANE.

    Distance, not a signed x displacement. The first version of this reported
    ``min(x - x[0])``, which sees only BACKWARD travel -- so ``stand_walk_stand``, which walks
    forward at 0.87 m/s for five seconds, reported 0.02 m and read as a robot standing still
    while commanded to walk. A signed axis cannot describe travel a case is free to do in any
    direction.
    """
    d = np.asarray(pos, dtype=np.float64)[:, :2] - np.asarray(pos, dtype=np.float64)[0, :2]
    return float(np.max(np.linalg.norm(d, axis=-1)))


def clip_real_steps(clip: str, case: str) -> int:
    """How many steps the case actually plays, so a truncated bag can be told from a survival.

    Read from the library rather than assumed, because the cases differ (``terrain`` is 331
    steps where the flat ones are ~1000).
    """
    import h5py

    sidecar = os.path.join(REPO_ROOT, clip.rsplit(".h5", 1)[0] + ".json")
    with open(sidecar) as fh:
        import json
        index = json.load(fh)["cases"][case]["clip_index"]
    with h5py.File(os.path.join(REPO_ROOT, clip), "r") as f:
        return int(np.asarray(f["clip_len"][index]))


def score_bag(path: str, terrain_min_z: float, real_steps: int) -> dict:
    """Verdict for one obelisk bag, over the ticks where the clip was still advancing.

    Args:
        path: The bag directory.
        terrain_min_z: Lowest terrain surface in the world the bag was recorded in.

    Returns:
        A row for :func:`report`.
    """
    run = Run(read_bag(path, TOPICS))
    if not run.armed.any():
        return {"name": os.path.basename(path), "stack": "obelisk", "note": "never armed"}
    a = int(np.argmax(run.armed))
    step = run.clip_step[a:]
    # The clip's own 20 s: playback SATURATES on its last frame rather than stopping, and what
    # happens during that hold is a different experiment. Score up to the last advancing tick.
    live = np.flatnonzero(step < np.nanmax(step)) if np.isfinite(step).any() else np.array([])
    end = int(live[-1]) + 1 if live.size else len(step)
    pos, quat = run.base_pos[a:][:end], run.base_quat[a:][:end]
    t = run.t[a:][:end] - run.t[a]
    # Arming does not wait for the estimator, so the first few ticks carry no base position.
    # Start scoring where a measurement exists; see run_metrics.first_measured_sample.
    lag = rm.first_measured_sample(pos)
    pos, quat, t = pos[lag:], quat[lag:], t[lag:]
    step = step[:end][lag:]
    usable = rm.base_position_is_usable(pos)
    reason, t_fall, i = rm.verdict(pos, quat, t, terrain_min_z, use_height=usable)
    return {
        "name": os.path.basename(path)[:19],
        "stack": "obelisk/" + ("cpp" if path.rstrip("/").endswith("cpp") else "py"),
        "ticks": end, "secs": float(t[-1]) if end else 0.0,
        "reason": reason, "t_fall": t_fall,
        "clip_step": int(step[i]) if i is not None else int(np.nanmax(step)),
        "est_lag": lag,
        "travel": _travel(pos) if usable else float("nan"),
        "height_scored": usable,
        # Did playback actually reach the end of the clip? A bag cut short by the recorder
        # stopping is not evidence either way, and must not be counted as a survival.
        "complete": bool(reason) or int(np.nanmax(step)) >= real_steps - 2,
    }


def run_sim2sim(case: str, clip: str, scene: str, robot: str, run_dir: str,
                out_dir: str) -> dict:
    """Play the case through the MuJoCo sim2sim runner and score its replay stream.

    Scored from the STREAM (``root_pos`` / ``base_quat_xyzw`` per tick) rather than from the
    runner's own verdict, so both stacks go through one criterion -- then cross-checked
    against ``sim_meta.yaml``, which is that criterion's other evaluation site.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "transfer/sim/gen_tracker_runner.py",
           "--mode", "eval", "--scene", scene, "--robot_name", robot,
           "--load_run", run_dir, "--clip", clip, "--clip_case", case,
           "--clip_ego_anchor", "--headless", "--zero_noise", "--providers", "cpu",
           "--log", "--log_dir", out_dir, "--dump_replay", out_dir]
    print("[sim2sim] " + " ".join(cmd[1:]))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:]); print(proc.stderr[-2000:])
        return {"name": case, "stack": "sim2sim", "note": f"runner exited {proc.returncode}"}

    from deploy.replay_stream import load
    arrays, _ = load(out_dir)
    pos, quat = arrays["root_pos"], arrays["base_quat_xyzw"]
    meta_path = os.path.join(out_dir, "sim_meta.yaml")
    meta = yaml.safe_load(open(meta_path)) if os.path.exists(meta_path) else {}
    dt = float(meta.get("dt", 0.02))
    t = np.arange(len(pos)) * dt
    tmz = float(meta.get("terrain_min_z", 0.0))
    reason, t_fall, i = rm.verdict(pos, quat, t, tmz)
    rm.assert_matches_runner(meta, reason, t_fall)
    return {"name": case, "stack": "sim2sim", "ticks": len(pos),
            "secs": float(t[-1]) if len(t) else 0.0, "reason": reason, "t_fall": t_fall,
            "clip_step": len(pos) - 1 if i is None else i,
            "travel": _travel(pos), "height_scored": True}


def report(rows: list[dict], case: str) -> int:
    """Print the table. Returns the number of runs that fell."""
    print(f"\n=== {case} : did it fall inside the clip? ===")
    print(f"{'run':<21} {'stack':<12} {'secs':>6} {'verdict':<30} {'travel':>7}")
    fell = 0
    for r in rows:
        if "note" in r:
            print(f"{r['name']:<21} {r['stack']:<12} {'':>6} {r['note']:<30}")
            continue
        if r["reason"]:
            fell += 1
            v = f"FELL ({r['reason']}) t={r['t_fall']:.1f}s step={r['clip_step']}"
        elif not r.get("complete", True):
            # NOT a survival. A recorder stopped early, or a run torn off part way, leaves a
            # bag that never reached the end of the clip -- and calling that "survived"
            # inflates the denominator with runs that were never given the chance to fall.
            v = f"INCOMPLETE: bag ends at step {r['clip_step']}"
        else:
            v = f"survived to step {r['clip_step']}"
        note = "" if r.get("height_scored", True) else "  [tilt only: no odometry]"
        print(f"{r['name']:<21} {r['stack']:<12} {r['secs']:>6.1f} {v:<30} {r['travel']:>7.2f}{note}")
    # Incomplete bags are excluded from the DENOMINATOR, not counted as survivals.
    obk = [r for r in rows if r["stack"].startswith("obelisk") and "note" not in r
           and r.get("complete", True)]
    sim = [r for r in rows if r["stack"] == "sim2sim" and "note" not in r]
    dropped = sum(1 for r in rows if "note" not in r and not r.get("complete", True))
    if obk:
        n = sum(1 for r in obk if r["reason"])
        print(f"\nobelisk: {n}/{len(obk)} fell inside the clip")
    if sim:
        n = sum(1 for r in sim if r["reason"])
        print(f"sim2sim: {n}/{len(sim)} fell inside the clip")
    if dropped:
        print(f"({dropped} incomplete bag(s) excluded -- they never reached the clip's end)")
    return fell


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="walk_backward")
    p.add_argument("--repeats", type=int, default=5,
                   help="How many of the newest matching obelisk bags to score.")
    p.add_argument("--bag_root", default=str(DEFAULT_BAG_ROOT))
    p.add_argument("--clip", default=DEFAULT_CLIP)
    p.add_argument("--scene", default="terrain_scene",
                   help="MuJoCo scene. Must be the one the obelisk world ran "
                        "(g1_world.yml's model_xml_path), or the two are not comparable.")
    p.add_argument("--robot", default="g1_perceptive")
    p.add_argument("--run_dir", default="2026-08-20_16-36-10_mode11_decor_plan_v1")
    p.add_argument("--out", default=None, help="Where the MuJoCo run writes (default: a temp).")
    p.add_argument("--no-sim", dest="sim", action="store_false", default=True,
                   help="Score existing bags only; do not run MuJoCo.")
    args = p.parse_args()

    terrain_min_z = rm.terrain_min_surface_z(args.robot, args.scene)
    print(f"[compare] scene {args.robot}/{args.scene}: terrain floor at z={terrain_min_z:.3f} m")

    rows: list[dict] = []
    if args.sim:
        out = args.out or os.path.join("/tmp", f"clip_ab_{args.case}")
        rows.append(run_sim2sim(args.case, args.clip, args.scene, args.robot, args.run_dir, out))

    real_steps = clip_real_steps(args.clip, args.case)
    bags = sorted(b for b in glob.glob(os.path.join(args.bag_root, f"*_clip_{args.case}*"))
                  if _is_case(b, args.case))
    for b in bags[-args.repeats:] if args.repeats > 0 else []:
        try:
            rows.append(score_bag(b, terrain_min_z, real_steps))
        except Exception as exc:                   # noqa: BLE001 -- a torn bag is not fatal
            rows.append({"name": os.path.basename(b)[:19], "stack": "obelisk",
                         "note": f"unreadable: {type(exc).__name__}"})
    if not rows:
        print(f"[compare] nothing to score: no bags matching *_clip_{args.case}* in "
              f"{args.bag_root}, and --no-sim was given.")
        return 1
    report(rows, args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
