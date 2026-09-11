# Copyright 2026 Zachary Olkin. All rights reserved.

"""Step through live samples on real Isaac tiles, one keypress at a time.

Not a replay tool -- ``live_select.py`` records only the win/loss table, no per-step
trajectory, so there is nothing to step through EXACTLY.  What this gives instead: the same
real clip-library tile (``terrains.isaac_tile_course``, the geometry a table row like "tile 6
K=1 0.700" was actually measured on -- no hand-authored terrain, no walls, none of that
perceptual difference), driven live in the MuJoCo viewer with ``--gaussian_noise`` so each
keypress is a genuinely fresh, independently-sampled draw under the same stochastic mechanism
``live_select.py`` uses. That is what "get a sense of what they look like" can mean without a
capture/replay feature that does not exist yet.

Each run is one live physics episode -- attaching a viewer perturbs ``mjData`` enough to
diverge from a headless run (see ``mjeval/runner.py``'s module docstring), so treat this as
"what does the policy typically do here", not "here is scored episode #7".

**Stops on time or a fall, NOT on reaching the goal.** ``gen_tracker_runner.py`` has no
course/goal concept of its own -- it runs for ``course.budget_s`` (or until ``robot.failed()``
trips the fall-height/tilt check) regardless of whether the robot arrived early, so a fast
success just idles in place for the remainder of the budget rather than ending the window. The
goal band IS drawn (a dotted teal rectangle, mjeval's own marker) so arrival is visible even
though it doesn't end the run.

Run with ``env_isaaclab`` active, from ``transfer/sim``::

    python watch_tiles.py --clip_lib ../../clips/g1_mb_v5_body_mode11_decor.h5 \\
        --load_run 2026-08-20_16-36-10_mode11_decor_plan_v1 \\
        --fm_onnx ../../logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx \\
        --tiles 6,20

Then at the prompt: Enter = next fresh draw on the current tile, a tile id = switch tile,
``z`` = toggle zero-noise (deterministic, the hardware default) vs gaussian (stochastic,
what the table's success rates were measured under), ``q`` = quit.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _d in (_HERE, os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import terrains  # noqa: E402

#: gen_tracker_runner.py imports robot_rl.flow_matching; in a worktree the editable install
#: otherwise resolves to the MAIN checkout's copy, which silently lacks whatever this worktree
#: added (see CLAUDE.md's worktree gotcha). Shadowing it here means the subprocess this script
#: launches cannot hit that trap even if the caller forgot to set PYTHONPATH themselves.
_CHILD_ENV = dict(os.environ)
_shadow = os.pathsep.join([os.path.join(_ROOT, "source", "robot_rl"),
                           os.path.join(_ROOT, "source", "motion_gen")])
_CHILD_ENV["PYTHONPATH"] = (_shadow + os.pathsep + _CHILD_ENV["PYTHONPATH"]
                            if _CHILD_ENV.get("PYTHONPATH") else _shadow)


def _build_courses(clip_lib: str, tiles: list[int], speed: float, approach: float,
                   max_offset: int, scene_tag: str, robot_name: str) -> dict[int, terrains.Course]:
    """Build and write one scene per tile, the same construction ``live_select.py`` uses.

    Args:
        clip_lib: Clip library ``.h5``.
        tiles: Tile ids to offer.
        speed: Commanded forward speed, m/s.
        approach: Standing-approach distance before the tile, m.
        max_offset: Goal-anchor max offset passed to ``isaac_tile_course``.
        scene_tag: Suffix on the written scene's name.
        robot_name: Robot asset name.

    Returns:
        ``{tile_id: Course}``, already written to ``robots/<robot_name>/``.
    """
    out = {}
    for t in tiles:
        course = terrains.isaac_tile_course(clip_lib, t, speeds=(speed,), approach=approach,
                                            goal_anchor_max_offset=max_offset)
        course = dataclasses.replace(course, name=f"tile{t}{scene_tag}")
        terrains.write_scene(course, robot_name)
        print(f"  tile {t}: {course.klass}, budget {course.budget_s:.2f}s, "
              f"anchor {course.goal_anchor_xyz}")
        out[t] = course
    return out


def main() -> int:
    """Entry point: build the offered tiles, then loop on stdin."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--load_run", required=True, help="tracker run dir name")
    p.add_argument("--fm_onnx", required=True)
    p.add_argument("--tiles", default="6,20", help="comma list of tile ids to offer")
    p.add_argument("--history_from", default="generator",
                   choices=["generator", "measured"],
                   help="What fills the generator's N history rows: its own served nodes "
                        "('generator', the deployed closed loop) or the ROBOT'S OWN achieved "
                        "state ('measured'), bar the base position and linear velocity, "
                        "which no robot can measure. Forwarded to gen_tracker_runner.py.")
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--approach", type=float, default=2.5)
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--rt_factor", type=float, default=1.0,
                   help="viewer playback speed; 1.0 = real time")
    p.add_argument("--zero_noise", action="store_true",
                   help="start in deterministic x0=0 mode (the hardware default) instead of "
                        "the stochastic gaussian mode the live_select table was measured "
                        "under. Toggle live with 'z'.")
    p.add_argument("--scene_tag", default="_watch")
    p.add_argument("--robot_name", default="g1_perceptive")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--steer_kp", type=float, default=1.5,
                   help="P gain (1/s) used when steering is toggled ON with 's'. Ported from "
                        "mjeval.runner.steer_yaw_rate. OFF-DISTRIBUTION ON STAIRS -- the "
                        "training library commands zero turn rate on every stair clip, so "
                        "this is a genuinely different measurement there, not a cleaner one.")
    p.add_argument("--steer_wz_max", type=float, default=0.68)
    p.add_argument("--steer", action="store_true", help="start with steering ON")
    args = p.parse_args()

    tiles = [int(t) for t in args.tiles.split(",") if t.strip()]
    print(f"[watch] building {len(tiles)} tile scene(s)...")
    courses = _build_courses(args.clip_lib, tiles, args.speed, args.approach, args.max_offset,
                             args.scene_tag, args.robot_name)

    cur = tiles[0]
    zero_noise = bool(args.zero_noise)
    steer = bool(args.steer)
    n = 0
    print(f"\nEnter = next draw on tile {cur}; a tile id ({tiles}) = switch; "
          "z = toggle zero/gaussian noise; s = toggle steering; q = quit\n")
    if steer:
        print("[watch] steering starts ON -- OFF-DISTRIBUTION ON STAIRS, see --steer_kp's help")
    while True:
        mode = "zero-noise (deterministic)" if zero_noise else "gaussian (stochastic)"
        steer_tag = f"steer P={args.steer_kp:.2f}" if steer else "no steer"
        raw = input(f"[tile {cur}, draw {n + 1}, {mode}, {steer_tag}] > ").strip().lower()
        if raw == "q":
            break
        if raw == "z":
            zero_noise = not zero_noise
            continue
        if raw == "s":
            steer = not steer
            if steer:
                print("[watch] steering ON -- OFF-DISTRIBUTION ON STAIRS, see --steer_kp's help")
            continue
        if raw:
            try:
                t = int(raw)
            except ValueError:
                print(f"  not a tile id, 'z', 's', or 'q': {raw!r}")
                continue
            if t not in courses:
                print(f"  tile {t} was not in --tiles {tiles}; restart with it included")
                continue
            cur, n = t, 0
        course = courses[cur]
        n += 1
        # mjeval's own rule for where a goal marker sits on the ground: the flight height
        # gained on a climb (floor_z stays 0 there), or the already-lowered floor on a
        # descent -- see mjeval/view_cell.py's target_z.
        goal_z = course.floor_z if course.flight_height_m < 0 else course.flight_height_m
        argv = [sys.executable, os.path.join(_HERE, "gen_tracker_runner.py"),
                "--scene", course.name, "--robot_name", args.robot_name,
                "--total_time", f"{course.budget_s:.2f}",
                "--const_cmd", f"{args.speed}", "0", "0",
                "--rt_factor", f"{args.rt_factor}",
                "--load_run", args.load_run,
                "--fm_onnx", args.fm_onnx,
                "--fall_height", f"{args.fall_height}",
                "--fall_tilt_deg", f"{args.fall_tilt_deg}",
                "--goal_xy", f"{course.target_xy[0]}", f"{course.target_xy[1]}",
                "--goal_half_width", f"{course.goal_half_width_m}",
                "--goal_z", f"{goal_z}",
                # RESTATED here because this is a subprocess boundary, which is where flags
                # go missing: unforwarded it would revert to the child's argparse default
                # and every episode would silently be the closed-loop arm.
                "--history_from", args.history_from,
                "--allow_robot_asset_mismatch"]
        if steer:
            argv += ["--steer_kp", f"{args.steer_kp}", "--steer_wz_max", f"{args.steer_wz_max}"]
        if not zero_noise:
            argv.append("--gaussian_noise")
        print(f"  $ {' '.join(argv)}")
        subprocess.run(argv, cwd=_HERE, env=_CHILD_ENV, check=False)

    print("[watch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
