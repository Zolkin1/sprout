# Copyright 2026 Zachary Olkin. All rights reserved.

"""Watch ONE cell of the eval grid -- the exact episode that produced its table row.

The grid is 400 numbers and a number cannot say *how* a robot failed.  This replays a chosen
``(course, speed, arm)`` in the MuJoCo viewer so a fall can be looked at.

**It captures and replays rather than driving the viewer live, and that is the whole design.**
``runner.py`` is headless-only on purpose: attaching the viewer perturbs ``mjData`` enough to
change the trajectory -- one course measured here survived 16 s headless and fell reproducibly
at 6.04 s with the viewer attached.  So a live re-run would show a DIFFERENT episode from the
one in the table, which is worse than useless for a spot check: it invites a conclusion about a
rollout that was never scored.  Instead this re-runs the cell HEADLESS with a per-step capture,
checks the resulting row against the recorded one, and then plays the captured trajectory back.
Playback only writes ``qpos`` and calls ``mj_forward``, so the viewer cannot perturb anything --
there is no physics and no policy in the loop.

**Re-running is sound because the sweep is deterministic**: ``zero_noise`` integrates the flow
from ``x0 = 0``, no domain randomisation or depth noise is applied, and MuJoCo is deterministic.
Thirteen runs of one episode across 1-, 4- and 8-way concurrency came back bit-identical to
0.000e+00 m.  :func:`verify_against_table` asserts it per cell anyway rather than trusting it,
because "the replay is the scored episode" is the one claim this tool exists to make.

``--live`` is the deliberate escape hatch: physics in the viewer, camera free to fly, all of
``gen_tracker_runner``'s viz available (the FM ribbon, the depth cloud, the velocity arrows) --
and a trajectory that will not match the table.  It says so before every run.

Run from ``transfer/sim/mjeval`` with ``env_isaaclab`` active::

    python view_cell.py --run <out> --list                      # the grid, both arms
    python view_cell.py --run <out> --list --disagree           # only cells the arms differ on
    python view_cell.py --run <out> --course mjeval_dn_w196_modal --speed 1.947 --arm finetune
    python view_cell.py --run <out> --terrain_idx 9 --speed_idx 6 --arm finetune --rt 0.25
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import evaluate  # noqa: E402
import schema  # noqa: E402
import terrains  # noqa: E402

#: Colours for the drawn overlays, RGBA.  The target ring is the report's "success" teal and
#: the plan ribbon its "baseline" violet, so the viewer and the write-up agree on what is what.
TARGET_RGBA = (0.06, 0.48, 0.38, 0.55)
PLAN_RGBA = (0.54, 0.49, 0.66, 0.85)

#: The GHOST arm's plan ribbon. Deliberately a different hue from PLAN_RGBA:
#: two ribbons in one colour read as one plan with a kink in it.
GHOST_PLAN_RGBA = (1.0, 0.55, 0.10, 0.85)
ROOT_RGBA = (0.72, 0.52, 0.17, 0.9)

#: Ghost colours per arm, matching the written report's palette so the two agree on which arm
#: is which without a legend.  Translucent, so the solid arm reads through the overlap -- which
#: is the whole point when the two are drawn at the same place.
GHOST_RGBA: dict[str, tuple] = {
    "baseline": (0.54, 0.49, 0.66, 0.42),
    "finetune": (0.06, 0.48, 0.38, 0.42),
    "ft_v3": (0.80, 0.42, 0.12, 0.42),
    "baseline_meas": (0.72, 0.20, 0.30, 0.42),
}


def load_rows(run_dir: str, label: str) -> dict[str, np.ndarray]:
    """One arm's episode table.

    Args:
        run_dir: The eval run directory.
        label: Arm label, e.g. ``"baseline"``.

    Returns:
        The columns dict.

    Raises:
        SystemExit: When the arm has no ``episodes.h5``.
    """
    return load_table(run_dir, label)[0]


def load_table(run_dir: str, label: str) -> tuple[dict[str, np.ndarray], dict]:
    """One arm's episode table AND the attributes recording how it was run.

    Args:
        run_dir: The eval run directory.
        label: Arm label.

    Returns:
        ``(columns, attrs)``.

    Raises:
        SystemExit: When the arm has no ``episodes.h5``.
    """
    path = os.path.join(run_dir, label, "episodes.h5")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found -- has that arm run?")
    return schema.read_table(path)


def find_row(cols: dict[str, np.ndarray], terrain_idx: int, speed_idx: int) -> int:
    """Index of one cell's row.

    Args:
        cols: An arm's columns.
        terrain_idx: Course index into :func:`terrains.catalogue`.
        speed_idx: Index into that course's speed ladder.

    Returns:
        The row index.

    Raises:
        SystemExit: When the cell is not in the table.
    """
    hit = np.flatnonzero((cols["terrain_idx"] == terrain_idx)
                         & (cols["speed_idx"] == speed_idx))
    if hit.size != 1:
        raise SystemExit(f"cell (terrain {terrain_idx}, speed {speed_idx}) matched "
                         f"{hit.size} rows; expected exactly one")
    return int(hit[0])


def resolve_cell(args, courses: tuple) -> tuple[int, int]:
    """Turn the CLI's course/speed selectors into indices.

    Accepts a course by name or index and a speed by value or index, because a cell is read off
    the report by name and off the table by index and neither should need translating by hand.

    Args:
        args: Parsed CLI arguments.
        courses: :func:`terrains.catalogue` output.

    Returns:
        ``(terrain_idx, speed_idx)``.

    Raises:
        SystemExit: On an unknown course name, or a speed matching no ladder entry.
    """
    if args.terrain_idx is not None:
        terrain_idx = args.terrain_idx
        if not 0 <= terrain_idx < len(courses):
            raise SystemExit(f"--terrain_idx {terrain_idx} outside 0..{len(courses) - 1}")
    else:
        names = [c.name for c in courses]
        if args.course not in names:
            raise SystemExit(f"unknown course {args.course!r}. Available:\n  "
                             + "\n  ".join(f"{i:2d} {n}" for i, n in enumerate(names)))
        terrain_idx = names.index(args.course)

    ladder = courses[terrain_idx].speeds
    if args.speed_idx is not None:
        speed_idx = args.speed_idx
        if not 0 <= speed_idx < len(ladder):
            raise SystemExit(f"--speed_idx {speed_idx} outside 0..{len(ladder) - 1}")
    else:
        near = [i for i, v in enumerate(ladder) if abs(v - args.speed) < 5e-3]
        if not near:
            raise SystemExit(f"--speed {args.speed} matches no rung of this course's ladder "
                             f"{ladder}. Pass --speed_idx to choose by position.")
        speed_idx = near[0]
    return terrain_idx, speed_idx


def print_grid(run_dir: str, labels: list[str], disagree_only: bool) -> None:
    """Print every cell's verdict for each arm, so a cell can be picked to look at.

    Args:
        run_dir: The eval run directory.
        labels: Arm labels, in column order.
        disagree_only: Print only cells whose arms reached different verdicts -- which is where
            a spot check has the most to learn, since the geometry and the command were
            identical and only the tracker differed.

    Cells the run did not cover are SKIPPED rather than fatal: a smoke run truncates the
    catalogue (``--max_courses`` / ``--max_speeds``) and a sharded one holds a residue class,
    so demanding the full 20x10 grid meant the browser could not open the table it was
    pointed at. A cell present for some arms and missing for others is still an error -- that
    is a torn run, not a truncated one.
    """
    tables = {lab: load_rows(run_dir, lab) for lab in labels}
    courses = terrains.catalogue()
    names = {v: k for k, v in terrains.VERDICT_CODES.items()}
    short = {"success": "OK  ", "fall": "FALL", "timeout": "time", "interrupted": "intr"}

    head = f"{'idx':>3s} {'course':<28s} {'spd_i':>5s} {'cmd':>6s}  " \
           + "  ".join(f"{lab:<10s}" for lab in labels)
    print(head)
    print("-" * len(head))
    shown = 0
    for terrain_idx, course in enumerate(courses):
        for speed_idx, speed in enumerate(course.speeds):
            present = [lab for lab in labels
                       if np.any((tables[lab]["terrain_idx"] == terrain_idx)
                                 & (tables[lab]["speed_idx"] == speed_idx))]
            if not present:
                continue                      # a cell this run did not cover
            if len(present) != len(labels):
                raise SystemExit(
                    f"cell (terrain {terrain_idx}, speed {speed_idx}) is in {present} but "
                    f"not in {[la for la in labels if la not in present]}. The arms cover "
                    "different cells, so no paired comparison over this run is valid.")
            verdicts = []
            for lab in labels:
                r = find_row(tables[lab], terrain_idx, speed_idx)
                verdicts.append(names[int(tables[lab]["verdict"][r])])
            if disagree_only and len(set(verdicts)) == 1:
                continue
            cells = "  ".join(f"{short.get(v, v):<10s}" for v in verdicts)
            print(f"{terrain_idx:3d} {course.name:<28s} {speed_idx:5d} {speed:6.3f}  {cells}")
            shown += 1
    print("-" * len(head))
    print(f"{shown} cell(s)" + (" where the arms disagree" if disagree_only else ""))


def capture(run_dir: str, label: str, terrain_idx: int, speed_idx: int, args
            ) -> tuple[dict, list, list]:
    """Re-run one cell headless, capturing a replayable trajectory.

    Args:
        run_dir: The eval run directory (for the arm's artifact paths).
        label: Arm label.
        terrain_idx: Course index.
        speed_idx: Speed-ladder index.
        args: Parsed CLI arguments, for the physical thresholds.

    Returns:
        ``(summary, frames, plans)``.  ``frames`` is a list of ``(time, qpos_copy)``; ``plans``
        is a list of ``(step, [F, 3] world root path)`` captured once per re-plan.
    """
    import runner  # noqa: PLC0415 - mujoco + the deploy runtime; --list must not pay for it

    arm = {a.label: a for a in evaluate.ARMS}[label]
    fm_onnx = generator_from_manifest(run_dir, arm.fm_onnx)
    course = terrains.catalogue()[terrain_idx]
    speed = course.speeds[speed_idx]

    # **The run's own attributes decide the config, not this CLI's defaults.**  A grid may have
    # been swept with the steering controller on or with a different command hold, and a replay
    # under different settings is a different episode -- which `verify_against_table` would
    # correctly refuse, leaving the operator with a tool that simply stops working on half the
    # runs.  Reading it back means the viewer reproduces whatever was actually run, and the
    # config is printed so it is never silently assumed.
    attrs = load_table(run_dir, label)[1]
    steer_kp = float(attrs.get("steer_kp", 0.0))
    steer_wz_max = float(attrs.get("steer_wz_max", runner.STEER_WZ_MAX))
    cmd_hold_s = float(attrs.get("cmd_hold_s", terrains.CMD_HOLD_S))
    # Read from the TABLE, like the three above, and for a sharper reason: the history source
    # and the noise draw change the trajectory completely while changing no shape, so a
    # replay that defaulted them would re-run a different experiment and blame the mismatch
    # on the checker. An older table names neither, and the defaults are what it ran.
    history_from = str(attrs.get("history_from", "generator"))
    zero_noise = bool(attrs.get("zero_noise", True))
    fm_seed = int(attrs.get("fm_seed", 0))
    print(f"  config from the run: steer_kp {steer_kp:g}"
          + (f" (wz clip {steer_wz_max:g} rad/s)" if steer_kp > 0 else " -- steering off")
          + f", cmd_hold {cmd_hold_s:g} s, history_from {history_from}"
          + (", x0=0" if zero_noise else f", GAUSSIAN x0 seed {fm_seed}"), flush=True)

    cfg = runner.ArmRun(
        arm=arm.code, label=arm.label,
        tracker_onnx=arm.tracker_onnx, fm_onnx=fm_onnx,
        out_dir=os.path.join(run_dir, arm.label),
        robot_name=args.robot_name, providers=tuple(args.providers.split(",")),
        fall_height=args.fall_height, fall_tilt_deg=args.fall_tilt_deg,
        spawn_z_offset=args.spawn_z_offset,
        steer_kp=steer_kp, steer_wz_max=steer_wz_max, cmd_hold_s=cmd_hold_s,
        history_from=history_from, zero_noise=zero_noise, fm_seed=fm_seed,
    )
    policy, fm_lag, spawn = runner.load_arm(cfg)
    rig = runner.CourseRig(course, policy, cfg, spawn)

    frames: list[tuple[float, np.ndarray]] = []
    plans: list[tuple[int, np.ndarray]] = []
    last_plan_id = [-1]

    def record(robot, pol) -> None:
        """Capture one control step's pose, and the plan when it has just been re-made."""
        frames.append((float(robot.mj_data.time), robot.mj_data.qpos.copy()))
        plan = pol.last_plan()
        if plan is not None and int(plan["plan_id"]) != last_plan_id[0]:
            last_plan_id[0] = int(plan["plan_id"])
            path = pol.predicted_root_path_w()
            if path is not None:
                plans.append((len(frames) - 1, np.asarray(path, dtype=float).copy()))

    cmd = np.array([float(speed), 0.0, 0.0])
    verdict, failure, acc = runner.run_episode(rig, policy, fm_lag, spawn, cmd, None,
                                               recorder=record, steer_kp=steer_kp,
                                               steer_wz_max=steer_wz_max,
                                               cfg_hold=cmd_hold_s,
                                               # The same cell key run_arm used, or the
                                               # replay draws different noise than the row.
                                               episode_seed=(fm_seed, terrain_idx, speed_idx))
    row = acc.row(episode_id=terrain_idx * runner.EPISODE_ID_STRIDE + speed_idx,
                  terrain_idx=terrain_idx, speed_idx=speed_idx, arm=arm.code,
                  seed=cfg.recorded_seed, verdict=verdict, failure_code=failure)
    summary = {"course": course, "speed": speed, "verdict": verdict, "failure": failure,
               "row": row, "model": rig.robot.mj_model}
    return summary, frames, plans


def generator_from_manifest(run: str, default: str) -> str:
    """The generator the RUN scored, from its manifest -- not this module's default.

    `evaluate.py --fm_onnx` lets a run name its own generator, and every AWR iteration
    exports one, so the module constant is no longer "the" generator. Reading it here instead
    of the manifest replayed a DIFFERENT model than the one scored, which
    `verify_against_table` then correctly refused -- on 24 columns, with `max_tilt_rad` 1.225
    against 1.376 and the fall 5 steps later. That refusal is the only reason it was noticed,
    and it should not have needed to fire.

    Args:
        run: The run directory.
        default: What to fall back to when the manifest predates the field.

    Returns:
        The generator path.

    Raises:
        SystemExit: If the manifest names a generator that is no longer on disk. A silent
            fallback to the default would reproduce exactly the bug this function fixes.
    """
    import json                                                     # noqa: PLC0415

    mf = os.path.join(run, "manifest.json")
    if not os.path.exists(mf):
        return default
    with open(mf) as f:
        got = (json.load(f).get("artifacts") or {}).get("generator") or {}
    name = str(got.get("name") or "")
    if not name:
        return default
    if not os.path.exists(name):
        raise SystemExit(
            f"{mf} was scored with generator {name}, which is not on disk now. Replaying "
            "with anything else would show a rollout that was never scored -- restore that "
            "file, or re-run the sweep.")
    return name


def verify_against_table(row: dict, cols: dict[str, np.ndarray], idx: int) -> list[str]:
    """Check the captured re-run reproduced the recorded row.

    The claim this tool makes is "you are watching the episode that produced that number", and
    it is only true if the re-run landed on the same row.  Timing columns are excluded because
    they are wall clock and legitimately vary with load -- everything else must match exactly.

    **The comparison is made THROUGH the column's own dtype**, which is the whole subtlety.  The
    table stores ``f4``, so a stored 2.6 reads back as 2.5999999046325684 while the fresh row
    carries python's 2.6; comparing those as float64 reports every populated column as a
    mismatch.  The question being asked is not "are these two float64s equal" but "would this
    row have been WRITTEN identically", so each side is cast to the storage dtype first.  Doing
    it the naive way reports a perfect replay as a total mismatch, which is exactly the kind of
    false alarm that trains people to ignore a check.

    Args:
        row: The re-run's row.
        cols: The recorded arm's columns.
        idx: The recorded row's index.

    Returns:
        Human-readable descriptions of every column that differs; empty when the replay is the
        scored episode.
    """
    skip = {"fm_ms_mean", "trk_ms_mean"}
    out = []
    for name, dtype in schema.MJ_EPISODE_COLUMNS:
        if name in skip:
            continue
        want = np.asarray(cols[name], dtype=dtype)[idx]
        got = np.asarray(row[name], dtype=dtype)
        if np.isnan(want) and np.isnan(got):
            continue
        if want != got:
            out.append(f"{name}: table {float(want):.9g} != replay {float(got):.9g}")
    return out


def _goal_rect(scn, course, z: float) -> None:
    """Outline the goal rectangle on the ground.

    An outline of small spheres rather than a filled slab because a filled one z-fights with
    the platform it sits on, and the band's EDGE is what is being judged.

    Args:
        scn: The viewer's ``user_scn``.
        course: The course, for its goal centre and half-width.
        z: Height to draw at.
    """
    gx, gy = course.target_xy
    hw, hd = course.goal_half_width_m, terrains.GOAL_DEPTH / 2.0
    corners = [(gx - hd, gy - hw), (gx + hd, gy - hw), (gx + hd, gy + hw), (gx - hd, gy + hw)]
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.12))
        for k in range(n):
            if scn.ngeom >= scn.maxgeom:
                return
            f = k / n
            pos = np.array([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, z])
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.028, 0, 0]), pos, np.eye(3).ravel(),
                                np.array(TARGET_RGBA, dtype=np.float32))
            scn.ngeom += 1


def _dots(scn, pts: np.ndarray, rgba: tuple, size: float, stride: int = 1) -> None:
    """Draw a point sequence as spheres.

    Args:
        scn: The viewer's ``user_scn``.
        pts: ``[N, 3]`` world positions.
        rgba: Colour.
        size: Sphere radius in metres.
        stride: Draw every Nth point.
    """
    for p in pts[::stride]:
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([size, 0, 0]), np.asarray(p, dtype=float),
                            np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
        scn.ngeom += 1


#: Group of the robot's VISUAL meshes.  Group 0 is the collision proxy (a box where the foot
#: is), group 2 the terrain, group 3 the camera housings -- so group 1 is the only one that
#: looks like the robot, and it is what ``depth_camera.SELF_SCAN_GROUP`` casts too.
ROBOT_VISUAL_GROUP: int = 1


#: ``MjvGeom`` fields carried when a geom is copied between scenes.  Everything the renderer
#: reads, listed rather than introspected so a field added by a future MuJoCo shows up as a
#: visibly wrong ghost rather than as a silent omission.
_GEOM_FIELDS: tuple[str, ...] = (
    "type", "dataid", "objtype", "objid", "category", "matid", "texcoord", "segid",
    "size", "pos", "mat", "rgba", "emission", "specular", "shininess", "reflectance",
    "label", "camdist", "modelrbound", "transparent",
)


def _ghost(scn, model, gdata, opt, scratch, rgba: tuple) -> None:
    """Draw a second robot into the viewer's decor layer, from its own ``mjData``.

    The viewer renders exactly one ``mjData``, so the comparison arm cannot be a second
    simulated body.  It is drawn instead as its real visual meshes -- posed by its own forward
    kinematics -- because the thing being compared is a gait, and the shape of a limb is most
    of that signal.

    **MuJoCo builds those geoms, and they are copied rather than hand-made.**  A mesh geom's
    scene ``dataid`` is NOT ``model.geom_dataid``: the visualizer keeps two render buffers per
    mesh (the mesh and its convex hull), so the scene index is twice the model's.  Building the
    geom by hand and assigning ``model.geom_dataid`` therefore draws every limb as some OTHER
    limb's mesh -- which renders as a plausible-looking jumble of robot parts rather than as an
    error, and was caught only by drawing a known standing pose beside itself.  Going through
    ``mjv_updateScene`` uses MuJoCo's own mapping and cannot drift from it.

    Args:
        scn: The viewer's ``user_scn``, drawn into.
        model: The shared model -- both arms ran the same course.
        gdata: The ghost arm's ``MjData``, already forwarded to this frame.
        opt: Visualisation options selecting the robot's visual group.
        scratch: A scratch :class:`mujoco.MjvScene` reused every frame, so the per-frame cost is
            a scene update rather than an allocation.
        rgba: Colour, translucent so the primary arm reads through the overlap.
    """
    mujoco.mjv_updateScene(model, gdata, opt, None, mujoco.MjvCamera(),
                           int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scratch)
    for i in range(scratch.ngeom):
        if scn.ngeom >= scn.maxgeom:
            return
        src, dst = scratch.geoms[i], scn.geoms[scn.ngeom]
        for field in _GEOM_FIELDS:
            value = getattr(src, field)
            if isinstance(value, np.ndarray):
                getattr(dst, field)[:] = value
            else:
                setattr(dst, field, value)
        dst.rgba[:] = np.array(rgba, dtype=np.float32)
        # Decor, so the viewer's own picking and labelling -- which index the real model -- do
        # not resolve a click to the ghost.
        dst.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        dst.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
        dst.objid = -1
        dst.segid = -1
        scn.ngeom += 1


def replay(summary: dict, frames: list, plans: list, rt: float, loop: bool,
           ghost: dict | None = None) -> None:
    """Play the captured trajectory in the viewer.

    Writes ``qpos`` and calls ``mj_forward`` only: no physics is stepped and no policy runs, so
    the viewer cannot perturb what is being shown.  That is exactly why this is a replay and not
    a re-drive.

    Args:
        summary: :func:`capture`'s summary.
        frames: ``(time, qpos)`` per control step.
        plans: ``(step, [F, 3])`` per re-plan.
        rt: Real-time multiple; 1.0 is real time, 0.25 quarter speed.
        loop: Restart at the end instead of holding the last frame.
        ghost: ``{"label", "frames", "rgba", "offset_y", "verdict"}`` for a second arm drawn
            translucently, or ``None``.  **The two episodes are different lengths** -- one may
            fall while the other is still climbing -- so the shorter track holds its final pose
            rather than looping or vanishing, and the console says which ended when.  Vanishing
            would read as "it left", which is the opposite of what a fall means.
    """
    import mujoco.viewer  # noqa: PLC0415 - opens a GL context; --list must not

    course, model = summary["course"], summary["model"]
    data = mujoco.MjData(model)
    plan_at = {step: path for step, path in plans}
    target_z = course.floor_z if course.flight_height_m < 0 else course.flight_height_m

    gdata = ghost_opt = ghost_scn = None
    ghost_plan_at: dict[int, np.ndarray] = {}
    n_steps = len(frames)
    if ghost is not None:
        # Optional, so `view_cell`'s own caller -- which passes no plans -- is unchanged.
        ghost_plan_at = {step: path for step, path in ghost.get("plans", ())}
        gdata = mujoco.MjData(model)
        ghost_opt = mujoco.MjvOption()
        ghost_opt.geomgroup[:] = 0
        ghost_opt.geomgroup[ROBOT_VISUAL_GROUP] = 1
        ghost_scn = mujoco.MjvScene(model, maxgeom=2048)
        n_steps = max(n_steps, len(ghost["frames"]))
        print(f"  ghost  {ghost['label']}: {len(ghost['frames'])} steps, verdict "
              f"{ghost['verdict']}"
              + (f", offset y {ghost['offset_y']:+.2f} m" if ghost["offset_y"] else
                 " (overlaid at true position)"), flush=True)

    print(f"\n  replaying {len(frames)} steps ({frames[-1][0]:.2f} s) at {rt:g}x real time. "
          "Close the window to exit.\n", flush=True)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 130.0
        current_plan = None
        current_ghost_plan = None
        while viewer.is_running():
            wall0 = time.time()
            for i in range(n_steps):
                if not viewer.is_running():
                    return
                # Each track holds its OWN last frame once it has ended, so a fall stays down
                # on screen while the other arm carries on.
                t, qpos = frames[min(i, len(frames) - 1)]
                data.qpos[:] = qpos
                data.time = t
                mujoco.mj_forward(model, data)
                if i in plan_at:
                    current_plan = plan_at[i]
                if i in ghost_plan_at:
                    current_ghost_plan = ghost_plan_at[i]

                viewer.user_scn.ngeom = 0
                if ghost is not None:
                    _, gq = ghost["frames"][min(i, len(ghost["frames"]) - 1)]
                    gdata.qpos[:] = gq
                    gdata.qpos[1] += ghost["offset_y"]
                    mujoco.mj_forward(model, gdata)
                    _ghost(viewer.user_scn, model, gdata, ghost_opt, ghost_scn,
                           ghost["rgba"])
                _goal_rect(viewer.user_scn, course, target_z)
                if current_plan is not None:
                    _dots(viewer.user_scn, current_plan, PLAN_RGBA, 0.022, stride=2)
                if current_ghost_plan is not None:
                    # Shifted with the ghost body, so the ribbon stays attached to the robot
                    # it belongs to when --offset_y separates the two arms.
                    gp = current_ghost_plan.copy()
                    gp[:, 1] += ghost["offset_y"]
                    _dots(viewer.user_scn, gp, GHOST_PLAN_RGBA, 0.022, stride=2)
                _dots(viewer.user_scn, qpos[:3].reshape(1, 3), ROOT_RGBA, 0.05)
                # Track the robot rather than leaving it to walk out of frame; the operator can
                # still orbit, because only the lookat point is written.
                viewer.cam.lookat[:] = qpos[:3]
                viewer.sync()

                if rt > 0:
                    ahead = (i * (frames[1][0] - frames[0][0] if len(frames) > 1 else 0.02)) \
                        / rt - (time.time() - wall0)
                    if ahead > 0:
                        time.sleep(ahead)
            if not loop:
                break
        # Hold the final frame so a fall can be inspected rather than flashing past.
        while viewer.is_running() and not loop:
            viewer.sync()
            time.sleep(0.05)


def live(run_dir: str, label: str, terrain_idx: int, speed_idx: int, args) -> None:
    """Drive the cell in the viewer with physics running -- NOT the scored episode.

    Args:
        run_dir: Unused; kept so the two entry points take the same arguments.
        label: Arm label.
        terrain_idx: Course index.
        speed_idx: Speed-ladder index.
        args: Parsed CLI arguments.
    """
    import subprocess  # noqa: PLC0415

    arm = {a.label: a for a in evaluate.ARMS}[label]
    course = terrains.catalogue()[terrain_idx]
    speed = course.speeds[speed_idx]
    argv = [sys.executable, os.path.join(os.path.dirname(_HERE), "gen_tracker_runner.py"),
            "--scene", course.name, "--robot_name", args.robot_name,
            "--total_time", f"{course.budget_s:.2f}",
            "--const_cmd", f"{speed}", "0", "0",
            "--rt_factor", f"{args.rt}",
            "--load_run", os.path.basename(os.path.dirname(os.path.dirname(arm.tracker_onnx))),
            "--fm_onnx", generator_from_manifest(args.run, arm.fm_onnx),
            "--fall_height", f"{args.fall_height}",
            "--fall_tilt_deg", f"{args.fall_tilt_deg}",
            "--allow_robot_asset_mismatch", "--viz"]
    print("\n  !! --live drives PHYSICS in the viewer, which perturbs mjData. The trajectory")
    print("     will NOT match the table row: one course here survived 16 s headless and fell")
    print("     at 6.04 s headed, reproducibly. Use the default replay to see the scored run.\n")
    print(f"  $ {' '.join(argv)}\n")
    subprocess.run(argv, cwd=os.path.dirname(_HERE), check=False)


def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="an mjeval run directory (holds baseline/, finetune/)")
    p.add_argument("--arm", default="finetune", help="which arm to watch")
    p.add_argument("--list", action="store_true", help="print the grid and exit")
    p.add_argument("--disagree", action="store_true",
                   help="with --list, show only cells whose arms reached different verdicts")
    p.add_argument("--course", default=None, help="course NAME, e.g. mjeval_dn_w196_modal")
    p.add_argument("--terrain_idx", type=int, default=None, help="course INDEX (0..19)")
    p.add_argument("--speed", type=float, default=None, help="commanded speed, must be a rung")
    p.add_argument("--speed_idx", type=int, default=None, help="rung index (0..9)")
    p.add_argument("--rt", type=float, default=1.0,
                   help="playback speed; 0.25 is quarter speed, 0 is as fast as it renders")
    p.add_argument("--loop", action="store_true", help="repeat instead of holding the last frame")
    p.add_argument("--compare", action="store_true",
                   help="draw BOTH arms at once: --arm solid, the other translucent. Same "
                        "course, same command, so only the tracker differs")
    p.add_argument("--offset_y", type=float, default=0.0,
                   help="with --compare, shift the ghost arm sideways (m). 0 overlays them at "
                        "true position, which is what shows the divergence directly")
    p.add_argument("--live", action="store_true",
                   help="drive physics in the viewer instead of replaying -- NOT the scored run")
    p.add_argument("--robot_name", default="g1_perceptive")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--spawn_z_offset", type=float, default=0.03)
    return p


def _refuse(drift: list[str], label: str) -> bool:
    """Report a failed verification, if there was one.

    Args:
        drift: :func:`verify_against_table`'s output.
        label: Which arm it was checking.

    Returns:
        True when the caller should abort rather than show the rollout.
    """
    if not drift:
        print(f"  replay [{label}] verified against the table on all "
              f"{len(schema.MJ_EPISODE_COLUMNS) - 2} non-timing columns "
              f"-- this IS the scored episode", flush=True)
        return False
    print(f"\n  [FATAL] the [{label}] replay did NOT reproduce the recorded row:", flush=True)
    for line in drift:
        print(f"    {line}")
    print("  Refusing to show a rollout that is not the one that was scored.", flush=True)
    return True


def main() -> int:
    """Resolve the cell, capture it, verify it and play it.

    Returns:
        Process exit code; 1 when the replay did not reproduce the recorded row.
    """
    args = build_parser().parse_args()
    run_dir = os.path.abspath(args.run)
    # The arms that actually RAN here, not every arm the table declares. A run is free to
    # cover a subset (`evaluate.py --arm baseline,baseline_meas` does), and listing a
    # declared-but-absent one used to abort the whole browse on its missing episodes.h5 --
    # so the tool could not show you the grid of the run you had.
    labels = [a.label for a in evaluate.ARMS
              if os.path.isfile(os.path.join(run_dir, a.label, "episodes.h5"))]
    if not labels:
        raise SystemExit(
            f"no arm under {run_dir} has an episodes.h5. Declared arms: "
            f"{[a.label for a in evaluate.ARMS]}")
    absent = [a.label for a in evaluate.ARMS if a.label not in labels]

    if args.list:
        if absent:
            print(f"[INFO] arms present: {labels}; not run here: {absent}")
        print_grid(run_dir, labels, args.disagree)
        return 0

    courses = terrains.catalogue()
    if args.course is None and args.terrain_idx is None:
        raise SystemExit("pass --course or --terrain_idx (or --list to browse the grid)")
    if args.speed is None and args.speed_idx is None:
        raise SystemExit("pass --speed or --speed_idx")
    terrain_idx, speed_idx = resolve_cell(args, courses)
    course = courses[terrain_idx]

    cols = load_rows(run_dir, args.arm)
    idx = find_row(cols, terrain_idx, speed_idx)
    names = {v: k for k, v in terrains.VERDICT_CODES.items()}
    print(f"\n  cell   {course.name}  @  {course.speeds[speed_idx]:.3f} m/s   [{args.arm}]")
    print(f"  table  verdict {names[int(cols['verdict'][idx])]}   "
          f"steps {int(cols['steps_survived'][idx])}   "
          f"progress {float(cols['progress_frac'][idx]):.3f}   "
          f"reached_target {bool(cols['reached_target'][idx])}   "
          f"traversed {bool(cols['traversed'][idx])}")
    print(f"  course {course.note}", flush=True)

    if args.live:
        live(run_dir, args.arm, terrain_idx, speed_idx, args)
        return 0

    summary, frames, plans = capture(run_dir, args.arm, terrain_idx, speed_idx, args)
    if _refuse(verify_against_table(summary["row"], cols, idx), args.arm):
        return 1
    if not frames:
        raise SystemExit("captured no frames; the episode ended before the first fold")

    ghost = None
    if args.compare:
        other = next((a.label for a in evaluate.ARMS if a.label != args.arm), None)
        if other is None:
            raise SystemExit("--compare needs two arms in the ARMS table")
        ocols = load_rows(run_dir, other)
        oidx = find_row(ocols, terrain_idx, speed_idx)
        print(f"\n  cell   {course.name}  @  {course.speeds[speed_idx]:.3f} m/s   [{other}]")
        print(f"  table  verdict {names[int(ocols['verdict'][oidx])]}   "
              f"steps {int(ocols['steps_survived'][oidx])}   "
              f"progress {float(ocols['progress_frac'][oidx]):.3f}", flush=True)
        osummary, oframes, _oplans = capture(run_dir, other, terrain_idx, speed_idx, args)
        if _refuse(verify_against_table(osummary["row"], ocols, oidx), other):
            return 1
        ghost = {"label": other, "frames": oframes, "verdict": osummary["verdict"],
                 "rgba": GHOST_RGBA.get(other, (0.55, 0.55, 0.60, 0.45)),
                 "offset_y": args.offset_y}

    replay(summary, frames, plans, args.rt, args.loop, ghost=ghost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
