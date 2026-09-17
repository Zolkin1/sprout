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

Two modes draw what the generator could have done instead, and they answer different questions.

``--plans K`` shows the OPTIONS.  At every re-plan it draws K more plans sampled from the same
conditioning, each in its own colour, with the served one in violet -- so a re-plan where the
generator was choosing between climbing a flight and hopping onto a box is visible as such,
which no aggregate number can show.  It costs K generator calls per re-plan, no extra rollouts,
and cannot perturb the scored episode (see :func:`alternative_plans`).  Pair it with
``--replan_hold_s`` -- a re-plan interval is 0.24 s, so the fan is otherwise gone before it can
be read.

``--choose K`` makes it INTERACTIVE: physics runs, the sim pauses at every re-plan showing the
served plan and K alternatives, and you pick which one the robot follows.  Picks compound --
the chosen plan becomes the history the next re-plan conditions on -- so this is the mode for
asking "what if it had committed to the stairs here".  It is necessarily not the scored
episode; see :func:`choose`.

``--branch K`` shows the CONSEQUENCES.  It rolls K more episodes identical to the scored one
except for ONE re-plan's draw and draws where each ended up, teal for survived and red for
fell.  The grid runs the generator at its mode (``x0 = 0``), so a cell that falls may be one
draw away from clearing the terrain or may fall on every draw, and those are different
problems.  See :func:`capture_branches` for why exactly one draw differs.

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
import signal
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

#: Branch colours, by REALIZED OUTCOME rather than by index.  The question a sampled branch
#: answers is "would a different draw here have survived", so the eye should sort them into two
#: piles and not into K.  Teal is the report's success colour, red its failure one.
SAMPLE_SURVIVED_RGBA = (0.06, 0.62, 0.48, 0.80)
SAMPLE_FELL_RGBA = (0.78, 0.18, 0.20, 0.70)

#: The instant the branches diverge, drawn on the reference's own path.  Bright and neutral:
#: it belongs to neither pile.
BRANCH_RGBA = (0.98, 0.85, 0.10, 0.95)

#: Candidate-plan colours, by INDEX rather than by outcome.  These are options the generator
#: had at one instant and none of them has an outcome, so the eye should be able to follow one
#: ribbon rather than sort them into piles.  Chosen to stay apart from the served plan's violet
#: and from the terrain.
ALT_PLAN_RGBA: tuple[tuple[float, float, float, float], ...] = (
    (0.20, 0.60, 1.00, 0.85),   # blue
    (0.20, 0.85, 0.35, 0.85),   # green
    (0.95, 0.25, 0.75, 0.85),   # magenta
    (0.98, 0.75, 0.10, 0.85),   # amber
    (0.15, 0.85, 0.85, 0.85),   # cyan
    (1.00, 0.45, 0.15, 0.85),   # orange
    (0.65, 0.45, 1.00, 0.85),   # violet-blue
    (0.85, 0.85, 0.85, 0.85),   # light grey
)

#: Geoms held back from the branch-ribbon budget for the goal rectangle, the plan ribbon, the
#: root marker and a possible ghost.  The ribbon stride is derived from what is left, because
#: ``_dots`` truncates SILENTLY at ``maxgeom`` and a half-drawn branch reads as a short one --
#: which is the same picture as an early fall.
_OVERLAY_GEOM_RESERVE: int = 400


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


def _build_cell(run_dir: str, label: str, terrain_idx: int, speed_idx: int, args) -> dict:
    """Load one cell's arm and rig it on its course, under the RUN's own configuration.

    Split out of :func:`capture` because the sampled-branch view rolls the same cell many
    times and loading the pair costs seconds: everything here is per-CELL, and nothing in it
    depends on the noise draws, so it is built once and reused by every branch.

    Args:
        run_dir: The eval run directory (for the arm's artifact paths).
        label: Arm label.
        terrain_idx: Course index.
        speed_idx: Speed-ladder index.
        args: Parsed CLI arguments, for the physical thresholds.

    Returns:
        ``{"arm", "cfg", "policy", "fm_lag", "spawn", "rig", "course", "speed", "zero_noise",
        "fm_seed"}``.
    """
    import runner  # noqa: PLC0415 - mujoco + the deploy runtime; --list must not pay for it

    arm = {a.label: a for a in evaluate.ARMS}[label]
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
        tracker_onnx=arm.tracker_onnx, fm_onnx=arm.fm_onnx,
        out_dir=os.path.join(run_dir, arm.label),
        robot_name=args.robot_name, providers=tuple(args.providers.split(",")),
        fall_height=args.fall_height, fall_tilt_deg=args.fall_tilt_deg,
        spawn_z_offset=args.spawn_z_offset,
        steer_kp=steer_kp, steer_wz_max=steer_wz_max, cmd_hold_s=cmd_hold_s,
        history_from=history_from, zero_noise=zero_noise, fm_seed=fm_seed,
    )
    policy, fm_lag, spawn = runner.load_arm(cfg)
    rig = runner.CourseRig(course, policy, cfg, spawn)
    return {"arm": arm, "cfg": cfg, "policy": policy, "fm_lag": fm_lag, "spawn": spawn,
            "rig": rig, "course": course, "speed": speed, "zero_noise": zero_noise,
            "fm_seed": fm_seed}


def _roll(cell: dict, terrain_idx: int, speed_idx: int, recorder) -> tuple[str, str, dict]:
    """One episode of an already-built cell, at the sweep's own episode key.

    Args:
        cell: :func:`_build_cell`'s dict.
        terrain_idx: Course index, part of the episode key.
        speed_idx: Speed-ladder index, part of the episode key.
        recorder: ``(robot, policy) -> None``, called once per control step at the fold.

    Returns:
        ``(verdict, failure_code, row)``.
    """
    import runner  # noqa: PLC0415

    cfg, arm = cell["cfg"], cell["arm"]
    cmd = np.array([float(cell["speed"]), 0.0, 0.0])
    verdict, failure, acc = runner.run_episode(
        cell["rig"], cell["policy"], cell["fm_lag"], cell["spawn"], cmd, None,
        recorder=recorder, steer_kp=cfg.steer_kp, steer_wz_max=cfg.steer_wz_max,
        cfg_hold=cfg.cmd_hold_s,
        # The same cell key run_arm used, or the replay draws different noise than the row.
        episode_seed=(cell["fm_seed"], terrain_idx, speed_idx))
    row = acc.row(episode_id=terrain_idx * runner.EPISODE_ID_STRIDE + speed_idx,
                  terrain_idx=terrain_idx, speed_idx=speed_idx, arm=arm.code,
                  seed=cfg.recorded_seed, verdict=verdict, failure_code=failure)
    return verdict, failure, row


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
        is a list of ``(step, [F, 3] world root path)`` captured once per re-plan.  With
        ``--plans K`` the summary also carries ``alt_plans``, ``(step, [paths])`` per re-plan.
    """
    cell = _build_cell(run_dir, label, terrain_idx, speed_idx, args)
    frames, plans, replans = [], [], []
    n_alt = int(getattr(args, "plans", 0) or 0)
    alts: list = []
    served_feet: list = []
    rng = np.random.default_rng(int(getattr(args, "plan_seed", 0)))
    fk = None
    if getattr(args, "feet", False):
        fk = PlanFk(cell["rig"].robot.mj_model, cell["rig"].robot, cell["policy"])
        print(f"  deriving foot paths ({', '.join(FOOT_BODIES)}) from each plan by MuJoCo "
              "kinematics -- the window carries joint angles, not feet", flush=True)
    if n_alt:
        print(f"  sampling {n_alt} alternative plan(s) at every re-plan -- the generator is "
              "re-run on the conditioning it already used, so the episode is untouched",
              flush=True)
    verdict, failure, row = _roll(
        cell, terrain_idx, speed_idx,
        _frame_recorder(frames, plans, replans, alts, n_alt, rng, fk, served_feet))
    summary = {"course": cell["course"], "speed": cell["speed"], "verdict": verdict,
               "failure": failure, "row": row, "model": cell["rig"].robot.mj_model,
               "alt_plans": alts, "frames": frames, "served_feet": served_feet}
    return summary, frames, plans


#: Bodies drawn as the feet.  The ankle ROLL link is the last body in each leg chain, so its
#: frame moves with the sole; the pitch link above it barely swings and would understate a step.
FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")


class PlanFk:
    """World foot positions for a predicted window, by MuJoCo forward kinematics.

    **The plan does not contain a foot path.**  This generator's window is
    ``base_pose(9) + base_vel(6) + joint_pos(29)`` and carries no end-effector block, so a foot
    trajectory has to be COMPUTED from the plan rather than read out of it -- which is worth
    saying because the drawn feet are then a derived quantity, exact given the plan but not
    something the generator itself ever represented.

    MuJoCo's own kinematics does it, rather than a second FK implementation: the model is
    already loaded, it is the same one the episode was simulated in, and the joint permutation
    goes through ``deploy.tracker_session.convert_joint_order`` so the Isaac-to-MuJoCo order is
    stated in exactly one place in this repo.  ``mj_kinematics`` alone is enough for body
    positions -- a full ``mj_forward`` would also solve constraints nothing here reads.

    Args:
        model: The course's ``mj_model``.
        robot: The rig's ``Robot``, for its MuJoCo joint order.
        policy: The loaded pair, for the Isaac joint order.

    Raises:
        SystemExit: If a foot body is not in the model, which would otherwise draw nothing and
            read as a plan with no feet in it.
    """

    def __init__(self, model, robot, policy) -> None:
        from deploy.tracker_session import convert_joint_order  # noqa: PLC0415

        self.model = model
        self.data = mujoco.MjData(model)
        isaac = list(policy.get_joint_names())
        self.perm = np.asarray(
            convert_joint_order(np.arange(len(isaac)), isaac, list(robot.joint_names)),
            dtype=int)
        self.body_ids = []
        for name in FOOT_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise SystemExit(f"no body {name!r} in this model; --feet would silently draw "
                                 "nothing. Known feet: " + ", ".join(FOOT_BODIES))
            self.body_ids.append(bid)

    def feet(self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray,
             joint_isaac: np.ndarray) -> np.ndarray:
        """Foot world positions over a whole window.

        Args:
            root_pos: ``[F, 3]`` world base positions.
            root_quat_xyzw: ``[F, 4]`` world base orientations, xyzw (what the window lift
                returns; MuJoCo wants wxyz and the swap happens here).
            joint_isaac: ``[F, nj]`` joint angles in ISAAC order -- the window's ``joint_pos``
                block is absolute and already in that order.

        Returns:
            ``[n_feet, F, 3]`` world positions.
        """
        pos = np.asarray(root_pos, dtype=float)
        quat = np.asarray(root_quat_xyzw, dtype=float)[:, [3, 0, 1, 2]]
        joints = np.asarray(joint_isaac, dtype=float)[:, self.perm]
        out = np.empty((len(self.body_ids), len(pos), 3), dtype=float)
        for i in range(len(pos)):
            self.data.qpos[:3] = pos[i]
            self.data.qpos[3:7] = quat[i]
            self.data.qpos[7:] = joints[i]
            mujoco.mj_kinematics(self.model, self.data)
            for b, bid in enumerate(self.body_ids):
                out[b, i] = self.data.xpos[bid]
        return out

    def served(self, policy) -> np.ndarray | None:
        """The feet of the plan the policy is currently serving.

        Args:
            policy: The loaded pair, at a step where it has a live prediction.

        Returns:
            ``[n_feet, F, 3]``, or ``None`` when the plan has no pose or no joint block.
        """
        lifted = policy.predicted_base_pose_w()
        joints = policy.predicted_joint_pos()
        if lifted is None or joints is None:
            return None
        return self.feet(lifted[0], lifted[1], joints)


def alternative_plans(policy, k: int, rng, fk: PlanFk | None = None) -> list[dict]:
    """``k`` more plans the generator could have emitted at the re-plan it just made.

    **The conditioning is not rebuilt, it is REUSED.**  ``_run_fm`` stashes exactly what it
    asked the graph -- the history nodes, the exteroception tokens, the command tokens and the
    null-history flag -- so this re-runs ``sample_window`` on that same input with a different
    ``x0``.  Rebuilding the conditioning from the robot's state instead would be a second
    implementation of ``_run_fm``, and any drift in it would show up as candidates that differ
    from the served plan for a reason that is not the noise draw.

    **It cannot perturb the episode**, which is what lets this run inside the scored replay
    rather than beside it: ``sample_window`` is a pure call on the session, the draws come from
    a private RNG rather than the policy's, and ``_pred`` is rebound to a private copy of its
    own contents afterwards -- so even if a future session returned a view of a reused device
    buffer, the served plan would still be the one the tracker acted on.

    Args:
        policy: The loaded ``GenTrackerPolicy``, at a control step where it has just re-planned.
        k: How many alternatives to draw.
        rng: The private generator for their ``x0`` draws.
        fk: A :class:`PlanFk` to derive each candidate's foot paths with, or ``None`` to skip
            the kinematics entirely.

    Returns:
        Up to ``k`` ``{"root": [F, 3], "feet": [n_feet, F, 3] or None}``, in draw order.  Empty
        when there is no live plan or the window carries no ``base_pose`` block to lift.
    """
    from deploy.fm_session import build_history_tokens, sample_window  # noqa: PLC0415
    from deploy.window_np import base_pose_block_to_world, window_block_slices  # noqa: PLC0415

    plan = policy.last_plan()
    if plan is None:
        return []
    ctx = policy.ctx
    sl = window_block_slices(len(policy._fm_to_isaac), ctx.target_num_extra_frames,
                             policy.reference_blocks).get("base_pose")
    if sl is None:
        return []
    jsl = window_block_slices(len(policy._fm_to_isaac), ctx.target_num_extra_frames,
                              policy.reference_blocks).get("joint_pos")
    history = build_history_tokens(ctx, plan["history_nodes"])
    history_null = np.array([bool(plan["history_null"])])
    anchor_pos, anchor_quat = policy._anchor_pos, policy._anchor_quat
    out: list[dict] = []
    try:
        for _ in range(int(k)):
            noise = rng.standard_normal((1, ctx.F, ctx.D))
            pred = np.asarray(sample_window(ctx, history, plan["extero_tokens"],
                                            plan["cmd_tokens"], noise,
                                            history_null=history_null))
            pos, quat = base_pose_block_to_world(pred[0, :, sl], anchor_pos, anchor_quat)
            feet = None
            if fk is not None and jsl is not None:
                feet = fk.feet(pos, quat, pred[0, :, jsl])
            # The raw window travels with the drawn paths so a candidate can be SELECTED --
            # `choose` assigns it straight to `_pred`, and re-deriving it from the ribbon
            # would be impossible (the lift to world is not invertible without the anchor).
            out.append({"root": np.asarray(pos, dtype=float).copy(), "feet": feet,
                        "window": pred.copy()})
    finally:
        # Rebind rather than restore: `saved` is now this function's own array, so nothing the
        # session does to its buffers can reach the plan the tracker is serving.
        policy._pred = np.array(policy._pred, copy=True)
    return out


def _frame_recorder(frames: list, plans: list, replans: list, alts: list | None = None,
                    n_alt: int = 0, alt_rng=None, fk: PlanFk | None = None,
                    served_feet: list | None = None):
    """A recorder capturing the pose per step, and the plan and step at each re-plan.

    ``replans`` is what the branch view needs and the plain replay does not: it is the
    positional map from a NOISE DRAW to the control step that consumed it.  It is built from
    ``plan_id`` transitions rather than from the tape's own counter because the draw happens
    inside ``get_action`` while the recorder runs at the fold, so the tape index read here
    would be ahead by one on exactly the steps that matter.

    Args:
        frames: Appended ``(time, qpos_copy)`` per control step.
        plans: Appended ``(step, [F, 3] world root path)`` per re-plan that exposed one.
        replans: Appended ``step`` per re-plan, whether or not it exposed a path.
        alts: Appended ``(step, [candidate, ...])`` per re-plan when ``n_alt`` is set.
        n_alt: How many alternative plans to draw at each re-plan; 0 draws none and costs
            nothing.
        alt_rng: Private RNG for the alternatives, so they never touch the policy's stream.
        fk: A :class:`PlanFk` for the foot paths, or ``None`` to compute none.
        served_feet: Appended ``(step, [n_feet, F, 3])`` for the SERVED plan when ``fk`` is
            given.

    Returns:
        The recorder callable.
    """
    last_plan_id = [-1]

    def record(robot, pol) -> None:
        """Capture one control step's pose, and the plan when it has just been re-made."""
        frames.append((float(robot.mj_data.time), robot.mj_data.qpos.copy()))
        plan = pol.last_plan()
        if plan is not None and int(plan["plan_id"]) != last_plan_id[0]:
            last_plan_id[0] = int(plan["plan_id"])
            replans.append(len(frames) - 1)
            path = pol.predicted_root_path_w()
            if path is not None:
                plans.append((len(frames) - 1, np.asarray(path, dtype=float).copy()))
            if fk is not None and served_feet is not None:
                got = fk.served(pol)
                if got is not None:
                    served_feet.append((len(frames) - 1, got))
            if n_alt > 0 and alts is not None:
                alts.append((len(frames) - 1, alternative_plans(pol, n_alt, alt_rng, fk)))

    return record


def capture_branches(run_dir: str, label: str, terrain_idx: int, speed_idx: int, args,
                     cols: dict[str, np.ndarray], row_idx: int
                     ) -> tuple[dict, list, list, list] | None:
    """Re-run one cell, then re-run it ``K`` more times with ONE re-plan sampled differently.

    **Exactly one noise draw differs between a branch and the scored episode.**  The reference
    roll is taped (``NoiseTape`` wraps ``GenTrackerPolicy._sample_noise``, the single place the
    sampler's ``x0`` comes from), and each branch replays that whole tape with entry ``n``
    replaced by a fresh Gaussian draw.  So the branch is bit-identical to the scored episode up
    to re-plan ``n``, diverges there, and every LATER re-plan is served the reference's own draw
    again -- the divergence after ``n`` is the changed state feeding the generator, not more
    fresh noise.  Under the grid's ``zero_noise=True`` that also means the branches run the
    deployed rule (``x0 = 0``) everywhere except the one sampled re-plan, so a surviving branch
    says "a different draw HERE would have been enough", which is the question worth asking.

    A branch that outlives the reference makes more re-plans than the tape holds; those run off
    the end and are served fresh, which under ``zero_noise`` is the deployed zero and otherwise
    is a genuine second difference.  It is reported rather than hidden.

    **The self-check is not decoration.**  Replaying the tape UNCHANGED must reproduce the
    scored row exactly; if it does not, the tape is not the only stochastic input and every
    colour in the viewer is meaningless.  It costs one rollout and it is what licenses the
    whole picture.

    Args:
        run_dir: The eval run directory.
        label: Arm label.
        terrain_idx: Course index.
        speed_idx: Speed-ladder index.
        args: Parsed CLI arguments; reads ``samples``, ``replan`` and ``sample_seed``.
        cols: The arm's recorded columns, for the verification.
        row_idx: The recorded row's index.

    Returns:
        ``(summary, frames, plans, samples)``, or ``None`` when a verification failed and the
        caller should abort.  Each sample is ``{"k", "path", "fell", "verdict", "row",
        "branch_step", "ran_off_tape"}``.
    """
    from fm_noise_attribution import NoiseTape  # noqa: PLC0415 - numpy only, but keep it lazy

    cell = _build_cell(run_dir, label, terrain_idx, speed_idx, args)
    tape = NoiseTape(cell["policy"])
    try:
        frames: list[tuple[float, np.ndarray]] = []
        plans: list[tuple[int, np.ndarray]] = []
        replans: list[int] = []
        served_feet: list = []
        # --feet composes with this mode too: it draws the SCORED plan's feet under the
        # branch ribbons. Wired rather than refused, but wired rather than left to do nothing
        # silently, which is what an unread flag does.
        fk = (PlanFk(cell["rig"].robot.mj_model, cell["rig"].robot, cell["policy"])
              if getattr(args, "feet", False) else None)
        tape.reset(None)
        verdict, failure, row = _roll(
            cell, terrain_idx, speed_idx,
            _frame_recorder(frames, plans, replans, None, 0, None, fk, served_feet))
        ref_tape = [np.asarray(z).copy() for z in tape.recorded]
        if _refuse(verify_against_table(row, cols, row_idx), label):
            return None
        if not frames:
            raise SystemExit("captured no frames; the episode ended before the first fold")
        if len(ref_tape) != len(replans):
            raise SystemExit(
                f"the reference made {len(replans)} re-plan(s) but consumed {len(ref_tape)} "
                "noise draw(s). The branch index would name a different re-plan than the one "
                "it is drawn at; refusing rather than showing a mislabelled divergence.")

        n, why = _resolve_replan(args, replans, frames, len(ref_tape), cell["course"])
        branch_step = replans[n]
        t_branch, q_branch = frames[branch_step]
        print(f"\n  re-plans {len(ref_tape)}; branching at re-plan {n} "
              f"(control step {branch_step}, t={t_branch:.2f} s, "
              f"root x={float(q_branch[0]):+.2f} m) -- {why}", flush=True)

        # ---- self-check: the tape replayed unchanged must BE the scored episode ------------
        tape.reset(list(ref_tape))
        chk_verdict, _chk_failure, chk_row = _roll(cell, terrain_idx, speed_idx, lambda *_: None)
        drift = verify_against_table(chk_row, cols, row_idx)
        if drift:
            print("\n  [FATAL] replaying the reference's own noise tape did NOT reproduce the "
                  "scored row:", flush=True)
            for line in drift:
                print(f"    {line}")
            print("  The noise draw is therefore not the only free variable, so a branch would "
                  "differ from the reference in more than the one draw it claims to. "
                  "Refusing.", flush=True)
            return None
        # Say what the check actually established, which differs between the two noise
        # regimes. Under x0=0 the tape is all zeros, so replaying it is not a strong test of
        # the tape -- but the branches are then exactly-one-draw-different by CONSTRUCTION,
        # which is the stronger statement anyway. Claiming the Gaussian sentence in both
        # places would be claiming a control that was not run.
        if cell["zero_noise"]:
            print(f"  self-check: the hook is transparent -- the run reproduces the scored row "
                  f"({chk_verdict}). At x0=0 every taped draw is zero, so a branch differs from "
                  "the reference at exactly one draw by construction", flush=True)
        else:
            print(f"  self-check: replaying the tape unchanged reproduces the scored row "
                  f"({chk_verdict}) -- the draw is the only free variable", flush=True)

        # ---- the branches ------------------------------------------------------------------
        samples: list[dict] = []
        k_total = int(args.samples)
        print(f"  rolling {k_total} branch(es), each the scored episode with re-plan {n} "
              f"drawn differently", flush=True)
        for k in range(k_total):
            rng = np.random.default_rng(int(args.sample_seed) * 1_000_003 + k)
            inject = list(ref_tape)
            inject[n] = rng.standard_normal(ref_tape[n].shape)
            path: list[np.ndarray] = []
            tape.reset(inject)
            b_verdict, _b_failure, b_row = _roll(
                cell, terrain_idx, speed_idx,
                lambda robot, _pol: path.append(robot.mj_data.qpos[:3].copy()))
            samples.append({
                "k": k, "path": np.asarray(path, dtype=float), "verdict": b_verdict,
                "fell": b_verdict != "success", "row": b_row, "branch_step": branch_step,
                "ran_off_tape": tape.n_fresh > 0,
            })
            print(f"    branch {k}: {b_verdict:<8s} steps {int(b_row['steps_survived']):>4d} "
                  f"progress {float(b_row['progress_frac']):.3f}", flush=True)
    finally:
        tape.detach()

    summary = {"course": cell["course"], "speed": cell["speed"], "verdict": verdict,
               "failure": failure, "row": row, "model": cell["rig"].robot.mj_model,
               "branch_step": branch_step, "replan": n, "n_replans": len(ref_tape),
               "zero_noise": cell["zero_noise"], "served_feet": served_feet}
    _print_branch_table(summary, samples, row)
    return summary, frames, plans, samples


#: Fallback lead, in re-plans, when a course names no obstacle.  Only a fallback: leads
#: measured back from the END are the wrong quantity, and two runs showed why -- five re-plans
#: (~1.2 s) before a fall gave a 0.077 m endpoint spread on one course and 0.059 m on another.
#: By then the robot is already going over and no draw can help, so the view is a bundle of red
#: lines on top of each other, which says nothing about whether the re-plan was a decision.
_DEFAULT_REPLAN_LEAD: int = 5


def _print_plan_options(summary: dict, plans: list, frames: list) -> None:
    """Per re-plan, what the served plan and its alternatives each proposed.

    The two shape columns are what separate a staircase plan from a jump plan without having to
    squint at the ribbons: ``dz`` is the window's NET climb and ``zmax`` its highest point above
    the start, so a flight of stairs is ``dz`` positive with ``zmax`` tracking it, while a jump
    is a ``zmax`` well above a ``dz`` that comes back down.  ``spread`` is how far apart the
    candidates' endpoints are -- near zero means the generator had one idea here and the fan in
    the viewer is telling you nothing, which is itself the answer.

    Args:
        summary: :func:`capture`'s summary, carrying ``alt_plans``.
        plans: ``(step, [F, 3])`` the served plan per re-plan.
        frames: The captured frames, for the time and root position at each re-plan.
    """
    alts = summary.get("alt_plans") or []
    if not alts:
        return
    served = {step: path for step, path in plans}
    k = max(len(cand) for _step, cand in alts)
    names = ", ".join(f"[{i}] {_COLOUR_NAMES[i % len(_COLOUR_NAMES)]}" for i in range(k))
    print(f"\n  candidate colours: {names}; the SERVED plan stays violet")
    print(f"  {'replan':>7}{'t':>7}{'x':>7}{'served':>18}   candidates dz/zmax (m)"
          f"{'spread':>9}")
    for j, (step, cand) in enumerate(alts):
        t, qpos = frames[min(step, len(frames) - 1)]
        s = served.get(step)
        s_txt = (f"{_dz(s)[0]:+.3f}/{_dz(s)[1]:+.3f}" if s is not None else "--")
        cells = "  ".join(f"{_dz(p['root'])[0]:+.2f}/{_dz(p['root'])[1]:+.2f}" for p in cand)
        ends = [p["root"][-1] for p in cand] + ([s[-1]] if s is not None else [])
        spread = 0.0
        if len(ends) > 1:
            e = np.stack(ends)
            spread = float(np.linalg.norm(e[:, None] - e[None], axis=-1).max())
        print(f"  {j:>7d}{t:>7.2f}{float(qpos[0]):>7.2f}{s_txt:>18}   {cells}{spread:>9.3f}")


#: Human names for :data:`ALT_PLAN_RGBA`, so the console legend and the viewer agree.
_COLOUR_NAMES: tuple[str, ...] = ("blue", "green", "magenta", "amber", "cyan", "orange",
                                  "violet-blue", "grey")


def _dz(path: np.ndarray) -> tuple[float, float]:
    """A plan's net climb and its peak climb, both relative to its own first node.

    Args:
        path: ``[F, 3]`` world root path.

    Returns:
        ``(net, peak)`` in metres.
    """
    z = np.asarray(path, dtype=float)[:, 2]
    return float(z[-1] - z[0]), float(z.max() - z[0])


def _resolve_replan(args, replans: list[int], frames: list, n_draws: int, course
                    ) -> tuple[int, str]:
    """Which re-plan to branch at, defaulting to the approach to the course's obstacle.

    **The default is anchored to the TERRAIN, not to the ending**, because the decision these
    courses are built to probe is how the generator planned the feature -- and that plan is
    made on the approach, several seconds before a fall resolves it.  So the default is the
    last re-plan made before the root reached ``obstacle_x_m``: branch there and the K draws
    each get their own attempt at the stairs or the box, which is the comparison worth
    drawing.  Anchoring to the end instead measures how committed a falling robot already is.

    Args:
        args: Parsed CLI arguments; reads ``replan``.
        replans: Control step of each re-plan, in order.
        frames: The captured ``(time, qpos)`` frames, for the root position at each re-plan.
        n_draws: How many noise draws the reference consumed.
        course: The course, for ``obstacle_x_m``.

    Returns:
        ``(re-plan index, why)``, the second being a short phrase for the console.

    Raises:
        SystemExit: On an out-of-range ``--replan``.
    """
    if args.replan is not None:
        if not 0 <= args.replan < n_draws:
            raise SystemExit(f"--replan {args.replan} outside 0..{n_draws - 1}; this episode "
                             f"made {n_draws} re-plan(s)")
        return int(args.replan), "asked for"
    obstacle_x = getattr(course, "obstacle_x_m", None)
    if obstacle_x is not None:
        before = [i for i, step in enumerate(replans)
                  if float(frames[step][1][0]) < float(obstacle_x)]
        if before:
            return before[-1], f"last re-plan before the obstacle at x={float(obstacle_x):.2f} m"
    # Either the course names no obstacle or the robot never got near it; fall back to the end.
    return max(0, n_draws - _DEFAULT_REPLAN_LEAD), f"{_DEFAULT_REPLAN_LEAD} re-plans from the end"


def _print_branch_table(summary: dict, samples: list[dict], ref_row: dict) -> None:
    """Print the reference and every branch, and how far apart the branches ended up.

    The spread is the headline: a near-zero one means this re-plan was not a decision at all
    and the colours are telling you nothing, which is a result and not a failure of the tool.

    Args:
        summary: The reference summary.
        samples: The branch dicts.
        ref_row: The reference's row.
    """
    print(f"\n  {'#':<5}{'verdict':<10}{'steps':>7}{'progress':>10}{'end x':>9}  colour")
    print(f"  {'ref':<5}{summary['verdict']:<10}{int(ref_row['steps_survived']):>7d}"
          f"{float(ref_row['progress_frac']):>10.3f}{float(ref_row['progress_x_m']):>9.2f}"
          "  (the scored episode)")
    for s in samples:
        colour = "red (fell)" if s["fell"] else "teal (survived)"
        print(f"  {s['k']:<5d}{s['verdict']:<10}{int(s['row']['steps_survived']):>7d}"
              f"{float(s['row']['progress_frac']):>10.3f}"
              f"{float(s['row']['progress_x_m']):>9.2f}  {colour}")
    ends = np.stack([s["path"][-1] for s in samples if len(s["path"])]) if samples else None
    if ends is not None and len(ends) > 1:
        spread = float(np.linalg.norm(ends[:, None] - ends[None], axis=-1).max())
        print(f"  max endpoint spread among branches: {spread:.3f} m")
    n_survived = sum(1 for s in samples if not s["fell"])
    print(f"  {n_survived}/{len(samples)} branch(es) survived; the scored episode's verdict "
          f"was {summary['verdict']!r}")
    print(f"  step to another decision with --replan 0..{summary['n_replans'] - 1} "
          f"(this was {summary['replan']})")
    off = sum(1 for s in samples if s["ran_off_tape"])
    if off and not summary["zero_noise"]:
        # Only a caveat under Gaussian sampling. At x0 = 0 the draw served past the end of the
        # tape is the same zero the tape holds, so an overrun changes nothing and saying it
        # does would invent a confound.
        print(f"  note: {off} branch(es) outlived the reference and ran off the end of its "
              "tape; those extra re-plans were served FRESH draws, so they differ from the "
              "reference in more than the one branch draw")


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
           ghost: dict | None = None, samples: list | None = None,
           replan_hold_s: float = 0.0) -> None:
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
        replan_hold_s: Wall seconds to hold on each re-plan frame.  A re-plan interval is 0.24 s
            and the candidate fan is redrawn at each one, so at any watchable playback rate it
            is gone before it can be read; this is the stepping control, without the keyboard
            plumbing a passive viewer cannot have.
        samples: :func:`capture_branches`' branch list, or ``None``.  Each is drawn as a root
            PATH rather than as a body: the viewer renders one ``mjData``, and K translucent
            humanoids at the same place would occlude the terrain that explains the divergence.
            A path appears only from its branch step, because before that every branch IS the
            reference and drawing them over it would suggest a spread that does not exist yet.
    """
    import mujoco.viewer  # noqa: PLC0415 - opens a GL context; --list must not

    course, model = summary["course"], summary["model"]
    data = mujoco.MjData(model)
    plan_at = {step: path for step, path in plans}
    alt_at = {step: cand for step, cand in (summary.get("alt_plans") or ())}
    feet_at = {step: feet for step, feet in (summary.get("served_feet") or ())}
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
    # A branch that outlived the reference keeps moving after the robot has stopped, so the
    # playback runs to the longest track rather than to the scored one.
    if samples:
        n_steps = max([n_steps] + [len(s["path"]) for s in samples])

    print(f"\n  replaying {len(frames)} steps ({frames[-1][0]:.2f} s) at {rt:g}x real time. "
          "Close the window to exit.\n", flush=True)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 130.0
        # `_dots` truncates silently at `maxgeom`, and a branch cut off halfway reads exactly
        # like a branch that ended halfway -- the one confusion this view must not create. So
        # the stride is DERIVED from the budget actually available rather than guessed.
        sample_stride = 1
        if samples:
            budget = max(120, viewer.user_scn.maxgeom - _OVERLAY_GEOM_RESERVE)
            drawn = sum(max(0, len(s["path"]) - s["branch_step"]) for s in samples)
            sample_stride = max(1, int(math.ceil(drawn / budget)))
            print(f"  drawing {len(samples)} branch path(s), every {sample_stride} step(s) "
                  f"({drawn} points into a {budget}-geom budget)", flush=True)
        current_plan = None
        current_ghost_plan = None
        current_alts: list = []
        current_feet = None
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
                if i in alt_at:
                    current_alts = alt_at[i]
                if i in feet_at:
                    current_feet = feet_at[i]
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
                if samples:
                    # The divergence instant, on the reference's own path -- without it the
                    # ribbons look like they were always apart.
                    b_step = samples[0]["branch_step"]
                    if i >= b_step:
                        _dots(viewer.user_scn,
                              frames[min(b_step, len(frames) - 1)][1][:3].reshape(1, 3),
                              BRANCH_RGBA, 0.07)
                    for s in samples:
                        path = s["path"]
                        if len(path) <= s["branch_step"] or i < s["branch_step"]:
                            continue
                        rgba = SAMPLE_FELL_RGBA if s["fell"] else SAMPLE_SURVIVED_RGBA
                        # Only as far as the playhead: the branches then GROW out of the
                        # branch point as the episode runs, which is the divergence itself.
                        head = min(i, len(path) - 1)
                        _dots(viewer.user_scn, path[s["branch_step"]:head + 1], rgba, 0.020,
                              stride=sample_stride)
                        _dots(viewer.user_scn, path[head].reshape(1, 3),
                              (rgba[0], rgba[1], rgba[2], 1.0), 0.055)
                # The alternatives go down BEFORE the served plan, so where two coincide the
                # violet one is on top -- the question is always "what did it pick out of
                # these", and the pick must never be the ribbon that got covered.
                for c_i, cand in enumerate(current_alts):
                    rgba = ALT_PLAN_RGBA[c_i % len(ALT_PLAN_RGBA)]
                    _dots(viewer.user_scn, cand["root"], rgba, 0.018)
                    # Feet smaller than the root ribbon they belong to: a plan reads as one
                    # object with a thin pair of tracks, not as three ribbons of equal weight.
                    if cand.get("feet") is not None:
                        for foot in cand["feet"]:
                            _dots(viewer.user_scn, foot, rgba, 0.011)
                if current_plan is not None:
                    _dots(viewer.user_scn, current_plan, PLAN_RGBA, 0.022, stride=2)
                if current_feet is not None:
                    for foot in current_feet:
                        _dots(viewer.user_scn, foot, PLAN_RGBA, 0.013)
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

                if replan_hold_s > 0.0 and i in alt_at:
                    t_hold = time.time()
                    while (time.time() - t_hold) < replan_hold_s and viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    # The playback clock is wall-referenced, so without this the steps after a
                    # hold would sprint to catch up and the hold would cost them instead of
                    # costing wall time.
                    wall0 += time.time() - t_hold

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
            "--fm_onnx", arm.fm_onnx,
            "--fall_height", f"{args.fall_height}",
            "--fall_tilt_deg", f"{args.fall_tilt_deg}",
            "--allow_robot_asset_mismatch", "--viz"]
    print("\n  !! --live drives PHYSICS in the viewer, which perturbs mjData. The trajectory")
    print("     will NOT match the table row: one course here survived 16 s headless and fell")
    print("     at 6.04 s headed, reproducibly. Use the default replay to see the scored run.\n")
    print(f"  $ {' '.join(argv)}\n")
    subprocess.run(argv, cwd=os.path.dirname(_HERE), check=False)


def _draw_choice(scn, course, target_z: float, served, served_feet, alts: list) -> None:
    """Draw the goal, the served plan and the candidate fan into a scene.

    Args:
        scn: The viewer's ``user_scn``, cleared by the caller.
        course: The course, for its goal rectangle.
        target_z: Height to draw the goal at.
        served: ``[F, 3]`` served root path, or ``None``.
        served_feet: ``[n_feet, F, 3]`` served foot paths, or ``None``.
        alts: Candidate dicts from :func:`alternative_plans`.
    """
    _goal_rect(scn, course, target_z)
    for c_i, cand in enumerate(alts):
        rgba = ALT_PLAN_RGBA[c_i % len(ALT_PLAN_RGBA)]
        _dots(scn, cand["root"], rgba, 0.018)
        if cand.get("feet") is not None:
            for foot in cand["feet"]:
                _dots(scn, foot, rgba, 0.011)
    if served is not None:
        _dots(scn, served, PLAN_RGBA, 0.022, stride=2)
    if served_feet is not None:
        for foot in served_feet:
            _dots(scn, foot, PLAN_RGBA, 0.013)


class QuitChoosing(Exception):
    """Raised to end a ``--choose`` session from the prompt.

    Deliberately NOT ``KeyboardInterrupt``: ``runner.run_episode`` catches that and converts
    it into a quiet ``verdict = "interrupted"``, so a ``q`` raised as one would be swallowed by
    the episode loop and the next re-plan would simply prompt again -- which is
    indistinguishable, at the terminal, from the key not working.
    """


def _prompt_choice(replan_idx: int, served, alts: list, auto: int | None) -> int:
    """Print the options and read the pick.  ``0`` is the plan the policy chose.

    Args:
        replan_idx: Which re-plan this is, for the heading.
        served: The served root path.
        alts: Candidate dicts.
        auto: Take this index every time instead of asking; ``None`` asks.

    Returns:
        ``0`` for the served plan, ``1..K`` for an alternative.

    Raises:
        QuitChoosing: On ``q``, or on EOF (the terminal went away mid-session).
    """
    rows = [("served", "violet", served)] + [
        (f"alt {i + 1}", _COLOUR_NAMES[i % len(_COLOUR_NAMES)], c["root"])
        for i, c in enumerate(alts)]
    ends = np.stack([r[2][-1] for r in rows])
    spread = float(np.linalg.norm(ends[:, None] - ends[None], axis=-1).max())
    print(f"\n=== re-plan {replan_idx} ===")
    print(f"  {'#':<4}{'plan':<9}{'colour':<9}{'dz':>8}{'zmax':>8}{'fwd':>8}{'lat':>8}")
    for i, (name, colour, path) in enumerate(rows):
        dz, zmax = _dz(path)
        fwd = float(path[-1][0] - path[0][0])
        lat = float(path[-1][1] - path[0][1])
        print(f"  {i:<4}{name:<9}{colour:<9}{dz:>8.3f}{zmax:>8.3f}{fwd:>8.3f}{lat:>8.3f}")
    print(f"  max endpoint spread: {spread:.3f} m"
          + ("   (the options are the same plan -- this re-plan is not a decision)"
             if spread < 0.05 else ""))
    if auto is not None:
        return min(int(auto), len(alts))
    while True:
        try:
            raw = input(f"  pick [0-{len(alts)}], Enter=0 (keep served), "
                        "q=quit (or ctrl-c twice): ").strip()
        except EOFError:
            # The terminal went away mid-session (window closed, shell exited). Ending is the
            # only correct move: looping would block on a stdin that can never answer.
            raise QuitChoosing("stdin closed") from None
        if raw.lower() in ("q", "quit"):
            raise QuitChoosing("user ended the plan-choosing session")
        if raw == "":
            return 0
        if raw.isdigit() and int(raw) <= len(alts):
            return int(raw)
        print("  not a listed option")


def choose(run_dir: str, label: str, terrain_idx: int, speed_idx: int, args) -> int:
    """Drive the cell LIVE, pausing at every re-plan to pick which plan to follow.

    **This is not the scored episode and cannot be.**  Choosing a plan changes what the robot
    does next, so physics has to run in the loop -- and this file's whole premise elsewhere is
    that a replay is safe precisely because it does not.  Attaching the viewer perturbs
    ``mjData`` on its own (one course measured here survived 16 s headless and fell
    reproducibly at 6.04 s headed), so even picking ``0`` every time will not reproduce the
    table row.  It is said before every run rather than left to be inferred from a number that
    disagrees.

    **How a pick takes effect.**  ``_run_fm`` is wrapped: the original runs first, so the
    conditioning is built and stashed exactly as the deployed loop builds it, then the
    candidates are sampled from that same conditioning and the chosen window is assigned to
    ``policy._pred``.  ``observe`` reads ``_pred`` immediately after ``_run_fm`` returns and
    stashes the served node into the generator's own history, so a chosen plan becomes the
    history the next re-plan conditions on -- the choice compounds, exactly as the generator's
    own output does.

    Args:
        run_dir: The eval run directory.
        label: Arm label.
        terrain_idx: Course index.
        speed_idx: Speed-ladder index.
        args: Parsed CLI; reads ``choose``, ``choose_every``, ``choose_auto``, ``plan_seed``,
            ``feet`` and ``rt``.

    Returns:
        Process exit code.
    """
    import runner  # noqa: PLC0415

    cell = _build_cell(run_dir, label, terrain_idx, speed_idx, args)
    policy, robot = cell["policy"], cell["rig"].robot
    course, model = cell["course"], robot.mj_model
    fk = PlanFk(model, robot, policy) if args.feet else None
    rng = np.random.default_rng(int(args.plan_seed))
    k, every = int(args.choose), max(1, int(args.choose_every))
    auto = args.choose_auto
    target_z = course.floor_z if course.flight_height_m < 0 else course.flight_height_m
    dt = float(policy.get_dt())
    # Without a terminal `input()` raises EOFError deep inside the physics loop, which reads
    # as a crash in the runtime rather than as "this mode needs somewhere to type". Checked
    # up front, and named with the flag that makes it scriptable.
    if auto is None and not sys.stdin.isatty():
        raise SystemExit(
            "--choose asks which plan to take, so it needs a terminal to answer on, and this "
            "process has no tty (piped, backgrounded, or launched by a tool). Run it in a "
            "shell, or pass --choose_auto I to take option I every time without asking.")

    print("\n  !! --choose runs PHYSICS in the loop and lets you change the plan, so this is")
    print("     NOT the scored episode and will not reproduce the table row. Use the default")
    print("     replay for that. Picks compound: a chosen plan becomes the history the next")
    print("     re-plan conditions on.\n", flush=True)

    live = {"served": None, "feet": None, "alts": [], "n": 0, "picks": []}
    viewer = None
    orig_run_fm = policy._run_fm

    def hooked_run_fm(state, history_null: bool = False) -> None:
        """Sample the plan as usual, then offer the alternatives and install the pick."""
        orig_run_fm(state, history_null=history_null)
        idx = live["n"]
        live["n"] += 1
        live["served"] = policy.predicted_root_path_w()
        live["feet"] = fk.served(policy) if fk is not None else None
        if idx % every:
            live["alts"] = []
            return
        live["alts"] = alternative_plans(policy, k, rng, fk)
        if not live["alts"]:
            return
        if viewer is not None:
            viewer.user_scn.ngeom = 0
            _draw_choice(viewer.user_scn, course, target_z, live["served"], live["feet"],
                         live["alts"])
            # Sync BEFORE blocking, or the fan being chosen between is not on screen while
            # the terminal waits for the answer.
            viewer.sync()
        pick = _prompt_choice(idx, live["served"], live["alts"], auto)
        live["picks"].append(pick)
        if pick > 0:
            chosen = live["alts"][pick - 1]
            policy._pred = np.asarray(chosen["window"], dtype=np.float64)
            live["served"], live["feet"] = chosen["root"], chosen["feet"]

    policy._run_fm = hooked_run_fm
    wall = [time.time()]

    def record(_robot, _pol) -> None:
        """Draw and pace one control step."""
        if viewer is None:
            return
        if not viewer.is_running():
            raise QuitChoosing("viewer window closed")
        viewer.user_scn.ngeom = 0
        _draw_choice(viewer.user_scn, course, target_z, live["served"], live["feet"],
                     live["alts"])
        _dots(viewer.user_scn, robot.mj_data.qpos[:3].reshape(1, 3), ROOT_RGBA, 0.05)
        viewer.cam.lookat[:] = robot.mj_data.qpos[:3]
        viewer.sync()
        if args.rt > 0:
            ahead = dt / args.rt - (time.time() - wall[0])
            if ahead > 0:
                time.sleep(ahead)
        wall[0] = time.time()

    ctx = None
    interrupts = [0]
    prev_sigint = signal.getsignal(signal.SIGINT)

    def on_sigint(_sig, _frame):
        """First ctrl-c unwinds politely; a second one leaves, whatever is stuck.

        The polite path is not guaranteed to work here and that is the point.  A ctrl-c
        raised while the prompt is blocked unwinds into ``run_episode``, which catches
        ``KeyboardInterrupt`` and turns it into ``verdict = "interrupted"`` -- so the episode
        ends but the process continues into the viewer teardown, and a passive viewer whose
        window has already died can block there.  The observed symptom is a session that
        ignores ctrl-c entirely, so the second press bypasses every handler and exits.
        """
        interrupts[0] += 1
        if interrupts[0] >= 2:
            print("\n  second ctrl-c -- exiting immediately.", flush=True)
            os._exit(130)
        print("\n  ctrl-c: ending the session (press again to exit immediately).", flush=True)
        raise QuitChoosing("interrupted")

    try:
        signal.signal(signal.SIGINT, on_sigint)
        if auto is None:
            import mujoco.viewer  # noqa: PLC0415
            ctx = mujoco.viewer.launch_passive(model, robot.mj_data, show_left_ui=False,
                                               show_right_ui=False)
            viewer = ctx.__enter__()
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 6.0, -18.0, 130.0
        verdict, failure, _acc = "interrupted", "", None
        try:
            verdict, failure, _acc = _roll(cell, terrain_idx, speed_idx, record)
        except (QuitChoosing, KeyboardInterrupt) as exc:
            print(f"\n  ended early: {exc}", flush=True)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        policy._run_fm = orig_run_fm
        if ctx is not None:
            ctx.__exit__(None, None, None)

    n_changed = sum(1 for p in live["picks"] if p > 0)
    print(f"\n  verdict {verdict}" + (f" ({failure})" if failure else ""))
    print(f"  {len(live['picks'])} decision(s) offered, {n_changed} changed from the plan the "
          "policy chose")
    print("  Reminder: physics ran in the viewer, so this episode is not the scored one.",
          flush=True)
    return 0


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
    p.add_argument("--plans", type=int, default=0, metavar="K",
                   help="at EVERY re-plan, draw K other plans the generator could have emitted "
                        "from the same conditioning, each in its own colour, with the served "
                        "one in violet. Shows what options it was choosing between -- a "
                        "staircase plan against a jump plan. Costs K generator calls per "
                        "re-plan and no extra rollouts, so the scored episode is untouched")
    p.add_argument("--plan_seed", type=int, default=0,
                   help="seeds the candidate plans' draws, so a session is reproducible")
    p.add_argument("--feet", action="store_true",
                   help="also draw each plan's FOOT paths, in the plan's own colour. The "
                        "window carries joint angles and no end-effector block, so these are "
                        "derived from it by MuJoCo kinematics -- exact given the plan, but a "
                        "quantity the generator never itself represented. Works with or "
                        "without --plans; a step onto a tread and a hop onto a box differ far "
                        "more in the feet than in the root")
    p.add_argument("--replan_hold_s", type=float, default=0.0, metavar="SEC",
                   help="hold each re-plan frame this long, so the candidate fan can be read; "
                        "a re-plan interval is 0.24 s and is otherwise gone before you see it")
    p.add_argument("--choose", type=int, default=0, metavar="K",
                   help="INTERACTIVE: drive the cell live and pause at each re-plan to pick "
                        "among the served plan and K sampled alternatives. The pick is "
                        "installed and the robot follows it, so picks compound. Physics runs "
                        "in the viewer, so this is NOT the scored episode")
    p.add_argument("--choose_every", type=int, default=1, metavar="N",
                   help="with --choose, only offer a choice every Nth re-plan; the rest keep "
                        "the policy's own plan. 45 prompts in an episode is a lot")
    p.add_argument("--choose_auto", type=int, default=None, metavar="I",
                   help="with --choose, never ask -- take option I every time (0 = the served "
                        "plan). Runs headless, which is how the mode is smoke-tested")
    p.add_argument("--branch", type=int, default=0, metavar="K",
                   help="a different question: ROLL K episodes identical to the scored one "
                        "except for ONE re-plan's draw, and draw where each ended up (teal "
                        "survived, red fell). Answers whether a draw here changes the outcome, "
                        "where --plans answers what the options looked like. Costs K+2 rollouts")
    p.add_argument("--replan", type=int, default=None, metavar="N",
                   help="with --branch, which re-plan the branches diverge at; the episode's "
                        "re-plan count is printed so the next one is a keystroke away. "
                        "Default: the last re-plan before the root reached the course's "
                        "obstacle -- the plan for the feature, which is the decision these "
                        "courses are built to probe")
    p.add_argument("--sample_seed", type=int, default=0,
                   help="seeds the branches' fresh draws, so a session is reproducible")
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

    # Every extra thing this can draw is a set of anonymous extra paths, and two sets at once
    # are unreadable and, worse, unlabelled -- so exactly one may be on at a time.
    extra = [n for n, v in (("--plans", args.plans), ("--branch", args.branch),
                            ("--choose", args.choose), ("--compare", args.compare)) if v]
    if len(extra) > 1:
        raise SystemExit(f"{' and '.join(extra)} cannot be combined: each draws its own extra "
                         "paths into one viewer, varying a different thing (the noise draw at "
                         "one instant, the outcome of a draw, your pick, the tracker). Run "
                         "them separately.")

    if args.choose > 0:
        return choose(run_dir, args.arm, terrain_idx, speed_idx, args)

    if args.branch > 0:
        args.samples = args.branch          # capture_branches' own name for the count
        got = capture_branches(run_dir, args.arm, terrain_idx, speed_idx, args, cols, idx)
        if got is None:
            return 1
        summary, frames, plans, samples = got
        replay(summary, frames, plans, args.rt, args.loop, samples=samples,
               replan_hold_s=args.replan_hold_s)
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

    _print_plan_options(summary, plans, frames)
    replay(summary, frames, plans, args.rt, args.loop, ghost=ghost,
           replan_hold_s=args.replan_hold_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
