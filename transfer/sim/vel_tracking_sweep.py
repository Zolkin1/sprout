#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Velocity tracking across a terrain feature: the paper's tracking figure, measured.

One long straight lane per terrain class, a step velocity command, and N repeats that differ
in NOTHING but the generator's noise draw.  The claim it exists to support has two halves:

* on FLAT ground the pair tracks the commanded velocity closely, and
* on a BOX / STAIRS UP / STAIRS DOWN it still tracks the commanded velocity while the root
  height does what the terrain requires.

So every trial records the achieved root velocity, the COMMANDED root velocity and the root
position in 3-D, per control step, and ``plot_vel_tracking.py`` turns one output directory
into the figure.  Verdict and the position at which an episode ended travel with it, so a
failure is drawn where it happened rather than dropped.

**The lane is deliberately long.**  ``--approach`` 10 m of flat before the feature and
``--runout`` 10 m of flat past it, against ``mjeval``'s 2.5 m: the point here is a settled,
steady-state velocity BEFORE the terrain and another one after, so that the excursion in
between is attributable to the terrain rather than to the opening transient.  That also means
these are NOT mjeval courses and their numbers are not comparable to a catalogue run -- the
approach alone differs by 7.5 m, and `fm_variance_probe` measured start position as worth
+-0.04 of success over a quarter metre.

**No stairwell walls.**  ``mjeval``'s stair courses ring the flight with physical walls, which
would mechanically hold a drifting robot on the treads -- exactly the thing the steering
controller is here to do and the thing the figure is claiming. A wall would make the lateral
half of the result an artifact of the scene.

**Steering is ON by default and is OFF-DISTRIBUTION ON STAIRS.**  Measured over the training
library, the two stair families (3500 clips) carry ``|wz| max = 0.0000``: the generator has
never once been commanded a turn on a staircase.  It is on anyway because without it a few
degrees of accumulated heading error over 20 m of lane becomes a lateral miss, and the figure
would be reporting drift rather than velocity tracking (on one real stairs-up tile
``--steer_kp 1.5`` is worth ~+0.27 of success).  The consequence for the figure is that the
commanded ``wz`` is NOT zero on the terrain panels, which is why it is plotted.

**Every repeat differs only in the flow-matching noise draw** -- no domain randomisation, no
depth noise, no clip, the same standing spawn, the same command, and MuJoCo is deterministic.
``--zero_noise_control`` is the blocking check: it integrates the flow from ``x0 = 0`` every
trial, so every column of the trace must be bit-identical across repeats.  Note that
``x0 = 0`` is also what the hardware controller runs and is NOT a neutral sample of the
conditional -- it sits near its mode -- so it is a control here and not an arm.

**DO NOT read ``episodes.h5``'s ``vel_mse_vx_pre`` / ``vel_mse_vx_post`` as this lane's
approach-vs-feature tracking split.** Those are ``mjeval.accumulate``'s columns and they split
at ``obstacle_x_m``, which on mjeval's 2.5 m approach really does separate the run-up from the
obstacle. Here the approach is 10 m by design, so ``pre`` spans the ENTIRE opening transient
plus most of the steady walk and ``post`` spans the feature plus the whole runout -- and the
transient dominates: measured on the shipped grid, ``pre`` reads 0.873 m/s RMSE on FLAT ground
at 2.2 m/s, which looks like a tracking failure and is the acceleration from a standing start.
The columns are correct and are left alone, because changing them would move every mjeval
number; they are simply answering a different question than their names suggest at this
approach. The number this harness is for comes from windowing ``traces.npz`` on downrange x --
settled (past the ramp-in, before the feature), at the feature, and runout -- and the FLAT
column is the control that makes it readable: flat shows no feature excursion at all
(0.114 against 0.118 settled), so an excursion on terrain is the terrain rather than the
choice of window.

Run from ``transfer/sim`` (no GPU needed for MuJoCo, but the generator wants one;
``env_transfer`` is enough -- no isaaclab)::

    TRK=../../logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported
    python vel_tracking_sweep.py \\
        --tracker_onnx $TRK/tracker.onnx \\
        --fm_onnx $TRK/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx \\
        --speed 0.8 --speed 1.5 --trials 50 \\
        --out ../../logs/vel_tracking/v1
    python plot_vel_tracking.py ../../logs/vel_tracking/v1

The generator named there is deliberate: that ``exported/`` directory holds a dozen of them,
including the DEPLOYED ``..._lag48_aug_fm.onnx``, which is not the one to run -- see "WHICH
ARTIFACTS TO USE" in the root ``CLAUDE.md``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _d in (_HERE, os.path.join(_HERE, "mjeval"), os.path.dirname(_HERE),
           os.path.dirname(os.path.dirname(_HERE))):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import accumulate as mjacc  # noqa: E402  - transfer/sim/mjeval/accumulate.py
import generate_stairs_scene as gss  # noqa: E402
import runner as mjrunner  # noqa: E402
import schema as mjschema  # noqa: E402
import terrains  # noqa: E402

sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))
import provenance  # noqa: E402  - the repo's own git-snapshot helper

#: The terrain classes this harness builds.  ``flat`` is the control arm of the figure.
TERRAIN_KINDS: tuple[str, ...] = ("flat", "box", "stairs_up", "stairs_down")

#: Integer codes for the trace's ``ep_terrain`` column, so the ``.npz`` needs no string dtype.
TERRAIN_CODES: dict[str, int] = {k: i for i, k in enumerate(TERRAIN_KINDS)}

#: How far BEHIND the spawn a descending course's raised approach slab extends (m).  The robot
#: spawns at x = 0 on the slab, so this only has to cover the backwards half of the spawn
#: jitter and the pelvis footprint; it is ``mjeval``'s own value.
SLAB_BEHIND_M: float = 4.0

#: Seconds of settling added to every episode's time budget on top of the distance term.
BUDGET_SETTLE_S: float = 4.0


@dataclass(frozen=True)
class Geometry:
    """The terrain numbers, one set shared by every course in a run.

    Every default is a MEASURED training mode from :mod:`generate_stairs_scene` -- the modal
    riser and tread, the trained box depth and the median jump-on width -- so the figure's
    headline is the terrain the policy was trained on and the one the shipped
    ``*_terrain_scene.xml`` puts under a joystick.

    **``box_depth`` defaults to the TRAINED 0.46 m, not to mjeval's 1.20 m.**  That 1.20 is the
    one deliberate out-of-distribution move in the mjeval catalogue and it is there for a
    reason -- a trained box is under one stride deep, so the robot never stands on top and "up
    then down" is a single ballistic event rather than the two a mode-selection probe wants.
    It is the wrong default HERE: it made the box column of the first sweep a stress case
    wearing the label "box", scoring 0.64/0.44/0.36 against a policy that handles the trained
    box far better.  Pass ``--box_depth 1.2`` for the mjeval geometry deliberately; the value
    used is recorded in the manifest either way.

    ``box_height`` 0.40 m is on the trained grid (:data:`gss.TRAINED_BOX_HEIGHTS`) and is the
    height at which the single-box mjeval course still succeeds while the two-box one does not.

    Attributes:
        box_height: Box top height (m).
        box_depth: Box extent along travel (m).
        box_width: Box lateral extent (m).
        stair_rise: Step height (m).
        stair_run: Tread depth (m).
        stair_steps: Number of risers, up and down alike.
        stair_width: Lateral flight width (m).
    """

    box_height: float = 0.40
    box_depth: float = gss.TRAINED_BOX_DEPTH
    box_width: float = gss.TRAINED_BOX_WIDTH
    stair_rise: float = gss.DEFAULT_RISE
    stair_run: float = gss.DEFAULT_RUN
    stair_steps: int = 5
    stair_width: float = 2.44


@dataclass(frozen=True)
class Terrain:
    """One built course plus the surface polyline the plotter draws it from.

    The profile is emitted by the SAME function that emits the geoms, from the same numbers,
    because a figure whose drawn staircase disagrees with the simulated one is worse than no
    figure: every velocity excursion would be attributed to the wrong step.

    Attributes:
        kind: One of :data:`TERRAIN_KINDS`.
        course: The mjeval :class:`terrains.Course`, with this run's approach and runout.
        profile_x: Downrange samples of the walkable top surface (m), ascending.
        profile_z: Surface height at each ``profile_x`` (m).  Vertical faces appear as two
            samples sharing an x, so the polyline can be drawn with ``plot`` or filled.
        x_enter: World x of the near face of the feature (m) -- the first riser or box face.
            On ``flat`` this is the nominal marker at ``--approach``, where a feature would
            have been, so the panels stay comparable.
        x_exit: World x past which the terrain is flat again (m).
        goal_x: World x the episode ends at: ``x_exit + runout``, the goal band's NEAR edge.
    """

    kind: str
    course: terrains.Course
    profile_x: tuple[float, ...]
    profile_z: tuple[float, ...]
    x_enter: float
    x_exit: float
    goal_x: float


def _flat_lead(approach: float) -> tuple[list[float], list[float]]:
    """The flat run-up shared by every course: 2 m behind the spawn out to the feature."""
    return [-2.0, approach], [0.0, 0.0]


def build_terrain(kind: str, *, approach: float, runout: float, geom: Geometry,
                  speeds: tuple[float, ...], tag: str = "") -> Terrain:
    """Build one course and its surface profile.

    The goal band's NEAR edge is placed at ``x_exit + runout``, so an episode that stops on
    arrival has travelled exactly ``runout`` metres of flat ground past the feature -- which
    is what ``--runout`` is asking for.  ``terrains.GOAL_DEPTH / 2`` is added to the target
    CENTRE to achieve that, because :meth:`terrains.Course.in_goal` is a band about the
    centre rather than a half-plane.

    Args:
        kind: One of :data:`TERRAIN_KINDS`.
        approach: Flat metres before the feature's near face.
        runout: Flat metres past the feature before the episode ends.
        geom: The terrain numbers.
        speeds: The commanded speeds this course will be run at; recorded on the course.
        tag: Suffix for the scene name, to keep two concurrent processes off one file.

    Returns:
        The assembled :class:`Terrain`.

    Raises:
        ValueError: On an unknown ``kind``.
    """
    px, pz = _flat_lead(approach)
    common = dict(variant=0, speeds=tuple(speeds), goal_half_width_m=terrains.GOAL_HALF_WIDTH_FLOOR)

    if kind == "flat":
        x_enter = x_exit = float(approach)
        goal_x = x_exit + runout
        px += [goal_x + 5.0]
        pz += [0.0]
        course = terrains.Course(
            name=f"veltrack_flat{tag}", klass="box", geoms=(), floor_z=0.0,
            target_xy=(goal_x + terrains.GOAL_DEPTH / 2.0, 0.0),
            course_length_m=goal_x, flight_height_m=0.0, obstacle_x_m=x_enter,
            note=(f"FLAT control lane, {approach:.1f} m + {runout:.1f} m; the marker at "
                  f"x={x_enter:.1f} m is where a feature would be"),
            params={"approach_m": approach, "runout_m": runout}, **common)
        # klass is a label only -- there is no flat class in CLASS_CODES and inventing one
        # would change every consumer's terrain_class encoding.  `params["kind"]` is the
        # authority for this harness and is what the trace records.

    elif kind == "box":
        x_enter, x_exit = float(approach), float(approach + geom.box_depth)
        goal_x = x_exit + runout
        geoms = (f'        <!-- box {geom.box_depth}x{geom.box_width}x{geom.box_height} m '
                 f'at x={x_enter:.2f} -->',
                 terrains._slab("box0", x_enter, x_exit, 0.0, geom.box_width, 0.0,
                                geom.box_height))
        px += [x_enter, x_exit, x_exit, goal_x + 5.0]
        pz += [geom.box_height, geom.box_height, 0.0, 0.0]
        course = terrains.Course(
            # The DEPTH is in the name because it is now a first-class arm of this harness
            # (trained 0.46 vs mjeval's 1.20), and the scene file is keyed by name: two box
            # lanes differing only in depth would otherwise write one file. Sequential runs
            # rewrite before loading and are safe either way; `--scene_tag` remains the guard
            # for anything else that differs, including approach and runout.
            name=(f"veltrack_box_h{int(round(geom.box_height * 100)):03d}"
                  f"_d{int(round(geom.box_depth * 100)):03d}{tag}"),
            klass="box", geoms=geoms, floor_z=0.0,
            target_xy=(goal_x + terrains.GOAL_DEPTH / 2.0, 0.0),
            course_length_m=goal_x, flight_height_m=geom.box_height, obstacle_x_m=x_enter,
            note=(f"one box {geom.box_height:.2f} m high, {geom.box_depth:.2f} m deep"
                  + ("" if abs(geom.box_depth - gss.TRAINED_BOX_DEPTH) < 1e-9 else
                     f" (depth is OOD -- trained {gss.TRAINED_BOX_DEPTH} m)")),
            params={"approach_m": approach, "runout_m": runout,
                    "height_m": geom.box_height, "depth_m": geom.box_depth,
                    "width_m": geom.box_width}, **common)

    elif kind == "stairs_up":
        rise, run, steps = geom.stair_rise, geom.stair_run, geom.stair_steps
        x_enter = float(approach)
        x_exit = float(approach + steps * run)
        top_z = steps * rise
        goal_x = x_exit + runout
        # The platform has to carry the whole runout AND the far half of the goal band, or
        # the episode would end in mid-air past the platform edge.
        platform_depth = runout + terrains.GOAL_DEPTH + 2.0
        lane = gss.Lane(name="flight", y=0.0, width=geom.stair_width, steps=steps)
        geoms = tuple(gss._lane_geoms(lane, rise, run, x_enter, platform_depth,
                                      descend=False))
        for m in range(1, steps + 1):
            px += [x_enter + (m - 1) * run, x_enter + m * run]
            pz += [m * rise, m * rise]
        px += [x_enter + steps * run + platform_depth]
        pz += [top_z]
        course = terrains.Course(
            name=f"veltrack_up_s{steps}{tag}", klass="stairs_up", geoms=geoms, floor_z=0.0,
            target_xy=(goal_x + terrains.GOAL_DEPTH / 2.0, 0.0),
            course_length_m=goal_x, flight_height_m=top_z, obstacle_x_m=x_enter,
            note=(f"ascending {steps} x ({rise:.3f}/{run:.3f} m) = {top_z:.2f} m, width "
                  f"{geom.stair_width:.2f} m, NO stairwell walls"),
            params={"approach_m": approach, "runout_m": runout, "rise_m": rise,
                    "run_m": run, "steps": steps, "width_m": geom.stair_width,
                    "platform_depth_m": platform_depth, "walls": False}, **common)

    elif kind == "stairs_down":
        rise, run, steps = geom.stair_rise, geom.stair_run, geom.stair_steps
        drop = steps * rise
        floor_z = -drop
        x_enter = float(approach)
        # The last riser lands on the floor plane itself, so there are `steps - 1` pillars
        # and `steps` risers -- the same convention `_lane_geoms`' descent uses.
        x_exit = float(approach + (steps - 1) * run)
        goal_x = x_exit + runout
        geoms = [f'        <!-- descending {steps} x {rise:.3f} m = {drop:.2f} m drop; '
                 f'floor at {floor_z:.2f} -->',
                 # The raised approach slab, its top at z = 0 because that is the height the
                 # robot keyframe and --spawn_z_offset are both written against.  Wider than
                 # the flight, so a lateral excursion on the approach is not a drop-off.
                 terrains._slab("top_platform", -SLAB_BEHIND_M, x_enter, 0.0,
                                geom.stair_width + 2.0, floor_z, 0.0)]
        for m in range(1, steps):
            geoms.append(terrains._slab(f"down{m}", x_enter + (m - 1) * run,
                                        x_enter + m * run, 0.0, geom.stair_width,
                                        floor_z, -m * rise))
            px += [x_enter + (m - 1) * run, x_enter + m * run]
            pz += [-m * rise, -m * rise]
        px += [x_exit, goal_x + 5.0]
        pz += [floor_z, floor_z]
        course = terrains.Course(
            name=f"veltrack_dn_s{steps}{tag}", klass="stairs_down", geoms=tuple(geoms),
            floor_z=floor_z, target_xy=(goal_x + terrains.GOAL_DEPTH / 2.0, 0.0),
            course_length_m=goal_x, flight_height_m=-drop, obstacle_x_m=x_enter,
            note=(f"descending {steps} x ({rise:.3f}/{run:.3f} m) = {drop:.2f} m, width "
                  f"{geom.stair_width:.2f} m, NO stairwell walls; dug DOWN so the spawn "
                  f"stays at z=0"),
            params={"approach_m": approach, "runout_m": runout, "rise_m": rise,
                    "run_m": run, "steps": steps, "width_m": geom.stair_width,
                    "drop_m": drop, "slab_behind_m": SLAB_BEHIND_M, "walls": False},
            **common)
    else:
        raise ValueError(f"unknown terrain {kind!r}; expected one of {TERRAIN_KINDS}")

    course.params["kind"] = kind
    return Terrain(kind=kind, course=course, profile_x=tuple(px), profile_z=tuple(pz),
                   x_enter=x_enter, x_exit=x_exit, goal_x=float(goal_x))


def episode_budget_s(course: terrains.Course, speed: float, *, hold_s: float,
                     speed_frac: float) -> float:
    """Seconds one episode is given, from the course length and the commanded speed.

    Generous on purpose: a timeout here is a wasted trial rather than a finding, because the
    figure is about the velocity a surviving robot achieves.  ``speed_frac`` is the fraction
    of the commanded speed the robot is assumed to average -- 0.5 by default, so a robot that
    walks the lane at half the speed it was asked for still finishes.

    Args:
        course: The course, for ``course_length_m``.
        speed: Commanded forward speed (m/s).
        hold_s: The opening zero-command hold (s), paid for rather than eaten out of margin.
        speed_frac: Fraction of ``speed`` assumed as the average traverse rate.

    Returns:
        The budget in seconds.
    """
    rate = max(float(speed) * float(speed_frac), 0.2)
    return float(hold_s) + BUDGET_SETTLE_S + float(course.course_length_m) / rate


class TraceRecorder:
    """Per-control-step trace of what was commanded and what the root did.

    Wired as ``run_episode(probe=...)`` rather than ``recorder=...`` because the COMMAND is
    what the figure is about and only the probe is handed the state dict the policy was given
    -- which is where the live command lives once the steering controller starts varying it.
    Reconstructing it here from the clock and the goal instead would be a second statement of
    ``steer_yaw_rate``, and the two would drift.

    **The velocity frame is :mod:`accumulate`'s, by import rather than by restatement.** The
    commanded ``(vx, vy)`` is issued in the root's YAW frame -- gravity-aligned, zero roll and
    pitch -- so the achieved velocity has to be resolved there too; rotating through the FULL
    root quaternion bleeds vertical velocity into the downrange channel whenever the torso
    pitches, which is exactly what happens on a staircase and would show up as a tracking
    error that is really a frame error.  ``wz`` is the WORLD-frame angular z, which is
    yaw-invariant and therefore the same number in either frame -- and is NOT ``qvel[5]``,
    the body-local rate MuJoCo stores.
    """

    def __init__(self, base_frame: str, decim: int = 1) -> None:
        """Start an empty trace.

        Args:
            base_frame: The base LINK frame whose twist is read (``policy.base_frame_name``).
            decim: Keep every ``decim``-th control step.  1 keeps all of them, which is what
                a 50 Hz trace over a 60 s lane costs 3000 rows of.
        """
        self.base_frame = str(base_frame)
        self.decim = max(1, int(decim))
        self._n = 0
        self.rows: list[tuple[float, ...]] = []

    def __call__(self, robot, policy, state, is_replan) -> None:
        """Fold one control step.  Signature is ``run_episode``'s ``probe`` contract."""
        self._n += 1
        if (self._n - 1) % self.decim:
            return
        data = robot.mj_data
        qpos = np.asarray(data.qpos, dtype=np.float64)
        lin_w, ang_w = robot.get_base_world_velocity(self.base_frame)
        lin_yaw = mjacc.rotate_inverse(mjacc.yaw_quat(qpos[3:7]), lin_w)
        cmd = np.asarray(state["command"], dtype=np.float64).reshape(3)
        self.rows.append((
            float(data.time),
            float(qpos[0]), float(qpos[1]), float(qpos[2]),
            float(lin_yaw[0]), float(lin_yaw[1]), float(ang_w[2]),
            float(cmd[0]), float(cmd[1]), float(cmd[2]),
        ))

    def array(self) -> np.ndarray:
        """The trace as ``(steps, 10)`` float32: t, xyz, achieved vx/vy/wz, commanded."""
        if not self.rows:
            return np.zeros((0, 10), dtype=np.float32)
        return np.asarray(self.rows, dtype=np.float32)


#: Column order of :meth:`TraceRecorder.array`, and of the ``.npz``'s per-step arrays.
TRACE_COLUMNS: tuple[str, ...] = ("t", "x", "y", "z", "vx", "vy", "wz",
                                  "cmd_vx", "cmd_vy", "cmd_wz")


def artifact_fingerprint(path: str) -> dict:
    """Identity of one artifact: its path, size, mtime and content sha256.

    The PATH is not identity. Exported names are deliberately stable -- ``<run>_fm.onnx`` is
    the same filename whether it was exported today or last week, and re-exporting a
    fine-tuned checkpoint overwrites in place -- so a recorded path alone cannot tell you
    which bytes produced a result. The hash can, and it is what makes "regenerate this figure"
    a checkable claim rather than a hopeful one.

    Args:
        path: File to fingerprint.

    Returns:
        A dict with ``path``, ``bytes``, ``mtime`` and ``sha256``; ``sha256`` is ``None`` when
        the file cannot be read, which is reported rather than raised -- provenance must never
        be the thing that fails a run.
    """
    out = {"path": os.path.abspath(path), "bytes": None, "mtime": None, "sha256": None}
    try:
        st = os.stat(path)
        out["bytes"], out["mtime"] = int(st.st_size), int(st.st_mtime)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    except OSError as exc:
        print(f"[WARN] could not fingerprint {path}: {exc}")
    return out


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracker_onnx", required=True, help="Absolute path to tracker.onnx.")
    p.add_argument("--fm_onnx", required=True, help="Absolute path to the generator .onnx.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--terrain", action="append", choices=TERRAIN_KINDS, default=None,
                   help=f"Repeatable. Default: all of {', '.join(TERRAIN_KINDS)}.")
    p.add_argument("--speed", type=float, action="append", default=None,
                   help="Repeatable commanded forward speed (m/s). Default: 0.8 and 1.5.")
    p.add_argument("--trials", type=int, default=50,
                   help="Repeats per (terrain, speed). Seeds are 0..trials-1, so a longer "
                        "run is a SUPERSET of a shorter one rather than a different sample.")
    p.add_argument("--approach", type=float, default=10.0,
                   help="Flat metres before the feature's near face.")
    p.add_argument("--runout", type=float, default=10.0,
                   help="Flat metres past the feature before the episode ends.")
    p.add_argument("--cmd_hold_s", type=float, default=terrains.CMD_HOLD_S,
                   help="Seconds of ZERO command before the step to --speed.")
    p.add_argument("--steer_kp", type=float, default=1.5,
                   help="P gain (1/s) on heading error toward the goal, which is what keeps "
                        "the robot in line with the feature. 0 disables steering; see the "
                        "module docstring on why it is off-distribution on stairs.")
    p.add_argument("--steer_wz_max", type=float, default=mjrunner.STEER_WZ_MAX,
                   help="Symmetric clip on the steering yaw rate (rad/s).")
    p.add_argument("--budget_speed_frac", type=float, default=0.5,
                   help="Fraction of the commanded speed assumed as the average traverse "
                        "rate when sizing each episode's time budget.")
    p.add_argument("--trace_decim", type=int, default=1,
                   help="Keep every Nth control step in the trace. 1 keeps all of them.")
    p.add_argument("--box_height", type=float, default=Geometry.box_height)
    p.add_argument("--box_depth", type=float, default=Geometry.box_depth)
    p.add_argument("--box_width", type=float, default=Geometry.box_width)
    p.add_argument("--stair_rise", type=float, default=Geometry.stair_rise)
    p.add_argument("--stair_run", type=float, default=Geometry.stair_run)
    p.add_argument("--stair_steps", type=int, default=Geometry.stair_steps)
    p.add_argument("--stair_width", type=float, default=Geometry.stair_width)
    p.add_argument("--zero_noise_control", action="store_true",
                   help="CONTROL: integrate the flow from x0=0 for every trial, so every "
                        "repeat is bit-identical. Every trace column must then be identical "
                        "across repeats; anything else means something other than the flow "
                        "noise varies between them, and the sweep measures that too.")
    p.add_argument("--scene_tag", default="",
                   help="Suffix for the emitted scene names. REQUIRED to be distinct when two "
                        "of these run concurrently: the scene is written to "
                        "robots/g1/<robot>_<course>.xml before it is loaded, and for a speed "
                        "sweep the geometry is identical, so the race tears the file silently "
                        "rather than failing to load.")
    p.add_argument("--providers", default="auto", help="onnxruntime providers.")
    p.add_argument("--num_threads", type=int, default=8,
                   help="onnxruntime intra-op threads. 0 is ORT's default of one per core, "
                        "which on a 128-core shared box is a 254-thread process at 6000% CPU "
                        "for a job whose heavy half runs on the GPU -- and it throttles "
                        "itself: episodes measured at 40 s stretched past 4 minutes once the "
                        "box was oversubscribed, this process being the largest contributor.")
    p.add_argument("--fall_height", type=float, default=0.2,
                   help="Pelvis clearance (m) above the scene's lowest terrain below which "
                        "the episode is a fall.")
    p.add_argument("--fall_tilt_deg", type=float, default=70.0, help="Tilt bound from upright.")
    return p


def run_cell(terrain: Terrain, speed: float, speed_idx: int, args, cfg: mjrunner.ArmRun,
             policy, fm_lag, spawn, table: mjschema.EpisodeTable, traces: list[np.ndarray],
             episodes: list[dict]) -> dict:
    """Run every trial of one (terrain, speed) cell.

    The pair is loaded ONCE for the whole sweep and the noise stream is re-keyed per trial:
    loading the ONNX pair costs seconds, and ``run_episode(episode_seed=...)`` restarts the
    stream rather than resuming it -- without which a trial's noise would depend on how many
    re-plans every trial before it made, and a trial that falls early makes fewer, so the
    tables would drift while looking entirely clean.

    **The key is the SEED ALONE, not ``(seed, terrain, speed)``.** Trial ``s`` therefore opens
    on the same noise tape in every cell, which makes the columns of the figure PAIRED on the
    draw: a difference between flat and stairs at one seed is the terrain, not a different
    sample of the generator. The pairing decays as the episodes diverge in length -- the tape
    is consumed per re-plan -- so it is a pairing of the openings, which is where the mode is
    chosen.

    Args:
        terrain: The built course, already carrying this cell's time budget.
        speed: Commanded forward speed (m/s).
        speed_idx: This speed's index in the run's ladder, for the ``speed_idx`` column.
        args: Parsed CLI.
        cfg: The arm.
        policy: The loaded pair.
        fm_lag: The generator's camera-lag ring.
        spawn: The standing spawn.
        table: Episode table to append the standard mjeval row to.
        traces: Accumulates one ``(steps, 10)`` array per trial.
        episodes: Accumulates one per-trial dict of the trace's own scalars.

    Returns:
        This cell's summary.
    """
    rig = mjrunner.CourseRig(terrain.course, policy, cfg, spawn)
    cmd = np.array([float(speed), 0.0, 0.0], dtype=np.float64)
    print(f"[vt] {terrain.kind} @ {speed:.2f} m/s: {terrain.course.note}")
    print(f"[vt]   feature x=[{terrain.x_enter:.2f}, {terrain.x_exit:.2f}] m, goal line "
          f"x={terrain.goal_x:.2f} m, budget {terrain.course.budget_s:.1f} s, "
          f"steer_kp={cfg.steer_kp}, trials={args.trials}")
    t0 = time.time()
    verdicts: list[str] = []
    for seed in range(int(args.trials)):
        trace = TraceRecorder(policy.base_frame_name, decim=args.trace_decim)
        verdict, failure, acc = mjrunner.run_episode(
            rig, policy, fm_lag, spawn, cmd, None,
            steer_kp=cfg.steer_kp, steer_wz_max=cfg.steer_wz_max,
            cfg_hold=cfg.cmd_hold_s, probe=trace,
            # The whole lane is the measurement, so arrival ends the episode (that IS
            # `runout` metres past the feature, by construction of the goal line) but a
            # tracking violation does not -- there is no spec loaded, so nothing latches.
            stop_on_goal=True, episode_seed=seed)
        arr = trace.array()
        row = acc.row(episode_id=TERRAIN_CODES[terrain.kind] * 1_000_000
                      + int(round(speed * 100)) * 1000 + seed,
                      terrain_idx=TERRAIN_CODES[terrain.kind], speed_idx=int(speed_idx),
                      arm=0, seed=seed, verdict=verdict, failure_code=failure)
        table.append(row)
        traces.append(arr)
        last = arr[-1] if arr.size else np.zeros(10, dtype=np.float32)
        episodes.append({
            "kind": terrain.kind, "terrain": TERRAIN_CODES[terrain.kind],
            "speed": float(speed), "seed": int(seed),
            "verdict": terrains.VERDICT_CODES[verdict],
            "failure_code": terrains.FAILURE_CODES[failure],
            "steps": int(arr.shape[0]),
            "end_t": float(last[0]), "end_x": float(last[1]),
            "end_y": float(last[2]), "end_z": float(last[3]),
        })
        verdicts.append(verdict)
        if seed < 3 or (seed + 1) % 10 == 0:
            print(f"[vt]   seed {seed:>3}  {verdict:<9}{failure:<7} "
                  f"x={row['progress_x_m']:6.2f}m  t={last[0]:5.1f}s  "
                  f"({time.time() - t0:.0f}s wall)", flush=True)
    del rig
    v = np.array([terrains.VERDICT_CODES[x] for x in verdicts])
    cell = {
        "kind": terrain.kind, "speed": float(speed), "n": int(v.size),
        "course": terrain.course.name,
        "success_rate": float((v == terrains.VERDICT_CODES["success"]).mean()),
        "fall_rate": float((v == terrains.VERDICT_CODES["fall"]).mean()),
        "timeout_rate": float((v == terrains.VERDICT_CODES["timeout"]).mean()),
        "budget_s": float(terrain.course.budget_s),
        "wall_s": float(time.time() - t0),
    }
    print(f"[vt]   -> success {cell['success_rate']:.3f}  fall {cell['fall_rate']:.3f}  "
          f"timeout {cell['timeout_rate']:.3f}  ({cell['wall_s']:.0f}s)\n", flush=True)
    return cell


def write_traces(path: str, traces: list[np.ndarray], episodes: list[dict]) -> None:
    """Write every trial's trace to one ``.npz``, flat plus an index.

    Flat-and-indexed rather than one array per episode: the traces are ragged (an episode
    ends when it arrives or falls), and a ragged ``.npz`` needs ``allow_pickle`` to read back,
    which makes the file unloadable by anything that refuses pickles.

    Args:
        path: Destination ``.npz``.
        traces: One ``(steps, 10)`` array per trial, in episode order.
        episodes: One dict per trial, same order, from :func:`run_cell`.
    """
    lens = np.array([t.shape[0] for t in traces], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]) if lens.size else lens
    flat = (np.concatenate(traces, axis=0) if traces
            else np.zeros((0, len(TRACE_COLUMNS)), dtype=np.float32))
    out = {name: flat[:, i] for i, name in enumerate(TRACE_COLUMNS)}
    out["ep_start"] = starts.astype(np.int64)
    out["ep_len"] = lens
    for key, dtype in (("terrain", np.int64), ("speed", np.float64), ("seed", np.int64),
                       ("verdict", np.int64), ("failure_code", np.int64),
                       ("end_t", np.float64), ("end_x", np.float64), ("end_y", np.float64),
                       ("end_z", np.float64)):
        out[f"ep_{key}"] = np.array([e[key] for e in episodes], dtype=dtype)
    np.savez_compressed(path, **out)


def write_outputs(out_dir: str, table: mjschema.EpisodeTable, traces: list[np.ndarray],
                  episodes: list[dict], meta: dict, cells: list[dict],
                  built: dict[str, Terrain]) -> None:
    """Write the table, the traces and the summary for the cells finished SO FAR.

    Called after every cell rather than once at the end, and it is not a nicety: a sweep is
    tens of minutes of GPU time and a crash used to take all of it. One did -- a SIGFPE in
    native code eleven cells into a twelve-cell run left an output directory holding a log and
    nothing else, and the finished cells were not recoverable from it. Rewriting three files
    per cell costs well under a second against several minutes of episodes, and every consumer
    already reads a partial run correctly: ``summary.cells`` is the authority for what a run
    contains, and the plotter and the reducers key off it.

    Args:
        out_dir: The run directory.
        table: Episode rows so far.
        traces: One array per episode so far, in the same order.
        episodes: Per-episode scalars, same order.
        meta: Run provenance.
        cells: Cell summaries finished so far.
        built: Every terrain the run will cover, so the profiles are present from the first
            write -- a partial run still draws its finished columns.
    """
    table.write(os.path.join(out_dir, "episodes.h5"), meta)
    write_traces(os.path.join(out_dir, "traces.npz"), traces, episodes)
    summary = {
        "meta": meta,
        "cells": cells,
        "terrains": {k: {"kind": t.kind, "course": t.course.name,
                         "profile_x": list(t.profile_x), "profile_z": list(t.profile_z),
                         "x_enter": t.x_enter, "x_exit": t.x_exit, "goal_x": t.goal_x,
                         "flight_height_m": float(t.course.flight_height_m),
                         "floor_z": float(t.course.floor_z),
                         "note": t.course.note, "params": t.course.params}
                     for k, t in built.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)


def main() -> int:
    """Build each course, run every cell, and write the table, the traces and the summary."""
    args = build_parser().parse_args()
    kinds = tuple(dict.fromkeys(args.terrain or TERRAIN_KINDS))
    speeds = tuple(args.speed if args.speed else (0.8, 1.5))
    os.makedirs(args.out, exist_ok=True)
    for role, path in (("tracker", args.tracker_onnx), ("generator", args.fm_onnx)):
        if not os.path.isfile(path):
            raise SystemExit(f"no {role} at {path}")
    if args.approach <= 0.0 or args.runout <= 0.0:
        raise SystemExit("--approach and --runout must both be positive")

    geom = Geometry(box_height=args.box_height, box_depth=args.box_depth,
                    box_width=args.box_width, stair_rise=args.stair_rise,
                    stair_run=args.stair_run, stair_steps=args.stair_steps,
                    stair_width=args.stair_width)
    cfg = mjrunner.ArmRun(
        arm=0, label="veltrack", tracker_onnx=os.path.abspath(args.tracker_onnx),
        fm_onnx=os.path.abspath(args.fm_onnx), out_dir=args.out,
        providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
        num_threads=int(args.num_threads),
        fall_height=float(args.fall_height), fall_tilt_deg=float(args.fall_tilt_deg),
        cmd_hold_s=float(args.cmd_hold_s), steer_kp=float(args.steer_kp),
        steer_wz_max=float(args.steer_wz_max),
        zero_noise=bool(args.zero_noise_control), fm_seed=0,
    )
    meta = {
        "harness": "vel_tracking_sweep",
        # Exactly how this run was invoked. Recorded because every knob below is a DEFAULT
        # until it is not, and a figure regenerated from a different default is not the same
        # figure -- `box_depth` moved from 1.20 to the trained 0.46 mid-investigation, which
        # would be invisible from a summary that only stored the resolved value.
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "tracker_onnx": cfg.tracker_onnx, "fm_onnx": cfg.fm_onnx,
        "tracker_artifact": artifact_fingerprint(cfg.tracker_onnx),
        "fm_artifact": artifact_fingerprint(cfg.fm_onnx),
        "terrains": list(kinds), "speeds": [float(s) for s in speeds],
        "trials": int(args.trials), "approach_m": float(args.approach),
        "runout_m": float(args.runout), "cmd_hold_s": float(args.cmd_hold_s),
        "steer_kp": float(args.steer_kp), "steer_wz_max": float(args.steer_wz_max),
        "budget_speed_frac": float(args.budget_speed_frac),
        "trace_decim": int(args.trace_decim),
        "zero_noise_control": bool(args.zero_noise_control),
        "fall_height": float(args.fall_height), "fall_tilt_deg": float(args.fall_tilt_deg),
        "scene_tag": str(args.scene_tag), "stair_walls": False,
        "geometry": {k: getattr(geom, k) for k in geom.__dataclass_fields__},
        "trace_columns": list(TRACE_COLUMNS),
        "verdict_codes": terrains.VERDICT_CODES, "failure_codes": terrains.FAILURE_CODES,
        "terrain_codes": TERRAIN_CODES,
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    built = {k: build_terrain(k, approach=args.approach, runout=args.runout, geom=geom,
                              speeds=speeds, tag=args.scene_tag) for k in kinds}
    if args.zero_noise_control:
        print("[vt] ZERO-NOISE CONTROL: every trial integrates the flow from x0 = 0, so "
              "every trace column must be identical across repeats.")
    if args.steer_kp > 0.0 and any(k.startswith("stairs") for k in kinds):
        print("[vt] NOTE: steering is ON and is OFF-DISTRIBUTION on stairs (|wz| = 0.0000 "
              "on all 3500 training stair clips). The stair panels are a different "
              "measurement from an unsteered run, not a cleaner one.")

    # Commit + status + diff under <out>/git/, best-effort, exactly as the recording and
    # flow-matching runs do it.
    provenance.git_snapshot(args.out, repo_root=_REPO)
    policy, fm_lag, spawn = mjrunner.load_arm(cfg)
    table = mjschema.EpisodeTable()
    traces: list[np.ndarray] = []
    episodes: list[dict] = []
    cells = []
    t_start = time.time()
    for kind in kinds:
        for i_s, speed in enumerate(speeds):
            terrain = built[kind]
            budget = episode_budget_s(terrain.course, speed, hold_s=args.cmd_hold_s,
                                      speed_frac=args.budget_speed_frac)
            cell_terrain = replace(terrain,
                                   course=replace(terrain.course, budget_s_override=budget))
            cells.append(run_cell(cell_terrain, speed, i_s, args, cfg, policy, fm_lag, spawn,
                                  table, traces, episodes))
            # Checkpoint after every cell: see write_outputs on why this is not optional.
            write_outputs(args.out, table, traces, episodes, meta, cells, built)

    total = sum(c["n"] for c in cells)
    h5 = os.path.join(args.out, "episodes.h5")
    npz = os.path.join(args.out, "traces.npz")
    print(f"[vt] {total} episodes in {time.time() - t_start:.0f}s -> {h5}")
    print(f"[vt] traces -> {npz}  ({sum(t.shape[0] for t in traces)} control steps)")
    print(f"[vt] summary -> {os.path.join(args.out, 'summary.json')}")
    print(f"[vt] figure:  python plot_vel_tracking.py {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
