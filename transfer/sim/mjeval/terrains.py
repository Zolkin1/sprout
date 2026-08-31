# Copyright 2026 Zachary Olkin. All rights reserved.

"""The eval terrain catalogue: 20 hand-authored MuJoCo courses, and where "through" is.

These are deliberately NOT the tiles the policy trained on.  Every clip tile came out of the
Motion Bricks generator and was then decorated; a course here is authored from the MEASURED
training distribution (``generate_stairs_scene``'s constants) with one parameter moved at a
time, so a failure names a geometry rather than a tile id.

Each course is a straight lane along +x.  The robot spawns at the world origin facing +x on
ground at ``z = 0``, walks a flat approach, meets exactly one obstacle class, and is done when
its root enters a circle past the far side.  There is nothing else in the scene -- no side
lanes, no decor, no walls -- so nothing but the course can explain a plan.

**Three conventions, each of which a consumer depends on.**

* **Every terrain geom is group 2** and carries no ``contype``/``conaffinity``.  Group 2 is
  simultaneously what the depth cameras cast (``depth_camera.TERRAIN_GROUP``), what
  ``Robot.compute_height_scan`` filters on, and what ``Robot.terrain_min_surface_z`` reads for
  the fall datum -- so one attribute makes a box physical, visible and fall-relevant at once.
  Setting ``contype=0`` would make it visual-only; ``mj_ray`` ignores those fields anyway.
* **Descending courses are dug DOWN, not spawned UP.**  The robot's keyframe (pelvis at
  ``(0, 0, 0.79)``, ``robots/g1/g1_perceptive.xml``) and ``--spawn_z_offset`` are both stated
  against ground at ``z = 0``, so a course that starts high would need both restated.  Instead
  the ground PLANE is lowered to the bottom of the flight and a raised approach slab carries
  the robot at ``z = 0``.  The cost is stated in :attr:`Course.fall_datum_z`: the scene minimum
  is then the bottom floor, so the height limb of the fall test is loose while the robot is
  still up top and only the 70-degree tilt cone is guarding it.
* **The target is 2-D.**  ``(x, y)`` only, at whatever height the course exits at.  A sphere
  would fail a robot standing at the right place on the wrong step, and "got through" is a
  question about ground covered.

**The traversal gate exists because a circle alone can be walked around.**  These lanes are
finite in y and the floor is infinite, so a robot that drifts wide reaches the target without
ever touching the course.  :meth:`Course.traversed` additionally requires the root to have
crossed a height threshold in the right direction -- 80% of the flight, which no gait
oscillation on flat reaches and no genuine climb misses.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, replace

_HERE = os.path.dirname(os.path.abspath(__file__))
# `generate_stairs_scene` is the sibling that owns the emitter and every measured constant.
# By path rather than by install, exactly as `gen_tracker_runner` and `stair_probe` reach
# their own siblings -- `transfer/sim` is not a package on sys.path.
sys.path.insert(0, os.path.dirname(_HERE))

import generate_stairs_scene as gss  # noqa: E402

#: Flat run-up (m) before the first riser / box face on every course.  2.5 m is
#: ``generate_stairs_scene.DEFAULT_X_START``: long enough that the far camera (35 deg down,
#: boresight meeting the ground at 1.8 m) sees the obstacle for several re-plans before the
#: robot arrives, short enough that a 0.6 m/s trial is not mostly approach.
X_START: float = 2.5

#: How far past the far edge of the last walkable geom the goal line sits (m).
TARGET_LEAD: float = 1.5

#: EXTRA flat ground a DESCENDING course requires past the bottom of its flight (m), on top of
#: :data:`TARGET_LEAD`.
#:
#: Without it the goal band's near edge sits only ``TARGET_LEAD - GOAL_DEPTH / 2`` = 0.5 m past
#: the last riser, which a robot can enter while still stumbling off the flight -- so "got down
#: the stairs" and "fell down the stairs in roughly the right direction" score the same.  Three
#: extra metres puts the near edge 3.5 m out, which is several strides of recovered walking, so
#: the episode has to end with the robot under control rather than merely past the bottom.
#:
#: Ascending courses do not need it: their goal sits on a flat top platform the robot has to be
#: standing on, which already demands the same thing.
DESCENT_RUNOUT: float = 3.0

#: Where the descending stairwell's walls STOP, past the bottom riser (m).  Its own constant
#: rather than :data:`TARGET_LEAD`, which it used to borrow: a real stairwell's walls end at the
#: foot of the stairs, so moving the GOAL must not drag the walls out with it.
DESCENT_WALL_RUNOUT: float = 1.5

#: **The goal is a RECTANGLE, not a circle, and it is deliberately wide.**  A 0.5 m circle on
#: the lane centreline was the first design and it was wrong: measured over the first 400-episode
#: grid, most timeouts were a robot that had cleared the terrain cleanly and then passed BESIDE
#: the disc, because nothing in this stack steers -- the command is a body-frame velocity and a
#: few degrees of accumulated heading error over six metres is more than half a metre of lateral
#: miss.  Scoring that as "did not get through" measures the absence of a heading controller,
#: which is not what the eval is for.
#:
#: So the goal is a band across the lane: any ``|y - goal_y|`` inside :data:`GOAL_HALF_WIDTH`
#: and any ``x`` inside :data:`GOAL_DEPTH` of the goal line counts.
#:
#: Half-width floor, in metres.  The widest terrain feature in the catalogue is the descending
#: courses' approach slab at 3.96 m (half 1.98), so 2.5 m clears everything with margin and
#: :func:`goal_half_width` widens it further per course where the geometry demands.
GOAL_HALF_WIDTH_FLOOR: float = 2.5

#: Extra half-width (m) beyond a course's own widest feature.
GOAL_WIDTH_MARGIN: float = 1.0

#: Flank every staircase, ascending and descending, with walls TOUCHING the tread edge --
#: what a real stairwell has, and the case the deployed pair has to survive.
#:
#: **Visual only, and not negotiable** (``generate_stairs_scene.lane_wall_geoms`` says the same
#: and refuses to emit a solid one).  Isaac's decoration carries no collider, nothing in
#: training has ever pushed back on the robot, and there is no obstacle term anywhere in the
#: MDP.  A solid wall this close would be swept by the reference's OWN arm swing, blocking a
#: wrist and firing ``ee_pose_error`` -- ending the episode for something no policy could fix,
#: and measuring the harness rather than the robot.  What a wall is here for is what the
#: generator PLANS with one in frame, which needs no contact: the wall-as-box misread
#: reproduces in MuJoCo on a joystick and is the open exteroception question.
#:
#: Turning this off changes what the cameras see and therefore what the generator plans, so a
#: grid run with it is NOT comparable to one run without.  ``view_cell.py`` will refuse rather
#: than mislead: it rebuilds the scene from this catalogue and checks the replay against the
#: recorded row, so an older grid replayed under a changed catalogue fails loudly.
STAIR_WALLS: bool = True

#: Lateral clearance (m) from tread edge to wall face.  Zero = touching, as asked.
STAIR_WALL_GAP: float = 0.0

#: Stairwell wall height (m) and thickness (m).  Height is a VIEWER choice above ~0.9 m: the
#: far camera's frame cuts a wall at z = 0.87 m at 3 m range, so 1.0 m and 3.0 m walls are
#: byte-identical images.
STAIR_WALL_HEIGHT: float = 2.0
STAIR_WALL_THICKNESS: float = 0.15

#: How far before the first riser the wall begins (m).  The approach is when the deciding plan
#: is made, so the wall has to be in frame before the robot arrives.
STAIR_WALL_LEAD: float = 1.0

#: Depth of the goal band along travel (m), centred on the goal line.  At the fastest commanded
#: speed the root moves 0.052 m per control step, so a 2 m band cannot be stepped over.
GOAL_DEPTH: float = 2.0

#: Fraction of the flight height the root must clear (ascending) or drop (descending) for the
#: course to count as traversed.  0.8 rather than 1.0 because the pelvis does not rise by the
#: full flight height -- the legs absorb part of it -- and rather than 0.5 because half a
#: 0.25 m box is inside a normal gait's vertical excursion.
TRAVERSAL_FRAC: float = 0.8

#: Pelvis height (m) at spawn: the standing config's value plus the runner's default
#: ``--spawn_z_offset``.  Used only as the datum for the traversal gate, which is a
#: DIFFERENCE, so a few cm of error in it is harmless.
SPAWN_BASE_Z: float = 0.79

#: Slowest speed (m/s) the timeout budget assumes the robot averages.  Every course gets ONE
#: budget, computed from this, shared by all ten speeds.  Scaling the budget with the
#: commanded speed -- what ``stair_probe.trial_seconds`` does, correctly, for a human-watched
#: trial -- would make "timeout" mean something different at each point of the very axis
#: being swept: a 2.6 m/s cell would be judged against a tighter clock than a 0.6 m/s one.
BUDGET_SPEED: float = 0.5

#: Seconds added to every budget for the spawn transient and the command step-in.
BUDGET_SETTLE: float = 4.0

#: Seconds of ZERO velocity command at the start of every episode, before the command steps to
#: its value.  The robot spawns from a solved standing pose AT REST, and without this it is
#: told the full speed on the very first control step -- so at 2.6 m/s it is asked to go from 0
#: to 2.6 instantly.  Measured on the first 400-episode grid, that is where the baseline
#: actually failed: 59% of its falls (94 of 159) never reached the terrain at all, and in the
#: fast half of the ladder the median fall was 0.40 m from spawn.
#:
#: A HOLD then a STEP rather than a ramp, because a ramp is its own off-distribution shape --
#: every terrain clip in the library holds ONE constant command for its whole length, with no
#: ramp, from frame 0.  What the library does contain is standing-to-walk transitions, on the
#: ``flat`` family, whose commands start at 0.0.  So the transient is moved onto flat ground
#: where it is trained, and the robot arrives at the terrain already walking at speed, which is
#: the condition every terrain clip was generated under.
CMD_HOLD_S: float = 0.5

# --- the swept command ladders ---------------------------------------------------------
#
# Per FAMILY, not shared, and measured from the TRAINING LIBRARY itself
# (`clips/g1_mb_v5_body_mode11_decor.h5`, `target_vel[:, 0, 0]` -- the command is a per-clip
# constant on every terrain family, so frame 0 IS the clip's command).  Reading it off an
# EVAL's `cmd_speed_mean` instead is a trap: the 264-clip eval subset sampled only two stair
# levels, 0.600 and 2.600, so its median of 1.60 is the midpoint of a bimodal pair and not a
# statement about the library at all.
#
#     terrain + transition (the stair families)   3500 clips, 94 distinct levels,
#         a UNIFORM GRID 0.600 .. 2.600 step 0.0215 -- 350 clips in each of ten equal bins.
#     box                                         3500 clips, 1829 distinct levels,
#         CONTINUOUS, 79% forward 0.503 .. 3.499 and 21% backward (the dropped `jump_off`).
#         NOT uniform: it peaks at 1.10-1.40 (p50 1.49) with a long thin tail to 3.50.
#
# So the two ladders are chosen differently on purpose:
#
# * STAIR_SPEEDS are ten EXACT points of the trained grid (a 1-in-10 subsample, both ends
#   included), so every stair speed this harness commands is one the library actually
#   contains.  Uniform spacing is not an assumption here -- it is the trained density.
# * BOX_SPEEDS span the full trained FORWARD range uniformly.  That is deliberately NOT the
#   trained density, which is peaked: even spacing is what a success-vs-speed CURVE needs, and
#   matching the density would spend three of ten levels between 1.10 and 1.40.  The cost is
#   that the fast cells are thinner in training than the plot's even spacing suggests, and the
#   report says so.
#
# Nothing goes past 2.600 on stairs.  Above it the library's only answer to "fast forward
# approach" is a jump, which is the documented mechanism behind reading a staircase as a box --
# so commanding a staircase at 3 m/s would measure a coverage hole rather than the policy.
STAIR_SPEEDS: tuple[float, ...] = (0.600, 0.804, 1.049, 1.253, 1.457,
                                   1.743, 1.947, 2.151, 2.396, 2.600)
BOX_SPEEDS: tuple[float, ...] = (0.50, 0.83, 1.17, 1.50, 1.83, 2.17, 2.50, 2.83, 3.17, 3.50)

#: Course classes, as stored in the ``terrain_class`` column.
CLASS_CODES: dict[str, int] = {
    "stairs_up": 0, "stairs_down": 1, "box": 2, "two_box_course": 3, "up_into_down": 4,
}

#: Verdicts, as stored in the ``verdict`` column.
VERDICT_CODES: dict[str, int] = {"success": 0, "fall": 1, "timeout": 2, "interrupted": 3}

#: ``Robot.failure_reason()`` strings, as stored in the ``failure_code`` column.
FAILURE_CODES: dict[str, int] = {"": 0, "height": 1, "tilt": 2}


@dataclass(frozen=True)
class Course:
    """One authored terrain, everything a trial needs to run and be scored.

    Attributes:
        name: Scene name, without the robot prefix.  The emitted file is
            ``robots/g1/<robot>_<name>.xml`` -- :func:`write_scene` puts it where
            ``Robot._get_model_data`` looks, which is the only place it looks.
        klass: Course class; a key of :data:`CLASS_CODES`.
        variant: Index within the class (which of the five stairs-up, etc.).
        geoms: The ``<geom .../>`` lines for everything but the floor.
        floor_z: World z (m) of the ground plane.  0.0 everywhere except the descending
            courses, which lower it to the bottom of their flight (see the module docstring).
        target_xy: Centre of the goal rectangle (m); its ``x`` is the goal line.
        goal_half_width_m: Half the goal band's lateral extent (m).  Set from the course's own
            widest terrain feature, floored at :data:`GOAL_HALF_WIDTH_FLOOR`.
        course_length_m: Downrange distance from spawn to the target, the denominator of
            ``progress_frac``.
        flight_height_m: Signed height change the course imposes on the root: positive for a
            climb, negative for a descent.  ``0.0`` disables the traversal gate.
        obstacle_x_m: World x of the first riser / box face.  Splits the velocity-tracking
            metric into approach and on-course halves.
        speeds: The ten commanded forward speeds for this course.
        note: One line for the manifest, saying what this course varies.
        goal_anchor_xyz: Where the failure critic measures its three goal features FROM, in
            this course's world frame, or ``None`` when no anchor is available.  This is NOT
            ``target_xy``: that is the far platform's CENTRE and owns ``in_goal`` / success,
            whereas the critic's anchor is where the reference CLIP finishes, and the two
            differ by a family-systematic 0.44-3.17 m (see :mod:`goal_anchor`).  Keeping them
            separate is what lets the critic be scored here without moving any success number
            already measured.
        goal_anchor_source: ``"clip"`` when recovered from the library's own terminal pose,
            ``"none"`` when absent.  A course with no anchor cannot be critic-scored, and
            saying so beats defaulting to ``target_xy``, which would read as an anchor while
            being off by most of a normalized unit.
    """

    name: str
    klass: str
    variant: int
    geoms: tuple[str, ...]
    floor_z: float
    target_xy: tuple[float, float]
    course_length_m: float
    flight_height_m: float
    obstacle_x_m: float
    speeds: tuple[float, ...]
    note: str
    goal_half_width_m: float = GOAL_HALF_WIDTH_FLOOR
    params: dict = field(default_factory=dict)
    goal_anchor_xyz: tuple[float, float, float] | None = None
    goal_anchor_source: str = "none"
    budget_s_override: float | None = None

    @property
    def class_code(self) -> int:
        """This course's :data:`CLASS_CODES` value."""
        return CLASS_CODES[self.klass]

    @property
    def budget_s(self) -> float:
        """The timeout, in seconds, shared by all ten speeds on this course.

        Includes :data:`CMD_HOLD_S`, so the opening standing hold is paid for rather than
        eaten out of the margin.

        Derived from :data:`BUDGET_SPEED` so it is generous for the slowest trial and
        therefore not the binding constraint at any speed -- which is what makes a timeout at
        2.6 m/s a finding about the policy rather than about the clock.

        **``budget_s_override`` replaces this with the ISAAC episode length**, and that is the
        right thing whenever something fitted on Isaac rollouts is being read here -- the
        failure critic above all.  Isaac's eval adds a ``clip_end`` termination at
        ``clip_len - 1 - max_offset``, so that is the only window the critic ever saw; a
        MuJoCo run that keeps going past it reads the critic in a regime no training episode
        reached.  Measured on the shipped tiles, this default runs 1.4-3.6x longer than the
        clip, so the gap is not marginal.  See ``goal_anchor.tile_episode_seconds``.

        The distance-derived default is also anti-correlated with difficulty on a box tile:
        tile 40's clips are the LONGEST of the three feature tiles (10.4 s) while its course
        is the shortest, so it gets the tightest budget of the three.
        """
        if self.budget_s_override is not None:
            return float(self.budget_s_override)
        return self.course_length_m / BUDGET_SPEED + BUDGET_SETTLE + CMD_HOLD_S

    @property
    def fall_datum_z(self) -> float:
        """Scene minimum surface z (m), i.e. what ``Robot.terrain_min_surface_z`` will return.

        Recorded so the manifest can state how loose the height limb of the fall test is on
        this course: the robot is guarded at ``fall_datum_z + fall_height`` no matter how high
        it currently stands, so on a descending course a collapse up top is caught by the tilt
        cone alone.
        """
        return self.floor_z

    def in_goal(self, x: float, y: float) -> bool:
        """Is the root inside the goal rectangle?

        Args:
            x: Root world x (m).
            y: Root world y (m).

        Returns:
            True when the root is within :data:`GOAL_DEPTH` / 2 of the goal line and within
            :attr:`goal_half_width_m` of the lane centre.
        """
        gx, gy = self.target_xy
        return (abs(x - gx) <= GOAL_DEPTH / 2.0
                and abs(y - gy) <= self.goal_half_width_m)

    def goal_distance(self, x: float, y: float) -> float:
        """Distance from ``(x, y)`` to the goal rectangle, 0 inside it.

        Reported per episode so a near-miss timeout is distinguishable from a robot that never
        left the spawn.  The rectangle's own metric, not the centre's, or a robot stopped at the
        band's edge would read as half a lane away.

        Args:
            x: Root world x (m).
            y: Root world y (m).

        Returns:
            Euclidean distance to the nearest point of the rectangle (m).
        """
        gx, gy = self.target_xy
        dx = max(0.0, abs(x - gx) - GOAL_DEPTH / 2.0)
        dy = max(0.0, abs(y - gy) - self.goal_half_width_m)
        return float(math.hypot(dx, dy))

    def traversed(self, max_base_z: float, min_base_z: float) -> bool:
        """Did the root actually go over (or down) the course, rather than around it?

        Args:
            max_base_z: Highest pelvis z reached during the episode (m).
            min_base_z: Lowest pelvis z reached during the episode (m).

        Returns:
            True when the gate is disabled, or when the root moved at least
            :data:`TRAVERSAL_FRAC` of the flight height in the course's direction.
        """
        if self.flight_height_m == 0.0:
            return True
        gate = TRAVERSAL_FRAC * abs(self.flight_height_m)
        if self.flight_height_m > 0:
            return bool(max_base_z >= SPAWN_BASE_Z + gate)
        return bool(min_base_z <= SPAWN_BASE_Z - gate)


def width_note(width: float) -> str:
    """Whether this flight width sits in a trained band, and which one.

    Duplicated from ``stair_probe.width_note`` rather than imported: that module is a CLI
    whose import pulls in the whole probe, and the bands themselves live in
    ``generate_stairs_scene`` and are read from there by both.

    Args:
        width: Flight width (m).

    Returns:
        A one-line verdict naming the band, or saying there is none.
    """
    kinds = ("narrow transition", "widened transition", "pure")
    for kind, (lo, hi) in zip(kinds, gss.TRAINED_STAIR_WIDTHS):
        if lo <= width <= hi:
            return f"in the trained {kind} band {lo}-{hi} m"
    return f"in the HOLE between trained bands {gss.TRAINED_STAIR_WIDTHS} m -- deliberate"


def step_note(rise: float, run: float, ascending: bool) -> str:
    """Whether this step geometry is one the library trained, for this direction.

    Reported per DIRECTION because the trained tread sets differ between them --
    ``TRAINED_TREADS_UP`` carries 0.209-0.28 and the descending set 0.178-0.25 -- so one
    authored staircase can be in-distribution walked up and out of it walked down.

    Args:
        rise: Step height (m).
        run: Tread depth (m).
        ascending: True for the climbing block, False for the descending one.

    Returns:
        A one-line verdict naming which of the two parameters is off-grid, if either.
    """
    treads = gss.TRAINED_TREADS_UP if ascending else (0.178, 0.19, 0.22, 0.25)
    near = lambda v, grid: any(abs(v - g) < 1e-3 for g in grid)  # noqa: E731
    bad = []
    if not near(rise, gss.TRAINED_RISERS):
        bad.append(f"riser {rise:.3f} m is outside the trained {gss.TRAINED_RISERS}")
    if not near(run, treads):
        bad.append(f"tread {run:.3f} m is outside the trained {treads}")
    if not bad:
        return f"fully trained for this direction; steepness {rise / run:.3f}"
    return "; ".join(bad) + f" -- steepness {rise / run:.3f}"


def height_note(height: float) -> str:
    """Whether this box height is one the training library actually carried.

    Args:
        height: Box top height (m).

    Returns:
        A one-line verdict against :data:`generate_stairs_scene.TRAINED_BOX_HEIGHTS`.
    """
    grid = gss.TRAINED_BOX_HEIGHTS
    if any(abs(height - h) < 1e-6 for h in grid):
        return f"an exact trained height (grid {grid[0]}-{grid[-1]} m)"
    if grid[0] <= height <= grid[-1]:
        return f"INSIDE the trained range {grid[0]}-{grid[-1]} m but on no trained tile"
    return f"OUTSIDE the trained range {grid[0]}-{grid[-1]} m -- deliberate"


def _slab(name: str, x_lo: float, x_hi: float, y: float, width: float,
          z_bottom: float, z_top: float) -> str:
    """One axis-aligned group-2 box geom, from its extents rather than its centre.

    MuJoCo boxes are centre + HALF extents; a course is naturally written as the interval it
    occupies.  Converting in one place is what stops a half/full extent slip, which is
    shape-silent and puts a step edge half a tread from where the target was computed.

    Args:
        name: Geom name.
        x_lo: Near face, world x (m).
        x_hi: Far face, world x (m).
        y: Lateral centre (m).
        width: Full lateral extent (m).
        z_bottom: Bottom face, world z (m).
        z_top: Top face, world z (m).

    Returns:
        The formatted ``<geom .../>`` line.
    """
    return (
        f'        <geom name="{name}" type="box" '
        f'size="{(x_hi - x_lo) / 2.0:.4f} {width / 2.0:.4f} {(z_top - z_bottom) / 2.0:.4f}" '
        f'pos="{(x_lo + x_hi) / 2.0:.4f} {y:.4f} {(z_bottom + z_top) / 2.0:.4f}" '
        f'group="2" material="box_material"/>'
    )


def _stairs_up(variant: int, width: float, rise: float, run: float, label: str, steps: int,
               platform_depth: float) -> Course:
    """An ascending flight onto a platform, with nothing past it.

    Built through :func:`generate_stairs_scene._lane_geoms` with ``descend=False`` so the
    tread geometry is emitted by the same code the in-distribution playground uses and cannot
    drift from it.

    Args:
        variant: Index within the stairs-up class.
        width: Lateral flight width (m) -- one of the two swept factors.
        rise: Step height (m) -- half of the other swept factor.
        run: Tread depth (m) -- the other half.
        label: Short name for the (rise, run) level, used in the scene filename.
        steps: Number of risers.
        platform_depth: Flat platform depth past the last tread (m).

    Returns:
        The assembled :class:`Course`.
    """
    lane = gss.Lane(name="flight", y=0.0, width=width, steps=steps)
    geoms = gss._lane_geoms(lane, rise, run, X_START, platform_depth, descend=False)
    if STAIR_WALLS:
        geoms += gss.lane_wall_geoms(lane, rise, run, X_START, platform_depth, descend=False,
                                     gap=STAIR_WALL_GAP, height=STAIR_WALL_HEIGHT,
                                     lead=STAIR_WALL_LEAD,
                                     thickness=STAIR_WALL_THICKNESS)
    top_x = X_START + steps * run
    target_x = top_x + min(TARGET_LEAD, platform_depth - GOAL_DEPTH / 2.0)
    return Course(
        name=f"mjeval_up_w{int(round(width * 100)):03d}_{label}",
        klass="stairs_up", variant=variant, geoms=tuple(geoms), floor_z=0.0,
        target_xy=(target_x, 0.0), course_length_m=target_x,
        flight_height_m=steps * rise, obstacle_x_m=X_START,
        speeds=STAIR_SPEEDS,
        note=(f"ascending {label} flight ({rise:.3f}/{run:.3f} m), width {width:.2f} m "
              f"-- {width_note(width)}; {step_note(rise, run, ascending=True)}"),
        params={"width_m": width, "rise_m": rise, "run_m": run, "step_label": label,
                "steepness": round(rise / run, 4), "steps": steps,
                "platform_depth_m": platform_depth,
                "walls": STAIR_WALLS, "wall_gap_m": STAIR_WALL_GAP if STAIR_WALLS else None},
    )


def _descend_wall_geoms(width: float, run: float, steps: int, floor_z: float,
                        approach: float) -> list[str]:
    """Stairwell walls for a DESCENDING course, which the ascending emitter cannot make.

    ``generate_stairs_scene.lane_wall_geoms`` builds walls standing up from ``z = 0``, which is
    right when the terrain rises from there.  A descending course is dug BELOW the spawn plane,
    so its walls have to reach down to the bottom floor instead.

    Each side is emitted as TWO boxes rather than one: over the raised approach slab the wall
    stands on the slab (``z`` from 0), and past the first riser it drops to the bottom floor.
    One box spanning the whole length would have to start at ``floor_z`` throughout, which would
    bury it inside the approach slab -- the slab is ``width + 2.0`` wide, deliberately, so that
    the robot does not spawn within a stride of a drop-off, and it is therefore wider than the
    flight the walls flank.

    Args:
        width: Flight width (m); the wall faces sit at ``+/- width / 2``.
        run: Tread depth (m).
        steps: Riser count.
        floor_z: World z of the bottom floor (m), negative.
        approach: How far behind the spawn the top slab extends (m).

    Returns:
        MuJoCo geom lines for the four boxes.
    """
    off = width / 2.0 + STAIR_WALL_THICKNESS / 2.0
    x_lead0, x_lead1 = X_START - STAIR_WALL_LEAD, X_START
    x_desc1 = X_START + (steps - 1) * run + DESCENT_WALL_RUNOUT
    out = [f'        <!-- descending stairwell walls, {STAIR_WALL_GAP} m clearance, visual only -->']
    for side, sign in (("left", 1.0), ("right", -1.0)):
        y = sign * off
        # Over the slab: stands on it. Past the first riser: down to the bottom floor.
        for tag, x0, x1, z0 in (("lead", x_lead0, x_lead1, 0.0),
                                ("desc", x_lead1, x_desc1, floor_z)):
            out.append(
                f'        <geom name="dnwall_{side}_{tag}" type="box" material="box_material" '
                f'{gss.DECOR_GROUP_ATTRS} '
                f'size="{(x1 - x0) / 2.0:.4f} {STAIR_WALL_THICKNESS / 2.0:.4f} '
                f'{(STAIR_WALL_HEIGHT - z0) / 2.0:.4f}" '
                f'pos="{(x0 + x1) / 2.0:.4f} {y:.4f} {(z0 + STAIR_WALL_HEIGHT) / 2.0:.4f}"/>')
    return out


def _stairs_down(variant: int, width: float, rise: float, run: float, label: str,
                 steps: int, approach: float = 4.0) -> Course:
    """A descending flight, dug below the spawn plane.

    The robot stands on a raised approach slab whose TOP is ``z = 0`` -- the height the
    keyframe and ``--spawn_z_offset`` are both written against -- walks off it, and descends
    to a ground plane at ``-steps * rise``.  Emitted here rather than through ``_lane_geoms``
    because that builds pillars up from ``z = 0``, and these hang down from it.

    The slab is deliberately WIDER than the flight (``width + 2.0``) and starts well behind
    the spawn: a slab exactly as wide as the treads would put a drop-off within a stride of
    the robot's start, which is a different experiment.

    Args:
        variant: Index within the stairs-down class.
        width: Lateral flight width (m) -- one of the two swept factors.
        rise: Step height (m) -- half of the other swept factor.
        run: Tread depth (m) -- the other half.
        label: Short name for the (rise, run) level, used in the scene filename.
        steps: Number of risers.
        approach: How far behind the spawn the top slab extends (m).

    Returns:
        The assembled :class:`Course`.
    """
    drop = steps * rise
    floor_z = -drop
    geoms = [f'        <!-- descending flight: {steps} down, width {width} m, '
             f'drop {drop:.2f} m; floor at {floor_z:.2f} -->']
    # The top slab. Its far face IS the first riser.
    geoms.append(_slab("top_platform", -approach, X_START, 0.0, width + 2.0, floor_z, 0.0))
    # steps-1 pillars; the last riser lands on the floor plane itself, so both this and
    # `_lane_geoms`' descent have exactly `steps` risers.
    for m in range(1, steps):
        geoms.append(_slab(f"down{m}", X_START + (m - 1) * run, X_START + m * run,
                           0.0, width, floor_z, -m * rise))
    if STAIR_WALLS:
        geoms += _descend_wall_geoms(width, run, steps, floor_z, approach)
    bottom_x = X_START + (steps - 1) * run
    target_x = bottom_x + TARGET_LEAD + DESCENT_RUNOUT
    return Course(
        name=f"mjeval_dn_w{int(round(width * 100)):03d}_{label}",
        klass="stairs_down", variant=variant, geoms=tuple(geoms), floor_z=floor_z,
        target_xy=(target_x, 0.0), course_length_m=target_x,
        flight_height_m=-drop, obstacle_x_m=X_START,
        speeds=STAIR_SPEEDS,
        note=(f"descending {label} flight ({rise:.3f}/{run:.3f} m), width {width:.2f} m "
              f"-- {width_note(width)}; {step_note(rise, run, ascending=False)}"),
        params={"width_m": width, "rise_m": rise, "run_m": run, "step_label": label,
                "steepness": round(rise / run, 4), "steps": steps, "drop_m": drop,
                "walls": STAIR_WALLS, "wall_gap_m": STAIR_WALL_GAP if STAIR_WALLS else None,
                "bottom_x_m": round(bottom_x, 4), "runout_m": DESCENT_RUNOUT},
    )


def _box_course(variant: int, height: float, depth: float, width: float,
                n_boxes: int = 1, gap: float = 3.0) -> Course:
    """One or more free-standing boxes: jump on, walk across the top, jump off FORWARD.

    The dismount is always forward.  A backwards one is the ``jump_off`` family, which was
    dropped from the fine-tuned arm's generator training set -- scoring it would measure a
    deliberate omission rather than the policy.

    ``depth`` is the one deliberate out-of-distribution move in the catalogue and it is here
    because the alternative is worse: every trained box is 0.46-0.48 m deep along travel,
    which is under one stride, so a robot never stands on top and "jump up then down" is a
    single ballistic event rather than the two the spec asks for.  The value is recorded in
    ``params`` so the deviation is in the manifest rather than in this comment alone.

    Args:
        variant: Index within the box class.
        height: Box top height (m) -- the swept parameter.
        depth: Extent along travel (m).
        width: Lateral extent (m).
        n_boxes: How many boxes in a row.
        gap: Flat ground between consecutive boxes (m).

    Returns:
        The assembled :class:`Course`.
    """
    geoms = [f'        <!-- {n_boxes} box(es) {depth}x{width}x{height} m, {gap} m apart -->']
    x = X_START
    for i in range(n_boxes):
        geoms.append(_slab(f"box{i}", x, x + depth, 0.0, width, 0.0, height))
        far = x + depth
        x = far + gap
    target_x = far + TARGET_LEAD
    klass = "two_box_course" if n_boxes > 1 else "box"
    name = (f"mjeval_box2_h{int(round(height * 100)):03d}" if n_boxes > 1
            else f"mjeval_box_h{int(round(height * 100)):03d}")
    return Course(
        name=name, klass=klass, variant=variant, geoms=tuple(geoms), floor_z=0.0,
        target_xy=(target_x, 0.0), course_length_m=target_x,
        flight_height_m=height, obstacle_x_m=X_START,
        speeds=BOX_SPEEDS,
        note=(f"{n_boxes} box(es) {height:.2f} m high, {depth:.2f} m deep "
              f"({height_note(height)}; depth is OOD -- trained {gss.TRAINED_BOX_DEPTH} m)"),
        params={"height_m": height, "depth_m": depth, "width_m": width,
                "n_boxes": n_boxes, "gap_m": gap,
                "depth_is_ood": depth > 0.6},
    )


def _up_into_down(variant: int, width: float, rise: float, run: float, steps: int,
                  platform_depth: float) -> Course:
    """Up a flight, across a platform, down the mirrored flight to the floor.

    ``platform_depth`` is the swept parameter and the two authored values are a designed
    contrast, not two arbitrary geometries: the trained transition tiles carry platforms of
    6.2-10.1 m (p50 8.8), so 8.0 m is the in-distribution control and 2.0 m is a platform
    short enough to present as the top of a box rather than as a landing.

    Args:
        variant: Index within the class.
        width: Lateral flight width (m).
        rise: Step height (m).
        run: Tread depth (m).
        steps: Risers per flight.
        platform_depth: Flat platform between the flights (m) -- the swept parameter.

    Returns:
        The assembled :class:`Course`.
    """
    lane = gss.Lane(name="flight", y=0.0, width=width, steps=steps)
    geoms = gss._lane_geoms(lane, rise, run, X_START, platform_depth, descend=True)
    if STAIR_WALLS:
        geoms += gss.lane_wall_geoms(lane, rise, run, X_START, platform_depth, descend=True,
                                     gap=STAIR_WALL_GAP, height=STAIR_WALL_HEIGHT,
                                     lead=STAIR_WALL_LEAD,
                                     thickness=STAIR_WALL_THICKNESS)
    x_desc = X_START + steps * run + platform_depth
    bottom_x = x_desc + (steps - 1) * run
    target_x = bottom_x + TARGET_LEAD
    return Course(
        name=f"mjeval_updn_p{int(round(platform_depth * 10)):03d}",
        klass="up_into_down", variant=variant, geoms=tuple(geoms), floor_z=0.0,
        target_xy=(target_x, 0.0), course_length_m=target_x,
        flight_height_m=steps * rise, obstacle_x_m=X_START,
        speeds=STAIR_SPEEDS,
        note=(f"up {steps} then down {steps}, platform {platform_depth:.1f} m "
              f"({'in-distribution' if platform_depth >= 6.0 else 'SHORT -- reads as a box top'})"),
        params={"width_m": width, "rise_m": rise, "run_m": run, "steps": steps,
                "platform_depth_m": platform_depth},
    )


# --- the catalogue ---------------------------------------------------------------------
#
# Every number below is a MEASURED trained value or a deliberate one-axis move off it, and
# every source constant lives in `generate_stairs_scene` rather than being restated here.
#
# The stair block is a 2 x 3 CROSSED design -- two widths by three step geometries -- run in
# both directions on the same six geometries.  Crossed rather than one-at-a-time because the
# two axes are not independent in the failure this eval is chasing: a narrow flight presents a
# first riser that looks like the face of a free-standing box, and a steep one presents a
# taller riser to look at, so "narrow AND steep" is the corner that has to be sampled rather
# than inferred from two marginals.
#
#: Flight widths (m), a TWO-level factor.  The trained library has three width clusters --
#: 0.809-1.102, 1.723-4.141 and 5.00-5.29 -- with a 0.621 m hole between the first two.
#: 1.22 m is 4 ft, the narrowest real staircase found anywhere, and it lands INSIDE that hole;
#: 1.96 m is the decorated transition median and is squarely in the trained band.  So the two
#: levels are "narrowest real, and unrepresented" against "the middle of what was trained",
#: which is the sharpest two-point contrast the width axis offers.
EVAL_STAIR_WIDTHS: tuple[float, ...] = (1.22, 1.96)

#: Step geometry, a THREE-level factor crossed with width: ``(rise, run, label)``.
#:
#: Trained values, measured off the same library:  risers ``(0.13, 0.131, 0.15, 0.17)``;
#: treads ``(0.209, 0.22, 0.221, 0.25, 0.28)`` ascending and ``(0.178, 0.19, 0.22, 0.25)``
#: descending.  Steepness (rise/run) therefore spans 0.464 to 0.813 in training.
#:
#: **The library tops out below a real staircase.**  Its steepest riser is 0.17 m = 6.7 in,
#: while a code-typical commercial stair is 7 in / 11 in (0.178 / 0.279) and residential codes
#: permit steeper still (up to 7.75 in rise on a 10 in run).  So the third level below is not
#: a synthetic extrapolation -- it is the ordinary staircase the robot will be asked to climb,
#: and its riser is outside every trained value.  That is the finding this axis exists to test.
#:
#: **The same three are used ASCENDING and DESCENDING**, so direction is a clean factor.  Two
#: of the three treads sit outside the descending trained set, and that is deliberate rather
#: than overlooked: a real staircase has ONE geometry and is walked both ways, so the library's
#: up/down tread asymmetry is a property of the generator and not of the world.  Where a level
#: is off-distribution for a direction, :func:`step_note` says so and the manifest records it.
EVAL_STEP_GEOMETRY: tuple[tuple[float, float, str], ...] = (
    (0.150, 0.250, "modal"),        # steepness 0.600 -- trained riser mode, the only tread in
                                    # BOTH trained sets.  The in-distribution control.
    (0.170, 0.209, "steepest"),     # steepness 0.813 -- max trained riser on min trained
                                    # ascending tread: the hardest fully in-distribution stair.
    (0.178, 0.280, "code_7x11"),    # steepness 0.636 -- a real 7 in / 11 in commercial stair.
                                    # The tread is authored at the TRAINED 0.280 rather than at
                                    # 11 in exactly (0.2794): 0.6 mm is the same tread, and
                                    # snapping it leaves the RISER as the single off-grid axis,
                                    # which is what makes this level readable. 0.178 m is above
                                    # every trained riser, so an ordinary staircase is steeper
                                    # in rise than anything the library contains.
)

#: Box top heights (m).  Trained heights are a 0.05 grid from 0.25 to 0.75 with 0.50 MISSING,
#: so 0.50 is an interpolation probe -- inside the range, on no trained tile -- while 0.25 and
#: 0.75 are the exact endpoints.
EVAL_BOX_HEIGHTS: tuple[float, ...] = (0.25, 0.40, 0.50, 0.65, 0.75)

#: Extent along travel (m) for the eval boxes.  See :func:`_box_course` -- deliberately far
#: past the trained 0.46-0.48 so there is a top to walk across.
EVAL_BOX_DEPTH: float = 1.20

#: Platform depths (m) for the two up-into-down courses.
EVAL_PLATFORM_DEPTHS: tuple[float, ...] = (2.0, 8.0)

#: Risers per flight.  Trained flights run 3..39; six is the shortest that is
#: unambiguously a staircase rather than a kerb, and it keeps the 0.6 m/s trials short.
_STEPS = 6


def _widest_feature_m(course: Course) -> float:
    """The largest lateral extent among a course's terrain boxes (m).

    Parsed back out of the emitted geom lines rather than threaded through every builder: the
    geoms are the authority on what was actually written, and a builder that forgot to pass its
    width would otherwise produce a goal narrower than its own terrain.

    Args:
        course: The course, before its goal width is set.

    Returns:
        The widest full y-extent, or 0.0 when the course has no boxes.
    """
    widest = 0.0
    for line in course.geoms:
        if 'type="box"' not in line:
            continue
        size = line.split('size="', 1)[1].split('"', 1)[0].split()
        widest = max(widest, 2.0 * float(size[1]))
    return widest


def _with_goal_width(course: Course) -> Course:
    """Return ``course`` with its goal band widened to clear its own terrain.

    Args:
        course: A freshly built course.

    Returns:
        The same course with :attr:`Course.goal_half_width_m` resolved.
    """
    half = max(GOAL_HALF_WIDTH_FLOOR,
               _widest_feature_m(course) / 2.0 + GOAL_WIDTH_MARGIN)
    return replace(course, goal_half_width_m=round(half, 3))


def catalogue() -> tuple[Course, ...]:
    """The 18 authored courses, in a stable order that indexes the ``terrain_idx`` column.

    Returns:
        Six ascending flights (2 widths x 3 step geometries), six descending on the SAME six
        geometries, five single boxes, one two-box course and two up-into-down platforms --
        twenty courses.
    """
    out: list[Course] = []
    i = 0
    for w in EVAL_STAIR_WIDTHS:
        for rise, run, label in EVAL_STEP_GEOMETRY:
            out.append(_stairs_up(i, w, rise, run, label, _STEPS, platform_depth=3.0))
            i += 1
    i = 0
    for w in EVAL_STAIR_WIDTHS:
        for rise, run, label in EVAL_STEP_GEOMETRY:
            out.append(_stairs_down(i, w, rise, run, label, _STEPS))
            i += 1
    for i, h in enumerate(EVAL_BOX_HEIGHTS):
        out.append(_box_course(i, h, EVAL_BOX_DEPTH, gss.TRAINED_BOX_WIDTH))
    out.append(_box_course(0, 0.40, EVAL_BOX_DEPTH, gss.TRAINED_BOX_WIDTH,
                           n_boxes=2, gap=3.0))
    for i, p in enumerate(EVAL_PLATFORM_DEPTHS):
        out.append(_up_into_down(i, 1.96, 0.150, 0.250, _STEPS, platform_depth=p))
    return tuple(_with_goal_width(c) for c in out)


def scene_path(course: Course, robot_name: str = "g1_perceptive") -> str:
    """Where :func:`write_scene` puts this course's XML.

    ``Robot._get_model_data`` resolves ``<robot>_<scene>.xml`` against ``transfer/sim`` --
    that is the only directory it looks in -- so scenes must land there and cannot live in
    the run directory.  The run directory gets copies, for provenance.

    Args:
        course: The course.
        robot_name: Robot model prefix.

    Returns:
        Absolute path to the scene XML.
    """
    return os.path.join(os.path.dirname(_HERE), "robots", "g1",
                        f"{robot_name}_{course.name}.xml")


def write_scene(course: Course, robot_name: str = "g1_perceptive") -> str:
    """Emit one course's scene XML and return its path.

    Args:
        course: The course to emit.
        robot_name: Robot model to ``<include>``.

    Returns:
        The path written.
    """
    floor = ('        <geom name="floor" size="0 0 0.05" type="plane" '
             f'material="groundplane" pos="0 0 {course.floor_z:.4f}" group="2"/>')
    header = (
        f"Generated by mjeval/terrains.py -- an OUT-OF-DISTRIBUTION eval course, not a\n"
        f"         training tile.  class={course.klass} variant={course.variant}\n"
        f"         {course.note}\n"
        f"         target ({course.target_xy[0]:.2f}, {course.target_xy[1]:.2f}) r="
        f"half-width {course.goal_half_width_m:.2f} m x depth {GOAL_DEPTH} m; "
        f"budget {course.budget_s:.1f} s; floor z={course.floor_z:.2f}"
    )
    xml = gss.scene_xml(f"{robot_name}.xml", header, "\n".join(course.geoms),
                        floor_geom=floor, model=course.name)
    path = scene_path(course, robot_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(xml)
    return path


def manifest_rows() -> list[dict]:
    """The catalogue as plain dicts, for the run manifest.

    Returns:
        One dict per course carrying its identity, geometry parameters, target, budget and
        speed ladder -- everything needed to read a result without re-running this module.
    """
    rows = []
    for idx, c in enumerate(catalogue()):
        rows.append({
            "terrain_idx": idx, "name": c.name, "class": c.klass,
            "class_code": c.class_code, "variant": c.variant,
            "target_xy": list(c.target_xy),
            "goal_half_width_m": round(c.goal_half_width_m, 3),
            "goal_depth_m": GOAL_DEPTH,
            "course_length_m": round(c.course_length_m, 4),
            "flight_height_m": round(c.flight_height_m, 4),
            "obstacle_x_m": c.obstacle_x_m, "floor_z": c.floor_z,
            "fall_datum_z": c.fall_datum_z,
            "budget_s": round(c.budget_s, 3), "speeds": list(c.speeds),
            "note": c.note, "params": c.params,
        })
    return rows


#: Manifest family -> course class.  An explicit table rather than a substring scan, which is
#: what this used to be and which was wrong three ways at once: ``family_jump_on.h5`` contains
#: no "box" so all 20 of those tiles raised, ``family_stairs_dn_pure.h5`` contains no
#: "stairs_down" so all 15 of those raised -- while ``family_stairs_up_pure.h5`` DID resolve,
#: purely because "stairs_up" happens to be a substring of it.  That asymmetry is the tell: a
#: scan that silently succeeds on one "_pure" family and refuses its mirror is matching
#: spelling, not meaning.
#:
#: The excluded families are absent on purpose rather than mapped to something plausible:
#: ``jump_off`` / ``jump_off_front`` jump BACKWARDS or dismount, which no forward-facing camera
#: sees and which :mod:`goal_anchor` refuses anyway, and ``distractor`` is skipped from FM
#: training by default.  A tile from one of those names itself in the error.
FAMILY_KLASS = {
    "stairs_up": "stairs_up",
    "stairs_up_pure": "stairs_up",
    "stairs_down": "stairs_down",
    "stairs_dn_pure": "stairs_down",
    "jump_on": "box",
}


def klass_for_family(family: str, terrain_id: int | None = None) -> str:
    """Resolve a manifest family string to a :data:`CLASS_CODES` key.

    Args:
        family: The manifest's own ``family`` entry, e.g. ``"family_jump_on.h5"``.
        terrain_id: Tile id, for the error message only.

    Returns:
        The course class.

    Raises:
        ValueError: If the family maps to no class.  Excluded families are named as excluded
            rather than reported as unknown, since those two need different responses.
    """
    stem = family.removeprefix("family_").removesuffix(".h5")
    if stem in FAMILY_KLASS:
        return FAMILY_KLASS[stem]
    where = "" if terrain_id is None else f"tile {terrain_id} "
    if stem in ("jump_off", "jump_off_front", "distractor"):
        raise ValueError(
            f"{where}is family {stem!r}, which is deliberately EXCLUDED from course building: "
            "jump_off/jump_off_front move backwards or dismount (unobservable to a "
            "forward-facing camera, and refused by mjeval.goal_anchor), and distractor is "
            "skipped from FM training by default. Pass klass= explicitly to override.")
    raise ValueError(
        f"{where}family {family!r} maps to no CLASS_CODES key. Known families: "
        f"{sorted(FAMILY_KLASS)}. Add it to FAMILY_KLASS or pass klass= explicitly.")


def isaac_clip_course(clip_path: str, clip_id: int, *, max_offset: int = 50,
                      name: str | None = None, klass: str | None = None) -> Course:
    """A :class:`Course` that IS one CLIP -- its terrain, start, goal, duration and command.

    :func:`isaac_tile_course` builds one course per TILE, which forces a reduction of the
    tile's ~50 clips to a single number per field.  Any two fields reduced differently imply a
    trajectory no clip ever ran: a ``max`` duration beside a ``median`` endpoint understates
    the pace, and the resulting configuration measured 0 successes in 6 reference episodes on
    tile 6 -- an infeasible course rather than a policy result.  Isaac has no such problem
    because it runs one clip per episode.  Neither does this.

    Every number here comes from the SAME clip:

    * **terrain** -- the clip's tile, rebuilt from the manifest exactly as before;
    * **origin** -- the clip's own start ``(x, y)`` and the walkable surface under it, so the
      robot begins where the reference begins rather than a fixed ``approach`` distance out;
    * **goal** -- the clip's terminal pelvis position.  NOT minus an approach margin:
      ``in_goal`` already awards success within ``GOAL_DEPTH / 2`` = 1.0 m, which is exactly
      Isaac's ``approach_margin_m``, and subtracting it here too would put the line a full
      metre nearer than Isaac's while still reporting as matched;
    * **budget** -- ``(clip_len - 1 - max_offset) * dt``, the ``clip_end`` window;
    * **command** -- ``target_vel``, the signal Isaac's teacher was actually driven by.  A pace
      inferred as distance/duration is NOT this and is badly wrong (0.86 m/s inferred against
      a 1.60 m/s real command on tile 6), because the clip climbs and turns and its mean ground
      speed is far below what it was told to do.

    The caller still owes the SPAWN, because building it needs the policy's joint order and
    this module has no policy.  ``params["clip_start_*"]`` carries everything for it, and
    ``params["clip_joint_names"]`` carries the order those angles are in -- which is the
    ROBOT's per-limb order and differs from the policy's interleaved order in 27 of 29
    positions, so a consumer must convert by NAME rather than assume.

    Args:
        clip_path: Clip library ``.h5``.
        clip_id: Which clip.
        max_offset: The tracker's furthest reference-window offset.
        name: Scene name; defaults to ``clip<id>``.
        klass: One of :data:`CLASS_CODES`; inferred from the tile's family when omitted.

    Returns:
        The course.

    Raises:
        ValueError: If the clip's tile has no walkable surface under the clip's start, which
            would put the robot in the air or inside geometry.
    """
    import generate_clip_tile_scene as gcts  # noqa: PLC0415

    import goal_anchor as _ga  # noqa: PLC0415

    ep = _ga.clip_episode(clip_path, clip_id, max_offset=max_offset)
    entry, _info = gcts.load_tile(clip_path, ep.terrain_id)
    overlays = gcts.emittable_overlays(entry)

    # The scene origin is the CLIP's start, not a derived approach point.
    sx, sy = float(ep.start_pos[0]), float(ep.start_pos[1])
    sz = gcts.terrain_top_at(gcts.walkable(overlays), sx, sy)
    if sz is None:
        raise ValueError(
            f"clip {clip_id} starts at ({sx:.2f}, {sy:.2f}) on tile {ep.terrain_id}, where "
            "the tile has no walkable surface -- the robot would spawn in the air or inside "
            "geometry. This clip cannot be run as a course.")
    lines, floor = gcts.tile_geoms(entry, overlays, (sx, sy, sz), infinite_floor=True)
    floor_z = float(floor.split('pos="0 0 ')[1].split('"')[0])

    goal = (float(ep.goal_xyz[0]) - sx, float(ep.goal_xyz[1]) - sy,
            float(ep.goal_xyz[2]) - sz)
    if klass is None:
        klass = klass_for_family(str(entry.get("family", "")), ep.terrain_id)

    step = gcts.first_step(overlays)
    obstacle_x = (float(step["pos_xy"][0]) - float(step["size_xyz"][0]) / 2.0 - sx
                  if step is not None else 0.0)

    return Course(
        name=name or f"clip{int(clip_id)}",
        klass=klass,
        variant=int(ep.terrain_id),
        geoms=tuple(lines),
        floor_z=floor_z,
        target_xy=(goal[0], goal[1]),
        course_length_m=float(math.hypot(goal[0], goal[1])),
        # The pelvis CLIMB, not the goal's height above the spawn ground. `traversed` tests
        # `max_base_z >= SPAWN_BASE_Z + TRAVERSAL_FRAC * flight_height_m`, i.e. it measures a
        # RISE from the spawn pelvis -- so passing the goal's absolute height here inflates the
        # gate by one pelvis height (1.612 against the true 0.826 on clip 300) and no episode
        # can ever be marked traversed. That fails as a TIMEOUT after the robot has already
        # walked through the goal band, which reads as a locomotion result.
        flight_height_m=float(goal[2] - (float(ep.start_pos[2]) - sz)),
        obstacle_x_m=float(obstacle_x),
        speeds=(float(ep.command[0][0]),),
        note=(f"CLIP {clip_id} of tile {ep.terrain_id} ({entry.get('family', '?')}): "
              f"{ep.duration_s:.2f} s at cmd {[round(float(v), 2) for v in ep.command[0]]}, "
              f"{math.hypot(goal[0], goal[1]):.2f} m to the clip's own endpoint. Start, goal, "
              "duration and command all come from THIS clip -- no aggregation over the tile."),
        goal_anchor_xyz=goal,
        goal_anchor_source="clip",
        budget_s_override=float(ep.duration_s) + CMD_HOLD_S,
        params={"clip_path": clip_path, "terrain_id": int(ep.terrain_id),
                "clip_id": int(clip_id), "spawn_tile_xy": [sx, sy], "spawn_tile_z": sz,
                "clip_duration_s": float(ep.duration_s),
                "clip_command": ep.command.tolist(),
                "clip_constant_command": bool(ep.constant_command),
                "clip_start_pos": [ep.start_pos[0] - sx, ep.start_pos[1] - sy,
                                   ep.start_pos[2] - sz],
                "clip_start_quat_wxyz": list(ep.start_quat_wxyz),
                "clip_start_joint_pos": list(ep.start_joint_pos),
                "clip_start_lin_vel": list(ep.start_lin_vel),
                "clip_start_ang_vel": list(ep.start_ang_vel),
                "clip_start_joint_vel": list(ep.start_joint_vel),
                "clip_joint_names": list(ep.joint_names)},
    )


def isaac_tile_course(clip_path: str, terrain_id: int, *, speeds: tuple[float, ...],
                      approach: float = 2.5, name: str | None = None,
                      klass: str | None = None,
                      goal_anchor_max_offset: int | None = None,
                      isaac_matched: bool = False) -> Course:
    """A :class:`Course` that IS one tile of a clip library, geometry for geometry.

    Every other course in this module is deliberately NOT a trained tile -- authored to vary
    one parameter so a failure names a geometry rather than a tile id.  This one is the
    opposite, and it exists for a different question: when you want to know what the
    generator's own sampling variance does, the terrain has to be IN distribution, or the
    variance you measure is confounded with perceptive distribution shift.

    Nothing here re-derives geometry.  ``generate_clip_tile_scene`` reads the library's
    ``terrain_manifest`` -- the artifact the clip generators emit and the RL importer re-reads
    -- so this is the one producer rather than a second opinion, and
    ``test_clip_tile_scene.py`` already pins that scene against the Isaac loader's own tile.

    Args:
        clip_path: Clip library ``.h5`` carrying the ``terrain_manifest``.
        terrain_id: Which tile.  ``None`` is not accepted here: a course must name its tile.
        speeds: The commanded forward speeds to run on it.
        approach: Metres before the first riser to spawn, laterally centred on the flight, so
            driving ``+x`` walks the feature.
        name: Scene name; defaults to ``tile<id>``.
        klass: One of :data:`CLASS_CODES`; inferred from the manifest family when omitted.
        goal_anchor_max_offset: The tracker's furthest reference-window offset (50 for every
            current tracker).  When given, the failure critic's goal anchor is recovered from
            this library and attached; when ``None`` it is not, and the course cannot be
            critic-scored.  Opt-in because recovering it reads one row per clip on the tile,
            which every non-critic caller would pay for nothing.  It is stated rather than
            defaulted because it sets ``last_usable_step`` and a wrong value yields a
            plausible anchor about a metre off.
        isaac_matched: Run this course the way the ISAAC eval runs its episodes -- which
            means moving the time budget AND the goal line together, because moving either
            alone is incoherent:

            * budget becomes ``max(clip_len - 1 - max_offset) * dt + CMD_HOLD_S``, the
              ``clip_end`` window every Isaac episode lived inside and therefore the only
              window the failure critic was fitted on (the default runs 1.4-3.6x longer);
            * the goal moves from ``target_xy`` (the far platform's CENTRE) to the clip's own
              endpoint, which is what Isaac's goal region is centred on.

            **Moving only the budget makes success unreachable.** At 0.8 m/s, tile 6's
            ``target_x`` is 11.3 s away against a 7.5 s Isaac budget, and tile 20's is 9.5 s
            against 8.1 s -- so every episode on both would time out by construction while
            looking like a policy result. Needs ``goal_anchor_max_offset``.

            Off by default: it changes what "success" MEANS on a course, so every mjeval
            number measured before it stays comparable only as long as it stays opt-in.

    Returns:
        The course, with an infinite floor plane so ``floor_z`` means what it means for every
        other course here.

    Raises:
        ValueError: If the tile has no walkable step to aim at, or its family maps to no
            known class -- either would make the traversal gate and the fall datum meaningless.
    """
    import generate_clip_tile_scene as gcts  # noqa: PLC0415

    entry, _info = gcts.load_tile(clip_path, terrain_id)
    overlays = gcts.emittable_overlays(entry)
    sx, sy, _why = gcts.spawn_point(entry, overlays, approach)
    sz = gcts.terrain_top_at(gcts.walkable(overlays), sx, sy)
    lines, floor = gcts.tile_geoms(entry, overlays, (sx, sy, sz), infinite_floor=True)

    step = gcts.first_step(overlays)
    if step is None:
        raise ValueError(f"tile {terrain_id} has no walkable step; it is not a feature course")
    obstacle_x = float(step["pos_xy"][0]) - float(step["size_xyz"][0]) / 2.0 - sx

    if klass is None:
        klass = klass_for_family(str(entry.get("family", "")), terrain_id)

    # The far platform: walk the walkable tops downrange of the flight and take the farthest
    # one's centre as the goal, so "traversed" means "got off the feature and onto the far
    # side" rather than a distance guessed from the tile size.
    tops = [ov for ov in gcts.walkable(overlays)
            if float(ov["pos_xy"][0]) - sx > obstacle_x]
    if not tops:
        raise ValueError(f"tile {terrain_id} has nothing downrange of its first step")
    far = max(tops, key=lambda ov: float(ov["pos_xy"][0]))
    target_x = float(far["pos_xy"][0]) - sx
    target_z = gcts.terrain_top_at(gcts.walkable(overlays),
                                   float(far["pos_xy"][0]), float(far["pos_xy"][1]))

    # The floor plane's z, read back out of the emitted geom rather than recomputed, so this
    # cannot disagree with the scene it describes.
    floor_z = float(floor.split('pos="0 0 ')[1].split('"')[0])

    # The critic's goal anchor, if asked for. Tile-local -> this course's world frame is the
    # same subtraction `tile_geoms` applied to the geometry, so the anchor lands in the frame
    # the robot actually walks in.
    anchor_xyz, anchor_src = None, "none"
    if goal_anchor_max_offset is not None:
        import goal_anchor as _ga  # noqa: PLC0415

        ta = _ga.tile_anchors(clip_path, max_offset=int(goal_anchor_max_offset),
                              terrain_ids=[int(terrain_id)])[int(terrain_id)]
        anchor_xyz = (ta.anchor[0] - sx, ta.anchor[1] - sy, ta.anchor[2] - sz)
        anchor_src = "clip"

    budget_override = None
    if isaac_matched:
        if goal_anchor_max_offset is None:
            raise ValueError("isaac_matched needs goal_anchor_max_offset: the Isaac episode "
                             "length IS clip_len - 1 - max_offset and the goal IS the clip's "
                             "endpoint at that step, so the offset is the quantity being "
                             "matched and guessing it defeats the point.")
        import goal_anchor as _ga2  # noqa: PLC0415
        budget_override = _ga2.tile_episode_seconds(
            clip_path, max_offset=int(goal_anchor_max_offset),
            terrain_ids=[int(terrain_id)])[int(terrain_id)] + CMD_HOLD_S
        # The goal moves WITH the budget. See the arg docstring: separately they are
        # incoherent, and the incoherence presents as a policy result rather than an error.
        # The goal line IS the anchor, and the approach margin is already there: Isaac awards
        # success on a half-plane at `anchor_x - approach_margin_m` (1.0 m for every family in
        # the fitted goal_regions.json), and `in_goal` awards it at `|x - target_x| <=
        # GOAL_DEPTH / 2`, which is also 1.0 m. Subtracting the margin HERE as well would
        # double-count it and put the success line a full metre nearer than Isaac's -- a
        # silently easier course that still reports as matched.
        #
        # The two shapes differ (Isaac a half-plane, mjeval a band), but `run_episode` breaks
        # on the first `in_goal`, so a robot can never leave the band by overshooting it.
        target_x = float(anchor_xyz[0])
        target_z = float(anchor_xyz[2]) + sz
        # The PACE has to move too, and for the same reason the goal did. In Isaac the clip
        # drives the robot, so it covers its own distance in its own time by construction. In
        # MuJoCo a constant command sets the pace, and the shipped 0.8 m/s is SLOWER than the
        # clip on tile 6 (0.86 m/s) -- so timing that command against the clip's budget leaves
        # -0.0 s of slack and manufactures timeouts that read as policy failures. Exposed
        # rather than forced: `speeds` is the caller's, and a speed SWEEP against a fixed
        # budget is still a legitimate experiment, just not an Isaac-matched one.
        # From the ANCHOR, not the margin-adjusted goal: this is what pace the CLIP
        # actually moved at, and dividing the shortened distance by the full duration
        # would under-report it by the margin (0.86 -> 0.72 m/s on tile 6).
        clip_pace = float(anchor_xyz[0]) / max(budget_override - CMD_HOLD_S, 1e-6)
    return Course(
        name=name or f"tile{terrain_id}",
        klass=klass,
        variant=int(terrain_id),
        geoms=tuple(lines),
        floor_z=floor_z,
        target_xy=(target_x, 0.0),
        course_length_m=float(target_x),
        flight_height_m=float(target_z - sz),
        obstacle_x_m=float(obstacle_x),
        speeds=tuple(speeds),
        note=(f"REAL clip-library tile {terrain_id} ({entry.get('family', '?')}), "
              f"{entry['size_x']:.2f} x {entry['size_y']:.2f} m, first riser at "
              f"x={obstacle_x:+.2f} m, net {target_z - sz:+.3f} m over the feature; "
              "in-distribution terrain, so a variance measured here is not confounded with "
              "perceptive distribution shift"),
        goal_anchor_xyz=anchor_xyz,
        goal_anchor_source=anchor_src,
        budget_s_override=budget_override,
        params={"clip_path": clip_path, "terrain_id": int(terrain_id),
                "approach": float(approach), "spawn_tile_xy": [sx, sy], "spawn_tile_z": sz,
                **({"clip_pace_mps": float(clip_pace)} if isaac_matched else {})},
    )
