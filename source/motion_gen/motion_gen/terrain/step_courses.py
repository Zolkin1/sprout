# Copyright 2026 Zachary Olkin. All rights reserved.

"""One FIXED free-form course: mjeval's staircase, walked up and back down.

A sibling of :mod:`~motion_gen.terrain.box2`, and for the same reason: the randomised
families exist to present a DISTRIBUTION, and this exists to present ONE obstacle that can be
overfitted to, evaluated on repeatedly and compared against MuJoCo.  The geometry is therefore
fixed and every constant is stated here rather than drawn.

**UP-INTO-DOWN** -- six risers up, a platform, six risers back down to the floor.  Every
number is mjeval's, so the two halves are the same course and a rate is comparable across the
two simulators; :data:`UPDOWN_RISE_M` and its neighbours each name the mjeval constant they
mirror.  What the course probes is the TRANSITION: the rise, the tread, the width and the
flight length are all in distribution, and only walking up into a descent is not.

**The CURB course was removed on 2026-09-09.**  It was a single wide step, authored here with
no MuJoCo twin, built to ask "how low does a box have to get before it stops being a box".
Nothing was ever fine-tuned on it.  Its geometry, its holdout and its libraries are in the
git history at ``0080cebe``; the constants it shared with box2 (the speed ladder and band)
live on in :mod:`~motion_gen.terrain.box2`.

**Lateral geometry.**  Up-into-down takes the free-form ``stairs`` family's shape -- flight,
wall band, authored ground -- but **not its lateral MARGIN**, and that one number is
load-bearing.  The family adds ``SECTION_MARGIN_M`` a side, which leaves 0.65 m of
ground-level apron beside the flight; ``mdp.freeform_goal_reached`` tests a downrange line and
a lateral band and never a up-into-down the line sits 0.5 m past the bottom riser, which is mjeval's own placement for
this class -- and note mjeval's ``stairs_down`` class adds a 3 m ``DESCENT_RUNOUT`` on top,
precisely because a robot can enter the band while still stumbling off the flight, while
``up_into_down`` does not.  So a success here is "reached the bottom", NOT "recovered
walking".  :data:`UPDOWN_RUNOUT_M` is the knob if that is ever the wrong bargain, and it
carries its own family for the same reason box2's v2 does.
"""

from __future__ import annotations

from motion_gen.terrain.freeform import (
    FREEFORM_SPAWN_JITTER_M,
    GOAL_HALF_WIDTH_FLOOR_M,
    GOAL_INSET_M,
    SECTION_MARGIN_M,
    SPAWN_INSET_M,
    WALL_T_M,
    _base_tile,
    _fit_goal_half_width,
    _ground_overlay,
    _overlay,
    _route,
    spawn_area,
)

# ---------------------------------------------------------------------------------------------
# Shared with box2 and with mjeval
# ---------------------------------------------------------------------------------------------
#: Flat run-up (m) from the spawn to the obstacle, on BOTH courses.  ``mjeval.terrains.X_START``
#: and ``generate_stairs_scene.DEFAULT_X_START``: long enough that the far camera (15 deg down,
#: boresight meeting the ground 4.25 m out) has the obstacle in frame for several re-plans
#: before the robot arrives, short enough that a 0.6 m/s trial is not mostly approach.
APPROACH_M: float = 2.5

#: How far past the far edge of the last walkable geom mjeval puts the goal RECTANGLE's centre
#: (m), ``mjeval.terrains.TARGET_LEAD``.
TARGET_LEAD_M: float = 1.5

#: Depth of that rectangle along travel (m), ``mjeval.terrains.GOAL_DEPTH``.  The RL side tests
#: a LINE, so the line goes at ``lead - depth / 2`` past the far edge: the rectangle's near
#: edge, which is the boundary the two harnesses have to agree on.
GOAL_DEPTH_M: float = 2.0


def _spawn_x() -> float:
    """Tile-local x of the spawn (m).

    Both courses inset the spawn the way every free-form section does, so a distance measured
    from the spawn here is the same quantity mjeval measures from its course origin.

    Returns:
        ``SECTION_MARGIN_M + SPAWN_INSET_M``.
    """
    return SECTION_MARGIN_M + SPAWN_INSET_M


# =============================================================================================
# UP-INTO-DOWN: mjeval's staircase, walked both ways
# =============================================================================================
#: Step rise (m).  ``mjeval.terrains.EVAL_STEP_GEOMETRY[0]``, the "modal" level: the trained
#: riser mode, and the tread below is the only one in BOTH the ascending and the descending
#: trained sets.  This is the in-distribution control of mjeval's step-geometry axis, which is
#: what makes the up-into-down TRANSITION the only thing being probed here.
UPDOWN_RISE_M: float = 0.150

#: Tread depth (m).  See :data:`UPDOWN_RISE_M`; steepness 0.600.
UPDOWN_RUN_M: float = 0.250

#: Risers per flight, ``mjeval.terrains._STEPS``.  Trained flights run 3-39; six is the
#: shortest that is unambiguously a staircase rather than a kerb -- which is exactly the
#: distinction that separates a staircase from a single kerb.
UPDOWN_STEPS: int = 6

#: Flight width (m).  ``mjeval.terrains.EVAL_STAIR_WIDTHS[1]``, the decorated transition
#: median and squarely inside the trained band -- and the only width mjeval builds its
#: up-into-down courses at, so matching it is what makes this the same course.
UPDOWN_WIDTH_M: float = 1.96

#: The platform depths (m) mjeval sweeps, ``mjeval.terrains.EVAL_PLATFORM_DEPTHS``.
UPDOWN_PLATFORM_DEPTHS_M: tuple[float, ...] = (2.0, 8.0)

#: EXTRA flat ground past the bottom riser (m), beyond what the goal line already needs.
#:
#: 0.0 is mjeval's ``up_into_down``, and matching it is the default.  mjeval's ``stairs_down``
#: class adds 3.0 m (``DESCENT_RUNOUT``) for a stated reason -- without it the goal band's near
#: edge is 0.5 m past the last riser, which a robot can enter while still stumbling off the
#: flight, so "got down" and "fell down in roughly the right direction" score the same.  The
#: knob is here so that bargain can be changed; a non-zero value gets its own family, because
#: more distance after the bottom is strictly more chances to fall and the two rates are then
#: not comparable for a reason that has nothing to do with the policy.
UPDOWN_RUNOUT_M: float = 0.0

#: Visual-only stairwell walls flanking the flight, as mjeval has them
#: (``mjeval.terrains.STAIR_WALLS``, ``STAIR_WALL_GAP`` = 0 so the inner face touches the tread
#: edge).  They are ``decor=True`` here, which is what makes them visual-only in Isaac: no
#: collider and invisible to the height scan, while the depth cameras cast against them.
#:
#: **That is deliberately NOT what the free-form ``stairs`` family does** -- it authors
#: COLLIDABLE walls -- and the reason to differ is that this course exists to be compared with
#: MuJoCo, where they are visual-only and, per ``lane_wall_geoms``, not negotiable: a solid
#: wall this close is swept by the reference's own arm swing, which blocks a wrist, fires
#: ``ee_pose_error`` and ends the episode for something no policy could fix.
#:
#: **Rendering them needs ``scene.terrain.render_decor=True``**, which the fine-tune tasks set
#: and the RL teacher does not.  ``render_decor`` is invisible in every artifact
#: (``terrain_fingerprint`` hashes the manifest, so decor-on and decor-off runs export
#: byte-identical metadata), so the manifest record below is the only durable statement that
#: this course HAS walls.
UPDOWN_WALLS: bool = True

#: Ground beside the flight, per side (m) -- the WALL BAND and nothing else.
#:
#: Deliberately NOT the free-form ``stairs`` family's ``WALL_T_M + SECTION_MARGIN_M``; see the
#: module docstring for why that 0.5 m is a route past the staircase on this course and not on
#: that one.  The consequence is neat rather than merely acceptable: the apron IS exactly where
#: the (visual-only) walls stand, so every walkable point outside the flight is under a wall.
#: Whether the stairwell walls are real COLLISION geometry rather than visual-only decor.
#:
#: ``True`` here and ``False`` in mjeval, and the divergence is deliberate rather than an
#: oversight. mjeval's ``STAIR_WALLS`` note calls visual-only "not negotiable" for one specific
#: reason: a solid wall this close to the tread edge is swept by the reference's own arm swing,
#: which blocks a wrist and fires ``ee_pose_error`` -- ending the episode for something no
#: policy could fix. **That termination does not exist on the fine-tune task**, whose only
#: enders are the time-out, the absolute height floor, the tilt cone and the goal, so the
#: failure the flag was protecting against cannot happen here.
#:
#: What making them solid buys: they enter the collider SDF, so the terrain-consistency reward
#: charges a plan that puts a limb inside a wall -- on a 1.96 m corridor that is a real part of
#: planning a staircase and was previously unpenalised. They also become physical, so the
#: apron under them stops being a walkable bypass by GEOMETRY as well as by the goal band, and
#: the privileged height scan sees them.
#:
#: **A tile with solid walls is NOT an mjeval clone**, and :func:`make_updown_tile` withholds
#: the ``mjeval_course`` claim accordingly -- the two are then different physics, so a rate
#: from one cannot be read against a rate from the other.
UPDOWN_WALL_COLLIDABLE: bool = True

UPDOWN_APRON_M: float = WALL_T_M

#: Stairwell wall height (m) and how far before the first riser it begins (m).
#: ``mjeval.terrains.STAIR_WALL_HEIGHT`` / ``STAIR_WALL_LEAD``.  The height is a VIEWER choice
#: above ~0.9 m -- at 3 m the far camera's frame already cuts the wall, so 1.0 m and 3.0 m
#: walls are byte-identical images -- and the lead matters because the approach is when the
#: plan that decides stairs-or-box is made.
UPDOWN_WALL_HEIGHT_M: float = 2.0
UPDOWN_WALL_LEAD_M: float = 1.0

#: Approach distances (m) from the pelvis to the first riser, one tile each -- so an
#: initial-condition grid over distance is a grid over TILES and the reset needs no per-episode
#: override, exactly as in box2.  Numerically box2's grid, restated rather than imported so an
#: edit there cannot move this library's terrain fingerprint, and equal to it on purpose so a
#: cell at 2.5 m is the same starting state on both courses.
#: Clearance kept between a top-platform spawn and the platform's own edges, in metres --
#: the side walls and the two flights alike.
#:
#: **The walls are the reason this is larger than box2's 0.30 m.** They are visual-only, so
#: nothing collides, but they are 2.0 m tall and sit only ``WALL_T_M`` outside the 1.96 m
#: corridor -- and the whole point of these courses is that the generator misreads a vertical
#: surface as terrain. Spawning with a wall 0.30 m off the pelvis fills the near camera with
#: it before the episode has begun, which tests the wall rather than the descent. 0.40 m
#: leaves ``1.96/2 - 0.40 = 0.58`` m of lateral range, still most of the corridor.
#:
#: Downrange it is applied to BOTH platform edges, so a spawn can sit anywhere from just past
#: the ascent to just before the descent but never ON either flight -- a pelvis over a riser
#: would drop the robot onto a step edge at reset.
UPDOWN_SPAWN_EDGE_BUFFER_M: float = 0.40

#: Relative draw weights for ``(ground start, top platform)``.
#:
#: Equal, i.e. half the envs start at the bottom and half on top. The bottom half is the whole
#: course; the top half is the DESCENT alone, which is the half the deployed pair has never
#: done well (mjeval scores stairs-down 0.233 and the hardware run never tested stairs at
#: all) and the half a bottom-spawned episode only reaches if the ascent went right. Without
#: it, descent data is conditioned on a successful ascent and there is very little of it.
UPDOWN_SPAWN_WEIGHTS: tuple[float, float] = (1.0, 1.0)

UPDOWN_TRAIN_APPROACHES_M: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0)

#: The held-out approach distance (m): interior to the training range and off its grid.
UPDOWN_HOLDOUT_APPROACH_M: float = 2.25

#: Commanded forward speeds (m/s) this library is INTENDED to be recorded at, one run each --
#: six of the ten points of ``mjeval.terrains.STAIR_SPEEDS``, which are themselves exact points
#: of the library's own uniform 0.600-2.600 stair grid.  So every speed here is one the
#: training set actually contains.  Provenance only: the speed is a command, pinned per run.
#: The commanded-speed RANGE this course is recorded over (m/s), sampled CONTINUOUSLY.
#:
#: **A range rather than a ladder, because the runtime never had a ladder.**
#: ``FmTrackingCommand._reset_freeform_command`` draws ``uniform_(min, max)`` per episode, so
#: a discrete list of six speeds was a description of nothing -- it reached the build banner
#: and the file's provenance block and was never enforced anywhere. Declaring the interval
#: says what actually happens.
#:
#: 0.6 is the free-form command's own floor; 2.5 is chosen for this course rather than
#: inherited, and it is well past what the flight was designed for -- which is the point: the
#: policy has to pick a mode for a 6 x 0.15 m staircase at a speed it was never shown one at.
UPDOWN_SPEED_RANGE_MPS: tuple[float, float] = (0.6, 2.5)

#: The held-out commanded speed (m/s).  Interior to :data:`UPDOWN_SPEED_RANGE_MPS`, so the
#: holdout is off the training GRID (its own approach distance) without also being off the
#: command distribution -- two confounded axes would make a holdout gap unattributable.
UPDOWN_HOLDOUT_SPEED_MPS: float = 1.150


def updown_family(platform_depth_m: float, runout_m: float = UPDOWN_RUNOUT_M) -> str:
    """The family tag for one up-into-down variant.

    Mirrors mjeval's own course naming (``mjeval_updn_p020`` / ``mjeval_updn_p080``) so the two
    halves are recognisable as the same course by eye as well as by test.  A non-default runout
    appends its own tag rather than reusing the base name; see :data:`UPDOWN_RUNOUT_M`.

    Args:
        platform_depth_m: Flat platform between the two flights (m).
        runout_m: Extra flat ground past the bottom riser (m).

    Returns:
        e.g. ``"updown_p020"``, or ``"updown_p020_r030"`` with a 3 m runout.
    """
    tag = f"updown_p{int(round(float(platform_depth_m) * 10)):03d}"
    if abs(float(runout_m) - UPDOWN_RUNOUT_M) >= 1e-9:
        tag += f"_r{int(round(float(runout_m) * 10)):03d}"
    return tag


def updown_feature_x(approach_m: float, platform_depth_m: float) -> dict[str, float]:
    """Every downrange landmark of one up-into-down section, tile-local (m).

    The ascending flight puts riser ``n`` at ``[x0 + (n-1) * run, x0 + n * run]`` with its top
    at ``n * rise``; the platform is flush with the last tread; the descending flight needs
    only ``steps - 1`` treads because the last step down lands on the floor -- which is what
    makes both flights exactly :data:`UPDOWN_STEPS` risers.  That is
    ``generate_stairs_scene._lane_geoms``' construction, restated.

    Args:
        approach_m: Flat run-up from the spawn to the first riser (m).
        platform_depth_m: Flat platform between the two flights (m).

    Returns:
        ``first_riser``, ``platform_x0``, ``platform_x1``, ``bottom_x`` and ``top_z``.
    """
    x0 = _spawn_x() + float(approach_m)
    plat0 = x0 + UPDOWN_STEPS * UPDOWN_RUN_M
    plat1 = plat0 + float(platform_depth_m)
    return {"first_riser": x0, "platform_x0": plat0, "platform_x1": plat1,
            "bottom_x": plat1 + (UPDOWN_STEPS - 1) * UPDOWN_RUN_M,
            "top_z": UPDOWN_STEPS * UPDOWN_RISE_M}


def make_updown_tile(terrain_id: int, approach_m: float, platform_depth_m: float, *,
                     runout_m: float = UPDOWN_RUNOUT_M,
                     walls: bool = UPDOWN_WALLS,
                     wall_collidable: bool = UPDOWN_WALL_COLLIDABLE,
                     family: str | None = None) -> dict:
    """One up-into-down section: 6 risers up, a platform, 6 risers back down to the floor.

    No z offset is applied, unlike the free-form ``stairs`` family: the bottom of the descent
    IS the ground, so the lowest walkable surface already sits at ``z = 0`` and the absolute-z
    fall termination has nothing to fire on.

    Args:
        terrain_id: The tile id.
        approach_m: Flat run-up from the spawn to the first riser (m).
        platform_depth_m: Flat platform between the two flights (m).
        runout_m: Extra flat ground past the bottom riser (m); see :data:`UPDOWN_RUNOUT_M`.
        walls: Author the stairwell walls at all; see :data:`UPDOWN_WALLS`.
        wall_collidable: Author them as collision geometry rather than decor; see
            :data:`UPDOWN_WALL_COLLIDABLE`.
        family: The manifest family.  ``None`` derives it with :func:`updown_family`, which is
            what keeps the two platform depths from pooling.

    Returns:
        The manifest entry, carrying its ``route`` block and an ``updown`` record.

    Raises:
        ValueError: On a non-positive approach or platform depth, a negative runout, or if the
            tile's ground cannot hold the goal band.
    """
    if not float(approach_m) > 0.0:
        raise ValueError(f"approach must be positive, got {approach_m}. It is measured from "
                         "the pelvis to the first RISER, so 0 spawns on the flight.")
    if not float(platform_depth_m) > 0.0:
        raise ValueError(f"platform depth must be positive, got {platform_depth_m}; without a "
                         "platform the two flights are one flight over a peak.")
    if float(runout_m) < 0.0:
        raise ValueError(f"runout must be non-negative, got {runout_m}.")

    f = updown_feature_x(approach_m, platform_depth_m)
    size_y = UPDOWN_WIDTH_M + 2.0 * UPDOWN_APRON_M
    cy = size_y / 2.0
    # mjeval's mapping: the rectangle's centre sits TARGET_LEAD past the bottom riser, and the
    # RL side's LINE goes at that rectangle's near edge.
    goal_x = f["bottom_x"] + float(runout_m) + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0
    size_x = goal_x + GOAL_INSET_M + SECTION_MARGIN_M

    overlays = [_ground_overlay(size_x, size_y)]
    # Every tread is a solid pillar from the ground to its own top -- `box_clip.stair_tile_boxes`
    # and `_lane_geoms` share that convention, so the riser faces are real vertical surfaces
    # with no gap for a foot to catch.
    for n in range(1, UPDOWN_STEPS + 1):
        x0 = f["first_riser"] + (n - 1) * UPDOWN_RUN_M
        overlays.append(_overlay(x0 + UPDOWN_RUN_M / 2.0, cy, 0.0,
                                 UPDOWN_RUN_M, UPDOWN_WIDTH_M, n * UPDOWN_RISE_M))
    overlays.append(_overlay((f["platform_x0"] + f["platform_x1"]) / 2.0, cy, 0.0,
                             f["platform_x1"] - f["platform_x0"], UPDOWN_WIDTH_M, f["top_z"]))
    for m in range(1, UPDOWN_STEPS):
        x0 = f["platform_x1"] + (m - 1) * UPDOWN_RUN_M
        overlays.append(_overlay(x0 + UPDOWN_RUN_M / 2.0, cy, 0.0, UPDOWN_RUN_M,
                                 UPDOWN_WIDTH_M, (UPDOWN_STEPS - m) * UPDOWN_RISE_M))

    if walls:
        # `lane_wall_geoms`' own span: it runs one tread PAST the bottom of the descent, which
        # is mjeval's arithmetic (`steps * run * 2 + platform_depth` from the first riser) and
        # is mirrored rather than tidied so the two sides can be pinned equal.
        wx0 = f["first_riser"] - UPDOWN_WALL_LEAD_M
        wx1 = f["first_riser"] + UPDOWN_STEPS * UPDOWN_RUN_M * 2.0 + float(platform_depth_m)
        for sign in (1.0, -1.0):
            overlays.append(_overlay(
                (wx0 + wx1) / 2.0, cy + sign * (UPDOWN_WIDTH_M / 2.0 + WALL_T_M / 2.0), 0.0,
                wx1 - wx0, WALL_T_M, UPDOWN_WALL_HEIGHT_M,
                kind="obstacle", walkable_top=False,
                decor=not bool(wall_collidable)))

    # Half the CORRIDOR, floored at what the RL termination clamps up to -- not mjeval's 2.5 m.
    # See the module docstring for why the two harnesses differ here on purpose.
    # Two spawn areas: the ground start, and the TOP PLATFORM between the two flights.
    #
    # **The ground one is authored at exactly `FREEFORM_SPAWN_JITTER_M`.** Declaring any area
    # REPLACES `pos_jitter_m` for the whole tile (`resets._draw_spawn_areas`), so sizing it to
    # anything else silently changes the ground spawns and makes them incomparable to a run
    # without areas. box2 paid this once at 4x the intended lateral spread.
    buf = float(UPDOWN_SPAWN_EDGE_BUFFER_M)
    half_y = UPDOWN_WIDTH_M / 2.0 - buf
    half_x = float(platform_depth_m) / 2.0 - buf
    if half_y <= 0.0 or half_x <= 0.0:
        raise ValueError(
            f"a {buf:.2f} m spawn buffer leaves no usable platform: the corridor is "
            f"{UPDOWN_WIDTH_M:.2f} m wide and the platform {platform_depth_m:.2f} m deep. "
            "Lower UPDOWN_SPAWN_EDGE_BUFFER_M or deepen the platform.")
    w_ground, w_top = UPDOWN_SPAWN_WEIGHTS
    areas = [
        spawn_area(_spawn_x(), cy, 0.0,
                   FREEFORM_SPAWN_JITTER_M[0], FREEFORM_SPAWN_JITTER_M[1],
                   weight=w_ground),
        # z is the platform's TOP, which is what the reset stands the robot on. One riser low
        # here is a PhysX ejection kick on every episode -- see `spawn_z_offset`'s history.
        spawn_area((f["platform_x0"] + f["platform_x1"]) / 2.0, cy, f["top_z"],
                   half_x, half_y, weight=w_top),
    ]
    route = _route((_spawn_x(), cy), 0.0, 0.0, (goal_x, cy),
                   _fit_goal_half_width(max(UPDOWN_WIDTH_M / 2.0, GOAL_HALF_WIDTH_FLOOR_M),
                                        overlays, (goal_x, cy)),
                   0.0, spawn_areas=areas)
    fam = family if family is not None else updown_family(platform_depth_m, runout_m)
    tile = _base_tile(terrain_id, str(fam), size_x, size_y, overlays, route)
    tile["updown"] = {
        "approach_m": float(approach_m),
        "platform_depth_m": float(platform_depth_m),
        "rise_m": UPDOWN_RISE_M, "run_m": UPDOWN_RUN_M, "steps": int(UPDOWN_STEPS),
        "width_m": UPDOWN_WIDTH_M,
        "flight_height_m": f["top_z"],
        "first_riser_x": f["first_riser"],
        "platform_x": [f["platform_x0"], f["platform_x1"]],
        "bottom_x": f["bottom_x"],
        "walls": bool(walls),
        "wall_collidable": bool(walls and wall_collidable),
        "spawn_areas": len(areas),
        "top_spawn_half_xy": [half_x, half_y],
        "goal_rect_centre_x": goal_x + GOAL_DEPTH_M / 2.0,
        "goal_depth_m": GOAL_DEPTH_M,
        # Only the DEFAULT runout is mjeval's course, and only then does the tile claim to be
        # it -- so a lengthened-runout recording cannot be read as an mjeval clone.
        # Claimed only by a tile that really IS the mjeval course: its runout AND its wall
        # collidability both have to match, because solid walls are different physics and a
        # rate measured against them cannot be read as an mjeval rate.
        "mjeval_course": (f"mjeval_updn_p{int(round(float(platform_depth_m) * 10)):03d}"
                          if (abs(float(runout_m) - UPDOWN_RUNOUT_M) < 1e-9
                              and not (walls and wall_collidable)) else ""),
    }
    # Recorded ONLY when it is not the default, and that asymmetry is deliberate:
    # `fingerprint_terrain_manifest` hashes the whole tile dict, so a key added unconditionally
    # moves the fingerprint of geometry that has not changed by a millimetre and strands every
    # rollout already recorded against it (box2 paid this once). A variant that IS the default
    # needs no field to say so; one that is not, does.
    if abs(float(runout_m) - UPDOWN_RUNOUT_M) >= 1e-9:
        tile["updown"]["runout_m"] = float(runout_m)
    return tile


def generate_updown_manifest(approaches_m: tuple[float, ...], platform_depth_m: float, *,
                             runout_m: float = UPDOWN_RUNOUT_M,
                             walls: bool = UPDOWN_WALLS,
                             wall_collidable: bool = UPDOWN_WALL_COLLIDABLE,
                             family: str | None = None) -> list[dict]:
    """One tile per approach distance at one platform depth, in the order given.

    The platform depth is a GEOMETRY axis and does not vary within a manifest: its two values
    are different courses of different lengths, and pooling their success rates would compare
    two obstacles under one name.  That is why it selects the family.

    Args:
        approaches_m: Pelvis-to-first-riser distances (m).
        platform_depth_m: Flat platform between the two flights (m).
        runout_m: See :data:`UPDOWN_RUNOUT_M`.
        walls: See :data:`UPDOWN_WALLS`.
        family: See :func:`make_updown_tile`.

    Returns:
        The manifest tile list, ``terrain_id`` running 0..n-1 in the order given.

    Raises:
        ValueError: On an empty approach set, or via :func:`make_updown_tile`.
    """
    if not approaches_m:
        raise ValueError("no approach distances given; an up-into-down library needs at least "
                         "one tile.")
    return [make_updown_tile(i, a, platform_depth_m, runout_m=runout_m, walls=walls,
                             wall_collidable=wall_collidable, family=family)
            for i, a in enumerate(approaches_m)]
