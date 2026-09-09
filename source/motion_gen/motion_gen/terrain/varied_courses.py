# Copyright 2026 Zachary Olkin. All rights reserved.
"""Randomised out-of-distribution courses for the MULTI-TERRAIN fine-tune.

Two course types, authored DIRECTLY rather than as a variation layer over
:mod:`~motion_gen.terrain.box2` and :mod:`~motion_gen.terrain.step_courses`.  Those two are
mjeval clones whose whole value is that they are pinned -- fixed geometry, one tile per initial
condition, byte-comparable fingerprints -- and they stay that way, because every baseline this
fine-tune will be read against was measured on them.  What is wanted here is the opposite
thing, so it is a separate generator instead of a flag on theirs:

* :func:`make_boxes_tile` -- one or two boxes, drawn height / depth / width / gap.  An angled
  approach is expressible and tested but is OFF by default; see
  :data:`APPROACH_YAW_IS_OFF_BY_DEFAULT`.
* :func:`make_updown_tile` -- one flight up, a platform, one flight back down, with drawn
  riser count, corridor width and stairwell walls.

**Three properties are shared by every tile here, and each is a deliberate break from the
pinned courses.**

1. **A tile is not an initial condition.**  On the pinned courses the approach distance IS the
   tile axis -- five tiles, five distances -- so a cell is a state and the held-out cell is an
   interpolation probe.  Here every tile carries the whole approach distribution as a CHAIN of
   ground spawn areas along its own route, so geometry is the only thing a tile index selects.
   The per-episode approach distance is still recoverable downstream, from the chunk table's
   ``goal_downrange_m`` at ``replan_index == 0``; it is NOT recoverable from the episode
   table's ``goal_travel_m``, which reads the route's DECLARED start and is therefore constant
   per tile.

2. **The commanded speed is a continuous band, declared by the course.**  Every fine-tune
   library here stamps :data:`COMMANDED_SPEED_RANGE_MPS` into ``generator_config``, which
   ``clip_io.read_commanded_speed_range`` reads back.  The pinned two-box course declared a
   six-rung LADDER that was provenance only and enforced nowhere -- the runtime has always
   drawn ``uniform_(min, max)`` per episode -- so a range is what actually happens.

3. **Boxes are met at an angle by turning the APPROACH, never by rotating the box.**  The
   overlay schema has a ``yaw`` field and the RL loader hard-refuses a non-zero one
   (``clip_terrain_loader._overlays_from_manifest``): :class:`OverlayVolumeCfg` has no yaw and
   ``build_overlay`` only translates, so a yawed box is axis-aligned in Isaac while
   ``generate_clip_tile_scene`` emits a rotation quaternion for it -- one manifest, two
   different worlds, no error anywhere.  Supporting it for real means widening
   ``overlay_sub_boxes``' ``(cx, cy, cz, sx, sy, sz)`` contract, which every renderer and the
   analytic box SDF behind the terrain-consistency reward depend on.  So the route runs at
   ``start_yaw`` to the tile's +x and the geometry stays axis-aligned, which from the robot's
   point of view is the same encounter.  The cost, stated because it is real: every box in a
   tile shares the angle, so two boxes at different angles to each other is not expressible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

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
# Families and the shared command band
# ---------------------------------------------------------------------------------------------
#: Family tag for the varied box tiles.  Distinct from every pinned family so no report, no
#: per-family threshold and no fingerprint comparison can pool a varied tile with a clone.
BOXES_FAMILY: str = "varied_boxes"

#: Family tag for the varied up-into-down tiles.
UPDOWN_FAMILY: str = "varied_updown"

#: The commanded forward-speed band (m/s), sampled CONTINUOUSLY per episode, shared by every
#: course in a multi-terrain library.
#:
#: One band for the whole library rather than one per family, because the runtime draws the
#: speed per EPISODE from the env's own ``freeform_speed_min/max`` and knows nothing about which
#: tile the episode landed on -- a per-family band would be a claim the runtime cannot honour.
#: The numbers are the up-into-down course's: 0.6 is the free-form command's own floor and 2.5
#: is well past what a 6 x 0.15 m flight or a 0.40 m box was designed for, which is the point.
COMMANDED_SPEED_RANGE_MPS: tuple[float, float] = (0.6, 2.5)

#: Goal rectangle depth (m) and how far past the last obstacle its centre sits -- mjeval's
#: mapping, kept because the goal LINE convention downstream is unchanged: the line goes at the
#: rectangle's near edge, so ``mdp.freeform_goal_reached``'s ``x >= goal_x`` test and a
#: rectangle test agree on the boundary that decides the outcome.
GOAL_DEPTH_M: float = 2.0
TARGET_LEAD_M: float = 1.5

#: Goal band half-extent (m), the SAME on every tile of every family here.
#:
#: **It does not scale with the drawn corridor, and that is the whole point.**  The arrival
#: tolerance reaches the runtime twice -- as each tile's ``route.goal_half_width``, which the
#: ``freeform_goal_reached`` TERMINATION tests, and as the single ``default.half_width_m`` of
#: the ``goal_regions.json`` spec that ``goal_region.in_goal`` reads.  Nothing reconciles them:
#: the spec is one number for the library, so a band that varied per tile could only be
#: declared at some summary of itself, and both directions of that mismatch are silent.  Too
#: wide and a robot standing beside the flight scores as arrived (the up-into-down spec's own
#: note records box2's 2.5 m being 2.6x too wide for that reason); too narrow and a legitimate
#: arrival off the centreline never fires ``in_goal`` while the termination still ends the
#: episode, so it records as a time-out.
#:
#: Fixed at the free-form floor, which every tile here can hold: the narrowest corridor is
#: 1.219 m and its walkable width at the goal is that plus two aprons, so the band never has to
#: be cut by :func:`~motion_gen.terrain.freeform._fit_goal_half_width`.  Pinned by a test that
#: reads it off the built tiles rather than trusting this comment.
GOAL_HALF_WIDTH_M: float = GOAL_HALF_WIDTH_FLOOR_M

#: Extra flat ground past the last obstacle, BEYOND what the goal line needs (m).
#:
#: Non-zero on purpose, and it is the ``DESCENT_RUNOUT`` argument: without it the goal band's
#: near edge sits half a goal-depth past the last riser, which a robot can enter while still
#: stumbling off the flight -- so "got down" and "fell down in roughly the right direction"
#: score the same.  3.0 m is what mjeval's own ``stairs_down`` class adds.
RUNOUT_M: float = 3.0


# ---------------------------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------------------------
@dataclass
class VariedCoursesCfg:
    """Every distribution the two varied courses draw from.

    Attributes:
        seed: Base seed.  A tile's draws depend on ``(seed, terrain_id, salt)`` and nothing
            else, so a tile is reproducible from its index and the library can be rebuilt tile
            by tile without a global iteration order.
        approach_range_m: Pelvis-to-first-obstacle distance at spawn (m), as an interval
            covered by the ground spawn chain rather than as a per-tile constant.
        boxes_per_tile_range: Inclusive box-count range.  ``(1, 2)`` gives the single-box and
            two-box courses from one generator, as two draws of the same axis.
        box_height_range_m: Box top height (m), drawn continuously.  Wider than the trained
            0.25-0.75 grid at the top end on purpose.
        box_depth_range_m: Extent along travel (m).  Spans the trained 0.46 m -- under one
            stride, so the jump up and the jump down collapse into one ballistic event -- and
            the pinned course's 1.20 m, which has a top the robot must stand on.  Having both
            in one library puts that boundary inside the training distribution.
        box_width_range_m: Lateral extent (m), inside the trained 0.81-7.30 band.
        box_gap_range_m: Flat ground between consecutive boxes (m).  Ignored at one box.
        box_max_approach_deg: How far the route may run off the tile's +x.  **0.0, i.e. OFF**
            -- see :data:`APPROACH_YAW_IS_OFF_BY_DEFAULT`.  The mechanism is kept and tested
            because it is the only way an angled encounter can be authored at all (an overlay
            yaw is refused; see the module docstring), but nothing ships with it on.
        updown_steps_range: Inclusive riser count per flight.  The low end is BELOW the trained
            3-39, and at one riser the course is a curb up and a curb down -- deliberate, and
            named here so a reader does not take "up into down" to imply a staircase.
        updown_width_range_m: Corridor width (m).  ``(1.219, 3.048)`` is 4 to 10 feet, which
            brackets both of mjeval's stair widths (1.22 and 1.96).
        updown_rise_m: Riser height (m).  Fixed at the trained mode: the riser is the axis the
            step-geometry sweep already covers, and varying it here would confound this
            library's terrain axes with that one's.
        updown_run_m: Tread depth (m); the only tread in both trained sets.
        updown_platform_range_m: Flat platform between the two flights (m).
        updown_wall_sides: The wall draws.  One or two walls, never none -- an unwalled flight
            has ground-level apron beside it and no vertical surface to stop a robot walking
            past the staircase, which scores as a traverse.  With at least one wall the open
            side is at most one, and :data:`UPDOWN_APRON_M` keeps even that unwalkable.
        updown_wall_height_range_m: Stairwell wall height (m).  Above ~0.9 m this is a viewer
            choice rather than a model one -- at 3 m the far camera's frame already cuts the
            wall, so a 1.0 m and a 3.0 m wall are byte-identical images.
        spawn_edge_buffer_m: Keep-out from the edges of any raised spawn surface (m).
    """

    seed: int = 0
    approach_range_m: tuple[float, float] = (1.5, 4.0)

    boxes_per_tile_range: tuple[int, int] = (1, 2)
    box_height_range_m: tuple[float, float] = (0.25, 0.75)
    box_depth_range_m: tuple[float, float] = (0.45, 1.40)
    box_width_range_m: tuple[float, float] = (1.20, 4.00)
    box_gap_range_m: tuple[float, float] = (1.50, 4.00)
    box_max_approach_deg: float = 0.0

    updown_steps_range: tuple[int, int] = (1, 10)
    updown_width_range_m: tuple[float, float] = (1.219, 3.048)
    updown_rise_m: float = 0.150
    updown_run_m: float = 0.250
    updown_platform_range_m: tuple[float, float] = (1.50, 3.00)
    updown_wall_sides: tuple[tuple[str, ...], ...] = (
        ("left", "right"), ("left",), ("right",))
    updown_wall_height_range_m: tuple[float, float] = (1.2, 2.4)

    spawn_edge_buffer_m: float = 0.30

    #: Relative draw weight of the GROUND chain against each raised surface.  Equal, so on a
    #: two-box tile the split is ground / box 1 / box 2 in thirds and on an up-into-down tile
    #: it is ground / platform in halves -- matching what the pinned courses established.
    #:
    #: **Why spawn on the raised surfaces at all.** The pinned two-box course measured it: of
    #: 352 successful episodes, EVERY one reached the goal by dismounting a box, so a correct
    #: two-jump traverse never appeared in the data and no reweighting could select for it.
    #: Starting on the obstacle makes the dismount its own frequent isolated problem.
    surface_weight: float = 1.0

    #: Salts keeping one tile's independent draws from sharing a stream.
    salts: dict[str, int] = field(default_factory=lambda: {"tile": 11, "box": 101})


#: Why :attr:`VariedCoursesCfg.box_max_approach_deg` ships at 0.
#:
#: An angled approach is expressible here and is the RIGHT way to author an off-angle box
#: encounter -- the route turns, the geometry stays axis-aligned, both simulators agree.  It is
#: nonetheless off by default, as a scope decision rather than a technical one: a non-zero yaw
#: puts the route off the tile axis, and every quantity derived from it then has a lateral
#: component that was exactly zero before -- the tile's own y extent, the spawn chain's
#: placement, each area's heading, the goal's position, and the steering controller's projection
#: of the root onto the route ray.  Each is handled and tested, but they are five new ways for a
#: first multi-terrain run to fail for a reason that is not about the policy.
#:
#: With it at 0 the whole lateral path collapses: ``slope`` is 0, the drift term is 0, every
#: spawn sits on the tile centreline and every heading is the frame's.  Turn it up once the
#: axis-aligned arm has a number to be compared against.
APPROACH_YAW_IS_OFF_BY_DEFAULT: str = (
    "route yaw is available and tested but ships at 0; see this constant's docstring")

#: Ground beside an up-into-down flight, per side (m) -- the WALL BAND and nothing else.
#:
#: **Deliberately not the free-form ``stairs`` family's ``WALL_T_M + SECTION_MARGIN_M``.**  That
#: 0.65 m is ground-level walkable apron, and because the goal band spans the full corridor at
#: ground level, a robot that drifts onto it and walks PAST the staircase scores a success.  On
#: the free-form family the collidable walls block it on the two-wall draw and leave it open on
#: the others; here the apron is the wall band exactly, so every walkable point outside the
#: flight is under a wall whether or not that side drew one.
UPDOWN_APRON_M: float = WALL_T_M


def tile_rng(cfg: VariedCoursesCfg, terrain_id: int, salt: int = 0) -> np.random.Generator:
    """A generator private to one tile and one draw, reproducible from its index.

    Args:
        cfg: The course config, for its seed.
        terrain_id: The tile.
        salt: Distinguishes independent draws within a tile.

    Returns:
        A seeded generator.
    """
    # A SEQUENCE, not an arithmetic mix. The `(seed * k + tid) ^ salt` idiom used elsewhere in
    # this package is a weak mixer and it collides here: `(X + 2) ^ 101` equals `(X + 3) ^ 102`,
    # so tile 2's first box drew exactly tile 3's second box and the two tiles came out mirror
    # images of each other. `SeedSequence` spreads all three fields properly.
    return np.random.default_rng([int(cfg.seed), int(terrain_id), int(salt)])


def _spawn_x() -> float:
    """Tile-local x of the FARTHEST ground spawn (m).

    The chain runs forward from here toward the obstacle, so the route's declared start is the
    longest approach in the tile.  That is the conservative choice for everything derived from
    the declared route -- ``goal_travel_m`` and the episode time-out horizon both read it -- so
    no episode gets a budget sized for a shorter run than it actually has.

    Returns:
        ``SECTION_MARGIN_M + SPAWN_INSET_M``.
    """
    return SECTION_MARGIN_M + SPAWN_INSET_M


def _ground_spawn_chain(first_obstacle_x: float, y0: float, yaw: float,
                        cfg: VariedCoursesCfg, weight: float) -> list[dict]:
    """Ground spawn areas covering the approach interval, along the route ray.

    A single wide rectangle cannot do this on an ANGLED tile: spawn extents are tile-frame
    (``resets.reset_on_freeform_section`` draws ``area.x + jitter * area.half_x`` with no
    rotation), so a rectangle spanning 2.5 m downrange at 20 degrees would have to be 0.9 m
    wide laterally to contain the route -- putting most spawns off the approach centreline.
    A CHAIN of ordinary-sized areas placed on the ray has every spawn exactly on it.

    The spacing is one full jitter width, so the covered intervals abut with no gap and no
    overlap and the pooled draw over the chain is uniform on the interval.

    Each area's ``yaw`` is the route's, because the reset takes the spawn HEADING from
    ``frame_yaw + area.yaw`` and never from ``route.start_yaw`` -- leaving it 0 on an angled
    tile would spawn the robot up to 20 degrees off its own route and ask the yaw controller to
    recover before the first step.

    Args:
        first_obstacle_x: Tile-local x of the near face of the first obstacle (m).
        y0: Route y at ``first_obstacle_x`` (m).
        yaw: Route heading relative to the tile frame (rad).
        cfg: The course config.
        weight: TOTAL draw weight for the ground, split evenly across the chain -- so the
            ground's share does not depend on how many areas it took to cover the interval.

    Returns:
        The spawn-area dicts, farthest approach first.
    """
    lo, hi = (float(v) for v in cfg.approach_range_m)
    if not 0.0 < lo <= hi:
        raise ValueError(f"approach_range_m must be a positive ordered pair, got {(lo, hi)}.")
    step = 2.0 * FREEFORM_SPAWN_JITTER_M[0]
    n = max(int(math.ceil((hi - lo) / step)), 1)
    slope = math.tan(float(yaw))
    areas: list[dict] = []
    for i in range(n):
        # Centres of n abutting intervals tiling [lo, hi]; a single-area chain sits at the
        # interval's midpoint rather than at either end.
        a = lo + (hi - lo) * (i + 0.5) / n
        x = float(first_obstacle_x) - a
        areas.append(spawn_area(x, float(y0) - slope * a, 0.0,
                                (hi - lo) / (2.0 * n), FREEFORM_SPAWN_JITTER_M[1],
                                yaw=float(yaw), weight=float(weight) / n))
    return areas


# ---------------------------------------------------------------------------------------------
# The varied box course
# ---------------------------------------------------------------------------------------------
def make_boxes_tile(terrain_id: int, cfg: VariedCoursesCfg) -> dict:
    """One varied box section: 1-2 drawn boxes met at a drawn approach angle.

    Args:
        terrain_id: The tile id.
        cfg: The course config.

    Returns:
        The manifest entry, carrying its ``route`` block and a ``varied_boxes`` record of every
        drawn quantity, so the tile says what it is without anyone re-deriving it from geometry.

    Raises:
        ValueError: If the drawn goal band does not fit the tile's ground, or the approach
            interval is not a positive ordered pair.
    """
    rng = tile_rng(cfg, terrain_id, cfg.salts["tile"])
    n_boxes = int(rng.integers(cfg.boxes_per_tile_range[0], cfg.boxes_per_tile_range[1] + 1))
    yaw = float(rng.uniform(-1.0, 1.0)) * math.radians(float(cfg.box_max_approach_deg))
    slope = math.tan(yaw)

    approach_hi = float(cfg.approach_range_m[1])
    first_x = _spawn_x() + approach_hi

    boxes: list[dict] = []
    x = first_x
    for k in range(n_boxes):
        brng = tile_rng(cfg, terrain_id, cfg.salts["box"] + k)
        depth = float(brng.uniform(*cfg.box_depth_range_m))
        height = float(brng.uniform(*cfg.box_height_range_m))
        width = float(brng.uniform(*cfg.box_width_range_m))
        boxes.append({"x0": x, "x1": x + depth, "depth_m": depth,
                      "height_m": height, "width_m": width})
        x += depth
        if k + 1 < n_boxes:
            x += float(brng.uniform(*cfg.box_gap_range_m))

    last_x1 = boxes[-1]["x1"]
    goal_x = last_x1 + RUNOUT_M + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0

    # The route drifts laterally as it runs, so the tile's y extent is set by the whole ray plus
    # the widest box and the goal band -- taken from the run rather than from any one feature.
    half_widths = [b["width_m"] / 2.0 for b in boxes] + [GOAL_HALF_WIDTH_M]
    drift = abs(slope) * (goal_x - _spawn_x())
    half_y = max(half_widths) + drift
    cy = half_y + SECTION_MARGIN_M
    size_y = 2.0 * cy
    size_x = goal_x + GOAL_INSET_M + SECTION_MARGIN_M

    # The route's y is measured at the FIRST BOX and runs back to the spawn and on to the goal,
    # so `first_x` is the ray's reference point and every other y follows from the slope.
    def route_y(x_at: float) -> float:
        """Route y at a tile-local x (m)."""
        return cy + slope * (x_at - first_x)

    overlays = [_ground_overlay(size_x, size_y)]
    for b in boxes:
        overlays.append(_overlay((b["x0"] + b["x1"]) / 2.0, route_y((b["x0"] + b["x1"]) / 2.0),
                                 0.0, b["depth_m"], b["width_m"], b["height_m"]))

    buf = float(cfg.spawn_edge_buffer_m)
    areas = _ground_spawn_chain(first_x, cy, yaw, cfg, weight=float(cfg.surface_weight))
    for b in boxes:
        hx = (b["depth_m"] / 2.0) - buf
        hy = (b["width_m"] / 2.0) - buf
        if hx <= 0.0 or hy <= 0.0:
            # A shallow box has no room for a stance on top, which is a property of the DRAW
            # rather than a mistake -- the depth range spans the trained 0.46 m deliberately.
            # It gets no top spawn; the ground chain still covers it.
            continue
        cx = (b["x0"] + b["x1"]) / 2.0
        areas.append(spawn_area(cx, route_y(cx), b["height_m"], hx, hy,
                                yaw=yaw, weight=float(cfg.surface_weight)))

    goal_xy = (goal_x, route_y(goal_x))
    route = _route((_spawn_x(), route_y(_spawn_x())), 0.0, yaw, goal_xy,
                   _fit_goal_half_width(GOAL_HALF_WIDTH_M, overlays, goal_xy), 0.0,
                   spawn_areas=areas)
    tile = _base_tile(terrain_id, BOXES_FAMILY, size_x, size_y, overlays, route)
    tile["varied_boxes"] = {
        "n_boxes": n_boxes,
        "approach_deg": math.degrees(yaw),
        "approach_range_m": [float(v) for v in cfg.approach_range_m],
        "boxes": [{k: float(v) for k, v in b.items()} for b in boxes],
        "goal_rect_centre_x": goal_x + GOAL_DEPTH_M / 2.0,
        "goal_depth_m": GOAL_DEPTH_M,
        "runout_m": RUNOUT_M,
        "spawn_areas": len(areas),
        "ground_spawn_areas": len(areas) - sum(
            1 for b in boxes
            if (b["depth_m"] / 2.0) - buf > 0.0 and (b["width_m"] / 2.0) - buf > 0.0),
    }
    return tile


# ---------------------------------------------------------------------------------------------
# The varied up-into-down course
# ---------------------------------------------------------------------------------------------
def make_updown_tile(terrain_id: int, cfg: VariedCoursesCfg) -> dict:
    """One varied up-into-down section: a drawn flight up, a platform, the same flight down.

    Both flights carry the same riser count, because a real staircase has one geometry and is
    walked both ways -- the library's own up/down tread asymmetry is a property of the generator
    and not of the world.

    Args:
        terrain_id: The tile id.
        cfg: The course config.

    Returns:
        The manifest entry, carrying its ``route`` block and a ``varied_updown`` record.

    Raises:
        ValueError: If the drawn corridor cannot hold a platform spawn with its edge buffer.
    """
    rng = tile_rng(cfg, terrain_id, cfg.salts["tile"])
    steps = int(rng.integers(cfg.updown_steps_range[0], cfg.updown_steps_range[1] + 1))
    width = float(rng.uniform(*cfg.updown_width_range_m))
    platform = float(rng.uniform(*cfg.updown_platform_range_m))
    sides = tuple(cfg.updown_wall_sides[int(rng.integers(0, len(cfg.updown_wall_sides)))])
    wall_h = float(rng.uniform(*cfg.updown_wall_height_range_m))

    rise, run = float(cfg.updown_rise_m), float(cfg.updown_run_m)
    approach_hi = float(cfg.approach_range_m[1])
    first_riser = _spawn_x() + approach_hi
    plat0 = first_riser + steps * run
    plat1 = plat0 + platform
    # The descending flight needs only `steps - 1` treads: the last step down lands on the
    # floor, which is what makes both flights exactly `steps` risers.
    bottom_x = plat1 + (steps - 1) * run
    top_z = steps * rise

    size_y = width + 2.0 * UPDOWN_APRON_M
    cy = size_y / 2.0
    goal_x = bottom_x + RUNOUT_M + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0
    size_x = goal_x + GOAL_INSET_M + SECTION_MARGIN_M

    overlays = [_ground_overlay(size_x, size_y)]
    # Every tread is a solid pillar from the ground to its own top, so a riser face is a real
    # vertical surface with no gap for a foot to catch.
    for n in range(1, steps + 1):
        x0 = first_riser + (n - 1) * run
        overlays.append(_overlay(x0 + run / 2.0, cy, 0.0, run, width, n * rise))
    overlays.append(_overlay((plat0 + plat1) / 2.0, cy, 0.0, plat1 - plat0, width, top_z))
    for m in range(1, steps):
        x0 = plat1 + (m - 1) * run
        overlays.append(_overlay(x0 + run / 2.0, cy, 0.0, run, width, (steps - m) * rise))

    # COLLIDABLE walls (`decor=False`). They enter the collider SDF, so the terrain-consistency
    # reward charges a plan that puts a limb inside one -- on a corridor this narrow that is a
    # real part of planning a staircase. The termination the visual-only choice was protecting
    # against (`ee_pose_error` on an arm swept into a wall) does not exist on the fine-tune
    # task, whose only enders are the time-out, the height floor, the tilt cone and the goal.
    wall_x0 = first_riser - 1.0
    wall_x1 = bottom_x + run
    for side in sides:
        sign = 1.0 if side == "left" else -1.0
        overlays.append(_overlay(
            (wall_x0 + wall_x1) / 2.0, cy + sign * (width / 2.0 + WALL_T_M / 2.0), 0.0,
            wall_x1 - wall_x0, WALL_T_M, wall_h,
            kind="obstacle", walkable_top=False, decor=False))

    buf = float(cfg.spawn_edge_buffer_m)
    areas = _ground_spawn_chain(first_riser, cy, 0.0, cfg, weight=float(cfg.surface_weight))
    hx, hy = platform / 2.0 - buf, width / 2.0 - buf
    if hx <= 0.0 or hy <= 0.0:
        raise ValueError(
            f"tile {terrain_id}: a {buf:.2f} m spawn buffer leaves no usable platform -- the "
            f"corridor is {width:.2f} m wide and the platform {platform:.2f} m deep. Raise "
            "updown_platform_range_m / updown_width_range_m or lower spawn_edge_buffer_m.")
    # z is the platform's TOP, which is what the reset stands the robot on. One riser low here
    # is a PhysX ejection kick on every episode.
    areas.append(spawn_area((plat0 + plat1) / 2.0, cy, top_z, hx, hy,
                            weight=float(cfg.surface_weight)))

    goal_xy = (goal_x, cy)
    route = _route((_spawn_x(), cy), 0.0, 0.0, goal_xy,
                   _fit_goal_half_width(GOAL_HALF_WIDTH_M, overlays, goal_xy), 0.0,
                   spawn_areas=areas)
    tile = _base_tile(terrain_id, UPDOWN_FAMILY, size_x, size_y, overlays, route)
    tile["varied_updown"] = {
        "steps": steps, "rise_m": rise, "run_m": run,
        "width_m": width, "platform_depth_m": platform,
        "flight_height_m": top_z,
        "first_riser_x": first_riser, "platform_x": [plat0, plat1], "bottom_x": bottom_x,
        "wall_sides": list(sides), "wall_height_m": wall_h, "wall_collidable": True,
        "apron_m": UPDOWN_APRON_M,
        "approach_range_m": [float(v) for v in cfg.approach_range_m],
        "goal_rect_centre_x": goal_x + GOAL_DEPTH_M / 2.0,
        "goal_depth_m": GOAL_DEPTH_M,
        "runout_m": RUNOUT_M,
        "spawn_areas": len(areas),
        "ground_spawn_areas": len(areas) - 1,
    }
    return tile


# ---------------------------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------------------------
def generate_varied_manifest(n_boxes: int, n_updown: int,
                             cfg: VariedCoursesCfg | None = None) -> list[dict]:
    """One multi-terrain manifest: ``n_boxes`` box tiles then ``n_updown`` up-into-down tiles.

    ``terrain_id`` is assigned over the CONCATENATION, so the ids are contiguous and a family
    occupies one block -- which is what lets a report reduce per family by an id range, the way
    the in-distribution routes already do.

    Args:
        n_boxes: Varied box tiles.
        n_updown: Varied up-into-down tiles.
        cfg: The course config; the defaults if omitted.

    Returns:
        The manifest, one entry per tile.

    Raises:
        ValueError: If both counts are zero, or either is negative.
    """
    cfg = cfg if cfg is not None else VariedCoursesCfg()
    if n_boxes < 0 or n_updown < 0:
        raise ValueError(f"tile counts must be non-negative, got {(n_boxes, n_updown)}.")
    if n_boxes + n_updown == 0:
        raise ValueError("a multi-terrain library needs at least one tile.")
    tiles = [make_boxes_tile(i, cfg) for i in range(n_boxes)]
    tiles += [make_updown_tile(n_boxes + i, cfg) for i in range(n_updown)]
    return tiles


def family_blocks(n_boxes: int, n_updown: int) -> dict[str, tuple[int, int]]:
    """The inclusive ``terrain_id`` range each family occupies.

    The reduction key for a per-family report, derived from the same two counts the manifest
    was built from rather than re-scanned out of the file.

    Args:
        n_boxes: Varied box tiles.
        n_updown: Varied up-into-down tiles.

    Returns:
        ``{family: (lo, hi)}``, omitting a family with no tiles.
    """
    out: dict[str, tuple[int, int]] = {}
    if n_boxes:
        out[BOXES_FAMILY] = (0, n_boxes - 1)
    if n_updown:
        out[UPDOWN_FAMILY] = (n_boxes, n_boxes + n_updown - 1)
    return out


def generator_config(n_boxes: int, n_updown: int,
                     cfg: VariedCoursesCfg | None = None) -> dict:
    """The ``generator_config`` block a varied library stamps.

    Carries :data:`COMMANDED_SPEED_RANGE_MPS` under a key
    ``clip_io.read_commanded_speed_range`` looks for, so the course owns its command band
    instead of it living in whichever launcher happened to pass an override.

    Args:
        n_boxes: Varied box tiles.
        n_updown: Varied up-into-down tiles.
        cfg: The course config; the defaults if omitted.

    Returns:
        A JSON-serialisable dict.
    """
    cfg = cfg if cfg is not None else VariedCoursesCfg()
    return {
        "varied_courses": {
            "commanded_speed_range_mps": [float(v) for v in COMMANDED_SPEED_RANGE_MPS],
            "seed": int(cfg.seed),
            "n_boxes": int(n_boxes),
            "n_updown": int(n_updown),
            "family_blocks": {k: list(v) for k, v in family_blocks(n_boxes, n_updown).items()},
            "approach_range_m": [float(v) for v in cfg.approach_range_m],
        }
    }


def goal_region_spec(n_boxes: int, n_updown: int,
                     cfg: VariedCoursesCfg | None = None) -> dict:
    """The DECLARED ``goal_regions.json`` spec for a varied library.

    Declared from the authored courses rather than fitted from clip-arm episodes, because this
    is a manifest-only library with no clips to fit from and on an authored course the goal
    region is a property of the geometry.

    ``half_width_m`` is :data:`GOAL_HALF_WIDTH_M`, which is exactly every tile's own
    ``route.goal_half_width`` -- so ``goal_region.in_goal`` and the ``freeform_goal_reached``
    termination agree, which is the thing a single library-wide number can only do if the tiles
    were built to one band.  ``approach_margin_m`` is 0 because ``freeform_goal_anchor_world``
    already puts the anchor on the goal LINE the tile declares.

    Args:
        n_boxes: Varied box tiles.
        n_updown: Varied up-into-down tiles.
        cfg: The course config; the defaults if omitted.

    Returns:
        A JSON-serialisable spec.
    """
    cfg = cfg if cfg is not None else VariedCoursesCfg()
    return {
        "default": {
            "approach_margin_m": 0.0,
            "half_width_m": float(GOAL_HALF_WIDTH_M),
            # The height test is OFF. A varied library's goals sit at ground level on both
            # families -- past the runout, past the last box -- so there is nothing for it to
            # catch, and a non-zero tolerance here is what made an in-distribution `jump_on`
            # goal scoreable from the floor.
            "height_tol_m": 0.0,
        },
        "by_family": {},
        "provenance": {
            "declared": True,
            "source": "motion_gen.terrain.varied_courses",
            "note": (
                "DECLARED from the authored courses, not fitted: a manifest-only library has "
                "no clips to fit from. half_width_m is GOAL_HALF_WIDTH_M, which every tile of "
                "both families is BUILT to -- the band deliberately does not scale with the "
                "drawn corridor or box width, because this spec is one number for the whole "
                "library and a per-tile band could only be declared at some summary of "
                "itself. Too wide scores a robot beside the obstacle as arrived; too narrow "
                "records a legitimate arrival as a time-out, since the termination still "
                "fires. Neither mismatch reports itself."),
            "goal_half_width_m": float(GOAL_HALF_WIDTH_M),
            "goal_depth_m": float(GOAL_DEPTH_M),
            "runout_m": float(RUNOUT_M),
            "families": {k: list(v) for k, v in family_blocks(n_boxes, n_updown).items()},
            "approach_range_m": [float(v) for v in cfg.approach_range_m],
            "commanded_speed_range_mps": [float(v) for v in COMMANDED_SPEED_RANGE_MPS],
            "seed": int(cfg.seed),
        },
    }
