# Copyright 2026 Zachary Olkin. All rights reserved.
"""A REAL building entrance, measured on site -- the geometry, and the Isaac tile built from it.

The MuJoCo scene (``transfer/sim/generate_entry_stairs_scene.py``) came first. These dimensions
are RESTATED from it rather than imported, for the same reason ``box2`` and ``step_courses``
restate mjeval's: ``source/motion_gen`` is the offline generation package and stays free of
``transfer`` exactly as it stays free of IsaacLab -- and ``transfer`` runs in ``env_transfer``,
which does not carry ``motion_gen`` at all, so an import in either direction breaks one of them.

What makes the restatement safe is a test that imports BOTH and requires them to agree:
``transfer/sim/tests/test_entry_stairs_matches_mjeval.py``. A constant edited on one side fails
there instead of silently producing an Isaac fine-tune and a MuJoCo evaluation that run on two
different staircases.

Dimensions are given in INCHES because that is how they were taken, and converted exactly once
below. That rule is not style: a 10 mm camera-mount error survived in this repo for weeks
because one number was written in two units.

    approach  ->  5 x (6.00" rise, 11.00" run), 200" wide      -> +30.00"
              ->  100" landing, 200" wide
              ->  the walls CLOSE IN to a 100" hallway
              ->  3 x (5.25" rise, 13.00" run), 100" wide      -> +15.75"
              ->  the walls OPEN OUT into an 8 x 8 m walled lobby

**Why this tile earns its place beside the drawn courses.** Every other course in the
fine-tune set draws ONE staircase geometry per tile, because that is what a real staircase has.
This one is the exception that a real building produces anyway:

* **The two flights disagree on both dimensions and on their ratio** -- 6.00/11.00 against
  5.25/13.00. A generator that reads the first flight and carries that answer into the second
  gets the second wrong, and nothing else in the library asks it to re-read mid-traverse.
* **The corridor NARROWS, 200" to 100", immediately before a riser.** A vertical surface inside
  camera range is exactly what the generator has been measured to misread as a box
  (``terrain-dialed-clutter``); here it cannot avoid one, and it arrives with a step behind it.
* **The second flight is shallower AND deeper** -- 5.25" on a 13.00" tread is close to flat, so
  the failure worth watching is a step-up planned where none is needed.

Both flights sit inside the trained riser band (6.00" = 0.152 m and 5.25" = 0.133 m against a
trained {0.130, 0.155, 0.180}) but the SECOND flight's tread, 13.00" = 0.330 m, is the top of
the trained range while the first's 11.00" = 0.279 m is the bottom of it. So the tile spans the
grid rather than leaving it: what is out of distribution here is the SEQUENCE, not the steps.
"""

from __future__ import annotations

from types import SimpleNamespace

from .freeform import (
    GOAL_INSET_M,
    SECTION_MARGIN_M,
    WALL_T_M,
    _base_tile,
    _fit_goal_half_width,
    _ground_overlay,
    _overlay,
    _route,
    spawn_area,
)
# The goal geometry the drawn courses use, taken from there rather than re-chosen: a tile whose
# goal rectangle is a different size from its neighbours' would make the per-tile-type breakdown
# compare two different questions.
from .varied_courses import (
    GOAL_DEPTH_M,
    GOAL_HALF_WIDTH_M,
    RUNOUT_M,
    TARGET_LEAD_M,
    VariedCoursesCfg,
    _ground_spawn_chain,
    _spawn_x,
)

#: Inches to metres. The ONE conversion; every dimension below is stated in inches.
IN: float = 0.0254

#: Family names in the manifest, so the eval's per-tile-type breakdown can name them.
#:
#: The two DIRECTIONS are separate families, not one. They are different questions -- climbing
#: a flight you can see and descending one whose treads fall away below the camera's view are
#: the stack's easiest and hardest stair modes respectively (``STAIRS DOWN BAD`` is the
#: standing hardware finding) -- so pooling them reports a rate that moves with the up/down
#: mix rather than with the policy.
ENTRY_STAIRS_FAMILY: str = "entry_stairs"
ENTRY_STAIRS_UP_FAMILY: str = "entry_stairs_up"
ENTRY_STAIRS_DOWN_FAMILY: str = "entry_stairs_down"

#: The WIDE lower flight: five risers at 6.00 in on an 11.00 in tread, 200 in across.
WIDE_STEPS: int = 5
WIDE_RISE_IN: float = 6.00
WIDE_RUN_IN: float = 11.00
WIDE_WIDTH_IN: float = 200.0

#: The landing between the flights, at the wide flight's width.
LANDING_LEN_IN: float = 100.0

#: The NARROW upper flight: three risers at 5.25 in on a 13.00 in tread, in a 100 in hallway.
HALL_STEPS: int = 3
HALL_RISE_IN: float = 5.25
HALL_RUN_IN: float = 13.00
HALL_WIDTH_IN: float = 100.0

#: Flat ground in front of the building (m), and the length of the lobby's counterpart at the
#: other end. NOT measured -- the building sits on however much ground it sits on.
#:
#: It is sized by the LIBRARY's spawn convention, not chosen: the ground spawn chain covers
#: ``approach_range_m`` measured back from the first riser, and its farthest area sits at
#: ``_spawn_x()``, so the flat has to be at least ``_spawn_x() + approach_range_m[1]`` =
#: 1.25 + 4.0 = 5.25 m or the chain hangs off the tile. 5.5 with a little room, and
#: :func:`make_entry_stairs_tile` raises rather than letting a widened range outgrow it.
#:
#: **It was 3.0, and that was too short in a way that reads as a policy failure.** With a
#: single spawn area the run-up was 0.3-2.7 m, against the drawn courses' guaranteed 1.5-4.0:
#: a third of episodes opened with under a metre of ground before the first riser, which is
#: not enough to reach the commanded speed the tile then scores the robot at. Matching the
#: shared MuJoCo constant is what keeps the two sims one building; the SPAWN is Isaac's own.
APPROACH_M: float = 5.5

#: The indoor CLEARING the hallway opens into, at the top level (m). A lobby: the hallway
#: constrains the robot laterally and then RELEASES it, which is the opposite transition to the
#: narrowing and the one a building actually makes on the way in.
#:
#: In metres rather than inches because, unlike the flights and the landing, it was not measured
#: -- it is a room-sized space chosen to be clearly bigger than the 200" section, so "the walls
#: opened up" is unambiguous to a depth camera rather than a marginal widening. Restated from
#: the MuJoCo scene like everything else here, and compared against it by the agreement test.
#:
#: 8.0 comfortably holds either job an end section can be given (see
#: :func:`make_entry_stairs_tile`): the goal's run-out going UP, and the spawn chain going DOWN.
CLEARING_LEN_M: float = 8.0
CLEARING_WIDTH_M: float = 8.0

#: Stairwell wall height (m), above whichever surface the wall stands on. SOLID, like the drawn
#: courses and unlike mjeval's visual-only default: the objection there is ``ee_pose_error``
#: firing on a blocked wrist, a termination the fine-tune task does not carry.
WALL_H_M: float = 1.2


def geometry_m() -> dict:
    """Every measured dimension, converted once, in metres.

    The single accessor both simulators read, so neither restates a number.

    Returns:
        A dict of the converted dimensions plus the derived flight heights.
    """
    return {
        "wide_steps": WIDE_STEPS,
        "wide_rise_m": WIDE_RISE_IN * IN,
        "wide_run_m": WIDE_RUN_IN * IN,
        "wide_width_m": WIDE_WIDTH_IN * IN,
        "landing_len_m": LANDING_LEN_IN * IN,
        "hall_steps": HALL_STEPS,
        "hall_rise_m": HALL_RISE_IN * IN,
        "hall_run_m": HALL_RUN_IN * IN,
        "hall_width_m": HALL_WIDTH_IN * IN,
        "wide_height_m": WIDE_STEPS * WIDE_RISE_IN * IN,
        "hall_height_m": HALL_STEPS * HALL_RISE_IN * IN,
        "total_height_m": (WIDE_STEPS * WIDE_RISE_IN + HALL_STEPS * HALL_RISE_IN) * IN,
    }


def _wall(cx: float, cy: float, base_z: float, sx: float, sy: float) -> dict:
    """One collidable wall standing on the surface at ``base_z``.

    COLLIDABLE (``decor=False``), so it enters the collider SDF and the terrain-consistency
    reward charges a plan that puts a limb inside one -- on a corridor this narrow that is a
    real part of planning the staircase, and it is the same choice ``varied_courses`` makes.

    The base is the surface the wall stands on, NOT the tile floor, mirroring the MuJoCo scene:
    the hallway's side walls stand on the LANDING and the lobby's on the CLEARING, so each
    rises from the surface the robot is walking on.

    Args:
        cx / cy: Footprint centre (m).
        base_z: The surface the wall stands on (m).
        sx / sy: Footprint extents (m); the height is :data:`WALL_H_M`.

    Returns:
        The overlay dict.
    """
    return _overlay(cx, cy, base_z, sx, sy, WALL_H_M,
                    kind="obstacle", walkable_top=False, decor=False)


def _approach_range(cfg) -> tuple[float, float]:
    """The library's spawn-to-first-obstacle interval (m), from ``cfg`` or the shared default.

    Args:
        cfg: A course config.

    Returns:
        ``(lo, hi)`` in metres.
    """
    rng = getattr(cfg, "approach_range_m", None) or VariedCoursesCfg().approach_range_m
    return float(rng[0]), float(rng[1])


def make_entry_stairs_tile(terrain_id: int, cfg, descending: bool = False) -> dict:
    """The measured entrance as one Isaac tile, climbed or descended.

    Takes ``cfg`` for the shared knobs a course tile needs -- the spawn buffer, the surface
    weight and the approach interval -- but draws NOTHING. This tile is one measured building,
    so more tiles of it would be more episodes on the same staircase, which the spawn chain
    already provides. That is why a multi-terrain config gives it a handful of tiles rather
    than a share proportional to the others.

    **The two directions are separate TILES, not a reversed route.** Every tile's frame runs
    downrange along +x and ``_freeform_route_table`` refuses a ``goal_x`` at or behind its own
    ``start_x``, so "go back down" cannot be expressed as a route on the ascending tile -- it
    has to be the building authored the other way round. ``assert_yaw_free`` says the same
    thing from the other side: no tile in these libraries is turned.

    Descending is NOT the mirror image, and that is the part worth reading. A flight of ``n``
    risers needs ``n`` treads going up and only ``n - 1`` coming down, because the last step
    down lands on the surface already there -- the landing, or the floor. Mirroring the
    ascending tile instead leaves a tread flush with the landing and quietly makes the landing
    one run deeper. ``make_updown_tile`` takes the same care for the same reason.

    Both end sections are built long enough for EITHER job, so the building is identical in
    the two directions and in MuJoCo, and only the end you start from changes:

    * going UP, the flat is the approach and the lobby holds the goal;
    * going DOWN, the lobby is the approach and the flat holds the goal.

    Args:
        terrain_id: The tile id.
        cfg: A course config, for ``spawn_edge_buffer_m``, ``surface_weight`` and
            ``approach_range_m``.
        descending: Start at the top and go down, rather than at the bottom and go up.

    Returns:
        The manifest entry, with its ``route`` block and an ``entry_stairs`` record.

    Raises:
        ValueError: If an end section is too short for the spawn chain or the goal run-out, or
            if the spawn buffer leaves no usable landing to stand on.
    """
    g = geometry_m()
    w_rise, w_run, w_wide = g["wide_rise_m"], g["wide_run_m"], g["wide_width_m"]
    h_rise, h_run, h_wide = g["hall_rise_m"], g["hall_run_m"], g["hall_width_m"]
    landing_len = g["landing_len_m"]
    z_landing, z_top = g["wide_height_m"], g["total_height_m"]
    y_room = CLEARING_WIDTH_M / 2.0

    # The tile is as wide as the LOBBY, the widest thing in it -- the entrance flight is 5.08 m
    # and the lobby 8.0. Apron on all sides so a robot cannot walk around the structure and
    # reach the goal band from beside it.
    size_y = max(w_wide, CLEARING_WIDTH_M + 2.0 * WALL_T_M) + 2.0 * SECTION_MARGIN_M
    cy = size_y / 2.0

    # What each end has to be able to hold. Checked against BOTH ends rather than against the
    # one this direction uses, so the building does not silently change shape with `descending`.
    approach_hi = _approach_range(cfg)[1]
    need_approach = _spawn_x() + approach_hi
    need_runout = RUNOUT_M + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0 + GOAL_INSET_M + SECTION_MARGIN_M
    need = max(need_approach, need_runout)
    for name, length in (("APPROACH_M", APPROACH_M), ("CLEARING_LEN_M", CLEARING_LEN_M)):
        if length < need - 1e-9:
            raise ValueError(
                f"tile {terrain_id}: {name} is {length:.2f} m but an end section must hold "
                f"{need:.2f} m -- the spawn chain needs {need_approach:.2f} "
                f"(_spawn_x() {_spawn_x():.2f} + approach {approach_hi:.2f}) and the goal "
                f"run-out {need_runout:.2f}. Both ends are sized for either job so the "
                "building is the same shape in both directions.")

    overlays = [_ground_overlay(0.0, 0.0)]        # placeholder; sized once size_x is known

    def tread(x0: float, run: float, width: float, top_z: float) -> None:
        """One tread, as a solid pillar from the ground to its own top."""
        overlays.append(_overlay(x0 + run / 2.0, cy, 0.0, run, width, top_z))

    if not descending:
        x_wide0 = APPROACH_M
        x_wide1 = x_wide0 + g["wide_steps"] * w_run
        x_land0, x_land1 = x_wide1, x_wide1 + landing_len
        x_hall0 = x_land1
        x_hall1 = x_hall0 + g["hall_steps"] * h_run
        x_lobby0, x_lobby1 = x_hall1, x_hall1 + CLEARING_LEN_M
        size_x = x_lobby1 + SECTION_MARGIN_M
        # Up: every riser needs its own tread, the last of them at the level it arrives on.
        for n in range(1, g["wide_steps"] + 1):
            tread(x_wide0 + (n - 1) * w_run, w_run, w_wide, n * w_rise)
        for n in range(1, g["hall_steps"] + 1):
            tread(x_hall0 + (n - 1) * h_run, h_run, h_wide, z_landing + n * h_rise)
        first_obstacle_x, spawn_z = x_wide0, 0.0
        goal_from_x, goal_z = x_hall1, z_top
    else:
        x_lobby0, x_lobby1 = 0.0, CLEARING_LEN_M
        x_hall0 = x_lobby1
        # Down: `steps - 1` treads. The last step down lands on the surface already there.
        x_hall1 = x_hall0 + (g["hall_steps"] - 1) * h_run
        x_land0, x_land1 = x_hall1, x_hall1 + landing_len
        x_wide0 = x_land1
        x_wide1 = x_wide0 + (g["wide_steps"] - 1) * w_run
        size_x = x_wide1 + APPROACH_M + SECTION_MARGIN_M
        for m in range(1, g["hall_steps"]):
            tread(x_hall0 + (m - 1) * h_run, h_run, h_wide,
                  z_landing + (g["hall_steps"] - m) * h_rise)
        for m in range(1, g["wide_steps"]):
            tread(x_wide0 + (m - 1) * w_run, w_run, w_wide, (g["wide_steps"] - m) * w_rise)
        first_obstacle_x, spawn_z = x_hall0, z_top
        goal_from_x, goal_z = x_wide1, 0.0

    overlays[0] = _ground_overlay(size_x, size_y)
    # The landing, at the wide flight's width and height, and the lobby at the hall's top.
    overlays.append(_overlay((x_land0 + x_land1) / 2.0, cy, 0.0,
                             landing_len, w_wide, z_landing))
    overlays.append(_overlay((x_lobby0 + x_lobby1) / 2.0, cy, 0.0,
                             CLEARING_LEN_M, CLEARING_WIDTH_M, z_top))

    # ---- THE NARROWING, 200" to 100" ------------------------------------------------------
    # Two RETURN FACES at the landing's HALL end: frontal surfaces square to travel, with the
    # hallway's mouth between them and a riser immediately beyond. This is the feature the tile
    # is for -- a vertical surface inside camera range is what the generator has been measured
    # to misread as a box, and here it cannot avoid one. Side walls alone are NOT this: they
    # run parallel to travel and give the robot nothing frontal to read.
    #
    # At the landing's hall end, never the wide flight's: the measured entrance is 200" wide
    # for the flight AND the landing, and the walls close only for the upper flight. Narrowing
    # the landing too removes what the course asks -- arriving on a 5 m landing and having to
    # commit to a 2.5 m hallway.
    face_x = x_land1 - WALL_T_M / 2.0 if not descending else x_land0 + WALL_T_M / 2.0
    hall_lo = min(x_hall0, x_hall1) - (WALL_T_M if not descending else 0.0)
    hall_hi = max(x_hall0, x_hall1) + (0.0 if not descending else WALL_T_M)
    for sign in (1.0, -1.0):
        y_in, y_out = cy + sign * h_wide / 2.0, cy + sign * w_wide / 2.0
        overlays.append(_wall(face_x, (y_in + y_out) / 2.0, z_landing,
                              WALL_T_M, abs(y_out - y_in)))
        # The hallway's side walls, which STOP AT THE LOBBY. Running them on through it would
        # make the room a corridor and delete the widening -- the transition this course was
        # extended to include, and the one the descending direction meets first.
        overlays.append(_wall((hall_lo + hall_hi) / 2.0,
                              cy + sign * (h_wide / 2.0 + WALL_T_M / 2.0), z_landing,
                              hall_hi - hall_lo, WALL_T_M))

    # ---- THE LOBBY'S OWN PERIMETER --------------------------------------------------------
    # So it reads as INDOORS rather than as a plateau: side walls, a far wall at its closed
    # end, and the hall-end wall in two segments with the hallway's mouth left open between
    # them. All standing on the clearing floor, not on the ground.
    far_x = (x_lobby1 - WALL_T_M / 2.0) if not descending else (x_lobby0 + WALL_T_M / 2.0)
    mouth_x = (x_lobby0 + WALL_T_M / 2.0) if not descending else (x_lobby1 - WALL_T_M / 2.0)
    for sign in (1.0, -1.0):
        overlays.append(_wall((x_lobby0 + x_lobby1) / 2.0,
                              cy + sign * (y_room + WALL_T_M / 2.0), z_top,
                              CLEARING_LEN_M, WALL_T_M))
        y_in, y_out = cy + sign * h_wide / 2.0, cy + sign * y_room
        overlays.append(_wall(mouth_x, (y_in + y_out) / 2.0, z_top,
                              WALL_T_M, abs(y_out - y_in)))
    overlays.append(_wall(far_x, cy, z_top, WALL_T_M, CLEARING_WIDTH_M))

    # ---- spawns ---------------------------------------------------------------------------
    buf = float(getattr(cfg, "spawn_edge_buffer_m", 0.30))
    weight = float(getattr(cfg, "surface_weight", 1.0))

    # A CHAIN over `approach_range_m`, not one wide rectangle: the library's own convention, so
    # the approach distribution is the same question on this tile as on a box or up-into-down
    # one. The single rectangle this replaced gave 0.3-2.7 m against the chain's 1.5-4.0, and
    # a third of episodes opened with under a metre before the first riser.
    # The chain reads `approach_range_m` off the cfg directly, so hand it the RESOLVED
    # interval rather than requiring every caller's config to declare one.
    areas = _ground_spawn_chain(first_obstacle_x, cy, 0.0,
                                SimpleNamespace(approach_range_m=_approach_range(cfg)), weight)
    if descending:
        # The chain is authored on the ground; going down, the approach is the LOBBY.
        for a in areas:
            a["z"] = spawn_z
    # Plus one on the LANDING, the same reasoning the drawn courses use: the far flight is
    # otherwise reached only by an episode that already cleared the near one, so it is rare
    # exactly where it is hardest.
    hx = landing_len / 2.0 - buf
    hy = w_wide / 2.0 - buf
    if hx <= 0.0 or hy <= 0.0:
        raise ValueError(
            f"tile {terrain_id}: a {buf:.2f} m spawn buffer leaves no usable landing -- it is "
            f"{landing_len:.2f} m deep and {w_wide:.2f} m wide.")
    areas.append(spawn_area((x_land0 + x_land1) / 2.0, cy, z_landing, hx, hy, weight=weight))

    goal_x = goal_from_x + RUNOUT_M + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0
    # GOAL_HALF_WIDTH_M, never half the corridor. The band reaches the runtime twice -- as this
    # tile's `route.goal_half_width` and as the ONE `default.half_width_m` of the library's
    # goal-regions spec -- and a spec is a single number for the whole library, so a tile that
    # sizes its own band to its own corridor makes the two disagree with nothing to reconcile
    # them. Half this hallway is 1.27 m against the drawn courses' 0.75, a 1.7x mismatch in the
    # direction that scores a robot standing beside the goal as arrived.
    goal_xy = (goal_x, cy)
    start_x = min(a["x"] - a["half_x"] for a in areas if abs(a["z"] - spawn_z) < 1e-9)
    route = _route((max(start_x, 0.0), cy), spawn_z, 0.0, goal_xy,
                   _fit_goal_half_width(GOAL_HALF_WIDTH_M, overlays, goal_xy),
                   goal_z, spawn_areas=areas)
    family = ENTRY_STAIRS_DOWN_FAMILY if descending else ENTRY_STAIRS_UP_FAMILY
    tile = _base_tile(terrain_id, family, size_x, size_y, overlays, route)
    tile["entry_stairs"] = {
        **g,
        "descending": bool(descending),
        "x_wide_flight": [x_wide0, x_wide1],
        "x_landing": [x_land0, x_land1],
        "x_hall_flight": [x_hall0, x_hall1],
        "x_lobby": [x_lobby0, x_lobby1],
        "approach_m": APPROACH_M,
        "approach_range_m": list(_approach_range(cfg)),
        "clearing_len_m": CLEARING_LEN_M,
        "clearing_width_m": CLEARING_WIDTH_M,
        "wall_height_m": WALL_H_M,
        "wall_collidable": True,
        "runout_m": RUNOUT_M,
        "goal_rect_centre_x": goal_x + GOAL_DEPTH_M / 2.0,
        "goal_depth_m": GOAL_DEPTH_M,
        "spawn_areas": len(areas),
        "ground_spawn_areas": len(areas) - 1,
    }
    return tile
