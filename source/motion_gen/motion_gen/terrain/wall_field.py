# Copyright 2026 Zachary Olkin. All rights reserved.

"""The WALL-FIELD course: one large flat tile with walls scattered across it.

    from motion_gen.terrain.wall_field import WallFieldCfg, generate_wall_manifest
    manifest = generate_wall_manifest(24, WallFieldCfg())

**What this course is for.** The root ``CLAUDE.md`` is explicit that nothing in this stack
avoids obstacles and that "the robot walked at a wall" is partly CORRECT behaviour. What IS a
real fault is planning a JUMP or a CLIMB at one -- a terrain-CLASSIFICATION failure, where the
generator reads a vertical face as a climbable box. This course trains that out, and only that:
every goal is reachable in a STRAIGHT LINE without crossing a wall, so the policy is never asked
to route around anything. The walls are there to be seen and ignored.

That is why the corridor is a hard constraint of the draw rather than a reward term. A course
whose goal required going around a wall would be asking for a planner this stack deliberately
does not have, and the fine-tune would grade a failure the policy cannot act on.

**The walls are AXIS-ALIGNED, and that is a loader constraint, not a taste.** ``_overlay``
writes ``yaw`` as an explicit ``0.0`` because the RL loader hard-refuses a non-zero one -- it
would be axis-aligned in Isaac and rotated in MuJoCo, so the two simulators would disagree about
the world. A wall therefore runs along x or along y and the randomness is in which, where, how
long and how tall.

**The walls are COLLIDABLE** (``kind="obstacle"``, ``walkable_top=False``, ``decor=False``),
matching :func:`freeform.make_stairs_tile`'s stairwell walls. ``WallFieldCfg.wall_collidable``
flips the whole field to the visual-only channel in one line. Four consequences, all of which
that function already documents and which apply here unchanged:

* the generator was FITTED on non-colliding decoration -- every wall in its training library is
  ``decor=True`` -- so the robot has never been pushed back on by one;
* ``undesired_contacts`` is **-0.5** effective (``G1TeacherEnvCfg.__post_init__`` overwrites the
  declared -0.1) against a ``clf_reward`` of 10, and its sensor list is thighs / torso /
  backpack / elbows -- **no wrists and no hands**, so an arm swept into a wall costs nothing;
* a collidable wall IS visible to the height scan, where a decor one is not. The privileged
  observation changes too, so a scan-conditioned number measured here is not comparable to one
  measured on a decor-walled library;
* the reference is generated, not followed from a clip, so a blocked limb shows up as tracking
  error against a reference the terrain will not permit. :data:`WALL_CORRIDOR_HALF_WIDTH_M` is
  what keeps that rare, and it is sized for the arm sweep rather than for the torso.

**The depth cameras see the walls either way, and the height scan is what changes.** The
cameras ray-cast ``/World/ground`` AND ``/World/decor`` (``g1_depth_env_cfg``'s
``mesh_prim_paths``); the height scanner is a ``RayCaster`` pointed at ``/World/ground`` alone.
So the visual-only channel is fully visible to the generator -- which is the observation this
course is about -- and collidability buys physics and the scan, not perception.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .freeform import (
    FREEFORM_SPAWN_JITTER_M,
    GOAL_HALF_WIDTH_FLOOR_M,
    SECTION_MARGIN_M,
    SPAWN_INSET_M,
    _base_tile,
    _fit_goal_half_width,
    _ground_overlay,
    _overlay,
    _route,
    spawn_area,
)

#: Family tag. Distinct from every other family so no report, threshold or fingerprint
#: comparison can pool a wall tile with a box or a staircase.
WALLS_FAMILY: str = "wall_field"

#: The commanded forward-speed band (m/s), sampled per EPISODE by the runtime.
#:
#: The band the OTHER FINE-TUNE ARMS ACTUALLY RUN -- boxes, updown, indist and multi all declare
#: [0.4, 2.5] in their configs -- so a wall arm and a box arm are commanded the same way and a
#: difference between them is the terrain.
#:
#: This said "the varied courses' band" and quoted 0.6-2.5, which is the CONSTANT in
#: ``varied_courses.py`` but not what any arm was run at: the yaml overrides it at build time.
#: Comparing against the module constant rather than against the shipped configs is how the
#: claim came to be false while reading as careful.
#:
#: With the approach controller on, this is a CRUISE CEILING rather than an arrival speed --
#: the command decays to the floor and then to zero regardless of what was drawn. The runtime
#: draws per EPISODE from the env's own ``freeform_speed_min/max`` and knows nothing about which
#: tile it landed on, so a per-family band would be a claim it cannot honour.
COMMANDED_SPEED_RANGE_MPS: tuple[float, float] = (0.4, 2.5)

#: Half-width of the protected corridor from spawn to goal (m).
#:
#: No wall may intrude within this distance of the straight spawn-to-goal segment, which is what
#: makes "reachable without crossing a wall" a property of the DRAW rather than something the
#: policy has to discover. Sized for the ARM, not the torso: the reference sweeps the wrists
#: well outside the footprint, and with no wrist in ``undesired_contacts`` a swept arm inside a
#: solid wall is not penalised -- it just fails to track. 1.10 m clears the widest reference arm
#: extent with margin at the 0.15 m wall thickness.
WALL_CORRIDOR_HALF_WIDTH_M: float = 1.10

#: Radius of the DWELL disc the robot has to stop inside and hold (m). 2 m diameter.
#:
#: **Declared here rather than in the task cfg because three things have to agree about it**:
#: this course's goal SPEC (``goal_region_spec`` below, which is what the offline grading's
#: ``in_goal`` reads), the ``freeform_goal_dwell`` TERMINATION that ends the episode, and the
#: debug MARKER that draws it. ``g1_finetune_env_cfg.WALLS_GOAL_RADIUS_M`` imports this, and the
#: marker reads the radius back off the termination -- so all three trace to this line. Three
#: independent copies of a region is exactly the shape of drift this repo has paid for before.
#:
#: It must stay >= ``G1WallsFinetuneEnvCfg``'s stop radius, or the command zeroes before the
#: robot is inside the region it has to hold and there is a band where it is told to stand
#: still and scored as not yet arrived. ``freeform_goal_dwell.__init__`` refuses that pairing.
#:
#: **A wider disc is a weaker stopping test, and the near-wall tiles are where that shows.**
#: With the goal 0.5-1.5 m from the far wall, a 1.0 m radius lets the robot satisfy the disc
#: while standing up to 1.0 m SHORT of the goal, i.e. up to 2.5 m from the wall. It still has
#: to stop and hold, which is the behaviour being trained; it just does not have to stop
#: precisely. Narrow it if the fine-tune learns to halt early rather than to halt close.
DWELL_RADIUS_M: float = 1.0

#: Seconds the robot must hold the disc, consecutively, for the episode to count as a success.
#:
#: Declared beside the radius so the goal SPEC this module writes and the TERMINATION the task
#: wires cannot disagree; ``g1_finetune_env_cfg`` imports both.
DWELL_S: float = 5.0

#: Goal band half-extent stamped into every tile's route and into the goal spec (m).
#:
#: Tied to :data:`DWELL_RADIUS_M` rather than to
#: ``freeform.GOAL_HALF_WIDTH_FLOOR_M`` (0.75, what every traverse course uses), so the region
#: the offline grading scores and the region the episode turns on are the same size. They are
#: still different SHAPES -- ``in_goal`` is a downrange half-plane crossed with a lateral band,
#: the dwell test is a disc -- and that difference is not cosmetic; see the grading gap recorded
#: in ``G1WallsFinetuneEnvCfg``. Matching the number removes the one part of the discrepancy
#: that is gratuitous.
#:
#: ``freeform._validate_route`` checks the band against the tile's walkable ground at
#: ``max(declared, GOAL_HALF_WIDTH_FLOOR_M)``, so widening it tightens that check rather than
#: loosening it -- the floor spans the whole room, so it passes, but a future narrower room
#: would be caught here rather than at runtime.
GOAL_HALF_WIDTH_M: float = DWELL_RADIUS_M

#: Clear ground kept around the spawn rectangle (m).
SPAWN_CLEARANCE_M: float = 2.0


@dataclass
class WallFieldCfg:
    """Draw parameters for the wall field.

    Attributes:
        seed: Base seed; every tile derives its own stream from this and its id.
        room_x_range_m: Interior length of the room along the tile's x (m), drawn per tile.
        room_y_range_m: Interior width of the room (m), drawn per tile.
        perimeter_height_range_m: Height of the four enclosing walls (m).
        near_wall_fraction: Fraction of tiles whose goal is placed close to the room's FAR
            wall, so the robot has to stop in front of it. The rest put the goal in open floor
            with :attr:`open_goal_clearance_m` of space, so the course is not exclusively a
            stopping-at-a-wall drill and the two cases can be reported apart.
        near_wall_gap_range_m: Gap from the goal centre to the far wall's inner face on a
            near-wall tile (m).
        open_goal_clearance_m: Clear floor kept around the goal on the other tiles (m).
        n_walls_range: Inclusive range on the wall count per tile.
        wall_len_range_m: Wall length along its own axis (m).
        wall_thickness_m: Wall thickness (m). Fixed rather than drawn: thickness is the one
            wall dimension the far camera cannot resolve at range, so varying it would add a
            parameter no observation can see.
        wall_height_range_m: Wall height (m). The floor is deliberately above the tallest
            trained box (the varied box family tops out near 0.6 m) so that "this is not a box"
            is decidable from the image rather than a coin flip.
        goal_bearing_max_deg: Maximum |bearing| of the ROUTE within the tile frame (deg). This
            rotates the whole spawn-goal ray relative to the field's axes, and since the walls
            are axis-aligned it is what makes them meet the route at varied angles. It is NOT
            the steering difficulty -- see :attr:`spawn_heading_error_deg` for that.
        goal_bearing_min_deg: Minimum |bearing| (deg).
        spawn_heading_error_deg: The robot's initial heading error off its own route (deg),
            drawn per tile and signed. THIS is what the yaw controller has to remove.
            ``mdp.events.resets.reset_on_freeform_section`` takes the spawn heading from
            ``frame_yaw + area.yaw`` and never from ``route.start_yaw``, so the spawn can be
            misaligned while the goal still sits exactly on the route -- which
            ``freeform._validate_route`` requires, and for a good reason: a goal off its own
            route would have the steering aim down one line while the termination watched
            another, and read as a policy that walks past its goal.
        wall_min_gap_m: Minimum gap between two walls (m), so a draw cannot fuse several into
            one long barrier that the corridor then has to thread.
        wall_lateral_range_m: Signed |lateral offset| of a wall centre from the route
            centreline (m). The floor is raised to just outside the corridor at draw time, so
            setting it below :attr:`corridor_half_width_m` is harmless rather than a trap. The
            ceiling is what keeps walls inside useful camera range -- see the placement comment
            in :func:`make_wall_tile`.
        wall_along_margin_m: How far before the spawn and past the goal walls may be drawn (m).
        corridor_half_width_m: Protected corridor half-width (m); see
            :data:`WALL_CORRIDOR_HALF_WIDTH_M`.
        wall_collidable: Solid terrain when True; the visual-only ``decor`` channel when False.
            See the module docstring for what changes.
        spawn_half_extent_m: Half-extents of the spawn rectangle (m).
            **:data:`freeform.FREEFORM_SPAWN_JITTER_M`, and not a number of this course's
            own.** A declared spawn area REPLACES ``pos_jitter_m`` for the whole tile, so an
            area sized to anything else silently changes the spawn distribution relative to
            every other arm -- the terrain CLAUDE.md records box2 paying this at 4x, and this
            course shipped 0.35 against everyone else's 0.25 (1.4x the initial cross-track
            error) until it was measured. It is also what makes a ground-only draw
            bit-identical to the pre-spawn-area libraries the earlier numbers came from.
        max_draw_attempts: Rejection-sampling budget per wall before the tile settles for
            however many it placed.
        salts: Per-purpose RNG salts, so adding a draw does not shift an unrelated one. The
            wall salt is used as ``salts["wall"] + k`` for the k-th wall, so the others sit far
            above any reachable ``k`` -- at the original spacing ``wall + 16`` collided with
            ``spawn`` and a tile with 17+ walls would have correlated a wall draw with the
            heading-error draw.
    """

    seed: int = 0
    room_x_range_m: tuple[float, float] = (11.0, 17.0)
    room_y_range_m: tuple[float, float] = (8.0, 13.0)
    perimeter_height_range_m: tuple[float, float] = (1.8, 2.8)
    near_wall_fraction: float = 0.5
    near_wall_gap_range_m: tuple[float, float] = (0.5, 1.5)
    open_goal_clearance_m: float = 2.0
    n_walls_range: tuple[int, int] = (3, 7)
    wall_len_range_m: tuple[float, float] = (2.0, 6.0)
    wall_thickness_m: float = 0.15
    wall_height_range_m: tuple[float, float] = (1.2, 2.6)
    goal_bearing_max_deg: float = 18.0
    goal_bearing_min_deg: float = 0.0
    spawn_heading_error_deg: tuple[float, float] = (8.0, 28.0)
    wall_min_gap_m: float = 0.8
    wall_lateral_range_m: tuple[float, float] = (1.4, 4.0)
    wall_along_margin_m: float = 2.0
    corridor_half_width_m: float = WALL_CORRIDOR_HALF_WIDTH_M
    wall_collidable: bool = True
    spawn_half_extent_m: tuple[float, float] = FREEFORM_SPAWN_JITTER_M
    max_draw_attempts: int = 400
    salts: dict = field(default_factory=lambda: {"tile": 11, "goal": 23, "wall": 37,
                                                 "spawn": 9001, "room": 9101})


def tile_rng(cfg: WallFieldCfg, terrain_id: int, salt: int = 0) -> np.random.Generator:
    """A per-tile, per-purpose random stream.

    Keyed on ``(seed, terrain_id, salt)`` so a tile's geometry depends only on its id -- adding
    a tile to the library cannot change an existing one, which is what makes a holdout drawn
    from a different seed a comparable sample rather than a reshuffle.

    Args:
        cfg: The course config.
        terrain_id: The tile id.
        salt: A per-purpose offset from :attr:`WallFieldCfg.salts`.

    Returns:
        The generator.
    """
    return np.random.default_rng((int(cfg.seed), int(terrain_id), int(salt)))


def _segment_box_gap(p0: tuple[float, float], p1: tuple[float, float],
                     cx: float, cy: float, hx: float, hy: float) -> float:
    """Shortest distance from the segment ``p0->p1`` to an axis-aligned box, 0 when they meet.

    Exact for an axis-aligned box: the distance from a point to the box is the norm of the
    per-axis overshoot, so the segment's distance is the minimum of that over the segment. It is
    evaluated by dense sampling rather than in closed form because the closed form for a
    segment-vs-rectangle distance has several cases and this runs a few hundred times per tile.

    Sampling is safe HERE and would not be in general: the walls are metres long and the
    corridor is over a metre wide, so a 2 cm sample step cannot step over an intrusion. The step
    is derived from the segment length rather than fixed, so a longer route does not sample
    coarser.

    Args:
        p0: Segment start (m).
        p1: Segment end (m).
        cx: Box centre x (m).
        cy: Box centre y (m).
        hx: Box half-extent in x (m).
        hy: Box half-extent in y (m).

    Returns:
        The distance in metres; ``0.0`` when the segment touches or enters the box.
    """
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(2, int(length / 0.02) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = p0[0] + t * (p1[0] - p0[0])
    ys = p0[1] + t * (p1[1] - p0[1])
    dx = np.maximum(np.abs(xs - cx) - hx, 0.0)
    dy = np.maximum(np.abs(ys - cy) - hy, 0.0)
    return float(np.min(np.hypot(dx, dy)))


def _boxes_overlap(a: tuple[float, float, float, float],
                   b: tuple[float, float, float, float], gap: float) -> bool:
    """Whether two axis-aligned boxes are closer than ``gap``.

    Args:
        a: ``(cx, cy, hx, hy)`` of the first box (m).
        b: ``(cx, cy, hx, hy)`` of the second (m).
        gap: Required clear separation (m).

    Returns:
        True when the boxes are within ``gap`` of each other on both axes.
    """
    return (abs(a[0] - b[0]) < a[2] + b[2] + gap) and (abs(a[1] - b[1]) < a[3] + b[3] + gap)


def make_wall_tile(terrain_id: int, cfg: WallFieldCfg) -> dict:
    """One enclosed room: flat floor, four perimeter walls, interior walls, one goal inside.

    The draw order is GOAL FIRST, then walls rejected against it. The corridor is the invariant
    this course rests on, and making it a post-hoc search would let a dense draw fail to find
    one and quietly emit a tile whose goal is unreachable in a straight line -- exactly the case
    the policy has no machinery for.

    Half the tiles (``near_wall_fraction``) put the goal a drawn 0.5-1.5 m in front of the
    room's FAR wall, which is the whole point of the course: the robot has to stop rather than
    walk into it. The far wall is used rather than an authored stop-wall because the room
    already has one and it is already perpendicular-ish to a route drawn within +-18 deg --
    an authored perpendicular wall could not be axis-aligned at a non-zero bearing, and the
    loader refuses a rotated overlay.

    Args:
        terrain_id: The tile id.
        cfg: The course config.

    Returns:
        The manifest entry, carrying a ``wall_field`` record of every drawn quantity.

    Raises:
        ValueError: If the drawn room cannot hold the route with its clearances, or a config
            range is not a positive ordered pair.
    """
    if cfg.goal_bearing_min_deg > cfg.goal_bearing_max_deg:
        raise ValueError("goal_bearing_min_deg must not exceed goal_bearing_max_deg; got "
                         f"({cfg.goal_bearing_min_deg}, {cfg.goal_bearing_max_deg}).")
    half_t = float(cfg.wall_thickness_m) / 2.0
    #: The floor extends a wall thickness plus a margin past the interior on every side, so the
    #: perimeter walls stand ON ground rather than overhanging the tile edge.
    pad = float(cfg.wall_thickness_m) + SECTION_MARGIN_M

    rrng = tile_rng(cfg, terrain_id, cfg.salts["room"])
    room_x = float(rrng.uniform(*cfg.room_x_range_m))
    room_y = float(rrng.uniform(*cfg.room_y_range_m))
    wall_h = float(rrng.uniform(*cfg.perimeter_height_range_m))
    x0, y0 = pad, pad                      # interior lower corner
    x1, y1 = pad + room_x, pad + room_y    # interior upper corner
    size_x, size_y = x1 + pad, y1 + pad

    grng = tile_rng(cfg, terrain_id, cfg.salts["goal"])
    spawn_x = x0 + SPAWN_INSET_M + float(cfg.spawn_half_extent_m[0])
    spawn_y = (y0 + y1) / 2.0
    spawn = (spawn_x, spawn_y)

    # ORDER MATTERS: the goal's lateral offset is `dist * sin(bearing)` and `dist` is set by
    # how far down the room the goal goes, so a bearing drawn before the room is consulted can
    # put the goal inside a side wall. The bearing is therefore CLAMPED to what this room can
    # hold rather than drawn and then rejected -- a rejection here would raise on a perfectly
    # reasonable small room, which it did on the holdout seed.
    #
    # THE CLAMP ITSELF BIASES THE BEARING, and saying otherwise was wrong: compressing
    # U[0, max] into U[0, b_geom] pulls the mean down (8.3 deg against the 9.0 a uniform draw
    # would give, clamped on ~40% of tiles) and it does so UNEVENLY between the two goal
    # kinds -- `lat_room` is 1.0 m on a near-wall tile and 2.0 m on an open one, so near-wall
    # tiles keep more of their bearing range and are steered harder. The two sub-populations
    # this course reports apart therefore differ in steering difficulty as well as in goal
    # placement. `bearing_was_clamped` and `bearing_max_deg_geometric` are recorded per tile so
    # a report can condition on it; none does yet.
    near_wall = bool(grng.random() < float(cfg.near_wall_fraction))
    gap = float(grng.uniform(*cfg.near_wall_gap_range_m)) if near_wall else float("nan")
    lat_room = 1.0 if near_wall else float(cfg.open_goal_clearance_m)
    reach_x = (x1 - (gap if near_wall else float(cfg.open_goal_clearance_m))) - spawn_x
    if reach_x <= 1.0:
        raise ValueError(
            f"tile {terrain_id}: room is {room_x:.1f} m long, which leaves {reach_x:.2f} m "
            "between the spawn and the goal. Raise room_x_range_m.")
    lat_budget = max(0.0, (room_y / 2.0) - lat_room)
    # ATAN, not asin. At bearing b the goal sits at `dist = reach_x / cos(b)`, so its lateral
    # offset is `reach_x * tan(b)` -- using asin under-corrects by `1/cos(b) - 1` (5.1% at
    # 18 deg), the guard below then raises, and `generate_wall_manifest` has no per-tile
    # recovery so one bad tile kills the build. Measured over 300 seeds before the fix: 35 of
    # 7165 tiles raised, hitting ~12% of seeds.
    b_geom = math.degrees(math.atan(lat_budget / reach_x))
    b_max = min(float(cfg.goal_bearing_max_deg), b_geom)
    b_min = min(float(cfg.goal_bearing_min_deg), b_max)
    mag = float(grng.uniform(b_min, b_max))
    bearing = math.radians(mag) * (1.0 if grng.random() < 0.5 else -1.0)
    ux, uy = math.cos(bearing), math.sin(bearing)

    if near_wall:
        # Stop a drawn gap short of the far wall's INNER face.
        dist = (x1 - gap - spawn_x) / max(ux, 1e-6)
    else:
        reach = reach_x / max(ux, 1e-6)
        lo = min(5.0, max(2.0, 0.55 * reach))
        dist = float(grng.uniform(lo, reach)) if reach > lo else reach
    goal = (spawn_x + dist * ux, spawn_y + dist * uy)

    if not (y0 + lat_room <= goal[1] <= y1 - lat_room) or dist <= 1.0:
        raise ValueError(
            f"tile {terrain_id}: goal {goal} at {math.degrees(bearing):+.1f} deg still does "
            f"not fit the room's y extent [{y0:.2f}, {y1:.2f}] with {lat_room:.2f} m of "
            "clearance after the bearing clamp -- the clamp is wrong, not the draw.")

    srng = tile_rng(cfg, terrain_id, cfg.salts["spawn"])
    err_mag = float(srng.uniform(*cfg.spawn_heading_error_deg))
    heading_err = math.radians(err_mag) * (1.0 if srng.random() < 0.5 else -1.0)

    overlays = [_ground_overlay(size_x, size_y)]
    decor = not bool(cfg.wall_collidable)

    def _wall(cx: float, cy: float, sx: float, sy: float, h: float) -> dict:
        """One wall overlay in this tile's chosen collision channel."""
        return _overlay(cx, cy, 0.0, sx, sy, h, kind="obstacle",
                        walkable_top=False, decor=decor)

    # The four enclosing walls, centred ON the interior boundary so their inner faces are the
    # interior. The x-running pair is grown by a thickness at each end to close the corners.
    span_x = room_x + 2.0 * float(cfg.wall_thickness_m)
    perim = [
        _wall(x0 - half_t, (y0 + y1) / 2.0, float(cfg.wall_thickness_m), room_y, wall_h),
        _wall(x1 + half_t, (y0 + y1) / 2.0, float(cfg.wall_thickness_m), room_y, wall_h),
        _wall((x0 + x1) / 2.0, y0 - half_t, span_x, float(cfg.wall_thickness_m), wall_h),
        _wall((x0 + x1) / 2.0, y1 + half_t, span_x, float(cfg.wall_thickness_m), wall_h),
    ]
    overlays.extend(perim)

    # Interior walls, drawn in the ROUTE's frame so they line the corridor at camera range
    # rather than scattering into the corners -- what limits exposure is angular size, not
    # range (see the terrain CLAUDE.md), and a wall in a far corner is a few pixels.
    nx, ny = -uy, ux
    corridor = float(cfg.corridor_half_width_m)
    lat_lo = max(float(cfg.wall_lateral_range_m[0]), corridor + half_t + 0.05)
    lat_hi = max(lat_lo + 0.1, float(cfg.wall_lateral_range_m[1]))
    #: Interior walls keep away from the goal so the only thing near it is the intended far
    #: wall. On an open-goal tile that is the full clearance; on a near-wall tile it is a
    #: metre, which still leaves the drawn 0.5-1.5 m gap to the perimeter as the close feature.
    goal_keepout = float(cfg.open_goal_clearance_m) if not near_wall else 1.0
    placed: list[tuple[float, float, float, float]] = []
    walls: list[dict] = []
    n_target = int(grng.integers(cfg.n_walls_range[0], cfg.n_walls_range[1] + 1))
    for k in range(n_target):
        wrng = tile_rng(cfg, terrain_id, cfg.salts["wall"] + k)
        for _ in range(int(cfg.max_draw_attempts)):
            along_x = bool(wrng.random() < 0.5)
            length = float(wrng.uniform(*cfg.wall_len_range_m))
            height = float(wrng.uniform(*cfg.wall_height_range_m))
            hx, hy = (length / 2.0, half_t) if along_x else (half_t, length / 2.0)
            s_along = float(wrng.uniform(-float(cfg.wall_along_margin_m),
                                         dist + float(cfg.wall_along_margin_m)))
            d = float(wrng.uniform(lat_lo, lat_hi)) * (1.0 if wrng.random() < 0.5 else -1.0)
            cx = spawn[0] + s_along * ux + d * nx
            cy = spawn[1] + s_along * uy + d * ny
            # Inside the room, clear of the perimeter by one min-gap.
            if not (x0 + hx + cfg.wall_min_gap_m <= cx <= x1 - hx - cfg.wall_min_gap_m):
                continue
            if not (y0 + hy + cfg.wall_min_gap_m <= cy <= y1 - hy - cfg.wall_min_gap_m):
                continue
            if _segment_box_gap(spawn, goal, cx, cy, hx, hy) < corridor:
                continue
            if _segment_box_gap(spawn, spawn, cx, cy, hx, hy) < SPAWN_CLEARANCE_M:
                continue
            if _segment_box_gap(goal, goal, cx, cy, hx, hy) < goal_keepout:
                continue
            if any(_boxes_overlap((cx, cy, hx, hy), p, float(cfg.wall_min_gap_m))
                   for p in placed):
                continue
            placed.append((cx, cy, hx, hy))
            walls.append({"cx": cx, "cy": cy, "size_x": 2.0 * hx, "size_y": 2.0 * hy,
                          "height_m": height, "along": "x" if along_x else "y"})
            overlays.append(_wall(cx, cy, 2.0 * hx, 2.0 * hy, height))
            break

    areas = [spawn_area(spawn_x, spawn_y, 0.0,
                        float(cfg.spawn_half_extent_m[0]), float(cfg.spawn_half_extent_m[1]),
                        yaw=bearing + heading_err, weight=1.0)]
    route = _route(spawn, 0.0, bearing, goal,
                   _fit_goal_half_width(GOAL_HALF_WIDTH_M, overlays, goal), 0.0,
                   spawn_areas=areas)
    tile = _base_tile(terrain_id, WALLS_FAMILY, size_x, size_y, overlays, route)
    tile["wall_field"] = {
        "room_x_m": room_x,
        "room_y_m": room_y,
        "perimeter_height_m": wall_h,
        "near_wall_goal": near_wall,
        "goal_wall_gap_m": gap,
        "n_walls_requested": n_target,
        "n_walls_placed": len(walls),
        "goal_distance_m": dist,
        "route_bearing_deg": math.degrees(bearing),
        "bearing_max_deg_geometric": b_geom,
        "bearing_was_clamped": bool(b_geom < float(cfg.goal_bearing_max_deg)),
        "spawn_heading_error_deg": math.degrees(heading_err),
        "corridor_half_width_m": corridor,
        "wall_collidable": bool(cfg.wall_collidable),
        "wall_thickness_m": float(cfg.wall_thickness_m),
        "interior_xy": [x0, y0, x1, y1],
        "walls": [{k: (v if isinstance(v, str) else float(v)) for k, v in w.items()}
                  for w in walls],
    }
    return tile


def generate_wall_manifest(n_tiles: int, cfg: WallFieldCfg | None = None) -> list[dict]:
    """A wall-field manifest with contiguous ``terrain_id``.

    Args:
        n_tiles: How many tiles.
        cfg: The course config; the defaults if omitted.

    Returns:
        The manifest, one entry per tile.

    Raises:
        ValueError: If ``n_tiles`` is not positive.
    """
    cfg = cfg if cfg is not None else WallFieldCfg()
    if n_tiles <= 0:
        raise ValueError(f"a wall-field library needs at least one tile, got {n_tiles}.")
    return [make_wall_tile(i, cfg) for i in range(n_tiles)]


def family_blocks(n_tiles: int) -> dict[str, tuple[int, int]]:
    """The inclusive ``terrain_id`` range the family occupies.

    Args:
        n_tiles: How many tiles.

    Returns:
        ``{family: (lo, hi)}``, empty when there are no tiles.
    """
    return {WALLS_FAMILY: (0, n_tiles - 1)} if n_tiles else {}


def generator_config(n_tiles: int, cfg: WallFieldCfg | None = None) -> dict:
    """The ``generator_config`` block a wall-field library stamps.

    Carries :data:`COMMANDED_SPEED_RANGE_MPS` under the key
    ``clip_io.read_commanded_speed_range`` looks for, so the course owns its command band rather
    than it living in whichever launcher happened to pass an override.

    Args:
        n_tiles: How many tiles.
        cfg: The course config; the defaults if omitted.

    Returns:
        A JSON-serialisable dict.
    """
    cfg = cfg if cfg is not None else WallFieldCfg()
    return {
        # THE BAND MUST ALSO BE STAMPED UNDER "freeform", and this is not redundancy.
        # `clip_io.read_commanded_speed_range` searches a FIXED list of block names
        # ("step_course", "box2", "freeform", "varied_courses", "indist_routes") and knows
        # nothing about a new course's own block. Stamped only under "wall_field" it read back
        # None, and the two consumers failed in OPPOSITE directions: `run_finetune_loop.sh`
        # silently fell back to its own 0.5-2.2 default, while `run_quick_eval.sh`'s
        # `terrain_axis.py --check_speeds` hard-refused with exit 2. `multi_courses` stamps
        # both blocks for exactly this reason.
        "freeform": {
            "commanded_speed_range_mps": [float(v) for v in COMMANDED_SPEED_RANGE_MPS],
            "section_counts": {"wall_field": int(n_tiles)},
        },
        "wall_field": {
            "commanded_speed_range_mps": [float(v) for v in COMMANDED_SPEED_RANGE_MPS],
            "seed": int(cfg.seed),
            "n_tiles": int(n_tiles),
            "family_blocks": {k: list(v) for k, v in family_blocks(n_tiles).items()},
            "room_x_range_m": [float(v) for v in cfg.room_x_range_m],
            "room_y_range_m": [float(v) for v in cfg.room_y_range_m],
            "near_wall_fraction": float(cfg.near_wall_fraction),
            "near_wall_gap_range_m": [float(v) for v in cfg.near_wall_gap_range_m],
            "n_walls_range": [int(v) for v in cfg.n_walls_range],
            "wall_height_range_m": [float(v) for v in cfg.wall_height_range_m],
            "wall_collidable": bool(cfg.wall_collidable),
            "corridor_half_width_m": float(cfg.corridor_half_width_m),
            "wall_lateral_range_m": [float(v) for v in cfg.wall_lateral_range_m],
            "route_bearing_deg_range": [float(cfg.goal_bearing_min_deg),
                                        float(cfg.goal_bearing_max_deg)],
            "spawn_heading_error_deg": [float(v) for v in cfg.spawn_heading_error_deg],
        }
    }


def goal_region_spec(n_tiles: int, cfg: WallFieldCfg | None = None) -> dict:
    """The DECLARED ``goal_regions.json`` spec for a wall-field library.

    Declared from the authored course rather than fitted from clip-arm episodes: this is a
    manifest-only library with no clips to fit from, and on an authored course the goal region
    is a property of the geometry.

    Args:
        n_tiles: How many tiles.
        cfg: The course config; the defaults if omitted.

    Returns:
        A JSON-serialisable spec.
    """
    cfg = cfg if cfg is not None else WallFieldCfg()
    return {
        "default": {
            "approach_margin_m": 0.0,
            "half_width_m": float(GOAL_HALF_WIDTH_M),
            # NOT 0. `height_tol_m` names two quantities with opposite conventions at zero:
            # `terminations.freeform_goal_reached` guards its height clause with
            # `if height_tol_m > 0`, so 0 disables it, while `goal_region.in_goal` applies
            # `|height_err| <= height_tol_m` unconditionally -- an exact-equality test that
            # floating point never satisfies. Written as 0.0 once on the varied courses and it
            # cost a whole arm: every arrival recorded as a non-arrival over 8400 episodes AND
            # the progress reward's success term was identically zero for 5 iterations.
            #
            # The ground here is flat everywhere, so 0.5 m is pure slack on posture: a robot
            # arriving upright reads ~0 and one arriving in a crouch still counts.
            "height_tol_m": 0.5,
            # THE DWELL RULE. Declaring these makes `eval.accumulate` score `reached_goal` as
            # "held the disc for dwell_s" instead of "was ever inside the half-plane", which is
            # what puts the offline grading on the same definition as the runtime termination.
            # Without them the reward would pay for crossing the goal LINE and walking on into
            # the wall -- the exact behaviour this course exists to remove.
            #
            # Both must match `G1WallsFinetuneEnvCfg`: the radius is DWELL_RADIUS_M, which that
            # cfg imports, and the seconds are restated here because the spec is a data file
            # the env cannot write into. `test_wall_dwell_contract.py` pins the pair.
            "dwell_radius_m": float(DWELL_RADIUS_M),
            "dwell_s": float(DWELL_S),
        },
        "by_family": {},
        "provenance": {
            "declared": True,
            "source": "motion_gen.terrain.wall_field",
            "note": (
                "DECLARED from the authored course, not fitted: a manifest-only library has no "
                "clips to fit from. half_width_m is GOAL_HALF_WIDTH_M, which every tile is "
                "BUILT to and which g1_finetune_env_cfg floors at the same number, so the "
                "clamp in freeform_goal_reached is a no-op -- though that term is INERT on "
                "this course (see G1WallsFinetuneTerminationCfg), which wires the dwell rule "
                "below instead. half_width_m is tied to DWELL_RADIUS_M so the region the "
                "grading scores and the region the episode turns on are the same size; they "
                "remain different SHAPES, a half-plane and a disc, which is why the dwell "
                "fields are declared rather than relying on half_width_m alone."),
            "goal_half_width_m": float(GOAL_HALF_WIDTH_M),
            "spawn_heading_error_deg": [float(v) for v in cfg.spawn_heading_error_deg],
            "families": {k: list(v) for k, v in family_blocks(n_tiles).items()},
        },
    }
