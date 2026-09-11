# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate a MuJoCo sim2sim staircase scene for the G1 humanoid.

Emits a scene XML (``robots/g1/<robot>_<scene>.xml``) that mirrors
``g1_21j_basic_scene.xml`` -- same lights / ground plane / assets / ankle sensors --
but adds a handful of staircases. Each lane is **symmetric**: it climbs to a flat
platform and then descends back to the floor on the far side, so it can be walked up
one side and down the other (and looks the same entered from either end) instead of
dead-ending on a plateau. ``--no_descent`` gives the old ascend-only shape.
The scene is the sim2sim counterpart of the ``G1-teacher-clf-terrain`` training
terrain: an *in-distribution playground* (not tied to any one clip) for driving the
flow-matching generator + RL tracker over stairs with a joystick and watching the
pelvis height scan ride the treads.

Geometry is pinned to the training terrain library (see the measured constants below).
What varies between staircases is the lateral tread **width** and the step **count**.

**Width comes in THREE clusters with two wide holes between them.**  The TRANSITION families
(``stairs_up`` / ``stairs_down``, a flight set between two platforms) split into 12 tiles at
0.81-1.10 m and 28 at 1.72-4.14 m -- ``motion_gen.terrain.decorate`` widened the latter -- and
the PURE families are 5.00-5.29 m, the full tile.  A lane at 1.4 m or at 4.5 m matches no
trained tile.  The constants below say so, and they described the pre-decoration library until
2026-08-27; see :data:`TRAINED_STAIR_WIDTHS` for the measurement.

Each staircase is built from full-height box "pillars" (step ``n`` spans ``z in
[0, n*rise]``), so the assembled solid has clean tread tops and vertical risers with
no gaps for a foot to catch.  Every terrain geom is placed in **collision group 2**:
that is the group the sim2sim height-scan raycast filters on
(``Robot.compute_height_scan`` casts only against group 2), and it matches the ground
plane's group so the robot's default-collision foot contacts still land on the treads.

Run from ``transfer/sim/`` (so the output lands next to the other scenes)::

    python generate_stairs_scene.py                      # -> robots/g1/g1_21j_terrain_scene.xml
    python generate_stairs_scene.py --robot g1_21j_compute
    python generate_stairs_scene.py --scene my_stairs --lane 2.5:6 --lane 1.2:4

Then drive it with the generator + tracker runner::

    python gen_tracker_runner.py --mode joystick --scene terrain_scene --load_run <run>
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

# Training terrain step geometry, MEASURED from the MB library the current policy trains on
# (clips/g1_mb_v1_nodistract.h5 terrain_manifest, 70 stair tiles):
#     riser  in {0.13, 0.15, 0.17}      tread in {0.22, 0.25, 0.28}      n_steps 3..39
# The defaults below are the MODAL pair. The previous defaults (0.135 / 0.233) came from the
# older DP library clips/g1_terrain_v2.h5 and sat BETWEEN grid points, i.e. slightly
# out-of-distribution for every trained tile.
TRAINED_RISERS = (0.13, 0.131, 0.15, 0.17)
#: 0.209 was MISSING here and is not a rounding of 0.22: four stairs_up transition tiles
#: (1, 3, 8, 15) carry it, and it is the tread the clip library walks at HALF speed --
#: 0.32 m/s against 0.61-0.69 on every other tread. So the one geometry whose reference
#: behaves differently was the one the playground could not build.
#: UP and DOWN staircases use DIFFERENT tread sets, and the old constant listed only the UP
#: one.  Measured identically on g1_mb_v5_body.h5, g1_mb_v5_body_mode11.h5 and the decorated
#: library -- so 0.178 / 0.19 did not arrive with any of those passes, they were always there
#: and were simply never written down:
#:
#:     stairs_up, stairs_up_pure     0.209  0.22  0.221  0.25  0.28
#:     stairs_down, stairs_dn_pure   0.178  0.19  0.22   0.25
#:
#: A descending flight is therefore steeper per tread than any ascending one the robot has
#: seen, which a probe sweeping ``--run`` should know before calling 0.19 out-of-distribution.
TRAINED_TREADS = (0.178, 0.19, 0.209, 0.22, 0.221, 0.25, 0.28)
#: The ascending half of that split, for a probe that only builds up-staircases.
TRAINED_TREADS_UP = (0.209, 0.22, 0.221, 0.25, 0.28)

# Lateral step WIDTH, measured from the manifest of the library the deployed pair trains on
# (clips/g1_mb_v5_body_mode11_decor.h5, 2026-08-27), as the narrowest flight in each tile:
#
#     stairs_up       20 tiles   0.825 - 4.141 m   (p50 1.961)
#     stairs_down     20 tiles   0.809 - 3.918 m   (p50 2.070)
#     stairs_up_pure  15 tiles   4.998 - 5.286 m   (p50 5.115)
#     stairs_dn_pure  15 tiles   5.097 - 5.282 m   (p50 5.200)
#
# STILL THREE CLUSTERS, not the two the old constants named and not the continuum the first
# correction of them claimed.  Over the 40 transition tiles the sorted widths are
#
#     0.809 .. 1.102   (12 tiles)   <- 0.621 m HOLE ->   1.723 .. 4.141   (28 tiles)
#
# and the pure staircases sit alone above 4.998, past a second 0.857 m hole.  `decorate.py`
# widened the flight on 28 of the 40, not on the 13 `wide_stairs` alone: the `matched` arm
# (15 tiles) widens it too, and only `narrow_landing` (12) leaves it where the generator put
# it.  Those 12 ARE the lower cluster -- verified tile by tile against the pre-decoration
# library, where every transition flight was 0.801-1.156 m (p50 0.863).
#
# The old constants here read 0.80-1.16 m, which is that pre-decoration library to three
# decimals.  They were a hand-kept copy of a measurement the decoration pass then moved:
# everything from 1.72 to 4.14 m was being flagged as matching no trained tile while sitting
# in the larger of the two transition clusters.  Same defect shape as the stale ROS URDF and
# the stale metadata fixture.
TRAINED_STAIR_WIDTHS: tuple[tuple[float, float], ...] = ((0.80, 1.11), (1.72, 4.15),
                                                         (5.00, 5.29))
#: Median of the WIDENED transition cluster -- the middle of what a decorated flight looks like.
TRAINED_TRANSITION_WIDTH = 1.96
#: REAL staircase widths, tape-measured off buildings on 2026-08-27 -- not from any library.
#: This is the playground's authority, because the playground exists to be driven at things
#: the robot will actually meet:
#:
#:     4 ft (1.22 m)   the narrowest found anywhere
#:     5 ft (1.52 m)   the common indoor case
#:     7 ft (2.13 m)   outdoor, and upward from there
#:
#: **The narrowest REAL staircase is 40% wider than the library's narrowest trained flight**
#: (0.81 m), and 4-5 ft lands squarely inside the library's 0.62 m hole (1.10 - 1.72 m), where
#: no stair tile exists at all. So the two most common real widths are extrapolation for this
#: generator, and `_warn_off_grid` will say so on every lane below -- correctly, and on
#: purpose. That is the finding, not a misconfiguration.
REAL_STAIR_WIDTHS: dict[str, float] = {"min": 1.22, "indoor": 1.52, "outdoor": 2.44}

#: Median of the NARROW transition cluster (the 12 `narrow_landing` tiles, 0.809-1.102 m).
#: The median rather than the minimum on purpose: 0.809 is one tile, and a probe wants the
#: shape the robot met repeatedly. This is the end where a flight is barely wider than the
#: robot, which is the interesting end for a stairs-read-as-box question.
TRAINED_NARROW_FLIGHT_WIDTH = 0.86

DEFAULT_RISE = 0.15  # step height (m) -- modal riser
DEFAULT_RUN = 0.25   # step depth / tread run (m) -- modal tread

# Box (jump-on / jump-off) geometry, re-measured on clips/g1_mb_v5_body_mode11_decor.h5
# (2026-08-27), over the leading walkable overlay of each box tile:
#
#     jump_on         20 tiles   depth_x 0.46      width_y 0.90-4.90   height 0.25-0.75
#     jump_off        15 tiles   depth_x 0.48      width_y 0.95-3.63   height 0.25-0.75
#     jump_off_front  15 tiles   depth_x 0.48-3.79 width_y 0.81-7.30   height 0.25-0.75
#
# The previous constants (0.75 x 0.44 m footprint) were measured on a different library AND
# had the two axes the other way round: the trained box presents a SHALLOW face across a WIDE
# span, not a deep narrow block.  That matters here specifically -- the depth of the face is
# what a forward camera sees as the obstacle, and a 0.75 m one is nothing the robot met.
# Changing these numbers moves nothing on disk; ``robots/g1/g1_perceptive_terrain_scene.xml``
# keeps the geometry it was generated with until someone reruns this script.
#: A 0.05 m grid from 0.25 to 0.75.  Three tiles sit off it -- 0.453 (jump_off), 0.459
#: (jump_off_front), 0.469 (jump_on), one each -- and they are NOT folded in: rounding them
#: to a shared 0.47 would invent a grid point no tile has and make the off-grid warning below
#: pass for a height nothing was trained on.
TRAINED_BOX_HEIGHTS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75)
TRAINED_BOX_DEPTH = 0.46   # x extent (m), the face the robot approaches
TRAINED_BOX_WIDTH = 2.59   # y extent (m), the jump_on median

# Default playground: three staircases spanning the training width range, with the
# medium one straight ahead of the spawn (y = 0) and the wide / narrow ones off to
# the sides so a joystick operator can steer to whichever width they want to test.
DEFAULT_X_START = 2.5      # flat approach (m) before the first riser (robot spawns at x = 0)
DEFAULT_PLATFORM_DEPTH = 2.0  # flat top platform depth (m) past the last tread


@dataclass
class Lane:
    """One staircase lane: a run of ascending treads plus a flat top platform.

    Attributes:
        name: Short label used to name the emitted geoms (``<name>_step<n>`` etc.).
        y: Lateral centre of the lane in world metres (robot spawns at ``y = 0``).
        width: Lateral tread width (m); the training terrain samples this in ``[1, 3]``.
        steps: Number of steps (training samples ``[4, 8]``).
    """

    name: str
    y: float
    width: float
    steps: int
    rise: float | None = None   # None -> the scene-wide --rise
    run: float | None = None    # None -> the scene-wide --run


# One lane per corner of the trained (riser, tread) grid, at REAL widths.
#
# Every lane is now at least 4 ft, because the playground had drifted into measuring a robot
# against staircases that do not exist: the lane straight ahead of spawn was 0.90 m (2.95 ft)
# and `slow_tread` 0.86 m (2.82 ft), both NARROWER than the narrowest real staircase found,
# and the one realistic lane was called `narrow` while being the second widest of the four.
# Hand-labelled probe trials (`stair_probe.py`) put the cost of that plainly: at 0.86 m the
# generator planned a JUMP on 3 of 3 approaches, and across 1.22 / 1.52 / 2.13 / 2.44 m it
# planned stairs on 11 of 12. Driving the old centre lane was driving the one width where the
# training library says a raised thing is more likely a box than a staircase.
#
# The four (riser, tread) pairs are unchanged -- shallow-deep, modal, steep-shallow and the
# slow-gait tread -- so the scene still spans that grid; only the widths and the names moved.
DEFAULT_LANES: tuple[Lane, ...] = (
    # Straight ahead of spawn: the NARROWEST staircase that actually exists. This is the hard
    # case now, and unlike 0.86 m it is a hard case the robot will meet.
    Lane(name="center", y=0.0, width=REAL_STAIR_WIDTHS["min"], steps=6,
         rise=0.15, run=0.25),
    Lane(name="indoor", y=-3.5, width=REAL_STAIR_WIDTHS["indoor"], steps=4,
         rise=0.17, run=0.22),
    Lane(name="outdoor", y=4.0, width=REAL_STAIR_WIDTHS["outdoor"], steps=8,
         rise=0.13, run=0.28),
    # The SLOW-GAIT geometry, which the other three lanes cannot express. Measured over the
    # library: tread 0.209 / riser 0.131 is walked at 0.32 m/s, while every tread of 0.22 and
    # above is walked at 0.61-0.69 -- a 2x split with nothing in between, across a geometry
    # sweep that is otherwise continuous. Its width used to be tile 15's 0.86 m; at a real
    # width the tread is the only thing that makes it special, which is the point of it.
    Lane(name="slow_tread", y=8.0, width=REAL_STAIR_WIDTHS["indoor"], steps=6,
         rise=0.131, run=0.209),
)


@dataclass
class Box:
    """One free-standing box to jump on / off, matching the training box tiles.

    Attributes:
        name: Label for the emitted geom.
        x: World x (m) of the box centre.
        y: World y (m) of the box centre.
        height: Top surface height (m); the training grid is ``TRAINED_BOX_HEIGHTS``.
        depth: x extent (m).
        width: y extent (m).
    """

    name: str
    x: float
    y: float
    height: float
    depth: float = TRAINED_BOX_DEPTH
    width: float = TRAINED_BOX_WIDTH


# A box lane off to the side: four heights spanning the trained range, spaced far enough
# apart to approach, jump on, and dismount each one independently.
DEFAULT_BOXES: tuple[Box, ...] = tuple(
    Box(name=f"box{i}", x=3.0 + 3.0 * i, y=-6.5, height=h)
    for i, h in enumerate((0.25, 0.40, 0.55, 0.70))
)


def _box_geoms(box: Box) -> list[str]:
    """MuJoCo geom line for one box (a box geom resting on the floor)."""
    return [
        f'        <geom name="{box.name}" type="box" material="box_material" group="2" '
        f'size="{box.depth / 2:.4f} {box.width / 2:.4f} {box.height / 2:.4f}" '
        f'pos="{box.x:.4f} {box.y:.4f} {box.height / 2:.4f}"/>'
    ]


def _lane_geoms(lane: Lane, rise: float, run: float, x_start: float, platform_depth: float,
                descend: bool = True) -> list[str]:
    """Return the MuJoCo ``<geom>`` XML lines for one staircase lane.

    Builds ``lane.steps`` ascending full-height box pillars (step ``n`` occupies
    ``x in [x_start + (n-1)*run, x_start + n*run]`` and ``z in [0, n*rise]``) up to a
    flat top platform flush with the last tread and, when ``descend`` is set, a mirrored
    flight back down to the floor past the far platform edge. The lane is then
    traversable end-to-end (approach -> up -> platform -> down -> floor) and symmetric:
    it presents the same staircase whichever end it is entered from, so both the
    ``stairs_up`` and ``stairs_down`` trained skills get exercised in one pass instead of
    dead-ending on a plateau with a full-height drop off the far edge.

    The descent needs only ``steps - 1`` pillars -- the last step down lands on the floor
    -- so both flights have exactly ``lane.steps`` risers. Descending pillars are
    full-height from ``z = 0`` like the ascending ones, keeping clean tread tops and
    vertical risers with no gaps for a foot to catch.

    All boxes are group 2 so the height-scan ray hits them and the robot's
    default-collision feet rest on them.

    Args:
        lane: The lane geometry (lateral centre, width, step count, optional rise/run).
        rise: Scene-wide per-step height (m); overridden by ``lane.rise``.
        run: Scene-wide per-step depth / tread run (m); overridden by ``lane.run``.
        x_start: World x (m) of the first riser (front of step 1).
        platform_depth: Depth (m) of the flat platform between the two flights.
        descend: Emit the mirrored descending flight past the platform.

    Returns:
        A list of formatted ``<geom .../>`` XML strings for this lane.
    """
    rise = lane.rise if lane.rise is not None else rise
    run = lane.run if lane.run is not None else run
    hy = lane.width / 2.0

    def _pillar(name: str, cx: float, top_z: float) -> str:
        """One full-height tread pillar spanning ``z in [0, top_z]``, centred at ``cx``."""
        return (
            f'        <geom name="{name}" type="box" '
            f'size="{run / 2.0:.4f} {hy:.4f} {top_z / 2.0:.4f}" '
            f'pos="{cx:.4f} {lane.y:.4f} {top_z / 2.0:.4f}" '
            f'group="2" material="box_material"/>'
        )

    flights = f"{lane.steps} up + {lane.steps} down" if descend else f"{lane.steps} up"
    lines: list[str] = [
        f'        <!-- {lane.name} staircase: {flights}, width {lane.width} m -->'
    ]
    for n in range(1, lane.steps + 1):
        lines.append(_pillar(f"{lane.name}_step{n}", x_start + (n - 0.5) * run, n * rise))

    top_z = lane.steps * rise
    x_platform_start = x_start + lane.steps * run
    lines.append(
        f'        <geom name="{lane.name}_platform" type="box" '
        f'size="{platform_depth / 2.0:.4f} {hy:.4f} {top_z / 2.0:.4f}" '
        f'pos="{x_platform_start + platform_depth / 2.0:.4f} {lane.y:.4f} {top_z / 2.0:.4f}" '
        f'group="2" material="box_material"/>'
    )

    if descend:
        x_desc = x_platform_start + platform_depth  # far platform edge = first descending riser
        for m in range(1, lane.steps):
            lines.append(_pillar(f"{lane.name}_down{m}", x_desc + (m - 0.5) * run,
                                 (lane.steps - m) * rise))
    return lines


#: Group and contact attributes for visual-only geometry, matching the tile generator and
#: Isaac's ``/World/decor``: the depth cameras cast group 1, and nothing collides with it.
DECOR_GROUP_ATTRS = 'group="1" contype="0" conaffinity="0"'


#: Lateral gap from a flight's edge to a flanking wall face (m).  A real stairwell wall is
#: essentially at the edge of the treads, which is the case this exists to test.
#:
#: For comparison, measured on the decorated library's 40 stairs transition tiles: the walls
#: ring a RENDERED FLOOR 9.35 m wide while the walkable tile is a median 3.82 m, so there are
#: two populations and only one of them is realistic. The FAR-side wall is 4.98-7.05 m from
#: the flight (median 6.36) -- effectively absent. The NEAR-side wall, where a tile has one,
#: sits 0.0-1.52 m away (median 0.59), which IS stairwell-like -- but only 18 of the 40 tiles
#: have a near-side wall at all. So the trained distribution is roughly half "wall beside the
#: stairs" and half "nothing within five metres", and the value below is deliberately at the
#: tight end of the realistic half rather than at its median.
DEFAULT_LANE_WALL_GAP = 0.15
#: Wall height (m). Above roughly 0.9 m this is a VIEWER choice, not a model one: at 3 m the
#: depth frame's top edge already cuts the wall, so taller walls are byte-identical images.
DEFAULT_LANE_WALL_HEIGHT = 2.0
#: How far before the first riser the wall starts (m), so it is in frame during the approach
#: -- which is when the plan that decides stairs-or-box is made.
DEFAULT_LANE_WALL_LEAD = 1.0


def lane_wall_geoms(lane: Lane, rise: float, run: float, x_start: float,
                    platform_depth: float, descend: bool,
                    gap: float = DEFAULT_LANE_WALL_GAP,
                    height: float = DEFAULT_LANE_WALL_HEIGHT,
                    lead: float = DEFAULT_LANE_WALL_LEAD,
                    thickness: float = 0.15) -> list[str]:
    """Two walls flanking one staircase, the way a real stairwell does.

    **Visual only, with no option to make them solid.**  Isaac's decoration carries no
    collider, so nothing in training has ever pushed back on the robot, and there is no
    obstacle term anywhere in the MDP.  Worse than merely untrained: a collidable wall this
    close would be swept by the reference's own arm swing, which blocks a wrist, fires
    ``ee_pose_error`` and ends the episode for a reason the policy cannot fix.  What this is
    for is watching what the generator PLANS with a wall in frame, which needs no contact.

    The walls span the whole lane -- lead-in, ascent, platform and descent -- because a
    stairwell does, and because the approach is when the deciding plan is made.

    Args:
        lane: The staircase to flank.
        rise: Scene-wide per-step height (m); overridden by ``lane.rise``.
        run: Scene-wide tread run (m); overridden by ``lane.run``.
        x_start: World x of the first riser.
        platform_depth: Depth of the platform between the flights.
        descend: Whether the lane carries a descending flight (it is longer if so).
        gap: Lateral clearance from the tread edge to the wall face (m).
        height: Wall height (m).
        lead: How far before the first riser the wall begins (m).
        thickness: Wall thickness (m).

    Returns:
        MuJoCo geom lines for the two walls.
    """
    rn = lane.run if lane.run is not None else run
    span = lane.steps * rn * (2 if descend else 1) + platform_depth
    x0, x1 = x_start - lead, x_start + span
    cx, half_len = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    # Face at lane edge + gap, so `gap` is the clearance a foot sees, not a centre distance.
    off = lane.width / 2.0 + gap + thickness / 2.0
    return [
        f'        <!-- {lane.name} stairwell wall, {gap} m clearance, visual only -->'
    ] + [
        f'        <geom name="{lane.name}_wall_{side}" type="box" material="box_material" '
        f'{DECOR_GROUP_ATTRS} '
        f'size="{half_len:.4f} {thickness / 2:.4f} {height / 2:.4f}" '
        f'pos="{cx:.4f} {lane.y + sign * off:.4f} {height / 2:.4f}"/>'
        for side, sign in (("left", 1.0), ("right", -1.0))
    ]


def corner_wall_geoms(x: float, y: float, height: float, length: float,
                      thickness: float = 0.15, solid: bool = False) -> list[str]:
    """Two walls meeting at 90 deg, for driving at in the playground.

    One leg runs ACROSS the lane at ``x`` and one runs back along it from that leg's ``+y``
    end, which is the corner an indoor space presents: two surfaces at different ranges in
    one frame, harder than a single flat face.

    **Visual only by default.** Isaac's decoration carries no collider, so nothing in
    training has ever pushed back on the robot, and there is no obstacle term anywhere in
    the MDP. A solid wall would test a contact response the robot does not have and teach
    you about MuJoCo contacts rather than about the generator; what this is for is watching
    the PLAN. ``solid=True`` if you want it to push back anyway.

    Args:
        x: World x of the across-lane leg (m).
        y: World y of its centre (m); the side leg runs back from ``y + length / 2``.
        height: Wall height (m). Barely observable -- the far camera's frame cuts a wall
            near waist height at these ranges -- but it changes what a viewer sees.
        length: Length of each leg (m).
        thickness: Wall thickness (m).
        solid: Emit colliding geometry instead of decoration.

    Returns:
        MuJoCo geom lines.
    """
    attrs = 'group="2"' if solid else DECOR_GROUP_ATTRS
    half = length / 2.0
    return [
        f'        <geom name="play_wall_across" type="box" material="box_material" {attrs} '
        f'size="{thickness / 2:.4f} {half:.4f} {height / 2:.4f}" '
        f'pos="{x:.4f} {y:.4f} {height / 2:.4f}"/>',
        f'        <geom name="play_wall_side" type="box" material="box_material" {attrs} '
        f'size="{half:.4f} {thickness / 2:.4f} {height / 2:.4f}" '
        f'pos="{x - half:.4f} {y + half:.4f} {height / 2:.4f}"/>',
    ]


def build_scene_xml(robot_xml: str, lanes: tuple[Lane, ...], rise: float, run: float,
                    x_start: float, platform_depth: float,
                    boxes: tuple[Box, ...] = (), descend: bool = True,
                    walls: list[str] | None = None,
                    wall_lanes: tuple[str, ...] = (),
                    wall_gap: float = DEFAULT_LANE_WALL_GAP,
                    wall_height: float = DEFAULT_LANE_WALL_HEIGHT) -> str:
    """Assemble the full MuJoCo scene XML string.

    Mirrors ``g1_21j_basic_scene.xml`` (visual / asset / ankle sensors / lights +
    ground plane), including the requested robot model and appending one staircase
    per lane to the worldbody.

    Args:
        robot_xml: Robot model file to ``<include>`` (e.g. ``g1_21j.xml``).
        lanes: Staircase lanes to build.
        rise: Per-step height (m).
        run: Per-step depth / tread run (m).
        x_start: World x (m) of the first riser.
        platform_depth: Depth (m) of each flat platform.
        boxes: Free-standing jump-on/off boxes to add (see :data:`DEFAULT_BOXES`).
        descend: Give every lane a mirrored descending flight past its platform.
        walls: Extra geom lines (see :func:`corner_wall_geoms`), or ``None``.
        wall_lanes: Names of lanes to flank with stairwell walls. Leaving some lanes bare
            makes one scene a within-scene A/B rather than needing two.
        wall_gap: Lateral clearance from tread edge to wall face (m).
        wall_height: Wall height (m).

    Returns:
        The complete scene XML as a string.
    """
    stair_lines: list[str] = []
    for lane in lanes:
        stair_lines.extend(_lane_geoms(lane, rise, run, x_start, platform_depth, descend))
        if lane.name in wall_lanes:
            stair_lines.extend(lane_wall_geoms(lane, rise, run, x_start, platform_depth,
                                               descend, gap=wall_gap, height=wall_height))
    for box in boxes:
        stair_lines.extend(_box_geoms(box))
    stair_lines.extend(walls or [])
    header = (
        f"Generated by generate_stairs_scene.py: in-distribution sim2sim terrain.\n"
        f"         Geometry measured from the MB training library:\n"
        f"         risers {TRAINED_RISERS} m, treads {TRAINED_TREADS} m, widths "
        f"{TRAINED_STAIR_WIDTHS} m, box heights\n"
        f"         {TRAINED_BOX_HEIGHTS} m at {TRAINED_BOX_DEPTH}x{TRAINED_BOX_WIDTH} m "
        f"footprint.\n"
        f"         Scene-wide rise/run = {rise}/{run}; each lane may override "
        f"(see the lane table)."
    )
    return scene_xml(robot_xml, header, "\n".join(stair_lines))


#: The infinite checkered ground plane, in the raycast/collision group (2).
GROUND_PLANE = ('        <geom name="floor" size="0 0 0.05" type="plane" '
                'material="groundplane" group="2"/>')


def scene_xml(robot_xml: str, header: str, body_geoms: str, floor_geom: str = GROUND_PLANE,
              model: str = "g1_terrain_scene", tile_local_origin: tuple = None) -> str:
    """The scene boilerplate every generated sim2sim scene shares.

    Mirrors ``g1_21j_basic_scene.xml`` -- visual settings, assets, ankle-site sensors, light
    -- so a generated scene differs from the hand-written one only in its terrain.  Factored
    out so ``generate_clip_tile_scene.py`` cannot drift from it; the two generators differ
    in what geometry they emit, never in the scene around it.

    Args:
        robot_xml: Robot model file to ``<include>`` (e.g. ``g1_29j.xml``).
        header: Provenance comment body. **Must not contain ``- -`` without the space**:
            XML forbids it in a comment, and while MuJoCo tolerates it, Python's
            ElementTree rejects the whole file -- emitted XML must parse in both.
        body_geoms: Pre-formatted ``<geom .../>`` lines for the worldbody.
        floor_geom: The ground geom line; a finite tile floor replaces the infinite plane.
        model: MuJoCo model name.
        tile_local_origin: ``(x, y, z)`` in the CLIP LIBRARY's tile-local frame that this
            scene put at its world origin, or ``None`` for a scene with no such shift.
            Emitted as a ``<custom><numeric>`` so a consumer can read it off the compiled
            model instead of parsing the provenance comment -- clip playback needs it to
            place a clip's world-frame poses in the scene, and getting it by hand means
            being 8.6 m wrong with no error.

    Returns:
        The complete scene XML as a string.
    """
    origin = ""
    if tile_local_origin is not None:
        origin = ('\n    <custom>\n        <numeric name="tile_local_origin" data="'
                  + " ".join(f"{float(v):.6f}" for v in tile_local_origin)
                  + '"/>\n    </custom>\n')
    return f"""<mujoco model="{model}">
    <!-- {header} -->
    <include file="{robot_xml}"/>
{origin}
    <statistic center="1.0 0.7 1.0" extent="0.8"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="-140" elevation="-20" offwidth="1920" offheight="1080"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        <material name="box_material" shininess="0.1" specular="0.5" roughness="0.3" rgba="0.75 0.6 0.42 1"/>
    </asset>

    <sensor>
        <framepos objtype="site" objname="left_ankle_site" name="left_ankle_pos"/>
        <framepos objtype="site" objname="right_ankle_site" name="right_ankle_pos"/>
    </sensor>

    <worldbody>
        <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
{floor_geom}
{body_geoms}
    </worldbody>
</mujoco>
"""


def _parse_lane(spec: str, index: int) -> Lane:
    """Parse a ``width:steps[:y]`` CLI lane spec into a :class:`Lane`.

    Args:
        spec: ``"width:steps"`` or ``"width:steps:y"`` (metres / count / metres).
        index: Position of this ``--lane`` flag, used to name the lane when no name
            is given (``lane0``, ``lane1``, ...).

    Returns:
        The parsed lane.

    Raises:
        argparse.ArgumentTypeError: If the spec is malformed.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"lane spec '{spec}' must be 'width:steps' or 'width:steps:y'")
    try:
        width = float(parts[0])
        steps = int(parts[1])
        y = float(parts[2]) if len(parts) == 3 else 0.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad lane spec '{spec}': {exc}") from exc
    return Lane(name=f"lane{index}", y=y, width=width, steps=steps)


def _parse_box(spec: str, index: int) -> Box:
    """Parse a ``height[:x][:y]`` box spec (metres)."""
    parts = spec.split(":")
    try:
        height = float(parts[0])
        x = float(parts[1]) if len(parts) > 1 else 3.0 + 3.0 * index
        y = float(parts[2]) if len(parts) > 2 else -6.5
    except (ValueError, IndexError) as exc:
        raise argparse.ArgumentTypeError(f"bad box spec '{spec}': {exc}") from exc
    return Box(name=f"box{index}", x=x, y=y, height=height)


def main() -> None:
    """Parse CLI arguments and write the generated scene XML."""
    parser = argparse.ArgumentParser(description="Generate a MuJoCo G1 sim2sim staircase scene.")
    parser.add_argument("--robot", default="g1_perceptive",
                        help="Robot model to include (e.g. g1_21j, g1_21j_compute). Names the output.")
    parser.add_argument("--scene", default="terrain_scene",
                        help="Scene name; output is robots/g1/<robot>_<scene>.xml.")
    parser.add_argument("--out", default=None,
                        help="Explicit output path (overrides the robots/g1/<robot>_<scene>.xml default).")
    parser.add_argument("--rise", type=float, default=DEFAULT_RISE, help="Step height (m).")
    parser.add_argument("--run", type=float, default=DEFAULT_RUN, help="Step depth / tread run (m).")
    parser.add_argument("--x_start", type=float, default=DEFAULT_X_START,
                        help="Flat approach (m) before the first riser (robot spawns at x=0).")
    parser.add_argument("--platform_depth", type=float, default=DEFAULT_PLATFORM_DEPTH,
                        help="Flat platform depth (m) between the ascending and descending flights.")
    parser.add_argument("--no_descent", dest="descend", action="store_false", default=True,
                        help="Emit ascending stairs only, dead-ending on the top platform. Default "
                             "is symmetric: every lane also gets a mirrored flight back down to the "
                             "floor, so it can be walked up one side and down the other (and is "
                             "identical entered from either end).")
    parser.add_argument("--lane", action="append", dest="lanes", type=str, default=None,
                        help="Add a staircase 'width:steps[:y]' (repeatable). Omit for the default 3 lanes.")
    parser.add_argument("--box", action="append", dest="boxes", type=str, default=None,
                        help="Add a jump box 'height[:x][:y]' in metres (repeatable). Omit for the "
                             f"default row at heights {(0.25, 0.40, 0.55, 0.70)} spanning the "
                             f"trained grid {TRAINED_BOX_HEIGHTS}. Use --no_boxes for none.")
    parser.add_argument("--wall_lanes", type=str, default="center,indoor",
                        help="Comma-separated lane names to flank with STAIRWELL walls, or "
                             "'' for none. Real staircases have walls right at the treads and "
                             "the trained library only half does -- its far-side wall sits "
                             "5-7 m away and fewer than half the tiles carry a near-side one. "
                             "The default walls two lanes and leaves the others bare, so one "
                             "scene is a within-scene A/B instead of two scenes. Visual only, "
                             "always: see lane_wall_geoms for why a solid one would test the "
                             "wrong thing.")
    parser.add_argument("--wall_gap", type=float, default=DEFAULT_LANE_WALL_GAP,
                        help="Clearance from the tread edge to the wall face (m).")
    parser.add_argument("--lane_wall_height", type=float, default=DEFAULT_LANE_WALL_HEIGHT,
                        help="Stairwell wall height (m). Above ~0.9 m this is a viewer choice: "
                             "the depth frame's top edge already cuts the wall at 3 m.")
    parser.add_argument("--wall", action="store_true", default=False,
                        help="Add two walls meeting at 90 deg, for driving at. VISUAL ONLY "
                             "unless --wall_solid: Isaac's decoration never collides and "
                             "there is no obstacle term in the MDP, so a solid one tests a "
                             "contact response the robot does not have. Watch the PLAN.")
    parser.add_argument("--wall_xy", type=float, nargs=2, default=(-5.0, 0.0),
                        metavar=("X", "Y"),
                        help="Where the across-lane leg sits (m). Default is BEHIND the "
                             "spawn: the lanes are at y=+4/0/-3.5 and the boxes at y=-6.5, "
                             "so straight ahead puts the wall on the stairs and confounds "
                             "the two. Behind is flat, clear, and the ground plane is "
                             "infinite -- turn around and drive at it.")
    parser.add_argument("--wall_height", type=float, default=2.0, help="Wall height (m).")
    parser.add_argument("--wall_length", type=float, default=4.0,
                        help="Length of each leg (m).")
    parser.add_argument("--wall_solid", action="store_true", default=False,
                        help="Make the walls collide -- a different experiment from the "
                             "trained one; see --wall.")
    parser.add_argument("--no_boxes", action="store_true", default=False,
                        help="Emit stairs only (no jump boxes).")
    args = parser.parse_args()

    if args.lanes:
        lanes = tuple(_parse_lane(spec, i) for i, spec in enumerate(args.lanes))
    else:
        lanes = DEFAULT_LANES

    if args.no_boxes:
        boxes: tuple[Box, ...] = ()
    elif args.boxes:
        boxes = tuple(_parse_box(spec, i) for i, spec in enumerate(args.boxes))
    else:
        boxes = DEFAULT_BOXES
    for b in boxes:
        if round(b.height, 3) not in TRAINED_BOX_HEIGHTS:
            print(f"[WARN] box '{b.name}' height {b.height} m is OUTSIDE the trained set "
                  f"{TRAINED_BOX_HEIGHTS} -- deliberate OOD test?")
    for ln in lanes:
        r, rn = ln.rise if ln.rise is not None else args.rise, ln.run if ln.run is not None else args.run
        if round(r, 3) not in TRAINED_RISERS or round(rn, 3) not in TRAINED_TREADS:
            print(f"[WARN] lane '{ln.name}' rise/run {r}/{rn} is off the trained grid "
                  f"{TRAINED_RISERS}/{TRAINED_TREADS} -- deliberate OOD test?")
        # Width had no check at all, which is how a 2.2 m centre lane -- wider than any
        # trained flight and narrower than any trained pure staircase -- went unremarked.
        if not any(lo <= ln.width <= hi for lo, hi in TRAINED_STAIR_WIDTHS):
            print(f"[WARN] lane '{ln.name}' width {ln.width} m is OUTSIDE the trained bands "
                  f"{TRAINED_STAIR_WIDTHS} m (decorated transition flights vs full-tile pure "
                  "staircases) -- deliberate OOD test?")

    walls = None
    if args.wall:
        wx, wy = args.wall_xy
        walls = corner_wall_geoms(wx, wy, args.wall_height, args.wall_length,
                                  solid=args.wall_solid)
        print(f"[INFO] corner wall: across-lane leg at x={wx:.2f}, y={wy:.2f}, side leg back "
              f"from y={wy + args.wall_length / 2:.2f}; {args.wall_length:.1f} m legs, "
              f"{args.wall_height:.1f} m tall, "
              f"{'SOLID' if args.wall_solid else 'visual only (non-colliding)'}")
    wall_lanes = tuple(n.strip() for n in args.wall_lanes.split(",") if n.strip())
    unknown = [n for n in wall_lanes if n not in {ln.name for ln in lanes}]
    if unknown:
        # Silently walling nothing would look exactly like walls that do not matter.
        raise SystemExit(f"[ERROR] --wall_lanes names {unknown}, which are not lanes in this "
                         f"scene: {sorted(ln.name for ln in lanes)}")
    xml = build_scene_xml(f"{args.robot}.xml", lanes, args.rise, args.run,
                          args.x_start, args.platform_depth, boxes, args.descend,
                          walls=walls, wall_lanes=wall_lanes, wall_gap=args.wall_gap,
                          wall_height=args.lane_wall_height)

    if args.out is not None:
        out_path = args.out
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, "robots", "g1", f"{args.robot}_{args.scene}.xml")
    with open(out_path, "w") as f:
        f.write(xml)

    # Per lane: `steps` ascending pillars + 1 platform + (steps - 1) descending pillars.
    total_geoms = sum(2 * ln.steps if args.descend else ln.steps + 1 for ln in lanes) + len(boxes)
    profile = "up -> platform -> down (symmetric)" if args.descend else \
              "up -> platform (dead end, --no_descent)"
    print(f"[INFO] Wrote {out_path}")
    print(f"[INFO] {len(lanes)} staircase(s) + {len(boxes)} box(es), {total_geoms} terrain "
          f"geoms (group 2). Scene-wide rise={args.rise} m run={args.run} m; lanes may override.")
    print(f"[INFO] Lane profile: {profile}")
    for lane in lanes:
        # Report each lane's EFFECTIVE geometry, not the scene-wide default -- with per-lane
        # overrides those differ, and the emitted geoms follow the lane.
        r = lane.rise if lane.rise is not None else args.rise
        rn = lane.run if lane.run is not None else args.run
        span = lane.steps * rn * (2 if args.descend else 1) + args.platform_depth
        band = next((f"trained band {lo}-{hi} m" for lo, hi in TRAINED_STAIR_WIDTHS
                     if lo <= lane.width <= hi), "OFF the trained grid")
        walled = ("  WALLED both sides at "
                  f"{args.wall_gap} m" if lane.name in wall_lanes else "  (no walls)")
        print(f"       - {lane.name}: width {lane.width} m ({lane.width / 0.3048:.2f} ft, "
              f"{band}), {lane.steps} steps"
              f"{' up + ' + str(lane.steps) + ' down' if args.descend else ''}, "
              f"rise {r} m / run {rn} m (top {lane.steps * r:.3f} m) at y={lane.y}; "
              f"x span {args.x_start:.2f}-{args.x_start + span:.2f} m{walled}")
    narrowest = min(ln.width for ln in lanes) if lanes else 0.0
    if narrowest < REAL_STAIR_WIDTHS["min"]:
        print(f"[WARN] the narrowest lane is {narrowest:.2f} m "
              f"({narrowest / 0.3048:.2f} ft) -- under the {REAL_STAIR_WIDTHS['min']} m "
              "(4 ft) narrowest REAL staircase measured. Deliberate probe, or a leftover?")
    for b in boxes:
        print(f"       - {b.name}: height {b.height} m, {b.depth}x{b.width} m at "
              f"(x={b.x}, y={b.y})")


if __name__ == "__main__":
    main()
