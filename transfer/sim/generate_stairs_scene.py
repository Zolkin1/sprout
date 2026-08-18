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

**Width is not a continuum.**  The current library's stair steps fall into two clusters and
nothing in between: the TRANSITION families (``stairs_up`` / ``stairs_down``, a narrow flight
between two platforms) are 0.80-1.16 m wide, and the PURE families are 5.00-5.29 m -- the
full tile.  The old defaults here (1.5 / 2.2 / 3.0 m) sat in the gap, matching no trained
tile, because they came from ``clips/g1_terrain_v2.h5``'s ``width_range == [1.0, 3.0]`` and
that library is long gone.

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
TRAINED_RISERS = (0.13, 0.15, 0.17)
TRAINED_TREADS = (0.22, 0.25, 0.28)

# Lateral step WIDTH, measured from the same manifest (clips/g1_mb_v5_body.h5, 2026-08-13),
# over every box overlay with a tread-sized x extent. Two disjoint clusters, per family:
#
#     stairs_up      109 steps   0.80 - 1.04 m   (p50 0.89)
#     stairs_down    105 steps   0.80 - 1.16 m   (p50 0.86)
#     stairs_up_pure 383 steps   5.00 - 5.29 m   (p50 5.11)
#     stairs_dn_pure 483 steps   5.10 - 5.28 m   (p50 5.20)
#
# The transition flights are the narrow ones because they are a staircase set INTO a tile
# between two platforms, which is the case a robot crossing a terrain transition sees; the
# pure tiles are stairs spanning the whole tile. A lane between the two clusters is
# out-of-distribution for every trained tile, which is what _warn_off_grid reports.
TRAINED_STAIR_WIDTHS: tuple[tuple[float, float], ...] = ((0.80, 1.16), (5.00, 5.29))
#: Median transition-flight width -- the default for the lane the robot faces at spawn.
TRAINED_TRANSITION_WIDTH = 0.90
DEFAULT_RISE = 0.15  # step height (m) -- modal riser
DEFAULT_RUN = 0.25   # step depth / tread run (m) -- modal tread

# Box (jump-on / jump-off) geometry, measured from the same manifest (50 box tiles):
#     height in {0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70}
#     footprint  depth_x 0.73-0.80 m, width_y 0.43-0.45 m  (one box per tile)
TRAINED_BOX_HEIGHTS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70)
TRAINED_BOX_DEPTH = 0.75   # x extent (m)
TRAINED_BOX_WIDTH = 0.44   # y extent (m)

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


# One lane per corner of the trained (riser, tread) grid plus the modal pair straight ahead,
# so a single scene exercises shallow-deep, modal and steep-shallow treads.
DEFAULT_LANES: tuple[Lane, ...] = (
    Lane(name="wide", y=4.0, width=3.0, steps=8, rise=0.13, run=0.28),
    # The lane the robot faces at spawn, at the trained TRANSITION width. It was 2.2 m,
    # which is roughly two and a half times the flight the policy ever climbed in Isaac --
    # a foot could land anywhere across it, so nothing made the robot stay on the treads.
    Lane(name="center", y=0.0, width=TRAINED_TRANSITION_WIDTH, steps=6, rise=0.15, run=0.25),
    Lane(name="narrow", y=-3.5, width=1.5, steps=4, rise=0.17, run=0.22),
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


def build_scene_xml(robot_xml: str, lanes: tuple[Lane, ...], rise: float, run: float,
                    x_start: float, platform_depth: float,
                    boxes: tuple[Box, ...] = (), descend: bool = True) -> str:
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

    Returns:
        The complete scene XML as a string.
    """
    stair_lines: list[str] = []
    for lane in lanes:
        stair_lines.extend(_lane_geoms(lane, rise, run, x_start, platform_depth, descend))
    for box in boxes:
        stair_lines.extend(_box_geoms(box))
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
              model: str = "g1_terrain_scene") -> str:
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

    Returns:
        The complete scene XML as a string.
    """
    return f"""<mujoco model="{model}">
    <!-- {header} -->
    <include file="{robot_xml}"/>

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
    parser.add_argument("--robot", default="g1_21j",
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
            print(f"[WARN] lane '{ln.name}' width {ln.width} m falls BETWEEN the trained "
                  f"clusters {TRAINED_STAIR_WIDTHS} m (narrow transition flights vs "
                  "full-tile pure staircases) -- no trained tile looks like this.")

    xml = build_scene_xml(f"{args.robot}.xml", lanes, args.rise, args.run,
                          args.x_start, args.platform_depth, boxes, args.descend)

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
        print(f"       - {lane.name}: width {lane.width} m, {lane.steps} steps"
              f"{' up + ' + str(lane.steps) + ' down' if args.descend else ''}, "
              f"rise {r} m / run {rn} m (top {lane.steps * r:.3f} m) at y={lane.y}; "
              f"x span {args.x_start:.2f}-{args.x_start + span:.2f} m")
    for b in boxes:
        print(f"       - {b.name}: height {b.height} m, {b.depth}x{b.width} m at "
              f"(x={b.x}, y={b.y})")


if __name__ == "__main__":
    main()
