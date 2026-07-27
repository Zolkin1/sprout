# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate a MuJoCo sim2sim staircase scene for the G1 humanoid.

Emits a scene XML (``robots/g1/<robot>_<scene>.xml``) that mirrors
``g1_21j_basic_scene.xml`` -- same lights / ground plane / assets / ankle sensors --
but adds a handful of ascending staircases, each leading up to a flat top platform.
The scene is the sim2sim counterpart of the ``G1-teacher-clf-terrain`` training
terrain: an *in-distribution playground* (not tied to any one clip) for driving the
flow-matching generator + RL tracker over stairs with a joystick and watching the
pelvis height scan ride the treads.

Geometry is pinned to the training terrain library ``clips/g1_terrain_v2.h5``
(``generator_config.hd_grid == [[0.135, 0.233]]``): every step has a **0.135 m rise**
and a **0.233 m run** (tread depth).  What varies between staircases is the lateral
tread **width** and the step **count** -- exactly the axes the training terrain
randomizes (``width_range == [1.0, 3.0]``, ``num_steps_range == [4, 8]``).

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
    Lane(name="center", y=0.0, width=2.2, steps=6, rise=0.15, run=0.25),
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


def _lane_geoms(lane: Lane, rise: float, run: float, x_start: float, platform_depth: float) -> list[str]:
    rise = lane.rise if lane.rise is not None else rise
    run = lane.run if lane.run is not None else run
    """Return the MuJoCo ``<geom>`` XML lines for one staircase lane.

    Builds ``lane.steps`` full-height box pillars (step ``n`` occupies
    ``x in [x_start + (n-1)*run, x_start + n*run]`` and ``z in [0, n*rise]``) followed
    by a flat top platform flush with the last tread. All boxes are group 2 so the
    height-scan ray hits them and the robot's default-collision feet rest on them.

    Args:
        lane: The lane geometry (lateral centre, width, step count).
        rise: Per-step height (m).
        run: Per-step depth / tread run (m).
        x_start: World x (m) of the first riser (front of step 1).
        platform_depth: Depth (m) of the flat platform past the last tread.

    Returns:
        A list of formatted ``<geom .../>`` XML strings for this lane.
    """
    hy = lane.width / 2.0
    lines: list[str] = [f'        <!-- {lane.name} staircase: {lane.steps} steps, width {lane.width} m -->']
    for n in range(1, lane.steps + 1):
        top_z = n * rise
        cx = x_start + (n - 0.5) * run
        lines.append(
            f'        <geom name="{lane.name}_step{n}" type="box" '
            f'size="{run / 2.0:.4f} {hy:.4f} {top_z / 2.0:.4f}" '
            f'pos="{cx:.4f} {lane.y:.4f} {top_z / 2.0:.4f}" '
            f'group="2" material="box_material"/>'
        )
    top_z = lane.steps * rise
    cx = x_start + lane.steps * run + platform_depth / 2.0
    lines.append(
        f'        <geom name="{lane.name}_platform" type="box" '
        f'size="{platform_depth / 2.0:.4f} {hy:.4f} {top_z / 2.0:.4f}" '
        f'pos="{cx:.4f} {lane.y:.4f} {top_z / 2.0:.4f}" '
        f'group="2" material="box_material"/>'
    )
    return lines


def build_scene_xml(robot_xml: str, lanes: tuple[Lane, ...], rise: float, run: float,
                    x_start: float, platform_depth: float,
                    boxes: tuple[Box, ...] = ()) -> str:
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
        platform_depth: Depth (m) of each flat top platform.
        boxes: Free-standing jump-on/off boxes to add (see :data:`DEFAULT_BOXES`).

    Returns:
        The complete scene XML as a string.
    """
    stair_lines: list[str] = []
    for lane in lanes:
        stair_lines.extend(_lane_geoms(lane, rise, run, x_start, platform_depth))
    for box in boxes:
        stair_lines.extend(_box_geoms(box))
    stairs = "\n".join(stair_lines)
    return f"""<mujoco model="g1_terrain_scene">
    <!-- Generated by generate_stairs_scene.py -- in-distribution sim2sim terrain.
         Geometry measured from the MB training library (clips/g1_mb_v1_nodistract.h5):
         risers {TRAINED_RISERS} m, treads {TRAINED_TREADS} m, box heights
         {TRAINED_BOX_HEIGHTS} m at {TRAINED_BOX_DEPTH}x{TRAINED_BOX_WIDTH} m footprint.
         Scene-wide rise/run = {rise}/{run}; each lane may override (see the lane table). -->
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
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" group="2"/>
{stairs}
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
                        help="Flat top-platform depth (m) past the last tread.")
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

    xml = build_scene_xml(f"{args.robot}.xml", lanes, args.rise, args.run,
                          args.x_start, args.platform_depth, boxes)

    if args.out is not None:
        out_path = args.out
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, "robots", "g1", f"{args.robot}_{args.scene}.xml")
    with open(out_path, "w") as f:
        f.write(xml)

    total_geoms = sum(lane.steps + 1 for lane in lanes) + len(boxes)
    print(f"[INFO] Wrote {out_path}")
    print(f"[INFO] {len(lanes)} staircase(s) + {len(boxes)} box(es), {total_geoms} terrain "
          f"geoms (group 2). Scene-wide rise={args.rise} m run={args.run} m; lanes may override.")
    for lane in lanes:
        # Report each lane's EFFECTIVE geometry, not the scene-wide default -- with per-lane
        # overrides those differ, and the emitted geoms follow the lane.
        r = lane.rise if lane.rise is not None else args.rise
        rn = lane.run if lane.run is not None else args.run
        print(f"       - {lane.name}: width {lane.width} m, {lane.steps} steps, "
              f"rise {r} m / run {rn} m (top {lane.steps * r:.3f} m) at y={lane.y}")
    for b in boxes:
        print(f"       - {b.name}: height {b.height} m, {b.depth}x{b.width} m at "
              f"(x={b.x}, y={b.y})")


if __name__ == "__main__":
    main()
