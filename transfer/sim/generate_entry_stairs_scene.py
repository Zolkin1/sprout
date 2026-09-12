# Copyright 2026 Zachary Olkin. All rights reserved.
"""A REAL building entrance, measured on site: a wide flight, a landing, then a narrow one.

Two flights the robot must cross in sequence, with the corridor NARROWING between them --
which is the part no generated course in this repo has, and the part a real doorway always
has. Dimensions were taken with a tape measure and are given in INCHES here because that is
how they were measured; the module converts once, at the top, so no number is restated in two
units (that is how a 10 mm camera-mount error survived in this repo for weeks).

    approach  ->  5 x (6.00" rise, 11.00" run), 200" wide      -> +30.00"
              ->  200" landing, 200" wide
              ->  the walls close in to a 100" hallway
              ->  3 x (5.25" rise, 13.00" run), 100" wide      -> +15.75"
              ->  an 8 x 8 m walled indoor clearing

Why this course is interesting and not just another staircase:

* **The two flights disagree on both dimensions**, and neither matches the other's ratio --
  6.00/11.00 against 5.25/13.00. The trained library samples a single rise/run per flight, so
  a course that changes it mid-traverse asks the generator to re-read the geometry rather
  than carry the first flight's answer into the second.
* **The narrowing is the real test.** 200" -> 100" is a hard lateral constraint arriving
  immediately before a riser, and `terrain-dialed-clutter` records that a vertical surface
  within camera range is what the generator misreads as a box. Here it cannot avoid one.
* **The second flight is SHALLOWER and DEEPER.** 5.25" rise on a 13.00" tread is close to
  flat, and the failure mode worth watching is the robot planning a step-up it does not need.

Walls are SOLID by default, unlike mjeval's stair walls: the objection there is
``ee_pose_error`` firing on a blocked wrist, a termination the fine-tune task does not carry
(see ``step_courses.UPDOWN_WALL_COLLIDABLE``). Pass ``--visual_walls`` for decor-only.

Usage::

    python generate_entry_stairs_scene.py                    # writes the scene, prints its name
    python generate_entry_stairs_scene.py --visual_walls
"""

from __future__ import annotations

import argparse
import os

from generate_stairs_scene import DECOR_GROUP_ATTRS, scene_xml

#: Metres per inch.  Converted ONCE; every dimension below stays in the units it was measured
#: in, and nothing downstream sees an inch.
IN: float = 0.0254

# --- the wide lower flight (the one in the photograph) --------------------------------
WIDE_STEPS: int = 5
WIDE_RISE_IN: float = 6.00
WIDE_RUN_IN: float = 11.00
WIDE_WIDTH_IN: float = 200.0

# --- the landing between the flights --------------------------------------------------
#: Half the measured 200", by request: it puts the narrowing and the upper flight closer to
#: the top of the lower one, so the generator gets less flat ground to settle on between two
#: flights that disagree about their geometry. That is the harder and more building-like case.
LANDING_LEN_IN: float = 100.0

# --- the narrow upper flight (behind the doors; the doors themselves are not modelled) --
HALL_STEPS: int = 3
HALL_RISE_IN: float = 5.25
HALL_RUN_IN: float = 13.00
HALL_WIDTH_IN: float = 100.0

#: Flat ground before the first riser (m).  Room to reach commanded speed, and for the far
#: camera to see the flight before the feet are on it.
#:
#: 5.5 rather than the 3.0 it started at, and the number comes from the ISAAC side: the tile
#: there spawns over a chain covering ``approach_range_m`` (up to 4.0 m back from the first
#: riser) whose farthest area sits 1.25 m in, so anything under 5.25 m hangs the chain off the
#: tile.  Held equal here by ``test_entry_stairs_matches_mjeval.py`` -- the two simulators show
#: the same building standing on the same amount of ground, and a robot driven at this
#: staircase in MuJoCo gets the run-up its Isaac counterpart was trained with.
APPROACH_M: float = 5.5
#: The indoor CLEARING the hallway opens into, at the top level (m).  A lobby: the hallway
#: constrains the robot laterally and then releases it, which is the opposite transition to
#: the narrowing and is the one a building actually makes on the way in.
#:
#: Given in metres rather than inches because, unlike the flights and the landing, it was not
#: measured -- it is a room-sized space chosen to be clearly bigger than the 200" section, so
#: "the walls opened up" is unambiguous to a depth camera rather than a marginal widening.
CLEARING_LEN_M: float = 8.0
CLEARING_WIDTH_M: float = 8.0
#: Wall height (m).  Waist-high is enough to be a lateral constraint and to fill the near
#: camera; the far camera's frame cuts a wall near this height at these ranges anyway.
WALL_H_M: float = 1.2
#: Wall thickness (m).
WALL_T_M: float = 0.15


def _step_pillars(name: str, x0: float, y: float, width: float, steps: int,
                  rise: float, run: float, z0: float) -> list[str]:
    """One flight, as solid pillars from the GROUND to each tread's own top.

    Pillars rather than slabs, matching ``box_clip.stair_tile_boxes`` and
    ``generate_stairs_scene._lane_geoms``: a riser face is then a real vertical surface with
    no gap under the nose for a toe to catch, and the height scan reads the tread rather than
    seeing through it.

    Args:
        name: Geom name prefix.
        x0: Downrange edge of the FIRST riser (m).
        y: Lateral centre (m).
        width: Tread width (m).
        steps: Number of risers.
        rise: Per-step height (m).
        run: Tread depth (m).
        z0: Height of the ground this flight starts from (m).

    Returns:
        MuJoCo ``<geom>`` lines.
    """
    out: list[str] = []
    for n in range(1, steps + 1):
        top = z0 + n * rise
        out.append(
            f'        <geom name="{name}_step{n}" type="box" material="box_material" '
            f'group="2" size="{run / 2:.4f} {width / 2:.4f} {top / 2:.4f}" '
            f'pos="{x0 + (n - 0.5) * run:.4f} {y:.4f} {top / 2:.4f}"/>')
    return out


def _slab(name: str, x0: float, x1: float, y: float, width: float, top: float) -> list[str]:
    """A walkable slab from the ground up to ``top``, spanning ``x0..x1``."""
    return [
        f'        <geom name="{name}" type="box" material="box_material" group="2" '
        f'size="{(x1 - x0) / 2:.4f} {width / 2:.4f} {top / 2:.4f}" '
        f'pos="{(x0 + x1) / 2:.4f} {y:.4f} {top / 2:.4f}"/>']


def _wall(name: str, x0: float, x1: float, y: float, z0: float, height: float,
          thickness: float, solid: bool) -> list[str]:
    """A wall running along x, centred laterally at ``y``, standing on ``z0``."""
    attrs = 'group="2"' if solid else DECOR_GROUP_ATTRS
    return [
        f'        <geom name="{name}" type="box" material="box_material" {attrs} '
        f'size="{(x1 - x0) / 2:.4f} {thickness / 2:.4f} {height / 2:.4f}" '
        f'pos="{(x0 + x1) / 2:.4f} {y:.4f} {z0 + height / 2:.4f}"/>']


def _return_face(name: str, x: float, y0: float, y1: float, z0: float, height: float,
                 thickness: float, solid: bool) -> list[str]:
    """A wall running along y -- the face that closes the wide section down to the hallway."""
    attrs = 'group="2"' if solid else DECOR_GROUP_ATTRS
    return [
        f'        <geom name="{name}" type="box" material="box_material" {attrs} '
        f'size="{thickness / 2:.4f} {(y1 - y0) / 2:.4f} {height / 2:.4f}" '
        f'pos="{x:.4f} {(y0 + y1) / 2:.4f} {z0 + height / 2:.4f}"/>']


def geometry(solid_walls: bool = True) -> tuple[list[str], dict[str, float]]:
    """Every geom of the course, plus the landmarks a caller may want to print.

    Returns:
        ``(geom lines, landmarks)``.
    """
    w_rise, w_run, w_wide = WIDE_RISE_IN * IN, WIDE_RUN_IN * IN, WIDE_WIDTH_IN * IN
    h_rise, h_run, h_wide = HALL_RISE_IN * IN, HALL_RUN_IN * IN, HALL_WIDTH_IN * IN
    landing = LANDING_LEN_IN * IN

    x_wide0 = APPROACH_M
    x_wide1 = x_wide0 + WIDE_STEPS * w_run
    z_landing = WIDE_STEPS * w_rise
    x_land1 = x_wide1 + landing
    x_hall1 = x_land1 + HALL_STEPS * h_run
    z_top = z_landing + HALL_STEPS * h_rise
    x_end = x_hall1 + CLEARING_LEN_M

    g: list[str] = []
    # The wide lower flight, then the landing it rises onto.
    g += _step_pillars("wide", x_wide0, 0.0, w_wide, WIDE_STEPS, w_rise, w_run, 0.0)
    g += _slab("landing", x_wide1, x_land1, 0.0, w_wide, z_landing)
    # The narrow upper flight, standing on the landing.
    g += _step_pillars("hall", x_land1, 0.0, h_wide, HALL_STEPS, h_rise, h_run, z_landing)
    # The clearing the hallway opens into. One slab, wider than the hallway on both sides.
    g += _slab("clearing", x_hall1, x_end, 0.0, CLEARING_WIDTH_M, z_top)

    # **The narrowing.** Two return faces closing 200" down to 100", placed at the landing's
    # far edge where the hallway begins -- so the lateral constraint arrives immediately
    # before the upper flight's first riser, which is what the real doorway does.
    for sgn, side in ((1.0, "left"), (-1.0, "right")):
        y_in, y_out = sgn * h_wide / 2.0, sgn * w_wide / 2.0
        g += _return_face(f"narrow_{side}", x_land1 - WALL_T_M / 2.0,
                          min(y_in, y_out), max(y_in, y_out),
                          z_landing, WALL_H_M, WALL_T_M, solid_walls)
        # The hallway's side walls stop AT the clearing -- running them further would make
        # the room a corridor and delete the widening this course was extended to include.
        g += _wall(f"hall_wall_{side}", x_land1 - WALL_T_M, x_hall1,
                   sgn * (h_wide / 2.0 + WALL_T_M / 2.0),
                   z_landing, WALL_H_M, WALL_T_M, solid_walls)

    # The clearing's own perimeter, so it reads as INDOORS rather than as a plateau: side
    # walls, a far wall, and a near wall in two segments with the hallway's mouth left open
    # between them. Standing on the clearing floor, not on the ground.
    y_room = CLEARING_WIDTH_M / 2.0
    for sgn, side in ((1.0, "left"), (-1.0, "right")):
        g += _wall(f"room_wall_{side}", x_hall1, x_end,
                   sgn * (y_room + WALL_T_M / 2.0), z_top, WALL_H_M, WALL_T_M, solid_walls)
        # the near wall's segment on this side of the opening
        g += _return_face(f"room_near_{side}", x_hall1 + WALL_T_M / 2.0,
                          min(sgn * h_wide / 2.0, sgn * y_room),
                          max(sgn * h_wide / 2.0, sgn * y_room),
                          z_top, WALL_H_M, WALL_T_M, solid_walls)
    g += _return_face("room_far", x_end - WALL_T_M / 2.0, -y_room, y_room,
                      z_top, WALL_H_M, WALL_T_M, solid_walls)

    marks = {"x_wide_first_riser": x_wide0, "x_landing": x_wide1, "z_landing": z_landing,
             "x_hall_first_riser": x_land1, "x_hall_top": x_hall1, "z_top": z_top,
             "x_end": x_end, "wide_width_m": w_wide, "hall_width_m": h_wide,
             "clearing_w": CLEARING_WIDTH_M}
    return g, marks


def main() -> None:
    """Write the scene beside the other generated ones and print what it contains."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="g1_perceptive")
    p.add_argument("--scene", default="entry_stairs_scene")
    p.add_argument("--out", default=None)
    p.add_argument("--visual_walls", action="store_true",
                   help="Author the walls as DECOR (no collider). Default is SOLID: the "
                        "objection to solid walls is `ee_pose_error` firing on a blocked "
                        "wrist, which the fine-tune task does not have.")
    a = p.parse_args()

    geoms, m = geometry(solid_walls=not a.visual_walls)
    header = (f"Measured building entrance. Wide flight {WIDE_STEPS} x "
              f"{WIDE_RISE_IN}in rise / {WIDE_RUN_IN}in run at {WIDE_WIDTH_IN}in wide; "
              f"{LANDING_LEN_IN}in landing; then {HALL_STEPS} x {HALL_RISE_IN}in / "
              f"{HALL_RUN_IN}in in a {HALL_WIDTH_IN}in hallway. Walls "
              f"{'DECOR' if a.visual_walls else 'SOLID'}. Generated by "
              f"generate_entry_stairs_scene.py; do not edit by hand.")
    xml = scene_xml(f"{a.robot}.xml", header, "\n".join(geoms),
                    model=f"g1_{a.scene}")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "robots", "g1", f"{a.robot}_{a.scene}.xml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(xml)
    print(f"[entry] wrote {out}")
    print(f"[entry]   wide flight  x {m['x_wide_first_riser']:.3f} -> {m['x_landing']:.3f} m, "
          f"{WIDE_STEPS} x {WIDE_RISE_IN * IN:.4f} m, {m['wide_width_m']:.3f} m wide")
    print(f"[entry]   landing      x {m['x_landing']:.3f} -> {m['x_hall_first_riser']:.3f} m "
          f"at z {m['z_landing']:.4f} m")
    print(f"[entry]   hallway      x {m['x_hall_first_riser']:.3f} -> {m['x_hall_top']:.3f} m, "
          f"{HALL_STEPS} x {HALL_RISE_IN * IN:.4f} m, {m['hall_width_m']:.3f} m wide "
          f"(narrowed from {m['wide_width_m']:.3f})")
    print(f"[entry]   clearing     x {m['x_hall_top']:.3f} -> {m['x_end']:.3f} m at "
          f"z {m['z_top']:.4f} m, {m['clearing_w']:.2f} m wide, walled with the hallway "
          f"mouth open")


if __name__ == "__main__":
    main()
