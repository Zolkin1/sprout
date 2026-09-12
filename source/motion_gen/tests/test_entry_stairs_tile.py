# Copyright 2026 Zachary Olkin. All rights reserved.
"""The measured entry-stairs tile, in both directions.

The cross-simulator agreement test (``transfer/sim/tests/test_entry_stairs_matches_mjeval.py``)
pins the tape-measured DIMENSIONS against the MuJoCo scene. This file pins what the Isaac tile
builds out of them, which is a different question with its own ways of going wrong.

Three failures worth the file, each of which builds, validates and renders perfectly:

* **The goal off the end of the surface.** This is the first course whose traverse ends above
  ground level, and the goal placement it inherits was written for courses that end on the
  floor -- where the run-out is the tile's own ground slab, already there and free. Sized
  independently it put the goal line 0.50 m past the clearing at a ``goal_z`` with no floor
  under it, so ``in_goal`` could never fire and every episode read as a time-out.
* **The run-up too short to run up.** One wide spawn rectangle gave 0.3-2.7 m before the first
  riser against the drawn courses' guaranteed 1.5-4.0, so a third of episodes opened with
  under a metre -- not enough to reach the commanded speed the tile then scores the robot at.
* **Descending built as the mirror image.** A flight of ``n`` risers needs ``n`` treads going
  up and ``n - 1`` coming down, because the last step down lands on the surface already there.
  Mirroring leaves a tread flush with the landing and makes the landing one run deeper.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from motion_gen.terrain.entry_stairs import (
    APPROACH_M,
    CLEARING_LEN_M,
    CLEARING_WIDTH_M,
    ENTRY_STAIRS_FAMILY,
    WALL_H_M,
    geometry_m,
    make_entry_stairs_tile,
)
from motion_gen.terrain.freeform import (
    SECTION_MARGIN_M,
    WALL_T_M,
    validate_manifest,
    walkable_ground_span,
)
from motion_gen.terrain.varied_courses import (
    GOAL_DEPTH_M,
    GOAL_HALF_WIDTH_M,
    RUNOUT_M,
    TARGET_LEAD_M,
    VariedCoursesCfg,
)

TOL = 1e-9
#: Both directions, for everything that must hold of either.
DIRECTIONS = (False, True)


def _cfg(**kw) -> SimpleNamespace:
    """The knobs the tile reads off a course config."""
    return SimpleNamespace(**{"spawn_edge_buffer_m": 0.30, "surface_weight": 1.0, **kw})


def _tile(descending: bool = False, **kw) -> dict:
    """The tile, built the way a library builds it."""
    return make_entry_stairs_tile(0, _cfg(**kw), descending=descending)


def _walls(tile: dict) -> list[dict]:
    """Every collidable wall of the tile."""
    return [ov for ov in tile["overlays"] if ov["kind"] == "obstacle"]


def _surfaces(tile: dict) -> list[dict]:
    """Every walkable overlay, ordered downrange."""
    return sorted((ov for ov in tile["overlays"] if ov["walkable_top"] and not ov["decor"]),
                  key=lambda ov: ov["pos_xy"][0])


def _walkable_tops(tile: dict, x: float, y: float) -> list[float]:
    """Every walkable top face directly under ``(x, y)``, low to high (m)."""
    tops = []
    for ov in tile["overlays"]:
        if ov["decor"] or not ov["walkable_top"]:
            continue
        cx, cy = ov["pos_xy"]
        sx, sy, sz = ov["size_xyz"]
        if (cx - sx / 2 - TOL <= x <= cx + sx / 2 + TOL
                and cy - sy / 2 - TOL <= y <= cy + sy / 2 + TOL):
            tops.append(ov["base_z"] + sz)
    return sorted(tops)


def _first_riser_x(tile: dict) -> float:
    """Tile-local x of the near face of the first riser the route meets."""
    e = tile["entry_stairs"]
    return e["x_hall_flight"][0] if e["descending"] else e["x_wide_flight"][0]


def _last_riser_x(tile: dict) -> float:
    """Tile-local x where the route leaves the last riser."""
    e = tile["entry_stairs"]
    return e["x_wide_flight"][1] if e["descending"] else e["x_hall_flight"][1]


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_manifest_validates(descending: bool) -> None:
    validate_manifest([_tile(descending)], families=(ENTRY_STAIRS_FAMILY,))


# ---------------------------------------------------------------------------------------------
# The goal must be reachable, in either direction
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_goal_stands_on_a_surface_at_its_own_declared_height(descending: bool) -> None:
    """The goal line, the surface under it and ``goal_z`` must be the same place.

    ``goal_z`` is what ``goal_region.in_goal`` tests ``|root_z - goal_z|`` against. If the
    highest walkable surface under the goal is not that height, arrival is unsatisfiable while
    the termination still ends the episode, and it records as a time-out.
    """
    t = _tile(descending)
    r = t["route"]
    tops = _walkable_tops(t, r["goal_x"], r["goal_y"])
    assert tops, "the goal is over a hole in the tile's own ground"
    assert max(tops) == pytest.approx(r["goal_z"]), (
        f"the goal is at z={r['goal_z']:.3f} m but the highest walkable surface under it is at "
        f"{max(tops):.3f} m (all of them: {tops}).")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_whole_goal_band_is_on_that_surface(descending: bool) -> None:
    """Both edges of the band, not just its centre.

    ``freeform_goal_reached`` tests ``|y - goal_y| <= goal_half_width``, so the band's EDGES
    are scoring positions too. ``walkable_ground_span`` deliberately does not filter by height,
    so it cannot answer this and the check has to be against the surface at the goal's level.
    """
    t = _tile(descending)
    r = t["route"]
    for edge in (r["goal_y"] - r["goal_half_width"], r["goal_y"] + r["goal_half_width"]):
        tops = _walkable_tops(t, r["goal_x"], edge)
        assert tops and max(tops) == pytest.approx(r["goal_z"]), (
            f"the goal band's edge at y={edge:.3f} is not on the goal surface (tops {tops})")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_goal_sits_the_same_distance_past_the_last_riser_as_a_drawn_course(
        descending: bool) -> None:
    """Otherwise the per-tile-type breakdown compares differently-placed finish lines."""
    t = _tile(descending)
    past = t["route"]["goal_x"] - _last_riser_x(t)
    assert past == pytest.approx(RUNOUT_M + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0)


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_surface_runs_past_the_goal_with_room_to_stop(descending: bool) -> None:
    """A goal band flush against the far wall would score arrival and a collision together."""
    t = _tile(descending)
    r = t["route"]
    beyond = [ov for ov in _surfaces(t)
              if ov["base_z"] + ov["size_xyz"][2] == pytest.approx(r["goal_z"])]
    far = max(ov["pos_xy"][0] + ov["size_xyz"][0] / 2 for ov in beyond)
    assert far >= r["goal_x"] + 1.0, (
        f"the goal surface ends at x={far:.3f}, only {far - r['goal_x']:.2f} m past the goal")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_goal_band_is_the_librarys_and_not_this_corridors(descending: bool) -> None:
    """A per-tile band cannot be declared in a per-LIBRARY goal spec, in either direction.

    Half this hallway is 1.27 m against the drawn courses' 0.75 -- a 1.7x mismatch, in the
    direction that scores a robot standing beside the goal as arrived.
    """
    t = _tile(descending)
    assert t["route"]["goal_half_width"] == pytest.approx(GOAL_HALF_WIDTH_M)
    assert t["route"]["goal_half_width"] != pytest.approx(geometry_m()["hall_width_m"] / 2.0)


# ---------------------------------------------------------------------------------------------
# The approach: the library's own spawn convention
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_run_up_is_the_librarys_approach_interval(descending: bool) -> None:
    """THE fix: a guaranteed run-up, not a lucky one.

    One wide spawn rectangle gave 0.3-2.7 m before the first riser; the chain the drawn
    courses use covers ``approach_range_m`` exactly. Under a metre of ground is not enough to
    reach the commanded speed the tile then scores the robot at, and the shortfall reads as a
    policy that cannot climb rather than as a course that never let it start.
    """
    t = _tile(descending)
    lo_want, hi_want = VariedCoursesCfg().approach_range_m
    first = _first_riser_x(t)
    start_z = t["route"]["start_z"]
    chain = [a for a in t["route"]["spawn_areas"] if abs(a["z"] - start_z) < TOL]
    assert len(chain) > 1, "the approach is a single rectangle, not a chain"
    lo = min(first - (a["x"] + a["half_x"]) for a in chain)
    hi = max(first - (a["x"] - a["half_x"]) for a in chain)
    assert lo == pytest.approx(lo_want) and hi == pytest.approx(hi_want), (
        f"run-up spans {lo:.2f}..{hi:.2f} m, not the declared {lo_want}..{hi_want}")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_approach_chain_tiles_its_interval_without_gaps(descending: bool) -> None:
    """A gap is a run-up distance that never occurs; an overlap is one drawn twice."""
    t = _tile(descending)
    start_z = t["route"]["start_z"]
    chain = sorted((a for a in t["route"]["spawn_areas"] if abs(a["z"] - start_z) < TOL),
                   key=lambda a: a["x"])
    for a, b in zip(chain, chain[1:]):
        assert a["x"] + a["half_x"] == pytest.approx(b["x"] - b["half_x"]), (
            "the approach chain's intervals do not abut")


def test_an_approach_range_too_long_for_the_end_section_is_refused() -> None:
    """Rather than hanging the chain off the tile, where a spawn falls to its death."""
    with pytest.raises(ValueError, match="end section must hold"):
        make_entry_stairs_tile(0, _cfg(approach_range_m=(1.5, 20.0)))


def test_both_end_sections_could_hold_either_job() -> None:
    """What makes the building the SAME shape in both directions.

    Each end is sized for the longer of the two jobs it might be given -- the spawn chain or
    the goal run-out -- so ``descending`` chooses which end you start from and changes nothing
    about the geometry between them.
    """
    from motion_gen.terrain.varied_courses import _spawn_x

    need = max(_spawn_x() + VariedCoursesCfg().approach_range_m[1],
               RUNOUT_M + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0 + 0.75 + SECTION_MARGIN_M)
    assert APPROACH_M >= need and CLEARING_LEN_M >= need


# ---------------------------------------------------------------------------------------------
# The flights, and the asymmetry between up and down
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_route_climbs_or_descends_one_riser_at_a_time(descending: bool) -> None:
    """Every step between consecutive surfaces is exactly one riser of its own flight.

    This is what a mirrored descending tile gets wrong: it leaves a tread flush with the
    landing, i.e. a zero-height "step", and silently deepens the landing by one run.
    """
    t = _tile(descending)
    g = geometry_m()
    levels = [ov["base_z"] + ov["size_xyz"][2] for ov in _surfaces(t)
              if ov["size_xyz"][1] < t["size_y"] - TOL]      # skip the tile-wide ground slab
    levels = [0.0] + levels if not descending else levels + [0.0]
    # Collapse flat continuations before measuring: the landing is flush with the wide
    # flight's top tread and the lobby with the hall's, so consecutive equal levels are one
    # surface, not a zero-height step.
    flat: list[float] = []
    for v in levels:
        if not flat or abs(v - flat[-1]) > 1e-9:
            flat.append(v)
    steps = [abs(b - a) for a, b in zip(flat, flat[1:])]
    for s in steps:
        assert s == pytest.approx(g["wide_rise_m"]) or s == pytest.approx(g["hall_rise_m"]), (
            f"a {s:.4f} m step is neither a {g['wide_rise_m']:.4f} m wide riser nor a "
            f"{g['hall_rise_m']:.4f} m hall riser; levels were {[round(v, 3) for v in flat]}")
    assert sum(1 for s in steps if s == pytest.approx(g["wide_rise_m"])) == g["wide_steps"]
    assert sum(1 for s in steps if s == pytest.approx(g["hall_rise_m"])) == g["hall_steps"]


def test_going_down_authors_one_fewer_tread_per_flight() -> None:
    """The last step down lands on the surface already there -- the landing, or the floor."""
    g = geometry_m()
    up, down = _tile(False), _tile(True)
    treads = lambda t, w: sum(  # noqa: E731
        1 for ov in _surfaces(t) if ov["size_xyz"][1] == pytest.approx(w)
        and ov["size_xyz"][0] == pytest.approx(g["wide_run_m" if w > 3 else "hall_run_m"]))
    assert treads(up, g["wide_width_m"]) == g["wide_steps"]
    assert treads(down, g["wide_width_m"]) == g["wide_steps"] - 1
    assert treads(up, g["hall_width_m"]) == g["hall_steps"]
    assert treads(down, g["hall_width_m"]) == g["hall_steps"] - 1


def test_the_two_directions_are_the_same_building() -> None:
    """Same landing, same lobby, same flight widths and the same total height -- only the end
    you start from changes."""
    g = geometry_m()
    up, down = _tile(False), _tile(True)
    for t in (up, down):
        x0, x1 = t["entry_stairs"]["x_landing"]
        assert x1 - x0 == pytest.approx(g["landing_len_m"])
        lo, hi = t["entry_stairs"]["x_lobby"]
        assert hi - lo == pytest.approx(CLEARING_LEN_M)
    assert up["size_y"] == pytest.approx(down["size_y"])
    assert len(_walls(up)) == len(_walls(down))


# ---------------------------------------------------------------------------------------------
# The structure
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("descending", DIRECTIONS)
def test_every_overlay_and_spawn_area_is_inside_the_tile(descending: bool) -> None:
    """Geometry over the edge is built over void, and a spawn there falls at reset."""
    t = _tile(descending)
    for ov in t["overlays"]:
        cx, cy = ov["pos_xy"]
        sx, sy, _ = ov["size_xyz"]
        assert cx - sx / 2 >= -TOL and cx + sx / 2 <= t["size_x"] + TOL, f"{ov} off in x"
        assert cy - sy / 2 >= -TOL and cy + sy / 2 <= t["size_y"] + TOL, f"{ov} off in y"
    for a in t["route"]["spawn_areas"]:
        assert a["x"] - a["half_x"] >= -TOL and a["x"] + a["half_x"] <= t["size_x"] + TOL
        assert a["y"] - a["half_y"] >= -TOL and a["y"] + a["half_y"] <= t["size_y"] + TOL


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_each_spawn_area_stands_on_the_surface_it_declares(descending: bool) -> None:
    """A spawn at the wrong z is a reset that drops the robot or buries it."""
    t = _tile(descending)
    for a in t["route"]["spawn_areas"]:
        for x in (a["x"] - a["half_x"], a["x"], a["x"] + a["half_x"]):
            for y in (a["y"] - a["half_y"], a["y"], a["y"] + a["half_y"]):
                tops = _walkable_tops(t, x, y)
                assert tops and max(tops) == pytest.approx(a["z"]), (
                    f"spawn area at z={a['z']:.3f} has surface {tops} under ({x:.2f}, {y:.2f})")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_landing_keeps_the_wide_flights_width(descending: bool) -> None:
    """THE correction this tile already needed once.

    The measured entrance is 200 in wide for the flight AND the landing; the walls close to
    100 in only for the upper flight. Narrowing the landing removes what the course asks --
    arriving on a 5 m landing and having to commit to a 2.5 m hallway.
    """
    t = _tile(descending)
    g = geometry_m()
    x0, x1 = t["entry_stairs"]["x_landing"]
    lo, hi = walkable_ground_span(t["overlays"], (x0 + x1) / 2.0, t["size_y"] / 2.0)
    assert hi - lo >= g["wide_width_m"] - TOL, (
        f"the landing is {hi - lo:.3f} m wide, not the measured {g['wide_width_m']:.3f}")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_narrowing_is_a_FRONTAL_face_not_just_side_walls(descending: bool) -> None:
    """The feature the whole tile exists for, and the one the first port left out.

    A vertical surface inside camera range is what the generator has been measured to misread
    as a box, and side walls alone give the robot nothing FRONTAL to read -- they run parallel
    to travel. The narrowing has to be a return face square to it, with the hallway's mouth
    between the two halves.
    """
    t = _tile(descending)
    g = geometry_m()
    faces = [w for w in _walls(t)
             if w["size_xyz"][0] == pytest.approx(WALL_T_M)
             and w["base_z"] == pytest.approx(g["wide_height_m"])]
    assert len(faces) == 2, f"expected two return faces, got {len(faces)}"
    for f in faces:
        assert f["size_xyz"][1] == pytest.approx(
            (g["wide_width_m"] - g["hall_width_m"]) / 2.0), (
            "a return face must close the full 200-to-100 in step on its side")
    lo_f, hi_f = sorted(faces, key=lambda f: f["pos_xy"][1])
    gap = ((hi_f["pos_xy"][1] - hi_f["size_xyz"][1] / 2)
           - (lo_f["pos_xy"][1] + lo_f["size_xyz"][1] / 2))
    assert gap == pytest.approx(g["hall_width_m"]), (
        f"the mouth between the return faces is {gap:.3f} m, not the hallway's "
        f"{g['hall_width_m']:.3f} m")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_hallway_walls_stop_at_the_lobby(descending: bool) -> None:
    """Running them on through it makes the room a corridor and deletes the WIDENING -- the
    transition the course was extended to include, and the one a descent meets first."""
    t = _tile(descending)
    g = geometry_m()
    lobby_lo, lobby_hi = t["entry_stairs"]["x_lobby"]
    side = [w for w in _walls(t)
            if w["base_z"] == pytest.approx(g["wide_height_m"])
            and w["size_xyz"][1] == pytest.approx(WALL_T_M)]
    assert len(side) == 2, "the hallway has no side walls"
    for w in side:
        x0, x1 = w["pos_xy"][0] - w["size_xyz"][0] / 2, w["pos_xy"][0] + w["size_xyz"][0] / 2
        assert x1 <= lobby_lo + TOL or x0 >= lobby_hi - TOL, (
            f"a hallway wall spans [{x0:.2f}, {x1:.2f}] and runs into the lobby "
            f"[{lobby_lo:.2f}, {lobby_hi:.2f}]")


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_the_lobby_is_wider_than_the_hallway_and_its_mouth_is_open(descending: bool) -> None:
    """The widening, and the one gap that must NOT be walled: the way through."""
    t = _tile(descending)
    g = geometry_m()
    assert CLEARING_WIDTH_M > g["hall_width_m"] * 2.0, "the lobby barely widens"
    cy = t["size_y"] / 2.0
    lobby_lo, lobby_hi = t["entry_stairs"]["x_lobby"]
    lo, hi = walkable_ground_span(t["overlays"], (lobby_lo + lobby_hi) / 2.0, cy)
    assert hi - lo >= CLEARING_WIDTH_M - TOL
    mouth_x = lobby_hi if t["entry_stairs"]["descending"] else lobby_lo
    blocked = [w for w in _walls(t)
               if w["pos_xy"][0] - w["size_xyz"][0] / 2 < mouth_x + WALL_T_M
               and w["pos_xy"][0] + w["size_xyz"][0] / 2 > mouth_x - WALL_T_M
               and abs(w["pos_xy"][1] - cy) < g["hall_width_m"] / 2.0]
    assert not blocked, f"{len(blocked)} wall(s) block the way through the lobby"


@pytest.mark.parametrize("descending", DIRECTIONS)
def test_every_wall_is_collidable_and_stands_on_the_surface_it_belongs_to(
        descending: bool) -> None:
    """Visual-only walls leave the corridor a suggestion rather than a constraint, and a wall
    based at the wrong level either floats in view or buries itself."""
    t = _tile(descending)
    g = geometry_m()
    walls = _walls(t)
    assert len(walls) == 9, (
        "expected 2 return faces + 2 hallway walls + 2 lobby sides + 2 mouth segments + 1 far "
        f"wall, got {len(walls)}")
    levels = (g["wide_height_m"], g["total_height_m"])
    for w in walls:
        assert not w["decor"], "a decor wall has no collider -- the corridor would not constrain"
        assert not w["walkable_top"]
        assert w["size_xyz"][2] == pytest.approx(WALL_H_M)
        assert any(w["base_z"] == pytest.approx(lv) for lv in levels), (
            f"a wall stands at z={w['base_z']:.3f}, neither the landing nor the lobby")


def test_a_spawn_buffer_that_swallows_the_landing_is_refused() -> None:
    """Rather than emitting a zero- or negative-extent spawn area."""
    with pytest.raises(ValueError, match="no usable landing"):
        make_entry_stairs_tile(0, _cfg(spawn_edge_buffer_m=2.0))
