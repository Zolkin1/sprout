# Copyright 2026 Zachary Olkin. All rights reserved.
"""The randomised multi-terrain fine-tune courses.

Every test here is aimed at a failure that would train cleanly and mean something other than
what was asked for -- a spawn off its own route, a walkable bypass past the obstacle, a wall
draw that can come out empty, or two tiles that are secretly the same tile.
"""

from __future__ import annotations

import math

import pytest

from motion_gen.terrain.freeform import (
    FREEFORM_SPAWN_JITTER_M,
    WALL_T_M,
    validate_manifest,
)
from motion_gen.terrain.varied_courses import (
    BOXES_FAMILY,
    COMMANDED_SPEED_RANGE_MPS,
    GOAL_HALF_WIDTH_M,
    UPDOWN_APRON_M,
    UPDOWN_FAMILY,
    VariedCoursesCfg,
    family_blocks,
    generate_varied_manifest,
    generator_config,
    goal_region_spec,
    make_boxes_tile,
    make_updown_tile,
    tile_rng,
)

SEEDS = (0, 7, 21)
TOL = 1e-9


def _manifest(seed: int = 0, n_boxes: int = 8, n_updown: int = 8) -> list[dict]:
    """A small manifest at one seed."""
    return generate_varied_manifest(n_boxes, n_updown, VariedCoursesCfg(seed=seed))


# ---------------------------------------------------------------------------------------------
# The manifest is well formed for every consumer
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_the_manifest_validates(seed: int) -> None:
    tiles = _manifest(seed)
    validate_manifest(tiles, families=(BOXES_FAMILY, UPDOWN_FAMILY))


@pytest.mark.parametrize("seed", SEEDS)
def test_every_overlay_and_spawn_area_is_inside_its_tile(seed: int) -> None:
    """Nothing hangs off the tile, in either axis.

    A box or a spawn area over the edge is not an error anywhere -- the floor is rendered from
    ``size_x`` / ``size_y``, so geometry outside it is simply built over void and a robot
    spawned there falls at reset for a reason that reads as a policy failure.
    """
    for t in _manifest(seed):
        sx, sy = t["size_x"], t["size_y"]
        for i, ov in enumerate(t["overlays"]):
            cx, cy = ov["pos_xy"]
            ex, ey, _ = ov["size_xyz"]
            assert -TOL <= cx - ex / 2.0 and cx + ex / 2.0 <= sx + TOL, (
                f"tile {t['terrain_id']} overlay {i} off the tile in x")
            assert -TOL <= cy - ey / 2.0 and cy + ey / 2.0 <= sy + TOL, (
                f"tile {t['terrain_id']} overlay {i} off the tile in y")
        for j, a in enumerate(t["route"]["spawn_areas"]):
            assert -TOL <= a["x"] - a["half_x"] and a["x"] + a["half_x"] <= sx + TOL, (
                f"tile {t['terrain_id']} spawn area {j} off the tile in x")
            assert -TOL <= a["y"] - a["half_y"] and a["y"] + a["half_y"] <= sy + TOL, (
                f"tile {t['terrain_id']} spawn area {j} off the tile in y")


@pytest.mark.parametrize("seed", SEEDS)
def test_no_overlay_carries_a_yaw(seed: int) -> None:
    """The RL loader hard-refuses a yawed overlay, so authoring one breaks the whole library.

    The angle is expressed by the ROUTE instead; see the module docstring.  A yawed box would
    be axis-aligned in Isaac and rotated in MuJoCo -- one manifest, two worlds.
    """
    for t in _manifest(seed):
        for i, ov in enumerate(t["overlays"]):
            assert abs(float(ov["yaw"])) <= TOL, (
                f"tile {t['terrain_id']} overlay {i} has yaw {ov['yaw']}")


# ---------------------------------------------------------------------------------------------
# The spawn chain: on the route, gapless, and weight-preserving
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_every_ground_spawn_area_sits_on_the_route_ray(seed: int) -> None:
    """A ground spawn is on the approach centreline, whatever the tile's angle.

    Spawn extents are TILE-frame -- the reset draws ``area.x + jitter * area.half_x`` with no
    rotation -- so one wide rectangle spanning the approach interval on a 20 degree tile would
    have to be nearly a metre wide laterally to contain the route, putting most spawns off the
    centreline.  The chain exists so each area's centre is exactly on the ray.
    """
    cfg = VariedCoursesCfg(seed=seed, box_max_approach_deg=20.0)
    for t in (make_boxes_tile(i, cfg) for i in range(12)):
        r = t["route"]
        assert abs(r["start_yaw"]) > 1e-6, "the angled path must actually be exercised here"
        slope = math.tan(r["start_yaw"])
        for j, a in enumerate(r["spawn_areas"]):
            if a["z"] > TOL:          # a box top, not part of the ground chain
                continue
            want = r["start_y"] + slope * (a["x"] - r["start_x"])
            assert abs(a["y"] - want) < 1e-6, (
                f"tile {t['terrain_id']} ground area {j} is {abs(a['y'] - want):.3f} m off the "
                f"route ray")


@pytest.mark.parametrize("seed", SEEDS)
def test_every_spawn_area_carries_the_routes_heading(seed: int) -> None:
    """The reset takes the spawn HEADING from ``frame_yaw + area.yaw``, never from the route.

    So an angled tile whose areas left ``yaw`` at 0 would spawn the robot up to 20 degrees off
    its own route and ask the yaw controller to recover before the first step -- a handicap
    that looks like a policy result.
    """
    cfg = VariedCoursesCfg(seed=seed, box_max_approach_deg=20.0)
    for t in (make_boxes_tile(i, cfg) for i in range(12)):
        r = t["route"]
        assert abs(r["start_yaw"]) > 1e-6, "the angled path must actually be exercised here"
        for j, a in enumerate(r["spawn_areas"]):
            assert abs(a["yaw"] - r["start_yaw"]) < 1e-9, (
                f"tile {t['terrain_id']} area {j} yaw {a['yaw']} != route {r['start_yaw']}")


@pytest.mark.parametrize("seed", SEEDS)
def test_the_ground_chain_tiles_the_approach_interval_exactly(seed: int) -> None:
    """The covered approach intervals abut: no gap, no overlap, and the union is the range.

    A gap makes some approach distances unreachable while the config claims otherwise; an
    overlap makes the pooled draw non-uniform, over-weighting whatever the chain double-covers.
    """
    cfg = VariedCoursesCfg(seed=seed)
    lo, hi = cfg.approach_range_m
    for t in generate_varied_manifest(6, 6, cfg):
        key = "varied_boxes" if t["family"] == BOXES_FAMILY else "varied_updown"
        first = (t[key]["boxes"][0]["x0"] if key == "varied_boxes"
                 else t[key]["first_riser_x"])
        ground = [a for a in t["route"]["spawn_areas"] if a["z"] <= TOL]
        spans = sorted(((first - (a["x"] + a["half_x"]), first - (a["x"] - a["half_x"]))
                        for a in ground), key=lambda s: s[0])
        assert abs(spans[0][0] - lo) < 1e-9, f"tile {t['terrain_id']} starts at {spans[0][0]}"
        assert abs(spans[-1][1] - hi) < 1e-9, f"tile {t['terrain_id']} ends at {spans[-1][1]}"
        for a, b in zip(spans, spans[1:]):
            assert abs(a[1] - b[0]) < 1e-9, (
                f"tile {t['terrain_id']}: chain gap/overlap of {b[0] - a[1]:.4f} m")


@pytest.mark.parametrize("seed", SEEDS)
def test_the_grounds_total_weight_does_not_depend_on_the_chain_length(seed: int) -> None:
    """However many areas it took to cover the interval, the ground's SHARE is one unit.

    Otherwise a longer approach range would silently starve the raised-surface spawns -- and
    those are the ones the pinned two-box course measured as essential, since of 352 successful
    episodes every one reached the goal by dismounting a box.
    """
    cfg = VariedCoursesCfg(seed=seed)
    for t in generate_varied_manifest(6, 6, cfg):
        ground = [a for a in t["route"]["spawn_areas"] if a["z"] <= TOL]
        assert abs(sum(a["weight"] for a in ground) - cfg.surface_weight) < 1e-9, (
            f"tile {t['terrain_id']} ground weight {sum(a['weight'] for a in ground)}")
        for a in t["route"]["spawn_areas"]:
            if a["z"] > TOL:
                assert abs(a["weight"] - cfg.surface_weight) < 1e-9


def test_a_longer_approach_range_adds_areas_rather_than_widening_them() -> None:
    """The chain grows in COUNT; each area keeps the ordinary jitter extent.

    Widening the areas instead would change the spawn jitter for the whole tile, which
    ``_draw_spawn_areas`` uses in place of ``pos_jitter_m`` -- so it would silently make these
    runs incomparable to every earlier one.
    """
    short = make_updown_tile(0, VariedCoursesCfg(approach_range_m=(2.0, 2.5)))
    long = make_updown_tile(0, VariedCoursesCfg(approach_range_m=(1.5, 4.0)))
    n_short = sum(1 for a in short["route"]["spawn_areas"] if a["z"] <= TOL)
    n_long = sum(1 for a in long["route"]["spawn_areas"] if a["z"] <= TOL)
    assert n_long > n_short
    for t in (short, long):
        for a in t["route"]["spawn_areas"]:
            if a["z"] <= TOL:
                assert a["half_y"] == pytest.approx(FREEFORM_SPAWN_JITTER_M[1])


# ---------------------------------------------------------------------------------------------
# The draws stay in their declared ranges
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_the_box_draws_stay_in_range(seed: int) -> None:
    cfg = VariedCoursesCfg(seed=seed)
    for t in (make_boxes_tile(i, cfg) for i in range(40)):
        b = t["varied_boxes"]
        assert cfg.boxes_per_tile_range[0] <= b["n_boxes"] <= cfg.boxes_per_tile_range[1]
        assert abs(b["approach_deg"]) <= cfg.box_max_approach_deg + TOL
        for x in b["boxes"]:
            assert cfg.box_height_range_m[0] <= x["height_m"] <= cfg.box_height_range_m[1]
            assert cfg.box_depth_range_m[0] <= x["depth_m"] <= cfg.box_depth_range_m[1]
            assert cfg.box_width_range_m[0] <= x["width_m"] <= cfg.box_width_range_m[1]
        for p, q in zip(b["boxes"], b["boxes"][1:]):
            gap = q["x0"] - p["x1"]
            assert cfg.box_gap_range_m[0] - TOL <= gap <= cfg.box_gap_range_m[1] + TOL


@pytest.mark.parametrize("seed", SEEDS)
def test_the_updown_draws_stay_in_range(seed: int) -> None:
    cfg = VariedCoursesCfg(seed=seed)
    for t in (make_updown_tile(i, cfg) for i in range(40)):
        u = t["varied_updown"]
        assert cfg.updown_steps_range[0] <= u["steps"] <= cfg.updown_steps_range[1]
        assert cfg.updown_width_range_m[0] <= u["width_m"] <= cfg.updown_width_range_m[1]
        assert (cfg.updown_platform_range_m[0] <= u["platform_depth_m"]
                <= cfg.updown_platform_range_m[1])
        # The staircase geometry is a DRAWN CELL now, not two fixed scalars, so the height
        # must agree with whichever cell this tile drew -- and the cell must be one that was
        # offered. A rise that is merely "in range" would pass while pairing a trained riser
        # with an untrained tread, which is the failure drawing them independently produces.
        cell = (u["rise_m"], u["run_m"])
        offered = {(round(r, 6), round(t, 6))
                   for r, t in tuple(cfg.updown_cells) + tuple(cfg.updown_measured_cells)}
        assert (round(cell[0], 6), round(cell[1], 6)) in offered, f"unoffered cell {cell}"
        assert u["flight_height_m"] == pytest.approx(u["steps"] * cell[0])


@pytest.mark.parametrize("seed", SEEDS)
def test_rise_and_run_are_one_draw_never_two(seed: int) -> None:
    """The pair must come from the offered list, never from a cross product of its axes.

    Drawn independently, a 0.180 m riser lands on a 0.2092 m tread -- a ladder, and no shape
    check downstream would question it because each number is individually in range. The
    trained grid is rectangular so most crosses ARE offered; this catches the measured cells,
    whose rises and runs appear in no other pair.
    """
    cfg = VariedCoursesCfg(seed=seed, updown_measured_frac=0.5)
    offered = {(round(r, 6), round(t, 6))
               for r, t in tuple(cfg.updown_cells) + tuple(cfg.updown_measured_cells)}
    seen = set()
    for t in (make_updown_tile(i, cfg) for i in range(60)):
        u = t["varied_updown"]
        seen.add((round(u["rise_m"], 6), round(u["run_m"], 6)))
    assert seen <= offered, f"cells drawn that were never offered: {sorted(seen - offered)}"
    assert len(seen) > 1, "the draw collapsed onto one cell"


def test_the_measured_cells_are_where_they_were_measured() -> None:
    """Pins the conversion, because the numbers are inches in the real world.

    12.5 x 5.5, 12 x 7 and 13 x 5 inches. Two of the three land ON trained cells -- which is
    why they are called `measured` and not `ood`, and why a report about them cannot claim
    generalisation.
    """
    inch = 0.0254
    want = {(round(5.5 * inch, 4), round(12.5 * inch, 4)),
            (round(7.0 * inch, 4), round(12.0 * inch, 4)),
            (round(5.0 * inch, 4), round(13.0 * inch, 4))}
    got = {(round(r, 4), round(t, 4)) for r, t in VariedCoursesCfg().updown_measured_cells}
    assert got == want
    # Their DISTANCE to the trained grid, pinned rather than counted: this is the whole reason
    # the set is called `measured` and not `ood`, so the numbers belong in the test.
    trained = VariedCoursesCfg().updown_cells
    def gap(cell):
        return min(max(abs(cell[0] - r), abs(cell[1] - t)) for r, t in trained)
    inch = 0.0254
    assert gap((7.0 * inch, 12.0 * inch)) == pytest.approx(0.0022, abs=2e-4), "7x12 is (0.180, 0.305)"
    assert gap((5.0 * inch, 13.0 * inch)) == pytest.approx(0.0030, abs=2e-4), "5x13 is (0.130, 0.330)"
    # Only this one is new geometry -- and it INTERPOLATES, sitting inside the grid's hull on
    # both axes rather than beyond it, so it is not an extrapolation test either.
    assert gap((5.5 * inch, 12.5 * inch)) == pytest.approx(0.0125, abs=2e-4)
    lo_r, hi_r = min(r for r, _ in trained), max(r for r, _ in trained)
    lo_t, hi_t = min(t for _, t in trained), max(t for _, t in trained)
    assert lo_r <= 5.5 * inch <= hi_r and lo_t <= 12.5 * inch <= hi_t


def test_a_flight_is_never_unwalled() -> None:
    """One or two walls, never none -- asserted on the CONFIG and on the drawn tiles.

    An unwalled flight has ground-level apron beside it and no vertical surface stopping a
    robot from walking PAST the staircase, which the goal band then scores as a traverse.  The
    free-form ``stairs`` family draws an empty side list one time in four, which is exactly
    what must not happen here.
    """
    cfg = VariedCoursesCfg()
    assert all(len(s) >= 1 for s in cfg.updown_wall_sides), cfg.updown_wall_sides
    seen: set[int] = set()
    for seed in SEEDS:
        for t in (make_updown_tile(i, VariedCoursesCfg(seed=seed)) for i in range(40)):
            n = len(t["varied_updown"]["wall_sides"])
            assert n >= 1, f"tile {t['terrain_id']} drew no walls"
            seen.add(n)
    assert seen == {1, 2}, f"both wall counts must be reachable, saw {seen}"


def test_the_updown_apron_leaves_no_walkable_bypass() -> None:
    """The ground beside the flight is the WALL BAND and nothing more.

    The free-form stairs family adds ``SECTION_MARGIN_M`` on top, leaving 0.65 m of walkable
    apron; with the goal band spanning the corridor at ground level, a robot that drifts onto
    it and walks past the staircase scores a success.
    """
    assert UPDOWN_APRON_M == pytest.approx(WALL_T_M)
    for seed in SEEDS:
        for t in (make_updown_tile(i, VariedCoursesCfg(seed=seed)) for i in range(20)):
            u = t["varied_updown"]
            assert t["size_y"] == pytest.approx(u["width_m"] + 2.0 * WALL_T_M)


def test_a_box_too_shallow_to_stand_on_gets_no_top_spawn() -> None:
    """A shallow box is a legitimate DRAW, not a mistake, so it must not raise.

    The depth range spans the trained 0.46 m deliberately -- under one stride, no top to stand
    on -- and such a box simply carries no top spawn area while the ground chain still covers
    it.
    """
    cfg = VariedCoursesCfg(box_depth_range_m=(0.45, 0.46),
                           boxes_per_tile_range=(2, 2))
    t = make_boxes_tile(0, cfg)
    raised = [a for a in t["route"]["spawn_areas"] if a["z"] > TOL]
    assert raised == []
    assert t["varied_boxes"]["spawn_areas"] == t["varied_boxes"]["ground_spawn_areas"]


def test_a_corridor_too_narrow_for_a_platform_spawn_is_refused() -> None:
    """Refused with the numbers and the knobs named, rather than emitting a negative extent.

    A negative ``half_x`` would make the reset draw spawns OUTSIDE the platform, off its edge.
    """
    cfg = VariedCoursesCfg(updown_width_range_m=(0.5, 0.5),
                           updown_platform_range_m=(0.4, 0.4))
    with pytest.raises(ValueError, match="no usable platform"):
        make_updown_tile(0, cfg)


# ---------------------------------------------------------------------------------------------
# Reproducibility, and the mixer that was not
# ---------------------------------------------------------------------------------------------
def test_a_seed_reproduces_a_library_and_a_different_seed_does_not() -> None:
    assert _manifest(3) == _manifest(3)
    assert _manifest(3) != _manifest(4)


def test_adjacent_tiles_are_not_secretly_the_same_tile() -> None:
    """The draws must not collide across tile ids.

    They did: with the ``(seed * k + tid) ^ salt`` mixer the other terrain modules use,
    ``(X + 2) ^ 101 == (X + 3) ^ 102``, so tile 2's first box drew exactly tile 3's second box
    and the two tiles came out mirror images.  Nothing about either tile looked wrong.
    """
    for seed in SEEDS:
        cfg = VariedCoursesCfg(seed=seed)
        sigs = set()
        for i in range(40):
            b = make_boxes_tile(i, cfg)["varied_boxes"]
            sigs.add(tuple(round(v, 9) for x in b["boxes"]
                           for v in (x["height_m"], x["depth_m"], x["width_m"])))
        assert len(sigs) == 40, f"seed {seed}: only {len(sigs)} distinct box draws in 40 tiles"


def test_the_tile_rng_separates_all_three_of_its_fields() -> None:
    """Seed, tile and salt each move the stream on their own."""
    cfg = VariedCoursesCfg(seed=5)
    base = tile_rng(cfg, 3, 7).random(4).tolist()
    assert base != tile_rng(VariedCoursesCfg(seed=6), 3, 7).random(4).tolist()
    assert base != tile_rng(cfg, 4, 7).random(4).tolist()
    assert base != tile_rng(cfg, 3, 8).random(4).tolist()
    assert base == tile_rng(VariedCoursesCfg(seed=5), 3, 7).random(4).tolist()


# ---------------------------------------------------------------------------------------------
# The library contract
# ---------------------------------------------------------------------------------------------
def test_the_families_occupy_contiguous_id_blocks() -> None:
    """A report reduces per family by id range, the way the in-distribution routes do."""
    tiles = _manifest(0, n_boxes=5, n_updown=7)
    blocks = family_blocks(5, 7)
    assert blocks == {BOXES_FAMILY: (0, 4), UPDOWN_FAMILY: (5, 11)}
    for t in tiles:
        lo, hi = blocks[t["family"]]
        assert lo <= t["terrain_id"] <= hi
    assert [t["terrain_id"] for t in tiles] == list(range(12))


def test_an_empty_family_is_omitted_from_the_blocks() -> None:
    assert family_blocks(4, 0) == {BOXES_FAMILY: (0, 3)}
    assert family_blocks(0, 4) == {UPDOWN_FAMILY: (0, 3)}


def test_the_generator_config_declares_the_command_band_where_the_reader_looks() -> None:
    """``clip_io.read_commanded_speed_range`` searches named blocks, so the key must match.

    A band declared under a block the reader does not know is a band nothing enforces -- which
    is exactly what the pinned two-box course's six-rung speed LADDER was.
    """
    cfg = generator_config(3, 4)
    band = cfg["varied_courses"]["commanded_speed_range_mps"]
    assert band == [float(v) for v in COMMANDED_SPEED_RANGE_MPS]
    lo, hi = band
    assert 0.0 < lo < hi


def test_an_empty_library_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one tile"):
        generate_varied_manifest(0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        generate_varied_manifest(-1, 4)


def test_a_bad_approach_range_is_refused() -> None:
    with pytest.raises(ValueError, match="positive ordered pair"):
        make_updown_tile(0, VariedCoursesCfg(approach_range_m=(4.0, 1.5)))


# ---------------------------------------------------------------------------------------------
# The goal band and the spec that has to agree with it
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_every_tile_carries_the_SAME_goal_band(seed: int) -> None:
    """One band across both families, unscaled by the drawn corridor or box width.

    The tolerance reaches the runtime twice -- as each tile's ``route.goal_half_width`` (the
    ``freeform_goal_reached`` termination) and as the single ``default.half_width_m`` of the
    goal-regions spec (``goal_region.in_goal``) -- and nothing reconciles them.  A band that
    varied per tile could only be declared at some summary of itself: too wide scores a robot
    beside the obstacle as arrived, too narrow records a real arrival as a time-out because the
    termination fires anyway.  Both are silent.
    """
    bands = {round(t["route"]["goal_half_width"], 9)
             for t in generate_varied_manifest(20, 20, VariedCoursesCfg(seed=seed))}
    assert bands == {round(GOAL_HALF_WIDTH_M, 9)}, (
        f"seed {seed}: tiles disagree on the goal band -> {sorted(bands)}")


@pytest.mark.parametrize("seed", SEEDS)
def test_the_goal_band_is_never_cut_by_the_ground_fit(seed: int) -> None:
    """``_fit_goal_half_width`` can only narrow, so a tile too small to hold the band would
    silently reintroduce the per-tile disagreement above.  The narrowest corridor plus its two
    aprons must therefore still hold it."""
    for t in generate_varied_manifest(20, 20, VariedCoursesCfg(seed=seed)):
        assert t["route"]["goal_half_width"] == pytest.approx(GOAL_HALF_WIDTH_M), (
            f"tile {t['terrain_id']} ({t['family']}) band cut to "
            f"{t['route']['goal_half_width']:.3f}")


def test_the_declared_spec_matches_what_the_tiles_were_built_to() -> None:
    """The spec is declared, not fitted, so nothing downstream checks it -- this does."""
    spec = goal_region_spec(6, 6)
    assert spec["default"]["half_width_m"] == pytest.approx(GOAL_HALF_WIDTH_M)
    assert spec["default"]["approach_margin_m"] == 0.0
    # A REAL band, not 0: see `test_the_goal_spec_never_declares_a_zero_height_tolerance` --
    # 0 is an exact-equality test in `in_goal`, and it voided a whole arm.
    assert spec["default"]["height_tol_m"] == pytest.approx(0.5)
    assert spec["provenance"]["declared"] is True
    for t in generate_varied_manifest(6, 6):
        assert t["route"]["goal_half_width"] == pytest.approx(spec["default"]["half_width_m"])


def test_the_shipped_default_authors_no_rotation_at_all() -> None:
    """Every tile as SHIPPED runs straight down its own axis.

    The angled mechanism above is kept and tested, but a first multi-terrain run does not use
    it: a non-zero yaw gives a lateral component to five things that were exactly zero -- the
    tile's y extent, the spawn chain's placement, each area's heading, the goal's position and
    the steering controller's projection onto the route -- and each is a way to fail for a
    reason that is not about the policy.
    """
    assert VariedCoursesCfg().box_max_approach_deg == 0.0
    for seed in SEEDS:
        for t in generate_varied_manifest(20, 20, VariedCoursesCfg(seed=seed)):
            r = t["route"]
            assert r["start_yaw"] == 0.0, f"tile {t['terrain_id']} is angled"
            assert r["goal_y"] == pytest.approx(r["start_y"]), (
                f"tile {t['terrain_id']} goal is off the spawn's centreline")
            for a in r["spawn_areas"]:
                assert a["yaw"] == 0.0
                assert a["y"] == pytest.approx(r["start_y"]), (
                    f"tile {t['terrain_id']} spawn area off the centreline")
            if t["family"] == BOXES_FAMILY:
                for b, ov in zip(t["varied_boxes"]["boxes"], t["overlays"][1:]):
                    assert ov["pos_xy"][1] == pytest.approx(r["start_y"]), (
                        f"tile {t['terrain_id']} box off the centreline")


def test_the_goal_spec_never_declares_a_zero_height_tolerance() -> None:
    """0 is not "off" for `in_goal`, and writing it there voids an entire arm.

    ``height_tol_m`` names two quantities with opposite conventions at zero:
    ``terminations.freeform_goal_reached`` guards its height clause with ``if height_tol_m >
    0``, so 0 disables the test, while ``goal_region.in_goal`` applies
    ``|height_err| <= height_tol_m`` unconditionally -- so 0 is an exact-equality test that
    floating point never satisfies.

    Written as 0.0 once, it made every arrival on the varied tiles record as a non-arrival:
    the eval read 0.0000 on both arms over 8400 episodes, and because the progress reward's
    success term is ``gate * C * reached_goal`` it was identically zero through all five
    training iterations too.  Nothing failed; the numbers were simply all zero.
    """
    tol = goal_region_spec(4, 4)["default"]["height_tol_m"]
    assert tol > 0.0, "0 is an exact-equality test in `in_goal`, not a disabled one"
    # Matches every other course's spec, and `GoalTolerances`' own default.
    assert tol == pytest.approx(0.5)


def test_the_goal_spec_tolerances_are_all_usable() -> None:
    """A zero band on any axis is unsatisfiable, not permissive."""
    d = goal_region_spec(4, 4)["default"]
    assert d["half_width_m"] > 0.0
    assert d["height_tol_m"] > 0.0
    # The margin is the one that MAY be zero: `in_goal` tests `downrange >= -margin`, so 0
    # means "at or past the line", which is exactly the intent.
    assert d["approach_margin_m"] >= 0.0
