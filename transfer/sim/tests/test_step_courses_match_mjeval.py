# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Isaac up-into-down tile and the MuJoCo up-into-down course must be the SAME staircase.

``motion_gen.terrain.step_courses`` restates mjeval's geometry rather than importing it,
because ``source/motion_gen`` is the offline generation package and stays free of ``transfer``
exactly as it stays free of IsaacLab.  This is the file that makes the restatement safe: it
imports BOTH and requires them to agree, so a constant edited on one side fails here instead of
silently producing an Isaac fine-tune and a MuJoCo evaluation run on two different staircases.
It is the same device ``test_box2_matches_mjeval.py`` applies to the two-box course.

The comparison is made in each frame's OWN terms -- distances from the spawn -- because the two
place their origins differently: mjeval's course has the spawn at ``x = 0`` and the first riser
at ``X_START``, while a free-form tile is corner-anchored and insets its spawn by
``SECTION_MARGIN_M + SPAWN_INSET_M``.  Comparing absolute x would only be comparing that inset.

**The CURB has no mjeval twin and is not tested here**; it is authored in ``step_courses`` and
nothing off-sim has to agree with it.  What this file does check about it is the one thing that
crosses the boundary: its step heights against the trained box grid, which is the claim that
makes the course out of distribution at all.

This test lives under ``transfer/sim`` rather than ``source/motion_gen`` because it is the half
that may import mjeval; it needs ``source/motion_gen`` on the path, which the repo's pytest
invocation supplies (``PYTHONPATH=source/robot_rl:source/motion_gen``).
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.dirname(_HERE)
# ``transfer`` itself is needed because ``generate_stairs_scene`` imports ``sim.robot`` as a
# package -- mjeval's own modules import each other by bare name, so both levels go on.
for _d in (os.path.join(_SIM, "mjeval"), _SIM, os.path.dirname(_SIM)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

sc = pytest.importorskip("motion_gen.terrain.step_courses")
freeform = pytest.importorskip("motion_gen.terrain.freeform")
mj_terrains = pytest.importorskip("terrains")
gss = pytest.importorskip("generate_stairs_scene")


def _course(platform_depth_m: float):
    """The mjeval up-into-down course at one platform depth, found by NAME not by index."""
    want = f"mjeval_updn_p{int(round(platform_depth_m * 10)):03d}"
    hits = [c for c in mj_terrains.catalogue() if c.name == want]
    assert len(hits) == 1, f"expected exactly one {want}, got {len(hits)}"
    return hits[0]


@pytest.fixture(scope="module")
def courses():
    """Both up-into-down courses, keyed by platform depth."""
    return {d: _course(d) for d in sc.UPDOWN_PLATFORM_DEPTHS_M}


# --- shared constants -----------------------------------------------------------------
def test_the_approach_and_goal_constants_are_mjevals():
    """The run-up, the goal lead and the goal rectangle's depth are mjeval's own numbers."""
    assert sc.APPROACH_M == pytest.approx(mj_terrains.X_START)
    # And pin the hop: mjeval takes it from the scene generator, so a change there must land.
    assert sc.APPROACH_M == pytest.approx(gss.DEFAULT_X_START)
    assert sc.TARGET_LEAD_M == pytest.approx(mj_terrains.TARGET_LEAD)
    assert sc.GOAL_DEPTH_M == pytest.approx(mj_terrains.GOAL_DEPTH)


# --- up-into-down ---------------------------------------------------------------------
def test_the_step_geometry_is_mjevals_modal_level(courses):
    """Rise, run, riser count and width all come off the built course, not off arithmetic."""
    for depth, course in courses.items():
        p = course.params
        assert sc.UPDOWN_RISE_M == pytest.approx(p["rise_m"]), depth
        assert sc.UPDOWN_RUN_M == pytest.approx(p["run_m"]), depth
        assert sc.UPDOWN_STEPS == int(p["steps"]), depth
        assert sc.UPDOWN_WIDTH_M == pytest.approx(p["width_m"]), depth
        assert sc.UPDOWN_PLATFORM_DEPTHS_M == pytest.approx(
            mj_terrains.EVAL_PLATFORM_DEPTHS)
    # The rise/run pair must stay mjeval's "modal" level -- the in-distribution control of its
    # step-geometry axis. Any other level would make the up-into-down TRANSITION confounded
    # with an off-grid riser.
    modal = mj_terrains.EVAL_STEP_GEOMETRY[0]
    assert (sc.UPDOWN_RISE_M, sc.UPDOWN_RUN_M) == pytest.approx(modal[:2])
    assert modal[2] == "modal"
    assert sc.UPDOWN_WIDTH_M == pytest.approx(mj_terrains.EVAL_STAIR_WIDTHS[1])
    assert sc.UPDOWN_STEPS == mj_terrains._STEPS


def test_every_landmark_sits_at_the_same_distance_from_the_spawn(courses):
    """First riser, platform, bottom of the descent -- all measured from the spawn.

    mjeval's spawn is the course origin and its first riser is at ``X_START``, so the tile built
    at ``approach_m = X_START`` is the one that should line up landmark for landmark.
    """
    off_x = freeform.SECTION_MARGIN_M + freeform.SPAWN_INSET_M
    for depth, course in courses.items():
        f = sc.updown_feature_x(mj_terrains.X_START, depth)
        rel = {k: v - off_x for k, v in f.items() if k != "top_z"}

        assert rel["first_riser"] == pytest.approx(mj_terrains.X_START), depth
        assert rel["first_riser"] == pytest.approx(course.obstacle_x_m), depth
        # mjeval's own arithmetic, restated from `_up_into_down` / `_lane_geoms`.
        plat0 = mj_terrains.X_START + sc.UPDOWN_STEPS * sc.UPDOWN_RUN_M
        assert rel["platform_x0"] == pytest.approx(plat0), depth
        assert rel["platform_x1"] == pytest.approx(plat0 + depth), depth
        assert rel["bottom_x"] == pytest.approx(
            plat0 + depth + (sc.UPDOWN_STEPS - 1) * sc.UPDOWN_RUN_M), depth
        # And against the course itself: its target is TARGET_LEAD past the bottom.
        assert rel["bottom_x"] == pytest.approx(
            course.target_xy[0] - mj_terrains.TARGET_LEAD), depth
        assert f["top_z"] == pytest.approx(course.flight_height_m), depth


def test_the_goal_line_is_the_near_edge_of_mjevals_rectangle(courses):
    """The tile's goal LINE sits where mjeval's goal RECTANGLE begins.

    The two terminations differ in shape -- mjeval tests containment in a rectangle, the RL side
    a downrange line plus a lateral band -- so the mapping that makes them agree on the boundary
    that decides an outcome is "line at the rectangle's near edge".  This pins it.
    """
    off_x = freeform.SECTION_MARGIN_M + freeform.SPAWN_INSET_M
    for depth, course in courses.items():
        tile = sc.make_updown_tile(0, mj_terrains.X_START, depth)
        goal_rel = tile["route"]["goal_x"] - off_x
        near_edge = course.target_xy[0] - mj_terrains.GOAL_DEPTH / 2.0
        assert goal_rel == pytest.approx(near_edge), depth


def test_the_goal_band_is_the_corridor_and_deliberately_not_mjevals(courses):
    """The band is half the flight, NOT mjeval's 2.5 m -- and it is above the RL floor.

    mjeval widened its goal because nothing in that stack steered and most timeouts were a robot
    that had cleared the terrain and passed beside the target.  Here the yaw controller runs and
    the corridor has no apron, so a band matching the flight measures "got to the bottom of the
    stairs" rather than "ended up roughly past them".  A regression to mjeval's width would
    silently re-admit the wide-band reading, so pin the difference rather than the agreement.
    """
    for depth, course in courses.items():
        tile = sc.make_updown_tile(0, mj_terrains.X_START, depth)
        half = tile["route"]["goal_half_width"]
        assert half == pytest.approx(sc.UPDOWN_WIDTH_M / 2.0), depth
        assert half < course.goal_half_width_m, depth
        # Above what `mdp.freeform_goal_reached` clamps up to, or the declared band is not what
        # the termination measures.
        assert half > freeform.GOAL_HALF_WIDTH_FLOOR_M, depth


def test_the_apron_beside_the_flight_is_the_wall_band_and_nothing_more(courses):
    """A drifting robot must not be able to walk PAST the staircase and still score.

    ``freeform_goal_reached`` tests a downrange line and a lateral band and NEVER a height, and
    this course's goal is at floor level past the descent -- so ground-level apron beside the
    flight is a route around the obstacle for its whole length, not a bounded overhang.  The
    free-form ``stairs`` family adds ``SECTION_MARGIN_M`` a side (0.65 m of apron); this one
    must not, so pin the width rather than merely the fact that it is small.

    mjeval has an infinite floor plane and tells going-over from going-around with
    ``Course.traversed``; nothing here does, which is why the geometry has to.
    """
    for depth in courses:
        tile = sc.make_updown_tile(0, mj_terrains.X_START, depth)
        assert sc.UPDOWN_APRON_M == pytest.approx(freeform.WALL_T_M)
        assert tile["size_y"] == pytest.approx(sc.UPDOWN_WIDTH_M + 2.0 * freeform.WALL_T_M)
        # The family's own number would reopen it, so state what is being avoided.
        assert tile["size_y"] < (sc.UPDOWN_WIDTH_M + 2.0 * freeform.WALL_T_M
                                 + 2.0 * freeform.SECTION_MARGIN_M)
        # Measured where it matters: beside the flight, and at the goal.
        for x in (tile["updown"]["first_riser_x"] + sc.UPDOWN_RUN_M / 2.0,
                  tile["route"]["goal_x"]):
            lo, hi = freeform.walkable_ground_span(tile["overlays"], x, tile["size_y"] / 2.0)
            apron = ((hi - lo) - sc.UPDOWN_WIDTH_M) / 2.0
            assert apron == pytest.approx(sc.UPDOWN_APRON_M), (depth, x, lo, hi)
        # Every walkable point outside the flight is under a wall, which is what makes the
        # apron unusable rather than merely narrow.
        walls = [o for o in tile["overlays"] if o["kind"] == "obstacle"]
        for w in walls:
            outer = abs(w["pos_xy"][1] - tile["size_y"] / 2.0) + w["size_xyz"][1] / 2.0
            assert outer == pytest.approx(tile["size_y"] / 2.0), depth


def test_the_walls_span_mjevals_own_length_and_only_a_VISUAL_one_claims_the_clone(courses):
    """Wall geometry, and the invariant that survived making them solid.

    mjeval's ``STAIR_WALLS`` calls visual-only "not negotiable", for one specific reason: a
    solid wall this close to the tread edge is swept by the reference's own arm swing, which
    blocks a wrist and fires ``ee_pose_error`` -- ending the episode for something no policy
    could fix.  **That termination does not exist on the fine-tune task** (its enders are the
    time-out, the absolute height floor, the tilt cone and the goal), so the failure the flag
    guarded against cannot happen there, and solid walls buy a real signal: they enter the
    collider SDF, so the terrain reward charges a plan that puts a limb inside one.

    What still has to hold is the CLAIM: solid walls are different physics, so a tile carrying
    them must not advertise itself as an mjeval clone -- a rate measured against solid walls
    cannot be read against an mjeval rate. That implication is what this pins, in both
    directions, rather than the flag's value.
    """
    for depth in courses:
        for collidable in (False, True):
            tile = sc.make_updown_tile(0, mj_terrains.X_START, depth,
                                       wall_collidable=collidable)
            walls = [o for o in tile["overlays"] if o["kind"] == "obstacle"]
            assert len(walls) == 2, depth
            for w in walls:
                assert w["decor"] is (not collidable), (depth, collidable)
                assert w["walkable_top"] is False
            claim = tile["updown"]["mjeval_course"]
            assert bool(claim) is (not collidable), (
                f"a tile with collidable={collidable} walls claims mjeval_course={claim!r}; "
                "solid walls are different physics and must not advertise the clone.")
        # And the geometry itself is mjeval's under either flag -- only the collider differs.
        tile = sc.make_updown_tile(0, mj_terrains.X_START, depth)
        walls = [o for o in tile["overlays"] if o["kind"] == "obstacle"]
        for w in walls:
            assert w["size_xyz"][1] == pytest.approx(freeform.WALL_T_M)
            assert w["size_xyz"][2] == pytest.approx(mj_terrains.STAIR_WALL_HEIGHT)
            # mjeval's span: lead before the first riser, `steps * run * 2 + platform` after it.
            span = sc.UPDOWN_STEPS * sc.UPDOWN_RUN_M * 2.0 + depth
            assert w["size_xyz"][0] == pytest.approx(span + mj_terrains.STAIR_WALL_LEAD), depth
            # Face TOUCHING the tread edge, zero clearance -- so the centre sits half a
            # thickness beyond it.  `STAIR_WALL_GAP` is a face clearance, not a centre offset.
            assert mj_terrains.STAIR_WALL_GAP == pytest.approx(0.0)
            face = abs(w["pos_xy"][1] - tile["size_y"] / 2.0) - freeform.WALL_T_M / 2.0
            assert face == pytest.approx(sc.UPDOWN_WIDTH_M / 2.0), depth
        assert sc.UPDOWN_WALL_HEIGHT_M == pytest.approx(mj_terrains.STAIR_WALL_HEIGHT)
        assert sc.UPDOWN_WALL_LEAD_M == pytest.approx(mj_terrains.STAIR_WALL_LEAD)
        assert sc.UPDOWN_WALLS is mj_terrains.STAIR_WALLS


def test_the_two_flights_have_the_same_riser_count(courses):
    """Both flights are ``UPDOWN_STEPS`` risers; the descent emits one FEWER tread.

    The last step down lands on the floor, so ``steps - 1`` treads make ``steps`` risers.  Off
    by one here means an asymmetric staircase that still looks plausible in a viewport.
    """
    for depth in courses:
        tile = sc.make_updown_tile(0, mj_terrains.X_START, depth)
        treads = [o for o in tile["overlays"]
                  if o["kind"] == "box" and o["walkable_top"]
                  and o["size_xyz"][0] == pytest.approx(sc.UPDOWN_RUN_M)]
        assert len(treads) == 2 * sc.UPDOWN_STEPS - 1, depth
        tops = sorted(round(o["base_z"] + o["size_xyz"][2], 9) for o in treads)
        want = sorted([round(n * sc.UPDOWN_RISE_M, 9) for n in range(1, sc.UPDOWN_STEPS + 1)]
                      + [round(n * sc.UPDOWN_RISE_M, 9)
                         for n in range(1, sc.UPDOWN_STEPS)])
        assert tops == pytest.approx(want), depth
        # The lowest walkable surface is the ground, so no z shift is owed and the absolute-z
        # fall termination has nothing to fire on.
        assert tile["route"]["start_z"] == pytest.approx(0.0)
        assert tile["route"]["goal_z"] == pytest.approx(0.0)


def test_every_tile_carries_the_same_staircase_and_differs_only_in_approach(courses):
    """Across the IC grid the obstacle is invariant; only the spawn distance moves.

    The design rests on this -- an IC grid over distance must not be a grid over terrain as
    well, or the critic is asked to generalise over two things at once.
    """
    for depth in courses:
        tiles = sc.generate_updown_manifest(
            sc.UPDOWN_TRAIN_APPROACHES_M + (sc.UPDOWN_HOLDOUT_APPROACH_M,), depth)
        shapes = set()
        for t in tiles:
            u = t["updown"]
            shapes.add((u["rise_m"], u["run_m"], u["steps"], u["width_m"],
                        u["platform_depth_m"], u["flight_height_m"], u["walls"]))
            # Every landmark follows from the approach alone.
            f = sc.updown_feature_x(u["approach_m"], depth)
            assert u["first_riser_x"] == pytest.approx(f["first_riser"])
            assert u["bottom_x"] == pytest.approx(f["bottom_x"])
        assert len(shapes) == 1, f"staircase differs across tiles at p={depth}: {shapes}"


def test_the_two_platform_depths_cannot_pool():
    """Each depth gets its own family, and the tag mirrors mjeval's course name.

    p080 is 6 m longer than p020 with the same staircase, so more distance is strictly more
    chances to fall and a p080 rate is lower than a p020 rate for a reason that is not the
    policy.  `binarize` thresholds per family and every report groups by it, so a shared family
    would silently average the two into one number.
    """
    fams = {d: sc.updown_family(d) for d in sc.UPDOWN_PLATFORM_DEPTHS_M}
    assert len(set(fams.values())) == len(fams), fams
    for d, fam in fams.items():
        assert fam == f"updown_p{int(round(d * 10)):03d}"
        assert _course(d).name == f"mjeval_{fam.replace('updown', 'updn')}"
        assert sc.make_updown_tile(0, 2.5, d)["family"] == fam
        # The clone CLAIM belongs to mjeval's own configuration, which is visual walls --
        # solid ones are different physics and withhold it (see the wall test).
        assert sc.make_updown_tile(0, 2.5, d, wall_collidable=False
                                   )["updown"]["mjeval_course"] == _course(d).name


def test_a_lengthened_runout_takes_its_own_family_and_disclaims_mjeval():
    """A non-default runout is a different course and must say so, in both places.

    It also must not add a field to the DEFAULT tile: `fingerprint_terrain_manifest` hashes the
    tile dict, so a key added unconditionally moves the fingerprint of geometry that has not
    changed and strands every rollout already recorded against it (box2 paid this once).
    """
    # `wall_collidable=False` throughout, so the only thing under test is the RUNOUT: the
    # wall flag withholds the mjeval claim on its own, and leaving it at the default would
    # make this test pass for the wrong reason.
    base = sc.make_updown_tile(0, 2.5, 2.0, wall_collidable=False)
    assert "runout_m" not in base["updown"]
    assert base["updown"]["mjeval_course"] == "mjeval_updn_p020"

    longer = sc.make_updown_tile(0, 2.5, 2.0, runout_m=3.0, wall_collidable=False)
    assert longer["updown"]["runout_m"] == pytest.approx(3.0)
    assert longer["updown"]["mjeval_course"] == ""
    assert longer["family"] != base["family"]
    assert longer["family"] == "updown_p020_r030"
    # The staircase does not move; only the goal and the tile length do.
    assert longer["updown"]["bottom_x"] == pytest.approx(base["updown"]["bottom_x"])
    assert longer["route"]["start_x"] == pytest.approx(base["route"]["start_x"])
    assert longer["route"]["goal_x"] - base["route"]["goal_x"] == pytest.approx(3.0)
    assert longer["size_x"] - base["size_x"] == pytest.approx(3.0)
    # And mjeval's own descending class is where the 3 m comes from.
    assert mj_terrains.DESCENT_RUNOUT == pytest.approx(3.0)


# --- the curb -------------------------------------------------------------------------
def test_the_curb_is_out_of_distribution_in_the_way_it_claims_to_be():
    """The step heights against the trained box grid, and the top against the trained depth.

    These are the claims the whole course rests on, and both live on the ``transfer`` side --
    ``TRAINED_BOX_DEPTH`` is only in ``generate_stairs_scene`` -- so this is where they can be
    checked rather than asserted in a docstring.
    """
    trained = freeform.TRAINED_BOX_HEIGHTS
    below = [h for h in sc.CURB_TRAIN_HEIGHTS_M if h < min(trained)]
    assert len(below) == 3, (
        f"the point of the height axis is rungs BELOW every trained box ({min(trained)} m); "
        f"got {below} of {list(sc.CURB_TRAIN_HEIGHTS_M)}")
    # ...and exactly one rung AT the bottom of the grid, as the anchor.
    assert max(sc.CURB_TRAIN_HEIGHTS_M) == pytest.approx(min(trained))
    # 0.15 m is a trained stair RISER, which is what makes the context the variable rather
    # than the height. If the riser set moves, this course's middle rung should move with it.
    assert 0.15 in freeform.TRAINED_RISERS
    assert 0.15 in sc.CURB_TRAIN_HEIGHTS_M
    # The top is a SURFACE, not a box top: far past the trained box depth, and past box2's
    # own deliberately-OOD depth too.
    assert sc.CURB_TOP_DEPTH_M > 5.0 * gss.TRAINED_BOX_DEPTH
    assert sc.CURB_TOP_DEPTH_M > 3.0 * mj_terrains.EVAL_BOX_DEPTH


def test_the_curb_spans_the_corridor_so_it_cannot_be_walked_around():
    """The step is as wide as the tile; a curb runs across your path.

    box2's boxes deliberately do NOT -- a 2.59 m box in a 6 m corridor leaves 1.7 m of floor
    either side -- so this is the one lateral difference between the two courses and it is the
    difference between an obstacle and an object.
    """
    for h in sc.CURB_TRAIN_HEIGHTS_M:
        tile = sc.make_curb_tile(0, h)
        step = [o for o in tile["overlays"]
                if o["size_xyz"][2] == pytest.approx(h) and o["base_z"] == pytest.approx(0.0)]
        assert len(step) == 1, h
        assert step[0]["size_xyz"][1] == pytest.approx(tile["size_y"]), h
        assert step[0]["size_xyz"][0] == pytest.approx(sc.CURB_TOP_DEPTH_M), h
        # The goal sits ON the step, so the robot never comes back down.
        assert tile["route"]["goal_z"] == pytest.approx(h), h
        assert tile["route"]["goal_x"] < step[0]["pos_xy"][0] + step[0]["size_xyz"][0] / 2.0


def test_the_curb_shares_box2s_lateral_geometry_and_approach():
    """Same corridor width, same run-up, same goal band -- so box2 is its comparator.

    ``box2`` is imported here only to compare; ``step_courses`` deliberately does not import it,
    so that an edit to box2's grid cannot move this library's terrain fingerprint.  That makes
    this the check that the two are still the SAME corridor.
    """
    box2 = pytest.importorskip("motion_gen.terrain.box2")
    assert sc.CURB_HALF_WIDTH_M == pytest.approx(box2.GOAL_HALF_WIDTH_M)
    assert sc.CURB_TRAIN_SPEEDS_MPS == pytest.approx(box2.TRAIN_SPEEDS_MPS)
    assert sc.CURB_HOLDOUT_SPEED_MPS == pytest.approx(box2.HOLDOUT_SPEED_MPS)
    curb = sc.make_curb_tile(0, 0.20)
    b2 = box2.make_box2_tile(0, sc.APPROACH_M)
    assert curb["size_y"] == pytest.approx(b2["size_y"])
    assert curb["route"]["start_x"] == pytest.approx(b2["route"]["start_x"])
    assert curb["route"]["start_y"] == pytest.approx(b2["route"]["start_y"])
    assert curb["route"]["goal_half_width"] == pytest.approx(b2["route"]["goal_half_width"])
    # The first obstacle face is at the same distance out on both courses.
    assert curb["curb"]["riser_x"][0] == pytest.approx(box2.box_faces(sc.APPROACH_M)[0][0])
    # And up-into-down starts its staircase at the same place, so all three are comparable
    # at the shared 2.5 m approach.
    assert sc.updown_feature_x(sc.APPROACH_M, 2.0)["first_riser"] == pytest.approx(
        curb["curb"]["riser_x"][0])
    # The approach GRID is box2's too, restated rather than imported -- equal on purpose.
    assert sc.UPDOWN_TRAIN_APPROACHES_M == pytest.approx(box2.TRAIN_SPAWN_DISTANCES_M)
    assert sc.UPDOWN_HOLDOUT_APPROACH_M == pytest.approx(box2.HOLDOUT_SPAWN_DISTANCE_M)


# --- the holdouts ---------------------------------------------------------------------
def test_both_holdouts_are_interior_and_off_grid():
    """A held-out cell must test interpolation, not extrapolation past the training range."""
    for grid, held in ((sc.CURB_TRAIN_HEIGHTS_M, sc.CURB_HOLDOUT_HEIGHT_M),
                       (sc.CURB_TRAIN_SPEEDS_MPS, sc.CURB_HOLDOUT_SPEED_MPS),
                       (sc.UPDOWN_TRAIN_APPROACHES_M, sc.UPDOWN_HOLDOUT_APPROACH_M)):
        assert held not in grid, (grid, held)
        assert min(grid) < held < max(grid), (grid, held)
    # The updown SPEED is deliberately NOT held out: its band is continuous, so the holdout
    # speed is inside it by construction. What that file holds out is the approach DISTANCE
    # (checked above) -- holding out the command too would confound the two axes and make a
    # holdout gap unattributable to either.
    lo, hi = sc.UPDOWN_SPEED_RANGE_MPS
    assert lo < sc.UPDOWN_HOLDOUT_SPEED_MPS < hi, (sc.UPDOWN_SPEED_RANGE_MPS,
                                                   sc.UPDOWN_HOLDOUT_SPEED_MPS)


def test_the_updown_speed_band_stays_inside_the_trained_stair_range():
    """The commanded band must not reach past what the library ever showed on STAIRS.

    Above the ladder's top the library's only answer to a fast forward approach is a JUMP,
    which is the documented mechanism behind reading a staircase as a box -- so a stair course
    commanded past it measures a coverage hole rather than the policy.

    **The band is CONTINUOUS, so it deliberately samples BETWEEN rungs**, and that is why this
    tests the ENDPOINTS rather than membership. Interpolating inside a covered interval is not
    the same hole: every speed drawn has trained neighbours a few hundredths either side.
    `FmTrackingCommand._reset_freeform_command` draws `uniform_(min, max)` per episode, so a
    discrete ladder was never enforced anywhere and asserting membership described nothing
    that ran.
    """
    ladder = mj_terrains.STAIR_SPEEDS
    lo, hi = sc.UPDOWN_SPEED_RANGE_MPS
    assert lo >= min(ladder) - 1e-9, (lo, min(ladder))
    assert hi <= max(ladder) + 1e-9, (
        f"the band reaches {hi} m/s but the library only ever showed stairs up to "
        f"{max(ladder)} m/s; past that its answer to a fast approach is a jump.")
    # And the gap to the ceiling, so a future widening has to move this number knowingly:
    # 2.5 against 2.600 is 0.1 m/s of margin, which is deliberate and thin.
    assert hi <= 2.5 + 1e-9, ("the band was widened past 2.5 m/s; the trained stair ceiling "
                              "is 2.600, so there is very little room left -- confirm that "
                              "intentionally.")
