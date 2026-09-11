# Copyright 2026 Zachary Olkin. All rights reserved.

"""The free-form terrain generator's schema, geometry invariants and labelling.

Every assertion here stands for a failure that is silent or far from its cause: a manifest the RL
loader refuses hours into a run, a descending section below the world floor that fires the fall
termination on a good descent, a wall buried half a thickness inside the flight it flanks, or a
"regulation" label on a riser/tread pair no real staircase has.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from motion_gen import features as F
from motion_gen.terrain import freeform as ff

CFG = ff.FreeformConfig(seed=3, num_flat=4, num_boxes=8, num_stairs=16)


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    """One generated manifest, shared by every test that only reads it."""
    man = ff.generate_freeform_manifest(CFG)
    ff.validate_manifest(man)
    return man


def _tiles(manifest: list[dict], family: str) -> list[dict]:
    """The manifest's tiles of one family."""
    return [m for m in manifest if m["family"] == family]


# ---------------------------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------------------------
def test_every_tile_carries_the_schema_the_loader_parses(manifest):
    """The keys ``clip_terrain_loader`` indexes without a ``.get``, plus the two it refuses on."""
    assert len(manifest) == CFG.num_tiles
    for m in manifest:
        assert {"terrain_id", "type", "size_x", "size_y", "overlays"} <= set(m)
        assert m["type"] == "terrain", "a new type string would need edits in five places"
        assert m["has_floor"] is False, "the section authors its own ground"
        assert m["frame_yaw"] == 0.0, "stated, not omitted -- the anchor mask resolves in it"
        assert m["generator"] == ff.GENERATOR_TAG


def test_every_overlay_would_survive_overlays_from_manifest(manifest):
    """The two refusals in ``_overlays_from_manifest``, plus ``build_overlay``'s extent check."""
    for m in manifest:
        for ov in m["overlays"]:
            assert ov["kind"] in ff.OVERLAY_KINDS
            assert ov["yaw"] == 0.0, "a yawed overlay is unrotated in Isaac, rotated in MuJoCo"
            assert all(float(v) > 0.0 for v in ov["size_xyz"])
            assert ov["variant"] in ("solid", "floating", "floating_center_leg",
                                     "floating_corner_legs")


def test_overlay_kinds_match_the_rl_loaders_frozenset():
    """The copied ``OVERLAY_KINDS`` must equal the real one, or a kind is accepted here and
    refused at load."""
    blocks = pytest.importorskip(
        "robot_rl.tasks.manager_based.robot_rl.terrains.blocks",
        reason="needs isaaclab (no simulator)")
    assert ff.OVERLAY_KINDS == blocks.OVERLAY_KINDS


def test_validate_manifest_refuses_a_yawed_overlay(manifest):
    """The generator never emits one, but the validator is the gate for a hand edit."""
    bad = [dict(manifest[0])]
    bad[0]["overlays"] = [dict(manifest[0]["overlays"][0], yaw=0.3)]
    with pytest.raises(ValueError, match="yaw"):
        ff.validate_manifest(bad)


def test_validate_manifest_refuses_a_duplicate_terrain_id(manifest):
    """A duplicate id builds one tile and orphans the other, with no message anywhere."""
    with pytest.raises(ValueError, match="duplicate"):
        ff.validate_manifest([manifest[0], dict(manifest[1], terrain_id=manifest[0]["terrain_id"])])


# ---------------------------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------------------------
def test_route_fields_are_exactly_what_the_rl_side_reads():
    """The producing and consuming halves name the same seven fields.

    ``_freeform_route_table`` raises on a missing one, at the first reset of a run rather than at
    generation, so this is where the two halves are held together.
    """
    resets = pytest.importorskip(
        "robot_rl.tasks.manager_based.robot_rl.mdp.events.resets",
        reason="needs isaaclab (no simulator)")
    assert ff.ROUTE_FIELDS == resets.FREEFORM_ROUTE_FIELDS


def test_every_route_is_declared_complete_and_inside_its_tile(manifest):
    """A start off the tile spawns the robot over the void between tiles; a goal at or behind the
    start truncates every episode on its first step."""
    for m in manifest:
        r = m["route"]
        assert set(ff.ROUTE_FIELDS) <= set(r)
        assert r["goal_z"] is not None
        for xk, yk in (("start_x", "start_y"), ("goal_x", "goal_y")):
            assert 0.0 <= r[xk] <= m["size_x"] and 0.0 <= r[yk] <= m["size_y"]
        assert r["goal_x"] > r["start_x"]
        assert r["goal_half_width"] > 0.0
        assert ff.route_length(r) > 5.0
        # The RL reset jitters the spawn by +-0.25 m in the tile frame; it may not fall off.
        assert r["start_x"] >= 0.25 and r["start_y"] >= 0.25


def test_the_goal_lies_on_its_own_route(manifest):
    """Otherwise the steering aims down one line while the goal termination watches another,
    which reads as a policy that walks past its goal."""
    for m in manifest:
        _along, lateral = ff.route_project(m["route"], (m["route"]["goal_x"],
                                                        m["route"]["goal_y"]))
        assert lateral == pytest.approx(0.0, abs=1e-9)


def test_the_goal_band_stays_on_walkable_ground_on_every_family(manifest):
    """The band is SIZED by the section but CENTRED on the goal, and an angled route moves the
    goal off the tile's centreline -- so containment is not implied by anything else here.

    Checked against the EFFECTIVE band. ``mdp.freeform_goal_reached`` clamps the declared
    half-width up to :data:`GOAL_HALF_WIDTH_FLOOR_M`, so the widened band is what the
    termination measures, and that termination is wired ``time_out=True``: a robot over the
    void past the goal line would score a SUCCESSFUL truncation with its value bootstrapped.

    This is what the flat family got wrong. At ``goal_half_width = size_y / 2`` all 8 tiles of
    the default config at seed 0 covered 27.853 m of void and excluded 27.853 m of real ground,
    the same number twice because the two errors are one lateral offset.
    """
    for m in manifest:
        r = m["route"]
        half = max(r["goal_half_width"], ff.GOAL_HALF_WIDTH_FLOOR_M)
        lo, hi = ff.walkable_ground_span(m["overlays"], r["goal_x"], r["goal_y"])
        assert r["goal_y"] - half >= lo - 1e-9, f"tile {m['terrain_id']} covers void below"
        assert r["goal_y"] + half <= hi + 1e-9, f"tile {m['terrain_id']} covers void above"


def test_the_flat_goals_really_are_off_centre(manifest):
    """Without this the containment test above is vacuous on the one family that failed it:
    a goal on the centreline is contained by ``size_y / 2`` for free."""
    offsets = [abs(m["route"]["goal_y"] - m["size_y"] / 2.0) for m in _tiles(manifest, "flat")]
    assert max(offsets) > 0.5, f"no flat goal is off the tile centreline: {offsets}"
    # And the fit is what binds there, not the tile: the band is narrower than half the tile by
    # (at least) that offset.
    for m in _tiles(manifest, "flat"):
        off = abs(m["route"]["goal_y"] - m["size_y"] / 2.0)
        assert m["route"]["goal_half_width"] <= m["size_y"] / 2.0 - off + 1e-9


def test_validate_manifest_refuses_a_goal_band_over_the_void(manifest):
    """The rule the generator now obeys, restated as a gate -- this is the exact edit that was
    shipped: a flat band sized from the tile while centred on an off-centre goal."""
    flat = max(_tiles(manifest, "flat"),
               key=lambda m: abs(m["route"]["goal_y"] - m["size_y"] / 2.0))
    bad = dict(flat, route=dict(flat["route"], goal_half_width=flat["size_y"] / 2.0))
    with pytest.raises(ValueError, match="leaves the walkable ground"):
        ff.validate_manifest([bad])


def test_the_goal_band_is_refused_below_the_floor_the_rl_side_clamps_up_to():
    """A declared band under ``GOAL_HALF_WIDTH_FLOOR_M`` is not what the termination measures:
    it is widened back out, which is how a fitted band would leave the ground anyway."""
    overlays = [ff._ground_overlay(10.0, 1.0)]                 # a 1 m strip: 0.5 m either side
    with pytest.raises(ValueError, match="floor the RL termination clamps up to"):
        ff._fit_goal_half_width(0.4, overlays, (5.0, 0.5))


def test_walkable_ground_span_ignores_walls_and_merges_what_it_touches():
    """Decoration and non-walkable volumes are not floor; a box sitting on the ground extends
    nothing.  The stairwell walls are the live case -- they are collidable, so they would
    otherwise widen the interval by their own thickness."""
    ground = ff._ground_overlay(10.0, 4.0)
    box = ff._overlay(5.0, 2.0, 0.0, 1.0, 1.0, 0.4)            # on the ground, inside it
    wall = ff._overlay(5.0, 4.075, 0.0, 10.0, 0.15, 2.0,
                       kind="obstacle", walkable_top=False)     # beyond the ground's +y edge
    assert ff.walkable_ground_span([ground, box, wall], 5.0, 2.0) == pytest.approx((0.0, 4.0))
    # A point over nothing is a hole in the tile's own ground, not a 0-width interval.
    with pytest.raises(ValueError, match="no walkable overlay"):
        ff.walkable_ground_span([ground], 5.0, 9.0)


def test_route_project_reports_downrange_and_signed_lateral(manifest):
    """The two numbers the runtime derives everything else from."""
    r = _tiles(manifest, "stairs")[0]["route"]          # start_yaw == 0, so the algebra is plain
    assert r["start_yaw"] == 0.0
    along, lateral = ff.route_project(r, (r["start_x"] + 2.0, r["start_y"] + 0.4))
    assert along == pytest.approx(2.0)
    assert lateral == pytest.approx(0.4)
    # Not clamped: a point behind the start is negative, not zero.
    behind, _ = ff.route_project(r, (r["start_x"] - 1.0, r["start_y"]))
    assert behind == pytest.approx(-1.0)


def test_route_lookahead_recedes_and_does_not_clamp_at_the_goal(manifest):
    """Matches ``freeform_yaw_rate``: the aim point recedes with the robot rather than being
    reached, so the heading error stays meaningful through the goal band."""
    m = _tiles(manifest, "boxes")[0]
    r = m["route"]
    here = (r["start_x"], r["start_y"])
    assert ff.route_lookahead(r, here, 0.0) == pytest.approx(here)
    for lead in (0.5, 2.0, 5.0):
        aim = ff.route_lookahead(r, here, lead)
        along, lateral = ff.route_project(r, aim)
        assert along == pytest.approx(lead)
        assert lateral == pytest.approx(0.0, abs=1e-9)
    past_goal = ff.route_lookahead(r, (r["goal_x"], r["goal_y"]), 5.0)
    along, _ = ff.route_project(r, past_goal)
    assert along > ff.route_length(r), "the aim point must not be clamped to the goal"


def test_route_lookahead_is_measured_from_the_projection_not_the_robot(manifest):
    """A robot standing off to the side aims at the same point as one on the line."""
    r = _tiles(manifest, "flat")[0]["route"]
    on_line = ff.route_lookahead(r, (r["start_x"], r["start_y"]), 4.0)
    ux, uy = ff.route_direction(r)
    beside = (r["start_x"] - 1.5 * uy, r["start_y"] + 1.5 * ux)   # 1.5 m to the left
    assert ff.route_lookahead(r, beside, 4.0) == pytest.approx(on_line)


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------
def test_a_tile_depends_only_on_seed_and_terrain_id(manifest):
    """Regenerating with more tiles must leave every id that did not move byte-identical.

    The importer and both viewers rebuild the world from the manifest and never see this RNG, so
    a tile whose geometry depended on how many tiles preceded it would be irreproducible from
    the file alone.
    """
    again = ff.generate_freeform_manifest(ff.FreeformConfig(
        seed=CFG.seed, num_flat=CFG.num_flat, num_boxes=CFG.num_boxes, num_stairs=CFG.num_stairs))
    assert again == manifest
    # Growing the LAST family leaves every earlier tile alone.
    grown = ff.generate_freeform_manifest(ff.FreeformConfig(
        seed=CFG.seed, num_flat=CFG.num_flat, num_boxes=CFG.num_boxes,
        num_stairs=CFG.num_stairs + 5))
    assert grown[:CFG.num_tiles] == manifest


def test_a_different_seed_moves_the_geometry(manifest):
    """The complement: the seed has to reach the draws."""
    other = ff.generate_freeform_manifest(ff.FreeformConfig(
        seed=CFG.seed + 1, num_flat=CFG.num_flat, num_boxes=CFG.num_boxes,
        num_stairs=CFG.num_stairs))
    assert other != manifest


# ---------------------------------------------------------------------------------------------
# Stairs geometry
# ---------------------------------------------------------------------------------------------
def test_five_platforms_and_four_flights(manifest):
    """The family's shape, and the flights alternate with the platforms in x."""
    for m in _tiles(manifest, "stairs"):
        plats, flights = m["stair_platforms"], m["stair_flights"]
        assert len(plats) == 5 and len(flights) == 4
        for i, f in enumerate(flights):
            assert plats[i]["x1"] == pytest.approx(f["x0"])
            assert f["x1"] == pytest.approx(plats[i + 1]["x0"])
            assert f["z_from"] == pytest.approx(plats[i]["top_z"])
            assert f["z_to"] == pytest.approx(plats[i + 1]["top_z"])


def test_lowest_platform_top_is_exactly_zero(manifest):
    """Flights go up AND down, so without the offset a descending section walks below the world
    floor and an absolute-world-z fall termination fires on a perfectly good descent."""
    for m in _tiles(manifest, "stairs"):
        tops = [p["top_z"] for p in m["stair_platforms"]]
        assert min(tops) == pytest.approx(0.0, abs=1e-12)
        assert min(tops) >= 0.0
        for ov in m["overlays"]:
            assert ov["base_z"] >= -ff.GROUND_THICKNESS_M - 1e-9
    # And the family must actually descend somewhere, or the offset rule is untested.
    assert any(f["direction"] == "down"
               for m in _tiles(manifest, "stairs") for f in m["stair_flights"])
    assert any(m["route"]["goal_z"] < m["route"]["start_z"] for m in _tiles(manifest, "stairs"))


def test_riser_and_tread_are_uniform_within_a_flight(manifest):
    """IBC 1011.5.4: uniform within a flight (3/8 in of variation allowed; this emits none).

    Checked on the emitted TREAD BOXES, not on the recorded scalars -- the scalars could agree
    while the layout that consumes them did not.
    """
    for m in _tiles(manifest, "stairs"):
        cy = m["size_y"] / 2.0
        for f in m["stair_flights"]:
            treads = [ov for ov in m["overlays"]
                      if ov["kind"] == "box" and abs(ov["pos_xy"][1] - cy) < 1e-9
                      and f["x0"] - 1e-9 <= ov["pos_xy"][0] - ov["size_xyz"][0] / 2.0
                      and ov["pos_xy"][0] + ov["size_xyz"][0] / 2.0 <= f["x1"] + 1e-9
                      and abs(ov["size_xyz"][0] - f["tread"]) < 1e-9]
            assert len(treads) == f["steps"] - 1, "n risers emit n-1 intermediate treads"
            if not treads:
                continue
            assert len({round(ov["size_xyz"][0], 12) for ov in treads}) == 1
            tops = sorted(ov["base_z"] + ov["size_xyz"][2] for ov in treads)
            if len(tops) > 1:
                assert np.allclose(np.diff(tops), f["riser"], atol=1e-9)
            lo, hi = sorted((f["z_from"], f["z_to"]))
            assert lo < tops[0] and tops[-1] < hi


def test_stair_ranges_are_honoured(manifest):
    """Width, platform length and step count stay inside the config's own ranges."""
    for m in _tiles(manifest, "stairs"):
        assert CFG.stair_width_range[0] <= m["stair_width"] <= CFG.stair_width_range[1]
        for p in m["stair_platforms"]:
            assert CFG.platform_len_range[0] <= p["length"] <= CFG.platform_len_range[1]
        for f in m["stair_flights"]:
            # The height cap only ever REDUCES a flight, so the low end is the binding check.
            assert CFG.steps_range[0] <= f["steps"] <= CFG.steps_range[1]
            assert f["variant"] in ("solid", "floating")
        assert max(p["top_z"] for p in m["stair_platforms"]) <= CFG.max_structure_height_m + 1e-9


def test_an_unreachable_height_cap_is_refused_rather_than_exceeded(manifest):
    """The shortening loop bottoms out at one riser per flight, so a cap under that floor cannot
    be met -- and ``validate_manifest`` cannot catch it either, since the manifest does not carry
    the cap.  It used to return quietly: this exact config generated and VALIDATED a tile
    spanning 0.714 m against its own 0.5 m cap, 43% over.
    """
    cfg = ff.FreeformConfig(seed=3, num_flat=0, num_boxes=0, num_stairs=6,
                            max_structure_height_m=0.5)
    with pytest.raises(ValueError, match="is unreachable"):
        ff.generate_freeform_manifest(cfg)
    # The floor is set by the DIRECTIONS drawn, not by the config: the offending tile's four
    # one-riser flights all descend, so their risers sum instead of cancelling.
    arm = ff.SAMPLING_ARMS[cfg.index_in_family(2) % len(ff.SAMPLING_ARMS)]
    flights = [dict(f, steps=1) for f in ff._draw_stair_flights(2, cfg, arm)]
    levels = ff._platform_levels(flights)
    assert max(levels) - min(levels) == pytest.approx(0.7137, abs=1e-3)
    assert {f["direction"] for f in flights} == {"down"}


def test_the_shipped_height_cap_never_binds_that_way(manifest):
    """The complement: at the shipped 4.0 m the refusal is unreachable, so this fix cannot cost
    a tile.  Measured over 960 tiles / 40 seeds -- 0 refusals and 0 violations; the one-riser
    floor reaches 0.720 m there, which is the number a smaller cap has to clear."""
    assert ff.FreeformConfig().max_structure_height_m == 4.0
    for m in _tiles(manifest, "stairs"):
        floor_levels = ff._platform_levels([dict(f, steps=1) for f in m["stair_flights"]])
        assert max(floor_levels) - min(floor_levels) <= CFG.max_structure_height_m


def test_walls_touch_the_flight_face(manifest):
    """A wall's offset is a FACE clearance, not a centre offset.

    ``generate_stairs_scene.lane_wall_geoms`` uses ``width / 2 + gap + thickness / 2``;
    ``ground_variants.WALL_GAP_M`` is a CENTRE offset, and copying that convention buries half
    the wall thickness inside the flight.  "Touching the width" is ``gap == 0`` in the face
    convention.
    """
    seen = 0
    for m in _tiles(manifest, "stairs"):
        cy, w = m["size_y"] / 2.0, m["stair_width"]
        walls = [ov for ov in m["overlays"] if ov["kind"] == "obstacle"]
        assert len(walls) == len(m["stair_wall_sides"])
        assert set(m["stair_wall_sides"]) <= {"left", "right"}
        for side in m["stair_wall_sides"]:
            sign = 1.0 if side == "left" else -1.0
            face = cy + sign * w / 2.0
            assert any(abs((ov["pos_xy"][1] - sign * ov["size_xyz"][1] / 2.0) - face) < 1e-9
                       for ov in walls)
            seen += 1
        for ov in walls:
            assert ov["decor"] is False, "stairwell walls are COLLIDABLE by decision"
            assert ov["walkable_top"] is False
    assert seen > 0, "no wall was drawn at all -- the wall arm never ran"
    counts = {len(m["stair_wall_sides"]) for m in _tiles(manifest, "stairs")}
    assert counts == {0, 1, 2}, f"the 0/1/2-wall arms did not all appear: {counts}"


def test_the_wall_band_is_reserved_whether_or_not_a_wall_is_drawn(manifest):
    """Otherwise the wall draw changes the footprint, and the pooled CLASS footprint every tile
    is padded to moves with it."""
    for m in _tiles(manifest, "stairs"):
        assert m["size_y"] == pytest.approx(
            m["stair_width"] + 2.0 * ff.WALL_T_M + 2.0 * ff.SECTION_MARGIN_M)


def test_floating_flights_use_the_overlay_variant(manifest):
    """The legacy ``StairDecorCfg.float_thickness`` tread builder is dead; floating rides
    ``variant="floating"`` + ``slab_thickness``, which every renderer expands identically."""
    floating = [f for m in _tiles(manifest, "stairs") for f in m["stair_flights"]
                if f["variant"] == "floating"]
    assert floating, "no floating flight was drawn"
    for f in floating:
        assert (CFG.stair_slab_thickness_range[0] <= f["slab_thickness"]
                <= min(CFG.stair_slab_thickness_range[1], f["riser"]))
    for m in _tiles(manifest, "stairs"):
        for ov in m["overlays"]:
            if ov["variant"] == "floating":
                assert 0.0 < ov["slab_thickness"] <= ov["size_xyz"][2]


# ---------------------------------------------------------------------------------------------
# Stair step sampling: the two arms
# ---------------------------------------------------------------------------------------------
def test_the_two_sampling_arms_are_labelled_and_split_about_evenly(manifest):
    """The label is a REQUIREMENT: success is reported separately for in-band and OOD geometry."""
    arms = [m["stair_sampling_arm"] for m in _tiles(manifest, "stairs")]
    assert set(arms) == set(ff.SAMPLING_ARMS)
    assert abs(arms.count("regulation") - arms.count("trained")) <= 1
    for m in _tiles(manifest, "stairs"):
        for f in m["stair_flights"]:
            assert f["sampling"] == m["stair_sampling_arm"]
            assert (f["code"] is None) == (f["sampling"] == "trained")


def test_regulation_flights_obey_blondel_and_their_own_code_box(manifest):
    """Sampling riser and tread independently inside a code box produces pairs no real staircase
    has; the Blondel coupling is what makes this arm mean anything."""
    codes = {c.name: c for c in ff.STAIR_CODES}
    lo, hi = ff.BLONDEL_RANGE_M
    n = 0
    for m in _tiles(manifest, "stairs"):
        for f in m["stair_flights"]:
            if f["sampling"] != "regulation":
                continue
            n += 1
            c = codes[f["code"]]
            assert c.riser_min - 1e-12 <= f["riser"] <= c.riser_max + 1e-12
            assert c.tread_min - 1e-12 <= f["tread"] <= c.tread_max + 1e-12
            assert lo - 1e-12 <= 2.0 * f["riser"] + f["tread"] <= hi + 1e-12
            if c.pitch_deg_range is not None:
                pitch = math.degrees(math.atan2(f["riser"], f["tread"]))
                assert c.pitch_deg_range[0] - 1e-9 <= pitch <= c.pitch_deg_range[1] + 1e-9
    assert n > 0
    assert {f["code"] for m in _tiles(manifest, "stairs") for f in m["stair_flights"]
            if f["code"]} == set(codes), "a code was never drawn"


def test_trained_flights_land_on_the_measured_grid(manifest):
    """And ascending / descending draw from their OWN tread set: the library used different ones,
    so a pooled draw would put a descending tread under an ascending flight."""
    n = 0
    for m in _tiles(manifest, "stairs"):
        for f in m["stair_flights"]:
            if f["sampling"] != "trained":
                continue
            n += 1
            assert f["riser"] in ff.TRAINED_RISERS
            assert f["tread"] in (ff.TRAINED_TREADS_UP if f["direction"] == "up"
                                  else ff.TRAINED_TREADS_DOWN)
    assert n > 0


def test_sample_regulation_step_raises_rather_than_clamping_an_infeasible_box():
    """A clamped pair satisfies the box and breaks the coupling -- silently making the arm a
    statement about a rectangle."""
    impossible = ff.StairCode(name="impossible", riser_min=0.30, riser_max=0.40,
                              tread_min=0.25, tread_max=0.30, pitch_deg_range=None, cite="test")
    with pytest.raises(ValueError, match="no longer intersect"):
        ff.sample_regulation_step(np.random.default_rng(0), impossible)


def test_trained_constants_match_generate_stairs_scene():
    """The copies here vs the file that OWNS them.

    ``transfer/sim/generate_stairs_scene.py`` is the source of truth for every trained constant
    (it owns the MuJoCo playground emitter and the measurements).  It is not importable from
    ``motion_gen`` -- ``transfer/sim`` is not a package -- so the numbers are copied, and this is
    what stops the copy from rotting the way the ROS URDF and the metadata fixture did.
    """
    path = F._REPO_ROOT / "transfer" / "sim" / "generate_stairs_scene.py"
    spec = importlib.util.spec_from_file_location("_gss_for_test", path)
    gss = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: ``dataclasses`` resolves a string annotation through
    # ``sys.modules[cls.__module__]`` and raises on a module that is not there yet.
    sys.modules[spec.name] = gss
    spec.loader.exec_module(gss)
    assert ff.TRAINED_RISERS == gss.TRAINED_RISERS
    assert ff.TRAINED_TREADS == gss.TRAINED_TREADS
    assert ff.TRAINED_TREADS_UP == gss.TRAINED_TREADS_UP
    assert ff.TRAINED_BOX_HEIGHTS == gss.TRAINED_BOX_HEIGHTS
    # The DOWN half is stated only in that file's TRAINED_TREADS comment, so pin the split.
    assert set(ff.TRAINED_TREADS_UP) | set(ff.TRAINED_TREADS_DOWN) == set(gss.TRAINED_TREADS)
    assert set(ff.TRAINED_TREADS_DOWN) <= set(gss.TRAINED_TREADS)


def test_the_real_stair_widths_are_inside_the_sampled_range():
    """The playground's tape-measured widths -- 4 ft, 5 ft, outdoor -- must all be reachable.

    The training library has a 0.62 m HOLE at 1.10-1.72 m, where the two most common real widths
    sit; sampling continuously across 1.219-3.048 m is what fills it.
    """
    path = F._REPO_ROOT / "transfer" / "sim" / "generate_stairs_scene.py"
    spec = importlib.util.spec_from_file_location("_gss_widths", path)
    gss = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gss
    spec.loader.exec_module(gss)
    lo, hi = ff.FreeformConfig().stair_width_range
    for name, w in gss.REAL_STAIR_WIDTHS.items():
        assert lo - 0.01 <= w <= hi, f"the {name} real width {w} is outside [{lo}, {hi}]"


# ---------------------------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------------------------
def test_boxes_never_overlap(manifest):
    """Guaranteed by construction -- boxes occupy disjoint x bands separated by a clear gap -- so
    this is the check that the construction still holds."""
    for m in _tiles(manifest, "boxes"):
        boxes = m["boxes"]
        for a, b in zip(boxes[:-1], boxes[1:]):
            assert b["x0"] - a["x1"] >= CFG.box_gap_range[0] - 1e-9
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) <= 0.0


def test_a_box_never_seals_the_corridor(manifest):
    """The two side clearances sum to ``corridor_width - box_width``, so bounding the width by
    ``corridor - 2 * clearance`` leaves at least ``clearance`` on ONE side of every box."""
    for m in _tiles(manifest, "boxes"):
        corridor = m["corridor_width"]
        for b in m["boxes"]:
            box_w = b["size_xyz"][1]
            assert box_w <= corridor - 2.0 * CFG.box_side_clearance_m + 1e-9
            # Measured against the ROUTE, which is what the corridor is a band around.
            _along, lateral = ff.route_project(m["route"], b["pos_xy"])
            left = corridor / 2.0 - (lateral + box_w / 2.0)
            right = (lateral - box_w / 2.0) + corridor / 2.0
            assert min(left, right) >= -1e-6, "the box left the corridor"
            assert max(left, right) >= CFG.box_side_clearance_m - 1e-6


def test_boxes_are_axis_aligned_and_on_the_trained_height_grid(manifest):
    """Height is the one axis kept in distribution; the layout deliberately is not."""
    for m in _tiles(manifest, "boxes"):
        for b in m["boxes"]:
            assert b["size_xyz"][2] in ff.TRAINED_BOX_HEIGHTS
            assert b["base_z"] == 0.0
            assert b["role"] in ("jump", "walk_along")
        assert all(ov["yaw"] == 0.0 for ov in m["overlays"])


def test_the_route_is_angled_so_an_axis_aligned_box_is_met_off_square(manifest):
    """The approach angle is the whole mechanism: the box cannot be rotated, so the route is."""
    yaws = [abs(m["route"]["start_yaw"]) for m in _tiles(manifest, "boxes")]
    assert max(yaws) > math.radians(5.0), "no boxes section is angled; every box met head-on"
    assert max(yaws) <= math.radians(CFG.box_max_approach_deg) + 1e-12
    # Every box centre sits on the route, offset only laterally.
    for m in _tiles(manifest, "boxes"):
        for b in m["boxes"]:
            _along, lateral = ff.route_project(m["route"], b["pos_xy"])
            assert lateral == pytest.approx(b["route_offset"], abs=1e-9)


def test_both_box_roles_appear(manifest):
    """A long box is what makes 'walk along it first, then jump off' reachable at all."""
    roles = {b["role"] for m in _tiles(manifest, "boxes") for b in m["boxes"]}
    assert roles == {"jump", "walk_along"}
    for m in _tiles(manifest, "boxes"):
        for b in m["boxes"]:
            if b["role"] == "walk_along":
                assert b["size_xyz"][0] >= CFG.box_long_depth_range[0]


# ---------------------------------------------------------------------------------------------
# Flat
# ---------------------------------------------------------------------------------------------
def test_flat_is_one_authored_ground_overlay(manifest):
    """Authored, not left to ``has_floor`` -- see the module docstring for the class-footprint
    artefact that would otherwise put a 6 m section on a 46 m floor."""
    for m in _tiles(manifest, "flat"):
        assert len(m["overlays"]) == 1
        ov = m["overlays"][0]
        assert ov["walkable_top"] is True and ov["decor"] is False
        assert ov["base_z"] + ov["size_xyz"][2] == pytest.approx(0.0)
        assert ov["size_xyz"][:2] == [m["size_x"], m["size_y"]]
        assert CFG.flat_size_x_range[0] <= m["size_x"] <= CFG.flat_size_x_range[1]


def test_flat_routes_are_not_all_straight_down_the_tile(manifest):
    """A route along the tile axis never asks the yaw controller for anything, and turning IS in
    distribution on flat (the training library commands up to 0.80 rad/s there)."""
    yaws = [abs(m["route"]["start_yaw"]) for m in _tiles(manifest, "flat")]
    assert max(yaws) > math.radians(2.0)
    assert max(yaws) <= math.radians(CFG.flat_max_approach_deg) + 1e-12


# ---------------------------------------------------------------------------------------------
# Every family
# ---------------------------------------------------------------------------------------------
def test_every_family_authors_its_ground_at_z_zero(manifest):
    """One datum across the three families: the ground's top face is z = 0 everywhere, and the
    route's ``start_z`` is the surface actually under the spawn."""
    for m in manifest:
        ground = m["overlays"][0]
        assert ground["decor"] is False and ground["walkable_top"] is True
        assert ground["base_z"] + ground["size_xyz"][2] == pytest.approx(0.0)
        assert ground["size_xyz"][0] == pytest.approx(m["size_x"])
        assert ground["size_xyz"][1] == pytest.approx(m["size_y"])
        if m["family"] != "stairs":
            assert m["route"]["start_z"] == 0.0 and m["route"]["goal_z"] == 0.0
    for m in _tiles(manifest, "stairs"):
        assert m["route"]["start_z"] == pytest.approx(m["stair_platforms"][0]["top_z"])
        assert m["route"]["goal_z"] == pytest.approx(m["stair_platforms"][-1]["top_z"])


def test_every_overlay_stays_inside_its_tile(manifest):
    """Tiles are packed into a shared field with void between them, so geometry that leaves its
    footprint lands on a neighbour."""
    for m in manifest:
        for ov in m["overlays"]:
            px, py = ov["pos_xy"]
            ex, ey, _ez = ov["size_xyz"]
            assert -1e-9 <= px - ex / 2.0 and px + ex / 2.0 <= m["size_x"] + 1e-9
            assert -1e-9 <= py - ey / 2.0 and py + ey / 2.0 <= m["size_y"] + 1e-9
