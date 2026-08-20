# Copyright 2026 Zachary Olkin. All rights reserved.

"""A width edit must never move a surface out from under the feet.

The pass rewrites terrain that 10 000 recorded clips already walk on, and the motion cannot
adapt. So the tests are about the two directions an edit can be wrong:

* growing a surface must produce a STRICT SUPERSET of what it replaced -- the failure mode
  the plan names is "widen and re-centre", which looks like growth and is partly a move;
* shrinking one must still contain the whole foot envelope, allowance included.

Both are checked in ``_resize_y`` at edit time rather than by a sweep afterwards, so a bad
edit cannot be written in the first place.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from motion_gen.terrain.decorate import (ARMS, DecorateConfig, TREAD_WIDTH_FRAC,
                                         BOX_VARIANTS, _resize_y, _wall_segments,
                                         add_walls, classify, make_flat_rooms,
                                         classify_overlays, decorate_box_tile,
                                         decorate_transition_tile, has_width_variation,
                                         span, tile_rng)
from motion_gen.terrain.footprint import TileFootprint


def _overlay(pos_y: float, size_y: float, pos_x: float = 1.0, size_x: float = 0.3) -> dict:
    """One walkable box."""
    return {"kind": "box", "pos_xy": [pos_x, pos_y], "size_xyz": [size_x, size_y, 0.15],
            "base_z": 0.0, "walkable_top": True, "yaw": 0.0}


def _tile(terrain_id: int = 0, size_y: float = 4.0) -> dict:
    """A transition tile: two treads plus two full-width platforms, as the library has."""
    return {
        "terrain_id": terrain_id, "type": "transition", "size_x": 20.0, "size_y": size_y,
        "has_floor": True, "frame_yaw": 0.0, "family": "family_stairs_up.h5",
        "overlays": [
            _overlay(1.5, 0.9, pos_x=5.0), _overlay(1.5, 0.9, pos_x=6.0),
            _overlay(size_y / 2, size_y, pos_x=1.0, size_x=4.0),
            _overlay(size_y / 2, size_y, pos_x=15.0, size_x=4.0),
        ],
    }


def _footprint(terrain_id: int = 0, lo: float = 1.2, hi: float = 1.8) -> TileFootprint:
    """An envelope already including its allowance."""
    return TileFootprint(terrain_id=terrain_id, n_clips=10, y_min=lo + 0.3, y_max=hi - 0.3,
                         x_min=0.0, x_max=20.0, y_min_sized=lo, y_max_sized=hi)


def test_treads_and_platforms_are_separated_by_width():
    """The classifier is the whole basis of the pass; on the real library the two classes
    are 2 m apart, so the threshold is not a judgement call."""
    tile = _tile()
    kinds = [classify(o, tile["size_y"]) for o in tile["overlays"]]
    assert kinds == ["tread", "tread", "platform", "platform"]
    assert TREAD_WIDTH_FRAC < 1.0


def test_a_tile_draws_the_same_arm_regardless_of_iteration_order():
    """Seeded per terrain_id, not per position, so adding a tile cannot reshuffle the rest."""
    cfg = DecorateConfig(seed=7)
    first = [tile_rng(cfg.seed, t).choice(ARMS) for t in range(20)]
    second = [tile_rng(cfg.seed, t).choice(ARMS) for t in reversed(range(20))][::-1]
    assert first == second


def test_widening_produces_a_superset():
    """Growth may only add surface.

    Roles come from ``report.roles``, captured before the edit. Re-deriving them from the
    rewritten widths inspects the wrong overlays -- a widened tread now spans nearly the tile
    and reads as a platform, a narrowed platform reads as a tread. The first version of this
    test did that and failed on a narrowed platform it thought was a tread, which is how the
    same bug was found in the pass's own foot-support check.
    """
    for tid in range(30):
        tile = _tile(terrain_id=tid)
        before = [span(o) for o in tile["overlays"]]
        report = decorate_transition_tile(tile, _footprint(tid), DecorateConfig(seed=1))
        if report.arm == "narrow_landing":
            continue
        for o, role, (old_lo, old_hi) in zip(tile["overlays"], report.roles, before):
            if role != "tread":
                continue
            lo, hi = span(o)
            assert lo <= old_lo + 1e-9 and hi >= old_hi - 1e-9, f"tread shrank on tile {tid}"


def test_narrowing_still_covers_the_foot_envelope():
    """A landing may shrink, but not past where the feet actually go."""
    for tid in range(30):
        tile = _tile(terrain_id=tid)
        fp = _footprint(tid)
        report = decorate_transition_tile(tile, fp, DecorateConfig(seed=2))
        if report.arm == "wide_stairs":
            continue
        for o, role in zip(tile["overlays"], report.roles):
            if role != "platform":
                continue
            lo, hi = span(o)
            assert lo <= fp.y_min_sized + 1e-9, f"tile {tid} platform cut past the envelope"
            assert hi >= fp.y_max_sized - 1e-9, f"tile {tid} platform cut past the envelope"


def test_resize_refuses_a_widening_that_is_not_a_superset():
    """'Widen and re-centre' in miniature: same width, moved. Must be rejected."""
    o = _overlay(1.5, 0.9)
    with pytest.raises(ValueError, match="not a superset"):
        _resize_y(o, 1.5, 2.4, must_contain=(1.5, 2.4), widening=True)


def test_resize_refuses_dropping_the_required_span():
    """A narrowing that leaves the feet off the edge."""
    o = _overlay(2.0, 4.0)
    with pytest.raises(ValueError, match="does not cover"):
        _resize_y(o, 1.9, 2.1, must_contain=(1.0, 3.0), widening=False)


def test_a_degenerate_span_is_refused():
    """A zero-width overlay is not a thin surface, it is a build error downstream."""
    with pytest.raises(ValueError, match="degenerate"):
        _resize_y(_overlay(1.5, 0.9), 2.0, 2.0, must_contain=(2.0, 2.0), widening=False)


def test_an_envelope_outside_the_tile_is_an_error():
    """The manifest and the clips disagreeing about a tile must not be papered over."""
    tile = _tile(size_y=4.0)
    away = TileFootprint(terrain_id=0, n_clips=1, y_min=9.0, y_max=9.5, x_min=0.0,
                         x_max=1.0, y_min_sized=8.7, y_max_sized=9.8)
    with pytest.raises(ValueError, match="does not intersect"):
        decorate_transition_tile(tile, away, DecorateConfig())


def test_nothing_but_lateral_geometry_changes():
    """x, z, base_z, kind and walkable_top are none of this pass's business."""
    tile = _tile()
    before = copy.deepcopy(tile)
    decorate_transition_tile(tile, _footprint(), DecorateConfig(seed=3))
    for old, new in zip(before["overlays"], tile["overlays"]):
        assert old["size_xyz"][0] == new["size_xyz"][0]
        assert old["size_xyz"][2] == new["size_xyz"][2]
        assert old["pos_xy"][0] == new["pos_xy"][0]
        assert old["base_z"] == new["base_z"]
        assert old["kind"] == new["kind"]
        assert old["walkable_top"] == new["walkable_top"]
    assert {k: v for k, v in before.items() if k != "overlays"} == \
           {k: v for k, v in tile.items() if k != "overlays"}


def test_roles_are_captured_before_the_edit_not_after():
    """The classifier is width-based and the pass changes widths, so the two disagree.

    Pinned because the disagreement is silent: re-deriving roles from a decorated manifest
    returns a plausible answer for every overlay, just the wrong one.
    """
    tile = _tile(terrain_id=0, size_y=4.0)
    report = decorate_transition_tile(tile, _footprint(0), DecorateConfig(seed=5))
    if report.arm == "matched":
        after = classify_overlays(tile)
        assert after != report.roles, (
            "on a matched tile the widened treads and narrowed platforms should swap "
            "classes; if they no longer do, this test has stopped covering anything"
        )


def test_every_arm_is_reachable():
    """A weight vector that silently never produces an arm would halve the variety."""
    cfg = DecorateConfig(seed=11)
    drawn = {decorate_transition_tile(_tile(terrain_id=t), _footprint(t), cfg).arm
             for t in range(60)}
    assert drawn == set(ARMS)


def test_an_undecorated_manifest_reports_no_width_variation():
    """The structural answer to "has this been decorated?".

    Straight out of generation every stair platform is exactly its tile's width, because both
    generator paths stretch it there as a safety net. That is what makes the absence of
    variation a reliable signal, and why it is checked on the geometry rather than on an
    attribute somebody could set by hand.
    """
    assert not has_width_variation([_tile(terrain_id=t) for t in range(5)])


def test_a_decorated_manifest_reports_variation_and_cannot_be_decorated_twice():
    """Re-running is not idempotent -- the arms are seeded per tile, so a second pass narrows
    an already narrowed platform relative to its reduced width."""
    tiles = [_tile(terrain_id=t) for t in range(30)]
    for tile in tiles:
        decorate_transition_tile(tile, _footprint(int(tile["terrain_id"])),
                                 DecorateConfig(seed=4))
    assert has_width_variation(tiles), (
        "with 30 tiles across three arms at least one platform must have been narrowed"
    )


def test_variation_is_not_claimed_from_a_tile_of_another_type():
    """Box and terrain tiles are not touched by this pass, so they must not answer for it."""
    box = _tile(terrain_id=0)
    box["type"] = "box"
    box["overlays"][2]["size_xyz"][1] = 1.0        # narrow, but on a box tile
    assert not has_width_variation([box])


def _box_tile(terrain_id: int = 0) -> dict:
    """A box tile: a jump box and its landing platform, as the library has them."""
    return {
        "terrain_id": terrain_id, "type": "box", "size_x": 20.0, "size_y": 4.0,
        "has_floor": True, "frame_yaw": 0.0, "family": "family_box.h5",
        "overlays": [_overlay(2.0, 1.074, pos_x=13.5, size_x=0.463),
                     _overlay(2.0, 1.074, pos_x=16.1, size_x=4.712)],
    }


def test_box_variants_change_only_the_variant_field():
    """The walkable top and the footprint must survive untouched.

    That invariant is the entire safety argument: every variant puts its top at
    base_z + size_z and keeps the legs inside the solid box's own footprint, so no foot that
    was supported before is unsupported after and the height scan reads the same surface.
    Only what the DEPTH CAMERAS see changes.
    """
    tile = _box_tile()
    before = copy.deepcopy(tile)
    decorate_box_tile(tile, DecorateConfig(seed=0))
    for old, new in zip(before["overlays"], tile["overlays"]):
        assert new["variant"] in BOX_VARIANTS
        assert old["size_xyz"] == new["size_xyz"]
        assert old["pos_xy"] == new["pos_xy"]
        assert old["base_z"] == new["base_z"]
        assert old["walkable_top"] == new["walkable_top"]
        assert "variant" not in old


def test_box_variant_weights_must_sum_to_one():
    """A typo'd distribution would silently starve a variant, or oversample one."""
    cfg = DecorateConfig(box_variant_fracs=(0.5, 0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="must sum to 1"):
        decorate_box_tile(_box_tile(), cfg)


def test_every_box_variant_is_reachable():
    """All four, or the feature is narrower than it claims."""
    cfg = DecorateConfig(seed=3)
    seen = set()
    for t in range(120):
        tile = _box_tile(terrain_id=t)
        decorate_box_tile(tile, cfg)
        seen.update(o["variant"] for o in tile["overlays"])
    assert seen == set(BOX_VARIANTS)


def test_box_variants_are_seeded_per_tile():
    """Same reason the arms are: adding a tile must not reshuffle the others."""
    cfg = DecorateConfig(seed=9)
    a = [decorate_box_tile(_box_tile(terrain_id=t), cfg) for t in range(15)]
    b = [decorate_box_tile(_box_tile(terrain_id=t), cfg) for t in reversed(range(15))][::-1]
    assert a == b


def test_the_box_variant_stream_is_independent_of_the_arm_stream():
    """Both are seeded from terrain_id; if they shared a stream, a tile's box variant would
    be correlated with which arm a transition tile of the same id happened to draw."""
    cfg = DecorateConfig(seed=0)
    arm_stream = [tile_rng(cfg.seed, t).random() for t in range(10)]
    box_stream = [tile_rng(cfg.seed ^ 0x5EED, t).random() for t in range(10)]
    assert arm_stream != box_stream


def test_box_widening_only_ever_adds_surface():
    """A recorded landing cannot be invalidated by making its target bigger."""
    cfg = DecorateConfig(seed=2, box_widen_frac=1.0)
    for tid in range(30):
        tile = _box_tile(terrain_id=tid)
        before = [span(o) for o in tile["overlays"]]
        decorate_box_tile(tile, cfg, _footprint(tid, lo=1.6, hi=2.4))
        for o, (old_lo, old_hi) in zip(tile["overlays"], before):
            lo, hi = span(o)
            assert lo <= old_lo + 1e-9 and hi >= old_hi - 1e-9, f"box shrank on tile {tid}"


def test_boxes_are_left_alone_without_a_footprint():
    """Widening needs the envelope; without it only the variant may be assigned.

    Silently widening to the tile instead would be the one edit that CANNOT be checked
    against where the feet go.
    """
    tile = _box_tile()
    before = copy.deepcopy(tile["overlays"])
    decorate_box_tile(tile, DecorateConfig(seed=0, box_widen_frac=1.0), None)
    for old, new in zip(before, tile["overlays"]):
        assert old["size_xyz"] == new["size_xyz"]
        assert old["pos_xy"] == new["pos_xy"]


def test_not_every_box_tile_is_widened():
    """The narrow-target case must survive, or the variety is just moved rather than added."""
    cfg = DecorateConfig(seed=1)
    modes = []
    for tid in range(60):
        tile = _box_tile(terrain_id=tid)
        modes.append(decorate_box_tile(tile, cfg, _footprint(tid, lo=1.6, hi=2.4)))
    widened = sum(1 for m in modes if m.endswith("widened"))
    assert 0 < widened < len(modes), (
        f"{widened} of {len(modes)} widened; both cases must occur"
    )


def test_a_wall_with_no_window_is_one_box():
    """The common case must not pay for the window machinery."""
    boxes = _wall_segments("y", 0.1, 0.2, 0.0, 10.0, 2.0, 0.0, None)
    assert len(boxes) == 1
    assert boxes[0]["size_xyz"] == [10.0, pytest.approx(0.1), 2.0]
    assert boxes[0]["walkable_top"] is False
    # _wall_segments builds geometry; add_walls decides whether it is decor or collidable.
    assert "decor" not in boxes[0]


def test_a_window_becomes_the_four_boxes_around_it():
    """Neither simulator does CSG, so a hole is the pieces AROUND it."""
    boxes = _wall_segments("y", 0.0, 0.1, 0.0, 10.0, 3.0, 0.0, (4.0, 6.0, 1.0, 2.0))
    assert len(boxes) == 4
    # nothing may occupy the hole
    for b in boxes:
        x0 = b["pos_xy"][0] - b["size_xyz"][0] / 2
        x1 = b["pos_xy"][0] + b["size_xyz"][0] / 2
        z0, z1 = b["base_z"], b["base_z"] + b["size_xyz"][2]
        overlaps_x = x0 < 6.0 - 1e-9 and x1 > 4.0 + 1e-9
        overlaps_z = z0 < 2.0 - 1e-9 and z1 > 1.0 + 1e-9
        assert not (overlaps_x and overlaps_z), f"piece {b} intrudes into the window"


def test_a_window_flush_with_an_edge_drops_the_degenerate_piece():
    """build_overlay raises on a zero extent, so an empty piece must never be emitted."""
    boxes = _wall_segments("y", 0.0, 0.1, 0.0, 10.0, 3.0, 0.0, (0.0, 2.0, 0.0, 1.0))
    assert all(b["size_xyz"][0] > 0 and b["size_xyz"][2] > 0 for b in boxes)
    assert len(boxes) == 2, "the before-piece and the under-piece are both degenerate here"


def test_walls_are_marked_decor_and_never_walkable():
    """The three properties all hang off these two fields."""
    tile = _tile(terrain_id=0, size_y=6.0)
    fp = _footprint(0, lo=2.5, hi=3.5)
    n_sides, n = add_walls(tile, fp, DecorateConfig(seed=0, wall_frac=1.0))
    assert n > 0 and n_sides > 0
    walls = [o for o in tile["overlays"] if o.get("decor")]
    assert len(walls) == n
    assert all(w["walkable_top"] is False for w in walls)


def test_every_wall_sits_ON_a_tile_edge():
    """The fix for walls appearing mid-tile.

    An earlier version placed each wall a fixed clearance outside the FOOT ENVELOPE, which
    is the tile edge only when the motion fills the tile. On a pure-stair tile with 3.8 m of
    unused width it put the wall through the middle instead. The rule is now the tile's own
    boundary, and the envelope is only a veto on WHETHER a side is walled.
    """
    cfg = DecorateConfig(seed=1, wall_frac=1.0)
    for tid in range(40):
        tile = _tile(terrain_id=tid, size_y=6.0)
        add_walls(tile, _footprint(tid, lo=2.5, hi=3.5), cfg)
        for w in [o for o in tile["overlays"] if o.get("decor")]:
            lo_y, hi_y = span(w)
            lo_x = w["pos_xy"][0] - w["size_xyz"][0] / 2.0
            hi_x = w["pos_xy"][0] + w["size_xyz"][0] / 2.0
            on_edge = (abs(lo_y) < 1e-6 or abs(hi_y - 6.0) < 1e-6
                       or abs(lo_x) < 1e-6 or abs(hi_x - 20.0) < 1e-6)
            assert on_edge, f"tile {tid}: wall at y[{lo_y},{hi_y}] x[{lo_x},{hi_x}] is not on an edge"
            assert lo_y >= -1e-9 and hi_y <= 6.0 + 1e-9, "wall outside the tile in y"
            assert lo_x >= -1e-9 and hi_x <= 20.0 + 1e-9, "wall outside the tile in x"


def test_a_tile_can_get_all_four_walls():
    """A full room must be reachable, not just a pair of side walls."""
    cfg = DecorateConfig(seed=0, wall_frac=1.0, wall_window_frac=0.0)
    tile = _tile(terrain_id=0, size_y=6.0)
    # motion well clear of every edge, so all four sides qualify
    roomy = TileFootprint(terrain_id=0, n_clips=1, y_min=2.5, y_max=3.5, x_min=5.0,
                          x_max=15.0, y_min_sized=2.2, y_max_sized=3.8)
    n_sides, _ = add_walls(tile, roomy, cfg)
    assert n_sides == 4, f"expected a full room, got {n_sides} walls"


def test_a_side_the_motion_reaches_is_not_walled():
    """What stops a wall across the mouth of a corridor.

    A transition tile is traversed along +x, so its x-extremes are where the robot enters and
    leaves; those sides must never be walled however high wall_frac is.
    """
    cfg = DecorateConfig(seed=0, wall_frac=1.0, wall_window_frac=0.0)
    tile = _tile(terrain_id=0, size_y=6.0)
    corridor = TileFootprint(terrain_id=0, n_clips=1, y_min=2.5, y_max=3.5, x_min=0.0,
                             x_max=20.0, y_min_sized=2.2, y_max_sized=3.8)
    add_walls(tile, corridor, cfg)
    for w in [o for o in tile["overlays"] if o.get("decor")]:
        assert w["size_xyz"][0] > 1.0, (
            "an x-edge wall was built across the corridor the motion runs along"
        )


def test_walls_can_be_made_collidable_on_request():
    """Off by default -- see DecorateConfig.wall_collidable for why -- but not hardwired."""
    tile = _tile(terrain_id=0, size_y=6.0)
    add_walls(tile, _footprint(0, lo=2.5, hi=3.5),
              DecorateConfig(seed=0, wall_frac=1.0, wall_collidable=True))
    walls = [o for o in tile["overlays"] if o.get("walkable_top") is False]
    assert walls, "no walls were built"
    assert all(w.get("decor") is False for w in walls)


def test_a_tile_with_no_clear_band_gets_no_wall():
    """A tile whose motion fills it must not be given a wall on top of the feet."""
    tile = _tile(terrain_id=0, size_y=4.0)
    tight = TileFootprint(terrain_id=0, n_clips=1, y_min=0.1, y_max=3.9, x_min=0.0,
                          x_max=20.0, y_min_sized=0.0, y_max_sized=4.0)
    assert add_walls(tile, tight, DecorateConfig(seed=0, wall_frac=1.0)) == (0, 0)


def test_walls_have_their_own_random_stream():
    """A tile's walls must not be correlated with the width arm it drew."""
    cfg = DecorateConfig(seed=0)
    arm = [tile_rng(cfg.seed, t).random() for t in range(10)]
    wall = [tile_rng(cfg.seed ^ 0x2A11, t).random() for t in range(10)]
    box = [tile_rng(cfg.seed ^ 0x5EED, t).random() for t in range(10)]
    assert len({tuple(arm), tuple(wall), tuple(box)}) == 3


def test_flat_rooms_are_authored_about_the_TILE_CENTRE():
    """Flat tiles have no manifest entry and no fixed size.

    The loader synthesises them at ``motion travel + 2 * flat_buffer``, and flat_buffer is a
    RUNTIME argument -- 2.0 in assemble_clip_terrain, 20.0 in ClipTerrainImporterCfg, which is
    what training uses. Absolute coordinates resolved offline would land somewhere else the
    moment it changed, so a room is authored around zero and the loader adds the centre.
    """
    rooms = make_flat_rooms(12, (18.9, 16.8), DecorateConfig(seed=0, wall_frac=1.0,
                                                             flat_room_frac=1.0))
    for room in rooms:
        xs = [o["pos_xy"][0] for o in room]
        ys = [o["pos_xy"][1] for o in room]
        assert min(xs) < 0 < max(xs), "a room should straddle x=0, not start at it"
        assert min(ys) < 0 < max(ys), "a room should straddle y=0, not start at it"


def test_a_flat_room_stands_off_the_motion_it_surrounds():
    """A wall inside the travel would be a wall the robot walks through."""
    travel = (18.9, 16.8)
    cfg = DecorateConfig(seed=2, wall_frac=1.0, flat_room_frac=1.0,
                         flat_room_standoff_range=(3.0, 8.0))
    for room in make_flat_rooms(12, travel, cfg):
        for o in room:
            far_x = abs(o["pos_xy"][0]) + o["size_xyz"][0] / 2.0
            far_y = abs(o["pos_xy"][1]) + o["size_xyz"][1] / 2.0
            assert far_x >= travel[0] / 2.0 or far_y >= travel[1] / 2.0, (
                f"wall {o} sits entirely inside the {travel} travel box"
            )


def test_some_flat_variants_are_open_ground():
    """Not every flat tile should be a room, or the FM never sees unbounded flat."""
    rooms = make_flat_rooms(24, (18.9, 16.8), DecorateConfig(seed=1, flat_room_frac=0.6))
    assert any(not r for r in rooms) and any(r for r in rooms)


def test_flat_rooms_are_decor_by_default():
    """Same channel as the tile walls; the flag flips both together."""
    rooms = make_flat_rooms(6, (18.9, 16.8),
                            DecorateConfig(seed=0, wall_frac=1.0, flat_room_frac=1.0))
    boxes = [o for r in rooms for o in r]
    assert boxes and all(o["decor"] is True for o in boxes)
    solid = make_flat_rooms(6, (18.9, 16.8),
                            DecorateConfig(seed=0, wall_frac=1.0, flat_room_frac=1.0,
                                           wall_collidable=True))
    assert all(o["decor"] is False for r in solid for o in r)
