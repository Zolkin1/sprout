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
                                         _resize_y, classify, classify_overlays,
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
