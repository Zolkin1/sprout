# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Isaac two-box tile and the MuJoCo two-box course must be the SAME obstacle.

``motion_gen.terrain.box2`` restates mjeval's geometry rather than importing it, because
``source/motion_gen`` is the offline generation package and stays free of ``transfer`` exactly
as it stays free of IsaacLab.  This is the file that makes the restatement safe: it imports
BOTH and requires them to agree, so a constant edited on one side fails here instead of
silently producing an Isaac fine-tune and a MuJoCo evaluation run on two different obstacles.

The comparison is made in each frame's OWN terms -- distances from the spawn -- because the two
place their origins differently: mjeval's course has the spawn at ``x = 0`` and the first box
face at ``X_START``, while a free-form tile is corner-anchored and insets its spawn by
``SECTION_MARGIN_M + SPAWN_INSET_M``.  Comparing absolute x would only be comparing that inset.

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

box2 = pytest.importorskip("motion_gen.terrain.box2")
freeform = pytest.importorskip("motion_gen.terrain.freeform")
mj_terrains = pytest.importorskip("terrains")
gss = pytest.importorskip("generate_stairs_scene")

#: The catalogue index of ``mjeval_box2_h040``.  Resolved by NAME below rather than trusted.
_COURSE_NAME = "mjeval_box2_h040"


@pytest.fixture(scope="module")
def course():
    """The mjeval two-box course, found by name."""
    hits = [c for c in mj_terrains.catalogue() if c.name == _COURSE_NAME]
    assert len(hits) == 1, f"expected exactly one {_COURSE_NAME}, got {len(hits)}"
    return hits[0]


def test_the_scalar_geometry_constants_agree(course):
    """Height, depth, width and gap are the same numbers on both sides."""
    p = course.params
    assert box2.BOX_HEIGHT_M == pytest.approx(p["height_m"])
    assert box2.BOX_DEPTH_M == pytest.approx(p["depth_m"])
    assert box2.BOX_WIDTH_M == pytest.approx(p["width_m"])
    assert box2.BOX_GAP_M == pytest.approx(p["gap_m"])
    assert p["n_boxes"] == 2, "this course must stay a TWO-box course"
    # The width is mjeval's via generate_stairs_scene, so pin that hop too rather than only the
    # value it happens to have.
    assert box2.BOX_WIDTH_M == pytest.approx(gss.TRAINED_BOX_WIDTH)


def test_the_goal_constants_agree(course):
    """The goal lead, depth and half-width are mjeval's own resolved values."""
    assert box2.TARGET_LEAD_M == pytest.approx(mj_terrains.TARGET_LEAD)
    assert box2.GOAL_DEPTH_M == pytest.approx(mj_terrains.GOAL_DEPTH)
    # `_with_goal_width` resolves the half-width from the course's widest feature; taking it off
    # the built course rather than recomputing the rule is what keeps this honest.
    assert box2.GOAL_HALF_WIDTH_M == pytest.approx(course.goal_half_width_m)


def test_the_boxes_sit_at_the_same_distances_from_the_spawn(course):
    """Both faces of both boxes, measured from the spawn, match mjeval's.

    mjeval's spawn is the course origin and its first face is at ``X_START``, so the tile built
    at ``spawn_distance_m = X_START`` is the one that should line up face for face.
    """
    d = mj_terrains.X_START
    faces = box2.box_faces(d)
    off_x = freeform.SECTION_MARGIN_M + freeform.SPAWN_INSET_M
    rel = [[x - off_x for x in pair] for pair in faces]

    want0 = [d, d + box2.BOX_DEPTH_M]
    want1 = [want0[1] + box2.BOX_GAP_M, want0[1] + box2.BOX_GAP_M + box2.BOX_DEPTH_M]
    assert rel[0] == pytest.approx(want0)
    assert rel[1] == pytest.approx(want1)
    # And against mjeval's own numbers, not just against box2's own arithmetic.
    assert rel[0][0] == pytest.approx(mj_terrains.X_START)
    assert rel[1][1] == pytest.approx(course.course_length_m - mj_terrains.TARGET_LEAD)


def test_the_goal_line_is_the_near_edge_of_mjevals_rectangle(course):
    """The tile's goal LINE sits where mjeval's goal RECTANGLE begins.

    The two terminations differ in shape -- mjeval tests containment in a rectangle, the RL side
    tests a downrange line plus a lateral band -- so the mapping that makes them agree on the
    boundary that decides an outcome is "line at the rectangle's near edge".  This pins it.
    """
    d = mj_terrains.X_START
    tile = box2.make_box2_tile(0, d)
    off_x = freeform.SECTION_MARGIN_M + freeform.SPAWN_INSET_M
    goal_rel = tile["route"]["goal_x"] - off_x
    near_edge = course.target_xy[0] - mj_terrains.GOAL_DEPTH / 2.0
    assert goal_rel == pytest.approx(near_edge)
    assert tile["route"]["goal_half_width"] == pytest.approx(course.goal_half_width_m)


def test_every_tile_carries_the_same_boxes_and_differs_only_in_approach():
    """Across the IC grid the obstacle is invariant; only the spawn distance moves.

    The whole design rests on this -- an IC grid over distance must not be a grid over terrain
    as well, or the critic is being asked to generalise over two things at once.
    """
    tiles = box2.generate_box2_manifest(
        box2.TRAIN_SPAWN_DISTANCES_M + (box2.HOLDOUT_SPAWN_DISTANCE_M,))
    spans = set()
    for t in tiles:
        boxes = [o for o in t["overlays"] if o["size_xyz"][2] > 0.0
                 and o["size_xyz"][2] == pytest.approx(box2.BOX_HEIGHT_M)]
        assert len(boxes) == 2, f"tile {t['terrain_id']} has {len(boxes)} boxes, expected 2"
        spans.add(tuple(round(v, 9) for b in boxes for v in b["size_xyz"]))
        d = t["box2"]["spawn_distance_m"]
        assert t["box2"]["box_faces_x"] == [list(p) for p in box2.box_faces(d)]
    assert len(spans) == 1, f"box dimensions differ across tiles: {spans}"


def test_the_ic_grid_holds_its_holdout_out():
    """The held-out cell is off the training grid on BOTH axes."""
    assert box2.HOLDOUT_SPAWN_DISTANCE_M not in box2.TRAIN_SPAWN_DISTANCES_M
    assert box2.HOLDOUT_SPEED_MPS not in box2.TRAIN_SPEEDS_MPS
    # Interior, not extrapolation -- see the constant's docstring for why that is the choice.
    assert min(box2.TRAIN_SPAWN_DISTANCES_M) < box2.HOLDOUT_SPAWN_DISTANCE_M
    assert box2.HOLDOUT_SPAWN_DISTANCE_M < max(box2.TRAIN_SPAWN_DISTANCES_M)
    assert min(box2.TRAIN_SPEEDS_MPS) < box2.HOLDOUT_SPEED_MPS < max(box2.TRAIN_SPEEDS_MPS)


# --- the v2 variant ------------------------------------------------------------------
# v2 exists to lengthen the RUN-OUT, not to change the obstacle. These pin both halves of
# that claim, and pin that v1 is untouched -- the fingerprint hashes the whole tile dict, so
# a field added for v2's sake moved v1's hash once already (aa50da093337 -> 826a2af3918a) on
# geometry that had not changed at all.


def test_v2_moves_only_the_goal_and_grows_the_tile_to_match():
    """Boxes, spawn and lateral extent identical; goal and tile length both +3 m."""
    for dist in box2.TRAIN_SPAWN_DISTANCES_M:
        v1 = box2.make_box2_tile(0, dist)
        v2 = box2.make_box2_tile(0, dist, target_lead_m=box2.TARGET_LEAD_V2_M,
                                 family=box2.BOX2_V2_FAMILY)
        assert v1["box2"]["box_faces_x"] == v2["box2"]["box_faces_x"], dist
        assert v1["route"]["start_x"] == v2["route"]["start_x"], dist
        assert v1["size_y"] == v2["size_y"], dist
        assert v2["route"]["goal_x"] - v1["route"]["goal_x"] == pytest.approx(
            box2.GOAL_SHIFT_V2_M), dist
        assert v2["size_x"] - v1["size_x"] == pytest.approx(box2.GOAL_SHIFT_V2_M), dist


def test_v2_carries_its_own_family_so_nothing_pools_the_two():
    """A longer run-out is strictly more chances to fall, so the rates are incomparable.

    `binarize` thresholds per family and every report groups by it, so a shared family would
    silently average a v1 and a v2 recording into one number.
    """
    v1 = box2.make_box2_tile(0, 3.0)
    v2 = box2.make_box2_tile(0, 3.0, target_lead_m=box2.TARGET_LEAD_V2_M,
                             family=box2.BOX2_V2_FAMILY)
    assert v1["family"] == box2.BOX2_FAMILY
    assert v2["family"] == box2.BOX2_V2_FAMILY
    assert v1["family"] != v2["family"]


def test_only_v1_claims_to_be_the_mjeval_course():
    """v2 is deliberately NOT that course; a recording must not read as an mjeval clone."""
    assert box2.make_box2_tile(0, 3.0)["box2"]["mjeval_course"] == "mjeval_box2_h040"
    v2 = box2.make_box2_tile(0, 3.0, target_lead_m=box2.TARGET_LEAD_V2_M,
                             family=box2.BOX2_V2_FAMILY)
    assert v2["box2"]["mjeval_course"] == ""


def test_the_default_tile_records_no_target_lead_field():
    """Adding it unconditionally moved v1's terrain fingerprint on unchanged geometry.

    `fingerprint_terrain_manifest` hashes the tile dict, so an extra key is a hash change --
    which would strand every v1 rollout already recorded, none of which a freshly built v1
    library could then reproduce. The default needs no field to say it is the default.
    """
    assert "target_lead_m" not in box2.make_box2_tile(0, 3.0)["box2"]
    v2 = box2.make_box2_tile(0, 3.0, target_lead_m=box2.TARGET_LEAD_V2_M,
                             family=box2.BOX2_V2_FAMILY)
    assert v2["box2"]["target_lead_m"] == pytest.approx(box2.TARGET_LEAD_V2_M)
