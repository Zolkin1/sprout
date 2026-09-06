# Copyright 2026 Zachary Olkin. All rights reserved.

"""The mjeval TWO-BOX course, as a free-form Isaac terrain section.

This is a deliberate departure from :mod:`~motion_gen.terrain.freeform`, whose families are
RANDOMISED so a fine-tune sees a distribution.  Here the geometry is FIXED and is a clone of one
specific MuJoCo eval course -- ``mjeval_box2_h040``, two 0.40 m boxes 3 m apart -- because the
question being asked is not "does the generator handle boxes" but "can it be taught to pick the
right MODE on THIS obstacle".  Overfitting to it is the point, so the terrain must not vary.

**Why this course.**  Measured over 20 flow-noise seeds x 10 speeds in MuJoCo, the deployed pair
scores **0.060 unsteered / 0.145 steered** on it -- and 0/10 at every speed on the deterministic
``x0 = 0`` draw the hardware runs.  It is the worst course in the mjeval catalogue and the only
one that fails at every speed, while the SINGLE 0.40 m box (identical geometry, one box) scores
0.70.  So the second box is what breaks it, the failure is a mode choice rather than a capability
limit, and the generator visibly proposes different modes there (``view_cell.py --plans``).

**The geometry is MIRRORED, not re-derived.**  Every constant below is mjeval's, restated here
because ``source/motion_gen`` must not import ``transfer`` (it is the offline generation package
and stays free of the deploy stack, exactly as it stays free of IsaacLab).  The two copies are
held together by a parity test rather than by an import --
``transfer/sim/tests/test_box2_matches_mjeval.py`` imports BOTH and requires them to agree, which
is the same device ``freeform.OVERLAY_KINDS`` uses against the RL loader's own table.  A silent
divergence here would mean the Isaac fine-tune and the MuJoCo evaluation were run on two
different obstacles while every artifact said otherwise.

**What varies across tiles is the SPAWN DISTANCE and nothing else.**  One tile per approach
distance, identical boxes, so an initial-condition grid over distance is a grid over TILES and
needs no per-episode override: the reset already reads the spawn off the manifest's route block
(``mdp.reset_on_freeform_section``).  The other IC axis, commanded speed, is a COMMAND and is
pinned per recording run by ``freeform_speed_min == freeform_speed_max``.
"""

from __future__ import annotations

from motion_gen.terrain.freeform import (
    GOAL_INSET_M,
    SECTION_MARGIN_M,
    SPAWN_INSET_M,
    _base_tile,
    _fit_goal_half_width,
    _ground_overlay,
    _overlay,
    _route,
)

#: Family tag for these tiles.  Distinct from :data:`freeform.FAMILIES` on purpose: a consumer
#: that splits by family must be able to tell a fixed course from a randomised section, and the
#: eval's family tables key off exactly this string.
BOX2_FAMILY: str = "box2"

# --------------------------------------------------------------------------------------------
# mjeval's geometry.  Mirrored from ``transfer/sim/mjeval/terrains.py`` and
# ``transfer/sim/generate_stairs_scene.py``; pinned by test_box2_matches_mjeval.py.
# --------------------------------------------------------------------------------------------

#: Box top height (m).  ``EVAL_BOX_HEIGHTS[1]``, and the height the two-box course is built at.
BOX_HEIGHT_M: float = 0.40

#: Extent along travel (m), ``EVAL_BOX_DEPTH``.  **Deliberately out of distribution**: every
#: trained box is 0.46-0.48 m deep, under one stride, so a robot never stands on top and the
#: jump up and the jump down collapse into one ballistic event instead of the two this course
#: exists to separate.  Carried here so the deviation lives in the manifest, not in a comment.
BOX_DEPTH_M: float = 1.20

#: Lateral extent (m), ``generate_stairs_scene.TRAINED_BOX_WIDTH`` -- the jump_on median.
BOX_WIDTH_M: float = 2.59

#: Flat ground between the two boxes (m).  The whole reason this course is hard: it is long
#: enough to demand a landing, a re-approach and a second commitment.
BOX_GAP_M: float = 3.00

#: How far past the last box face mjeval puts the goal CENTRE (m), ``TARGET_LEAD``.
TARGET_LEAD_M: float = 1.50

#: Depth of mjeval's goal RECTANGLE along travel (m), ``GOAL_DEPTH``.
GOAL_DEPTH_M: float = 2.00

#: Half-extent of mjeval's goal band (m).  ``_with_goal_width`` resolves this course to
#: ``max(GOAL_HALF_WIDTH_FLOOR=2.5, BOX_WIDTH_M / 2 + GOAL_WIDTH_MARGIN=1.0)`` = 2.5.
GOAL_HALF_WIDTH_M: float = 2.50

# --------------------------------------------------------------------------------------------
# The initial-condition grid.
# --------------------------------------------------------------------------------------------

#: Spawn distances (m) from the pelvis to the FIRST box face, one tile each.  Brackets mjeval's
#: own 2.5 m approach on both sides.
TRAIN_SPAWN_DISTANCES_M: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0)

#: The HELD-OUT approach distance (m).  Interior to the training range and off its grid, so the
#: held-out cell tests interpolation over the IC rather than extrapolation past it -- the latter
#: would confound "the critic did not generalise" with "the state was never seen".
HOLDOUT_SPAWN_DISTANCE_M: float = 2.25

#: Commanded forward speeds (m/s), one recording run each.  The first four are the rungs of
#: mjeval's own box ladder where the 20-seed sweep found ANY successes; the last two scored
#: 0.000 at both steering settings and are included deliberately, as the saturated-failure
#: anchor a critic needs in order to have a boundary to place.
TRAIN_SPEEDS_MPS: tuple[float, ...] = (0.50, 0.83, 1.17, 1.50, 1.83, 2.17)

#: The HELD-OUT commanded speed (m/s).  Between rungs 2 and 3 of the training ladder, where the
#: MuJoCo sweep puts success in the interesting 0.25-0.75 band rather than at either saturation.
HOLDOUT_SPEED_MPS: float = 1.00


def box_faces(spawn_distance_m: float) -> tuple[tuple[float, float], ...]:
    """The two boxes' near and far faces, tile-local, for a given approach distance.

    Args:
        spawn_distance_m: Pelvis-to-first-face distance at spawn (m).

    Returns:
        ``((x0, x1), (x0, x1))`` for box 0 and box 1.
    """
    off_x = SECTION_MARGIN_M + SPAWN_INSET_M
    a0 = off_x + float(spawn_distance_m)
    a1 = a0 + BOX_DEPTH_M
    b0 = a1 + BOX_GAP_M
    return ((a0, a1), (b0, b0 + BOX_DEPTH_M))


def make_box2_tile(terrain_id: int, spawn_distance_m: float) -> dict:
    """One two-box section, spawning ``spawn_distance_m`` before the first box.

    **The goal is mjeval's, mapped rather than re-invented.**  mjeval scores arrival inside a
    RECTANGLE -- centre ``far + TARGET_LEAD``, depth :data:`GOAL_DEPTH_M`, half-width
    :data:`GOAL_HALF_WIDTH_M` -- while ``mdp.freeform_goal_reached`` tests a downrange LINE plus
    a lateral band (``x >= goal_x`` and ``|y - goal_y| <= half_width``).  The line is therefore
    put at the rectangle's NEAR edge, so the two agree on the boundary that decides the outcome.
    They differ only on a robot that runs past the rectangle's far edge, which mjeval scores a
    failure and this scores a success -- and here that case is not reachable, because arrival
    ends the episode the moment the line is crossed.

    Args:
        terrain_id: The tile id.
        spawn_distance_m: Pelvis-to-first-face distance at spawn (m).

    Returns:
        The manifest entry, carrying its ``route`` block and a ``box2`` record of the geometry.

    Raises:
        ValueError: If the tile's ground cannot hold the goal band (via
            :func:`~motion_gen.terrain.freeform._fit_goal_half_width`).
    """
    (a0, a1), (b0, b1) = box_faces(spawn_distance_m)
    off_x = SECTION_MARGIN_M + SPAWN_INSET_M

    # The y half-extent is set by the GOAL BAND, not by the boxes: at 2.5 m it is wider than the
    # 2.59 m box (1.295 m a side), and `_fit_goal_half_width` refuses a band the ground cannot
    # hold rather than letting the RL side clamp it back out over void.
    half_y = max(GOAL_HALF_WIDTH_M, BOX_WIDTH_M / 2.0)
    off_y = SECTION_MARGIN_M + half_y
    size_y = 2.0 * half_y + 2.0 * SECTION_MARGIN_M

    goal_x = b1 + TARGET_LEAD_M - GOAL_DEPTH_M / 2.0
    size_x = goal_x + GOAL_INSET_M + SECTION_MARGIN_M

    overlays = [_ground_overlay(size_x, size_y)]
    for x0, x1 in ((a0, a1), (b0, b1)):
        overlays.append(_overlay((x0 + x1) / 2.0, off_y, 0.0,
                                 x1 - x0, BOX_WIDTH_M, BOX_HEIGHT_M))

    route = _route((off_x, off_y), 0.0, 0.0, (goal_x, off_y),
                   _fit_goal_half_width(GOAL_HALF_WIDTH_M, overlays, (goal_x, off_y)), 0.0)
    tile = _base_tile(terrain_id, BOX2_FAMILY, size_x, size_y, overlays, route)
    tile["box2"] = {
        "spawn_distance_m": float(spawn_distance_m),
        "height_m": BOX_HEIGHT_M, "depth_m": BOX_DEPTH_M, "width_m": BOX_WIDTH_M,
        "gap_m": BOX_GAP_M,
        "box_faces_x": [[a0, a1], [b0, b1]],
        "goal_rect_centre_x": b1 + TARGET_LEAD_M,
        "goal_depth_m": GOAL_DEPTH_M,
        "mjeval_course": "mjeval_box2_h040",
    }
    return tile


def generate_box2_manifest(spawn_distances_m: tuple[float, ...]) -> list[dict]:
    """One tile per approach distance, in the order given.

    Args:
        spawn_distances_m: Pelvis-to-first-face distances (m).

    Returns:
        The manifest tile list, ``terrain_id`` running 0..n-1 in the order given -- so the
        recording's ``terrain_id`` column IS the index into this tuple, and the IC a row was
        collected at is recoverable from the table alone.

    Raises:
        ValueError: On an empty or non-positive distance set.
    """
    if not spawn_distances_m:
        raise ValueError("no spawn distances given; a box2 library needs at least one tile.")
    bad = [d for d in spawn_distances_m if not d > 0.0]
    if bad:
        raise ValueError(f"spawn distances must be positive, got {bad}. The distance is measured "
                         "from the pelvis to the first box FACE, so 0 spawns inside the box.")
    return [make_box2_tile(i, d) for i, d in enumerate(spawn_distances_m)]
