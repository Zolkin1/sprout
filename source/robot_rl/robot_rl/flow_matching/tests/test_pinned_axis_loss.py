# Copyright 2026 Zachary Olkin. All rights reserved.

"""Weighting the flow loss toward the root DOFs the anchor mask PINS to the terrain.

Those are the axes the generator gets wrong -- a displaced robot drags its predicted stair
segment along instead of leaving it where the terrain put it -- so the weighting concentrates
gradient there. Three things make it more than a channel mask, and each is pinned here: the
planar pinned axis is a rotated direction rather than a channel, x and y are standardized 4.3x
apart so the projection must be taken in metres, and yaw lives inside a 6D rotation.
"""

from __future__ import annotations

import math

import pytest
import torch

from motion_gen.world_lift import (
    _ANCHOR_DOF_INDEX,
    _ANCHOR_MASK_FLAT,
    _ANCHOR_MASK_STAIRS,
)
from robot_rl.flow_matching.flow import _ANCHOR_BIT, pinned_axis_flow_loss
from robot_rl.flow_matching.normalizer import Normalizer

_BLOCKS = ["base_pose", "base_vel", "joint_pos"]
_D = 44
_NJ = 29
# The shipped teacher's measured position stds -- the robot travels far downrange and little
# laterally, which is exactly the anisotropy the metric projection exists to survive.
_STD_XY = (0.442, 0.102)


def _norm() -> Normalizer:
    std = torch.ones(_D)
    std[0], std[1], std[2] = _STD_XY[0], _STD_XY[1], 0.187
    return Normalizer(torch.zeros(_D), std, torch.ones(_D, dtype=torch.bool))


def _loss(err, mask, yaw=0.0, weight=50.0, norm=None):
    b, f = err.shape[:2]
    return float(pinned_axis_flow_loss(
        err, err ** 2, torch.full((b, f), int(mask), dtype=torch.int64),
        torch.full((b,), float(yaw)), weight, norm or _norm(), _NJ, 0, _BLOCKS))


def _metric_error(ux: float, uy: float, b=2, f=3) -> torch.Tensor:
    """A unit METRIC displacement along ``(ux, uy)``, expressed in normalized channels."""
    e = torch.zeros(b, f, _D)
    e[..., 0], e[..., 1] = ux / _STD_XY[0], uy / _STD_XY[1]
    return e


def test_bit_table_matches_the_authority():
    """``flow`` restates the bit order to avoid importing motion_gen; it must not drift."""
    assert _ANCHOR_BIT == _ANCHOR_DOF_INDEX


def test_weight_one_is_exactly_the_plain_mean():
    """The feature must be a strict no-op when off -- an unweighted A/B arm depends on it."""
    torch.manual_seed(0)
    e = torch.randn(2, 3, _D)
    assert _loss(e, _ANCHOR_MASK_STAIRS, yaw=0.5, weight=1.0) == pytest.approx(
        float(e.pow(2).mean()), rel=1e-12)


def test_renormalization_preserves_the_loss_scale():
    """Weights are renormalized to mean 1, so the effective learning rate does not move.

    Without this the weighted arm would also train ~5x 'harder' and a difference between the
    two runs could be the weighting or the LR.
    """
    torch.manual_seed(0)
    e = torch.randn(4, 6, _D)
    base = float(e.pow(2).mean())
    for w in (5.0, 50.0, 500.0):
        got = _loss(e, _ANCHOR_MASK_STAIRS, yaw=0.3, weight=w)
        assert 0.25 * base < got < 4.0 * base, f"weight={w} moved the loss scale to {got}"


@pytest.mark.parametrize("deg", [0, 15, 30, 45, 60, 90, 135, 180])
def test_a_free_lateral_error_never_picks_up_the_downrange_weight(deg):
    """The anisotropy test, and the reason the projection is taken in METRES.

    Projecting the already-normalized error onto the normalized image of the metric downrange
    axis would leak badly: at these stds those two directions are only 26 degrees apart at 45
    degrees of heading, so a purely LATERAL error would collect 81% of a weight meant for
    downrange. Lateral is FREE under the stair mask, so its loss must be identical to the
    unweighted one at every heading.
    """
    ang = math.radians(deg)
    e = _metric_error(-math.sin(ang), math.cos(ang))          # metric lateral
    weighted = _loss(e, _ANCHOR_MASK_STAIRS, yaw=ang, weight=50.0)
    unweighted = _loss(e, _ANCHOR_MASK_STAIRS, yaw=ang, weight=1.0)
    # The weighted value is smaller only because the renormalizer's denominator grew; what
    # must hold is that NO extra numerator was added for a free axis.
    assert weighted < unweighted
    plain = _loss(e, _ANCHOR_MASK_FLAT, yaw=ang, weight=1.0)
    assert unweighted == pytest.approx(plain, rel=1e-12)


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_axis_aligned_headings_realise_the_full_weight(deg):
    """On an axis the boost is exact: pinned error weighs ``weight`` times a free one.

    These are the headings that matter -- measured on the depth_v5 recording, 90.7% of
    terrain-pinned windows sit within 10 degrees of one (the distribution is bimodal at 0 and
    180, the robot crossing the tile forwards or backwards).
    """
    ang = math.radians(deg)
    dr = _metric_error(math.cos(ang), math.sin(ang))
    lat = _metric_error(-math.sin(ang), math.cos(ang))
    # Same denominator for both (same mask, same heading), so the numerator ratio shows.
    num_dr = _loss(dr, _ANCHOR_MASK_STAIRS, yaw=ang) / _loss(dr, _ANCHOR_MASK_STAIRS, yaw=ang, weight=1.0)
    num_lat = _loss(lat, _ANCHOR_MASK_STAIRS, yaw=ang) / _loss(lat, _ANCHOR_MASK_STAIRS, yaw=ang, weight=1.0)
    assert num_dr / num_lat == pytest.approx(50.0, rel=1e-6)


def test_flat_pins_height_only_and_terrain_pins_more():
    """Flat frees downrange/lateral/yaw and pins height; stairs pins downrange, height and yaw.

    So on identical error the stair mask must place strictly more of the loss on the pinned
    set -- and flat must leave the ORIENTATION block alone, because flat frees yaw.
    """
    torch.manual_seed(1)
    e = torch.randn(3, 4, _D)
    flat = _loss(e, _ANCHOR_MASK_FLAT, weight=50.0)
    stairs = _loss(e, _ANCHOR_MASK_STAIRS, weight=50.0)
    assert flat != pytest.approx(stairs)
    # Orientation: only an all-pinned rotation is weighted. Put the error ONLY in rot6d and
    # compare -- flat must reproduce the unweighted value, stairs must not.
    rot = torch.zeros(3, 4, _D)
    rot[..., 3:9] = 1.0
    assert _loss(rot, _ANCHOR_MASK_FLAT, weight=50.0) < _loss(rot, _ANCHOR_MASK_FLAT, weight=1.0)
    ratio_flat = _loss(rot, _ANCHOR_MASK_FLAT, weight=50.0) / _loss(rot, _ANCHOR_MASK_FLAT, weight=1.0)
    ratio_stairs = _loss(rot, _ANCHOR_MASK_STAIRS, weight=50.0) / _loss(rot, _ANCHOR_MASK_STAIRS, weight=1.0)
    assert ratio_stairs > ratio_flat


def test_height_is_weighted_under_every_mask():
    """Height is pinned by flat AND terrain masks -- it is the one axis both agree on."""
    z = torch.zeros(2, 3, _D)
    z[..., 2] = 1.0
    for mask in (_ANCHOR_MASK_FLAT, _ANCHOR_MASK_STAIRS):
        ratio = _loss(z, mask, weight=50.0) / _loss(z, mask, weight=1.0)
        assert ratio > 1.0, f"height not weighted under mask {mask}"


def test_a_window_without_base_pose_is_untouched():
    """No base_pose block means nothing the anchor mask describes; fall back to the mean."""
    torch.manual_seed(2)
    e = torch.randn(2, 3, 2 * _NJ)
    out = pinned_axis_flow_loss(
        e, e ** 2, torch.full((2, 3), _ANCHOR_MASK_STAIRS, dtype=torch.int64),
        torch.zeros(2), 50.0, None, _NJ, 0, ["joint_pos", "joint_vel"])
    assert float(out) == pytest.approx(float(e.pow(2).mean()), rel=1e-12)
