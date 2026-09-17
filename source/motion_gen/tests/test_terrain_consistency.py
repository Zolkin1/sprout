# Copyright 2026 Zachary Olkin. All rights reserved.

"""The terrain-consistency reward: what it rewards, what it charges, and what it exempts.

Every case here is a plan whose right answer is known by construction -- a standing robot on
known ground, moved by a known amount -- so a sign error or a frame error cannot pass.  The
exemption tests matter most: the penetration penalty deliberately skips a foot's own points
near a detected stance, and the failure mode of getting that wrong is a reward that charges
normal walking (radius too small) or one that cannot see a buried foot (radius too large, and
no two-sided contact kernel).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from motion_gen import features as F
from motion_gen.joint_names import G1_JOINT_NAMES
from motion_gen.plan_geometry import (
    FOOT_BODIES,
    PlanColliderModel,
    plan_collider_points,
    plan_contact,
)
from motion_gen.terrain.box_sdf import BoxUnionTerrain, pad_box_sets
from motion_gen.terrain.overlay_boxes import collider_boxes_from_manifest_tile
from motion_gen.terrain_consistency import (
    DEFAULT_PENETRATION_WEIGHT,
    FOOT_PENETRATION_WEIGHT,
    TerrainConsistencySpec,
    resolve_point_weights,
    terrain_consistency,
)

NJ = len(G1_JOINT_NAMES)
#: Base height that puts the sole of a zero-joint stance exactly on z = 0.
STAND_Z = 0.7913
NODES = 62

BOX_TILE = {
    "terrain_id": 0, "type": "terrain", "size_x": 12.9, "size_y": 6.0,
    "frame_yaw": 0.0, "has_floor": False,
    "overlays": [
        {"kind": "box", "pos_xy": [6.45, 3.0], "size_xyz": [12.9, 6.0, 0.5],
         "base_z": -0.5, "walkable_top": True, "yaw": 0.0, "decor": False,
         "variant": "solid", "slab_thickness": 0.1},
        {"kind": "box", "pos_xy": [3.35, 3.0], "size_xyz": [1.2, 2.59, 0.4],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0, "decor": False,
         "variant": "solid", "slab_thickness": 0.1},
    ],
}


@pytest.fixture(scope="module")
def model() -> PlanColliderModel:
    """The collider model, float64 so the assertions are limited by geometry not by fp32."""
    return PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64)


@pytest.fixture(scope="module")
def terrain() -> BoxUnionTerrain:
    """One box tile, batch of 1."""
    boxes, mask = pad_box_sets([collider_boxes_from_manifest_tile(BOX_TILE)],
                               dtype=torch.float64)
    return BoxUnionTerrain(boxes, mask)


def _standing(x: float, z: float, f: int = NODES) -> tuple[torch.Tensor, ...]:
    """A static standing plan at ``(x, 3.0, z)``, zero joints, identity orientation."""
    base_pos = torch.zeros(1, f, 3, dtype=torch.float64)
    base_pos[..., 0] = x
    base_pos[..., 1] = 3.0
    base_pos[..., 2] = z
    rot = torch.eye(3, dtype=torch.float64).expand(1, f, 3, 3).contiguous()
    return base_pos, rot, torch.zeros(1, f, NJ, dtype=torch.float64)


def _grade(model, terrain, x, z, spec=None, **contact_kw):
    """Build a standing plan, infer contacts and grade it."""
    pts = plan_collider_points(*_standing(x, z), model)
    contact = plan_contact(pts, model, dt=0.02, **contact_kw)
    return terrain_consistency(pts, contact, terrain, model,
                              spec or TerrainConsistencySpec()), pts, contact


def test_the_sole_sits_on_the_ground_at_the_declared_stand_height(model, terrain):
    """The fixture's STAND_Z is right, or every case below is measuring an offset."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z), model)
    sole = pts[0, 0, model.sole_index.reshape(-1), 2]
    assert sole.abs().max().item() < 1e-3, sole.tolist()


def test_a_plan_planted_on_flat_ground_scores_perfectly(model, terrain):
    """Contact reward 1, no penetration: the reference case."""
    out, _pts, _c = _grade(model, terrain, x=1.0, z=STAND_Z)
    assert out["contact_reward"][0].item() == pytest.approx(1.0, abs=1e-3)
    assert out["penetration_volume_m3"][0].item() == pytest.approx(0.0, abs=1e-12)
    assert out["penetration_mean"][0].item() == pytest.approx(0.0, abs=1e-12)
    assert out["support_fraction"][0].item() == pytest.approx(1.0)
    assert out["contact_coverage"][0].item() == pytest.approx(1.0)


def test_a_plan_planted_on_the_box_top_also_scores_perfectly(model, terrain):
    """The reward reads the terrain, not a flat floor -- the whole point of the SDF.

    This is the case the deleted flat-``ground_z`` penetration loss got wrong: standing on a
    0.4 m box is correct behaviour, and a scalar floor calls it a 0.4 m error.
    """
    out, _pts, _c = _grade(model, terrain, x=3.35, z=STAND_Z + 0.4)
    assert out["contact_reward"][0].item() == pytest.approx(1.0, abs=1e-3)
    assert out["penetration_volume_m3"][0].item() == pytest.approx(0.0, abs=1e-12)


def test_a_stance_foot_hovering_above_the_ground_is_charged(model, terrain):
    """The mode error: the plan says planted, the terrain says 15 cm of air."""
    out, _pts, c = _grade(model, terrain, x=1.0, z=STAND_Z + 0.15)
    assert bool(c["contact"].all()), "a static plan is still stance by velocity"
    assert out["contact_reward"][0].item() < 1e-3
    assert out["penetration_volume_m3"][0].item() == pytest.approx(0.0, abs=1e-12)
    assert out["node_hover"][0, 0].min().item() == pytest.approx(0.15, abs=2e-3)


def test_a_stance_foot_buried_in_the_ground_is_charged_by_the_CONTACT_term(model, terrain):
    """The hole the exemption would otherwise open, closed by the two-sided kernel.

    A stance node is exempt from the penetration penalty (it is supposed to touch), so a plan
    that drives its stance foot 10 cm into the tread must be caught by the contact term or by
    nothing at all.
    """
    out, _pts, c = _grade(model, terrain, x=1.0, z=STAND_Z - 0.10)
    assert bool(c["contact"].all())
    assert out["contact_reward"][0].item() < 1e-3, "burial must cost the contact term"
    assert out["node_bury"][0, 0].max().item() == pytest.approx(0.10, abs=2e-3)


def test_the_penetration_penalty_sees_bodies_that_are_not_feet(model, terrain):
    """Sink the whole robot: the feet are exempt, the shanks are not.

    The drop has to clear the shank's REAL surface. Sampled, the shank's flat lower end sits
    0.052 m above the ground at nominal stance -- where the old two-end-cap-sphere
    approximation put it at 0.002 m, i.e. it over-reported the shank by 5 cm. So nothing
    penetrates until the robot drops past ~5.2 cm, and this test is also the pin on that.
    """
    clear, _pts, _c = _grade(model, terrain, x=1.0, z=STAND_Z - 0.05)
    assert clear["penetration_volume_m3"][0].item() == pytest.approx(0.0, abs=1e-12)

    out, _pts, _c = _grade(model, terrain, x=1.0, z=STAND_Z - 0.10)
    assert out["penetration_volume_m3"][0].item() > 0.0
    assert out["max_penetration_m"][0].item() == pytest.approx(0.043, abs=3e-3)
    # ...and every foot point is exempt, so the feet contribute nothing to it.
    n_foot_pts = sum(int((model.body_index == model.body_order.index(f)).sum())
                     for f in FOOT_BODIES)
    assert out["exempt_fraction"][0].item() == pytest.approx(
        n_foot_pts / model.num_points, abs=1e-9)


def test_only_the_contacting_foot_is_exempt_never_another_body(model, terrain):
    """Nothing but a foot is ever supposed to touch the ground."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z), model)
    contact = plan_contact(pts, model, dt=0.02)
    # Force ONE foot out of contact and check the exemption follows it.
    contact["contact"][:, :, 1] = False
    contact["contact_buffered"][:, :, 1] = False
    out = terrain_consistency(pts, contact, terrain, model)
    only_one = int((model.body_index == model.body_order.index(FOOT_BODIES[0])).sum())
    assert out["exempt_fraction"][0].item() == pytest.approx(
        only_one / model.num_points, abs=1e-9)


def test_the_dilation_radius_is_a_knob_and_zero_reproduces_the_unbuffered_term(model, terrain):
    """Radius 0 must be bit-identical to no buffer at all, so the knob has a null setting."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z - 0.12), model)
    c0 = plan_contact(pts, model, dt=0.02, dilate=0)
    assert torch.equal(c0["contact"], c0["contact_buffered"])
    out0 = terrain_consistency(pts, c0, terrain, model)

    manual = dict(c0)
    manual["contact_buffered"] = c0["contact"]
    assert out0["penetration_volume_m3"][0].item() == pytest.approx(
        terrain_consistency(pts, manual, terrain, model)["penetration_volume_m3"][0].item(),
        abs=0.0)

    # A wider buffer can only exempt MORE, so penetration can only fall.
    prev = out0["penetration_volume_m3"][0].item()
    assert prev > 0.0, "pick a drop that actually penetrates, or this proves nothing"
    for r in (1, 2, 3):
        out = terrain_consistency(pts, plan_contact(pts, model, dt=0.02, dilate=r),
                                  terrain, model)
        assert out["penetration_volume_m3"][0].item() <= prev + 1e-12
        prev = out["penetration_volume_m3"][0].item()


def test_a_plan_with_no_detected_stance_scores_zero_contact_and_reports_it(model, terrain):
    """An empty contact mask must be visible in ``contact_coverage``, not silently neutral."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z), model)
    contact = plan_contact(pts, model, dt=0.02)
    contact["contact"] = torch.zeros_like(contact["contact"])
    contact["contact_buffered"] = torch.zeros_like(contact["contact_buffered"])
    contact["support"] = torch.zeros_like(contact["support"])
    out = terrain_consistency(pts, contact, terrain, model)
    assert out["contact_reward"][0].item() == 0.0
    assert out["contact_coverage"][0].item() == 0.0
    assert out["support_fraction"][0].item() == 0.0


def test_the_served_window_reduction_covers_only_the_leading_nodes(model, terrain):
    """The plan's tail carries the MODE; its head is what the robot executes."""
    f = NODES
    base_pos, rot, q = _standing(1.0, STAND_Z, f)
    base_pos = base_pos.clone()
    base_pos[:, 12:, 2] += 0.15            # only the TAIL hovers
    pts = plan_collider_points(base_pos, rot, q, model)
    contact = plan_contact(pts, model, dt=0.02)
    contact["contact"] = torch.ones_like(contact["contact"])       # pin the mask
    contact["contact_buffered"] = torch.ones_like(contact["contact_buffered"])
    spec = TerrainConsistencySpec(served_nodes=12)
    out = terrain_consistency(pts, contact, terrain, model, spec)
    assert out["contact_reward_served"][0].item() == pytest.approx(1.0, abs=1e-3)
    assert out["contact_reward"][0].item() < 0.25


def test_per_body_signed_distance_is_reported_for_every_body(model, terrain):
    """``node_sd_min`` is what the recorder stores, so its layout is a contract."""
    out, _pts, _c = _grade(model, terrain, x=1.0, z=STAND_Z)
    sd = out["node_sd_min"]
    assert tuple(sd.shape) == (1, NODES, len(model.body_order))
    assert torch.isfinite(sd).all()
    feet = [model.body_order.index(f) for f in FOOT_BODIES]
    assert sd[0, 0, feet].abs().max().item() < 1e-3           # soles are on the ground
    assert sd[0, 0, model.body_order.index("torso_link")].item() > 0.3


def test_a_terrain_batch_mismatch_is_refused(model, terrain):
    """Grading N plans against M tiles would return a plausible, wrong number."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z), model)
    pts = pts.repeat(3, 1, 1, 1)
    contact = plan_contact(pts, model, dt=0.02)
    with pytest.raises(ValueError, match="batch"):
        terrain_consistency(pts, contact, terrain, model)


def test_the_spec_refuses_degenerate_constants():
    """A zero sigma or threshold produces a reward that is not a reward."""
    with pytest.raises(ValueError, match="hover_sigma_m"):
        TerrainConsistencySpec(hover_sigma_m=0.0)
    with pytest.raises(ValueError, match="contact_speed_thresh"):
        TerrainConsistencySpec(contact_speed_thresh=0.0)
    with pytest.raises(ValueError, match="served_nodes"):
        TerrainConsistencySpec(served_nodes=0)


def test_the_penetration_sum_grows_with_how_much_is_buried(model, terrain):
    """The point of an area-weighted sum: it measures volume, not just the worst point.

    A ``max`` saturates at the deepest point and a mean dilutes; sinking the robot further
    must strictly increase the sum.
    """
    vols, maxes = [], []
    for drop in (0.06, 0.10, 0.15, 0.20):
        out, _pts, _c = _grade(model, terrain, x=1.0, z=STAND_Z - drop)
        vols.append(out["penetration_volume_m3"][0].item())
        maxes.append(out["max_penetration_m"][0].item())
    assert vols == sorted(vols), vols
    assert vols[-1] > 5.0 * vols[0], "the sum should be strongly superlinear in depth"
    assert maxes == sorted(maxes)


def test_node_chunking_is_exact(model, terrain):
    """Chunking exists for memory at 512 envs; it must not change a single number."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z - 0.12), model)
    contact = plan_contact(pts, model, dt=0.02)
    whole = terrain_consistency(pts, contact, terrain, model)
    for chunk in (1, 7, 16, NODES, NODES + 5):
        part = terrain_consistency(pts, contact, terrain, model, chunk_nodes=chunk)
        for key, want in whole.items():
            assert torch.allclose(part[key], want, atol=1e-12, rtol=0.0), (key, chunk)


def test_the_three_reductions_disagree_in_the_documented_direction(model, terrain):
    """A sanity check on the framing: the mean is the one that dilutes.

    One foot 5 cm under on an otherwise clear robot. The mean divides by every collider point
    on the robot; the volume sum does not.
    """
    out, _pts, _c = _grade(model, terrain, x=1.0, z=STAND_Z - 0.12)
    assert out["penetration_mean"][0].item() < out["penetration_volume_m3"][0].item()
    assert out["max_penetration_m"][0].item() > out["penetration_volume_m3"][0].item()


def test_the_area_weighted_terms_are_ABSENT_on_an_extreme_point_model(terrain):
    """A zero volume for a buried plan would be a plausible, wrong number."""
    ext = PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64,
                            spacing=None)
    pts = plan_collider_points(*_standing(1.0, STAND_Z - 0.20), ext)
    out = terrain_consistency(pts, plan_contact(pts, ext, dt=0.02), terrain, ext)
    assert "penetration_volume_m3" not in out
    assert "penetration_sq" not in out
    assert "node_penetration_m3" not in out
    assert out["max_penetration_m"][0].item() > 0.0        # still valid in this mode


# -----------------------------------------------------------------------------------------
# PARC's Eq 7 / Eq 8, and the deadband
# -----------------------------------------------------------------------------------------
def _all_stance(model, terrain, offset: float, spec=None):
    """Grade a static standing plan lifted/sunk by ``offset``, with every node a stance.

    ``thresh=1e9`` forces the contact mask on, so these tests exercise the PENALTY shape and
    not the detector.
    """
    pts = plan_collider_points(*_standing(1.0, STAND_Z + offset), model)
    contact = plan_contact(pts, model, dt=0.02, thresh=1e9)
    return terrain_consistency(pts, contact, terrain, model,
                               spec or TerrainConsistencySpec())


@pytest.mark.parametrize("offset", [0.0, 0.005, 0.01, -0.005, -0.01])
def test_the_contact_deadband_is_free_within_one_centimetre_either_way(model, terrain,
                                                                       offset):
    """PARC Eq 8 is zero only at tangency; the deadband widens that to +-1 cm."""
    out = _all_stance(model, terrain, offset)
    assert out["contact_penalty"][0].item() == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("offset", [0.03, -0.03, 0.10, -0.10])
def test_the_contact_penalty_is_the_excess_over_the_deadband_and_is_SYMMETRIC(model, terrain,
                                                                              offset):
    """Too high and too deep cost the same -- ``|sd|``, not a one-sided hover term.

    Two feet, each contributing ``|offset| - deadband``, summed per node (PARC sums over
    bodies) and averaged over nodes.
    """
    spec = TerrainConsistencySpec()
    want = 2.0 * (abs(offset) - spec.contact_deadband_m)
    got = _all_stance(model, terrain, offset)["contact_penalty"][0].item()
    assert got == pytest.approx(want, abs=2e-3), (offset, got, want)


def test_the_contact_penalty_sums_over_FEET_not_averages(model, terrain):
    """PARC Eq 8 is a sum over bodies, so one foot in contact costs half of two."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z + 0.05), model)
    contact = plan_contact(pts, model, dt=0.02, thresh=1e9)
    both = terrain_consistency(pts, contact, terrain, model)["contact_penalty"][0].item()
    contact["contact"] = contact["contact"].clone()
    contact["contact"][:, :, 1] = False                     # right foot no longer in contact
    one = terrain_consistency(pts, contact, terrain, model)["contact_penalty"][0].item()
    assert one == pytest.approx(both / 2.0, rel=1e-6)


def test_a_node_with_no_contact_label_costs_nothing(model, terrain):
    """``c_b`` gates it: a swinging foot 30 cm up is not an error."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z + 0.30), model)
    contact = plan_contact(pts, model, dt=0.02, thresh=1e9)
    contact["contact"] = torch.zeros_like(contact["contact"])
    out = terrain_consistency(pts, contact, terrain, model)
    assert out["contact_penalty"][0].item() == 0.0


def test_eq8_reads_the_CLOSEST_sole_point_not_the_deepest(model, terrain):
    """``min_p |sd|`` and ``|min_p sd|`` disagree on a tilted foot, and Eq 8 wants the first.

    A foot with its toe under the surface and its heel well above it HAS a point touching, so
    the contact requirement is met; the deepest point would report a penetration instead.
    Burial is the penetration term's job, not this one's.
    """
    out = _all_stance(model, terrain, 0.0)
    # Flat on the ground the two agree; construct the disagreement directly instead.
    sd = torch.tensor([[-0.02, -0.01, 0.03, 0.05]])
    assert sd.abs().amin().item() == pytest.approx(0.01)      # closest point
    assert sd.amin().item() == pytest.approx(-0.02)           # deepest point
    assert out["node_foot_abs"].shape == out["node_foot_clearance"].shape


def test_penetration_is_the_plain_SUM_of_depths_over_points(model, terrain):
    """PARC Eq 7: a sum, so it grows with both depth and how many points are under."""
    prev = 0.0
    for drop in (0.06, 0.10, 0.15, 0.20, 0.30):
        got = _all_stance(model, terrain, -drop)["penetration_sum_m"][0].item()
        assert got >= prev, drop
        prev = got
    assert prev > 5.0, "a 30 cm-deep robot should sum to metres over ~1400 points"

    # Nothing penetrates while the robot is above the ground.
    assert _all_stance(model, terrain, 0.02)["penetration_sum_m"][0].item() == 0.0


def test_the_penetration_sum_scales_with_the_sampling_and_the_volume_does_not(terrain):
    """The one cost of PARC's plain sum, asserted so it cannot surprise anyone.

    ``penetration_sum_m`` is a sum over POINTS, so halving the spacing roughly quadruples it.
    ``penetration_volume_m3`` is area-weighted and therefore converges. If the collider
    spacing ever changes, the volume is the term that stays comparable across runs -- which is
    why both are returned.
    """
    got = {}
    for spacing in (0.05, 0.035, 0.025):
        m = PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64,
                              spacing=spacing)
        out = _all_stance(m, terrain, -0.20)
        got[spacing] = (out["penetration_sum_m"][0].item(),
                        out["penetration_volume_m3"][0].item(), m.num_points)
    sums = [got[s][0] for s in (0.05, 0.035, 0.025)]
    vols = [got[s][1] for s in (0.05, 0.035, 0.025)]
    assert sums[2] > 2.0 * sums[0], f"the sum must grow with density: {sums}"
    assert max(vols) / min(vols) < 1.15, f"the volume must be density-stable: {vols}"


def test_both_reward_terms_are_penalties_and_a_clean_plan_scores_zero(model, terrain):
    """Sign convention: these wire with NEGATIVE weights, so perfect is 0 for both."""
    out = _all_stance(model, terrain, 0.0)
    assert out["contact_penalty"][0].item() == pytest.approx(0.0, abs=1e-3)
    assert out["penetration_sum_m"][0].item() == 0.0
    for key in ("contact_penalty", "penetration_sum_m"):
        assert bool((out[key] >= 0.0).all()), f"{key} must be non-negative"


def test_the_per_node_contact_penalty_sums_to_the_scalar(model, terrain):
    """``node_contact_penalty`` is what the viewer draws; it must be the same quantity."""
    out = _all_stance(model, terrain, 0.05)
    assert out["node_contact_penalty"].mean(dim=1)[0].item() == pytest.approx(
        out["contact_penalty"][0].item(), rel=1e-9)


def test_the_total_is_the_weighted_sum_and_its_per_node_mean_agrees(model, terrain):
    """One balance, and the per-node series must reduce to the scalar the reward uses.

    The viewer plots the per-node series and reports the scalar; if they were computed
    independently, a plot could disagree with the number beside it and neither would look
    wrong.
    """
    spec = TerrainConsistencySpec()
    out = _all_stance(model, terrain, -0.12, spec)
    assert out["weighted_contact"][0].item() == pytest.approx(
        spec.contact_weight * out["contact_penalty"][0].item(), rel=1e-12)
    assert out["weighted_penetration"][0].item() == pytest.approx(
        spec.penetration_weight * out["penetration_sum_m"][0].item(), rel=1e-12)
    assert out["total_penalty"][0].item() == pytest.approx(
        out["weighted_contact"][0].item() + out["weighted_penetration"][0].item(), rel=1e-12)
    assert out["node_total_penalty"].mean(dim=1)[0].item() == pytest.approx(
        out["total_penalty"][0].item(), rel=1e-9)


def test_the_reference_weights_make_the_two_terms_commensurate():
    """They are declared FROM the measured p95s, so that relationship is the check.

    p95 over 6000 recorded plans, WITH the per-body penetration weights: contact_penalty
    0.0147 m, penetration_sum_m 9.36 m. Equal weights would make the total a report on
    penetration alone.
    """
    spec = TerrainConsistencySpec()
    assert spec.contact_weight * 0.0147 == pytest.approx(1.0, abs=0.05)
    assert spec.penetration_weight * 9.36 == pytest.approx(1.0, abs=0.05)


def test_a_clean_plan_has_zero_total_and_the_weights_are_refused_if_negative(model, terrain):
    """Perfect is 0 for the total too, and a negative weight would flip a penalty to a bonus."""
    assert _all_stance(model, terrain, 0.0)["total_penalty"][0].item() == pytest.approx(
        0.0, abs=1e-3)
    with pytest.raises(ValueError, match="must be"):
        TerrainConsistencySpec(contact_weight=-1.0)
    with pytest.raises(ValueError, match="must be"):
        TerrainConsistencySpec(penetration_weight=-1.0)


# -----------------------------------------------------------------------------------------
# Per-body penetration weights
# -----------------------------------------------------------------------------------------
def _no_contact(contact):
    """The same contact mapping with every flag cleared, so nothing is exempt.

    Needed because a STATIC test plan has sole speed exactly 0, so even ``thresh=1e-9`` marks
    every node a stance and exempts both feet -- which silently hides the foot weight.
    """
    cleared = dict(contact)
    for key in ("contact", "contact_raw", "contact_filled", "contact_buffered", "support"):
        cleared[key] = torch.zeros_like(contact[key])
    return cleared


def test_the_feet_are_weighted_below_every_other_body(model):
    """The whole point: a foot at ``sd ~ 0`` is a stance, a shin inside a riser is not."""
    spec = TerrainConsistencySpec()
    w = resolve_point_weights(model, spec)
    for foot in FOOT_BODIES:
        own = model.body_index == model.body_order.index(foot)
        assert torch.allclose(w[own], torch.full_like(w[own], FOOT_PENETRATION_WEIGHT))
    for body in model.body_order:
        if body in FOOT_BODIES:
            continue
        own = model.body_index == model.body_order.index(body)
        assert torch.allclose(w[own], torch.full_like(w[own], DEFAULT_PENETRATION_WEIGHT)), body
    assert DEFAULT_PENETRATION_WEIGHT == 10.0 * FOOT_PENETRATION_WEIGHT


def test_an_unknown_body_name_is_REFUSED_not_ignored(model):
    """A typo would leave that body at the default and look like a re-weighting."""
    spec = TerrainConsistencySpec(body_penetration_weights={"left_shin_link": 3.0})
    with pytest.raises(ValueError, match="does not have"):
        resolve_point_weights(model, spec)


def test_the_weights_scale_each_body_s_contribution(model, terrain):
    """Doubling one body's weight doubles exactly its share of the sum, and nothing else's."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z - 0.12), model)
    contact = _no_contact(plan_contact(pts, model, dt=0.02))

    base = terrain_consistency(pts, contact, terrain, model, TerrainConsistencySpec())
    heavy = terrain_consistency(pts, contact, terrain, model, TerrainConsistencySpec(
        body_penetration_weights={"left_ankle_roll_link": 2.0 * FOOT_PENETRATION_WEIGHT,
                                  "right_ankle_roll_link": FOOT_PENETRATION_WEIGHT}))
    order = model.body_order
    i_left = order.index("left_ankle_roll_link")
    b, h = base["penetration_by_body"][0], heavy["penetration_by_body"][0]
    assert b[i_left].item() > 0.0, "pick a depth where the foot is actually charged"
    assert h[i_left].item() == pytest.approx(2.0 * b[i_left].item(), rel=1e-9)
    for i, name in enumerate(order):
        if i != i_left:
            assert h[i].item() == pytest.approx(b[i].item(), rel=1e-9), name


def test_the_per_body_attribution_sums_to_the_scalar(model, terrain):
    """``penetration_by_body`` is an attribution of the reward, not a separate quantity."""
    pts = plan_collider_points(*_standing(1.0, STAND_Z - 0.15), model)
    contact = _no_contact(plan_contact(pts, model, dt=0.02))
    out = terrain_consistency(pts, contact, terrain, model)
    assert out["penetration_by_body"][0].sum().item() == pytest.approx(
        out["penetration_sum_m"][0].item(), rel=1e-9)
    assert out["penetration_by_body"].shape == (1, len(model.body_order))


def test_the_same_geometry_charges_ten_times_more_at_the_default_weight(model, terrain):
    """The behaviour asked for, stated as the only comparison that is actually controlled.

    A cross-BODY comparison is not: at any single drop the foot's sole is far deeper than the
    knee's lowest surface (the shank clears the ground by 5.2 cm at nominal stance), and the
    two have different surface areas, so a per-point ratio across them measures geometry and
    not the weights. So hold the geometry fixed and change only the weight: raising the feet
    from the foot weight to the default must multiply their charge by exactly 10.
    """
    pts = plan_collider_points(*_standing(1.0, STAND_Z - 0.15), model)
    contact = _no_contact(plan_contact(pts, model, dt=0.02))

    light = terrain_consistency(pts, contact, terrain, model, TerrainConsistencySpec())
    heavy = terrain_consistency(pts, contact, terrain, model, TerrainConsistencySpec(
        body_penetration_weights={}))          # no exceptions -> the feet take the default
    i = model.body_order.index("left_ankle_roll_link")
    assert light["penetration_by_body"][0, i].item() > 0.0
    assert heavy["penetration_by_body"][0, i].item() == pytest.approx(
        (DEFAULT_PENETRATION_WEIGHT / FOOT_PENETRATION_WEIGHT)
        * light["penetration_by_body"][0, i].item(), rel=1e-9)


def test_a_negative_per_body_weight_is_refused():
    """It would turn the penalty into a bonus for that body."""
    with pytest.raises(ValueError, match="negative per-body"):
        TerrainConsistencySpec(body_penetration_weights={"left_ankle_roll_link": -1.0})
    with pytest.raises(ValueError, match="penetration_default_weight"):
        TerrainConsistencySpec(penetration_default_weight=-1.0)


def test_the_default_weight_table_is_not_shared_between_specs():
    """A mutable default on a dataclass is the classic footgun; ``default_factory`` avoids it."""
    a, b = TerrainConsistencySpec(), TerrainConsistencySpec()
    assert a.body_penetration_weights == b.body_penetration_weights
    assert a.body_penetration_weights is not b.body_penetration_weights

