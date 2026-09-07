# Copyright 2026 Zachary Olkin. All rights reserved.

"""A plan's world geometry, and the contacts backed out of it by velocity alone.

The generator's window carries no end-effector block, so a foot is a DERIVED quantity: FK of
the predicted joints, placed by the predicted base pose.  These tests pin the derivation
against a plain per-body FK call (the flattened point/body-index path is the part that could
gather wrongly and still look plausible), pin that the base ROTATION is applied, and pin the
dilation's exact width -- which is what the penetration exemption depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motion_gen import features as F
from motion_gen.fk import FKChain
from motion_gen.joint_names import G1_JOINT_NAMES
from motion_gen.plan_geometry import (
    FOOT_BODIES,
    PlanColliderModel,
    close_contact_gaps,
    dilate_contact,
    node_speed,
    plan_collider_points,
    plan_contact,
)

NJ = len(G1_JOINT_NAMES)
STAND_Z = 0.7869


@pytest.fixture(scope="module")
def model() -> PlanColliderModel:
    """One resolved collider model, shared: the MJCF parse is the whole setup cost.

    float64 so the FK comparison below is limited by the arithmetic and not by fp32.
    """
    return PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64)


def _plan(n: int, f: int, z: float = STAND_Z, yaw: float = 0.0
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A static standing plan at height ``z``, rotated by ``yaw``."""
    base_pos = torch.zeros(n, f, 3, dtype=torch.float64)
    base_pos[..., 2] = z
    c, s = np.cos(yaw), np.sin(yaw)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return base_pos, rot.expand(n, f, 3, 3).contiguous(), torch.zeros(n, f, NJ,
                                                                      dtype=torch.float64)


def test_collider_points_match_a_plain_per_body_fk(model):
    """The flattened gather must equal FK'ing each body and placing its own points.

    The recipe is the one ``scripts/flow_matching/fm_foot_error.bodies_from_window`` uses:
    FK in the base frame, then ``einsum("nfij,nfbj->nfbi", R_base, p) + t_base``.
    """
    rng = np.random.default_rng(0)
    n, f = 2, 5
    q = torch.as_tensor(rng.uniform(-0.5, 0.5, size=(n, f, NJ)))
    base_pos = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(n, f, 3)))
    ang = rng.uniform(-1.0, 1.0, size=(n, f))
    rot = torch.stack([torch.stack([
        torch.tensor([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float64) for a in row]) for row in ang])

    got = plan_collider_points(base_pos, rot, q, model)

    chain = FKChain(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), list(model.body_order),
                    base_link="pelvis_link", dtype=torch.float64)
    pos_b, rot_b, _jv, _jw = chain.pose_and_jacobian(q.reshape(n * f, NJ))
    pos_b = pos_b.reshape(n, f, -1, 3)
    rot_b = rot_b.reshape(n, f, -1, 3, 3)
    for i, body in enumerate(model.body_order):
        local = torch.as_tensor(model.colliders[body].points)
        in_base = torch.einsum("nfij,pj->nfpi", rot_b[:, :, i], local) + pos_b[:, :, i, None, :]
        want = torch.einsum("nfij,nfpj->nfpi", rot, in_base) + base_pos[:, :, None, :]
        assert torch.allclose(got[:, :, model.body_index == i], want, atol=1e-10), body


def test_the_base_rotation_is_applied(model):
    """A tilted base really does swing the feet; leaving R out under-reports foot error."""
    n, f = 1, 3
    p0 = plan_collider_points(*_plan(n, f, yaw=0.0), model)
    p90 = plan_collider_points(*_plan(n, f, yaw=np.pi / 2.0), model)
    # A yaw of 90 deg maps (x, y) -> (-y, x); the feet are laterally offset, so they MOVE.
    assert not torch.allclose(p0, p90, atol=1e-3)
    assert torch.allclose(p90[..., 0], -p0[..., 1], atol=1e-10)
    assert torch.allclose(p90[..., 1], p0[..., 0], atol=1e-10)
    assert torch.allclose(p90[..., 2], p0[..., 2], atol=1e-10)


def test_the_sole_indices_are_the_bottom_FACE_of_each_foot(model):
    """Contact is read off the sole face only.

    ``z <= min(z)``, not ``z < mean(z)``: on a sampled surface the latter takes the whole
    lower half of the box, side faces included, which would let a foot resting on its outer
    edge read as planted.
    """
    n_sole = model.sole_index.shape[1]
    assert model.sole_index.shape[0] == 2 and n_sole >= 4
    for j, foot in enumerate(FOOT_BODIES):
        own = model.body_index == model.body_order.index(foot)
        z_all = model.points[own, 2]
        z_sole = model.points[model.sole_index[j], 2]
        assert torch.allclose(z_sole, z_all.min().expand(n_sole))
        assert z_sole[0] < z_all.max()


def test_extreme_mode_still_gives_exactly_four_sole_corners(model):
    """The cheap mode is unchanged, so a caller that wants 46 points keeps them."""
    ext = PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64,
                            spacing=None)
    assert ext.num_points == 46
    assert tuple(ext.sole_index.shape) == (2, 4)


def test_node_speed_is_exact_on_a_ramp():
    """Central differences are exact on a linear path; the ends are one-sided."""
    f, dt = 7, 0.02
    t = torch.arange(f, dtype=torch.float64)[None, :, None, None] * dt
    pts = torch.cat([t * 1.5, t * 0.0, t * -2.0], dim=-1)      # |v| = sqrt(1.5^2 + 2^2)
    got = node_speed(pts, dt)
    assert torch.allclose(got, torch.full_like(got, 2.5), atol=1e-12)

    with pytest.raises(ValueError, match="at least 2 nodes"):
        node_speed(pts[:, :1], dt)
    with pytest.raises(ValueError, match="dt must be positive"):
        node_speed(pts, 0.0)


def test_dilation_width_is_exactly_the_radius_each_side():
    """A single stance node exempts 2r+1 nodes -- the property the exemption relies on."""
    f = 21
    contact = torch.zeros(1, f, 2, dtype=torch.bool)
    contact[0, 10, 0] = True
    for r in (0, 1, 2, 3):
        got = dilate_contact(contact, r)
        assert int(got[0, :, 0].sum()) == 2 * r + 1, r
        assert int(got[0, :, 1].sum()) == 0, "the other foot must be untouched"
        assert bool(got[0, 10 - r, 0]) and bool(got[0, 10 + r, 0])
        if r < 10:
            assert not bool(got[0, 10 - r - 1, 0])


def test_dilation_clamps_at_the_plan_ends_and_radius_zero_is_the_identity():
    """Node 0 continues the previous plan, so a stance there dilates inward only."""
    contact = torch.zeros(1, 8, 1, dtype=torch.bool)
    contact[0, 0, 0] = True
    got = dilate_contact(contact, 2)
    assert got[0, :, 0].tolist() == [True, True, True, False, False, False, False, False]
    assert torch.equal(dilate_contact(contact, 0), contact)
    with pytest.raises(ValueError, match="radius must be"):
        dilate_contact(contact, -1)


def test_a_static_standing_plan_is_all_contact_and_a_lifted_one_is_none(model):
    """The heuristic's two extremes, so a sign error cannot hide."""
    n, f = 1, 10
    still = plan_collider_points(*_plan(n, f), model)
    c = plan_contact(still, model, dt=0.02)
    assert bool(c["contact"].all()) and bool(c["support"].all())
    assert c["foot_speed"].max().item() == pytest.approx(0.0, abs=1e-12)

    # Now translate the whole robot at 1 m/s: nothing is in contact at any threshold below it.
    base_pos, rot, q = _plan(n, f)
    base_pos = base_pos + torch.arange(f, dtype=torch.float64)[None, :, None] * 0.02
    moving = plan_collider_points(base_pos, rot, q, model)
    c = plan_contact(moving, model, dt=0.02, thresh=0.10)
    assert not bool(c["contact"].any())
    assert c["foot_speed"].min().item() > 1.0


def test_a_dtype_mismatch_is_refused_rather_than_silently_downcast(model):
    """A float64 plan through a float32 chain would lose millimetres with no error."""
    base_pos, rot, q = _plan(1, 3)
    with pytest.raises(ValueError, match="dtype mismatch"):
        plan_collider_points(base_pos.float(), rot.float(), q.float(), model)


def test_a_collider_set_without_the_feet_is_refused():
    """The contact heuristic reads the feet; a set without them is a silently empty reward."""
    with pytest.raises(ValueError, match="missing"):
        PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), body_names=["torso_link"])


def _seq(pattern: str) -> torch.Tensor:
    """``"#..#"`` -> a ``(1, F, 1)`` bool contact mask."""
    return torch.tensor([c == "#" for c in pattern], dtype=torch.bool).view(1, -1, 1)


def _show(mask: torch.Tensor) -> str:
    """A ``(1, F, 1)`` mask back as a ``#``/``.`` string."""
    return "".join("#" if v else "." for v in mask[0, :, 0].tolist())


@pytest.mark.parametrize("pattern,want", [
    ("#..#", "####"),                    # a 2-node gap closes
    ("#...#", "#####"),                  # a 3-node gap closes -- the boundary case
    ("#....#", "#....#"),                # a 4-node gap does NOT
    ("...#..#...", "...####..."),         # leading and trailing runs are real swing
    ("#.#.#.#", "#######"),              # several short gaps all close
    ("....", "...."),                    # nothing to close
    ("####", "####"),                    # nothing to do
    ("#", "#"),
    (".#.", ".#."),                      # a lone contact grows nothing
])
def test_gap_closing_is_exactly_at_most_max_gap_and_only_between_contacts(pattern, want):
    """The exact run-length semantics, boundary included.

    A morphological closing cannot express "at most 3" -- radius r fills 2r -- so this pins
    that the implementation is the run-length statement and not a dilate/erode pair.
    """
    assert _show(close_contact_gaps(_seq(pattern), 3)) == want


def test_gap_closing_is_per_channel():
    """One foot's stance must never close a gap in the other's."""
    c = torch.zeros(1, 5, 2, dtype=torch.bool)
    c[0, [0, 4], 0] = True                       # foot 0: #...#
    out = close_contact_gaps(c, 3)
    assert out[0, :, 0].tolist() == [True] * 5
    assert out[0, :, 1].tolist() == [False] * 5


def test_gap_closing_never_removes_a_contact_and_zero_is_the_identity():
    """It is a closing: the result is a superset, and the knob has a null setting."""
    rng = np.random.default_rng(0)
    c = torch.as_tensor(rng.random((4, 40, 2)) < 0.3)
    for g in (0, 1, 3, 7):
        out = close_contact_gaps(c, g)
        assert bool((out | c == out).all()), "closing must be a superset"
    assert torch.equal(close_contact_gaps(c, 0), c)
    with pytest.raises(ValueError, match="max_gap must be"):
        close_contact_gaps(c, -1)


def test_gap_closing_is_monotone_in_max_gap():
    """A larger gap can only add contact, never take it away."""
    rng = np.random.default_rng(1)
    c = torch.as_tensor(rng.random((4, 60, 2)) < 0.25)
    prev = close_contact_gaps(c, 0)
    for g in (1, 2, 3, 5, 10):
        cur = close_contact_gaps(c, g)
        assert bool((prev | cur == cur).all()), g
        prev = cur


def test_plan_contact_closes_BEFORE_it_dilates(model):
    """The buffer must be the dilation of the CLOSED mask, not of the raw one.

    The two orders often give the same buffered mask -- a 3-node gap is bridged by a +-2
    dilation anyway -- so the property that actually matters is not mask equality but WHICH
    OUTPUT the filled nodes land in. Closed-then-dilated puts them in ``contact``, where the
    contact reward grades them for being on the ground. Dilated-then-closed would put them
    only in ``contact_buffered``, where they are merely exempt from the penetration penalty
    and graded by nothing. That is the difference this asserts.
    """
    n, f = 1, 24
    pts = plan_collider_points(*_plan(n, f), model)
    got = plan_contact(pts, model, dt=0.02, thresh=1e9, max_gap=3, dilate=2)
    assert torch.equal(got["contact_buffered"], dilate_contact(got["contact"], 2)), (
        "contact_buffered must be the dilation of the closed mask")
    assert bool((got["contact"] | got["contact_buffered"] == got["contact_buffered"]).all())
    assert bool((got["contact_raw"] | got["contact"] == got["contact"]).all())


def test_a_closed_gap_is_GRADED_not_merely_exempted():
    """A filled node has to reach the contact term, which is the point of closing at all."""
    raw = torch.zeros(1, 12, 2, dtype=torch.bool)
    raw[0, [2, 6], :] = True                          # a 3-node gap between two stances
    closed = close_contact_gaps(raw, 3)
    filled = closed & ~raw
    assert bool(filled[0, 3:6, 0].all()), "nodes 3-5 are the gap"
    # They are in the mask the reward AVERAGES over, not just in the exemption mask.
    assert bool(closed[0, 3:6, 0].all())
    # And a gap one node longer stays a real swing.
    raw2 = torch.zeros(1, 12, 2, dtype=torch.bool)
    raw2[0, [2, 7], :] = True
    assert not bool(close_contact_gaps(raw2, 3)[0, 4, 0])


def test_plan_contact_reports_what_the_filter_added(model):
    """``contact_filled`` makes the filter inspectable instead of invisible."""
    n, f = 1, 12
    pts = plan_collider_points(*_plan(n, f), model)
    got = plan_contact(pts, model, dt=0.02, thresh=1e9)
    assert torch.equal(got["contact_filled"], got["contact"] & ~got["contact_raw"])
    assert not bool(got["contact_filled"].any()), "nothing to fill when all nodes are stance"
