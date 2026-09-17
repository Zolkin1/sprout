# Copyright 2026 Zachary Olkin. All rights reserved.

"""A predicted window's world-frame collision geometry, and the contacts implied by it.

The flow-matching generator's window is ``base_pose(9) + base_vel(6) + joint_pos(29)`` and
carries **no end-effector block**, so a foot -- let alone a foot's shape -- is not something
that can be read out of a plan.  It has to be computed, and this is where.

Two steps, both batched over ``(N, F)`` and both device-portable:

1. :func:`plan_collider_points` -- FK the predicted joints through
   :class:`~motion_gen.fk.FKChain`, place each body's collider points
   (:mod:`motion_gen.collision`) in the body frame, and lift the whole thing into the world
   with the predicted base pose.  The base ROTATION is applied, not just the translation: a
   tilted base really does swing the feet, and leaving it out under-reports foot error (the
   same point ``scripts/flow_matching/fm_foot_error.py`` makes).
2. :func:`plan_contact` -- mark a foot as in contact at a node when its slowest sole corner
   is slower than a threshold.

**Why velocity and not proximity.**  The contact flag gates a reward that grades geometry
against the terrain, so it must come from a channel that knows nothing about the terrain, or
the reward is a tautology: define contact by proximity and rewarding proximity says nothing.
Velocity is that independent channel, and the statement it makes is falsifiable -- "this plan
says the foot is planted, and it is 15 cm above the ground".

**Per sole CORNER, not per ankle.**  A foot pivoting over the toe has a moving ankle and a
stationary toe, so an ankle-origin speed misses both ends of a stance phase. Measured on 200
flat clips of ``g1_mb_v5_body_mode11_flat.h5`` against the library's own (there, valid)
contact flag, min-over-corners speed by central finite difference gives **zero false
positives** up to 0.20 m/s, with recall 0.48 at 0.10 m/s and 0.68 at 0.20 -- i.e. it is a
high-precision, low-recall detector, which is the useful direction.  Recall is what
:func:`dilate_contact` exists to absorb.
"""

from __future__ import annotations

import numpy as np
import torch

from .collision import BodyColliders, collider_points, flatten_colliders
from .fk import FKChain

#: Bodies whose SOLE is the ground-contact geometry, in left/right order.
#:
#: The same two frames the clip libraries record as ``contact_frames`` and the same two
#: ``motion_gen.features.FOOT_FRAMES`` names, so the contact heuristic here and the recorded
#: flag it is calibrated against are talking about the same links.
FOOT_BODIES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")

#: Default contact speed threshold (m/s) -- the slowest sole corner below this is stance.
#:
#: **0.05, and the choice is about COVERAGE, not severity.**  Measured over 20k recorded plans,
#: ``contact_reward`` is flat at 0.906-0.910 across every threshold from 0.03 to 0.10, so
#: moving it does not make the term stricter; it changes how much of the plan gets graded::
#:
#:     thresh   coverage   support   plans with NO stance node
#:      0.02      0.016     0.032          18.28%
#:      0.03      0.036     0.070           3.94%
#:      0.04      0.062     0.119           0.68%
#:      0.05      0.093     0.175           0.10%
#:      0.08      0.204     0.360           0.00%
#:      0.10      0.279     0.471           0.00%
#:
#: **The floor is the last column, and it is a hard one.**  A plan with no marked stance has
#: nothing for the contact term to average, so it scores 0 -- which reads as "every stance was
#: badly placed" when it actually means "no stance was detected".  Below 0.03 that fires on a
#: fifth of all plans and the median score drops to 0.870 from that artefact alone.  0.05
#: keeps it at one plan in a thousand.
#:
#: On the calibration side the whole range is safe: against the (valid) flat-clip contact flag,
#: precision is 1.000 at every threshold up to 0.20, so lowering it only trades recall away.
#: A false positive would be a node where the reward demands ground under an airborne foot;
#: a false negative only costs coverage.
#:
#: The plan's own sole speed for scale: q05 0.036, q10 0.052, q25 0.092, q50 0.181 m/s. The
#: generator's plans are smoother than a clip's, whose feet reach 0.10 m/s in 48% of frames
#: against the plan's 28%, so a threshold read off clip data over-marks a plan.
CONTACT_SPEED_THRESH: float = 0.05

#: Default surface-sample spacing (m) for the collider point sets.
#:
#: 3.5 cm gives **1462 points** over the whole robot and **41 on each foot's sole**, which is
#: what makes the area-weighted penetration sum an integral rather than a point count.
#: ``None`` falls back to the 46 extreme points, which are still the right thing for a cheap
#: ``max``.
#:
#: Chosen for cost, and the sampling is comfortably converged by here: the recovered total
#: surface area moves **0.25%** between 3.0 and 3.5 cm (1.16109 -> 1.16398 m^2) while the
#: point count drops 22.5% (1886 -> 1462).  Point count scales as ``1 / spacing^2``, so this
#: is the knob to turn if the reward ever costs too much per re-plan -- 4 cm would be another
#: 19% off.
DEFAULT_COLLIDER_SPACING: float = 0.035

#: Longest run of non-contact nodes that is CLOSED into the stance around it.
#:
#: The velocity test fragments a single stance phase: a foot planted for 20 nodes can dip
#: above the threshold for a node or two mid-stance (the sole rocks, the finite difference
#: picks up the pelvis moving over it), which splits one contact into several.  A gap that
#: short is a detection artefact, not a real lift-off -- 3 nodes is 0.06 s, and no real swing
#: phase is that brief.
#:
#: This is a better answer to the detector's ~50% recall than the exemption buffer is: it
#: recovers the INTERIOR of a stance phase, where the foot really is planted and really should
#: be graded for being on the ground, instead of merely declining to charge it.  Both are
#: kept -- the buffer covers the ENDS of a stance (touchdown and toe-off), which have no
#: bounding contact on one side and so cannot be closed.
#:
#: Only gaps bounded by contact on BOTH sides are filled: a leading or trailing run is a real
#: swing, not a gap.
CONTACT_MAX_GAP_NODES: int = 3

#: Nodes each side of a detected stance that are exempted from the penetration penalty.
#:
#: The buffer for the ~50% recall.  An undetected touchdown or toe-off lives in exactly these
#: nodes, and a foot genuinely in contact there sits at ``sd ~ 0``, which the penetration
#: penalty would otherwise charge as if it were a plan driving a foot into the ground.  Two
#: nodes is 0.04 s at the control dt.
CONTACT_DILATE_NODES: int = 2


class PlanColliderModel:
    """The FK chain + collider point set a plan's geometry is evaluated with.

    Built once and reused: parsing the MJCF and resolving the chain is the whole setup cost,
    and a reward term evaluated every re-plan must not pay it per call.

    Attributes are the flattened form :func:`~motion_gen.collision.flatten_colliders`
    produces, because a per-body reduction (or a per-body contact exemption) is then a
    scatter over ``body_index`` rather than a Python loop over bodies.
    """

    def __init__(
        self,
        mjcf_path: str,
        joint_names: list[str],
        body_names: list[str] | None = None,
        base_link: str = "pelvis_link",
        dtype: torch.dtype = torch.float32,
        spacing: float | None = DEFAULT_COLLIDER_SPACING,
    ):
        """Resolve the chain and the collider points.

        Args:
            mjcf_path: The MJCF, e.g. ``motion_gen.features.DEFAULT_MJCF``.
            joint_names: The plan's joint order (the policy/Isaac order).
            body_names: Bodies to test, or ``None`` for every body in the MJCF that has a
                collider.  ``None`` is the default so the set is a property of the robot file.
            base_link: The link the window's ``base_pose`` refers to.
            dtype: Working dtype.
            spacing: Surface-sample spacing (m); ``None`` for the 46 extreme points.

        Raises:
            ValueError: If a foot body ends up outside the collider set -- the contact
                heuristic reads the feet, so a set without them would silently produce a
                reward with no contact term at all.
        """
        self.colliders: dict[str, BodyColliders] = collider_points(
            mjcf_path, body_names, spacing=spacing)
        self.body_order, pts, radii, areas, body_index = flatten_colliders(self.colliders)
        missing = [f for f in FOOT_BODIES if f not in self.body_order]
        if missing:
            raise ValueError(
                f"the collider set is missing {missing}, but the contact heuristic reads the "
                "feet. A set without them yields a reward whose contact term is empty and "
                "whose penetration term is never exempted."
            )
        self.chain = FKChain(mjcf_path, list(joint_names), list(self.body_order),
                             base_link=base_link, dtype=dtype)
        self.spacing = spacing
        self.points = torch.as_tensor(pts, dtype=dtype)               # (P, 3) body frame
        self.radii = torch.as_tensor(radii, dtype=dtype)              # (P,)
        self.areas = torch.as_tensor(areas, dtype=dtype)              # (P,) m^2 per point
        self.body_index = torch.as_tensor(body_index, dtype=torch.long)
        self.foot_index = torch.as_tensor(
            [self.body_order.index(f) for f in FOOT_BODIES], dtype=torch.long)
        # The points on each foot's BOTTOM FACE, as indices into the flat point array.
        # Contact is read off the sole; every other point on the foot belongs to the
        # penetration test. Selected by ``z == min(z)`` rather than ``z < mean(z)``: on a
        # sampled surface the latter takes the whole lower half of the box, side faces
        # included, which would let a foot resting on its outer edge read as planted.
        sole = []
        for f in FOOT_BODIES:
            local = self.colliders[f].points
            own = np.nonzero(body_index == self.body_order.index(f))[0]
            zs = local[:, 2]
            sole.append(own[zs <= zs.min() + 1e-9])
        counts = {len(s_) for s_ in sole}
        if len(counts) != 1:
            raise ValueError(
                f"the two feet sampled to different sole-point counts {counts}; they are the "
                "same geometry, so this means the sampling is not deterministic.")
        self.sole_index = torch.as_tensor(np.stack(sole, axis=0), dtype=torch.long)
        self.dtype = dtype
        self.device = torch.device("cpu")

    def to(self, device: torch.device | str) -> "PlanColliderModel":
        """Move every tensor to ``device`` in place, and return self."""
        self.chain = self.chain.to(device)
        self.points = self.points.to(device)
        self.radii = self.radii.to(device)
        self.areas = self.areas.to(device)
        self.body_index = self.body_index.to(device)
        self.foot_index = self.foot_index.to(device)
        self.sole_index = self.sole_index.to(device)
        self.device = torch.device(device)
        return self

    @property
    def num_points(self) -> int:
        """Total collider sample points ``P``."""
        return int(self.points.shape[0])

    @property
    def total_area_m2(self) -> float:
        """Total sampled surface area (m^2); ``0`` in extreme-point mode."""
        return float(self.areas.sum())


def plan_collider_points(
    base_pos: torch.Tensor,
    base_rot: torch.Tensor,
    joint_pos: torch.Tensor,
    model: PlanColliderModel,
) -> torch.Tensor:
    """World-frame collider points for every node of a predicted window.

    Takes the base ROTATION MATRIX rather than the window's rot6d on purpose: the window
    layout and its orientation encoding belong to the package that defines the window, so
    the caller converts (``robot_rl.flow_matching.transforms.rot6d_to_matrix``, or the
    deploy numpy twin) and this stays layout-agnostic.

    Args:
        base_pos: ``(N, F, 3)`` predicted base position, WORLD frame.
        base_rot: ``(N, F, 3, 3)`` predicted base orientation, WORLD frame.
        joint_pos: ``(N, F, nj)`` predicted joint angles, in ``model``'s joint order.
        model: The resolved :class:`PlanColliderModel`.  Its dtype must match the inputs'.

    Returns:
        ``(N, F, P, 3)`` world-frame collider points, in ``model.dtype``.

    Raises:
        ValueError: On a shape disagreement between the three inputs, or a dtype that does
            not match the model's.
    """
    if base_pos.ndim != 3 or base_pos.shape[-1] != 3:
        raise ValueError(f"base_pos must be (N, F, 3); got {tuple(base_pos.shape)}")
    if base_rot.shape[:2] != base_pos.shape[:2] or base_rot.shape[-2:] != (3, 3):
        raise ValueError(f"base_rot must be (N, F, 3, 3); got {tuple(base_rot.shape)}")
    if joint_pos.shape[:2] != base_pos.shape[:2]:
        raise ValueError(
            f"joint_pos {tuple(joint_pos.shape)} disagrees with base_pos "
            f"{tuple(base_pos.shape)} on (N, F)")
    # REFUSED rather than cast. ``FKChain`` bakes its dtype at construction, so casting here
    # would silently run a float64 caller's plan through a float32 chain and hand back
    # float32 -- a precision downgrade with no error, in a quantity measured in millimetres.
    if base_pos.dtype != model.dtype or joint_pos.dtype != model.dtype:
        raise ValueError(
            f"dtype mismatch: the model was built as {model.dtype} but base_pos is "
            f"{base_pos.dtype} and joint_pos is {joint_pos.dtype}. Build the "
            f"PlanColliderModel with dtype={base_pos.dtype} instead of casting here.")

    n, f = base_pos.shape[:2]
    q = joint_pos.reshape(n * f, joint_pos.shape[-1])
    pos_b, rot_b, _jv, _jw = model.chain.pose_and_jacobian(q)     # (N*F, K, 3), (N*F, K, 3, 3)
    k = pos_b.shape[1]
    pos_b = pos_b.reshape(n, f, k, 3)
    rot_b = rot_b.reshape(n, f, k, 3, 3)

    # Each point in its own body's frame -> the base frame, gathered per point.
    bi = model.body_index                                          # (P,)
    p_local = model.points                                         # (P, 3)
    # (N, F, P, 3) = R_body[..., bi] @ p_local + t_body[..., bi]
    r_pt = rot_b[:, :, bi]                                         # (N, F, P, 3, 3)
    t_pt = pos_b[:, :, bi]                                         # (N, F, P, 3)
    in_base = torch.einsum("nfpij,pj->nfpi", r_pt, p_local) + t_pt

    return torch.einsum("nfij,nfpj->nfpi", base_rot.to(in_base.dtype), in_base) \
        + base_pos[:, :, None, :]


def node_speed(points: torch.Tensor, dt: float) -> torch.Tensor:
    """Speed of each point along the node axis, by central finite difference.

    Central differences, one-sided at the two ends.  Finite differences rather than the
    window's velocity channels because the shipped window carries no ``joint_vel``, so there
    is no analytic point velocity to read -- and measured against the analytic
    ``v_ankle + omega x r`` on the same clips they score identically, so nothing is lost.

    Args:
        points: ``(N, F, P, 3)`` world-frame points; ``F >= 2``.
        dt: Node spacing (s).  The plan is contiguous control steps, so 0.02 s.

    Returns:
        ``(N, F, P)`` speeds (m/s).

    Raises:
        ValueError: If ``F < 2`` (no derivative exists) or ``dt <= 0``.
    """
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"points must be (N, F, P, 3); got {tuple(points.shape)}")
    f = points.shape[1]
    if f < 2:
        raise ValueError(f"need at least 2 nodes to difference; got F={f}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")
    vel = torch.empty_like(points)
    vel[:, 1:-1] = (points[:, 2:] - points[:, :-2]) / (2.0 * dt)
    vel[:, 0] = (points[:, 1] - points[:, 0]) / dt
    vel[:, -1] = (points[:, -1] - points[:, -2]) / dt
    return vel.norm(dim=-1)


def close_contact_gaps(contact: torch.Tensor, max_gap: int) -> torch.Tensor:
    """Fill runs of non-contact of at most ``max_gap`` nodes that lie BETWEEN contacts.

    A morphological closing would be the usual tool, but its structuring element fills gaps of
    even length only (radius ``r`` fills ``2r``), so it cannot express "at most 3".  This is
    the exact run-length statement instead, and it is still vectorised: for every node take the
    index of the nearest contact at or before it and the nearest at or after it, and fill when
    both exist and the gap between them is short enough.

    Per channel, so one foot's stance can never close a gap in the other's.

    Args:
        contact: ``(N, F, C)`` bool contact mask.
        max_gap: Longest fillable run of non-contact; ``0`` returns the input unchanged.

    Returns:
        ``(N, F, C)`` bool, a superset of ``contact``.

    Raises:
        ValueError: If ``max_gap`` is negative.
    """
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0; got {max_gap}")
    if max_gap == 0:
        return contact
    f = contact.shape[1]
    idx = torch.arange(f, device=contact.device).view(1, f, 1).expand_as(contact)
    # Nearest contact at or before each node (-1 where there is none yet), and at or after
    # (F where there is none left). A node inside a fillable gap has both.
    prev = torch.cummax(torch.where(contact, idx, torch.full_like(idx, -1)), dim=1).values
    nxt = torch.cummin(torch.where(contact, idx, torch.full_like(idx, f)).flip(1),
                       dim=1).values.flip(1)
    fill = (prev >= 0) & (nxt < f) & ((nxt - prev - 1) <= max_gap)
    return contact | fill


def dilate_contact(contact: torch.Tensor, radius: int) -> torch.Tensor:
    """Widen a contact mask by ``radius`` nodes each side, clamped at the plan's ends.

    Args:
        contact: ``(N, F, C)`` bool contact mask.
        radius: Nodes to dilate each way; ``0`` returns the input unchanged.

    Returns:
        ``(N, F, C)`` bool, ``True`` where any node within ``radius`` was in contact.

    Raises:
        ValueError: If ``radius`` is negative.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0; got {radius}")
    if radius == 0:
        return contact
    # Max-pool along F with 'same' padding. Clamping at the ends (rather than treating
    # off-plan nodes as not-in-contact) is deliberate: node 0 of a plan continues the
    # previous one, so a stance at node 0 was almost certainly a stance at node -1 too.
    x = contact.to(torch.float32).permute(0, 2, 1)                 # (N, C, F)
    x = torch.nn.functional.max_pool1d(x, kernel_size=2 * radius + 1, stride=1,
                                       padding=radius)
    return x.permute(0, 2, 1) > 0.5


def plan_contact(
    points: torch.Tensor,
    model: PlanColliderModel,
    dt: float,
    thresh: float = CONTACT_SPEED_THRESH,
    dilate: int = CONTACT_DILATE_NODES,
    max_gap: int = CONTACT_MAX_GAP_NODES,
) -> dict[str, torch.Tensor]:
    """Contact flags for both feet at every node, from sole-corner speed alone.

    Three stages, in this order, and the order matters:

    1. threshold the slowest sole point's speed -> ``contact_raw``;
    2. **close** gaps of at most ``max_gap`` nodes that sit between two contacts ->
       ``contact``, which is what the reward grades.  A short gap mid-stance is the detector
       fragmenting one stance phase, not a real lift-off;
    3. **dilate** by ``dilate`` nodes -> ``contact_buffered``, which exempts a foot from the
       penetration penalty.  Dilating the CLOSED mask, not the raw one, so the buffer sits
       around whole stance phases rather than around each fragment.

    Args:
        points: ``(N, F, P, 3)`` world-frame collider points from
            :func:`plan_collider_points`.
        model: The model those points were built with (supplies the sole indices).
        dt: Node spacing (s).
        thresh: Speed below which a sole point counts as stationary (m/s).
        max_gap: Longest run of non-contact closed into the stance around it.
        dilate: Nodes each side of a stance to include in ``contact_buffered``.

    Returns:
        Mapping with

        * ``corner_speed`` ``(N, F, 2, n_sole)`` -- per sole point (m/s);
        * ``foot_speed`` ``(N, F, 2)`` -- the slowest sole point per foot;
        * ``contact_raw`` ``(N, F, 2)`` bool -- straight off the threshold, before closing;
        * ``contact`` ``(N, F, 2)`` bool -- gap-closed; **what the contact reward grades**;
        * ``contact_filled`` ``(N, F, 2)`` bool -- exactly what the closing ADDED, so the
          filter's effect is inspectable rather than invisible;
        * ``contact_buffered`` ``(N, F, 2)`` bool -- dilated, what exempts a foot from the
          penetration penalty;
        * ``support`` ``(N, F)`` bool -- at least one foot in contact.
    """
    speed = node_speed(points, dt)                                 # (N, F, P)
    corner = speed[:, :, model.sole_index]                         # (N, F, 2, n_sole)
    foot = corner.amin(dim=-1)                                     # (N, F, 2)
    raw = foot < float(thresh)
    contact = close_contact_gaps(raw, int(max_gap))
    return {
        "corner_speed": corner,
        "foot_speed": foot,
        "contact_raw": raw,
        "contact": contact,
        "contact_filled": contact & ~raw,
        "contact_buffered": dilate_contact(contact, int(dilate)),
        "support": contact.any(dim=-1),
    }
